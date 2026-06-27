//! The Agent orchestrator coordinates between the Provider and the Tools.

pub mod magi_adapter;
pub mod magi_wiring;
pub mod messages;
pub mod provider;

use crate::agent::messages::{Content, Message, Role};
use crate::agent::provider::{Provider, ResponseChunk};
use crate::memory::clock::Clock;
use crate::memory::config::MemoryConfig;
use crate::memory::context::assemble_selective;
use crate::memory::decay::purge_expired_archives;
use crate::memory::embedding::EmbeddingProvider;
use crate::memory::judge::LlmDistillJudge;
use crate::memory::profile::{distill, render_profile};
use crate::memory::retrieval::reembed_pending;
use crate::memory::salience::assign_salience;
use crate::memory::store::{Memory, SqliteVectorStore, VectorStore};
use crate::memory::MemoryKind;
use crate::system::database::MemoryStore;
use crate::tools::Tool;
use anyhow::Result;
use futures::StreamExt;
use sha2::{Digest, Sha256};
use std::sync::Arc;
use tokio::sync::oneshot;
use tokio::time::{timeout, Duration};

const APPROVAL_TIMEOUT_SECS: u64 = 300; // 5 minutes

/// Approval request sent to the UI.
pub struct ApprovalRequest {
    pub tool_name: String,
    /// Carries tool input for the approval prompt; reserved.
    #[allow(dead_code)]
    pub input: serde_json::Value,
    pub tx: oneshot::Sender<bool>,
}

/// A piece of a streamed assistant turn forwarded to the UI. `Content` is answer
/// text (persisted); `Reasoning` is a thinking model's chain-of-thought, shown
/// live but never persisted (#24).
#[derive(Debug, Clone, PartialEq)]
pub enum StreamPiece {
    Content(String),
    Reasoning(String),
}

/// Tiered-memory subsystem handle for `selective` mode (Task 12).
///
/// Absent (`None` on `Agent`) ⇒ legacy `load_all` behavior: the full
/// conversation history is sent to the provider on every turn, preserving
/// the behavior of all 188 pre-existing tests (REQ-28/SC-32).
struct MemorySubsystem {
    store: Arc<SqliteVectorStore>,
    embedder: Arc<dyn EmbeddingProvider>,
    clock: Arc<dyn Clock>,
    cfg: MemoryConfig,
    /// Scope tag for memory isolation; defaults to `"root"`. Multi-scope is
    /// activated in L3 (Agent Society — AS-REQ-09).
    scope: String,
    /// System-prompt text for token-budget accounting; empty when none is set.
    system: String,
    /// Number of selective turns processed since `set_memory_subsystem`. Used
    /// to determine when to fire the `distill_every_n_turns` trigger (Task 13b).
    turns_since_open: usize,
}

/// The Agent orchestrator.
pub struct Agent {
    provider: Arc<dyn Provider>,
    tools: Vec<Box<dyn Tool>>,
    history: Vec<Message>,
    /// Optional persistent memory store.
    memory: Option<Arc<dyn MemoryStore>>,
    /// Current session ID in the memory store.
    session_id: Option<String>,
    /// Optional channel to request approval for tools.
    approval_tx: Option<tokio::sync::mpsc::Sender<ApprovalRequest>>,
    /// Safeguard for infinite loops (max tool calls per query)
    pub max_tool_calls: usize,
    /// Optional tiered-memory subsystem for `selective` mode (Task 12).
    /// `None` ⇒ legacy `load_all` behavior; all 188 pre-existing tests rely on
    /// this path being byte-identical to the original implementation.
    memory_subsystem: Option<MemorySubsystem>,
}

impl Agent {
    /// Creates a new Agent.
    pub fn new(provider: Arc<dyn Provider>) -> Self {
        Self {
            provider,
            tools: Vec::new(),
            history: Vec::new(),
            memory: None,
            session_id: None,
            approval_tx: None,
            max_tool_calls: 15,
            memory_subsystem: None,
        }
    }

    /// Sets the persistent memory store and session for the agent.
    pub fn set_memory(&mut self, memory: Arc<dyn MemoryStore>, session_id: String) {
        self.memory = Some(memory);
        self.session_id = Some(session_id);
    }

    /// Replaces the active LLM provider (e.g. after a mid-session `/login` swaps
    /// the startup `StaticProvider` for a live `AnthropicProvider`).
    pub fn set_provider(&mut self, provider: Arc<dyn Provider>) {
        self.provider = provider;
    }

    /// Whether the active provider is the canned `StaticProvider` (no API key).
    /// Used by `/login` to decide whether the prior history is canned noise (safe
    /// to clear) vs a real conversation (must be kept).
    pub fn provider_is_static(&self) -> bool {
        self.provider.is_static()
    }

    /// Loads history from the persistent memory store.
    pub async fn load_history(&mut self) -> Result<()> {
        if let (Some(memory), Some(sid)) = (&self.memory, &self.session_id) {
            let messages = memory.get_messages(sid).await?;
            self.history = messages;
        }
        Ok(())
    }

    /// Set approval channel.
    pub fn set_approval_channel(&mut self, tx: tokio::sync::mpsc::Sender<ApprovalRequest>) {
        self.approval_tx = Some(tx);
    }

    /// Registers a tool with the agent.
    pub fn register_tool(&mut self, tool: Box<dyn Tool>) {
        self.tools.push(tool);
    }

    /// Registers `tool`, replacing any existing tool with the same `name()`.
    /// Used to refresh a provider-bound tool (e.g. `consult`) after the active
    /// provider changes via `/login`, so it never holds stale credentials.
    pub fn register_or_replace_tool(&mut self, tool: Box<dyn Tool>) {
        if let Some(slot) = self.tools.iter_mut().find(|t| t.name() == tool.name()) {
            *slot = tool;
        } else {
            self.tools.push(tool);
        }
    }

    /// Normalizes tool input recursively to detect semantically identical calls.
    pub fn normalize_input(val: &serde_json::Value, depth: usize) -> Result<String> {
        const MAX_DEPTH: usize = 10;
        if depth > MAX_DEPTH {
            return Err(anyhow::anyhow!("JSON nesting limit exceeded"));
        }

        match val {
            serde_json::Value::Object(map) => {
                let mut sorted_keys: Vec<_> = map.keys().collect();
                sorted_keys.sort();
                let mut parts = Vec::new();
                for k in sorted_keys {
                    let v = map.get(k).unwrap();
                    parts.push(format!("{}:{}", k, Self::normalize_input(v, depth + 1)?));
                }
                Ok(format!("{{{}}}", parts.join(",")))
            }
            serde_json::Value::Array(arr) => {
                let mut normalized_elements = Vec::new();
                for v in arr {
                    normalized_elements.push(Self::normalize_input(v, depth + 1)?);
                }
                Ok(format!("[{}]", normalized_elements.join(",")))
            }
            serde_json::Value::String(s) => Ok(s.trim().to_string()),
            _ => Ok(val.to_string()),
        }
    }

    /// Strips terminal escape sequences and control characters for security.
    pub fn sanitize_text(text: &str) -> String {
        let mut result = String::with_capacity(text.len());
        let mut chars = text.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '\x1B' {
                if let Some('[') = chars.peek() {
                    chars.next();
                    for next in chars.by_ref() {
                        if next.is_ascii_alphabetic() {
                            break;
                        }
                    }
                    continue;
                }
            }
            if c.is_control() && c != '\n' && c != '\r' && c != '\t' {
                continue;
            }
            result.push(c);
        }
        result
    }

    /// Clears the conversation history.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Attaches the tiered-memory subsystem, enabling `selective` mode inside
    /// `query_streaming`.
    ///
    /// Once set, each turn is persisted as a vector-indexed episodic memory and
    /// the bounded context assembler (`assemble_selective`) replaces the full
    /// history on every provider call. The legacy persistence path
    /// (`memory.add_message`) is preserved in both modes.
    ///
    /// # Parameters
    /// - `store` — encrypted vector store (shares the SQLite connection of
    ///   `EncryptedSqliteMemory`, so the file stays a single on-disk asset).
    /// - `embedder` — text-to-vector backend (Ollama default; any openai-compat
    ///   endpoint works by pointing `base_url`).
    /// - `clock` — wall-clock abstraction; inject `FixedClock` in tests for
    ///   deterministic decay (D-18 / R-06).
    /// - `cfg` — memory configuration (context budget, weights, mode, …).
    pub fn set_memory_subsystem(
        &mut self,
        store: Arc<SqliteVectorStore>,
        embedder: Arc<dyn EmbeddingProvider>,
        clock: Arc<dyn Clock>,
        cfg: MemoryConfig,
    ) {
        self.memory_subsystem = Some(MemorySubsystem {
            store,
            embedder,
            clock,
            cfg,
            scope: "root".into(),
            system: String::new(),
            turns_since_open: 0,
        });
    }

    /// Runs best-effort maintenance on session open: re-embeds pending memories
    /// (stored without a vector due to a transient embedding failure) and purges
    /// expired archives.
    ///
    /// Errors are intentionally swallowed — maintenance failures must never
    /// block session startup (REQ-29).
    pub async fn on_session_open(&self) -> Result<()> {
        if let Some(s) = &self.memory_subsystem {
            let _ = reembed_pending(&*s.store, &*s.embedder, &s.cfg, &s.scope).await;
            let _ = purge_expired_archives(&*s.store, &*s.clock, &s.cfg).await;
        }
        Ok(())
    }

    /// Runs a best-effort distillation pass on session close (Task 13b).
    ///
    /// If the tiered-memory subsystem is attached and
    /// `cfg.distill_on_session_close` is true, runs [`distill`] once.
    /// Errors are swallowed — a distillation failure must never prevent clean
    /// shutdown (REQ-29).
    ///
    /// # Errors
    /// Always returns `Ok(())` (errors swallowed internally).
    pub async fn on_session_close(&self) -> anyhow::Result<()> {
        let Some(sub) = self.memory_subsystem.as_ref() else {
            return Ok(());
        };
        if !sub.cfg.distill_on_session_close || !sub.cfg.distill_enabled {
            return Ok(());
        }
        let judge = LlmDistillJudge::new(self.provider.clone());
        let _ = distill(
            &*sub.store,
            &judge,
            &*sub.embedder,
            &*sub.clock,
            &sub.cfg,
            &sub.scope,
        )
        .await
        .map_err(|e| eprintln!("on_session_close distill: {e}"));
        Ok(())
    }

    /// Process a user message and returns the final assistant response.
    /// This method supports streaming via a sender channel.
    pub async fn query_streaming(
        &mut self,
        text: &str,
        chunk_tx: tokio::sync::mpsc::Sender<StreamPiece>,
    ) -> Result<String> {
        let user_msg = Message::user(text);
        self.history.push(user_msg.clone());

        // Persist user message (both modes keep the message-store path for
        // backward compatibility and load_history support).
        if let (Some(memory), Some(sid)) = (&self.memory, &self.session_id) {
            memory.add_message(sid, &user_msg).await?;
        }

        // ── Selective tiered-memory path ──────────────────────────────────────
        // When a subsystem is attached and `mode == "selective"`, each turn is
        // persisted to the vector store and the assembler builds a bounded context
        // instead of sending the full history.
        //
        // The load_all path below is untouched — byte-identical to the original
        // implementation — so all 188 pre-existing tests pass unchanged (REQ-28).
        if self
            .memory_subsystem
            .as_ref()
            .is_some_and(|s| s.cfg.mode == "selective")
        {
            // Snapshot Arc handles (ref-count increments only; O(1)) so the mutable
            // loop body below can borrow `self` freely without conflicting field
            // borrows on `memory_subsystem`.
            let (store, embedder, clock, cfg, scope, system_str) = {
                let s = self.memory_subsystem.as_ref().unwrap();
                (
                    s.store.clone(),
                    s.embedder.clone(),
                    s.clock.clone(),
                    s.cfg.clone(),
                    s.scope.clone(),
                    s.system.clone(),
                )
            };
            let session_id_str = self.session_id.clone().unwrap_or_default();

            // Render the profile fresh from the store so promoted preferences
            // appear in the context immediately after distillation (Task 13b/REQ-16).
            let live_profile = render_profile(&*store, &cfg, &scope)
                .await
                .unwrap_or_default();

            // Write user turn to vector store (best-effort; REQ-29).
            write_turn_to_memory(
                &store,
                &*embedder,
                &*clock,
                &cfg,
                &scope,
                &session_id_str,
                text,
                Role::User,
            )
            .await;

            // Assemble bounded context: system + profile + top-k recalls + turn.
            let assembled = assemble_selective(
                &*store,
                &*embedder,
                &*clock,
                &cfg,
                &system_str,
                &live_profile,
                &user_msg,
                &scope,
            )
            .await
            .map_err(|e| anyhow::anyhow!("context assembly failed: {e}"))?;
            let mut working = assembled.messages;

            // ── Selective tool loop ───────────────────────────────────────────
            let mut tool_call_count = 0;
            let mut last_normalized_tool: Option<(String, String)> = None;
            let mut repeat_count = 0;

            loop {
                let mut stream = self.provider.stream_messages(&working, &self.tools).await?;
                let mut full_text = String::new();
                let mut last_message: Option<Message> = None;

                while let Some(chunk_result) = stream.next().await {
                    match chunk_result? {
                        ResponseChunk::TextDelta(delta) => {
                            let sanitized = Self::sanitize_text(&delta);
                            full_text.push_str(&sanitized);
                            if chunk_tx
                                .send(StreamPiece::Content(sanitized))
                                .await
                                .is_err()
                            {
                                return Err(anyhow::anyhow!(
                                    "TUI connection closed during streaming"
                                ));
                            }
                        }
                        ResponseChunk::ReasoningDelta(delta) => {
                            let sanitized = Self::sanitize_text(&delta);
                            if chunk_tx
                                .send(StreamPiece::Reasoning(sanitized))
                                .await
                                .is_err()
                            {
                                return Err(anyhow::anyhow!(
                                    "TUI connection closed during streaming"
                                ));
                            }
                        }
                        ResponseChunk::MessageDone(msg) => {
                            last_message = Some(msg);
                        }
                        _ => {}
                    }
                }

                let response = last_message
                    .ok_or_else(|| anyhow::anyhow!("Stream ended without MessageDone"))?;
                // Both legacy persistence paths (self.history + memory.add_message) are
                // kept so load_history and the existing message store stay consistent.
                self.history.push(response.clone());
                working.push(response.clone());
                if let (Some(memory), Some(sid)) = (&self.memory, &self.session_id) {
                    memory.add_message(sid, &response).await?;
                }

                let mut tool_results = Vec::new();
                let mut requested_tool = false;

                for content in &response.content {
                    if let Content::ToolUse { id, name, input } = content {
                        requested_tool = true;
                        tool_call_count += 1;
                        if tool_call_count > self.max_tool_calls {
                            return Err(anyhow::anyhow!("Maximum tool call limit reached"));
                        }

                        let normalized_input = Self::normalize_input(input, 0)?;
                        if let Some((ref last_name, ref last_norm_input)) = last_normalized_tool {
                            if last_name == name && last_norm_input == &normalized_input {
                                repeat_count += 1;
                                if repeat_count >= 3 {
                                    return Err(anyhow::anyhow!("Repetitive tool call detected"));
                                }
                            } else {
                                repeat_count = 0;
                            }
                        }
                        last_normalized_tool = Some((name.clone(), normalized_input));

                        let approved = if let Some(ref tx) = self.approval_tx {
                            let (oneshot_tx, oneshot_rx) = oneshot::channel();
                            let _ = tx
                                .send(ApprovalRequest {
                                    tool_name: name.clone(),
                                    input: input.clone(),
                                    tx: oneshot_tx,
                                })
                                .await;
                            match timeout(Duration::from_secs(APPROVAL_TIMEOUT_SECS), oneshot_rx)
                                .await
                            {
                                Ok(Ok(res)) => res,
                                _ => false,
                            }
                        } else {
                            true
                        };

                        if !approved {
                            tool_results.push(Content::ToolResult {
                                tool_use_id: id.clone(),
                                content: "Execution denied or timed out.".to_string(),
                                is_error: true,
                            });
                            continue;
                        }

                        let tool_result =
                            if let Some(tool) = self.tools.iter().find(|t| t.name() == name) {
                                match tool.execute(input.clone()).await {
                                    Ok(val) => Content::ToolResult {
                                        tool_use_id: id.clone(),
                                        content: val.to_string(),
                                        is_error: false,
                                    },
                                    Err(e) => Content::ToolResult {
                                        tool_use_id: id.clone(),
                                        content: e.to_string(),
                                        is_error: true,
                                    },
                                }
                            } else {
                                Content::ToolResult {
                                    tool_use_id: id.clone(),
                                    content: format!("Tool '{}' not found", name),
                                    is_error: true,
                                }
                            };
                        tool_results.push(tool_result);
                    }
                }

                if requested_tool {
                    let tool_res_msg = Message {
                        role: Role::User,
                        content: tool_results,
                    };
                    self.history.push(tool_res_msg.clone());
                    working.push(tool_res_msg.clone());
                    if let (Some(memory), Some(sid)) = (&self.memory, &self.session_id) {
                        memory.add_message(sid, &tool_res_msg).await?;
                    }
                } else {
                    // Write assistant response to the vector store (best-effort; REQ-29).
                    write_turn_to_memory(
                        &store,
                        &*embedder,
                        &*clock,
                        &cfg,
                        &scope,
                        &session_id_str,
                        &full_text,
                        Role::Assistant,
                    )
                    .await;

                    // Increment turn counter and fire distill pass when cadence is reached
                    // (Task 13b / REQ-17). Errors are non-fatal (CP2-Z).
                    let should_distill = {
                        let sub = self.memory_subsystem.as_mut().unwrap();
                        sub.turns_since_open = sub.turns_since_open.saturating_add(1);
                        let n = sub.cfg.distill_every_n_turns;
                        sub.cfg.distill_enabled && n > 0 && sub.turns_since_open.is_multiple_of(n)
                    };
                    if should_distill {
                        let judge = LlmDistillJudge::new(self.provider.clone());
                        let _ = distill(&*store, &judge, &*embedder, &*clock, &cfg, &scope)
                            .await
                            .map_err(|e| eprintln!("distill: {e}"));
                    }

                    for content in response.content.iter().rev() {
                        if let Content::Text { text } = content {
                            return Ok(text.clone());
                        }
                    }
                    return Ok(String::new());
                }
            }
        }

        // ── load_all path (original, byte-identical) ──────────────────────────
        // No memory subsystem or mode != "selective" → send the full history to the
        // provider exactly as before.  This path is the sole path exercised by the
        // 188 pre-existing tests (REQ-28 / SC-32).
        let mut tool_call_count = 0;
        let mut last_normalized_tool: Option<(String, String)> = None;
        let mut repeat_count = 0;

        loop {
            let mut stream = self
                .provider
                .stream_messages(&self.history, &self.tools)
                .await?;
            let mut full_text = String::new();
            let mut last_message: Option<Message> = None;

            while let Some(chunk_result) = stream.next().await {
                match chunk_result? {
                    ResponseChunk::TextDelta(delta) => {
                        let sanitized = Self::sanitize_text(&delta);
                        full_text.push_str(&sanitized);
                        if chunk_tx
                            .send(StreamPiece::Content(sanitized))
                            .await
                            .is_err()
                        {
                            return Err(anyhow::anyhow!("TUI connection closed during streaming"));
                        }
                    }
                    ResponseChunk::ReasoningDelta(delta) => {
                        // Forwarded for live display only — NOT added to `full_text`,
                        // so the persisted assistant message excludes the thinking.
                        let sanitized = Self::sanitize_text(&delta);
                        if chunk_tx
                            .send(StreamPiece::Reasoning(sanitized))
                            .await
                            .is_err()
                        {
                            return Err(anyhow::anyhow!("TUI connection closed during streaming"));
                        }
                    }
                    ResponseChunk::MessageDone(msg) => {
                        last_message = Some(msg);
                    }
                    _ => {}
                }
            }

            let response =
                last_message.ok_or_else(|| anyhow::anyhow!("Stream ended without MessageDone"))?;
            self.history.push(response.clone());

            // Persist assistant response
            if let (Some(memory), Some(sid)) = (&self.memory, &self.session_id) {
                memory.add_message(sid, &response).await?;
            }

            let mut tool_results = Vec::new();
            let mut requested_tool = false;

            for content in &response.content {
                if let Content::ToolUse { id, name, input } = content {
                    requested_tool = true;
                    tool_call_count += 1;
                    if tool_call_count > self.max_tool_calls {
                        return Err(anyhow::anyhow!("Maximum tool call limit reached"));
                    }

                    let normalized_input = Self::normalize_input(input, 0)?;
                    if let Some((ref last_name, ref last_norm_input)) = last_normalized_tool {
                        if last_name == name && last_norm_input == &normalized_input {
                            repeat_count += 1;
                            if repeat_count >= 3 {
                                return Err(anyhow::anyhow!("Repetitive tool call detected"));
                            }
                        } else {
                            repeat_count = 0;
                        }
                    }
                    last_normalized_tool = Some((name.clone(), normalized_input));

                    let approved = if let Some(ref tx) = self.approval_tx {
                        let (oneshot_tx, oneshot_rx) = oneshot::channel();
                        let _ = tx
                            .send(ApprovalRequest {
                                tool_name: name.clone(),
                                input: input.clone(),
                                tx: oneshot_tx,
                            })
                            .await;
                        match timeout(Duration::from_secs(APPROVAL_TIMEOUT_SECS), oneshot_rx).await
                        {
                            Ok(Ok(res)) => res,
                            _ => false,
                        }
                    } else {
                        true
                    };

                    if !approved {
                        tool_results.push(Content::ToolResult {
                            tool_use_id: id.clone(),
                            content: "Execution denied or timed out.".to_string(),
                            is_error: true,
                        });
                        continue;
                    }

                    let tool_result =
                        if let Some(tool) = self.tools.iter().find(|t| t.name() == name) {
                            match tool.execute(input.clone()).await {
                                Ok(val) => Content::ToolResult {
                                    tool_use_id: id.clone(),
                                    content: val.to_string(),
                                    is_error: false,
                                },
                                Err(e) => Content::ToolResult {
                                    tool_use_id: id.clone(),
                                    content: e.to_string(),
                                    is_error: true,
                                },
                            }
                        } else {
                            Content::ToolResult {
                                tool_use_id: id.clone(),
                                content: format!("Tool '{}' not found", name),
                                is_error: true,
                            }
                        };
                    tool_results.push(tool_result);
                }
            }

            if requested_tool {
                let tool_res_msg = Message {
                    role: Role::User,
                    content: tool_results,
                };
                self.history.push(tool_res_msg.clone());
                // Persist tool results
                if let (Some(memory), Some(sid)) = (&self.memory, &self.session_id) {
                    memory.add_message(sid, &tool_res_msg).await?;
                }
            } else {
                for content in response.content.iter().rev() {
                    if let Content::Text { text } = content {
                        return Ok(text.clone());
                    }
                }
                return Ok(String::new());
            }
        }
    }
}

/// Persists one conversational turn to the encrypted vector store (best-effort).
///
/// On embedding failure the record is stored with an empty vector, which marks it
/// for lazy re-embed on the next `on_session_open` call. A write error must never
/// abort the user's turn (REQ-29).
///
/// # Parameters
/// - `store` — encrypted vector store to insert the record into.
/// - `embedder` — text-to-vector backend; `document_prefix` is applied before embedding.
/// - `clock` — wall-clock abstraction for `created_at` / `last_accessed_at`.
/// - `cfg` — memory config for salience heuristic.
/// - `scope` — isolation scope (always `"root"` in Task 12).
/// - `session_id` — owning session UUID.
/// - `text` — raw turn text (not yet prefixed).
/// - `role` — `Role::User` or `Role::Assistant`; stored in the ID hash for uniqueness.
///
/// # Note
/// This is intentionally a module-level free function (not a method on `Agent`) so
/// it can be called from inside the `query_streaming` async loop without creating
/// conflicting borrows on `self`.
#[allow(clippy::too_many_arguments)]
async fn write_turn_to_memory(
    store: &SqliteVectorStore,
    embedder: &dyn EmbeddingProvider,
    clock: &dyn Clock,
    cfg: &MemoryConfig,
    scope: &str,
    session_id: &str,
    text: &str,
    role: Role,
) {
    if text.trim().is_empty() {
        return;
    }
    let now = clock.now();
    let salience = assign_salience(MemoryKind::Episodic, text, role.clone(), cfg);

    // Apply document prefix before embedding (D-04 / REQ-34).
    let doc_prefix = embedder.document_prefix();
    let prefixed = if doc_prefix.is_empty() {
        text.to_string()
    } else {
        format!("{doc_prefix}{text}")
    };

    // Best-effort embed — on failure store an empty vector marked for lazy re-embed
    // (REQ-29: write failures must never abort the user's turn).
    let (embedding, model_id, dim) = match embedder.embed(&[prefixed]).await {
        Ok(v) if !v.is_empty() => {
            let d = v[0].len();
            (
                v.into_iter().next().unwrap(),
                embedder.model_id().to_string(),
                d,
            )
        }
        _ => (Vec::new(), String::new(), 0),
    };

    // Use a content-hash-based ID so identical texts within the same second still
    // produce distinct records if the role differs.
    let id = format!(
        "turn:{:x}",
        Sha256::digest(format!("{now}:{role:?}:{text}").as_bytes())
    );

    let m = Memory {
        id,
        session_id: session_id.to_string(),
        kind: MemoryKind::Episodic,
        text: text.to_string(),
        embedding,
        model_id,
        dim,
        created_at: now,
        salience,
        access_count: 0,
        last_accessed_at: now,
        superseded_by: None,
        evicted_at: None,
        scope: scope.to_string(),
        distilled_at: None,
    };

    // Swallow write errors — a store failure must not abort the turn (REQ-29).
    let _ = store.insert(&m).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use async_trait::async_trait;
    use futures::stream::{self, BoxStream};
    use serde_json::json;

    pub struct MockProvider;

    #[async_trait]
    impl Provider for MockProvider {
        async fn stream_messages(
            &self,
            _messages: &[Message],
            _tools: &[Box<dyn Tool>],
        ) -> Result<BoxStream<'static, Result<ResponseChunk>>> {
            let chunks = vec![
                Ok(ResponseChunk::TextDelta("Summary content.".to_string())),
                Ok(ResponseChunk::MessageDone(Message::assistant(
                    "Summary content.",
                ))),
            ];
            Ok(Box::pin(stream::iter(chunks)))
        }
        async fn send_messages(
            &self,
            _messages: &[Message],
            _tools: &[Box<dyn Tool>],
        ) -> Result<Message> {
            Ok(Message::assistant("Summary content."))
        }
    }

    #[tokio::test]
    async fn test_agent_normalization_depth_limit() {
        let mut deep_json = json!({"path": "."});
        for i in 0..20 {
            deep_json = json!({format!("level_{}", i): deep_json});
        }
        let result = Agent::normalize_input(&deep_json, 0);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_agent_text_sanitization() {
        let input = "\x1B[31mDangerous\x1B[0m Text\x07";
        assert_eq!(Agent::sanitize_text(input), "Dangerous Text");
    }

    #[tokio::test]
    async fn test_agent_encrypted_persistence_integration() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let db_path = tmp_dir.path().join("test_persist.db");
        let password = "master_password_123".to_string();

        let msg_text = "Persist this message";
        let _sid = "session_1".to_string();

        // Scope 1: Create agent and save a message
        let sid = {
            let mut agent = Agent::new(Arc::new(MockProvider));
            let memory = Arc::new(
                crate::system::database::EncryptedSqliteMemory::new(
                    db_path.clone(),
                    password.clone(),
                )
                .unwrap(),
            );

            // Create session
            let id = memory.create_session("test_proj").await.unwrap();
            agent.set_memory(memory.clone(), id.clone());

            let user_msg = Message::user(msg_text);
            agent.history.push(user_msg.clone());
            memory.add_message(&id, &user_msg).await.unwrap();
            id
        };

        // Scope 2: Recreate agent and verify loading
        {
            let mut agent = Agent::new(Arc::new(MockProvider));
            let memory = Arc::new(
                crate::system::database::EncryptedSqliteMemory::new(db_path, password).unwrap(),
            );
            agent.set_memory(memory, sid);

            // Load history
            agent.load_history().await.unwrap();

            assert_eq!(agent.history.len(), 1);
            if let Content::Text { text } = &agent.history[0].content[0] {
                assert_eq!(text, msg_text);
            } else {
                panic!("Expected text content");
            }
        }
    }

    #[tokio::test]
    async fn test_query_streaming_forwards_deltas_to_channel_before_final() {
        use tokio::sync::mpsc;

        struct TwoDeltaProvider;

        #[async_trait]
        impl Provider for TwoDeltaProvider {
            async fn stream_messages(
                &self,
                _messages: &[Message],
                _tools: &[Box<dyn Tool>],
            ) -> Result<BoxStream<'static, Result<ResponseChunk>>> {
                let chunks = vec![
                    Ok(ResponseChunk::ReasoningDelta("thinking".to_string())),
                    Ok(ResponseChunk::TextDelta("Hello ".to_string())),
                    Ok(ResponseChunk::TextDelta("world".to_string())),
                    Ok(ResponseChunk::MessageDone(Message::assistant(
                        "Hello world",
                    ))),
                ];
                Ok(Box::pin(stream::iter(chunks)))
            }
        }

        let mut agent = Agent::new(Arc::new(TwoDeltaProvider));
        let (chunk_tx, mut chunk_rx) = mpsc::channel::<StreamPiece>(8);

        let collector = tokio::spawn(async move {
            let mut received = Vec::new();
            while let Some(piece) = chunk_rx.recv().await {
                received.push(piece);
            }
            received
        });

        let final_text = agent.query_streaming("Hi", chunk_tx).await.unwrap();
        let received = collector.await.unwrap();

        // Reasoning is forwarded as a distinct piece (for live display) and is NOT
        // part of the persisted final answer text; content is forwarded as Content.
        assert_eq!(
            received,
            vec![
                StreamPiece::Reasoning("thinking".to_string()),
                StreamPiece::Content("Hello ".to_string()),
                StreamPiece::Content("world".to_string()),
            ]
        );
        assert_eq!(final_text, "Hello world");
    }

    #[tokio::test]
    async fn test_set_provider_swaps_the_active_provider() {
        // A-S1 (#9): a mid-session set_provider replaces the active provider.
        struct FixedProvider(&'static str);

        #[async_trait]
        impl Provider for FixedProvider {
            async fn stream_messages(
                &self,
                _messages: &[Message],
                _tools: &[Box<dyn Tool>],
            ) -> Result<BoxStream<'static, Result<ResponseChunk>>> {
                let text = self.0.to_string();
                Ok(Box::pin(stream::iter(vec![Ok(
                    ResponseChunk::MessageDone(Message::assistant(&text)),
                )])))
            }
        }

        let mut agent = Agent::new(Arc::new(FixedProvider("from-A")));
        agent.set_provider(Arc::new(FixedProvider("from-B")));
        let (tx, _rx) = tokio::sync::mpsc::channel::<StreamPiece>(8);
        let out = agent.query_streaming("hi", tx).await.unwrap();
        assert_eq!(out, "from-B", "set_provider must swap the active provider");
    }

    #[tokio::test]
    async fn test_provider_is_static_reflects_provider() {
        // StaticProvider → true (canned, safe to clear on login); other → false.
        let static_agent = Agent::new(Arc::new(crate::agent::provider::StaticProvider));
        assert!(static_agent.provider_is_static());
        let real_agent = Agent::new(Arc::new(MockProvider));
        assert!(!real_agent.provider_is_static());
    }

    // ── Task-12 helpers ────────────────────────────────────────────────────────

    /// Provider that records every `messages` slice it receives.
    /// Used to assert what context the agent sent in each call.
    struct CapturingProvider {
        calls: Arc<std::sync::Mutex<Vec<Vec<Message>>>>,
    }

    impl CapturingProvider {
        fn new() -> (Self, Arc<std::sync::Mutex<Vec<Vec<Message>>>>) {
            let calls = Arc::new(std::sync::Mutex::new(Vec::<Vec<Message>>::new()));
            (
                Self {
                    calls: calls.clone(),
                },
                calls,
            )
        }
    }

    #[async_trait]
    impl Provider for CapturingProvider {
        async fn stream_messages(
            &self,
            messages: &[Message],
            _tools: &[Box<dyn Tool>],
        ) -> Result<BoxStream<'static, Result<ResponseChunk>>> {
            self.calls.lock().unwrap().push(messages.to_vec());
            let chunks = vec![
                Ok(ResponseChunk::TextDelta("ok".to_string())),
                Ok(ResponseChunk::MessageDone(Message::assistant("ok"))),
            ];
            Ok(Box::pin(stream::iter(chunks)))
        }
    }

    /// Deterministic bag-of-words embedder for tests (no HTTP calls).
    /// Matches the pattern used in `context.rs` and `retrieval.rs` tests.
    fn bow(text: &str, dim: usize) -> Vec<f32> {
        let mut v = vec![0f32; dim];
        for w in text.to_lowercase().split_whitespace() {
            let h = w
                .bytes()
                .fold(0usize, |a, b| a.wrapping_mul(31).wrapping_add(b as usize))
                % dim;
            v[h] += 1.0;
        }
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 {
            for x in &mut v {
                *x /= n;
            }
        }
        v
    }

    struct FakeEmbedder {
        dim: usize,
        model: String,
    }

    #[async_trait]
    impl EmbeddingProvider for FakeEmbedder {
        async fn embed(
            &self,
            texts: &[String],
        ) -> std::result::Result<Vec<Vec<f32>>, crate::memory::error::EmbeddingError> {
            Ok(texts.iter().map(|t| bow(t, self.dim)).collect())
        }
        fn model_id(&self) -> &str {
            &self.model
        }
        fn dim(&self) -> usize {
            self.dim
        }
        fn query_prefix(&self) -> &str {
            ""
        }
        fn document_prefix(&self) -> &str {
            ""
        }
    }

    // ── D1/D2 tests ────────────────────────────────────────────────────────────

    /// Embedder that always returns an auth error — used to simulate transient
    /// failures in `assemble_selective`.
    struct ErrorEmbedder;

    #[async_trait]
    impl EmbeddingProvider for ErrorEmbedder {
        async fn embed(
            &self,
            _texts: &[String],
        ) -> std::result::Result<Vec<Vec<f32>>, crate::memory::error::EmbeddingError> {
            Err(crate::memory::error::EmbeddingError::Auth)
        }
        fn model_id(&self) -> &str {
            "error-model"
        }
        fn dim(&self) -> usize {
            16
        }
        fn query_prefix(&self) -> &str {
            ""
        }
        fn document_prefix(&self) -> &str {
            ""
        }
    }

    /// D1 (REQ-29): when `assemble_selective` encounters a transient error (e.g.
    /// embedder auth failure), `query_streaming` must NOT propagate the error —
    /// it must fall back to the `load_all` history path and still return a
    /// successful response.
    #[tokio::test]
    async fn test_selective_mode_falls_back_to_history_on_embedder_error() {
        use crate::memory::clock::FixedClock;
        use crate::memory::config::MemoryConfig;
        use crate::memory::store::SqliteVectorStore;
        use crate::system::database::EncryptedSqliteMemory;
        use tokio::sync::mpsc;

        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mem = EncryptedSqliteMemory::new(tmp.path().to_path_buf(), "pw".into()).unwrap();
        let vstore = Arc::new(SqliteVectorStore::new(mem.shared_conn(), mem.data_key()).unwrap());
        let clock = Arc::new(FixedClock::new(1_000_000));
        let cfg = MemoryConfig {
            mode: "selective".into(),
            ..MemoryConfig::default()
        };

        // Use an embedder that always errors — assemble_selective will fail.
        let embedder = Arc::new(ErrorEmbedder);

        let (cap, _calls) = CapturingProvider::new();
        let mut agent = Agent::new(Arc::new(cap));
        agent.set_memory_subsystem(vstore, embedder, clock, cfg);

        let (tx, mut rx) = mpsc::channel::<StreamPiece>(16);
        // Must succeed (fallback), not return an Err.
        let result = agent.query_streaming("hello", tx).await;
        assert!(
            result.is_ok(),
            "D1: selective mode with embedder error must fall back, not propagate Err: {result:?}"
        );

        // Provider must still have returned a response.
        let mut got_response = false;
        while let Ok(piece) = rx.try_recv() {
            if let StreamPiece::Content(text) = piece {
                if !text.is_empty() {
                    got_response = true;
                }
            }
        }
        assert!(
            got_response,
            "D1: a response must be delivered even after embedder error (fallback path)"
        );
    }

    /// D2: when the context assembler produces a truncation notice (D-17 oversized
    /// turn), the notice must be forwarded to `chunk_tx` before the provider call.
    #[tokio::test]
    async fn test_selective_mode_forwards_assembler_notices() {
        use crate::memory::clock::FixedClock;
        use crate::memory::config::MemoryConfig;
        use crate::memory::store::SqliteVectorStore;
        use crate::system::database::EncryptedSqliteMemory;
        use tokio::sync::mpsc;

        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mem = EncryptedSqliteMemory::new(tmp.path().to_path_buf(), "pw".into()).unwrap();
        let vstore = Arc::new(SqliteVectorStore::new(mem.shared_conn(), mem.data_key()).unwrap());
        let embedder = Arc::new(FakeEmbedder {
            dim: 16,
            model: "fake".into(),
        });
        let clock = Arc::new(FixedClock::new(1_000_000));
        // Very tight budget so that a long turn triggers a truncation notice.
        let cfg = MemoryConfig {
            mode: "selective".into(),
            context_budget_tokens: 20,    // 20 tokens ≈ 70 chars
            response_headroom_tokens: 0,
            safety_margin_ratio: 0.0,
            oversized_turn_policy: "truncate".into(),
            ..MemoryConfig::default()
        };

        let (cap, _calls) = CapturingProvider::new();
        let mut agent = Agent::new(Arc::new(cap));
        agent.set_memory_subsystem(vstore, embedder, clock, cfg);

        // An oversized turn (well over the 20-token budget).
        let long_input = "x".repeat(500);
        let (tx, mut rx) = mpsc::channel::<StreamPiece>(32);
        agent.query_streaming(&long_input, tx).await.unwrap();

        // Collect all streamed pieces.
        let mut pieces = Vec::new();
        while let Ok(p) = rx.try_recv() {
            pieces.push(p);
        }

        // At least one piece must be a `[memory: …]` notice.
        let notice_found = pieces.iter().any(|p| {
            if let StreamPiece::Content(text) = p {
                text.starts_with("[memory:")
            } else {
                false
            }
        });
        assert!(
            notice_found,
            "D2: a truncation notice must be forwarded to chunk_tx when the turn exceeds the budget; \
             got pieces: {pieces:?}"
        );
    }

    // ── Task-12 tests (SC-32 / SC-22 / SC-18) ─────────────────────────────────

    /// SC-32: An agent with NO memory subsystem set behaves exactly as today.
    /// The full conversation history grows turn by turn and is sent to the
    /// provider unchanged.  This test must pass in RED, GREEN, and REFACTOR.
    #[tokio::test]
    async fn test_load_all_mode_is_unchanged() {
        let mut agent = Agent::new(Arc::new(MockProvider));
        let (tx1, _rx1) = tokio::sync::mpsc::channel::<StreamPiece>(8);
        agent.query_streaming("hello", tx1).await.unwrap();
        // After 1 turn: User("hello") + Asst("Summary content.") = 2 messages.
        assert_eq!(
            agent.history.len(),
            2,
            "load_all: history must have 2 messages after 1 turn"
        );

        let (tx2, _rx2) = tokio::sync::mpsc::channel::<StreamPiece>(8);
        agent.query_streaming("world", tx2).await.unwrap();
        // After 2 turns: 4 messages total.
        assert_eq!(
            agent.history.len(),
            4,
            "load_all: history must grow to 4 messages after 2 turns"
        );

        // First user turn must still be present in the full history.
        let has_hello = agent.history.iter().any(|m| {
            m.role == Role::User
                && m.content
                    .iter()
                    .any(|c| matches!(c, Content::Text { text } if text == "hello"))
        });
        assert!(
            has_hello,
            "load_all: full history must contain the first user turn"
        );
    }

    /// SC-32 / SC-18: In `selective` mode the provider receives the assembled
    /// bounded context, not the growing full history.  Assembled context never
    /// contains an `Assistant`-role message (only `User`-role context messages
    /// are produced by the assembler).
    ///
    /// Fails in RED because `query_streaming` has no selective branch yet.
    #[tokio::test]
    async fn test_selective_mode_sends_assembled_context_not_full_history() {
        use crate::memory::clock::FixedClock;
        use crate::memory::config::MemoryConfig;
        use crate::memory::store::SqliteVectorStore;
        use crate::system::database::EncryptedSqliteMemory;

        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mem = EncryptedSqliteMemory::new(tmp.path().to_path_buf(), "pw".into()).unwrap();
        let vstore = Arc::new(SqliteVectorStore::new(mem.shared_conn(), mem.data_key()).unwrap());
        let embedder = Arc::new(FakeEmbedder {
            dim: 16,
            model: "fake".into(),
        });
        let clock = Arc::new(FixedClock::new(1_000_000));
        let cfg = MemoryConfig {
            mode: "selective".into(),
            ..MemoryConfig::default()
        };

        let (cap, calls) = CapturingProvider::new();
        let mut agent = Agent::new(Arc::new(cap));
        agent.set_memory_subsystem(vstore, embedder, clock, cfg);

        let (tx1, _rx1) = tokio::sync::mpsc::channel::<StreamPiece>(8);
        agent.query_streaming("first query", tx1).await.unwrap();

        let (tx2, _rx2) = tokio::sync::mpsc::channel::<StreamPiece>(8);
        agent.query_streaming("second query", tx2).await.unwrap();

        let all_calls = calls.lock().unwrap();
        assert!(
            all_calls.len() >= 2,
            "provider must have been called at least twice"
        );
        // Assembled context is exclusively User-role messages; no Assistant turns
        // appear (the assembler produces preamble + current turn, both User-role).
        let turn2_msgs = &all_calls[1];
        let has_assistant = turn2_msgs.iter().any(|m| m.role == Role::Assistant);
        assert!(
            !has_assistant,
            "SC-18/SC-32: selective mode must send assembled context without \
             Assistant messages (got {} messages in turn 2)",
            turn2_msgs.len()
        );
    }

    /// SC-22: In `selective` mode, a fact stated in turn 1 is written to the
    /// vector store and visible to the assembler for turn 2.
    ///
    /// Fails in RED because `write_turn_to_memory` is a no-op stub.
    #[tokio::test]
    async fn test_fact_written_in_one_turn_is_recalled_in_a_later_turn() {
        use crate::memory::clock::FixedClock;
        use crate::memory::config::MemoryConfig;
        use crate::memory::store::{SqliteVectorStore, VectorStore};
        use crate::system::database::EncryptedSqliteMemory;

        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mem = EncryptedSqliteMemory::new(tmp.path().to_path_buf(), "pw".into()).unwrap();
        let vstore = Arc::new(SqliteVectorStore::new(mem.shared_conn(), mem.data_key()).unwrap());
        let embedder = Arc::new(FakeEmbedder {
            dim: 32,
            model: "fake".into(),
        });
        let clock = Arc::new(FixedClock::new(1_000_000));
        let cfg = MemoryConfig {
            mode: "selective".into(),
            context_budget_tokens: 2000,
            response_headroom_tokens: 0,
            safety_margin_ratio: 0.0,
            top_k: 10,
            ..MemoryConfig::default()
        };

        let (cap, calls) = CapturingProvider::new();
        let mut agent = Agent::new(Arc::new(cap));
        agent.set_memory_subsystem(vstore.clone(), embedder, clock, cfg);

        // Turn 1: plant a fact.
        let (tx1, _rx1) = tokio::sync::mpsc::channel::<StreamPiece>(8);
        agent
            .query_streaming("favorite color is blue", tx1)
            .await
            .unwrap();

        // The write path must have persisted at least one episodic memory.
        let mems = vstore.active("root").await.unwrap();
        assert!(
            !mems.is_empty(),
            "SC-22 (write path): vector store must be non-empty after turn 1 \
             (write_turn_to_memory stub → fails in RED)"
        );
        let has_fact = mems.iter().any(|m| m.text.contains("blue"));
        assert!(
            has_fact,
            "SC-22: the 'blue' fact from turn 1 must be in the vector store"
        );

        // Turn 2: the selective path must NOT send the full history.
        let (tx2, _rx2) = tokio::sync::mpsc::channel::<StreamPiece>(8);
        agent
            .query_streaming("what is favorite color", tx2)
            .await
            .unwrap();

        let all_calls = calls.lock().unwrap();
        assert!(
            all_calls.len() >= 2,
            "provider must have been called at least twice"
        );
        let turn2_msgs = &all_calls[1];
        let has_assistant = turn2_msgs.iter().any(|m| m.role == Role::Assistant);
        assert!(
            !has_assistant,
            "SC-22: turn 2 must use assembled context (no Assistant messages)"
        );
    }

    /// Ensures `on_session_open` compiles, runs without error on an empty store,
    /// and does not block session startup on maintenance failures (REQ-29).
    #[tokio::test]
    async fn test_on_session_open_is_noop_when_store_is_empty() {
        use crate::memory::clock::FixedClock;
        use crate::memory::config::MemoryConfig;
        use crate::memory::store::SqliteVectorStore;
        use crate::system::database::EncryptedSqliteMemory;

        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mem = EncryptedSqliteMemory::new(tmp.path().to_path_buf(), "pw".into()).unwrap();
        let vstore = Arc::new(SqliteVectorStore::new(mem.shared_conn(), mem.data_key()).unwrap());
        let embedder = Arc::new(FakeEmbedder {
            dim: 8,
            model: "fake".into(),
        });
        let clock = Arc::new(FixedClock::new(1_000_000));
        let cfg = MemoryConfig::default();

        let mut agent = Agent::new(Arc::new(MockProvider));
        agent.set_memory_subsystem(vstore, embedder, clock, cfg);
        // Must complete without error even when the store is empty.
        agent.on_session_open().await.unwrap();
    }

    #[tokio::test]
    async fn test_agent_history_resilience_to_key_rotation() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let db_path = tmp_dir.path().join("resilient_test.db");

        let key_a = "api_key_alpha".to_string();
        let key_b = "api_key_beta".to_string();
        let msg_text = "Secrets are safe";

        // Scope 1: Save with Key A
        let sid = {
            let memory =
                crate::system::database::EncryptedSqliteMemory::new(db_path.clone(), key_a)
                    .unwrap();
            let id = memory.create_session("test").await.unwrap();
            memory
                .add_message(&id, &Message::user(msg_text))
                .await
                .unwrap();
            id
        };

        // Scope 2: Attempt to load with Key B (Simulate rotation before fix)
        {
            let memory =
                crate::system::database::EncryptedSqliteMemory::new(db_path, key_b).unwrap();
            let result = memory.get_messages(&sid).await;

            // Expected failure: Key B cannot decrypt what Key A encrypted
            assert!(
                result.is_err(),
                "Recovery with a different key SHOULD fail currently"
            );
            assert!(result
                .unwrap_err()
                .to_string()
                .contains("Decryption failed"));
        }
    }

    // ── Task-13b tests ─────────────────────────────────────────────────────────

    /// test_distill_triggers_after_n_turns: after `distill_every_n_turns = 2`
    /// selective turns, episodic memories get `distilled_at` set.
    ///
    /// Fails in RED because the distill trigger is a noop stub.
    #[tokio::test]
    async fn test_distill_triggers_after_n_turns() {
        use crate::memory::clock::FixedClock;
        use crate::memory::config::MemoryConfig;
        use crate::memory::store::{SqliteVectorStore, VectorStore};
        use crate::system::database::EncryptedSqliteMemory;
        use tokio::sync::mpsc;

        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mem = EncryptedSqliteMemory::new(tmp.path().to_path_buf(), "pw".into()).unwrap();
        let vstore = Arc::new(SqliteVectorStore::new(mem.shared_conn(), mem.data_key()).unwrap());
        let embedder = Arc::new(FakeEmbedder {
            dim: 8,
            model: "fake".into(),
        });
        let clock = Arc::new(FixedClock::new(1_000_000));
        let cfg = MemoryConfig {
            mode: "selective".into(),
            distill_every_n_turns: 2,
            distill_enabled: true,
            ..MemoryConfig::default()
        };

        // MockProvider.send_messages returns "Summary content." — the distill
        // judge uses this to extract preferences (non-empty, so distilled_at gets set).
        let mut agent = Agent::new(Arc::new(MockProvider));
        agent.set_memory_subsystem(vstore.clone(), embedder, clock, cfg);

        // Turn 1 — trigger threshold not reached.
        let (tx1, _rx1) = mpsc::channel::<StreamPiece>(8);
        agent.query_streaming("first turn", tx1).await.unwrap();

        let mems_after_1 = vstore.active("root").await.unwrap();
        let distilled_after_1 = mems_after_1
            .iter()
            .filter(|m| m.distilled_at.is_some())
            .count();
        assert_eq!(
            distilled_after_1, 0,
            "no memories should be distilled after turn 1 (trigger fires at 2)"
        );

        // Turn 2 — distill trigger fires (turns_since_open == 2, 2 % 2 == 0).
        let (tx2, _rx2) = mpsc::channel::<StreamPiece>(8);
        agent.query_streaming("second turn", tx2).await.unwrap();

        let mems_after_2 = vstore.active("root").await.unwrap();
        let distilled_after_2 = mems_after_2
            .iter()
            .filter(|m| m.distilled_at.is_some())
            .count();
        assert!(
            distilled_after_2 > 0,
            "at least one memory must be distilled after turn 2 \
             (distill_every_n_turns = 2); got 0 distilled"
        );
    }

    /// test_promoted_preference_appears_in_assembled_context: after a preference
    /// is promoted via the distill judge (provider returns "- use rust"),
    /// the next turn's assembled context contains "use rust".
    ///
    /// Fails in RED because both the distill trigger and live profile are stubs.
    #[tokio::test]
    async fn test_promoted_preference_appears_in_assembled_context() {
        use crate::memory::clock::FixedClock;
        use crate::memory::config::MemoryConfig;
        use crate::memory::store::SqliteVectorStore;
        use crate::system::database::EncryptedSqliteMemory;
        use tokio::sync::mpsc;

        // A provider that returns "- use rust" from stream_messages (and thus
        // from send_messages via the default impl), and records every
        // stream_messages call for inspection.
        struct PrefCapProvider {
            calls: Arc<std::sync::Mutex<Vec<Vec<Message>>>>,
        }

        #[async_trait]
        impl Provider for PrefCapProvider {
            async fn stream_messages(
                &self,
                messages: &[Message],
                _tools: &[Box<dyn crate::tools::Tool>],
            ) -> Result<futures::stream::BoxStream<'static, Result<ResponseChunk>>> {
                self.calls.lock().unwrap().push(messages.to_vec());
                Ok(Box::pin(futures::stream::iter(vec![
                    Ok(ResponseChunk::TextDelta("- use rust".to_string())),
                    Ok(ResponseChunk::MessageDone(Message::assistant("- use rust"))),
                ])))
            }
        }

        let calls = Arc::new(std::sync::Mutex::new(Vec::<Vec<Message>>::new()));
        let provider = Arc::new(PrefCapProvider {
            calls: calls.clone(),
        });

        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mem = EncryptedSqliteMemory::new(tmp.path().to_path_buf(), "pw".into()).unwrap();
        let vstore = Arc::new(SqliteVectorStore::new(mem.shared_conn(), mem.data_key()).unwrap());
        let embedder = Arc::new(FakeEmbedder {
            dim: 8,
            model: "fake".into(),
        });
        let clock = Arc::new(FixedClock::new(1_000_000));
        let cfg = MemoryConfig {
            mode: "selective".into(),
            distill_every_n_turns: 1, // distill after every turn
            distill_enabled: true,
            context_budget_tokens: 4000,
            response_headroom_tokens: 0,
            safety_margin_ratio: 0.0,
            top_k: 0, // no episodic recall; only profile contributes "use rust"
            ..MemoryConfig::default()
        };

        let mut agent = Agent::new(provider);
        agent.set_memory_subsystem(vstore.clone(), embedder, clock, cfg);

        // Turn 1: provider returns "- use rust". After the turn, distill fires
        // → judge calls send_messages → "- use rust" → "use rust" promoted to profile.
        let (tx1, _rx1) = mpsc::channel::<StreamPiece>(8);
        agent.query_streaming("hello", tx1).await.unwrap();

        // ── Diagnostic: check intermediate state after turn 1 ────────────────
        {
            let calls_after_1 = calls.lock().unwrap().len();
            let mems = vstore.active("root").await.unwrap();
            let pref_count = mems
                .iter()
                .filter(|m| m.kind == crate::memory::MemoryKind::Preference)
                .count();
            assert!(
                calls_after_1 >= 2,
                "distill must fire a provider call after turn 1; got {calls_after_1} calls"
            );
            assert!(
                pref_count > 0,
                "promote_to_profile must have inserted a preference after turn 1; \
                 got {pref_count} preferences (total mems: {})",
                mems.len()
            );
        }
        // ── End diagnostic ───────────────────────────────────────────────────

        // Turn 2: render_profile must return "- use rust\n" as live_profile,
        // and assemble_selective must include it in the preamble messages.
        //
        // Capture the call index before turn 2 so we can pinpoint turn 2's main
        // stream_messages call even when the per-turn distill trigger adds further
        // calls after it (distill_every_n_turns = 1 means distill fires after
        // EVERY turn, so locked.last() would be the distill call, not the main one).
        let n_before_t2 = calls.lock().unwrap().len();
        let (tx2, _rx2) = mpsc::channel::<StreamPiece>(8);
        agent
            .query_streaming("what are my prefs", tx2)
            .await
            .unwrap();

        // calls[n_before_t2] is turn 2's main (assembled context) provider call.
        let locked = calls.lock().unwrap();
        let t2_main = locked
            .get(n_before_t2)
            .expect("turn 2's main stream_messages call must exist");
        let all_text: String = t2_main
            .iter()
            .flat_map(|m| m.content.iter())
            .filter_map(|c| {
                if let Content::Text { text } = c {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        assert!(
            all_text.contains("use rust"),
            "turn 2's assembled context must contain the promoted preference 'use rust'; \
             got:\n{all_text}"
        );
    }
}
