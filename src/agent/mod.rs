//! The Agent orchestrator coordinates between the Provider and the Tools.

pub mod magi_adapter;
pub mod magi_wiring;
pub mod messages;
pub mod provider;

use crate::agent::messages::{Content, Message, Role};
use crate::agent::provider::{Provider, ResponseChunk};
use crate::memory::clock::Clock;
use crate::memory::config::MemoryConfig;
use crate::memory::decay::purge_expired_archives;
use crate::memory::embedding::EmbeddingProvider;
use crate::memory::retrieval::reembed_pending;
use crate::memory::store::SqliteVectorStore;
use crate::system::database::MemoryStore;
use crate::tools::Tool;
use anyhow::Result;
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
// Narrow allow: all fields are read in `on_session_open` (GREEN: also in
// `query_streaming`). The struct itself is constructed in `set_memory_subsystem`.
#[allow(dead_code)]
struct MemorySubsystem {
    store: Arc<SqliteVectorStore>,
    embedder: Arc<dyn EmbeddingProvider>,
    clock: Arc<dyn Clock>,
    cfg: MemoryConfig,
    /// Scope tag for memory isolation; defaults to `"root"`. Multi-scope is
    /// activated in L3 (Agent Society — AS-REQ-09).
    scope: String,
    /// Serialized preference profile always injected into the assembled context.
    /// Empty until Task 13 (distiller) wires content.
    profile: String,
    /// System-prompt text for token-budget accounting; empty when none is set.
    system: String,
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
    // Narrow allow: read in `query_streaming` selective branch (GREEN phase).
    #[allow(dead_code)]
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
    // Narrow allow: called from `main.rs` in GREEN phase; dead in RED binary target.
    #[allow(dead_code)]
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
            profile: String::new(),
            system: String::new(),
        });
    }

    /// Runs best-effort maintenance on session open: re-embeds pending memories
    /// (stored without a vector due to a transient embedding failure) and purges
    /// expired archives.
    ///
    /// Errors are intentionally swallowed — maintenance failures must never
    /// block session startup (REQ-29).
    // Narrow allow: called from `main.rs` in GREEN phase; dead in RED binary target.
    #[allow(dead_code)]
    pub async fn on_session_open(&self) -> Result<()> {
        if let Some(s) = &self.memory_subsystem {
            let _ = reembed_pending(&*s.store, &*s.embedder, &s.cfg, &s.scope).await;
            let _ = purge_expired_archives(&*s.store, &*s.clock, &s.cfg).await;
        }
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

        // Persist user message
        if let (Some(memory), Some(sid)) = (&self.memory, &self.session_id) {
            memory.add_message(sid, &user_msg).await?;
        }

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

            use futures::StreamExt;
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
// Narrow allow: called from `query_streaming` selective branch (GREEN phase).
// The stub body is intentionally a no-op in RED so the TDD tests fail correctly.
#[allow(dead_code, clippy::too_many_arguments)]
async fn write_turn_to_memory(
    _store: &SqliteVectorStore,
    _embedder: &dyn EmbeddingProvider,
    _clock: &dyn Clock,
    _cfg: &MemoryConfig,
    _scope: &str,
    _session_id: &str,
    _text: &str,
    _role: Role,
) {
    // RED PHASE STUB — implemented in GREEN (Task 12).
    // Intentionally a no-op so the three new tests fail for the right reason:
    //   - test_selective_mode_sends_assembled_context_not_full_history: the
    //     selective branch doesn't exist yet, so the provider receives the full
    //     history (contains an Assistant message) → assertion fails.
    //   - test_fact_written_in_one_turn_is_recalled_in_a_later_turn: nothing is
    //     written to the store → vstore.active("root") is empty → assertion fails.
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
}
