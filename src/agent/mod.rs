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
use crate::memory::error::MemoryError;
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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::oneshot;
use tokio::time::{timeout, Duration};
use tokio_util::sync::CancellationToken;

/// Process-monotonic counter for write-path turn-ID uniqueness (G2).
///
/// Each [`write_turn_to_memory`] call includes this counter in the ID hash so
/// two calls with identical text + role + clock-second always produce distinct
/// IDs and neither is silently dropped.
///
/// `migrate_from_messages` uses `"mig:…"` IDs independently of this counter
/// (it does not call `write_turn_to_memory`) and is unaffected.
static TURN_SEQ: AtomicU64 = AtomicU64::new(0);

const APPROVAL_TIMEOUT_SECS: u64 = 300; // 5 minutes

/// Default per-query tool-call cap (interactive and headless-normal tiers).
///
/// Single source of truth for the "normal" cap used by both [`Agent::new`] and
/// [`AgentRunConfig::default`], so the interactive path and an unconfigured run
/// share the same limit.
pub const DEFAULT_MAX_TOOL_CALLS: usize = 15;

/// Error message returned when a single query exceeds its tool-call cap.
///
/// Exposed as a `pub const` so an out-of-loop caller (the headless runner) can
/// distinguish "cap reached" (a non-error terminal state, `stop_reason =
/// max_tool_calls`) from a genuine runtime error by comparing the returned
/// error's `Display` against this exact string — a shared contract rather than
/// a brittle inline literal match.
pub const MAX_TOOL_CALLS_ERROR: &str = "Maximum tool call limit reached";

/// Observes an agent run from **outside** the tool loop, for the headless runner.
///
/// Absent ([`AgentRunConfig::observer`] is `None`) ⇒ the interactive tool loop
/// runs byte-for-byte unchanged. When present it becomes **authoritative for
/// every tool call in all tiers** (REQ-H06/H07/H09): it replaces the
/// `requires_approval()` / `approval_tx` gate, so tools that opt out of
/// interactive approval (`project_knowledge`, an auto-approve `consult`) are
/// still gated by the tier — the interactive gate alone cannot express a tier
/// because those tools never reach it.
///
/// It also captures per-call and final-turn data the [`StreamPiece`] stream does
/// not carry (tool results with wall-clock timing, and the final-turn text-block
/// count for the deterministic `stop_reason` of REQ-H23b).
///
/// All methods take `&self`; an implementor uses interior mutability. They are
/// invoked from inside `Agent::query_streaming` on the run's task.
pub trait RunObserver: Send + Sync {
    /// Decides whether `tool_name` may run. Consulted for **every** tool call,
    /// before execution, replacing the interactive approval decision.
    ///
    /// # Returns
    /// `true` to run the tool; `false` to deny it (the call is recorded with
    /// `ok = false` and the agent continues — a denial never aborts the loop).
    fn authorize(&self, tool_name: &str) -> bool;

    /// Records one resolved tool call (executed or tier-denied).
    ///
    /// # Parameters
    /// - `id` — the tool-use id, correlating this call with the assistant
    ///   `ToolUse` block that requested it (used to assemble the transcript).
    /// - `name` — the tool name.
    /// - `input` — the JSON input the tool was invoked with.
    /// - `result` — the tool result (or the denial / not-found message).
    /// - `ok` — `true` on successful execution; `false` on failure or denial.
    /// - `ms` — wall-clock execution time in milliseconds (`0` for a denied
    ///   call, which never executes).
    fn on_tool_call(
        &self,
        id: &str,
        name: &str,
        input: &serde_json::Value,
        result: &str,
        ok: bool,
        ms: u64,
    );

    /// Records the number of `TextDelta` blocks emitted in the agent's FINAL
    /// turn (REQ-H23b: `response_empty` ⇔ this count is zero). Counts raw
    /// stream blocks, not the emptiness of the concatenated text.
    fn on_final_turn(&self, text_block_count: usize);
}

/// Per-run configuration for [`Agent::query_streaming`].
///
/// [`AgentRunConfig::default`] reproduces the interactive behavior exactly
/// ([`DEFAULT_MAX_TOOL_CALLS`], repetitive-guard enabled, no observer), so the
/// TUI and the existing tests pass `AgentRunConfig::default()` and keep the
/// interactive path byte-for-byte unchanged. It has a hand-written [`Default`]
/// impl rather than a derived one because a derived `usize` default is `0`,
/// which would silently reduce the interactive cap to zero.
///
/// No field ever relaxes a **hard** barrier (`bash::is_command_allowed`, the
/// metacharacter ban, `PathGuard::validate`) — those live inside each tool and
/// are enforced regardless of this configuration.
pub struct AgentRunConfig {
    /// Maximum tool calls for this run. Interactive default:
    /// [`DEFAULT_MAX_TOOL_CALLS`]; the headless runner passes the tier-resolved
    /// cap (elevated under `--full-auto`).
    pub max_tool_calls: usize,
    /// When `true`, the 3-identical-call repetitive **soft** guard is disabled
    /// (REQ-H08, `--full-auto` only). Never disables any hard barrier.
    pub disable_repetitive_guard: bool,
    /// Optional external observer/authorizer (headless runner). `None` ⇒ the
    /// interactive `requires_approval()` / `approval_tx` path runs unchanged.
    pub observer: Option<Arc<dyn RunObserver>>,
    /// Cooperative run cancellation, passed to every `Tool::execute` (REQ-H36).
    /// The headless runner fires this when a wall-clock `--timeout` elapses so an
    /// in-flight `bash` subprocess tree is killed. [`AgentRunConfig::default`]
    /// installs a fresh, never-cancelled token, so the interactive/TUI path is
    /// unaffected.
    pub cancel: CancellationToken,
}

impl Default for AgentRunConfig {
    fn default() -> Self {
        Self {
            max_tool_calls: DEFAULT_MAX_TOOL_CALLS,
            disable_repetitive_guard: false,
            observer: None,
            cancel: CancellationToken::new(),
        }
    }
}

/// Approval request sent to the UI.
pub struct ApprovalRequest {
    pub tool_name: String,
    /// Carries tool input for the approval prompt; reserved.
    #[allow(dead_code)]
    pub input: serde_json::Value,
    pub tx: oneshot::Sender<bool>,
}

/// A piece of a streamed assistant turn forwarded to the UI.
///
/// - `Content` — answer text (persisted); forwarded to the TUI as a `StreamDelta`.
/// - `Reasoning` — thinking model chain-of-thought; shown live, never persisted (#24).
/// - `Notice` — non-content operational notice (e.g. memory fallback warning,
///   truncation advisory).  Rendered in a distinct style (dimmed/yellow, prefixed
///   `⚠ `) in the TUI so it stands out from model output without corrupting the
///   ratatui frame via stderr.
///
/// Memory assembler truncation notices and agent-loop warnings use `Notice` so they
/// are routed through the channel rather than written to raw stderr while the TUI
/// is in `EnterAlternateScreen` + raw mode.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamPiece {
    Content(String),
    Reasoning(String),
    /// A non-content operational notice (memory warning, truncation advisory).
    /// Rendered distinctly in the TUI; never persisted to conversation history.
    Notice(String),
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
    /// By design this Agent injects no `Role::System` message on any code path
    /// (neither `selective` nor `load_all`), so this field stays `String::new()`.
    /// The field and the assembler slot exist per REQ-13 for callers that do set
    /// a system prompt in the future.
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

    /// Borrows the current conversation history.
    ///
    /// Exposed read-only so an out-of-loop caller (the headless runner) can
    /// project the finished run into a normalized transcript without the Agent
    /// itself depending on the headless output types. Includes the user turn,
    /// each assistant turn (with any `ToolUse` blocks), and the `User`-role
    /// tool-result messages, in order.
    ///
    /// Consumed by the headless runner, whose production caller lands in MS2
    /// Task 6; until then the plain (non-test) binary has no live path here, so
    /// `dead_code` is allowed only for `not(test)`.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn history(&self) -> &[Message] {
        &self.history
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
        // TIMING NOTE (L317): on_session_close runs in the agent-runner spawned
        // task AFTER UiEvent::Quit is processed (post event-loop break).  The TUI
        // teardown (disable_raw_mode + LeaveAlternateScreen) happens synchronously
        // in the main task immediately after run_app returns.  Because distillation
        // involves an async LLM call (tens–hundreds of ms), the TUI is reliably
        // torn down before this error path is reached.  eprintln! is therefore safe
        // here — there is no chunk_tx available at this call site in any case.
        .map_err(|e| eprintln!("on_session_close distill: {e}"));
        Ok(())
    }

    /// Process a user message and returns the final assistant response.
    /// This method supports streaming via a sender channel.
    pub async fn query_streaming(
        &mut self,
        text: &str,
        chunk_tx: tokio::sync::mpsc::Sender<StreamPiece>,
        config: AgentRunConfig,
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

            // Assemble bounded context: system + profile + top-k recalls + turn.
            // G1: assemble BEFORE writing the user turn to the store so the current
            // turn cannot recall itself into its own context (self-recall prevention).
            // D1 (REQ-29): on a transient assembly failure fall back to the full
            // history rather than aborting the turn.  Only `BudgetUnsatisfiable`
            // (a misconfigured budget, not a transient error) propagates as Err.
            let (working_messages, assembly_notices) = match assemble_selective(
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
            {
                Ok(assembled) => (assembled.messages, assembled.notices),
                Err(MemoryError::BudgetUnsatisfiable) => {
                    return Err(anyhow::anyhow!("context assembly failed: budget unsatisfiable (system+profile exceed budget — check config)"));
                }
                Err(e) => {
                    // D1 (REQ-29): transient error (embedding/storage/crypto) —
                    // route a concise notice through chunk_tx (not eprintln! which
                    // corrupts the ratatui frame while in EnterAlternateScreen) and
                    // fall back to the full history so the turn still completes.
                    let summary = summarize_assembly_error(&e);
                    let _ = chunk_tx.send(StreamPiece::Notice(summary)).await;
                    (self.history.clone(), vec![])
                }
            };

            // D2 (D-17 gap): forward assembler truncation notices to the TUI as
            // StreamPiece::Notice so they render with the ⚠ prefix/style, distinct
            // from model Content.
            for notice in assembly_notices {
                let _ = chunk_tx.send(StreamPiece::Notice(notice)).await;
            }

            // Write user turn to vector store AFTER assembly (G1: self-recall prevention).
            // Best-effort; REQ-29: failures must not abort the turn.
            write_turn_to_memory(
                &store,
                &*embedder,
                &*clock,
                &cfg,
                &scope,
                &session_id_str,
                text,
                Role::User,
                Some(chunk_tx.clone()),
            )
            .await;

            // Run the shared tool loop. Returns (full_text, final_text):
            // - full_text: sanitized delta-accumulated text for write_turn_to_memory.
            // - final_text: text from MessageDone content, returned to the caller.
            let (full_text, final_text) = self
                .run_tool_loop(working_messages, &chunk_tx, &config)
                .await?;

            // ── Selective-specific terminal work (after the tool loop) ─────────
            // Persist the final synthesizing assistant turn to the vector store
            // (best-effort; REQ-29).
            //
            // DESIGN NOTE (H4 / REQ-01): the semantic index intentionally embeds
            // only the user turn (written above, before the tool loop) and this
            // synthesizing final assistant turn — the two meaningful conversational
            // units per REQ-01.  Raw intermediate tool-use and tool-result messages
            // are persisted to the legacy message store only (via memory.add_message
            // inside run_tool_loop) and are NOT separately embedded here.
            //
            // Rationale: the final assistant turn synthesizes all tool outcomes into
            // a coherent response, making raw intermediate messages redundant for
            // semantic retrieval.  Embedding them separately would flood the index
            // with low-value file-dump / command-output noise, inflate embedding
            // cost, and increase data egress (R-02 privacy).  The benchmark (SC-29)
            // validated recall accuracy with exactly this design.
            write_turn_to_memory(
                &store,
                &*embedder,
                &*clock,
                &cfg,
                &scope,
                &session_id_str,
                &full_text,
                Role::Assistant,
                Some(chunk_tx.clone()),
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
                if let Err(e) = distill(&*store, &judge, &*embedder, &*clock, &cfg, &scope).await {
                    // Non-fatal — route through chunk_tx so the notice renders inside
                    // the ratatui frame instead of corrupting it via raw stderr.
                    let _ = chunk_tx
                        .send(StreamPiece::Notice(format!(
                            "memory: distillation failed (non-fatal) — {}",
                            e
                        )))
                        .await;
                }
            }

            return Ok(final_text);
        }

        // ── load_all path (original, byte-identical) ──────────────────────────
        // No memory subsystem or mode != "selective" → send the full history to the
        // provider exactly as before.  This path is the sole path exercised by the
        // 188 pre-existing tests (REQ-28 / SC-32).
        //
        // `working` is seeded from `self.history` (which already includes the new
        // user message). `run_tool_loop` extends both `working` and `self.history`
        // in lock-step, so the provider sees the same growing context as before.
        let working = self.history.clone();
        let (_full_text, final_text) = self.run_tool_loop(working, &chunk_tx, &config).await?;
        Ok(final_text)
    }

    /// Inner tool-loop shared by the `selective` and `load_all` execution paths.
    ///
    /// Streams provider responses and dispatches tool use until the model produces
    /// a terminal (no-tool) turn, then returns. Both paths build their own `working`
    /// context slice and delegate here, eliminating the previous duplication while
    /// preserving every invariant of the original loops (REQ-28).
    ///
    /// # Parameters
    /// - `working` — mutable context slice sent to the provider on each iteration.
    ///   Extended in-place as assistant responses and tool-result messages are
    ///   appended. `self.history` is updated in parallel so `load_history` and the
    ///   message store stay consistent across both paths.
    /// - `chunk_tx` — streaming sender for UI pieces (shared reference).
    ///
    /// # Returns
    /// A tuple `(full_text, final_text)` where:
    /// - `full_text` — text accumulated from `TextDelta` chunks (sanitized); used
    ///   by the `selective` caller for [`write_turn_to_memory`].
    /// - `final_text` — text extracted from the terminal `MessageDone` content,
    ///   matching the pre-refactor return value in both branches.
    ///
    /// # Invariants preserved
    /// `max_tool_calls` cap, 3× repetition abort, approval/timeout/deny semantics,
    /// `sanitize_text` on every delta, `MessageDone`-missing error, TUI-closed
    /// error, and `self.history` + `memory.add_message` persistence order are all
    /// identical to the original per-path loops.
    async fn run_tool_loop(
        &mut self,
        mut working: Vec<Message>,
        chunk_tx: &tokio::sync::mpsc::Sender<StreamPiece>,
        config: &AgentRunConfig,
    ) -> Result<(String, String)> {
        let mut tool_call_count = 0;
        let mut last_normalized_tool: Option<(String, String)> = None;
        let mut repeat_count = 0;

        loop {
            let mut stream = self.provider.stream_messages(&working, &self.tools).await?;
            let mut full_text = String::new();
            // REQ-H23b: count TextDelta blocks in THIS turn (reset each iteration);
            // the terminal turn's count is the run's `final_turn_text_blocks`.
            let mut turn_text_blocks = 0usize;
            let mut last_message: Option<Message> = None;

            while let Some(chunk_result) = stream.next().await {
                match chunk_result? {
                    ResponseChunk::TextDelta(delta) => {
                        let sanitized = Self::sanitize_text(&delta);
                        turn_text_blocks += 1;
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
                    if tool_call_count > config.max_tool_calls {
                        return Err(anyhow::anyhow!(MAX_TOOL_CALLS_ERROR));
                    }

                    let normalized_input = Self::normalize_input(input, 0)?;
                    if let Some((ref last_name, ref last_norm_input)) = last_normalized_tool {
                        if last_name == name && last_norm_input == &normalized_input {
                            repeat_count += 1;
                            // REQ-H08: `--full-auto` silences this SOFT guard (only).
                            // No hard barrier is affected.
                            if repeat_count >= 3 && !config.disable_repetitive_guard {
                                return Err(anyhow::anyhow!("Repetitive tool call detected"));
                            }
                        } else {
                            repeat_count = 0;
                        }
                    }
                    last_normalized_tool = Some((name.clone(), normalized_input));

                    // Authorization decision. When a headless observer is
                    // attached it is AUTHORITATIVE for EVERY tool (REQ-H06/H07/
                    // H09) — the only place a tier can gate tools that opt out of
                    // interactive approval (`project_knowledge`, an auto-approve
                    // `consult`), which never reach the `approval_tx` gate. With
                    // no observer the original interactive path runs unchanged.
                    let approved = if let Some(observer) = config.observer.as_deref() {
                        observer.authorize(name)
                    } else {
                        // Per-tool approval policy: look up the tool's
                        // `requires_approval()` flag before deciding whether to
                        // prompt.  Safe tools (false) are auto-approved without
                        // emitting any ApprovalRequest — this eliminates one prompt
                        // per stored memory fact when the model issues multiple
                        // project_knowledge / view / ls / grep calls in sequence.
                        // Dangerous tools (bash / edit / consult) keep the default
                        // (true) and go through the existing gate below.  Unknown
                        // tool names default to true (safe-by-default); they fail
                        // as "tool not found" on execute.
                        let needs_approval = self
                            .tools
                            .iter()
                            .find(|t| t.name() == name)
                            .is_none_or(|t| t.requires_approval());

                        if needs_approval {
                            if let Some(ref tx) = self.approval_tx {
                                let (oneshot_tx, oneshot_rx) = oneshot::channel();
                                let _ = tx
                                    .send(ApprovalRequest {
                                        tool_name: name.clone(),
                                        input: input.clone(),
                                        tx: oneshot_tx,
                                    })
                                    .await;
                                match timeout(
                                    Duration::from_secs(APPROVAL_TIMEOUT_SECS),
                                    oneshot_rx,
                                )
                                .await
                                {
                                    Ok(Ok(res)) => res,
                                    _ => false,
                                }
                            } else {
                                // SECURITY: no approval channel is wired (headless
                                // / test mode, `approval_tx == None`) — there is no
                                // UI to ask, so the call proceeds even for a tool
                                // that requires approval. Pre-existing behavior: the
                                // interactive TUI ALWAYS sets `approval_tx` (see
                                // `run_tui_ext`), so bash / edit / consult are
                                // genuinely gated in production; only non-interactive
                                // callers reach this auto path.
                                true
                            }
                        } else {
                            // Auto-approve: tool opted out of the gate.
                            // Emit an announcement notice if the tool provides one,
                            // so the user knows a potentially slow operation starts.
                            if let Some(tool) = self.tools.iter().find(|t| t.name() == name) {
                                if let Some(notice) = tool.approval_notice() {
                                    // Best-effort: a failed send (TUI gone) must not
                                    // abort the turn.
                                    let _ = chunk_tx.send(StreamPiece::Notice(notice)).await;
                                }
                            }
                            true
                        }
                    };

                    if !approved {
                        // A tier denial gets a clear audit message; the interactive
                        // path keeps its original "denied or timed out" wording.
                        let denial_msg = if config.observer.is_some() {
                            format!(
                                "Tool '{name}' denied: not authorized in the current \
                                 authorization tier"
                            )
                        } else {
                            "Execution denied or timed out.".to_string()
                        };
                        if let Some(observer) = config.observer.as_deref() {
                            observer.on_tool_call(id, name, input, &denial_msg, false, 0);
                        }
                        tool_results.push(Content::ToolResult {
                            tool_use_id: id.clone(),
                            content: denial_msg,
                            is_error: true,
                        });
                        continue;
                    }

                    // Execute, measuring wall-clock time so the observer records a
                    // faithful per-call duration.
                    let started = Instant::now();
                    let (result_content, is_error) =
                        if let Some(tool) = self.tools.iter().find(|t| t.name() == name) {
                            match tool.execute(input.clone(), &config.cancel).await {
                                Ok(val) => (val.to_string(), false),
                                Err(e) => (e.to_string(), true),
                            }
                        } else {
                            (format!("Tool '{}' not found", name), true)
                        };
                    let elapsed_ms =
                        u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
                    if let Some(observer) = config.observer.as_deref() {
                        observer.on_tool_call(
                            id,
                            name,
                            input,
                            &result_content,
                            !is_error,
                            elapsed_ms,
                        );
                    }
                    tool_results.push(Content::ToolResult {
                        tool_use_id: id.clone(),
                        content: result_content,
                        is_error,
                    });
                }
            }

            if requested_tool {
                let tool_res_msg = Message {
                    role: Role::User,
                    content: tool_results,
                };
                self.history.push(tool_res_msg.clone());
                working.push(tool_res_msg.clone());
                // Persist the tool-result message to the legacy message store so
                // load_history and the full-history (load_all) path stay complete.
                // This message is intentionally NOT written to the vector store —
                // only the user turn and the final synthesizing assistant turn are
                // embedded in selective mode (H4 / REQ-01 / SC-29).  See the
                // write_turn_to_memory call after run_tool_loop in query_streaming.
                if let (Some(memory), Some(sid)) = (&self.memory, &self.session_id) {
                    memory.add_message(sid, &tool_res_msg).await?;
                }
            } else {
                // Terminal (no-tool) turn: report its TextDelta block count for
                // the deterministic REQ-H23b `response_empty` signal.
                if let Some(observer) = config.observer.as_deref() {
                    observer.on_final_turn(turn_text_blocks);
                }
                // Extract final text from MessageDone content — matches the
                // pre-refactor return path used by both the selective and load_all
                // branches (distinct from `full_text` which is delta-accumulated).
                let final_text = response
                    .content
                    .iter()
                    .rev()
                    .find_map(|c| {
                        if let Content::Text { text } = c {
                            Some(text.clone())
                        } else {
                            None
                        }
                    })
                    .unwrap_or_default();
                return Ok((full_text, final_text));
            }
        }
    }
}

/// Produces a concise, single-line summary of a `MemoryError` suitable for a TUI
/// `StreamPiece::Notice` (≤80 chars, no raw JSON blobs, TUI-safe UTF-8).
///
/// This keeps the notice readable on screen without dumping e.g. a full 404 JSON
/// body into the conversation pane.
fn summarize_assembly_error(e: &MemoryError) -> String {
    let raw = e.to_string();
    // Truncate long messages (e.g. full HTTP body) at 80 chars and append "…".
    const NOTICE_MAX: usize = 80;
    if raw.len() <= NOTICE_MAX {
        format!(
            "memory: context assembly failed — using full history ({})",
            raw
        )
    } else {
        let truncated: String = raw.chars().take(NOTICE_MAX).collect();
        format!(
            "memory: context assembly failed — using full history ({}…)",
            truncated
        )
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
/// - `notice_tx` — optional sender for routing non-fatal write-error notices to the
///   TUI as `StreamPiece::Notice` (instead of raw stderr that corrupts the ratatui
///   frame).  `None` is accepted so call sites outside the streaming context (tests,
///   future callers) are not forced to supply a channel.
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
    notice_tx: Option<tokio::sync::mpsc::Sender<StreamPiece>>,
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

    // G2 (live write path): include a monotonic counter so two calls with the
    // SAME text + role + clock-second always produce distinct IDs and neither is
    // silently dropped by a UNIQUE-constraint failure.
    //
    // `migrate_from_messages` uses its own `"mig:…"` content-hash scheme for
    // idempotent replay (CP2-M) and does NOT call `write_turn_to_memory`, so the
    // migration dedup contract is unchanged.
    let seq = TURN_SEQ.fetch_add(1, Ordering::Relaxed);
    let id = format!(
        "turn:{:x}",
        Sha256::digest(format!("{now}:{role:?}:{seq}:{text}").as_bytes())
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

    // F6b / G2 — swallow write errors intentionally (REQ-29 degradation).
    //
    // `TURN_SEQ` (G2) ensures each live write has a unique ID, so the UNIQUE
    // constraint is no longer triggered by duplicate content. Genuine write errors
    // (disk full, crypto failure) are non-fatal to preserve REQ-29 ("never abort a
    // turn because of a memory failure").
    //
    // When a notice_tx is supplied (in-TUI streaming context), route the warning
    // through StreamPiece::Notice so it renders inside the ratatui frame instead of
    // being written to raw stderr which corrupts the EnterAlternateScreen display.
    // When no channel is available (test contexts, post-teardown callers), fall back
    // to eprintln! for operational visibility.
    if let Err(e) = store.insert(&m).await {
        if let Some(tx) = notice_tx {
            let _ = tx
                .send(StreamPiece::Notice(
                    "memory: turn not persisted (non-fatal)".to_string(),
                ))
                .await;
        } else {
            eprintln!("WARN [magi-rs]: memory insert failed (non-fatal): {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolResult;
    use anyhow::Result;
    use async_trait::async_trait;
    use futures::stream::{self, BoxStream};
    use serde_json::{json, Value};

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
                    zeroize::Zeroizing::new(password.clone()),
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
                crate::system::database::EncryptedSqliteMemory::new(
                    db_path,
                    zeroize::Zeroizing::new(password),
                )
                .unwrap(),
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

        let final_text = agent
            .query_streaming("Hi", chunk_tx, AgentRunConfig::default())
            .await
            .unwrap();
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
        let out = agent
            .query_streaming("hi", tx, AgentRunConfig::default())
            .await
            .unwrap();
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
        let mem = EncryptedSqliteMemory::new(
            tmp.path().to_path_buf(),
            zeroize::Zeroizing::new("pw".to_string()),
        )
        .unwrap();
        let vstore =
            Arc::new(SqliteVectorStore::new(mem.shared_conn(), mem.data_key().unwrap()).unwrap());
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
        let result = agent
            .query_streaming("hello", tx, AgentRunConfig::default())
            .await;
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
        let mem = EncryptedSqliteMemory::new(
            tmp.path().to_path_buf(),
            zeroize::Zeroizing::new("pw".to_string()),
        )
        .unwrap();
        let vstore =
            Arc::new(SqliteVectorStore::new(mem.shared_conn(), mem.data_key().unwrap()).unwrap());
        let embedder = Arc::new(FakeEmbedder {
            dim: 16,
            model: "fake".into(),
        });
        let clock = Arc::new(FixedClock::new(1_000_000));
        // Very tight budget so that a long turn triggers a truncation notice.
        let cfg = MemoryConfig {
            mode: "selective".into(),
            context_budget_tokens: 20, // 20 tokens ≈ 70 chars
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
        agent
            .query_streaming(&long_input, tx, AgentRunConfig::default())
            .await
            .unwrap();

        // Collect all streamed pieces.
        let mut pieces = Vec::new();
        while let Ok(p) = rx.try_recv() {
            pieces.push(p);
        }

        // At least one piece must be a `StreamPiece::Notice` (D2 routing).
        let notice_found = pieces.iter().any(|p| matches!(p, StreamPiece::Notice(_)));
        assert!(
            notice_found,
            "D2: a truncation notice must be forwarded to chunk_tx as StreamPiece::Notice when \
             the turn exceeds the budget; got pieces: {pieces:?}"
        );
    }

    /// D1-notice routing: when `assemble_selective` encounters a transient error (e.g.
    /// embedder auth failure), `query_streaming` must emit a `StreamPiece::Notice`
    /// through `chunk_tx` (NOT write to stderr) AND still complete the turn via the
    /// load_all fallback path (REQ-29 / SC-30).
    #[tokio::test]
    async fn test_selective_d1_fallback_routes_notice_not_eprintln() {
        use crate::memory::clock::FixedClock;
        use crate::memory::config::MemoryConfig;
        use crate::memory::store::SqliteVectorStore;
        use crate::system::database::EncryptedSqliteMemory;
        use tokio::sync::mpsc;

        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mem = EncryptedSqliteMemory::new(
            tmp.path().to_path_buf(),
            zeroize::Zeroizing::new("pw".to_string()),
        )
        .unwrap();
        let vstore =
            Arc::new(SqliteVectorStore::new(mem.shared_conn(), mem.data_key().unwrap()).unwrap());
        let clock = Arc::new(FixedClock::new(1_000_000));
        let cfg = MemoryConfig {
            mode: "selective".into(),
            ..MemoryConfig::default()
        };

        let (cap, _calls) = CapturingProvider::new();
        let mut agent = Agent::new(Arc::new(cap));
        agent.set_memory_subsystem(vstore, Arc::new(ErrorEmbedder), clock, cfg);

        let (tx, mut rx) = mpsc::channel::<StreamPiece>(16);
        // Must succeed (fallback) — REQ-29.
        let result = agent
            .query_streaming("hello", tx, AgentRunConfig::default())
            .await;
        assert!(
            result.is_ok(),
            "D1-notice: selective mode with embedder error must fall back, not Err: {result:?}"
        );

        let mut pieces = Vec::new();
        while let Ok(p) = rx.try_recv() {
            pieces.push(p);
        }

        // A Notice piece must have been emitted (not an eprintln! to stderr).
        let got_notice = pieces.iter().any(|p| matches!(p, StreamPiece::Notice(_)));
        assert!(
            got_notice,
            "D1-notice: a StreamPiece::Notice must be emitted through chunk_tx on \
             assembly failure (not eprintln!); got pieces: {pieces:?}"
        );

        // The turn must also have produced a Content piece (provider still responded).
        let got_content = pieces
            .iter()
            .any(|p| matches!(p, StreamPiece::Content(s) if !s.is_empty()));
        assert!(
            got_content,
            "D1-notice: provider response (Content piece) must arrive after fallback; \
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
        agent
            .query_streaming("hello", tx1, AgentRunConfig::default())
            .await
            .unwrap();
        // After 1 turn: User("hello") + Asst("Summary content.") = 2 messages.
        assert_eq!(
            agent.history.len(),
            2,
            "load_all: history must have 2 messages after 1 turn"
        );

        let (tx2, _rx2) = tokio::sync::mpsc::channel::<StreamPiece>(8);
        agent
            .query_streaming("world", tx2, AgentRunConfig::default())
            .await
            .unwrap();
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
        let mem = EncryptedSqliteMemory::new(
            tmp.path().to_path_buf(),
            zeroize::Zeroizing::new("pw".to_string()),
        )
        .unwrap();
        let vstore =
            Arc::new(SqliteVectorStore::new(mem.shared_conn(), mem.data_key().unwrap()).unwrap());
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
        agent
            .query_streaming("first query", tx1, AgentRunConfig::default())
            .await
            .unwrap();

        let (tx2, _rx2) = tokio::sync::mpsc::channel::<StreamPiece>(8);
        agent
            .query_streaming("second query", tx2, AgentRunConfig::default())
            .await
            .unwrap();

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
        let mem = EncryptedSqliteMemory::new(
            tmp.path().to_path_buf(),
            zeroize::Zeroizing::new("pw".to_string()),
        )
        .unwrap();
        let vstore =
            Arc::new(SqliteVectorStore::new(mem.shared_conn(), mem.data_key().unwrap()).unwrap());
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
            .query_streaming("favorite color is blue", tx1, AgentRunConfig::default())
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
            .query_streaming("what is favorite color", tx2, AgentRunConfig::default())
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
        let mem = EncryptedSqliteMemory::new(
            tmp.path().to_path_buf(),
            zeroize::Zeroizing::new("pw".to_string()),
        )
        .unwrap();
        let vstore =
            Arc::new(SqliteVectorStore::new(mem.shared_conn(), mem.data_key().unwrap()).unwrap());
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
            let memory = crate::system::database::EncryptedSqliteMemory::new(
                db_path.clone(),
                zeroize::Zeroizing::new(key_a),
            )
            .unwrap();
            let id = memory.create_session("test").await.unwrap();
            memory
                .add_message(&id, &Message::user(msg_text))
                .await
                .unwrap();
            id
        };

        // Scope 2: Attempt to open with Key B (a different DB master).
        // Under the DEK/KEK envelope model (REQ-V35) a wrong master now fails at
        // OPEN — the KEK cannot unwrap the DEK, so `new` returns Err — rather
        // than opening and failing later on a per-record decrypt. Crucially, the
        // failed open must NEVER wipe or corrupt the existing data.
        {
            let result = crate::system::database::EncryptedSqliteMemory::new(
                db_path.clone(),
                zeroize::Zeroizing::new(key_b),
            );
            assert!(
                result.is_err(),
                "opening with a different DB master must fail (wrong KEK cannot unwrap the DEK)"
            );
        }

        // Scope 3: Re-open with the correct Key A — the data survives the failed
        // wrong-key attempt intact (never-wipe guarantee).
        {
            let key_a_again = "api_key_alpha".to_string();
            let memory = crate::system::database::EncryptedSqliteMemory::new(
                db_path,
                zeroize::Zeroizing::new(key_a_again),
            )
            .unwrap();
            let msgs = memory.get_messages(&sid).await.unwrap();
            assert_eq!(
                msgs,
                vec![Message::user(msg_text)],
                "the correct master still recovers the data after a failed wrong-key open"
            );
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
        let mem = EncryptedSqliteMemory::new(
            tmp.path().to_path_buf(),
            zeroize::Zeroizing::new("pw".to_string()),
        )
        .unwrap();
        let vstore =
            Arc::new(SqliteVectorStore::new(mem.shared_conn(), mem.data_key().unwrap()).unwrap());
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
        agent
            .query_streaming("first turn", tx1, AgentRunConfig::default())
            .await
            .unwrap();

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
        agent
            .query_streaming("second turn", tx2, AgentRunConfig::default())
            .await
            .unwrap();

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
    #[ignore = "placeholder: wired in Task 13b"]
    async fn test_promoted_preference_placeholder() {}

    // ── G1 / G2 tests ─────────────────────────────────────────────────────────

    /// G1: in selective mode, the user's current turn must NOT be recalled into
    /// its own assembler context (self-recall prevention).
    ///
    /// With the bug: `write_turn_to_memory` runs BEFORE `assemble_selective`,
    /// so the current turn is in the store during recall and can appear in the
    /// preamble. After the fix: the write happens AFTER assembly — the store is
    /// empty on the very first turn, so no self-recall is possible.
    #[tokio::test]
    async fn test_selective_mode_current_turn_not_self_recalled() {
        use crate::memory::clock::FixedClock;
        use crate::memory::config::MemoryConfig;
        use crate::memory::store::SqliteVectorStore;
        use crate::system::database::EncryptedSqliteMemory;

        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mem = EncryptedSqliteMemory::new(
            tmp.path().to_path_buf(),
            zeroize::Zeroizing::new("pw".to_string()),
        )
        .unwrap();
        let vstore =
            Arc::new(SqliteVectorStore::new(mem.shared_conn(), mem.data_key().unwrap()).unwrap());
        let embedder = Arc::new(FakeEmbedder {
            dim: 32,
            model: "fake".into(),
        });
        let clock = Arc::new(FixedClock::new(1_000_000));
        // Budget large enough to include recalls; top_k > 0 so recall runs.
        let cfg = MemoryConfig {
            mode: "selective".into(),
            context_budget_tokens: 4000,
            response_headroom_tokens: 0,
            safety_margin_ratio: 0.0,
            top_k: 5,
            ..MemoryConfig::default()
        };

        let (cap, calls) = CapturingProvider::new();
        let mut agent = Agent::new(Arc::new(cap));
        agent.set_memory_subsystem(vstore, embedder, clock, cfg);

        // Distinctive text that would self-recall if written before assembly.
        let distinctive = "alpha bravo charlie unique self recall prevention test";
        let (tx, _rx) = tokio::sync::mpsc::channel::<StreamPiece>(8);
        agent
            .query_streaming(distinctive, tx, AgentRunConfig::default())
            .await
            .unwrap();

        let locked = calls.lock().unwrap();
        assert!(!locked.is_empty(), "G1: provider must have been called");

        // Assembled context: at most [User(preamble), User(current_turn)].
        // If ≥2 messages, the preamble (all but the last) must NOT contain the
        // current turn text (which only belongs in the last message slot).
        let turn1 = &locked[0];
        if turn1.len() >= 2 {
            let preamble_text: String = turn1[..turn1.len() - 1]
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
                .join(" ");
            assert!(
                !preamble_text.contains("alpha bravo charlie unique"),
                "G1: current turn text must not appear in the recall preamble \
                 on its own first turn (self-recall); preamble was:\n{preamble_text}"
            );
        }
        // Single message: only the current turn → no preamble → correct by construction.
    }

    /// G2: two `write_turn_to_memory` calls with identical text + role in the
    /// same FixedClock second must produce TWO distinct stored records.
    ///
    /// Before fix: ID = `Sha256("{now}:{role:?}:{text}")` — same for both calls
    /// → second INSERT fails UNIQUE constraint (silently swallowed) → 1 record.
    /// After fix: ID includes a monotonic counter → always unique → 2 records.
    #[tokio::test]
    async fn test_write_turn_produces_distinct_ids_for_same_text_in_same_second() {
        use crate::memory::clock::FixedClock;
        use crate::memory::config::MemoryConfig;
        use crate::memory::store::{SqliteVectorStore, VectorStore};
        use crate::system::database::EncryptedSqliteMemory;

        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mem = EncryptedSqliteMemory::new(
            tmp.path().to_path_buf(),
            zeroize::Zeroizing::new("pw".to_string()),
        )
        .unwrap();
        let vstore =
            Arc::new(SqliteVectorStore::new(mem.shared_conn(), mem.data_key().unwrap()).unwrap());
        let embedder = FakeEmbedder {
            dim: 8,
            model: "fake".into(),
        };
        let clock = FixedClock::new(1_000_000); // fixed — same second for both calls
        let cfg = MemoryConfig::default();

        // Both calls: same text, same role, same clock-second.
        write_turn_to_memory(
            &vstore,
            &embedder,
            &clock,
            &cfg,
            "root",
            "session1",
            "same text same second",
            Role::User,
            None,
        )
        .await;
        write_turn_to_memory(
            &vstore,
            &embedder,
            &clock,
            &cfg,
            "root",
            "session1",
            "same text same second",
            Role::User,
            None,
        )
        .await;

        let all = vstore.active("root").await.unwrap();
        assert_eq!(
            all.len(),
            2,
            "G2: two identical write_turn_to_memory calls in the same FixedClock second \
             must produce 2 distinct stored records (not 1 deduped record)"
        );
    }

    // ── Per-tool approval-policy tests ─────────────────────────────────────────
    //
    // RED: fails to compile because `fn requires_approval` is not a member of
    // the `Tool` trait yet.  After the GREEN commit both tests must pass.

    /// Provider that emits one `ToolUse` on its first `stream_messages` call,
    /// then returns a plain text response on every subsequent call (after the
    /// agent feeds back the tool result).
    struct SingleToolCallProvider {
        tool_name: String,
        call_count: Arc<std::sync::Mutex<u32>>,
    }

    impl SingleToolCallProvider {
        fn new(tool_name: &str) -> (Self, Arc<std::sync::Mutex<u32>>) {
            let count = Arc::new(std::sync::Mutex::new(0u32));
            (
                Self {
                    tool_name: tool_name.to_string(),
                    call_count: count.clone(),
                },
                count,
            )
        }
    }

    #[async_trait]
    impl Provider for SingleToolCallProvider {
        async fn stream_messages(
            &self,
            _messages: &[Message],
            _tools: &[Box<dyn Tool>],
        ) -> Result<BoxStream<'static, Result<ResponseChunk>>> {
            let mut n = self.call_count.lock().unwrap();
            *n += 1;
            let call = *n;
            drop(n);

            if call == 1 {
                // First call: ask the agent to call the registered tool.
                let msg = Message {
                    role: Role::Assistant,
                    content: vec![Content::ToolUse {
                        id: "approval-policy-test-1".to_string(),
                        name: self.tool_name.clone(),
                        input: serde_json::json!({}),
                    }],
                };
                Ok(Box::pin(stream::iter(vec![Ok(
                    ResponseChunk::MessageDone(msg),
                )])))
            } else {
                // Subsequent calls: return the final text response.
                Ok(Box::pin(stream::iter(vec![
                    Ok(ResponseChunk::TextDelta("done".to_string())),
                    Ok(ResponseChunk::MessageDone(Message::assistant("done"))),
                ])))
            }
        }
    }

    /// Tool that records whether its `execute` method was called.
    ///
    /// `requires_approval()` returns `false` when `safe = true` and `true`
    /// (the default) when `safe = false`. If `notice` is `Some`, the tool
    /// also emits an `approval_notice` when auto-approved.
    struct TrackingTool {
        name_str: String,
        executed: Arc<std::sync::Mutex<bool>>,
        safe: bool,
        notice: Option<String>,
    }

    impl TrackingTool {
        fn new(name: &str, safe: bool) -> (Self, Arc<std::sync::Mutex<bool>>) {
            let executed = Arc::new(std::sync::Mutex::new(false));
            (
                Self {
                    name_str: name.to_string(),
                    executed: executed.clone(),
                    safe,
                    notice: None,
                },
                executed,
            )
        }

        /// Constructor for a safe (auto-approved) tool that also returns a notice.
        fn with_notice(name: &str, notice: &str) -> (Self, Arc<std::sync::Mutex<bool>>) {
            let executed = Arc::new(std::sync::Mutex::new(false));
            (
                Self {
                    name_str: name.to_string(),
                    executed: executed.clone(),
                    safe: true, // auto-approved — notice only fires on auto-approve
                    notice: Some(notice.to_string()),
                },
                executed,
            )
        }
    }

    #[async_trait]
    impl Tool for TrackingTool {
        fn name(&self) -> &str {
            &self.name_str
        }
        fn description(&self) -> &str {
            "tracking tool for approval-policy tests"
        }
        fn input_schema(&self) -> Value {
            json!({"type": "object", "properties": {}})
        }
        async fn execute(&self, _args: Value, _cancel: &CancellationToken) -> ToolResult<Value> {
            *self.executed.lock().unwrap() = true;
            Ok(json!({"executed": true}))
        }
        /// Safe tools opt out of the approval gate; dangerous tools keep the default.
        fn requires_approval(&self) -> bool {
            !self.safe
        }
        fn approval_notice(&self) -> Option<String> {
            self.notice.clone()
        }
    }

    /// Adversarial: a safe tool (`requires_approval() = false`) is auto-approved
    /// and executes even when a blanket-denier `approval_tx` is connected.
    /// No `ApprovalRequest` must be emitted for the safe tool.
    ///
    /// Fails in RED: `fn requires_approval` is not a member of trait `Tool`.
    #[tokio::test]
    async fn test_safe_tool_auto_approved_despite_blanket_denier() {
        use tokio::sync::mpsc;

        let (approval_tx, mut approval_rx) = mpsc::channel::<ApprovalRequest>(8);

        let (tool, executed) = TrackingTool::new("safe_op", true /* safe */);
        let (provider, _count) = SingleToolCallProvider::new("safe_op");
        let mut agent = Agent::new(Arc::new(provider));
        agent.register_tool(Box::new(tool));
        agent.set_approval_channel(approval_tx);

        let (chunk_tx, _rx) = mpsc::channel::<StreamPiece>(8);
        agent
            .query_streaming("do the safe thing", chunk_tx, AgentRunConfig::default())
            .await
            .unwrap();

        // Safe tool must have executed (auto-approved despite denier).
        assert!(
            *executed.lock().unwrap(),
            "safe tool (requires_approval=false) must execute \
             even when a blanket-denier approval_tx is connected"
        );

        // No ApprovalRequest must have been emitted for a safe tool.
        assert!(
            approval_rx.try_recv().is_err(),
            "safe tool must NOT emit an ApprovalRequest (no-prompt guarantee)"
        );
    }

    /// Adversarial: a dangerous tool (`requires_approval() = true`) is denied
    /// and does NOT execute when a blanket-denier `approval_tx` is connected.
    ///
    /// Fails in RED: `fn requires_approval` is not a member of trait `Tool`.
    #[tokio::test]
    async fn test_dangerous_tool_denied_by_blanket_denier() {
        use tokio::sync::mpsc;

        let (approval_tx, mut approval_rx) = mpsc::channel::<ApprovalRequest>(8);
        // Spawn a denier: immediately denies every ApprovalRequest it receives.
        tokio::spawn(async move {
            while let Some(req) = approval_rx.recv().await {
                let _ = req.tx.send(false);
            }
        });

        let (tool, executed) = TrackingTool::new("dangerous_op", false /* NOT safe */);
        let (provider, _count) = SingleToolCallProvider::new("dangerous_op");
        let mut agent = Agent::new(Arc::new(provider));
        agent.register_tool(Box::new(tool));
        agent.set_approval_channel(approval_tx);

        let (chunk_tx, _rx) = mpsc::channel::<StreamPiece>(8);
        agent
            .query_streaming(
                "do the dangerous thing",
                chunk_tx,
                AgentRunConfig::default(),
            )
            .await
            .unwrap();

        // Dangerous tool must NOT have executed (denied by approval_tx denier).
        assert!(
            !*executed.lock().unwrap(),
            "dangerous tool (requires_approval=true) must NOT execute \
             when the approval denier rejects the request"
        );
    }

    // ── approval_notice() agent-level tests ─────────────────────────────────────
    //
    // Verify that `run_tool_loop` emits a `StreamPiece::Notice` via `chunk_tx`
    // BEFORE executing a tool that is auto-approved AND returns Some from
    // `approval_notice()`.

    /// Auto-approved tool with `approval_notice = Some(msg)` MUST emit a
    /// `StreamPiece::Notice` in the chunk stream BEFORE the tool executes.
    ///
    /// RED: fails until the notice-emission block is added to `run_tool_loop`.
    #[tokio::test]
    async fn test_auto_approved_tool_with_notice_emits_stream_notice() {
        use tokio::sync::mpsc;

        const NOTICE_TEXT: &str = "auto-launch notice: consensus in progress";

        let (tool, _executed) = TrackingTool::with_notice("notice_op", NOTICE_TEXT);
        let (provider, _count) = SingleToolCallProvider::new("notice_op");
        let mut agent = Agent::new(Arc::new(provider));
        agent.register_tool(Box::new(tool));
        // No approval_tx → auto-approve path for all tools (or it would time out).

        let (chunk_tx, mut chunk_rx) = mpsc::channel::<StreamPiece>(32);
        agent
            .query_streaming("do the notice thing", chunk_tx, AgentRunConfig::default())
            .await
            .unwrap();

        // Collect all stream pieces.
        let mut pieces = Vec::new();
        while let Ok(p) = chunk_rx.try_recv() {
            pieces.push(p);
        }

        // There must be at least one Notice with the expected text.
        let has_notice = pieces.iter().any(|p| {
            if let StreamPiece::Notice(msg) = p {
                msg.contains(NOTICE_TEXT)
            } else {
                false
            }
        });
        assert!(
            has_notice,
            "auto-approved tool with approval_notice must emit StreamPiece::Notice \
             with the notice text before execution; pieces: {pieces:?}"
        );
    }

    /// Auto-approved tool with `approval_notice = None` must NOT emit any Notice.
    ///
    /// RED: the existing behavior for silent auto-approved tools must be unchanged —
    /// no spurious Notice pieces when `approval_notice()` returns `None`.
    #[tokio::test]
    async fn test_auto_approved_tool_without_notice_emits_no_stream_notice() {
        use tokio::sync::mpsc;

        let (tool, _executed) = TrackingTool::new("silent_op", true /* safe, no notice */);
        let (provider, _count) = SingleToolCallProvider::new("silent_op");
        let mut agent = Agent::new(Arc::new(provider));
        agent.register_tool(Box::new(tool));

        let (chunk_tx, mut chunk_rx) = mpsc::channel::<StreamPiece>(32);
        agent
            .query_streaming("do the silent thing", chunk_tx, AgentRunConfig::default())
            .await
            .unwrap();

        let mut pieces = Vec::new();
        while let Ok(p) = chunk_rx.try_recv() {
            pieces.push(p);
        }

        let has_notice = pieces.iter().any(|p| matches!(p, StreamPiece::Notice(_)));
        assert!(
            !has_notice,
            "auto-approved tool with approval_notice=None must NOT emit StreamPiece::Notice; \
             pieces: {pieces:?}"
        );
    }

    /// Gate-required tool (requires_approval=true) with a (hypothetical) approval_notice
    /// MUST NOT emit a Notice in the stream — the notice is only for auto-approved launches.
    ///
    /// The test wires a blanket-approver so the tool executes (no deny), but the
    /// notice path is gated on `!needs_approval`, so no Notice must appear.
    #[tokio::test]
    async fn test_gated_tool_approval_notice_not_emitted_even_when_approved() {
        use tokio::sync::mpsc;

        // A dangerous tool (requires_approval=true) that also has a notice — the
        // notice must NOT fire since the tool goes through the approval gate.
        struct DangerousNoticeeTool;
        #[async_trait]
        impl Tool for DangerousNoticeeTool {
            fn name(&self) -> &str {
                "dangerous_noticeee"
            }
            fn description(&self) -> &str {
                "dangerous tool that also has a notice field"
            }
            fn input_schema(&self) -> Value {
                json!({"type": "object", "properties": {}})
            }
            async fn execute(
                &self,
                _args: Value,
                _cancel: &CancellationToken,
            ) -> ToolResult<Value> {
                Ok(json!({"executed": true}))
            }
            fn requires_approval(&self) -> bool {
                true // gated — goes through the approval prompt
            }
            fn approval_notice(&self) -> Option<String> {
                Some("this should NOT appear — tool is gated".into())
            }
        }

        let (approval_tx, mut approval_rx) = mpsc::channel::<ApprovalRequest>(8);
        // Approve every request so the tool actually runs.
        tokio::spawn(async move {
            while let Some(req) = approval_rx.recv().await {
                let _ = req.tx.send(true);
            }
        });

        let (provider, _count) = SingleToolCallProvider::new("dangerous_noticeee");
        let mut agent = Agent::new(Arc::new(provider));
        agent.register_tool(Box::new(DangerousNoticeeTool));
        agent.set_approval_channel(approval_tx);

        let (chunk_tx, mut chunk_rx) = mpsc::channel::<StreamPiece>(32);
        agent
            .query_streaming(
                "do the dangerous noticeee thing",
                chunk_tx,
                AgentRunConfig::default(),
            )
            .await
            .unwrap();

        let mut pieces = Vec::new();
        while let Ok(p) = chunk_rx.try_recv() {
            pieces.push(p);
        }

        let has_notice = pieces.iter().any(|p| matches!(p, StreamPiece::Notice(_)));
        assert!(
            !has_notice,
            "gated tool (requires_approval=true) must NOT emit StreamPiece::Notice \
             even if it has an approval_notice — notice is only for auto-approved launches; \
             pieces: {pieces:?}"
        );
    }

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
        let mem = EncryptedSqliteMemory::new(
            tmp.path().to_path_buf(),
            zeroize::Zeroizing::new("pw".to_string()),
        )
        .unwrap();
        let vstore =
            Arc::new(SqliteVectorStore::new(mem.shared_conn(), mem.data_key().unwrap()).unwrap());
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
        agent
            .query_streaming("hello", tx1, AgentRunConfig::default())
            .await
            .unwrap();

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
            .query_streaming("what are my prefs", tx2, AgentRunConfig::default())
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
