//! The Agent orchestrator coordinates between the Provider and the Tools.

pub mod magi_adapter;
pub mod magi_wiring;
pub mod messages;
pub mod mode_classifier;
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
use magi_core::schema::Mode;
use magi_rs::magi::gate::{evaluate, GateTelemetry, GateThresholds, GateVerdict, NoGateTelemetry};
use magi_rs::magi::mode::{agent_chosen_mode, input_for_dispatch, resolve_mode_guarded, ModeConfig};
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
///
/// **Stability contract:** this exact string is the wire format between this
/// module (producer, both `Err(anyhow::anyhow!(MAX_TOOL_CALLS_ERROR))` sites
/// below) and `crate::headless_runner::run_query` (consumer, its
/// `message == MAX_TOOL_CALLS_ERROR` comparison). Changing the string value is
/// a breaking change to that contract. It is pinned end-to-end by
/// `headless_runner::tests::test_runner_max_tool_calls_when_cap_exhausted`
/// (cap exhausted ⇒ `StopReason::MaxToolCalls`, no error payload) and
/// `test_runner_max_tool_calls_priority_over_denied`.
pub const MAX_TOOL_CALLS_ERROR: &str = "Maximum tool call limit reached";

/// Registered name of the multi-perspective MAGI tool, mirroring
/// `ConsultTool::name()` (`src/tools/consult.rs`) — the single string the
/// forced-consult injection and its at-most-once guard compare against
/// (REQ-H22). Kept independent of the tool's own literal (no cross-crate-layer
/// import from `agent` into `tools`' private constants), same precedent as
/// `headless_runner::CONSULT_TOOL`.
const CONSULT_TOOL_NAME: &str = "consult";

/// Synthetic tool-use id of the runner-injected forced consult (REQ-H22).
/// Unlike the pre-refactor post-loop pass, this now correlates a REAL
/// assistant `ToolUse` / `ToolResult` message pair appended to `history` —
/// the model's next provider turn genuinely sees it and can react.
const FORCED_CONSULT_TOOL_USE_ID: &str = "forced-consult";

/// `ToolResult` content recorded — WITHOUT going through the observer, so it
/// never appears twice in the run's tool-call audit — when the model itself
/// requests `consult` after the forced consult already ran for this query.
/// REQ-H22 guarantees exactly one consult invocation per forced run: the
/// model's ToolUse block still needs an answer (the API contract requires
/// one), so it gets this message, but the tool is never re-executed.
const CONSULT_ALREADY_FORCED_MESSAGE: &str =
    "consult already ran once for this forced query; no further invocations";

/// Consecutive vetoes that close `consult` for the rest of the TURN (REQ-A20c).
///
/// **The exact sequence, spelled out because "the second veto is terminal" reads
/// two ways and the trace is not obvious:**
///
/// | Call | Counter on entry | What happens |
/// |---|---|---|
/// | 1st, trivial | 0 | EVALUATED, vetoed ⇒ counter 1 |
/// | 2nd, trivial | 1 | EVALUATED, vetoed ⇒ counter 2 |
/// | 3rd onward | 2 | NOT evaluated: the door is closed and the result says so |
///
/// So: two vetoes actually occur, and starting from the third call the door is
/// closed. The second is not rejected sight-unseen — it is evaluated, vetoed,
/// and that is the last one. `1` here would block the second call WITHOUT
/// evaluating it, which is one fewer veto than the spec describes.
const MAX_CONSECUTIVE_VETOES: u8 = 2;

/// Text returned to the agent after a veto (REQ-A20e).
///
/// **Does not reveal the threshold**: saying how many characters are missing is
/// a direct invitation to pad the content until it passes, which would produce
/// exactly the expensive consult the gate exists to avoid, plus the cost of the
/// padding.
fn veto_message(mode: &Mode) -> String {
    format!(
        "This content does not warrant a three-perspective consensus in {mode} mode: it is \
         too short. Retrying with the same content gives the same result — answer directly \
         instead."
    )
}

/// SECOND veto: also says the door closed for the rest of the turn (REQ-A20f).
fn veto_message_terminal(mode: &Mode) -> String {
    format!(
        "{} And this is the second veto this turn: `consult` is now disabled for the rest \
         of it. The next turn starts clean.",
        veto_message(mode)
    )
}

/// Calls made AFTER the door already closed: not even re-evaluated. Same rule
/// as the other two messages — never names the threshold.
fn consult_disabled_message() -> String {
    "`consult` is disabled for the rest of this turn (two consecutive vetoes). Answer \
     directly; the next turn starts clean."
        .to_string()
}

/// Records ONE gate evaluation (REQ-A20 telemetry, SC-A20h).
///
/// **Not an `eprintln!`**: this is telemetry to calibrate thresholds, not a
/// user-facing message — mixing the two channels would make the interactive
/// path noisy.
fn log_gate(config: &AgentRunConfig, mode: &Mode, chars: usize, threshold: usize, vetoed: bool) {
    // The APPLIED threshold travels in the line (SC-A20h). Without it the
    // telemetry cannot do the one thing it exists for: with `[magi.complexity]`
    // configurable, "vetoed at 40 chars" does not say whether the threshold was
    // 50 or 500, and calibrating is exactly comparing the two numbers.
    config
        .gate_telemetry
        .on_gate_evaluation(mode, chars, threshold, vetoed);
}

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
    /// Records only calls that reached the authorize/execute path. A model-issued
    /// `consult` that is short-circuited by the forced-consult lock (REQ-H22, see
    /// [`CONSULT_ALREADY_FORCED_MESSAGE`]) is answered directly and is
    /// deliberately NOT recorded here — it neither authorized nor executed, so the
    /// audit shows exactly the one forced `consult` that actually ran.
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

    /// Records token usage from a `ResponseChunk::Usage`, whenever the backend
    /// reports it (the built-in providers emit at most one usage chunk per
    /// `stream_messages` call, but that is a provider convention, not a
    /// guarantee this method enforces). Each call is **additive**: an implementor
    /// accumulates across calls for a whole-run total, so a backend that happened
    /// to report usage in several chunks simply sums correctly.
    ///
    /// Default empty body: an observer that does not care about usage (or a
    /// future implementor written before this method existed) needs no change.
    /// Not implementing this method never affects `authorize`/`on_tool_call`/
    /// `on_final_turn` semantics.
    fn on_usage(&self, _input_tokens: u64, _output_tokens: u64) {}
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
    /// Effective system-prompt text for this run (REQ-H12b), forwarded to every
    /// `Provider::stream_messages` call. `None` ⇒ no system prompt is sent — the
    /// interactive default — so the TUI path is unaffected. The headless runner
    /// sets this from the resolved `SystemPolicy` (`magi_rs::headless::types`):
    /// the operator default, or — only when explicitly enabled — the caller
    /// override.
    pub system: Option<String>,
    /// When `true`, [`Agent::run_tool_loop`] injects exactly one **in-loop**
    /// `consult` tool call before the first provider turn (REQ-H22 forced
    /// consult: `magi query --consult` / envelope `consult:true`). `false` (the
    /// default) never injects anything — the interactive/TUI path and an
    /// unconfigured headless run are unaffected. See the rustdoc on
    /// [`Agent::run_tool_loop`] for the full authorization/at-most-once
    /// contract.
    pub force_consult: bool,
    /// Complexity-gate thresholds for an autonomous `consult` (REQ-A20/A20b).
    /// [`AgentRunConfig::default`] uses [`GateThresholds::builtin`] — the gate
    /// is ACTIVE by default (REQ-A20b): a security-relevant gate that turns
    /// itself off by omission is a gate that's effectively off. Only
    /// [`Agent::dispatch_consult_through_gate`] (the model-issued `ToolUse`
    /// path) ever evaluates this; the forced pre-loop injection above
    /// (REQ-H22) never does — see that method's rustdoc for why that
    /// call-site distinction IS REQ-A20's contract.
    pub gate_thresholds: GateThresholds,
    /// Mode resolution inputs the funnel needs without re-reading `magi.toml`
    /// per turn (REQ-A07/A07c/A07d). [`AgentRunConfig::default`] leaves both
    /// fields at their least-surprising value — no configured default mode,
    /// `untrusted_content` off — so an unconfigured run keeps inferring/
    /// defaulting exactly as it did before this field existed.
    pub mode_config: ModeConfig,
    /// Sink for the gate's telemetry (REQ-A20, SC-A20h). **Deliberately
    /// separate from [`RunObserver`]**: the observer is `None` in the TUI —
    /// the surface that autonomously routes to `consult` the most — so a
    /// signal SC-A20h requires *always* recorded cannot depend on it.
    /// [`AgentRunConfig::default`] installs [`NoGateTelemetry`] (zero
    /// recording), so this field is purely additive: no existing run changes
    /// behavior by having it.
    pub gate_telemetry: Arc<dyn GateTelemetry>,
}

impl Default for AgentRunConfig {
    fn default() -> Self {
        Self {
            max_tool_calls: DEFAULT_MAX_TOOL_CALLS,
            disable_repetitive_guard: false,
            observer: None,
            cancel: CancellationToken::new(),
            system: None,
            force_consult: false,
            gate_thresholds: GateThresholds::builtin(),
            mode_config: ModeConfig::default(),
            gate_telemetry: Arc::new(NoGateTelemetry),
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
                // Sort (key, value) pairs directly instead of sorting keys and then
                // re-looking them up: this yields the same key order without ever
                // needing a `map.get(k)` that could (in principle) miss.
                let mut sorted_entries: Vec<_> = map.iter().collect();
                sorted_entries.sort_by_key(|(k, _)| *k);
                let mut parts = Vec::new();
                for (k, v) in sorted_entries {
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
        // Snapshot Arc handles (ref-count increments only; O(1)) so the mutable loop
        // body below can borrow `self` freely without conflicting field borrows on
        // `memory_subsystem`. Folding the "is selective mode" check into the same
        // `Option` match that produces the snapshot avoids a separate `unwrap()` on
        // a second, independent `as_ref()` call.
        let selective_snapshot = self.memory_subsystem.as_ref().and_then(|s| {
            (s.cfg.mode == "selective").then(|| {
                (
                    s.store.clone(),
                    s.embedder.clone(),
                    s.clock.clone(),
                    s.cfg.clone(),
                    s.scope.clone(),
                )
            })
        });

        if let Some((store, embedder, clock, cfg, scope)) = selective_snapshot {
            // `unwrap_or_default()` cannot panic: a `None` session_id yields an
            // empty string (a persistence run without a bound session is a no-op
            // downstream), so selective mode never depends on a bound session.
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
            //
            // The `system` argument is always `""` here: the real system-prompt
            // injection point is `AgentRunConfig::system`, forwarded straight to
            // `Provider::stream_messages` (REQ-H12b) for BOTH the selective and
            // load_all paths from the single call site in `run_tool_loop`. This
            // assembler's own `system` parameter remains part of `assemble_selective`'s
            // stable signature (its budget-accounting/preamble role is exercised
            // directly by `context.rs`'s tests) but is not fed from `Agent` state —
            // the former `MemorySubsystem.system` field was always `""` (dead: no
            // setter ever populated it) and has been removed rather than wired to
            // `config.system`, which would have sent the same system text to the
            // provider twice (once via the API's dedicated system channel, once
            // folded into this preamble as ordinary message text).
            let (working_messages, assembly_notices) = match assemble_selective(
                &*store,
                &*embedder,
                &*clock,
                &cfg,
                "",
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
                .run_tool_loop(working_messages, &chunk_tx, &config, text)
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
            //
            // `memory_subsystem` is guaranteed `Some` here — `selective_snapshot`
            // above only produced `Some(..)` (entering this `if let` block at all)
            // when `self.memory_subsystem` was `Some`, and nothing in between
            // clears it — but the `None` arm still returns a safe, no-op default
            // instead of unwrapping, so this can never panic even if that
            // invariant is ever violated by a future edit.
            let should_distill = if let Some(sub) = self.memory_subsystem.as_mut() {
                sub.turns_since_open = sub.turns_since_open.saturating_add(1);
                let n = sub.cfg.distill_every_n_turns;
                sub.cfg.distill_enabled && n > 0 && sub.turns_since_open.is_multiple_of(n)
            } else {
                false
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
        let (_full_text, final_text) = self
            .run_tool_loop(working, &chunk_tx, &config, text)
            .await?;
        Ok(final_text)
    }

    /// Authorizes, executes, and records ONE tool call, returning the
    /// `Content::ToolResult` to fold into the requesting turn's results.
    ///
    /// Shared by the model-driven `ToolUse` dispatch inside [`Self::run_tool_loop`]
    /// and its pre-loop forced-consult injection (REQ-H22), so both sites run
    /// through IDENTICAL approval/execution/observer-recording logic — a forced
    /// consult is authorized and executed exactly like a model-requested one.
    ///
    /// # Authorization
    /// A headless observer ([`AgentRunConfig::observer`]), when present, is
    /// AUTHORITATIVE for every tool (REQ-H06/H07/H09) — the only mechanism that
    /// can gate a tool opting out of interactive approval (`project_knowledge`,
    /// an auto-approve `consult`), which never reaches `approval_tx`. Without an
    /// observer the original interactive `approval_tx` / `requires_approval()`
    /// gate runs unchanged.
    ///
    /// # Parameters
    /// - `id` — the tool-use id correlating this call with its `ToolResult`.
    /// - `name` — the tool name to look up in `self.tools`.
    /// - `input` — the JSON input to execute the tool with.
    /// - `config` — the run configuration (observer, cancellation, approval).
    /// - `chunk_tx` — streaming sender, used only to forward an auto-approved
    ///   tool's [`Tool::approval_notice`].
    ///
    /// # Returns
    /// `Content::ToolResult` — a denial (`is_error = true`) when not approved,
    /// otherwise the tool's own result (`is_error` reflecting `Tool::execute`).
    /// A name with no matching registered tool resolves to a "not found" error
    /// result rather than panicking.
    async fn authorize_and_execute_tool(
        &mut self,
        id: &str,
        name: &str,
        input: &serde_json::Value,
        config: &AgentRunConfig,
        chunk_tx: &tokio::sync::mpsc::Sender<StreamPiece>,
    ) -> Content {
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
                            tool_name: name.to_string(),
                            input: input.clone(),
                            tx: oneshot_tx,
                        })
                        .await;
                    match timeout(Duration::from_secs(APPROVAL_TIMEOUT_SECS), oneshot_rx).await {
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
                format!("Tool '{name}' denied: not authorized in the current authorization tier")
            } else {
                "Execution denied or timed out.".to_string()
            };
            if let Some(observer) = config.observer.as_deref() {
                observer.on_tool_call(id, name, input, &denial_msg, false, 0);
            }
            return Content::ToolResult {
                tool_use_id: id.to_string(),
                content: denial_msg,
                is_error: true,
            };
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
        let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        if let Some(observer) = config.observer.as_deref() {
            observer.on_tool_call(id, name, input, &result_content, !is_error, elapsed_ms);
        }
        Content::ToolResult {
            tool_use_id: id.to_string(),
            content: result_content,
            is_error,
        }
    }

    /// Runs a model-issued `consult` request through the complexity gate
    /// (REQ-A20), producing the `ToolResult` to fold into the turn.
    ///
    /// # The call-site distinction this method IS
    ///
    /// This is called **only** from the `ToolUse` loop in [`Self::run_tool_loop`]
    /// — the model deciding, on its own, to invoke `consult`. The forced
    /// pre-loop injection (REQ-H22) calls [`Self::authorize_and_execute_tool`]
    /// directly and never reaches this method at all. That is the entire
    /// mechanism behind REQ-A20's "the gate vetoes the autonomous route, never
    /// an explicit one": `/consult` (TUI), `magi consult` (CLI) and a forced
    /// consult all bypass the gate structurally, by never calling this method,
    /// not by any flag this method could check. If a future refactor ever
    /// unifies the two call sites "to simplify", an explicit consult starts
    /// getting vetoed — see `a_forced_injection_bypasses_the_gate_while_a_model_choice_does_not`.
    ///
    /// # Outcomes
    /// - The door is already closed this turn (`consecutive_vetoes >=
    ///   [`MAX_CONSECUTIVE_VETOES`]`): not even re-evaluated, zero model calls.
    /// - [`GateVerdict::Veto`]: `consecutive_vetoes` increments; zero model
    ///   calls; the result names the mode and, on the second veto, that the
    ///   door is now closed for the rest of the turn (REQ-A20c).
    /// - [`GateVerdict::Dispatch`]: `consecutive_vetoes` resets to `0` (a
    ///   dispatched consult spends three model calls — the exact cost the gate
    ///   exists to avoid — so resetting on it never opens a padding shortcut),
    ///   and the resolved `(Mode, ModeSource)` is injected into a CLONED input
    ///   (`magi_rs::magi::mode::input_for_dispatch`) before the real dispatch,
    ///   so [`ConsultTool`](crate::tools::consult::ConsultTool) reads the same
    ///   mode the gate just evaluated instead of re-resolving it.
    ///
    /// There is deliberately **no anti-mode-shopping guard**: the agent can
    /// choose the mode via the tool's `input_schema` (REQ-A07b), so a veto
    /// under `Design` (higher threshold) followed by a retry under `Analysis`
    /// (lower) that then dispatches is not evasion — the built-in thresholds
    /// say a shorter `Analysis` is legitimately enough for that lens, and the
    /// gate's own `Dispatch` verdict says so. A guard here would second-guess
    /// the gate's own configuration, and the plain reset-on-success rule
    /// already covers the case that matters: a mode change that fails to pass
    /// is still a `Veto` (counts, and closes the door on the second one).
    ///
    /// # Errors
    /// Propagates [`magi_rs::magi::mode::ModeError`] from
    /// [`resolve_mode_guarded`] (`untrusted_content` active with no declared
    /// mode) — the same fail-closed behavior the rest of the run already
    /// applies to a malformed turn.
    async fn dispatch_consult_through_gate(
        &mut self,
        id: &str,
        input: &serde_json::Value,
        config: &AgentRunConfig,
        chunk_tx: &tokio::sync::mpsc::Sender<StreamPiece>,
        consecutive_vetoes: &mut u8,
    ) -> Result<Content> {
        if *consecutive_vetoes >= MAX_CONSECUTIVE_VETOES {
            // REQ-A20c: the door already closed this turn. Not even the gate
            // itself runs again — zero model calls, same as a fresh veto.
            return Ok(Content::ToolResult {
                tool_use_id: id.to_string(),
                content: consult_disabled_message(),
                is_error: false,
            });
        }

        let query = input
            .get("query")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        // Explicit (level 1) is never populated here: there is no human on the
        // autonomous route. `classifier: None` — this call site never infers
        // (REQ-A07d); the agent already had its chance to pick a mode via its
        // own `mode` argument (`agent_chosen_mode`), which is level 3, not a
        // classification call.
        let res = resolve_mode_guarded(
            None,
            config.mode_config.default_mode,
            agent_chosen_mode(input),
            config.mode_config.untrusted_content,
            None,
            query,
        )
        .await?;

        match evaluate(query, &res.mode, &config.gate_thresholds) {
            GateVerdict::Veto { mode } => {
                *consecutive_vetoes += 1;
                log_gate(
                    config,
                    &mode,
                    query.chars().count(),
                    config.gate_thresholds.for_mode(&mode),
                    true,
                );
                let content = if *consecutive_vetoes >= MAX_CONSECUTIVE_VETOES {
                    veto_message_terminal(&mode)
                } else {
                    veto_message(&mode)
                };
                Ok(Content::ToolResult {
                    tool_use_id: id.to_string(),
                    content,
                    is_error: false,
                })
            }
            GateVerdict::Dispatch => {
                // A dispatched consult already spends three model calls, so
                // resetting here never opens a "pad it until it passes"
                // shortcut (REQ-A20c) — that cost IS what the gate exists to
                // avoid, so paying it buys the reset honestly.
                *consecutive_vetoes = 0;
                log_gate(
                    config,
                    &res.mode,
                    query.chars().count(),
                    config.gate_thresholds.for_mode(&res.mode),
                    false,
                );
                let tool_input = input_for_dispatch(input, &res);
                Ok(self
                    .authorize_and_execute_tool(id, CONSULT_TOOL_NAME, &tool_input, config, chunk_tx)
                    .await)
            }
        }
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
    /// - `prompt` — the original user query text for this turn. Used ONLY by the
    ///   [`AgentRunConfig::force_consult`] injection (REQ-H22) to build the
    ///   forced consult's `{"query": prompt}` input; ignored otherwise.
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
    ///
    /// # Forced consult (REQ-H22)
    /// When `config.force_consult` is `true`, BEFORE the first provider call this
    /// method injects exactly one synthetic `consult` tool call — a real
    /// `Assistant` `ToolUse` message immediately followed by a `User`
    /// `ToolResult` message, appended to both `working` and `self.history`
    /// exactly like a model-requested call. This means: it consumes one
    /// `max_tool_calls` slot, it is authorized through the same
    /// [`RunObserver::authorize`] tier gate (so `default` denies it, `ok =
    /// false`, never elevated), it is executed via the SAME registered
    /// `consult` tool a proactive call would use (so authorization/execution/
    /// recording are identical), and — because it is appended to `working`
    /// before the loop's first `stream_messages` call — the model's very first
    /// turn already sees the result and can react to it. If the tool is not
    /// registered on this agent, the call is recorded as a "not found" failure
    /// instead of panicking.
    ///
    /// Once injected (successfully or not), `forced_consult_done` locks out any
    /// further `consult` request for the rest of THIS run: if the model also
    /// requests `consult` afterward, that `ToolUse` is answered with
    /// [`CONSULT_ALREADY_FORCED_MESSAGE`] and neither authorized nor executed
    /// again — recorded only in the conversation, NOT re-added to the observer's
    /// audit trail (so `RunOutcome.tool_calls` still shows exactly one `consult`
    /// entry). This is REQ-H22's "no se re-dispara aunque el agente quisiera".
    ///
    /// The injection deliberately does **not** seed `last_normalized_tool`/
    /// `repeat_count` (the 3x-identical-call guard below): it is a runner
    /// injection, not a model-issued call, so it must not count toward — or
    /// shift the threshold for — the model's own repetition tracking. A
    /// model-issued `consult` that happens to match the forced call's input
    /// is tracked exactly like any other first occurrence.
    async fn run_tool_loop(
        &mut self,
        mut working: Vec<Message>,
        chunk_tx: &tokio::sync::mpsc::Sender<StreamPiece>,
        config: &AgentRunConfig,
        prompt: &str,
    ) -> Result<(String, String)> {
        // SEQUENTIALITY REQUIRED: this counter — like `repeat_count` just below, and like
        // the veto counter REQ-A20c adds in MS2 — is a flat local that assumes the
        // `ToolUse` loop further down dispatches one at a time. Parallelising that loop
        // breaks all of them at once, and silently: nothing in a `let mut x = 0` says
        // "sequential". Pinned by `tool_dispatch_is_sequential_within_a_turn`, so the
        // change breaks the suite instead of surfacing as miscounts under load.
        let mut tool_call_count = 0;
        let mut last_normalized_tool: Option<(String, String)> = None;
        // SEQUENTIALITY REQUIRED — see the note on `tool_call_count` above.
        let mut repeat_count = 0;
        // SEQUENTIALITY REQUIRED — see the note on `tool_call_count` above. This
        // is the third counter that note already promised: MS2's veto counter
        // (REQ-A20c). Same flat local, same lifetime (the turn: born on entry,
        // gone on exit via any of the four return paths, no `Drop` to write),
        // same pin (`tool_dispatch_is_sequential_within_a_turn`). It does NOT
        // live on `Agent` or on `ConsultTool` (an `Arc` shared across
        // sessions) — putting it there would let a turn vetoed in one session
        // disable `consult` in another.
        let mut consecutive_vetoes: u8 = 0;
        // REQ-H22: locks out any further `consult` request once the forced
        // pre-loop injection below has run (success, denial, or "not found").
        // Defensive/redundant by construction: the injection block runs iff
        // `config.force_consult`, and sets this to `true` unconditionally, so
        // when `force_consult` is set this is always `true` by the time the loop
        // reads it. It is kept explicit (rather than reusing `config.force_consult`
        // alone) so a future reader sees the "already fired" intent directly.
        let mut forced_consult_done = false;

        if config.force_consult {
            let input = serde_json::json!({ "query": prompt });
            tool_call_count += 1;
            if tool_call_count > config.max_tool_calls {
                // See `MAX_TOOL_CALLS_ERROR`'s rustdoc: this exact string is a
                // stability contract with `headless_runner::run_query`.
                return Err(anyhow::anyhow!(MAX_TOOL_CALLS_ERROR));
            }
            // Deliberately NOT seeding `last_normalized_tool` here: this is a
            // runner injection, not a model-issued call, so it must not count
            // toward the model's own 3x-identical repetitive-call guard below
            // (REQ-H22). Seeding it would let a genuinely distinct model call
            // be miscounted as a repeat of this forced one.
            let result_content = self
                .authorize_and_execute_tool(
                    FORCED_CONSULT_TOOL_USE_ID,
                    CONSULT_TOOL_NAME,
                    &input,
                    config,
                    chunk_tx,
                )
                .await;

            let tool_use_msg = Message {
                role: Role::Assistant,
                content: vec![Content::ToolUse {
                    id: FORCED_CONSULT_TOOL_USE_ID.to_string(),
                    name: CONSULT_TOOL_NAME.to_string(),
                    input: input.clone(),
                }],
            };
            self.history.push(tool_use_msg.clone());
            working.push(tool_use_msg.clone());
            if let (Some(memory), Some(sid)) = (&self.memory, &self.session_id) {
                memory.add_message(sid, &tool_use_msg).await?;
            }

            let tool_res_msg = Message {
                role: Role::User,
                content: vec![result_content],
            };
            self.history.push(tool_res_msg.clone());
            working.push(tool_res_msg.clone());
            if let (Some(memory), Some(sid)) = (&self.memory, &self.session_id) {
                memory.add_message(sid, &tool_res_msg).await?;
            }

            forced_consult_done = true;
        }

        loop {
            let mut stream = self
                .provider
                .stream_messages(&working, &self.tools, config.system.as_deref())
                .await?;
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
                    ResponseChunk::Usage {
                        input_tokens,
                        output_tokens,
                    } => {
                        // Headless-only signal: no observer ⇒ no-op (default empty
                        // body), so the interactive path is unaffected. Forwarded
                        // per turn; the observer accumulates the run total.
                        if let Some(observer) = config.observer.as_deref() {
                            observer.on_usage(input_tokens, output_tokens);
                        }
                    }
                    ResponseChunk::ToolUseInputDelta { .. } => {}
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
                        // See `MAX_TOOL_CALLS_ERROR`'s rustdoc: this exact
                        // string is a stability contract with
                        // `headless_runner::run_query`.
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

                    // REQ-H22: once the forced consult has run (successfully or
                    // not), any further `consult` request for the rest of THIS
                    // run — model-issued or not — is answered but never
                    // re-authorized/re-executed/re-recorded (see the
                    // `# Forced consult` rustdoc on this method). Such a blocked
                    // request still consumed one `max_tool_calls` slot (counted
                    // at the top of this iteration) and its ToolUse/ToolResult
                    // stay in the conversation history, but it is deliberately
                    // absent from the observer audit trail (`on_tool_call` is not
                    // called), so `RunOutcome.tool_calls` shows exactly the one
                    // forced consult that actually ran.
                    let result_content =
                        if config.force_consult && forced_consult_done && name == CONSULT_TOOL_NAME
                        {
                            Content::ToolResult {
                                tool_use_id: id.clone(),
                                content: CONSULT_ALREADY_FORCED_MESSAGE.to_string(),
                                is_error: true,
                            }
                        } else if name == CONSULT_TOOL_NAME {
                            // REQ-A20: this is the model's OWN request (the
                            // `ToolUse` loop), which is exactly the call site the
                            // complexity gate is meant to see. The forced
                            // pre-loop injection above never reaches this branch
                            // — it calls `authorize_and_execute_tool` directly,
                            // a few lines up, bypassing the gate entirely. Two
                            // distinct call sites to the SAME tool is the whole
                            // property REQ-A20 rests on; see
                            // `dispatch_consult_through_gate`'s rustdoc.
                            self.dispatch_consult_through_gate(
                                id,
                                input,
                                config,
                                chunk_tx,
                                &mut consecutive_vetoes,
                            )
                            .await?
                        } else {
                            self.authorize_and_execute_tool(id, name, input, config, chunk_tx)
                                .await
                        };
                    tool_results.push(result_content);
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
    //
    // `embed` is called with a single-element input slice, so per its contract
    // (see `EmbeddingProvider::embed` rustdoc) an `Ok` result has exactly one
    // vector. Rather than indexing `v[0]` (panics on an out-of-contract empty
    // `Ok`) and separately `unwrap()`-ing `v.into_iter().next()`, consume the
    // iterator once and match on the `Option` it yields — the `None` arm covers
    // both an empty `Ok` and an `Err` with the same safe fallback.
    let (embedding, model_id, dim) = match embedder.embed(&[prefixed]).await {
        Ok(v) => match v.into_iter().next() {
            Some(first) => {
                let d = first.len();
                (first, embedder.model_id().to_string(), d)
            }
            None => (Vec::new(), String::new(), 0),
        },
        Err(_) => (Vec::new(), String::new(), 0),
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
            _system: Option<&str>,
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
            _system: Option<&str>,
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
                _system: Option<&str>,
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
                _system: Option<&str>,
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

    // ── Feature E: AgentRunConfig::system reaches the Provider (REQ-H12b) ──────

    /// Provider that records every `system` argument it received across calls.
    struct SystemCapturingProvider {
        seen: Arc<std::sync::Mutex<Vec<Option<String>>>>,
    }

    #[async_trait]
    impl Provider for SystemCapturingProvider {
        async fn stream_messages(
            &self,
            _messages: &[Message],
            _tools: &[Box<dyn Tool>],
            system: Option<&str>,
        ) -> Result<BoxStream<'static, Result<ResponseChunk>>> {
            self.seen.lock().unwrap().push(system.map(str::to_string));
            Ok(Box::pin(stream::iter(vec![Ok(
                ResponseChunk::MessageDone(Message::assistant("ok")),
            )])))
        }
    }

    #[tokio::test]
    async fn test_query_streaming_forwards_configured_system_to_provider() {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut agent = Agent::new(Arc::new(SystemCapturingProvider { seen: seen.clone() }));
        let (tx, _rx) = tokio::sync::mpsc::channel::<StreamPiece>(8);
        let config = AgentRunConfig {
            system: Some("You are a headless test assistant.".to_string()),
            ..AgentRunConfig::default()
        };
        agent.query_streaming("hi", tx, config).await.unwrap();
        assert_eq!(
            seen.lock().unwrap().as_slice(),
            &[Some("You are a headless test assistant.".to_string())]
        );
    }

    #[tokio::test]
    async fn test_query_streaming_default_config_sends_no_system() {
        // Interactive default: AgentRunConfig::default().system == None must
        // reach the provider as None — the TUI path stays byte-for-byte
        // unaffected by the headless system-prompt channel.
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut agent = Agent::new(Arc::new(SystemCapturingProvider { seen: seen.clone() }));
        let (tx, _rx) = tokio::sync::mpsc::channel::<StreamPiece>(8);
        agent
            .query_streaming("hi", tx, AgentRunConfig::default())
            .await
            .unwrap();
        assert_eq!(seen.lock().unwrap().as_slice(), &[None]);
    }

    // ── Feature C: usage accumulation via RunObserver::on_usage ────────────────

    /// Provider that replays a fixed script of turns (tool-call, then terminal
    /// text), emitting a `ResponseChunk::Usage` on every turn — used to exercise
    /// per-turn `RunObserver::on_usage` accumulation across ≥2 provider calls.
    struct UsageScriptedProvider {
        /// `(input_tokens, output_tokens)` per call, consumed in order.
        turns: std::sync::Mutex<std::collections::VecDeque<(u64, u64)>>,
    }

    #[async_trait]
    impl Provider for UsageScriptedProvider {
        async fn stream_messages(
            &self,
            _messages: &[Message],
            _tools: &[Box<dyn Tool>],
            _system: Option<&str>,
        ) -> Result<BoxStream<'static, Result<ResponseChunk>>> {
            let (input_tokens, output_tokens) =
                self.turns.lock().unwrap().pop_front().unwrap_or((0, 0));
            let is_last = self.turns.lock().unwrap().is_empty();
            let mut chunks = vec![Ok(ResponseChunk::Usage {
                input_tokens,
                output_tokens,
            })];
            if is_last {
                // Terminal turn: no ToolUse ⇒ run_tool_loop returns.
                chunks.push(Ok(ResponseChunk::MessageDone(Message::assistant("done"))));
            } else {
                chunks.push(Ok(ResponseChunk::MessageDone(Message {
                    role: Role::Assistant,
                    content: vec![Content::ToolUse {
                        id: "u1".to_string(),
                        name: "no-such-tool".to_string(),
                        input: serde_json::json!({}),
                    }],
                })));
            }
            Ok(Box::pin(stream::iter(chunks)))
        }
    }

    /// Minimal [`RunObserver`] spy: authorizes every tool, accumulates every
    /// `on_usage` call, ignores `on_tool_call`/`on_final_turn`.
    #[derive(Default)]
    struct UsageSpyObserver {
        total: std::sync::Mutex<(u64, u64)>,
    }

    impl RunObserver for UsageSpyObserver {
        fn authorize(&self, _tool_name: &str) -> bool {
            true
        }
        fn on_tool_call(
            &self,
            _id: &str,
            _name: &str,
            _input: &serde_json::Value,
            _result: &str,
            _ok: bool,
            _ms: u64,
        ) {
        }
        fn on_final_turn(&self, _text_block_count: usize) {}
        fn on_usage(&self, input_tokens: u64, output_tokens: u64) {
            let mut t = self.total.lock().unwrap();
            t.0 += input_tokens;
            t.1 += output_tokens;
        }
    }

    #[tokio::test]
    async fn test_run_tool_loop_accumulates_usage_via_observer_across_turns() {
        // Two turns: turn 1 (a tool call) reports (10, 2); turn 2 (terminal)
        // reports (5, 3). The observer must see BOTH calls and the run's total
        // usage is the sum across turns, not just the last one.
        let provider = Arc::new(UsageScriptedProvider {
            turns: std::sync::Mutex::new(std::collections::VecDeque::from([(10, 2), (5, 3)])),
        });
        let mut agent = Agent::new(provider);
        let observer = Arc::new(UsageSpyObserver::default());
        let config = AgentRunConfig {
            observer: Some(observer.clone() as Arc<dyn RunObserver>),
            ..AgentRunConfig::default()
        };
        let (tx, _rx) = tokio::sync::mpsc::channel::<StreamPiece>(8);
        agent.query_streaming("hi", tx, config).await.unwrap();
        assert_eq!(*observer.total.lock().unwrap(), (15, 5));
    }

    #[tokio::test]
    async fn test_on_usage_default_impl_is_a_no_op_for_observers_that_ignore_it() {
        // A RunObserver written before on_usage existed compiles unchanged and
        // calling the default method never panics.
        struct MinimalObserver;
        impl RunObserver for MinimalObserver {
            fn authorize(&self, _tool_name: &str) -> bool {
                true
            }
            fn on_tool_call(
                &self,
                _id: &str,
                _name: &str,
                _input: &serde_json::Value,
                _result: &str,
                _ok: bool,
                _ms: u64,
            ) {
            }
            fn on_final_turn(&self, _text_block_count: usize) {}
        }
        MinimalObserver.on_usage(1, 1);
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
            _system: Option<&str>,
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
            _system: Option<&str>,
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

    // ── Interactive-path regression (MS2 T8, REQ-H28) ───────────────────────────
    //
    // The headless MS2 work threaded an `AgentRunConfig` (T3), a
    // `Tool::execute(cancel)` argument (T4) and a `.magi/` discovery step (T7)
    // through the shared agent loop. The TUI callers were updated to pass
    // `AgentRunConfig::default()` (max_tool_calls = 15, repetitive guard ENABLED,
    // no observer) + a never-cancelled token. These tests pin that the interactive
    // path (observer = None) behaves byte-for-byte as it did in v0.9.0: the
    // approval gate still mediates a dangerous tool, and the repetitive-call guard
    // still fires — neither is auto-approved nor silenced by the new plumbing.

    /// Provider that requests the SAME identical tool call on EVERY turn, so the
    /// agent's 3-repetition guard is the terminal condition (well before the
    /// 15-call cap). `id` is constant, but `normalize_input` compares only
    /// `(name, normalized_input)`, so the repetition is detected regardless.
    struct AlwaysSameToolProvider {
        tool_name: String,
    }

    #[async_trait]
    impl Provider for AlwaysSameToolProvider {
        async fn stream_messages(
            &self,
            _messages: &[Message],
            _tools: &[Box<dyn Tool>],
            _system: Option<&str>,
        ) -> Result<BoxStream<'static, Result<ResponseChunk>>> {
            let msg = Message {
                role: Role::Assistant,
                content: vec![Content::ToolUse {
                    id: "repeat-id".to_string(),
                    name: self.tool_name.clone(),
                    input: json!({"same": "input"}),
                }],
            };
            Ok(Box::pin(stream::iter(vec![Ok(
                ResponseChunk::MessageDone(msg),
            )])))
        }
    }

    /// REGRESSION: on the interactive path (`AgentRunConfig::default()`, no
    /// observer) a dangerous tool (`requires_approval() == true`) is STILL gated
    /// through the `approval_tx` prompt — it is NOT auto-approved — and the run
    /// executes it only after the UI approves, returning the model's normal text.
    ///
    /// This is the v0.9.0 approval-gate behavior; the MS2 `AgentRunConfig`/observer
    /// plumbing must not have changed it for the observer-less interactive path.
    #[tokio::test]
    async fn test_interactive_path_regression_approval_gate_still_gates_dangerous_tool() {
        use tokio::sync::mpsc;

        // Approver that records every tool name it is asked to authorize, then
        // approves — proving an ApprovalRequest was actually emitted (the gate
        // ran) rather than the tool being silently auto-approved.
        let gated: Arc<std::sync::Mutex<Vec<String>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (approval_tx, mut approval_rx) = mpsc::channel::<ApprovalRequest>(8);
        let gated_seen = gated.clone();
        tokio::spawn(async move {
            while let Some(req) = approval_rx.recv().await {
                gated_seen.lock().unwrap().push(req.tool_name.clone());
                let _ = req.tx.send(true);
            }
        });

        let (tool, executed) = TrackingTool::new("gated_op", false /* dangerous */);
        let (provider, _count) = SingleToolCallProvider::new("gated_op");
        let mut agent = Agent::new(Arc::new(provider));
        agent.register_tool(Box::new(tool));
        agent.set_approval_channel(approval_tx);

        let (chunk_tx, _rx) = mpsc::channel::<StreamPiece>(8);
        let response = agent
            .query_streaming("use the gated tool", chunk_tx, AgentRunConfig::default())
            .await
            .expect("interactive run must complete once approval is granted");

        // The gate ran: exactly one ApprovalRequest for the dangerous tool.
        assert_eq!(
            *gated.lock().unwrap(),
            vec!["gated_op".to_string()],
            "interactive path must route a dangerous tool through the approval gate \
             (one ApprovalRequest), NOT auto-approve it"
        );
        // Approval was honored: the tool executed.
        assert!(
            *executed.lock().unwrap(),
            "the approved dangerous tool must execute on the interactive path"
        );
        // Response semantics unchanged: the model's final text is returned.
        assert_eq!(
            response, "done",
            "interactive run must return the model's normal final text after the tool call"
        );
    }

    /// REGRESSION: on the interactive path (`AgentRunConfig::default()`, whose
    /// `disable_repetitive_guard == false`) three identical consecutive tool calls
    /// STILL abort the run with "Repetitive tool call detected". The `--full-auto`
    /// soft-guard silencing (REQ-H08) must NOT leak into the interactive/Default
    /// path.
    #[tokio::test]
    async fn test_interactive_path_regression_repetitive_guard_still_fires() {
        use tokio::sync::mpsc;

        let (tool, _executed) = TrackingTool::new("loop_op", true /* auto-approve */);
        let provider = AlwaysSameToolProvider {
            tool_name: "loop_op".to_string(),
        };
        let mut agent = Agent::new(Arc::new(provider));
        agent.register_tool(Box::new(tool));

        let (chunk_tx, _rx) = mpsc::channel::<StreamPiece>(8);
        let err = agent
            .query_streaming("loop forever", chunk_tx, AgentRunConfig::default())
            .await
            .expect_err("the repetitive-call guard must abort the run on the interactive path");

        let msg = err.to_string();
        assert!(
            msg.contains("Repetitive tool call detected"),
            "interactive path must fire the repetitive-call guard (not be silenced); got: {msg}"
        );
        // The guard — not the tool-call cap — is the terminal condition, matching
        // v0.9.0 semantics (guard fires at the 3rd repeat, well under the 15 cap).
        assert!(
            !msg.contains(MAX_TOOL_CALLS_ERROR),
            "the guard must terminate the run before the max-tool-calls cap; got: {msg}"
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
                _system: Option<&str>,
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

    /// Rendezvous window for [`OverlapProbeTool`].
    ///
    /// Under the sequential dispatch this guards, nobody ever arrives, so this window is
    /// spent in full on every run — that fixed half second is the price of the guarantee.
    /// It is deliberately generous anyway: if it were too short, a *parallel* loop whose
    /// second dispatch merely started late would leave the peak at 1 and the test would go
    /// green with the property already broken. A guardian's false negative is worse than
    /// its cost.
    const OVERLAP_RENDEZVOUS: std::time::Duration = std::time::Duration::from_millis(500);

    /// Records the peak number of `execute` calls that were ever in flight at once.
    struct OverlapProbeTool {
        /// Executions inside `execute` right now.
        live: Arc<std::sync::atomic::AtomicUsize>,
        /// High-water mark of `live`.
        peak: Arc<std::sync::atomic::AtomicUsize>,
        /// Total calls so far, which is what decides who waits.
        ///
        /// Separate from `live` on purpose: under sequential dispatch the second execution
        /// also finds `live == 1`, so keying the wait off `live` makes BOTH of them sit
        /// through the full window and doubles the test's fixed cost for nothing. Only the
        /// first call needs to offer a window for someone to overlap it.
        calls: Arc<std::sync::atomic::AtomicUsize>,
        /// Meeting point between the two executions. See the comment in `execute`.
        second_arrived: Arc<tokio::sync::Notify>,
    }

    #[async_trait]
    impl Tool for OverlapProbeTool {
        fn name(&self) -> &str {
            "overlap_probe"
        }
        fn description(&self) -> &str {
            "records overlap between tool executions"
        }
        fn input_schema(&self) -> Value {
            json!({"type": "object", "properties": {}})
        }

        async fn execute(&self, _args: Value, _cancel: &CancellationToken) -> ToolResult<Value> {
            use std::sync::atomic::Ordering;

            // SYNCHRONISATION, not a `sleep`. The first execution waits for a second one to
            // release it. Under SEQUENTIAL dispatch nobody ever does, the timeout expires,
            // and the peak stays at 1 — detected deterministically. Under PARALLEL dispatch
            // the second arrives while the first is still inside, so the peak reaches 2.
            //
            // A bare `sleep` would invert the failure: if a parallel scheduler took longer
            // than the window, the two executions would not overlap and the test would pass
            // green with the loop already parallelised.
            let ordinal = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            let now = self.live.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(now, Ordering::SeqCst);
            if ordinal == 1 {
                let _ =
                    tokio::time::timeout(OVERLAP_RENDEZVOUS, self.second_arrived.notified()).await;
            } else {
                self.second_arrived.notify_waiters();
            }
            self.live.fetch_sub(1, Ordering::SeqCst);
            Ok(json!({"ok": true}))
        }
    }

    /// Emits TWO `ToolUse` blocks in one assistant turn, then a plain text turn.
    struct TwoToolUseProvider {
        /// Turns served so far; the first is the two-tool turn.
        turn: Arc<std::sync::Mutex<u32>>,
    }

    #[async_trait]
    impl Provider for TwoToolUseProvider {
        async fn stream_messages(
            &self,
            _messages: &[Message],
            _tools: &[Box<dyn Tool>],
            _system: Option<&str>,
        ) -> Result<BoxStream<'static, Result<ResponseChunk>>> {
            let mut n = self.turn.lock().unwrap();
            *n += 1;
            let call = *n;
            drop(n);

            if call == 1 {
                let msg = Message {
                    role: Role::Assistant,
                    content: vec![
                        Content::ToolUse {
                            id: "seq-a".to_string(),
                            name: "overlap_probe".to_string(),
                            input: json!({}),
                        },
                        Content::ToolUse {
                            id: "seq-b".to_string(),
                            name: "overlap_probe".to_string(),
                            input: json!({}),
                        },
                    ],
                };
                Ok(Box::pin(stream::iter(vec![Ok(
                    ResponseChunk::MessageDone(msg),
                )])))
            } else {
                Ok(Box::pin(stream::iter(vec![
                    Ok(ResponseChunk::TextDelta("done".to_string())),
                    Ok(ResponseChunk::MessageDone(Message::assistant("done"))),
                ])))
            }
        }
    }

    /// SC-A20l: two `ToolUse` blocks from the same turn execute ONE AFTER THE OTHER.
    ///
    /// `tool_call_count` and `repeat_count` are flat locals whose correctness rests on this
    /// property, and MS2 adds a third one — the veto counter of REQ-A20c (Task 3.2).
    /// Parallelising the loop breaks all of them at once, and it would break them silently:
    /// nothing in their declarations says "sequential", which is why this test exists and
    /// why each declaration carries a comment pointing here.
    #[tokio::test]
    async fn tool_dispatch_is_sequential_within_a_turn() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let live = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));

        let provider = TwoToolUseProvider {
            turn: Arc::new(std::sync::Mutex::new(0)),
        };
        let mut agent = Agent::new(Arc::new(provider));
        agent.register_tool(Box::new(OverlapProbeTool {
            live: Arc::clone(&live),
            peak: Arc::clone(&peak),
            calls: Arc::new(AtomicUsize::new(0)),
            second_arrived: Arc::new(tokio::sync::Notify::new()),
        }));

        let (chunk_tx, _rx) = tokio::sync::mpsc::channel(64);
        let _ = agent
            .query_streaming("two tools", chunk_tx, AgentRunConfig::default())
            .await;

        assert_eq!(
            peak.load(Ordering::SeqCst),
            1,
            "the tool loop dispatched in parallel: the per-turn counters stop being correct",
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Task 3.2 — veto counter and the terminal rule (REQ-A20c)
    //
    // Five tests inherited from Task 3.1 (its Step 1 pasted them alongside its
    // own three pure ones), plus one written here to close SC-A20i, which the
    // reassigned five leave unnamed. `run_turn_with_autonomous_consults`
    // (PLURAL) is an orphan nobody defines in the plan — writing it is part of
    // this task's own Step 1, same as `agent_after_turn_ending_in` and
    // `run_turn_with_consults_on`, neither of which any prior task names.
    // ─────────────────────────────────────────────────────────────────────────

    /// The observable outcome of a turn, for the gate's tests.
    ///
    /// `tool_calls_counted` and `exit` are NOT in the plan's pasted contract
    /// block (`examples/ms2_contracts.rs`'s normative stub only lists
    /// `veto_count`/`consult_disabled_for_rest_of_turn`/`magi_calls`/
    /// `gate_log`) even though `a_veto_still_consumes_the_turn_budget` reads
    /// both — registered plan debt #7, verified against the code before
    /// adding these two fields.
    struct TurnOutcome {
        /// Vetoes that were actually EVALUATED and cut short (derived from
        /// `gate_log`, which only grows on a real evaluation — a call made
        /// after the door already closed never reaches `evaluate` at all).
        veto_count: usize,
        /// Whether the terminal rule (REQ-A20c) closed `consult` for the rest
        /// of the turn. Derived by replaying `gate_log` through the SAME
        /// reset-on-dispatch / increment-on-veto state machine
        /// `dispatch_consult_through_gate` runs in production, so it reflects
        /// the state at the END of the turn — a dispatch in the middle
        /// re-opens the door, exactly like SC-A20m requires.
        consult_disabled_for_rest_of_turn: bool,
        /// Real invocations of the `consult` double's `execute` — zero
        /// whenever the gate vetoed (SC-A20).
        magi_calls: usize,
        /// Lines the gate's telemetry sink recorded, in order (SC-A20h).
        gate_log: Vec<String>,
        /// How many `consult` calls the turn actually accounted for before
        /// terminating — vetoes, disabled-door continuations and dispatches
        /// all count (SC-A20f): every processed call produces exactly one
        /// `Content::ToolResult`, so counting those (not raw `ToolUse`
        /// blocks, which an overflowing call still leaves in history before
        /// erroring) gives the count `max_tool_calls` actually saw.
        tool_calls_counted: usize,
        /// How the run ended.
        exit: Exit,
    }

    /// Exit paths of a turn, for [`the_counter_dies_with_the_turn_on_every_exit_path`].
    ///
    /// `PartialEq`/`Eq` beyond the plan's pasted `#[derive(Debug, Clone,
    /// Copy)]`: `a_veto_still_consumes_the_turn_budget` asserts
    /// `out.exit == Exit::MaxToolCalls` via `assert_eq!`, which needs it.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Exit {
        FinalAnswer,
        MaxToolCalls,
        Cancelled,
        Error,
    }

    /// Test double standing in for `ConsultTool`: counts genuine invocations
    /// without calling any model, registered under [`CONSULT_TOOL_NAME`] so
    /// the gate's call-site distinction is exercised against the SAME name
    /// production code dispatches to.
    struct CountingConsultTool {
        calls: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl CountingConsultTool {
        /// Builds a fresh double and returns it alongside a handle to its
        /// call counter.
        fn new() -> (Self, Arc<std::sync::atomic::AtomicUsize>) {
            use std::sync::atomic::AtomicUsize;
            let calls = Arc::new(AtomicUsize::new(0));
            (
                Self {
                    calls: calls.clone(),
                },
                calls,
            )
        }
    }

    #[async_trait]
    impl Tool for CountingConsultTool {
        fn name(&self) -> &str {
            CONSULT_TOOL_NAME
        }
        fn description(&self) -> &str {
            "test double standing in for consult"
        }
        fn input_schema(&self) -> Value {
            json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "mode": {"type": "string"}
                },
                "required": ["query"]
            })
        }
        async fn execute(&self, _args: Value, _cancel: &CancellationToken) -> ToolResult<Value> {
            use std::sync::atomic::Ordering;
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(json!({"report": "ok", "degraded": false}))
        }
    }

    /// Test double for [`GateTelemetry`]: records every line the sink
    /// receives, in the exact format `log_gate` writes (SC-A20h).
    struct RecordingGateTelemetry {
        lines: Arc<std::sync::Mutex<Vec<String>>>,
    }

    impl GateTelemetry for RecordingGateTelemetry {
        fn on_gate_evaluation(&self, mode: &Mode, chars: usize, threshold: usize, vetoed: bool) {
            self.lines.lock().unwrap().push(format!(
                "gate {mode} chars={chars} threshold={threshold} {}",
                if vetoed { "veto" } else { "dispatch" }
            ));
        }
    }

    /// Emits ONE `ToolUse` of `consult` per entry of `script`, one PER TURN
    /// (a separate `stream_messages` call each), then a plain-text turn once
    /// `script` is exhausted so the loop ends normally.
    struct SequentialConsultProvider {
        script: Vec<String>,
        turn: std::sync::Mutex<usize>,
    }

    impl SequentialConsultProvider {
        fn new(contents: &[&str]) -> Self {
            Self {
                script: contents.iter().map(|s| (*s).to_string()).collect(),
                turn: std::sync::Mutex::new(0),
            }
        }
    }

    #[async_trait]
    impl Provider for SequentialConsultProvider {
        async fn stream_messages(
            &self,
            _messages: &[Message],
            _tools: &[Box<dyn Tool>],
            _system: Option<&str>,
        ) -> Result<BoxStream<'static, Result<ResponseChunk>>> {
            let mut n = self.turn.lock().unwrap();
            let idx = *n;
            *n += 1;
            drop(n);

            if let Some(q) = self.script.get(idx) {
                let msg = Message {
                    role: Role::Assistant,
                    content: vec![Content::ToolUse {
                        id: format!("seq-consult-{idx}"),
                        name: CONSULT_TOOL_NAME.to_string(),
                        input: json!({"query": q}),
                    }],
                };
                Ok(Box::pin(stream::iter(vec![Ok(
                    ResponseChunk::MessageDone(msg),
                )])))
            } else {
                Ok(Box::pin(stream::iter(vec![
                    Ok(ResponseChunk::TextDelta("done".to_string())),
                    Ok(ResponseChunk::MessageDone(Message::assistant("done"))),
                ])))
            }
        }
    }

    /// Emits ALL of `contents` as separate `ToolUse` blocks within the SAME
    /// assistant turn (one `stream_messages` call), then a plain-text turn
    /// (SC-A20i).
    struct AllAtOnceConsultProvider {
        script: Vec<String>,
        served: std::sync::Mutex<bool>,
    }

    impl AllAtOnceConsultProvider {
        fn new(contents: &[&str]) -> Self {
            Self {
                script: contents.iter().map(|s| (*s).to_string()).collect(),
                served: std::sync::Mutex::new(false),
            }
        }
    }

    #[async_trait]
    impl Provider for AllAtOnceConsultProvider {
        async fn stream_messages(
            &self,
            _messages: &[Message],
            _tools: &[Box<dyn Tool>],
            _system: Option<&str>,
        ) -> Result<BoxStream<'static, Result<ResponseChunk>>> {
            let mut served = self.served.lock().unwrap();
            if !*served {
                *served = true;
                drop(served);
                let content = self
                    .script
                    .iter()
                    .enumerate()
                    .map(|(i, q)| Content::ToolUse {
                        id: format!("batch-consult-{i}"),
                        name: CONSULT_TOOL_NAME.to_string(),
                        input: json!({"query": q}),
                    })
                    .collect();
                let msg = Message {
                    role: Role::Assistant,
                    content,
                };
                Ok(Box::pin(stream::iter(vec![Ok(
                    ResponseChunk::MessageDone(msg),
                )])))
            } else {
                Ok(Box::pin(stream::iter(vec![
                    Ok(ResponseChunk::TextDelta("done".to_string())),
                    Ok(ResponseChunk::MessageDone(Message::assistant("done"))),
                ])))
            }
        }
    }

    /// Emits ONE `ToolUse` per `(query, mode)` pair, one per turn — the
    /// `mode` travels in the tool input exactly as the agent's own choice via
    /// the `input_schema` would (REQ-A07b, `agent_chosen_mode`).
    struct ScriptedModeConsultProvider {
        script: Vec<(String, Mode)>,
        turn: std::sync::Mutex<usize>,
    }

    impl ScriptedModeConsultProvider {
        fn new(pairs: &[(&str, Mode)]) -> Self {
            Self {
                script: pairs.iter().map(|(q, m)| ((*q).to_string(), *m)).collect(),
                turn: std::sync::Mutex::new(0),
            }
        }
    }

    #[async_trait]
    impl Provider for ScriptedModeConsultProvider {
        async fn stream_messages(
            &self,
            _messages: &[Message],
            _tools: &[Box<dyn Tool>],
            _system: Option<&str>,
        ) -> Result<BoxStream<'static, Result<ResponseChunk>>> {
            let mut n = self.turn.lock().unwrap();
            let idx = *n;
            *n += 1;
            drop(n);

            if let Some((q, m)) = self.script.get(idx) {
                let msg = Message {
                    role: Role::Assistant,
                    content: vec![Content::ToolUse {
                        id: format!("mode-consult-{idx}"),
                        name: CONSULT_TOOL_NAME.to_string(),
                        input: json!({"query": q, "mode": m.to_string()}),
                    }],
                };
                Ok(Box::pin(stream::iter(vec![Ok(
                    ResponseChunk::MessageDone(msg),
                )])))
            } else {
                Ok(Box::pin(stream::iter(vec![
                    Ok(ResponseChunk::TextDelta("done".to_string())),
                    Ok(ResponseChunk::MessageDone(Message::assistant("done"))),
                ])))
            }
        }
    }

    /// Emits one vetoable `ToolUse` per entry of `contents` (closing the door
    /// after two, if `contents` has at least two trivial entries), then
    /// concludes the turn according to `exit` — all four exit paths converge
    /// on the SAME provider so `agent_after_turn_ending_in` can drive them
    /// uniformly.
    struct ClosingTurnProvider {
        contents: Vec<String>,
        exit: Exit,
        turn: std::sync::Mutex<usize>,
    }

    impl ClosingTurnProvider {
        fn new(contents: &[&str], exit: Exit) -> Self {
            Self {
                contents: contents.iter().map(|s| (*s).to_string()).collect(),
                exit,
                turn: std::sync::Mutex::new(0),
            }
        }
    }

    #[async_trait]
    impl Provider for ClosingTurnProvider {
        async fn stream_messages(
            &self,
            _messages: &[Message],
            _tools: &[Box<dyn Tool>],
            _system: Option<&str>,
        ) -> Result<BoxStream<'static, Result<ResponseChunk>>> {
            let mut n = self.turn.lock().unwrap();
            let idx = *n;
            *n += 1;
            drop(n);

            let q = if idx < self.contents.len() {
                Some(self.contents[idx].clone())
            } else if matches!(self.exit, Exit::MaxToolCalls) {
                // Keep hammering the (already closed) door so the CAP — not
                // an early final answer — is what ends this turn.
                self.contents.last().cloned()
            } else {
                None
            };

            if let Some(q) = q {
                let msg = Message {
                    role: Role::Assistant,
                    content: vec![Content::ToolUse {
                        id: format!("closing-{idx}"),
                        name: CONSULT_TOOL_NAME.to_string(),
                        input: json!({"query": q}),
                    }],
                };
                return Ok(Box::pin(stream::iter(vec![Ok(
                    ResponseChunk::MessageDone(msg),
                )])));
            }

            match self.exit {
                Exit::FinalAnswer | Exit::MaxToolCalls => Ok(Box::pin(stream::iter(vec![
                    Ok(ResponseChunk::TextDelta("done".to_string())),
                    Ok(ResponseChunk::MessageDone(Message::assistant("done"))),
                ]))),
                Exit::Cancelled => Err(anyhow::anyhow!("provider aborted: cancelled by timeout")),
                Exit::Error => Err(anyhow::anyhow!("provider aborted: simulated upstream failure")),
            }
        }
    }

    /// Shared core: runs one turn on `agent` with a freshly wired gate-log
    /// sink and `consult` double, and assembles the [`TurnOutcome`].
    async fn run_and_observe(
        agent: &mut Agent,
        config: AgentRunConfig,
        magi_calls: &Arc<std::sync::atomic::AtomicUsize>,
        gate_log_sink: &Arc<std::sync::Mutex<Vec<String>>>,
    ) -> anyhow::Result<TurnOutcome> {
        use std::sync::atomic::Ordering;

        let (tx, _rx) = tokio::sync::mpsc::channel(64);
        let result = agent.query_streaming("start", tx, config).await;

        let exit = match &result {
            Ok(_) => Exit::FinalAnswer,
            Err(e) if e.to_string() == MAX_TOOL_CALLS_ERROR => Exit::MaxToolCalls,
            Err(e) if e.to_string().contains("cancelled") => Exit::Cancelled,
            Err(_) => Exit::Error,
        };

        let gate_log = gate_log_sink.lock().unwrap().clone();
        let veto_count = gate_log.iter().filter(|l| l.ends_with("veto")).count();
        let mut running = 0usize;
        for line in &gate_log {
            if line.ends_with("veto") {
                running += 1;
            } else {
                running = 0;
            }
        }
        let consult_disabled_for_rest_of_turn = running >= usize::from(MAX_CONSECUTIVE_VETOES);

        let tool_calls_counted = agent
            .history()
            .iter()
            .flat_map(|m| &m.content)
            .filter(|c| matches!(c, Content::ToolResult { .. }))
            .count();

        Ok(TurnOutcome {
            veto_count,
            consult_disabled_for_rest_of_turn,
            magi_calls: magi_calls.load(Ordering::SeqCst),
            gate_log,
            tool_calls_counted,
            exit,
        })
    }

    /// Corre UN turno con los contenidos dados como consults autorruteados,
    /// uno por turno (SC-A20f/SC-A20m).
    async fn run_turn_with_consults(contents: &[&str]) -> anyhow::Result<TurnOutcome> {
        let provider = SequentialConsultProvider::new(contents);
        let (tool, magi_calls) = CountingConsultTool::new();
        let mut agent = Agent::new(Arc::new(provider));
        agent.register_tool(Box::new(tool));
        let gate_log_sink = Arc::new(std::sync::Mutex::new(Vec::new()));
        let config = AgentRunConfig {
            gate_telemetry: Arc::new(RecordingGateTelemetry {
                lines: gate_log_sink.clone(),
            }),
            ..AgentRunConfig::default()
        };
        run_and_observe(&mut agent, config, &magi_calls, &gate_log_sink).await
    }

    /// Igual, pero emitiendo los `ToolUse` en UN solo bloque de respuesta
    /// (SC-A20i).
    async fn run_turn_with_two_tooluse_blocks(contents: &[&str]) -> anyhow::Result<TurnOutcome> {
        let provider = AllAtOnceConsultProvider::new(contents);
        let (tool, magi_calls) = CountingConsultTool::new();
        let mut agent = Agent::new(Arc::new(provider));
        agent.register_tool(Box::new(tool));
        let gate_log_sink = Arc::new(std::sync::Mutex::new(Vec::new()));
        let config = AgentRunConfig {
            gate_telemetry: Arc::new(RecordingGateTelemetry {
                lines: gate_log_sink.clone(),
            }),
            ..AgentRunConfig::default()
        };
        run_and_observe(&mut agent, config, &magi_calls, &gate_log_sink).await
    }

    /// Corre un turno con un único consult autorruteado.
    async fn run_turn_with_autonomous_consult(content: &str) -> anyhow::Result<TurnOutcome> {
        run_turn_with_consults(&[content]).await
    }

    /// **PLURAL** — an orphan the plan names in several tasks but nobody
    /// defines (the pre-flight sweep listed it as such); writing it is this
    /// task's own Step 1. Each `(query, mode)` pair is dispatched on its OWN
    /// turn, with `mode` carried in the tool input as the AGENT's own choice
    /// (`agent_chosen_mode`), not a human `--mode`.
    async fn run_turn_with_autonomous_consults(
        pairs: &[(&str, Mode)],
    ) -> anyhow::Result<TurnOutcome> {
        let provider = ScriptedModeConsultProvider::new(pairs);
        let (tool, magi_calls) = CountingConsultTool::new();
        let mut agent = Agent::new(Arc::new(provider));
        agent.register_tool(Box::new(tool));
        let gate_log_sink = Arc::new(std::sync::Mutex::new(Vec::new()));
        let config = AgentRunConfig {
            gate_telemetry: Arc::new(RecordingGateTelemetry {
                lines: gate_log_sink.clone(),
            }),
            ..AgentRunConfig::default()
        };
        run_and_observe(&mut agent, config, &magi_calls, &gate_log_sink).await
    }

    /// Ends a FIRST turn via each of the four exit paths, closing the door
    /// with two consecutive vetoes beforehand whenever `contents` has (at
    /// least) two trivial entries — so all four scenarios genuinely close the
    /// door before concluding, not just the ones that happen to need it.
    /// Returns the `Agent` wrapped for reuse across a SECOND turn
    /// ([`run_turn_with_consults_on`]).
    async fn agent_after_turn_ending_in(
        exit: Exit,
        contents: &[&str],
    ) -> tokio::sync::Mutex<Agent> {
        let provider = ClosingTurnProvider::new(contents, exit);
        let (tool, _calls) = CountingConsultTool::new();
        let mut agent = Agent::new(Arc::new(provider));
        agent.register_tool(Box::new(tool));

        let max_tool_calls = if matches!(exit, Exit::MaxToolCalls) {
            contents.len()
        } else {
            DEFAULT_MAX_TOOL_CALLS
        };
        let config = AgentRunConfig {
            max_tool_calls,
            ..AgentRunConfig::default()
        };
        let (tx, _rx) = tokio::sync::mpsc::channel(64);
        let _ = agent.query_streaming("first turn", tx, config).await;
        tokio::sync::Mutex::new(agent)
    }

    /// Runs a FRESH turn on an `Agent` reused from [`agent_after_turn_ending_in`],
    /// with its own provider/tool/telemetry, so a stale counter from the prior
    /// turn would be the only thing that could make this one see it.
    async fn run_turn_with_consults_on(
        agent: &tokio::sync::Mutex<Agent>,
        contents: &[&str],
    ) -> anyhow::Result<TurnOutcome> {
        let mut guard = agent.lock().await;
        let provider = SequentialConsultProvider::new(contents);
        guard.set_provider(Arc::new(provider));
        let (tool, magi_calls) = CountingConsultTool::new();
        guard.register_or_replace_tool(Box::new(tool));
        let gate_log_sink = Arc::new(std::sync::Mutex::new(Vec::new()));
        let config = AgentRunConfig {
            gate_telemetry: Arc::new(RecordingGateTelemetry {
                lines: gate_log_sink.clone(),
            }),
            ..AgentRunConfig::default()
        };
        run_and_observe(&mut guard, config, &magi_calls, &gate_log_sink).await
    }

    /// SC-A20 / SC-A20c: the gate vetoes the autonomous route without erroring.
    #[tokio::test]
    async fn the_gate_vetoes_the_autonomous_route_without_erroring() {
        let outcome = run_turn_with_autonomous_consult("trivial").await;
        assert!(
            outcome.is_ok(),
            "the veto comes back as a normal ToolResult, not an error"
        );
        assert_eq!(outcome.unwrap().magi_calls, 0, "zero model calls");
    }

    /// SC-A20e: the veto message discourages retrying and never names the
    /// threshold.
    #[tokio::test]
    async fn the_veto_message_discourages_retry_without_naming_the_threshold() {
        let msg = veto_message(&Mode::Analysis);
        assert!(
            msg.to_lowercase().contains("same result"),
            "must say retrying is pointless"
        );
        assert!(
            !msg.contains(&magi_rs::magi::GATE_ANALYSIS.to_string()),
            "revealing how many characters are missing is a direct invitation to pad content"
        );
    }

    /// SC-A20f / SC-A20m: two CONSECUTIVE vetoes are terminal; a success in
    /// between resets.
    #[tokio::test]
    async fn two_consecutive_vetoes_are_terminal_but_a_success_resets() {
        let out = run_turn_with_consults(&["trivial", "tambien trivial"])
            .await
            .unwrap();
        assert!(out.consult_disabled_for_rest_of_turn);

        let long = "x".repeat(magi_rs::magi::GATE_ANALYSIS + 10);
        let out = run_turn_with_consults(&["trivial", &long, "trivial otra vez"])
            .await
            .unwrap();
        assert!(
            !out.consult_disabled_for_rest_of_turn,
            "veto → runs → veto is not a loop: it's a turn with one trivial question and \
             one that genuinely warranted consensus"
        );
    }

    /// A veto does NOT loosen the turn's caps. Complements SC-A20f from the
    /// other side: SC-A20f pins that the consult PATH closes on the second
    /// veto; this pins that the WHOLE TURN stays capped in the meantime —
    /// every processed call (veto, disabled continuation, or dispatch)
    /// consumes exactly one `max_tool_calls` slot.
    #[tokio::test]
    async fn a_veto_still_consumes_the_turn_budget() {
        // Many more invocations than `max_tool_calls`, all vetoable.
        let script: Vec<(&str, Mode)> = vec![("x", Mode::Analysis); 40];
        let out = run_turn_with_autonomous_consults(&script).await.unwrap();

        assert!(
            out.tool_calls_counted >= usize::from(MAX_CONSECUTIVE_VETOES),
            "each veto counts as an invocation: max_tool_calls counts what the model asked for"
        );
        // TERMINATION, not a lower bound: 40 vetoable invocations against a
        // `max_tool_calls` of 15 MUST end exactly at the cap.
        assert_eq!(out.exit, Exit::MaxToolCalls);
        assert_eq!(
            out.tool_calls_counted, DEFAULT_MAX_TOOL_CALLS,
            "all 15 were consumed: vetoes AND denials count alike"
        );
    }

    /// SC-A20g: the counter dies with the turn, through all FOUR exit paths.
    #[tokio::test]
    async fn the_counter_dies_with_the_turn_on_every_exit_path() {
        for exit in [
            Exit::FinalAnswer,
            Exit::MaxToolCalls,
            Exit::Cancelled,
            Exit::Error,
        ] {
            let agent = agent_after_turn_ending_in(exit, &["trivial", "trivial"]).await;
            let out = run_turn_with_consults_on(&agent, &["trivial"])
                .await
                .unwrap();
            assert!(
                !out.consult_disabled_for_rest_of_turn,
                "turn {exit:?}: must not persist"
            );
        }
    }

    /// SC-A20h: every evaluation is logged — in EVERY surface, with or
    /// without an observer.
    #[tokio::test]
    async fn every_gate_evaluation_is_logged() {
        let long = "x".repeat(magi_rs::magi::GATE_ANALYSIS + 1);
        let log = run_turn_with_consults(&["trivial", &long])
            .await
            .unwrap()
            .gate_log;
        // Exercised WITHOUT an observer, which is the TUI's configuration:
        // the gate's telemetry cannot depend on a channel the highest-traffic
        // surface does not have.
        assert!(
            !log.is_empty(),
            "without a RunObserver the telemetry must still be recorded — that coupling was \
             the bug"
        );
        assert_eq!(log.len(), 2, "both the veto and the pass get recorded");
        assert!(log[0].contains("analysis") && log[0].contains("veto"));
        assert!(
            log[0].contains(&magi_rs::magi::GATE_ANALYSIS.to_string()),
            "SC-A20h requires the APPLIED threshold: the length alone doesn't say which side \
             it fell on, and calibrating is exactly comparing the two"
        );
        assert!(
            log[1].contains(&magi_rs::magi::GATE_ANALYSIS.to_string()),
            "also on the line that dispatches: one side alone doesn't calibrate anything"
        );
    }

    /// SC-A20i: several `ToolUse` blocks in the SAME turn count exactly like
    /// separate turns, and two concurrent sessions never contaminate each
    /// other — proven by actually running them concurrently, not just
    /// asserted from the shape of the code.
    #[tokio::test]
    async fn multiple_tooluse_blocks_in_one_turn_count_like_separate_turns() {
        let out = run_turn_with_two_tooluse_blocks(&["trivial", "tambien trivial"])
            .await
            .unwrap();
        assert_eq!(
            out.veto_count, 2,
            "both blocks of the same turn get evaluated and vetoed"
        );
        assert!(out.consult_disabled_for_rest_of_turn);

        let long = "x".repeat(magi_rs::magi::GATE_ANALYSIS + 5);
        let (closed_door, clean_session) = tokio::join!(
            run_turn_with_two_tooluse_blocks(&["trivial", "trivial"]),
            run_turn_with_consults(&[long.as_str()]),
        );
        assert!(closed_door.unwrap().consult_disabled_for_rest_of_turn);
        assert!(
            !clean_session.unwrap().consult_disabled_for_rest_of_turn,
            "the other session never even noticed"
        );
    }

    /// Task 3.2's namesake test. `authorize_and_execute_tool` (the forced
    /// consult injection, REQ-H22) and the `ToolUse` loop (the model's own
    /// choice) are two DISTINCT entrances to the same `consult` tool, and
    /// only the second passes through the complexity gate (REQ-A20). A prior
    /// attempt at this test (Task 3.1) simulated both sides with hardcoded
    /// literals and could not fail under any change — see the `#[cfg(test)]`
    /// comment in `src/magi/gate.rs` naming this task as the gap's owner.
    /// This drives BOTH real call sites against a real `Agent`.
    #[tokio::test]
    async fn a_forced_injection_bypasses_the_gate_while_a_model_choice_does_not() {
        use std::sync::atomic::Ordering;

        // (a) FORCED: `force_consult = true` injects consult via
        // `authorize_and_execute_tool` directly, BEFORE the loop even starts —
        // never touching `dispatch_consult_through_gate`. Trivial content
        // that would be vetoed anywhere else must still dispatch here.
        let (forced_tool, forced_calls) = CountingConsultTool::new();
        let mut forced_agent = Agent::new(Arc::new(MockProvider));
        forced_agent.register_tool(Box::new(forced_tool));
        let (tx, _rx) = tokio::sync::mpsc::channel(64);
        let forced_config = AgentRunConfig {
            force_consult: true,
            ..AgentRunConfig::default()
        };
        forced_agent
            .query_streaming("trivial", tx, forced_config)
            .await
            .unwrap();
        assert_eq!(
            forced_calls.load(Ordering::SeqCst),
            1,
            "the forced injection must dispatch even though its content is trivial: REQ-A20 \
             forbids vetoing it — it never reaches the gate at all"
        );

        // (b) MODEL-ISSUED: the SAME trivial content, requested by the model
        // through the ordinary `ToolUse` loop, DOES reach the gate and gets
        // vetoed.
        let (model_tool, model_calls) = CountingConsultTool::new();
        let provider = SequentialConsultProvider::new(&["trivial"]);
        let mut model_agent = Agent::new(Arc::new(provider));
        model_agent.register_tool(Box::new(model_tool));
        let (tx2, _rx2) = tokio::sync::mpsc::channel(64);
        model_agent
            .query_streaming("start", tx2, AgentRunConfig::default())
            .await
            .unwrap();
        assert_eq!(
            model_calls.load(Ordering::SeqCst),
            0,
            "the model's own request for the SAME trivial content must be vetoed: it enters \
             through the ToolUse loop, which the gate DOES see"
        );
    }
}
