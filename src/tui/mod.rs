//! This module implements the Terminal User Interface using Ratatui.

use crate::agent::{Agent, AgentRunConfig, ApprovalRequest, StreamPiece};
use crossterm::{
    event::{self, DisableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use magi_core::error::MagiError;
use magi_core::reporting::MagiReport;
use magi_core::schema::Mode;
use magi_rs::magi::kind::ProviderKind;
use magi_rs::magi::mode::{normalize_label, resolve_mode_guarded, ModeClassifier, ModeError};
use magi_rs::redact::redact_foreign_error;
use magi_rs::vault::{SecretStore, VaultError};
use ratatui::{
    backend::{Backend, CrosstermBackend},
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
    Frame, Terminal,
};
use std::io;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

/// A vault-backed secret store shared with `main.rs`, used by the `/login`
/// and `/logout` handlers. `None` in an ephemeral (no-persistence) session.
///
/// The concrete `VaultStore` behind this trait object is `Send` but
/// deliberately **not** `Sync` (its mask rotates on every access); the
/// `Mutex` supplies the exclusion `Mutex<T>: Sync` requires of `T: Send`.
pub type SharedSecretStore = Arc<Mutex<dyn SecretStore + Send>>;

/// Persists a freshly-minted `ANTHROPIC_API_KEY` in the vault (SC-V36).
///
/// Extracted from the `/login` event handler so the storage step is
/// unit-testable without driving a real OAuth flow or terminal (MAGI run 8).
/// A poisoned `store` mutex is recovered (`into_inner`) rather than
/// panicking, mirroring the `seal`/`unseal` poison-recovery pattern used
/// elsewhere in this codebase (`system::database`).
///
/// # Returns
/// [`AgentResponse::Info`] on success (stored, or — with no vault attached —
/// an explicit "ephemeral session" notice, never a silent no-op);
/// [`AgentResponse::Error`] if the vault write itself fails.
fn handle_login(store: Option<&SharedSecretStore>, api_key: &str) -> AgentResponse {
    match store {
        Some(ss) => {
            let mut guard = ss.lock().unwrap_or_else(|p| p.into_inner());
            match guard.set("ANTHROPIC_API_KEY", api_key) {
                Ok(()) => AgentResponse::Info("API key stored in the vault.".to_string()),
                Err(e) => AgentResponse::Error(e.to_string()),
            }
        }
        None => AgentResponse::Info("ephemeral session: key not persisted".to_string()),
    }
}

/// Removes `ANTHROPIC_API_KEY` from the vault (the `/logout` analogue of
/// [`handle_login`]). An absent key, or no vault attached at all, reports
/// "no stored session" rather than an error — logging out twice, or logging
/// out of an ephemeral session, is not a failure.
fn handle_logout(store: Option<&SharedSecretStore>) -> AgentResponse {
    match store {
        Some(ss) => {
            let mut guard = ss.lock().unwrap_or_else(|p| p.into_inner());
            match guard.remove("ANTHROPIC_API_KEY") {
                Ok(()) => AgentResponse::Info("Logged out successfully.".to_string()),
                Err(VaultError::SecretNotFound(_)) => {
                    AgentResponse::Info("no stored session".to_string())
                }
                Err(e) => AgentResponse::Error(e.to_string()),
            }
        }
        None => AgentResponse::Info("no stored session".to_string()),
    }
}

/// Fallback text for a `/consult` issued with no trio available AND no precomputed
/// reason (Task 4.3, REQ-A06/SC-A06b).
///
/// Structurally unreachable given `run()`'s invariant — `consult` is `None` if and
/// only if building the MAGI trio failed, and that failure always produces a message
/// via `trio_unavailable_message` — kept anyway so a future change to that invariant
/// degrades to a generic-but-honest reply instead of silently sending nothing (B9).
const CONSULT_UNAVAILABLE_FALLBACK: &str =
    "MAGI consult is unavailable for this session: no trio was built.";

/// The `/consult` response when no MAGI trio is available for this session (Task 4.3,
/// REQ-A06, SC-A06b).
///
/// Extracted from the `UiEvent::Consult` handler for the same reason as
/// [`handle_login`]/[`handle_trio_rebuild_failure`]: the full TUI event loop is
/// intractable to drive in a test, so the exact text this arm sends is verified here
/// as a plain `fn`. `reason` must be the SAME text already pushed to the startup
/// notices (via `trio_unavailable_message` in `main.rs`) — never a second,
/// independently-worded message, or the notice and the `/consult` reply would look
/// like two unrelated problems to the user.
fn consult_unavailable_response(reason: &str) -> AgentResponse {
    AgentResponse::Error(reason.to_string())
}

/// The `/consult` response BODY for a successful `MagiReport` (REQ-A12c, SC-A12f, fix
/// round 4, finding 2).
///
/// **Before this fix, `UiEvent::Consult` built its body from `report.report` directly
/// — bypassing `annotate_report_text` entirely.** A human typing `/consult` is
/// arguably the most direct "first use" surface REQ-A12c describes, and they got NONE
/// of this task's keyless-auth guidance: a partial failure (`degraded = true`) with a
/// seat rejected on auth under a keyless kind rendered as an unexplained `[DEGRADED:
/// …]` banner over the raw report, same as any other partial failure.
///
/// Reuses [`crate::tools::consult::annotate_report_text`] — the SAME function
/// `ConsultTool::execute`/`analyze_direct` already call — so the TUI never carries its
/// own, fourth copy of this wording (B3). The `[DEGRADED: …]` banner is TUI-only
/// presentation (it is not part of the JSON `report`/`degraded` shape the other two
/// surfaces emit) and stays here, prepended to the shared annotated text.
///
/// Extracted from the `UiEvent::Consult` handler for the same reason as
/// [`consult_unavailable_response`]/[`handle_login`]: the full TUI event loop is
/// intractable to drive in a test.
fn tui_consult_success_body(report: &MagiReport, kind: ProviderKind) -> String {
    let annotated = crate::tools::consult::annotate_report_text(report, kind);
    if report.degraded {
        format!(
            "[DEGRADED: fewer than 3 agents responded — consensus may be unreliable]\n\n{annotated}"
        )
    } else {
        annotated
    }
}

/// The `/consult` response BODY for a failed `Magi::analyze()` call (REQ-A12c,
/// SC-A12f, fix round 4, finding 2).
///
/// **Before this fix, `UiEvent::Consult` sent a hardcoded generic string** — "MAGI
/// consult failed — check your provider/credentials and try again." — on EVERY
/// failure, including a total failure (`MagiError::InsufficientAgents`) under a
/// keyless kind, where [`crate::tools::consult::explain_magi_error`] would have named
/// the configuration as a probable cause. Reuses that SAME function — the one
/// `ConsultTool::execute`/`analyze_direct` already call — so the hint reads
/// identically on all three surfaces (B3), never a fourth wording.
///
/// `explain_magi_error` already redacts the underlying `err` via
/// `redact_foreign_error` before this function ever sees it (B11) — nothing here
/// needs to redact again.
///
/// Extracted for the same reason as [`tui_consult_success_body`].
fn tui_consult_error_body(err: &MagiError, kind: ProviderKind) -> String {
    format!(
        "MAGI consult failed: {}",
        crate::tools::consult::explain_magi_error(err, kind)
    )
}

/// Handles a failed post-`/login` MAGI trio rebuild (I4, fix round 2).
///
/// Extracted from the `/login` event handler for the same reason as
/// [`handle_login`]/[`handle_logout`]: the full TUI event loop is intractable
/// to test directly, so the logic that decides what happens on a rebuild
/// failure is tested here as a plain `fn`.
///
/// Two things must happen together, and both are about not lying to the
/// user by omission:
///
/// - **The error text is redacted.** `e` is a foreign [`ProviderError`] from
///   magi-core — its `Display` can cite the endpoint URL, and magi-core does
///   not know this crate's redaction rule (REQ-A16c). Every other foreign-
///   error surface in this codebase routes through [`redact_foreign_error`]
///   for exactly this reason; this one must too.
/// - **Nothing keeps using the OLD credentials.** By the time this runs,
///   [`handle_login`] already wrote a NEW key to the vault, so the running
///   session and the vault now disagree about what the current credential
///   is. Leaving `consult_magi_runner` and the registered `consult` tool
///   pointed at whatever built successfully before this attempt — possibly
///   nothing, possibly a stale provider from an earlier session — would let
///   consult keep answering under a diverged credential while the user was
///   told the rebuild failed. Dropping both makes the direct `/consult` path
///   and the autonomous tool path fail closed the same way an unconfigured
///   trio always does (REQ-A06), instead of silently using the wrong thing.
///
/// [`ProviderError`]: magi_core::error::ProviderError
fn handle_trio_rebuild_failure(
    e: &magi_core::error::ProviderError,
    consult_magi_runner: &mut Option<Arc<magi_core::orchestrator::Magi>>,
    runner_agent: &mut Agent,
) -> magi_rs::redact::SafeErrorText {
    *consult_magi_runner = None;
    runner_agent.remove_tool("consult");
    redact_foreign_error(e)
}

/// Different interaction modes for the TUI.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum AppMode {
    Normal,
    Selection,
    Visual, // Mode for selecting text within a message
}

/// Parses a trimmed input line as a `/consult` command. `Some(query)` for
/// `/consult <query>` (empty string for bare `/consult`), `None` otherwise.
/// Requires a space boundary so `/consultation` is treated as normal input.
pub(crate) fn parse_consult_command(trimmed: &str) -> Option<&str> {
    let rest = trimmed.strip_prefix("/consult")?;
    if rest.is_empty() {
        return Some("");
    }
    Some(rest.strip_prefix(' ')?.trim())
}

/// Flag that declares an explicit `--mode` on the TUI's `/consult` (REQ-A07b).
const TUI_MODE_FLAG: &str = "--mode";

/// Why `/consult`'s flags failed to parse (REQ-A07b/A07d).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TuiConsultParseError {
    /// The line was not a `/consult` command at all.
    NotAConsultCommand,
    /// `--mode` appeared with no value after it.
    MissingModeValue,
    /// The `--mode` value does not name one of the three valid modes.
    UnknownMode(String),
    /// Any other `--flag`-shaped leading token — in particular
    /// `--untrusted-content`: the TUI never exposes that mark (SC-A07t) because a
    /// human already chose the content here, so the classification guard it would
    /// gate doesn't apply.
    UnsupportedFlag(String),
}

/// A parsed `/consult` command (REQ-A07b): the explicit mode, if any, and the
/// free-text query with the recognized flags stripped out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TuiConsultCommand {
    /// Explicit mode declared with `--mode`; `None` when omitted.
    pub mode: Option<Mode>,
    /// The query text, with any recognized leading flags removed.
    pub query: String,
}

/// Splits `s` into its first whitespace-delimited token and the (left-trimmed)
/// remainder, or `None` if `s` is empty after trimming leading whitespace.
///
/// `O(n)` in `s.len()`: a single forward scan for the first whitespace byte.
fn split_leading_token(s: &str) -> Option<(&str, &str)> {
    let s = s.trim_start();
    if s.is_empty() {
        return None;
    }
    match s.find(char::is_whitespace) {
        Some(i) => Some((&s[..i], s[i..].trim_start())),
        None => Some((s, "")),
    }
}

/// Parses `/consult [--mode <value>] <query>` — the only flag this surface
/// exposes (REQ-A07b). Any other `--flag`-shaped leading token is rejected,
/// `--untrusted-content` included: that mark belongs to the surfaces a gate
/// automates (CLI, envelope), never to the TUI, where a human already chose the
/// content (REQ-A07d/SC-A07t).
///
/// `--mode`'s value is validated with the same closed, model-output
/// normalization as an inferred label ([`normalize_label`]) rather than the
/// config rule: a human typing at a prompt is closer to that case than to a
/// `magi.toml` value, and reusing it avoids a third, slightly different
/// acceptance rule for the same three labels.
///
/// A REPEATED `--mode` is last-value-wins (each occurrence overwrites the
/// previous one) — a DELIBERATE mirror of clap's own default behavior for a
/// non-multiple arg (`HeadlessArgs.mode: Option<CliMode>` on the CLI surface
/// behaves identically), not an oversight, despite this parser otherwise
/// failing closed on every unrecognized `--flag`.
///
/// # Errors
/// See [`TuiConsultParseError`]'s variants.
///
/// Consumed in production by the `KeyCode::Enter` handler in [`run_app`] (REQ-A07d): it
/// replaces the older `parse_consult_command`, which only stripped the `/consult ` prefix
/// and left any `--mode design` text embedded in the query — the flag never worked and
/// silently polluted the prompt. Also covered by `every_surface_accepts_an_explicit_mode` and
/// `untrusted_content_is_declarable_where_the_threat_lives` (in `main.rs`), plus the
/// edge-case tests in this module's `tests`.
pub(crate) fn parse_tui_consult(trimmed: &str) -> Result<TuiConsultCommand, TuiConsultParseError> {
    let mut rest =
        parse_consult_command(trimmed).ok_or(TuiConsultParseError::NotAConsultCommand)?;
    let mut mode = None;

    while let Some((token, after)) = split_leading_token(rest) {
        if token == TUI_MODE_FLAG {
            let (value, after_value) =
                split_leading_token(after).ok_or(TuiConsultParseError::MissingModeValue)?;
            mode = Some(
                normalize_label(value)
                    .ok_or_else(|| TuiConsultParseError::UnknownMode(value.to_string()))?,
            );
            rest = after_value;
            continue;
        }
        if token.starts_with("--") {
            return Err(TuiConsultParseError::UnsupportedFlag(token.to_string()));
        }
        break;
    }

    Ok(TuiConsultCommand {
        mode,
        query: rest.to_string(),
    })
}

/// Resolves the effective mode for a parsed `/consult` command and hands back the
/// (already flag-stripped) query alongside it — the exact pair the `UiEvent::Consult`
/// handler passes to `Magi::analyze` (REQ-A07d).
///
/// Extracted into its own function, same precedent as `handle_login`/`handle_logout`
/// above: the full TUI event loop is intractable to test directly, so the logic that
/// decides the mode is tested here as a plain `async fn`.
///
/// `agent_chosen` is always `None`: `/consult` is a human-written command, never something
/// the agent routed to on its own, so there is no third level to feed in. `untrusted` comes
/// from the OPERATOR's `magi.toml` only — never from this command line, because
/// [`parse_tui_consult`] already rejects `--untrusted-content` here (SC-A07t): a human
/// already chose the content, so the surface never carries the mark.
///
/// # Errors
/// [`ModeError::UntrustedContentRequiresExplicitMode`] if the operator declared
/// `untrusted_content = true` in `[magi]` and neither `cmd.mode` nor `default_mode` names a
/// lens (REQ-A07r) — the classification level is exactly what that configuration blocks.
async fn resolve_tui_consult_mode(
    cmd: TuiConsultCommand,
    default_mode: Option<Mode>,
    untrusted_content: bool,
    classifier: &dyn ModeClassifier,
) -> Result<(Mode, String), ModeError> {
    let resolution = resolve_mode_guarded(
        cmd.mode,
        default_mode,
        None,
        untrusted_content,
        Some(classifier),
        &cmd.query,
    )
    .await?;
    Ok((resolution.mode, cmd.query))
}

/// Braille spinner frames for the "thinking" activity indicator.
pub(crate) const SPINNER_FRAMES: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// Advances the spinner frame index, wrapping around the frame set.
pub(crate) fn next_spinner_frame(frame: usize) -> usize {
    (frame + 1) % SPINNER_FRAMES.len()
}

/// The compact "thinking" indicator line, ending with the current spinner glyph.
pub(crate) fn thinking_indicator(frame: usize) -> String {
    format!(
        "🤔 MAGI Pensando… {}",
        SPINNER_FRAMES[frame % SPINNER_FRAMES.len()]
    )
}

/// True if `trimmed` is the `/toggle-show-thinking` command (toggles between the
/// compact thinking indicator and the verbose reasoning stream).
pub(crate) fn parse_toggle_show_thinking(trimmed: &str) -> bool {
    trimmed.trim() == "/toggle-show-thinking"
}

/// True if `trimmed` is the `/init-config` command. Mirrors `parse_toggle_show_thinking`.
///
/// **Still recognized after retirement (REQ-A22)** — only what happens once recognized
/// changed, from scaffolding a `magi.toml` to showing [`init_config_retired_message`].
/// Making this return `false` instead would fall through to sending the literal text
/// `/init-config` to the agent as a chat message, which is a worse outcome than a clear
/// retirement notice.
pub(crate) fn parse_init_config(trimmed: &str) -> bool {
    trimmed.trim() == "/init-config"
}

/// Message shown when `/init-config` is used (REQ-A22, SC-A22): the command is
/// retired, and this names the replacement instead of silently doing nothing or
/// falling through to being sent to the agent — same treatment as `reject_init_config`
/// in `main.rs` for the CLI `--init-config` flag.
pub(crate) fn init_config_retired_message() -> String {
    "System: /init-config was retired; run `magi init` instead.".to_string()
}

/// Events that can happen in the UI.
pub enum UiEvent {
    Input(String),
    Clear,
    Login,
    Logout,
    /// Trigger a forced MAGI multi-perspective analysis. `mode` is the explicit
    /// `--mode` declared on the command, if any (REQ-A07b); `None` lets
    /// `resolve_tui_consult_mode` fall back to `[magi].default_mode` or classify.
    Consult {
        query: String,
        mode: Option<Mode>,
    },
    Quit,
}

/// Messages from the Agent to the UI.
#[derive(Debug)]
pub enum AgentResponse {
    Text(String),
    Error(String),
    Info(String),
    /// An incremental text delta from the streaming provider.
    StreamDelta(String),
    /// An incremental reasoning (chain-of-thought) delta — shown live, never
    /// persisted (#24).
    ReasoningDelta(String),
    /// A non-content operational notice (memory warning, truncation advisory).
    /// Rendered in a distinct style (⚠ prefix, dimmed/yellow) — never persisted.
    Notice(String),
}

/// Represents the state of the TUI application.
pub struct App {
    /// The input string currently being typed.
    pub input: String,
    /// Current cursor position in the input string (byte index).
    pub cursor_position: usize,
    /// Selection start position (if any)
    pub selection_start: Option<usize>,
    /// History of messages to display.
    pub messages: Vec<String>,
    /// Channel to send events to the agent runner.
    pub event_tx: mpsc::Sender<UiEvent>,
    /// Channel to receive responses from the agent.
    pub response_rx: mpsc::Receiver<AgentResponse>,
    /// Channel to receive approval requests from the agent.
    pub approval_rx: mpsc::Receiver<ApprovalRequest>,
    /// Pending approval request
    pub pending_approval: Option<ApprovalRequest>,
    /// Current UI mode
    pub mode: AppMode,
    /// Index of the selected message in Selection mode
    pub selected_index: usize,
    /// Cursor position within the selected message (Visual mode)
    pub visual_cursor: usize,
    /// Selection start within the selected message (Visual mode)
    pub visual_selection_start: Option<usize>,
    /// Whether the agent is currently streaming a response.
    pub streaming: bool,
    /// Conversation scrollback offset: wrapped lines scrolled UP from the bottom
    /// (Normal mode). `0` follows the tail (newest content visible).
    pub scroll_offset: usize,
    /// Max scroll offset computed at the last render — cached so key handlers can
    /// clamp `scroll_offset` without recomputing the wrapped-line layout.
    pub last_max_scroll: usize,
    /// Visible height (lines) of the conversation pane at the last render — used to
    /// size a PageUp/PageDown step.
    pub last_viewport_height: usize,
    /// Show the full reasoning text (verbose/debug, mode A) vs a compact
    /// "thinking…" indicator (mode B, the default). Toggled by
    /// `/toggle-show-thinking`.
    pub show_thinking: bool,
    /// Whether a reasoning model is currently thinking (the compact indicator is
    /// shown while true).
    pub thinking_active: bool,
    /// Current spinner frame for the thinking indicator.
    pub spinner_frame: usize,
}

impl App {
    pub fn new(
        event_tx: mpsc::Sender<UiEvent>,
        response_rx: mpsc::Receiver<AgentResponse>,
        approval_rx: mpsc::Receiver<ApprovalRequest>,
    ) -> Self {
        Self {
            input: String::new(),
            cursor_position: 0,
            selection_start: None,
            messages: Vec::new(),
            event_tx,
            response_rx,
            approval_rx,
            pending_approval: None,
            mode: AppMode::Normal,
            selected_index: 0,
            visual_cursor: 0,
            visual_selection_start: None,
            streaming: false,
            scroll_offset: 0,
            last_max_scroll: 0,
            last_viewport_height: 0,
            show_thinking: false,
            thinking_active: false,
            spinner_frame: 0,
        }
    }

    /// Toggles between the compact thinking indicator (mode B, default) and the
    /// verbose reasoning stream (mode A, for debugging). Returns the new value.
    pub fn toggle_show_thinking(&mut self) -> bool {
        self.show_thinking = !self.show_thinking;
        self.show_thinking
    }

    /// Handles a reasoning (chain-of-thought) delta. In verbose mode it streams the
    /// text into the assistant message; in the default compact mode it only raises
    /// the activity indicator (advancing the spinner) and never shows the text.
    pub fn on_reasoning(&mut self, delta: String) {
        if self.show_thinking {
            self.append_stream_delta(delta);
        } else {
            // Compact mode: just raise the indicator; the spinner animates in the
            // render loop (driven by the ~50ms tick, not the delta rate).
            self.thinking_active = true;
        }
    }

    /// Moves the cursor to the left, respecting Unicode character boundaries.
    pub fn move_cursor_left(&mut self, select: bool) {
        if select && self.selection_start.is_none() {
            self.selection_start = Some(self.cursor_position);
        } else if !select {
            self.selection_start = None;
        }

        if self.cursor_position > 0 {
            let indices = self.input.char_indices().rev();
            for (idx, _) in indices {
                if idx < self.cursor_position {
                    self.cursor_position = idx;
                    return;
                }
            }
            self.cursor_position = 0;
        }
    }

    /// Moves the cursor to the right, respecting Unicode character boundaries.
    pub fn move_cursor_right(&mut self, select: bool) {
        if select && self.selection_start.is_none() {
            self.selection_start = Some(self.cursor_position);
        } else if !select {
            self.selection_start = None;
        }

        if self.cursor_position < self.input.len() {
            let indices = self.input.char_indices();
            for (idx, _) in indices {
                if idx > self.cursor_position {
                    self.cursor_position = idx;
                    return;
                }
            }
            self.cursor_position = self.input.len();
        }
    }

    /// Inserts a character at the current cursor position.
    pub fn insert_char(&mut self, c: char) {
        self.delete_selection();
        // Ensure cursor is at char boundary before insert
        if !self.input.is_char_boundary(self.cursor_position) {
            self.cursor_position = 0; // Emergency fallback
        }
        self.input.insert(self.cursor_position, c);
        self.cursor_position += c.len_utf8();
    }

    /// Deletes the character before the current cursor position.
    pub fn delete_char(&mut self) {
        if self.selection_start.is_some() {
            self.delete_selection();
            return;
        }

        if self.cursor_position > 0 {
            self.move_cursor_left(false);
            let prev_pos = self.cursor_position;
            if self.input.is_char_boundary(prev_pos) {
                self.input.remove(prev_pos);
            }
        }
    }

    /// Deletes the currently selected text.
    pub fn delete_selection(&mut self) {
        if let Some(start) = self.selection_start {
            let end = self.cursor_position;
            let (from, to) = if start < end {
                (start, end)
            } else {
                (end, start)
            };
            if self.input.is_char_boundary(from) && self.input.is_char_boundary(to) {
                self.input.drain(from..to);
                self.cursor_position = from;
            }
            self.selection_start = None;
        }
    }

    /// Returns the selected text if any.
    pub fn get_selected_text(&self) -> Option<String> {
        self.selection_start.and_then(|start| {
            let end = self.cursor_position;
            let (from, to) = if start < end {
                (start, end)
            } else {
                (end, start)
            };
            if self.input.is_char_boundary(from) && self.input.is_char_boundary(to) {
                Some(self.input[from..to].to_string())
            } else {
                None
            }
        })
    }

    /// Appends a message to the UI history.
    pub fn push_message(&mut self, message: String) {
        self.messages.push(message);
        // New content → snap back to the tail so it's visible.
        self.scroll_offset = 0;
    }

    /// Appends an operational notice to the UI history with the `⚠ ` prefix so
    /// it is visually distinct from model output.  The prefix is also used by the
    /// Normal-mode renderer to apply a dimmed/yellow style.
    pub fn push_notice(&mut self, text: String) {
        // Ensure the text is valid UTF-8 (it always is, but make the intent explicit).
        let notice = format!("⚠ {}", text);
        self.messages.push(notice);
        self.scroll_offset = 0;
    }

    /// Appends a streaming delta to the in-progress assistant message,
    /// creating the line on the first delta. Append-only; never byte-indexes.
    pub fn append_stream_delta(&mut self, delta: String) {
        // Answer content arriving means the thinking phase is over.
        self.thinking_active = false;
        // Streaming content → follow the tail so the live reply stays visible.
        self.scroll_offset = 0;
        if self.streaming {
            if let Some(last) = self.messages.last_mut() {
                last.push_str(&delta);
                return;
            }
        }
        self.messages.push(format!("Magi Agent: {}", delta));
        self.streaming = true;
    }

    /// Marks the end of a streamed assistant turn.
    pub fn finalize_stream(&mut self) {
        self.streaming = false;
        // Turn over → drop any lingering "thinking…" indicator (covers turns that
        // reasoned but produced no content: empty answer, error, or tool-only).
        self.thinking_active = false;
    }
}

/// Session-scoped MAGI mode/reporting parameters for the TUI's whole run
/// (REQ-A07c/REQ-A12c) — the owned counterpart of `headless_runner`'s
/// `MagiRuntimeParams`, used everywhere `/consult` is resolved instead of a
/// borrowed reference: `run_tui_ext`'s event loop lives inside a
/// `tokio::spawn`'d `'static` task, so the classifier must be owned
/// (`Arc<dyn ModeClassifier>`), not borrowed.
///
/// # Fields
/// - `mode_classifier` — consulted by [`resolve_tui_consult_mode`] only when
///   `/consult` has no explicit `--mode` and no `[magi].default_mode` is set
///   (REQ-A07c).
/// - `default_mode` — `[magi].default_mode`, resolved once at startup
///   (REQ-A15).
/// - `untrusted_content` — `[magi].untrusted_content` only; the TUI never
///   exposes this as a command-line flag (REQ-A07d/SC-A07t).
/// - `magi_kind` (REQ-A12c) — the [`ProviderKind`] the trio runs under. Feeds
///   [`tui_consult_success_body`]/[`tui_consult_error_body`] so the explicit
///   `/consult` command gets the SAME keyless-auth guidance
///   `ConsultTool`/headless already have.
pub struct TuiMagiRuntimeConfig {
    /// Consulted only when `/consult` declares no mode and none is
    /// configured (REQ-A07c).
    pub mode_classifier: Arc<dyn ModeClassifier>,
    /// `[magi].default_mode`, resolved once at startup (REQ-A15).
    pub default_mode: Option<Mode>,
    /// `[magi].untrusted_content` (REQ-A07d/SC-A07t).
    pub untrusted_content: bool,
    /// The `ProviderKind` the trio runs under (REQ-A12c).
    pub magi_kind: ProviderKind,
}

/// The MAGI `consult` tool's wiring for the TUI's whole run: whether a live
/// trio is available, what to tell the user when it is not, and how the
/// tool auto-approves — the same three values consulted both at startup and
/// again on every post-`/login` trio rebuild (I-5).
///
/// # Fields
/// - `consult` — the live orchestrator, or `None` if the trio failed to
///   build at startup (REQ-A06).
/// - `consult_unavailable_message` (Task 4.3, REQ-A06/SC-A06b) — the SAME
///   text already pushed to `startup_notices` when `consult` is `None`.
///   Read only when a `/consult` is issued with no trio available, so a
///   later `/consult` echoes the exact reason the startup notice already
///   gave instead of a second, independently-worded message.
/// - `magi_auto_approve` — whether the registered `consult` tool
///   auto-approves an autonomous invocation, mirrored into every rebuilt
///   `ConsultTool` after `/login` (I-5).
pub struct TuiConsultWiring {
    /// The live orchestrator, or `None` if the trio failed to build.
    pub consult: Option<std::sync::Arc<magi_core::orchestrator::Magi>>,
    /// Echoed verbatim by a `/consult` issued with no trio available.
    pub consult_unavailable_message: Option<String>,
    /// Whether the registered `consult` tool auto-approves.
    pub magi_auto_approve: bool,
}

/// # Parameters (REQ-A07d additions over the pre-MS2 signature)
///
/// - `consult_wiring` — the [`TuiConsultWiring`] bundle: the live trio (if
///   any), its unavailability message, and the tool's auto-approve flag.
/// - `magi_runtime` — the [`TuiMagiRuntimeConfig`] bundle: the mode
///   classifier, `[magi].default_mode`, the `untrusted_content` guard, and
///   the trio's `ProviderKind`.
pub async fn run_tui_ext(
    agent: Agent,
    startup_notices: Vec<String>,
    consult_wiring: TuiConsultWiring,
    secret_store: Option<SharedSecretStore>,
    magi_runtime: TuiMagiRuntimeConfig,
) -> anyhow::Result<()> {
    let TuiConsultWiring {
        consult,
        consult_unavailable_message,
        magi_auto_approve,
    } = consult_wiring;
    let TuiMagiRuntimeConfig {
        mode_classifier,
        default_mode,
        untrusted_content,
        magi_kind,
    } = magi_runtime;

    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        let _ = disable_raw_mode();
        let mut stdout = io::stdout();
        let _ = execute!(stdout, LeaveAlternateScreen, DisableMouseCapture);
        let _ =
            Terminal::new(CrosstermBackend::new(io::stdout())).and_then(|mut t| t.show_cursor());
        original_hook(panic_info);
    }));

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let (event_tx, mut event_rx) = mpsc::channel(100);
    let (response_tx, response_rx) = mpsc::channel(100);
    let (approval_tx, approval_rx) = mpsc::channel(100);

    for notice in startup_notices {
        let _ = response_tx.send(AgentResponse::Info(notice)).await;
    }

    let mut runner_agent = agent;
    runner_agent.set_approval_channel(approval_tx);

    let mut consult_magi_runner = consult;

    tokio::spawn(async move {
        while let Some(event) = event_rx.recv().await {
            match event {
                UiEvent::Input(text) => {
                    // Stream-bridge: `chunk_tx` is owned by `query_streaming`; when
                    // the method returns it is dropped, which closes the sender end of
                    // the channel. The forwarder task then drains any remaining deltas
                    // and exits its `recv()` loop naturally. `forwarder.await` joins
                    // the task before the end-of-turn marker is sent, guaranteeing
                    // all deltas arrive at the UI before `Text("")` (end-of-turn
                    // convention) or `Error(...)`.
                    let (chunk_tx, mut chunk_rx) = mpsc::channel::<StreamPiece>(100);
                    let forward_tx = response_tx.clone();
                    let forwarder = tokio::spawn(async move {
                        while let Some(piece) = chunk_rx.recv().await {
                            let resp = match piece {
                                StreamPiece::Content(s) => AgentResponse::StreamDelta(s),
                                StreamPiece::Reasoning(s) => AgentResponse::ReasoningDelta(s),
                                // Operational notices (memory warnings, truncation advisories)
                                // are routed through the channel instead of eprintln! to avoid
                                // corrupting the ratatui frame while in EnterAlternateScreen.
                                StreamPiece::Notice(s) => AgentResponse::Notice(s),
                            };
                            if forward_tx.send(resp).await.is_err() {
                                break;
                            }
                        }
                    });

                    // Interactive path: default config = normal cap, repetitive
                    // guard on, no headless observer (byte-for-byte unchanged).
                    let result = runner_agent
                        .query_streaming(&text, chunk_tx, AgentRunConfig::default())
                        .await;
                    // Join the forwarder: ensures all deltas are forwarded before the
                    // end-of-turn marker below is enqueued.
                    let _ = forwarder.await;

                    // `Text("")` signals end-of-turn to `run_app`; it calls
                    // `finalize_stream` instead of pushing an empty message line.
                    match result {
                        Ok(_) => {
                            let _ = response_tx.send(AgentResponse::Text(String::new())).await;
                        }
                        Err(e) => {
                            let _ = response_tx.send(AgentResponse::Error(e.to_string())).await;
                        }
                    }
                }
                UiEvent::Clear => {
                    runner_agent.clear_history();
                }
                UiEvent::Consult { query, mode } => {
                    let magi = match consult_magi_runner.as_ref() {
                        Some(m) => m.clone(),
                        None => {
                            // REQ-A06/SC-A06b: echoes the SAME text the startup
                            // notice already gave — never a second, independently-
                            // worded message (`consult_unavailable_message` is
                            // `None` only if the trio built fine, in which case
                            // this arm is unreachable; the fallback exists so a
                            // reachability change never sends silence, B9).
                            let reason = consult_unavailable_message
                                .as_deref()
                                .unwrap_or(CONSULT_UNAVAILABLE_FALLBACK);
                            let _ = response_tx.send(consult_unavailable_response(reason)).await;
                            continue;
                        }
                    };
                    // Cap forced /consult input too (the tool path caps in execute; this
                    // direct path bypasses it) — reject before any model call, INCLUDING a
                    // classification call.
                    if query.len() > crate::tools::consult::MAX_QUERY_LEN {
                        let _ = response_tx
                            .send(AgentResponse::Error(format!(
                                "consult query too large ({} bytes; max {})",
                                query.len(),
                                crate::tools::consult::MAX_QUERY_LEN
                            )))
                            .await;
                        continue;
                    }
                    // REQ-A07d: fails closed if the operator declared
                    // `untrusted_content = true` and neither `--mode` nor
                    // `default_mode` named a lens — before any model call.
                    let (mode, query) = match resolve_tui_consult_mode(
                        TuiConsultCommand { mode, query },
                        default_mode,
                        untrusted_content,
                        mode_classifier.as_ref(),
                    )
                    .await
                    {
                        Ok(resolved) => resolved,
                        Err(e) => {
                            let _ = response_tx.send(AgentResponse::Error(e.to_string())).await;
                            continue;
                        }
                    };
                    let _ = response_tx
                        .send(AgentResponse::Info(
                            "MAGI deliberating — 3 model calls…".to_string(),
                        ))
                        .await;
                    // MAGI FIX: joined spawn (awaited inline → serial, no finalize-order
                    // regression) isolates a panic in magi-core's analyze into a recoverable
                    // JoinError so the runner survives (see plan Task 6 iteration-3).
                    let join = tokio::spawn(async move { magi.analyze(&mode, &query).await }).await;
                    match join {
                        Ok(Ok(report)) => {
                            // REQ-A12c/SC-A12f (fix round 4, finding 2): routed through
                            // the SAME `annotate_report_text` `ConsultTool`/headless use
                            // — see `tui_consult_success_body`'s own doc for what this
                            // replaces.
                            let body = tui_consult_success_body(&report, magi_kind);
                            // Sanitize the verbatim report (LLM-generated) before rendering —
                            // strips ANSI escapes / control chars, matching the TextDelta path.
                            let body = crate::agent::Agent::sanitize_text(&body);
                            let _ = response_tx.send(AgentResponse::Text(body)).await;
                        }
                        Ok(Err(e)) => {
                            // REQ-A12c/SC-A12f (fix round 4, finding 2): routed through
                            // the SAME `explain_magi_error` `ConsultTool`/headless use —
                            // see `tui_consult_error_body`'s own doc. `body` is already
                            // redacted (B11) before either use below.
                            let body = tui_consult_error_body(&e, magi_kind);
                            eprintln!("[consult] analyze failed: {body}");
                            let _ = response_tx.send(AgentResponse::Error(body)).await;
                        }
                        Err(join_err) => {
                            eprintln!("[consult] analyze panicked: {join_err}");
                            let _ = response_tx
                                .send(AgentResponse::Error(
                                    "MAGI consult crashed unexpectedly; the session is still alive."
                                        .to_string(),
                                ))
                                .await;
                        }
                    }
                }
                UiEvent::Login => {
                    let oauth = crate::services::oauth::OAuthService::new();
                    let url = oauth.get_authorize_url();
                    let _ = response_tx.send(AgentResponse::Info(url)).await;

                    match oauth.start_callback_server().await {
                        Ok(code) => {
                            let _ = response_tx
                                .send(AgentResponse::Info("Authenticating...".to_string()))
                                .await;
                            match oauth.exchange_code_for_token(&code).await {
                                Ok(token) => match oauth.create_raw_api_key(&token).await {
                                    Ok(api_key) => {
                                        // SC-V36: persist the freshly-minted key in the vault
                                        // (or report an ephemeral session — never a keyring).
                                        let login_result =
                                            handle_login(secret_store.as_ref(), &api_key);
                                        let failed =
                                            matches!(login_result, AgentResponse::Error(_));
                                        let _ = response_tx.send(login_result).await;
                                        if !failed {
                                            // #9: rebuild the running agent's provider in-session
                                            // so replies use the new key without a restart.
                                            let model = std::env::var("ANTHROPIC_MODEL")
                                                .unwrap_or_else(|_| {
                                                    crate::DEFAULT_MODEL.to_string()
                                                });
                                            // #16: only the canned StaticProvider history is safe to
                                            // clear; a re-login over a live provider must keep the
                                            // real conversation. Read before the swap, build banner
                                            // before `model` moves.
                                            let was_static = runner_agent.provider_is_static();
                                            let banner = if was_static {
                                                format!("Successfully logged in! Now using Magi API (model: {model}) — no restart needed; prior canned replies cleared.")
                                            } else {
                                                format!("Re-authenticated. Now using Magi API (model: {model}) — conversation kept.")
                                            };
                                            let provider_arc: std::sync::Arc<
                                                dyn crate::agent::provider::Provider,
                                            > = std::sync::Arc::new(
                                                crate::agent::provider::AnthropicProvider::new(
                                                    api_key.clone(),
                                                    model.clone(),
                                                ),
                                            );
                                            runner_agent.set_provider(provider_arc);
                                            // I-5 + MAGI, updated Task 4.1: rebuild the consult
                                            // orchestrator over the new credentials so BOTH the
                                            // forced /consult handle and the registered auto-path
                                            // tool use them (register_or_replace adds it if it was
                                            // absent, e.g. after a static -> login transition).
                                            // Native `ClaudeProvider` (REQ-A01) replaces the retired
                                            // `MagiCoreProviderAdapter` — it gets the system prompt
                                            // through its OWN channel instead of folded into the
                                            // user turn — wrapped in `RetryProvider` (REQ-A03) like
                                            // every other native seat. Single-shared-provider shape,
                                            // matching the `Magi::new` path this replaces (no
                                            // per-agent overrides on the OAuth-login rebuild).
                                            let ceiling = std::time::Duration::from_secs(
                                                magi_rs::magi::AGENT_TIMEOUT_SECS,
                                            );
                                            let client_timeout =
                                                magi_rs::magi::derive_client_timeout(
                                                    ceiling.as_secs(),
                                                );
                                            let mut retry =
                                                magi_core::provider::RetryConfig::default();
                                            retry.operation_budget =
                                                magi_rs::magi::derive_operation_budget(
                                                    ceiling.as_secs(),
                                                );
                                            match magi_core::providers::claude::ClaudeProvider::with_timeout(
                                                api_key,
                                                model,
                                                client_timeout,
                                            ) {
                                                Ok(native) => {
                                                    let wrapped: std::sync::Arc<
                                                        dyn magi_core::provider::LlmProvider,
                                                    > = std::sync::Arc::new(
                                                        magi_core::provider::RetryProvider::with_config(
                                                            std::sync::Arc::new(native),
                                                            retry,
                                                        ),
                                                    );
                                                    let new_magi = std::sync::Arc::new(
                                                        magi_core::orchestrator::Magi::new(wrapped),
                                                    );
                                                    runner_agent.register_or_replace_tool(Box::new(
                                                        crate::tools::consult::ConsultTool::new(
                                                            new_magi.clone(),
                                                            magi_auto_approve,
                                                        ),
                                                    ));
                                                    consult_magi_runner = Some(new_magi);
                                                }
                                                Err(e) => {
                                                    let safe = handle_trio_rebuild_failure(
                                                        &e,
                                                        &mut consult_magi_runner,
                                                        &mut runner_agent,
                                                    );
                                                    let _ = response_tx
                                                        .send(AgentResponse::Error(format!(
                                                            "logged in, but the MAGI trio could \
                                                             not be rebuilt: {safe}"
                                                        )))
                                                        .await;
                                                }
                                            }
                                            if was_static {
                                                runner_agent.clear_history();
                                            }
                                            let _ =
                                                response_tx.send(AgentResponse::Info(banner)).await;
                                        }
                                    }
                                    Err(e) => {
                                        let _ = response_tx
                                            .send(AgentResponse::Error(format!(
                                                "Failed to create API key: {}",
                                                e
                                            )))
                                            .await;
                                    }
                                },
                                Err(e) => {
                                    let _ = response_tx
                                        .send(AgentResponse::Error(format!(
                                            "OAuth exchange failed: {}",
                                            e
                                        )))
                                        .await;
                                }
                            }
                        }
                        Err(e) => {
                            let _ = response_tx
                                .send(AgentResponse::Error(format!(
                                    "Callback server error: {}",
                                    e
                                )))
                                .await;
                        }
                    }
                }
                UiEvent::Logout => {
                    // Mirrors the CLI --logout path in main.rs: removes
                    // ANTHROPIC_API_KEY from the vault (SC-V37 — no keyring
                    // is ever consulted).
                    let _ = response_tx.send(handle_logout(secret_store.as_ref())).await;
                }
                UiEvent::Quit => break,
            }
        }
        // Best-effort distillation pass on exit (Task 13b / REQ-17).
        let _ = runner_agent.on_session_close().await;
    });

    let app = App::new(event_tx, response_rx, approval_rx);
    let res = run_app(&mut terminal, app).await;

    let _ = disable_raw_mode();
    let _ = execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    );
    let _ = terminal.show_cursor();

    if let Err(err) = res {
        eprintln!("TUI Error: {:?}", err)
    }
    Ok(())
}

async fn run_app<B: Backend>(terminal: &mut Terminal<B>, mut app: App) -> io::Result<()> {
    loop {
        terminal.draw(|f| ui(f, &mut app))?;

        while let Ok(response) = app.response_rx.try_recv() {
            match response {
                AgentResponse::StreamDelta(delta) => app.append_stream_delta(delta),
                // Mode-aware: verbose streams the text; compact (default) raises a
                // "thinking…" indicator without showing the chain-of-thought.
                AgentResponse::ReasoningDelta(delta) => app.on_reasoning(delta),
                AgentResponse::Text(t) => {
                    if t.is_empty() {
                        app.finalize_stream();
                    } else {
                        app.push_message(format!("Magi Agent: {}", t));
                    }
                }
                AgentResponse::Error(e) => {
                    app.finalize_stream();
                    app.push_message(format!("Error: {}", e));
                }
                AgentResponse::Info(i) => {
                    app.finalize_stream();
                    app.push_message(format!("System: {}", i));
                }
                AgentResponse::Notice(n) => {
                    // Operational notices (memory warnings, truncation advisories) do
                    // not end a streamed turn — the assistant is still responding.
                    // Rendered with the ⚠ prefix and dimmed/yellow styling in
                    // Normal mode so they are visually distinct from model output.
                    app.push_notice(n);
                }
            }
        }

        while let Ok(req) = app.approval_rx.try_recv() {
            app.push_message(format!("APPROVAL REQUIRED: Execute {}?", req.tool_name));
            app.push_message("Press 'y' to approve, 'c' or 'Esc' to deny.".to_string());
            app.pending_approval = Some(req);
        }

        if event::poll(std::time::Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }

                match app.mode {
                    AppMode::Selection => {
                        match key.code {
                            KeyCode::Up if app.selected_index > 0 => {
                                app.selected_index -= 1;
                            }
                            KeyCode::Down
                                if app.selected_index < app.messages.len().saturating_sub(1) =>
                            {
                                app.selected_index += 1;
                            }
                            KeyCode::Enter => {
                                app.mode = AppMode::Visual;
                                app.visual_cursor = 0;
                                app.visual_selection_start = None;
                            }
                            KeyCode::Char('y') => {
                                if let Some(msg) = app.messages.get(app.selected_index) {
                                    if let Ok(mut clipboard) = arboard::Clipboard::new() {
                                        let _ = clipboard.set_text(msg.clone());
                                        app.push_message("System: Message copied".to_string());
                                    }
                                }
                                app.mode = AppMode::Normal;
                            }
                            KeyCode::Esc | KeyCode::Char('q') => {
                                app.mode = AppMode::Normal;
                            }
                            _ => {}
                        }
                        continue;
                    }
                    AppMode::Visual => {
                        let msg = app
                            .messages
                            .get(app.selected_index)
                            .cloned()
                            .unwrap_or_default();
                        match key.code {
                            KeyCode::Left => {
                                if key.modifiers.contains(KeyModifiers::SHIFT)
                                    && app.visual_selection_start.is_none()
                                {
                                    app.visual_selection_start = Some(app.visual_cursor);
                                } else if !key.modifiers.contains(KeyModifiers::SHIFT) {
                                    app.visual_selection_start = None;
                                }
                                if app.visual_cursor > 0 {
                                    let indices = msg.char_indices().rev();
                                    for (idx, _) in indices {
                                        if idx < app.visual_cursor {
                                            app.visual_cursor = idx;
                                            break;
                                        }
                                    }
                                }
                            }
                            KeyCode::Right => {
                                if key.modifiers.contains(KeyModifiers::SHIFT)
                                    && app.visual_selection_start.is_none()
                                {
                                    app.visual_selection_start = Some(app.visual_cursor);
                                } else if !key.modifiers.contains(KeyModifiers::SHIFT) {
                                    app.visual_selection_start = None;
                                }
                                if app.visual_cursor < msg.len() {
                                    let indices = msg.char_indices();
                                    for (idx, _) in indices {
                                        if idx > app.visual_cursor {
                                            app.visual_cursor = idx;
                                            break;
                                        }
                                    }
                                }
                            }
                            KeyCode::Enter => {
                                if let (Some(msg_ref), Some(start)) = (
                                    app.messages.get(app.selected_index),
                                    app.visual_selection_start,
                                ) {
                                    let (from, to) = if start < app.visual_cursor {
                                        (start, app.visual_cursor)
                                    } else {
                                        (app.visual_cursor, start)
                                    };
                                    if msg_ref.is_char_boundary(from)
                                        && msg_ref.is_char_boundary(to)
                                    {
                                        if let Ok(mut clipboard) = arboard::Clipboard::new() {
                                            let _ =
                                                clipboard.set_text(msg_ref[from..to].to_string());
                                            app.push_message("System: Fragment copied".to_string());
                                        }
                                    }
                                }
                                app.mode = AppMode::Normal;
                            }
                            KeyCode::Esc | KeyCode::Char('q') => {
                                app.mode = AppMode::Selection;
                            }
                            _ => {}
                        }
                        continue;
                    }
                    AppMode::Normal => {
                        match (key.code, key.modifiers) {
                            (KeyCode::Char('v'), KeyModifiers::CONTROL) => {
                                if let Ok(mut clipboard) = arboard::Clipboard::new() {
                                    if let Ok(text) = clipboard.get_text() {
                                        for c in text.chars() {
                                            app.insert_char(c);
                                        }
                                    }
                                }
                                continue;
                            }
                            (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                                if let Some(selected) = app.get_selected_text() {
                                    if let Ok(mut clipboard) = arboard::Clipboard::new() {
                                        let _ = clipboard.set_text(selected);
                                        app.push_message("System: Selection copied".to_string());
                                    }
                                    continue;
                                } else {
                                    let _ = app.event_tx.send(UiEvent::Quit).await;
                                    return Ok(());
                                }
                            }
                            (KeyCode::Char('s'), KeyModifiers::CONTROL) => {
                                if !app.messages.is_empty() {
                                    app.mode = AppMode::Selection;
                                    app.selected_index = app.messages.len().saturating_sub(1);
                                }
                                continue;
                            }
                            (KeyCode::Left, m) => {
                                app.move_cursor_left(m.contains(KeyModifiers::SHIFT));
                                continue;
                            }
                            (KeyCode::Right, m) => {
                                app.move_cursor_right(m.contains(KeyModifiers::SHIFT));
                                continue;
                            }
                            (KeyCode::PageUp, _) => {
                                let page = app.last_viewport_height.saturating_sub(1).max(1);
                                app.scroll_offset =
                                    (app.scroll_offset + page).min(app.last_max_scroll);
                                continue;
                            }
                            (KeyCode::PageDown, _) => {
                                let page = app.last_viewport_height.saturating_sub(1).max(1);
                                app.scroll_offset = app.scroll_offset.saturating_sub(page);
                                continue;
                            }
                            (KeyCode::Home, _) => {
                                app.scroll_offset = app.last_max_scroll;
                                continue;
                            }
                            (KeyCode::End, _) => {
                                app.scroll_offset = 0;
                                continue;
                            }
                            (KeyCode::Up, _) => {
                                app.scroll_offset =
                                    (app.scroll_offset + 1).min(app.last_max_scroll);
                                continue;
                            }
                            (KeyCode::Down, _) => {
                                app.scroll_offset = app.scroll_offset.saturating_sub(1);
                                continue;
                            }
                            _ => {}
                        }

                        if let Some(req) = app.pending_approval.take() {
                            match key.code {
                                KeyCode::Char('y') | KeyCode::Char('Y') => {
                                    let _ = req.tx.send(true);
                                    app.push_message("User: Approved".to_string());
                                }
                                KeyCode::Char('c') | KeyCode::Char('C') | KeyCode::Esc => {
                                    let _ = req.tx.send(false);
                                    app.push_message("User: Denied".to_string());
                                }
                                _ => {
                                    app.pending_approval = Some(req);
                                }
                            }
                            continue;
                        }

                        match key.code {
                            KeyCode::Enter => {
                                let input = app.input.drain(..).collect::<String>();
                                app.cursor_position = 0;
                                let trimmed = input.trim();
                                if !trimmed.is_empty() {
                                    // REQ-A07b: `parse_tui_consult` replaces the older
                                    // `parse_consult_command` here — that one only stripped
                                    // the `/consult ` prefix, so `--mode design` in
                                    // `/consult --mode design <question>` fell straight
                                    // through into the query text instead of being read as
                                    // a flag, and the analysis always ran `Analysis`
                                    // regardless. `parse_consult_command` stays (used
                                    // internally by `parse_tui_consult` and by its own
                                    // pre-existing test) but is no longer this call site's
                                    // parser.
                                    match parse_tui_consult(trimmed) {
                                        Ok(cmd) => {
                                            if cmd.query.is_empty() {
                                                app.push_message(
                                                    "Usage: /consult [--mode <code-review|design|analysis>] <question> — forces MAGI multi-perspective analysis (3 model calls)"
                                                        .to_string(),
                                                );
                                            } else {
                                                let echoed = match cmd.mode {
                                                    Some(m) => {
                                                        format!(
                                                            "User: /consult --mode {m} {}",
                                                            cmd.query
                                                        )
                                                    }
                                                    None => format!("User: /consult {}", cmd.query),
                                                };
                                                app.push_message(echoed);
                                                let _ = app
                                                    .event_tx
                                                    .send(UiEvent::Consult {
                                                        query: cmd.query,
                                                        mode: cmd.mode,
                                                    })
                                                    .await;
                                            }
                                            continue;
                                        }
                                        Err(TuiConsultParseError::NotAConsultCommand) => {
                                            // Not a `/consult` line at all — fall through to
                                            // the other command checks below, unchanged.
                                        }
                                        Err(TuiConsultParseError::MissingModeValue) => {
                                            app.push_message(
                                                "System: --mode needs a value (code-review, design, or analysis)"
                                                    .to_string(),
                                            );
                                            continue;
                                        }
                                        Err(TuiConsultParseError::UnknownMode(bad)) => {
                                            app.push_message(format!(
                                                "System: unknown mode {bad:?} (valid: code-review, design, analysis)"
                                            ));
                                            continue;
                                        }
                                        Err(TuiConsultParseError::UnsupportedFlag(flag)) => {
                                            app.push_message(format!(
                                                "System: unsupported flag {flag:?} for /consult"
                                            ));
                                            continue;
                                        }
                                    }
                                    if parse_toggle_show_thinking(trimmed) {
                                        let verbose = app.toggle_show_thinking();
                                        let mode = if verbose {
                                            "VERBOSE (full reasoning shown — for debugging)"
                                        } else {
                                            "COMPACT (activity indicator only)"
                                        };
                                        app.push_message(format!(
                                            "System: thinking display -> {mode}"
                                        ));
                                        continue;
                                    }
                                    if parse_init_config(trimmed) {
                                        app.push_message(init_config_retired_message());
                                        continue;
                                    }
                                    match trimmed {
                                        "/exit" | "/quit" => {
                                            let _ = app.event_tx.send(UiEvent::Quit).await;
                                            return Ok(());
                                        }
                                        "/clear" => {
                                            app.messages.clear();
                                            let _ = app.event_tx.send(UiEvent::Clear).await;
                                            continue;
                                        }
                                        "/login" => {
                                            let _ = app.event_tx.send(UiEvent::Login).await;
                                            continue;
                                        }
                                        "/logout" => {
                                            let _ = app.event_tx.send(UiEvent::Logout).await;
                                            continue;
                                        }
                                        "/help" => {
                                            app.push_message("Available commands:".to_string());
                                            app.push_message(
                                                "  /login, /logout - Identity management"
                                                    .to_string(),
                                            );
                                            app.push_message(
                                                "  /exit, /quit    - Exit the application"
                                                    .to_string(),
                                            );
                                            app.push_message(
                                                "  /clear          - Clear session history"
                                                    .to_string(),
                                            );
                                            app.push_message(
                                                "  /consult <q>    - Force MAGI multi-perspective analysis (3 model calls)"
                                                    .to_string(),
                                            );
                                            app.push_message(
                                                "  /help           - Show this help message"
                                                    .to_string(),
                                            );
                                            app.push_message(
                                                "  /toggle-show-thinking - Reasoning display: indicator (default) <-> verbose"
                                                    .to_string(),
                                            );
                                            app.push_message(
                                                "  (run `magi init` from a shell to scaffold .magi/ and its magi.toml)"
                                                    .to_string(),
                                            );
                                            continue;
                                        }
                                        _ => {}
                                    }
                                    app.push_message(format!("User: {}", trimmed));
                                    let _ = app
                                        .event_tx
                                        .send(UiEvent::Input(trimmed.to_string()))
                                        .await;
                                }
                            }
                            KeyCode::Char(c) => {
                                app.insert_char(c);
                            }
                            KeyCode::Backspace => {
                                app.delete_char();
                            }
                            KeyCode::Esc => {
                                let _ = app.event_tx.send(UiEvent::Quit).await;
                                return Ok(());
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }
}

/// Terminal display width of a char (0 for combining marks, 2 for wide CJK/emoji),
/// defaulting to 0 for control chars (which the agent already sanitizes out).
fn char_display_width(c: char) -> usize {
    UnicodeWidthChar::width(c).unwrap_or(0)
}

/// Hard-splits a single token into chunks no wider than `width` display columns,
/// pushing each chunk to `out`. Used for words longer than the whole line width.
fn push_hard_split(word: &str, width: usize, out: &mut Vec<String>) {
    let mut chunk = String::new();
    let mut chunk_w = 0usize;
    for ch in word.chars() {
        let cw = char_display_width(ch);
        if chunk_w + cw > width && !chunk.is_empty() {
            out.push(std::mem::take(&mut chunk));
            chunk_w = 0;
        }
        chunk.push(ch);
        chunk_w += cw;
    }
    if !chunk.is_empty() {
        out.push(chunk);
    }
}

/// Word-wraps `text` so each returned line fits in `width` terminal DISPLAY
/// COLUMNS (not chars/bytes — CJK/emoji count as 2). Existing `\n` are preserved
/// as hard breaks. `width == 0` is treated as no-op.
///
/// A line that already fits is returned **unchanged**, preserving its leading
/// indentation and internal alignment spaces (markdown bullets, ASCII tables,
/// box-drawing — e.g. the consult report). Only lines wider than `width` are
/// reflowed at word boundaries; a single word wider than `width` is hard-split.
fn wrap_message(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![text.to_string()];
    }
    let mut out: Vec<String> = Vec::new();
    for paragraph in text.split('\n') {
        if paragraph.is_empty() {
            out.push(String::new());
            continue;
        }
        // Fast path: keep a fitting line byte-for-byte (indentation + spacing intact).
        if UnicodeWidthStr::width(paragraph) <= width {
            out.push(paragraph.to_string());
            continue;
        }
        // Over-width line: reflow by word at display-column granularity.
        let mut line = String::new();
        let mut line_w: usize = 0;
        for word in paragraph.split_whitespace() {
            let w = UnicodeWidthStr::width(word);
            if w > width {
                if !line.is_empty() {
                    out.push(std::mem::take(&mut line));
                    line_w = 0;
                }
                push_hard_split(word, width, &mut out);
                continue;
            }
            let need = if line.is_empty() { w } else { w + 1 };
            if line_w + need > width {
                out.push(std::mem::take(&mut line));
                line_w = 0;
            }
            if !line.is_empty() {
                line.push(' ');
                line_w += 1;
            }
            line.push_str(word);
            line_w += w;
        }
        if !line.is_empty() {
            out.push(line);
        }
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

/// Selection index for the conversation `List` given the current UI mode.
///
/// In Selection / Visual mode the user-chosen index is used.  In Normal mode
/// the LAST message is selected so ratatui auto-scrolls the pane to keep the
/// newest message visible (follow-tail behavior). Empty history → `None` so
/// the list renders without an out-of-bounds selection.
fn effective_selection(mode: AppMode, selected_index: usize, messages_len: usize) -> Option<usize> {
    if messages_len == 0 {
        return None;
    }
    match mode {
        AppMode::Selection | AppMode::Visual => Some(selected_index),
        AppMode::Normal => Some(messages_len - 1),
    }
}

/// Highlight-symbol prefix for the conversation `List`.
///
/// Returns `">> "` only in Selection / Visual modes (where the user is
/// actively picking a message); Normal mode returns `""` so the auto-scroll
/// pin from `effective_selection` is invisible.
fn effective_highlight_symbol(mode: AppMode) -> &'static str {
    if matches!(mode, AppMode::Selection | AppMode::Visual) {
        ">> "
    } else {
        ""
    }
}

/// Maximum scroll offset (lines hidden above the bottom window) for a conversation
/// of `total` wrapped lines in a viewport `height` lines tall. `0` when it all fits.
fn max_scroll(total: usize, height: usize) -> usize {
    total.saturating_sub(height)
}

/// Index range of the visible slice of a `total`-line conversation in a viewport
/// `height` lines tall, scrolled `offset` lines UP from the bottom.
///
/// `offset == 0` pins to the bottom (follow-tail). The offset is clamped so the
/// top line never scrolls past view. A conversation shorter than the viewport
/// shows everything (`0..total`); a zero height or empty buffer yields `0..0`.
fn scroll_window(total: usize, height: usize, offset: usize) -> std::ops::Range<usize> {
    if height == 0 || total == 0 {
        return 0..0;
    }
    if total <= height {
        return 0..total;
    }
    let max_off = total - height;
    let clamped = offset.min(max_off);
    let start = total - height - clamped;
    start..(start + height)
}

/// Maximum number of visible content rows the input box may grow to before it
/// stops stealing space from the conversation pane (a long prompt then scrolls
/// internally, keeping the cursor — at the tail — in view).
const MAX_INPUT_ROWS: usize = 6;

/// Number of content rows the input box should show for `input` wrapped to
/// `width` columns, clamped to `[1, max]`. Lets the pane grow with a long or
/// multi-line prompt instead of truncating it, without taking the whole screen.
fn input_pane_rows(input: &str, width: usize, max: usize) -> usize {
    wrap_message(input, width).len().clamp(1, max.max(1))
}

fn ui(f: &mut Frame, app: &mut App) {
    let area = f.size();
    // Input content width = full width minus the layout margin (1 each side) and
    // the box borders (1 each side). Compute it up front so the input pane height
    // can grow with the wrapped prompt before the layout is split.
    let input_content_w = (area.width as usize).saturating_sub(4);
    let input_rows = input_pane_rows(&app.input, input_content_w, MAX_INPUT_ROWS);
    let input_pane_height = (input_rows as u16).saturating_add(2); // + top/bottom borders

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([Constraint::Min(1), Constraint::Length(input_pane_height)].as_ref())
        .split(area);

    let inner_width = chunks[0].width.saturating_sub(2) as usize; // subtract left + right borders
    let inner_height = chunks[0].height.saturating_sub(2) as usize; // subtract top + bottom borders

    if app.mode == AppMode::Normal {
        // Flatten every message into one wrapped-line buffer (a blank separator line
        // between messages), then render a scrollable window of it. This gives
        // line-level scrollback — a single message taller than the viewport (e.g. the
        // consult report) is fully reachable via PgUp/PgDn/Home/End.
        let mut all_lines: Vec<String> = Vec::new();
        for (i, m) in app.messages.iter().enumerate() {
            if i > 0 {
                all_lines.push(String::new());
            }
            all_lines.extend(wrap_message(m, inner_width));
        }
        // Compact "thinking…" indicator (mode B): a transient last line while a
        // reasoning model is thinking, with an animated spinner advanced per frame.
        if app.thinking_active {
            if !all_lines.is_empty() {
                all_lines.push(String::new());
            }
            all_lines.extend(wrap_message(
                &thinking_indicator(app.spinner_frame),
                inner_width,
            ));
            app.spinner_frame = next_spinner_frame(app.spinner_frame);
        }
        let total = all_lines.len();
        // Cache viewport bounds so key handlers can clamp scroll_offset (see Normal keys).
        app.last_viewport_height = inner_height;
        app.last_max_scroll = max_scroll(total, inner_height);
        if app.scroll_offset > app.last_max_scroll {
            app.scroll_offset = app.last_max_scroll;
        }
        let range = scroll_window(total, inner_height, app.scroll_offset);
        // Render notice lines (⚠ prefix) with a dimmed yellow style so they are
        // visually distinct from model Content and from system messages.
        let notice_style = Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::DIM);
        let visible: Vec<Line> = all_lines[range]
            .iter()
            .map(|l| {
                if l.starts_with('⚠') {
                    Line::styled(l.clone(), notice_style)
                } else {
                    Line::from(l.clone())
                }
            })
            .collect();
        let title = if app.scroll_offset > 0 {
            format!(
                "Conversation History  [scrolled ↑{} · PgDn/End → bottom]",
                app.scroll_offset
            )
        } else {
            "Conversation History".to_string()
        };
        let conversation = Paragraph::new(Text::from(visible))
            .block(Block::default().borders(Borders::ALL).title(title));
        f.render_widget(conversation, chunks[0]);
    } else {
        // Selection / Visual: per-message List so the user can pick a message to copy.
        let messages: Vec<ListItem> = app
            .messages
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let mut style = Style::default();
                if i == app.selected_index {
                    style = style
                        .bg(Color::Blue)
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD);
                }
                let lines: Vec<Line> = wrap_message(m, inner_width)
                    .into_iter()
                    .map(Line::from)
                    .collect();
                ListItem::new(Text::from(lines)).style(style)
            })
            .collect();

        let mut state = ListState::default();
        if let Some(idx) = effective_selection(app.mode, app.selected_index, app.messages.len()) {
            state.select(Some(idx));
        }

        let messages_list = List::new(messages)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Conversation History"),
            )
            .highlight_symbol(effective_highlight_symbol(app.mode));
        f.render_stateful_widget(messages_list, chunks[0], &mut state);
    }

    let input_title = match app.mode {
        AppMode::Selection => {
            "SELECT MESSAGE (Enter to select text, 'y' to copy whole, Esc to exit)"
        }
        AppMode::Visual => "VISUAL SELECTION MODE",
        _ if app.pending_approval.is_some() => "WAITING FOR APPROVAL (y/c)",
        _ => "Input (↑↓/PgUp/PgDn/Home/End Scroll, Ctrl+S Copy, Shift+←→ Select)",
    };
    let input_block = Block::default().borders(Borders::ALL).title(input_title);

    if input_rows <= 1 {
        // Single-line prompt: keep the in-place selection highlight and the simple
        // cursor (no behavior change for the common case).
        let mut input_text = Text::raw(app.input.as_str());
        if let Some(start) = app.selection_start {
            let (from, to) = if start < app.cursor_position {
                (start, app.cursor_position)
            } else {
                (app.cursor_position, start)
            };
            if app.input.is_char_boundary(from) && app.input.is_char_boundary(to) {
                let spans = vec![
                    Span::raw(&app.input[..from]),
                    Span::styled(
                        &app.input[from..to],
                        Style::default().bg(Color::White).fg(Color::Black),
                    ),
                    Span::raw(&app.input[to..]),
                ];
                input_text = Text::from(Line::from(spans));
            }
        }
        f.render_widget(Paragraph::new(input_text).block(input_block), chunks[1]);
        if app.mode == AppMode::Normal {
            let col = UnicodeWidthStr::width(&app.input[..app.cursor_position]) as u16;
            f.set_cursor(chunks[1].x + col + 1, chunks[1].y + 1);
        }
    } else {
        // Long / multi-line prompt: pre-wrap with the same algorithm so the cursor
        // (derived from the prefix) lines up exactly, and tail the wrapped lines so
        // the cursor — at the end while typing — stays visible. The in-place
        // selection highlight is omitted here (composing, not selecting within).
        let wrapped = wrap_message(&app.input, input_content_w);
        let total = wrapped.len();
        let shown = total.min(MAX_INPUT_ROWS);
        let start = total - shown;
        let visible: Vec<Line> = wrapped[start..]
            .iter()
            .map(|l| Line::from(l.clone()))
            .collect();
        f.render_widget(
            Paragraph::new(Text::from(visible)).block(input_block),
            chunks[1],
        );
        if app.mode == AppMode::Normal {
            let prefix = wrap_message(&app.input[..app.cursor_position], input_content_w);
            let cur_row_abs = prefix.len().saturating_sub(1);
            let cur_col =
                UnicodeWidthStr::width(prefix.last().map(String::as_str).unwrap_or("")) as u16;
            let cur_row_vis = cur_row_abs.saturating_sub(start) as u16;
            f.set_cursor(chunks[1].x + cur_col + 1, chunks[1].y + cur_row_vis + 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magi_core::providers::claude::ClaudeProvider;
    use magi_core::schema::AgentName;
    use magi_core::test_support::RoutingMockProvider;
    use magi_core::verdict_markers::{VERDICT_CLOSE, VERDICT_OPEN};

    /// Canonical mage response (magi-core 3.x verdict-marker contract) — same shape
    /// `src/tools/consult.rs`'s own `agent_json` builds, needed here to drive a REAL
    /// `Magi::analyze()` for the fix-round-4 tests below (a `MagiReport` cannot be
    /// hand-constructed: it is `#[non_exhaustive]`).
    fn agent_json(agent: &str) -> String {
        let verdict = format!(
            r#"{{"agent":"{agent}","verdict":"approve","confidence":0.9,"summary":"s","reasoning":"r","findings":[],"recommendation":"rec"}}"#
        );
        format!("{VERDICT_OPEN}\n{verdict}\n{VERDICT_CLOSE}")
    }

    /// Builds an in-memory [`SharedSecretStore`] fixture for the
    /// `handle_login`/`handle_logout` regression tests (MAGI run 8, Balthasar
    /// — the full TUI event loop is intractable to test directly, so the
    /// logic is extracted into these plain functions and tested here).
    fn vault_fixture() -> SharedSecretStore {
        let conn = rusqlite::Connection::open_in_memory().expect("mem db");
        let dek =
            magi_rs::vault::MaskedDek::new(zeroize::Zeroizing::new(vec![5u8; 32])).expect("32B");
        let store = magi_rs::vault::wire(Arc::new(Mutex::new(conn)), dek).expect("wire");
        Arc::new(Mutex::new(store)) as SharedSecretStore
    }

    #[test]
    fn test_handle_login_stores_key_in_vault() {
        let ss = vault_fixture();
        let resp = handle_login(Some(&ss), "sk-fresh-key");
        assert!(matches!(resp, AgentResponse::Info(_)));
        let mut guard = ss.lock().unwrap();
        assert_eq!(
            guard.get("ANTHROPIC_API_KEY").unwrap().as_str(),
            "sk-fresh-key"
        );
    }

    #[test]
    fn test_handle_login_without_vault_reports_ephemeral() {
        let resp = handle_login(None, "sk-fresh-key");
        match resp {
            AgentResponse::Info(msg) => assert!(msg.to_lowercase().contains("ephemeral")),
            other => panic!("expected an Info notice, got {other:?}"),
        }
    }

    #[test]
    fn test_handle_logout_removes_key() {
        let ss = vault_fixture();
        {
            let mut guard = ss.lock().unwrap();
            guard.set("ANTHROPIC_API_KEY", "sk-to-remove").unwrap();
        }
        let resp = handle_logout(Some(&ss));
        assert!(matches!(resp, AgentResponse::Info(_)));
        let mut guard = ss.lock().unwrap();
        assert!(matches!(
            guard.get("ANTHROPIC_API_KEY"),
            Err(VaultError::SecretNotFound(_))
        ));
    }

    #[test]
    fn test_handle_logout_absent_key_reports_no_session() {
        let ss = vault_fixture();
        let resp = handle_logout(Some(&ss));
        match resp {
            AgentResponse::Info(msg) => assert!(msg.to_lowercase().contains("no stored session")),
            other => panic!("expected an Info notice, got {other:?}"),
        }
        // No vault attached at all is the same "nothing to log out of" case.
        let resp2 = handle_logout(None);
        match resp2 {
            AgentResponse::Info(msg) => assert!(msg.to_lowercase().contains("no stored session")),
            other => panic!("expected an Info notice, got {other:?}"),
        }
    }

    /// A [`magi_core::provider::LlmProvider`] double never actually called in
    /// the `handle_trio_rebuild_failure` tests below — it exists only so
    /// `Magi::new` has something to wrap into an `Arc<Magi>` fixture.
    struct UnusedProvider;

    #[async_trait::async_trait]
    impl magi_core::provider::LlmProvider for UnusedProvider {
        async fn complete(
            &self,
            _system_prompt: &str,
            _user_prompt: &str,
            _config: &magi_core::provider::CompletionConfig,
        ) -> Result<String, magi_core::error::ProviderError> {
            Err(magi_core::error::ProviderError::external(
                "unused in this test",
                magi_core::error::ExternalErrorKind::Network,
            ))
        }
        fn name(&self) -> &str {
            "unused"
        }
        fn model(&self) -> &str {
            "unused"
        }
    }

    /// A no-op [`crate::tools::Tool`] double standing in for the real
    /// `consult` tool — cheap to register/remove without a real `Magi`.
    struct NamedTool(&'static str);

    #[async_trait::async_trait]
    impl crate::tools::Tool for NamedTool {
        fn name(&self) -> &str {
            self.0
        }
        fn description(&self) -> &str {
            "test double"
        }
        fn input_schema(&self) -> serde_json::Value {
            serde_json::json!({})
        }
        async fn execute(
            &self,
            _input: serde_json::Value,
            _cancel: &tokio_util::sync::CancellationToken,
        ) -> crate::tools::ToolResult<serde_json::Value> {
            Ok(serde_json::Value::Null)
        }
    }

    /// I4 (fix round 2): a failed rebuild clears BOTH `consult_magi_runner`
    /// and the registered `consult` tool — leaving either one pointed at the
    /// pre-failure state would let something keep routing against
    /// credentials that no longer match what was just written to the vault.
    #[test]
    fn test_handle_trio_rebuild_failure_clears_runner_and_removes_tool() {
        let mut consult_magi_runner = Some(Arc::new(magi_core::orchestrator::Magi::new(Arc::new(
            UnusedProvider,
        ))));
        let mut runner_agent = Agent::new(Arc::new(crate::agent::provider::StaticProvider));
        runner_agent.register_tool(Box::new(NamedTool("consult")));
        runner_agent.register_tool(Box::new(NamedTool("bash")));

        let e = magi_core::error::ProviderError::external(
            "boom",
            magi_core::error::ExternalErrorKind::Network,
        );
        let _ = handle_trio_rebuild_failure(&e, &mut consult_magi_runner, &mut runner_agent);

        assert!(
            consult_magi_runner.is_none(),
            "a failed rebuild must not leave the OLD Magi handle in place"
        );
        assert!(
            !runner_agent.has_tool("consult"),
            "the stale consult tool must be gone"
        );
        assert!(
            runner_agent.has_tool("bash"),
            "an unrelated tool must survive"
        );
    }

    /// I4 (fix round 2): the returned text is redacted — `e`'s `Display` can
    /// carry a URL (a foreign `ProviderError`'s message is third-party text
    /// this crate does not author), and the credential in it must not reach
    /// the user-facing error banner.
    #[test]
    fn test_handle_trio_rebuild_failure_redacts_the_error_text() {
        let mut consult_magi_runner = Some(Arc::new(magi_core::orchestrator::Magi::new(Arc::new(
            UnusedProvider,
        ))));
        let mut runner_agent = Agent::new(Arc::new(crate::agent::provider::StaticProvider));

        let e = magi_core::error::ProviderError::external(
            "connect to https://svc-user:s3cr3t-pass@evil.example.com/v1 failed",
            magi_core::error::ExternalErrorKind::Network,
        );
        let safe = handle_trio_rebuild_failure(&e, &mut consult_magi_runner, &mut runner_agent);

        assert!(
            !safe.as_str().contains("s3cr3t-pass"),
            "the credential must not survive redaction: {}",
            safe.as_str()
        );
        assert!(
            safe.as_str().contains("evil.example.com"),
            "the host stays visible — only the userinfo is redacted: {}",
            safe.as_str()
        );
    }

    /// SC-A06b: the TUI's `/consult`-without-a-trio reply is verbatim `reason`,
    /// wrapped as an error — never a re-worded summary of it.
    #[test]
    fn test_consult_unavailable_response_echoes_the_reason_verbatim() {
        let reason = "El consenso MAGI no está disponible — no se pudieron construir \
                       estos asientos:\n  Melchior: falta la credencial OPENAI_API_KEY";
        match consult_unavailable_response(reason) {
            AgentResponse::Error(text) => assert_eq!(text, reason),
            other => panic!("expected AgentResponse::Error, got {other:?}"),
        }
    }

    /// B9: the structurally-unreachable no-message case still answers with SOME
    /// honest text instead of the empty string / a panic.
    #[test]
    fn test_consult_unavailable_fallback_is_non_empty() {
        assert!(!CONSULT_UNAVAILABLE_FALLBACK.is_empty());
    }

    // -----------------------------------------------------------------------
    // Fix round 4, finding 2 — REQ-A12c/SC-A12f: the TUI's explicit `/consult`
    // routed through the same annotation/explanation as ConsultTool/headless.
    // -----------------------------------------------------------------------

    /// SC-A12f: a partial failure (one seat rejected on auth) reaches the TUI's
    /// `/consult` body with the SAME keyless-auth explanation `ConsultTool`/headless
    /// already surface — not the raw, unexplained report `UiEvent::Consult` rendered
    /// before this fix.
    #[tokio::test]
    async fn tui_consult_success_body_carries_the_keyless_hint_when_a_seat_fails_on_auth() {
        let auth_err = ClaudeProvider::map_status_to_error(401, "x", vec![], None);
        let provider = RoutingMockProvider::new()
            .with_agent_responses(AgentName::Melchior, vec![Ok(agent_json("melchior"))])
            .with_agent_responses(AgentName::Balthasar, vec![Ok(agent_json("balthasar"))])
            .with_agent_responses(AgentName::Caspar, vec![Err(auth_err)]);
        let magi = magi_core::orchestrator::Magi::new(Arc::new(provider));
        let report = magi
            .analyze(&Mode::Analysis, "should we migrate X to Y?")
            .await
            .expect("2 of 3 succeed ⇒ Ok, degraded");

        let body = tui_consult_success_body(&report, ProviderKind::Ollama);
        assert!(body.contains("DEGRADED"), "{body}");
        assert!(
            body.contains("keyless") && body.contains("openai-compat"),
            "the explanation must reach the TUI body: {body}"
        );
    }

    /// SC-A12f negative control: the same partial failure under a credentialed kind
    /// renders WITHOUT the hint — proving the guard, not just the annotation call,
    /// survives on this surface.
    #[tokio::test]
    async fn tui_consult_success_body_omits_the_hint_under_a_credentialed_kind() {
        let auth_err = ClaudeProvider::map_status_to_error(401, "x", vec![], None);
        let provider = RoutingMockProvider::new()
            .with_agent_responses(AgentName::Melchior, vec![Ok(agent_json("melchior"))])
            .with_agent_responses(AgentName::Balthasar, vec![Ok(agent_json("balthasar"))])
            .with_agent_responses(AgentName::Caspar, vec![Err(auth_err)]);
        let magi = magi_core::orchestrator::Magi::new(Arc::new(provider));
        let report = magi
            .analyze(&Mode::Analysis, "should we migrate X to Y?")
            .await
            .expect("2 of 3 succeed ⇒ Ok, degraded");

        let body = tui_consult_success_body(&report, ProviderKind::OpenAiCompat);
        assert!(
            !body.contains("keyless"),
            "openai-compat carries a credential: no hint: {body}"
        );
    }

    /// SC-A12f: a TOTAL failure (0 of 3 seats) under a keyless kind carries the
    /// probable-cause hint in the TUI's error body — not the hardcoded generic
    /// string `UiEvent::Consult` sent unconditionally before this fix.
    #[tokio::test]
    async fn tui_consult_error_body_carries_the_keyless_hint_on_total_failure_under_a_keyless_kind()
    {
        let mk = || ClaudeProvider::map_status_to_error(401, "x", vec![], None);
        let provider = RoutingMockProvider::new()
            .with_agent_responses(AgentName::Melchior, vec![Err(mk())])
            .with_agent_responses(AgentName::Balthasar, vec![Err(mk())])
            .with_agent_responses(AgentName::Caspar, vec![Err(mk())]);
        let magi = magi_core::orchestrator::Magi::new(Arc::new(provider));
        let err = magi
            .analyze(&Mode::Analysis, "should we migrate X to Y?")
            .await
            .expect_err("0 of 3 succeed ⇒ Err(InsufficientAgents)");

        let body = tui_consult_error_body(&err, ProviderKind::Ollama);
        assert!(
            body.contains("keyless") && body.contains("openai-compat"),
            "the probable-cause hint must reach the TUI body: {body}"
        );
    }

    /// SC-A12f negative control: the same total failure under a credentialed kind
    /// gets no hint — the guard is real, not the hint being unconditional.
    #[tokio::test]
    async fn tui_consult_error_body_omits_the_hint_under_a_credentialed_kind() {
        let mk = || ClaudeProvider::map_status_to_error(401, "x", vec![], None);
        let provider = RoutingMockProvider::new()
            .with_agent_responses(AgentName::Melchior, vec![Err(mk())])
            .with_agent_responses(AgentName::Balthasar, vec![Err(mk())])
            .with_agent_responses(AgentName::Caspar, vec![Err(mk())]);
        let magi = magi_core::orchestrator::Magi::new(Arc::new(provider));
        let err = magi
            .analyze(&Mode::Analysis, "should we migrate X to Y?")
            .await
            .expect_err("0 of 3 succeed ⇒ Err(InsufficientAgents)");

        let body = tui_consult_error_body(&err, ProviderKind::OpenAiCompat);
        assert!(
            !body.contains("keyless"),
            "openai-compat carries a credential: no hint: {body}"
        );
    }

    #[tokio::test]
    async fn test_app_cursor_logic() {
        let (event_tx, _) = mpsc::channel(1);
        let (_, response_rx) = mpsc::channel(1);
        let (_, approval_rx) = mpsc::channel(1);
        let mut app = App::new(event_tx, response_rx, approval_rx);

        app.insert_char('a');
        app.insert_char('c');
        assert_eq!(app.input, "ac");
        assert_eq!(app.cursor_position, 2);

        app.move_cursor_left(false);
        app.insert_char('b');
        assert_eq!(app.input, "abc");
        assert_eq!(app.cursor_position, 2);

        app.delete_char();
        assert_eq!(app.input, "ac");
        assert_eq!(app.cursor_position, 1);
    }

    #[tokio::test]
    async fn test_unicode_character_boundary_panic() {
        let (event_tx, _) = mpsc::channel(1);
        let (_, response_rx) = mpsc::channel(1);
        let (_, approval_rx) = mpsc::channel(1);
        let mut app = App::new(event_tx, response_rx, approval_rx);

        app.insert_char('á');
        assert_eq!(app.cursor_position, 2);

        app.move_cursor_left(false);
        assert_eq!(app.cursor_position, 0);

        app.insert_char('x');
        assert_eq!(app.input, "xá");
    }

    #[test]
    fn test_wrap_message_normal_word_wrap() {
        let out = wrap_message("the quick brown fox jumps over the lazy dog", 12);
        // No line should exceed the width.
        for line in &out {
            assert!(line.chars().count() <= 12, "line {line:?} > 12");
        }
        // Joining with single spaces reconstructs the original text.
        assert_eq!(out.join(" "), "the quick brown fox jumps over the lazy dog");
    }

    #[test]
    fn test_wrap_message_preserves_embedded_newlines() {
        // Existing \n in the text becomes a hard line break — wrap each paragraph independently.
        let out = wrap_message("hello world\n\nsecond paragraph here", 20);
        // The blank line between paragraphs is preserved as an empty entry.
        assert!(
            out.iter().any(|l| l.is_empty()),
            "expected an empty line for the blank paragraph: {out:?}"
        );
        assert!(out.iter().any(|l| l == "hello world"));
        assert!(out.iter().any(|l| l.contains("second paragraph")));
    }

    #[test]
    fn test_wrap_message_breaks_oversized_word() {
        // A single word longer than width must be split into chunks of <= width chars each,
        // not infinite-loop and not exceed width.
        let out = wrap_message("supercalifragilisticexpialidocious", 5);
        for line in &out {
            assert!(line.chars().count() <= 5, "line {line:?} > 5");
        }
        assert!(!out.is_empty());
        // The chunks, concatenated, must equal the original word.
        assert_eq!(out.join(""), "supercalifragilisticexpialidocious");
    }

    #[test]
    fn test_wrap_message_handles_multibyte_utf8() {
        // Spanish accents: chars().count() not byte length. Width is measured in CHARS, not bytes.
        let out = wrap_message("La capital de Venezuela es Caracas — está al norte", 18);
        for line in &out {
            assert!(line.chars().count() <= 18, "line {line:?} > 18 chars");
        }
    }

    #[test]
    fn test_wrap_message_width_zero_yields_at_least_one_line() {
        // Defensive: width 0 must not panic / loop. A single line containing the original text is acceptable.
        let out = wrap_message("anything", 0);
        assert_eq!(out, vec!["anything".to_string()]);
    }

    #[test]
    fn test_wrap_message_empty_input() {
        let out = wrap_message("", 80);
        // An empty input should produce one empty line so the message still renders as a row.
        assert_eq!(out, vec!["".to_string()]);
    }

    #[test]
    fn test_wrap_message_preserves_leading_indent_when_fits() {
        // Follow-up #1: a line that fits must be returned UNCHANGED, preserving its
        // leading indentation (markdown bullets / nested lists / the consult report).
        assert_eq!(
            wrap_message("  - bullet item", 80),
            vec!["  - bullet item".to_string()]
        );
    }

    #[test]
    fn test_wrap_message_preserves_internal_spacing_when_fits() {
        // Follow-up #1: multi-space alignment runs (ASCII tables) must survive, not
        // collapse to single spaces, when the line fits.
        assert_eq!(
            wrap_message("col1    col2", 80),
            vec!["col1    col2".to_string()]
        );
    }

    #[test]
    fn test_wrap_message_wraps_by_display_width_for_wide_chars() {
        // Follow-up #3: wrapping is measured in terminal display columns, not chars.
        // Each CJK glyph is 2 columns wide, so width 6 fits exactly 3 of them per line.
        assert_eq!(
            wrap_message("あいうえお", 6),
            vec!["あいう".to_string(), "えお".to_string()]
        );
    }

    #[test]
    fn test_effective_selection_normal_mode_follows_tail() {
        // Normal mode auto-pins the last message so the list auto-scrolls to bottom.
        assert_eq!(effective_selection(AppMode::Normal, 0, 5), Some(4));
        assert_eq!(effective_selection(AppMode::Normal, 99, 5), Some(4)); // ignores stale idx
    }

    #[test]
    fn test_effective_selection_selection_and_visual_use_index() {
        // Selection / Visual modes use the user's chosen index verbatim.
        assert_eq!(effective_selection(AppMode::Selection, 2, 5), Some(2));
        assert_eq!(effective_selection(AppMode::Visual, 0, 5), Some(0));
        assert_eq!(effective_selection(AppMode::Visual, 4, 5), Some(4));
    }

    #[test]
    fn test_effective_selection_empty_messages_yields_none() {
        // No messages → no selection (avoid out-of-bounds; the List renders nothing).
        assert_eq!(effective_selection(AppMode::Normal, 0, 0), None);
        assert_eq!(effective_selection(AppMode::Selection, 0, 0), None);
        assert_eq!(effective_selection(AppMode::Visual, 0, 0), None);
    }

    #[test]
    fn test_effective_highlight_symbol_by_mode() {
        // Selection / Visual show the ">> " marker; Normal hides it.
        assert_eq!(effective_highlight_symbol(AppMode::Selection), ">> ");
        assert_eq!(effective_highlight_symbol(AppMode::Visual), ">> ");
        assert_eq!(effective_highlight_symbol(AppMode::Normal), "");
    }

    #[test]
    fn test_max_scroll_is_overflow_above_viewport() {
        assert_eq!(max_scroll(10, 4), 6); // 6 lines hidden above the bottom window
        assert_eq!(max_scroll(4, 4), 0); // exactly fits → nothing to scroll
        assert_eq!(max_scroll(3, 10), 0); // shorter than viewport → 0 (saturating)
    }

    #[test]
    fn test_scroll_window_offset_zero_pins_to_bottom() {
        // Follow-tail: offset 0 shows the LAST `height` lines.
        assert_eq!(scroll_window(10, 5, 0), 5..10);
    }

    #[test]
    fn test_scroll_window_offset_scrolls_up_by_lines() {
        assert_eq!(scroll_window(10, 5, 2), 3..8);
    }

    #[test]
    fn test_scroll_window_clamps_offset_to_top() {
        // Offset past the top is clamped so the first line stays visible.
        assert_eq!(scroll_window(10, 5, 999), 0..5);
    }

    #[test]
    fn test_scroll_window_shorter_than_viewport_shows_all() {
        assert_eq!(scroll_window(3, 5, 0), 0..3);
        assert_eq!(scroll_window(3, 5, 99), 0..3); // offset ignored when it all fits
    }

    #[test]
    fn test_scroll_window_degenerate_zero_height_or_empty() {
        assert_eq!(scroll_window(10, 0, 0), 0..0);
        assert_eq!(scroll_window(0, 5, 0), 0..0);
    }

    #[test]
    fn test_input_pane_rows_minimum_one() {
        // Empty / short input still occupies a single visible row.
        assert_eq!(input_pane_rows("", 10, 6), 1);
        assert_eq!(input_pane_rows("short", 10, 6), 1);
    }

    #[test]
    fn test_input_pane_rows_grows_with_wrapped_input() {
        // 25 chars at width 10 wrap to 3 rows.
        assert_eq!(input_pane_rows(&"a".repeat(25), 10, 6), 3);
    }

    #[test]
    fn test_input_pane_rows_clamped_to_max() {
        // 100 chars at width 10 want 10 rows but are capped at the max.
        assert_eq!(input_pane_rows(&"a".repeat(100), 10, 6), 6);
    }

    #[test]
    fn test_parse_toggle_thinking_command() {
        assert!(super::parse_toggle_show_thinking("/toggle-show-thinking"));
        assert!(super::parse_toggle_show_thinking(
            "  /toggle-show-thinking  "
        ));
        assert!(!super::parse_toggle_show_thinking("/toggle"));
        assert!(!super::parse_toggle_show_thinking("hello"));
    }

    #[test]
    fn test_parse_init_config_command() {
        // S-13. The command is still RECOGNIZED after retirement (REQ-A22) — only what
        // happens once recognized changes (see `test_init_config_retired_message_...`).
        // A literal `parse_init_config` returning `false` would fall through to being
        // sent to the agent as a normal chat message, which is worse than a clear
        // retirement notice.
        assert!(parse_init_config("/init-config"));
        assert!(parse_init_config("  /init-config  "));
        assert!(!parse_init_config("/init-configurator"));
        assert!(!parse_init_config("/help"));
    }

    /// SC-A22: `/init-config` was retired (REQ-A22) — mirrors `reject_init_config` in
    /// `main.rs` for the CLI flag. Retiring is not muting: a message pointing at the
    /// replacement, not a silent fall-through to "send this text to the agent".
    #[test]
    fn test_init_config_retired_message_points_at_magi_init() {
        let msg = init_config_retired_message();
        assert!(
            msg.contains("magi init"),
            "a message that doesn't name the replacement turns a one-line migration \
             into a search: {msg}"
        );
    }

    #[test]
    fn test_spinner_frame_cycles() {
        // Advancing wraps around the frame set; never out of range.
        let n = SPINNER_FRAMES.len();
        assert_eq!(next_spinner_frame(0), 1);
        assert_eq!(next_spinner_frame(n - 1), 0);
    }

    #[test]
    fn test_thinking_indicator_has_label_and_a_spinner_glyph() {
        let s = thinking_indicator(0);
        assert!(s.contains("Pensando"), "indicator text: {s:?}");
        assert!(
            s.ends_with(SPINNER_FRAMES[0]),
            "ends with spinner glyph: {s:?}"
        );
    }

    #[test]
    fn test_show_thinking_defaults_off_and_toggles() {
        let (event_tx, _) = mpsc::channel(1);
        let (_, response_rx) = mpsc::channel(1);
        let (_, approval_rx) = mpsc::channel(1);
        let mut app = App::new(event_tx, response_rx, approval_rx);
        assert!(
            !app.show_thinking,
            "default is the compact indicator (mode B)"
        );
        assert!(app.toggle_show_thinking()); // → verbose (A)
        assert!(app.show_thinking);
        assert!(!app.toggle_show_thinking()); // → compact (B)
        assert!(!app.show_thinking);
    }

    #[test]
    fn test_reasoning_compact_mode_shows_indicator_not_text() {
        let (event_tx, _) = mpsc::channel(1);
        let (_, response_rx) = mpsc::channel(1);
        let (_, approval_rx) = mpsc::channel(1);
        let mut app = App::new(event_tx, response_rx, approval_rx);
        // Default mode B: reasoning sets the activity indicator, never the message text.
        app.on_reasoning("secret thoughts".to_string());
        assert!(app.thinking_active);
        assert!(
            app.messages.is_empty(),
            "reasoning text must NOT appear in messages in compact mode"
        );
        // Content arriving clears the indicator and starts the real reply.
        app.append_stream_delta("Answer".to_string());
        assert!(!app.thinking_active);
        assert!(app.messages.last().unwrap().contains("Answer"));
    }

    #[test]
    fn test_reasoning_verbose_mode_appends_text() {
        let (event_tx, _) = mpsc::channel(1);
        let (_, response_rx) = mpsc::channel(1);
        let (_, approval_rx) = mpsc::channel(1);
        let mut app = App::new(event_tx, response_rx, approval_rx);
        app.show_thinking = true; // mode A (verbose, for debug)
        app.on_reasoning("visible thoughts".to_string());
        assert!(
            app.messages.last().unwrap().contains("visible thoughts"),
            "verbose mode streams the reasoning into the message"
        );
    }

    #[test]
    fn test_finalize_stream_clears_thinking_indicator() {
        let (event_tx, _) = mpsc::channel(1);
        let (_, response_rx) = mpsc::channel(1);
        let (_, approval_rx) = mpsc::channel(1);
        let mut app = App::new(event_tx, response_rx, approval_rx);
        // A turn that reasons but ends with NO content (empty answer / error /
        // tool-only) must not leave the spinner stuck on screen forever.
        app.on_reasoning("thinking".to_string());
        assert!(app.thinking_active);
        app.finalize_stream();
        assert!(!app.thinking_active, "end-of-turn must drop the indicator");
    }

    #[test]
    fn test_parse_consult_command() {
        assert_eq!(
            super::parse_consult_command("/consult should we X?"),
            Some("should we X?")
        );
        assert_eq!(super::parse_consult_command("/consult"), Some(""));
        assert_eq!(super::parse_consult_command("hello"), None);
        assert_eq!(super::parse_consult_command("/consultation"), None);
    }

    #[test]
    fn test_parse_tui_consult_accepts_an_explicit_mode_and_keeps_the_query() {
        let cmd = super::parse_tui_consult("/consult --mode design ¿esto o aquello?").unwrap();
        assert_eq!(cmd.mode, Some(Mode::Design));
        assert_eq!(cmd.query, "¿esto o aquello?");
    }

    #[test]
    fn test_parse_tui_consult_accepts_a_bare_consult_with_no_mode() {
        let cmd = super::parse_tui_consult("/consult").unwrap();
        assert_eq!(cmd.mode, None);
        assert_eq!(cmd.query, "");
    }

    /// m2: pins last-value-wins on a repeated `--mode` as a DECISION, not an oversight —
    /// this parser fails closed on everything else it doesn't recognize, so a reader
    /// needs a test (and the rustdoc line above `parse_tui_consult`) to tell the two
    /// apart. Mirrors clap's own behavior for a non-multiple arg.
    #[test]
    fn test_parse_tui_consult_last_mode_flag_wins_on_repeat() {
        let cmd = super::parse_tui_consult("/consult --mode design --mode analysis query").unwrap();
        assert_eq!(cmd.mode, Some(Mode::Analysis));
        assert_eq!(cmd.query, "query");
    }

    #[test]
    fn test_parse_tui_consult_rejects_untrusted_content() {
        assert_eq!(
            super::parse_tui_consult("/consult --untrusted-content x"),
            Err(super::TuiConsultParseError::UnsupportedFlag(
                "--untrusted-content".to_string()
            )),
            "la TUI no expone la marca: ahí hay un humano que eligió el contenido"
        );
    }

    #[test]
    fn test_parse_tui_consult_rejects_mode_without_a_value() {
        assert_eq!(
            super::parse_tui_consult("/consult --mode"),
            Err(super::TuiConsultParseError::MissingModeValue)
        );
    }

    #[test]
    fn test_parse_tui_consult_rejects_an_unknown_mode_label() {
        assert_eq!(
            super::parse_tui_consult("/consult --mode banana query"),
            Err(super::TuiConsultParseError::UnknownMode(
                "banana".to_string()
            ))
        );
    }

    #[test]
    fn test_parse_tui_consult_rejects_a_non_consult_line() {
        assert_eq!(
            super::parse_tui_consult("hello"),
            Err(super::TuiConsultParseError::NotAConsultCommand)
        );
    }

    // -------------------------------------------------------------------
    // resolve_tui_consult_mode — REQ-A07d, fix round 1 (`/consult`'s own
    // dispatch, not just its acceptance).
    // -------------------------------------------------------------------

    /// Doble de [`ModeClassifier`] que PANICS si se lo invoca — la forma más fuerte
    /// de probar "cero llamadas de clasificación": un `CountingClassifier` en 0
    /// podría ocultar un bug donde se llama pero se descarta el resultado; este no
    /// deja esa salida.
    struct NeverClassifier;

    #[async_trait::async_trait]
    impl ModeClassifier for NeverClassifier {
        async fn classify(&self, _content: &str) -> Option<Mode> {
            panic!("classifier must not be called when a mode is already declared");
        }
    }

    /// Doble que siempre devuelve una etiqueta fija, para el camino de inferencia.
    struct FixedClassifier(Mode);

    #[async_trait::async_trait]
    impl ModeClassifier for FixedClassifier {
        async fn classify(&self, _content: &str) -> Option<Mode> {
            Some(self.0)
        }
    }

    /// SC-A07b (mitad TUI) — Fix round 1. Antes de este fix, el handler de
    /// `UiEvent::Consult` corría `Mode::Analysis` sin condición y la entrada del
    /// loop usaba `parse_consult_command`, que no entendía `--mode` en absoluto:
    /// `/consult --mode design ¿esto o aquello?` dejaba el texto `--mode design`
    /// DENTRO de la pregunta y de todos modos analizaba en `Analysis`. Este test
    /// fija las dos mitades del arreglo: el modo resuelto es `Design`, y la query
    /// que llegaría a `Magi::analyze` ya no contiene el flag.
    #[tokio::test]
    async fn an_explicit_mode_reaches_analyze_as_declared_and_strips_the_flag_from_the_query() {
        let cmd = super::parse_tui_consult("/consult --mode design ¿esto o aquello?").unwrap();
        let (mode, query) = super::resolve_tui_consult_mode(cmd, None, false, &NeverClassifier)
            .await
            .unwrap();
        assert_eq!(
            mode,
            Mode::Design,
            "el --mode explícito debe ganar, nunca Analysis por defecto"
        );
        assert_eq!(query, "¿esto o aquello?");
        assert!(
            !query.contains("--mode"),
            "el flag no debe sobrevivir en el texto que llega a analyze: {query:?}"
        );
    }

    /// SC-A07k (mitad TUI): `[magi].default_mode` le gana a la clasificación —
    /// sin `--mode` en el comando, si el operador declaró un default, se usa ESE
    /// y el clasificador nunca se invoca.
    #[tokio::test]
    async fn configured_default_mode_wins_without_classifying() {
        let cmd = super::parse_tui_consult("/consult ¿esto o aquello?").unwrap();
        let (mode, query) =
            super::resolve_tui_consult_mode(cmd, Some(Mode::CodeReview), false, &NeverClassifier)
                .await
                .unwrap();
        assert_eq!(mode, Mode::CodeReview);
        assert_eq!(query, "¿esto o aquello?");
    }

    /// Sin `--mode` y sin `default_mode`, el clasificador SÍ se consulta y su
    /// respuesta se usa — el camino de inferencia sigue vivo en esta superficie.
    #[tokio::test]
    async fn without_mode_or_config_the_classifier_is_consulted() {
        let cmd = super::parse_tui_consult("/consult ¿esto o aquello?").unwrap();
        let (mode, _query) =
            super::resolve_tui_consult_mode(cmd, None, false, &FixedClassifier(Mode::Design))
                .await
                .unwrap();
        assert_eq!(mode, Mode::Design);
    }

    /// SC-A07r (mitad TUI): el operador declaró `untrusted_content = true` en su
    /// `magi.toml` y ni `--mode` ni `default_mode` nombran una lente — falla
    /// cerrado, sin clasificar.
    #[tokio::test]
    async fn operator_declared_untrusted_content_fails_closed_without_a_mode() {
        let cmd = super::parse_tui_consult("/consult ¿esto o aquello?").unwrap();
        let err = super::resolve_tui_consult_mode(cmd, None, true, &NeverClassifier)
            .await
            .expect_err("sin modo declarado, la marca del operador debe fallar cerrado");
        assert!(matches!(
            err,
            ModeError::UntrustedContentRequiresExplicitMode
        ));
    }

    #[test]
    fn test_push_notice_stores_with_warning_prefix() {
        // AgentResponse::Notice must be rendered with the ⚠ prefix so it is
        // visually distinct from model Content (REQ-29 / D1/D2 routing).
        let (event_tx, _) = mpsc::channel(1);
        let (_, response_rx) = mpsc::channel(1);
        let (_, approval_rx) = mpsc::channel(1);
        let mut app = App::new(event_tx, response_rx, approval_rx);

        app.push_notice("memory: context assembly failed".to_string());

        assert_eq!(
            app.messages.len(),
            1,
            "push_notice must add exactly one message"
        );
        let stored = &app.messages[0];
        assert!(
            stored.starts_with("⚠ "),
            "notice message must start with the ⚠ prefix; got: {stored:?}"
        );
        assert!(
            stored.contains("context assembly failed"),
            "notice message must contain the original text; got: {stored:?}"
        );
    }

    #[test]
    fn test_push_notice_scrolls_to_tail() {
        // push_notice must snap back to the tail, consistent with push_message.
        let (event_tx, _) = mpsc::channel(1);
        let (_, response_rx) = mpsc::channel(1);
        let (_, approval_rx) = mpsc::channel(1);
        let mut app = App::new(event_tx, response_rx, approval_rx);
        app.scroll_offset = 5;
        app.push_notice("any notice".to_string());
        assert_eq!(
            app.scroll_offset, 0,
            "push_notice must reset scroll_offset to 0 (follow-tail)"
        );
    }

    #[test]
    fn test_full_report_renders_without_panic() {
        let (event_tx, _) = mpsc::channel(1);
        let (_, response_rx) = mpsc::channel(1);
        let (_, approval_rx) = mpsc::channel(1);
        let mut app = App::new(event_tx, response_rx, approval_rx);

        let report = format!(
            "+{}+\n|  MAGI VERDICT  |\n+{}+\nMelchior: APPROVE — café ☕ {}\n",
            "=".repeat(50),
            "=".repeat(50),
            "x".repeat(500)
        );
        // Push each line of the report as a separate message (simulates how
        // AgentResponse::Text is rendered line-by-line in run_app).
        for line in report.lines() {
            app.push_message(line.to_string());
        }
        // Must not panic; assert that the VERDICT line is present.
        assert!(
            app.messages.iter().any(|m| m.contains("MAGI VERDICT")),
            "expected a message containing 'MAGI VERDICT', got: {:?}",
            app.messages
        );
    }
}
