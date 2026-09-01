// Author: Julian Bolivar
// Version: 0.18.0
// Date: 2026-08-31

// The strict lint set of `src/logging/`, applied HERE because this file
// implements `NoticeSink` and its contract says an implementation must not
// panic. `Cargo.toml` sets `panic = "abort"` in release, so a panicking sink
// kills the process — and a contract stated in a rustdoc that no lint enforces
// in the file where it is violated is a convention, not a guarantee.
#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
#![cfg_attr(not(test), deny(clippy::panic))]

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
use magi_rs::magi::mode::{
    normalize_label, resolve_mode_guarded, ModeClassifier, ModeError, ModeResolution, ModeSources,
};
use magi_rs::notices::{emit_notices_into, Notice};
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
use tokio_util::sync::CancellationToken;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

/// A vault-backed secret store shared with `main.rs`, used by the `/login`
/// and `/logout` handlers. `None` in an ephemeral (no-persistence) session.
///
/// The concrete `VaultStore` behind this trait object is `Send` but
/// deliberately **not** `Sync` (its mask rotates on every access); the
/// `Mutex` supplies the exclusion `Mutex<T>: Sync` requires of `T: Send`.
pub type SharedSecretStore = Arc<Mutex<dyn SecretStore + Send>>;

/// The last-resort mouth for startup notices when NO logging layer exists.
///
/// # Why the terminal cannot use `stderr` for this
///
/// `init_logging` is guarded on a discovered `.magi/` workspace, so a session
/// started outside one installs no subscriber and
/// [`magi_rs::notices::emit_notices_into`] takes its fallback branch. Writing
/// to `stderr` there puts the lines on the PRIMARY buffer, which
/// `EnterAlternateScreen` swaps out immediately afterwards — so the one message
/// that case exists to deliver, "no `.magi/` state directory found — run
/// `magi init`", is written and hidden in the same breath, and reappears only
/// once the user quits. This writes into the transcript instead, where the
/// layer's screen branch would have put it.
///
/// The screen POLICY is not re-implemented here: `emit_notices_into` has
/// already ordered, deduplicated and filtered to `WARN` and above by the time
/// anything reaches this writer.
struct NoticeTranscript {
    /// The response channel `run_app` drains into the message list.
    tx: mpsc::Sender<AgentResponse>,
    /// Bytes seen since the last newline. `emit_notices_into` writes one
    /// `writeln!` per notice, so this is normally empty between notices; it
    /// exists because `Write` may be handed any split of the bytes and a
    /// notice cut across two calls must not become two lines.
    partial: Vec<u8>,
}

impl NoticeTranscript {
    /// Wraps the response channel as a `Write`.
    fn new(tx: mpsc::Sender<AgentResponse>) -> Self {
        Self {
            tx,
            partial: Vec::new(),
        }
    }

    /// Puts one finished line in the transcript.
    ///
    /// `try_send` cannot fail in practice at the one call site: this runs
    /// before `run_app`, so the receiver is alive and the 100-slot channel is
    /// empty. If it ever did, `stderr` is still correct HERE and only here —
    /// the alternate screen is not up yet — so the line is degraded to the
    /// mouth it would have had, never dropped.
    fn line(&self, text: String) {
        if let Err(mpsc::error::TrySendError::Full(AgentResponse::Notice(text)))
        | Err(mpsc::error::TrySendError::Closed(AgentResponse::Notice(text))) =
            self.tx.try_send(AgentResponse::Notice(text))
        {
            eprintln!("{text}");
        }
    }
}

impl io::Write for NoticeTranscript {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.partial.extend_from_slice(buf);
        while let Some(newline) = self.partial.iter().position(|b| *b == b'\n') {
            let mut finished: Vec<u8> = self.partial.drain(..=newline).collect();
            finished.pop();
            if finished.last() == Some(&b'\r') {
                finished.pop();
            }
            self.line(String::from_utf8_lossy(&finished).into_owned());
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        if !self.partial.is_empty() {
            let trailing = std::mem::take(&mut self.partial);
            self.line(String::from_utf8_lossy(&trailing).into_owned());
        }
        Ok(())
    }
}

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

/// The mode classifier's one-time notices, routed through the TUI's notice channel instead of
/// stderr (REQ-A07c's two notices: the cost heads-up and the expiry report).
///
/// **Why this type exists.** `ProcessNoticeSink` writes with `eprintln!`, which is correct for
/// the headless CLI and wrong for the TUI: from the moment `run_tui_ext` enters raw mode and
/// the alternate screen, an external write to stderr lands on top of the ratatui frame and
/// desynchronises its previous-frame diff buffer. That is precisely why
/// [`StreamPiece::Notice`](crate::agent::StreamPiece) exists — and the classifier fires on the
/// *first* `/consult` without `--mode` under the default scaffold, so it was not an exotic
/// corner. The destination is injected rather than hardcoded because the same classifier code
/// runs headless, where stderr **is** the right destination.
///
/// **Why the channel is attached late.** `main.rs` builds the classifier before `run_tui_ext`
/// creates the response channel, so the sink is constructed first and connected afterwards.
/// Until it is connected — i.e. before raw mode is entered — a notice still has to go
/// somewhere, and stderr is then the correct place: the frame it could corrupt does not exist
/// yet.
pub struct TuiNoticeSink {
    /// Keys already emitted, for the "once per process" contract of [`NoticeSink::once`].
    seen: Mutex<std::collections::BTreeSet<&'static str>>,
    /// The TUI's response channel, present once [`TuiNoticeSink::attach`] has run.
    tx: Mutex<Option<mpsc::Sender<AgentResponse>>>,
    /// Notices that could not be delivered through `tx` (closed, or full and later closed)
    /// while the alternate screen might still be up — see [`Self::flush`] for why a fallback
    /// never prints for itself.
    pending: Arc<Mutex<PendingNotices>>,
}

/// The state [`TuiNoticeSink::pending`] moves through exactly once, in exactly one direction.
///
/// **Why a state, not a flag plus a buffer.** A prior version of this sink fell back to
/// `eprintln!` directly from inside `once` — including from the `tokio::spawn`ed task queued
/// when the channel was momentarily full (MS2 gate S7-f finding, Caspar): that spawned task's
/// own "the channel closed while I waited" check and its `eprintln!` were two steps with an
/// `.await` between them, and nothing synchronized either step against `run_tui_ext` calling
/// `LeaveAlternateScreen` on a different task. The fallback could print while the alternate
/// screen was still up, or concurrently with the very call that was tearing it down — a real,
/// if low-probability, frame corruption.
///
/// The fix is not "check a flag, then write" (that is the same two-step race with extra
/// steps) — it is "the check AND the write happen under the SAME lock as the state
/// transition, so there is exactly one linearization point". While `Buffering`, a fallback
/// defers its message into the `Vec` under the lock and returns; nothing is printed. Once
/// [`TuiNoticeSink::flush`] — called by `run_tui_ext` strictly AFTER `LeaveAlternateScreen` —
/// swaps the state to `Flushed` under that same lock, EVERY fallback that observes `Flushed`
/// (whether it ran before, during, or after the swap) knows the terminal has already been
/// restored, because `flush` is the only thing that produces `Flushed` and it is only called
/// post-teardown. There is no window in which "closed" and "still on screen" can be true at
/// once from the writer's point of view.
enum PendingNotices {
    /// The alternate screen may still be active; fallbacks accumulate here instead of
    /// printing.
    Buffering(Vec<String>),
    /// `run_tui_ext` has already left the alternate screen; a fallback observing this state
    /// prints immediately — doing so is now always safe.
    Flushed,
}

impl TuiNoticeSink {
    /// A fresh, unattached sink.
    #[must_use]
    pub fn new() -> Self {
        Self {
            seen: Mutex::new(std::collections::BTreeSet::new()),
            tx: Mutex::new(None),
            pending: Arc::new(Mutex::new(PendingNotices::Buffering(Vec::new()))),
        }
    }

    /// Connects the sink to the TUI's response channel; every later notice is rendered in the
    /// frame instead of written to stderr.
    ///
    /// # Parameters
    /// * `tx` - the sender half of the channel `run_tui_ext` already owns.
    pub fn attach(&self, tx: mpsc::Sender<AgentResponse>) {
        *self.tx.lock().unwrap_or_else(|p| p.into_inner()) = Some(tx);
    }

    /// Routes one fallback message through [`PendingNotices`]: buffered while the alternate
    /// screen might still be active, printed immediately once [`Self::flush`] has run. See
    /// the enum's doc for why the check and the write must share one lock with the state
    /// transition.
    fn defer_or_print(pending: &Mutex<PendingNotices>, msg: String) {
        let mut guard = pending.lock().unwrap_or_else(|p| p.into_inner());
        match &mut *guard {
            PendingNotices::Flushed => {
                drop(guard);
                eprintln!("{msg}");
            }
            PendingNotices::Buffering(queued) => queued.push(msg),
        }
    }

    /// Marks the sink as safe to print immediately from now on, and returns every message
    /// that had to be deferred before this call. **Must be called by `run_tui_ext` AFTER
    /// `LeaveAlternateScreen` — never before** — the caller is responsible for printing the
    /// returned messages (this method has no I/O of its own, which is what keeps it testable
    /// without capturing stderr). A second call returns an empty `Vec`: once `Flushed`, later
    /// fallbacks print themselves immediately instead of accumulating here.
    pub(crate) fn flush(&self) -> Vec<String> {
        let mut guard = self.pending.lock().unwrap_or_else(|p| p.into_inner());
        match std::mem::replace(&mut *guard, PendingNotices::Flushed) {
            PendingNotices::Buffering(queued) => queued,
            PendingNotices::Flushed => Vec::new(),
        }
    }

    /// Test-only, non-destructive peek at how many notices are currently buffered — lets a
    /// test poll for "the background fallback landed" without calling [`Self::flush`] itself,
    /// which would prematurely flip the sink to `Flushed` and hide the very race window under
    /// test.
    #[cfg(test)]
    pub(crate) fn pending_len(&self) -> usize {
        match &*self.pending.lock().unwrap_or_else(|p| p.into_inner()) {
            PendingNotices::Buffering(queued) => queued.len(),
            PendingNotices::Flushed => 0,
        }
    }
}

impl Default for TuiNoticeSink {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::agent::mode_classifier::NoticeSink for TuiNoticeSink {
    /// Emits `msg` the first time it is called with `key`.
    ///
    /// A notice is **never dropped silently** (B9), and it never corrupts a LIVE frame either
    /// — the two fallbacks used to be conflated (MAGI S7 fix round, finding 1) and that was a
    /// real bug: `tx.try_send(..).is_ok()` failing on a **full** channel fell back to
    /// `eprintln!` exactly like a closed one, even though a full channel means the TUI is very
    /// much alive and about to have its frame written over — precisely the corruption this
    /// sink exists to prevent (see the module rustdoc). The two cases are distinguished by
    /// `TrySendError`'s variant, not lumped into "any send failure":
    /// - **No channel attached yet** (`self.tx` is `None`) — raw mode has not been entered, so
    ///   there is no frame to protect; stderr is correct — printed directly, no deferral needed.
    /// - **`TrySendError::Closed`** — the receiver is gone, which only happens once `run_app`
    ///   has returned and dropped it, i.e. once teardown has *started*. That is not the same as
    ///   teardown having *finished*: `LeaveAlternateScreen` runs on a different task and is not
    ///   synchronized with this one, so printing here directly used to be a second instance of
    ///   the same race the `Full` branch has (MS2 gate S7-f finding, Caspar). This now routes
    ///   through [`Self::defer_or_print`] instead of printing unconditionally.
    /// - **`TrySendError::Full`** — the frame IS at risk, so this never prints inline. The
    ///   classifier (`agent::mode_classifier::ProviderClassifier`) only ever emits two distinct
    ///   keys total in a process's lifetime (the cost heads-up, the expiry report), deduplicated
    ///   by `seen` above — so a bounded background task per full-channel event cannot
    ///   accumulate; it waits for room with a real `.send().await` and, if the channel closes
    ///   before room frees up (the run ended), defers through the same
    ///   [`Self::defer_or_print`] path rather than printing directly from the spawned task —
    ///   see [`PendingNotices`] for why that indirection is what actually closes the race.
    fn once(&self, key: &'static str, msg: &magi_rs::logging::auditor::Audited) {
        {
            let mut seen = self.seen.lock().unwrap_or_else(|p| p.into_inner());
            if !seen.insert(key) {
                return;
            }
        }
        self.route(msg.as_str());
    }

    /// Delivers without deduplicating; the auditor already did it, by
    /// `(secret, target)`, which a `'static` key cannot express.
    fn emit(&self, msg: &magi_rs::logging::auditor::Audited) {
        self.route(msg.as_str());
    }
}

impl TuiNoticeSink {
    /// The routing every notice shares, whatever deduplicated it.
    ///
    /// Extracted when `emit` arrived: two copies of the three-way fallback below
    /// is two chances for one of them to lose the distinction between a closed
    /// channel and a full one, which is the bug this whole comment records.
    fn route(&self, msg: &str) {
        let tx = self
            .tx
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .cloned();
        let Some(tx) = tx else {
            eprintln!("{msg}");
            return;
        };
        match tx.try_send(AgentResponse::Notice(msg.to_string())) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Closed(_)) => {
                Self::defer_or_print(&self.pending, msg.to_string());
            }
            Err(mpsc::error::TrySendError::Full(_)) => {
                // The frame is live and momentarily saturated — printing here is exactly the
                // corruption this sink exists to prevent, so this waits for room instead
                // (`spawn`, not a blocking wait, since `once` is a sync trait method called
                // from inside an async caller and must not stall its executor thread).
                let msg = msg.to_string();
                let pending = Arc::clone(&self.pending);
                tokio::spawn(async move {
                    if tx.send(AgentResponse::Notice(msg.clone())).await.is_err() {
                        // The channel closed while this was queued (the run ended before
                        // room freed up) — deferred rather than printed here directly, because
                        // this task's own "found it closed" and "write to stderr" are two
                        // steps with nothing synchronizing them against `LeaveAlternateScreen`
                        // on the main task; see `PendingNotices`.
                        Self::defer_or_print(&pending, msg);
                    }
                });
            }
        }
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

/// Validates a `/consult` query's size before any model call, INCLUDING a
/// classification call (REQ-A11b, SC-A11c) — the SAME `check_query_size` the
/// `ConsultTool` path and the headless direct path apply, against the SAME
/// `MagiConfig`-resolved cap (`TuiMagiRuntimeConfig::max_query_bytes`).
///
/// Extracted from the `UiEvent::Consult` handler for the same reason as
/// [`consult_unavailable_response`]/[`handle_login`]: the full TUI event loop is
/// intractable to drive in a test, so the decision is tested here as a plain fn.
///
/// # Returns
/// `Ok(())` when the query passes; `Err(AgentResponse::Error(_))` with the exact
/// text `check_query_size` produced, ready to send back to the user.
fn tui_consult_size_check(query: &str, max_query_bytes: usize) -> Result<(), AgentResponse> {
    crate::tools::consult::check_query_size(query, max_query_bytes)
        .map_err(|e| AgentResponse::Error(e.to_string()))
}

/// The `[DEGRADED: ...]` banner text (REQ-A12c, SC-A12f, fix round 4, finding 2) —
/// TUI-only presentation, not part of the JSON `report`/`degraded` shape the other
/// two surfaces emit (see [`tui_consult_success_body`]'s own doc).
///
/// A named `const` rather than an inline literal (B3/B4): [`tui_consult_success_body`]
/// needs it standalone (never pre-joined to the report) so a caller that has to
/// truncate can reserve room for it BEFORE cutting the report — see
/// [`tui_consult_success_reply`] for why that ordering is the entire point of a fix
/// applied in this task.
const DEGRADED_BANNER: &str =
    "[DEGRADED: fewer than 3 agents responded — consensus may be unreliable]";

/// The `/consult` response BODY pieces for a successful `MagiReport` (REQ-A12c,
/// SC-A12f, fix round 4, finding 2; restructured in a later fix — see below).
///
/// **Before the fix round 4 changes, `UiEvent::Consult` built its body from
/// `report.report` directly — bypassing `annotate_report_text` entirely.** A human
/// typing `/consult` is arguably the most direct "first use" surface REQ-A12c
/// describes, and they got NONE of this task's keyless-auth guidance: a partial
/// failure (`degraded = true`) with a seat rejected on auth under a keyless kind
/// rendered as an unexplained `[DEGRADED: …]` banner over the raw report, same as
/// any other partial failure.
///
/// Reuses [`crate::tools::consult::annotate_report_text`] — the SAME function
/// `ConsultTool::execute`/`analyze_direct` already call — so the TUI never carries its
/// own, fourth copy of this wording (B3).
///
/// **Returns the banner and the annotated report SEPARATE, not pre-joined.** A prior
/// version of this function returned one already-joined `String`
/// (`banner + "\n\n" + annotated`), and the production call site truncated THAT whole
/// string with [`crate::tools::consult::truncate_report`]. That function's three
/// levels all cut the kept region starting from the verdict anchor onward — which
/// sits AFTER a pre-joined banner — so a sufficiently long degraded report silently
/// lost its banner on truncation: a thin, unreliable consensus rendered
/// indistinguishable from a full-strength one, which is worse than an obviously
/// broken reply because there is nothing telling the user to distrust it. Keeping the
/// pieces separate lets [`tui_consult_success_reply`] reserve room for the banner
/// BEFORE truncating the report, instead of after.
///
/// Extracted from the `UiEvent::Consult` handler for the same reason as
/// [`consult_unavailable_response`]/[`handle_login`]: the full TUI event loop is
/// intractable to drive in a test.
struct ConsultSuccessBody {
    /// The caveat that must survive any truncation applied afterward — present
    /// only when `report.degraded`.
    banner: Option<&'static str>,
    /// The annotated report text (REQ-A12c) — never the raw `MagiReport::report`
    /// on its own.
    annotated: String,
}

fn tui_consult_success_body(report: &MagiReport, kind: ProviderKind) -> ConsultSuccessBody {
    ConsultSuccessBody {
        banner: report.degraded.then_some(DEGRADED_BANNER),
        annotated: crate::tools::consult::annotate_report_text(report, kind),
    }
}

/// The truncation-safe `/consult` reply for a successful `MagiReport` — the exact
/// combination `UiEvent::Consult`'s success arm performs (fix: see
/// [`tui_consult_success_body`]'s own doc for the defect this replaces).
///
/// When [`ConsultSuccessBody::banner`] is present, `cap` is applied via
/// [`crate::tools::consult::truncate_report_with_preserved_prefix`], which reserves
/// room for the banner FIRST and truncates the report under whatever remains — the
/// banner is the highest-value text in the reply (it says how much to trust
/// everything below it) and is never the thing sacrificed to fit the cap. Without a
/// banner this is exactly [`crate::tools::consult::truncate_report`] on the annotated
/// text, unchanged from before this fix.
///
/// Extracted from the `UiEvent::Consult` handler for the same reason as every other
/// `tui_consult_*` helper in this file: the full TUI event loop is intractable to
/// drive in a test, so the exact bytes the user receives are verified here instead.
fn tui_consult_success_reply(
    report: &MagiReport,
    kind: ProviderKind,
    cap: usize,
) -> crate::tools::consult::Truncated {
    let pieces = tui_consult_success_body(report, kind);
    match pieces.banner {
        Some(banner) => crate::tools::consult::truncate_report_with_preserved_prefix(
            banner,
            &pieces.annotated,
            cap,
        ),
        None => crate::tools::consult::truncate_report(&pieces.annotated, cap),
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

/// The truncation-safe `/consult` reply for a FAILED `Magi::analyze()` call — the error-arm
/// counterpart of [`tui_consult_success_reply`] (REQ-A11b/SC-A11d, S7 gate re-review fix).
///
/// **Before this fix, the error arm of `UiEvent::Consult` sent [`tui_consult_error_body`]'s
/// output straight to `AgentResponse::Error` with no cap at all.** The success arm has been
/// bounded since the REQ-A11b work landed, but the error arm was never brought in line with
/// it: a foreign provider's HTTP error body can be arbitrarily long (`ProviderError::Http`
/// carries the response body verbatim), so an unbounded error reply defeated the exact
/// budget the success path exists to enforce, on the surface most likely to actually hit
/// it — an HTTP error page or a misconfigured endpoint's response.
///
/// Reuses [`crate::tools::consult::truncate_report`] directly rather than the
/// preserved-prefix variant [`tui_consult_success_reply`] uses for the `[DEGRADED: ...]`
/// banner: [`tui_consult_error_body`] never carries a `MagiReport`'s verdict/finding
/// structure, so there is no anchor to preserve — `truncate_report` degrades to its
/// byte-only level for text with no contractual anchors, which is exactly what an
/// arbitrary error string needs.
///
/// Also runs the result through [`crate::agent::Agent::sanitize_text`] (sixth-pass gate
/// finding, S7): the success arm's call site sanitizes `tui_consult_success_reply`'s output
/// before rendering it, but this error arm did not, even though `tui_consult_error_body`
/// embeds `explain_magi_error`'s text — which in turn can carry a `ProviderError::Http`
/// body composed by a foreign HTTP endpoint we do not control. `explain_magi_error` already
/// redacts credentials (B11) via `redact_foreign_error`, but redaction is not sanitization:
/// it strips secrets, not ANSI escapes or control characters. Sanitizing AFTER truncation,
/// same order as the success arm, so a cut mid-escape-sequence is still cleaned up rather
/// than leaving a dangling `ESC` at the tail.
fn tui_consult_error_reply(
    err: &MagiError,
    kind: ProviderKind,
    cap: usize,
) -> crate::tools::consult::Truncated {
    let mut truncated =
        crate::tools::consult::truncate_report(&tui_consult_error_body(err, kind), cap);
    truncated.text = crate::agent::Agent::sanitize_text(&truncated.text);
    truncated
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
/// - **The error text is redacted.** `e` is a foreign [`ProviderError`] from magi-core — its `Display` can cite the endpoint URL, and magi-core does not know this crate's redaction rule (REQ-A16c). Every other foreign- error surface in this codebase routes through [`redact_foreign_error`] for exactly this reason; this one must too.
/// - **Nothing keeps using the OLD credentials.** By the time this runs, [`handle_login`] already wrote a NEW key to the vault, so the running session and the vault now disagree about what the current credential is. Leaving `consult_magi_runner` and the registered `consult` tool pointed at whatever built successfully before this attempt — possibly nothing, possibly a stale provider from an earlier session — would let consult keep answering under a diverged credential while the user was told the rebuild failed. Dropping both makes the direct `/consult` path and the autonomous tool path fail closed the same way an unconfigured trio always does (REQ-A06), instead of silently using the wrong thing.
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

/// Rows the status line takes while an operation is running.
const STATUS_ROW_HEIGHT_SHOWN: u16 = 1;
/// Rows the status line takes while there is nothing to show (REQ-L26).
const STATUS_ROW_HEIGHT_COLLAPSED: u16 = 0;

/// What the row says while a `/consult` is being deliberated.
///
/// A program constant, never a composed `String` — see [`StatusRow`] for why
/// the type of the text is the guarantee rather than the convention.
const STATUS_CONSULTING_THE_TRIO: &str = "consulting the trio…";

/// The ephemeral status row: one line while a long operation runs, and nothing
/// at all the rest of the time.
///
/// **It lives OUTSIDE [`App::messages`]** (REQ-L25). The transcript is
/// append-only and both Selection and Visual index into it, so a line that
/// appears and disappears would need mutability there and would change two
/// modes that have no business knowing an operation is in flight.
///
/// **The text is `&'static str`, and that is the guarantee rather than a
/// convention.** The row is ephemeral, so it never passes through the auditor —
/// which means nothing composed at runtime may reach it. A program constant
/// cannot carry a credential, a path or a provider's error body, and the type
/// is what enforces it, the same argument
/// [`SecretName`](magi_rs::logging::auditor::SecretName) makes for a secret's
/// name. If a counter is ever wanted ("seat 2 of 3") it is added as its OWN
/// field with its own bounded format, never concatenated into this text.
///
/// **Cloning shares one row, and that is the point.** The task that starts the
/// operation and the task that draws the frame are different tasks: the event
/// loop holds one handle and [`App`] holds another, both naming the same line.
/// The shared cell is how the renderer gets to see what the other task set.
///
/// **Two concurrent setters are possible and nothing here prevents them.** The
/// state is interior-mutable, so `set` takes `&self`: an exclusive borrow would
/// exclude per HANDLE and not per row, promising a guarantee the type cannot
/// keep. What actually happens with two overlapping `set` calls is that the
/// later one overwrites the line, and the first guard to drop collapses the row
/// while the other operation is still running — one row, one line, last writer
/// wins. It is pinned by
/// `test_two_handles_on_one_status_row_do_not_exclude_each_other`. There is one
/// setter today; whoever adds the second — R-L15 names a startup probe as the
/// other candidate — has to decide the policy rather than assume the borrow
/// checker already did.
///
/// # Example
///
/// ```ignore
/// let row = StatusRow::new();
/// {
///     let _showing = row.set("consulting the trio…");
///     // …the long operation runs here; the row is one line tall…
/// }
/// // The guard is gone, so the row is collapsed again.
/// assert_eq!(row.height(), 0);
/// ```
#[derive(Clone, Default)]
pub struct StatusRow {
    /// The line currently shown, or `None` while the row is collapsed.
    shown: Arc<Mutex<Option<&'static str>>>,
}

impl StatusRow {
    /// A collapsed row.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Shows `text` until the returned guard is dropped.
    ///
    /// # Parameters
    ///
    /// * `text` — a program constant naming the operation, supplied by whoever
    ///   STARTS it. Only operations known to be long qualify (R-L15).
    ///
    /// # Returns
    ///
    /// The guard whose `Drop` clears the row — on success, on failure, on
    /// cancellation and on a panic alike (REQ-L27).
    ///
    /// # Why `&self`
    ///
    /// The row is interior-mutable because two tasks share it, so `&mut self`
    /// would exclude per handle rather than per row and promise a guarantee
    /// this type cannot keep. See [`StatusRow`] for what two overlapping setters
    /// actually do.
    ///
    /// # Complexity
    ///
    /// `O(1)`.
    pub fn set(&self, text: &'static str) -> StatusGuard<'_> {
        *self.cell() = Some(text);
        StatusGuard { row: self }
    }

    /// The rows this line occupies: one while something is shown, zero
    /// otherwise (REQ-L26).
    ///
    /// # Complexity
    ///
    /// `O(1)`.
    #[must_use]
    pub fn height(&self) -> u16 {
        if self.current().is_some() {
            STATUS_ROW_HEIGHT_SHOWN
        } else {
            STATUS_ROW_HEIGHT_COLLAPSED
        }
    }

    /// What the row says, or `None` while it is collapsed.
    ///
    /// # Complexity
    ///
    /// `O(1)`.
    #[must_use]
    pub fn current(&self) -> Option<&'static str> {
        *self.cell()
    }

    /// The shared cell, with a poisoned lock recovered rather than unwrapped —
    /// a panic in one task must not take the row down for the other.
    fn cell(&self) -> std::sync::MutexGuard<'_, Option<&'static str>> {
        self.shown.lock().unwrap_or_else(|p| p.into_inner())
    }
}

/// Clears the [`StatusRow`] when it is dropped (REQ-L27).
///
/// **A drop guard rather than a clear in each exit branch**, because the
/// branches are not enumerable: a failure, a cancellation and a panic all have
/// to give the line back, and the one that gets forgotten is whichever branch
/// is added last. Precedent in this repository: `AbortOnDrop` in `src/task.rs`.
///
/// **On a RELEASE panic this never runs, and the requirement is still met by
/// something else.** `Cargo.toml` sets `panic = "abort"` for the release
/// profile, so a panic in the shipped binary tears the process down without
/// unwinding and no `Drop` is called. What clears the row there is the process
/// exiting — there is no session left to leave a stale line in, so the cost is
/// nil. It is written down because the gap between REQ-L27's wording and what a
/// build profile does is exactly the kind of thing someone later "fixes" by
/// scattering explicit clears through the exit branches, which is the design
/// this guard replaced. The unwinding path is real under the test profile,
/// which is where `test_the_status_row_is_cleared_even_when_the_operation_panics`
/// observes it.
pub struct StatusGuard<'a> {
    /// The row this guard is holding open.
    row: &'a StatusRow,
}

impl Drop for StatusGuard<'_> {
    fn drop(&mut self) {
        *self.row.cell() = None;
    }
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
/// **Returns the whole [`ModeResolution`], not just its `mode`.** REQ-A08 requires the level
/// the mode came from to be visible on all four surfaces, and this one used to drop it: a bare
/// `/consult` running `CodeReview` because the operator configured it, because the classifier
/// inferred it, or because a classification timed out and fell to the default were three
/// materially different situations that rendered identically. Only the resolver knows which
/// happened; a caller cannot re-derive it.
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
) -> Result<(ModeResolution, String), ModeError> {
    let resolution = resolve_mode_guarded(
        ModeSources {
            explicit: cmd.mode,
            configured: default_mode,
            ..ModeSources::default()
        },
        untrusted_content,
        Some(classifier),
        &cmd.query,
    )
    .await?;
    Ok((resolution, cmd.query))
}

/// The line the TUI shows when it dispatches an explicit `/consult`, naming the effective mode
/// and the level it came from (REQ-A08).
///
/// **Why it is folded into the "deliberating" line rather than added to the report.** The reply
/// path reserves room for the `[DEGRADED: …]` banner before truncating the report, and appending
/// to it would change that byte accounting; this line is emitted before the analysis instead, so
/// it appears identically whether the consult succeeds or fails — which is the case REQ-A08's
/// auditability actually needs, since a failed run is when "which lens ran?" is hardest to
/// answer.
///
/// **The source is rendered with `ModeSource`'s `Display`** (loop 1 fix round CE, F26) — a
/// dedicated human-facing label added to `src/magi/mode.rs` rather than the derived `Debug` this
/// line used before. The five strings are pinned to match `Debug`'s output exactly, so this
/// change is not observable to anyone reading the terminal; what changes is that the label is no
/// longer borrowed from a trait meant for developers.
///
/// # Parameters
/// * `res` - the resolution `resolve_tui_consult_mode` produced for this command.
///
/// # Returns
/// The text sent as an [`AgentResponse::Info`] just before the three model calls start.
#[must_use]
fn tui_consult_dispatch_notice(res: &ModeResolution) -> String {
    format!(
        "MAGI deliberating in {} mode ({}) — 3 model calls…",
        res.mode, res.source
    )
}

/// The notice sent when the spawned `magi.analyze` task panics (loop 1 fix round CE, F24).
///
/// **Replaces a pre-existing `eprintln!`** that predates this gate's merge-base and wrote
/// straight to stderr while the TUI holds raw mode and the alternate screen — the same
/// frame-corruption class F3 fixed for the mode classifier's notices. Unlike its sibling arm
/// (`Ok(Err(e))`, whose `eprintln!` was a pure duplicate of the `AgentResponse::Error` sent
/// right after and was simply dropped), this one carries [`tokio::task::JoinError`] detail —
/// e.g. which panic message the task exited with — that the generic "crashed unexpectedly"
/// error text sent alongside it does not transport. Routed through
/// [`AgentResponse::Notice`] instead of dropped, so the diagnostic survives losing its
/// stderr line.
///
/// **Also runs the result through [`crate::agent::Agent::sanitize_text`]** (S7 gate
/// re-review finding, Caspar): this is the third and last arm of the `match join { ... }`
/// block in `run_app`'s `/consult` handler, and the only one that still sent its text to
/// the UI unsanitized — the success arm sanitizes at its call site and the error arm's
/// `tui_consult_error_reply` sanitizes internally. `{join_err}` embeds whatever panic
/// message `magi.analyze` exited with, which this crate does not compose and does not
/// control, and it lands in a terminal sitting in raw mode on the alternate screen —
/// exactly the combination `sanitize_text` exists to guard.
///
/// Empirically, `tokio::task::JoinError`'s `Display` already Debug-escapes a `String`/
/// `&str` panic payload (`{:?}` turns a raw ESC or other control byte into the literal
/// text `\u{1b}`), verified against the tokio 1.53.1 this crate pins — so no panic
/// triggered through the public `panic!` API can put a raw control byte in this
/// function's output today. Sanitizing anyway is still correct: that guarantee lives
/// inside a dependency this crate does not control and did not design for this purpose,
/// the cost is a linear pass over a short string, and it keeps this arm identical in
/// shape to its two siblings instead of exempt for a reason a future reader would have
/// to rediscover.
#[must_use]
fn consult_panic_notice(join_err: &tokio::task::JoinError) -> String {
    crate::agent::Agent::sanitize_text(&format!("[consult] analyze panicked: {join_err}"))
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
    /// Indices into `messages` that are operational notices (memory warning, truncation
    /// advisory, …), as pushed by [`App::push_notice`] — the STRUCTURED record of which
    /// entries deserve the notice style.
    ///
    /// Before this field existed, the Normal-mode renderer decided styling by sniffing a
    /// leading `⚠` glyph out of the rendered text (MAGI S7 finding 2): any model or user
    /// message that happened to start with the same glyph was silently mis-styled as a
    /// notice, and the coupling between "this is a notice" and "this text starts with ⚠"
    /// was invisible — nothing signalled that changing the prefix in [`App::push_notice`]
    /// would also have to be mirrored in the renderer's `starts_with` check. Tracking the
    /// index here instead makes notice-ness structured data the renderer looks up, not a
    /// property it infers from content.
    pub notice_indices: std::collections::HashSet<usize>,
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
    /// Index into `messages` of the entry currently receiving stream deltas — `None` when
    /// not streaming.
    ///
    /// Tracked explicitly rather than inferred (MS2 gate S7-f finding, Balthasar):
    /// `append_stream_delta` used to assume its target was always `messages.last_mut()`, but
    /// `push_notice` legitimately interleaves a NEW message into `messages` while `streaming`
    /// is still `true` — an operational notice does not end the turn, see the `run_app` call
    /// site — which left the notice, not the in-progress reply, as `messages.last_mut()`. The
    /// next delta then appended onto the notice's text instead of the reply's, corrupting
    /// both. Tracking the index directly means ANY interleaved push (not just notices) is
    /// safe: the delta always lands where the stream actually started, regardless of what
    /// else got appended after it.
    pub stream_target: Option<usize>,
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
    /// The ephemeral status line drawn between the transcript and the input.
    ///
    /// A handle onto the row the event loop's task also holds, not a copy of
    /// it: whoever starts a long operation sets the text there, and this side
    /// only reads it while drawing (see [`StatusRow`]).
    pub status_row: StatusRow,
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
            notice_indices: std::collections::HashSet::new(),
            event_tx,
            response_rx,
            approval_rx,
            pending_approval: None,
            mode: AppMode::Normal,
            selected_index: 0,
            visual_cursor: 0,
            visual_selection_start: None,
            streaming: false,
            stream_target: None,
            scroll_offset: 0,
            last_max_scroll: 0,
            last_viewport_height: 0,
            show_thinking: false,
            thinking_active: false,
            spinner_frame: 0,
            status_row: StatusRow::new(),
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
    ///
    /// Delegates to [`Self::insert_str`] (a single-character `&str` view of `c`, via
    /// `char::encode_utf8` over a stack buffer — no allocation) so the selection-clearing and
    /// char-boundary emergency-fallback logic lives in exactly one place (B3) rather than
    /// being duplicated between the single-character and batched insertion paths.
    pub fn insert_char(&mut self, c: char) {
        let mut buf = [0u8; 4];
        self.insert_str(c.encode_utf8(&mut buf));
    }

    /// Inserts `text` at the current cursor position in a single operation.
    ///
    /// MS2 gate S7 seventh-pass fix: clipboard paste (`Ctrl+V`) used to call [`Self::insert_char`]
    /// once per pasted character, and `insert_char` calls `String::insert` — an O(n) tail-shift.
    /// Repeated N times for an N-character paste, that is O(N^2): a 50 KB clipboard paste (a
    /// routine terminal action) would perform on the order of a billion byte-shifts and freeze
    /// the 50ms-poll event loop for seconds. `insert_str` places the whole payload with one
    /// `String::insert_str` call — O(n) total for the whole paste, matching `insert_char`'s
    /// per-character cost for a single character.
    ///
    /// Mirrors `insert_char`'s two invariants exactly, just once instead of per character:
    /// clears any active selection first (so a paste over a selection replaces it, not appends
    /// to it), and falls back the cursor to `0` if it is not on a char boundary — the same
    /// emergency guard `insert_char` uses, kept here rather than only in the single-character
    /// path so every mutator in this file upholds the UTF-8 boundary-safety invariant
    /// independently (`char_indices`/`is_char_boundary`, never a raw byte index).
    pub fn insert_str(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.delete_selection();
        if !self.input.is_char_boundary(self.cursor_position) {
            self.cursor_position = 0; // Emergency fallback, mirrors insert_char.
        }
        self.input.insert_str(self.cursor_position, text);
        self.cursor_position += text.len();
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

    /// Appends an operational notice to the UI history with the `⚠ ` prefix so it is
    /// visually distinct from model output at a glance in a plain-text copy/paste.
    ///
    /// The prefix is cosmetic text, not the styling mechanism: the Normal-mode renderer
    /// looks up `notice_indices` (recorded here) rather than sniffing the prefix back out
    /// of the rendered line — see `notice_indices`'s rustdoc for why that distinction
    /// matters (MAGI S7 finding 2).
    pub fn push_notice(&mut self, text: String) {
        // Ensure the text is valid UTF-8 (it always is, but make the intent explicit).
        let notice = format!("⚠ {}", text);
        self.messages.push(notice);
        self.notice_indices.insert(self.messages.len() - 1);
        self.scroll_offset = 0;
    }

    /// Appends a streaming delta to the in-progress assistant message,
    /// creating the line on the first delta. Append-only; never byte-indexes.
    ///
    /// Targets `stream_target` explicitly rather than `messages.last_mut()` (MS2 gate S7-f
    /// finding, Balthasar): a notice or another message can legitimately land at the end of
    /// `messages` while a stream is still in progress (see `stream_target`'s doc), and writing
    /// blindly to "whatever is last" would then corrupt that interleaved entry instead of the
    /// reply actually being streamed.
    pub fn append_stream_delta(&mut self, delta: String) {
        // Answer content arriving means the thinking phase is over.
        self.thinking_active = false;
        // Streaming content → follow the tail so the live reply stays visible.
        self.scroll_offset = 0;
        if let Some(target) = self
            .stream_target
            .and_then(|idx| self.messages.get_mut(idx))
        {
            target.push_str(&delta);
            return;
        }
        self.messages.push(format!("Magi Agent: {}", delta));
        self.stream_target = Some(self.messages.len() - 1);
        self.streaming = true;
    }

    /// Marks the end of a streamed assistant turn.
    pub fn finalize_stream(&mut self) {
        self.streaming = false;
        self.stream_target = None;
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
/// - `mode_classifier` — consulted by [`resolve_tui_consult_mode`] only when `/consult` has no explicit `--mode` and no `[magi].default_mode` is set (REQ-A07c).
/// - `default_mode` — `[magi].default_mode`, resolved once at startup (REQ-A15).
/// - `untrusted_content` — `[magi].untrusted_content` only; the TUI never exposes this as a command-line flag (REQ-A07d/SC-A07t).
/// - `magi_kind` (REQ-A12c) — the [`ProviderKind`] the trio runs under. Feeds [`tui_consult_success_body`]/[`tui_consult_error_body`] so the explicit `/consult` command gets the SAME keyless-auth guidance `ConsultTool`/headless already have.
/// - `max_query_bytes` (REQ-A11b) — `MagiConfig::effective_max_query_bytes()`, resolved once at startup. Checked by the explicit `/consult` command via `crate::tools::consult::check_query_size` — the SAME cap the tool path and the headless direct path enforce (SC-A11c).
/// - `tool_result_cap` (REQ-A11b) — `MagiConfig::effective_tool_result_cap()`, resolved once at startup. Bounds the explicit `/consult` command's body via [`tui_consult_success_reply`] before it reaches the terminal — which reserves room for the `[DEGRADED: ...]` banner ahead of the report, rather than handing the whole joined string to `crate::tools::consult::truncate_report` the way a prior fix did (see [`tui_consult_success_body`]'s own doc).
pub struct TuiMagiRuntimeConfig {
    /// Consulted only when `/consult` declares no mode and none is
    /// configured (REQ-A07c).
    pub mode_classifier: Arc<dyn ModeClassifier>,
    /// The sink `mode_classifier` writes its two one-time notices to, so
    /// [`run_tui_ext`] can connect it to the response channel once that
    /// channel exists. It MUST be the same instance the classifier was built
    /// with, or its notices keep going to stderr and over the frame — see
    /// [`TuiNoticeSink`].
    pub classifier_notices: Arc<TuiNoticeSink>,
    /// `[magi].default_mode`, resolved once at startup (REQ-A15).
    pub default_mode: Option<Mode>,
    /// `[magi].untrusted_content` (REQ-A07d/SC-A07t).
    pub untrusted_content: bool,
    /// The `ProviderKind` the trio runs under (REQ-A12c).
    pub magi_kind: ProviderKind,
    /// Effective input cap (REQ-A11b), checked before any model call.
    pub max_query_bytes: usize,
    /// Effective output cap (REQ-A11b), applied to the `/consult` reply.
    pub tool_result_cap: usize,
}

/// The MAGI `consult` tool's wiring for the TUI's whole run: whether a live
/// trio is available, what to tell the user when it is not, and how the
/// tool auto-approves — the same three values consulted both at startup and
/// again on every post-`/login` trio rebuild (I-5).
///
/// # Fields
/// - `consult` — the live orchestrator, or `None` if the trio failed to build at startup (REQ-A06).
/// - `consult_unavailable_message` (Task 4.3, REQ-A06/SC-A06b) — the SAME text already pushed to `startup_notices` when `consult` is `None`. Read only when a `/consult` is issued with no trio available, so a later `/consult` echoes the exact reason the startup notice already gave instead of a second, independently-worded message.
/// - `magi_auto_approve` — whether the registered `consult` tool auto-approves an autonomous invocation, mirrored into every rebuilt `ConsultTool` after `/login` (I-5).
/// - `agent_timeout_secs` — `[magi].agent_timeout_secs` as read from config, UNRESOLVED (may be `None`). Fed to [`post_login_agent_timeout_secs`] on every post-`/login` rebuild, which applies the SAME precedence `build_magi_orchestrator` (`main.rs`) already uses at startup. Before this field existed, the rebuild ignored config entirely and hardcoded [`magi_rs::magi::AGENT_TIMEOUT_SECS`] — a configured ceiling silently stopped applying after a `/login` even though it kept being honored everywhere else in the process.
pub struct TuiConsultWiring {
    /// The live orchestrator, or `None` if the trio failed to build.
    pub consult: Option<std::sync::Arc<magi_core::orchestrator::Magi>>,
    /// Echoed verbatim by a `/consult` issued with no trio available.
    pub consult_unavailable_message: Option<String>,
    /// Whether the registered `consult` tool auto-approves.
    pub magi_auto_approve: bool,
    /// `[magi].agent_timeout_secs`, unresolved; see
    /// [`post_login_agent_timeout_secs`].
    pub agent_timeout_secs: Option<u64>,
}

/// Resolves the wall-clock ceiling used to rebuild the MAGI trio's native
/// `ClaudeProvider` after `/login` (I-5) — the SAME precedence
/// `build_magi_orchestrator` (`main.rs`) already uses at startup:
/// `cfg.magi.agent_timeout_secs.unwrap_or(AGENT_TIMEOUT_SECS)`.
///
/// **Fixes M1**: before this fix the rebuild handler hardcoded
/// [`magi_rs::magi::AGENT_TIMEOUT_SECS`] unconditionally, so a configured
/// `[magi].agent_timeout_secs` silently stopped applying after a `/login`
/// even though it kept being honored everywhere else in the process.
///
/// Extracted from the `/login` event handler for the same reason as
/// [`handle_login`]/[`handle_trio_rebuild_failure`] above: the full TUI
/// event loop is intractable to drive in a test, so the ceiling-selection
/// decision is tested here as a plain `fn`.
fn post_login_agent_timeout_secs(configured: Option<u64>) -> u64 {
    configured.unwrap_or(magi_rs::magi::AGENT_TIMEOUT_SECS)
}

/// The [`ProviderKind`] the trio runs under AFTER a successful `/login` rebuild (REQ-A12c).
///
/// **It does not depend on `session_kind`, and that is the whole point.** The `/login` handler
/// rebuilds the trio on a native `ClaudeProvider` unconditionally, whatever `[magi].kind` said
/// at startup — so a session that began on the documented no-config default
/// ([`ProviderKind::Ollama`], keyless) is genuinely Anthropic-credentialed afterwards. The
/// session's `magi_kind` used to keep its startup value forever, and it feeds the keyless-auth
/// guidance in [`tui_consult_success_body`]/[`tui_consult_error_body`]: a later 401 (revoked
/// key, clock skew) was then explained as *"your endpoint is keyless — configure
/// `openai-compat`"*, which sends the user to debug a configuration that is no longer in play.
/// Wrong advice is worse than none.
///
/// `session_kind` is still taken so the signature says what the transformation is — the kind
/// before the rebuild maps to the kind after it — rather than reading as a constant that
/// happens to be called from one place.
///
/// # Parameters
/// * `session_kind` - the kind the trio ran under before this `/login`.
///
/// # Returns
/// Always [`ProviderKind::Anthropic`]: that is what the rebuild actually constructs.
#[must_use]
fn post_login_magi_kind(session_kind: ProviderKind) -> ProviderKind {
    let _ = session_kind;
    ProviderKind::Anthropic
}

/// The [`AgentRunConfig`] for ONE interactive chat turn (REQ-A20/A20b, REQ-A07/A07d).
///
/// `AgentRunConfig::default()` is the interactive baseline and must stay exactly that (it is
/// pinned by the interactive-path regression tests); the operator's `magi.toml` is overlaid on
/// top of it. Before this existed, the chat loop passed `AgentRunConfig::default()` verbatim,
/// so `[magi.complexity]`, `[magi].default_mode` and `[magi].untrusted_content` were inert on
/// the busiest autonomous-consult surface in the product.
///
/// Extracted from the event loop for the same reason as
/// [`handle_login`]/[`post_login_agent_timeout_secs`]: the loop itself is intractable to drive
/// in a test, so the decision it makes is tested here as a plain `fn`.
///
/// # Parameters
/// * `autonomous` - the operator's autonomous-consult configuration, resolved once at startup.
///
/// # Returns
/// Interactive semantics, with the gate thresholds, mode configuration and telemetry sink the
/// operator declared.
#[must_use]
fn tui_agent_run_config(autonomous: &crate::AutonomousRunConfig) -> AgentRunConfig {
    autonomous.apply(AgentRunConfig::default())
}

/// # Parameters (REQ-A07d additions over the pre-MS2 signature)
///
/// - `consult_wiring` — the [`TuiConsultWiring`] bundle: the live trio (if any), its unavailability message, and the tool's auto-approve flag.
/// - `magi_runtime` — the [`TuiMagiRuntimeConfig`] bundle: the mode classifier, `[magi].default_mode`, the `untrusted_content` guard, and the trio's `ProviderKind`. It serves the EXPLICIT `/consult` command only.
/// - `autonomous` — the operator's autonomous-consult configuration (`[magi.complexity]`, `[magi].default_mode`, `[magi].untrusted_content`, the gate telemetry sink). It serves the chat loop's SELF-ROUTED consults, which is the other surface entirely — hence its own parameter rather than more fields on `magi_runtime`.
pub async fn run_tui_ext(
    agent: Agent,
    consult_wiring: TuiConsultWiring,
    secret_store: Option<SharedSecretStore>,
    magi_runtime: TuiMagiRuntimeConfig,
    autonomous: crate::AutonomousRunConfig,
    startup_notices: Vec<Notice>,
) -> anyhow::Result<()> {
    let TuiConsultWiring {
        consult,
        consult_unavailable_message,
        magi_auto_approve,
        agent_timeout_secs: configured_agent_timeout_secs,
    } = consult_wiring;
    let TuiMagiRuntimeConfig {
        mode_classifier,
        classifier_notices,
        default_mode,
        untrusted_content,
        // `mut`: a successful `/login` rebuilds the trio on Anthropic, and the session's kind
        // must follow it — see `post_login_magi_kind`.
        mut magi_kind,
        max_query_bytes,
        tool_result_cap,
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

    // The channels are built BEFORE the terminal is taken over, because the sink
    // has to be attached before then and it needs `response_tx` to attach to.
    let (event_tx, mut event_rx) = mpsc::channel(100);
    let (response_tx, response_rx) = mpsc::channel(100);
    let (approval_tx, approval_rx) = mpsc::channel(100);

    // **Attached BEFORE `EnterAlternateScreen`, and the order is the whole
    // point.** An unattached sink writes to stderr, which is right only while
    // there is no frame to write over. This used to sit AFTER the terminal was
    // taken over, which was harmless for as long as the sink was reached by the
    // mode classifier alone — it cannot fire that early. Since MS2 the logging
    // layer's screen branch delivers into the same sink from `init_logging`, so
    // those lines became a window in which a degradation would have landed on
    // top of the frame: the corruption `PendingNotices` closes from the teardown
    // end, reopened at the setup end. Nothing emits there today, which is
    // exactly what makes it worth fixing now rather than after it does.
    classifier_notices.attach(response_tx.clone());

    // **Announced HERE, and the two neighbours above and below are the reason.**
    // The layer's screen branch delivers a `WARN`/`ERROR` into the sink just
    // attached, so the line lands in the 100-slot channel and `run_app`'s
    // `AgentResponse::Notice` arm puts it in the transcript. Announced any
    // earlier — which is where `run()` used to do it — the sink's `tx` is still
    // `None`, `route` takes its stderr branch, and `EnterAlternateScreen` covers
    // the result about a millisecond later: a mistyped passphrase's "running
    // WITHOUT persistence for this session" would be written, hidden, and only
    // reappear once the user quits, after a whole conversation spent believing
    // the session was being saved. Announced any later, the same stderr branch
    // writes on top of the frame instead (REQ-L39). The window is one statement
    // wide and both walls are guarded.
    //
    // The fallback is the transcript rather than `stderr`, for the OTHER half
    // of the same defect: outside a `.magi/` workspace no layer is installed at
    // all, so the notices take `emit_notices_into`'s no-subscriber branch, and
    // `stderr` there is the primary buffer the alternate screen is about to
    // cover. See `NoticeTranscript`.
    let mut transcript = NoticeTranscript::new(response_tx.clone());
    emit_notices_into(startup_notices, &mut transcript);
    let _ = io::Write::flush(&mut transcript);

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Cancelled right after `run_app` returns, below — races an in-flight OAuth callback
    // wait so quitting mid-login does not force the process to sit out
    // `OAUTH_CALLBACK_TIMEOUT_SECS` (MS2 gate S7 finding; see
    // `await_login_callback_or_quit`).
    let quit_token = CancellationToken::new();

    // The startup notices arrive as a list, but they are no longer PRINTED as one: the
    // announcement above turns each into a `tracing` event and the layer's screen branch
    // decides which of them a human sees (REQ-L19) — which is the whole point of the
    // reclassification, since a list pushed straight into the transcript cannot tell a lost
    // capability from a line counting memories.
    let mut runner_agent = agent;
    runner_agent.set_approval_channel(approval_tx);

    let mut consult_magi_runner = consult;

    // The event loop below is a `'static` spawned task, so it takes ownership of `autonomous`;
    // this clone (three `Copy` fields plus an `Arc`) keeps the telemetry sink reachable here so
    // the session's gate evaluations can be reported after the terminal is restored.
    let gate_telemetry = autonomous.clone();

    // Cloned into the event loop below for `UiEvent::Login` to race against; `quit_token`
    // itself stays here so `run_tui_ext` can cancel it once `run_app` returns.
    let quit_token_for_loop = quit_token.clone();

    // The ephemeral status row, created HERE because both sides need a handle:
    // the event loop below sets it when it starts a long operation, and `App`
    // reads it on every frame. Two handles, one row (see `StatusRow`).
    let status_row = StatusRow::new();
    let status_row_for_loop = status_row.clone();

    // The handle is kept and joined (via `join_event_loop_then_drain`) before the gate
    // telemetry is drained below, instead of being dropped here — a detached spawn would let
    // the drain race the task's tail (its last `UiEvent` or `on_session_close`'s best-effort
    // distillation pass), silently losing whatever it recorded after the drain ran (MS2 gate
    // S7 finding).
    let event_loop = tokio::spawn(async move {
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

                    // Interactive path: the default config (normal cap, repetitive
                    // guard on, no headless observer) with the operator's
                    // autonomous-consult configuration overlaid — see
                    // `tui_agent_run_config`.
                    let result = runner_agent
                        .query_streaming(&text, chunk_tx, tui_agent_run_config(&autonomous))
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
                    // The one long operation this surface starts (R-L15): a
                    // deliberation by three mages, plus the classification call
                    // that may precede it. The text comes from here, the side
                    // that KNOWS what is running, and the guard gives the row
                    // back on every way out of this arm — the early `continue`s
                    // below, an error, a panic inside `analyze` (REQ-L27).
                    let _showing = status_row_for_loop.set(STATUS_CONSULTING_THE_TRIO);
                    // Cap forced /consult input too (the tool path caps in execute; this
                    // direct path bypasses it) — reject before any model call, INCLUDING a
                    // classification call.
                    if let Err(resp) = tui_consult_size_check(&query, max_query_bytes) {
                        let _ = response_tx.send(resp).await;
                        continue;
                    }
                    // REQ-A07d: fails closed if the operator declared
                    // `untrusted_content = true` and neither `--mode` nor
                    // `default_mode` named a lens — before any model call.
                    let (resolution, query) = match resolve_tui_consult_mode(
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
                    let mode = resolution.mode;
                    // REQ-A08: the effective mode AND the level it came from, before the
                    // analysis, so it is shown on the failing runs too.
                    let _ = response_tx
                        .send(AgentResponse::Info(tui_consult_dispatch_notice(
                            &resolution,
                        )))
                        .await;
                    // MAGI FIX: joined spawn (awaited inline → serial, no finalize-order
                    // regression) isolates a panic in magi-core's analyze into a recoverable
                    // JoinError so the runner survives (see plan Task 6 iteration-3).
                    let join = tokio::spawn(async move { magi.analyze(&mode, &query).await }).await;
                    match join {
                        Ok(Ok(report)) => {
                            // REQ-A12c/SC-A12f (fix round 4, finding 2): routed through
                            // the SAME `annotate_report_text` `ConsultTool`/headless use.
                            // REQ-A11b/SC-A11d: bounds the reply the same way the other
                            // two routes bound their `report` field — this is the TUI's
                            // OTHER consult surface (the explicit command, not the
                            // agent-routed tool), and a wall of text is worth capping on
                            // its own merits even though this reply never re-enters
                            // `runner_agent`'s history the way a `ToolResult` would.
                            //
                            // `tui_consult_success_reply` reserves room for the
                            // `[DEGRADED: ...]` banner BEFORE truncating the report —
                            // see its own doc (and `tui_consult_success_body`'s) for the
                            // defect this replaces: a plain `truncate_report` on the
                            // already-joined banner+report string could cut the banner
                            // away on a long degraded report, leaving a thin consensus
                            // rendered as if it were full-strength.
                            let truncated =
                                tui_consult_success_reply(&report, magi_kind, tool_result_cap);
                            // Sanitize the verbatim report (LLM-generated) before rendering —
                            // strips ANSI escapes / control chars, matching the TextDelta path.
                            let body = crate::agent::Agent::sanitize_text(&truncated.text);
                            let _ = response_tx.send(AgentResponse::Text(body)).await;
                        }
                        Ok(Err(e)) => {
                            // REQ-A12c/SC-A12f (fix round 4, finding 2): routed through
                            // the SAME `explain_magi_error` `ConsultTool`/headless use —
                            // see `tui_consult_error_body`'s own doc. `body` is already
                            // redacted (B11) before either use below.
                            //
                            // Loop 1 fix round CE, F24: the pre-existing `eprintln!` here
                            // was dropped rather than routed — it was a PURE duplicate of
                            // `body`, which the very next line already sends through
                            // `AgentResponse::Error`. Nothing was lost by removing it.
                            //
                            // REQ-A11b/SC-A11d (S7 gate re-review finding): capped through
                            // `tui_consult_error_reply` the same way the success arm above
                            // is capped — see that function's own doc for the defect this
                            // closes (an unbounded provider error body bypassing
                            // `tool_result_cap`).
                            let truncated = tui_consult_error_reply(&e, magi_kind, tool_result_cap);
                            let _ = response_tx.send(AgentResponse::Error(truncated.text)).await;
                        }
                        Err(join_err) => {
                            // Loop 1 fix round CE, F24: replaces a pre-existing `eprintln!`
                            // that wrote over the ratatui frame while raw mode is active —
                            // the same class F3 fixed for the classifier's notices. Unlike
                            // the arm above, this one's `eprintln!` carried `JoinError`
                            // detail the generic error text below does not, so it is routed
                            // through the notice channel instead of dropped.
                            let _ = response_tx
                                .send(AgentResponse::Notice(consult_panic_notice(&join_err)))
                                .await;
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

                    // MS2 gate S7 finding: raced against `quit_token_for_loop` instead of
                    // awaited directly, so a user who quits while this is pending does not
                    // strand the process for up to OAUTH_CALLBACK_TIMEOUT_SECS (600s) behind
                    // an already-queued UiEvent::Quit — see `await_login_callback_or_quit`.
                    match await_login_callback_or_quit(
                        oauth.start_callback_server(),
                        &quit_token_for_loop,
                    )
                    .await
                    {
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
                                            // M1 fix: the ceiling comes from
                                            // `configured_agent_timeout_secs`
                                            // (`[magi].agent_timeout_secs`),
                                            // not the hardcoded built-in — see
                                            // `post_login_agent_timeout_secs`.
                                            let agent_timeout_secs = post_login_agent_timeout_secs(
                                                configured_agent_timeout_secs,
                                            );
                                            let client_timeout =
                                                magi_rs::magi::derive_client_timeout(
                                                    agent_timeout_secs,
                                                );
                                            let mut retry =
                                                magi_core::provider::RetryConfig::default();
                                            retry.operation_budget =
                                                magi_rs::magi::derive_operation_budget(
                                                    agent_timeout_secs,
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
                                                    // REQ-A12c: the rebuild is Anthropic, so
                                                    // BOTH the session's own `magi_kind` and
                                                    // the rebuilt tool's must say so — a stale
                                                    // `Ollama` here explains a later 401 as a
                                                    // keyless-endpoint problem that no longer
                                                    // exists.
                                                    magi_kind = post_login_magi_kind(magi_kind);
                                                    runner_agent.register_or_replace_tool(Box::new(
                                                        crate::tools::consult::ConsultTool::new(
                                                            new_magi.clone(),
                                                            magi_auto_approve,
                                                        )
                                                        .with_kind(magi_kind)
                                                        .with_max_query_bytes(max_query_bytes)
                                                        .with_output_cap(tool_result_cap),
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

    let mut app = App::new(event_tx, response_rx, approval_rx);
    app.status_row = status_row;
    let res = run_app(&mut terminal, app).await;

    // `run_app` returning — by ANY exit path — means the user is done with the terminal.
    // Cancel here, before `join_event_loop_then_drain` awaits the event loop below: if
    // `UiEvent::Login` is mid-wait on the OAuth callback, this makes
    // `await_login_callback_or_quit` abandon it immediately instead of the loop sitting
    // out the OAuth timeout behind an already-queued `UiEvent::Quit` (MS2 gate S7 finding).
    quit_token.cancel();

    let _ = disable_raw_mode();
    let _ = execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    );
    let _ = terminal.show_cursor();

    // MS2 gate S7-f finding (Caspar): any classifier notice that had to defer instead of
    // printing (because the channel closed or was full while the alternate screen might still
    // have been up) is safe to print only now, strictly AFTER `LeaveAlternateScreen` above —
    // `flush` is what makes that "strictly after" hold even under concurrent scheduling, not
    // just program order; see `PendingNotices`.
    for msg in classifier_notices.flush() {
        eprintln!("{msg}");
    }

    // SC-A20h: the TUI has no structured run log, and while it holds the alternate screen no
    // line may reach the terminal (that is the whole reason `StreamPiece::Notice` exists). So
    // the session's gate evaluations are reported HERE — after raw mode and the alternate
    // screen are gone and stderr is safe again. Silent when nothing was evaluated.
    //
    // `join_event_loop_then_drain` awaits `event_loop` first (MS2 gate S7 finding): draining
    // straight from `gate_telemetry` here would race the still-running event-loop task.
    report_gate_telemetry(&join_event_loop_then_drain(event_loop, &gate_telemetry).await);

    if let Err(err) = res {
        eprintln!("TUI Error: {:?}", err)
    }
    Ok(())
}

/// Writes the session's gate evaluations to stderr, with the header that states what the sample
/// can be used for (SC-A20h).
///
/// Separate from [`run_tui_ext`] so the "silent when empty" rule is testable without a terminal.
///
/// # Parameters
/// * `lines` - the drained telemetry lines, in the order the evaluations occurred.
fn report_gate_telemetry(lines: &[String]) {
    if lines.is_empty() {
        return;
    }
    eprintln!("{GATE_TELEMETRY_HEADER}");
    for line in lines {
        eprintln!("{line}");
    }
}

/// Header printed above the session's gate telemetry.
///
/// It names the sampling bias on purpose: the lines only cover consults the agent **chose to
/// route**, so they answer "of what the agent wanted to consult, how much did we stop?" and not
/// "how many valuable consults are we missing?" — reading them as the second question is how a
/// threshold gets lowered on evidence that does not exist.
const GATE_TELEMETRY_HEADER: &str =
    "complexity gate — evaluations from this session (agent-routed consults only; \
     says nothing about consults the agent never routed):";

/// Awaits `event_loop` before draining `telemetry` (MS2 gate S7 finding: the loop's
/// `tokio::spawn` handle used to be dropped immediately, so [`run_tui_ext`] could call
/// `drain_telemetry` while the task was still processing its final `UiEvent` or still inside
/// `on_session_close`'s best-effort distillation pass — silently losing any gate evaluation
/// recorded after the drain ran).
///
/// `event_loop` is guaranteed to finish rather than hang: every `run_app` exit path either
/// sends `UiEvent::Quit` before returning (which the loop's `while let Some(event) =
/// event_rx.recv()` breaks on) or drops `app` — and with it `event_tx` — which closes the
/// channel and ends the `recv()` loop the same way. A panic inside the task surfaces as an
/// `Err` from `.await`, swallowed here (best-effort, matching the rest of this shutdown path)
/// rather than propagated.
///
/// **This "guaranteed to finish" was true but misleading before the MS2 gate S7 fix**: it
/// bounded the wait at `OAUTH_CALLBACK_TIMEOUT_SECS` (600s) if `UiEvent::Login` was still
/// awaiting the OAuth callback when `Quit` was queued behind it, which is "not indefinite"
/// but is still up to ten minutes of an unresponsive process after the user already asked to
/// quit. `run_tui_ext` now cancels `quit_token` right after `run_app` returns, before calling
/// this function, so that wait — if in flight — is abandoned promptly instead
/// (`await_login_callback_or_quit`).
///
/// # Parameters
/// * `event_loop` - the join handle of the spawned event-loop task.
/// * `telemetry` - the sink to drain once `event_loop` has completed.
///
/// # Returns
/// Every gate-evaluation line recorded up to and including whatever the event loop's tail
/// (including `on_session_close`) produced.
async fn join_event_loop_then_drain(
    event_loop: tokio::task::JoinHandle<()>,
    telemetry: &crate::AutonomousRunConfig,
) -> Vec<String> {
    let _ = event_loop.await;
    telemetry.drain_telemetry()
}

/// Races an in-flight OAuth callback wait against the user quitting (MS2 gate S7 finding).
///
/// `UiEvent::Login` used to await `oauth.start_callback_server()` directly inside the event
/// loop's sequential dispatch, so a user who quit while the browser round-trip was still
/// pending had the queued `UiEvent::Quit` sit unprocessed behind it — bounded by
/// `OAUTH_CALLBACK_TIMEOUT_SECS` (600s), not indefinite, but ten minutes of an unresponsive
/// process is not an acceptable response to "the user already asked to quit".
///
/// `quit_token` is cancelled by [`run_tui_ext`] as soon as `run_app` returns — every exit
/// path from `run_app` means the user is done with the terminal, regardless of which key
/// triggered it. `CancellationToken::cancelled()` is level-triggered rather than a one-shot
/// notify, so this resolves promptly whether the cancellation happened before this function
/// was even called (e.g. `Login` was still queued behind other events) or arrives mid-wait.
/// `biased` favors the cancellation arm so an already-cancelled token wins immediately rather
/// than racing a `callback` that also happens to be ready.
///
/// Dropping `callback` on the cancellation branch is what actually frees the port: the
/// callback server (`axum::serve` over a `TcpListener`) is a plain awaited future inside
/// `callback`, not a detached task, so dropping it drops the listener and unbinds
/// `REDIRECT_PORT` — no explicit signal needs to reach `oauth.rs` for that to happen.
///
/// # Errors
/// The callback's own error if it resolves first; a cancellation error naming the reason if
/// `quit_token` fires first.
async fn await_login_callback_or_quit<F>(
    callback: F,
    quit_token: &CancellationToken,
) -> anyhow::Result<String>
where
    F: std::future::Future<Output = anyhow::Result<String>>,
{
    tokio::select! {
        biased;
        () = quit_token.cancelled() => Err(anyhow::anyhow!(
            "login cancelled: quit requested before the OAuth callback arrived"
        )),
        res = callback => res,
    }
}

async fn run_app<B: Backend>(terminal: &mut Terminal<B>, mut app: App) -> io::Result<()> {
    loop {
        terminal.draw(|f| ui(f, &mut app))?;

        // The health window expires by TIME, not by event, so something has to
        // wake up and say so: a degradation that stops producing events — which
        // is precisely what a service going down looks like — would otherwise
        // hold its pending transition forever. This loop is that something. It
        // already wakes on the `poll` timeout below even when nobody is typing,
        // which is exactly the condition the window needs, so no timer is added.
        //
        // `health_flush` is NOT called here. It belongs after
        // `LeaveAlternateScreen`, where `bootstrap_headless` already calls it:
        // writing to the terminal while the alternate screen is held is what
        // REQ-L39 forbids.
        if let Some(handle) = magi_rs::logging::installed() {
            handle.health_tick(std::time::Instant::now());
        }

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
                                        // Batched (`insert_str`), not per-character: an
                                        // `insert_char` loop here is O(N^2) on the pasted
                                        // length and freezes the event loop on a large paste
                                        // (MS2 gate S7 seventh-pass finding).
                                        app.insert_str(&text);
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
                                let input = std::mem::take(&mut app.input);
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
                                            app.notice_indices.clear();
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

/// Flattens `messages` into wrap-rendered lines for the Normal-mode conversation pane,
/// alongside a parallel per-line "is this an operational notice" flag — one blank
/// separator line between messages, matching the layout `ui()` used to build inline.
///
/// Notice-ness is looked up from `notice_indices` (the structured record [`App::push_notice`]
/// produces) rather than sniffed from a leading glyph in the rendered text (MAGI S7 finding
/// 2): a model or user message that happens to start with the same glyph must never be
/// mis-styled, and a long notice's wrapped CONTINUATION lines must stay styled too — a
/// first-line-only glyph check could not express that second property either, since
/// [`wrap_message`] only leaves the `⚠` prefix on the first of a notice's wrapped lines.
///
/// # Parameters
/// * `messages` - the conversation history, in display order.
/// * `notice_indices` - indices into `messages` that are operational notices.
/// * `width` - wrap width in terminal columns, forwarded to [`wrap_message`].
///
/// # Returns
/// `(lines, is_notice)` — same length, one entry per rendered line.
fn flatten_history_lines(
    messages: &[String],
    notice_indices: &std::collections::HashSet<usize>,
    width: usize,
) -> (Vec<String>, Vec<bool>) {
    let mut lines = Vec::new();
    let mut is_notice = Vec::new();
    for (i, m) in messages.iter().enumerate() {
        if i > 0 {
            lines.push(String::new());
            is_notice.push(false);
        }
        let notice = notice_indices.contains(&i);
        let wrapped = wrap_message(m, width);
        is_notice.extend(std::iter::repeat_n(notice, wrapped.len()));
        lines.extend(wrapped);
    }
    (lines, is_notice)
}

fn ui(f: &mut Frame, app: &mut App) {
    let area = f.size();
    // Input content width = full width minus the layout margin (1 each side) and
    // the box borders (1 each side). Compute it up front so the input pane height
    // can grow with the wrapped prompt before the layout is split.
    let input_content_w = (area.width as usize).saturating_sub(4);
    let input_rows = input_pane_rows(&app.input, input_content_w, MAX_INPUT_ROWS);
    let input_pane_height = (input_rows as u16).saturating_add(2); // + top/bottom borders

    // The ephemeral status row sits between the transcript and the input, and
    // is `Length(0)` while there is nothing to show (REQ-L26). The layout was
    // already variable-height — `input_pane_height` grows with the wrapped
    // prompt — so a row that appears and disappears introduces no new class of
    // behaviour here. Selection and Visual never show it (REQ-L25): those two
    // modes navigate the transcript, and an operation in flight is not part of
    // it.
    let status_height = if app.mode == AppMode::Normal {
        app.status_row.height()
    } else {
        STATUS_ROW_HEIGHT_COLLAPSED
    };
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints(
            [
                Constraint::Min(1),
                Constraint::Length(status_height),
                Constraint::Length(input_pane_height),
            ]
            .as_ref(),
        )
        .split(area);
    // Named so the rest of this function reads as panes rather than indices —
    // the input pane moved from `chunks[1]` to `chunks[2]` when the row was
    // inserted, and every one of its uses had to move with it.
    let status_area = chunks[1];
    let input_area = chunks[2];

    let inner_width = chunks[0].width.saturating_sub(2) as usize; // subtract left + right borders
    let inner_height = chunks[0].height.saturating_sub(2) as usize; // subtract top + bottom borders

    if app.mode == AppMode::Normal {
        // Flatten every message into one wrapped-line buffer (a blank separator line
        // between messages), then render a scrollable window of it. This gives
        // line-level scrollback — a single message taller than the viewport (e.g. the
        // consult report) is fully reachable via PgUp/PgDn/Home/End.
        let (mut all_lines, mut line_is_notice) =
            flatten_history_lines(&app.messages, &app.notice_indices, inner_width);
        // Compact "thinking…" indicator (mode B): a transient last line while a
        // reasoning model is thinking, with an animated spinner advanced per frame.
        if app.thinking_active {
            if !all_lines.is_empty() {
                all_lines.push(String::new());
                line_is_notice.push(false);
            }
            let indicator_lines = wrap_message(&thinking_indicator(app.spinner_frame), inner_width);
            line_is_notice.extend(std::iter::repeat_n(false, indicator_lines.len()));
            all_lines.extend(indicator_lines);
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
        // Render notice lines with a dimmed yellow style so they are visually distinct
        // from model Content and from system messages — driven by `line_is_notice`
        // (structured, see `flatten_history_lines`), not a glyph sniffed from the text.
        let notice_style = Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::DIM);
        let visible: Vec<Line> = all_lines[range.clone()]
            .iter()
            .zip(&line_is_notice[range])
            .map(|(l, &notice)| {
                if notice {
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

    // Borderless and one line tall: the row is a hint that something is
    // running, not a widget competing with the transcript for attention. The
    // text is handed to ratatui whole and is never byte-indexed (G5) — it is a
    // `&'static str` that may well contain multi-byte characters, and the
    // backend measures it by grapheme.
    //
    // **The mode and the emptiness are decided ONCE, by `status_height` above,
    // and are deliberately not re-tested here.** A second `if` on the same two
    // facts read as a safety net and was dead: `status_area` is zero rows tall
    // in exactly those cases, so the render draws nothing. Two gates on one
    // decision is also where they drift apart — the layout would keep reserving
    // a row that the render had stopped filling.
    if let Some(text) = app.status_row.current() {
        f.render_widget(
            Paragraph::new(text).style(Style::default().add_modifier(Modifier::DIM)),
            status_area,
        );
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
        f.render_widget(Paragraph::new(input_text).block(input_block), input_area);
        if app.mode == AppMode::Normal {
            let col = UnicodeWidthStr::width(&app.input[..app.cursor_position]) as u16;
            f.set_cursor(input_area.x + col + 1, input_area.y + 1);
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
            input_area,
        );
        if app.mode == AppMode::Normal {
            let prefix = wrap_message(&app.input[..app.cursor_position], input_content_w);
            let cur_row_abs = prefix.len().saturating_sub(1);
            let cur_col =
                UnicodeWidthStr::width(prefix.last().map(String::as_str).unwrap_or("")) as u16;
            let cur_row_vis = cur_row_abs.saturating_sub(start) as u16;
            f.set_cursor(input_area.x + cur_col + 1, input_area.y + cur_row_vis + 1);
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

    /// A [`magicore::provider::LlmProvider`] double never actually called in the
    /// `handle_trio_rebuild_failure` tests below — it exists only so `Magi::new` has something
    /// to wrap into an `Arc<Magi>` fixture.
    struct UnusedProvider;

    #[async_trait::async_trait]
    impl magi_core::provider::LlmProvider for UnusedProvider {
        async fn complete(
            &self,
            _system_prompt: &str,
            _user_prompt: &str,
            _config: &magi_core::provider::CompletionConfig,
        ) -> Result<magi_core::provider::Completion, magi_core::error::ProviderError> {
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

    /// M1: a configured `[magi].agent_timeout_secs` must survive a
    /// post-`/login` MAGI trio rebuild instead of silently reverting to the
    /// built-in default — everywhere else in the process (in particular
    /// `build_magi_orchestrator`'s startup build) it keeps being honored.
    ///
    /// `CONFIGURED_CEILING_SECS` is deliberately distinguishable from
    /// `magi_rs::magi::AGENT_TIMEOUT_SECS` (90s): a test that only checked
    /// "some value came back" would still pass if the rebuild kept ignoring
    /// config, so the assertion pins the DIRECTION — the resolved ceiling
    /// must equal the value the operator configured, not the built-in.
    #[test]
    fn post_login_rebuild_ceiling_honors_a_configured_agent_timeout() {
        const CONFIGURED_CEILING_SECS: u64 = 45;
        assert_ne!(
            CONFIGURED_CEILING_SECS,
            magi_rs::magi::AGENT_TIMEOUT_SECS,
            "the fixture must differ from the built-in default, or this test \
             cannot tell a fixed rebuild from a still-broken one"
        );

        assert_eq!(
            post_login_agent_timeout_secs(Some(CONFIGURED_CEILING_SECS)),
            CONFIGURED_CEILING_SECS,
            "a configured [magi].agent_timeout_secs must survive a /login \
             rebuild (M1) instead of silently reverting to the built-in \
             default"
        );
    }

    /// Loop 1, F1 — the chat loop's turn configuration carries the operator's `magi.toml`.
    ///
    /// The event loop used to pass `AgentRunConfig::default()` verbatim, so
    /// `[magi.complexity]`, `[magi].default_mode` and `[magi].untrusted_content` were inert on
    /// the surface that self-routes the most consults. The fixture values are deliberately
    /// distinguishable from the built-ins: a test that only checked "some value came back"
    /// would still pass against the broken version.
    #[test]
    fn the_chat_loop_turn_config_carries_the_operator_configuration() {
        const CONFIGURED_ANALYSIS: usize = 11;
        assert_ne!(
            CONFIGURED_ANALYSIS,
            magi_rs::magi::GATE_ANALYSIS,
            "the fixture must differ from the built-in, or this test cannot tell a wired \
             config from an ignored one"
        );

        let cfg = crate::config::MagiConfig::from_toml_str(
            "[magi]\n\
             default_mode = \"design\"\n\
             untrusted_content = true\n\
             [magi.complexity]\n\
             analysis = 11\n",
        )
        .expect("fixture parses");
        let run = tui_agent_run_config(&crate::AutonomousRunConfig::from_magi_config(&cfg));

        assert_eq!(run.gate_thresholds.analysis, CONFIGURED_ANALYSIS);
        assert_eq!(run.mode_config.default_mode, Some(Mode::Design));
        assert!(run.mode_config.untrusted_content);
    }

    /// Interactive semantics are the BASELINE, not a casualty: everything the operator did not
    /// configure keeps the value `AgentRunConfig::default()` gives it.
    #[test]
    fn the_chat_loop_turn_config_keeps_interactive_defaults() {
        let run = tui_agent_run_config(&crate::AutonomousRunConfig::from_magi_config(
            &crate::config::MagiConfig::default(),
        ));
        let baseline = AgentRunConfig::default();
        assert_eq!(run.max_tool_calls, baseline.max_tool_calls);
        assert_eq!(
            run.disable_repetitive_guard,
            baseline.disable_repetitive_guard
        );
        assert!(run.observer.is_none(), "the TUI has no headless observer");
        assert!(!run.force_consult);
        assert!(run.system.is_none());
    }

    /// SC-A20h's TUI half: the session report is silent when nothing was evaluated, so a user
    /// who never triggered an autonomous consult sees no trailing noise on exit.
    #[test]
    fn the_gate_telemetry_report_is_silent_when_nothing_was_evaluated() {
        report_gate_telemetry(&[]);
        assert!(
            GATE_TELEMETRY_HEADER.contains("agent-routed"),
            "the header must state the sampling bias, or the numbers get read as an answer to \
             a question they cannot answer"
        );
    }

    /// MS2 gate S7 finding: `run_tui_ext` used to drop the event loop's `tokio::spawn`
    /// `JoinHandle` and drain the gate telemetry immediately, racing the still-running task. A
    /// gate evaluation recorded in the task's tail (e.g. after the last `UiEvent` but before
    /// `on_session_close` returns) could be lost to a drain that ran first.
    ///
    /// `join_event_loop_then_drain` closes that window by awaiting the handle before draining.
    /// This double stands in for the event-loop task: it sleeps, THEN records one evaluation —
    /// exactly the "telemetry written by the task's tail" shape the race loses. Under
    /// `start_paused = true` the sleep does not consume real wall-clock time; tokio
    /// auto-advances the paused clock once every other task is parked on it, so the assertion
    /// is deterministic rather than a timing guess.
    #[tokio::test(start_paused = true)]
    async fn join_event_loop_then_drain_waits_for_the_spawned_tasks_tail() {
        let autonomous =
            crate::AutonomousRunConfig::from_magi_config(&crate::config::MagiConfig::default());
        let run_cfg = tui_agent_run_config(&autonomous);
        let telemetry = run_cfg.gate_telemetry;

        let handle = tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            telemetry.on_gate_evaluation(&Mode::Analysis, 999, 1, true);
        });

        let lines = join_event_loop_then_drain(handle, &autonomous).await;

        assert_eq!(
            lines.len(),
            1,
            "the evaluation the spawned task recorded AFTER its sleep must survive the drain"
        );
        assert!(lines[0].contains("chars=999"));
    }

    /// Companion to the test above: absence of `[magi].agent_timeout_secs`
    /// must still fall back to the built-in default, matching
    /// `build_magi_orchestrator`'s own `unwrap_or(AGENT_TIMEOUT_SECS)`
    /// precedence — the fix must not turn `None` into a panic or a zero
    /// ceiling.
    #[test]
    fn post_login_rebuild_ceiling_falls_back_to_the_builtin_when_unconfigured() {
        assert_eq!(
            post_login_agent_timeout_secs(None),
            magi_rs::magi::AGENT_TIMEOUT_SECS
        );
    }

    /// MS2 gate S7 finding: `UiEvent::Login` used to await
    /// `oauth.start_callback_server()` directly inside the event loop's sequential
    /// dispatch. That call is already bounded (RF-4.2, `OAUTH_CALLBACK_TIMEOUT_SECS` =
    /// 600s) but not CANCELLABLE — a user who presses Esc to quit while the OAuth
    /// round-trip is still pending had the queued `UiEvent::Quit` sit unprocessed
    /// behind it, so the process could not exit for up to 10 minutes.
    ///
    /// This exercises `await_login_callback_or_quit` directly against a callback future
    /// that never resolves on its own (`std::future::pending`, standing in for a real
    /// but very slow OAuth wait) and asserts cancellation wins promptly — the whole
    /// point being that the caller never has to wait for the callback future at all.
    /// Wrapped in an outer `tokio::time::timeout` so a regression back to awaiting the
    /// callback inline fails this test instead of hanging the test suite.
    #[tokio::test]
    async fn await_login_callback_or_quit_returns_promptly_when_quit_fires_first() {
        let quit_token = CancellationToken::new();
        let cancel_for_task = quit_token.clone();
        let handle = tokio::spawn(async move {
            await_login_callback_or_quit(std::future::pending(), &cancel_for_task).await
        });
        // Give the spawned task a chance to start polling the pending future before
        // cancelling, so this genuinely exercises "cancel arrives while the callback
        // wait is in flight" rather than "cancel arrives before select! ever runs".
        tokio::task::yield_now().await;
        quit_token.cancel();

        let result = tokio::time::timeout(std::time::Duration::from_secs(5), handle)
            .await
            .expect("cancellation must resolve the select! promptly, not hang")
            .expect("the spawned task must not panic");
        assert!(
            result.is_err(),
            "a cancelled login must surface an error, not a fabricated OAuth code"
        );
    }

    /// The reverse race: when the callback resolves before any cancellation, its
    /// result passes through unchanged — cancellation support must not swallow a
    /// legitimate, fast OAuth completion.
    #[tokio::test]
    async fn await_login_callback_or_quit_returns_the_callback_result_when_it_wins() {
        let quit_token = CancellationToken::new();
        let result = await_login_callback_or_quit(
            std::future::ready(Ok("auth-code-123".to_string())),
            &quit_token,
        )
        .await;
        assert_eq!(result.unwrap(), "auth-code-123");
    }

    /// SC-A06b: the TUI's `/consult`-without-a-trio reply is verbatim `reason`,
    /// wrapped as an error — never a re-worded summary of it.
    #[test]
    fn test_consult_unavailable_response_echoes_the_reason_verbatim() {
        let reason = "MAGI consensus is not available — these seats could not be \
                       built:\n  Melchior: missing the OPENAI_API_KEY credential";
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

    /// SC-A11c: the explicit `/consult` command rejects with the SAME cap the
    /// tool path and the headless direct path enforce — no re-derived copy.
    #[test]
    fn tui_consult_size_check_rejects_over_the_configured_cap() {
        let over = "x".repeat(101);
        let err = tui_consult_size_check(&over, 100).expect_err("101 over a 100-byte cap");
        match err {
            AgentResponse::Error(msg) => {
                assert!(msg.contains("101") && msg.contains("100"), "{msg}");
            }
            other => panic!("expected AgentResponse::Error, got {other:?}"),
        }
    }

    /// Edge case (B13): within the cap, and empty, are the two boundaries.
    #[test]
    fn tui_consult_size_check_accepts_within_the_cap_and_rejects_empty() {
        assert!(tui_consult_size_check("hello", 100).is_ok());
        assert!(
            tui_consult_size_check("   ", 100).is_err(),
            "SC-A11b: an empty/whitespace-only query is rejected too"
        );
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

        // Driven through `tui_consult_success_reply` — the actual function the
        // production call site invokes — under the default output cap, so this is
        // the real production path with truncation simply not engaging (the report
        // is well under `TOOL_RESULT_CAP_BYTES`).
        let body = tui_consult_success_reply(
            &report,
            ProviderKind::Ollama,
            magi_rs::magi::TOOL_RESULT_CAP_BYTES,
        )
        .text;
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

        let body = tui_consult_success_reply(
            &report,
            ProviderKind::OpenAiCompat,
            magi_rs::magi::TOOL_RESULT_CAP_BYTES,
        )
        .text;
        assert!(
            !body.contains("keyless"),
            "openai-compat carries a credential: no hint: {body}"
        );
    }

    /// Builds a real, DEGRADED `MagiReport` (2 of 3 seats succeed, one fails on
    /// an ordinary network error unrelated to auth so no keyless-auth hint noise
    /// enters the assertions below) via `Magi::analyze` — the same construction
    /// [`tui_consult_success_body_carries_the_keyless_hint_when_a_seat_fails_on_auth`]
    /// uses, factored out because the three tests below all need one.
    async fn degraded_report_fixture() -> MagiReport {
        let provider = RoutingMockProvider::new()
            .with_agent_responses(AgentName::Melchior, vec![Ok(agent_json("melchior"))])
            .with_agent_responses(AgentName::Balthasar, vec![Ok(agent_json("balthasar"))])
            .with_agent_responses(
                AgentName::Caspar,
                vec![Err(magi_core::error::ProviderError::external(
                    "unreachable",
                    magi_core::error::ExternalErrorKind::Network,
                ))],
            );
        let magi = magi_core::orchestrator::Magi::new(Arc::new(provider));
        let report = magi
            .analyze(&Mode::Analysis, "should we migrate X to Y?")
            .await
            .expect("2 of 3 succeed ⇒ Ok, degraded");
        assert!(report.degraded, "test setup: this report must be degraded");
        report
    }

    /// Task 6.2 fix (disclosed defect): a DEGRADED report long enough to trigger
    /// real truncation must still carry the `[DEGRADED: ...]` banner in the
    /// final, capped reply produced by [`tui_consult_success_reply`] — the exact
    /// function the production `UiEvent::Consult` success arm calls.
    ///
    /// **RED, before this fix:** the production arm instead called
    /// `tui_consult_success_body` (which JOINED the banner onto the report
    /// BEFORE returning) followed by a bare `crate::tools::consult::
    /// truncate_report` on that joined string. `truncate_report`'s three levels
    /// all cut the kept region starting at the verdict anchor onward — which
    /// sits AFTER a pre-joined banner — so a long enough degraded report
    /// silently lost the ONE piece of text that tells the user how much to
    /// trust everything below it. Confirmed red against that exact two-step
    /// call by temporarily reverting this fix: the assertion below failed with
    /// `truncated.text` containing the truncation mark but NOT "DEGRADED".
    #[tokio::test]
    async fn degraded_banner_survives_truncation_on_the_production_path() {
        let report = degraded_report_fixture().await;

        // Tiny enough to force real truncation of a genuine `MagiReport`, but
        // comfortably above the floor `truncate_report_with_preserved_prefix`
        // needs to guarantee BOTH the banner and some report content survive
        // (75 bytes reserved for the banner + 38 for `mark_overhead()` = 113).
        const CAP: usize = 300;
        let out = tui_consult_success_reply(&report, ProviderKind::Ollama, CAP);

        assert!(
            out.text.len() <= CAP,
            "must not exceed the configured cap: {} > {CAP}: {}",
            out.text.len(),
            out.text
        );
        assert!(
            out.text.contains(magi_rs::magi::TRUNCATION_MARK),
            "test setup: the report must actually have been truncated for this \
             test to exercise anything: {}",
            out.text
        );
        assert!(
            out.text.starts_with(DEGRADED_BANNER),
            "the banner must survive truncation, and lead the reply: {}",
            out.text
        );
    }

    /// Negative direction (B13): a NON-degraded report must never grow a
    /// banner. Without this, an unconditional prepend of `DEGRADED_BANNER`
    /// would pass the positive test above while silently mislabeling every
    /// full-strength consensus as degraded.
    #[tokio::test]
    async fn non_degraded_report_never_grows_a_banner_even_when_truncated() {
        let provider = RoutingMockProvider::new()
            .with_agent_responses(AgentName::Melchior, vec![Ok(agent_json("melchior"))])
            .with_agent_responses(AgentName::Balthasar, vec![Ok(agent_json("balthasar"))])
            .with_agent_responses(AgentName::Caspar, vec![Ok(agent_json("caspar"))]);
        let magi = magi_core::orchestrator::Magi::new(Arc::new(provider));
        let report = magi
            .analyze(&Mode::Analysis, "should we migrate X to Y?")
            .await
            .expect("3 of 3 succeed ⇒ Ok, not degraded");
        assert!(
            !report.degraded,
            "test setup: this report must NOT be degraded"
        );

        // Same tiny cap as the positive test, so truncation fires here too —
        // proving the absence of a banner is not just an artifact of an
        // untruncated reply.
        const CAP: usize = 300;
        let out = tui_consult_success_reply(&report, ProviderKind::Ollama, CAP);

        assert!(
            out.text.len() <= CAP,
            "must not exceed the configured cap: {} > {CAP}",
            out.text.len()
        );
        assert!(
            out.text.contains(magi_rs::magi::TRUNCATION_MARK),
            "test setup: the report must actually have been truncated for this \
             test to exercise anything: {}",
            out.text
        );
        assert!(
            !out.text.contains("DEGRADED"),
            "a full-strength consensus must never render with a DEGRADED \
             banner: {}",
            out.text
        );
    }

    /// Boundary case (B13): a degraded report capped EXACTLY at the smallest
    /// value where `truncate_report_with_preserved_prefix` still guarantees
    /// both the banner and real report content — `DEGRADED_BANNER.len() + 2`
    /// (the `"\n\n"` join) reserved for the banner, plus `mark_overhead()` for
    /// the report's own truncation mark. One byte below this, the report side
    /// has no budget left even for its mark and the whole combined text is
    /// returned untruncated instead (documented on
    /// `truncate_report_with_preserved_prefix`'s own rustdoc) — exercising
    /// exactly at the floor is where an off-by-one in the reserved-budget
    /// arithmetic would show up.
    #[tokio::test]
    async fn degraded_banner_survives_truncation_exactly_at_the_viable_floor() {
        let report = degraded_report_fixture().await;

        let cap = DEGRADED_BANNER.len() + 2 + magi_rs::magi::mark_overhead();
        let out = tui_consult_success_reply(&report, ProviderKind::Ollama, cap);

        assert!(
            out.text.len() <= cap,
            "must not exceed the configured cap: {} > {cap}: {}",
            out.text.len(),
            out.text
        );
        assert!(
            out.text.starts_with(DEGRADED_BANNER),
            "the banner must survive truncation exactly at the viable floor: {}",
            out.text
        );
        assert!(
            out.text.contains(magi_rs::magi::TRUNCATION_MARK),
            "at this floor the report side has room for nothing but its own \
             mark: {}",
            out.text
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

    /// Loop 1, F5 / REQ-A12c: after a `/login` the trio is Anthropic, so the session's kind
    /// must stop claiming the endpoint is keyless.
    ///
    /// The two assertions are the defect from both ends: the transformation itself, and the
    /// user-visible consequence — the same 401 that legitimately earns the keyless hint under
    /// the startup kind must NOT earn it once the trio has been rebuilt on credentials.
    #[tokio::test]
    async fn a_login_rebuild_stops_the_session_claiming_a_keyless_endpoint() {
        assert_eq!(
            post_login_magi_kind(ProviderKind::Ollama),
            ProviderKind::Anthropic,
            "the rebuild constructs a native ClaudeProvider, whatever [magi].kind said"
        );

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

        assert!(
            tui_consult_error_body(&err, ProviderKind::Ollama).contains("keyless"),
            "control: the hint IS correct while the session really is keyless"
        );
        assert!(
            !tui_consult_error_body(&err, post_login_magi_kind(ProviderKind::Ollama))
                .contains("keyless"),
            "after /login the endpoint carries a credential, so pointing the user at a \
             keyless-configuration problem sends them to debug the wrong thing"
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

    // -----------------------------------------------------------------------
    // S7 gate re-review finding (Balthasar): the error arm bypassed
    // `tool_result_cap` entirely — only the success arm was bounded.
    // -----------------------------------------------------------------------

    /// A foreign provider's HTTP error body can be arbitrarily long — built here via the
    /// SAME `map_status_to_error` real dispatch uses to construct `ProviderError::Http`,
    /// whose `body` field carries the response text verbatim. Before this fix,
    /// `UiEvent::Consult`'s error arm sent this straight through with no cap.
    #[test]
    fn tui_consult_error_reply_truncates_a_long_provider_error_body_under_the_cap() {
        let big_body = "x".repeat(5_000);
        let provider_err = ClaudeProvider::map_status_to_error(500, &big_body, vec![], None);
        let err = MagiError::Provider(provider_err);
        let cap = 256;

        let out = tui_consult_error_reply(&err, ProviderKind::OpenAiCompat, cap);

        assert!(
            out.text.len() <= cap,
            "must not exceed the configured cap: {} > {cap}",
            out.text.len()
        );
        assert!(
            out.text.contains(magi_rs::magi::TRUNCATION_MARK),
            "a body this large must show the truncation mark: {}",
            out.text
        );
    }

    /// Negative control: an error well under the cap is returned intact and unmarked — the
    /// cap is a ceiling on the rare long case, not a rewrite of every error reply.
    #[test]
    fn tui_consult_error_reply_leaves_a_short_error_untouched() {
        let err = MagiError::InsufficientAgents {
            succeeded: 0,
            required: 2,
        };
        let cap = 4096;
        let body = tui_consult_error_body(&err, ProviderKind::OpenAiCompat);

        let out = tui_consult_error_reply(&err, ProviderKind::OpenAiCompat, cap);

        assert_eq!(out.text, body, "well under the cap, nothing should change");
        assert!(
            !out.text.contains(magi_rs::magi::TRUNCATION_MARK),
            "short error must not be marked as truncated: {}",
            out.text
        );
    }

    /// Sixth-pass gate finding (S7, Balthasar): the error arm sent `AgentResponse::Error`
    /// straight to the terminal with no [`crate::agent::Agent::sanitize_text`] pass, unlike
    /// the success arm (see the call site in `run_app`, which sanitizes `tui_consult_success_reply`'s
    /// output before rendering it). A `ProviderError::Http` body is text composed by ANOTHER
    /// crate from a response we do not control — exactly the untrusted path `sanitize_text`
    /// exists to cover — so an ANSI escape or control character embedded in an HTTP error page
    /// reached the terminal verbatim.
    #[test]
    fn tui_consult_error_reply_strips_ansi_escapes_and_control_chars() {
        let hostile_body = "\x1B[31mmalicious\x1B[0m content\x07 here";
        let provider_err = ClaudeProvider::map_status_to_error(500, hostile_body, vec![], None);
        let err = MagiError::Provider(provider_err);
        let cap = 4096;

        let out = tui_consult_error_reply(&err, ProviderKind::OpenAiCompat, cap);

        assert!(
            !out.text.contains('\x1B'),
            "ANSI escape must be stripped: {:?}",
            out.text
        );
        assert!(
            !out.text.contains('\x07'),
            "control character must be stripped: {:?}",
            out.text
        );
        assert!(
            out.text.contains("malicious")
                && out.text.contains("content")
                && out.text.contains("here"),
            "readable content must survive sanitization: {}",
            out.text
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

    /// I2: the notice sink must be attached BEFORE the alternate screen is
    /// entered, not after.
    ///
    /// An unattached sink is the branch that writes to stderr, and that is
    /// correct only while there is no frame to write over. The sink went live at
    /// `init_logging` when the screen branch was connected, so every line
    /// between `EnterAlternateScreen` and `attach` was a window in which a
    /// degradation would have landed on top of the ratatui frame — the exact
    /// corruption `PendingNotices` exists to prevent, reintroduced from the
    /// other end. Nothing emitted in that window, which is what makes it worth
    /// a guard: a latent ordering defect is invisible until the day something
    /// does.
    ///
    /// Asserted on byte offsets in the source, like
    /// `the_startup_notices_are_emitted_before_the_terminal_is_taken_over` in
    /// `main.rs`, and for the same reason: the property IS an order, and driving
    /// `run_tui_ext` far enough to observe it needs a real terminal. Both
    /// needles are single-line and the `\r` is stripped, so the guard reads the
    /// same on a CRLF checkout as on an LF one.
    #[test]
    fn test_the_notice_sink_is_attached_before_the_alternate_screen() {
        let source = include_str!("mod.rs").replace('\r', "");
        let start = source
            .find("\npub async fn run_tui_ext(")
            .expect("run_tui_ext must still bring the terminal up");
        let end = source[start..]
            .find("\nfn report_gate_telemetry")
            .map(|offset| start + offset)
            .expect("the item after run_tui_ext must still bound its body");
        let body = &source[start..end];
        let attached = body
            .find("classifier_notices.attach(")
            .expect("the sink must still be attached");
        let alternate = body
            .find("EnterAlternateScreen)")
            .expect("the alternate screen must still be entered");
        assert!(
            attached < alternate,
            "the sink is still unattached when the alternate screen goes up, so a notice \
             arriving in that window writes over the frame"
        );
    }

    /// C1, second half: with NO logging layer installed, a startup `WARN` still
    /// reaches the transcript instead of the buffer about to be swapped out.
    ///
    /// This is the first-run case and the one that matters most: outside a
    /// `.magi/` workspace `init_logging` never runs, so `emit_notices_into`
    /// takes its no-subscriber branch and writes to the fallback. With `stderr`
    /// as that fallback, "no `.magi/` state directory found — run `magi init`"
    /// went to the primary buffer and `EnterAlternateScreen` covered it for the
    /// session — which is what v0.18.0 did NOT do, because it pushed the list
    /// into the channel directly.
    ///
    /// Driven through the real `emit_notices_into`, not by feeding
    /// [`NoticeTranscript`] bytes by hand: the half worth guarding is that the
    /// screen policy still applies on this branch — the `WARN` arrives, the
    /// `INFO` does not — and a hand-fed writer would assert that the writer
    /// splits lines, which nothing was ever going to get wrong.
    ///
    /// **What this does NOT check is that production passes this fallback**,
    /// and mutation proved it: reverting `run_tui_ext` to `stderr` leaves this
    /// green. `test_the_startup_notices_are_emitted_inside_the_attached_window`
    /// is what holds the wiring, by naming the argument and not only the call.
    ///
    /// `LevelFilter::current()` is process-global, and this asserts it is `OFF`
    /// rather than assuming: nextest gives every test its own process, but a
    /// plain `cargo test` does not, and a subscriber installed by a neighbour
    /// would send these notices to the layer and leave the channel empty for a
    /// reason that has nothing to do with the property.
    #[test]
    fn a_startup_warning_reaches_the_transcript_when_no_layer_exists() {
        assert_eq!(
            tracing::level_filters::LevelFilter::current(),
            tracing::level_filters::LevelFilter::OFF,
            "a subscriber is installed in this process, so this cannot observe \
             the no-layer branch; run under `cargo nextest`, not `cargo test`"
        );
        let (tx, mut rx) = mpsc::channel(8);
        let mut transcript = NoticeTranscript::new(tx);
        emit_notices_into(
            vec![
                Notice::warn("no .magi/ state directory found — run `magi init`"),
                Notice::info("memory: 0 active, 0 archived, 0 pending re-embed"),
            ],
            &mut transcript,
        );
        let _ = io::Write::flush(&mut transcript);

        let mut seen = Vec::new();
        while let Ok(response) = rx.try_recv() {
            match response {
                AgentResponse::Notice(text) => seen.push(text),
                other => panic!("the fallback must produce notices, not {other:?}"),
            }
        }
        assert_eq!(
            seen.len(),
            1,
            "the screen policy still decides on this branch: the WARN arrives \
             and the INFO does not; got {seen:?}"
        );
        assert!(
            seen[0].contains("no .magi/ state directory found"),
            "and it must be the warning, verbatim: {seen:?}"
        );
    }

    /// MS2 gate S4, finding 1: nothing reaches the transcript unaudited.
    ///
    /// **Why the transcript and not just any screen.** In Selection mode `y`
    /// copies a displayed message to the SYSTEM CLIPBOARD, so a credential
    /// that lands here is exfiltration rather than an ugly frame. That is what
    /// separates this mouth from the others and why the audit belongs at the
    /// boundary itself.
    ///
    /// **Driven through the writer by hand, deliberately.** The one production
    /// producer — `emit_notices_into` — audits before it writes, so a test
    /// going through it would stay green with this boundary wide open and
    /// would be exactly the guardian-that-cannot-fail this repository keeps
    /// finding. [`NoticeTranscript`] is an `io::Write`: a `&str` written
    /// straight to it is the one shape that escapes the `Audited` type, which
    /// is REQ-L48's whole mechanism, so the type has to redact for itself.
    ///
    /// The canary is **shapeless** — no `sk-` prefix, no URL authority around
    /// it — so only the auditor's exact pass over the registered variants can
    /// catch it. A pattern-only defence leaves this in the clear.
    #[test]
    fn a_registered_secret_never_reaches_the_transcript_in_the_clear() {
        const CANARY: magi_rs::logging::auditor::SecretName =
            magi_rs::logging::auditor::SecretName::new("TRANSCRIPT_CANARY");
        const VALUE: &str = "quaywoodenlanternfeatherbridge";
        assert!(
            magi_rs::logging::process_auditor().register_secret(CANARY, &[VALUE]),
            "the canary must be long enough for the exact pass to carry it"
        );

        let (tx, mut rx) = mpsc::channel(8);
        let mut transcript = NoticeTranscript::new(tx);
        let raw = format!("warning: vault entry BASE_URL_PASSWORD reads {VALUE}\n");
        io::Write::write_all(&mut transcript, raw.as_bytes()).expect("the writer never fails");

        let mut seen = Vec::new();
        while let Ok(response) = rx.try_recv() {
            match response {
                AgentResponse::Notice(text) => seen.push(text),
                other => panic!("the transcript must produce notices, not {other:?}"),
            }
        }
        assert!(
            !seen.is_empty(),
            "the line must still be delivered — redaction is not suppression"
        );
        assert!(
            seen.iter().all(|line| !line.contains(VALUE)),
            "a registered secret reached the clipboard-copyable transcript in \
             the clear: {seen:?}"
        );
        assert!(
            seen.iter().any(|line| line.contains("SECURITY")
                && line.contains(CANARY.as_str())
                && !line.contains(VALUE)),
            "masking and the alarm that says masking happened travel together — \
             both, never one: {seen:?}"
        );
    }

    /// C1: the startup notices are announced INSIDE the attached window — after
    /// `attach`, before `EnterAlternateScreen` — and nowhere else.
    ///
    /// Announcing them earlier (which is what `run()` used to do, one statement
    /// before the handoff) reaches a sink whose `tx` is still `None`, so
    /// [`TuiNoticeSink::route`] takes its stderr branch and writes to the
    /// PRIMARY buffer. `EnterAlternateScreen` swaps that buffer out about a
    /// millisecond later, so a wrong passphrase's "running WITHOUT persistence
    /// for this session" is written, instantly covered, and stays invisible for
    /// the whole conversation — reappearing only after the user quits.
    /// Announcing them LATER, once the frame exists, is the other failure: then
    /// the same stderr branch writes ON TOP of the frame.
    ///
    /// Both bounds therefore have to hold at once, which is why the two
    /// assertions live in one guard: satisfying either alone reopens the other
    /// defect. The companion guard in `main.rs`
    /// (`the_startup_notices_are_handed_to_the_tui_rather_than_announced_early`)
    /// closes the third way out — announcing them in `run()` as well as here.
    ///
    /// Asserted on byte offsets for the same reason as the guard above: the
    /// property IS an order, and driving `run_tui_ext` far enough to observe it
    /// needs a real terminal. Every needle is single-line and the `\r` is
    /// stripped, so this reads the same on a CRLF checkout as on an LF one.
    #[test]
    fn test_the_startup_notices_are_emitted_inside_the_attached_window() {
        let source = include_str!("mod.rs").replace('\r', "");
        let start = source
            .find("\npub async fn run_tui_ext(")
            .expect("run_tui_ext must still bring the terminal up");
        let end = source[start..]
            .find("\nfn report_gate_telemetry")
            .map(|offset| start + offset)
            .expect("the item after run_tui_ext must still bound its body");
        let body = &source[start..end];
        let attached = body
            .find("classifier_notices.attach(")
            .expect("the sink must still be attached");
        // The needle carries the ARGUMENT, not just the call. Mutation is what
        // said so: with only the call named, swapping the fallback back to
        // `stderr` left this green and left
        // `a_startup_warning_reaches_the_transcript_when_no_layer_exists` green
        // too, because that one drives the writer rather than the wiring. The
        // fallback is half the fix; a guard that cannot see which one is
        // passed guards the other half only.
        let emitted = body
            .find("emit_notices_into(startup_notices, &mut transcript);")
            .expect(
                "run_tui_ext must announce the startup notices it was handed, into the \
                 transcript fallback",
            );
        let built = body
            .find("NoticeTranscript::new(response_tx")
            .expect("the fallback must be the session's own response channel");
        let alternate = body
            .find("EnterAlternateScreen)")
            .expect("the alternate screen must still be entered");
        assert!(
            attached < built,
            "the transcript fallback is built before the sink is attached, which is out of \
             order even though nothing observes it today"
        );
        assert!(
            attached < emitted,
            "the startup notices are announced while the sink is still \
             unattached, so every WARN and ERROR goes to the primary screen \
             and is hidden by EnterAlternateScreen for the whole session"
        );
        assert!(
            emitted < alternate,
            "the startup notices are announced after the alternate screen is \
             up, so the no-layer fallback writes over the frame"
        );
    }

    /// R15: the event loop is what expires the health window while the user is
    /// looking at the screen.
    ///
    /// Until this call existed, `headless_runner.rs` held the only `health_tick`
    /// in the tree, so a pending recovery surfaced in a TUI session only AFTER
    /// it ended — which makes SC-L17's `✓ restored` unreachable at the one time
    /// it means anything. No timer is needed: the loop already wakes on its own
    /// `poll` timeout, which is exactly the "runs even when nobody is typing"
    /// the window requires.
    ///
    /// Asserted against the source because the property is *where* the call
    /// sits: a test that called `health_tick` itself would pass with the loop
    /// never calling it, which is the shape of guardian this repository keeps
    /// finding.
    ///
    /// **Bounded by the loop body, not by the function.** Spanning all of
    /// `run_app` would leave the guard green with the call hoisted above `loop`
    /// or dropped below it — a single tick per session instead of one per pass,
    /// which is precisely the failure that leaves the stability window never
    /// expiring. `loop {` opens the body and `if event::poll(` is unambiguously
    /// inside it, so a call outside the loop lands outside that pair either way.
    ///
    /// The `\r` comes out because `include_str!` returns the file's bytes
    /// untouched while rustc normalises CRLF inside a source literal, so a
    /// needle spanning lines would compare two different things on a Windows
    /// checkout.
    #[test]
    fn test_the_event_loop_expires_the_health_window_on_every_pass() {
        let source = include_str!("mod.rs").replace('\r', "");
        let start = source
            .find("\nasync fn run_app<B: Backend>")
            .expect("run_app must still be the event loop");
        let opens = source[start..]
            .find("\n    loop {")
            .map(|offset| start + offset)
            .expect("run_app must still be a loop");
        let polls = source[opens..]
            .find("\n        if event::poll(")
            .map(|offset| opens + offset)
            .expect("the loop must still block on the poll timeout");
        let loop_body = &source[opens..polls];
        assert!(
            loop_body.contains("health_tick("),
            "the health window must be expired on every pass of the ratatui loop, or a \
             recovery is only ever shown after the session it belongs to has ended"
        );
    }

    /// Draws one frame in `mode` with `row` attached, and returns every cell of
    /// the resulting buffer as one string.
    ///
    /// Renders through the REAL `ui()` rather than asking a helper what it
    /// would have done: "does not appear on screen" is a claim about the frame,
    /// and a helper that agrees with itself proves nothing about the layout.
    fn frame_text(mode: AppMode, row: StatusRow) -> String {
        let (event_tx, _events) = mpsc::channel(1);
        let (_responses, response_rx) = mpsc::channel(1);
        let (_approvals, approval_rx) = mpsc::channel(1);
        let mut app = App::new(event_tx, response_rx, approval_rx);
        app.mode = mode;
        app.status_row = row;
        app.messages.push("Magi Agent: hello".to_string());
        let mut terminal =
            Terminal::new(ratatui::backend::TestBackend::new(60, 12)).expect("test terminal");
        terminal.draw(|f| ui(f, &mut app)).expect("draw");
        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect()
    }

    #[test]
    fn test_the_status_row_collapses_to_zero_height_when_idle() {
        // REQ-L26: the row is not a blank line kept in reserve — while there is
        // nothing to say it takes no row at all, which is why the layout can
        // afford to carry it on every frame.
        let row = StatusRow::new();
        assert_eq!(row.height(), 0, "an idle row must occupy no line");
        let observer = row.clone();
        {
            let _showing = row.set(STATUS_CONSULTING_THE_TRIO);
            assert_eq!(
                observer.height(),
                1,
                "a running operation must occupy exactly one line"
            );
        }
        assert_eq!(row.height(), 0, "and the line is given back when it ends");
    }

    #[test]
    fn test_the_status_row_is_cleared_even_when_the_operation_panics() {
        // REQ-L27: success is the easy branch. A panic is the one an explicit
        // clear at each exit point always forgets, and a row left behind by it
        // says an operation is running for the rest of the session.
        let row = StatusRow::new();
        let observer = row.clone();
        let was_shown = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = Arc::clone(&was_shown);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _showing = row.set(STATUS_CONSULTING_THE_TRIO);
            flag.store(observer.height() == 1, std::sync::atomic::Ordering::SeqCst);
            panic!("the operation failed");
        }));
        assert!(
            outcome.is_err(),
            "the panic must reach the caller unchanged"
        );
        assert!(
            was_shown.load(std::sync::atomic::Ordering::SeqCst),
            "the row has to be shown while the operation runs, or clearing it proves nothing"
        );
        assert_eq!(
            observer.height(),
            0,
            "the drop guard must clear the row on a panic too"
        );
    }

    /// I3: two handles onto one row do NOT exclude each other, and this is what
    /// they do instead.
    ///
    /// `set` cannot prevent this — the handles are separate values, so no borrow
    /// rule relates them — which is exactly why the behaviour is pinned here and
    /// spelled out in [`StatusRow`]'s own rustdoc rather than left to whoever
    /// adds the second setter. R-L15 names two candidate operations, so a second
    /// setter is a plausible edit, not a hypothetical one.
    #[test]
    fn test_two_handles_on_one_status_row_do_not_exclude_each_other() {
        /// A second operation's text. A test-local constant rather than a
        /// production one: nothing in the tree starts a second long operation
        /// yet, and a constant added for a test alone is API surface with no
        /// consumer.
        const OTHER_OPERATION: &str = "probing model windows…";
        let first = StatusRow::new();
        let second = first.clone();
        let observer = first.clone();
        let holding_first = first.set(STATUS_CONSULTING_THE_TRIO);
        let holding_second = second.set(OTHER_OPERATION);
        assert_eq!(
            observer.current(),
            Some(OTHER_OPERATION),
            "the later setter wins: one row, one line, and the last writer owns it"
        );
        drop(holding_second);
        assert_eq!(
            observer.height(),
            0,
            "and the first drop collapses the row even though the other operation is \
             still running — the row is shared state, not a stack"
        );
        drop(holding_first);
        assert_eq!(observer.height(), 0);
    }

    #[test]
    fn test_the_status_row_does_not_appear_in_selection_or_visual_mode() {
        // REQ-L25: the row lives outside `App::messages`, and the two modes
        // that navigate the transcript are not modified. The positive half is
        // what keeps this honest — without it the test passes against a row
        // that never renders anywhere.
        let row = StatusRow::new();
        let observer = row.clone();
        let _showing = row.set(STATUS_CONSULTING_THE_TRIO);
        assert!(
            frame_text(AppMode::Normal, observer.clone()).contains(STATUS_CONSULTING_THE_TRIO),
            "Normal mode must show the row while the operation runs"
        );
        assert!(
            !frame_text(AppMode::Selection, observer.clone()).contains(STATUS_CONSULTING_THE_TRIO),
            "Selection mode must be left exactly as it was"
        );
        assert!(
            !frame_text(AppMode::Visual, observer).contains(STATUS_CONSULTING_THE_TRIO),
            "Visual mode must be left exactly as it was"
        );
    }

    /// MS2 gate S7 seventh-pass finding (Caspar): clipboard paste (`Ctrl+V`) used to insert
    /// each pasted character individually via `insert_char`, which calls `String::insert` — an
    /// O(n) tail-shift per character, making an N-character paste O(N^2). A 50 KB code block
    /// (a routine terminal paste) would perform on the order of a billion byte-shifts,
    /// freezing the 50ms-poll event loop for seconds. `insert_str` must place the whole
    /// clipboard payload with one `String::insert_str` call regardless of its length, and must
    /// still honor the char-boundary invariant every other mutator in this file relies on.
    #[tokio::test]
    async fn test_insert_str_places_a_large_multibyte_paste_at_the_cursor_in_one_call() {
        let (event_tx, _) = mpsc::channel(1);
        let (_, response_rx) = mpsc::channel(1);
        let (_, approval_rx) = mpsc::channel(1);
        let mut app = App::new(event_tx, response_rx, approval_rx);

        app.insert_char('á');
        app.insert_char('c');
        // Cursor sits after "ác" (byte 3) — move it back to between the two characters
        // (byte 2), so the pasted payload lands in the middle of existing multi-byte content.
        app.move_cursor_left(false);

        // A large, multi-byte payload — the shape of a real clipboard paste of a code block
        // with non-ASCII content (emoji in a comment, accented identifiers, etc.).
        let pasted: String = "🎉x".repeat(2000);
        app.insert_str(&pasted);

        let expected = format!("á{pasted}c");
        assert_eq!(app.input, expected);
        assert_eq!(app.cursor_position, "á".len() + pasted.len());
    }

    /// Companion to the paste-batching test above: a paste that lands while text is selected
    /// must replace the selection exactly once — [`App::insert_str`] mirrors
    /// [`App::insert_char`]'s `delete_selection` call, not a per-character repeat of it.
    #[tokio::test]
    async fn test_insert_str_replaces_an_active_selection() {
        let (event_tx, _) = mpsc::channel(1);
        let (_, response_rx) = mpsc::channel(1);
        let (_, approval_rx) = mpsc::channel(1);
        let mut app = App::new(event_tx, response_rx, approval_rx);

        app.insert_char('a');
        app.insert_char('b');
        app.insert_char('c');
        app.selection_start = Some(0); // selects the whole "abc", cursor already at 3

        app.insert_str("xyz");
        assert_eq!(app.input, "xyz");
        assert_eq!(app.cursor_position, 3);
        assert!(app.selection_start.is_none());
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

    /// MAGI S7 fix round, finding 2: a message that legitimately STARTS WITH the notice
    /// glyph but was never pushed via `push_notice` must not be styled as a notice — the
    /// old `l.starts_with('⚠')` check could not tell the two apart because it never looked
    /// at how the message was produced, only at its rendered text.
    #[test]
    fn test_flatten_history_lines_does_not_style_a_message_that_merely_starts_with_the_glyph() {
        let messages = vec!["⚠ this is model output, not a real notice".to_string()];
        let notice_indices = std::collections::HashSet::new(); // never pushed via push_notice
        let (lines, is_notice) = flatten_history_lines(&messages, &notice_indices, 80);
        assert_eq!(
            lines,
            vec!["⚠ this is model output, not a real notice".to_string()]
        );
        assert_eq!(
            is_notice,
            vec![false],
            "notice styling must come from notice_indices, not the leading glyph"
        );
    }

    /// The counterpart: a genuine notice's WRAPPED CONTINUATION lines must stay styled
    /// too, not just the first wrapped line (a first-line-only glyph check could not
    /// express this, since `wrap_message` leaves the `⚠` prefix only on the first line).
    #[test]
    fn test_flatten_history_lines_styles_every_wrapped_line_of_a_long_notice() {
        let long_notice = format!("⚠ {}", "x".repeat(50));
        let messages = vec![long_notice];
        let mut notice_indices = std::collections::HashSet::new();
        notice_indices.insert(0usize);

        let (lines, is_notice) = flatten_history_lines(&messages, &notice_indices, 10);
        assert!(
            lines.len() > 1,
            "the notice must actually wrap to multiple lines for this test to be meaningful: {lines:?}"
        );
        assert!(
            is_notice.iter().all(|&n| n),
            "every wrapped line of a notice message must be styled, including continuation \
             lines: {is_notice:?}"
        );
    }

    /// A real user/assistant message (no notice) yields an `is_notice` vector of the same
    /// length as its wrapped lines, all `false` — and the one-blank-line-per-message
    /// separator is itself never styled as a notice.
    #[test]
    fn test_flatten_history_lines_separator_between_messages_is_never_a_notice() {
        let messages = vec!["first".to_string(), "second".to_string()];
        let mut notice_indices = std::collections::HashSet::new();
        notice_indices.insert(1usize); // "second" IS a notice

        let (lines, is_notice) = flatten_history_lines(&messages, &notice_indices, 80);
        assert_eq!(
            lines,
            vec!["first".to_string(), String::new(), "second".to_string()]
        );
        assert_eq!(is_notice, vec![false, false, true]);
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
        let cmd = super::parse_tui_consult("/consult --mode design this or that?").unwrap();
        assert_eq!(cmd.mode, Some(Mode::Design));
        assert_eq!(cmd.query, "this or that?");
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
            "the TUI does not expose the flag: there is a human there who chose the content"
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

    /// A [`ModeClassifier`] double that PANICS if invoked — the strongest way to test "zero
    /// classification calls": a `CountingClassifier` at 0 could hide a bug where it is called
    /// but the result is discarded; this one leaves no such escape.
    struct NeverClassifier;

    #[async_trait::async_trait]
    impl ModeClassifier for NeverClassifier {
        async fn classify(&self, _content: &str) -> Option<Mode> {
            panic!("classifier must not be called when a mode is already declared");
        }
    }

    /// A double that always returns a fixed label, for the inference path.
    struct FixedClassifier(Mode);

    #[async_trait::async_trait]
    impl ModeClassifier for FixedClassifier {
        async fn classify(&self, _content: &str) -> Option<Mode> {
            Some(self.0)
        }
    }

    /// SC-A07b (TUI half) — Fix round 1. Before this fix, the `UiEvent::Consult` handler ran
    /// `Mode::Analysis` unconditionally and the loop input used `parse_consult_command`, which
    /// did not understand `--mode` at all: `/consult --mode design this or that?` left the text
    /// `--mode design` INSIDE the question and still analyzed in `Analysis`. This test pins the
    /// two halves of the fix: the resolved mode is `Design`, and the query that would reach
    /// `Magi::analyze` no longer contains the flag.
    #[tokio::test]
    async fn an_explicit_mode_reaches_analyze_as_declared_and_strips_the_flag_from_the_query() {
        let cmd = super::parse_tui_consult("/consult --mode design this or that?").unwrap();
        let (res, query) = super::resolve_tui_consult_mode(cmd, None, false, &NeverClassifier)
            .await
            .unwrap();
        assert_eq!(
            res.mode,
            Mode::Design,
            "the explicit --mode must win, never default to Analysis"
        );
        assert_eq!(query, "this or that?");
        assert!(
            !query.contains("--mode"),
            "the flag must not survive in the text that reaches analyze: {query:?}"
        );
    }

    /// SC-A07k (TUI half): `[magi].default_mode` beats classification — without `--mode` in the
    /// command, if the operator declared a default, THAT one is used and the classifier is
    /// never invoked.
    #[tokio::test]
    async fn configured_default_mode_wins_without_classifying() {
        let cmd = super::parse_tui_consult("/consult this or that?").unwrap();
        let (res, query) =
            super::resolve_tui_consult_mode(cmd, Some(Mode::CodeReview), false, &NeverClassifier)
                .await
                .unwrap();
        assert_eq!(res.mode, Mode::CodeReview);
        assert_eq!(
            res.source,
            magi_rs::magi::mode::ModeSource::Configured,
            "REQ-A08: the level must survive the trip, or a reader cannot tell a configured              lens from an inferred one"
        );
        assert_eq!(query, "this or that?");
    }

    /// Without `--mode` and without `default_mode`, the classifier IS consulted and its answer
    /// is used — the inference path remains alive on this surface.
    #[tokio::test]
    async fn without_mode_or_config_the_classifier_is_consulted() {
        let cmd = super::parse_tui_consult("/consult this or that?").unwrap();
        let (res, _query) =
            super::resolve_tui_consult_mode(cmd, None, false, &FixedClassifier(Mode::Design))
                .await
                .unwrap();
        assert_eq!(res.mode, Mode::Design);
        assert_eq!(res.source, magi_rs::magi::mode::ModeSource::Inferred);
    }

    /// Loop 1, F4 / REQ-A08: the `/consult` reply states the effective mode AND the level it
    /// came from.
    ///
    /// Without the level, three materially different situations render identically to the
    /// user: the operator configured `code-review`, the classifier inferred it, or a
    /// classification timed out and everything fell to `Analysis`. With inference active the
    /// same prompt can resolve differently across two runs, so this is what makes "did the lens
    /// I asked for actually run?" answerable at all.
    #[test]
    fn the_consult_dispatch_notice_names_the_mode_and_its_source() {
        for (source, needle) in [
            (magi_rs::magi::mode::ModeSource::Explicit, "Explicit"),
            (magi_rs::magi::mode::ModeSource::Configured, "Configured"),
            (magi_rs::magi::mode::ModeSource::Inferred, "Inferred"),
            (magi_rs::magi::mode::ModeSource::Default, "Default"),
        ] {
            let notice = super::tui_consult_dispatch_notice(&ModeResolution {
                mode: Mode::CodeReview,
                source,
                classification_attempted: false,
            });
            assert!(
                notice.contains("code-review"),
                "the effective mode must be named: {notice}"
            );
            assert!(
                notice.contains(needle),
                "the level must be named, or a configured lens is indistinguishable from an \
                 inferred one: {notice}"
            );
        }
    }

    /// The notice keeps saying what it always said — that three model calls are about to run —
    /// so adding the audit information does not cost the user the cost warning.
    #[test]
    fn the_consult_dispatch_notice_still_announces_the_three_calls() {
        let notice = super::tui_consult_dispatch_notice(&ModeResolution {
            mode: Mode::Analysis,
            source: magi_rs::magi::mode::ModeSource::Default,
            classification_attempted: true,
        });
        assert!(
            notice.contains('3'),
            "the cost heads-up must survive: {notice}"
        );
    }

    /// SC-A07r (TUI half): the operator declared `untrusted_content = true` in their
    /// `magi.toml` and neither `--mode` nor `default_mode` names a lens — fails closed, without
    /// classifying.
    #[tokio::test]
    async fn operator_declared_untrusted_content_fails_closed_without_a_mode() {
        let cmd = super::parse_tui_consult("/consult this or that?").unwrap();
        let err = super::resolve_tui_consult_mode(cmd, None, true, &NeverClassifier)
            .await
            .expect_err("with no mode declared, the operator's flag must fail closed");
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

    /// MS2 gate S7-f finding (Balthasar): `push_notice` — and any other push — can
    /// legitimately land while `streaming` is still `true` (an operational notice does not
    /// end the turn — see the `AgentResponse::Notice` arm in `run_app`). Before `stream_target`
    /// existed, `append_stream_delta` blindly wrote to `messages.last_mut()`, so a delta
    /// arriving after the notice appended onto the NOTICE's text instead of the reply's,
    /// corrupting both the notice and the in-progress answer.
    #[test]
    fn a_notice_arriving_mid_stream_does_not_corrupt_the_streamed_reply() {
        let (event_tx, _) = mpsc::channel(1);
        let (_, response_rx) = mpsc::channel(1);
        let (_, approval_rx) = mpsc::channel(1);
        let mut app = App::new(event_tx, response_rx, approval_rx);

        app.append_stream_delta("Hello".to_string());
        app.push_notice("memory: context assembly failed".to_string());
        app.append_stream_delta(", world".to_string());

        assert_eq!(
            app.messages.len(),
            2,
            "the notice must stay its own entry, not get merged into the reply: {:?}",
            app.messages
        );
        assert_eq!(
            app.messages[0], "Magi Agent: Hello, world",
            "the second delta must land on the streamed reply, not the notice that \
             happened to be last: {:?}",
            app.messages
        );
        assert!(
            app.messages[1].starts_with("⚠ ") && app.messages[1].contains("context assembly"),
            "the notice text must be untouched by the delta that arrived after it: {:?}",
            app.messages
        );
    }

    /// Companion: `finalize_stream` must clear `stream_target`, or a delta belonging to the
    /// NEXT turn would silently append onto the previous turn's now-finalized message instead
    /// of starting a fresh one.
    #[test]
    fn finalize_stream_clears_the_target_so_the_next_turn_starts_a_new_message() {
        let (event_tx, _) = mpsc::channel(1);
        let (_, response_rx) = mpsc::channel(1);
        let (_, approval_rx) = mpsc::channel(1);
        let mut app = App::new(event_tx, response_rx, approval_rx);

        app.append_stream_delta("first turn".to_string());
        app.finalize_stream();
        app.append_stream_delta("second turn".to_string());

        assert_eq!(
            app.messages,
            vec![
                "Magi Agent: first turn".to_string(),
                "Magi Agent: second turn".to_string(),
            ],
            "a new turn must start a new message, not append onto the finalized one: {:?}",
            app.messages
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

    /// Loop 1 fix round CE, F24: the JoinError's own detail reaches the notice text, driven
    /// through a REAL panicked `tokio::spawn`, not a hand-rolled stand-in — a genuine
    /// `JoinError` cannot be constructed any other way (it has no public constructor), and a
    /// mock would not prove the real `Display` output is what ends up in the notice.
    #[tokio::test]
    async fn consult_panic_notice_carries_the_real_join_errors_detail() {
        let join_err = tokio::spawn(async {
            panic!("synthetic panic for F24 verification");
        })
        .await
        .expect_err("the spawned task must have panicked");

        let notice = consult_panic_notice(&join_err);

        assert!(
            notice.starts_with("[consult] analyze panicked: "),
            "{notice}"
        );
        assert!(
            notice.contains(&join_err.to_string()),
            "the JoinError's own Display must reach the notice verbatim: {notice}"
        );
    }

    /// S7 gate re-review finding (Caspar): the panic arm sent `consult_panic_notice`'s
    /// output straight to `AgentResponse::Notice` with no [`crate::agent::Agent::sanitize_text`]
    /// pass, unlike its two siblings — the success arm sanitizes at its call site (see `run_app`)
    /// and the error arm's `tui_consult_error_reply` sanitizes internally. Driven through a REAL
    /// panicked `tokio::spawn`, same reasoning as the test above: a genuine `JoinError` has no
    /// public constructor, so a hand-rolled stand-in would not prove what the real `Display`
    /// output looks like.
    ///
    /// **Note on what this proves.** A real `JoinError`'s `Display` already Debug-escapes a
    /// raw ESC/control byte embedded in a `String`/`&str` panic payload (`{:?}` renders it as
    /// literal `\u{1b}` text, not the raw byte) — verified directly against tokio 1.53.1, the
    /// version this crate pins. So these three assertions hold even without
    /// `consult_panic_notice`'s `sanitize_text` pass; this test does not reproduce a live
    /// injection through the JoinError path. It still earns its place: it locks in the
    /// end-to-end guarantee explicitly (this crate's own `sanitize_text`, not an incidental
    /// property of a dependency it does not control) and keeps this arm's coverage on par with
    /// its two siblings, one of which (`tui_consult_error_reply_strips_ansi_escapes_and_control_chars`)
    /// covers a source that CAN carry raw control bytes (a foreign HTTP error body).
    #[tokio::test]
    async fn consult_panic_notice_strips_ansi_escapes_and_control_chars() {
        let join_err = tokio::spawn(async {
            panic!("malicious\x1B[31m escape\x07 payload");
        })
        .await
        .expect_err("the spawned task must have panicked");

        let notice = consult_panic_notice(&join_err);

        assert!(
            !notice.contains('\x1B'),
            "ANSI escape must be stripped: {notice:?}"
        );
        assert!(
            !notice.contains('\x07'),
            "control character must be stripped: {notice:?}"
        );
        assert!(
            notice.contains("malicious") && notice.contains("escape") && notice.contains("payload"),
            "readable content must survive sanitization: {notice}"
        );
    }
}
