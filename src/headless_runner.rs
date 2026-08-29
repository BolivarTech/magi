// Author: Julian Bolivar
// Version: 0.17.0
// Date: 2026-08-27

//! Headless runner: wires the existing [`Agent`] tool loop to the non-interactive
//! path with per-tier auto-approval and a deterministic `stop_reason`
//! (REQ-H02, REQ-H09, REQ-H23/H23b).
//!
//! This module lives in the **binary** crate (not `src/headless/`, the library)
//! because it needs `crate::agent::Agent` and `crate::tools`, which the pure
//! `headless` library modules cannot reach. It consumes the library's resolved
//! parameters ([`Resolved`]), tier policy ([`Policy`]) and output contract types
//! ([`RunOutcome`]) through the `magi_rs::headless` public API.
//!
//! # How the Agent is driven
//! [`run_query`] builds an [`AgentRunConfig`] whose [`RunObserver`] is authoritative
//! for **every** tool call (all tiers). This is the only mechanism that can gate a
//! tool which opts out of interactive approval (`project_knowledge`, an
//! auto-approve `consult`): the interactive `approval_tx` gate never sees those,
//! so wiring `approval_tx` alone could not express a tier (REQ-H06/H07/H09). The
//! observer additionally captures the per-call outcome (with wall-clock timing)
//! and the final-turn text-block count that the [`StreamPiece`] stream does not
//! carry — the data needed to build a faithful, auditable [`RunOutcome`].
//!
//! The `magi query` / `magi consult` subcommand dispatch in `main.rs` (MS2 T7)
//! is the production caller of [`run_query`] / [`run_consult`] /
//! [`resolve_tier_timeout_default`]; every item is reachable in the non-test build.

use std::collections::HashMap;
use std::future;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use magi_core::orchestrator::Magi;
use magi_core::schema::Mode;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use magi_rs::headless::output::sanitize_error_message;
use magi_rs::headless::policy::{Policy, Tier};
use magi_rs::headless::resolution::Resolved;
use magi_rs::headless::types::{
    AppliedCaps, ErrorKind, ErrorPayload, RunOutcome, StopReason, Timings, ToolCallRecord,
    TranscriptEntry, Usage,
};
use magi_rs::magi::kind::ProviderKind;
use magi_rs::magi::mode::{resolve_mode_guarded, ModeClassifier, ModeError, ModeSources};
use magi_rs::magi::{BudgetTelemetry, TimeoutDecision};
use magi_rs::redact::redact_foreign_text;

use crate::agent::messages::{Content, Message, Role};
use crate::agent::mode_classifier::NoticeSink;
use crate::agent::{Agent, AgentRunConfig, RunObserver, StreamPiece, MAX_TOOL_CALLS_ERROR};
use crate::config::MagiConfig;
use crate::task::AbortOnDrop;
use crate::tools::consult::{
    annotate_report_text, check_query_size, explain_magi_error, report_to_consult_json,
    truncate_report, RunContext, StructuredVerdicts,
};

/// Dedup key for [`NoticeSink::once`] — the SC-A04d warning, distinct from the
/// mode-classifier's `classify.cost`/`classify.timeout` keys (`agent::mode_
/// classifier`) even though production shares one [`NoticeSink`] instance across
/// both: dedup is per-key, so sharing the sink cannot suppress one notice because
/// the other already fired.
const NOTICE_TIMEOUT_BELOW_FORMULA: &str = "timeout.below_formula";
/// Cap on a tool input written to the log.
///
/// The input is REDACTED first and capped after (REQ-H15c): capping first can
/// split a secret across the limit and ship the surviving prefix in the clear.
const TOOL_INPUT_LOG_CAP: usize = 2048;

/// Target recorded on the notices this module emits.
const NOTICE_TARGET: &str = "magi_rs::headless_runner";
/// Notices do not travel the appender's channel, so they reserve nothing.
const NO_RESERVATION: usize = 0;

/// Run-log line for the same condition on the `magi query` route (SC-A04d).
///
/// The exact seconds live in the stderr notice and in `applied_caps`; the log line only has to
/// say that the deadline is structurally too short, so a later `error.kind = timeout` can be
/// attributed to the configuration instead of to the model.
const TIMEOUT_BELOW_FORMULA_LOG: &str =
    "the run deadline is below the minimum a consult with its schema retry needs \
     (REQ-A04); a forced or proactive consult may not complete";

/// What every operator-facing notice is prefixed with on stderr.
///
/// stderr and not stdout: stdout carries the run's output, which a consumer
/// parses as JSON, and a notice written there would corrupt the very contract
/// S1 exists to protect. There is no alternate screen to overwrite here, which
/// is the reason the TUI needs a channel and this does not.
const NOTICE_PREFIX: &str = "notice: ";

/// Renders one agent notice for stderr, with any authority in it redacted.
///
/// The text is FOREIGN: it carries whatever the failing subsystem said, and
/// `summarize_assembly_error` truncates a raw error that can embed the
/// endpoint it was talking to, credentials and all. Every other foreign string
/// this binary prints goes through the same helper, and a notice is not an
/// exception just because it is short.
///
/// # Parameters
/// - `text` — the notice as the agent emitted it.
///
/// # Returns
/// The line to print, prefixed and redacted.
fn notice_line(text: &str) -> String {
    format!("{NOTICE_PREFIX}{}", redact_foreign_text(text).as_str())
}

/// Buffered capacity of the internal chunk channel; mirrors the interactive TUI
/// bridge so backpressure behaves identically. The channel is drained
/// concurrently, so this is a small smoothing buffer, not a bound on output.
const CHUNK_CHANNEL_CAPACITY: usize = 100;

/// Upper bound on how long [`run_query`] waits for the ttfb-measuring drain task after the
/// run itself has already concluded (by completion, timeout, or error) — MAGI S6 gate finding
/// 6.
///
/// In the well-behaved case this is a formality: `chunk_tx` is a parameter owned by
/// `Agent::query_streaming`'s future, so it is dropped — and `chunk_rx.recv()` returns `None`,
/// closing the drain loop — essentially synchronously with that future's own completion or
/// drop. This bound exists so `run_query`'s own wall-clock guarantee (REQ-H36: "the bound is
/// not a lie") does not silently depend on that invariant holding forever in a file outside
/// this module's edit surface (`src/agent/mod.rs`): if a future change there ever leaks a
/// `chunk_tx` clone into a longer-lived task, this run still returns instead of hanging past
/// its own deadline — it degrades to `ttfb_ms: None` (never measured) rather than blocking.
/// 500ms is generous relative to the sub-millisecond cost of the well-behaved case.
const DRAIN_GRACE: Duration = Duration::from_millis(500);

/// Normalized transcript role for a real user turn (REQ-H14).
const ROLE_USER: &str = "user";

/// Normalized transcript role for an assistant turn (REQ-H14).
const ROLE_ASSISTANT: &str = "assistant";

/// `error.message` when a run is aborted by the wall-clock `--timeout` (REQ-H36).
const TIMEOUT_MESSAGE: &str = "run exceeded the --timeout wall-clock limit";

/// Registered name of the multi-perspective MAGI `consult` tool (REQ-H21/H22).
/// Single source of truth for the name matched when capturing the consult
/// result out of a finished run's tool-call records ([`extract_consult_value`]).
/// The FORCING itself now happens in-loop
/// (`Agent::run_tool_loop`/`AgentRunConfig::force_consult`, REQ-H22) — this
/// module only reads the result back out.
const CONSULT_TOOL: &str = "consult";

/// `error.message` for a direct `magi consult` prompt that is empty or exceeds
/// the effective input cap (`MagiConfig::effective_max_query_bytes`, REQ-A11b;
/// REQ-H33: reject, never truncate).
const CONSULT_INPUT_INVALID_MESSAGE: &str = "consult prompt is empty or exceeds the maximum length";

/// Typed failure of a direct `magi consult` run, mapped to an [`ErrorKind`] by
/// the caller so an over-cap prompt becomes `input_invalid` (exit 2, REQ-H33)
/// and a wall-clock abort becomes `timeout` (exit 1, REQ-H36).
enum ConsultRunError {
    /// The prompt was empty or exceeded the effective input cap (→ `input_invalid`).
    InputInvalid,
    /// `untrusted_content` was active and no mode was declared by any surface
    /// (→ `input_invalid`, REQ-A07d/REQ-A07r): the run fails closed instead of
    /// classifying the hostile content.
    UntrustedContentRequiresMode(ModeError),
    /// The consult was cancelled by the `--timeout` deadline (→ `timeout`).
    Timeout,
    /// The MAGI orchestrator failed or panicked; the message is sanitized before
    /// use (→ `runtime`).
    Runtime(String),
}

/// Resolves the effective mode for a **direct** `magi consult` (REQ-A07c): a
/// declared `--mode`/envelope value wins outright, at zero classification cost
/// (SC-A07g); its absence costs exactly one classification call (SC-A07f),
/// which itself fails open to [`Mode::Analysis`] on any error, timeout, or
/// unrecognized reply ([`ModeClassifier::classify`] documents this — never a
/// hang, never a propagated error).
///
/// This is a narrower resolution than the full five-level gate
/// ([`resolve_mode_guarded`]): it only expresses "a human declared it" vs.
/// "classify it", with no `Configured`/`AgentChosen` level and no
/// `untrusted_content` guard. [`analyze_direct`] no longer calls this directly
/// — it goes through `resolve_mode_guarded` so the direct consult path also
/// honors `[magi].default_mode` and `untrusted_content` (REQ-A07d/A15) — but
/// this narrower helper stays `pub(crate)` because `main.rs`'s CLI-level test
/// coverage (`run_consult_cli`, SC-A07f/g) still calls it directly, to observe
/// a classification-call count without standing up a full `Arc<Magi>`
/// orchestrator.
///
/// `#[cfg(test)]`: with `analyze_direct` now going through
/// `resolve_mode_guarded` directly, production has no remaining caller — only
/// the `run_consult_cli` test does. Gating it keeps that narrower assertion
/// alive without leaving a `pub(crate)` item with zero non-test callers (which
/// `clippy -D warnings` would flag as dead code in a release build).
#[cfg(test)]
pub(crate) async fn resolve_direct_mode(
    explicit: Option<Mode>,
    classifier: &dyn ModeClassifier,
    content: &str,
) -> Mode {
    match explicit {
        Some(mode) => mode,
        None => classifier.classify(content).await.unwrap_or(Mode::Analysis),
    }
}

/// Session-scoped MAGI parameters that travel together, unchanged, through
/// every direct-consult call site (REQ-A07c/REQ-A12c): how to pick the mode
/// when none is declared for this specific call, and how to explain the
/// trio's provider kind once a result (or failure) comes back.
///
/// Bundled because [`analyze_direct`] and [`run_consult`] both thread these
/// values straight from the caller's already-resolved configuration to
/// several different call sites: `classifier`/`configured_mode`/
/// `untrusted_content` feed [`resolve_mode_guarded`] verbatim, `kind` feeds
/// [`report_to_consult_json`]/[`explain_magi_error`], and `magi_config`/
/// `timeout_decision` feed [`RunContext::build`] (fix round 1, Finding 1) on
/// the way back out. None is call-specific the way `explicit_mode` is (that
/// one stays its own parameter) — grouping only the values that are constant
/// for the lifetime of a run removes the parameter-count pressure both
/// functions used to carry (B3): this is what lets `RunContext::build`'s two
/// new consumers become a **field** each, not an 11th/12th parameter.
///
/// # Fields
/// - `kind` — the [`ProviderKind`] the trio runs under; feeds [`report_to_consult_json`]/[`explain_magi_error`] for provider-specific guidance (REQ-A12c).
/// - `classifier` — consulted **only** when neither a call's `explicit_mode` nor `configured_mode` is set, via [`resolve_mode_guarded`] (REQ-A07c).
/// - `configured_mode` — `[magi].default_mode`, if declared; wins over the classification level, below a call's `explicit_mode` (REQ-A15).
/// - `untrusted_content` — REQ-A07d's guard: with this `true` and no mode declared by any other level, the run fails closed ([`ConsultRunError::UntrustedContentRequiresMode`]) instead of classifying hostile content.
/// - `magi_config` — feeds `RunContext::build`'s `cfg.magi_endpoint_diverges()` (REQ-A11d). A reference, not a clone: the caller's `MagiConfig` already outlives the awaited call.
/// - `timeout_decision` — the REAL [`TimeoutDecision`] (SC-A04d), computed by the caller via `magi_rs::magi::resolve_run_timeout` for its `below_formula` flag — **not recomputed here**. Its `effective_secs` is deliberately NOT used to override the enforced `timeout` parameter this call already takes: this struct exists to make the JSON's telemetry honest, not to change which timeout is actually enforced (a larger, separate, pre-existing gap — see this fix round's report).
/// - `notice_sink` — where [`analyze_direct`] emits `timeout_decision.warning` (fix round 2, SC-A04d's other half: the JSON flag alone isn't the whole requirement — a human running the command by hand needs the same fact on stderr). Production shares ONE [`ProcessNoticeSink`](crate::agent::mode_classifier::ProcessNoticeSink) instance with the mode classifier's own notices (`run_consult_subcommand`) rather than opening a second output path — dedup is per-key, so the two notices cannot suppress each other.
///
/// `pub(crate)` because `main.rs`'s `run_consult_subcommand` constructs it, from its own
/// already-resolved `classifier`/`configured_mode`/`untrusted_content`.
///
/// This used to add "same reasoning as `resolve_direct_mode` above", and that reasoning is the
/// opposite one (S4 Loop 2, Balthasar): `resolve_direct_mode` is `#[cfg(test)]` precisely
/// because it has NO production caller left. Borrowing its justification for a struct that does
/// have one invites the next reader to conclude this is test scaffolding and gate it away.
pub(crate) struct MagiRuntimeParams<'a> {
    /// The `ProviderKind` the trio runs under (REQ-A12c).
    pub(crate) kind: ProviderKind,
    /// Consulted only when no mode was declared by any other level (REQ-A07c).
    pub(crate) classifier: &'a dyn ModeClassifier,
    /// `[magi].default_mode`, if declared (REQ-A15).
    pub(crate) configured_mode: Option<Mode>,
    /// REQ-A07d's guard against classifying untrusted content.
    pub(crate) untrusted_content: bool,
    /// Feeds `RunContext::build`'s `endpoint_divergence` (REQ-A11d).
    pub(crate) magi_config: &'a MagiConfig,
    /// Feeds `RunContext::build`'s `timeout_below_formula` (SC-A04d).
    pub(crate) timeout_decision: TimeoutDecision,
    /// Where `timeout_decision.warning` is emitted, if present (SC-A04d).
    pub(crate) notice_sink: &'a dyn NoticeSink,
    /// Builds the [`Audited`](magi_rs::logging::auditor::Audited) a notice has to be.
    pub(crate) auditor: &'a magi_rs::logging::auditor::Auditor,
    /// REQ-EA01/EA03: `consult --structured-verdicts`. It rides HERE rather than as one more
    /// positional argument down the chain, for the same reason `OpenAiSettings` and `ModeSources`
    /// are named structs: several same-typed arguments in a row are a silent transposition hazard.
    ///
    /// Carried as the enum rather than a `bool` for the reason the enum exists: a bare `false`
    /// at a call site says nothing about why. Only the headless CLI ever sets `Include` —
    /// `ConsultTool` builds its own literal `Omit` and never these params (REQ-EA02).
    pub(crate) structured_verdicts: StructuredVerdicts,
    /// How this run's per-mage budget was decided (REQ-EB02b/EB04), read off
    /// `HeadlessContext::budget` — `run_consult` stamps it into `RunOutcome.applied_caps.budget`
    /// next to `timeout_secs`, the same override the `run_query` path applies via
    /// [`RunWiring::budget`].
    pub(crate) budget: BudgetTelemetry,
}

/// Runs `prompt` directly through the 3-perspective MAGI consensus, off the agent
/// tool-loop (REQ-H21), honoring the same input cap ([`check_query_size`] against
/// `runtime.magi_config.effective_max_query_bytes()`, REQ-H33/REQ-A11b) and the
/// `--timeout`/cancellation plumbing of the enclosing run (REQ-H36).
///
/// The `analyze` call runs on a joined task so a panic in `magi-core` surfaces as
/// a recoverable [`ConsultRunError::Runtime`] instead of unwinding the caller; on
/// cancellation or deadline the task is **aborted** (not merely detached) and
/// [`ConsultRunError::Timeout`] is returned.
///
/// # Parameters
/// - `magi` — shared MAGI orchestrator (the same one wired for the interactive `consult` tool).
/// - `prompt` — the decision/content to analyze.
/// - `cancel` — cooperative cancellation fired by the enclosing run's deadline.
/// - `timeout` — optional wall-clock ceiling for this consult specifically.
/// - `explicit_mode` — the lens declared by a human (`--mode`/the envelope field), if any. Wins outright, at zero classification cost (REQ-A07c, SC-A07g).
/// - `runtime` — the session-scoped [`MagiRuntimeParams`] (mode classifier, `[magi].default_mode`, the `untrusted_content` guard, and the provider `kind`); see its rustdoc for how each field is used.
///
/// # SC-A04d's warning
/// If `runtime.timeout_decision.warning` is `Some`, it is emitted via `runtime.
/// notice_sink` **before** anything else runs — the warning is about the
/// operator's `--timeout` choice, independent of whether THIS particular query
/// turns out to be valid, so it fires even on the `InputInvalid` path below.
///
/// # Errors
/// - [`ConsultRunError::InputInvalid`] if `prompt` is empty or exceeds `runtime.magi_config.effective_max_query_bytes()`.
/// - [`ConsultRunError::UntrustedContentRequiresMode`] if `runtime.untrusted_content` is `true` and neither `explicit_mode` nor `runtime.configured_mode` was declared (REQ-A07d/REQ-A07r).
/// - [`ConsultRunError::Timeout`] if cancelled or the deadline elapsed.
/// - [`ConsultRunError::Runtime`] if the MAGI analysis failed or panicked.
async fn analyze_direct(
    magi: &Arc<Magi>,
    prompt: &str,
    cancel: &CancellationToken,
    timeout: Option<Duration>,
    explicit_mode: Option<Mode>,
    runtime: &MagiRuntimeParams<'_>,
) -> Result<Value, ConsultRunError> {
    // SC-A04d: the value is obeyed either way (REQ-A04, `timeout` above is never
    // overridden) — this is the human-facing half of the same fact `RunContext.
    // timeout_below_formula` already reports in the JSON (REQ-A11d covers the
    // pipeline consumer; this covers whoever runs the command by hand).
    if let Some(w) = &runtime.timeout_decision.warning {
        let (audited, _) = runtime
            .auditor
            .audit(w, NOTICE_TARGET, None, NO_RESERVATION);
        runtime
            .notice_sink
            .once(NOTICE_TIMEOUT_BELOW_FORMULA, &audited);
    }

    // SC-A11c: the SAME `check_query_size` the tool path and the TUI's explicit
    // `/consult` call, against the SAME `MagiConfig`-resolved cap — one function,
    // not three copies that could drift apart.
    if check_query_size(prompt, runtime.magi_config.effective_max_query_bytes()).is_err() {
        return Err(ConsultRunError::InputInvalid);
    }

    // The direct consult path has no agent and no tool-loop, so there is no
    // `agent_chosen` level to feed in (`None`, third argument) — only "a human
    // declared it" (explicit/configured) vs. "classify it" (REQ-A07c).
    let resolution = resolve_mode_guarded(
        ModeSources {
            explicit: explicit_mode,
            configured: runtime.configured_mode,
            ..ModeSources::default()
        },
        runtime.untrusted_content,
        Some(runtime.classifier),
        prompt,
    )
    .await
    .map_err(ConsultRunError::UntrustedContentRequiresMode)?;
    let mode = resolution.mode;

    let magi = Arc::clone(magi);
    let owned = prompt.to_string();
    let handle = tokio::spawn(async move { magi.analyze(&mode, &owned).await });
    // RAII backstop (same primitive as `ConsultTool::execute`, `src/tools/
    // consult.rs`): aborts the spawned analysis if THIS future is dropped
    // before the `select!` resolves — e.g. the enclosing `run_consult`/
    // `run_query` future is dropped outside of the explicit cancel/deadline
    // arms below. Without it, a dropped `JoinHandle` only detaches the task,
    // orphaning the three in-flight LLM calls. The explicit arms still call
    // `.abort()` too (belt-and-suspenders): they fire, and hand back a typed
    // `Timeout`, before this future itself is dropped by their caller.
    let abort_guard = AbortOnDrop::new(handle.abort_handle());

    // A `None` timeout parks forever, so the deadline arm never fires; the cancel
    // arm still aborts if the enclosing run is cancelled.
    let deadline = async {
        match timeout {
            Some(dur) => tokio::time::sleep(dur).await,
            None => future::pending::<()>().await,
        }
    };
    tokio::pin!(deadline);

    // **`biased;` makes the ORDER a decision, and the middle arm was in the wrong place**
    // (S4 Loop 2, Balthasar). With the deadline ahead of the join handle, a consult that
    // finished in the same poll as the timer expiring was thrown away and reported as a
    // `Timeout`: three model calls already paid for, discarded over a race the operator cannot
    // influence, to enforce a bound that had just been met anyway.
    //
    // Completion first, then the two stop signals. The wall-clock guarantee is untouched — if
    // the handle is not ready, the deadline arm is polled in the same pass and still fires — but
    // a finished answer is now never discarded in favour of a tie.
    //
    // `cancel` stays FIRST, deliberately. It is external cancellation, not this consult's own
    // budget: the caller is tearing down, and handing it a report it asked not to receive is a
    // different contract from delivering work that completed on time. That distinction is worth
    // more than the symmetry.
    //
    // **As it stands that arm is UNREACHABLE, and the ordering above is about the other two.**
    // `analyze_direct` has exactly one caller — `run_consult` — which creates the token two
    // lines before the call and never cancels it. So the deadline is the only stop signal today
    // and the completion-vs-timer order is the whole of the decision.
    //
    // Said carefully because the two previous attempts at this comment were both wrong, in
    // opposite directions, and each was written confidently (S4 Loop 2, Caspar then Balthasar).
    // The first justified the order by "external cancellation"; the second attributed that to
    // `run_query`, which does not call this function at all. The arm and its parameter stay:
    // `ConsultTool::execute` runs the same shape over a token that IS live, so the signature is
    // the one a second caller would need — but nothing here exercises it, and a comment that
    // implies otherwise is how this got documented wrong twice.
    //
    // **UNGUARDED, and said plainly rather than left to look tested.** Every other fix in this
    // gate carries a test that goes red when the fix is reverted; this one does not, because the
    // scenario cannot be expressed with the harness available. Constructing the tie needs a
    // paused clock, and under `start_paused` the analysis never completes at all — magi-core's
    // internal tasks do not participate in the auto-advance, so the double's `tokio::time::sleep`
    // never resolves and the run times out with or without a deadline. A test built anyway would
    // pass for a reason unrelated to arm order, which is worse than none.
    //
    // What stands in for it is that the change is small and total: three arms, one moved, and
    // `biased;` means the order is the whole behaviour. If someone reorders these again, this
    // paragraph is the only thing that will stop them.
    let joined = tokio::select! {
        biased;
        () = cancel.cancelled() => {
            abort_guard.abort();
            return Err(ConsultRunError::Timeout);
        }
        joined = handle => joined,
        () = &mut deadline => {
            abort_guard.abort();
            return Err(ConsultRunError::Timeout);
        }
    };

    match joined {
        Ok(Ok(report)) => {
            // The annotation (REQ-A12c) is applied BEFORE truncating —
            // `report_to_consult_json` renders `truncated.text` verbatim, so the
            // annotation step has to happen here, not inside `truncate_report`.
            let annotated = annotate_report_text(&report, runtime.kind);
            // REQ-A11b/SC-A11d (the `magi consult` headless direct route): bounds
            // the report the same way the tool-loop route does, with the SAME
            // truncation-level vocabulary surfaced via `report_truncated`.
            let truncated =
                truncate_report(&annotated, runtime.magi_config.effective_tool_result_cap());
            // `resolution` already carries the REAL `classification_attempted`
            // signal from `resolve_mode_guarded` above, unlike `ConsultTool::
            // execute`'s call site, which only has mode+source round-tripped
            // through the tool-call input. `runtime.magi_config`/
            // `runtime.timeout_decision` are the REAL config/decision the caller
            // resolved (fix round 1, Finding 1) — `RunContext::build` combines
            // them with `resolution` exactly as its own contract specifies.
            let ctx =
                RunContext::build(runtime.magi_config, &resolution, &runtime.timeout_decision);
            // REQ-EA01/EA03: the ONLY surface that can ask for them. `ConsultTool` never builds
            // these params, so the agent-facing path cannot reach `Include` even by accident.
            Ok(report_to_consult_json(
                &report,
                &truncated,
                &resolution,
                &ctx,
                runtime.structured_verdicts,
            ))
        }
        Ok(Err(e)) => Err(ConsultRunError::Runtime(explain_magi_error(
            &e,
            runtime.kind,
        ))),
        Err(join_err) => Err(ConsultRunError::Runtime(format!(
            "consult crashed: {join_err}"
        ))),
    }
}

/// Maps a [`ConsultRunError`] to the `(response, consult, stop_reason, error)`
/// tuple of a **direct** `magi consult` outcome (REQ-H21/H33/H36), reused by
/// [`run_consult`] so the direct path's error taxonomy stays in one place.
fn consult_error_outcome(
    err: ConsultRunError,
) -> (
    Option<String>,
    Option<Value>,
    StopReason,
    Option<ErrorPayload>,
) {
    let payload = match err {
        ConsultRunError::InputInvalid => ErrorPayload {
            message: CONSULT_INPUT_INVALID_MESSAGE.to_string(),
            kind: ErrorKind::InputInvalid,
        },
        // Same `input_invalid` family as the cap check above: `untrusted_content`
        // without a declared mode is a configuration problem, not a runtime one
        // (REQ-A07d/REQ-A07r) — `ModeError`'s `Display` already names the fix.
        ConsultRunError::UntrustedContentRequiresMode(e) => ErrorPayload {
            message: e.to_string(),
            kind: ErrorKind::InputInvalid,
        },
        ConsultRunError::Timeout => ErrorPayload {
            message: TIMEOUT_MESSAGE.to_string(),
            kind: ErrorKind::Timeout,
        },
        ConsultRunError::Runtime(message) => ErrorPayload {
            message: sanitize_error_message(&message),
            kind: ErrorKind::Runtime,
        },
    };
    (None, None, StopReason::Error, Some(payload))
}

/// Projects a direct consult into the normalized transcript (REQ-H14): the user
/// prompt, plus the MAGI report as a single assistant turn when one was produced.
/// No `tool_calls` appear — the direct path never runs the agent tool-loop.
fn build_consult_transcript(prompt: &str, report: Option<&str>) -> Vec<TranscriptEntry> {
    let mut transcript = vec![TranscriptEntry {
        role: ROLE_USER.to_string(),
        content: prompt.to_string(),
        tool_calls: None,
    }];
    if let Some(text) = report {
        transcript.push(TranscriptEntry {
            role: ROLE_ASSISTANT.to_string(),
            content: text.to_string(),
            tool_calls: None,
        });
    }
    transcript
}

/// Extracts the MAGI object of the last **successful** `consult` tool call from a
/// finished run's records (REQ-H22): the forced or proactive consult result, or
/// `None` when no consult succeeded (e.g. denied by the tier).
///
/// # The `.ok()` is a documented impossibility, not a swallowed error
///
/// B9 forbids discarding a parse failure, and S4 Loop 2 (Caspar) rightly asked about this one.
/// The filter above it already required `rec.ok`, and a `consult` record is `ok` only when
/// `ConsultTool::execute` returned — which returns `report_to_consult_json`'s value, serialized
/// by `serde_json` immediately before. So the string being parsed here is one this process
/// produced from a `Value` microseconds earlier; a failure would mean `serde_json` cannot read
/// back what it just wrote.
///
/// It grows no error channel, because there is no honest thing to report through one: this
/// function has no notices sink, and the caller's only recourse for a self-contradictory record
/// would be the same `None` it already returns. What it does carry is a `debug_assert!`, so the
/// impossibility is checked wherever checking is free rather than merely asserted in prose.
fn extract_consult_value(calls: &[(String, ToolCallRecord)]) -> Option<Value> {
    calls
        .iter()
        .rev()
        .find(|(_, rec)| rec.name == CONSULT_TOOL && rec.ok)
        .and_then(
            |(_, rec)| match serde_json::from_str::<Value>(&rec.result) {
                Ok(value) => Some(value),
                Err(e) => {
                    // The impossibility now SAYS SO when it stops being one (S4 Loop 2, Balthasar,
                    // whose counter-proposal beat the rejection it answered). The reasoning below
                    // still holds — there is no honest channel to report through from here — but a
                    // `debug_assert!` needs none: it costs nothing in release and turns a silent
                    // swallow into a loud failure in every test and dev build, which is exactly
                    // where an invariant that quietly stopped holding would otherwise hide.
                    debug_assert!(
                        false,
                        "a consult record marked ok holds JSON this process serialized moments \
                     earlier; if it no longer parses, the contract between ConsultTool::execute \
                     and this reader has broken: {e}"
                    );
                    None
                }
            },
        )
}

/// JSON key of the deadline verdict, shared with `report_to_consult_json`'s own literal.
///
/// It is written in two places because they sit on opposite sides of the tool boundary and
/// neither can see the other's constant: `tools::consult` builds the object, this module
/// corrects one value in it. Naming it here at least makes the coupling greppable.
const CONSULT_TIMEOUT_BELOW_FORMULA_KEY: &str = "timeout_below_formula";

/// Overwrites the consult object's `timeout_below_formula` with the value the RUN knows
/// (SC-A04d).
///
/// **`report_to_consult_json` stays the single source of the shape.** It cannot be the source
/// of this particular *value*: no `--timeout` concept reaches a tool-loop-dispatched consult, so
/// `ConsultTool::execute` hardcodes `false` — correctly, because from inside the tool it is
/// unknowable. The deadline belongs to the run, and the run is here. The key is only ever
/// replaced, never introduced: if the object is not an object, or does not already carry the
/// key, nothing is touched, so this can never invent a field the contract does not declare.
///
/// # Parameters
/// * `consult` - the extracted consult object, if a consult succeeded.
/// * `below_formula` - the run's own verdict on its deadline.
///
/// # Returns
/// `consult`, with the key corrected when there was one to correct.
fn apply_timeout_verdict(consult: Option<Value>, below_formula: bool) -> Option<Value> {
    let mut consult = consult?;
    if let Some(slot) = consult
        .as_object_mut()
        .and_then(|o| o.get_mut(CONSULT_TIMEOUT_BELOW_FORMULA_KEY))
    {
        *slot = Value::Bool(below_formula);
    }
    Some(consult)
}

/// Mutable state collected by [`RunTracker`] during a run.
///
/// Written only from inside the agent's task via the [`RunObserver`] callbacks,
/// then snapshotted once after the run — no lock is ever held across an `.await`.
#[derive(Default)]
struct TrackerState {
    /// Every resolved tool call, in execution order, paired with its tool-use id
    /// (the id correlates a call with the assistant `ToolUse` block that
    /// requested it, for transcript assembly).
    calls: Vec<(String, ToolCallRecord)>,
    /// Number of tools denied by the **tier** (distinct from execution failures),
    /// the authoritative `tier_denied` signal for REQ-H23b.
    tier_denials: usize,
    /// TextDelta block count of the agent's final turn (REQ-H23b `response_empty`).
    final_turn_text_blocks: usize,
    /// Token usage summed across every provider turn that reported one
    /// (`RunObserver::on_usage`); `(input_tokens, output_tokens)`.
    usage_total: (u64, u64),
}

/// [`RunObserver`] that enforces the tier policy for every tool and records the
/// run for a faithful [`RunOutcome`].
struct RunTracker {
    /// Tier policy consulted to authorize each tool call.
    policy: Policy,
    /// Collected run state behind a `std::sync::Mutex` (accessed only from
    /// synchronous observer callbacks, never across an `.await`).
    state: Mutex<TrackerState>,
}

/// Runs `f` with the tracker state, recovering a poisoned lock instead of
/// panicking (a poisoned lock never loses the collected data).
fn with_state<R>(state: &Mutex<TrackerState>, f: impl FnOnce(&mut TrackerState) -> R) -> R {
    let mut guard = state.lock().unwrap_or_else(PoisonError::into_inner);
    f(&mut guard)
}

impl RunObserver for RunTracker {
    fn authorize(&self, tool_name: &str) -> bool {
        let allowed = self.policy.approves(tool_name);
        if !allowed {
            with_state(&self.state, |s| s.tier_denials += 1);
        }
        allowed
    }

    fn on_tool_call(
        &self,
        id: &str,
        name: &str,
        input: &serde_json::Value,
        result: &str,
        ok: bool,
        ms: u64,
    ) {
        let record = ToolCallRecord {
            name: name.to_string(),
            input: input.clone(),
            result: result.to_string(),
            ms,
            ok,
        };
        with_state(&self.state, |s| s.calls.push((id.to_string(), record)));
    }

    fn on_final_turn(&self, text_block_count: usize) {
        with_state(&self.state, |s| s.final_turn_text_blocks = text_block_count);
    }

    fn on_usage(&self, input_tokens: u64, output_tokens: u64) {
        with_state(&self.state, |s| {
            s.usage_total.0 = s.usage_total.0.saturating_add(input_tokens);
            s.usage_total.1 = s.usage_total.1.saturating_add(output_tokens);
        });
    }
}

/// Resolves the effective wall-clock timeout for a run from its **tier policy**
/// (REQ-H36), applying the tier default when none was explicitly configured.
///
/// **Not to be confused with `magi_rs::magi::resolve_run_timeout`**, which derives REQ-A04's
/// minimum from `agent_timeout_secs` and is a different question entirely. The two used to
/// share a name, distinguishable only by import path, and that is the likeliest reason
/// `run_query_subcommand` went years calling this one where the consult route also needed the
/// other; hence the rename.
///
/// Precedence: an explicit `--timeout` / `[headless] timeout_secs` (already
/// resolved into [`Policy::timeout`]) wins; otherwise any **tool-executing**
/// tier (`--auto`/`--full-auto`) receives `full_auto_timeout_secs` — the
/// EFFECTIVE default (spec §11; an operator can lower it below
/// [`FULL_AUTO_TIMEOUT_SECS`](magi_rs::headless::limits::FULL_AUTO_TIMEOUT_SECS)
/// via `[headless] timeout_secs`) — since the
/// permissive tiers always carry a wall-clock ceiling; the read-only
/// `default` tier gets no timeout. The result is passed straight to
/// [`run_query`].
///
/// # Returns
/// `Some(duration)` when a ceiling applies; `None` for an unbounded run.
#[must_use]
pub fn resolve_tier_timeout_default(
    policy: &Policy,
    full_auto_timeout_secs: u64,
) -> Option<Duration> {
    if let Some(secs) = policy.timeout() {
        return Some(Duration::from_secs(secs));
    }
    match policy.tier() {
        Tier::Auto | Tier::FullAuto => Some(Duration::from_secs(full_auto_timeout_secs)),
        Tier::Default => None,
    }
}

/// Everything `main.rs` resolved for ONE `magi query` run that [`run_query`] cannot derive on
/// its own.
///
/// **Bundled rather than added as more parameters.** Those three values live one level down, in
/// the [`autonomous`](Self::autonomous) field — this struct carries `timeout`, `autonomous` and
/// `timeout_below_formula`, and naming the inner three here as if they were its own sent readers
/// looking for fields that are not on it (S4 Loop 2, Balthasar).
///
/// `run_query` already carried six parameters, and the values that had to arrive were three
/// (`gate_thresholds`, `mode_config`, `gate_telemetry`) —
/// enough to push the signature past the point where the only remaining move is an
/// `#[allow(clippy::too_many_arguments)]`, which this repo tracks as priority debt. Grouping
/// them behind [`AutonomousRunConfig`](crate::AutonomousRunConfig) keeps the arity where it was
/// AND makes the operator's configuration a value the caller must produce rather than a set of
/// fields it can quietly leave at their defaults.
///
/// # Fields
/// - `timeout` — wall-clock ceiling for the whole run (REQ-H36). `None` ⇒ unbounded. On elapse the agent future is dropped (cancelling the in-flight LLM stream), the run's [`CancellationToken`] fires so any `bash` subprocess *tree* is killed, and a partial outcome with `stop_reason = Error` / `error.kind = Timeout` is returned.
/// - `autonomous` — the operator's autonomous-consult configuration (`[magi.complexity]`, `[magi].default_mode`, `[magi].untrusted_content`, the gate telemetry sink), overlaid on the tier-resolved [`AgentRunConfig`] so the tool loop's self-routed `consult` obeys `magi.toml` (REQ-A20/A20b, REQ-A07/A07d).
/// - `timeout_below_formula` — whether `timeout` is shorter than REQ-A04's minimum for a run that can dispatch a consult (`classification_ceiling + 2 × agent_timeout_secs + slack`). The value is still obeyed — a wall-clock cap is the operator's call, not a safety invariant — but a consult that needs its schema retry cannot complete under it, and SC-A04d requires that fact to reach the JSON as well as stderr: whoever passes an explicit `--timeout` is running a pipeline, i.e. exactly who will not read stderr.
/// - `budget` — how `main.rs` derived this run's per-mage budget (REQ-EB02b/EB04), stamped into `RunOutcome.applied_caps.budget` so the decision is machine-readable rather than only a stderr notice.
pub struct RunWiring {
    /// Wall-clock ceiling for the whole run (REQ-H36).
    pub timeout: Option<Duration>,
    /// The operator's autonomous-consult configuration (REQ-A20/A07d).
    pub autonomous: crate::AutonomousRunConfig,
    /// Whether `timeout` falls below REQ-A04's minimum for a consult-capable run (SC-A04d).
    pub timeout_below_formula: bool,
    /// How this run's per-mage budget was decided (REQ-EB02b/EB04), read off
    /// `HeadlessContext::budget` and stamped into `RunOutcome.applied_caps.budget` — the
    /// placeholder `resolution::resolve` built the `Resolved` with cannot know this value.
    pub budget: BudgetTelemetry,
}

/// Wall-clock milliseconds elapsed since `start`, saturating instead of wrapping.
fn elapsed_ms(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// The `(response, stop_reason, error)` triple of a wall-clock timeout abort
/// (REQ-H36): no response, a first-class [`ErrorKind::Timeout`] payload. A
/// forced consult now runs strictly INSIDE the agent loop (REQ-H22:
/// `AgentRunConfig::force_consult`), sharing the loop's single
/// `tokio::time::timeout` wrapper — there is no separate forced-consult
/// deadline to reconcile, so this triple has exactly one caller: the run-level
/// timeout arm in [`run_query`].
fn timeout_outcome() -> (Option<String>, StopReason, Option<ErrorPayload>) {
    (
        None,
        StopReason::Error,
        Some(ErrorPayload {
            message: TIMEOUT_MESSAGE.to_string(),
            kind: ErrorKind::Timeout,
        }),
    )
}

/// Projects the finished conversation `history` into the normalized transcript
/// the output contract fixes (REQ-H14): one entry per user/assistant turn, with
/// an assistant turn's `ToolUse` blocks folded into its `tool_calls` (resolved
/// against `records` by tool-use id). The `User`-role tool-result carrier
/// messages are not emitted standalone — their outcomes already live inside the
/// requesting assistant entry's `tool_calls`, matching the fixed `RunOutcome`
/// shape.
fn build_transcript(
    history: &[Message],
    records: &HashMap<&str, &ToolCallRecord>,
) -> Vec<TranscriptEntry> {
    let mut transcript = Vec::new();
    for msg in history {
        match msg.role {
            Role::User => {
                // Every current production write path constructs a tool-result carrier
                // message with ONLY `Content::ToolResult` blocks (see `agent::mod.rs`'s
                // `tool_results.push(result_content)` / `content: vec![result_content]`
                // sites) — never mixed with `Content::Text` in the same message. But
                // `Message`/`Content` place no such constraint in the TYPE SYSTEM (MAGI S6
                // gate finding 7): CLAUDE.md's durable invariant is that a tool result must
                // never be COLLAPSED into text, which cuts the other way from what an
                // unconditional "any ToolResult block ⇒ discard the whole message" would do
                // if a message ever carried both. The distinction that matters is whether
                // there is real text to keep, not whether a ToolResult block is ALSO
                // present: a pure tool-result carrier (no text) is folded away as before;
                // anything with text is kept, so a hypothetical mixed message — from a
                // future write path, or a session loaded from the DB under `load_all`
                // (REQ-28) — cannot silently lose its real user text (B9).
                let text = msg.concat_text();
                let is_pure_tool_result_carrier = text.is_empty()
                    && msg
                        .content
                        .iter()
                        .any(|c| matches!(c, Content::ToolResult { .. }));
                if is_pure_tool_result_carrier {
                    continue;
                }
                transcript.push(TranscriptEntry {
                    role: ROLE_USER.to_string(),
                    content: text,
                    tool_calls: None,
                });
            }
            Role::Assistant => {
                let calls: Vec<ToolCallRecord> = msg
                    .content
                    .iter()
                    .filter_map(|c| match c {
                        Content::ToolUse { id, name, input } => Some(
                            records
                                .get(id.as_str())
                                .map(|r| (*r).clone())
                                // Defensive: a `ToolUse` with no recorded outcome
                                // should not occur (every call is recorded), but
                                // never drop the block silently.
                                .unwrap_or_else(|| ToolCallRecord {
                                    name: name.clone(),
                                    input: input.clone(),
                                    result: String::new(),
                                    ms: 0,
                                    ok: false,
                                }),
                        ),
                        _ => None,
                    })
                    .collect();
                let tool_calls = if calls.is_empty() { None } else { Some(calls) };
                transcript.push(TranscriptEntry {
                    role: ROLE_ASSISTANT.to_string(),
                    content: msg.concat_text(),
                    tool_calls,
                });
            }
        }
    }
    transcript
}

/// Awaits the ttfb-measuring drain task with [`DRAIN_GRACE`] as an upper bound, aborting it on
/// timeout so an unexpectedly long-lived `chunk_rx.recv()` loop cannot keep [`run_query`] from
/// returning (MAGI S6 gate finding 6 — see [`DRAIN_GRACE`]'s rustdoc for why this bound exists
/// instead of an unconditional `.await`).
///
/// # Parameters
/// - `drain` — the task spawned by `run_query` draining `chunk_rx` and computing
///   time-to-first-byte.
///
/// # Returns
/// The measured `ttfb_ms`, or `None` if the task panicked or did not finish within
/// [`DRAIN_GRACE`] (in which case it is aborted, so it cannot linger consuming resources).
async fn bounded_drain_result(mut drain: tokio::task::JoinHandle<Option<u64>>) -> Option<u64> {
    match tokio::time::timeout(DRAIN_GRACE, &mut drain).await {
        Ok(join_result) => join_result.ok().flatten(),
        Err(_elapsed) => {
            drain.abort();
            None
        }
    }
}

/// Runs `prompt` directly through the 3-perspective MAGI consensus for the
/// `magi consult` subcommand (REQ-H21), returning a [`RunOutcome`] whose `consult`
/// field holds the MAGI object.
///
/// This is the **forced, direct** path: it always runs the three perspectives via
/// `magi-core` (never the agent tool-loop), so it does **not** consult the tier
/// tool-gate — `magi consult` *is* the multi-perspective analysis. The outcome
/// reflects that direct nature: `tool_calls` is empty, `response` is the MAGI
/// report text, and `transcript` is the prompt plus that report.
///
/// An over-cap prompt (`check_query_size` against `MagiConfig::
/// effective_max_query_bytes()`) or an empty prompt is
/// rejected with `error.kind = input_invalid` (REQ-H33: reject, never truncate),
/// which the caller (Task 7) maps to exit 2. A `--timeout` abort yields
/// `error.kind = timeout` (REQ-H36); a MAGI failure yields a sanitized `runtime`.
///
/// # Parameters
/// - `resolved` — effective run parameters; supplies `model`/`provider`/ `applied_caps` for the output.
/// - `magi` — shared MAGI orchestrator (same one wired for the `consult` tool).
/// - `prompt` — the decision/content to analyze.
/// - `timeout` — optional wall-clock ceiling (REQ-H36).
/// - `explicit_mode` — the lens declared by a human (`--mode`/the envelope field); `None` triggers exactly one classification call, never more (REQ-A07c, SC-A07f/g).
/// - `runtime` — the session-scoped [`MagiRuntimeParams`] (mode classifier, `[magi].default_mode`, the `untrusted_content` guard, and the `kind` that feeds [`report_to_consult_json`] via [`analyze_direct`]).
///
/// # Gaps (documented, never fabricated)
/// - `usage` is `Usage { 0, 0 }`: `magi-core` does not surface token counts here.
/// - `timings.per_turn_ms` is empty and `ttfb_ms` is `None`: the direct consult is a single buffered analysis, not a streamed turn sequence.
pub async fn run_consult(
    resolved: Resolved,
    magi: Arc<Magi>,
    prompt: &str,
    timeout: Option<Duration>,
    explicit_mode: Option<Mode>,
    runtime: &MagiRuntimeParams<'_>,
) -> RunOutcome {
    // A fresh token: the direct consult has no enclosing agent run to inherit a cancellation
    // from — and nothing here ever cancels it, so `analyze_direct`'s cancel arm never fires.
    // The `timeout` argument on the next line is what stops a long run, through that function's
    // own deadline arm.
    //
    // This used to say the cancel arm "only fires if this run's own `timeout` deadline elapses",
    // which named the wrong arm for the right event (S4 Loop 2, Balthasar).
    let cancel = CancellationToken::new();
    let run_start = Instant::now();
    let result = analyze_direct(&magi, prompt, &cancel, timeout, explicit_mode, runtime).await;
    let total_ms = elapsed_ms(run_start);

    let (response, consult, stop_reason, error) = match result {
        Ok(value) => {
            let response = value
                .get("report")
                .and_then(Value::as_str)
                .map(str::to_string);
            (response, Some(value), StopReason::Done, None)
        }
        Err(err) => consult_error_outcome(err),
    };

    let transcript = build_consult_transcript(prompt, response.as_deref());

    tracing::info!(target: "magi_rs::headless", "consult complete: stop_reason={stop_reason:?}");

    RunOutcome {
        response,
        model: resolved.model,
        provider: resolved.provider,
        // Gap: magi-core does not surface token counts to this layer.
        usage: Usage {
            input_tokens: 0,
            output_tokens: 0,
        },
        timings: Timings {
            total_ms,
            ttfb_ms: None,
            per_turn_ms: Vec::new(),
        },
        stop_reason,
        // Direct path: no agent tool-loop, so no tool calls.
        tool_calls: Vec::new(),
        transcript,
        consult,
        applied_caps: AppliedCaps {
            // Same reasoning as `run_query`: `resolved.applied_caps.timeout_secs` is a
            // static `None` from `resolution::resolve`; the direct consult route knows
            // its own effective ceiling as the `timeout` parameter and must stamp it
            // here (MS2 gate S6 finding 1).
            timeout_secs: timeout.map(|d| d.as_secs()),
            budget: runtime.budget,
            ..resolved.applied_caps
        },
        error,
    }
}

/// Runs `agent` over `prompt` non-interactively under `policy`, returning a fully
/// assembled [`RunOutcome`] (REQ-H02/H09/H23b).
///
/// Auto-approval is per tier: a [`RunObserver`] answers each tool authorization
/// from [`Policy::approves`] with no interactive wait — an approved tool runs
/// (its hard barriers still apply inside the tool), a denied tool is recorded
/// with `ok = false` and the agent **continues** (a denial never aborts the run,
/// REQ-H09). Under `--full-auto` ([`Policy::silences_soft_guards`]) the run also
/// uses the elevated tool-call cap and disables the repetitive-call soft guard;
/// no hard barrier is ever affected.
///
/// `stop_reason` is deterministic (REQ-H23b), with priority
/// `Error > MaxToolCalls > Denied > Done`:
/// - the cap being hit ⇒ [`StopReason::MaxToolCalls`] (a terminal state, not an error — no `error` payload, maps to exit 0);
/// - any other run error (including a repetitive-guard abort) ⇒ [`StopReason::Error`] with a sanitized `error` payload;
/// - otherwise, at least one tier denial **and** an empty final turn (zero `TextDelta` blocks) ⇒ [`StopReason::Denied`]; else [`StopReason::Done`].
///
/// # Parameters
/// - `resolved` — effective run parameters; supplies the output `model`, `provider` and `applied_caps` (the cap actually enforced comes from `policy`).
/// - `policy` — tier policy driving authorization, the tool-call cap and the soft-guard silencing.
/// - `agent` — a constructed agent (provider + registered tools).
/// - `prompt` — the resolved user prompt.
/// - `wiring` — what `main.rs` resolved for THIS run and the runner cannot derive on its own: the wall-clock ceiling and the operator's autonomous-consult configuration. See [`RunWiring`].
///
/// # Forced consult (REQ-H22)
/// `resolved.consult == Some(true)` sets `AgentRunConfig::force_consult`, which
/// makes [`Agent::run_tool_loop`] inject exactly one **in-loop** `consult` tool
/// call before the first provider turn: a genuine `ToolUse`/`ToolResult`
/// message pair through the SAME registered `consult` tool a proactive
/// (`--auto`+) call would use, authorized by the SAME tier gate (the observer's
/// [`RunObserver::authorize`], so the tier wins — in `default` it is recorded
/// denied, `ok = false`, and `consult` stays `None`, never elevated), counted
/// against `max_tool_calls`, and fed back into the conversation so the model's
/// own next turn can react to it. If the model itself also requests `consult`
/// afterward, that request is answered but never re-executed (REQ-H22:
/// exactly one invocation per forced run). The caller (`agent` here) must have
/// the `consult` tool registered for the injection to succeed — `main.rs`
/// registers it whenever a MAGI orchestrator is wired, unconditionally, so
/// this is already true for every production headless run with a live
/// backend. The injected consult shares the run's single wall-clock deadline
/// automatically: it runs inside the same `tokio::time::timeout` that bounds
/// [`Agent::query_streaming`] below, so there is no separate budget to
/// reconcile — a mid-consult timeout surfaces as the ordinary run-level
/// `Timeout` (REQ-H36).
///
/// # Usage accounting
/// `RunOutcome.usage` is the sum of every `ResponseChunk::Usage` the provider
/// reported across the run's turns (`RunObserver::on_usage`, accumulated in
/// `RunTracker`). It reflects only the MAIN agent-loop tokens: a `consult` tool
/// call (forced or proactive) runs its 3 magi-core LLM calls through
/// `magi-core`'s own `LlmProvider`, which exposes no token counts to this
/// layer, so those tokens are **not** included in the total — the same gap
/// `run_consult` documents for the direct `magi consult` path. A backend that
/// never reports usage yields `{0, 0}`, never a fabricated value.
///
/// # Gaps (documented, never fabricated)
/// - `timings.per_turn_ms` is empty: turn boundaries are not observable from outside the loop. `total_ms` and (best-effort) `ttfb_ms` are measured.
pub async fn run_query(
    resolved: Resolved,
    policy: Policy,
    agent: &mut Agent,
    prompt: &str,
    wiring: &RunWiring,
) -> RunOutcome {
    let timeout = wiring.timeout;
    // SC-A04d: a deadline too short for a consult's schema retry is recorded up front, next to
    // the tier warnings, so the run log says WHY a later `Timeout` was structural rather than
    // leaving an opaque `error.kind = timeout` as the only trace.
    if wiring.timeout_below_formula {
        tracing::warn!(target: "magi_rs::headless", "{TIMEOUT_BELOW_FORMULA_LOG}");
    }
    // Tier warnings (only under --full-auto) are recorded up front.
    for warning in policy.warnings() {
        tracing::warn!(target: "magi_rs::headless", "{warning}");
    }

    let tracker = Arc::new(RunTracker {
        policy: policy.clone(),
        state: Mutex::new(TrackerState::default()),
    });
    // Cancellation fired on wall-clock timeout so in-flight tool subprocess trees
    // are killed (bash) and the run stops (REQ-H36).
    let cancel = CancellationToken::new();
    // Effective system prompt (REQ-H12b): `resolved.system` already carries the
    // origin-aware decision (operator default, or the caller override only when
    // `--allow-system-override` enabled it — `applied_caps.system_override_applied`
    // reports which). Empty text is equivalent to no system prompt at all; the
    // provider layer treats `Some("")` the same as `None`, but normalizing here
    // keeps the intent explicit at the call site.
    let system = resolved.system.text();
    // The tier-resolved half: everything that is a property of THIS run rather than of the
    // operator's `magi.toml`.
    let tier_config = AgentRunConfig {
        max_tool_calls: usize::try_from(policy.max_tool_calls()).unwrap_or(usize::MAX),
        disable_repetitive_guard: policy.silences_soft_guards(),
        observer: Some(Arc::clone(&tracker) as Arc<dyn RunObserver>),
        cancel: cancel.clone(),
        system: (!system.is_empty()).then(|| system.to_string()),
        // REQ-H22: `magi query --consult` / envelope `consult:true` forces
        // exactly one IN-LOOP consult (see the `# Forced consult` rustdoc
        // above) rather than the pre-refactor post-loop pass.
        force_consult: resolved.consult == Some(true),
        ..AgentRunConfig::default()
    };
    // …and the operator's half on top (REQ-A20/A20b, REQ-A07/A07d). Before this, the three
    // fields stayed at `AgentRunConfig::default()`, so `[magi.complexity]` never applied here
    // and `untrusted_content` could never fail a self-routed consult closed (SC-A07r).
    let config = wiring.autonomous.apply(tier_config);

    let (chunk_tx, mut chunk_rx) =
        tokio::sync::mpsc::channel::<StreamPiece>(CHUNK_CHANNEL_CAPACITY);

    let run_start = Instant::now();
    // Drain the stream concurrently: measure time-to-first-byte and prevent the
    // agent from blocking on channel backpressure. The task ends when
    // `query_streaming` drops `chunk_tx`.
    let drain = tokio::spawn(async move {
        let mut ttfb_ms: Option<u64> = None;
        while let Some(piece) = chunk_rx.recv().await {
            match &piece {
                StreamPiece::Content(_) if ttfb_ms.is_none() => {
                    ttfb_ms = Some(elapsed_ms(run_start));
                }
                // REQ-29's second half. The TUI renders these; here they used
                // to be consumed and dropped, so an operator whose embedder
                // was failing was told nothing at all and watched memories
                // pile up unembedded.
                StreamPiece::Notice(text) => eprintln!("{}", notice_line(text)),
                _ => {}
            }
        }
        ttfb_ms
    });

    // Race the run against the optional wall-clock deadline. On elapse the query
    // future is dropped (cancelling the LLM stream); firing `cancel` lets a
    // still-polled tool observe cancellation, and the bash group killer drops
    // with the future — killing any subprocess tree either way. `None` on the
    // outcome marks a timeout.
    let query_fut = agent.query_streaming(prompt, chunk_tx, config);
    let outcome_result: Option<Result<String, anyhow::Error>> = match timeout {
        Some(dur) => match tokio::time::timeout(dur, query_fut).await {
            Ok(r) => Some(r),
            Err(_elapsed) => {
                cancel.cancel();
                None
            }
        },
        None => Some(query_fut.await),
    };
    let ttfb_ms = bounded_drain_result(drain).await;

    let total_ms = elapsed_ms(run_start);

    // Snapshot the observer state once (the run is over; no more callbacks).
    let (calls, tier_denied, response_empty, usage_total) = with_state(&tracker.state, |s| {
        (
            s.calls.clone(),
            s.tier_denials > 0,
            s.final_turn_text_blocks == 0,
            s.usage_total,
        )
    });
    let tool_calls: Vec<ToolCallRecord> = calls.iter().map(|(_, rec)| rec.clone()).collect();
    let records_by_id: HashMap<&str, &ToolCallRecord> =
        calls.iter().map(|(id, rec)| (id.as_str(), rec)).collect();
    let transcript = build_transcript(agent.history(), &records_by_id);

    // Deterministic outcome mapping (priority Error > MaxToolCalls > Denied >
    // Done); a wall-clock timeout is a first-class `Error` (kind = timeout) and
    // dominates every other signal because it dropped the run mid-flight — a
    // forced consult that exhausts the deadline is dropped along with the rest
    // of the loop and produces this same `None` arm (REQ-H36).
    let (response, stop_reason, error) = match outcome_result {
        None => timeout_outcome(),
        Some(Ok(text)) => {
            let stop_reason = if tier_denied && response_empty {
                StopReason::Denied
            } else {
                StopReason::Done
            };
            (Some(text), stop_reason, None)
        }
        Some(Err(e)) => {
            let message = e.to_string();
            // `MAX_TOOL_CALLS_ERROR`'s exact string value is a stability
            // contract with `crate::agent`'s two producer sites (see its
            // rustdoc in `src/agent/mod.rs`), not a brittle inline literal —
            // pinned end-to-end by
            // `tests::test_runner_max_tool_calls_when_cap_exhausted` and
            // `tests::test_runner_max_tool_calls_priority_over_denied`.
            if message == MAX_TOOL_CALLS_ERROR {
                // Cap reached is a terminal state, not an error: no payload, and
                // `exit::exit_code` maps it to success.
                (None, StopReason::MaxToolCalls, None)
            } else {
                (
                    None,
                    StopReason::Error,
                    Some(ErrorPayload {
                        message: sanitize_error_message(&message),
                        kind: ErrorKind::Runtime,
                    }),
                )
            }
        }
    };

    // Record each tool call and a terminal summary to the run log (best-effort).
    for record in &tool_calls {
        // **Redact FIRST, then cap** (REQ-H15c). The order is not cosmetic:
        // capping first can split a secret across the limit and ship the
        // surviving prefix in the clear. The matchers are the ones `output.rs`
        // already owns; they are never reimplemented here.
        let redacted = magi_rs::headless::output::redact_secret_patterns(&record.input.to_string());
        tracing::debug!(
            target: "magi_rs::headless",
            tool = record.name.as_str(),
            "tool input: {}",
            magi_rs::headless::output::truncate_result(&redacted, TOOL_INPUT_LOG_CAP)
        );
    }
    tracing::info!(target: "magi_rs::headless", "run complete: stop_reason={stop_reason:?}");

    RunOutcome {
        response,
        model: resolved.model,
        provider: resolved.provider,
        // Summed from every ResponseChunk::Usage the provider reported across
        // the run's turns (RunObserver::on_usage). A proactive `consult` tool
        // call's 3 magi-core LLM calls are NOT included: magi-core's
        // `LlmProvider` exposes no token counts to this layer.
        usage: Usage {
            input_tokens: usage_total.0,
            output_tokens: usage_total.1,
        },
        timings: Timings {
            total_ms,
            ttfb_ms,
            per_turn_ms: Vec::new(),
        },
        stop_reason,
        tool_calls,
        transcript,
        // The MAGI object of whichever consult succeeded (forced or a proactive
        // `--auto`+ call); `None` when none ran or it was denied by the tier. The run's
        // deadline verdict is stamped in here (SC-A04d) because only the run knows it.
        consult: apply_timeout_verdict(extract_consult_value(&calls), wiring.timeout_below_formula),
        applied_caps: AppliedCaps {
            // `resolution::resolve` cannot know the caller's wall-clock ceiling (it is
            // resolved separately, into `wiring.timeout`); stamp the effective one here
            // rather than leaving the static `None` it was built with (MS2 gate S6
            // finding 1).
            timeout_secs: wiring.timeout.map(|d| d.as_secs()),
            budget: wiring.budget,
            ..resolved.applied_caps
        },
        error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use anyhow::Result;
    use async_trait::async_trait;
    use futures::stream::{self, BoxStream};
    use magi_rs::headless::limits::FULL_AUTO_TIMEOUT_SECS;
    use serde_json::{json, Value};
    use std::collections::{BTreeSet, VecDeque};
    use std::sync::atomic::{AtomicBool, Ordering};

    use magi_core::schema::AgentName;
    use magi_core::test_support::RoutingMockProvider;
    use magi_core::verdict_markers::{VERDICT_CLOSE, VERDICT_OPEN};

    use magi_rs::headless::policy::Tier;
    use magi_rs::headless::types::{AppliedCaps, SystemPolicy};

    use crate::agent::provider::{Provider, ResponseChunk};
    use crate::tools::consult::ConsultTool;
    use crate::tools::{Tool, ToolResult};

    /// A [`ModeClassifier`] double for tests that pass an explicit mode and
    /// therefore must never reach the classification branch of
    /// [`resolve_direct_mode`] (SC-A07g): panicking here turns a silent
    /// regression (calling the classifier when it should have been skipped)
    /// into a failing test instead of a mystery extra provider call.
    struct NeverClassifier;

    #[async_trait]
    impl ModeClassifier for NeverClassifier {
        async fn classify(&self, _content: &str) -> Option<Mode> {
            panic!("explicit mode must skip classification entirely (SC-A07g)");
        }
    }

    /// The counterpart to [`NeverClassifier`]: always succeeds, so a call with no
    /// explicit/configured mode genuinely ATTEMPTS classification
    /// (`ModeResolution::classification_attempted == true`) — fix round 1,
    /// Finding 1's `endpoint_divergence` tests need this to drive the real
    /// `classification_attempted` signal through `resolve_mode_guarded`, not a
    /// hand-built `ModeResolution`.
    struct AlwaysClassifies(Mode);

    #[async_trait]
    impl ModeClassifier for AlwaysClassifies {
        async fn classify(&self, _content: &str) -> Option<Mode> {
            Some(self.0)
        }
    }

    /// A [`TimeoutDecision`] that never triggers SC-A04d's warning — the neutral
    /// filler for tests in this module that do not care about `timeout_below_
    /// formula` (fix round 1, Finding 1's `MagiRuntimeParams.timeout_decision`).
    fn neutral_timeout_decision() -> TimeoutDecision {
        TimeoutDecision::obeyed(magi_rs::magi::AGENT_TIMEOUT_SECS)
    }

    /// Sink double: records emitted messages in memory instead of touching
    /// stderr, so [`analyze_direct`]'s SC-A04d warning (fix round 2) can be
    /// observed without redirecting real process stderr — which nextest's
    /// parallel execution makes unsafe to do globally, and no dependency in
    /// this crate does per-test (B14: no new one added for this). Mirrors
    /// `agent::mode_classifier`'s private test double of the same name — one
    /// per module rather than a cross-module test-only export (B13).
    #[derive(Default)]
    struct RecordingNoticeSink {
        /// Keys already emitted, for the dedup (same contract as
        /// [`crate::agent::mode_classifier::ProcessNoticeSink`]).
        seen: Mutex<BTreeSet<&'static str>>,
        /// The messages that WERE emitted, in order.
        messages: Mutex<Vec<String>>,
    }

    use magi_rs::logging::auditor::{Audited, Auditor};

    /// The auditor the test runtimes borrow.
    ///
    /// A `OnceLock` because `MagiRuntimeParams` borrows it for `'a` and a local
    /// would not outlive the literal. Sharing one across tests is safe here and
    /// only here: in MS1 the auditor holds no registered state, so there is
    /// nothing for one test to leak into another. **If a test ever registers a
    /// secret, it needs its own.**
    fn test_auditor() -> &'static Auditor {
        static A: std::sync::OnceLock<Auditor> = std::sync::OnceLock::new();
        A.get_or_init(Auditor::new)
    }

    impl NoticeSink for RecordingNoticeSink {
        fn once(&self, key: &'static str, msg: &Audited) {
            let mut seen = self.seen.lock().unwrap_or_else(PoisonError::into_inner);
            if seen.insert(key) {
                self.record(msg);
            }
        }

        fn emit(&self, msg: &Audited) {
            self.record(msg);
        }
    }

    impl RecordingNoticeSink {
        /// Stores what the sink was handed.
        ///
        /// **Records `as_str()` of an [`Audited`], never a `&str` it was given.**
        /// The migration's whole point is that the type is what proves the line
        /// went through the auditor; a double that still took a `&str` would
        /// keep compiling against the old contract and the tests would go on
        /// asserting about a guarantee the production path no longer has.
        fn record(&self, msg: &Audited) {
            self.messages
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(msg.as_str().to_string());
        }
    }

    impl RecordingNoticeSink {
        /// Every message emitted, joined — enough for a `contains`/`is_empty`
        /// assertion without exposing the `Vec` itself.
        fn emitted(&self) -> String {
            self.messages
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .join("\n")
        }
    }

    /// A canned MAGI orchestrator whose three perspectives all approve, over a
    /// `RoutingMockProvider` (no network) — deterministic, mirrors the double used
    /// in `consult.rs` so a direct/forced consult yields a non-degraded report.
    fn canned_magi() -> Arc<Magi> {
        // magi-core 3.x reads the verdict ONLY between markers; a bare JSON (valid format in
        // 2.x) is no longer parsed and the mage counts as failed.
        fn agent_json(agent: &str) -> String {
            let verdict = format!(
                r#"{{"agent":"{agent}","verdict":"approve","confidence":0.9,"summary":"s","reasoning":"r","findings":[],"recommendation":"rec"}}"#
            );
            format!(
                "{VERDICT_OPEN}
{verdict}
{VERDICT_CLOSE}"
            )
        }
        let provider = RoutingMockProvider::new()
            .with_agent_responses(AgentName::Melchior, vec![Ok(agent_json("melchior"))])
            .with_agent_responses(AgentName::Balthasar, vec![Ok(agent_json("balthasar"))])
            .with_agent_responses(AgentName::Caspar, vec![Ok(agent_json("caspar"))]);
        Arc::new(Magi::new(Arc::new(provider)))
    }

    /// A MAGI orchestrator whose every perspective blocks for `delay` before
    /// answering — used to make a forced/direct consult deterministically exhaust
    /// its wall-clock budget (the reply content is irrelevant; the deadline fires
    /// first and the analysis task is aborted).
    /// The cancel arm of `analyze_direct` works — reachable, and now exercised.
    ///
    /// Three review rounds went into describing that arm, and two of the three descriptions were
    /// wrong, because `run_consult` (its only caller) creates a token it never cancels. Melchior
    /// offered two ways out in S4 Loop 2: delete the arm, or drive it from a test double. Driving
    /// it is the better trade — the arm is a real safety path that `ConsultTool::execute` relies
    /// on in the same shape, and deleting a working guard because today's single caller happens
    /// not to use it trades a defence for a line count.
    ///
    /// **Mutation-verified (B16):** remove the `cancel.cancelled()` arm and this hangs on the
    /// hour-long analysis instead of returning; drop the `.abort()` inside it and the spawned
    /// task outlives the call.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_cancel_arm_stops_an_analysis_that_would_otherwise_run_for_an_hour() {
        let magi = slow_magi(Duration::from_secs(3_600));
        let cfg = MagiConfig::default();
        let sink = RecordingNoticeSink::default();
        let runtime = MagiRuntimeParams {
            kind: ProviderKind::OpenAiCompat,
            classifier: &NeverClassifier,
            configured_mode: None,
            untrusted_content: false,
            magi_config: &cfg,
            timeout_decision: neutral_timeout_decision(),
            notice_sink: &sink,
            auditor: test_auditor(),
            structured_verdicts: StructuredVerdicts::Omit,
            budget: BudgetTelemetry::default(),
        };

        let cancel = CancellationToken::new();
        // Cancelled BEFORE the call, so the arm is ready on the first poll: no sleep, no
        // tolerance, and no dependence on how loaded the box is (R-R05).
        cancel.cancel();

        // `None` timeout: the deadline arm parks forever, so nothing but cancellation can end
        // this — which is what makes the result attributable.
        let result = analyze_direct(
            &magi,
            "should we migrate X to Y?",
            &cancel,
            None,
            Some(Mode::Analysis),
            &runtime,
        )
        .await;

        assert!(
            matches!(result, Err(ConsultRunError::Timeout)),
            "a cancelled run must end promptly with the typed stop, not wait out the analysis"
        );
    }

    fn slow_magi(delay: Duration) -> Arc<Magi> {
        use magi_core::error::ProviderError;
        use magi_core::provider::{CompletionConfig, LlmProvider};

        /// An [`LlmProvider`] that sleeps `delay` on every completion.
        struct SlowLlm {
            /// How long each `complete` call blocks before returning.
            delay: Duration,
        }

        #[async_trait]
        impl LlmProvider for SlowLlm {
            async fn complete(
                &self,
                _system_prompt: &str,
                _user_prompt: &str,
                _config: &CompletionConfig,
            ) -> Result<magi_core::provider::Completion, ProviderError> {
                tokio::time::sleep(self.delay).await;
                Ok(magi_core::provider::Completion::new(format!(
                    "{VERDICT_OPEN}
{}
{VERDICT_CLOSE}",
                    r#"{"agent":"melchior","verdict":"approve","confidence":0.9,"summary":"s","reasoning":"r","findings":[],"recommendation":"rec"}"#
                )))
            }
            fn name(&self) -> &str {
                "slow"
            }
            fn model(&self) -> &str {
                "slow-model"
            }
        }

        Arc::new(Magi::new(Arc::new(SlowLlm { delay })))
    }

    /// A MAGI orchestrator whose every perspective blocks for `delay`, and whose
    /// `complete` future *flips `dropped` to `true` when it is dropped* rather
    /// than when it returns. Used to distinguish a genuinely **aborted** spawned
    /// task (its future tree, including the pending `sleep`, is dropped
    /// immediately) from one that was merely **detached** (a bare dropped
    /// `JoinHandle` keeps the task running to completion, so `dropped` would stay
    /// `false` at assertion time).
    fn slow_droppy_magi(
        delay: Duration,
        entered: Arc<AtomicBool>,
        dropped: Arc<AtomicBool>,
    ) -> Arc<Magi> {
        use magi_core::error::ProviderError;
        use magi_core::provider::{CompletionConfig, LlmProvider};

        /// Sets its `flag` to `true` on `Drop`, regardless of whether the
        /// surrounding future ever ran to completion.
        struct DropFlag(Arc<AtomicBool>);

        impl Drop for DropFlag {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        /// An [`LlmProvider`] that sleeps `delay` on every completion, holding a
        /// [`DropFlag`] live across the sleep so aborting the enclosing task
        /// (dropping its future tree) is observable.
        struct SlowDroppyLlm {
            /// How long each `complete` call blocks before returning.
            delay: Duration,
            /// Flipped to `true` as soon as a `complete` call is entered.
            ///
            /// Exists so the test can wait on the **condition** "the spawned
            /// analysis has actually started" instead of guessing a duration
            /// that is long enough for it. Without this signal a caller can
            /// only poll the outer future for an arbitrary while and hope;
            /// under CPU contention that guess is what makes the test flaky,
            /// because dropping before `complete` is entered means no
            /// [`DropFlag`] was ever constructed and `dropped` can never flip.
            entered: Arc<AtomicBool>,
            /// Flipped to `true` when an in-flight `complete` future is dropped.
            dropped: Arc<AtomicBool>,
        }

        #[async_trait]
        impl LlmProvider for SlowDroppyLlm {
            async fn complete(
                &self,
                _system_prompt: &str,
                _user_prompt: &str,
                _config: &CompletionConfig,
            ) -> Result<magi_core::provider::Completion, ProviderError> {
                let _spy = DropFlag(self.dropped.clone());
                // AFTER the guard exists, so observing `entered` guarantees a
                // `DropFlag` is live and an abort is therefore observable.
                self.entered.store(true, Ordering::SeqCst);
                tokio::time::sleep(self.delay).await;
                Ok(magi_core::provider::Completion::new(format!(
                    "{VERDICT_OPEN}
{}
{VERDICT_CLOSE}",
                    r#"{"agent":"melchior","verdict":"approve","confidence":0.9,"summary":"s","reasoning":"r","findings":[],"recommendation":"rec"}"#
                )))
            }
            fn name(&self) -> &str {
                "slow-droppy"
            }
            fn model(&self) -> &str {
                "slow-droppy-model"
            }
        }

        Arc::new(Magi::new(Arc::new(SlowDroppyLlm {
            delay,
            entered,
            dropped,
        })))
    }

    /// A [`Resolved`] stub with a **forced** consult (`consult = Some(true)`).
    fn forced_resolved() -> Resolved {
        Resolved {
            consult: Some(true),
            ..resolved_stub()
        }
    }

    /// One scripted provider turn.
    enum Turn {
        /// Emit a single assistant `ToolUse` block (triggers a tool call).
        Tool {
            id: String,
            name: String,
            input: Value,
        },
        /// Like `Tool`, plus a `ResponseChunk::Usage` chunk — exercises
        /// `RunObserver::on_usage` accumulation on a NON-terminal turn.
        ToolWithUsage {
            id: String,
            name: String,
            input: Value,
            input_tokens: u64,
            output_tokens: u64,
        },
        /// Emit streamed text plus a matching `MessageDone` (terminal turn).
        Text(String),
        /// Like `Text`, plus a `ResponseChunk::Usage` chunk — exercises
        /// `RunObserver::on_usage` accumulation across turns.
        TextWithUsage {
            text: String,
            input_tokens: u64,
            output_tokens: u64,
        },
        /// Emit a `MessageDone` with empty content and ZERO `TextDelta` blocks
        /// (a terminal turn with an empty response, for REQ-H23b).
        Empty,
        /// Fail the provider call (surfaces as a run error).
        Fail,
    }

    /// A provider that replays a fixed script of turns, one per `stream_messages`
    /// call. When the script is exhausted it returns an empty terminal turn so a
    /// loop always terminates.
    struct ScriptedProvider {
        turns: Mutex<VecDeque<Turn>>,
    }

    impl ScriptedProvider {
        fn new(turns: Vec<Turn>) -> Arc<Self> {
            Arc::new(Self {
                turns: Mutex::new(turns.into_iter().collect()),
            })
        }
    }

    #[async_trait]
    impl Provider for ScriptedProvider {
        async fn stream_messages(
            &self,
            _messages: &[Message],
            _tools: &[Box<dyn Tool>],
            _system: Option<&str>,
        ) -> Result<BoxStream<'static, Result<ResponseChunk>>> {
            let turn = self
                .turns
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .pop_front();
            match turn {
                Some(Turn::Tool { id, name, input }) => {
                    let msg = Message {
                        role: Role::Assistant,
                        content: vec![Content::ToolUse { id, name, input }],
                    };
                    Ok(Box::pin(stream::iter(vec![Ok(
                        ResponseChunk::MessageDone(msg),
                    )])))
                }
                Some(Turn::ToolWithUsage {
                    id,
                    name,
                    input,
                    input_tokens,
                    output_tokens,
                }) => {
                    let msg = Message {
                        role: Role::Assistant,
                        content: vec![Content::ToolUse { id, name, input }],
                    };
                    Ok(Box::pin(stream::iter(vec![
                        Ok(ResponseChunk::Usage {
                            input_tokens,
                            output_tokens,
                        }),
                        Ok(ResponseChunk::MessageDone(msg)),
                    ])))
                }
                Some(Turn::Text(text)) => Ok(Box::pin(stream::iter(vec![
                    Ok(ResponseChunk::TextDelta(text.clone())),
                    Ok(ResponseChunk::MessageDone(Message::assistant(&text))),
                ]))),
                Some(Turn::TextWithUsage {
                    text,
                    input_tokens,
                    output_tokens,
                }) => Ok(Box::pin(stream::iter(vec![
                    Ok(ResponseChunk::TextDelta(text.clone())),
                    Ok(ResponseChunk::Usage {
                        input_tokens,
                        output_tokens,
                    }),
                    Ok(ResponseChunk::MessageDone(Message::assistant(&text))),
                ]))),
                Some(Turn::Empty) | None => Ok(Box::pin(stream::iter(vec![Ok(
                    ResponseChunk::MessageDone(Message {
                        role: Role::Assistant,
                        content: vec![],
                    }),
                )]))),
                Some(Turn::Fail) => Err(anyhow::anyhow!("provider network failure")),
            }
        }
    }

    /// A provider that records the `system` argument of its one expected call
    /// and returns a fixed terminal turn — used to prove `run_query` forwards
    /// the resolved `SystemPolicy` text into `AgentRunConfig::system`.
    struct SystemCapturingProvider {
        seen: Mutex<Vec<Option<String>>>,
    }

    impl SystemCapturingProvider {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                seen: Mutex::new(Vec::new()),
            })
        }
    }

    #[async_trait]
    impl Provider for SystemCapturingProvider {
        async fn stream_messages(
            &self,
            _messages: &[Message],
            _tools: &[Box<dyn Tool>],
            system: Option<&str>,
        ) -> Result<BoxStream<'static, Result<ResponseChunk>>> {
            self.seen
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(system.map(str::to_string));
            Ok(Box::pin(stream::iter(vec![Ok(
                ResponseChunk::MessageDone(Message::assistant("ok")),
            )])))
        }
    }

    /// The runner sets `AgentRunConfig::system` from `resolved.system.text()` —
    /// a `SystemPolicy::CallerOverride` reaches the provider verbatim.
    #[tokio::test]
    async fn test_run_query_sets_system_from_caller_override_policy() {
        let provider = SystemCapturingProvider::new();
        let mut agent = Agent::new(provider.clone());
        let resolved = Resolved {
            system: SystemPolicy::CallerOverride("caller system prompt".to_string()),
            ..resolved_stub()
        };
        let policy = Policy::new(Tier::Default, 15, None);
        run_query(resolved, policy, &mut agent, "prompt", &wiring(None)).await;
        assert_eq!(
            provider.seen.lock().unwrap().as_slice(),
            &[Some("caller system prompt".to_string())]
        );
    }

    /// An empty `SystemPolicy::Operator("")` (the `resolved_stub()` default)
    /// must reach the provider as `None`, not `Some("")`.
    #[tokio::test]
    async fn test_run_query_sends_no_system_when_operator_default_is_empty() {
        let provider = SystemCapturingProvider::new();
        let mut agent = Agent::new(provider.clone());
        let policy = Policy::new(Tier::Default, 15, None);
        run_query(resolved_stub(), policy, &mut agent, "prompt", &wiring(None)).await;
        assert_eq!(provider.seen.lock().unwrap().as_slice(), &[None]);
    }

    /// A tool that always succeeds, echoing its name — enough to exercise tier
    /// authorization and the tool-call record (hard barriers are tested in each
    /// tool's own module).
    struct EchoTool {
        name: String,
    }

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            "echo tool for runner tests"
        }
        fn input_schema(&self) -> Value {
            json!({"type": "object", "properties": {}})
        }
        async fn execute(&self, _args: Value, _cancel: &CancellationToken) -> ToolResult<Value> {
            Ok(json!(format!("{}-ran", self.name)))
        }
    }

    /// Builds a `Turn::Tool` with an empty input object.
    fn tool_turn(id: &str, name: &str) -> Turn {
        Turn::Tool {
            id: id.to_string(),
            name: name.to_string(),
            input: json!({}),
        }
    }

    /// Registers an [`EchoTool`] named `name` on `agent`.
    fn register_echo(agent: &mut Agent, name: &str) {
        agent.register_tool(Box::new(EchoTool {
            name: name.to_string(),
        }));
    }

    /// A `Resolved` stub with distinguishable metadata for output assertions.
    fn resolved_stub() -> Resolved {
        Resolved {
            model: "test-model".to_string(),
            provider: "test-provider".to_string(),
            system: SystemPolicy::Operator(String::new()),
            max_tool_calls: 15,
            consult: None,
            applied_caps: AppliedCaps {
                max_tool_calls: 15,
                max_tool_calls_clamped: false,
                timeout_secs: None,
                system_override_applied: false,
                budget: BudgetTelemetry::default(),
            },
        }
    }

    /// A [`RunWiring`] with the built-in gate, no configured mode and the guard off — what an
    /// unconfigured `magi.toml` resolves to — so a test that is not about the operator's
    /// configuration reads exactly as it did before that configuration existed.
    ///
    /// One helper rather than a literal per call site: [`RunWiring`] gains fields as the
    /// milestone's per-run telemetry grows, and 25 literals would each have to be revisited.
    fn wiring(timeout: Option<Duration>) -> RunWiring {
        RunWiring {
            timeout,
            autonomous: crate::AutonomousRunConfig::from_magi_config(
                &crate::config::MagiConfig::default(),
            ),
            timeout_below_formula: false,
            budget: BudgetTelemetry::default(),
        }
    }

    /// Same, but with the operator's `magi.toml` in force — the shape a real `magi query` run
    /// gets, and the one that makes `[magi.complexity]`/`[magi].untrusted_content` observable
    /// from this surface.
    fn wiring_from_toml(toml: &str) -> RunWiring {
        RunWiring {
            timeout: None,
            autonomous: crate::AutonomousRunConfig::from_magi_config(
                &crate::config::MagiConfig::from_toml_str(toml).expect("fixture parses"),
            ),
            timeout_below_formula: false,
            budget: BudgetTelemetry::default(),
        }
    }

    /// `n` identical `edit` tool calls (same name + input) with distinct ids,
    /// to exercise the repetitive-call soft guard.
    fn identical_edit_turns(n: usize) -> Vec<Turn> {
        (0..n)
            .map(|i| Turn::Tool {
                id: format!("r{i}"),
                name: "edit".to_string(),
                input: json!({"same": "input"}),
            })
            .collect()
    }

    /// One notice line, formatted and redacted the way the drain loop emits it.
    ///
    /// This covers the FORMATTING half only, and says so because the name it
    /// carried before did not: "a notice reaches the operator" is a claim about
    /// the drain loop, and deleting the `StreamPiece::Notice` arm there leaves
    /// this test green. The wiring's guardian is S9's fourth assertion in
    /// `smoke/`, which runs the release binary against a failing embedder and
    /// looks for the notice in what the process actually printed.
    ///
    /// The drain loop consumed every `StreamPiece` and inspected only
    /// `Content`, for time-to-first-byte. `StreamPiece::Notice` was dropped on
    /// the floor, so REQ-29's second half held in the TUI and nowhere else: an
    /// operator running `magi query` with a failing embedder watched memories
    /// pile up unembedded and was told nothing at all. The smoke harness found
    /// it by asking for the notice and never seeing one.
    ///
    /// The text is foreign -- it carries whatever the failing subsystem said,
    /// which can embed a URL with credentials in it -- so it is redacted on
    /// the way out, exactly like every other foreign string this binary
    /// prints.
    #[test]
    fn a_notice_line_is_prefixed_and_its_authority_redacted() {
        let line =
            notice_line("memory: context assembly failed (POST https://u:pw@host/v1/embeddings)");
        assert!(line.starts_with(NOTICE_PREFIX), "got: {line}");
        assert!(!line.contains("pw"), "the credential survived: {line}");
        assert!(line.contains("context assembly failed"), "got: {line}");
    }

    /// The runner sums `RunObserver::on_usage` across every turn into
    /// `RunOutcome.usage` - a non-terminal tool turn reporting (10, 2) plus a
    /// terminal turn reporting (5, 3) must total (15, 5), not just the last turn.
    #[tokio::test]
    async fn test_run_query_sums_usage_across_turns_into_run_outcome() {
        let provider = ScriptedProvider::new(vec![
            Turn::ToolWithUsage {
                id: "t1".to_string(),
                name: "edit".to_string(),
                input: json!({}),
                input_tokens: 10,
                output_tokens: 2,
            },
            Turn::TextWithUsage {
                text: "answered".to_string(),
                input_tokens: 5,
                output_tokens: 3,
            },
        ]);
        let mut agent = Agent::new(provider);
        register_echo(&mut agent, "edit");

        let policy = Policy::new(Tier::Auto, 15, None);
        let outcome = run_query(resolved_stub(), policy, &mut agent, "prompt", &wiring(None)).await;

        assert_eq!(outcome.usage.input_tokens, 15);
        assert_eq!(outcome.usage.output_tokens, 5);
    }

    /// A run with no usage-emitting turns (the common case for a backend that
    /// omits usage) reports `{0, 0}` rather than a fabricated total.
    #[tokio::test]
    async fn test_run_query_usage_defaults_to_zero_when_provider_reports_none() {
        let provider = ScriptedProvider::new(vec![Turn::Text("hi".to_string())]);
        let mut agent = Agent::new(provider);
        let policy = Policy::new(Tier::Default, 15, None);
        let outcome = run_query(resolved_stub(), policy, &mut agent, "prompt", &wiring(None)).await;
        assert_eq!(outcome.usage.input_tokens, 0);
        assert_eq!(outcome.usage.output_tokens, 0);
    }

    /// MS2 gate S6 finding 1: `applied_caps.timeout_secs` must report the wall-clock
    /// ceiling the runner actually enforced, not a permanent `None`. `Resolved::
    /// applied_caps.timeout_secs` starts `None` (resolution.rs cannot know the caller's
    /// ceiling); `run_query` receives the effective one via `RunWiring::timeout` and
    /// must copy it into the outcome — a consumer reading `null` otherwise cannot tell
    /// "no cap" from "cap applied but not reported".
    #[tokio::test]
    async fn test_run_query_reports_the_effective_timeout_in_applied_caps() {
        let provider = ScriptedProvider::new(vec![Turn::Text("hi".to_string())]);
        let mut agent = Agent::new(provider);
        let policy = Policy::new(Tier::Default, 15, None);
        let outcome = run_query(
            resolved_stub(),
            policy,
            &mut agent,
            "prompt",
            &wiring(Some(Duration::from_secs(42))),
        )
        .await;
        assert_eq!(
            outcome.applied_caps.timeout_secs,
            Some(42),
            "a run with a 42s wall-clock ceiling must surface 42 in \
             applied_caps.timeout_secs, not null"
        );
    }

    /// The other half of the same contract: a run with no wall-clock ceiling
    /// (`wiring.timeout == None`) reports `applied_caps.timeout_secs == None`, which
    /// is how "no cap" stays distinguishable from the bug the sibling test pins.
    #[tokio::test]
    async fn test_run_query_reports_no_timeout_when_none_is_set() {
        let provider = ScriptedProvider::new(vec![Turn::Text("hi".to_string())]);
        let mut agent = Agent::new(provider);
        let policy = Policy::new(Tier::Default, 15, None);
        let outcome = run_query(resolved_stub(), policy, &mut agent, "prompt", &wiring(None)).await;
        assert_eq!(outcome.applied_caps.timeout_secs, None);
    }

    /// The placeholder from `resolution::resolve` must be OVERRIDDEN, not published.
    /// A defaulted budget in the envelope is not a missing value but a FALSE one:
    /// `operation_budget_secs: 0` and `floor_activation_threshold_secs: 0` read as real
    /// numbers, and a consumer auto-remediating on that threshold would retry with
    /// `--timeout 0`. Zero is indistinguishable from a genuine reading by shape, which is
    /// why this is a test rather than a review item.
    #[tokio::test]
    async fn run_query_publishes_the_wirings_budget_not_the_placeholder() {
        let expected = BudgetTelemetry {
            operation_budget_secs: 149,
            ceiling_floored: false,
            floor_activation_threshold_secs: 114,
            max_rotations_effective: 2,
            ceiling_above_sanity: false,
        };
        let provider = ScriptedProvider::new(vec![Turn::Text("hi".to_string())]);
        let mut agent = Agent::new(provider);
        let policy = Policy::new(Tier::Default, 15, None);
        let mut w = wiring(None);
        w.budget = expected;
        // `resolved_stub()` carries `BudgetTelemetry::default()`, exactly as `resolution::resolve`
        // produces it — so a run that forgets to stamp fails here rather than in production.
        let outcome = run_query(resolved_stub(), policy, &mut agent, "prompt", &w).await;
        assert_eq!(outcome.applied_caps.budget, expected);
    }

    /// Default tier denies `edit`; the agent still answers, so the run is `Done`
    /// (the denial is recorded, exit 0) — MS2.md Task 3 Step 1 / REQ-H23b.
    #[tokio::test]
    async fn test_runner_default_denies_edit_and_reports_without_aborting() {
        let provider = ScriptedProvider::new(vec![
            tool_turn("t1", "edit"),
            Turn::Text("answered anyway".to_string()),
        ]);
        let mut agent = Agent::new(provider);
        register_echo(&mut agent, "edit");

        let policy = Policy::new(Tier::Default, 15, None);
        let outcome = run_query(resolved_stub(), policy, &mut agent, "prompt", &wiring(None)).await;

        assert!(
            outcome.tool_calls.iter().any(|t| t.name == "edit" && !t.ok),
            "edit must be recorded as denied (ok=false) in default tier"
        );
        assert_eq!(
            outcome.stop_reason,
            StopReason::Done,
            "the agent answered despite the denial ⇒ Done (exit 0)"
        );
        assert_eq!(outcome.response.as_deref(), Some("answered anyway"));
        assert_eq!(outcome.model, "test-model");
        assert_eq!(outcome.provider, "test-provider");
    }

    /// SC-A20c: a complexity-gate veto of an autonomous `consult` is not an error, so the
    /// headless exit code stays the SUCCESS one — a CI pipeline reading a non-zero exit here
    /// would wrongly read a normal, cost-saving veto as a run failure.
    ///
    /// The model-issued `consult` ToolUse reaches [`Agent::dispatch_consult_through_gate`]
    /// directly (it never goes through [`Agent::authorize_and_execute_tool`], so the tier does
    /// not gate it — only the complexity gate does), so `Tier::Default` already exercises the
    /// real production path; no elevated tier is needed to observe the veto. The empty `query`
    /// left by `tool_turn`'s default `json!({})` input is unconditionally below every
    /// `[magi.complexity]` threshold (REQ-A20 requires every threshold `> 1`), so the veto is
    /// deterministic without needing to know the exact configured number.
    #[tokio::test]
    async fn test_gate_vetoed_consult_still_exits_success() {
        let provider = ScriptedProvider::new(vec![
            tool_turn("t1", "consult"),
            Turn::Text("answered without consensus".to_string()),
        ]);
        let mut agent = Agent::new(provider);
        register_echo(&mut agent, "consult");

        let policy = Policy::new(Tier::Default, 15, None);
        let outcome = run_query(resolved_stub(), policy, &mut agent, "prompt", &wiring(None)).await;

        assert_eq!(
            outcome.stop_reason,
            StopReason::Done,
            "the veto returns an ordinary ToolResult, not an error: the turn completes normally"
        );
        assert!(outcome.error.is_none(), "a veto is never an error payload");
        let response = outcome.response.as_deref().unwrap_or_default();
        assert!(
            response.starts_with("answered without consensus"),
            "the agent's own text must survive: {response}"
        );
        assert!(
            response.contains("NO CONSENSUS"),
            "SC-A20k: the answer must carry the no-consensus mark: {response}"
        );
        assert_eq!(
            crate::exit_code_for_outcome(&outcome),
            0,
            "a vetoed autonomous consult must not turn into a non-zero process exit code"
        );
    }

    /// Loop 1, F1 / SC-A07r: `[magi].untrusted_content = true` must fail a SELF-ROUTED consult
    /// closed on `magi query`'s tool loop, exactly as it already did on the direct
    /// `magi consult` route.
    ///
    /// This is the surface where the guard was inert: the runner built its `AgentRunConfig`
    /// with `..AgentRunConfig::default()`, so `mode_config.untrusted_content` was permanently
    /// `false` no matter what the operator declared. An operator who configured the guard
    /// specifically to stop hostile content from being classified got no protection here.
    #[tokio::test]
    async fn untrusted_content_from_the_toml_fails_a_self_routed_consult_closed() {
        let provider = ScriptedProvider::new(vec![
            tool_turn("t1", "consult"),
            Turn::Text("should never be reached".to_string()),
        ]);
        let mut agent = Agent::new(provider);
        register_echo(&mut agent, "consult");

        let policy = Policy::new(Tier::Default, 15, None);
        let wiring = wiring_from_toml("[magi]\nuntrusted_content = true\n");
        let outcome = run_query(resolved_stub(), policy, &mut agent, "prompt", &wiring).await;

        assert_eq!(
            outcome.stop_reason,
            StopReason::Error,
            "the guard must abort the turn, not let it classify hostile content"
        );
        let err = outcome
            .error
            .expect("a fail-closed run carries an error payload");
        assert_eq!(err.kind, ErrorKind::Runtime);
        assert!(
            err.message.contains("untrusted content"),
            "the message must name the guard so the operator knows what to declare: {}",
            err.message
        );
    }

    /// Loop 1, F1 / SC-A20d: the per-mode `0` off-switch must reach this surface.
    ///
    /// `analysis = 0` disables the veto for the default mode, so the same trivial self-routed
    /// consult that `test_gate_vetoed_consult_still_exits_success` sees vetoed is DISPATCHED
    /// here. With the thresholds stuck at the built-ins, the two runs would be identical —
    /// which is precisely the silence this finding is about: the operator turned the gate off
    /// for a mode and nothing changed.
    #[tokio::test]
    async fn a_configured_zero_threshold_lets_a_trivial_self_routed_consult_through() {
        let provider = ScriptedProvider::new(vec![
            tool_turn("t1", "consult"),
            Turn::Text("answered".to_string()),
        ]);
        let mut agent = Agent::new(provider);
        register_echo(&mut agent, "consult");

        // `Tier::Auto`, not `Tier::Default` — the tier is scaffolding here and the wrong one was
        // standing between this test and its own subject. `Default` auto-approves only the
        // read-only set, so `consult` was refused by the TIER after clearing the complexity
        // gate: the test could observe the gate's decision and never the consult itself, and its
        // name promised a consult that got "through" when nothing ran. `Auto` approves every
        // registered tool, so the whole path is exercised and the name becomes true.
        let policy = Policy::new(Tier::Auto, 15, None);
        let wiring = wiring_from_toml("[magi.complexity]\nanalysis = 0\n");
        let outcome = run_query(resolved_stub(), policy, &mut agent, "prompt", &wiring).await;

        assert_eq!(outcome.stop_reason, StopReason::Done);

        // Now assertable end to end: the consult RAN and succeeded.
        let consult = outcome
            .tool_calls
            .iter()
            .find(|rec| rec.name == "consult")
            .expect("the model asked for a consult, so a record exists either way");
        assert!(
            consult.ok,
            "with the veto disabled AND the tier approving, the consult must actually run: {}",
            consult.result
        );

        // **The discriminating signal is the gate TELEMETRY, and nothing in `tool_calls` is.**
        // Two earlier drafts of this precondition were themselves vacuous, and mutation testing
        // — removing `analysis = 0` so the built-in threshold vetoes a six-character prompt —
        // is what exposed both. `name == "consult"` is recorded whether the gate dispatched or
        // vetoed. The `result` does not help either: `Tier::Default` does not authorize
        // `consult`, so the tier refuses it FIRST and the stored text is always the tier
        // denial — the veto never reaches the record to be looked for.
        //
        // Finding that out is also what corrected the fixture: under `Tier::Default` the consult
        // only ever got past the complexity gate and was then refused, so the test could not
        // observe the thing its name promises. `Tier::Auto` approves every registered tool, the
        // whole path runs, and the `ok` assertion above became possible.
        //
        // `on_gate_evaluation` writes "veto" or "dispatch" per evaluation (SC-A20h) — the one
        // place the two outcomes differ, and what the neighbouring test reads for the
        // opposite case.
        let gate_lines = wiring.autonomous.drain_telemetry();
        assert_eq!(
            gate_lines.len(),
            1,
            "one evaluation, one line: {gate_lines:?}"
        );
        assert!(
            gate_lines[0].contains("dispatch"),
            "with `analysis = 0` the veto is disabled for that mode, so the gate records a dispatch: {}",
            gate_lines[0]
        );

        let response = outcome.response.as_deref().unwrap_or_default();
        assert!(
            !response.contains("NO CONSENSUS"),
            "the consult ran, so the answer must NOT carry the no-consensus mark: {response}"
        );
    }

    /// Loop 1, F1 / SC-A20h: every gate evaluation on this surface is recorded, so the
    /// thresholds can be calibrated from data instead of guessed again.
    #[tokio::test]
    async fn every_gate_evaluation_on_this_surface_is_recorded() {
        let provider = ScriptedProvider::new(vec![
            tool_turn("t1", "consult"),
            Turn::Text("answered".to_string()),
        ]);
        let mut agent = Agent::new(provider);
        register_echo(&mut agent, "consult");

        let policy = Policy::new(Tier::Default, 15, None);
        let wiring = wiring(None);
        let _ = run_query(resolved_stub(), policy, &mut agent, "prompt", &wiring).await;

        let lines = wiring.autonomous.drain_telemetry();
        assert_eq!(lines.len(), 1, "one evaluation, one line: {lines:?}");
        assert!(
            lines[0].contains("analysis") && lines[0].contains("veto"),
            "the line names the mode and the outcome: {}",
            lines[0]
        );
        assert!(
            lines[0].contains(&magi_rs::magi::GATE_ANALYSIS.to_string()),
            "the APPLIED threshold travels with the line, or it calibrates nothing: {}",
            lines[0]
        );
    }

    /// Loop 1, F7 / SC-A04d: a below-formula deadline reaches THIS run's JSON, not only
    /// stderr.
    ///
    /// Whoever sets an explicit `--timeout` is running a pipeline — exactly the consumer that
    /// never sees stderr of a process that already exited. The tool-loop consult cannot compute
    /// this itself (no `--timeout` concept reaches it), so the runner, which owns the deadline,
    /// corrects the one field the tool hardcodes to `false`.
    /// Drives one forced consult through a REAL [`ConsultTool`] — so the object under
    /// assertion is the one `report_to_consult_json` actually produces, not a stand-in — and
    /// returns its `consult` object.
    async fn forced_consult_json(timeout_below_formula: bool) -> Value {
        let provider = ScriptedProvider::new(vec![Turn::Text("answered".to_string())]);
        let mut agent = Agent::new(provider);
        agent.register_tool(Box::new(ConsultTool::new(canned_magi(), true)));

        let policy = Policy::new(Tier::Auto, 15, None);
        let w = RunWiring {
            timeout: None,
            autonomous: crate::AutonomousRunConfig::from_magi_config(&MagiConfig::default()),
            timeout_below_formula,
            budget: BudgetTelemetry::default(),
        };
        run_query(forced_resolved(), policy, &mut agent, "decide X vs Y", &w)
            .await
            .consult
            .expect("a forced consult in --auto populates the consult object")
    }

    #[tokio::test]
    async fn a_below_formula_deadline_reaches_the_consult_json_of_this_run() {
        let consult = forced_consult_json(true).await;
        assert_eq!(
            consult.get("timeout_below_formula"),
            Some(&Value::Bool(true)),
            "the run's own deadline verdict must travel in the run's own JSON: {consult}"
        );
    }

    /// The companion: a healthy deadline leaves the flag alone, so it means what it says
    /// rather than being set by the mere presence of a consult.
    #[tokio::test]
    async fn a_healthy_deadline_leaves_the_consult_json_flag_false() {
        let consult = forced_consult_json(false).await;
        assert_eq!(
            consult.get("timeout_below_formula"),
            Some(&Value::Bool(false))
        );
    }

    /// `--auto` auto-approves `edit` and `bash`; both execute (hard barriers
    /// still apply inside the real tools) — REQ-H07.
    #[tokio::test]
    async fn test_runner_auto_approves_edit_and_bash() {
        let provider = ScriptedProvider::new(vec![
            tool_turn("t1", "edit"),
            tool_turn("t2", "bash"),
            Turn::Text("done".to_string()),
        ]);
        let mut agent = Agent::new(provider);
        register_echo(&mut agent, "edit");
        register_echo(&mut agent, "bash");

        let policy = Policy::new(Tier::Auto, 15, None);
        let outcome = run_query(resolved_stub(), policy, &mut agent, "prompt", &wiring(None)).await;

        assert!(outcome.tool_calls.iter().any(|t| t.name == "edit" && t.ok));
        assert!(outcome.tool_calls.iter().any(|t| t.name == "bash" && t.ok));
        assert_eq!(outcome.stop_reason, StopReason::Done);
    }

    /// A tier denial AND an empty final turn (zero `TextDelta` blocks) ⇒
    /// `Denied` (→ exit 3) — REQ-H23b.
    #[tokio::test]
    async fn test_runner_denied_when_response_empty_and_tool_denied() {
        let provider = ScriptedProvider::new(vec![tool_turn("t1", "edit"), Turn::Empty]);
        let mut agent = Agent::new(provider);
        register_echo(&mut agent, "edit");

        let policy = Policy::new(Tier::Default, 15, None);
        let outcome = run_query(resolved_stub(), policy, &mut agent, "prompt", &wiring(None)).await;

        assert!(
            outcome.tool_calls.iter().any(|t| t.name == "edit" && !t.ok),
            "edit denied by the default tier"
        );
        assert_eq!(
            outcome.stop_reason,
            StopReason::Denied,
            "denial + empty final turn ⇒ Denied"
        );
    }

    /// Exhausting the tool-call cap ⇒ `MaxToolCalls` with NO error payload
    /// (a terminal state, exit 0) — REQ-H14.
    #[tokio::test]
    async fn test_runner_max_tool_calls_when_cap_exhausted() {
        // Cap 2, but the provider keeps requesting a tool ⇒ the 3rd trips the cap.
        let provider = ScriptedProvider::new(vec![
            tool_turn("t1", "ls"),
            tool_turn("t2", "ls"),
            tool_turn("t3", "ls"),
            tool_turn("t4", "ls"),
        ]);
        let mut agent = Agent::new(provider);
        register_echo(&mut agent, "ls");

        let policy = Policy::new(Tier::Auto, 2, None);
        let outcome = run_query(resolved_stub(), policy, &mut agent, "prompt", &wiring(None)).await;

        assert_eq!(outcome.stop_reason, StopReason::MaxToolCalls);
        assert!(
            outcome.error.is_none(),
            "reaching the cap is a terminal state, not an error"
        );
    }

    /// A provider failure ⇒ `Error` with a populated (sanitized) error payload.
    #[tokio::test]
    async fn test_runner_provider_error_maps_to_error() {
        let provider = ScriptedProvider::new(vec![Turn::Fail]);
        let mut agent = Agent::new(provider);

        let policy = Policy::new(Tier::Auto, 15, None);
        let outcome = run_query(resolved_stub(), policy, &mut agent, "prompt", &wiring(None)).await;

        assert_eq!(outcome.stop_reason, StopReason::Error);
        assert!(outcome.error.is_some(), "error payload must be populated");
        assert_eq!(outcome.response, None);
    }

    /// Priority `Error > Denied`: a denial occurs, then the provider fails ⇒ the
    /// terminal state is `Error`, not `Denied` — REQ-H14 priority.
    #[tokio::test]
    async fn test_runner_error_priority_over_denied() {
        let provider = ScriptedProvider::new(vec![tool_turn("t1", "edit"), Turn::Fail]);
        let mut agent = Agent::new(provider);
        register_echo(&mut agent, "edit");

        let policy = Policy::new(Tier::Default, 15, None);
        let outcome = run_query(resolved_stub(), policy, &mut agent, "prompt", &wiring(None)).await;

        assert!(
            outcome.tool_calls.iter().any(|t| t.name == "edit" && !t.ok),
            "edit was denied before the provider failed"
        );
        assert_eq!(
            outcome.stop_reason,
            StopReason::Error,
            "a run error dominates a tier denial (Error > Denied)"
        );
    }

    /// Priority `MaxToolCalls > Denied`: repeated denied tools still increment the
    /// call count, so the cap trips first ⇒ `MaxToolCalls` (not `Denied`).
    #[tokio::test]
    async fn test_runner_max_tool_calls_priority_over_denied() {
        let provider = ScriptedProvider::new(vec![
            tool_turn("t1", "edit"),
            tool_turn("t2", "edit"),
            tool_turn("t3", "edit"),
        ]);
        let mut agent = Agent::new(provider);
        register_echo(&mut agent, "edit");

        // Default denies edit; cap 2 ⇒ the 3rd (denied) call trips the cap.
        let policy = Policy::new(Tier::Default, 2, None);
        let outcome = run_query(resolved_stub(), policy, &mut agent, "prompt", &wiring(None)).await;

        assert_eq!(
            outcome.stop_reason,
            StopReason::MaxToolCalls,
            "the cap dominates the tier denials (MaxToolCalls > Denied)"
        );
    }

    /// Under `--full-auto` the repetitive-call soft guard is silenced: five
    /// identical calls do NOT abort, and the run reaches its terminal text ⇒
    /// `Done` — REQ-H08.
    #[tokio::test]
    async fn test_runner_full_auto_allows_repeated_identical_calls() {
        let mut turns = identical_edit_turns(5);
        turns.push(Turn::Text("done".to_string()));
        let provider = ScriptedProvider::new(turns);
        let mut agent = Agent::new(provider);
        register_echo(&mut agent, "edit");

        let policy = Policy::new(Tier::FullAuto, 50, None);
        let outcome = run_query(resolved_stub(), policy, &mut agent, "prompt", &wiring(None)).await;

        assert_eq!(
            outcome.stop_reason,
            StopReason::Done,
            "--full-auto silences the repetitive guard ⇒ no abort"
        );
        assert_eq!(
            outcome.tool_calls.len(),
            5,
            "all five identical calls executed under --full-auto"
        );
        assert!(outcome.tool_calls.iter().all(|t| t.ok));
    }

    /// Under `--auto` the repetitive-call soft guard is active: the same five
    /// identical calls trip it and the run collapses to `Error` — REQ-H08.
    #[tokio::test]
    async fn test_runner_auto_aborts_on_repeated_identical_calls() {
        let mut turns = identical_edit_turns(5);
        turns.push(Turn::Text("unreached".to_string()));
        let provider = ScriptedProvider::new(turns);
        let mut agent = Agent::new(provider);
        register_echo(&mut agent, "edit");

        let policy = Policy::new(Tier::Auto, 15, None);
        let outcome = run_query(resolved_stub(), policy, &mut agent, "prompt", &wiring(None)).await;

        assert_eq!(
            outcome.stop_reason,
            StopReason::Error,
            "the repetitive-guard abort collapses to Error under --auto"
        );
    }

    /// The transcript projects the user turn and the assistant turn, folding the
    /// tool call (with its recorded result and `ok`) into the assistant entry.
    #[tokio::test]
    async fn test_runner_transcript_folds_tool_call_into_assistant_entry() {
        let provider = ScriptedProvider::new(vec![
            tool_turn("call-1", "ls"),
            Turn::Text("here you go".to_string()),
        ]);
        let mut agent = Agent::new(provider);
        register_echo(&mut agent, "ls");

        let policy = Policy::new(Tier::Auto, 15, None);
        let outcome = run_query(resolved_stub(), policy, &mut agent, "prompt", &wiring(None)).await;

        // user prompt, assistant(tool-use), assistant(final text).
        let user = outcome
            .transcript
            .iter()
            .find(|e| e.role == "user")
            .expect("a user entry");
        assert_eq!(user.content, "prompt");

        let with_tool = outcome
            .transcript
            .iter()
            .find(|e| e.tool_calls.is_some())
            .expect("an assistant entry carrying the tool call");
        let calls = with_tool.tool_calls.as_ref().expect("tool_calls present");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "ls");
        assert!(calls[0].ok);
    }

    /// MAGI S6 gate finding 7: a `User`-role message that carries BOTH a real `Text` block
    /// and a `Content::ToolResult` block must not have its text silently discarded. No
    /// current production write path constructs this shape (see `build_transcript`'s
    /// updated rustdoc), but nothing in `Message`/`Content` rules it out, and the old
    /// "any ToolResult block ⇒ skip the whole message" check would have dropped the text
    /// here.
    #[test]
    fn test_build_transcript_keeps_text_from_a_user_message_that_also_carries_a_tool_result() {
        let history = vec![Message {
            role: Role::User,
            content: vec![
                Content::Text {
                    text: "please also note: X".to_string(),
                },
                Content::ToolResult {
                    tool_use_id: "call-1".to_string(),
                    content: "tool output".to_string(),
                    is_error: false,
                },
            ],
        }];
        let records: HashMap<&str, &ToolCallRecord> = HashMap::new();

        let transcript = build_transcript(&history, &records);

        assert_eq!(
            transcript.len(),
            1,
            "the text must not vanish: {transcript:?}"
        );
        assert_eq!(transcript[0].role, "user");
        assert_eq!(transcript[0].content, "please also note: X");
    }

    /// The overwhelmingly common case is unaffected: a PURE tool-result carrier (no text at
    /// all) is still folded away — its outcome already lives inside the requesting
    /// assistant entry's `tool_calls`.
    #[test]
    fn test_build_transcript_still_folds_away_a_pure_tool_result_carrier() {
        let history = vec![
            Message::user("prompt"),
            Message {
                role: Role::User,
                content: vec![Content::ToolResult {
                    tool_use_id: "call-1".to_string(),
                    content: "tool output".to_string(),
                    is_error: false,
                }],
            },
        ];
        let records: HashMap<&str, &ToolCallRecord> = HashMap::new();

        let transcript = build_transcript(&history, &records);

        assert_eq!(
            transcript.len(),
            1,
            "the pure tool-result carrier must still be folded away: {transcript:?}"
        );
        assert_eq!(transcript[0].content, "prompt");
    }

    /// MAGI S6 gate finding 8: `policy::READ_ONLY_TOOLS`/`READ_WRITE_TOOLS` must match the
    /// REAL `Tool::name()` implementations, not a third hand-typed literal list
    /// (`policy.rs`'s own `REAL_REGISTERED_TOOL_NAMES`, which was exactly as fictional as the
    /// two lists it checked — nothing failed if `main.rs` and `policy.rs` drifted, only if
    /// `policy.rs` disagreed with itself). The `headless` module stays pure (cannot import
    /// `crate::tools`, see its own module rustdoc) — but this file, `headless_runner.rs`,
    /// already lives in the BINARY crate alongside `crate::tools` (it already imports
    /// `crate::tools::consult::ConsultTool`/`crate::tools::bash::BashTool` for other tests),
    /// so it can construct each real tool and read its actual name.
    ///
    /// This does not close the drift risk completely: a BRAND NEW tool type registered in
    /// `main.rs` without a corresponding construction added here would still slip past both
    /// this test and the old one — there is no single "real registry" function in `main.rs` to
    /// call instead of enumerating tool types by hand, and `main.rs` is outside this file's
    /// edit surface. What this closes is the more common failure mode this project has already
    /// hit once (the nextest-filter drift documented in `CLAUDE.local.md`): an EXISTING tool's
    /// name changing without every list that names it being updated in step.
    #[tokio::test]
    async fn test_policy_tool_lists_match_real_tool_name_implementations() {
        use magi_rs::headless::policy::{READ_ONLY_TOOLS, READ_WRITE_TOOLS};

        use crate::system::database::MemoryStore;
        use crate::system::fs::{FileSystem, RealFileSystem};
        use crate::system::grep::MockGrep;
        use crate::tools::bash::BashTool;
        use crate::tools::grep::GrepTool;
        use crate::tools::knowledge::ProjectFactTool;
        use crate::tools::ls::ListTool;
        use crate::tools::read::FileReadTool;
        use crate::tools::write::FileWriteTool;

        /// Never actually called: `ProjectFactTool::new` only needs a value of the right
        /// type to construct, and this test never invokes any of its methods — it exists
        /// purely so `Tool::name()` can be called on a real `ProjectFactTool`.
        struct UnusedMemoryStore;

        #[async_trait::async_trait]
        impl MemoryStore for UnusedMemoryStore {
            async fn create_session(&self, _project_name: &str) -> anyhow::Result<String> {
                unimplemented!("not exercised by this test")
            }
            async fn add_message(
                &self,
                _session_id: &str,
                _message: &Message,
            ) -> anyhow::Result<()> {
                unimplemented!("not exercised by this test")
            }
            async fn get_messages(&self, _session_id: &str) -> anyhow::Result<Vec<Message>> {
                unimplemented!("not exercised by this test")
            }
            async fn list_sessions(&self) -> anyhow::Result<Vec<(String, String)>> {
                unimplemented!("not exercised by this test")
            }
            async fn set_knowledge(&self, _key: &str, _value: &str) -> anyhow::Result<()> {
                unimplemented!("not exercised by this test")
            }
            async fn get_knowledge(&self, _key: &str) -> anyhow::Result<Option<String>> {
                unimplemented!("not exercised by this test")
            }
            async fn list_knowledge_keys(&self) -> anyhow::Result<Vec<String>> {
                unimplemented!("not exercised by this test")
            }
        }

        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();
        let fs: Arc<dyn FileSystem> = Arc::new(RealFileSystem);

        let real_tools: Vec<Box<dyn Tool>> = vec![
            Box::new(ListTool::new(fs.clone(), root.clone()).expect("ListTool")),
            Box::new(FileReadTool::new(fs.clone(), root.clone()).expect("FileReadTool")),
            Box::new(FileWriteTool::new(fs.clone(), root.clone()).expect("FileWriteTool")),
            Box::new(GrepTool::new(Box::new(MockGrep::new()), root.clone()).expect("GrepTool")),
            Box::new(BashTool::new(root.clone()).expect("BashTool")),
            Box::new(ConsultTool::new(canned_magi(), false)),
            Box::new(ProjectFactTool::new(Arc::new(UnusedMemoryStore))),
        ];
        let mut real_names: Vec<&str> = real_tools.iter().map(|t| t.name()).collect();
        real_names.sort_unstable();

        let mut known: Vec<&str> = READ_ONLY_TOOLS
            .iter()
            .copied()
            .chain(READ_WRITE_TOOLS.iter().copied())
            .collect();
        known.sort_unstable();

        assert_eq!(
            known, real_names,
            "policy::READ_ONLY_TOOLS + READ_WRITE_TOOLS must match the real Tool::name() \
             implementations exactly — update the policy lists when a tool's name changes \
             or a new tool is registered in main.rs"
        );
    }

    // ── REQ-H36: wall-clock timeout ─────────────────────────────────────────────

    /// `resolve_tier_timeout_default` applies the tier default: the read-only `default`
    /// tier gets no ceiling, while the tool-executing tiers get the 900 s hard
    /// default when none was configured; an explicit config value always wins.
    #[test]
    fn test_resolve_tier_timeout_default_applies_tier_default() {
        // No configured timeout: default tier ⇒ None; Auto/FullAuto ⇒ 900 s.
        assert_eq!(
            resolve_tier_timeout_default(
                &Policy::new(Tier::Default, 15, None),
                FULL_AUTO_TIMEOUT_SECS
            ),
            None,
            "the read-only default tier carries no wall-clock ceiling"
        );
        assert_eq!(
            resolve_tier_timeout_default(
                &Policy::new(Tier::Auto, 15, None),
                FULL_AUTO_TIMEOUT_SECS
            ),
            Some(Duration::from_secs(FULL_AUTO_TIMEOUT_SECS))
        );
        assert_eq!(
            resolve_tier_timeout_default(
                &Policy::new(Tier::FullAuto, 50, None),
                FULL_AUTO_TIMEOUT_SECS
            ),
            Some(Duration::from_secs(FULL_AUTO_TIMEOUT_SECS))
        );
        // An explicit configured timeout wins over the tier default, in any tier.
        assert_eq!(
            resolve_tier_timeout_default(
                &Policy::new(Tier::Auto, 15, Some(5)),
                FULL_AUTO_TIMEOUT_SECS
            ),
            Some(Duration::from_secs(5))
        );
        assert_eq!(
            resolve_tier_timeout_default(
                &Policy::new(Tier::Default, 15, Some(7)),
                FULL_AUTO_TIMEOUT_SECS
            ),
            Some(Duration::from_secs(7))
        );
    }

    /// REQ-H36, spec §11: the tool-executing tiers must fall back to the
    /// EFFECTIVE `full_auto_timeout_secs` (an operator-lowered
    /// `[headless] timeout_secs`), not the `FULL_AUTO_TIMEOUT_SECS` constant,
    /// when no explicit `--timeout`/policy timeout was configured.
    #[test]
    fn test_resolve_tier_timeout_default_respects_a_custom_effective_full_auto_default() {
        let custom_default = 42u64;
        assert_ne!(
            custom_default, FULL_AUTO_TIMEOUT_SECS,
            "fixture must differ from the module constant to prove it is not used"
        );

        assert_eq!(
            resolve_tier_timeout_default(&Policy::new(Tier::Auto, 15, None), custom_default),
            Some(Duration::from_secs(custom_default)),
            "a custom (smaller) effective default must apply, not the module constant"
        );
        assert_eq!(
            resolve_tier_timeout_default(&Policy::new(Tier::FullAuto, 50, None), custom_default),
            Some(Duration::from_secs(custom_default))
        );
        // The read-only tier is still unbounded regardless of the effective default.
        assert_eq!(
            resolve_tier_timeout_default(&Policy::new(Tier::Default, 15, None), custom_default),
            None
        );
    }

    /// With `timeout = None` the run is never cancelled: a normal tool + text run
    /// completes with `Done` and no error payload.
    #[tokio::test]
    async fn test_run_query_none_timeout_does_not_cancel() {
        let provider = ScriptedProvider::new(vec![
            tool_turn("t1", "ls"),
            Turn::Text("finished".to_string()),
        ]);
        let mut agent = Agent::new(provider);
        register_echo(&mut agent, "ls");

        let policy = Policy::new(Tier::Auto, 15, None);
        let outcome = run_query(resolved_stub(), policy, &mut agent, "prompt", &wiring(None)).await;

        assert_eq!(
            outcome.stop_reason,
            StopReason::Done,
            "an unbounded run completes normally"
        );
        assert!(outcome.error.is_none(), "no timeout ⇒ no error payload");
        assert_eq!(outcome.response.as_deref(), Some("finished"));
        assert!(outcome.tool_calls.iter().any(|t| t.name == "ls" && t.ok));
    }

    /// Deterministic end-to-end wall-clock timeout: a real `bash` subprocess tree
    /// is killed when `--timeout` elapses, and the run reports
    /// `stop_reason = Error` / `error.kind = Timeout` (REQ-H36). Windows path.
    #[cfg(windows)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_run_query_timeout_kills_bash_and_reports_timeout_windows() {
        run_query_timeout_kills_bash().await;
    }

    /// POSIX process-group analog of the Windows timeout test above (CI-only).
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_run_query_timeout_kills_bash_and_reports_timeout_unix() {
        run_query_timeout_kills_bash().await;
    }

    /// Shared body: drive a real [`BashTool`] running a long `python worker.py`
    /// under a short `--timeout`; assert the outcome is a `Timeout` error and the
    /// subprocess tree was terminated (no DONE marker) via the group killer's
    /// `Drop` backstop (the query future is dropped by `tokio::time::timeout`).
    #[cfg(any(windows, unix))]
    async fn run_query_timeout_kills_bash() {
        use crate::tools::bash::BashTool;
        use crate::tools::proc_group::test_support::{
            measure_cold_start_ms, python_available, tree_kill_worker_with_sleep, wait_for_marker,
            CANCEL_FIRE_DELAY_MS, COLD_START_MARGIN_FACTOR, DONE_MARGIN_SECS,
            MARKER_WAIT_DEADLINE_MS,
        };

        // **Fails closed, where it used to skip** (S4 Loop 2, Balthasar). A `return` here is a
        // PASS: on a machine without an interpreter the suite reported green while REQ-H36's
        // guarantee — that the wall-clock bound really terminates the subprocess tree, rather
        // than merely stopping the wait — went unverified. That is the vacuous guardian in its
        // purest form: not an assertion that holds trivially, but a test that never ran and
        // still counted.
        //
        // Failing is the honest signal. Every platform in this project's CI ships an
        // interpreter, so this fires only where the environment genuinely cannot verify a
        // shipped promise, and that is worth a red build rather than a line on stderr nobody
        // reads.
        assert!(
            python_available(),
            "no python interpreter: this test spawns a REAL child to prove the --timeout kills \
             the process tree (REQ-H36), and skipping it would report that guarantee as verified \
             when nothing checked it"
        );
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().canonicalize().expect("canonicalize");

        // Size the kill from a MEASURED cold start instead of a constant. The constant
        // was calibrated on one machine, grown from ~2 s to 5 s after this exact
        // failure, and still fired before the worker existed on a CI runner — the kill
        // landing ahead of START makes the test fail on its own precondition, which
        // reads like a broken kill and is not one.
        let cold_ms = measure_cold_start_ms(&root).await;
        let fire_after_ms = (cold_ms * COLD_START_MARGIN_FACTOR).max(CANCEL_FIRE_DELAY_MS);
        // Derived from the same number, so DONE always sits a fixed distance AFTER
        // whenever the kill actually landed. Two independent constants are what let the
        // ordering invert on a slow machine.
        let worker_sleep_secs = fire_after_ms.div_ceil(1_000) + DONE_MARGIN_SECS;
        let post_kill_wait_ms = DONE_MARGIN_SECS * 1_000 + 3_000;
        std::fs::write(
            root.join("worker.py"),
            tree_kill_worker_with_sleep(worker_sleep_secs),
        )
        .expect("write worker");

        let provider = ScriptedProvider::new(vec![
            Turn::Tool {
                id: "b1".to_string(),
                name: "bash".to_string(),
                input: json!({ "command": "python worker.py" }),
            },
            Turn::Text("unreached".to_string()),
        ]);
        let mut agent = Agent::new(provider);
        agent.register_tool(Box::new(BashTool::new(root.clone()).expect("BashTool")));

        // The `--timeout` budget below must absorb python/shell cold-start under
        // full-suite CPU contention (see `CANCEL_FIRE_DELAY_MS` rustdoc) while
        // still firing well before the worker's sleep completes, so the
        // wall-clock timeout genuinely pre-empts live work rather than racing
        // a cold start that hasn't even written START yet.
        let policy = Policy::new(Tier::Auto, 15, None);
        let outcome = run_query(
            resolved_stub(),
            policy,
            &mut agent,
            "go",
            &wiring(Some(Duration::from_millis(fire_after_ms))),
        )
        .await;

        assert_eq!(
            outcome.stop_reason,
            StopReason::Error,
            "a wall-clock timeout maps to Error"
        );
        let err = outcome
            .error
            .expect("a timeout populates the error payload");
        assert_eq!(
            err.kind,
            ErrorKind::Timeout,
            "error.kind must be timeout (first-class)"
        );

        // The precondition, waited on as a CONDITION with a generous failure
        // deadline. It guards against a false pass: had the child never run, DONE
        // would be absent for a reason that proves nothing about the kill.
        let start_seen = wait_for_marker(&root.join("start.marker"), MARKER_WAIT_DEADLINE_MS).await;
        assert!(
            start_seen.is_some(),
            "the child really ran (START present) — measured cold start {cold_ms} ms, \
             kill armed at {fire_after_ms} ms; a miss here means the kill still \
             outran the spawn, not that tree-kill is broken"
        );

        // Prove the tree died: wait past when a surviving orphan would write DONE.
        // The margin is derived from the kill moment above, not a constant.
        let done = root.join("done.marker");
        tokio::time::sleep(Duration::from_millis(post_kill_wait_ms)).await;
        assert!(
            !done.exists(),
            "the bash subprocess tree must be dead — no DONE marker"
        );
    }

    // ── REQ-H21/H22: consult (direct + forced) ─────────────────────────────────

    /// REQ-EA01/EA03, the POSITIVE path through the runner: `Include` on the params actually
    /// reaches the emitted object.
    ///
    /// The other tests cover the emitter given its answer, the CLI given its flag, and the agent
    /// path's permanent `Omit`. None of them covers the segment between: a flag that parses, is
    /// stored on `MagiRuntimeParams` and is never applied would satisfy every one of them. That
    /// exact defect — a config bundle threaded past its consumer and quietly not used — has
    /// happened in this codebase before (`AutonomousRunConfig`), which is why the wiring gets its
    /// own test rather than being inferred from the parts (S1 gate, Balthasar).
    #[tokio::test]
    async fn test_run_consult_include_reaches_the_emitted_object() {
        let cfg = MagiConfig::default();
        let sink = RecordingNoticeSink::default();
        let outcome = run_consult(
            resolved_stub(),
            canned_magi(),
            "should we migrate X to Y?",
            None,
            Some(Mode::Analysis),
            &MagiRuntimeParams {
                kind: ProviderKind::OpenAiCompat,
                classifier: &NeverClassifier,
                configured_mode: None,
                untrusted_content: false,
                magi_config: &cfg,
                timeout_decision: neutral_timeout_decision(),
                notice_sink: &sink,
                auditor: test_auditor(),
                structured_verdicts: StructuredVerdicts::Include,
                budget: BudgetTelemetry::default(),
            },
        )
        .await;

        let consult = outcome.consult.expect("the MAGI object must be present");
        assert!(
            consult["agents"].as_array().is_some_and(|a| !a.is_empty()),
            "the seats that answered must reach the object: {consult:?}"
        );
        assert!(
            consult["consensus"]["consensus_verdict"].is_string(),
            "and so must the consensus: {consult:?}"
        );
    }

    /// `magi consult` direct path: the 3 perspectives run off the agent loop and
    /// `RunOutcome.consult` is the (non-null) MAGI object; `response` is the report
    /// and no tool calls are recorded (REQ-H21).
    #[tokio::test]
    async fn test_run_consult_direct_runs_three_perspectives_and_populates_consult() {
        let cfg = MagiConfig::default();
        let sink = RecordingNoticeSink::default();
        let outcome = run_consult(
            resolved_stub(),
            canned_magi(),
            "should we migrate X to Y?",
            None,
            Some(Mode::Analysis),
            &MagiRuntimeParams {
                kind: ProviderKind::OpenAiCompat,
                classifier: &NeverClassifier,
                configured_mode: None,
                untrusted_content: false,
                magi_config: &cfg,
                timeout_decision: neutral_timeout_decision(),
                notice_sink: &sink,
                auditor: test_auditor(),
                structured_verdicts: StructuredVerdicts::Omit,
                budget: BudgetTelemetry::default(),
            },
        )
        .await;

        assert_eq!(outcome.stop_reason, StopReason::Done);
        assert!(outcome.error.is_none(), "a healthy consult has no error");
        let consult = outcome.consult.expect("the MAGI object must be present");
        assert_eq!(consult["degraded"], json!(false), "3 agents ⇒ not degraded");
        assert!(
            consult["report"].as_str().is_some_and(|s| !s.is_empty()),
            "the consult object carries the MAGI report"
        );
        assert!(
            outcome.response.as_deref().is_some_and(|s| !s.is_empty()),
            "response reflects the direct report text"
        );
        assert!(
            outcome.tool_calls.is_empty(),
            "the direct path runs no agent tool-loop ⇒ no tool calls"
        );
    }

    /// Fix round 1, Finding 1 (SC-A11f/REQ-A11d): `endpoint_divergence` reaches
    /// the JSON from a REAL `run_consult` execution — a diverging `MagiConfig`
    /// AND a classifier that genuinely runs — not from a hand-built `RunContext`,
    /// which is exactly what let the previous hardcoded-`false` field sit under a
    /// green suite undetected.
    #[tokio::test]
    async fn endpoint_divergence_reaches_the_json_from_a_real_run() {
        let diverged = MagiConfig::from_toml_str(
            "base_url = \"http://a/v1\"\n[magi]\nbase_url = \"http://b/v1\"\n",
        )
        .expect("valid toml");
        let sink = RecordingNoticeSink::default();
        let outcome = run_consult(
            resolved_stub(),
            canned_magi(),
            "should we migrate X to Y?",
            None,
            // No explicit mode AND no configured mode: classification runs.
            None,
            &MagiRuntimeParams {
                kind: ProviderKind::OpenAiCompat,
                classifier: &AlwaysClassifies(Mode::Analysis),
                configured_mode: None,
                untrusted_content: false,
                magi_config: &diverged,
                timeout_decision: neutral_timeout_decision(),
                notice_sink: &sink,
                auditor: test_auditor(),
                structured_verdicts: StructuredVerdicts::Omit,
                budget: BudgetTelemetry::default(),
            },
        )
        .await;

        let consult = outcome.consult.expect("healthy consult");
        assert_eq!(
            consult["endpoint_divergence"], true,
            "the config diverges AND classification was genuinely attempted"
        );
    }

    /// The negative case for the test above: same diverging config, but an
    /// EXPLICIT mode means classification never runs, so `endpoint_divergence`
    /// must stay `false` — proves the field tracks the REAL attempt flag, not
    /// merely whether the config happens to diverge.
    #[tokio::test]
    async fn endpoint_divergence_stays_false_without_a_real_classification_attempt() {
        let diverged = MagiConfig::from_toml_str(
            "base_url = \"http://a/v1\"\n[magi]\nbase_url = \"http://b/v1\"\n",
        )
        .expect("valid toml");
        let sink = RecordingNoticeSink::default();
        let outcome = run_consult(
            resolved_stub(),
            canned_magi(),
            "should we migrate X to Y?",
            None,
            Some(Mode::Analysis), // explicit: classification is skipped
            &MagiRuntimeParams {
                kind: ProviderKind::OpenAiCompat,
                classifier: &NeverClassifier,
                configured_mode: None,
                untrusted_content: false,
                magi_config: &diverged,
                timeout_decision: neutral_timeout_decision(),
                notice_sink: &sink,
                auditor: test_auditor(),
                structured_verdicts: StructuredVerdicts::Omit,
                budget: BudgetTelemetry::default(),
            },
        )
        .await;

        let consult = outcome.consult.expect("healthy consult");
        assert_eq!(
            consult["endpoint_divergence"], false,
            "no classification attempted ⇒ the content never left for the principal, \
             even though the config diverges"
        );
    }

    /// Fix round 1, Finding 1 (SC-A04d): `timeout_below_formula` reaches the JSON
    /// from the REAL `TimeoutDecision` threaded through `MagiRuntimeParams`,
    /// driven through the production `run_consult` entry point.
    #[tokio::test]
    async fn timeout_below_formula_reaches_the_json_from_a_real_run() {
        let cfg = MagiConfig::default();
        let ceiling = magi_rs::magi::AGENT_TIMEOUT_SECS;
        // An `asked` well below `headless_consult_timeout_secs(ceiling)`.
        let decision = magi_rs::magi::resolve_run_timeout(
            Some(1),
            ceiling,
            0,
            false,
            magi_rs::magi::TimeoutMeasure::DerivesCeiling,
        );
        assert!(
            decision.below_formula,
            "test setup: this must actually trigger the formula check"
        );
        let sink = RecordingNoticeSink::default();

        let outcome = run_consult(
            resolved_stub(),
            canned_magi(),
            "should we migrate X to Y?",
            None,
            Some(Mode::Analysis),
            &MagiRuntimeParams {
                kind: ProviderKind::OpenAiCompat,
                classifier: &NeverClassifier,
                configured_mode: None,
                untrusted_content: false,
                magi_config: &cfg,
                timeout_decision: decision,
                notice_sink: &sink,
                auditor: test_auditor(),
                structured_verdicts: StructuredVerdicts::Omit,
                budget: BudgetTelemetry::default(),
            },
        )
        .await;

        let consult = outcome.consult.expect("healthy consult");
        assert_eq!(consult["timeout_below_formula"], true);
    }

    /// Fix round 2 (SC-A04d's other half): the warning reaches a REAL `analyze_
    /// direct` run's `notice_sink`, not just the JSON flag. Driven through
    /// `run_consult` with a REAL `TimeoutDecision` from `magi_rs::magi::resolve_run_timeout`,
    /// exactly like the JSON-flag test above — a hand-built `TimeoutDecision`
    /// asserted against in isolation would prove nothing about whether
    /// `analyze_direct` actually reads `runtime.notice_sink`.
    ///
    /// Asserts on "wall-clock deadline", not "--timeout": the message is deliberately
    /// source-agnostic (`resolve_run_timeout`'s rustdoc explains why — this same deadline can
    /// come from a tier default or `[headless] timeout_secs`, not only the flag), so do not
    /// "fix" this assertion back to the flag string.
    ///
    /// **`TimeoutMeasure::ConfiguredCeiling`, deliberately, not `DerivesCeiling`** (fix round 3):
    /// this test proves the PLUMBING (a populated `.warning` reaches `notice_sink`), not which
    /// route production actually takes. `DerivesCeiling` never populates `.warning` — see its
    /// rustdoc — because `prepare_headless` already reports that route's floored condition via
    /// `floored_ceiling_notice`, and this test would fail its own setup assertion if it used
    /// that variant.
    #[tokio::test]
    async fn sc_a04d_warning_reaches_the_notice_sink_from_a_real_run() {
        let cfg = MagiConfig::default();
        let ceiling = magi_rs::magi::AGENT_TIMEOUT_SECS;
        let decision = magi_rs::magi::resolve_run_timeout(
            Some(1),
            ceiling,
            0,
            false,
            magi_rs::magi::TimeoutMeasure::ConfiguredCeiling,
        );
        assert!(
            decision.below_formula && decision.warning.is_some(),
            "test setup: this must actually trigger the warning"
        );
        let sink = RecordingNoticeSink::default();

        let _outcome = run_consult(
            resolved_stub(),
            canned_magi(),
            "should we migrate X to Y?",
            None,
            Some(Mode::Analysis),
            &MagiRuntimeParams {
                kind: ProviderKind::OpenAiCompat,
                classifier: &NeverClassifier,
                configured_mode: None,
                untrusted_content: false,
                magi_config: &cfg,
                timeout_decision: decision,
                notice_sink: &sink,
                auditor: test_auditor(),
                structured_verdicts: StructuredVerdicts::Omit,
                budget: BudgetTelemetry::default(),
            },
        )
        .await;

        assert!(
            sink.emitted().contains("wall-clock deadline"),
            "the human-facing warning must reach the sink: {}",
            sink.emitted()
        );
    }

    /// The negative case, in both directions the spec names: a `--timeout` AT OR
    /// ABOVE the formula, and an ABSENT `--timeout`, must emit NOTHING. Without
    /// this, a sink that always fires (an unconditional `eprintln!` regardless of
    /// `decision.warning`) would pass the positive test above too.
    #[tokio::test]
    async fn sc_a04d_warning_stays_silent_at_or_above_the_formula_and_when_absent() {
        let cfg = MagiConfig::default();
        let ceiling = magi_rs::magi::AGENT_TIMEOUT_SECS;

        let generous = magi_rs::magi::resolve_run_timeout(
            Some(100_000),
            ceiling,
            0,
            false,
            magi_rs::magi::TimeoutMeasure::DerivesCeiling,
        );
        assert!(
            !generous.below_formula && generous.warning.is_none(),
            "test setup: this must NOT trigger the formula check"
        );
        let sink_generous = RecordingNoticeSink::default();
        run_consult(
            resolved_stub(),
            canned_magi(),
            "should we migrate X to Y?",
            None,
            Some(Mode::Analysis),
            &MagiRuntimeParams {
                kind: ProviderKind::OpenAiCompat,
                classifier: &NeverClassifier,
                configured_mode: None,
                untrusted_content: false,
                magi_config: &cfg,
                timeout_decision: generous,
                notice_sink: &sink_generous,
                auditor: test_auditor(),
                structured_verdicts: StructuredVerdicts::Omit,
                budget: BudgetTelemetry::default(),
            },
        )
        .await;
        assert!(
            sink_generous.emitted().is_empty(),
            "a --timeout at/above the formula must emit nothing: {}",
            sink_generous.emitted()
        );

        let absent = magi_rs::magi::resolve_run_timeout(
            None,
            ceiling,
            0,
            false,
            magi_rs::magi::TimeoutMeasure::DerivesCeiling,
        );
        assert!(
            absent.warning.is_none(),
            "test setup: no --timeout, no warning"
        );
        let sink_absent = RecordingNoticeSink::default();
        run_consult(
            resolved_stub(),
            canned_magi(),
            "should we migrate X to Y?",
            None,
            Some(Mode::Analysis),
            &MagiRuntimeParams {
                kind: ProviderKind::OpenAiCompat,
                classifier: &NeverClassifier,
                configured_mode: None,
                untrusted_content: false,
                magi_config: &cfg,
                timeout_decision: absent,
                notice_sink: &sink_absent,
                auditor: test_auditor(),
                structured_verdicts: StructuredVerdicts::Omit,
                budget: BudgetTelemetry::default(),
            },
        )
        .await;
        assert!(
            sink_absent.emitted().is_empty(),
            "an absent --timeout must emit nothing: {}",
            sink_absent.emitted()
        );
    }

    /// `magi consult` over the effective input cap is rejected as `input_invalid`
    /// (exit 2), NOT truncated: `consult`/`response` stay `None` (REQ-H33/SC-A11b).
    #[tokio::test]
    async fn test_run_consult_over_cap_is_input_invalid_not_truncated() {
        // `MagiConfig::default()` below resolves to `magi_rs::magi::MAX_QUERY_BYTES`
        // (REQ-A11b's raised cap, SC-A11) — not the retired 8 KiB `MAX_QUERY_LEN`.
        let big = "x".repeat(magi_rs::magi::MAX_QUERY_BYTES + 1);
        let cfg = MagiConfig::default();
        let sink = RecordingNoticeSink::default();
        let outcome = run_consult(
            resolved_stub(),
            canned_magi(),
            &big,
            None,
            Some(Mode::Analysis),
            &MagiRuntimeParams {
                kind: ProviderKind::OpenAiCompat,
                classifier: &NeverClassifier,
                configured_mode: None,
                untrusted_content: false,
                magi_config: &cfg,
                timeout_decision: neutral_timeout_decision(),
                notice_sink: &sink,
                auditor: test_auditor(),
                structured_verdicts: StructuredVerdicts::Omit,
                budget: BudgetTelemetry::default(),
            },
        )
        .await;

        assert_eq!(outcome.stop_reason, StopReason::Error);
        let err = outcome
            .error
            .expect("an over-cap prompt populates the error");
        assert_eq!(
            err.kind,
            ErrorKind::InputInvalid,
            "over-cap ⇒ input_invalid (→ exit 2)"
        );
        assert!(outcome.consult.is_none(), "rejected input ⇒ no MAGI object");
        assert!(outcome.response.is_none(), "rejected, not truncated");
    }

    /// SC-A11d (the `magi consult` headless direct route specifically): a report
    /// bigger than a tiny configured output cap comes back bounded, with the
    /// level surfaced in the JSON — proving `run_consult`/`analyze_direct`
    /// actually call `truncate_report` with `runtime.magi_config.
    /// effective_tool_result_cap()`, not merely that the function exists.
    #[tokio::test]
    async fn run_consult_bounds_the_report_when_it_exceeds_the_configured_output_cap() {
        let cap = magi_rs::magi::mark_overhead() + 20;
        let cfg = MagiConfig::builder()
            .tool_result_cap_bytes(Some(cap))
            .build()
            .unwrap();
        let sink = RecordingNoticeSink::default();
        let outcome = run_consult(
            resolved_stub(),
            canned_magi(),
            "should we migrate X to Y?",
            None,
            Some(Mode::Analysis),
            &MagiRuntimeParams {
                kind: ProviderKind::OpenAiCompat,
                classifier: &NeverClassifier,
                configured_mode: None,
                untrusted_content: false,
                magi_config: &cfg,
                timeout_decision: neutral_timeout_decision(),
                notice_sink: &sink,
                auditor: test_auditor(),
                structured_verdicts: StructuredVerdicts::Omit,
                budget: BudgetTelemetry::default(),
            },
        )
        .await;

        let consult = outcome.consult.expect("healthy consult");
        let report = consult["report"].as_str().expect("report string");
        assert!(
            report.len() <= cap,
            "the direct consult route must respect the configured output cap: \
             {} > {cap}",
            report.len()
        );
        assert_ne!(
            consult["report_truncated"], "none",
            "a report bigger than the cap must be marked truncated"
        );
    }

    /// Same contract as `test_run_query_reports_the_effective_timeout_in_applied_caps`,
    /// for the direct `magi consult` route: `run_consult` takes its wall-clock ceiling
    /// as the `timeout` parameter directly (not via `RunWiring`), and must copy it into
    /// `applied_caps.timeout_secs` rather than leaving `Resolved::applied_caps`'s
    /// static `None` untouched.
    #[tokio::test]
    async fn run_consult_reports_the_effective_timeout_in_applied_caps() {
        let cfg = MagiConfig::default();
        let sink = RecordingNoticeSink::default();
        let outcome = run_consult(
            resolved_stub(),
            canned_magi(),
            "should we migrate X to Y?",
            Some(Duration::from_secs(77)),
            Some(Mode::Analysis),
            &MagiRuntimeParams {
                kind: ProviderKind::OpenAiCompat,
                classifier: &NeverClassifier,
                configured_mode: None,
                untrusted_content: false,
                magi_config: &cfg,
                timeout_decision: neutral_timeout_decision(),
                notice_sink: &sink,
                auditor: test_auditor(),
                structured_verdicts: StructuredVerdicts::Omit,
                budget: BudgetTelemetry::default(),
            },
        )
        .await;

        assert_eq!(
            outcome.applied_caps.timeout_secs,
            Some(77),
            "run_consult's `timeout` parameter must surface in applied_caps.timeout_secs"
        );
    }

    /// Mirror of `run_query_publishes_the_wirings_budget_not_the_placeholder` for the direct
    /// `magi consult` route: its budget arrives through `MagiRuntimeParams::budget` rather than
    /// `RunWiring`, a SEPARATE literal — a fix applied to one does not reach the other, which is
    /// exactly how `timeout_secs` needed the same correction twice during the MS2 gate.
    #[tokio::test]
    async fn run_consult_publishes_the_runtimes_budget_not_the_placeholder() {
        let expected = BudgetTelemetry {
            operation_budget_secs: 149,
            ceiling_floored: false,
            floor_activation_threshold_secs: 114,
            max_rotations_effective: 2,
            ceiling_above_sanity: false,
        };
        let cfg = MagiConfig::default();
        let sink = RecordingNoticeSink::default();
        let outcome = run_consult(
            resolved_stub(),
            canned_magi(),
            "should we migrate X to Y?",
            None,
            Some(Mode::Analysis),
            &MagiRuntimeParams {
                kind: ProviderKind::OpenAiCompat,
                classifier: &NeverClassifier,
                configured_mode: None,
                untrusted_content: false,
                magi_config: &cfg,
                timeout_decision: neutral_timeout_decision(),
                notice_sink: &sink,
                auditor: test_auditor(),
                structured_verdicts: StructuredVerdicts::Omit,
                budget: expected,
            },
        )
        .await;

        assert_eq!(outcome.applied_caps.budget, expected);
    }

    /// Dropping the `run_consult` future itself — NOT via the `--timeout`
    /// cancellation token — must still abort the spawned 3-perspective MAGI
    /// analysis (the gap `AbortOnDrop` closes, mirroring `ConsultTool::execute`).
    /// A dropped future whose spawned task were merely *detached* (a bare
    /// `JoinHandle` drop) would let the analysis run to its full delay — the
    /// [`slow_droppy_magi`] double proves the opposite: its `complete` future is
    /// dropped promptly, well before its (huge) delay could ever elapse.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_run_consult_future_drop_aborts_spawned_analysis() {
        /// Ceiling for each of the two waits below.
        ///
        /// Generous on purpose: it is a **failure** deadline, not a measurement.
        /// The discriminating property is not "under 30s" — it is "does not
        /// take the analysis's full hour", and 30s settles that decisively
        /// while leaving room for an arbitrarily loaded CI box. A regression to
        /// detach-instead-of-abort keeps the flag `false` for 3600s, so it
        /// still fails, just after a bounded wait.
        const DEADLINE: Duration = Duration::from_secs(30);
        /// Gap between polls. Small enough to be prompt, large enough not to
        /// busy-spin one of the two worker threads.
        const POLL: Duration = Duration::from_millis(5);

        let entered = Arc::new(AtomicBool::new(false));
        let dropped = Arc::new(AtomicBool::new(false));
        // An hour: if the spawned task were merely detached rather than aborted,
        // `dropped` would still be `false` at every point this test can observe.
        let magi = slow_droppy_magi(Duration::from_secs(3_600), entered.clone(), dropped.clone());
        // Named binding (not an inline temporary): `fut` below is driven across
        // several statements before it is ever awaited, so the borrowed
        // `MagiRuntimeParams` (and the `MagiConfig` it borrows) must outlive the
        // statement that creates `fut`.
        let cfg = MagiConfig::default();
        let sink = RecordingNoticeSink::default();
        let runtime = MagiRuntimeParams {
            kind: ProviderKind::OpenAiCompat,
            classifier: &NeverClassifier,
            configured_mode: None,
            untrusted_content: false,
            magi_config: &cfg,
            timeout_decision: neutral_timeout_decision(),
            notice_sink: &sink,
            auditor: test_auditor(),
            structured_verdicts: StructuredVerdicts::Omit,
            budget: BudgetTelemetry::default(),
        };

        {
            let fut = run_consult(
                resolved_stub(),
                magi,
                "should we migrate X to Y?",
                None,
                Some(Mode::Analysis),
                &runtime,
            );
            tokio::pin!(fut);
            // Drive the future until the spawned analysis has REALLY entered
            // `complete` — waiting on the condition, never on a duration.
            //
            // The previous version polled for a flat 20ms and assumed that was
            // enough for `tokio::spawn` to get there. Under CPU contention it
            // is not, and then the future gets dropped before any `DropFlag`
            // exists — so `dropped` can never flip and the test fails for a
            // reason that has nothing to do with the behaviour under test.
            let start_by = Instant::now() + DEADLINE;
            while !entered.load(Ordering::SeqCst) {
                assert!(
                    Instant::now() < start_by,
                    "the spawned MAGI analysis never reached `complete` within \
                     {DEADLINE:?}; nothing about abort-vs-detach can be \
                     concluded from this run"
                );
                let _ = tokio::time::timeout(POLL, &mut fut).await;
            }
        } // `fut` dropped here, WITHOUT ever reaching a terminal state.

        // Wait for the abort to propagate down the task's future tree. Again a
        // condition, not a fixed tick: a starved runtime may need more than one
        // scheduling quantum, and that is not a defect.
        let abort_by = Instant::now() + DEADLINE;
        while !dropped.load(Ordering::SeqCst) {
            assert!(
                Instant::now() < abort_by,
                "dropping the run_consult future must abort the spawned MAGI \
                 analysis, not merely detach it: {DEADLINE:?} elapsed and the \
                 in-flight `complete` was still alive"
            );
            tokio::time::sleep(POLL).await;
        }
    }

    /// MAGI S6 gate finding 6: `bounded_drain_result` must return within `DRAIN_GRACE`
    /// (rather than hang) even when the drain task never sees its channel close — the
    /// pathological case a future leak of a `chunk_tx` clone elsewhere in the crate (outside
    /// this module's edit surface) could produce. The sender is deliberately kept alive so
    /// `rx.recv()` never resolves, simulating exactly that leak.
    #[tokio::test]
    async fn test_bounded_drain_result_degrades_instead_of_hanging_when_channel_never_closes() {
        let (_never_dropped_tx, mut rx) = tokio::sync::mpsc::channel::<()>(1);
        let drain: tokio::task::JoinHandle<Option<u64>> = tokio::spawn(async move {
            loop {
                rx.recv().await;
            }
        });

        let start = Instant::now();
        let result = bounded_drain_result(drain).await;
        let elapsed = start.elapsed();

        assert_eq!(
            result, None,
            "a drain that never observes its channel close must degrade to unmeasured"
        );
        // A generous failure ceiling (not a tight timing assertion, per CLAUDE.local.md's
        // "wait on conditions, never on durations"): the discriminating property is "returned
        // at all", not "returned in exactly DRAIN_GRACE".
        assert!(
            elapsed < DRAIN_GRACE * 10,
            "bounded_drain_result must return promptly bounded by DRAIN_GRACE, not hang \
             indefinitely: took {elapsed:?}"
        );
    }

    /// The well-behaved case is unaffected: a drain that finishes normally (its channel
    /// closes promptly, as `Agent::query_streaming` dropping `chunk_tx` does in production)
    /// still returns its measured value through the new bound.
    #[tokio::test]
    async fn test_bounded_drain_result_returns_the_measured_value_in_the_normal_case() {
        let drain: tokio::task::JoinHandle<Option<u64>> = tokio::spawn(async { Some(42) });
        assert_eq!(bounded_drain_result(drain).await, Some(42));
    }

    /// `query --consult` in `--auto` invokes consult **exactly once**, IN-LOOP —
    /// through the SAME registered `consult` tool a proactive call would use —
    /// and populates `RunOutcome.consult`, even though the agent itself did not
    /// request it (REQ-H22).
    #[tokio::test]
    async fn test_query_forced_consult_auto_runs_exactly_once_and_populates() {
        let provider = ScriptedProvider::new(vec![Turn::Text("here is my answer".to_string())]);
        let mut agent = Agent::new(provider);
        // Production always has this registered whenever a MAGI orchestrator is
        // wired (main.rs); the forced injection reuses it by name (REQ-H22).
        agent.register_tool(Box::new(ConsultTool::new(canned_magi(), true)));

        let policy = Policy::new(Tier::Auto, 15, None);
        let outcome = run_query(
            forced_resolved(),
            policy,
            &mut agent,
            "decide X vs Y",
            &wiring(None),
        )
        .await;

        let consult_calls = outcome
            .tool_calls
            .iter()
            .filter(|t| t.name == "consult")
            .count();
        assert_eq!(consult_calls, 1, "forced consult runs exactly once");
        assert!(
            outcome
                .tool_calls
                .iter()
                .any(|t| t.name == "consult" && t.ok),
            "the forced consult succeeded under --auto"
        );
        assert!(
            outcome.consult.is_some(),
            "the forced consult result is captured into RunOutcome.consult"
        );
    }

    /// `query --consult` in `default` DENIES the forced consult (the tier gate
    /// wins): it is recorded `ok = false` and `RunOutcome.consult` stays `None` —
    /// it is NOT elevated out of the read-only tier (REQ-H22 precedence). The
    /// tier gate fires before the tool is ever looked up, so this holds even
    /// though `consult` IS registered (matching production wiring).
    #[tokio::test]
    async fn test_query_forced_consult_default_denied_and_not_elevated() {
        let provider = ScriptedProvider::new(vec![Turn::Text("read-only answer".to_string())]);
        let mut agent = Agent::new(provider);
        agent.register_tool(Box::new(ConsultTool::new(canned_magi(), true)));

        let policy = Policy::new(Tier::Default, 15, None);
        let outcome = run_query(
            forced_resolved(),
            policy,
            &mut agent,
            "decide X vs Y",
            &wiring(None),
        )
        .await;

        assert!(
            outcome
                .tool_calls
                .iter()
                .any(|t| t.name == "consult" && !t.ok),
            "the forced consult is denied by the default tier (ok=false)"
        );
        assert!(
            outcome.consult.is_none(),
            "a denied consult leaves RunOutcome.consult null — not elevated"
        );
    }

    /// A forced consult does NOT double-fire: the runner-injected consult runs FIRST (before
    /// the model's first turn), and if the model ALSO requests `consult` afterward, that
    /// redundant request is answered but never re-executed or re-recorded —
    /// `RunOutcome.tool_calls` still shows exactly one `consult` entry (`REQ-H22: "does not re-
    /// fire even if the agent wanted to"`).
    #[tokio::test]
    async fn test_query_forced_consult_does_not_double_fire() {
        let provider = ScriptedProvider::new(vec![
            Turn::Tool {
                id: "c1".to_string(),
                name: "consult".to_string(),
                input: json!({ "query": "decide X vs Y" }),
            },
            Turn::Text("synthesized".to_string()),
        ]);
        let mut agent = Agent::new(provider);
        // A real ConsultTool over a canned MAGI: the FORCED (pre-loop) call
        // executes against this; the model's own subsequent request must be
        // blocked without invoking it a second time.
        agent.register_tool(Box::new(ConsultTool::new(canned_magi(), true)));

        let policy = Policy::new(Tier::Auto, 15, None);
        let outcome = run_query(
            forced_resolved(),
            policy,
            &mut agent,
            "decide X vs Y",
            &wiring(None),
        )
        .await;

        let consult_calls = outcome
            .tool_calls
            .iter()
            .filter(|t| t.name == "consult")
            .count();
        assert_eq!(
            consult_calls, 1,
            "the forced pass runs once; the model's own later request is blocked \
             without a second observer-recorded call"
        );
        assert!(
            outcome.consult.is_some(),
            "the single forced consult populates RunOutcome.consult"
        );
    }

    /// The forced consult is a genuine IN-LOOP tool call: it consumes one
    /// `max_tool_calls` slot rather than running "for free" outside the loop
    /// (REQ-H22). With the cap set to exactly 1, the injected consult exhausts
    /// it, so the model's very first (non-consult) tool request immediately
    /// exceeds the cap — proving the forced call occupied the one slot.
    #[tokio::test]
    async fn test_query_forced_consult_counts_against_max_tool_calls() {
        let provider = ScriptedProvider::new(vec![tool_turn("t1", "edit")]);
        let mut agent = Agent::new(provider);
        agent.register_tool(Box::new(ConsultTool::new(canned_magi(), true)));
        register_echo(&mut agent, "edit");

        let policy = Policy::new(Tier::Auto, 1, None);
        let outcome = run_query(
            forced_resolved(),
            policy,
            &mut agent,
            "decide X vs Y",
            &wiring(None),
        )
        .await;

        assert_eq!(
            outcome.stop_reason,
            StopReason::MaxToolCalls,
            "the forced consult already spent the single available slot, so the \
             model's next tool request trips the cap"
        );
        let consult_calls = outcome
            .tool_calls
            .iter()
            .filter(|t| t.name == "consult")
            .count();
        assert_eq!(
            consult_calls, 1,
            "the forced consult itself did run — it occupied the slot"
        );
        assert!(
            outcome.tool_calls.iter().all(|t| t.name != "edit"),
            "the cap trips before the model's tool is ever authorized/executed"
        );
    }

    /// REQ-H22 x repetitive-call guard: the runner-injected forced consult
    /// must NOT seed the model's own repetition tracking
    /// (`last_normalized_tool`/`repeat_count`). If it did, the model's own
    /// SUBSEQUENT identical `consult` requests would be miscounted as repeats
    /// of a call the model never made, tripping the "Repetitive tool call
    /// detected" abort one call earlier than it should.
    ///
    /// Script: 4 model-issued `consult` calls, identical to the forced
    /// consult's `{"query": prompt}` input, with `max_tool_calls` capped at
    /// exactly 4 (one slot for the forced call + three for the model). If the
    /// guard counts only GENUINE model repeats (correct), the model's 3rd
    /// identical call brings `repeat_count` to 2 — still under the 3-repeat
    /// threshold — so no abort; the model's 4th attempt then trips
    /// `max_tool_calls` first, giving `StopReason::MaxToolCalls`. If the
    /// forced call wrongly seeds the tracker (bug), the model's 3rd identical
    /// call is miscounted as the 3rd repeat and the run aborts early with
    /// `StopReason::Error` / "Repetitive tool call detected" instead.
    #[tokio::test]
    async fn test_forced_consult_does_not_seed_repetitive_guard_for_model_calls() {
        let consult_call = || Turn::Tool {
            id: "c".to_string(),
            name: "consult".to_string(),
            input: json!({ "query": "decide X vs Y" }),
        };
        let provider = ScriptedProvider::new(vec![
            consult_call(),
            consult_call(),
            consult_call(),
            consult_call(),
        ]);
        let mut agent = Agent::new(provider);
        agent.register_tool(Box::new(ConsultTool::new(canned_magi(), true)));

        // One slot for the forced consult + three for the model's own attempts.
        let policy = Policy::new(Tier::Auto, 4, None);
        let outcome = run_query(
            forced_resolved(),
            policy,
            &mut agent,
            "decide X vs Y",
            &wiring(None),
        )
        .await;

        assert_eq!(
            outcome.stop_reason,
            StopReason::MaxToolCalls,
            "the cap, not the repetitive guard, must be what stops the run — \
             the forced consult must not have seeded the model's repeat \
             tracking (got error: {:?})",
            outcome.error,
        );
    }

    /// A provider that reports whether its (only) call saw a `consult`
    /// `ToolResult` already present in the message history — proves the
    /// model's turn genuinely reacts to the forced consult's content, not
    /// merely that the loop continued after it.
    struct ConsultReactingProvider;

    #[async_trait]
    impl Provider for ConsultReactingProvider {
        async fn stream_messages(
            &self,
            messages: &[Message],
            _tools: &[Box<dyn Tool>],
            _system: Option<&str>,
        ) -> Result<BoxStream<'static, Result<ResponseChunk>>> {
            let saw_consult = messages.iter().any(|m| {
                m.content.iter().any(|c| {
                    matches!(c, Content::ToolResult { content, .. } if content.contains("degraded"))
                })
            });
            let text = if saw_consult {
                "reacted-to-consult"
            } else {
                "no-consult-seen"
            };
            Ok(Box::pin(stream::iter(vec![
                Ok(ResponseChunk::TextDelta(text.to_string())),
                Ok(ResponseChunk::MessageDone(Message::assistant(text))),
            ])))
        }
    }

    /// The model's turn AFTER the forced consult genuinely includes its result
    /// in the conversation it sees — this is the point of moving the forced
    /// pass in-loop (REQ-H22): the consult's `{"report":…,"degraded":…}`
    /// `ToolResult` is already in `working` by the time the FIRST provider
    /// call happens.
    #[tokio::test]
    async fn test_query_forced_consult_result_reaches_the_models_next_turn() {
        let mut agent = Agent::new(Arc::new(ConsultReactingProvider));
        agent.register_tool(Box::new(ConsultTool::new(canned_magi(), true)));

        let policy = Policy::new(Tier::Auto, 15, None);
        let outcome = run_query(
            forced_resolved(),
            policy,
            &mut agent,
            "decide X vs Y",
            &wiring(None),
        )
        .await;

        assert_eq!(
            outcome.response.as_deref(),
            Some("reacted-to-consult"),
            "the model's first provider turn must already see the forced \
             consult's ToolResult content"
        );
    }

    /// SC-A20c: a vetoed AUTONOMOUS consult — the model's own `ToolUse`, not an
    /// operator `--consult` — leaves the headless run at `StopReason::Done` with
    /// no error, so the process exit code stays the success one. `resolved_stub()`
    /// carries `consult: None` (unlike `forced_resolved()` above), so this goes
    /// through the ordinary tool loop, where `dispatch_consult_through_gate`
    /// (`src/agent/mod.rs`) is the ONLY call site the complexity gate sees
    /// (REQ-A20). MS2 has not yet wired `[magi.complexity]` into `Resolved` (see
    /// the "Task 3.2" comment inside `run_query` above), so the gate here runs
    /// with its BUILT-IN thresholds — this also pins that the unconfigured
    /// default still vetoes trivial content.
    ///
    /// The canned `Magi` is real (not a stub that would silently succeed if
    /// called): `dispatch_consult_through_gate` only reaches
    /// `authorize_and_execute_tool` — the sole place the `RunObserver` records a
    /// `tool_calls` entry — on `GateVerdict::Dispatch`, never on a veto. So an
    /// absent "consult" entry in `outcome.tool_calls` is direct evidence the
    /// canned MAGI was never invoked, not an assumption.
    #[tokio::test]
    async fn a_vetoed_autonomous_consult_leaves_the_headless_run_at_exit_success() {
        let provider = ScriptedProvider::new(vec![
            Turn::Tool {
                id: "c1".to_string(),
                name: "consult".to_string(),
                input: json!({ "query": "trivial" }),
            },
            Turn::Text("answered without consensus".to_string()),
        ]);
        let mut agent = Agent::new(provider);
        agent.register_tool(Box::new(ConsultTool::new(canned_magi(), true)));

        let policy = Policy::new(Tier::Auto, 15, None);
        let outcome = run_query(
            resolved_stub(),
            policy,
            &mut agent,
            "please look into this",
            &wiring(None),
        )
        .await;

        assert_eq!(
            outcome.stop_reason,
            StopReason::Done,
            "a veto is not an error: the turn must finish normally"
        );
        assert!(
            outcome.error.is_none(),
            "a veto must never populate RunOutcome.error"
        );
        assert!(
            outcome.tool_calls.iter().all(|t| t.name != "consult"),
            "a vetoed consult never reaches authorize_and_execute_tool, so it must \
             not appear in the observer-recorded tool_calls — proof the canned \
             MAGI was never invoked"
        );
        assert_eq!(
            crate::exit_code_for_outcome(&outcome),
            0,
            "the process exit code must stay the success one"
        );
    }

    /// Behavioral proof (REQ-H36): a forced consult shares the run's SINGLE
    /// wall-clock budget automatically, because it is now a genuine in-loop
    /// tool call bounded by the SAME `tokio::time::timeout` as the rest of the
    /// loop — there is no separate forced-consult deadline to reconcile. The
    /// agent loop would finish fast, but the registered `consult` tool blocks
    /// far past `--timeout`, so the WHOLE run times out — `stop_reason = Error`
    /// / `error.kind = Timeout`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_forced_consult_shares_single_wall_clock_budget() {
        let provider = ScriptedProvider::new(vec![Turn::Text("fast loop answer".to_string())]);
        let mut agent = Agent::new(provider);

        // A MAGI whose analysis blocks far longer than the run's --timeout, so the
        // forced consult is guaranteed to exhaust the shared budget.
        let magi = slow_magi(Duration::from_secs(30));
        agent.register_tool(Box::new(ConsultTool::new(magi, true)));

        let started = Instant::now();
        let policy = Policy::new(Tier::Auto, 15, None);
        let outcome = run_query(
            forced_resolved(),
            policy,
            &mut agent,
            "decide X vs Y",
            &wiring(Some(Duration::from_millis(300))),
        )
        .await;
        let elapsed = started.elapsed();

        assert_eq!(
            outcome.stop_reason,
            StopReason::Error,
            "loop + forced consult over the shared budget ⇒ Error"
        );
        assert_eq!(
            outcome
                .error
                .expect("a wall-clock timeout populates the error payload")
                .kind,
            ErrorKind::Timeout,
            "the combined wall-clock overrun is a first-class Timeout"
        );
        assert!(
            outcome.consult.is_none(),
            "a timed-out forced consult yields no MAGI object"
        );
        // The forced consult was cut by the shared deadline, NOT run to its 30 s
        // completion: a generous bound that never flakes but rejects an unbounded
        // (or 2×) run.
        assert!(
            elapsed < Duration::from_secs(5),
            "the forced consult must be bounded by the deadline, not run to completion"
        );
    }
}
