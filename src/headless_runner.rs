// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-18

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
//! [`resolve_run_timeout`]; every item is reachable in the non-test build.

use std::collections::HashMap;
use std::future;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use magi_core::orchestrator::Magi;
use magi_core::schema::Mode;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use magi_rs::headless::log::{LogEvent, LogLevel, RunLog};
use magi_rs::headless::output::sanitize_error_message;
use magi_rs::headless::policy::{Policy, Tier};
use magi_rs::headless::resolution::Resolved;
use magi_rs::headless::types::{
    ErrorKind, ErrorPayload, RunOutcome, StopReason, Timings, ToolCallRecord, TranscriptEntry,
    Usage,
};

use crate::agent::messages::{Content, Message, Role};
use crate::agent::{Agent, AgentRunConfig, RunObserver, StreamPiece, MAX_TOOL_CALLS_ERROR};
use crate::tools::consult::{report_to_consult_json, MAX_QUERY_LEN};

/// Buffered capacity of the internal chunk channel; mirrors the interactive TUI
/// bridge so backpressure behaves identically. The channel is drained
/// concurrently, so this is a small smoothing buffer, not a bound on output.
const CHUNK_CHANNEL_CAPACITY: usize = 100;

/// Normalized transcript role for a real user turn (REQ-H14).
const ROLE_USER: &str = "user";

/// Normalized transcript role for an assistant turn (REQ-H14).
const ROLE_ASSISTANT: &str = "assistant";

/// `error.message` when a run is aborted by the wall-clock `--timeout` (REQ-H36).
const TIMEOUT_MESSAGE: &str = "run exceeded the --timeout wall-clock limit";

/// Registered name of the multi-perspective MAGI `consult` tool (REQ-H21/H22).
/// Single source of truth for the name matched when forcing/capturing a consult.
const CONSULT_TOOL: &str = "consult";

/// JSON input key the `consult` tool expects (mirrors `ConsultTool::input_schema`).
const CONSULT_QUERY_KEY: &str = "query";

/// Synthetic tool-use id of a runner-injected forced consult (REQ-H22). It never
/// correlates to an assistant `ToolUse` block, so it appears in `tool_calls` but
/// not in the transcript — the forced pass is a runner action, not a model turn.
const FORCED_CONSULT_ID: &str = "forced-consult";

/// `error.message` for a direct/forced consult whose prompt is empty or exceeds
/// [`MAX_QUERY_LEN`] (REQ-H33: reject, never truncate).
const CONSULT_INPUT_INVALID_MESSAGE: &str = "consult prompt is empty or exceeds the maximum length";

/// Recorded result of a forced consult that a tool-executing tier authorized but
/// that has no MAGI orchestrator wired (defensive; production always wires one).
const CONSULT_UNAVAILABLE_MESSAGE: &str = "MAGI orchestrator is not configured for consult";

/// Typed failure of a direct/forced consult, mapped to an [`ErrorKind`] by the
/// caller so an over-cap prompt becomes `input_invalid` (exit 2, REQ-H33) and a
/// wall-clock abort becomes `timeout` (exit 1, REQ-H36).
enum ConsultRunError {
    /// The prompt was empty or exceeded [`MAX_QUERY_LEN`] (→ `input_invalid`).
    InputInvalid,
    /// The consult was cancelled by the `--timeout` deadline (→ `timeout`).
    Timeout,
    /// The MAGI orchestrator failed or panicked; the message is sanitized before
    /// use (→ `runtime`).
    Runtime(String),
}

/// Builds the `consult` tool input object (`{"query": <prompt>}`) recorded for a
/// forced consult, matching the schema of the interactive `consult` tool.
fn consult_input(prompt: &str) -> Value {
    json!({ CONSULT_QUERY_KEY: prompt })
}

/// Runs `prompt` directly through the 3-perspective MAGI consensus, off the agent
/// tool-loop (REQ-H21), honoring the same input cap ([`MAX_QUERY_LEN`], REQ-H33)
/// and the `--timeout`/cancellation plumbing of the enclosing run (REQ-H36).
///
/// The `analyze` call runs on a joined task so a panic in `magi-core` surfaces as
/// a recoverable [`ConsultRunError::Runtime`] instead of unwinding the caller; on
/// cancellation or deadline the task is **aborted** (not merely detached) and
/// [`ConsultRunError::Timeout`] is returned.
///
/// # Parameters
/// - `magi` — shared MAGI orchestrator (the same one wired for the interactive
///   `consult` tool).
/// - `prompt` — the decision/content to analyze.
/// - `cancel` — cooperative cancellation fired by the enclosing run's deadline.
/// - `timeout` — optional wall-clock ceiling for this consult specifically.
///
/// # Errors
/// - [`ConsultRunError::InputInvalid`] if `prompt` is empty or exceeds
///   [`MAX_QUERY_LEN`].
/// - [`ConsultRunError::Timeout`] if cancelled or the deadline elapsed.
/// - [`ConsultRunError::Runtime`] if the MAGI analysis failed or panicked.
async fn analyze_direct(
    magi: &Arc<Magi>,
    prompt: &str,
    cancel: &CancellationToken,
    timeout: Option<Duration>,
) -> Result<Value, ConsultRunError> {
    if prompt.trim().is_empty() || prompt.len() > MAX_QUERY_LEN {
        return Err(ConsultRunError::InputInvalid);
    }

    let magi = Arc::clone(magi);
    let owned = prompt.to_string();
    let handle = tokio::spawn(async move { magi.analyze(&Mode::Analysis, &owned).await });
    let aborter = handle.abort_handle();

    // A `None` timeout parks forever, so the deadline arm never fires; the cancel
    // arm still aborts if the enclosing run is cancelled.
    let deadline = async {
        match timeout {
            Some(dur) => tokio::time::sleep(dur).await,
            None => future::pending::<()>().await,
        }
    };
    tokio::pin!(deadline);

    let joined = tokio::select! {
        biased;
        () = cancel.cancelled() => {
            aborter.abort();
            return Err(ConsultRunError::Timeout);
        }
        () = &mut deadline => {
            aborter.abort();
            return Err(ConsultRunError::Timeout);
        }
        joined = handle => joined,
    };

    match joined {
        Ok(Ok(report)) => Ok(report_to_consult_json(&report)),
        Ok(Err(e)) => Err(ConsultRunError::Runtime(e.to_string())),
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
fn extract_consult_value(calls: &[(String, ToolCallRecord)]) -> Option<Value> {
    calls
        .iter()
        .rev()
        .find(|(_, rec)| rec.name == CONSULT_TOOL && rec.ok)
        .and_then(|(_, rec)| serde_json::from_str::<Value>(&rec.result).ok())
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
}

/// Resolves the effective wall-clock timeout for a run from its tier policy
/// (REQ-H36), applying the tier default when none was explicitly configured.
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
pub fn resolve_run_timeout(policy: &Policy, full_auto_timeout_secs: u64) -> Option<Duration> {
    if let Some(secs) = policy.timeout() {
        return Some(Duration::from_secs(secs));
    }
    match policy.tier() {
        Tier::Auto | Tier::FullAuto => Some(Duration::from_secs(full_auto_timeout_secs)),
        Tier::Default => None,
    }
}

/// Wall-clock milliseconds elapsed since `start`, saturating instead of wrapping.
fn elapsed_ms(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// The wall-clock budget left for a post-loop forced consult (REQ-H36).
///
/// The run shares ONE `--timeout` across the agent loop **and** the forced
/// consult, so the consult runs against the time REMAINING after `elapsed`,
/// never a fresh `dur` — a fresh deadline here would let `loop + forced consult`
/// reach up to **2×** `dur`, weakening the wall-clock DoS ceiling that
/// `--auto`/`--full-auto` rely on. `None` (no timeout) stays unbounded; an
/// already-exhausted budget saturates to [`Duration::ZERO`], which the consult
/// treats as an immediate timeout.
fn forced_consult_budget(timeout: Option<Duration>, elapsed: Duration) -> Option<Duration> {
    timeout.map(|dur| dur.checked_sub(elapsed).unwrap_or(Duration::ZERO))
}

/// The `(response, stop_reason, error)` triple of a wall-clock timeout abort
/// (REQ-H36): no response, a first-class [`ErrorKind::Timeout`] payload. Shared
/// by the loop-timeout and forced-consult-timeout arms so the timeout outcome is
/// defined in exactly one place.
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

/// Writes `event` to `run_log` if one is attached; a log write failure is
/// swallowed (best-effort — logging must never abort or contaminate the run).
///
/// Takes `Option<&mut &mut RunLog>` so a caller holding `Option<&mut RunLog>`
/// can reborrow it repeatedly with `.as_mut()` (a plain `as_deref_mut()` would
/// be a no-op reborrow of the same type).
fn log_event(run_log: Option<&mut &mut RunLog>, event: &LogEvent<'_>) {
    if let Some(log) = run_log {
        let _ = log.event(event);
    }
}

/// Concatenates the `Text` blocks of `msg` into a single string (tool-use /
/// tool-result blocks contribute no text).
fn join_text(msg: &Message) -> String {
    msg.content
        .iter()
        .filter_map(|c| match c {
            Content::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
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
                // A `User` message is either a real user turn (Text) or a
                // tool-result carrier (ToolResult); fold the latter away.
                let is_tool_result = msg
                    .content
                    .iter()
                    .any(|c| matches!(c, Content::ToolResult { .. }));
                if is_tool_result {
                    continue;
                }
                transcript.push(TranscriptEntry {
                    role: ROLE_USER.to_string(),
                    content: join_text(msg),
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
                    content: join_text(msg),
                    tool_calls,
                });
            }
        }
    }
    transcript
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
/// An over-cap prompt (`prompt.len() > MAX_QUERY_LEN`) or an empty prompt is
/// rejected with `error.kind = input_invalid` (REQ-H33: reject, never truncate),
/// which the caller (Task 7) maps to exit 2. A `--timeout` abort yields
/// `error.kind = timeout` (REQ-H36); a MAGI failure yields a sanitized `runtime`.
///
/// # Parameters
/// - `resolved` — effective run parameters; supplies `model`/`provider`/
///   `applied_caps` for the output.
/// - `magi` — shared MAGI orchestrator (same one wired for the `consult` tool).
/// - `prompt` — the decision/content to analyze.
/// - `timeout` — optional wall-clock ceiling (REQ-H36).
/// - `run_log` — optional JSONL run log; the terminal summary is recorded
///   best-effort.
///
/// # Gaps (documented, never fabricated)
/// - `usage` is `Usage { 0, 0 }`: `magi-core` does not surface token counts here.
/// - `timings.per_turn_ms` is empty and `ttfb_ms` is `None`: the direct consult is
///   a single buffered analysis, not a streamed turn sequence.
pub async fn run_consult(
    resolved: Resolved,
    magi: Arc<Magi>,
    prompt: &str,
    timeout: Option<Duration>,
    mut run_log: Option<&mut RunLog>,
) -> RunOutcome {
    // A fresh token: the direct consult has no enclosing agent run to inherit a
    // cancellation from, so `analyze_direct`'s cancel arm only fires if this run's
    // own `timeout` deadline elapses.
    let cancel = CancellationToken::new();
    let run_start = Instant::now();
    let result = analyze_direct(&magi, prompt, &cancel, timeout).await;
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

    let summary = format!("consult complete: stop_reason={stop_reason:?}");
    log_event(
        run_log.as_mut(),
        &LogEvent::Message {
            level: LogLevel::Info,
            text: &summary,
        },
    );

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
        applied_caps: resolved.applied_caps,
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
/// - the cap being hit ⇒ [`StopReason::MaxToolCalls`] (a terminal state, not an
///   error — no `error` payload, maps to exit 0);
/// - any other run error (including a repetitive-guard abort) ⇒
///   [`StopReason::Error`] with a sanitized `error` payload;
/// - otherwise, at least one tier denial **and** an empty final turn (zero
///   `TextDelta` blocks) ⇒ [`StopReason::Denied`]; else [`StopReason::Done`].
///
/// # Parameters
/// - `resolved` — effective run parameters; supplies the output `model`,
///   `provider` and `applied_caps` (the cap actually enforced comes from
///   `policy`).
/// - `policy` — tier policy driving authorization, the tool-call cap and the
///   soft-guard silencing.
/// - `agent` — a constructed agent (provider + registered tools).
/// - `prompt` — the resolved user prompt.
/// - `timeout` — optional wall-clock ceiling for the whole run (REQ-H36). `None`
///   ⇒ no timeout (the interactive default). On elapse the agent future is
///   dropped — cancelling the in-flight LLM stream — the run's
///   [`CancellationToken`] is fired so any in-flight `bash` subprocess *tree* is
///   killed (its group killer drops with the future), and a partial outcome with
///   `stop_reason = Error` / `error.kind = Timeout` is returned. The tier default
///   (900 s for `--auto`/`--full-auto`) is selected by the caller (Task 7) and
///   passed here; use [`resolve_run_timeout`] to apply it.
/// - `consult_magi` — MAGI orchestrator used to satisfy a **forced** consult
///   (`resolved.consult == Some(true)`, REQ-H22). `None` disables forcing (the
///   agent may still call the `consult` tool proactively under `--auto`+).
/// - `run_log` — optional JSONL run log; tier warnings and each tool call are
///   recorded best-effort.
///
/// # Forced consult (REQ-H22)
/// When `resolved.consult == Some(true)` and the run did not time out, the runner
/// guarantees **exactly one** consult: if the agent already invoked the `consult`
/// tool during the loop, that call is reused (never re-fired); otherwise the
/// runner injects a single consult through the **same tier gate** (the observer's
/// [`RunObserver::authorize`]), so the tier wins — in `default` the forced consult
/// is recorded denied (`ok = false`) and `consult` stays `None`, never elevated
/// out of the tier. Whichever consult succeeded (forced or proactive) populates
/// `RunOutcome.consult`. The injected consult shares the run's **single**
/// wall-clock budget: it runs against the time remaining after the agent loop
/// ([`forced_consult_budget`]), so `loop + forced consult` never exceeds one
/// `dur`; if that remaining budget elapses the whole run reports a `Timeout`
/// (REQ-H36).
///
/// **Trade-off (reported):** the injected forced consult runs as a post-loop pass
/// through the tracker rather than literally inside the provider-driven loop —
/// forcing an extra tool call *inside* the loop would require invasive changes to
/// `Agent` that risk the interactive path and the deterministic `stop_reason`
/// accounting. It still flows through the identical authorization/recording path,
/// so tier gating, `tier_denied` accounting and the audit record are faithful;
/// its only divergence is that it is not fed back into the agent's answer and does
/// not appear in `transcript` (it is a runner action, recorded in `tool_calls`).
///
/// # Gaps (documented, never fabricated)
/// - `usage` is `Usage { 0, 0 }`: the agent/provider do not surface token counts
///   to this layer.
/// - `timings.per_turn_ms` is empty: turn boundaries are not observable from
///   outside the loop. `total_ms` and (best-effort) `ttfb_ms` are measured.
pub async fn run_query(
    resolved: Resolved,
    policy: Policy,
    agent: &mut Agent,
    prompt: &str,
    timeout: Option<Duration>,
    consult_magi: Option<Arc<Magi>>,
    mut run_log: Option<&mut RunLog>,
) -> RunOutcome {
    // Tier warnings (only under --full-auto) are recorded up front.
    for warning in policy.warnings() {
        log_event(
            run_log.as_mut(),
            &LogEvent::Message {
                level: LogLevel::Warn,
                text: &warning,
            },
        );
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
    let config = AgentRunConfig {
        max_tool_calls: usize::try_from(policy.max_tool_calls()).unwrap_or(usize::MAX),
        disable_repetitive_guard: policy.silences_soft_guards(),
        observer: Some(Arc::clone(&tracker) as Arc<dyn RunObserver>),
        cancel: cancel.clone(),
        system: (!system.is_empty()).then(|| system.to_string()),
    };

    let (chunk_tx, mut chunk_rx) =
        tokio::sync::mpsc::channel::<StreamPiece>(CHUNK_CHANNEL_CAPACITY);

    let run_start = Instant::now();
    // Drain the stream concurrently: measure time-to-first-byte and prevent the
    // agent from blocking on channel backpressure. The task ends when
    // `query_streaming` drops `chunk_tx`.
    let drain = tokio::spawn(async move {
        let mut ttfb_ms: Option<u64> = None;
        while let Some(piece) = chunk_rx.recv().await {
            if ttfb_ms.is_none() {
                if let StreamPiece::Content(_) = piece {
                    ttfb_ms = Some(elapsed_ms(run_start));
                }
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
    let ttfb_ms = drain.await.ok().flatten();

    // Forced consult (REQ-H22): only when requested AND the run actually ran (a
    // timeout dropped the loop mid-flight ⇒ nothing to force onto). Injected
    // through the SAME tracker so its authorization, `tier_denied` accounting and
    // audit record are identical to a loop tool call — the tier gate wins.
    //
    // It shares the run's SINGLE wall-clock budget (REQ-H36): the consult runs
    // against the time REMAINING after the agent loop, not a fresh `dur`, so
    // `loop + forced consult` can never exceed one `dur`. If that remaining
    // budget elapses the consult reports a timeout, which dominates the outcome
    // mapping below exactly like a loop timeout.
    let mut forced_consult_timed_out = false;
    if resolved.consult == Some(true) && outcome_result.is_some() {
        let remaining = forced_consult_budget(timeout, run_start.elapsed());
        forced_consult_timed_out =
            force_consult_once(&tracker, consult_magi.as_ref(), prompt, &cancel, remaining).await;
    }

    let total_ms = elapsed_ms(run_start);

    // Snapshot the observer state once (the run is over; no more callbacks).
    let (calls, tier_denied, response_empty) = with_state(&tracker.state, |s| {
        (
            s.calls.clone(),
            s.tier_denials > 0,
            s.final_turn_text_blocks == 0,
        )
    });
    let tool_calls: Vec<ToolCallRecord> = calls.iter().map(|(_, rec)| rec.clone()).collect();
    let records_by_id: HashMap<&str, &ToolCallRecord> =
        calls.iter().map(|(id, rec)| (id.as_str(), rec)).collect();
    let transcript = build_transcript(agent.history(), &records_by_id);

    // Deterministic outcome mapping (priority Error > MaxToolCalls > Denied >
    // Done); a wall-clock timeout is a first-class `Error` (kind = timeout) and
    // dominates every other signal because it dropped the run mid-flight. A
    // forced consult that exhausted the shared budget is the same wall-clock
    // timeout and produces the identical outcome (REQ-H36).
    let (response, stop_reason, error) = match outcome_result {
        None => timeout_outcome(),
        Some(_) if forced_consult_timed_out => timeout_outcome(),
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
        let input_str = record.input.to_string();
        log_event(
            run_log.as_mut(),
            &LogEvent::ToolCall {
                level: LogLevel::Info,
                name: &record.name,
                ok: record.ok,
                ms: record.ms,
                input: &input_str,
            },
        );
    }
    let summary = format!("run complete: stop_reason={stop_reason:?}");
    log_event(
        run_log.as_mut(),
        &LogEvent::Message {
            level: LogLevel::Info,
            text: &summary,
        },
    );

    RunOutcome {
        response,
        model: resolved.model,
        provider: resolved.provider,
        // Gap: the agent/provider do not surface token counts to this layer.
        usage: Usage {
            input_tokens: 0,
            output_tokens: 0,
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
        // `--auto`+ call); `None` when none ran or it was denied by the tier.
        consult: extract_consult_value(&calls),
        applied_caps: resolved.applied_caps,
        error,
    }
}

/// Ensures **exactly one** consult for a forced run (REQ-H22), recorded through
/// `tracker` so the tier gate and accounting match a loop tool call.
///
/// No-op if the agent already invoked `consult` during the loop (the existing
/// call is reused — never re-fired). Otherwise the forced consult is authorized
/// via [`RunObserver::authorize`] — in `default` this denies and records
/// `ok = false` (the tier wins, `consult` stays `None`); in `--auto`+ it runs the
/// direct analysis and records the result. A `None` `magi` under an authorizing
/// tier records an `ok = false` "not configured" call (defensive; production
/// always wires the orchestrator when consult is possible).
///
/// # Returns
/// `true` iff the forced consult was cut short by the wall-clock `timeout`
/// (`ConsultRunError::Timeout`); the caller promotes that to the run-level
/// timeout outcome (REQ-H36). Every other path — no-op reuse, tier denial,
/// missing orchestrator, input-invalid, runtime failure — returns `false`.
async fn force_consult_once(
    tracker: &Arc<RunTracker>,
    magi: Option<&Arc<Magi>>,
    prompt: &str,
    cancel: &CancellationToken,
    timeout: Option<Duration>,
) -> bool {
    let already_consulted = with_state(&tracker.state, |s| {
        s.calls.iter().any(|(_, r)| r.name == CONSULT_TOOL)
    });
    if already_consulted {
        return false;
    }

    let input = consult_input(prompt);
    if !tracker.authorize(CONSULT_TOOL) {
        // Tier denial: recorded exactly like a loop denial; `consult` stays None.
        let denial = format!(
            "Tool '{CONSULT_TOOL}' denied: not authorized in the current authorization tier"
        );
        tracker.on_tool_call(FORCED_CONSULT_ID, CONSULT_TOOL, &input, &denial, false, 0);
        return false;
    }

    let Some(magi) = magi else {
        tracker.on_tool_call(
            FORCED_CONSULT_ID,
            CONSULT_TOOL,
            &input,
            CONSULT_UNAVAILABLE_MESSAGE,
            false,
            0,
        );
        return false;
    };

    let started = Instant::now();
    let (result, ok, timed_out) = match analyze_direct(magi, prompt, cancel, timeout).await {
        Ok(value) => (value.to_string(), true, false),
        Err(ConsultRunError::InputInvalid) => {
            (CONSULT_INPUT_INVALID_MESSAGE.to_string(), false, false)
        }
        Err(ConsultRunError::Timeout) => (TIMEOUT_MESSAGE.to_string(), false, true),
        Err(ConsultRunError::Runtime(message)) => (sanitize_error_message(&message), false, false),
    };
    let ms = elapsed_ms(started);
    tracker.on_tool_call(FORCED_CONSULT_ID, CONSULT_TOOL, &input, &result, ok, ms);
    timed_out
}

#[cfg(test)]
mod tests {
    use super::*;

    use anyhow::Result;
    use async_trait::async_trait;
    use futures::stream::{self, BoxStream};
    use magi_rs::headless::limits::FULL_AUTO_TIMEOUT_SECS;
    use serde_json::{json, Value};
    use std::collections::VecDeque;

    use magi_core::schema::AgentName;
    use magi_core::test_support::RoutingMockProvider;

    use magi_rs::headless::policy::Tier;
    use magi_rs::headless::types::{AppliedCaps, SystemPolicy};

    use crate::agent::provider::{Provider, ResponseChunk};
    use crate::tools::consult::ConsultTool;
    use crate::tools::{Tool, ToolResult};

    /// A canned MAGI orchestrator whose three perspectives all approve, over a
    /// `RoutingMockProvider` (no network) — deterministic, mirrors the double used
    /// in `consult.rs` so a direct/forced consult yields a non-degraded report.
    fn canned_magi() -> Arc<Magi> {
        fn agent_json(agent: &str) -> String {
            format!(
                r#"{{"agent":"{agent}","verdict":"approve","confidence":0.9,"summary":"s","reasoning":"r","findings":[],"recommendation":"rec"}}"#
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
            ) -> Result<String, ProviderError> {
                tokio::time::sleep(self.delay).await;
                Ok(r#"{"agent":"melchior","verdict":"approve","confidence":0.9,"summary":"s","reasoning":"r","findings":[],"recommendation":"rec"}"#.to_string())
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
        /// Emit streamed text plus a matching `MessageDone` (terminal turn).
        Text(String),
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
                Some(Turn::Text(text)) => Ok(Box::pin(stream::iter(vec![
                    Ok(ResponseChunk::TextDelta(text.clone())),
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
            },
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
        let outcome = run_query(
            resolved_stub(),
            policy,
            &mut agent,
            "prompt",
            None,
            None,
            None,
        )
        .await;

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
        let outcome = run_query(
            resolved_stub(),
            policy,
            &mut agent,
            "prompt",
            None,
            None,
            None,
        )
        .await;

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
        let outcome = run_query(
            resolved_stub(),
            policy,
            &mut agent,
            "prompt",
            None,
            None,
            None,
        )
        .await;

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
        let outcome = run_query(
            resolved_stub(),
            policy,
            &mut agent,
            "prompt",
            None,
            None,
            None,
        )
        .await;

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
        let outcome = run_query(
            resolved_stub(),
            policy,
            &mut agent,
            "prompt",
            None,
            None,
            None,
        )
        .await;

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
        let outcome = run_query(
            resolved_stub(),
            policy,
            &mut agent,
            "prompt",
            None,
            None,
            None,
        )
        .await;

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
        let outcome = run_query(
            resolved_stub(),
            policy,
            &mut agent,
            "prompt",
            None,
            None,
            None,
        )
        .await;

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
        let outcome = run_query(
            resolved_stub(),
            policy,
            &mut agent,
            "prompt",
            None,
            None,
            None,
        )
        .await;

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
        let outcome = run_query(
            resolved_stub(),
            policy,
            &mut agent,
            "prompt",
            None,
            None,
            None,
        )
        .await;

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
        let outcome = run_query(
            resolved_stub(),
            policy,
            &mut agent,
            "prompt",
            None,
            None,
            None,
        )
        .await;

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

    // ── REQ-H36: wall-clock timeout ─────────────────────────────────────────────

    /// `resolve_run_timeout` applies the tier default: the read-only `default`
    /// tier gets no ceiling, while the tool-executing tiers get the 900 s hard
    /// default when none was configured; an explicit config value always wins.
    #[test]
    fn test_resolve_run_timeout_applies_tier_default() {
        // No configured timeout: default tier ⇒ None; Auto/FullAuto ⇒ 900 s.
        assert_eq!(
            resolve_run_timeout(
                &Policy::new(Tier::Default, 15, None),
                FULL_AUTO_TIMEOUT_SECS
            ),
            None,
            "the read-only default tier carries no wall-clock ceiling"
        );
        assert_eq!(
            resolve_run_timeout(&Policy::new(Tier::Auto, 15, None), FULL_AUTO_TIMEOUT_SECS),
            Some(Duration::from_secs(FULL_AUTO_TIMEOUT_SECS))
        );
        assert_eq!(
            resolve_run_timeout(
                &Policy::new(Tier::FullAuto, 50, None),
                FULL_AUTO_TIMEOUT_SECS
            ),
            Some(Duration::from_secs(FULL_AUTO_TIMEOUT_SECS))
        );
        // An explicit configured timeout wins over the tier default, in any tier.
        assert_eq!(
            resolve_run_timeout(
                &Policy::new(Tier::Auto, 15, Some(5)),
                FULL_AUTO_TIMEOUT_SECS
            ),
            Some(Duration::from_secs(5))
        );
        assert_eq!(
            resolve_run_timeout(
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
    fn test_resolve_run_timeout_respects_custom_effective_full_auto_default() {
        let custom_default = 42u64;
        assert_ne!(
            custom_default, FULL_AUTO_TIMEOUT_SECS,
            "fixture must differ from the module constant to prove it is not used"
        );

        assert_eq!(
            resolve_run_timeout(&Policy::new(Tier::Auto, 15, None), custom_default),
            Some(Duration::from_secs(custom_default)),
            "a custom (smaller) effective default must apply, not the module constant"
        );
        assert_eq!(
            resolve_run_timeout(&Policy::new(Tier::FullAuto, 50, None), custom_default),
            Some(Duration::from_secs(custom_default))
        );
        // The read-only tier is still unbounded regardless of the effective default.
        assert_eq!(
            resolve_run_timeout(&Policy::new(Tier::Default, 15, None), custom_default),
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
        let outcome = run_query(
            resolved_stub(),
            policy,
            &mut agent,
            "prompt",
            None,
            None,
            None,
        )
        .await;

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
            python_available, tree_kill_worker, CANCEL_FIRE_DELAY_MS, POST_KILL_WAIT_MS,
        };

        if !python_available() {
            eprintln!("skipping: python interpreter not found — cannot spawn a real child");
            return;
        }
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().canonicalize().expect("canonicalize");
        std::fs::write(root.join("worker.py"), tree_kill_worker()).expect("write worker");

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
            Some(Duration::from_millis(CANCEL_FIRE_DELAY_MS)),
            None,
            None,
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

        // Prove the tree died: wait past when a surviving orphan would write
        // DONE — see `POST_KILL_WAIT_MS` rustdoc for the margin math.
        let done = root.join("done.marker");
        tokio::time::sleep(Duration::from_millis(POST_KILL_WAIT_MS)).await;
        assert!(
            root.join("start.marker").exists(),
            "the child really ran (START present)"
        );
        assert!(
            !done.exists(),
            "the bash subprocess tree must be dead — no DONE marker"
        );
    }

    // ── REQ-H21/H22: consult (direct + forced) ─────────────────────────────────

    /// `magi consult` direct path: the 3 perspectives run off the agent loop and
    /// `RunOutcome.consult` is the (non-null) MAGI object; `response` is the report
    /// and no tool calls are recorded (REQ-H21).
    #[tokio::test]
    async fn test_run_consult_direct_runs_three_perspectives_and_populates_consult() {
        let outcome = run_consult(
            resolved_stub(),
            canned_magi(),
            "should we migrate X to Y?",
            None,
            None,
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

    /// `magi consult` over `MAX_QUERY_LEN` is rejected as `input_invalid` (exit 2),
    /// NOT truncated: `consult`/`response` stay `None` (REQ-H33).
    #[tokio::test]
    async fn test_run_consult_over_cap_is_input_invalid_not_truncated() {
        let big = "x".repeat(MAX_QUERY_LEN + 1);
        let outcome = run_consult(resolved_stub(), canned_magi(), &big, None, None).await;

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

    /// `query --consult` in `--auto` invokes consult **exactly once** and populates
    /// `RunOutcome.consult`, even though the agent itself did not consult (REQ-H22).
    #[tokio::test]
    async fn test_query_forced_consult_auto_runs_exactly_once_and_populates() {
        let provider = ScriptedProvider::new(vec![Turn::Text("here is my answer".to_string())]);
        let mut agent = Agent::new(provider);

        let policy = Policy::new(Tier::Auto, 15, None);
        let outcome = run_query(
            forced_resolved(),
            policy,
            &mut agent,
            "decide X vs Y",
            None,
            Some(canned_magi()),
            None,
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
    /// it is NOT elevated out of the read-only tier (REQ-H22 precedence).
    #[tokio::test]
    async fn test_query_forced_consult_default_denied_and_not_elevated() {
        let provider = ScriptedProvider::new(vec![Turn::Text("read-only answer".to_string())]);
        let mut agent = Agent::new(provider);

        let policy = Policy::new(Tier::Default, 15, None);
        let outcome = run_query(
            forced_resolved(),
            policy,
            &mut agent,
            "decide X vs Y",
            None,
            Some(canned_magi()),
            None,
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

    /// A forced consult does NOT double-fire: when the agent already invoked the
    /// `consult` tool during the loop, exactly one consult call is recorded and no
    /// second (injected) pass runs (REQ-H22 exactly-once).
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
        // A real ConsultTool over a canned MAGI so the agent's proactive call
        // executes and is recorded (auto_approve is moot — the observer authorizes).
        agent.register_tool(Box::new(ConsultTool::new(canned_magi(), true)));

        let policy = Policy::new(Tier::Auto, 15, None);
        let outcome = run_query(
            forced_resolved(),
            policy,
            &mut agent,
            "decide X vs Y",
            None,
            Some(canned_magi()),
            None,
        )
        .await;

        let consult_calls = outcome
            .tool_calls
            .iter()
            .filter(|t| t.name == "consult")
            .count();
        assert_eq!(
            consult_calls, 1,
            "the agent's own consult is reused — the forced pass does not re-fire"
        );
        assert!(
            outcome.consult.is_some(),
            "the single (proactive) consult populates RunOutcome.consult"
        );
    }

    /// Structural proof of the single-deadline bound (REQ-H36): the forced consult
    /// gets the budget REMAINING after the loop, never a fresh `dur`. Against the
    /// pre-fix code — which handed the forced consult the full original `timeout` —
    /// `loop + forced consult` could reach 2× `dur`; this arithmetic rejects that.
    #[test]
    fn test_forced_consult_budget_is_remaining_not_fresh() {
        // 1000 ms budget, 600 ms already spent by the loop ⇒ only 400 ms left
        // (NOT a fresh 1000 ms).
        assert_eq!(
            forced_consult_budget(
                Some(Duration::from_millis(1000)),
                Duration::from_millis(600)
            ),
            Some(Duration::from_millis(400)),
            "the forced consult may only use the time remaining after the loop"
        );
        // An already-exhausted budget saturates to zero (immediate timeout),
        // never underflows.
        assert_eq!(
            forced_consult_budget(Some(Duration::from_millis(500)), Duration::from_millis(900)),
            Some(Duration::ZERO),
            "an over-budget loop leaves zero for the consult, not a wrapped value"
        );
        // No timeout ⇒ unbounded (the interactive/None path is unaffected).
        assert_eq!(
            forced_consult_budget(None, Duration::from_millis(900)),
            None,
            "a None timeout stays unbounded for the forced consult too"
        );
    }

    /// Behavioral proof (REQ-H36): a forced consult shares the run's SINGLE
    /// wall-clock budget. The agent loop finishes fast, but the forced consult
    /// would block far past the remaining budget, so the run times out as a whole
    /// — `stop_reason = Error` / `error.kind = Timeout`. Against the pre-fix code
    /// (a fresh deadline whose consult timeout was NOT propagated) this same run
    /// reported `Done`, so this test fails on the old 2×-budget code.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_forced_consult_shares_single_wall_clock_budget() {
        let provider = ScriptedProvider::new(vec![Turn::Text("fast loop answer".to_string())]);
        let mut agent = Agent::new(provider);

        // A MAGI whose analysis blocks far longer than the run's --timeout, so the
        // forced consult is guaranteed to exhaust the shared budget.
        let magi = slow_magi(Duration::from_secs(30));

        let started = Instant::now();
        let policy = Policy::new(Tier::Auto, 15, None);
        let outcome = run_query(
            forced_resolved(),
            policy,
            &mut agent,
            "decide X vs Y",
            Some(Duration::from_millis(300)),
            Some(magi),
            None,
        )
        .await;
        let elapsed = started.elapsed();

        assert_eq!(
            outcome.stop_reason,
            StopReason::Error,
            "loop + forced consult over the shared budget ⇒ Error (pre-fix: Done)"
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
