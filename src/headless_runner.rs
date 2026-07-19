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
//! Staged rollout: this module is fully implemented and exercised by its own
//! tests, but its production caller — the `magi query` / `magi consult`
//! subcommand dispatch in `main.rs` — lands in MS2 Task 6. Until then the plain
//! (non-test) binary build has no live path into it, so `dead_code` is allowed
//! **only** for `not(test)`; the test build reaches every item.

#![cfg_attr(not(test), allow(dead_code))]

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use magi_rs::headless::log::{LogEvent, LogLevel, RunLog};
use magi_rs::headless::output::sanitize_error_message;
use magi_rs::headless::policy::Policy;
use magi_rs::headless::resolution::Resolved;
use magi_rs::headless::types::{
    ErrorKind, ErrorPayload, RunOutcome, StopReason, Timings, ToolCallRecord, TranscriptEntry,
    Usage,
};

use crate::agent::messages::{Content, Message, Role};
use crate::agent::{Agent, AgentRunConfig, RunObserver, StreamPiece, MAX_TOOL_CALLS_ERROR};

/// Buffered capacity of the internal chunk channel; mirrors the interactive TUI
/// bridge so backpressure behaves identically. The channel is drained
/// concurrently, so this is a small smoothing buffer, not a bound on output.
const CHUNK_CHANNEL_CAPACITY: usize = 100;

/// Normalized transcript role for a real user turn (REQ-H14).
const ROLE_USER: &str = "user";

/// Normalized transcript role for an assistant turn (REQ-H14).
const ROLE_ASSISTANT: &str = "assistant";

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

/// Wall-clock milliseconds elapsed since `start`, saturating instead of wrapping.
fn elapsed_ms(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)
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
/// - `run_log` — optional JSONL run log; tier warnings and each tool call are
///   recorded best-effort.
///
/// # Gaps (documented, never fabricated)
/// - `usage` is `Usage { 0, 0 }`: the agent/provider do not surface token counts
///   to this layer.
/// - `timings.per_turn_ms` is empty: turn boundaries are not observable from
///   outside the loop. `total_ms` and (best-effort) `ttfb_ms` are measured.
/// - `consult` is `None` (forced consult is a later task).
pub async fn run_query(
    resolved: Resolved,
    policy: Policy,
    agent: &mut Agent,
    prompt: &str,
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
    let config = AgentRunConfig {
        max_tool_calls: usize::try_from(policy.max_tool_calls()).unwrap_or(usize::MAX),
        disable_repetitive_guard: policy.silences_soft_guards(),
        observer: Some(Arc::clone(&tracker) as Arc<dyn RunObserver>),
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

    let result = agent.query_streaming(prompt, chunk_tx, config).await;
    let ttfb_ms = drain.await.ok().flatten();
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

    // Deterministic outcome mapping (priority Error > MaxToolCalls > Denied > Done).
    let (response, stop_reason, error) = match &result {
        Ok(text) => {
            let stop_reason = if tier_denied && response_empty {
                StopReason::Denied
            } else {
                StopReason::Done
            };
            (Some(text.clone()), stop_reason, None)
        }
        Err(e) => {
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
        // Forced consult is a later task; a proactive consult (in --auto+) is
        // still recorded in `tool_calls` like any other tool.
        consult: None,
        applied_caps: resolved.applied_caps,
        error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use anyhow::Result;
    use async_trait::async_trait;
    use futures::stream::{self, BoxStream};
    use serde_json::{json, Value};
    use std::collections::VecDeque;

    use magi_rs::headless::policy::Tier;
    use magi_rs::headless::types::{AppliedCaps, SystemPolicy};

    use crate::agent::provider::{Provider, ResponseChunk};
    use crate::tools::{Tool, ToolResult};

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
        async fn execute(&self, _args: Value) -> ToolResult<Value> {
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
        let provider =
            ScriptedProvider::new(vec![tool_turn("t1", "edit"), Turn::Text("answered anyway".to_string())]);
        let mut agent = Agent::new(provider);
        register_echo(&mut agent, "edit");

        let policy = Policy::new(Tier::Default, 15, None);
        let outcome = run_query(resolved_stub(), policy, &mut agent, "prompt", None).await;

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
        let outcome = run_query(resolved_stub(), policy, &mut agent, "prompt", None).await;

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
        let outcome = run_query(resolved_stub(), policy, &mut agent, "prompt", None).await;

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
        let outcome = run_query(resolved_stub(), policy, &mut agent, "prompt", None).await;

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
        let outcome = run_query(resolved_stub(), policy, &mut agent, "prompt", None).await;

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
        let outcome = run_query(resolved_stub(), policy, &mut agent, "prompt", None).await;

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
        let outcome = run_query(resolved_stub(), policy, &mut agent, "prompt", None).await;

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
        let outcome = run_query(resolved_stub(), policy, &mut agent, "prompt", None).await;

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
        let outcome = run_query(resolved_stub(), policy, &mut agent, "prompt", None).await;

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
        let provider =
            ScriptedProvider::new(vec![tool_turn("call-1", "ls"), Turn::Text("here you go".to_string())]);
        let mut agent = Agent::new(provider);
        register_echo(&mut agent, "ls");

        let policy = Policy::new(Tier::Auto, 15, None);
        let outcome = run_query(resolved_stub(), policy, &mut agent, "prompt", None).await;

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
}
