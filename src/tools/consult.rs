// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-06-07

//! Tool that wraps `magi_core::Magi` to run 3-perspective consensus queries.
//! The agent routes here only for genuine multi-perspective decisions; trivial
//! or factual lookups are answered directly.

use crate::tools::{Tool, ToolError, ToolResult};
use async_trait::async_trait;
use magi_core::orchestrator::Magi;
use magi_core::reporting::MagiReport;
use magi_core::schema::Mode;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// Reject oversized consult input before incurring 3 model calls.
/// `pub(crate)` so the forced `/consult` TUI path and the headless direct/forced
/// consult path ([`crate::headless_runner`]) apply the same cap (REQ-H33).
pub(crate) const MAX_QUERY_LEN: usize = 8192;

/// Builds the stable `consult` JSON object from a finished MAGI report.
///
/// Single source of truth for the `{report, degraded}` shape shared by the
/// tool-loop [`ConsultTool::execute`] path and the headless direct/forced
/// consult path (REQ-H21/H22) — so the on-the-wire shape never drifts between
/// the two entry points.
///
/// # Parameters
/// * `report` - The finished multi-perspective consensus report.
///
/// # Returns
/// A JSON object `{"report": <markdown>, "degraded": <bool>}`.
pub(crate) fn report_to_consult_json(report: &MagiReport) -> Value {
    json!({ "report": report.report, "degraded": report.degraded })
}

/// RAII backstop that aborts a spawned task when the guard is dropped.
///
/// [`ConsultTool::execute`] runs the 3-perspective analysis on a `tokio::spawn`
/// task and awaits it under a `select!`. The explicit cancel arm aborts the
/// task on `--timeout`, but if the `execute` future itself is *dropped* before
/// either arm resolves (e.g. the caller drops the tool call), a bare spawned
/// task would keep running and orphan its three in-flight LLM calls. Holding
/// this guard across the `select!` aborts the task on that drop too, mirroring
/// the `GroupKiller` backstop the `bash` tool uses for its subprocess.
///
/// `pub(crate)` so [`crate::headless_runner`]'s direct `magi consult` path
/// (`analyze_direct`) reuses this exact primitive for its own spawned MAGI
/// analysis rather than duplicating it — same gap, same fix, one guard type.
pub(crate) struct AbortOnDrop {
    /// Abort handle of the guarded task.
    handle: tokio::task::AbortHandle,
}

impl AbortOnDrop {
    /// Wraps a task's abort handle so dropping the guard aborts the task.
    pub(crate) fn new(handle: tokio::task::AbortHandle) -> Self {
        Self { handle }
    }

    /// Aborts the guarded task now. Idempotent: aborting an already-finished or
    /// already-aborted task is a no-op, so `Drop` re-invoking it is harmless.
    pub(crate) fn abort(&self) {
        self.handle.abort();
    }
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

/// Notice emitted in the TUI when the `consult` tool is auto-approved.
/// Visible to the user so they know the 3-LLM consensus was launched.
const AUTO_LAUNCH_NOTICE: &str = "launched MAGI multi-perspective consensus — awaiting evaluation…";

/// Tool wrapping a `magi_core::Magi`. `execute` runs the 3-perspective consensus
/// (implemented in Task 4) and returns the verbatim report. The `description` is
/// what makes the main LLM self-route here only for multi-perspective decisions.
pub struct ConsultTool {
    magi: Arc<Magi>,
    description: String,
    /// When `true`, autonomous MAGI launches via the agent tool loop are
    /// auto-approved (no `ApprovalRequest` emitted). The explicit `/consult`
    /// TUI command path is NEVER gated regardless of this flag.
    auto_approve: bool,
}

impl ConsultTool {
    /// Creates a `ConsultTool` over a shared `Magi` orchestrator.
    ///
    /// # Parameters
    /// * `magi` - Shared `Magi` orchestrator that drives the 3-perspective consensus.
    /// * `auto_approve` - When `true`, the tool opts out of the approval gate for
    ///   autonomous launches (the agent tool loop will auto-approve it and emit a
    ///   TUI notice). Default is `false` — the agent asks before each launch.
    ///
    /// # Returns
    /// A new `ConsultTool` instance with a routing-tuned description.
    pub fn new(magi: Arc<Magi>, auto_approve: bool) -> Self {
        Self {
            magi,
            description: "Run a multi-perspective MAGI consensus (three independent \
                analyst agents) on a hard decision. Use ONLY for questions with genuine \
                trade-offs, design/architecture choices, or 'should we X vs Y given these \
                constraints?' decisions where a single answer is risky. Do NOT use for \
                trivial, factual, or lookup questions — answer those directly."
                .to_string(),
            auto_approve,
        }
    }
}

#[async_trait]
impl Tool for ConsultTool {
    fn name(&self) -> &str {
        "consult"
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The decision or content to analyze from three perspectives."
                },
                "mode": {
                    "type": "string",
                    "enum": ["code-review", "design", "analysis"],
                    "description": "Optional lens for the analysis (pick the one that \
                        matches what you're asking about). Omit to let the caller's \
                        configured/inferred lens apply instead."
                }
            },
            "required": ["query"]
        })
    }

    /// When `auto_approve = false` (the default), autonomous MAGI launches are
    /// gated — the agent prompts the user before each 3-LLM consensus call.
    /// When `auto_approve = true`, the agent tool loop auto-approves the call
    /// and emits an [`Self::approval_notice`] in the TUI instead.
    fn requires_approval(&self) -> bool {
        !self.auto_approve
    }

    /// Returns an announcement notice when the tool is auto-approved.
    ///
    /// The notice is sent as a `StreamPiece::Notice` **before** the tool runs,
    /// so the user knows the 3-LLM consensus was launched without a prompt.
    /// Returns `None` when `auto_approve = false` (the gate prompts the user
    /// instead, so no proactive notice is needed).
    fn approval_notice(&self) -> Option<String> {
        if self.auto_approve {
            Some(AUTO_LAUNCH_NOTICE.to_string())
        } else {
            None
        }
    }

    async fn execute(&self, args: Value, cancel: &CancellationToken) -> ToolResult<Value> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArguments("missing 'query' string".to_string()))?;
        if query.trim().is_empty() {
            return Err(ToolError::InvalidArguments(
                "query must not be empty".to_string(),
            ));
        }
        if query.len() > MAX_QUERY_LEN {
            return Err(ToolError::InvalidArguments(format!(
                "query too large ({} bytes; max {})",
                query.len(),
                MAX_QUERY_LEN
            )));
        }
        let magi = self.magi.clone();
        let q = query.to_string();
        // Joined spawn isolates a panic in magi-core's analyze into a recoverable
        // JoinError instead of unwinding into the agent tool loop.
        let handle = tokio::spawn(async move { magi.analyze(&Mode::Analysis, &q).await });
        // RAII backstop: aborts the spawned analysis if this `execute` future is
        // dropped before the `select!` resolves — a dropped tool call would
        // otherwise orphan the three in-flight LLM calls. The abort handle is
        // taken separately from `handle` (which the `select!` consumes), so
        // there is no borrow conflict.
        let abort_guard = AbortOnDrop::new(handle.abort_handle());
        // A proactive consult runs three MAGI LLM calls; on the run's `--timeout`
        // cancellation (REQ-H36) the task is **aborted** — not merely detached —
        // so those expensive API calls actually stop instead of being orphaned.
        // `biased` polls the cancel arm first, so an already-cancelled token
        // short-circuits before the analysis is awaited.
        let report = tokio::select! {
            biased;
            () = cancel.cancelled() => {
                abort_guard.abort();
                return Err(ToolError::ExecutionError(
                    "consult cancelled by timeout".to_string(),
                ));
            }
            joined = handle => match joined {
                Ok(Ok(report)) => report,
                Ok(Err(e)) => return Err(ToolError::ExecutionError(e.to_string())),
                Err(join_err) => {
                    return Err(ToolError::ExecutionError(format!(
                        "consult crashed: {join_err}"
                    )))
                }
            },
        };
        Ok(report_to_consult_json(&report))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magi_core::error::{ExternalErrorKind, ProviderError};
    use magi_core::provider::{CompletionConfig, LlmProvider};
    use magi_core::schema::AgentName;
    use magi_core::test_support::RoutingMockProvider;
    use magi_core::verdict_markers::{VERDICT_CLOSE, VERDICT_OPEN};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    /// Upper bound on how long a *cancelled* `execute` may take to return. The
    /// cancel path aborts the in-flight analysis, so it must resolve almost
    /// immediately; sized generously to absorb scheduler jitter while staying
    /// far below the blocking provider's [`BlockingProvider::SLEEP_SECS`] sleep,
    /// so a regression that awaited the full analysis would blow this budget.
    const CANCEL_RETURN_BUDGET_MS: u128 = 2_000;

    /// A provider whose `complete` blocks far longer than any test tolerates, so
    /// a MAGI analysis over it never finishes within the test. Used to prove that
    /// [`ConsultTool::execute`] returns on cancellation *without* waiting for the
    /// analysis: if the cancel token were ignored, `execute` would block on this
    /// sleep and overrun [`CANCEL_RETURN_BUDGET_MS`].
    struct BlockingProvider;

    impl BlockingProvider {
        /// Sleep duration of each `complete` call — long enough that awaiting the
        /// full analysis is unmistakably distinguishable from a prompt cancel.
        const SLEEP_SECS: u64 = 3_600;
    }

    #[async_trait]
    impl LlmProvider for BlockingProvider {
        async fn complete(
            &self,
            _system_prompt: &str,
            _user_prompt: &str,
            _config: &CompletionConfig,
        ) -> Result<String, ProviderError> {
            tokio::time::sleep(Duration::from_secs(Self::SLEEP_SECS)).await;
            Ok(String::new())
        }

        fn name(&self) -> &str {
            "blocking"
        }

        fn model(&self) -> &str {
            "blocking"
        }
    }

    /// Helper: constructs a `ConsultTool` with `auto_approve = false` (the default).
    fn dummy_tool() -> ConsultTool {
        ConsultTool::new(
            Arc::new(Magi::new(Arc::new(RoutingMockProvider::new()))),
            false,
        )
    }

    /// Respuesta canónica de un mage, en el formato que magi-core 3.x exige.
    ///
    /// Desde 3.0.0 el veredicto se lee **solo** entre [`VERDICT_OPEN`] y
    /// [`VERDICT_CLOSE`], cada marcador solo en su línea. Un JSON pelado —el
    /// formato que servía en 2.x— ya no se parsea y el mage cuenta como fallido.
    /// Se usan las constantes del crate en vez de literales para que un cambio
    /// de marcador rompa la compilación en vez de degradar el fixture en silencio.
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

    fn magi_all_ok() -> Arc<Magi> {
        let provider = RoutingMockProvider::new()
            .with_agent_responses(AgentName::Melchior, vec![Ok(agent_json("melchior"))])
            .with_agent_responses(AgentName::Balthasar, vec![Ok(agent_json("balthasar"))])
            .with_agent_responses(AgentName::Caspar, vec![Ok(agent_json("caspar"))]);
        Arc::new(Magi::new(Arc::new(provider)))
    }

    /// `ConsultTool` with `auto_approve = false` (default) MUST require approval.
    #[test]
    fn test_consult_tool_requires_approval_when_auto_approve_false() {
        let tool = dummy_tool(); // auto_approve = false
        assert!(
            tool.requires_approval(),
            "consult with auto_approve=false must still require approval"
        );
    }

    /// `ConsultTool` with `auto_approve = true` must NOT require approval.
    ///
    /// RED: fails until `requires_approval()` is wired to `!self.auto_approve`.
    #[test]
    fn test_consult_tool_does_not_require_approval_when_auto_approve_true() {
        let tool = ConsultTool::new(
            Arc::new(Magi::new(Arc::new(RoutingMockProvider::new()))),
            true,
        );
        assert!(
            !tool.requires_approval(),
            "consult with auto_approve=true must not require approval (auto-approved)"
        );
    }

    /// `ConsultTool` with `auto_approve = false` must return `None` from `approval_notice`.
    ///
    /// RED: fails until `approval_notice()` is wired to `auto_approve`.
    #[test]
    fn test_consult_approval_notice_is_none_when_auto_approve_false() {
        let tool = dummy_tool(); // auto_approve = false
        assert!(
            tool.approval_notice().is_none(),
            "consult with auto_approve=false must return None — user is prompted instead"
        );
    }

    /// `ConsultTool` with `auto_approve = true` must return `Some(notice)` from `approval_notice`.
    ///
    /// RED: fails until `approval_notice()` is wired to `auto_approve`.
    #[test]
    fn test_consult_approval_notice_is_some_when_auto_approve_true() {
        let tool = ConsultTool::new(
            Arc::new(Magi::new(Arc::new(RoutingMockProvider::new()))),
            true,
        );
        let notice = tool.approval_notice();
        assert!(
            notice.is_some(),
            "consult with auto_approve=true must return Some notice for TUI announcement"
        );
        let msg = notice.unwrap();
        assert!(
            msg.contains("MAGI") || msg.contains("consensus"),
            "auto-launch notice must mention MAGI or consensus; got: {msg:?}"
        );
    }

    #[test]
    fn test_consult_tool_contract() {
        let tool = dummy_tool();
        assert_eq!(tool.name(), "consult");
        let schema = tool.input_schema();
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["query"]["type"], "string");
        assert_eq!(schema["required"][0], "query");
        // `required` names ONLY `query`: `mode` must stay optional so an agent that
        // doesn't pick a lens still gets to consult (REQ-A07/A07b).
        assert_eq!(schema["required"].as_array().unwrap().len(), 1);
        let lower = tool.description().to_lowercase();
        assert!(!lower.is_empty());
        assert!(lower.contains("trade-off"));
        assert!(lower.contains("perspective") || lower.contains("perspectives"));
        assert!(lower.contains("decision") || lower.contains("decisions"));
    }

    /// REQ-A07b: the tool exposes `mode` in its own input schema so an agent that
    /// decides to consult can also pick the lens, from the same three-label
    /// vocabulary `magi_rs::magi::mode::normalize_label` accepts. No behavior change
    /// to `execute` — it still hardcodes `Mode::Analysis`; wiring the declared value
    /// into dispatch is Task 2.3/2.4's job, not this one's.
    #[test]
    fn test_consult_tool_schema_exposes_an_optional_mode_lens() {
        let tool = dummy_tool();
        let schema = tool.input_schema();
        assert_eq!(schema["properties"]["mode"]["type"], "string");
        assert_eq!(
            schema["properties"]["mode"]["enum"],
            json!(["code-review", "design", "analysis"])
        );
    }

    #[tokio::test]
    async fn test_execute_oversized_query_is_invalid_arguments() {
        let tool = ConsultTool::new(magi_all_ok(), false);
        let big = "x".repeat(9000);
        assert!(matches!(
            tool.execute(json!({"query": big}), &CancellationToken::new())
                .await
                .unwrap_err(),
            ToolError::InvalidArguments(_)
        ));
    }

    #[tokio::test]
    async fn test_execute_returns_consensus_report() {
        let tool = ConsultTool::new(magi_all_ok(), false);
        let out = tool
            .execute(
                json!({"query": "should we migrate X to Y?"}),
                &CancellationToken::new(),
            )
            .await
            .expect("3 agents → success");
        assert!(!out["report"].as_str().expect("report string").is_empty());
        assert_eq!(out["degraded"], json!(false));
    }

    #[tokio::test]
    async fn test_execute_empty_query_is_invalid_arguments() {
        let tool = ConsultTool::new(magi_all_ok(), false);
        assert!(matches!(
            tool.execute(json!({ "query": "   " }), &CancellationToken::new())
                .await
                .unwrap_err(),
            ToolError::InvalidArguments(_)
        ));
    }

    #[tokio::test]
    async fn test_execute_missing_query_is_invalid_arguments() {
        let tool = ConsultTool::new(magi_all_ok(), false);
        assert!(matches!(
            tool.execute(json!({}), &CancellationToken::new())
                .await
                .unwrap_err(),
            ToolError::InvalidArguments(_)
        ));
    }

    /// A pre-cancelled token makes `execute` return the cancellation error
    /// promptly, aborting the in-flight 3-LLM analysis instead of running it to
    /// completion (REQ-H36). Uses [`BlockingProvider`] so the analysis would
    /// otherwise block for an hour: returning within [`CANCEL_RETURN_BUDGET_MS`]
    /// proves the cancel path pre-empts the work rather than awaiting it.
    #[tokio::test]
    async fn test_execute_returns_cancellation_error_without_running_full_analysis() {
        let tool = ConsultTool::new(Arc::new(Magi::new(Arc::new(BlockingProvider))), false);
        let cancel = CancellationToken::new();
        cancel.cancel();

        let start = std::time::Instant::now();
        let err = tool
            .execute(json!({ "query": "should we migrate X to Y?" }), &cancel)
            .await
            .expect_err("a cancelled consult must return an error, not a report");
        let elapsed = start.elapsed();

        assert!(
            matches!(err, ToolError::ExecutionError(ref m) if m.contains("cancelled")),
            "cancelled consult must surface a typed cancellation error; got: {err:?}"
        );
        assert!(
            elapsed.as_millis() < CANCEL_RETURN_BUDGET_MS,
            "cancelled consult must return promptly (took {elapsed:?}); it awaited the full analysis"
        );
    }

    /// The `AbortOnDrop` backstop aborts its guarded task the instant the guard
    /// is dropped, so a dropped `execute` future cannot orphan the spawned
    /// analysis (the drop path the explicit cancel arm does not cover). A bare
    /// dropped `JoinHandle`/`AbortHandle` would merely detach the task, leaving
    /// it to run to completion — this asserts the join reports cancellation and
    /// the task never reached its completion store.
    #[tokio::test]
    async fn test_abort_on_drop_aborts_task_when_guard_dropped() {
        let ran_to_completion = Arc::new(AtomicBool::new(false));
        let flag = ran_to_completion.clone();
        let handle = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(BlockingProvider::SLEEP_SECS)).await;
            flag.store(true, Ordering::SeqCst);
        });
        {
            let _guard = AbortOnDrop::new(handle.abort_handle());
            // `_guard` drops here without ever calling `abort()` explicitly.
        }
        let joined = handle.await;
        assert!(
            joined.as_ref().err().map(|e| e.is_cancelled()).unwrap_or(false),
            "dropping the guard must abort the task (join must report cancellation); got {joined:?}"
        );
        assert!(
            !ran_to_completion.load(Ordering::SeqCst),
            "aborted task must not have reached its completion store"
        );
    }

    #[tokio::test]
    async fn test_execute_backend_failure_surfaces_error() {
        let p = RoutingMockProvider::new()
            .with_agent_responses(
                AgentName::Melchior,
                vec![Err(ProviderError::external(
                    "down",
                    ExternalErrorKind::Network,
                ))],
            )
            .with_agent_responses(
                AgentName::Balthasar,
                vec![Err(ProviderError::external(
                    "down",
                    ExternalErrorKind::Network,
                ))],
            )
            .with_agent_responses(
                AgentName::Caspar,
                vec![Err(ProviderError::external(
                    "down",
                    ExternalErrorKind::Network,
                ))],
            );
        let tool = ConsultTool::new(Arc::new(Magi::new(Arc::new(p))), false);
        assert!(matches!(
            tool.execute(json!({"query": "x"}), &CancellationToken::new())
                .await
                .unwrap_err(),
            ToolError::ExecutionError(_)
        ));
    }
}
