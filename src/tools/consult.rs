// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-06-07

//! Tool that wraps `magi_core::Magi` to run 3-perspective consensus queries.
//! The agent routes here only for genuine multi-perspective decisions; trivial
//! or factual lookups are answered directly.

use crate::tools::{Tool, ToolError, ToolResult};
use async_trait::async_trait;
use magi_core::orchestrator::Magi;
use magi_core::schema::Mode;
use serde_json::{json, Value};
use std::sync::Arc;

/// Reject oversized consult input before incurring 3 model calls.
/// `pub(crate)` so the forced `/consult` TUI path applies the same cap.
pub(crate) const MAX_QUERY_LEN: usize = 8192;

/// Notice emitted in the TUI when the `consult` tool is auto-approved.
/// Visible to the user so they know the 3-LLM consensus was launched.
const AUTO_LAUNCH_NOTICE: &str =
    "launched MAGI multi-perspective consensus — awaiting evaluation…";

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

    async fn execute(&self, args: Value) -> ToolResult<Value> {
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
        let report =
            match tokio::spawn(async move { magi.analyze(&Mode::Analysis, &q).await }).await {
                Ok(Ok(report)) => report,
                Ok(Err(e)) => return Err(ToolError::ExecutionError(e.to_string())),
                Err(join_err) => {
                    return Err(ToolError::ExecutionError(format!(
                        "consult crashed: {join_err}"
                    )))
                }
            };
        Ok(json!({ "report": report.report, "degraded": report.degraded }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magi_core::error::ProviderError;
    use magi_core::schema::AgentName;
    use magi_core::test_support::RoutingMockProvider;

    /// Helper: constructs a `ConsultTool` with `auto_approve = false` (the default).
    fn dummy_tool() -> ConsultTool {
        ConsultTool::new(Arc::new(Magi::new(Arc::new(RoutingMockProvider::new()))), false)
    }

    fn agent_json(agent: &str) -> String {
        format!(
            r#"{{"agent":"{agent}","verdict":"approve","confidence":0.9,"summary":"s","reasoning":"r","findings":[],"recommendation":"rec"}}"#
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
        let lower = tool.description().to_lowercase();
        assert!(!lower.is_empty());
        assert!(lower.contains("trade-off"));
        assert!(lower.contains("perspective") || lower.contains("perspectives"));
        assert!(lower.contains("decision") || lower.contains("decisions"));
    }

    #[tokio::test]
    async fn test_execute_oversized_query_is_invalid_arguments() {
        let tool = ConsultTool::new(magi_all_ok(), false);
        let big = "x".repeat(9000);
        assert!(matches!(
            tool.execute(json!({"query": big})).await.unwrap_err(),
            ToolError::InvalidArguments(_)
        ));
    }

    #[tokio::test]
    async fn test_execute_returns_consensus_report() {
        let tool = ConsultTool::new(magi_all_ok(), false);
        let out = tool
            .execute(json!({"query": "should we migrate X to Y?"}))
            .await
            .expect("3 agents → success");
        assert!(!out["report"].as_str().expect("report string").is_empty());
        assert_eq!(out["degraded"], json!(false));
    }

    #[tokio::test]
    async fn test_execute_empty_query_is_invalid_arguments() {
        let tool = ConsultTool::new(magi_all_ok(), false);
        assert!(matches!(
            tool.execute(json!({ "query": "   " })).await.unwrap_err(),
            ToolError::InvalidArguments(_)
        ));
    }

    #[tokio::test]
    async fn test_execute_missing_query_is_invalid_arguments() {
        let tool = ConsultTool::new(magi_all_ok(), false);
        assert!(matches!(
            tool.execute(json!({})).await.unwrap_err(),
            ToolError::InvalidArguments(_)
        ));
    }

    #[tokio::test]
    async fn test_execute_backend_failure_surfaces_error() {
        let p = RoutingMockProvider::new()
            .with_agent_responses(
                AgentName::Melchior,
                vec![Err(ProviderError::Network {
                    message: "down".into(),
                })],
            )
            .with_agent_responses(
                AgentName::Balthasar,
                vec![Err(ProviderError::Network {
                    message: "down".into(),
                })],
            )
            .with_agent_responses(
                AgentName::Caspar,
                vec![Err(ProviderError::Network {
                    message: "down".into(),
                })],
            );
        let tool = ConsultTool::new(Arc::new(Magi::new(Arc::new(p))), false);
        assert!(matches!(
            tool.execute(json!({"query": "x"})).await.unwrap_err(),
            ToolError::ExecutionError(_)
        ));
    }
}
