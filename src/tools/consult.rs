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

/// Tool wrapping a `magi_core::Magi`. `execute` runs the 3-perspective consensus
/// (implemented in Task 4) and returns the verbatim report. The `description` is
/// what makes the main LLM self-route here only for multi-perspective decisions.
pub struct ConsultTool {
    magi: Arc<Magi>,
    description: String,
}

impl ConsultTool {
    /// Creates a `ConsultTool` over a shared `Magi` orchestrator.
    ///
    /// # Parameters
    /// * `magi` - Shared `Magi` orchestrator that drives the 3-perspective consensus.
    ///
    /// # Returns
    /// A new `ConsultTool` instance with a routing-tuned description.
    pub fn new(magi: Arc<Magi>) -> Self {
        Self {
            magi,
            description: "Run a multi-perspective MAGI consensus (three independent \
                analyst agents) on a hard decision. Use ONLY for questions with genuine \
                trade-offs, design/architecture choices, or 'should we X vs Y given these \
                constraints?' decisions where a single answer is risky. Do NOT use for \
                trivial, factual, or lookup questions — answer those directly."
                .to_string(),
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

    async fn execute(&self, args: Value) -> ToolResult<Value> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArguments("missing 'query' string".to_string()))?;
        let report = self
            .magi
            .analyze(&Mode::Analysis, query)
            .await
            .map_err(|e| ToolError::ExecutionError(e.to_string()))?;
        Ok(json!({ "report": report.report, "degraded": report.degraded }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magi_core::error::ProviderError;
    use magi_core::schema::AgentName;
    use magi_core::test_support::RoutingMockProvider;

    fn dummy_tool() -> ConsultTool {
        ConsultTool::new(Arc::new(Magi::new(Arc::new(RoutingMockProvider::new()))))
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
        assert!(
            lower.contains("trade-off")
                || lower.contains("perspective")
                || lower.contains("decision")
        );
    }

    #[tokio::test]
    async fn test_execute_returns_consensus_report() {
        let tool = ConsultTool::new(magi_all_ok());
        let out = tool
            .execute(json!({"query": "should we migrate X to Y?"}))
            .await
            .expect("3 agents → success");
        assert!(!out["report"].as_str().expect("report string").is_empty());
        assert_eq!(out["degraded"], json!(false));
    }

    #[tokio::test]
    async fn test_execute_missing_query_is_invalid_arguments() {
        let tool = ConsultTool::new(magi_all_ok());
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
        let tool = ConsultTool::new(Arc::new(Magi::new(Arc::new(p))));
        assert!(matches!(
            tool.execute(json!({"query": "x"})).await.unwrap_err(),
            ToolError::ExecutionError(_)
        ));
    }
}
