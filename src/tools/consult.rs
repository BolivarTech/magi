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
// ConsultTool and its constructor are registered in main.rs (Task 5);
// allow dead-code warnings until that task lands.
#[allow(dead_code)]
pub struct ConsultTool {
    magi: Arc<Magi>,
    description: String,
}

#[allow(dead_code)]
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

    async fn execute(&self, _args: Value) -> ToolResult<Value> {
        // Implemented under Task 4 (true Red). Sentinel until then.
        let _ = (&self.magi, Mode::Analysis);
        Err(ToolError::ExecutionError(
            "consult execute not yet implemented".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_tool() -> ConsultTool {
        ConsultTool::new(Arc::new(Magi::new(Arc::new(
            magi_core::test_support::RoutingMockProvider::new(),
        ))))
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
}
