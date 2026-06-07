// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-06-07

//! Tool that wraps `magi_core::Magi` to run 3-perspective consensus queries.
//! The agent routes here only for genuine multi-perspective decisions; trivial
//! or factual lookups are answered directly.

use std::sync::Arc;
use async_trait::async_trait;
use magi_core::orchestrator::Magi;
use magi_core::schema::Mode;
use serde_json::{json, Value};
use crate::tools::{Tool, ToolError, ToolResult};

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
        assert!(lower.contains("trade-off") || lower.contains("perspective") || lower.contains("decision"));
    }
}
