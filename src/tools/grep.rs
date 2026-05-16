//! This module implements the GrepTool, which allows the agent to search for patterns.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;
use std::path::PathBuf;
use crate::tools::{Tool, ToolResult, ToolError};
use crate::system::grep::Grep;

/// Arguments for the `GrepTool`.
#[derive(Debug, Deserialize)]
struct GrepArgs {
    pattern: String,
    path: String,
}

/// A tool that searches for patterns.
pub struct GrepTool {
    grep: Box<dyn Grep>,
    workspace_root: PathBuf,
}

impl GrepTool {
    pub fn new(grep: Box<dyn Grep>, workspace_root: PathBuf) -> anyhow::Result<Self> {
        let root = workspace_root.canonicalize()?;
        Ok(Self { grep, workspace_root: root })
    }
}

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search for a pattern in the workspace using RipGrep."
    }

    fn input_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for."
                },
                "path": {
                    "type": "string",
                    "description": "Relative path to search within."
                }
            },
            "required": ["pattern", "path"]
        })
    }

    async fn execute(&self, args: Value) -> ToolResult<Value> {
        let args: GrepArgs = serde_json::from_value(args)
            .map_err(|e| ToolError::InvalidArguments(e.to_string()))?;

        let target_path = self.workspace_root.join(args.path);
        
        if !target_path.exists() {
             return Err(ToolError::ExecutionError("Path not found".to_string()));
        }

        let results = self.grep.search(&args.pattern, &target_path).await
            .map_err(|e| ToolError::ExecutionError(e.to_string()))?;

        Ok(serde_json::json!({ "results": results }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::grep::MockGrep;

    #[tokio::test]
    async fn test_grep_tool_execution() {
        let mut mock_grep = MockGrep::new();
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();

        mock_grep.expect_search()
            .times(1)
            .returning(|_, _| Box::pin(async move { Ok(vec!["match".to_string()]) }));

        let tool = GrepTool::new(Box::new(mock_grep), root.clone()).unwrap();
        let args = serde_json::json!({
            "pattern": "test",
            "path": "."
        });
        
        let result = tool.execute(args).await;
        assert!(result.is_ok());
    }
}
