//! This module implements the GrepTool, which allows the agent to search for patterns.

use crate::system::grep::Grep;
use crate::system::path_guard::PathGuard;
use crate::tools::{Tool, ToolError, ToolResult};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;
use std::path::{Path, PathBuf};

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
        Ok(Self {
            grep,
            workspace_root: root,
        })
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
        let args: GrepArgs =
            serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments(e.to_string()))?;

        let guard = PathGuard::new(self.workspace_root.clone())
            .map_err(|e| ToolError::ExecutionError(format!("Sandbox init failed: {}", e)))?;
        let target_path = guard
            .validate(Path::new(&args.path))
            .map_err(|e| ToolError::ExecutionError(format!("Security Violation: {}", e)))?;

        let results = self
            .grep
            .search(&args.pattern, &target_path)
            .await
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

        mock_grep
            .expect_search()
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

    #[tokio::test]
    async fn test_grep_rejects_symlink_escaping_workspace() {
        let mut mock_grep = MockGrep::new();
        mock_grep.expect_search().never();

        let work = tempfile::tempdir().unwrap();
        let root = work.path().canonicalize().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let outside_path = outside.path().canonicalize().unwrap();

        let link = root.join("link");

        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&outside_path, &link).unwrap();
        }
        #[cfg(windows)]
        {
            if std::os::windows::fs::symlink_dir(&outside_path, &link).is_err() {
                eprintln!("skipping: cannot create directory symlink without privilege");
                return;
            }
        }

        let tool = GrepTool::new(Box::new(mock_grep), root).unwrap();
        let args = serde_json::json!({ "pattern": "secret", "path": "link" });

        let result = tool.execute(args).await;
        assert!(
            result.is_err(),
            "symlink escaping the workspace must be rejected, got: {:?}",
            result
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Security") || msg.contains("sandbox") || msg.contains("traversal"),
            "error should signal a sandbox violation, got: {}",
            msg
        );
    }
}
