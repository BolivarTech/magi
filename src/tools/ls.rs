//! This module implements the ListTool (ls), which allows the agent to list files in a directory.
//! Hardened following MAGI review to prevent sandbox escapes and infinite loops.

use crate::system::fs::FileSystem;
use crate::tools::{Tool, ToolError, ToolResult};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;

/// Arguments for the `ListTool`.
#[derive(Debug, Deserialize)]
struct ListArgs {
    /// The path to list.
    path: String,
}

/// A tool that lists directory contents.
pub struct ListTool {
    fs: Arc<dyn FileSystem>,
    workspace_root: PathBuf,
}

impl ListTool {
    /// Creates a new `ListTool` anchored to the workspace root.
    pub fn new(fs: Arc<dyn FileSystem>, workspace_root: PathBuf) -> anyhow::Result<Self> {
        let root = workspace_root
            .canonicalize()
            .map_err(|e| anyhow::anyhow!("Invalid workspace root: {}", e))?;
        Ok(Self {
            fs,
            workspace_root: root,
        })
    }
}

#[async_trait]
impl Tool for ListTool {
    fn name(&self) -> &str {
        "ls"
    }

    fn description(&self) -> &str {
        "List files and directories in a given path. Sandboxed to workspace."
    }

    fn input_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to list."
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, args: Value) -> ToolResult<Value> {
        let args: ListArgs =
            serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments(e.to_string()))?;

        let target_path = self.workspace_root.join(args.path);

        // Security check: ensure path is within workspace
        let canonical_target = target_path
            .canonicalize()
            .map_err(|e| ToolError::ExecutionError(format!("Invalid path: {}", e)))?;

        if !canonical_target.starts_with(&self.workspace_root) {
            return Err(ToolError::ExecutionError(
                "Security Violation: Path traversal attempted".to_string(),
            ));
        }

        let entries = self
            .fs
            .list_directory(&canonical_target)
            .await
            .map_err(|e| ToolError::ExecutionError(e.to_string()))?;

        Ok(serde_json::json!({ "entries": entries }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::fs::MockFileSystem;

    /// `ListTool` is read-only and sandboxed — must NOT require approval.
    ///
    /// Fails in RED: `requires_approval` is not a method on the `Tool` trait yet.
    #[tokio::test]
    async fn test_list_tool_does_not_require_approval() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let mock_fs = MockFileSystem::new();
        let tool = ListTool::new(Arc::new(mock_fs), root).unwrap();
        assert!(
            !tool.requires_approval(),
            "ls (directory list) is read-only — must auto-approve"
        );
    }

    #[tokio::test]
    async fn test_ls_tool_hardened() {
        let mut mock_fs = MockFileSystem::new();
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();

        mock_fs
            .expect_list_directory()
            .times(1)
            .returning(|_| Box::pin(async move { Ok(vec!["file1.txt".to_string()]) }));

        let tool = ListTool::new(Arc::new(mock_fs), root.clone()).unwrap();
        let args = serde_json::json!({"path": "."});
        let result = tool.execute(args).await;

        assert!(result.is_ok());
    }
}
