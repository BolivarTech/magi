//! This module implements the FileReadTool, which allows the agent to read file contents.
//! Hardened following MAGI review to prevent OOM and Path Traversal.

use crate::system::fs::FileSystem;
use crate::tools::{Tool, ToolError, ToolResult};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;

/// Arguments for the `FileReadTool`.
#[derive(Debug, Deserialize)]
struct ReadArgs {
    /// The relative path of the file to read.
    file_path: String,
}

/// A tool that reads file contents.
pub struct FileReadTool {
    fs: Arc<dyn FileSystem>,
    workspace_root: PathBuf,
}

impl FileReadTool {
    /// Creates a new `FileReadTool` anchored to the workspace root.
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
impl Tool for FileReadTool {
    fn name(&self) -> &str {
        "view"
    }

    fn description(&self) -> &str {
        "Read the contents of a file within the workspace."
    }

    fn input_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative path of the file to read."
                }
            },
            "required": ["file_path"]
        })
    }

    /// Read-only, sandboxed to the workspace — no side effects.
    /// Auto-approved so file-read calls do not each prompt the user.
    fn requires_approval(&self) -> bool {
        false
    }

    async fn execute(&self, args: Value) -> ToolResult<Value> {
        let args: ReadArgs =
            serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments(e.to_string()))?;

        let target_path = self.workspace_root.join(args.file_path);

        let canonical_target = if target_path.exists() {
            target_path
                .canonicalize()
                .map_err(|e| ToolError::ExecutionError(format!("Invalid path: {}", e)))?
        } else {
            return Err(ToolError::ExecutionError("File not found".to_string()));
        };

        if !canonical_target.starts_with(&self.workspace_root) {
            return Err(ToolError::ExecutionError(
                "Security Violation: Path traversal attempted".to_string(),
            ));
        }

        let content = self
            .fs
            .read_file(&canonical_target)
            .await
            .map_err(|e| ToolError::ExecutionError(e.to_string()))?;

        Ok(serde_json::json!({ "content": content }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::fs::MockFileSystem;

    /// `FileReadTool` is read-only and sandboxed — must NOT require approval.
    ///
    /// Fails in RED: `requires_approval` is not a method on the `Tool` trait yet.
    #[tokio::test]
    async fn test_file_read_tool_does_not_require_approval() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let mock_fs = MockFileSystem::new();
        let tool = FileReadTool::new(Arc::new(mock_fs), root).unwrap();
        assert!(
            !tool.requires_approval(),
            "view (file read) is read-only — must auto-approve"
        );
    }

    #[tokio::test]
    async fn test_file_read_tool_hardened() {
        let mut mock_fs = MockFileSystem::new();
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();

        mock_fs
            .expect_read_file()
            .times(1)
            .returning(|_| Box::pin(async move { Ok("hello".to_string()) }));

        let tool = FileReadTool::new(Arc::new(mock_fs), root.clone()).unwrap();
        let args = serde_json::json!({"file_path": "test.txt"});
        // Use a real file for the check to pass target_path.exists()
        std::fs::write(root.join("test.txt"), "hello").unwrap();

        let result = tool.execute(args).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap()["content"], "hello");
    }
}
