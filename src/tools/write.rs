//! This module implements the FileWriteTool, which allows the agent to create and modify files.
//! It includes security sandboxing via PathGuard.

use crate::system::fs::FileSystem;
use crate::system::path_guard::PathGuard;
use crate::tools::{Tool, ToolError, ToolResult};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Arguments for the `FileWriteTool`.
#[derive(Debug, Deserialize)]
struct WriteArgs {
    /// The relative path of the file.
    file_path: String,
    /// Content to write.
    content: String,
}

/// A tool that writes file contents.
pub struct FileWriteTool {
    fs: Arc<dyn FileSystem>,
    workspace_root: PathBuf,
}

impl FileWriteTool {
    /// Creates a new `FileWriteTool` anchored to the workspace root.
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
impl Tool for FileWriteTool {
    fn name(&self) -> &str {
        "edit"
    }

    fn description(&self) -> &str {
        "Create or overwrite a file with the provided content."
    }

    fn input_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative path of the file."
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file."
                }
            },
            "required": ["file_path", "content"]
        })
    }

    async fn execute(&self, args: Value) -> ToolResult<Value> {
        let args: WriteArgs =
            serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments(e.to_string()))?;

        let guard = PathGuard::new(self.workspace_root.clone())
            .map_err(|e| ToolError::ExecutionError(format!("Sandbox init failed: {}", e)))?;
        let target_path = guard
            .validate(Path::new(&args.file_path))
            .map_err(|e| ToolError::ExecutionError(format!("Security Violation: {}", e)))?;

        self.fs
            .write_file(&target_path, &args.content)
            .await
            .map_err(|e| ToolError::ExecutionError(e.to_string()))?;

        Ok(serde_json::json!({ "status": "success" }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::fs::MockFileSystem;

    /// `FileWriteTool` modifies files on disk — MUST require approval.
    ///
    /// Fails in RED: `requires_approval` is not a method on the `Tool` trait yet.
    #[tokio::test]
    async fn test_file_write_tool_requires_approval() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let mock_fs = MockFileSystem::new();
        let tool = FileWriteTool::new(Arc::new(mock_fs), root).unwrap();
        assert!(
            tool.requires_approval(),
            "edit (file write) modifies disk — must always require approval"
        );
    }

    #[tokio::test]
    async fn test_file_write_tool_execution() {
        let mut mock_fs = MockFileSystem::new();
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();

        mock_fs
            .expect_write_file()
            .times(1)
            .returning(|_, _| Box::pin(async move { Ok(()) }));

        let tool = FileWriteTool::new(Arc::new(mock_fs), root.clone()).unwrap();
        let args = serde_json::json!({
            "file_path": "new.txt",
            "content": "hello"
        });

        let result = tool.execute(args).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_write_rejects_absolute_path_outside_workspace() {
        let mut mock_fs = MockFileSystem::new();
        mock_fs.expect_write_file().never();

        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let tool = FileWriteTool::new(Arc::new(mock_fs), root.clone()).unwrap();

        #[cfg(target_os = "windows")]
        let evil = r"C:\Windows\System32\evil.dll";
        #[cfg(not(target_os = "windows"))]
        let evil = "/etc/evil.dll";

        let args = serde_json::json!({ "file_path": evil, "content": "payload" });

        let result = tool.execute(args).await;
        assert!(
            result.is_err(),
            "absolute path outside workspace must be rejected, got: {:?}",
            result
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("sandbox") || msg.contains("Security"),
            "error should signal a sandbox violation, got: {}",
            msg
        );
    }

    #[tokio::test]
    async fn test_write_rejects_parent_dir_traversal() {
        let mut mock_fs = MockFileSystem::new();
        mock_fs.expect_write_file().never();

        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let tool = FileWriteTool::new(Arc::new(mock_fs), root).unwrap();

        let args = serde_json::json!({ "file_path": "sub/../../escape.txt", "content": "payload" });

        let result = tool.execute(args).await;
        assert!(result.is_err(), "parent-dir traversal must be rejected");
    }
}
