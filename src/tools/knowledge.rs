//! This tool allows the agent to persist technical knowledge about the project in SQLite.

use crate::system::database::MemoryStore;
use crate::tools::{Tool, ToolError, ToolResult};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;

pub struct ProjectFactTool {
    memory: Arc<dyn MemoryStore>,
}

impl ProjectFactTool {
    pub fn new(memory: Arc<dyn MemoryStore>) -> Self {
        Self { memory }
    }
}

#[async_trait]
impl Tool for ProjectFactTool {
    fn name(&self) -> &str {
        "project_knowledge"
    }

    fn description(&self) -> &str {
        "Manages technical facts and decisions about the project in persistent storage. \
         Use 'remember' to save context, 'recall' to search, and 'list' to see all keys."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["remember", "recall", "list"]
                },
                "key": { "type": "string" },
                "value": { "type": "string" }
            },
            "required": ["action"]
        })
    }

    /// Local encrypted-DB write — no filesystem or command risk.
    /// Auto-approved so `project_knowledge` calls do not each prompt the user.
    fn requires_approval(&self) -> bool {
        false
    }

    async fn execute(&self, input: Value) -> ToolResult<Value> {
        let action = input["action"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("Missing action".to_string()))?;

        match action {
            "remember" => {
                let key = input["key"]
                    .as_str()
                    .ok_or_else(|| ToolError::InvalidArguments("Missing key".to_string()))?;
                let val = input["value"]
                    .as_str()
                    .ok_or_else(|| ToolError::InvalidArguments("Missing value".to_string()))?;

                // MAGI FIX: DoS protection - limit size of stored facts (50KB)
                const MAX_VALUE_SIZE: usize = 50 * 1024;
                if val.len() > MAX_VALUE_SIZE {
                    return Err(ToolError::InvalidArguments(format!(
                        "Size limit exceeded: value must be < {} bytes",
                        MAX_VALUE_SIZE
                    )));
                }

                self.memory
                    .set_knowledge(key, val)
                    .await
                    .map_err(|e| ToolError::ExecutionError(e.to_string()))?;
                Ok(json!({ "status": "success", "message": format!("Fact '{}' remembered.", key) }))
            }
            "recall" => {
                let key = input["key"]
                    .as_str()
                    .ok_or_else(|| ToolError::InvalidArguments("Missing key".to_string()))?;
                match self
                    .memory
                    .get_knowledge(key)
                    .await
                    .map_err(|e| ToolError::ExecutionError(e.to_string()))?
                {
                    Some(val) => Ok(json!({ "key": key, "value": val })),
                    None => Ok(
                        json!({ "status": "not_found", "message": format!("No fact found for '{}'", key) }),
                    ),
                }
            }
            "list" => {
                let keys = self
                    .memory
                    .list_knowledge_keys()
                    .await
                    .map_err(|e| ToolError::ExecutionError(e.to_string()))?;
                Ok(json!({ "known_keys": keys }))
            }
            _ => Err(ToolError::InvalidArguments("Unknown action".to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::database::EncryptedSqliteMemory;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_knowledge_tool_execution() {
        let tmp = NamedTempFile::new().unwrap();
        let memory = Arc::new(
            EncryptedSqliteMemory::new(
                tmp.path().to_path_buf(),
                zeroize::Zeroizing::new("pass".to_string()),
            )
            .unwrap(),
        );
        let tool = ProjectFactTool::new(memory.clone());

        // Test Remember
        let res = tool
            .execute(json!({
                "action": "remember",
                "key": "test_key",
                "value": "test_value"
            }))
            .await
            .unwrap();

        assert_eq!(res["status"], "success");

        // Test Recall
        let res = tool
            .execute(json!({
                "action": "recall",
                "key": "test_key"
            }))
            .await
            .unwrap();

        assert_eq!(res["value"], "test_value");
    }

    /// `ProjectFactTool` is a local encrypted-DB write — must NOT require approval.
    ///
    /// Fails in RED: `requires_approval` is not a method on the `Tool` trait yet.
    #[tokio::test]
    async fn test_project_fact_tool_does_not_require_approval() {
        let tmp = NamedTempFile::new().unwrap();
        let memory = Arc::new(
            EncryptedSqliteMemory::new(
                tmp.path().to_path_buf(),
                zeroize::Zeroizing::new("pass".to_string()),
            )
            .unwrap(),
        );
        let tool = ProjectFactTool::new(memory);
        assert!(
            !tool.requires_approval(),
            "project_knowledge is a local encrypted write — must auto-approve"
        );
    }

    #[tokio::test]
    async fn test_knowledge_tool_size_limit() {
        let tmp = NamedTempFile::new().unwrap();
        let memory = Arc::new(
            EncryptedSqliteMemory::new(
                tmp.path().to_path_buf(),
                zeroize::Zeroizing::new("pass".to_string()),
            )
            .unwrap(),
        );
        let tool = ProjectFactTool::new(memory.clone());

        // Attempt to store a very large value (e.g., 1MB)
        let large_val = "X".repeat(1024 * 1024);
        let res = tool
            .execute(json!({
                "action": "remember",
                "key": "too_big",
                "value": large_val
            }))
            .await;

        assert!(res.is_err(), "Should fail when value exceeds size limit");
        if let Err(crate::tools::ToolError::InvalidArguments(msg)) = res {
            assert!(
                msg.contains("Size limit exceeded"),
                "Error message should mention size limit"
            );
        } else {
            panic!("Expected InvalidArguments error, got {:?}", res);
        }
    }
}
