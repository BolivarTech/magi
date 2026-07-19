//! This module defines the base traits and types for the tool system.
//! It follows the OOP paradigm using traits for polymorphism and structs for implementation.

use async_trait::async_trait;
use serde_json::Value;
use thiserror::Error;
use tokio_util::sync::CancellationToken;

pub mod bash;
pub mod consult;
pub mod grep;
pub mod knowledge;
pub mod ls;
pub(crate) mod proc_group;
pub mod read;
pub mod write;

/// Errors that can occur during tool execution.
/// Follows the strict error handling requirement from CLAUDE.md.
#[derive(Debug, Error)]
pub enum ToolError {
    /// Error occurring during the actual execution of the tool logic.
    #[error("Execution failed: {0}")]
    ExecutionError(String),
    /// Error occurring when arguments provided to the tool are invalid.
    #[error("Invalid arguments: {0}")]
    InvalidArguments(String),
}

/// The result of a tool execution, encapsulating either a successful JSON value or a ToolError.
pub type ToolResult<T> = Result<T, ToolError>;

/// Base trait for all tools in the Magi Agent ecosystem.
///
/// This trait defines the contract that all tools must follow. It uses `async_trait`
/// to allow for asynchronous execution, which is essential for I/O bound tools.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Returns the unique name of the tool.
    ///
    /// # Returns
    /// A string slice containing the tool's name.
    fn name(&self) -> &str;

    /// Returns a human-readable description of what the tool does.
    ///
    /// # Returns
    /// A string slice containing the tool's description.
    fn description(&self) -> &str;

    /// Returns the JSON schema for the tool's input.
    ///
    /// # Returns
    /// A `serde_json::Value` representing the JSON schema.
    fn input_schema(&self) -> Value;

    /// Executes the tool with the given JSON arguments.
    ///
    /// # Parameters
    /// * `args` - A `serde_json::Value` containing the arguments for the tool.
    /// * `cancel` - Cooperative cancellation for the run (REQ-H36). A tool that
    ///   owns a subprocess (`bash`) MUST race this token and kill its process
    ///   *tree* when it fires, so a wall-clock `--timeout` is a real bound. Fast,
    ///   non-subprocess tools may ignore it (they finish before it matters) or
    ///   abort cooperatively at their next await. The interactive/TUI path passes
    ///   a fresh, never-cancelled token, so this parameter is inert there.
    ///
    /// # Returns
    /// A `ToolResult<Value>` containing the result of the execution or an error.
    async fn execute(&self, args: Value, cancel: &CancellationToken) -> ToolResult<Value>;

    /// Whether executing this tool requires explicit user approval.
    ///
    /// Default `true` (safe-by-default): every new tool requires approval unless
    /// it explicitly opts out. Override to `false` only for cheap, side-effect-free
    /// or local-only operations (read-only filesystem queries; local encrypted-memory
    /// writes). Expensive or dangerous tools (shell execution, file writes, external
    /// multi-model consensus) MUST keep the default.
    ///
    /// # Returns
    /// `true` if the agent must prompt the user before running this tool;
    /// `false` if the tool may be executed without a prompt (auto-approved).
    fn requires_approval(&self) -> bool {
        true
    }

    /// Optional notice shown in the TUI when this tool is AUTO-APPROVED (launched
    /// without a user prompt). `None` (default) = launch silently. Tools that are
    /// slow or noteworthy when auto-run return `Some(message)` so the user knows
    /// that a potentially long operation is in progress.
    ///
    /// The message is sent as a `StreamPiece::Notice` **before** the tool executes,
    /// rendered in the TUI with the `⚠` prefix/style. It is never persisted to
    /// conversation history.
    ///
    /// # Returns
    /// `None` (default) for fast or silent tools;  `Some(notice_text)` for tools
    /// that want to announce their auto-launch (e.g. a 3-LLM consensus call).
    fn approval_notice(&self) -> Option<String> {
        None
    }
}

/// A mock implementation of a tool for testing and architectural validation.
///
/// `MockTool` allows us to verify the tool orchestration logic without
/// performing actual side effects.
#[allow(dead_code)]
pub struct MockTool {
    name: String,
    description: String,
}

impl MockTool {
    /// Creates a new `MockTool` instance.
    ///
    /// # Parameters
    /// * `name` - The name of the tool.
    /// * `description` - The description of the tool.
    ///
    /// # Returns
    /// A new instance of `MockTool`.
    #[allow(dead_code)]
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
        }
    }
}

#[async_trait]
impl Tool for MockTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn input_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {}
        })
    }

    async fn execute(&self, _args: Value, _cancel: &CancellationToken) -> ToolResult<Value> {
        Ok(serde_json::json!({"status": "success"}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tool_metadata() {
        let tool = MockTool::new("test_tool", "A test tool");
        assert_eq!(tool.name(), "test_tool");
        assert_eq!(tool.description(), "A test tool");
    }

    #[tokio::test]
    async fn test_tool_execution() {
        let tool = MockTool::new("test_tool", "A test tool");
        let result = tool
            .execute(
                serde_json::json!({"input": "test"}),
                &CancellationToken::new(),
            )
            .await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), serde_json::json!({"status": "success"}));
    }

    #[tokio::test]
    async fn test_error_variants() {
        let err = ToolError::ExecutionError("failed".to_string());
        assert_eq!(err.to_string(), "Execution failed: failed");

        let err = ToolError::InvalidArguments("bad args".to_string());
        assert_eq!(err.to_string(), "Invalid arguments: bad args");
    }

    // ── Per-tool approval policy (RED: fails to compile until `requires_approval`
    //   is added to the `Tool` trait) ─────────────────────────────────────────────

    /// Default `MockTool` must require approval (safe-by-default invariant).
    ///
    /// Fails in RED: `requires_approval` is not a method on the `Tool` trait yet.
    #[tokio::test]
    async fn test_approval_policy_default_requires_approval() {
        let tool = MockTool::new("test_dangerous", "a tool that executes commands");
        assert!(
            tool.requires_approval(),
            "default requires_approval must be true (safe-by-default)"
        );
    }

    /// A mock that explicitly opts out — verifies the override mechanism compiles
    /// and correctly returns `false`.
    ///
    /// Fails in RED: `fn requires_approval` is not a member of trait `Tool`.
    struct AutoApproveMock;

    #[async_trait]
    impl Tool for AutoApproveMock {
        fn name(&self) -> &str {
            "auto_safe"
        }
        fn description(&self) -> &str {
            "auto-approved safe mock"
        }
        fn input_schema(&self) -> Value {
            serde_json::json!({"type": "object", "properties": {}})
        }
        async fn execute(&self, _args: Value, _cancel: &CancellationToken) -> ToolResult<Value> {
            Ok(serde_json::json!({"status": "auto"}))
        }
        fn requires_approval(&self) -> bool {
            false
        }
    }

    #[tokio::test]
    async fn test_approval_policy_override_to_false() {
        let tool = AutoApproveMock;
        assert!(
            !tool.requires_approval(),
            "explicit override to false must be respected"
        );
    }

    // ── approval_notice() tests ────────────────────────────────────────────────

    /// Default `MockTool` must return `None` from `approval_notice` (silent
    /// auto-run is the default behavior for most tools).
    ///
    /// RED: fails until `approval_notice` is added to the `Tool` trait.
    #[tokio::test]
    async fn test_approval_notice_default_is_none() {
        let tool = MockTool::new("test_tool", "A test tool");
        assert!(
            tool.approval_notice().is_none(),
            "default approval_notice must be None (most tools are silent when auto-approved)"
        );
    }

    /// A tool that overrides `approval_notice` to return `Some` must have its
    /// message respected by the agent.
    ///
    /// RED: fails until `approval_notice` is added to the `Tool` trait.
    #[tokio::test]
    async fn test_approval_notice_override_to_some() {
        struct NoticeTool;
        #[async_trait]
        impl Tool for NoticeTool {
            fn name(&self) -> &str {
                "notice_tool"
            }
            fn description(&self) -> &str {
                "tool that announces itself"
            }
            fn input_schema(&self) -> Value {
                serde_json::json!({"type": "object", "properties": {}})
            }
            async fn execute(
                &self,
                _args: Value,
                _cancel: &CancellationToken,
            ) -> ToolResult<Value> {
                Ok(serde_json::json!({"done": true}))
            }
            fn requires_approval(&self) -> bool {
                false
            }
            fn approval_notice(&self) -> Option<String> {
                Some("notice: auto-launch in progress".into())
            }
        }

        let tool = NoticeTool;
        assert_eq!(
            tool.approval_notice(),
            Some("notice: auto-launch in progress".into()),
            "approval_notice override must return Some with the expected message"
        );
    }
}
