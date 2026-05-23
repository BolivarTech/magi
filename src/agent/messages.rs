//! This module defines the conversation history structures.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The role of the message sender.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

/// A single block of content within a message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Content {
    /// A simple text block.
    Text { text: String },
    /// A request to use a specific tool.
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    /// The result of a tool execution.
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default)]
        is_error: bool,
    },
}

/// A message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub role: Role,
    pub content: Vec<Content>,
}

impl Message {
    /// Creates a new user message with a single text block.
    pub fn user(text: &str) -> Self {
        Self {
            role: Role::User,
            content: vec![Content::Text {
                text: text.to_string(),
            }],
        }
    }

    /// Creates a new assistant message with a single text block.
    pub fn assistant(text: &str) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![Content::Text {
                text: text.to_string(),
            }],
        }
    }
}
