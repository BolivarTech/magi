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

    /// Concatenates every `Content::Text` block of this message, in order.
    ///
    /// `ToolUse`/`ToolResult` blocks contribute nothing: a caller that needs
    /// those iterates `content` directly. Shared by the mode classifier
    /// (`agent::mode_classifier::ProviderClassifier::classify`) and
    /// `headless_runner::build_transcript`, so the same five-line filter/join
    /// is not re-derived at each call site.
    ///
    /// `agent::magi_adapter::MagiCoreProviderAdapter::complete` used to carry its own
    /// inline copy of this same logic. Task 4.1 deleted `src/agent/magi_adapter.rs`
    /// outright — native MAGI providers (`magi_core::providers::*`) replace the
    /// adapter entirely, so the duplicate this doc used to schedule for removal is
    /// simply gone now, not migrated.
    #[must_use]
    pub fn concat_text(&self) -> String {
        self.content
            .iter()
            .filter_map(|c| match c {
                Content::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }
}
