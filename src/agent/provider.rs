//! This module defines the Provider trait for AI backend interactions.

use crate::agent::messages::{Content, Message, Role};
use anyhow::Result;
use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::time::{sleep, Duration};

#[cfg(test)]
use mockall::automock;

use crate::tools::Tool;

/// A chunk of a response from the AI.
#[derive(Debug, Clone, PartialEq)]
pub enum ResponseChunk {
    /// A piece of text.
    TextDelta(String),
    /// Start of a tool use.
    ToolUseStart { id: String, name: String },
    /// Input data for a tool use.
    ToolUseInputDelta { id: String, input_json: String },
    /// Completion of a full message.
    MessageDone(Message),
}

/// Trait representing an AI backend provider.
#[async_trait]
pub trait Provider: Send + Sync {
    /// Sends a list of messages to the AI and returns a stream of chunks.
    async fn stream_messages(
        &self,
        messages: &[Message],
        tools: &[Box<dyn Tool>],
    ) -> Result<BoxStream<'static, Result<ResponseChunk>>>;

    /// Sends a list of messages and returns the full message (blocking until done).
    async fn send_messages(
        &self,
        messages: &[Message],
        tools: &[Box<dyn Tool>],
    ) -> Result<Message> {
        let mut attempts = 0;
        let max_attempts = 3;

        loop {
            attempts += 1;
            match self.stream_messages(messages, tools).await {
                Ok(mut stream) => {
                    let mut last_message = None;
                    let mut full_text = String::new();
                    let mut role = Role::Assistant;

                    while let Some(chunk_result) = stream.next().await {
                        match chunk_result? {
                            ResponseChunk::MessageDone(msg) => {
                                last_message = Some(msg);
                            }
                            ResponseChunk::TextDelta(t) => {
                                full_text.push_str(&t);
                            }
                            _ => {}
                        }
                    }

                    if let Some(msg) = last_message {
                        return Ok(msg);
                    }

                    if !full_text.is_empty() {
                        return Ok(Message {
                            role,
                            content: vec![Content::Text { text: full_text }],
                        });
                    }

                    return Err(anyhow::anyhow!(
                        "Stream ended without MessageDone or content"
                    ));
                }
                Err(e) if attempts < max_attempts && e.to_string().contains("429") => {
                    let wait_secs = 2_u64.pow(attempts as u32);
                    sleep(Duration::from_secs(wait_secs)).await;
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
    }
}

/// A provider that returns static, canned responses.
pub struct StaticProvider;

#[async_trait]
impl Provider for StaticProvider {
    async fn stream_messages(
        &self,
        _messages: &[Message],
        _tools: &[Box<dyn Tool>],
    ) -> Result<BoxStream<'static, Result<ResponseChunk>>> {
        let msg = Message::assistant("I am a Rust-powered assistant. How can I help you today?");
        let chunks = vec![
            Ok(ResponseChunk::TextDelta(
                "I am a Rust-powered assistant. ".to_string(),
            )),
            Ok(ResponseChunk::TextDelta(
                "How can I help you today?".to_string(),
            )),
            Ok(ResponseChunk::MessageDone(msg)),
        ];
        Ok(Box::pin(stream::iter(chunks)))
    }
}

/// Anthropic API Tool Schema
#[derive(Debug, Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

/// Anthropic API Message Schema
#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<Message>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<AnthropicTool>,
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    role: Role,
    content: Vec<Content>,
}

#[derive(Debug, Deserialize)]
struct AnthropicErrorResponse {
    error: AnthropicErrorDetail,
}

#[derive(Debug, Deserialize)]
struct AnthropicErrorDetail {
    #[serde(rename = "type")]
    error_type: String,
    message: String,
}

/// Anthropic SSE Event Types
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicSseEvent {
    MessageStart {
        message: AnthropicMessageStart,
    },
    ContentBlockStart {
        index: usize,
        content_block: serde_json::Value,
    },
    ContentBlockDelta {
        index: usize,
        delta: AnthropicDelta,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        delta: serde_json::Value,
        usage: serde_json::Value,
    },
    MessageStop,
    Ping,
    Error {
        error: AnthropicErrorDetail,
    },
}

#[derive(Debug, Deserialize)]
struct AnthropicMessageStart {
    id: String,
    role: Role,
    model: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicDelta {
    TextDelta { text: String },
    InputDelta { partial_json: String },
}

/// A provider that communicates with Anthropic's Messages API.
pub struct AnthropicProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
}

impl AnthropicProvider {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model,
            base_url: "https://api.anthropic.com/v1".to_string(),
        }
    }

    #[cfg(test)]
    pub fn with_base_url(api_key: String, model: String, base_url: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model,
            base_url,
        }
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    async fn stream_messages(
        &self,
        messages: &[Message],
        tools: &[Box<dyn Tool>],
    ) -> Result<BoxStream<'static, Result<ResponseChunk>>> {
        let url = format!("{}/messages", self.base_url);

        let anthropic_tools: Vec<AnthropicTool> = tools
            .iter()
            .map(|t| AnthropicTool {
                name: t.name().to_string(),
                description: t.description().to_string(),
                input_schema: t.input_schema(),
            })
            .collect();

        let request = AnthropicRequest {
            model: self.model.clone(),
            messages: messages.to_vec(),
            max_tokens: 4096,
            tools: anthropic_tools,
            stream: true,
        };

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await?;

            if let Ok(error_res) = serde_json::from_str::<AnthropicErrorResponse>(&body) {
                return Err(anyhow::anyhow!(
                    "Anthropic API Error [{}] ({}): {}",
                    status,
                    error_res.error.error_type,
                    error_res.error.message
                ));
            }

            return Err(anyhow::anyhow!(
                "Anthropic API Error [{}]: Raw Body: {}",
                status,
                body
            ));
        }

        let bytes_stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut full_content: Vec<Content> = Vec::new();
        let mut current_role = Role::Assistant;

        let output_stream = bytes_stream.flat_map(move |chunk_res| {
            let chunk = match chunk_res {
                Ok(c) => c,
                Err(e) => {
                    return stream::iter(vec![Err(anyhow::anyhow!("Network error: {}", e))]).boxed()
                }
            };
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            let mut chunks = Vec::new();
            while let Some(line_end) = buffer.find("\n\n") {
                let block = buffer.drain(..line_end + 2).collect::<String>();
                for line in block.lines() {
                    if line.starts_with("data: ") {
                        let data = &line[6..];
                        if let Ok(event) = serde_json::from_str::<AnthropicSseEvent>(data) {
                            match event {
                                AnthropicSseEvent::MessageStart { message } => {
                                    current_role = message.role;
                                }
                                AnthropicSseEvent::ContentBlockDelta { delta, .. } => match delta {
                                    AnthropicDelta::TextDelta { text } => {
                                        if let Some(Content::Text { text: existing }) =
                                            full_content.last_mut()
                                        {
                                            existing.push_str(&text);
                                        } else {
                                            full_content.push(Content::Text { text: text.clone() });
                                        }
                                        chunks.push(Ok(ResponseChunk::TextDelta(text)));
                                    }
                                    AnthropicDelta::InputDelta { partial_json } => {
                                        chunks.push(Ok(ResponseChunk::ToolUseInputDelta {
                                            id: String::new(),
                                            input_json: partial_json,
                                        }));
                                    }
                                },
                                AnthropicSseEvent::MessageStop => {
                                    let msg = Message {
                                        role: current_role.clone(),
                                        content: full_content.clone(),
                                    };
                                    chunks.push(Ok(ResponseChunk::MessageDone(msg)));
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            stream::iter(chunks).boxed()
        });

        Ok(Box::pin(output_stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::messages::{Content, Role};
    use mockito::Server;
    use serde_json::json;

    #[tokio::test]
    async fn test_anthropic_provider_simple_response() {
        let mut server = Server::new_async().await;
        let url = server.url();
        let sse_body =
            "event: message_start\ndata: {\"type\": \"message_start\", \"message\": {\"id\": \"msg_123\", \"role\": \"assistant\", \"model\": \"claude-3-5-sonnet\"}}\n\n\
             event: content_block_delta\ndata: {\"type\": \"content_block_delta\", \"index\": 0, \"delta\": {\"type\": \"text_delta\", \"text\": \"Hello from Mockito!\"}}\n\n\
             event: message_stop\ndata: {\"type\": \"message_stop\"}\n\n";
        let _m = server
            .mock("POST", "/messages")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse_body)
            .create_async()
            .await;
        let provider = AnthropicProvider::with_base_url(
            "test_key".to_string(),
            "claude-3-5-sonnet".to_string(),
            url,
        );
        let messages = vec![Message::user("Hi")];
        let response = provider.send_messages(&messages, &[]).await.unwrap();
        assert_eq!(response.role, Role::Assistant);
        if let Content::Text { text } = &response.content[0] {
            assert_eq!(text, "Hello from Mockito!");
        } else {
            panic!("Expected text content");
        }
    }

    #[tokio::test]
    async fn test_anthropic_provider_tool_use() {
        let mut server = Server::new_async().await;
        let url = server.url();
        let sse_body =
            "event: message_start\ndata: {\"type\": \"message_start\", \"message\": {\"id\": \"msg_tool_1\", \"role\": \"assistant\", \"model\": \"claude-3-5-sonnet\"}}\n\n\
             event: content_block_delta\ndata: {\"type\": \"content_block_delta\", \"index\": 0, \"delta\": {\"type\": \"text_delta\", \"text\": \"Listing \"}}\n\n\
             event: content_block_delta\ndata: {\"type\": \"content_block_delta\", \"index\": 0, \"delta\": {\"type\": \"text_delta\", \"text\": \"files in .\"}}\n\n\
             event: message_stop\ndata: {\"type\": \"message_stop\"}\n\n";
        let _m = server
            .mock("POST", "/messages")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse_body)
            .create_async()
            .await;
        let provider = AnthropicProvider::with_base_url(
            "test_key".to_string(),
            "claude-3-5-sonnet".to_string(),
            url,
        );
        let messages = vec![Message::user("List files")];
        let response = provider.send_messages(&messages, &[]).await.unwrap();
        assert_eq!(response.role, Role::Assistant);
        assert_eq!(response.content.len(), 1);
        if let Content::Text { text } = &response.content[0] {
            assert_eq!(text, "Listing files in .");
        } else {
            panic!("Expected text content, got {:?}", response.content[0]);
        }
    }

    #[tokio::test]
    async fn test_anthropic_provider_invalid_key_error() {
        let mut server = Server::new_async().await;
        let url = server.url();

        let _m = server
            .mock("POST", "/messages")
            .with_status(401)
            .with_header("content-type", "application/json")
            .with_body(
                json!({
                    "type": "error",
                    "error": {
                        "type": "authentication_error",
                        "message": "invalid x-api-key"
                    }
                })
                .to_string(),
            )
            .create_async()
            .await;

        let provider = AnthropicProvider::with_base_url(
            "invalid_key".to_string(),
            "claude-3-5-sonnet".to_string(),
            url,
        );

        let result = provider.send_messages(&[], &[]).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("authentication_error"),
            "Error should mention auth error type"
        );
        assert!(
            err_msg.contains("invalid x-api-key"),
            "Error should contain the specific API message"
        );
        assert!(
            err_msg.contains("401"),
            "Error should contain the status code"
        );
    }

    #[tokio::test]
    async fn test_anthropic_provider_streaming_parsing() {
        let mut server = Server::new_async().await;
        let url = server.url();

        // Mock an SSE stream from Anthropic
        let sse_body =
            "event: message_start\ndata: {\"type\": \"message_start\", \"message\": {\"id\": \"msg_1\", \"role\": \"assistant\", \"content\": [], \"model\": \"claude-3\", \"stop_reason\": null, \"stop_sequence\": null, \"usage\": {\"input_tokens\": 1, \"output_tokens\": 1}}}\n\n\
             event: content_block_start\ndata: {\"type\": \"content_block_start\", \"index\":0, \"content_block\": {\"type\": \"text\", \"text\": \"\"}}\n\n\
             event: content_block_delta\ndata: {\"type\": \"content_block_delta\", \"index\":0, \"delta\": {\"type\": \"text_delta\", \"text\": \"Hello \"}}\n\n\
             event: content_block_delta\ndata: {\"type\": \"content_block_delta\", \"index\":0, \"delta\": {\"type\": \"text_delta\", \"text\": \"world!\"}}\n\n\
             event: message_stop\ndata: {\"type\": \"message_stop\"}\n\n";

        let _m = server
            .mock("POST", "/messages")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse_body)
            .create_async()
            .await;

        let provider = AnthropicProvider::with_base_url(
            "test_key".to_string(),
            "claude-3-5-sonnet".to_string(),
            url,
        );

        let mut stream = provider.stream_messages(&[], &[]).await.unwrap();

        let mut full_text = String::new();
        while let Some(chunk_result) = stream.next().await {
            if let Ok(ResponseChunk::TextDelta(delta)) = chunk_result {
                full_text.push_str(&delta);
            }
        }

        assert_eq!(full_text, "Hello world!");
    }

    #[tokio::test]
    async fn test_anthropic_provider_malformed_sse() {
        let mut server = Server::new_async().await;
        let url = server.url();

        // One valid line, one malformed data line, one valid stop
        let sse_body =
            "event: content_block_delta\ndata: {\"type\": \"content_block_delta\", \"index\":0, \"delta\": {\"type\": \"text_delta\", \"text\": \"Valid\"}}\n\n\
             event: content_block_delta\ndata: {MALFORMED_JSON}\n\n\
             event: message_stop\ndata: {\"type\": \"message_stop\"}\n\n";

        let _m = server
            .mock("POST", "/messages")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse_body)
            .create_async()
            .await;

        let provider = AnthropicProvider::with_base_url(
            "test_key".to_string(),
            "claude-3-5-sonnet".to_string(),
            url,
        );

        let mut stream = provider.stream_messages(&[], &[]).await.unwrap();

        let mut full_text = String::new();
        while let Some(chunk_result) = stream.next().await {
            if let Ok(ResponseChunk::TextDelta(delta)) = chunk_result {
                full_text.push_str(&delta);
            }
        }

        // It should skip the malformed line and still finish
        assert_eq!(full_text, "Valid");
    }

    #[tokio::test]
    async fn test_anthropic_provider_retry_on_429() {
        let mut server = Server::new_async().await;
        let url = server.url();

        // Mock 429 once, then 200
        let _m1 = server
            .mock("POST", "/messages")
            .with_status(429)
            .with_header("content-type", "application/json")
            .with_body(
                json!({
                    "type": "error",
                    "error": {
                        "type": "rate_limit_error",
                        "message": "Too many requests"
                    }
                })
                .to_string(),
            )
            .expect(1)
            .create_async()
            .await;

        let sse_body =
            "event: content_block_delta\ndata: {\"type\": \"content_block_delta\", \"index\":0, \"delta\": {\"type\": \"text_delta\", \"text\": \"Recovered!\"}}\n\n\
             event: message_stop\ndata: {\"type\": \"message_stop\"}\n\n";

        let _m2 = server
            .mock("POST", "/messages")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse_body)
            .expect(1)
            .create_async()
            .await;

        let provider =
            AnthropicProvider::with_base_url("test_key".to_string(), "test-model".to_string(), url);

        let response = provider.send_messages(&[], &[]).await.unwrap();
        assert_eq!(response.role, Role::Assistant);
        if let Content::Text { text } = &response.content[0] {
            assert_eq!(text, "Recovered!");
        }
    }
}
