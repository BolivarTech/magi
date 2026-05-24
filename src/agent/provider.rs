//! This module defines the Provider trait for AI backend interactions.

use crate::agent::messages::{Content, Message, Role};
use anyhow::Result;
use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::time::{sleep, Duration};

use crate::tools::Tool;

/// A chunk of a response from the AI.
#[derive(Debug, Clone, PartialEq)]
pub enum ResponseChunk {
    /// A piece of text.
    TextDelta(String),
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
    /// Retry wrapper; used by tests and available to non-streaming callers; production uses
    /// `query_streaming`.
    #[allow(dead_code)]
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
                    let role = Role::Assistant;

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

/// Maximum size the SSE accumulation buffer may reach before a complete event
/// boundary (`"\n\n"`) is found. Guards an unbounded `buffer: String` from OOM on
/// a malformed/hostile stream (audit finding W1). 8 MiB exceeds any legitimate
/// single Anthropic SSE event.
const MAX_SSE_BUFFER_BYTES: usize = 8 * 1024 * 1024;

/// Parses an accumulated `tool_use` input-JSON string. Empty/whitespace → a valid
/// empty object; well-formed JSON is parsed; **malformed JSON returns `Err`** so
/// the caller can log it instead of silently degrading to `{}` (#4).
fn parse_tool_input(acc: &str) -> Result<serde_json::Value, String> {
    if acc.trim().is_empty() {
        return Ok(serde_json::Value::Object(serde_json::Map::new()));
    }
    serde_json::from_str(acc).map_err(|e| e.to_string())
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
struct AnthropicErrorResponse {
    error: AnthropicErrorDetail,
}

#[derive(Debug, Deserialize)]
struct AnthropicErrorDetail {
    #[serde(rename = "type")]
    error_type: String,
    message: String,
}

/// Anthropic SSE Event Types.
/// Fields are deserialized from the wire protocol; not all are read at runtime.
#[allow(dead_code)]
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

/// Wire-protocol message start metadata; `id` and `model` are deserialized but not read.
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct AnthropicMessageStart {
    id: String,
    role: Role,
    model: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicDelta {
    TextDelta {
        text: String,
    },
    // Anthropic's real wire type is "input_json_delta", not "input_delta".
    // The explicit rename overrides the rename_all="snake_case" for this variant.
    #[serde(rename = "input_json_delta")]
    InputDelta {
        partial_json: String,
    },
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
        // Accumulates (id, name, partial_json) for an in-progress tool_use block.
        let mut current_tool: Option<(String, String, String)> = None;

        let output_stream = bytes_stream.flat_map(move |chunk_res| {
            let chunk = match chunk_res {
                Ok(c) => c,
                Err(e) => {
                    return stream::iter(vec![Err(anyhow::anyhow!("Network error: {}", e))]).boxed()
                }
            };
            if buffer.len() + chunk.len() > MAX_SSE_BUFFER_BYTES {
                return stream::iter(vec![Err(anyhow::anyhow!(
                    "SSE buffer would exceed {} bytes without an event boundary; aborting to avoid OOM (limit: 8 MiB)",
                    MAX_SSE_BUFFER_BYTES
                ))])
                .boxed();
            }
            // NOTE (follow-up, future version): `from_utf8_lossy` is applied per
            // network chunk, so a multi-byte UTF-8 character split across a chunk
            // boundary is replaced with U+FFFD. Harmless for SSE control bytes
            // ("\n\n", "data:"); can corrupt rare multi-byte body text. A proper
            // fix buffers raw bytes and decodes once at each event boundary.
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            let mut chunks = Vec::new();
            while let Some(line_end) = buffer.find("\n\n") {
                let block = buffer.drain(..line_end + 2).collect::<String>();
                for line in block.lines() {
                    if let Some(data) = line.strip_prefix("data: ") {
                        if let Ok(event) = serde_json::from_str::<AnthropicSseEvent>(data) {
                            match event {
                                AnthropicSseEvent::MessageStart { message } => {
                                    current_role = message.role;
                                }
                                AnthropicSseEvent::ContentBlockStart {
                                    content_block, ..
                                } => {
                                    // When the block is a tool_use, begin accumulating its input.
                                    if content_block
                                        .get("type")
                                        .and_then(|t| t.as_str())
                                        == Some("tool_use")
                                    {
                                        let id = content_block
                                            .get("id")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or_default()
                                            .to_string();
                                        let name = content_block
                                            .get("name")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or_default()
                                            .to_string();
                                        current_tool = Some((id, name, String::new()));
                                    }
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
                                        // Accumulate into the current tool's JSON buffer.
                                        if let Some((_, _, acc)) = current_tool.as_mut() {
                                            acc.push_str(&partial_json);
                                        }
                                        chunks.push(Ok(ResponseChunk::ToolUseInputDelta {
                                            id: String::new(),
                                            input_json: partial_json,
                                        }));
                                    }
                                },
                                AnthropicSseEvent::ContentBlockStop { .. } => {
                                    // Finalize the accumulated tool_use block and push it to content.
                                    if let Some((id, name, acc)) = current_tool.take() {
                                        let input = parse_tool_input(&acc).unwrap_or_else(|e| {
                                            eprintln!(
                                                "WARNING: malformed tool_use input JSON for tool '{}' (id {}): {}; using empty object",
                                                name, id, e
                                            );
                                            serde_json::Value::Object(serde_json::Map::new())
                                        });
                                        full_content.push(Content::ToolUse { id, name, input });
                                    }
                                }
                                AnthropicSseEvent::MessageStop => {
                                    // Defensively finalize any still-pending tool block
                                    // in case content_block_stop was absent.
                                    if let Some((id, name, acc)) = current_tool.take() {
                                        let input = parse_tool_input(&acc).unwrap_or_else(|e| {
                                            eprintln!(
                                                "WARNING: malformed tool_use input JSON for tool '{}' (id {}): {}; using empty object",
                                                name, id, e
                                            );
                                            serde_json::Value::Object(serde_json::Map::new())
                                        });
                                        full_content.push(Content::ToolUse { id, name, input });
                                    }
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

    #[test]
    fn test_parse_tool_input_empty_is_object() {
        // B-S1: empty / whitespace accumulates to a valid empty object.
        assert_eq!(parse_tool_input("").unwrap(), json!({}));
        assert_eq!(parse_tool_input("   ").unwrap(), json!({}));
    }

    #[test]
    fn test_parse_tool_input_valid_json() {
        // B-S2: well-formed JSON is parsed.
        assert_eq!(
            parse_tool_input(r#"{"path":"."}"#).unwrap(),
            json!({"path":"."})
        );
    }

    #[test]
    fn test_parse_tool_input_malformed_is_err() {
        // B-S3 (load-bearing): malformed JSON surfaces as Err, not a silent {}.
        assert!(parse_tool_input(r#"{"path":"#).is_err());
    }

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
    async fn test_anthropic_provider_sse_buffer_cap_aborts_without_separator() {
        let mut server = Server::new_async().await;
        let url = server.url();

        let oversized = "a".repeat(9 * 1024 * 1024);

        let _m = server
            .mock("POST", "/messages")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(oversized)
            .create_async()
            .await;

        let provider = AnthropicProvider::with_base_url(
            "test_key".to_string(),
            "claude-3-5-sonnet".to_string(),
            url,
        );

        let mut stream = provider.stream_messages(&[], &[]).await.unwrap();

        let mut saw_error = false;
        while let Some(chunk_result) = stream.next().await {
            if let Err(e) = chunk_result {
                let msg = e.to_string();
                assert!(
                    msg.contains("buffer") || msg.contains("8 MiB") || msg.contains("limit"),
                    "error should mention the SSE buffer cap, got: {}",
                    msg
                );
                saw_error = true;
                break;
            }
        }
        assert!(
            saw_error,
            "oversized separator-less stream must abort with an error"
        );
    }

    #[tokio::test]
    async fn test_anthropic_provider_streaming_assembles_tool_use() {
        let mut server = Server::new_async().await;
        let url = server.url();
        // Genuine tool-use SSE wire events: content_block_start carries id/name,
        // two input_json_delta chunks whose partial_json concatenates to {"path": "."},
        // then content_block_stop and message_stop.
        let sse_body = concat!(
            "event: message_start\n",
            "data: {\"type\": \"message_start\", \"message\": {\"id\": \"msg_tu\", \"role\": \"assistant\", \"model\": \"claude-3-5-sonnet\"}}\n\n",
            "event: content_block_start\n",
            "data: {\"type\": \"content_block_start\", \"index\": 0, \"content_block\": {\"type\": \"tool_use\", \"id\": \"toolu_abc\", \"name\": \"ls\", \"input\": {}}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\": \"content_block_delta\", \"index\": 0, \"delta\": {\"type\": \"input_json_delta\", \"partial_json\": \"{\\\"path\\\": \"}}\n\n",
            "event: content_block_delta\n",
            // partial_json piece 2 = "."} — concatenated with piece 1 gives {"path": "."}
            "data: {\"type\": \"content_block_delta\", \"index\": 0, \"delta\": {\"type\": \"input_json_delta\", \"partial_json\": \"\\\".\\\"}\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\": \"content_block_stop\", \"index\": 0}\n\n",
            "event: message_stop\n",
            "data: {\"type\": \"message_stop\"}\n\n",
        );
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
        let response = provider
            .send_messages(&[Message::user("list")], &[])
            .await
            .unwrap();
        let tool = response.content.iter().find_map(|c| match c {
            Content::ToolUse { id, name, input } => Some((id.clone(), name.clone(), input.clone())),
            _ => None,
        });
        let (id, name, input) =
            tool.expect("streaming response must assemble a ToolUse content block");
        assert_eq!(id, "toolu_abc");
        assert_eq!(name, "ls");
        assert_eq!(input, serde_json::json!({"path": "."}));
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
