//! This module defines the Provider trait for AI backend interactions.

use crate::agent::messages::{Content, Message, Role};
use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::time::{sleep, Duration};

use crate::tools::Tool;

/// A chunk of a response from the AI.
#[derive(Debug, Clone, PartialEq)]
pub enum ResponseChunk {
    /// A piece of answer text (persisted into the final message).
    TextDelta(String),
    /// A piece of a reasoning model's chain-of-thought. Surfaced for live display
    /// but NEVER added to the final message (not persisted).
    ReasoningDelta(String),
    /// Input data for a tool use.
    ToolUseInputDelta { id: String, input_json: String },
    /// Completion of a full message.
    MessageDone(Message),
    /// Token usage for the provider turn that just completed.
    ///
    /// Transient — never added to the final [`Message`] and never persisted.
    /// Emitted at most once per `stream_messages` call, only when the backend
    /// actually reports usage (a backend that omits it emits nothing — usage is
    /// never fabricated). Consumed by [`crate::agent::RunObserver::on_usage`] to
    /// accumulate the headless `RunOutcome.usage` totals across agent-loop turns.
    Usage {
        /// Prompt/context tokens billed for the request.
        input_tokens: u64,
        /// Tokens generated in the response.
        output_tokens: u64,
    },
}

/// Trait representing an AI backend provider.
#[async_trait]
pub trait Provider: Send + Sync {
    /// Sends a list of messages to the AI and returns a stream of chunks.
    ///
    /// `system` is the effective system-prompt text for this call, if any
    /// (`None`/empty ⇒ no system prompt is sent — the interactive path always
    /// passes `None`, REQ-H12b). Each implementation applies it however its wire
    /// protocol expects (a top-level field for Anthropic, a prepended `system`
    /// message for OpenAI-compatible backends).
    async fn stream_messages(
        &self,
        messages: &[Message],
        tools: &[Box<dyn Tool>],
        system: Option<&str>,
    ) -> Result<BoxStream<'static, Result<ResponseChunk>>>;

    /// Whether this provider is the canned `StaticProvider` (no API key).
    /// Default `false`; `StaticProvider` overrides to `true`. Lets callers tell
    /// canned startup state from a live provider (#16).
    fn is_static(&self) -> bool {
        false
    }

    /// Sends a list of messages and returns the full message (blocking until done).
    ///
    /// Retry wrapper; used by non-streaming callers (`LlmDistillJudge`, the MAGI
    /// consult adapter) and by tests; production `query_streaming` calls
    /// `stream_messages` directly. Retries up to 3 times, waiting with
    /// exponential backoff (2s, then 4s) between attempts, on rate-limit (429)
    /// or transient connection failures (see [`is_retryable_error`]). A
    /// persistent outage fails after the 3rd attempt with no further wait.
    ///
    /// `system` is forwarded unchanged to [`Provider::stream_messages`] on every
    /// attempt.
    async fn send_messages(
        &self,
        messages: &[Message],
        tools: &[Box<dyn Tool>],
        system: Option<&str>,
    ) -> Result<Message> {
        let mut attempts = 0;
        let max_attempts = 3;

        loop {
            attempts += 1;
            match self.stream_messages(messages, tools, system).await {
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
                Err(e) if attempts < max_attempts && is_retryable_error(&e.to_string()) => {
                    let wait_secs = 2_u64.pow(attempts as u32);
                    sleep(Duration::from_secs(wait_secs)).await;
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
    }
}

/// Maximum size the SSE accumulation buffer may reach before a complete event boundary — **any**
/// of [`SSE_EVENT_BOUNDARIES`], not only `"\n\n"` — is found. Guards an unbounded `Vec<u8>` from
/// OOM on a malformed/hostile stream (audit finding W1). 8 MiB exceeds any legitimate single
/// Anthropic SSE event.
///
/// Two drifts, both from the same cause and both found by S4 Loop 2 (Balthasar). It said `"\n\n"`
/// because the CRLF and CR forms were added below without this doc following, and it said
/// `String` because the buffer became a `Vec<u8>` when byte-level boundary scanning replaced
/// char-level. Each is harmless to the cap itself and each is the kind of drift that makes a
/// later reader "fix" something that already works — or, worse, reason about UTF-8 boundaries in
/// a buffer that no longer has them.
///
/// The second one arrived in the round that fixed the first: correcting one clause of a comment
/// is not reading the comment.
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

/// Byte sequences that terminate an SSE event — "a blank line" in the wire format (Loop 2 gate,
/// S5 finding 5, Caspar). The SSE spec treats CR, LF, and CRLF as interchangeable line
/// terminators, so a boundary is two of them in a row, in any of these three same-terminator
/// forms. `b"\n\n"` alone covers every provider this crate targets directly — Anthropic and
/// every OpenAI-compatible backend (Ollama, OpenAI, Groq, OpenRouter) emit LF — but a proxy or
/// gateway sitting between magi-rs and the endpoint may normalize line endings to CRLF, and
/// without the other two patterns [`drain_sse_events`] would never see a boundary in that
/// stream at all: not a parse error surfaced anywhere, a silent hang that looks exactly like a
/// dead endpoint (the malformed-line tolerance this function's caller relies on only swallows a
/// *complete*, malformed block — an incomplete one just accumulates, unbounded up to
/// [`MAX_SSE_BUFFER_BYTES`]).
///
/// No two of these three patterns can match starting at the same buffer position — their second
/// byte differs pairwise (`\r\n\r\n` and `\r\r` both start with `\r` but diverge at the second
/// byte; `\n\n` starts with `\n`, disjoint from both) — so scanning all three independently and
/// keeping the earliest match needs no further tie-break.
const SSE_EVENT_BOUNDARIES: [&[u8]; 3] = [b"\r\n\r\n", b"\n\n", b"\r\r"];

/// Earliest occurrence of any [`SSE_EVENT_BOUNDARIES`] pattern in `buffer`, as `(start, len)` —
/// the position to cut at and how many bytes the boundary itself occupies.
#[must_use]
fn next_sse_boundary(buffer: &[u8]) -> Option<(usize, usize)> {
    SSE_EVENT_BOUNDARIES
        .iter()
        .filter_map(|pattern| {
            buffer
                .windows(pattern.len())
                .position(|w| w == *pattern)
                .map(|pos| (pos, pattern.len()))
        })
        .min_by_key(|&(pos, _)| pos)
}

/// Drains complete SSE event blocks (terminated by any [`SSE_EVENT_BOUNDARIES`] pattern) from a
/// raw byte buffer, decoding each *complete* block as UTF-8. Buffering raw bytes until the
/// event boundary means a multi-byte UTF-8 character split across network chunks
/// is never decoded mid-character (#3). Incomplete trailing bytes stay buffered.
fn drain_sse_events(buffer: &mut Vec<u8>) -> Vec<String> {
    let mut blocks = Vec::new();
    while let Some((pos, len)) = next_sse_boundary(buffer) {
        let block: Vec<u8> = buffer.drain(..pos + len).collect();
        blocks.push(String::from_utf8_lossy(&block).into_owned());
    }
    blocks
}

/// Finalizes a pending `tool_use` block (if any): parses its accumulated input
/// JSON (malformed → warn + `{}`, #4) and pushes a `Content::ToolUse`. `None` is a
/// no-op. Shared by `content_block_stop`, `message_stop`, and (defensively) a new
/// `content_block_start` (#5/#6).
fn finalize_tool(tool: Option<(String, String, String)>, full_content: &mut Vec<Content>) {
    if let Some((id, name, acc)) = tool {
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

/// Returns `true` when a `send_messages` error should be retried.
///
/// Retryable conditions:
/// - `"429"` in the error string — upstream rate-limit.
/// - `"Could not reach the OpenAI-compatible backend"` — transient connection
///   failure (the stable prefix of [`connection_error_hint`]).
///
/// A persistent outage will still fail after `max_attempts` tries; the
/// distiller then retries on the next scheduled pass (CP2-Z).
fn is_retryable_error(msg: &str) -> bool {
    msg.contains("429") || msg.contains("Could not reach the OpenAI-compatible backend")
}

/// A provider that returns static, canned responses.
pub struct StaticProvider;

#[async_trait]
impl Provider for StaticProvider {
    async fn stream_messages(
        &self,
        _messages: &[Message],
        _tools: &[Box<dyn Tool>],
        _system: Option<&str>,
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

    fn is_static(&self) -> bool {
        true
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
    /// Effective system prompt (top-level field per the Messages API — Anthropic
    /// has no `Role::System` message). Omitted entirely when `None` (REQ-H12b).
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
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
        #[serde(default)]
        usage: AnthropicOutputUsage,
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
    /// Present on every real `message_start` event; absent in older test fixtures
    /// (`#[serde(default)]` keeps those parsing exactly as before).
    #[serde(default)]
    usage: Option<AnthropicStartUsage>,
}

/// `message_start.message.usage` — carries the input-token count known at the
/// start of the turn (`output_tokens` is present but not yet meaningful here;
/// the authoritative output count comes from `message_delta.usage`, below).
#[derive(Debug, Default, Deserialize)]
struct AnthropicStartUsage {
    #[serde(default)]
    input_tokens: u64,
}

/// `message_delta.usage` — carries the cumulative output-token count, updated
/// (possibly more than once) as the turn streams; the last value observed before
/// `message_stop` is the turn's final `output_tokens`.
#[derive(Debug, Default, Deserialize)]
struct AnthropicOutputUsage {
    #[serde(default)]
    output_tokens: u64,
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

// ─── OpenAI-compatible structs ────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct OpenAiRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OpenAiTool>,
    stream: bool,
    /// Requests a final SSE chunk carrying token usage (`choices: []`, `usage:
    /// {...}`) so the provider can surface `ResponseChunk::Usage`. A backend that
    /// ignores the field simply omits the usage chunk — nothing is fabricated.
    stream_options: OpenAiStreamOptions,
}

#[derive(Debug, Serialize)]
struct OpenAiStreamOptions {
    include_usage: bool,
}

#[derive(Debug, Serialize)]
struct OpenAiMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAiToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct OpenAiToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    function: OpenAiToolCallFn,
}

#[derive(Debug, Serialize)]
struct OpenAiToolCallFn {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize)]
struct OpenAiTool {
    #[serde(rename = "type")]
    kind: String,
    function: OpenAiFunction,
}

#[derive(Debug, Serialize)]
struct OpenAiFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

/// Maps magi `Message`s → OpenAI messages (RF-5). Coalesces per magi message:
/// an assistant turn's Text → `content` and its ToolUse blocks → ONE `tool_calls`
/// array (OpenAI requires parallel calls in a single assistant message). A user
/// turn's Text → `role:"user"`; each ToolResult → its own `role:"tool"` message.
///
/// # Ordering guarantee
/// Per-message ordering preserves user → assistant(tool_calls) → tool(results)
/// sequence, satisfying OpenAI's "tool message must follow the assistant
/// tool_calls" contract for the typical single-turn-per-message history magi
/// builds. User Text and ToolResult blocks are emitted **in content order**
/// (each as its own message), so a mixed `[Text, ToolResult]` content list
/// produces `[user, tool]` — not the inverted `[tool, user]` that deferred
/// Text accumulation would cause.
fn map_messages(messages: &[Message]) -> Vec<OpenAiMessage> {
    let mut out = Vec::new();
    for m in messages {
        match m.role {
            Role::Assistant => {
                let mut text: Option<String> = None;
                let mut calls: Vec<OpenAiToolCall> = Vec::new();
                for c in &m.content {
                    match c {
                        Content::Text { text: t } => {
                            text.get_or_insert_with(String::new).push_str(t);
                        }
                        Content::ToolUse { id, name, input } => calls.push(OpenAiToolCall {
                            id: id.clone(),
                            kind: "function".into(),
                            function: OpenAiToolCallFn {
                                name: name.clone(),
                                arguments: input.to_string(),
                            },
                        }),
                        Content::ToolResult { .. } => {} // assistants don't carry tool results
                    }
                }
                if text.is_some() || !calls.is_empty() {
                    out.push(OpenAiMessage {
                        role: "assistant".into(),
                        content: text,
                        tool_calls: if calls.is_empty() { None } else { Some(calls) },
                        tool_call_id: None,
                    });
                }
            }
            Role::User => {
                for c in &m.content {
                    match c {
                        Content::Text { text: t } => out.push(OpenAiMessage {
                            role: "user".into(),
                            content: Some(t.clone()),
                            tool_calls: None,
                            tool_call_id: None,
                        }),
                        Content::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } => out.push(OpenAiMessage {
                            role: "tool".into(),
                            content: Some(content.clone()),
                            tool_calls: None,
                            tool_call_id: Some(tool_use_id.clone()),
                        }),
                        Content::ToolUse { .. } => {} // users don't issue tool calls
                    }
                }
            }
        }
    }
    out
}

fn map_tools(tools: &[Box<dyn Tool>]) -> Vec<OpenAiTool> {
    tools
        .iter()
        .map(|t| OpenAiTool {
            kind: "function".into(),
            function: OpenAiFunction {
                name: t.name().to_string(),
                description: t.description().to_string(),
                parameters: t.input_schema(),
            },
        })
        .collect()
}

// ─── OpenAI stream-deserialization structs ────────────────────────────────────

/// One OpenAI Chat Completions SSE chunk (`data:` payload).
#[derive(Debug, Deserialize)]
struct OpenAiStreamChunk {
    #[serde(default)]
    choices: Vec<OpenAiChoice>,
    /// Present only on the trailing usage chunk requested by `stream_options:
    /// {include_usage: true}` (typically carries `choices: []` alongside it).
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}

/// Token usage reported on the trailing `stream_options.include_usage` chunk.
#[derive(Debug, Deserialize)]
struct OpenAiUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    #[serde(default)]
    delta: OpenAiStreamDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct OpenAiStreamDelta {
    #[serde(default)]
    content: Option<String>,
    /// Reasoning models (kimi, deepseek-r1, …) stream their chain-of-thought here
    /// with empty `content` until the answer. Surfaced live but never persisted.
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenAiToolCallDelta>>,
}

/// A streamed tool-call fragment, keyed by `index` across chunks (Task 5 assembly).
#[derive(Debug, Deserialize)]
struct OpenAiToolCallDelta {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<OpenAiFnDelta>,
}

#[derive(Debug, Default, Deserialize)]
struct OpenAiFnDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

// ─── OpenAI-compatible provider ───────────────────────────────────────────────

/// Connection settings for [`OpenAiCompatibleProvider`]. Named fields make the
/// base_url/api_key/model order a compile-time non-issue (no positional swap).
pub struct OpenAiSettings {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
}

/// Caps tool-call slots from a network-controlled `index` to avoid an unbounded
/// `Vec::resize` (OOM/DoS). 64 ≫ any real parallel-tool-call count. Used by the
/// Task 5 tool-call assembler.
const MAX_TOOL_CALL_SLOTS: usize = 64;

/// A provider that talks to any OpenAI Chat Completions-compatible backend
/// (OpenAI, Ollama, Groq, OpenRouter, …) via a configurable `base_url`.
pub struct OpenAiCompatibleProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
}

impl OpenAiCompatibleProvider {
    /// Constructs the provider with a fresh `reqwest::Client`. No `.timeout(...)`
    /// is set on the client — the OpenAI/Ollama stream is intentionally unbounded
    /// at the client level. Local Ollama can spend tens of seconds on cold-load
    /// before the first SSE event arrives; a total-request timeout (which is what
    /// `reqwest::Client::builder().timeout(...)` configures) would truncate healthy
    /// long streams. Stream-side termination is handled by the unfold state
    /// machine (finalize on `finish_reason` / `[DONE]` / stream-end) plus the
    /// 8 MiB [`MAX_SSE_BUFFER_BYTES`] cap that aborts on unbounded buffering
    /// without an event boundary. (MAGI Checkpoint 2 iter-2 fix.)
    pub fn new(s: OpenAiSettings) -> Self {
        // No total-request timeout: streaming generations (esp. local Ollama) run
        // long; reqwest `.timeout()` is a TOTAL deadline and would truncate healthy
        // streams (MAGI iter-2). Parity with AnthropicProvider (no timeout). Anti-OOM
        // is the bounded tool-call index (Task 5) and the SSE buffer cap, not a
        // timeout.
        Self {
            client: reqwest::Client::new(),
            base_url: s.base_url,
            api_key: s.api_key,
            model: s.model,
        }
    }
}

#[cfg(test)]
impl OpenAiCompatibleProvider {
    /// Test-only accessor exposing the real `reqwest::Client` this provider
    /// built (SC-A19 fix round 1). Hands back the actual client — never a
    /// fabricated stand-in — so a test can inspect its `Debug` output, which
    /// is the only way to observe `reqwest::Client`'s total-timeout
    /// configuration: the type exposes no public getter for it.
    fn client_for_test(&self) -> &reqwest::Client {
        &self.client
    }
}

/// Constructs an OpenAI-compatible provider as a trait object. Single
/// construction site for [`OpenAiSettings`], reused by the principal backend and
/// by per-agent MAGI overrides (same endpoint/key, different model) — RF-8.
///
/// # Arguments
/// * `base_url` - OpenAI-compatible endpoint base URL.
/// * `api_key`  - Bearer token (dummy `"ollama"` accepted by local Ollama).
/// * `model`    - Model name to request.
pub fn build_openai_provider(
    base_url: &str,
    api_key: &str,
    model: &str,
) -> std::sync::Arc<dyn Provider> {
    std::sync::Arc::new(OpenAiCompatibleProvider::new(OpenAiSettings {
        base_url: base_url.to_string(),
        api_key: api_key.to_string(),
        model: model.to_string(),
    }))
}

/// Actionable message when the OpenAI-compatible backend at `base_url` cannot be
/// reached (RF-8). Interpolates `base_url` (= `DEFAULT_OPENAI_BASE_URL` in the
/// no-config Ollama default) and points at the Anthropic opt-in escape hatch.
pub fn connection_error_hint(base_url: &str) -> String {
    format!(
        "Could not reach the OpenAI-compatible backend at {base_url}. \
         If you use Ollama, make sure it is running; if you point at OpenAI or another \
         service, check base_url and OPENAI_API_KEY; or set provider=\"anthropic\" in \
         magi.toml to use Anthropic."
    )
}

/// Streaming state for the OpenAI SSE `unfold`. Owns the byte source and the
/// accumulators for the in-progress assistant message.
struct OaiState {
    /// Boxed byte source derived from `reqwest::Response::bytes_stream()`. Each
    /// item is the raw chunk as `Vec<u8>` (or a network error). Mapping to
    /// `Vec<u8>` at the boundary avoids naming `bytes::Bytes` (not a direct dep)
    /// while preserving the streaming behavior.
    src: BoxStream<'static, Result<Vec<u8>>>,
    /// Raw SSE bytes accumulated until an event boundary (`"\n\n"`).
    buffer: Vec<u8>,
    /// Assembled content blocks for the final assistant message.
    full_content: Vec<Content>,
    /// Per-`index` tool-call accumulators `(id, name, args_json)`, filled by the
    /// streamed tool-call assembler and drained by [`OaiState::finalize`].
    tool_accs: Vec<(String, String, String)>,
    /// Chunks ready to yield from the stream.
    pending: std::collections::VecDeque<Result<ResponseChunk>>,
    /// Whether `MessageDone` was already emitted (idempotent finalize).
    done: bool,
    /// Whether the byte source is exhausted.
    src_done: bool,
    /// Whether a `ResponseChunk::Usage` was already emitted (defensive — the
    /// backend is expected to send the usage chunk at most once).
    usage_emitted: bool,
}

impl OaiState {
    /// Emits the assembled assistant message exactly once (idempotent via `done`).
    /// Drains any accumulated tool-call slots through [`finalize_tool`], skipping
    /// untouched slots (id and name both empty) left by a dropped over-cap index.
    fn finalize(&mut self) {
        if self.done {
            return;
        }
        for acc in std::mem::take(&mut self.tool_accs) {
            // Skip untouched slots (id AND name empty) so a gap left by a dropped
            // over-cap index never emits a ghost ToolUse with an empty name.
            if acc.0.is_empty() && acc.1.is_empty() {
                continue;
            }
            finalize_tool(Some(acc), &mut self.full_content);
        }
        self.pending
            .push_back(Ok(ResponseChunk::MessageDone(Message {
                role: Role::Assistant,
                content: self.full_content.clone(),
            })));
        self.done = true;
    }

    /// Drains complete SSE blocks from `buffer`, pushing `TextDelta` chunks and
    /// finalizing on a `finish_reason` or the `[DONE]` sentinel. Malformed `data:`
    /// payloads are swallowed so one bad event never aborts the stream.
    ///
    /// Once `self.done` is set (the stream was finalized in a previous chunk),
    /// any subsequent post-stop text/tool content is dropped (MAGI Loop 2 caveat
    /// C1) — except a trailing usage-only chunk, which is captured regardless of
    /// `self.done` because it legitimately arrives after `finish_reason`. The
    /// stream-end branch in `stream_messages` still calls `finalize()` directly;
    /// the `done` guard makes that path idempotent.
    fn process_buffer(&mut self) {
        for block in drain_sse_events(&mut self.buffer) {
            for line in block.lines() {
                // Tolerate `data:` with or without the optional space (MAGI iter-2).
                let Some(rest) = line.strip_prefix("data:") else {
                    continue;
                };
                let data = rest.trim_start();
                if data.trim() == "[DONE]" {
                    self.finalize();
                    continue;
                }
                let Ok(parsed) = serde_json::from_str::<OpenAiStreamChunk>(data) else {
                    continue;
                };
                // The trailing usage chunk (`stream_options.include_usage`)
                // legitimately arrives AFTER the `finish_reason` chunk that
                // triggers `finalize()` below, so it is captured unconditionally
                // — even once `self.done` is already set. Every OTHER kind of
                // post-stop content stays suppressed by the `self.done` check
                // just below (MAGI Loop 2 caveat C1: a misbehaving backend must
                // not leak ghost TextDelta/tool-call chunks after the stream is
                // finalized).
                if let Some(usage) = parsed.usage {
                    if !self.usage_emitted {
                        self.usage_emitted = true;
                        self.pending.push_back(Ok(ResponseChunk::Usage {
                            input_tokens: usage.prompt_tokens,
                            output_tokens: usage.completion_tokens,
                        }));
                    }
                }
                if self.done {
                    continue;
                }
                let Some(choice) = parsed.choices.into_iter().next() else {
                    continue;
                };
                // Reasoning (thinking) deltas: surface as a distinct ReasoningDelta
                // for live display, but NEVER add to `full_content` — the persisted
                // message stays answer-only (#24). Presentation is the TUI's job.
                if let Some(reasoning) = choice.delta.reasoning {
                    if !reasoning.is_empty() {
                        self.pending
                            .push_back(Ok(ResponseChunk::ReasoningDelta(reasoning)));
                    }
                }
                if let Some(text) = choice.delta.content {
                    if !text.is_empty() {
                        if let Some(Content::Text { text: existing }) = self.full_content.last_mut()
                        {
                            existing.push_str(&text);
                        } else {
                            self.full_content.push(Content::Text { text: text.clone() });
                        }
                        self.pending.push_back(Ok(ResponseChunk::TextDelta(text)));
                    }
                }
                if let Some(tcs) = choice.delta.tool_calls {
                    for tc in tcs {
                        if tc.index >= MAX_TOOL_CALL_SLOTS {
                            // bound (anti-OOM); warn instead of silent drop (RF-8, MAGI iter-2)
                            eprintln!(
                                "WARNING: tool_call index {} exceeds cap {}; dropping",
                                tc.index, MAX_TOOL_CALL_SLOTS
                            );
                            continue;
                        }
                        if self.tool_accs.len() <= tc.index {
                            self.tool_accs.resize(
                                tc.index + 1,
                                (String::new(), String::new(), String::new()),
                            );
                        }
                        let slot = &mut self.tool_accs[tc.index];
                        if let Some(id) = tc.id {
                            if !id.is_empty() {
                                slot.0 = id;
                            }
                        }
                        if let Some(f) = tc.function {
                            if let Some(name) = f.name {
                                if !name.is_empty() {
                                    slot.1 = name;
                                }
                            }
                            if let Some(args) = f.arguments {
                                // MAGI Loop 2 caveat C2: an `arguments`-only fragment
                                // arriving before any fragment carries id or name
                                // means we have nothing to attach the args to (the
                                // resulting ToolUse would lack id+name and is dropped
                                // by `finalize`'s untouched-slot skip — so the args
                                // are lost regardless). Drop the entire fragment
                                // here: do not accumulate, do not push a delta with
                                // an empty id.
                                if slot.0.is_empty() && slot.1.is_empty() {
                                    eprintln!(
                                        "WARNING: tool_call args fragment arrived at slot {} before id/name; skipping",
                                        tc.index
                                    );
                                    continue;
                                }
                                slot.2.push_str(&args);
                                // MAGI: OpenAI streams id+name in the first chunk; subsequent
                                // chunks carry args only — `slot.0` is already populated by the
                                // time any args arrive (args-before-id never occurs in a well-
                                // formed stream). The C2 guard above defends the misbehaving
                                // case; here `slot.0` is guaranteed non-empty when we reach
                                // the delta push, so the delta carries a real id.
                                self.pending.push_back(Ok(ResponseChunk::ToolUseInputDelta {
                                    id: slot.0.clone(),
                                    input_json: args,
                                }));
                            }
                        }
                    }
                }
                if choice.finish_reason.is_some() {
                    self.finalize();
                }
            }
        }
    }
}

#[async_trait]
impl Provider for OpenAiCompatibleProvider {
    async fn stream_messages(
        &self,
        messages: &[Message],
        tools: &[Box<dyn Tool>],
        system: Option<&str>,
    ) -> Result<BoxStream<'static, Result<ResponseChunk>>> {
        let url = format!("{}/chat/completions", self.base_url);
        let mut oai_messages = map_messages(messages);
        // OpenAI Chat Completions has no top-level system field (unlike
        // Anthropic) — a system prompt is a regular message, prepended so it
        // precedes every other turn (REQ-H12b).
        if let Some(sys) = system.filter(|s| !s.is_empty()) {
            oai_messages.insert(
                0,
                OpenAiMessage {
                    role: "system".to_string(),
                    content: Some(sys.to_string()),
                    tool_calls: None,
                    tool_call_id: None,
                },
            );
        }
        let request = OpenAiRequest {
            model: self.model.clone(),
            messages: oai_messages,
            tools: map_tools(tools),
            stream: true,
            stream_options: OpenAiStreamOptions {
                include_usage: true,
            },
        };
        let response = self
            .client
            .post(&url)
            .header("authorization", format!("Bearer {}", self.api_key))
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .with_context(|| connection_error_hint(&self.base_url))?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("OpenAI API Error [{}]: {}", status, body));
        }
        let st = OaiState {
            src: response
                .bytes_stream()
                .map(|r| {
                    r.map(|b| b.to_vec())
                        .map_err(|e| anyhow::anyhow!("Network error: {}", e))
                })
                .boxed(),
            buffer: Vec::new(),
            full_content: Vec::new(),
            tool_accs: Vec::new(),
            pending: std::collections::VecDeque::new(),
            done: false,
            src_done: false,
            usage_emitted: false,
        };
        let out = stream::unfold(st, |mut st| async move {
            loop {
                if let Some(item) = st.pending.pop_front() {
                    return Some((item, st));
                }
                if st.src_done {
                    return None;
                }
                match st.src.next().await {
                    Some(Ok(chunk)) => {
                        if st.buffer.len() + chunk.len() > MAX_SSE_BUFFER_BYTES {
                            st.pending.push_back(Err(anyhow::anyhow!(
                                "SSE buffer would exceed {} bytes without an event boundary; aborting to avoid OOM (limit: 8 MiB)",
                                MAX_SSE_BUFFER_BYTES
                            )));
                            st.src_done = true;
                        } else {
                            st.buffer.extend_from_slice(&chunk);
                            st.process_buffer();
                        }
                    }
                    Some(Err(e)) => {
                        // Error already carries the "Network error: …" context from
                        // the byte-stream map above.
                        st.pending.push_back(Err(e));
                        st.src_done = true;
                    }
                    None => {
                        // Stream-end (MAGI fix c + iter-2): flush a trailing event that
                        // lacks the final blank line, then emit MessageDone.
                        st.src_done = true;
                        if !st.buffer.is_empty() {
                            st.buffer.extend_from_slice(b"\n\n");
                            st.process_buffer();
                        }
                        st.finalize();
                    }
                }
            }
        });
        Ok(Box::pin(out))
    }
}

// ─── Anthropic provider ───────────────────────────────────────────────────────

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
        system: Option<&str>,
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
            // Anthropic takes the system prompt as a dedicated top-level field,
            // never as a message (REQ-H12b). Empty/`None` ⇒ omitted entirely.
            system: system.filter(|s| !s.is_empty()).map(str::to_string),
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

        let bytes_stream = response
            .bytes_stream()
            .map(|r| {
                r.map(|b| b.to_vec())
                    .map_err(|e| anyhow::anyhow!("Network error: {}", e))
            })
            .boxed();

        let state = AnthropicState {
            src: bytes_stream,
            buffer: Vec::new(),
            full_content: Vec::new(),
            current_role: Role::Assistant,
            current_tool: None,
            usage_input_tokens: 0,
            usage_output_tokens: 0,
            usage_seen: false,
            pending: std::collections::VecDeque::new(),
            done: false,
            src_done: false,
        };

        // Mirrors `OaiState`'s unfold shape: drain any pending chunks first, then
        // pull more bytes from the source, finalizing on stream-end (`None`) so a
        // connection that closes without a `message_stop` still flushes whatever
        // content/tool calls were assembled (Gap 2 fix) instead of silently
        // dropping them.
        let output_stream = stream::unfold(state, |mut st| async move {
            loop {
                if let Some(item) = st.pending.pop_front() {
                    return Some((item, st));
                }
                if st.src_done {
                    return None;
                }
                match st.src.next().await {
                    Some(Ok(chunk)) => {
                        if st.buffer.len() + chunk.len() > MAX_SSE_BUFFER_BYTES {
                            st.pending.push_back(Err(anyhow::anyhow!(
                                "SSE buffer would exceed {} bytes without an event boundary; aborting to avoid OOM (limit: 8 MiB)",
                                MAX_SSE_BUFFER_BYTES
                            )));
                            st.src_done = true;
                        } else {
                            // #3: buffer raw bytes and decode once at each event
                            // boundary, so a multi-byte UTF-8 character split across a
                            // network chunk is never decoded mid-character (the W1
                            // size cap above still applies in bytes).
                            st.buffer.extend_from_slice(&chunk);
                            st.process_buffer();
                        }
                    }
                    Some(Err(e)) => {
                        // Error already carries the "Network error: …" context from
                        // the byte-stream map above.
                        st.pending.push_back(Err(e));
                        st.src_done = true;
                    }
                    None => {
                        // Stream-end: flush a trailing event that lacks the final
                        // blank line, then finalize a truncated turn if
                        // `message_stop` never arrived (Gap 2 fix).
                        st.src_done = true;
                        if !st.buffer.is_empty() {
                            st.buffer.extend_from_slice(b"\n\n");
                            st.process_buffer();
                        }
                        st.finalize_truncated();
                    }
                }
            }
        });

        Ok(Box::pin(output_stream))
    }
}

/// Streaming state for the Anthropic SSE `unfold`. Owns the byte source and the
/// accumulators for the in-progress assistant message; mirrors [`OaiState`]'s
/// shape so both providers share the same stream-end-finalization guarantee.
struct AnthropicState {
    /// Boxed byte source derived from `reqwest::Response::bytes_stream()`,
    /// already mapped to `Vec<u8>`/`anyhow::Error` at the boundary.
    src: BoxStream<'static, Result<Vec<u8>>>,
    /// Raw SSE bytes accumulated until an event boundary (`"\n\n"`).
    buffer: Vec<u8>,
    /// Assembled content blocks for the final assistant message.
    full_content: Vec<Content>,
    /// Role carried by `message_start`; defaults to `Assistant` if the turn is
    /// truncated before `message_start` ever arrives.
    current_role: Role,
    /// Accumulates `(id, name, partial_json)` for an in-progress `tool_use` block.
    current_tool: Option<(String, String, String)>,
    /// `message_start.message.usage.input_tokens`, if reported.
    usage_input_tokens: u64,
    /// Cumulative `message_delta.usage.output_tokens`, if reported.
    usage_output_tokens: u64,
    /// Whether real usage data was observed (a `message_start` with a `usage`
    /// object, or any `message_delta` event) — distinct from the zero defaults
    /// left by `#[serde(default)]`. Gates the `Usage` emission on both the
    /// `message_stop` path and [`AnthropicState::finalize_truncated`] so
    /// neither a well-formed close nor an abnormal one ever fabricates a
    /// `(0, 0)` usage reading (REQ-H14).
    usage_seen: bool,
    /// Chunks ready to yield from the stream.
    pending: std::collections::VecDeque<Result<ResponseChunk>>,
    /// Whether `MessageDone` was already emitted (idempotent finalize).
    done: bool,
    /// Whether the byte source is exhausted.
    src_done: bool,
}

impl AnthropicState {
    /// Drains complete SSE blocks from `buffer`, pushing `TextDelta`/
    /// `ToolUseInputDelta` chunks, updating usage/content accumulators, and
    /// finalizing on `message_stop`. Malformed `data:` payloads and lines missing
    /// the `"data: "` prefix are swallowed so one bad event never aborts the
    /// stream (matches the pre-existing behavior).
    ///
    /// Once `self.done` is set (the stream was already finalized by a prior
    /// `message_stop` or by [`AnthropicState::finalize_truncated`]), every
    /// subsequent event is dropped before it can touch any accumulator — this
    /// mirrors [`OaiState::process_buffer`]'s post-finalize ghost-content guard
    /// (MAGI Loop 2 caveat C1) so a misbehaving backend that keeps sending
    /// `content_block_delta`/`content_block_start` after the turn already
    /// finalized can never leak a second `TextDelta`/tool chunk or corrupt the
    /// already-emitted message. Unlike `OaiState`, Anthropic has no trailing
    /// usage-only event that legitimately arrives after finalization (usage
    /// rides on `message_start`/`message_delta`, both pre-`message_stop`), so
    /// the guard is unconditional here.
    fn process_buffer(&mut self) {
        for block in drain_sse_events(&mut self.buffer) {
            for line in block.lines() {
                let Some(data) = line.strip_prefix("data: ") else {
                    continue;
                };
                let Ok(event) = serde_json::from_str::<AnthropicSseEvent>(data) else {
                    continue;
                };
                if self.done {
                    continue;
                }
                match event {
                    AnthropicSseEvent::MessageStart { message } => {
                        self.current_role = message.role;
                        if message.usage.is_some() {
                            self.usage_seen = true;
                        }
                        self.usage_input_tokens =
                            message.usage.map(|u| u.input_tokens).unwrap_or(0);
                    }
                    AnthropicSseEvent::ContentBlockStart { content_block, .. } => {
                        // Defensively finalize any still-open tool before starting a
                        // new block, so a missing content_block_stop never drops it (#6).
                        finalize_tool(self.current_tool.take(), &mut self.full_content);
                        // When the block is a tool_use, begin accumulating its input.
                        if content_block.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
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
                            self.current_tool = Some((id, name, String::new()));
                        }
                    }
                    AnthropicSseEvent::ContentBlockDelta { delta, .. } => match delta {
                        AnthropicDelta::TextDelta { text } => {
                            if let Some(Content::Text { text: existing }) =
                                self.full_content.last_mut()
                            {
                                existing.push_str(&text);
                            } else {
                                self.full_content.push(Content::Text { text: text.clone() });
                            }
                            self.pending.push_back(Ok(ResponseChunk::TextDelta(text)));
                        }
                        AnthropicDelta::InputDelta { partial_json } => {
                            // Accumulate into the current tool's JSON buffer and tag the
                            // chunk with the in-progress tool id (#6).
                            let id = if let Some((id, _, acc)) = self.current_tool.as_mut() {
                                acc.push_str(&partial_json);
                                id.clone()
                            } else {
                                String::new()
                            };
                            self.pending.push_back(Ok(ResponseChunk::ToolUseInputDelta {
                                id,
                                input_json: partial_json,
                            }));
                        }
                    },
                    AnthropicSseEvent::ContentBlockStop { .. } => {
                        // Finalize the accumulated tool_use block and push it to content.
                        finalize_tool(self.current_tool.take(), &mut self.full_content);
                    }
                    AnthropicSseEvent::MessageDelta { usage, .. } => {
                        // Cumulative output-token count; the last value observed
                        // before message_stop is authoritative.
                        self.usage_seen = true;
                        self.usage_output_tokens = usage.output_tokens;
                    }
                    AnthropicSseEvent::MessageStop => {
                        // A duplicate/late message_stop can never reach this arm:
                        // the top-of-loop `if self.done { continue; }` guard above
                        // already drops it once the first message_stop set `done`.
                        // Defensively finalize any still-pending tool block in case
                        // content_block_stop was absent.
                        finalize_tool(self.current_tool.take(), &mut self.full_content);
                        // Only emit Usage when real usage data was actually observed
                        // (same gate as `finalize_truncated`) — a degenerate stream
                        // that never reports usage must not fabricate a (0, 0)
                        // reading (MAGI re-gate WARNING 2).
                        if self.usage_seen {
                            self.pending.push_back(Ok(ResponseChunk::Usage {
                                input_tokens: self.usage_input_tokens,
                                output_tokens: self.usage_output_tokens,
                            }));
                        }
                        let msg = Message {
                            role: self.current_role.clone(),
                            content: self.full_content.clone(),
                        };
                        self.pending.push_back(Ok(ResponseChunk::MessageDone(msg)));
                        self.done = true;
                    }
                    AnthropicSseEvent::Error { error } => {
                        // Gap 1 fix: a well-formed mid-stream `error` event (e.g.
                        // "overloaded_error") surfaces as Err instead of being
                        // swallowed. The message is the API's own description, safe
                        // to show — no request header/key is echoed.
                        self.pending.push_back(Err(anyhow::anyhow!(
                            "Anthropic API error ({}): {}",
                            error.error_type,
                            error.message
                        )));
                    }
                    AnthropicSseEvent::Ping => {}
                }
            }
        }
    }

    /// Finalizes the assembled assistant message when the byte source closes
    /// without a prior `message_stop` event — a truncated/dropped connection.
    /// Idempotent via `done` (a no-op once `message_stop` already finalized the
    /// turn). Mirrors [`OaiState::finalize`]'s stream-end flush (Gap 2 fix):
    /// without this, a mid-turn disconnect silently dropped every
    /// already-streamed `TextDelta`/`ToolUseInputDelta`'s accumulated content —
    /// those chunks had already reached the caller, but no `MessageDone` ever
    /// followed, so `run_tool_loop` errored with "Stream ended without
    /// MessageDone" and the assistant's partial turn was lost, not persisted.
    fn finalize_truncated(&mut self) {
        if self.done {
            return;
        }
        finalize_tool(self.current_tool.take(), &mut self.full_content);
        // Unlike the unconditional message_stop emission above, only emit Usage
        // here when real usage data was actually observed — an abnormal close
        // must not fabricate a (0, 0) reading.
        if self.usage_seen {
            self.pending.push_back(Ok(ResponseChunk::Usage {
                input_tokens: self.usage_input_tokens,
                output_tokens: self.usage_output_tokens,
            }));
        }
        let msg = Message {
            role: self.current_role.clone(),
            content: self.full_content.clone(),
        };
        self.pending.push_back(Ok(ResponseChunk::MessageDone(msg)));
        self.done = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::messages::{Content, Role};
    use mockito::Server;
    use serde_json::json;
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

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

    #[test]
    fn test_drain_sse_events_handles_multibyte_split_across_chunks() {
        // C-S1 (load-bearing): 'é' (0xC3 0xA9) split across two pushes must not
        // corrupt to U+FFFD — bytes are buffered until the "\n\n" boundary.
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice("data: caf".as_bytes());
        buf.push(0xC3);
        assert!(
            drain_sse_events(&mut buf).is_empty(),
            "no event before the boundary"
        );
        buf.push(0xA9);
        buf.extend_from_slice(b"\n\n");
        assert_eq!(
            drain_sse_events(&mut buf),
            vec!["data: café\n\n".to_string()]
        );
        assert!(buf.is_empty());
    }

    #[test]
    fn test_drain_sse_events_multiple_events_and_remainder() {
        // C-S2: drain all complete blocks, leave the incomplete tail buffered.
        let mut buf: Vec<u8> = b"event: a\n\nevent: b\n\nevent: c-incomplete".to_vec();
        assert_eq!(
            drain_sse_events(&mut buf),
            vec!["event: a\n\n".to_string(), "event: b\n\n".to_string()]
        );
        assert_eq!(buf, b"event: c-incomplete".to_vec());
    }

    /// Loop 2 gate, S5 finding 5 (Caspar): a CRLF-framed stream — as a proxy or gateway between
    /// magi-rs and the endpoint may normalize line endings to — must still yield event
    /// boundaries. Before the fix this would never match (`\r\n\r\n` contains no literal `\n\n`
    /// substring, since the two `\n`s are separated by `\r`), and the buffer would grow
    /// unboundedly instead of ever draining an event.
    #[test]
    fn test_drain_sse_events_recognizes_crlf_framing() {
        let mut buf: Vec<u8> =
            b"event: a\r\ndata: 1\r\n\r\nevent: b\r\ndata: 2\r\n\r\nevent: c-incomplete\r\n"
                .to_vec();
        assert_eq!(
            drain_sse_events(&mut buf),
            vec![
                "event: a\r\ndata: 1\r\n\r\n".to_string(),
                "event: b\r\ndata: 2\r\n\r\n".to_string(),
            ]
        );
        assert_eq!(buf, b"event: c-incomplete\r\n".to_vec());
    }

    /// Same guarantee for the bare-CR line-terminator form (`\r\r`) — less likely in practice
    /// than CRLF, but the SSE spec permits CR alone as a line terminator too, and the fix covers
    /// it by the same mechanism.
    #[test]
    fn test_drain_sse_events_recognizes_bare_cr_framing() {
        let mut buf: Vec<u8> = b"event: a\rdata: 1\r\r".to_vec();
        assert_eq!(
            drain_sse_events(&mut buf),
            vec!["event: a\rdata: 1\r\r".to_string()]
        );
        assert!(buf.is_empty());
    }

    /// The multibyte-split guarantee (C-S1) must hold under CRLF framing too, not just `\n\n` —
    /// a fix that only widened the boundary set without preserving the byte-buffering discipline
    /// would reopen the original mid-character corruption bug for CRLF-framed streams.
    #[test]
    fn test_drain_sse_events_handles_multibyte_split_across_chunks_under_crlf() {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice("data: caf".as_bytes());
        buf.push(0xC3);
        assert!(
            drain_sse_events(&mut buf).is_empty(),
            "no event before the boundary"
        );
        buf.push(0xA9);
        buf.extend_from_slice(b"\r\n\r\n");
        assert_eq!(
            drain_sse_events(&mut buf),
            vec!["data: café\r\n\r\n".to_string()]
        );
        assert!(buf.is_empty());
    }

    #[test]
    fn test_finalize_tool_pushes_parsed_tooluse() {
        // A-S1: valid accumulated input is parsed into a ToolUse.
        let mut content: Vec<Content> = Vec::new();
        finalize_tool(
            Some(("id1".into(), "ls".into(), r#"{"path":"."}"#.into())),
            &mut content,
        );
        assert_eq!(
            content,
            vec![Content::ToolUse {
                id: "id1".into(),
                name: "ls".into(),
                input: json!({"path":"."}),
            }]
        );
    }

    #[test]
    fn test_finalize_tool_empty_input_is_object() {
        // A-S2: empty accumulated input becomes an empty object.
        let mut content: Vec<Content> = Vec::new();
        finalize_tool(Some(("id".into(), "n".into(), String::new())), &mut content);
        assert_eq!(
            content,
            vec![Content::ToolUse {
                id: "id".into(),
                name: "n".into(),
                input: json!({}),
            }]
        );
    }

    #[test]
    fn test_finalize_tool_none_is_noop() {
        // A-S3: no pending tool → nothing pushed.
        let mut content: Vec<Content> = Vec::new();
        finalize_tool(None, &mut content);
        assert!(content.is_empty());
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
        let response = provider.send_messages(&messages, &[], None).await.unwrap();
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
        let response = provider.send_messages(&messages, &[], None).await.unwrap();
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

        let result = provider.send_messages(&[], &[], None).await;
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

        let mut stream = provider.stream_messages(&[], &[], None).await.unwrap();

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

        let mut stream = provider.stream_messages(&[], &[], None).await.unwrap();

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

        let mut stream = provider.stream_messages(&[], &[], None).await.unwrap();

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
            .send_messages(&[Message::user("list")], &[], None)
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
        // #6a no-double-push: the normal start→delta→stop→message_stop flow must
        // assemble exactly ONE ToolUse (the start-time defensive finalize no-ops).
        let tool_count = response
            .content
            .iter()
            .filter(|c| matches!(c, Content::ToolUse { .. }))
            .count();
        assert_eq!(
            tool_count, 1,
            "normal flow must assemble exactly one ToolUse (no double-push)"
        );
    }

    #[tokio::test]
    async fn test_missing_content_block_stop_does_not_drop_prior_tool() {
        // B-S1 (#6a): two tool_use blocks with NO content_block_stop between them
        // (only message_stop at the end). Both must be assembled — the first tool
        // must not be dropped when the second content_block_start arrives.
        let mut server = Server::new_async().await;
        let url = server.url();
        let sse_body = concat!(
            "event: message_start\n",
            "data: {\"type\": \"message_start\", \"message\": {\"id\": \"m\", \"role\": \"assistant\", \"model\": \"x\"}}\n\n",
            "event: content_block_start\n",
            "data: {\"type\": \"content_block_start\", \"index\": 0, \"content_block\": {\"type\": \"tool_use\", \"id\": \"toolu_A\", \"name\": \"ls\", \"input\": {}}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\": \"content_block_delta\", \"index\": 0, \"delta\": {\"type\": \"input_json_delta\", \"partial_json\": \"{}\"}}\n\n",
            "event: content_block_start\n",
            "data: {\"type\": \"content_block_start\", \"index\": 1, \"content_block\": {\"type\": \"tool_use\", \"id\": \"toolu_B\", \"name\": \"view\", \"input\": {}}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\": \"content_block_delta\", \"index\": 1, \"delta\": {\"type\": \"input_json_delta\", \"partial_json\": \"{}\"}}\n\n",
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
        let provider = AnthropicProvider::with_base_url("k".to_string(), "x".to_string(), url);
        let response = provider
            .send_messages(&[Message::user("go")], &[], None)
            .await
            .unwrap();
        let ids: Vec<String> = response
            .content
            .iter()
            .filter_map(|c| match c {
                Content::ToolUse { id, .. } => Some(id.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            ids,
            vec!["toolu_A".to_string(), "toolu_B".to_string()],
            "a missing content_block_stop must not drop the first tool"
        );
    }

    #[tokio::test]
    async fn test_tool_input_delta_chunk_carries_tool_id() {
        // B-S2 (#6b): the ToolUseInputDelta chunk must carry the in-progress tool id.
        let mut server = Server::new_async().await;
        let url = server.url();
        let sse_body = concat!(
            "event: message_start\n",
            "data: {\"type\": \"message_start\", \"message\": {\"id\": \"m\", \"role\": \"assistant\", \"model\": \"x\"}}\n\n",
            "event: content_block_start\n",
            "data: {\"type\": \"content_block_start\", \"index\": 0, \"content_block\": {\"type\": \"tool_use\", \"id\": \"toolu_x\", \"name\": \"ls\", \"input\": {}}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\": \"content_block_delta\", \"index\": 0, \"delta\": {\"type\": \"input_json_delta\", \"partial_json\": \"{}\"}}\n\n",
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
        let provider = AnthropicProvider::with_base_url("k".to_string(), "x".to_string(), url);
        let mut stream = provider
            .stream_messages(&[Message::user("go")], &[], None)
            .await
            .unwrap();
        let mut delta_ids = Vec::new();
        while let Some(chunk) = stream.next().await {
            if let Ok(ResponseChunk::ToolUseInputDelta { id, .. }) = chunk {
                delta_ids.push(id);
            }
        }
        assert_eq!(
            delta_ids,
            vec!["toolu_x".to_string()],
            "ToolUseInputDelta chunk must carry the tool id"
        );
    }

    // ─── Task 3: map_messages / map_tools tests ───────────────────────────────

    #[test]
    fn test_map_user_and_assistant_text() {
        let v = serde_json::to_value(map_messages(&[
            Message::user("hi"),
            Message::assistant("yo"),
        ]))
        .unwrap();
        assert_eq!(v[0]["role"], "user");
        assert_eq!(v[0]["content"], "hi");
        assert_eq!(v[1]["role"], "assistant");
        assert_eq!(v[1]["content"], "yo");
    }

    #[test]
    fn test_map_parallel_tooluse_coalesced_into_one_assistant_message() {
        // MAGI fix a: two ToolUse in ONE assistant turn → ONE message, tool_calls len 2.
        let msgs = vec![Message {
            role: Role::Assistant,
            content: vec![
                Content::ToolUse {
                    id: "c1".into(),
                    name: "ls".into(),
                    input: json!({"path": "."}),
                },
                Content::ToolUse {
                    id: "c2".into(),
                    name: "view".into(),
                    input: json!({"path": "a"}),
                },
            ],
        }];
        let v = serde_json::to_value(map_messages(&msgs)).unwrap();
        assert_eq!(
            v.as_array().unwrap().len(),
            1,
            "parallel tool calls must be ONE assistant message"
        );
        assert_eq!(v[0]["role"], "assistant");
        assert_eq!(v[0]["tool_calls"].as_array().unwrap().len(), 2);
        assert_eq!(v[0]["tool_calls"][0]["id"], "c1");
        assert_eq!(v[0]["tool_calls"][0]["function"]["name"], "ls");
        assert_eq!(v[0]["tool_calls"][1]["id"], "c2");
    }

    #[test]
    fn test_map_assistant_text_plus_tooluse_one_message() {
        let msgs = vec![Message {
            role: Role::Assistant,
            content: vec![
                Content::Text {
                    text: "calling".into(),
                },
                Content::ToolUse {
                    id: "c1".into(),
                    name: "ls".into(),
                    input: json!({}),
                },
            ],
        }];
        let v = serde_json::to_value(map_messages(&msgs)).unwrap();
        assert_eq!(v.as_array().unwrap().len(), 1);
        assert_eq!(v[0]["content"], "calling");
        assert_eq!(v[0]["tool_calls"][0]["id"], "c1");
    }

    #[test]
    fn test_map_toolresult_becomes_tool_role() {
        // S-6: User + ToolResult → role:"tool" (one message per result).
        let msgs = vec![Message {
            role: Role::User,
            content: vec![Content::ToolResult {
                tool_use_id: "c1".into(),
                content: "out".into(),
                is_error: false,
            }],
        }];
        let v = serde_json::to_value(map_messages(&msgs)).unwrap();
        assert_eq!(v[0]["role"], "tool");
        assert_eq!(v[0]["tool_call_id"], "c1");
        assert_eq!(v[0]["content"], "out");
    }

    #[test]
    fn test_map_user_text_before_toolresult_preserves_order() {
        // Mixed User content [Text, ToolResult] must map in order: user then tool.
        let msgs = vec![Message {
            role: Role::User,
            content: vec![
                Content::Text { text: "ctx".into() },
                Content::ToolResult {
                    tool_use_id: "c1".into(),
                    content: "out".into(),
                    is_error: false,
                },
            ],
        }];
        let v = serde_json::to_value(map_messages(&msgs)).unwrap();
        assert_eq!(v.as_array().unwrap().len(), 2);
        assert_eq!(v[0]["role"], "user");
        assert_eq!(v[0]["content"], "ctx");
        assert_eq!(v[1]["role"], "tool");
        assert_eq!(v[1]["tool_call_id"], "c1");
    }

    // ─── Anthropic retry test (unmodified) ────────────────────────────────────

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

        let response = provider.send_messages(&[], &[], None).await.unwrap();
        assert_eq!(response.role, Role::Assistant);
        if let Content::Text { text } = &response.content[0] {
            assert_eq!(text, "Recovered!");
        }
    }

    // ─── OpenAiCompatibleProvider (Task 4: text streaming) ────────────────────

    #[tokio::test]
    async fn test_openai_streams_text_finalizes_on_done() {
        let mut server = Server::new_async().await;
        let url = server.url();
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hello \"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"world!\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        let _m = server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse)
            .create_async()
            .await;
        let p = OpenAiCompatibleProvider::new(OpenAiSettings {
            base_url: url,
            api_key: "k".into(),
            model: "m".into(),
        });
        let r = p
            .send_messages(&[Message::user("hi")], &[], None)
            .await
            .unwrap();
        assert_eq!(
            r.content,
            vec![Content::Text {
                text: "Hello world!".into()
            }]
        );
    }

    #[tokio::test]
    async fn test_openai_streams_reasoning_visibly_but_does_not_persist_it() {
        // v0.5.2 (#24): reasoning models (kimi-k2.6, deepseek-r1) emit their
        // chain-of-thought in `delta.reasoning` with empty `delta.content`. The
        // provider surfaces it as a distinct `ResponseChunk::ReasoningDelta` (so the
        // TUI can show it or just an activity indicator) WITHOUT persisting it — the
        // finalized message keeps only the `content` answer.
        let mut server = Server::new_async().await;
        let url = server.url();
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"\",\"reasoning\":\"Let me think\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"\",\"reasoning\":\" about it\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"Answer.\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        let _m = server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse)
            .create_async()
            .await;
        let p = OpenAiCompatibleProvider::new(OpenAiSettings {
            base_url: url,
            api_key: "k".into(),
            model: "m".into(),
        });
        let mut stream = p
            .stream_messages(&[Message::user("hi")], &[], None)
            .await
            .unwrap();
        let mut reasoning = String::new();
        let mut content = String::new();
        let mut final_msg: Option<Message> = None;
        while let Some(chunk) = stream.next().await {
            match chunk.unwrap() {
                ResponseChunk::ReasoningDelta(r) => reasoning.push_str(&r),
                ResponseChunk::TextDelta(t) => content.push_str(&t),
                ResponseChunk::MessageDone(m) => final_msg = Some(m),
                _ => {}
            }
        }
        // Reasoning is surfaced as its OWN signal (raw text, no presentation — the
        // TUI decides how to display it), kept separate from the answer content.
        assert_eq!(reasoning, "Let me think about it");
        assert_eq!(content, "Answer.");
        // Not persisted: the finalized message is content-only (no reasoning).
        assert_eq!(
            final_msg.expect("MessageDone").content,
            vec![Content::Text {
                text: "Answer.".into()
            }]
        );
    }

    #[tokio::test]
    async fn test_openai_finalizes_without_done_sentinel() {
        // MAGI fix c: backend omits [DONE]; stream-end must still flush a MessageDone.
        let mut server = Server::new_async().await;
        let url = server.url();
        let sse =
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n";
        let _m = server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse)
            .create_async()
            .await;
        let p = OpenAiCompatibleProvider::new(OpenAiSettings {
            base_url: url,
            api_key: "k".into(),
            model: "m".into(),
        });
        let r = p
            .send_messages(&[Message::user("hi")], &[], None)
            .await
            .unwrap();
        assert_eq!(r.content, vec![Content::Text { text: "hi".into() }]);
    }

    #[tokio::test]
    async fn test_openai_swallows_malformed_line() {
        let mut server = Server::new_async().await;
        let url = server.url();
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Valid\"}}]}\n\n",
            "data: {MALFORMED}\n\n",
            "data: [DONE]\n\n"
        );
        let _m = server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse)
            .create_async()
            .await;
        let p = OpenAiCompatibleProvider::new(OpenAiSettings {
            base_url: url,
            api_key: "k".into(),
            model: "m".into(),
        });
        let r = p
            .send_messages(&[Message::user("hi")], &[], None)
            .await
            .unwrap();
        assert_eq!(
            r.content,
            vec![Content::Text {
                text: "Valid".into()
            }]
        );
    }

    #[tokio::test]
    async fn test_openai_http_error_surfaces() {
        let mut server = Server::new_async().await;
        let url = server.url();
        let _m = server
            .mock("POST", "/chat/completions")
            .with_status(401)
            .with_body("{\"error\":{\"message\":\"bad key\"}}")
            .create_async()
            .await;
        let p = OpenAiCompatibleProvider::new(OpenAiSettings {
            base_url: url,
            api_key: "k".into(),
            model: "m".into(),
        });
        assert!(p
            .send_messages(&[Message::user("hi")], &[], None)
            .await
            .unwrap_err()
            .to_string()
            .contains("401"));
    }

    // ─── OpenAiCompatibleProvider (Task 5: tool_calls assembly) ───────────────

    #[tokio::test]
    async fn test_openai_assembles_fragmented_tool_call() {
        // S-5: id+name first, arguments fragmented → ONE Content::ToolUse {"path":"."}.
        let mut server = Server::new_async().await;
        let url = server.url();
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_x\",\"function\":{\"name\":\"ls\",\"arguments\":\"{\\\"path\\\":\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\".\\\"}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        let _m = server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse)
            .create_async()
            .await;
        let p = OpenAiCompatibleProvider::new(OpenAiSettings {
            base_url: url,
            api_key: "k".into(),
            model: "m".into(),
        });
        let r = p
            .send_messages(&[Message::user("list")], &[], None)
            .await
            .unwrap();
        let tool = r
            .content
            .iter()
            .find_map(|c| match c {
                Content::ToolUse { id, name, input } => {
                    Some((id.clone(), name.clone(), input.clone()))
                }
                _ => None,
            })
            .expect("must assemble a ToolUse");
        assert_eq!(tool.0, "call_x");
        assert_eq!(tool.1, "ls");
        assert_eq!(tool.2, json!({"path":"."}));
        assert_eq!(
            r.content
                .iter()
                .filter(|c| matches!(c, Content::ToolUse { .. }))
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn test_openai_bounds_tool_call_index() {
        // MAGI fix d: a hostile huge index must NOT trigger an unbounded resize; it's ignored.
        let mut server = Server::new_async().await;
        let url = server.url();
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":999999999,\"id\":\"x\",\"function\":{\"name\":\"ls\",\"arguments\":\"{}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        let _m = server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse)
            .create_async()
            .await;
        let p = OpenAiCompatibleProvider::new(OpenAiSettings {
            base_url: url,
            api_key: "k".into(),
            model: "m".into(),
        });
        let r = p
            .send_messages(&[Message::user("x")], &[], None)
            .await
            .unwrap();
        // out-of-bounds index ignored → no ToolUse, no OOM, clean MessageDone.
        assert_eq!(
            r.content
                .iter()
                .filter(|c| matches!(c, Content::ToolUse { .. }))
                .count(),
            0
        );
    }

    #[tokio::test]
    async fn test_openai_finalizes_on_stream_end_without_finish_or_done() {
        // I-1: body has content but NO finish_reason and NO [DONE] sentinel; the
        // stream-end branch must flush the buffer and finalize exactly one
        // MessageDone carrying the assembled text.
        let mut server = Server::new_async().await;
        let url = server.url();
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"only\"}}]}\n\n";
        let _m = server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse)
            .create_async()
            .await;
        let p = OpenAiCompatibleProvider::new(OpenAiSettings {
            base_url: url,
            api_key: "k".into(),
            model: "m".into(),
        });
        let mut stream = p
            .stream_messages(&[Message::user("hi")], &[], None)
            .await
            .unwrap();
        let mut done = Vec::new();
        while let Some(chunk) = stream.next().await {
            if let Ok(ResponseChunk::MessageDone(m)) = chunk {
                done.push(m);
            }
        }
        assert_eq!(done.len(), 1, "exactly one MessageDone on stream end");
        assert_eq!(
            done[0].content,
            vec![Content::Text {
                text: "only".into()
            }]
        );
    }

    // ─── MAGI Loop 2 caveats (C1 + C2) ────────────────────────────────────────

    #[tokio::test]
    async fn test_openai_suppresses_post_stop_text_deltas() {
        // C1: a misbehaving backend sends `delta.content` events AFTER the
        // finalize-causing `finish_reason:"stop"` + `[DONE]` sentinel. The
        // provider must emit exactly ONE MessageDone and NO TextDelta chunks
        // arriving AFTER that MessageDone. Pre-fix the post-stop deltas leak
        // through process_buffer because `done` is not checked before draining
        // further blocks.
        let mut server = Server::new_async().await;
        let url = server.url();
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"ghost1\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"ghost2\"}}]}\n\n",
        );
        let _m = server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse)
            .create_async()
            .await;
        let p = OpenAiCompatibleProvider::new(OpenAiSettings {
            base_url: url,
            api_key: "k".into(),
            model: "m".into(),
        });
        let mut stream = p
            .stream_messages(&[Message::user("hi")], &[], None)
            .await
            .unwrap();

        // Collect chunks in order, classifying each as either a TextDelta or
        // a MessageDone (so we can prove ordering).
        #[derive(PartialEq, Eq)]
        enum Kind {
            Text,
            Done,
        }
        let mut ordered: Vec<Kind> = Vec::new();
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(ResponseChunk::TextDelta(_)) => ordered.push(Kind::Text),
                Ok(ResponseChunk::MessageDone(_)) => ordered.push(Kind::Done),
                _ => {}
            }
        }

        let done_count = ordered.iter().filter(|k| **k == Kind::Done).count();
        assert_eq!(done_count, 1, "exactly one MessageDone");

        let done_pos = ordered
            .iter()
            .position(|k| *k == Kind::Done)
            .expect("MessageDone present");
        // No TextDelta may sit after MessageDone — post-stop ghost deltas must
        // be suppressed.
        for (i, k) in ordered.iter().enumerate() {
            if i > done_pos {
                assert!(
                    *k != Kind::Text,
                    "TextDelta at index {i} arrived after MessageDone at {done_pos}"
                );
            }
        }
    }

    // ─── Task 5 (Task 6 RF-8): build_openai_provider helper ──────────────────

    #[test]
    fn test_build_openai_provider_returns_non_static() {
        let p = build_openai_provider("http://localhost:11434/v1", "ollama", "phi4-mini");
        assert!(!p.is_static());
    }

    #[test]
    fn test_connection_error_hint_interpolates_base_url() {
        // S-8: the actionable hint contains the resolved base_url (DEFAULT in the
        // no-config case) and the Anthropic-opt-in escape hatch.
        let hint = connection_error_hint(crate::defaults::DEFAULT_OPENAI_BASE_URL);
        assert!(hint.contains("http://localhost:11434/v1"));
        assert!(hint.contains(crate::defaults::DEFAULT_OPENAI_BASE_URL));
        assert!(hint.contains("provider=\"anthropic\""));
    }

    // ─── is_retryable_error predicate ────────────────────────────────────────

    /// `is_retryable_error` must return `true` for a rate-limit (429) message
    /// and for a transient connection-failure message (the connection_error_hint
    /// substring), and `false` for unrelated errors (e.g. auth 401).
    ///
    /// This test is fast (no network, no sleeps): it exercises the pure predicate
    /// directly, not the full 3× backoff loop.
    #[test]
    fn test_is_retryable_error_matches_429_and_connection_hint() {
        // 429 rate-limit → retryable
        assert!(
            is_retryable_error("upstream replied 429 Too Many Requests"),
            "429 must be retryable"
        );
        // The connection_error_hint prefix → retryable (transient connection failure)
        let hint = connection_error_hint("http://localhost:11434/v1");
        assert!(
            is_retryable_error(&hint),
            "connection_error_hint must be retryable"
        );
        // Just the stable substring in isolation
        assert!(
            is_retryable_error("Could not reach the OpenAI-compatible backend at http://x/v1."),
            "connection-hint substring must be retryable regardless of URL"
        );
        // Unrelated auth error → NOT retryable
        assert!(
            !is_retryable_error("OpenAI API Error [401]: bad key"),
            "401 auth error must not be retryable"
        );
        // 500 server error → NOT retryable
        assert!(
            !is_retryable_error("OpenAI API Error [500]: internal error"),
            "500 error must not be retryable"
        );
    }

    #[tokio::test]
    async fn test_openai_skips_args_fragment_when_slot_empty() {
        // C2: a misbehaving backend streams a tool_call fragment that carries
        // ONLY `arguments` (no `id`, no `function.name`) before any fragment
        // populates the slot's id/name. The provider must skip the fragment —
        // no `ToolUseInputDelta` chunk and no `Content::ToolUse` are emitted.
        let mut server = Server::new_async().await;
        let url = server.url();
        let sse = concat!(
            // First tool_calls fragment: arguments only, no id, no function.name.
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"orphan\\\":true}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let _m = server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse)
            .create_async()
            .await;
        let p = OpenAiCompatibleProvider::new(OpenAiSettings {
            base_url: url,
            api_key: "k".into(),
            model: "m".into(),
        });
        let mut stream = p
            .stream_messages(&[Message::user("x")], &[], None)
            .await
            .unwrap();

        let mut tool_input_delta_count = 0usize;
        let mut tool_use_count = 0usize;
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(ResponseChunk::ToolUseInputDelta { .. }) => tool_input_delta_count += 1,
                Ok(ResponseChunk::MessageDone(m)) => {
                    tool_use_count += m
                        .content
                        .iter()
                        .filter(|c| matches!(c, Content::ToolUse { .. }))
                        .count();
                }
                _ => {}
            }
        }
        assert_eq!(
            tool_input_delta_count, 0,
            "no ToolUseInputDelta emitted for orphan args fragment"
        );
        assert_eq!(
            tool_use_count, 0,
            "no Content::ToolUse emitted for orphan args fragment"
        );
    }

    // ─── Feature E: system-prompt injection (REQ-H12b) ────────────────────────

    /// Captures the raw UTF-8 request body mockito received, via
    /// `with_body_from_request`, while still returning `sse_body` as the mocked
    /// response — letting a test assert exactly what the provider serialized
    /// (including a field's *absence*, which a `Matcher` alone cannot express).
    async fn capture_body_mock(
        server: &mut Server,
        path: &str,
        sse_body: &'static str,
    ) -> (mockito::Mock, Arc<Mutex<Option<String>>>) {
        let captured = Arc::new(Mutex::new(None));
        let captured_clone = captured.clone();
        let mock = server
            .mock("POST", path)
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body_from_request(move |req| {
                *captured_clone.lock().unwrap() =
                    Some(req.utf8_lossy_body().unwrap_or_default().into_owned());
                sse_body.as_bytes().to_vec()
            })
            .create_async()
            .await;
        (mock, captured)
    }

    const MESSAGE_STOP_SSE: &str = "event: message_stop\ndata: {\"type\": \"message_stop\"}\n\n";

    #[tokio::test]
    async fn test_anthropic_provider_sends_top_level_system_field_when_present() {
        let mut server = Server::new_async().await;
        let url = server.url();
        let (_m, captured) = capture_body_mock(&mut server, "/messages", MESSAGE_STOP_SSE).await;
        let provider = AnthropicProvider::with_base_url("k".into(), "m".into(), url);
        let mut stream = provider
            .stream_messages(
                &[Message::user("hi")],
                &[],
                Some("You are a test assistant."),
            )
            .await
            .unwrap();
        while stream.next().await.is_some() {}
        let body = captured.lock().unwrap().clone().expect("request captured");
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["system"], "You are a test assistant.");
    }

    #[tokio::test]
    async fn test_anthropic_provider_omits_system_field_when_none() {
        // Interactive path: AgentRunConfig::default().system == None must reach
        // the wire as NO `system` field at all (not an empty string).
        let mut server = Server::new_async().await;
        let url = server.url();
        let (_m, captured) = capture_body_mock(&mut server, "/messages", MESSAGE_STOP_SSE).await;
        let provider = AnthropicProvider::with_base_url("k".into(), "m".into(), url);
        let mut stream = provider
            .stream_messages(&[Message::user("hi")], &[], None)
            .await
            .unwrap();
        while stream.next().await.is_some() {}
        let body = captured.lock().unwrap().clone().expect("request captured");
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(
            parsed.get("system").is_none(),
            "no system field must be serialized when system=None, got: {body}"
        );
    }

    #[tokio::test]
    async fn test_anthropic_provider_omits_system_field_when_empty_string() {
        let mut server = Server::new_async().await;
        let url = server.url();
        let (_m, captured) = capture_body_mock(&mut server, "/messages", MESSAGE_STOP_SSE).await;
        let provider = AnthropicProvider::with_base_url("k".into(), "m".into(), url);
        let mut stream = provider
            .stream_messages(&[Message::user("hi")], &[], Some(""))
            .await
            .unwrap();
        while stream.next().await.is_some() {}
        let body = captured.lock().unwrap().clone().expect("request captured");
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(
            parsed.get("system").is_none(),
            "empty system string must not be serialized, got: {body}"
        );
    }

    #[tokio::test]
    async fn test_openai_provider_prepends_system_message_when_present() {
        let mut server = Server::new_async().await;
        let url = server.url();
        let (_m, captured) =
            capture_body_mock(&mut server, "/chat/completions", "data: [DONE]\n\n").await;
        let p = OpenAiCompatibleProvider::new(OpenAiSettings {
            base_url: url,
            api_key: "k".into(),
            model: "m".into(),
        });
        let mut stream = p
            .stream_messages(
                &[Message::user("hi")],
                &[],
                Some("You are a test assistant."),
            )
            .await
            .unwrap();
        while stream.next().await.is_some() {}
        let body = captured.lock().unwrap().clone().expect("request captured");
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        let messages = parsed["messages"].as_array().unwrap();
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "You are a test assistant.");
        assert_eq!(messages[1]["role"], "user");
    }

    #[tokio::test]
    async fn test_openai_provider_omits_system_message_when_none() {
        let mut server = Server::new_async().await;
        let url = server.url();
        let (_m, captured) =
            capture_body_mock(&mut server, "/chat/completions", "data: [DONE]\n\n").await;
        let p = OpenAiCompatibleProvider::new(OpenAiSettings {
            base_url: url,
            api_key: "k".into(),
            model: "m".into(),
        });
        let mut stream = p
            .stream_messages(&[Message::user("hi")], &[], None)
            .await
            .unwrap();
        while stream.next().await.is_some() {}
        let body = captured.lock().unwrap().clone().expect("request captured");
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        let messages = parsed["messages"].as_array().unwrap();
        assert_eq!(
            messages.len(),
            1,
            "no system message must be prepended when system=None, got: {body}"
        );
        assert_eq!(messages[0]["role"], "user");
    }

    // ─── Feature C: token usage (ResponseChunk::Usage) ────────────────────────

    #[tokio::test]
    async fn test_anthropic_provider_emits_usage_chunk_with_input_and_output_tokens() {
        let mut server = Server::new_async().await;
        let url = server.url();
        let sse_body = concat!(
            "event: message_start\n",
            "data: {\"type\": \"message_start\", \"message\": {\"id\": \"m\", \"role\": \"assistant\", \"model\": \"x\", \"usage\": {\"input_tokens\": 42, \"output_tokens\": 0}}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\": \"content_block_delta\", \"index\":0, \"delta\": {\"type\": \"text_delta\", \"text\": \"hi\"}}\n\n",
            "event: message_delta\n",
            "data: {\"type\": \"message_delta\", \"delta\": {}, \"usage\": {\"output_tokens\": 7}}\n\n",
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
        let provider = AnthropicProvider::with_base_url("k".into(), "m".into(), url);
        let mut stream = provider
            .stream_messages(&[Message::user("hi")], &[], None)
            .await
            .unwrap();
        let mut usage = None;
        while let Some(chunk) = stream.next().await {
            if let Ok(ResponseChunk::Usage {
                input_tokens,
                output_tokens,
            }) = chunk
            {
                usage = Some((input_tokens, output_tokens));
            }
        }
        assert_eq!(usage, Some((42, 7)));
    }

    #[tokio::test]
    async fn test_anthropic_provider_emits_no_usage_when_absent_from_wire() {
        // MAGI re-gate WARNING 2: a well-formed `message_stop` with no prior
        // `usage` anywhere on the wire (no `message_start.message.usage`, no
        // `message_delta.usage`) must NOT fabricate a `(0, 0)` Usage chunk — that
        // contradicts REQ-H14 and diverges from `OaiState`/`finalize_truncated`,
        // both of which gate emission on usage actually being observed.
        let mut server = Server::new_async().await;
        let url = server.url();
        let _m = server
            .mock("POST", "/messages")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(MESSAGE_STOP_SSE)
            .create_async()
            .await;
        let provider = AnthropicProvider::with_base_url("k".into(), "m".into(), url);
        let mut stream = provider
            .stream_messages(&[Message::user("hi")], &[], None)
            .await
            .unwrap();
        let mut saw_usage = false;
        while let Some(chunk) = stream.next().await {
            if let Ok(ResponseChunk::Usage { .. }) = chunk {
                saw_usage = true;
            }
        }
        assert!(
            !saw_usage,
            "no Usage chunk must be fabricated when the wire never reported usage"
        );
    }

    // ─── Feature D: mid-stream `error` events + truncated-stream finalization ──

    #[tokio::test]
    async fn test_anthropic_provider_surfaces_mid_stream_error_event() {
        // Gap 1 (MAGI re-gate): a mid-stream Anthropic `error` event (e.g. the
        // backend going overloaded) must surface as an Err on the stream — before
        // the fix it either failed to deserialize or fell through the `_ => {}`
        // catch-all and was silently swallowed.
        let mut server = Server::new_async().await;
        let url = server.url();
        let sse_body = concat!(
            "event: message_start\n",
            "data: {\"type\": \"message_start\", \"message\": {\"id\": \"m\", \"role\": \"assistant\", \"model\": \"x\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\": \"content_block_delta\", \"index\": 0, \"delta\": {\"type\": \"text_delta\", \"text\": \"partial\"}}\n\n",
            "event: error\n",
            "data: {\"type\": \"error\", \"error\": {\"type\": \"overloaded_error\", \"message\": \"Overloaded\"}}\n\n",
        );
        let _m = server
            .mock("POST", "/messages")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse_body)
            .create_async()
            .await;
        let provider = AnthropicProvider::with_base_url("k".into(), "m".into(), url);
        let mut stream = provider
            .stream_messages(&[Message::user("hi")], &[], None)
            .await
            .unwrap();

        let mut saw_error = false;
        while let Some(chunk) = stream.next().await {
            if let Err(e) = chunk {
                let msg = e.to_string();
                assert!(
                    msg.contains("overloaded_error") && msg.contains("Overloaded"),
                    "error message must surface the Anthropic error type and message, got: {}",
                    msg
                );
                saw_error = true;
            }
        }
        assert!(
            saw_error,
            "a mid-stream Anthropic `error` event must surface as Err, not be swallowed"
        );
    }

    #[tokio::test]
    async fn test_anthropic_provider_truncated_stream_flushes_partial_content_as_message_done() {
        // Gap 2 (MAGI re-gate): the byte source can close mid-turn (dropped
        // connection, proxy timeout) BEFORE a `message_stop` event ever arrives.
        // Before the fix the Anthropic stream had no stream-end hook (unlike the
        // OpenAI provider's `stream::unfold` `None` branch), so every
        // already-streamed TextDelta's accumulated content silently vanished — no
        // MessageDone was ever emitted, and the caller (`run_tool_loop`) only saw
        // "Stream ended without MessageDone", discarding the partial turn.
        let mut server = Server::new_async().await;
        let url = server.url();
        // No message_stop: the mock body ends right after the delta.
        let sse_body = concat!(
            "event: message_start\n",
            "data: {\"type\": \"message_start\", \"message\": {\"id\": \"m\", \"role\": \"assistant\", \"model\": \"x\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\": \"content_block_delta\", \"index\": 0, \"delta\": {\"type\": \"text_delta\", \"text\": \"Partial answer\"}}\n\n",
        );
        let _m = server
            .mock("POST", "/messages")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse_body)
            .create_async()
            .await;
        let provider = AnthropicProvider::with_base_url("k".into(), "m".into(), url);
        let mut stream = provider
            .stream_messages(&[Message::user("hi")], &[], None)
            .await
            .unwrap();

        let mut final_msg = None;
        let mut saw_usage = false;
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(ResponseChunk::MessageDone(msg)) => final_msg = Some(msg),
                Ok(ResponseChunk::Usage { .. }) => saw_usage = true,
                _ => {}
            }
        }
        let msg = final_msg.expect(
            "a truncated stream (no message_stop) must still flush a MessageDone with the \
             partial content instead of silently dropping it",
        );
        assert_eq!(
            msg.content,
            vec![Content::Text {
                text: "Partial answer".to_string()
            }]
        );
        assert!(
            !saw_usage,
            "no Usage chunk must be fabricated on a truncated stream that never reported usage"
        );
    }

    #[tokio::test]
    async fn test_anthropic_provider_drops_content_events_after_message_stop() {
        // MAGI re-gate WARNING 1: `AnthropicState` did not mirror `OaiState`'s
        // post-finalize ghost-content guard (`if self.done { continue }`) — a
        // misbehaving backend that sends more `content_block_delta` events AFTER
        // `message_stop` must not leak that text into the assembled message, and
        // must never trigger a second `MessageDone`.
        let mut server = Server::new_async().await;
        let url = server.url();
        let sse_body = concat!(
            "event: content_block_delta\n",
            "data: {\"type\": \"content_block_delta\", \"index\":0, \"delta\": {\"type\": \"text_delta\", \"text\": \"Hello\"}}\n\n",
            "event: message_stop\n",
            "data: {\"type\": \"message_stop\"}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\": \"content_block_delta\", \"index\":0, \"delta\": {\"type\": \"text_delta\", \"text\": \" ghost\"}}\n\n",
        );
        let _m = server
            .mock("POST", "/messages")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse_body)
            .create_async()
            .await;
        let provider = AnthropicProvider::with_base_url("k".into(), "m".into(), url);
        let mut stream = provider
            .stream_messages(&[Message::user("hi")], &[], None)
            .await
            .unwrap();

        let mut done_count = 0;
        let mut final_msg = None;
        let mut saw_ghost_text = false;
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(ResponseChunk::MessageDone(msg)) => {
                    done_count += 1;
                    final_msg = Some(msg);
                }
                Ok(ResponseChunk::TextDelta(t)) if t.contains("ghost") => {
                    saw_ghost_text = true;
                }
                _ => {}
            }
        }
        assert_eq!(done_count, 1, "MessageDone must be emitted exactly once");
        assert!(
            !saw_ghost_text,
            "a content_block_delta arriving after message_stop must not leak a TextDelta"
        );
        let msg = final_msg.expect("a MessageDone must still be emitted for the valid turn");
        assert_eq!(
            msg.content,
            vec![Content::Text {
                text: "Hello".to_string()
            }],
            "the assembled message must not include post-message_stop ghost content"
        );
    }

    #[tokio::test]
    async fn test_openai_provider_emits_usage_chunk_from_stream_options() {
        let mut server = Server::new_async().await;
        let url = server.url();
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":3}}\n\n",
            "data: [DONE]\n\n",
        );
        let _m = server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse)
            .create_async()
            .await;
        let p = OpenAiCompatibleProvider::new(OpenAiSettings {
            base_url: url,
            api_key: "k".into(),
            model: "m".into(),
        });
        let mut stream = p
            .stream_messages(&[Message::user("hi")], &[], None)
            .await
            .unwrap();
        let mut usage = None;
        while let Some(chunk) = stream.next().await {
            if let Ok(ResponseChunk::Usage {
                input_tokens,
                output_tokens,
            }) = chunk
            {
                usage = Some((input_tokens, output_tokens));
            }
        }
        assert_eq!(usage, Some((11, 3)));
    }

    #[tokio::test]
    async fn test_openai_provider_emits_no_usage_when_backend_omits_it() {
        // A backend that ignores stream_options.include_usage (or doesn't support
        // it) must not cause a fabricated Usage chunk — none is emitted at all.
        let mut server = Server::new_async().await;
        let url = server.url();
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let _m = server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse)
            .create_async()
            .await;
        let p = OpenAiCompatibleProvider::new(OpenAiSettings {
            base_url: url,
            api_key: "k".into(),
            model: "m".into(),
        });
        let mut stream = p
            .stream_messages(&[Message::user("hi")], &[], None)
            .await
            .unwrap();
        let mut saw_usage = false;
        while let Some(chunk) = stream.next().await {
            if let Ok(ResponseChunk::Usage { .. }) = chunk {
                saw_usage = true;
            }
        }
        assert!(!saw_usage, "no Usage chunk must be fabricated when absent");
    }

    #[tokio::test]
    async fn test_openai_provider_requests_stream_options_include_usage() {
        let mut server = Server::new_async().await;
        let url = server.url();
        let (_m, captured) =
            capture_body_mock(&mut server, "/chat/completions", "data: [DONE]\n\n").await;
        let p = OpenAiCompatibleProvider::new(OpenAiSettings {
            base_url: url,
            api_key: "k".into(),
            model: "m".into(),
        });
        let mut stream = p
            .stream_messages(&[Message::user("hi")], &[], None)
            .await
            .unwrap();
        while stream.next().await.is_some() {}
        let body = captured.lock().unwrap().clone().expect("request captured");
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["stream_options"]["include_usage"], true);
    }

    // ─── SC-A19: the principal provider survived the adapter removal ─────────
    //
    // Task 4.1 removed `src/agent/magi_adapter.rs` and rebuilt the MAGI trio on
    // magi-core's native providers. These tests pin that the retirement did NOT
    // drag the *principal* `OpenAiCompatibleProvider` along with it — exactly
    // what the migration discarded in spec §3 would have broken (`LlmProvider`
    // is request/response, has no streaming, and no tool calling).
    //
    // The "no total-request timeout" property is pinned TWICE, deliberately,
    // by two tests that prove different things (fix round 1 — the original
    // single stalling-socket test only caught a regression SHORTER than its
    // 2s stall, missing exactly the realistic shape: someone re-adding a 30s
    // or 300s total timeout, the latter being magi-core's `OllamaProvider`
    // default — which REQ-R30 still forbids from reaching a MAGI seat, and
    // which has no business on this path either):
    //   - `the_principal_providers_client_carries_no_total_timeout_marker`
    //     pins the CLIENT'S CONFIGURATION, instantly and for a timeout of any
    //     duration, via `reqwest::Client`'s `Debug` output.
    //   - `the_principal_provider_has_no_total_request_timeout` (below) pins
    //     that nothing else in the CALL PATH imposes a deadline either — a
    //     real stalling socket, not just client config.
    //
    // The third SC-A19 property — a malformed `data:` line must not abort the
    // stream — is already pinned above by `test_openai_swallows_malformed_line`
    // (predates this task). It is intentionally not duplicated here; see
    // task-4.2-report.md for the revert-and-retest evidence that it is not
    // vacuous.

    /// The literal marker `reqwest::Client`'s `Debug` impl emits when (and
    /// only when) a total-request timeout is set via
    /// `ClientBuilder::timeout(...)`.
    ///
    /// This is NOT the string `"timeout"`. `Client`'s `Debug` impl
    /// (`async_impl/client.rs:2727`) delegates to `ClientRef::fmt_fields`
    /// (`:2957`), which prints the total timeout via
    /// `RequestConfig<TotalTimeout>::fmt_as_field` (`config.rs:60`) — and that
    /// helper uses `std::any::type_name::<TotalTimeout>()` as the field NAME,
    /// not a hardcoded string. `ClientBuilder`'s own `Debug` impl is a
    /// DIFFERENT code path (`Config::fmt_fields`, `client.rs:2772`) that does
    /// print the literal field `"timeout"` — but `OpenAiCompatibleProvider`
    /// holds a built `Client`, never a `ClientBuilder`, so that path is not
    /// what these tests exercise.
    ///
    /// Verified empirically (not just read) against the pinned
    /// `reqwest = 0.13.4` (see `Cargo.lock`) by probing
    /// `format!("{:?}", client)` on three clients:
    /// - plain `Client::new()`: no timeout-shaped field at all.
    /// - `.timeout(Duration::from_secs(30))`:
    ///   `reqwest::config::TotalTimeout: 30s`.
    /// - `.connect_timeout(Duration::from_secs(7))`: no field at all —
    ///   `ClientRef` (what `Client`'s `Debug` reads) has no
    ///   `connect_timeout` field; only `ClientBuilder`'s `Debug` shows it.
    ///
    /// That last point is why this needle cannot collide: `connect_timeout`
    /// never appears in a `Client`'s `Debug` output at all, and the only
    /// other timeout-shaped field `ClientRef` can print — `read_timeout`
    /// (`client.rs:2988`, unused by this provider) — is a different token
    /// than `TotalTimeout` and cannot match it as a substring.
    const TOTAL_TIMEOUT_DEBUG_MARKER: &str = "reqwest::config::TotalTimeout";

    /// Positive-control timeout for
    /// [`the_principal_providers_client_carries_no_total_timeout_marker`].
    /// The exact duration is immaterial to what the control proves (any
    /// `Some(_)` total timeout must surface the marker) — `30` is chosen to
    /// echo the realistic regression shape named in the fix-round-1 finding
    /// (a re-added `.timeout(Duration::from_secs(30))`).
    const POSITIVE_CONTROL_TIMEOUT: Duration = Duration::from_secs(30);

    #[test]
    fn the_principal_providers_client_carries_no_total_timeout_marker() {
        // SC-A19 fix round 1: the 2s stalling-socket test below only proves
        // the absence of a timeout SHORTER than its stall. A regression that
        // reintroduces a 30s or 300s total timeout — the latter being
        // magi-core's `OllamaProvider` default, which REQ-R30 keeps off a
        // MAGI seat and which has no business here either —
        // sails straight through it. `reqwest::Client` exposes no public
        // timeout getter, but its `Debug` impl only ever prints
        // `TOTAL_TIMEOUT_DEBUG_MARKER` when a total timeout is actually set
        // (see that constant's rustdoc for the verified mechanism), so the
        // property is observable instantly and deterministically, for a
        // timeout of ANY duration, with no wall-clock at all.
        let client = OpenAiCompatibleProvider::new(OpenAiSettings {
            base_url: "http://127.0.0.1:1".into(),
            api_key: "k".into(),
            model: "m".into(),
        });
        let debug = format!("{:?}", client.client_for_test());
        assert!(
            !debug.contains(TOTAL_TIMEOUT_DEBUG_MARKER),
            "the principal provider's client must carry no total-request \
             timeout marker, got: {debug}"
        );

        // Positive control (mandatory, same test): prove the marker DOES
        // surface for a client that legitimately carries a total timeout.
        // Without this, a future reqwest release that stops emitting the
        // field would make the assertion above pass for the wrong reason —
        // silently switching the guardrail off, exactly the failure mode
        // this project's spec condemns repeatedly. With the control, that
        // release breaks this test instead of the guardrail going dark.
        let with_timeout = reqwest::Client::builder()
            .timeout(POSITIVE_CONTROL_TIMEOUT)
            .build()
            .expect("build a client with a total timeout for the positive control");
        let control_debug = format!("{with_timeout:?}");
        assert!(
            control_debug.contains(TOTAL_TIMEOUT_DEBUG_MARKER),
            "positive control: a client built WITH .timeout(...) must show \
             the marker in its Debug output, got: {control_debug}"
        );
    }

    /// Mid-response stall used to pin that nothing in the call path — beyond
    /// the client configuration already pinned by
    /// [`the_principal_providers_client_carries_no_total_timeout_marker`] —
    /// imposes a deadline on a slow-arriving SSE stream. Long enough that a
    /// deadline shorter than this turns the stall into a hard `Err`; short
    /// enough not to bloat `cargo nextest`'s wall-clock (this box already
    /// starves under load, see `CLAUDE.local.md`). This constant deliberately
    /// stays short — widening it to also catch a 30s/300s-shaped regression
    /// would pay real wall-clock on every suite run forever; that shape is
    /// instead covered, for free and deterministically, by the sibling
    /// `Debug`-marker test above.
    const NO_TIMEOUT_STALL: Duration = Duration::from_secs(2);

    /// Spawns a one-shot raw TCP server on an ephemeral loopback port that
    /// speaks just enough HTTP/1.1 to satisfy `reqwest`: it writes response
    /// headers immediately, stalls for [`NO_TIMEOUT_STALL`] BEFORE writing any
    /// SSE body byte — mirroring the cold-load scenario documented on
    /// `OpenAiCompatibleProvider::new` ("Ollama can spend tens of seconds on
    /// cold-load before the first SSE event arrives") — then finishes the
    /// stream and closes. `mockito::Server` (used by every other test in this
    /// module) serves its mock body immediately and cannot inject a mid-response
    /// stall, so this fixture needs a real socket.
    ///
    /// Returns the `http://127.0.0.1:<port>` base URL once the listener is
    /// bound and ready to accept a connection.
    async fn spawn_stalling_sse_server() -> String {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind an ephemeral loopback port for the stall fixture");
        let addr = listener
            .local_addr()
            .expect("a bound listener exposes its local address");
        tokio::spawn(async move {
            let Ok((socket, _)) = listener.accept().await else {
                return;
            };
            let (mut reader, mut writer) = socket.into_split();
            // Drain and discard the request concurrently with writing the
            // response: the client's request write must never block on this
            // fixture reading it. O(request size) — bounded by one small JSON
            // test message, never more than a few hundred bytes.
            tokio::spawn(async move {
                let mut sink = [0u8; 4096];
                while matches!(reader.read(&mut sink).await, Ok(n) if n > 0) {}
            });
            let head =
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n";
            if writer.write_all(head.as_bytes()).await.is_err() || writer.flush().await.is_err() {
                return;
            }
            sleep(NO_TIMEOUT_STALL).await;
            let body = concat!(
                "data: {\"choices\":[{\"delta\":{\"content\":\"post-stall\"},",
                "\"finish_reason\":\"stop\"}]}\n\n",
                "data: [DONE]\n\n",
            );
            let _ = writer.write_all(body.as_bytes()).await;
            let _ = writer.flush().await;
            let _ = writer.shutdown().await;
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn the_principal_provider_has_no_total_request_timeout() {
        // SC-A19 / REQ-A19b: the client must still tolerate a stall before the
        // first SSE byte — exactly what local Ollama's cold-load looks like
        // (`OpenAiCompatibleProvider::new` rustdoc). This proves the CALL PATH
        // imposes no deadline shorter than NO_TIMEOUT_STALL end-to-end (a real
        // socket, real send_messages() drive) — it is NOT the test that pins
        // "no total timeout of any duration"; that is
        // `the_principal_providers_client_carries_no_total_timeout_marker`
        // above, which checks the client's actual configuration and is not
        // bounded by wall-clock. A total-request timeout shorter than the
        // stall would surface as an `Err` here (verified by temporarily
        // reintroducing one — see task-4.2-report.md for the red output).
        let base_url = spawn_stalling_sse_server().await;
        let p = OpenAiCompatibleProvider::new(OpenAiSettings {
            base_url,
            api_key: "k".into(),
            model: "m".into(),
        });
        let result = p.send_messages(&[Message::user("hi")], &[], None).await;
        assert!(
            result.is_ok(),
            "a {:?} stall before the first SSE byte must not abort the stream: {:?}",
            NO_TIMEOUT_STALL,
            result.err()
        );
        assert_eq!(
            result.expect("checked is_ok above").content,
            vec![Content::Text {
                text: "post-stall".into()
            }]
        );
    }

    #[test]
    fn the_principal_provider_kept_its_tool_call_slot_cap() {
        // SC-A19: retiring the adapter must not have changed the anti-OOM
        // ceiling on streamed tool-call indices (Task 5 / RF-8).
        assert_eq!(MAX_TOOL_CALL_SLOTS, 64);
    }
}
