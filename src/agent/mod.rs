//! The Agent orchestrator coordinates between the Provider and the Tools.

pub mod magi_adapter;
pub mod messages;
pub mod provider;

use crate::agent::messages::{Content, Message, Role};
use crate::agent::provider::{Provider, ResponseChunk};
use crate::system::database::MemoryStore;
use crate::tools::Tool;
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::oneshot;
use tokio::time::{timeout, Duration};

const APPROVAL_TIMEOUT_SECS: u64 = 300; // 5 minutes

/// Approval request sent to the UI.
pub struct ApprovalRequest {
    pub tool_name: String,
    /// Carries tool input for the approval prompt; reserved.
    #[allow(dead_code)]
    pub input: serde_json::Value,
    pub tx: oneshot::Sender<bool>,
}

/// The Agent orchestrator.
pub struct Agent {
    provider: Arc<dyn Provider>,
    tools: Vec<Box<dyn Tool>>,
    history: Vec<Message>,
    /// Optional persistent memory store.
    memory: Option<Arc<dyn MemoryStore>>,
    /// Current session ID in the memory store.
    session_id: Option<String>,
    /// Optional channel to request approval for tools.
    approval_tx: Option<tokio::sync::mpsc::Sender<ApprovalRequest>>,
    /// Safeguard for infinite loops (max tool calls per query)
    pub max_tool_calls: usize,
}

impl Agent {
    /// Creates a new Agent.
    pub fn new(provider: Arc<dyn Provider>) -> Self {
        Self {
            provider,
            tools: Vec::new(),
            history: Vec::new(),
            memory: None,
            session_id: None,
            approval_tx: None,
            max_tool_calls: 15,
        }
    }

    /// Sets the persistent memory store and session for the agent.
    pub fn set_memory(&mut self, memory: Arc<dyn MemoryStore>, session_id: String) {
        self.memory = Some(memory);
        self.session_id = Some(session_id);
    }

    /// Replaces the active LLM provider (e.g. after a mid-session `/login` swaps
    /// the startup `StaticProvider` for a live `AnthropicProvider`).
    pub fn set_provider(&mut self, provider: Arc<dyn Provider>) {
        self.provider = provider;
    }

    /// Whether the active provider is the canned [`StaticProvider`] (no API key).
    /// Used by `/login` to decide whether the prior history is canned noise (safe
    /// to clear) vs a real conversation (must be kept).
    pub fn provider_is_static(&self) -> bool {
        self.provider.is_static()
    }

    /// Loads history from the persistent memory store.
    pub async fn load_history(&mut self) -> Result<()> {
        if let (Some(memory), Some(sid)) = (&self.memory, &self.session_id) {
            let messages = memory.get_messages(sid).await?;
            self.history = messages;
        }
        Ok(())
    }

    /// Set approval channel.
    pub fn set_approval_channel(&mut self, tx: tokio::sync::mpsc::Sender<ApprovalRequest>) {
        self.approval_tx = Some(tx);
    }

    /// Registers a tool with the agent.
    pub fn register_tool(&mut self, tool: Box<dyn Tool>) {
        self.tools.push(tool);
    }

    /// Normalizes tool input recursively to detect semantically identical calls.
    pub fn normalize_input(val: &serde_json::Value, depth: usize) -> Result<String> {
        const MAX_DEPTH: usize = 10;
        if depth > MAX_DEPTH {
            return Err(anyhow::anyhow!("JSON nesting limit exceeded"));
        }

        match val {
            serde_json::Value::Object(map) => {
                let mut sorted_keys: Vec<_> = map.keys().collect();
                sorted_keys.sort();
                let mut parts = Vec::new();
                for k in sorted_keys {
                    let v = map.get(k).unwrap();
                    parts.push(format!("{}:{}", k, Self::normalize_input(v, depth + 1)?));
                }
                Ok(format!("{{{}}}", parts.join(",")))
            }
            serde_json::Value::Array(arr) => {
                let mut normalized_elements = Vec::new();
                for v in arr {
                    normalized_elements.push(Self::normalize_input(v, depth + 1)?);
                }
                Ok(format!("[{}]", normalized_elements.join(",")))
            }
            serde_json::Value::String(s) => Ok(s.trim().to_string()),
            _ => Ok(val.to_string()),
        }
    }

    /// Strips terminal escape sequences and control characters for security.
    pub fn sanitize_text(text: &str) -> String {
        let mut result = String::with_capacity(text.len());
        let mut chars = text.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '\x1B' {
                if let Some('[') = chars.peek() {
                    chars.next();
                    for next in chars.by_ref() {
                        if next.is_ascii_alphabetic() {
                            break;
                        }
                    }
                    continue;
                }
            }
            if c.is_control() && c != '\n' && c != '\r' && c != '\t' {
                continue;
            }
            result.push(c);
        }
        result
    }

    /// Clears the conversation history.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Compacts history by summarization; reserved for future use.
    #[allow(dead_code)]
    pub async fn compact_history(&mut self) -> Result<()> {
        if self.history.is_empty() {
            return Ok(());
        }

        let summary_prompt = Message::user("Please provide a concise technical summary of our conversation so far, capturing all key context, decisions, and outcomes. This summary will be used to continue our session efficiently.");

        let mut temp_history = self.history.clone();
        temp_history.push(summary_prompt);

        let response = self.provider.send_messages(&temp_history, &[]).await?;

        self.history.clear();
        let summary_msg = Message {
            role: Role::Assistant,
            content: vec![Content::Text {
                text: format!(
                    "CONVERSATION SUMMARY: {}",
                    match &response.content[0] {
                        Content::Text { text } => text.clone(),
                        _ => "Summary extraction failed.".to_string(),
                    }
                ),
            }],
        };
        self.history.push(summary_msg.clone());

        // Persist summary if memory is available
        if let (Some(memory), Some(sid)) = (&self.memory, &self.session_id) {
            memory.add_message(sid, &summary_msg).await?;
        }

        Ok(())
    }

    /// Process a user message and returns the final assistant response.
    /// This method supports streaming via a sender channel.
    pub async fn query_streaming(
        &mut self,
        text: &str,
        chunk_tx: tokio::sync::mpsc::Sender<String>,
    ) -> Result<String> {
        let user_msg = Message::user(text);
        self.history.push(user_msg.clone());

        // Persist user message
        if let (Some(memory), Some(sid)) = (&self.memory, &self.session_id) {
            memory.add_message(sid, &user_msg).await?;
        }

        let mut tool_call_count = 0;
        let mut last_normalized_tool: Option<(String, String)> = None;
        let mut repeat_count = 0;

        loop {
            let mut stream = self
                .provider
                .stream_messages(&self.history, &self.tools)
                .await?;
            let mut full_text = String::new();
            let mut last_message: Option<Message> = None;

            use futures::StreamExt;
            while let Some(chunk_result) = stream.next().await {
                match chunk_result? {
                    ResponseChunk::TextDelta(delta) => {
                        let sanitized = Self::sanitize_text(&delta);
                        full_text.push_str(&sanitized);
                        if chunk_tx.send(sanitized).await.is_err() {
                            return Err(anyhow::anyhow!("TUI connection closed during streaming"));
                        }
                    }
                    ResponseChunk::MessageDone(msg) => {
                        last_message = Some(msg);
                    }
                    _ => {}
                }
            }

            let response =
                last_message.ok_or_else(|| anyhow::anyhow!("Stream ended without MessageDone"))?;
            self.history.push(response.clone());

            // Persist assistant response
            if let (Some(memory), Some(sid)) = (&self.memory, &self.session_id) {
                memory.add_message(sid, &response).await?;
            }

            let mut tool_results = Vec::new();
            let mut requested_tool = false;

            for content in &response.content {
                if let Content::ToolUse { id, name, input } = content {
                    requested_tool = true;
                    tool_call_count += 1;
                    if tool_call_count > self.max_tool_calls {
                        return Err(anyhow::anyhow!("Maximum tool call limit reached"));
                    }

                    let normalized_input = Self::normalize_input(input, 0)?;
                    if let Some((ref last_name, ref last_norm_input)) = last_normalized_tool {
                        if last_name == name && last_norm_input == &normalized_input {
                            repeat_count += 1;
                            if repeat_count >= 3 {
                                return Err(anyhow::anyhow!("Repetitive tool call detected"));
                            }
                        } else {
                            repeat_count = 0;
                        }
                    }
                    last_normalized_tool = Some((name.clone(), normalized_input));

                    let approved = if let Some(ref tx) = self.approval_tx {
                        let (oneshot_tx, oneshot_rx) = oneshot::channel();
                        let _ = tx
                            .send(ApprovalRequest {
                                tool_name: name.clone(),
                                input: input.clone(),
                                tx: oneshot_tx,
                            })
                            .await;
                        match timeout(Duration::from_secs(APPROVAL_TIMEOUT_SECS), oneshot_rx).await
                        {
                            Ok(Ok(res)) => res,
                            _ => false,
                        }
                    } else {
                        true
                    };

                    if !approved {
                        tool_results.push(Content::ToolResult {
                            tool_use_id: id.clone(),
                            content: "Execution denied or timed out.".to_string(),
                            is_error: true,
                        });
                        continue;
                    }

                    let tool_result =
                        if let Some(tool) = self.tools.iter().find(|t| t.name() == name) {
                            match tool.execute(input.clone()).await {
                                Ok(val) => Content::ToolResult {
                                    tool_use_id: id.clone(),
                                    content: val.to_string(),
                                    is_error: false,
                                },
                                Err(e) => Content::ToolResult {
                                    tool_use_id: id.clone(),
                                    content: e.to_string(),
                                    is_error: true,
                                },
                            }
                        } else {
                            Content::ToolResult {
                                tool_use_id: id.clone(),
                                content: format!("Tool '{}' not found", name),
                                is_error: true,
                            }
                        };
                    tool_results.push(tool_result);
                }
            }

            if requested_tool {
                let tool_res_msg = Message {
                    role: Role::User,
                    content: tool_results,
                };
                self.history.push(tool_res_msg.clone());
                // Persist tool results
                if let (Some(memory), Some(sid)) = (&self.memory, &self.session_id) {
                    memory.add_message(sid, &tool_res_msg).await?;
                }
            } else {
                for content in response.content.iter().rev() {
                    if let Content::Text { text } = content {
                        return Ok(text.clone());
                    }
                }
                return Ok(String::new());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use async_trait::async_trait;
    use futures::stream::{self, BoxStream};
    use serde_json::json;

    pub struct MockProvider;

    #[async_trait]
    impl Provider for MockProvider {
        async fn stream_messages(
            &self,
            _messages: &[Message],
            _tools: &[Box<dyn Tool>],
        ) -> Result<BoxStream<'static, Result<ResponseChunk>>> {
            let chunks = vec![
                Ok(ResponseChunk::TextDelta("Summary content.".to_string())),
                Ok(ResponseChunk::MessageDone(Message::assistant(
                    "Summary content.",
                ))),
            ];
            Ok(Box::pin(stream::iter(chunks)))
        }
        async fn send_messages(
            &self,
            _messages: &[Message],
            _tools: &[Box<dyn Tool>],
        ) -> Result<Message> {
            Ok(Message::assistant("Summary content."))
        }
    }

    #[tokio::test]
    async fn test_agent_history_compaction() {
        let mut agent = Agent::new(Arc::new(MockProvider));
        agent.history.push(Message::user("Msg 1"));
        agent.history.push(Message::assistant("Resp 1"));
        assert_eq!(agent.history.len(), 2);

        agent.compact_history().await.unwrap();

        assert_eq!(agent.history.len(), 1);
        if let Content::Text { text } = &agent.history[0].content[0] {
            assert!(text.contains("Summary"));
        } else {
            panic!("Expected summary");
        }
    }

    #[tokio::test]
    async fn test_agent_normalization_depth_limit() {
        let mut deep_json = json!({"path": "."});
        for i in 0..20 {
            deep_json = json!({format!("level_{}", i): deep_json});
        }
        let result = Agent::normalize_input(&deep_json, 0);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_agent_text_sanitization() {
        let input = "\x1B[31mDangerous\x1B[0m Text\x07";
        assert_eq!(Agent::sanitize_text(input), "Dangerous Text");
    }

    #[tokio::test]
    async fn test_agent_encrypted_persistence_integration() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let db_path = tmp_dir.path().join("test_persist.db");
        let password = "master_password_123".to_string();

        let msg_text = "Persist this message";
        let _sid = "session_1".to_string();

        // Scope 1: Create agent and save a message
        let sid = {
            let mut agent = Agent::new(Arc::new(MockProvider));
            let memory = Arc::new(
                crate::system::database::EncryptedSqliteMemory::new(
                    db_path.clone(),
                    password.clone(),
                )
                .unwrap(),
            );

            // Create session
            let id = memory.create_session("test_proj").await.unwrap();
            agent.set_memory(memory.clone(), id.clone());

            let user_msg = Message::user(msg_text);
            agent.history.push(user_msg.clone());
            memory.add_message(&id, &user_msg).await.unwrap();
            id
        };

        // Scope 2: Recreate agent and verify loading
        {
            let mut agent = Agent::new(Arc::new(MockProvider));
            let memory = Arc::new(
                crate::system::database::EncryptedSqliteMemory::new(db_path, password).unwrap(),
            );
            agent.set_memory(memory, sid);

            // Load history
            agent.load_history().await.unwrap();

            assert_eq!(agent.history.len(), 1);
            if let Content::Text { text } = &agent.history[0].content[0] {
                assert_eq!(text, msg_text);
            } else {
                panic!("Expected text content");
            }
        }
    }

    #[tokio::test]
    async fn test_query_streaming_forwards_deltas_to_channel_before_final() {
        use tokio::sync::mpsc;

        struct TwoDeltaProvider;

        #[async_trait]
        impl Provider for TwoDeltaProvider {
            async fn stream_messages(
                &self,
                _messages: &[Message],
                _tools: &[Box<dyn Tool>],
            ) -> Result<BoxStream<'static, Result<ResponseChunk>>> {
                let chunks = vec![
                    Ok(ResponseChunk::TextDelta("Hello ".to_string())),
                    Ok(ResponseChunk::TextDelta("world".to_string())),
                    Ok(ResponseChunk::MessageDone(Message::assistant(
                        "Hello world",
                    ))),
                ];
                Ok(Box::pin(stream::iter(chunks)))
            }
        }

        let mut agent = Agent::new(Arc::new(TwoDeltaProvider));
        let (chunk_tx, mut chunk_rx) = mpsc::channel::<String>(8);

        let collector = tokio::spawn(async move {
            let mut received = Vec::new();
            while let Some(delta) = chunk_rx.recv().await {
                received.push(delta);
            }
            received
        });

        let final_text = agent.query_streaming("Hi", chunk_tx).await.unwrap();
        let received = collector.await.unwrap();

        assert_eq!(received, vec!["Hello ".to_string(), "world".to_string()]);
        assert_eq!(final_text, "Hello world");
        assert_eq!(received.concat(), final_text);
    }

    #[tokio::test]
    async fn test_set_provider_swaps_the_active_provider() {
        // A-S1 (#9): a mid-session set_provider replaces the active provider.
        struct FixedProvider(&'static str);

        #[async_trait]
        impl Provider for FixedProvider {
            async fn stream_messages(
                &self,
                _messages: &[Message],
                _tools: &[Box<dyn Tool>],
            ) -> Result<BoxStream<'static, Result<ResponseChunk>>> {
                let text = self.0.to_string();
                Ok(Box::pin(stream::iter(vec![Ok(
                    ResponseChunk::MessageDone(Message::assistant(&text)),
                )])))
            }
        }

        let mut agent = Agent::new(Arc::new(FixedProvider("from-A")));
        agent.set_provider(Arc::new(FixedProvider("from-B")));
        let (tx, _rx) = tokio::sync::mpsc::channel::<String>(8);
        let out = agent.query_streaming("hi", tx).await.unwrap();
        assert_eq!(out, "from-B", "set_provider must swap the active provider");
    }

    #[tokio::test]
    async fn test_provider_is_static_reflects_provider() {
        // StaticProvider → true (canned, safe to clear on login); other → false.
        let static_agent = Agent::new(Arc::new(crate::agent::provider::StaticProvider));
        assert!(static_agent.provider_is_static());
        let real_agent = Agent::new(Arc::new(MockProvider));
        assert!(!real_agent.provider_is_static());
    }

    #[tokio::test]
    async fn test_agent_history_resilience_to_key_rotation() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let db_path = tmp_dir.path().join("resilient_test.db");

        let key_a = "api_key_alpha".to_string();
        let key_b = "api_key_beta".to_string();
        let msg_text = "Secrets are safe";

        // Scope 1: Save with Key A
        let sid = {
            let memory =
                crate::system::database::EncryptedSqliteMemory::new(db_path.clone(), key_a)
                    .unwrap();
            let id = memory.create_session("test").await.unwrap();
            memory
                .add_message(&id, &Message::user(msg_text))
                .await
                .unwrap();
            id
        };

        // Scope 2: Attempt to load with Key B (Simulate rotation before fix)
        {
            let memory =
                crate::system::database::EncryptedSqliteMemory::new(db_path, key_b).unwrap();
            let result = memory.get_messages(&sid).await;

            // Expected failure: Key B cannot decrypt what Key A encrypted
            assert!(
                result.is_err(),
                "Recovery with a different key SHOULD fail currently"
            );
            assert!(result
                .unwrap_err()
                .to_string()
                .contains("Decryption failed"));
        }
    }
}
