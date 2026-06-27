// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-06-27

//! LLM-backed implementation of [`DistillJudge`].
//!
//! The real implementation calls the configured chat provider with bounded prompts.
//! Non-determinism is isolated here so tests can inject mock providers (R-06).

use async_trait::async_trait;
use std::sync::Arc;

use crate::agent::messages::Message;
use crate::agent::provider::Provider;
use crate::memory::error::MemoryError;
use crate::memory::profile::DistillJudge;
use crate::memory::store::Memory;

/// [`DistillJudge`] backed by the configured chat provider (REQ-17, agnostic).
///
/// Bounded prompts only; API keys are never logged or serialised (REQ-21).
/// Provider errors are propagated as typed [`MemoryError`] so the distiller's
/// CP2-Z non-fatal handler catches them without panic.
pub struct LlmDistillJudge {
    provider: Arc<dyn Provider>,
}

impl LlmDistillJudge {
    /// Creates a new judge wrapping the given provider.
    pub fn new(provider: Arc<dyn Provider>) -> Self {
        Self { provider }
    }
}

#[async_trait]
impl DistillJudge for LlmDistillJudge {
    async fn summarize_preferences(&self, episodic: &[Memory]) -> Result<Vec<String>, MemoryError> {
        if episodic.is_empty() {
            return Ok(vec![]);
        }

        let texts = episodic
            .iter()
            .map(|m| m.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "From the following user turns, list durable user preferences, one per line, \
             or nothing if none:\n{texts}"
        );

        let response = self
            .provider
            .send_messages(&[Message::user(&prompt)], &[])
            .await
            .map_err(|e| MemoryError::Storage(e.to_string()))?;

        let text = response
            .content
            .iter()
            .filter_map(|c| {
                if let crate::agent::messages::Content::Text { text } = c {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        let prefs = text
            .lines()
            .map(|line| {
                // Strip common list prefixes: "- ", "* ", "1. ", "2. " …
                let stripped = line
                    .trim_start_matches(|c: char| c.is_ascii_digit())
                    .trim_start_matches('.')
                    .trim_start_matches(['-', '*'])
                    .trim();
                stripped.to_string()
            })
            .filter(|s| !s.is_empty())
            .collect();

        Ok(prefs)
    }

    async fn contradicts(&self, a: &str, b: &str) -> Result<bool, MemoryError> {
        let prompt = format!(
            "Does statement B contradict or supersede statement A about the same subject? \
             Answer only yes or no.\nA: {a}\nB: {b}"
        );

        let response = self
            .provider
            .send_messages(&[Message::user(&prompt)], &[])
            .await
            .map_err(|e| MemoryError::Storage(e.to_string()))?;

        let text = response
            .content
            .iter()
            .filter_map(|c| {
                if let crate::agent::messages::Content::Text { text } = c {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        Ok(text.to_lowercase().contains("yes"))
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use async_trait::async_trait;
    use futures::stream::{self, BoxStream};
    use std::sync::Arc;

    use crate::agent::messages::Message;
    use crate::agent::provider::ResponseChunk;
    use crate::memory::error::MemoryError;
    use crate::memory::store::Memory;
    use crate::memory::MemoryKind;

    // ── CannedProvider ───────────────────────────────────────────────────────

    /// A provider that always returns a fixed canned response from both
    /// `stream_messages` and `send_messages`. Used to test `LlmDistillJudge`
    /// in isolation without HTTP calls (R-06).
    struct CannedProvider(String);

    #[async_trait]
    impl crate::agent::provider::Provider for CannedProvider {
        async fn stream_messages(
            &self,
            _messages: &[Message],
            _tools: &[Box<dyn crate::tools::Tool>],
        ) -> Result<BoxStream<'static, Result<ResponseChunk>>> {
            let text = self.0.clone();
            Ok(Box::pin(stream::iter(vec![
                Ok(ResponseChunk::TextDelta(text.clone())),
                Ok(ResponseChunk::MessageDone(Message::assistant(&text))),
            ])))
        }

        async fn send_messages(
            &self,
            _messages: &[Message],
            _tools: &[Box<dyn crate::tools::Tool>],
        ) -> Result<Message> {
            Ok(Message::assistant(&self.0))
        }
    }

    /// A provider that always returns `Err` (for error-path testing).
    struct FailingProvider;

    #[async_trait]
    impl crate::agent::provider::Provider for FailingProvider {
        async fn stream_messages(
            &self,
            _messages: &[Message],
            _tools: &[Box<dyn crate::tools::Tool>],
        ) -> Result<BoxStream<'static, Result<ResponseChunk>>> {
            Err(anyhow::anyhow!("provider failed"))
        }

        async fn send_messages(
            &self,
            _messages: &[Message],
            _tools: &[Box<dyn crate::tools::Tool>],
        ) -> Result<Message> {
            Err(anyhow::anyhow!("provider failed"))
        }
    }

    fn make_episodic(text: &str) -> Memory {
        Memory {
            id: "test-id".into(),
            session_id: "s".into(),
            kind: MemoryKind::Episodic,
            text: text.into(),
            embedding: vec![],
            model_id: String::new(),
            dim: 0,
            created_at: 1_000,
            salience: 0.3,
            access_count: 0,
            last_accessed_at: 1_000,
            superseded_by: None,
            evicted_at: None,
            scope: "root".into(),
            distilled_at: None,
        }
    }

    /// `summarize_preferences` parses the provider's response into trimmed,
    /// de-bulleted preference strings, one per line.
    #[tokio::test]
    async fn test_judge_parses_preferences_one_per_line() {
        let provider = Arc::new(CannedProvider("- use rust\n- dark mode".into()));
        let judge = LlmDistillJudge::new(provider);
        let episodic = vec![make_episodic("some user turn")];

        let result = judge.summarize_preferences(&episodic).await.unwrap();
        assert_eq!(
            result,
            vec!["use rust".to_string(), "dark mode".to_string()],
            "summarize_preferences must parse the provider's response into \
             trimmed, de-bulleted preference strings"
        );
    }

    /// `contradicts` returns `true` when the provider says "Yes" (case-insensitive)
    /// and `false` when it says "No".
    #[tokio::test]
    async fn test_judge_contradicts_parses_yes() {
        let yes_judge = LlmDistillJudge::new(Arc::new(CannedProvider("Yes.".into())));
        assert!(
            yes_judge
                .contradicts("statement A", "statement B")
                .await
                .unwrap(),
            "a 'Yes.' response must be parsed as true"
        );

        let no_judge = LlmDistillJudge::new(Arc::new(CannedProvider("No".into())));
        assert!(
            !no_judge
                .contradicts("statement A", "statement B")
                .await
                .unwrap(),
            "a 'No' response must be parsed as false"
        );
    }

    /// A provider error is propagated as a typed `MemoryError` (CP2-Z: non-fatal).
    #[tokio::test]
    async fn test_judge_provider_error_is_typed_memory_error() {
        let judge = LlmDistillJudge::new(Arc::new(FailingProvider));
        let episodic = vec![make_episodic("some text")];

        let result = judge.summarize_preferences(&episodic).await;
        assert!(
            result.is_err(),
            "a failing provider must produce Err, not panic"
        );
        assert!(
            matches!(result.unwrap_err(), MemoryError::Storage(_)),
            "the error must be a typed MemoryError::Storage"
        );
    }
}
