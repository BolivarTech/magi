// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-06-07

//! Adapter that bridges magi-rs's `Provider` trait to magi-core's `LlmProvider`
//! trait, enabling MAGI consensus workflows to reuse the already-resolved backend.

use crate::agent::messages::{Content, Message};
use crate::agent::provider::Provider;
use async_trait::async_trait;
use magi_core::error::ProviderError;
use magi_core::provider::{CompletionConfig, LlmProvider};
use std::sync::Arc;

/// Delimiter that signals role separation when folding magi-core's distinct
/// `system_prompt` into a magi-rs user turn (magi-rs `Role` has no `System`).
const SYSTEM_FOLD_DELIMITER: &str = "\n\n---\n\n";

/// Adapts an `Arc<dyn Provider>` (magi-rs backend) to magi-core's `LlmProvider`.
///
/// `complete` folds magi-core's distinct `system_prompt` into a single magi-rs
/// user `Message` (magi-rs `Role` has no `System` variant) and drains
/// `Provider::send_messages` for the assembled reply text.
///
/// MAGI FIX (consensus-fidelity caveat): magi-core differentiates its three
/// personas via distinct *system* prompts. magi-rs cannot send a system role
/// without changing the `Provider` trait (out of scope), so the system text is
/// folded into the user turn behind [`SYSTEM_FOLD_DELIMITER`]. On weak/local
/// models this can weaken persona divergence and JSON adherence. Documented
/// limitation; revisit when magi-rs gains a system-prompt channel.
///
/// `CompletionConfig` (`max_tokens`/`temperature`) is intentionally NOT applied:
/// the magi-rs `Provider` trait exposes no per-call knobs. Deferred (CHANGELOG).
///
/// **Concurrency safety**: the `Provider` trait requires `Send + Sync` (enforced
/// by the trait bound on `Arc<dyn Provider>`), and `complete` takes `&self`, so
/// multiple concurrent consult calls (e.g. the forced `/consult` handle and the
/// auto-path `ConsultTool`) sharing the same `Arc<MagiCoreProviderAdapter>` are safe.
pub struct MagiCoreProviderAdapter {
    inner: Arc<dyn Provider>,
    name: String,
    model: String,
}

impl MagiCoreProviderAdapter {
    /// Creates the adapter over a resolved magi-rs backend.
    pub fn new(
        inner: Arc<dyn Provider>,
        name: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            inner,
            name: name.into(),
            model: model.into(),
        }
    }

    /// Folds a system + user prompt into one user-turn string with an explicit
    /// role-separating delimiter. Empty system → user prompt unchanged.
    pub(crate) fn fold_prompt(system_prompt: &str, user_prompt: &str) -> String {
        if system_prompt.is_empty() {
            user_prompt.to_string()
        } else {
            format!("{system_prompt}{SYSTEM_FOLD_DELIMITER}{user_prompt}")
        }
    }
}

#[async_trait]
impl LlmProvider for MagiCoreProviderAdapter {
    async fn complete(
        &self,
        system_prompt: &str,
        user_prompt: &str,
        _config: &CompletionConfig,
    ) -> Result<String, ProviderError> {
        let prompt = Self::fold_prompt(system_prompt, user_prompt);
        let messages = [Message::user(&prompt)];
        let reply = self
            .inner
            .send_messages(&messages, &[])
            .await
            .map_err(|e| ProviderError::Network {
                message: e.to_string(),
            })?;
        let text = reply
            .content
            .iter()
            .filter_map(|c| match c {
                Content::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");
        Ok(text)
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn model(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::provider::ResponseChunk;
    use crate::tools::Tool;
    use anyhow::{anyhow, Result};
    use futures::stream::{self, BoxStream, StreamExt};

    struct CannedProvider;
    #[async_trait]
    impl Provider for CannedProvider {
        async fn stream_messages(
            &self,
            _messages: &[Message],
            _tools: &[Box<dyn Tool>],
        ) -> Result<BoxStream<'static, Result<ResponseChunk>>> {
            let chunks = vec![
                Ok(ResponseChunk::TextDelta("hello ".to_string())),
                Ok(ResponseChunk::MessageDone(Message::assistant(
                    "hello world",
                ))),
            ];
            Ok(stream::iter(chunks).boxed())
        }
    }

    struct FailingProvider;
    #[async_trait]
    impl Provider for FailingProvider {
        async fn stream_messages(
            &self,
            _messages: &[Message],
            _tools: &[Box<dyn Tool>],
        ) -> Result<BoxStream<'static, Result<ResponseChunk>>> {
            Err(anyhow!("boom 401 unauthorized"))
        }
    }

    #[tokio::test]
    async fn test_adapter_complete_returns_assembled_text() {
        let adapter = MagiCoreProviderAdapter::new(
            Arc::new(CannedProvider),
            "anthropic",
            "claude-sonnet-4-6",
        );
        let out = adapter
            .complete(
                "you are a scientist",
                "should we X?",
                &CompletionConfig::default(),
            )
            .await
            .expect("complete should succeed");
        assert_eq!(out, "hello world");
        assert_eq!(adapter.name(), "anthropic");
        assert_eq!(adapter.model(), "claude-sonnet-4-6");
    }

    #[test]
    fn test_fold_uses_role_separating_delimiter() {
        assert_eq!(
            MagiCoreProviderAdapter::fold_prompt("SYS", "USR"),
            format!("SYS{SYSTEM_FOLD_DELIMITER}USR")
        );
        assert_eq!(MagiCoreProviderAdapter::fold_prompt("", "USR"), "USR");
    }

    // magi-core (verified v1.0.1; unchanged in 1.1.0 orchestrator.rs:709,1863) does NOT branch on the
    // ProviderError variant — every provider error becomes a failed agent — so even
    // an auth-shaped message maps to Network here (no behavioral consumer for the
    // distinction). If a future magi-core branches on the variant, revisit.
    #[tokio::test]
    async fn test_backend_error_maps_to_network_with_message_preserved() {
        let adapter = MagiCoreProviderAdapter::new(Arc::new(FailingProvider), "anthropic", "m");
        match adapter
            .complete("s", "u", &CompletionConfig::default())
            .await
        {
            Err(ProviderError::Network { message }) => assert!(message.contains("401")),
            other => panic!("expected Network error preserving the message, got {other:?}"),
        }
    }
}
