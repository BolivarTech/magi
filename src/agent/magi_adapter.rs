// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-06-07

//! Adapter that bridges magi-rs's `Provider` trait to magi-core's `LlmProvider`
//! trait, enabling MAGI consensus workflows to reuse the already-resolved backend.

use std::sync::Arc;
use async_trait::async_trait;
use magi_core::error::ProviderError;
use magi_core::provider::{CompletionConfig, LlmProvider};
use crate::agent::messages::{Content, Message};
use crate::agent::provider::Provider;

/// Delimiter that signals role separation when folding magi-core's distinct
/// `system_prompt` into a magi-rs user turn (magi-rs `Role` has no `System`).
const SYSTEM_FOLD_DELIMITER: &str = "\n\n---\n\n";

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

    // magi-core (verified v1.0.1 orchestrator.rs:709,1863) does NOT branch on the
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
