// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-06-26

//! Agnostic embedding provider trait and OpenAI-compatible HTTP implementation.
//!
//! REQ-02: backends differ only by `base_url`; selection is via `EmbeddingConfig`.
//! REQ-29: all errors are typed (`EmbeddingError`); never panics; never leaks the API key.
//! D-04:   asymmetric task prefixes (`query_prefix` / `document_prefix`) are applied
//!         by the caller helpers `embed_query` / `embed_documents`.

use crate::memory::error::EmbeddingError;

// Placeholder import so the test module compiles its `use` path.
// The full struct and trait arrive in the Green phase.
use crate::memory::config::EmbeddingConfig as _EmbeddingConfig;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::config::EmbeddingConfig;

    fn cfg(base: &str) -> EmbeddingConfig {
        EmbeddingConfig {
            provider: "openai".into(),
            base_url: base.into(),
            model: "nomic-embed-text".into(),
            dim: 3,
            query_prefix: "search_query: ".into(),
            document_prefix: "search_document: ".into(),
        }
    }

    #[tokio::test]
    async fn test_embed_returns_vector_of_configured_dim_and_reports_model() {
        let mut server = mockito::Server::new_async().await;
        let m = server
            .mock("POST", "/embeddings")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"data":[{"embedding":[0.1,0.2,0.3]}]}"#)
            .create_async()
            .await;
        let emb = OpenAiCompatibleEmbedder::new(&cfg(&server.url()), Some("ollama".into()));
        let out = emb.embed(&["search_document: hi".to_string()]).await.unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), 3);
        assert_eq!(emb.model_id(), "nomic-embed-text");
        m.assert_async().await;
    }

    #[tokio::test]
    async fn test_embed_auth_failure_is_typed_error_no_panic() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/embeddings")
            .with_status(401)
            .create_async()
            .await;
        let emb = OpenAiCompatibleEmbedder::new(&cfg(&server.url()), Some("bad".into()));
        assert!(matches!(
            emb.embed(&["x".into()]).await.unwrap_err(),
            EmbeddingError::Auth
        ));
    }

    #[tokio::test]
    async fn test_429_is_rate_limited_typed_error() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/embeddings")
            .with_status(429)
            .create_async()
            .await;
        let emb = OpenAiCompatibleEmbedder::new(&cfg(&server.url()), Some("k".into()));
        assert!(matches!(
            emb.embed(&["x".into()]).await.unwrap_err(),
            EmbeddingError::RateLimited
        ));
    }

    #[tokio::test]
    async fn test_prefixes_are_applied_to_outgoing_request_body() {
        let mut server = mockito::Server::new_async().await;
        let m = server
            .mock("POST", "/embeddings")
            .match_body(mockito::Matcher::PartialJson(serde_json::json!(
                {"input": ["search_query: weather"]}
            )))
            .with_status(200)
            .with_body(r#"{"data":[{"embedding":[0.0,0.0,0.0]}]}"#)
            .create_async()
            .await;
        let emb = OpenAiCompatibleEmbedder::new(&cfg(&server.url()), Some("ollama".into()));
        let _ = emb.embed_query("weather").await.unwrap();
        m.assert_async().await;
    }

    #[tokio::test]
    async fn test_error_messages_redact_key() {
        // No error string ever contains the api_key.
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/embeddings")
            .with_status(500)
            .with_body("boom")
            .create_async()
            .await;
        let emb =
            OpenAiCompatibleEmbedder::new(&cfg(&server.url()), Some("SECRET-KEY-123".into()));
        let msg = emb.embed(&["x".into()]).await.unwrap_err().to_string();
        assert!(!msg.contains("SECRET-KEY-123"));
    }
}
