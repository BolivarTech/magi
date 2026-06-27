// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-06-26

//! Agnostic embedding provider trait and OpenAI-compatible HTTP implementation.
//!
//! # Design
//! - REQ-02: backends differ only by `base_url`; selection is via [`EmbeddingConfig`].
//! - REQ-29: all errors are typed ([`EmbeddingError`]); never panics; the API key is
//!   never included in any error string.
//! - D-04:   asymmetric task prefixes (`query_prefix` / `document_prefix`) are applied
//!   by the helpers [`embed_query`][OpenAiCompatibleEmbedder::embed_query] and
//!   [`embed_documents`][OpenAiCompatibleEmbedder::embed_documents].
//! - No `.timeout(…)` is set on the `reqwest::Client` — local Ollama can spend tens of
//!   seconds on a cold-load before the first byte arrives; a total-request timeout would
//!   truncate healthy long calls (matches `OpenAiCompatibleProvider` convention).

use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::memory::config::EmbeddingConfig;
use crate::memory::error::EmbeddingError;

// ─── Trait ───────────────────────────────────────────────────────────────────

/// Agnostic embedding provider (REQ-02).
///
/// Backends differ only by the `base_url` they are constructed with; the trait
/// surface is identical for local Ollama, Qwen Cloud, OpenAI, or any other
/// OpenAI-compatible embeddings endpoint.
///
/// Callers should use the higher-level helpers
/// [`embed_query`][OpenAiCompatibleEmbedder::embed_query] and
/// [`embed_documents`][OpenAiCompatibleEmbedder::embed_documents] which apply the
/// asymmetric task prefixes (D-04). The raw `embed` method accepts texts that
/// **have already been prefixed** (or that need no prefix).
// Narrow allow: trait methods dim/query_prefix/document_prefix consumed by the
// retrieval and context modules in Tasks 4–6.
#[allow(dead_code)]
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Embeds `texts` that have already been prefixed by the caller.
    ///
    /// Returns one vector per input text, in order.
    ///
    /// # Errors
    /// See [`EmbeddingError`] for the typed failure cases; never panics.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError>;

    /// The model id reported in the configuration (e.g. `"nomic-embed-text"`).
    fn model_id(&self) -> &str;

    /// The configured vector dimension.
    ///
    /// Returns `0` when `dim = 0` was specified in config (autodetect mode) and
    /// no successful response has been observed yet. After the first successful
    /// [`embed`][Self::embed] call in autodetect mode the internal detected
    /// dimension is stored and this method reflects the detected value.
    fn dim(&self) -> usize;

    /// Prefix applied to query text before embedding (D-04).
    fn query_prefix(&self) -> &str;

    /// Prefix applied to stored document text before embedding (D-04).
    fn document_prefix(&self) -> &str;
}

// ─── Private HTTP wire types ──────────────────────────────────────────────────

/// JSON body sent to `POST {base_url}/embeddings`.
#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

/// Top-level JSON response from the embeddings endpoint.
#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedData>,
}

/// One embedding item in the response `data` array.
#[derive(Deserialize)]
struct EmbedData {
    embedding: Vec<f32>,
}

// ─── Implementation ───────────────────────────────────────────────────────────

/// OpenAI-compatible embedder: `POST {base_url}/embeddings` with `Authorization: Bearer`.
///
/// Targets any endpoint that speaks the OpenAI embeddings API surface — local
/// Ollama (default), Qwen Cloud, OpenAI, or any other compatible service.
///
/// # No request timeout
/// The `reqwest::Client` is built without `.timeout(…)`. Local Ollama can spend
/// tens of seconds loading a model on a cold start before responding; a total-
/// request deadline would truncate healthy calls.
///
/// # Key safety
/// The API key is stored internally and **never** included in any error string.
///
/// # Autodetect mode
/// When `dim = 0` in [`EmbeddingConfig`], the embedder accepts whatever dimension
/// the endpoint returns on the first call and records it atomically. Subsequent
/// calls do not enforce a dimension check (the store layer is responsible for
/// filtering by `model_id`/`dim` — D-06).
pub struct OpenAiCompatibleEmbedder {
    client: reqwest::Client,
    base_url: String,
    model: String,
    /// Configured dimension (0 = autodetect).
    configured_dim: usize,
    /// Detected dimension from the first successful response; only used when
    /// `configured_dim == 0`. Stored with relaxed ordering — it is a best-effort
    /// hint for [`dim()`][Self::dim], not a security primitive.
    detected_dim: AtomicUsize,
    query_prefix: String,
    document_prefix: String,
    /// Bearer token sent in the `Authorization` header.
    /// `None` or empty → header omitted (compatible with Ollama's no-auth mode).
    /// Never included in error messages.
    api_key: Option<String>,
}

impl OpenAiCompatibleEmbedder {
    /// Constructs an embedder from the given config and optional API key.
    ///
    /// The key is taken as `Option<String>` rather than `String` so callers can
    /// propagate the absence of an environment variable without synthesising a
    /// dummy. An empty `Some("")` is treated as absent.
    // Narrow allow: new/embed_documents/embed_query consumed by the vector store
    // and retrieval modules in Tasks 4–6. Only tests exercise them here.
    #[allow(dead_code)]
    pub fn new(cfg: &EmbeddingConfig, api_key: Option<String>) -> Self {
        // No total-request timeout: cold Ollama loads can take tens of seconds
        // before the first byte; a deadline would break healthy long embeds.
        Self {
            client: reqwest::Client::new(),
            base_url: cfg.base_url.trim_end_matches('/').to_string(),
            model: cfg.model.clone(),
            configured_dim: cfg.dim,
            detected_dim: AtomicUsize::new(0),
            query_prefix: cfg.query_prefix.clone(),
            document_prefix: cfg.document_prefix.clone(),
            api_key,
        }
    }

    /// Prefixes each text with [`document_prefix`][Self::document_prefix] and
    /// calls the embeddings endpoint (for storage).
    ///
    /// # Errors
    /// See [`EmbeddingError`].
    #[allow(dead_code)]
    pub async fn embed_documents(&self, raw: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let prefixed: Vec<String> = raw
            .iter()
            .map(|t| apply_prefix(&self.document_prefix, t))
            .collect();
        self.embed(&prefixed).await
    }

    /// Prefixes `raw` with [`query_prefix`][Self::query_prefix] and calls the
    /// embeddings endpoint (for a retrieval query).
    ///
    /// Returns a single embedding vector.
    ///
    /// # Errors
    /// See [`EmbeddingError`].
    #[allow(dead_code)]
    pub async fn embed_query(&self, raw: &str) -> Result<Vec<f32>, EmbeddingError> {
        let prefixed = vec![apply_prefix(&self.query_prefix, raw)];
        let mut vecs = self.embed(&prefixed).await?;
        // Safety: embed() returns exactly one vector when given one input.
        vecs.pop()
            .ok_or_else(|| EmbeddingError::Malformed("empty data array".into()))
    }

    /// Core HTTP call: `POST {base_url}/embeddings`.
    ///
    /// Validates dimensions when `configured_dim > 0`. The API key is only sent
    /// as a Bearer token and is never written into any error variant.
    async fn call_embeddings(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let url = format!("{}/embeddings", self.base_url);
        let body = EmbedRequest {
            model: &self.model,
            input: texts,
        };

        let mut builder = self.client.post(&url).json(&body);
        if let Some(key) = &self.api_key {
            if !key.is_empty() {
                builder = builder.header("authorization", format!("Bearer {key}"));
            }
        }

        let response = builder.send().await.map_err(|_| {
            // All network-level failures (connect refused, DNS, timeout, request
            // build errors) map to Timeout.  The error detail is intentionally
            // discarded here to guarantee the API key can never leak via the
            // error path (the key is only in the Authorization header, but
            // future-proofing justifies the discard).
            EmbeddingError::Timeout
        })?;

        let status = response.status();
        match status.as_u16() {
            401 | 403 => return Err(EmbeddingError::Auth),
            429 => return Err(EmbeddingError::RateLimited),
            s if s >= 400 => {
                // Read a short body snippet for diagnostics.
                // The api_key is never in the server's response body.
                let snippet = response
                    .text()
                    .await
                    .unwrap_or_default()
                    .chars()
                    .take(256)
                    .collect::<String>();
                return Err(EmbeddingError::Http(format!("HTTP {s} — {snippet}")));
            }
            _ => {}
        }

        let parsed: EmbedResponse = response
            .json()
            .await
            .map_err(|e| EmbeddingError::Malformed(e.to_string()))?;

        if parsed.data.is_empty() {
            return Err(EmbeddingError::Malformed("empty data array".into()));
        }

        let effective_dim = self.configured_dim;
        let mut out = Vec::with_capacity(parsed.data.len());
        for item in parsed.data {
            let got = item.embedding.len();
            if effective_dim > 0 && got != effective_dim {
                return Err(EmbeddingError::Dim {
                    expected: effective_dim,
                    got,
                });
            }
            if effective_dim == 0 {
                // Autodetect: record the first observed dimension (best-effort).
                self.detected_dim.store(got, Ordering::Relaxed);
            }
            out.push(item.embedding);
        }
        Ok(out)
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAiCompatibleEmbedder {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        self.call_embeddings(texts).await
    }

    fn model_id(&self) -> &str {
        &self.model
    }

    /// Returns the configured dimension, or the autodetected value if `dim = 0`
    /// was specified and at least one successful call has been made.
    fn dim(&self) -> usize {
        if self.configured_dim > 0 {
            self.configured_dim
        } else {
            self.detected_dim.load(Ordering::Relaxed)
        }
    }

    fn query_prefix(&self) -> &str {
        &self.query_prefix
    }

    fn document_prefix(&self) -> &str {
        &self.document_prefix
    }
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// Prepends `prefix` to `text`, reusing the allocation when the prefix is empty.
// Narrow allow: called by embed_documents/embed_query; both are dead_code in this task.
#[allow(dead_code)]
fn apply_prefix(prefix: &str, text: &str) -> String {
    if prefix.is_empty() {
        text.to_string()
    } else {
        format!("{prefix}{text}")
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

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
        let out = emb
            .embed(&["search_document: hi".to_string()])
            .await
            .unwrap();
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
        let emb = OpenAiCompatibleEmbedder::new(&cfg(&server.url()), Some("SECRET-KEY-123".into()));
        let msg = emb.embed(&["x".into()]).await.unwrap_err().to_string();
        assert!(!msg.contains("SECRET-KEY-123"));
    }

    // ── F1: Network error variant ───────────────────────────────────────────────

    /// F1: a connection-refused error (non-timeout) must produce
    /// `EmbeddingError::Network`, not `EmbeddingError::Timeout`.
    ///
    /// We connect to a port that nobody listens on (OS immediately refuses), which
    /// is not a timeout — `reqwest` surfaces it as `is_connect() == true`.
    #[tokio::test]
    async fn test_connection_refused_produces_network_error_not_timeout() {
        // Port 1 is almost always closed and the OS refuses immediately.
        let emb = OpenAiCompatibleEmbedder::new(
            &EmbeddingConfig {
                base_url: "http://127.0.0.1:1".into(),
                ..Default::default()
            },
            None,
        );
        let err = emb.embed(&["hello".into()]).await.unwrap_err();
        assert!(
            matches!(err, EmbeddingError::Network),
            "F1: connection-refused must produce Network, got: {err:?}"
        );
    }
}
