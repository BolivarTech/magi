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
//! - A 120 s overall request timeout is set on the `reqwest::Client` (G1). Unlike the
//!   chat SSE provider — where a total-request timeout would truncate healthy long streams
//!   — embedding calls are single non-streaming POSTs; 120 s is generous for cold Ollama
//!   model loads but still bounds indefinite hangs that block the user's turn.
//!   A hung server surfaces as [`EmbeddingError::Timeout`] and is handled by the agent's
//!   fallback (persist text-only; continue).

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
    /// Returns one vector per input text, in order. Implementations MUST uphold
    /// this as an invariant: an `Ok` result is never a `Vec` shorter than `texts`
    /// (in particular, `Ok` for a non-empty `texts` is never an empty `Vec`) —
    /// any per-text failure must surface as `Err`, not as a missing/short vector.
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
/// # Request timeout (G1)
/// The `reqwest::Client` is built with a **120 s overall request timeout**. Unlike
/// the chat SSE provider (where `.timeout(…)` would truncate healthy long streams),
/// embedding calls are single non-streaming POSTs. 120 s is generous for cold Ollama
/// model loads while still bounding indefinite hangs — a hung endpoint surfaces as
/// [`EmbeddingError::Timeout`] and triggers the agent's graceful fallback.
///
/// # Key safety
/// The API key is stored internally and **never** included in any error string.
///
/// # Autodetect mode
/// When `dim = 0` in [`EmbeddingConfig`], the embedder records the vector length
/// returned by the **first** successful call and enforces it on all subsequent calls
/// — a length mismatch on a later call returns [`EmbeddingError::Dim`].  The store
/// layer still filters by `model_id`/`dim` (D-06) as a second line of defence.
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
    pub fn new(cfg: &EmbeddingConfig, api_key: Option<String>) -> Result<Self, EmbeddingError> {
        // 120 s total-request timeout (G1): embedding calls are single-round-trip
        // non-streaming POSTs, unlike the chat SSE provider where a deadline would
        // truncate healthy long streams. 120 s accommodates cold Ollama model loads
        // while bounding indefinite hangs that would block the user's turn.
        //
        // W1: Client::builder() can fail on systems without TLS or with an invalid
        // certificate store. Return EmbeddingError::Network so callers apply the
        // graceful fallback (REQ-29) instead of panicking.
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .map_err(|_| EmbeddingError::Network)?;
        Ok(Self {
            client,
            // `cfg.base_url` is `Option<String>` since Task 1.1 (REQ-A21): absent means
            // "inherit the root `base_url`". This constructor only ever sees an
            // `EmbeddingConfig` in isolation — it has no access to the root value to
            // inherit from — so it falls back to the same Ollama default the field
            // itself used to carry before it became optional. Callers that need real
            // inheritance (the root `base_url` may differ from the Ollama default)
            // must resolve it first via `MagiConfig::effective_embedding_base_url` and
            // pass a config with `base_url` already populated; `main.rs` does this.
            base_url: cfg
                .base_url
                .as_deref()
                .unwrap_or(crate::defaults::DEFAULT_OPENAI_BASE_URL)
                .trim_end_matches('/')
                .to_string(),
            model: cfg.model.clone(),
            configured_dim: cfg.dim,
            detected_dim: AtomicUsize::new(0),
            query_prefix: cfg.query_prefix.clone(),
            document_prefix: cfg.document_prefix.clone(),
            api_key,
        })
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

        let response = builder.send().await.map_err(|e| {
            // The error detail (not the error *kind*) is discarded to guarantee
            // the API key can never leak via the error path.  The key is only in
            // the Authorization header which reqwest never surfaces in errors, but
            // future-proofing justifies the discard.
            if e.is_timeout() {
                EmbeddingError::Timeout
            } else {
                // Covers: connection refused, DNS, TLS, request build failures.
                EmbeddingError::Network
            }
        })?;

        let status = response.status();
        match status.as_u16() {
            401 | 403 => return Err(EmbeddingError::Auth),
            429 => return Err(EmbeddingError::RateLimited),
            200..=299 => {} // 2xx success — fall through to JSON parsing
            s => {
                // 3xx (if reqwest's redirect budget is exhausted or redirects are
                // disabled), 4xx (non-401/403/429), and 5xx are all treated as HTTP
                // errors (G5): anything not 2xx is a failure. The api_key is never
                // in the server's response body.
                let snippet = response
                    .text()
                    .await
                    .unwrap_or_default()
                    .chars()
                    .take(256)
                    .collect::<String>();
                return Err(EmbeddingError::Http(format!("HTTP {s} — {snippet}")));
            }
        }

        let parsed: EmbedResponse = response
            .json()
            .await
            .map_err(|e| EmbeddingError::Malformed(e.to_string()))?;

        if parsed.data.is_empty() {
            return Err(EmbeddingError::Malformed("empty data array".into()));
        }

        // In autodetect mode (configured_dim == 0), load any dimension that was
        // established by a previous successful call. Use Acquire ordering so we
        // see the latest value written by any thread that won the CAS (F4).
        let mut effective_dim = if self.configured_dim > 0 {
            self.configured_dim
        } else {
            self.detected_dim.load(Ordering::Acquire) // 0 if not yet detected
        };

        let mut out = Vec::with_capacity(parsed.data.len());
        for item in parsed.data {
            let got = item.embedding.len();
            if effective_dim > 0 {
                // Configured or previously-detected dimension: enforce consistency.
                if got != effective_dim {
                    return Err(EmbeddingError::Dim {
                        expected: effective_dim,
                        got,
                    });
                }
            } else {
                // Autodetect mode: a zero-length vector must be rejected before the
                // CAS — storing dim = 0 would make records unretrievable by the ANN
                // index and silently corrupt the cosine-similarity filter (G4).
                if got == 0 {
                    return Err(EmbeddingError::Malformed(
                        "zero-dimension embedding response".into(),
                    ));
                }
                // First successful response in autodetect mode. Use CAS so that
                // concurrent first-callers all converge on ONE dimension (F4).
                //
                // - If CAS succeeds: we established `got` as the canonical dim.
                // - If CAS fails: another thread already stored `winner`; enforce
                //   that our response matches (Dim error if it doesn't).
                match self.detected_dim.compare_exchange(
                    0,
                    got,
                    Ordering::AcqRel,  // success: release our store, acquire theirs
                    Ordering::Acquire, // failure: acquire the winner's value
                ) {
                    Ok(_) => {
                        // We won the race: `got` is now the established dim.
                        effective_dim = got;
                    }
                    Err(winner) => {
                        // Another thread won before us: enforce consistency.
                        if got != winner {
                            return Err(EmbeddingError::Dim {
                                expected: winner,
                                got,
                            });
                        }
                        effective_dim = winner;
                    }
                }
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
    ///
    /// Uses `Acquire` ordering so callers see the value established by any
    /// thread that won the CAS in [`call_embeddings`][OpenAiCompatibleEmbedder::call_embeddings] (F4).
    fn dim(&self) -> usize {
        if self.configured_dim > 0 {
            self.configured_dim
        } else {
            self.detected_dim.load(Ordering::Acquire)
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

// ─── Test-only constructors ───────────────────────────────────────────────────

#[cfg(test)]
impl OpenAiCompatibleEmbedder {
    /// Constructs an embedder with a caller-supplied `reqwest::Client`.
    ///
    /// **Test use only** — allows injecting short-timeout clients (G1) or
    /// no-redirect clients (G5) without modifying the production constructor.
    fn new_with_client(
        cfg: &EmbeddingConfig,
        api_key: Option<String>,
        client: reqwest::Client,
    ) -> Self {
        Self {
            client,
            // See the matching comment in `new` — `base_url` is `Option<String>` since
            // Task 1.1 (REQ-A21); this test-only constructor falls back the same way.
            base_url: cfg
                .base_url
                .as_deref()
                .unwrap_or(crate::defaults::DEFAULT_OPENAI_BASE_URL)
                .trim_end_matches('/')
                .to_string(),
            model: cfg.model.clone(),
            configured_dim: cfg.dim,
            detected_dim: AtomicUsize::new(0),
            query_prefix: cfg.query_prefix.clone(),
            document_prefix: cfg.document_prefix.clone(),
            api_key,
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::config::EmbeddingConfig;

    // ── W1: new() must return Result<Self, EmbeddingError>, not panic ─────────
    //
    // Red state: this test fails to *compile* against the current `-> Self`
    // signature. The fix: commit changes new() to `-> Result<Self, EmbeddingError>`
    // so that `.expect()` is replaced by `?`-propagation and the test becomes green.

    #[test]
    fn test_new_with_valid_config_returns_ok() {
        let cfg = EmbeddingConfig::default();
        // Type annotation drives the compile failure: new() currently returns Self,
        // not Result<Self, EmbeddingError>, so this line does not compile until the
        // fix: commit changes the signature.
        let result: Result<OpenAiCompatibleEmbedder, EmbeddingError> =
            OpenAiCompatibleEmbedder::new(&cfg, None);
        assert!(
            result.is_ok(),
            "W1: new() with a valid config must return Ok"
        );
    }

    fn cfg(base: &str) -> EmbeddingConfig {
        EmbeddingConfig {
            provider: "openai".into(),
            base_url: Some(base.into()),
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
        let emb =
            OpenAiCompatibleEmbedder::new(&cfg(&server.url()), Some("ollama".into())).unwrap();
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
        let emb = OpenAiCompatibleEmbedder::new(&cfg(&server.url()), Some("bad".into())).unwrap();
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
        let emb = OpenAiCompatibleEmbedder::new(&cfg(&server.url()), Some("k".into())).unwrap();
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
        let emb =
            OpenAiCompatibleEmbedder::new(&cfg(&server.url()), Some("ollama".into())).unwrap();
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
        let emb = OpenAiCompatibleEmbedder::new(&cfg(&server.url()), Some("SECRET-KEY-123".into()))
            .unwrap();
        let msg = emb.embed(&["x".into()]).await.unwrap_err().to_string();
        assert!(!msg.contains("SECRET-KEY-123"));
    }

    // ── Fix 2: Autodetect dim enforcement ─────────────────────────────────────

    /// Fix 2: when `dim = 0` (autodetect), the first successful response establishes
    /// the effective dimension; subsequent calls returning a different length must
    /// be rejected with `EmbeddingError::Dim`.
    ///
    /// Mocks are differentiated by body (`input` field) so mock ordering does not
    /// affect the test outcome.
    #[tokio::test]
    async fn test_autodetect_dim_enforced_on_second_call() {
        let mut server = mockito::Server::new_async().await;
        // Call 1 (input "hello"): 3-dim — autodetects dim = 3.
        let _m1 = server
            .mock("POST", "/embeddings")
            .match_body(mockito::Matcher::PartialJson(serde_json::json!(
                {"input": ["hello"]}
            )))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"data":[{"embedding":[0.1,0.2,0.3]}]}"#)
            .create_async()
            .await;
        // Call 2 (input "world"): 4-dim — must be rejected.
        let _m2 = server
            .mock("POST", "/embeddings")
            .match_body(mockito::Matcher::PartialJson(serde_json::json!(
                {"input": ["world"]}
            )))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"data":[{"embedding":[0.1,0.2,0.3,0.4]}]}"#)
            .create_async()
            .await;

        let emb = OpenAiCompatibleEmbedder::new(
            &EmbeddingConfig {
                dim: 0, // autodetect
                base_url: Some(server.url()),
                ..Default::default()
            },
            None,
        )
        .unwrap();

        // First call: autodetect establishes dim = 3.
        let first = emb.embed(&["hello".into()]).await.unwrap();
        assert_eq!(first[0].len(), 3, "Fix2: first response establishes dim=3");
        assert_eq!(emb.dim(), 3, "Fix2: dim() must reflect autodetected value");

        // Second call: 4-dim mismatch must be a typed error.
        let err = emb.embed(&["world".into()]).await.unwrap_err();
        assert!(
            matches!(
                err,
                EmbeddingError::Dim {
                    expected: 3,
                    got: 4
                }
            ),
            "Fix2: expected Dim{{expected:3,got:4}}, got: {err:?}"
        );
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
                base_url: Some("http://127.0.0.1:1".into()),
                ..Default::default()
            },
            None,
        )
        .unwrap();
        let err = emb.embed(&["hello".into()]).await.unwrap_err();
        assert!(
            matches!(err, EmbeddingError::Network),
            "F1: connection-refused must produce Network, got: {err:?}"
        );
    }

    // ── F4: CAS contract for autodetect dim ────────────────────────────────────

    /// F4: in autodetect mode (`dim = 0`), the first successful embed establishes
    /// the dimension and `dim()` reflects it; a subsequent response with a
    /// different length must be rejected even when the initial `detected_dim` was 0.
    ///
    /// This test documents the CAS (compare-and-swap) contract: once the first
    /// call establishes a dim, all subsequent calls — including concurrent ones
    /// that also saw `detected_dim == 0` — must converge on that single value.
    #[tokio::test]
    async fn test_autodetect_dim_cas_contract_established_before_second_call() {
        let mut server = mockito::Server::new_async().await;
        // First call: dim=4 established.
        let _m1 = server
            .mock("POST", "/embeddings")
            .match_body(mockito::Matcher::PartialJson(
                serde_json::json!({"input": ["first"]}),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"data":[{"embedding":[0.1,0.2,0.3,0.4]}]}"#)
            .create_async()
            .await;
        // Second call: dim=2 (mismatch) — must be a Dim error regardless of how
        // the first dim was stored (load/store vs CAS semantics).
        let _m2 = server
            .mock("POST", "/embeddings")
            .match_body(mockito::Matcher::PartialJson(
                serde_json::json!({"input": ["second"]}),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"data":[{"embedding":[0.5,0.6]}]}"#)
            .create_async()
            .await;

        let emb = OpenAiCompatibleEmbedder::new(
            &EmbeddingConfig {
                dim: 0, // autodetect
                base_url: Some(server.url()),
                ..Default::default()
            },
            None,
        )
        .unwrap();

        // Before any call, dim() must be 0 (nothing detected yet).
        assert_eq!(emb.dim(), 0, "F4: dim() before first call must be 0");

        // First call establishes dim=4.
        let first = emb.embed(&["first".into()]).await.unwrap();
        assert_eq!(first[0].len(), 4, "F4: first response has 4 components");
        assert_eq!(emb.dim(), 4, "F4: dim() after first call must be 4");

        // Second call with dim=2 must fail: established dim (4) != got (2).
        let err = emb.embed(&["second".into()]).await.unwrap_err();
        assert!(
            matches!(
                err,
                EmbeddingError::Dim {
                    expected: 4,
                    got: 2
                }
            ),
            "F4: CAS contract — Dim{{expected:4,got:2}} expected, got: {err:?}"
        );
    }

    // ── G1: embedding HTTP client timeout ────────────────────────────────────

    /// G1: a request to a server that stalls before sending a response produces
    /// `EmbeddingError::Timeout`. A raw `TcpListener` accepts the connection
    /// but holds it open (simulating a hung Ollama / cloud endpoint), while the
    /// short-deadline client fires before the server ever replies.
    ///
    /// Production verification: `new()` calls `reqwest::Client::builder().timeout(…)`
    /// with 120 s (confirmed by reading the constructor). This test exercises the
    /// timeout *mapping* (`e.is_timeout()` → `EmbeddingError::Timeout`) via a
    /// test-only client; the 120 s production value is validated by code review.
    #[tokio::test]
    async fn test_timeout_client_respects_deadline() {
        use tokio::net::TcpListener;

        // Spawn a TCP server that accepts but stalls before sending HTTP headers.
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            if let Ok((_socket, _)) = listener.accept().await {
                // Hold the socket alive well beyond the client timeout without
                // writing any bytes — reqwest waits for response headers and
                // fires its deadline first.
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            }
        });

        let base_url = format!("http://127.0.0.1:{}", addr.port());
        // 50 ms client deadline fires before the 5 s stall.
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(50))
            .build()
            .unwrap();
        let emb = OpenAiCompatibleEmbedder::new_with_client(
            &EmbeddingConfig {
                base_url: Some(base_url),
                ..Default::default()
            },
            None,
            client,
        );
        assert!(
            matches!(
                emb.embed(&["hello".into()]).await.unwrap_err(),
                EmbeddingError::Timeout
            ),
            "G1: stalled server beyond client deadline must produce Timeout"
        );
    }

    // ── G4: autodetect zero-dim response ─────────────────────────────────────

    /// G4: in autodetect mode (`dim = 0`), a server returning a zero-length
    /// embedding vector must produce `EmbeddingError::Malformed`, not store dim = 0.
    #[tokio::test]
    async fn test_autodetect_zero_dim_response_is_malformed() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/embeddings")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"data":[{"embedding":[]}]}"#)
            .create_async()
            .await;
        let emb = OpenAiCompatibleEmbedder::new(
            &EmbeddingConfig {
                dim: 0, // autodetect
                base_url: Some(server.url()),
                ..Default::default()
            },
            None,
        )
        .unwrap();
        assert!(
            matches!(
                emb.embed(&["x".into()]).await.unwrap_err(),
                EmbeddingError::Malformed(_)
            ),
            "G4: zero-length embedding in autodetect mode must produce Malformed"
        );
    }

    // ── G5: 3xx treated as HTTP error ────────────────────────────────────────

    /// G5: a 302 response (redirect not followed) must produce
    /// `EmbeddingError::Http`, not fall through to JSON parsing and produce Malformed.
    #[tokio::test]
    async fn test_redirect_response_produces_http_error() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/embeddings")
            .with_status(302)
            .with_body("Found")
            .create_async()
            .await;
        // Disable redirect-following so the 302 is returned to our status handler.
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        let emb = OpenAiCompatibleEmbedder::new_with_client(&cfg(&server.url()), None, client);
        assert!(
            matches!(
                emb.embed(&["x".into()]).await.unwrap_err(),
                EmbeddingError::Http(_)
            ),
            "G5: 302 response must produce EmbeddingError::Http, not Malformed"
        );
    }

    /// F4: in autodetect mode, a second call returning the SAME dim must succeed
    /// (convergence case — both the winner and any concurrent caller that saw
    /// the same response must be accepted).
    #[tokio::test]
    async fn test_autodetect_dim_cas_same_dim_on_second_call_succeeds() {
        let mut server = mockito::Server::new_async().await;
        let _m1 = server
            .mock("POST", "/embeddings")
            .match_body(mockito::Matcher::PartialJson(
                serde_json::json!({"input": ["a"]}),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"data":[{"embedding":[0.1,0.2,0.3]}]}"#)
            .create_async()
            .await;
        let _m2 = server
            .mock("POST", "/embeddings")
            .match_body(mockito::Matcher::PartialJson(
                serde_json::json!({"input": ["b"]}),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"data":[{"embedding":[0.4,0.5,0.6]}]}"#)
            .create_async()
            .await;

        let emb = OpenAiCompatibleEmbedder::new(
            &EmbeddingConfig {
                dim: 0,
                base_url: Some(server.url()),
                ..Default::default()
            },
            None,
        )
        .unwrap();
        let _ = emb.embed(&["a".into()]).await.unwrap();
        // Same dim=3 on second call — must succeed.
        let second = emb.embed(&["b".into()]).await.unwrap();
        assert_eq!(second[0].len(), 3, "F4: same-dim second call must succeed");
    }
}
