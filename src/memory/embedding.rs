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
    /// This item's position in the request `input` array, per the OpenAI-compatible
    /// contract. Not every compatible server sets it, so it is `Option`: absent on
    /// every item falls back to trusting response array order (see
    /// [`reorder_by_index`]); present on every item is authoritative and reorders
    /// the response before anything reads it positionally.
    #[serde(default)]
    index: Option<usize>,
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
            base_url: effective_base_url_or_default(cfg),
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

        // The trait contract (`EmbeddingProvider::embed`) documents that `Ok` is NEVER a
        // `Vec` shorter than `texts`, in order — this is what upholds it against a
        // server that silently drops or duplicates an item. Without this check a
        // response with fewer items than requested would zip against `texts`
        // positionally at the caller (`retrieval::reembed_pending` does exactly this,
        // pairing each embedding with a memory `id` by position) and mis-associate
        // every entry after the first gap with the wrong memory — a corruption that
        // looks like a valid, plausible embedding rather than a visible error. A
        // response with MORE items than requested is equally rejected: the extra
        // entries have no legitimate consumer, and it is at least as likely to be a
        // server bug than an item at the front is unexpectedly missing.
        if parsed.data.len() != texts.len() {
            return Err(EmbeddingError::Malformed(format!(
                "expected {} embeddings for {} inputs, got {}",
                texts.len(),
                texts.len(),
                parsed.data.len()
            )));
        }

        // Reorders by `index` when the server reports one on every item — see
        // `reorder_by_index` for why response array order alone is not trusted.
        let data = reorder_by_index(parsed.data)?;

        // In autodetect mode (configured_dim == 0), load any dimension that was
        // established by a previous successful call. Use Acquire ordering so we
        // see the latest value written by any thread that won the CAS (F4).
        let mut effective_dim = if self.configured_dim > 0 {
            self.configured_dim
        } else {
            self.detected_dim.load(Ordering::Acquire) // 0 if not yet detected
        };

        let mut out = Vec::with_capacity(data.len());
        for item in data {
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

/// Reorders `data` by each item's `index` field when EVERY item reports one,
/// verifying the indices form an exact permutation of `0..data.len()`; returns
/// `data` unchanged when no item reports an index at all.
///
/// Nothing in the OpenAI-compatible wire contract guarantees a server preserves
/// response array order under internal batching or parallelization — `index` is
/// the field the contract defines for this purpose, so when it is present it is
/// authoritative instead of a hint. A response that reports `index` on SOME but
/// not all items, or whose indices are not a valid permutation (a duplicate or an
/// out-of-range value), is malformed rather than guessed at: silently trusting a
/// partial or malformed index set would reintroduce exactly the positional
/// mis-association this function exists to prevent.
///
/// # Errors
/// [`EmbeddingError::Malformed`] if `index` is reported on some but not all items,
/// or the reported indices are not a permutation of `0..data.len()`.
fn reorder_by_index(mut data: Vec<EmbedData>) -> Result<Vec<EmbedData>, EmbeddingError> {
    let with_index = data.iter().filter(|d| d.index.is_some()).count();
    if with_index == 0 {
        // No server-reported ordering at all: fall back to trusting response array
        // order, same as before this function existed. The caller's own count check
        // (`call_embeddings`) still guarantees the LENGTH is right either way.
        return Ok(data);
    }
    if with_index != data.len() {
        return Err(EmbeddingError::Malformed(
            "embedding response mixes items with and without an index".into(),
        ));
    }

    data.sort_by_key(|d| d.index.unwrap_or(usize::MAX));
    let is_permutation = data.iter().enumerate().all(|(i, d)| d.index == Some(i));
    if !is_permutation {
        return Err(EmbeddingError::Malformed(
            "embedding response indices are not a valid permutation of the request order".into(),
        ));
    }
    Ok(data)
}

/// Resolves `cfg.base_url` for construction: blank or absent falls back to the
/// Ollama default (review round 2, C1 defense in depth; m6 — shared by `new` and
/// `new_with_client` to remove the ≥3-line duplicate B3 flags).
///
/// This constructor only ever sees an `EmbeddingConfig` in isolation — it has no
/// access to the root `base_url` to inherit from — so a genuinely absent OR
/// blank value falls back to the same Ollama default the field itself carried
/// before it became optional (`Option<String>`, Task 1.1/REQ-A21). Callers that
/// need real root inheritance resolve it first via
/// `MagiConfig::effective_embedding_base_url` and hand this constructor a config
/// with `base_url` already populated (`main.rs` does this); this is the second,
/// defensive layer for any other caller — including a future one — that hands an
/// isolated `EmbeddingConfig` straight through with a blank `Some("")`.
fn effective_base_url_or_default(cfg: &EmbeddingConfig) -> String {
    cfg.base_url
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(crate::defaults::DEFAULT_OPENAI_BASE_URL)
        .trim_end_matches('/')
        .to_string()
}

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
            base_url: effective_base_url_or_default(cfg),
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

    /// C1 defense in depth (review round 2): a blank `Some("")` — not just
    /// `None` — falls back to the Ollama default. `.unwrap_or(DEFAULT)` alone
    /// (the code before this fix) only catches `None`; `Some("")` sailed
    /// through unchanged and reached `format!("{}/embeddings", self.base_url)`
    /// as `"/embeddings"`.
    #[test]
    fn effective_base_url_or_default_treats_blank_and_whitespace_as_absent() {
        for blank in ["", "   "] {
            let c = EmbeddingConfig {
                base_url: Some(blank.into()),
                ..EmbeddingConfig::default()
            };
            assert_eq!(
                effective_base_url_or_default(&c),
                crate::defaults::DEFAULT_OPENAI_BASE_URL,
                "blank {blank:?} must fall back to the default"
            );
        }
        let absent = EmbeddingConfig::default();
        assert_eq!(
            effective_base_url_or_default(&absent),
            crate::defaults::DEFAULT_OPENAI_BASE_URL
        );
        let declared = EmbeddingConfig {
            base_url: Some("http://custom:1234/v1/".into()),
            ..EmbeddingConfig::default()
        };
        assert_eq!(
            effective_base_url_or_default(&declared),
            "http://custom:1234/v1",
            "a real value is trimmed of its trailing slash, same as before"
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

    // ── S9 review round: response count integrity + `index` ordering ─────────
    //
    // `call_embeddings` previously zipped `parsed.data` against `texts` purely by
    // response array position, with no check that the two were the same length.
    // A caller that zips the result positionally against its own input (e.g.
    // `retrieval::reembed_pending`, which pairs each embedding with a memory `id`)
    // would silently mis-associate every entry after the first gap — a corruption
    // that looks like a valid result rather than a visible error.

    /// A response with FEWER embeddings than requested inputs must be rejected,
    /// never silently returned as a short `Vec` — the trait's documented invariant
    /// ("`Ok` is never a `Vec` shorter than `texts`") depends on this.
    #[tokio::test]
    async fn test_embed_response_with_fewer_items_than_inputs_is_rejected() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/embeddings")
            .with_status(200)
            .with_header("content-type", "application/json")
            // Only ONE item back, but two inputs are requested below.
            .with_body(r#"{"data":[{"embedding":[0.1,0.2,0.3]}]}"#)
            .create_async()
            .await;
        let emb = OpenAiCompatibleEmbedder::new(&cfg(&server.url()), None).unwrap();
        let err = emb
            .embed(&["a".to_string(), "b".to_string()])
            .await
            .unwrap_err();
        assert!(
            matches!(err, EmbeddingError::Malformed(_)),
            "a short response must be rejected, not silently zipped against the \
             wrong number of inputs: got {err:?}"
        );
    }

    /// A response with MORE embeddings than requested inputs is equally rejected:
    /// the extra items have no legitimate consumer and the mismatch itself is the
    /// signal something is wrong, whichever side is at fault.
    #[tokio::test]
    async fn test_embed_response_with_more_items_than_inputs_is_rejected() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/embeddings")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"data":[{"embedding":[0.1,0.2,0.3]},{"embedding":[0.4,0.5,0.6]}]}"#)
            .create_async()
            .await;
        let emb = OpenAiCompatibleEmbedder::new(&cfg(&server.url()), None).unwrap();
        let err = emb.embed(&["only-one".to_string()]).await.unwrap_err();
        assert!(
            matches!(err, EmbeddingError::Malformed(_)),
            "an over-long response must be rejected too: got {err:?}"
        );
    }

    /// When the server reports `index` on every item, the response is reordered
    /// by it BEFORE anything reads it positionally — response array order alone
    /// is never trusted when the contract's own ordering field is available.
    #[tokio::test]
    async fn test_embed_response_out_of_order_is_reordered_by_index() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/embeddings")
            .with_status(200)
            .with_header("content-type", "application/json")
            // Server returns index 1 before index 0 — deliberately out of request order.
            .with_body(
                r#"{"data":[{"embedding":[2.0,2.0,2.0],"index":1},{"embedding":[1.0,1.0,1.0],"index":0}]}"#,
            )
            .create_async()
            .await;
        let emb = OpenAiCompatibleEmbedder::new(&cfg(&server.url()), None).unwrap();
        let out = emb
            .embed(&["first".to_string(), "second".to_string()])
            .await
            .unwrap();
        assert_eq!(
            out[0],
            vec![1.0, 1.0, 1.0],
            "index 0 must land at position 0 regardless of response array order"
        );
        assert_eq!(
            out[1],
            vec![2.0, 2.0, 2.0],
            "index 1 must land at position 1 regardless of response array order"
        );
    }

    /// Indices that are not a valid permutation of `0..len` (a duplicate, here)
    /// are malformed rather than guessed at — silently accepting them would
    /// reintroduce the same mis-association the count check exists to prevent.
    #[tokio::test]
    async fn test_embed_response_with_duplicate_indices_is_rejected() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/embeddings")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{"data":[{"embedding":[1.0,1.0,1.0],"index":0},{"embedding":[2.0,2.0,2.0],"index":0}]}"#,
            )
            .create_async()
            .await;
        let emb = OpenAiCompatibleEmbedder::new(&cfg(&server.url()), None).unwrap();
        let err = emb
            .embed(&["a".to_string(), "b".to_string()])
            .await
            .unwrap_err();
        assert!(
            matches!(err, EmbeddingError::Malformed(_)),
            "duplicate indices are not a valid permutation and must be rejected: got {err:?}"
        );
    }

    /// A response that omits `index` entirely keeps working exactly as before —
    /// the common case (most OpenAI-compatible servers) pays nothing new besides
    /// the count check.
    #[tokio::test]
    async fn test_embed_response_without_any_index_still_trusts_array_order() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/embeddings")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"data":[{"embedding":[1.0,1.0,1.0]},{"embedding":[2.0,2.0,2.0]}]}"#)
            .create_async()
            .await;
        let emb = OpenAiCompatibleEmbedder::new(&cfg(&server.url()), None).unwrap();
        let out = emb
            .embed(&["first".to_string(), "second".to_string()])
            .await
            .unwrap();
        assert_eq!(out[0], vec![1.0, 1.0, 1.0]);
        assert_eq!(out[1], vec![2.0, 2.0, 2.0]);
    }
}
