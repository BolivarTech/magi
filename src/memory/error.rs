// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-06-26

//! Domain error types for the memory subsystem.
//!
//! All errors are typed (never panic — REQ-29). `MemoryError` wraps
//! `EmbeddingError` via `#[from]` so the `?` operator propagates across the
//! embedding/storage boundary without boilerplate.

use thiserror::Error;

/// Errors from the embedding provider.
///
/// Typed variants let callers decide whether to retry (e.g. [`RateLimited`]),
/// fall back to text-only storage ([`Auth`], [`Http`]), or surface a config
/// problem ([`Dim`]).
///
/// [`RateLimited`]: EmbeddingError::RateLimited
/// [`Auth`]: EmbeddingError::Auth
/// [`Http`]: EmbeddingError::Http
/// [`Dim`]: EmbeddingError::Dim
// Narrow allow: variants wired up in later tasks (Task 3/4 embedder).
#[allow(dead_code)]
#[derive(Debug, Error)]
pub enum EmbeddingError {
    /// A non-auth HTTP error from the embedding endpoint (status text or body).
    #[error("embedding HTTP error: {0}")]
    Http(String),

    /// The API key was rejected (401 or 403).
    #[error("embedding auth failed (401/403)")]
    Auth,

    /// The endpoint returned 429 — the caller should back off before retrying.
    #[error("embedding rate-limited (429)")]
    RateLimited,

    /// The embedding endpoint did not respond within the deadline.
    #[error("embedding endpoint timed out")]
    Timeout,

    /// A network-level failure that is not a timeout (e.g. connection refused,
    /// DNS resolution failure, TLS handshake error).
    #[error("embedding network error")]
    Network,

    /// The endpoint returned a response that could not be decoded as expected.
    #[error("malformed embedding response: {0}")]
    Malformed(String),

    /// The returned vector dimension differs from the configured `dim`.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    Dim {
        /// The `dim` value declared in `[embedding]` config.
        expected: usize,
        /// The actual dimension returned by the endpoint.
        got: usize,
    },
}

/// Errors from the memory subsystem.
///
/// Domain errors; use `thiserror` so call-sites can match on variants.
/// `EmbeddingError` is automatically converted via [`#[from]`] so `?` works
/// across the embedding/storage boundary.
///
/// At application boundaries (`main.rs`, TUI handlers) these are wrapped in
/// `anyhow::Result` for ergonomic propagation.
// Narrow allow: variants wired up in later tasks (Task 3/4 store, context assembler).
#[allow(dead_code)]
#[derive(Debug, Error)]
pub enum MemoryError {
    /// An encrypted-storage or serialization failure.
    #[error("crypto error: {0}")]
    Crypto(String),

    /// A SQLite-level storage failure.
    #[error("storage error: {0}")]
    Storage(String),

    /// An error from the embedding provider (propagated via `?`).
    #[error("embedding error: {0}")]
    Embedding(#[from] EmbeddingError),

    /// The system prompt and user preference profile together already exceed
    /// the configured context budget — this is a configuration error.
    #[error("context budget exceeded by system+profile alone (config error)")]
    BudgetUnsatisfiable,

    /// A configuration value is invalid or out of range.
    #[error("invalid memory config: {0}")]
    Config(String),

    /// An LLM/provider failure during the distillation pass (e.g. connection
    /// refused, authentication error, rate-limit after retries). Non-fatal
    /// (CP2-Z): the distiller catches this variant, logs it, and retries the
    /// batch on the next scheduled pass.
    #[error("LLM provider error: {0}")]
    Llm(String),
}
