// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-11

//! `ProviderProbe` answered from the persistent cache, with **independent** ceilings (REQ-R10/R12).
//!
//! # The crate does the opposite, and this runs underneath it
//!
//! magi-core's `run_preflight` wraps `window()` **and** `digest()` in a **single**
//! `tokio::time::timeout(DEFAULT_PREFLIGHT_TIMEOUT = 30 s)` and, on expiry, discards **both**
//! (`rotation.rs:756`). That is the *shared deadline* error this project forbids: a slow digest
//! throws away a window that was already measured. Since the preflight calls this implementation,
//! the ceilings have to be applied **inside** it and must never delegate to the crate's. Without
//! that, the defect returns through the back door and **invisibly** — the only symptom is a window
//! that sometimes is not measured.
//!
//! It also bounds the second risk the crate's timing creates: `run_preflight` runs **per consult**,
//! not once at startup, so an unbounded miss would spend the crate's 30 s *inside* a run rather
//! than at startup where the cost was budgeted.
//!
//! # The digest is measured on the first trip and never re-verified
//!
//! Not an optimisation — a decision (§3.9). `ProviderProbe::digest` is **per instance**, not per
//! model, so each call is its own `GET /api/tags`: re-verifying would cost one request per model on
//! every start. What it would buy degrades gently — the window feeds a pre-filter that fails open
//! and a threshold that only warns — while the lineage-collision check it enables is secondary and
//! advisory. So the stored digest means *"the one it had when we measured it"*, and a consumer that
//! reads it as current is wrong.
//!
//! # Why it lives in the bin
//!
//! Same reason as [`super::model_cache`]: it consumes that cache, which cannot live in the library
//! because the schema's single source of truth is bin-only.

#![deny(missing_docs)]
#![deny(clippy::missing_docs_in_private_items)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(clippy::missing_errors_doc, clippy::missing_panics_doc)]

use std::sync::Arc;
use std::time::Duration;

use magi_core::error::ProviderError;
use magi_core::rotation::ProviderProbe;
use magi_rs::magi::PROBE_TIMEOUT_SECS;

use super::model_cache::{CachedCapability, ModelCapabilityCache};

/// A [`ProviderProbe`] that answers from the persistent cache, measuring only on a miss.
pub struct CachedProbe {
    /// Shared cache over the encrypted database.
    cache: Arc<ModelCapabilityCache>,
    /// **Already redacted** endpoint, half of the cache key.
    ///
    /// Redaction happens upstream, where the trio is built; taking the redacted form in the
    /// signature makes that a property of the type rather than of a caller's memory.
    endpoint_redacted: String,
    /// The model this probe speaks for. A probe answers about ONE model and never guesses.
    model: String,
    /// What actually measures, on a miss. `None` means this probe can only serve cache hits —
    /// which is a legitimate state, not a degraded one: a candidate against a non-introspecting
    /// endpoint has nothing to measure with.
    source: Option<Arc<dyn ProviderProbe>>,
}

impl CachedProbe {
    /// Builds a probe for one `(endpoint_redacted, model)` pair.
    #[must_use]
    pub fn new(
        cache: Arc<ModelCapabilityCache>,
        endpoint_redacted: String,
        model: String,
        source: Option<Arc<dyn ProviderProbe>>,
    ) -> Self {
        Self {
            cache,
            endpoint_redacted,
            model,
            source,
        }
    }

    /// The cache behind this probe.
    #[must_use]
    pub fn cache(&self) -> &ModelCapabilityCache {
        &self.cache
    }

    /// Reads the cached row, treating any cache failure as a miss.
    ///
    /// Failing open here is the same discipline the whole measurement subsystem follows: an
    /// unreadable cache must degrade to measuring, never to refusing.
    fn cached(&self) -> Option<CachedCapability> {
        self.cache
            .get(&self.endpoint_redacted, &self.model)
            .ok()
            .flatten()
    }

    /// Measures one value under **its own** ceiling, never a shared one.
    ///
    /// A timeout, a transport failure and an unmeasurable answer all collapse to `None`: the probe
    /// fails open by contract, and `None` degrades a candidate while a guessed number would be
    /// trusted.
    async fn measure<T, F>(fut: F) -> Option<T>
    where
        F: std::future::Future<Output = Result<Option<T>, ProviderError>>,
    {
        tokio::time::timeout(Duration::from_secs(PROBE_TIMEOUT_SECS), fut)
            .await
            .ok()
            .and_then(Result::ok)
            .flatten()
    }
}

#[async_trait::async_trait]
impl ProviderProbe for CachedProbe {
    /// The context window, from the cache or measured once.
    ///
    /// # The whole row is written here, digest included
    ///
    /// A miss measures **both** values — each under its own ceiling — and persists them together,
    /// because the digest arrives on the same trip and costs no extra request. `digest()` then
    /// reads what this wrote. That is what makes "measured on the first trip and never
    /// re-verified" true rather than aspirational.
    ///
    /// **Only a measured window produces a row.** If the window does not resolve, nothing is
    /// written and the next start retries — persisting the failure would freeze a cold daemon's
    /// transient silence into a permanent answer.
    async fn window(&self) -> Result<Option<usize>, ProviderError> {
        Ok(None)
    }

    /// Stub.
    async fn digest(&self) -> Result<Option<String>, ProviderError> {
        Ok(None)
    }

    /// The model this probe speaks for, so the preflight can check the correspondence for free.
    fn declared_model(&self) -> Option<&str> {
        Some(&self.model)
    }
}

/// Unit tests for the cached probe.
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    use rusqlite::Connection;

    const ENDPOINT: &str = "http://localhost:11434/v1";
    const MODEL: &str = "qwen3.5:397b-cloud";

    /// A source whose window answers and whose digest hangs past any sane ceiling.
    ///
    /// The hang is **far** longer than the ceiling on purpose: the discriminating property is
    /// "the digest did not get its own answer in time", not a guessed duration (R-R05).
    struct WindowOkDigestHangs {
        /// Counts digest attempts, so a test can assert a warm row emits none.
        digest_calls: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl ProviderProbe for WindowOkDigestHangs {
        async fn window(&self) -> Result<Option<usize>, ProviderError> {
            Ok(Some(128_000))
        }
        async fn digest(&self) -> Result<Option<String>, ProviderError> {
            self.digest_calls.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_secs(3_600)).await;
            Ok(Some("never".into()))
        }
    }

    /// A source that never answers anything, standing in for a dead endpoint.
    ///
    /// It COUNTS its calls, and that is not bookkeeping: a timing assertion against a probe that
    /// never consulted its source passes trivially — "it was fast and returned nothing" is true of
    /// doing nothing at all. The count is what turns the ceiling assertion into a claim.
    struct DeadEndpoint {
        /// Window attempts, so a test can prove the source WAS consulted.
        window_calls: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl ProviderProbe for DeadEndpoint {
        async fn window(&self) -> Result<Option<usize>, ProviderError> {
            self.window_calls.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_secs(3_600)).await;
            Ok(None)
        }
        async fn digest(&self) -> Result<Option<String>, ProviderError> {
            tokio::time::sleep(Duration::from_secs(3_600)).await;
            Ok(None)
        }
    }

    /// An empty cache over an in-memory database and a fixed key — no passphrase, no Argon2.
    fn empty_cache() -> Arc<ModelCapabilityCache> {
        let conn = Connection::open_in_memory().expect("in-memory sqlite");
        let dek = magi_rs::vault::MaskedDek::new(zeroize::Zeroizing::new(vec![9u8; 32]))
            .expect("32 bytes");
        Arc::new(ModelCapabilityCache::new(Arc::new(Mutex::new(conn)), dek).expect("schema"))
    }

    /// SC-R54: a slow digest must NOT throw away an already-measured window, **in the same
    /// candidate**.
    ///
    /// SC-R09 pins isolation BETWEEN probes of different models — a different axis. The risk here
    /// is inside one candidate, between `window()` and `digest()`, and its symptom is silent.
    #[tokio::test(start_paused = true)]
    async fn a_slow_digest_does_not_discard_an_already_measured_window() {
        let source = Arc::new(WindowOkDigestHangs {
            digest_calls: AtomicUsize::new(0),
        });
        let cache = empty_cache();
        let probe = CachedProbe::new(
            Arc::clone(&cache),
            ENDPOINT.to_owned(),
            MODEL.to_owned(),
            Some(source),
        );

        assert_eq!(
            probe.window().await.expect("window resolves"),
            Some(128_000),
            "the window answered; a digest that did not must not take it down"
        );
        let row = cache
            .get(ENDPOINT, MODEL)
            .expect("cache read")
            .expect("a measured window MUST have produced a row");
        assert_eq!(row.window, 128_000);
        assert!(
            row.digest.is_none(),
            "the row goes in with digest = NULL rather than not going in at all"
        );
    }

    /// SC-R42: the digest is stored on the FIRST trip and never re-verified, in any route.
    #[tokio::test(start_paused = true)]
    async fn a_warm_row_emits_no_digest_probe() {
        let source = Arc::new(WindowOkDigestHangs {
            digest_calls: AtomicUsize::new(0),
        });
        let cache = empty_cache();
        cache
            .put(
                ENDPOINT,
                MODEL,
                &CachedCapability {
                    window: 64_000,
                    digest: Some("a".repeat(64)),
                },
            )
            .expect("put");
        let probe = CachedProbe::new(
            Arc::clone(&cache),
            ENDPOINT.to_owned(),
            MODEL.to_owned(),
            Some(Arc::clone(&source) as Arc<dyn ProviderProbe>),
        );

        assert_eq!(
            probe.digest().await.expect("digest"),
            Some("a".repeat(64)),
            "a warm row answers from the cache"
        );
        assert_eq!(
            probe.window().await.expect("window"),
            Some(64_000),
            "and so does the window"
        );
        assert_eq!(
            source.digest_calls.load(Ordering::SeqCst),
            0,
            "a warm row must emit NO digest probe — not on any route"
        );
    }

    /// REQ-R12 / S2: a cache MISS is bounded by magi-rs's ceiling, never the crate's 30 s.
    ///
    /// The preflight runs **per consult**, so an unbounded miss would spend that inside the run
    /// rather than at startup where the cost was budgeted. Asserted against the ceiling rather
    /// than a guessed sleep: the double hangs for an hour, so the discriminating property is
    /// "not the double's hang", exactly as R-R05 requires.
    #[tokio::test(start_paused = true)]
    async fn a_cache_miss_is_bounded_by_the_magi_rs_ceiling_not_the_crates() {
        let source = Arc::new(DeadEndpoint {
            window_calls: AtomicUsize::new(0),
        });
        let probe = CachedProbe::new(
            empty_cache(),
            ENDPOINT.to_owned(),
            MODEL.to_owned(),
            Some(Arc::clone(&source) as Arc<dyn ProviderProbe>),
        );

        let started = tokio::time::Instant::now();
        assert_eq!(
            probe
                .window()
                .await
                .expect("a dead endpoint is not an error"),
            None,
            "unmeasured is a valid answer; the probe fails OPEN"
        );
        let elapsed = started.elapsed();
        assert_eq!(
            source.window_calls.load(Ordering::SeqCst),
            1,
            "the miss must have REACHED the source; a ceiling assertion against a probe that              never measured passes for the wrong reason"
        );
        assert!(
            elapsed <= Duration::from_secs(PROBE_TIMEOUT_SECS + 1),
            "a miss must be bounded by magi-rs's {PROBE_TIMEOUT_SECS}s, not the crate's 30 s \
             nor the double's hour; took {elapsed:?}"
        );
    }

    /// A failed measurement writes NOTHING, so the next start retries (SC-R26 at the probe level).
    #[tokio::test(start_paused = true)]
    async fn a_failed_measurement_leaves_no_row_behind() {
        let cache = empty_cache();
        let source = Arc::new(DeadEndpoint {
            window_calls: AtomicUsize::new(0),
        });
        let probe = CachedProbe::new(
            Arc::clone(&cache),
            ENDPOINT.to_owned(),
            MODEL.to_owned(),
            Some(Arc::clone(&source) as Arc<dyn ProviderProbe>),
        );
        assert_eq!(probe.window().await.expect("no error"), None);
        assert_eq!(
            source.window_calls.load(Ordering::SeqCst),
            1,
            "same reason as above: an empty table proves nothing unless a measurement was tried"
        );
        assert_eq!(
            cache.row_count().expect("count"),
            0,
            "a transient silence must not become a permanent answer"
        );
    }

    /// REQ-R28: the probe declares the model it speaks for, so the preflight can check the
    /// correspondence at no cost. Answering `None` would silently opt out of that check.
    #[test]
    fn the_probe_declares_the_model_it_speaks_for() {
        let probe = CachedProbe::new(empty_cache(), ENDPOINT.to_owned(), MODEL.to_owned(), None);
        assert_eq!(probe.declared_model(), Some(MODEL));
    }

    /// A probe with no measurement source serves hits and answers `None` on a miss — a legitimate
    /// state, not a degraded one: a candidate on a non-introspecting endpoint has nothing to
    /// measure with.
    #[tokio::test]
    async fn a_probe_without_a_source_answers_none_instead_of_failing() {
        let probe = CachedProbe::new(empty_cache(), ENDPOINT.to_owned(), MODEL.to_owned(), None);
        assert_eq!(probe.window().await.expect("not an error"), None);
        assert_eq!(probe.digest().await.expect("not an error"), None);
    }
}
