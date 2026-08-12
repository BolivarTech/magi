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
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::todo,
        clippy::unimplemented,
        clippy::indexing_slicing,
        clippy::string_slice
    )
)]

use std::sync::Arc;
use std::time::Duration;

use magi_core::error::ProviderError;
use magi_core::rotation::ProviderProbe;
use magi_rs::magi::probe::validate_digest;
use magi_rs::magi::{PROBE_TIMEOUT_SECS, PROBE_WINDOW_MAX, PROBE_WINDOW_MIN};

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

    /// Reads the cached row, treating any cache failure as a miss.
    ///
    /// Failing open here is the same discipline the whole measurement subsystem follows: an
    /// unreadable cache must degrade to measuring, never to refusing.
    fn cached(&self) -> Option<CachedCapability> {
        self.cache
            .get(&self.endpoint_redacted, &self.model)
            .ok()
            .flatten()
            // Re-checked on READ, not only on write, because the range is a pair of constants and
            // the cache outlives the process that filled it. Narrow `PROBE_WINDOW_MIN/MAX` in a
            // later version and every row written under the old pair is still on disk, valid by a
            // rule that no longer exists — and nothing re-verifies it, so the stale value would be
            // served for the life of the install. Costs a comparison on a value already in hand
            // (S2 Loop 2, Balthasar).
            .filter(|row| (PROBE_WINDOW_MIN..=PROBE_WINDOW_MAX).contains(&row.window))
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
    /// while the miss is already being handled. `digest()` then reads what this wrote. That is
    /// what makes "measured on the first trip and never re-verified" true rather than
    /// aspirational.
    ///
    /// **It is not free, and the comment here used to say it was** (S2 Loop 2, Balthasar).
    /// `ProviderProbe::digest` is a separate call over a separate endpoint — `/api/tags` against
    /// `/api/show` for the window — so it costs one more request per model. What the design
    /// avoids is not that request; it is a SECOND pass later, on every start, to re-verify
    /// something that only changes when the configuration does. Writing "costs no extra request"
    /// made the trade sound like a free lunch, which is how a later reader talks themselves into
    /// re-verifying "since it's cheap anyway".
    ///
    /// **Only a measured window produces a row.** If the window does not resolve, nothing is
    /// written and the next start retries — persisting the failure would freeze a cold daemon's
    /// transient silence into a permanent answer.
    ///
    /// **"Resolve" includes being in range**, on the same terms `probe_models` applies at startup
    /// (`probe.rs`: `Some(w) if (PROBE_WINDOW_MIN..=PROBE_WINDOW_MAX).contains(&w)`). This path
    /// used to skip that check, so the same daemon answer degraded to *not measured* during
    /// startup and was accepted here — and accepted meant **cached**, which the design never
    /// re-verifies, so one absurd reading became this process's permanent truth and every later
    /// one's. It then reached magi-core's condition #6 as if it were a fact. Found by S2 Loop 2
    /// (Caspar and Balthasar, independently).
    async fn window(&self) -> Result<Option<usize>, ProviderError> {
        if let Some(row) = self.cached() {
            return Ok(Some(row.window));
        }
        let Some(source) = self.source.as_ref() else {
            return Ok(None);
        };

        // TWO ceilings, sequential and independent. One `timeout` around both would let a slow
        // digest discard a window that already resolved — `rotation.rs:756`'s defect.
        let Some(window) = Self::measure(source.window()).await else {
            return Ok(None);
        };
        if !(PROBE_WINDOW_MIN..=PROBE_WINDOW_MAX).contains(&window) {
            // Out of range is a failed measurement, not a small one: return without caching, so
            // the next run asks again instead of inheriting this answer forever.
            return Ok(None);
        }
        // Validated with the SAME predicate the startup path uses, not a second copy of it: a
        // malformed digest is discarded and the window survives (REQ-R25 persists a digest only
        // if it resolved AND passed the format check). Without this the cache — which nothing
        // re-verifies — kept whatever the daemon said, and `corroborate_by_digest` compared
        // garbage for equality (S2 Loop 2, Caspar).
        let digest = validate_digest(Self::measure(source.digest()).await);

        // A cache write that fails must not fail the measurement: the value is already in hand.
        let _ = self.cache.put(
            &self.endpoint_redacted,
            &self.model,
            &CachedCapability { window, digest },
        );
        Ok(Some(window))
    }

    /// The weights digest, **from the cache only**.
    ///
    /// It is never measured here and never re-verified: `window()` wrote it on the first trip. A
    /// miss answers `None`, which is inert — an unresolved digest cannot collide with anything.
    async fn digest(&self) -> Result<Option<String>, ProviderError> {
        Ok(self.cached().and_then(|row| row.digest))
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

    /// Our two independent ceilings must fit, **summed**, inside the crate's single outer one.
    ///
    /// REQ-R12 has this probe apply its own ceiling to `window()` and to `digest()` separately,
    /// so a slow digest cannot throw away a window that already resolved — the shared-deadline
    /// defect `run_preflight` itself commits (`rotation.rs:756`, one `timeout` around both) and
    /// that `SC-R54` pins from the inside.
    ///
    /// **This pins it from the outside, which is the half that was open.** Our ceilings are
    /// sequential inside that outer 30 s, so the worst case we can spend is
    /// `2 × PROBE_TIMEOUT_SECS`. Raise that constant past half the crate's budget and the outer
    /// timeout fires first, discarding **both** measurements — reintroducing the exact defect
    /// from the opposite direction, with no test to notice, and a symptom (a window that
    /// sometimes is not measured) that looks like a flaky daemon rather than a config change.
    ///
    /// Anchored to `DEFAULT_PREFLIGHT_TIMEOUT` rather than to a literal `30` on purpose: the
    /// number belongs to magi-core, so a future release that lowers it must break this test
    /// rather than silently shrink the room we assumed. Found by S2 Loop 2 (Caspar).
    #[test]
    fn our_two_ceilings_fit_inside_the_crate_preflight_budget() {
        let ours = Duration::from_secs(PROBE_TIMEOUT_SECS) * 2;
        let theirs = magi_core::rotation::DEFAULT_PREFLIGHT_TIMEOUT;

        assert!(
            ours <= theirs,
            "window + digest may spend {ours:?} but run_preflight wraps both in {theirs:?}; \
             above half of it the crate's timeout wins and discards a window we already had"
        );
    }

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

    /// A source answering a window **outside** `[PROBE_WINDOW_MIN, PROBE_WINDOW_MAX]`.
    ///
    /// It counts digest calls for the same reason `WindowOkDigestHangs` does: an out-of-range
    /// window must be abandoned BEFORE the second trip, so proving the digest was never asked is
    /// what separates "rejected the reading" from "rejected it after paying for it anyway".
    struct WindowOutOfRange {
        /// The out-of-range value to answer.
        window: usize,
        /// Digest attempts, which must stay at zero.
        digest_calls: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl ProviderProbe for WindowOutOfRange {
        async fn window(&self) -> Result<Option<usize>, ProviderError> {
            Ok(Some(self.window))
        }
        async fn digest(&self) -> Result<Option<String>, ProviderError> {
            self.digest_calls.fetch_add(1, Ordering::SeqCst);
            Ok(Some("a".repeat(64)))
        }
    }

    /// A source whose window is fine and whose digest is not 64 lowercase hex.
    struct WindowOkDigestMalformed {
        /// What the daemon "answers" for the digest.
        digest: String,
    }

    #[async_trait::async_trait]
    impl ProviderProbe for WindowOkDigestMalformed {
        async fn window(&self) -> Result<Option<usize>, ProviderError> {
            Ok(Some(128_000))
        }
        async fn digest(&self) -> Result<Option<String>, ProviderError> {
            Ok(Some(self.digest.clone()))
        }
    }

    /// A malformed digest is dropped and the WINDOW survives — the same trade `probe_models`
    /// makes, now through the same predicate rather than a second copy of it.
    ///
    /// REQ-R25 persists a digest only if it resolved **and** passed the format check. This path
    /// skipped the check, and because nothing re-verifies the cache, whatever the daemon said was
    /// kept forever and handed to `corroborate_by_digest`, which compares digests for equality —
    /// so two identically-garbled answers would have read as a lineage collision. Found by S2
    /// Loop 2 (Caspar).
    ///
    /// **Mutation-verified (B16):** drop the `validate_digest` wrapper in `window()` and the
    /// first case goes red, because the malformed value reaches the row.
    #[tokio::test]
    async fn a_malformed_digest_is_dropped_without_costing_the_window() {
        for bad in [
            "not-hex".to_owned(),
            "A".repeat(64), // uppercase: the rule is LOWERCASE hex
            "a".repeat(63), // one short of the exact length
            "a".repeat(65), // one over
            String::new(),
        ] {
            let cache = empty_cache();
            let probe = CachedProbe::new(
                Arc::clone(&cache),
                ENDPOINT.to_owned(),
                MODEL.to_owned(),
                Some(Arc::new(WindowOkDigestMalformed {
                    digest: bad.clone(),
                })),
            );

            assert_eq!(
                probe.window().await.expect("fails open"),
                Some(128_000),
                "a bad digest must not cost the window it arrived beside"
            );
            let row = cache
                .get(ENDPOINT, MODEL)
                .expect("cache read")
                .expect("the window was measured, so there IS a row");
            assert_eq!(
                row.digest, None,
                "but {bad:?} is not 64 lowercase hex and must not have been persisted"
            );
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

    /// An out-of-range window is a FAILED measurement here, exactly as it is at startup — and
    /// above all it must not reach the cache.
    ///
    /// `probe_models` degrades a window outside `[PROBE_WINDOW_MIN, PROBE_WINDOW_MAX]` to *not
    /// measured*. This path did not, and the asymmetry was not merely inconsistent: what this
    /// path accepts it **writes**, and the design re-verifies nothing, so a single absurd reading
    /// outlived the process that took it and answered magi-core's condition #6 as a fact from
    /// then on. Found by S2 Loop 2 (Caspar and Balthasar, independently).
    ///
    /// **Mutation-verified (B16):** remove the range guard in `window()` and both halves go red.
    #[tokio::test]
    async fn an_out_of_range_window_is_refused_and_never_cached() {
        for window in [PROBE_WINDOW_MIN - 1, PROBE_WINDOW_MAX + 1, 0] {
            let source = Arc::new(WindowOutOfRange {
                window,
                digest_calls: AtomicUsize::new(0),
            });
            let cache = empty_cache();
            let probe = CachedProbe::new(
                Arc::clone(&cache),
                ENDPOINT.to_owned(),
                MODEL.to_owned(),
                Some(Arc::clone(&source) as Arc<dyn ProviderProbe>),
            );

            assert_eq!(
                probe
                    .window()
                    .await
                    .expect("the probe fails open, never errors"),
                None,
                "{window} is outside the admissible range, so it is NOT a measurement"
            );
            assert!(
                cache.get(ENDPOINT, MODEL).expect("cache read").is_none(),
                "and it must leave NO row: the cache is never re-verified, so a value written \
                 here is inherited by every later run of this install"
            );
            assert_eq!(
                source.digest_calls.load(Ordering::SeqCst),
                0,
                "the reading is abandoned before the second trip, not after paying for it"
            );
        }
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
            "the miss must have REACHED the source; a ceiling assertion against a probe that \
             never measured passes for the wrong reason"
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
