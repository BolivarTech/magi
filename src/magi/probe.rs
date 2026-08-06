// Author: Julian Bolivar Version: 1.0.0 Date: 2026-08-04

//! Measurement of context windows and model digest, composed over `ProviderProbe` (REQ-A24).
//!
//! # By composition, not by migration
//!
//! `ProviderProbe` is a trait **separate** from `LlmProvider`: an `OllamaProvider` from magi-
//! core is built **only** to call `.window()` and `.digest()` on it, never to generate. magi-rs
//! still completes with its own `Provider` — D-A07 and R-A02 remain intact. Only `ollama` is
//! measurable (`ProviderKind::is_probeable`); `openai-compat` and `anthropic` offer no
//! introspection and degrade to [`Measurement::NotMeasurable`].
//!
//! # The body-size cap (REQ-A16b / SC-A16c) — satisfied BY COMPOSITION
//!
//! REQ-A16b requires that a probe response body be read under a cap that cuts off
//! **while reading**, not by verifying it against a buffer already fully allocated. This module
//! DOES NOT
//! implement that cap, and it is not an oversight: `OllamaProvider::window()`/`.digest()` make
//! their own HTTP request inside magi-core and return to this module a `Result<Option<T>,
//! ProviderError>` **already resolved** — the raw body never crosses the composition boundary,
//! so there is nothing magi-rs can cap without reimplementing the entire transport (which R-A02
//! forbids).
//!
//! magi-core already does it: `read_probe_body` caps at `MAX_SHOW_BODY_BYTES` (1 MiB)
//! **during** reading, not afterwards. The property is verified, not just read, by `tests/magi_
//! core_contract.rs::magi_core_rejects_an_endless_probe_body_instead_of_accumulating_it`, which
//! hits a real server with an endless body and confirms that the reader cuts by size instead of
//! exhausting memory or waiting for a timeout. If that dependency stopped capping, that test
//! would say so before a hostile endpoint in production.
//!
//! What is indeed this module's responsibility, and is implemented here, is **validating the
//! ALREADY-RESOLVED VALUES**: a window outside `[PROBE_WINDOW_MIN, PROBE_WINDOW_MAX]` or a
//! digest that is not exactly 64 lowercase hex characters degrades to *not measured*, never
//! being used as-is nor clipped to the range boundary.
//!
//! # `ProbeError` is not defined in this file
//!
//! The task header lists `ProbeError` as a symbol to define here, but no path in this design
//! needs an error type: `probe_for` returns [`ProbeSeat`] (not `Result`), `probe_models`
//! returns a `BTreeMap` (not `Result`), and `derive_warn_tokens` returns `Option<usize>` (not
//! `Result`) — the probe **fails open everywhere**, so there is no error channel to propagate.
//! Fabricating a type with no caller just to complete the list would have violated the rule of
//! not inventing symbols without a consumer; it is documented here instead of silently created.

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

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

use magi_core::providers::ollama::OllamaProvider;
use magi_core::rotation::ProviderProbe;

use crate::magi::endpoint::ResolvedEndpoint;
use crate::magi::kind::ProviderKind;
use crate::magi::{PROBE_TIMEOUT_SECS, PROBE_WINDOW_MAX, PROBE_WINDOW_MIN, WARN_WINDOW_FRACTION};
use crate::redact::{redact_foreign_error, SafeErrorText};

/// Exact length of a SHA-256 digest in hexadecimal (REQ-A16b).
///
/// SHA-256 produces 32 bytes; in hexadecimal that is EXACTLY 64 characters. It is not a design
/// choice: it is the size of a SHA-256 fingerprint, and magi-core validates by the same number
/// in `parse_tags_digest`. Documented instead of repeated as a literal (B4).
const DIGEST_HEX_LEN: usize = 64;

/// Result of measuring a model (REQ-A24c). **Three states, not two.**
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Measurement {
    /// Measured: window in tokens (within `[PROBE_WINDOW_MIN, PROBE_WINDOW_MAX]`) and digest if
    /// it could be resolved and validated.
    Measured {
        /// Context window, already validated against the range in REQ-A16b.
        window: usize,
        /// SHA256 of the manifest — 64 lowercase hex — or `None` if it could not be resolved or
        /// did not pass format validation.
        digest: Option<String>,
    },
    /// The endpoint offers no introspection (`openai-compat`, `anthropic`). Definitive and
    /// expected (SC-A24b) — **it is not a failure**.
    NotMeasurable,
    /// The endpoint DOES offer introspection but this time did not answer in time, returned
    /// something out of range, or the probe could not be built.
    ///
    /// It is the common case of the **first start** with a cold Ollama daemon. Without
    /// distinguishing it from [`Self::NotMeasurable`], an "unknown window" on the first run
    /// would read as
    /// *"this doesn't work"* when in reality it is *"it hasn't loaded yet"*.
    NotMeasuredThisTime,
}

/// What came out of trying to build a probe for an `(endpoint, model)`. **Three states, not
/// two**: a non-measurable `kind` and a probe that could not be built have different
/// consequences, and collapsing them would assert something false about the server's capability
/// when what failed was our configuration.
pub enum ProbeSeat {
    /// Ready to probe.
    Ready(Arc<dyn ProviderProbe>),
    /// The `kind` offers no probe. Definitive (SC-A24b) — **it is not a failure**.
    NotProbeable,
    /// The `kind` IS measurable but the probe could not be built (malformed URL, HTTP client).
    /// Fixable: it is a problem with our config, not a statement about the server. The text is
    /// already drafted — it can never contain the resolved credential of the endpoint that was
    /// being probed (B11).
    Unbuildable(SafeErrorText),
}

/// Probe factory — the injection seam that R-A04 requires (no real network in this module's
/// tests, except the one that verifies real construction against a test server).
///
/// `probe_models` does not build an `OllamaProvider` inside: it requests it through this trait,
/// so a test can substitute the probe with a deterministic double.
pub trait ProbeFactory: Send + Sync {
    /// Builds the probe for an `(endpoint, model)`.
    ///
    /// Takes `&ResolvedEndpoint`, not `&str`: it is the newtype whose only constructor is
    /// `EndpointTemplate::resolve`, so a `base_url` with placeholders left unreplaced cannot
    /// reach here by construction — resolution happens at startup, after opening the vault and
    /// before probing or building the trio.
    fn probe_for(&self, kind: ProviderKind, base_url: &ResolvedEndpoint, model: &str) -> ProbeSeat;
}

/// Production: `OllamaProvider` **only as a probe**, never for completions (D-A07).
///
/// The `base_url` is passed as-is, with its `/v1` if it has one: `OllamaProvider::new` accepts
/// both forms (with and without `/v1`) and normalizes internally — probe requests always go
/// against the ROOT of the daemon (`{root}/api/show`, `{root}/api/tags`), never under `/v1`.
/// Verified by the test `the_real_factory_probes_the_daemon_root_not_the_v1_prefix` in this
/// module (not an intra-doc link because it lives in `#[cfg(test)]`, outside the tree `cargo
/// doc` walks) against a real test server, not just read against magi-core's code.
pub struct OllamaProbeFactory;

impl ProbeFactory for OllamaProbeFactory {
    fn probe_for(&self, kind: ProviderKind, base_url: &ResolvedEndpoint, model: &str) -> ProbeSeat {
        // Two different answers, never collapsed into an `Option`: a non-measurable kind and a
        // malformed URL under a measurable kind have different causes and different remedies.
        if !kind.is_probeable() {
            return ProbeSeat::NotProbeable;
        }
        match OllamaProvider::new(base_url.as_str(), model) {
            Ok(provider) => ProbeSeat::Ready(Arc::new(provider)),
            // `redact_foreign_error`, not raw `e.to_string()`: `base_url` is the ALREADY-
            // RESOLVED URL (REQ-A16c), and an error from another crate that interpolated it
            // would leak the real credential to whoever ends up showing this reason (B11).
            Err(e) => ProbeSeat::Unbuildable(redact_foreign_error(&e)),
        }
    }
}

/// Accepts a digest only if it is EXACTLY 64 lowercase hexadecimal characters (REQ-A16b).
/// Anything else is discarded — the measured window survives just the same.
///
/// It does not byte-index anything: `str::bytes()` is a total iterator over the bytes of a
/// valid UTF-8 string, never panicking on a character boundary (unlike `&s[a..b]`).
fn validate_digest(raw: Option<String>) -> Option<String> {
    raw.filter(|d| {
        d.len() == DIGEST_HEX_LEN
            && d.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    })
}

/// Measures the indicated models, composed over [`ProviderProbe`] (REQ-A24).
///
/// **Fails open, without exception**: error, timeout, unexpected scheme, or an endpoint that
/// never
/// answers degrades to [`Measurement::NotMeasuredThisTime`]. The ceiling is **per probe**
/// (SC-A24k): with a shared deadline, a slow probe would consume the budget of the others and
/// leave them unmeasured without any of them having failed.
///
/// **Dedup by `(endpoint, model)`.** In the default case the trio inherits the root endpoint
/// and
/// may share a model with the main one, so probing the same thing four times quadruples the
/// startup cost for the same number four times. The key is the model under the received
/// `base_url` — the caller already fixes a single endpoint per call, so deduplicating by model
/// within that call is deduplicating by the full pair.
///
/// The returned map has one entry per **requested** model, not per probe emitted: duplicates in
/// `models` share the result of the single probe that was launched.
///
/// # Complexity
///
/// Two `O(n)` passes over `models` (deduplicate, and re-expand at the end), with no nested
/// pass: each unique model triggers at most two HTTP requests (`window`, `digest`), all
/// concurrent via [`futures::future::join_all`].
pub async fn probe_models(
    kind: ProviderKind,
    base_url: &ResolvedEndpoint,
    models: &[&str],
    factory: &dyn ProbeFactory,
) -> BTreeMap<String, Measurement> {
    if !kind.is_probeable() {
        return models
            .iter()
            .map(|m| ((*m).to_string(), Measurement::NotMeasurable))
            .collect();
    }

    let unique: BTreeSet<&str> = models.iter().copied().collect();
    let deadline = Duration::from_secs(PROBE_TIMEOUT_SECS);

    let futures = unique.into_iter().map(|model| async move {
        let probe = match factory.probe_for(kind, base_url, model) {
            ProbeSeat::Ready(p) => p,
            // Capability the endpoint does not offer: definitive, and NOT a failure (SC-A24b).
            ProbeSeat::NotProbeable => return (model.to_string(), Measurement::NotMeasurable),
            // Our config, not its capability: fixable, so *not measured THIS TIME*.
            ProbeSeat::Unbuildable(_) => {
                return (model.to_string(), Measurement::NotMeasuredThisTime)
            }
        };

        // TWO independent ceilings, not one shared between `window` and `digest`: they are two
        // distinct HTTP requests that are NOT worth the same. The window feeds
        // `input_warn_tokens`; the digest only decorates a notice. With a `timeout` wrapping
        // both, a slow digest would throw away a perfectly good window — the same "shared
        // deadline" error that SC-A24k forbids among probes, one level lower.
        let window = tokio::time::timeout(deadline, probe.window())
            .await
            .ok()
            .and_then(|r| r.ok().flatten());

        let value = match window {
            Some(w) if (PROBE_WINDOW_MIN..=PROBE_WINDOW_MAX).contains(&w) => {
                // If the window did not come through, the digest is not even requested: it
                // saves one request in the case that matters most, which is the slow or down
                // endpoint.
                let digest = tokio::time::timeout(deadline, probe.digest())
                    .await
                    .ok()
                    .and_then(|r| r.ok().flatten());
                Measurement::Measured {
                    window: w,
                    digest: validate_digest(digest),
                }
            }
            // Out of range degrades to NOT MEASURED, never to the extreme: a clipped value
            // would be used as if it were real, and *not measured* has a planned and auditable
            // path.
            _ => Measurement::NotMeasuredThisTime,
        };
        (model.to_string(), value)
    });

    let measured: BTreeMap<String, Measurement> = futures::future::join_all(futures)
        .await
        .into_iter()
        .collect();

    // Re-expand: each REQUESTED model receives its probe's result, shared if there were
    // duplicates. The caller sees one entry per requested model, not per emitted probe — and
    // the output `BTreeMap` dedups by construction, so requesting `[a, a, b]` yields `{a, b}`.
    models
        .iter()
        .map(|m| {
            (
                (*m).to_string(),
                measured
                    .get(*m)
                    .cloned()
                    .unwrap_or(Measurement::NotMeasuredThisTime),
            )
        })
        .collect()
}

/// **Minimum** window measured among the mages, WITHOUT applying the warning fraction
/// (REQ-A24b). A non-measurable mage is **omitted** from the minimum instead of lowering it —
/// if none are measurable, returns `None`.
///
/// Separated from [`derive_warn_tokens`] (Task 5.2) so that the stale-composition notice
/// (SC-A24i, `main.rs::stale_composition_notice`) can compare `max_query_bytes` against the RAW
/// window number: applying the warning fraction there as well would shift the threshold of THAT
/// notice without REQ-A24b asking for it — SC-A24i compares against the **measured** window,
/// not against an already-reduced threshold.
///
/// # Complexity
/// One `O(n)` pass over the mages, with no nesting.
#[must_use]
pub fn min_mage_window(mages: &BTreeMap<String, Measurement>) -> Option<usize> {
    mages
        .values()
        .filter_map(|m| match m {
            Measurement::Measured { window, .. } => Some(*window),
            Measurement::NotMeasurable | Measurement::NotMeasuredThisTime => None,
        })
        .min()
}

/// Derives `input_warn_tokens` from the **minimum** of the measured windows of the mages
/// (REQ-A24b).
///
/// **From the MAGES, not the main one**: `input_warn_tokens` governs the input received by the
/// three mages, and the main model does not receive that payload. With a main model of large
/// window and mages of smaller window, deriving it from the main one would give a threshold
/// that never fires — it is the CALLER's responsibility to pass in only the trio's table here,
/// never the one that includes the main model.
///
/// A non-measurable mage is **omitted** from the minimum instead of lowering it. If none are
/// measurable, returns `None` and the caller falls back to the next level (declared key, then
/// default).
#[must_use]
pub fn derive_warn_tokens(mages: &BTreeMap<String, Measurement>) -> Option<usize> {
    let min = min_mage_window(mages)?;
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    Some((min as f64 * WARN_WINDOW_FRACTION) as usize)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    use std::time::Instant;

    use async_trait::async_trait;
    use magi_core::error::{ExternalErrorKind, ProviderError};
    use zeroize::Zeroizing;

    use super::*;
    use crate::magi::endpoint::{EndpointTemplate, Scope};
    use crate::vault::{SecretEntry, SecretStore, VaultError};

    /// Syntactic endpoint shared by tests that do not depend on a concrete value: `StubProbes`
    /// never does I/O, so it is never actually contacted.
    const SYNTHETIC_BASE_URL: &str = "http://localhost:11434/v1";

    /// Vault that should never be queried: these tests' URLs carry no placeholders, so
    /// `EndpointTemplate::resolve` never gets to request any entry ("Absent" case of
    /// `locate_userinfo` — see `src/magi/endpoint.rs`).
    ///
    /// # Documented coverage exclusion (REQ-A00, task 5.1 review / F2)
    ///
    /// The five methods of this `impl` appear as UNCOVERED functions in `cargo llvm-cov`, and
    /// it is structural, not a gap: `EndpointTemplate::resolve` can only call
    /// `SecretStore::get`, and only in the `UserinfoLocation::Found` branch — when the URL
    /// carries a `[user]`/`[password]` placeholder left unreplaced. No fixture in this module
    /// uses a URL with placeholders (they all resolve through the `Absent` branch, which
    /// returns before touching the vault), so not even `get` runs in practice. `set`, `remove`,
    /// `list`, and `contains` are not invoked by any path in `resolve`, with or without
    /// placeholder: they exist only because `SecretStore` is the full trait and a double has to
    /// implement it in full.
    struct NoSecrets;

    impl SecretStore for NoSecrets {
        fn set(&mut self, _name: &str, _value: &str) -> Result<(), VaultError> {
            unreachable!("estos tests no escriben al vault")
        }
        fn get(&mut self, name: &str) -> Result<Zeroizing<String>, VaultError> {
            Err(VaultError::SecretNotFound(name.to_string()))
        }
        fn remove(&mut self, _name: &str) -> Result<(), VaultError> {
            unreachable!("estos tests no borran del vault")
        }
        fn list(&mut self) -> Result<Vec<SecretEntry>, VaultError> {
            Ok(Vec::new())
        }
        fn contains(&mut self, _name: &str) -> Result<bool, VaultError> {
            Ok(false)
        }
    }

    /// Parses and resolves a fixture `base_url`, without placeholders. The test fails (does not
    /// degrade) if the fixture is malformed: a test helper that silently degraded would hide a
    /// broken fixture behind a result that looks valid.
    fn resolved(raw: &str) -> ResolvedEndpoint {
        EndpointTemplate::parse(raw)
            .expect("fixture de test bien formada")
            .resolve(&mut NoSecrets, Scope::Root)
            .expect("fixture de test sin placeholders")
    }

    /// The shared endpoint for tests that do not exercise real I/O.
    fn test_endpoint() -> ResolvedEndpoint {
        resolved(SYNTHETIC_BASE_URL)
    }

    /// Configurable [`ProviderProbe`] double — never built directly by a test, only through
    /// [`StubProbes`].
    struct StubProbe {
        /// The name by which this `StubProbe` was requested, to record its timing.
        model: String,
        /// What `window()` returns, already "measured" by the double.
        window: Option<usize>,
        /// What `digest()` returns, already "measured" by the double.
        digest: Option<String>,
        /// Artificial delay before `window()` resolves.
        delay: Duration,
        /// If `true`, `window()` returns a REAL `ProviderError::External` instead of
        /// `Ok(self.window)` — different from a timeout: here the probe DID answer, and what it
        /// answered was a typed error (F1, task 5.1 review).
        window_fails: bool,
        /// Same for `digest()`, independent of `window_fails`: a failing digest must not bring
        /// down a window already measured successfully.
        digest_fails: bool,
        /// Where to record how long `window()` took to resolve — including cancellation.
        timings: Arc<Mutex<BTreeMap<String, Duration>>>,
    }

    /// Records in `Drop` how long the call to `window()` lived, whether it finished normally or
    /// `tokio::time::timeout` cancelled it when the ceiling expired — it is the only honest way
    /// to measure "the slow probe exhausted ITS full ceiling" (SC-A24k), because a cancellation
    /// never reaches the end of the body of the function that was cancelled.
    struct TimingGuard {
        /// The model under which to record the elapsed time.
        model: String,
        /// When it started, on tokio's clock — so it respects the paused clock of the tests.
        start: tokio::time::Instant,
        /// The same shared map exposed by [`StubProbes::elapsed_of`].
        timings: Arc<Mutex<BTreeMap<String, Duration>>>,
    }

    impl TimingGuard {
        /// Starts the stopwatch for `model`.
        fn new(model: String, timings: Arc<Mutex<BTreeMap<String, Duration>>>) -> Self {
            Self {
                model,
                start: tokio::time::Instant::now(),
                timings,
            }
        }
    }

    impl Drop for TimingGuard {
        fn drop(&mut self) {
            let elapsed = self.start.elapsed();
            if let Ok(mut map) = self.timings.lock() {
                map.insert(self.model.clone(), elapsed);
            }
        }
    }

    /// An already-drafted reason for `Unbuildable`, for `StubProbes::always_unbuildable`. The
    /// only constructor of `SafeErrorText` is `redact_foreign_error`, so a double that needs to
    /// produce one cannot simply wrap a `String`.
    fn unbuildable_reason() -> SafeErrorText {
        redact_foreign_error(&std::io::Error::other("stub: construcción rechazada"))
    }

    #[async_trait]
    impl ProviderProbe for StubProbe {
        async fn window(&self) -> Result<Option<usize>, ProviderError> {
            let _guard = TimingGuard::new(self.model.clone(), Arc::clone(&self.timings));
            if !self.delay.is_zero() {
                tokio::time::sleep(self.delay).await;
            }
            if self.window_fails {
                return Err(ProviderError::external(
                    "stub: synthetic window failure",
                    ExternalErrorKind::Network,
                ));
            }
            Ok(self.window)
        }

        async fn digest(&self) -> Result<Option<String>, ProviderError> {
            if self.digest_fails {
                return Err(ProviderError::external(
                    "stub: synthetic digest failure",
                    ExternalErrorKind::Network,
                ));
            }
            Ok(self.digest.clone())
        }
    }

    /// Factory of [`StubProbe`]s — the injection seam of R-A04: without it, every test of
    /// `probe_models` would need a real HTTP server.
    struct StubProbes {
        /// Window emitted by every probe that is not the "slow" one nor derives its own (mode
        /// `derive_per_model`).
        default_window: Option<usize>,
        /// Digest emitted by every probe in the same case as `default_window`.
        default_digest: Option<String>,
        /// The model with the artificial delay, and how much — if there is one.
        slow: Option<(String, Duration)>,
        /// Instead of a fixed value, derives a different window per model (dedup test).
        derive_per_model: bool,
        /// If present, `probe_for` returns THIS seat instead of building a `Ready` one — to
        /// exercise the `NotProbeable`/`Unbuildable` arms of `probe_models` exactly as it would
        /// see them with the real factory, without depending on a server.
        seat_override: Option<SeatOverride>,
        /// If `true`, every `Ready` probe this factory builds fails `window()` with a real
        /// `ProviderError` (F1, task 5.1 review) — different from `seat_override`, which never
        /// gets to build a `StubProbe`.
        window_error: bool,
        /// Same for `digest()`.
        digest_error: bool,
        /// How many times `probe_for` was called — one per UNIQUE requested model, never per
        /// duplicate (SC of the dedup).
        built: Arc<AtomicUsize>,
        /// How long `window()` took for each model, including cancellation by timeout.
        timings: Arc<Mutex<BTreeMap<String, Duration>>>,
    }

    /// Which non-`Ready` seat `probe_for` should return, when `StubProbes` is configured for
    /// that. It exists to exercise the arms of `probe_models` that the real factory produces
    /// (`ProbeSeat::NotProbeable`, `ProbeSeat::Unbuildable`) without depending on a server:
    /// `StubProbes::measuring`/`without_window`/`one_slow`/`counting` always return `Ready`, so
    /// none of those other doubles exercise this branch.
    #[derive(Clone, Copy)]
    enum SeatOverride {
        /// Forces `ProbeSeat::NotProbeable` — the case of a factory whose idea of "measurable"
        /// differs from that of the `kind` that `probe_models` already checked.
        NotProbeable,
        /// Forces `ProbeSeat::Unbuildable` — the case of a measurable URL that could not be
        /// turned into an HTTP client.
        Unbuildable,
    }

    impl StubProbes {
        /// Every probe measures the same window and the same digest.
        fn measuring(window: usize, digest: String) -> Self {
            Self {
                default_window: Some(window),
                default_digest: Some(digest),
                slow: None,
                derive_per_model: false,
                seat_override: None,
                window_error: false,
                digest_error: false,
                built: Arc::new(AtomicUsize::new(0)),
                timings: Arc::new(Mutex::new(BTreeMap::new())),
            }
        }

        /// Every probe answers `window: None` — the case of an `/api/show` without
        /// `*.context_length`.
        fn without_window() -> Self {
            Self {
                default_window: None,
                default_digest: None,
                slow: None,
                derive_per_model: false,
                seat_override: None,
                window_error: false,
                digest_error: false,
                built: Arc::new(AtomicUsize::new(0)),
                timings: Arc::new(Mutex::new(BTreeMap::new())),
            }
        }

        /// `slow_model` delays `delay` before resolving; the rest measure a fixed window
        /// immediately.
        fn one_slow(slow_model: &str, delay: Duration) -> Self {
            Self {
                default_window: Some(128_000),
                default_digest: None,
                slow: Some((slow_model.to_string(), delay)),
                derive_per_model: false,
                seat_override: None,
                window_error: false,
                digest_error: false,
                built: Arc::new(AtomicUsize::new(0)),
                timings: Arc::new(Mutex::new(BTreeMap::new())),
            }
        }

        /// Each model measures a DIFFERENT window, derived from its own name — to distinguish
        /// probes of different models without relying on an external counter.
        fn counting() -> Self {
            Self {
                default_window: None,
                default_digest: None,
                slow: None,
                derive_per_model: true,
                seat_override: None,
                window_error: false,
                digest_error: false,
                built: Arc::new(AtomicUsize::new(0)),
                timings: Arc::new(Mutex::new(BTreeMap::new())),
            }
        }

        /// `probe_for` returns `ProbeSeat::NotProbeable` for every requested model.
        fn always_not_probeable() -> Self {
            Self {
                default_window: None,
                default_digest: None,
                slow: None,
                derive_per_model: false,
                seat_override: Some(SeatOverride::NotProbeable),
                window_error: false,
                digest_error: false,
                built: Arc::new(AtomicUsize::new(0)),
                timings: Arc::new(Mutex::new(BTreeMap::new())),
            }
        }

        /// `probe_for` returns `ProbeSeat::Unbuildable` for every requested model.
        fn always_unbuildable() -> Self {
            Self {
                default_window: None,
                default_digest: None,
                slow: None,
                derive_per_model: false,
                seat_override: Some(SeatOverride::Unbuildable),
                window_error: false,
                digest_error: false,
                built: Arc::new(AtomicUsize::new(0)),
                timings: Arc::new(Mutex::new(BTreeMap::new())),
            }
        }

        /// Every probe builds a `Ready`, but `window()` returns a REAL
        /// `ProviderError::External` —not a timeout—. Until the task 5.1 review no double
        /// distinguished "there was no time" from "the provider answered that it cannot": the
        /// two collapse to the same [`Measurement::NotMeasuredThisTime`], but via different
        /// code paths inside `probe_models` (`tokio::time::timeout` expiring vs.
        /// `Result::ok().flatten()` discarding that internal `Err`), and only the first had
        /// coverage.
        fn erroring_window() -> Self {
            Self {
                default_window: None,
                default_digest: None,
                slow: None,
                derive_per_model: false,
                seat_override: None,
                window_error: true,
                digest_error: false,
                built: Arc::new(AtomicUsize::new(0)),
                timings: Arc::new(Mutex::new(BTreeMap::new())),
            }
        }

        /// `window()` measures `window` normally; `digest()` returns a real
        /// `ProviderError::External`. Different from [`Self::measuring`] with an invalid-format
        /// digest (already covered by `a_malformed_body_degrades_without_panicking`): here the
        /// provider FAILS the request, not responding with a value that does not pass
        /// `validate_digest`.
        fn erroring_digest(window: usize) -> Self {
            Self {
                default_window: Some(window),
                default_digest: None,
                slow: None,
                derive_per_model: false,
                seat_override: None,
                window_error: false,
                digest_error: true,
                built: Arc::new(AtomicUsize::new(0)),
                timings: Arc::new(Mutex::new(BTreeMap::new())),
            }
        }

        /// How many distinct probes were built — one per unique requested model.
        fn probes_built(&self) -> usize {
            self.built.load(Ordering::SeqCst)
        }

        /// How long the probe for `model` took to resolve (or be cancelled).
        ///
        /// # Panics
        ///
        /// If `model` was never requested — a test that calls this on a model that was not
        /// measured has a bug in the test itself, not something to degrade.
        ///
        /// # Documented coverage exclusion (REQ-A00, task 5.1 review / F2)
        ///
        /// The panic closure below appears as an UNCOVERED function in `cargo llvm-cov`, and it
        /// is deliberate: no test in this module calls `elapsed_of` on a model that was not
        /// measured, so the panic path is never taken. A test that did take it would be
        /// demonstrating a bug in the test itself, not in production code — there is nothing to
        /// exercise here.
        fn elapsed_of(&self, model: &str) -> Duration {
            *self
                .timings
                .lock()
                .expect("el lock del stub no se envenena en estos tests")
                .get(model)
                .unwrap_or_else(|| panic!("no se registró timing para {model}"))
        }
    }

    impl ProbeFactory for StubProbes {
        fn probe_for(
            &self,
            _kind: ProviderKind,
            _base_url: &ResolvedEndpoint,
            model: &str,
        ) -> ProbeSeat {
            self.built.fetch_add(1, Ordering::SeqCst);
            match self.seat_override {
                Some(SeatOverride::NotProbeable) => return ProbeSeat::NotProbeable,
                Some(SeatOverride::Unbuildable) => {
                    return ProbeSeat::Unbuildable(unbuildable_reason());
                }
                None => {}
            }
            let delay = self
                .slow
                .as_ref()
                .filter(|(slow_model, _)| slow_model == model)
                .map_or(Duration::ZERO, |(_, d)| *d);
            let (window, digest) = if self.derive_per_model {
                // Deterministic and different between models of different name: enough for
                // `assert_ne!` without needing a random generator in a test.
                let derived = PROBE_WINDOW_MIN + model.bytes().map(usize::from).sum::<usize>();
                (Some(derived), None)
            } else {
                (self.default_window, self.default_digest.clone())
            };
            ProbeSeat::Ready(Arc::new(StubProbe {
                model: model.to_string(),
                window,
                digest,
                delay,
                window_fails: self.window_error,
                digest_fails: self.digest_error,
                timings: Arc::clone(&self.timings),
            }))
        }
    }

    /// SC-A16b (edge, task 5.1 review / F1): 63 characters is ONE LESS than the valid minimum —
    /// the real edge of the validation, different from the 3-character string that
    /// `a_malformed_body_degrades_without_panicking` already covers (that one only tests "very
    /// short").
    #[test]
    fn validate_digest_rejects_sixty_three_hex_chars() {
        let short = "a".repeat(DIGEST_HEX_LEN - 1);
        assert_eq!(validate_digest(Some(short)), None);
    }

    /// SC-A16b (edge): 65 characters is ONE MORE than the valid maximum.
    #[test]
    fn validate_digest_rejects_sixty_five_hex_chars() {
        let long = "a".repeat(DIGEST_HEX_LEN + 1);
        assert_eq!(validate_digest(Some(long)), None);
    }

    /// SC-A16b: the contract explicitly requires lowercase (REQ-A16b) — uppercase hexadecimal,
    /// although it represents the same value, is rejected just like anything else that does not
    /// match byte for byte.
    #[test]
    fn validate_digest_rejects_uppercase_hex() {
        let upper = "A".repeat(DIGEST_HEX_LEN);
        assert_eq!(validate_digest(Some(upper)), None);
    }

    /// SC-A16b: the exact length is NOT enough on its own — a character outside `[0-9a-f]` at
    /// position 64 is also rejected. `'g'` is the first ASCII character after `'f'`, so it is
    /// the closest neighbor to the valid range.
    #[test]
    fn validate_digest_rejects_a_non_hex_character_at_the_exact_length() {
        let mut bad = "a".repeat(DIGEST_HEX_LEN - 1);
        bad.push('g');
        assert_eq!(
            bad.len(),
            DIGEST_HEX_LEN,
            "el test debe seguir probando el borde exacto"
        );
        assert_eq!(validate_digest(Some(bad)), None);
    }

    /// SC-A16b (happy, exact edge): exactly 64 lowercase hex passes as-is, without modifying
    /// the value — the success case bounded by the four rejections above.
    #[test]
    fn validate_digest_accepts_exactly_sixty_four_lowercase_hex() {
        let valid = "f".repeat(DIGEST_HEX_LEN);
        assert_eq!(validate_digest(Some(valid.clone())), Some(valid));
    }

    /// SC-A24 / SC-A24b: what is measurable is measured; not measurable is NOT a failure.
    #[tokio::test]
    async fn ollama_is_measured_and_the_others_are_not_a_failure() {
        let m = probe_models(
            ProviderKind::Ollama,
            &test_endpoint(),
            &["m"],
            &StubProbes::measuring(128_000, "a".repeat(64)),
        )
        .await;
        assert!(matches!(m["m"], Measurement::Measured { .. }));

        // No network: the non-measurable kind is resolved inside `probe_models` itself, before
        // touching the factory, so not even a socket is built.
        let m = probe_models(
            ProviderKind::Anthropic,
            &test_endpoint(),
            &["m"],
            &OllamaProbeFactory,
        )
        .await;
        assert!(
            matches!(m["m"], Measurement::NotMeasurable),
            "no es un fallo: es la capacidad que ese endpoint no ofrece"
        );
    }

    /// SC-A16b: out of range degrades to NOT MEASURED, never to the range boundary.
    ///
    /// `PROBE_WINDOW_MIN - 1` is the REAL edge of the validation — different from `1`, which
    /// only tests "very small" without touching the exact limit that
    /// `(PROBE_WINDOW_MIN..=PROBE_WINDOW_MAX)` evaluates.
    #[tokio::test]
    async fn an_out_of_range_window_degrades_instead_of_being_clamped() {
        for absurd in [
            1_usize,
            PROBE_WINDOW_MIN - 1,
            PROBE_WINDOW_MAX + 1,
            999_999_999_999,
        ] {
            let m = probe_models(
                ProviderKind::Ollama,
                &test_endpoint(),
                &["m"],
                &StubProbes::measuring(absurd, "a".repeat(64)),
            )
            .await;
            assert!(
                matches!(m["m"], Measurement::NotMeasuredThisTime),
                "recortar al extremo convertiría un dato basura en uno plausible (ventana {absurd})"
            );
        }
    }

    /// SC-A16b (inclusive edges): `PROBE_WINDOW_MIN` and `PROBE_WINDOW_MAX` themselves are
    /// ACCEPTED as-is — the range is `[MIN, MAX]`, not `(MIN, MAX)`, and the previous test only
    /// exercises the rejection side. Without this one, a `<` that should be `<=` (or vice
    /// versa) in `(PROBE_WINDOW_MIN..=PROBE_WINDOW_MAX).contains(&w)` would pass the suite
    /// anyway.
    #[tokio::test]
    async fn the_window_range_boundaries_are_accepted_inclusive() {
        for edge in [PROBE_WINDOW_MIN, PROBE_WINDOW_MAX] {
            let m = probe_models(
                ProviderKind::Ollama,
                &test_endpoint(),
                &["m"],
                &StubProbes::measuring(edge, "a".repeat(64)),
            )
            .await;
            assert!(
                matches!(m["m"], Measurement::Measured { window, .. } if window == edge),
                "el rango es cerrado: {edge} debe aceptarse sin degradar (borde de PROBE_WINDOW_MIN/MAX)"
            );
        }
    }

    /// SC-A24d: malformed response degrades, does not break.
    #[tokio::test]
    async fn a_malformed_body_degrades_without_panicking() {
        let m = probe_models(
            ProviderKind::Ollama,
            &test_endpoint(),
            &["m"],
            &StubProbes::without_window(),
        )
        .await;
        assert!(matches!(m["m"], Measurement::NotMeasuredThisTime));

        let m = probe_models(
            ProviderKind::Ollama,
            &test_endpoint(),
            &["m"],
            &StubProbes::measuring(128_000, "abc".to_string()),
        )
        .await;
        match &m["m"] {
            Measurement::Measured { digest, .. } => {
                assert!(
                    digest.is_none(),
                    "un digest que no es 64 hex se descarta, la ventana sobrevive"
                );
            }
            other => panic!("esperaba Measured, salió {other:?}"),
        }
    }

    /// SC-A24d (extension, task 5.1 review / F1): a REAL `ProviderError` in `window()` degrades
    /// exactly the same as a timeout. Until now `StubProbe` could only resolve successfully or
    /// delay until the `tokio::time::timeout` of `probe_models` expired; the arm where the
    /// probe DID answer and what it answered was a typed `Err` —`.and_then(|r|
    /// r.ok().flatten())` discarding that `Err`, not an `Elapsed`— had never been exercised.
    #[tokio::test]
    async fn a_genuine_provider_error_on_window_degrades_like_a_timeout() {
        let m = probe_models(
            ProviderKind::Ollama,
            &test_endpoint(),
            &["m"],
            &StubProbes::erroring_window(),
        )
        .await;
        assert!(
            matches!(m["m"], Measurement::NotMeasuredThisTime),
            "un ProviderError real en window() debe fallar abierto, igual que un timeout"
        );
    }

    /// SC-A24d (extension, task 5.1 review / F1): a REAL `ProviderError` in `digest()` does not
    /// bring down the window already measured successfully — same principle of "one out-of-
    /// range/broken field does not contaminate the other" that
    /// `a_malformed_body_degrades_without_panicking` already proves for an invalid-FORMAT
    /// digest, here applied to a digest that EXPLICITLY fails with a typed error.
    #[tokio::test]
    async fn a_genuine_provider_error_on_digest_leaves_the_window_intact() {
        let m = probe_models(
            ProviderKind::Ollama,
            &test_endpoint(),
            &["m"],
            &StubProbes::erroring_digest(128_000),
        )
        .await;
        match &m["m"] {
            Measurement::Measured { window, digest } => {
                assert_eq!(
                    *window, 128_000,
                    "la ventana medida con éxito debe sobrevivir"
                );
                assert!(
                    digest.is_none(),
                    "un digest que falla degrada solo, sin arrastrar la ventana"
                );
            }
            other => panic!("esperaba Measured, salió {other:?}"),
        }
    }

    /// SC-A24c / SC-A24k: ceiling PER PROBE — a slow one does not drag the others down.
    ///
    /// Runs with the tokio clock PAUSED: the real ceiling is several seconds, and a test that
    /// actually slept that long would be exactly the defect this project already diagnosed
    /// twice under load (`nextest` in parallel). `probe_models` does not `tokio::spawn`, so the
    /// four probes live in the same task and the paused clock's auto-advance unblocks all of
    /// them without blocking a single real thread.
    #[tokio::test(start_paused = true)]
    async fn a_slow_probe_does_not_starve_the_others() {
        let started = Instant::now();
        let stub = StubProbes::one_slow("a", Duration::from_secs(PROBE_TIMEOUT_SECS + 5));
        let m = probe_models(
            ProviderKind::Ollama,
            &test_endpoint(),
            &["a", "b", "c", "d"],
            &stub,
        )
        .await;

        assert!(
            started.elapsed() < Duration::from_secs(PROBE_TIMEOUT_SECS + 1),
            "corren en paralelo: el peor caso es UN techo, no cuatro"
        );
        assert!(matches!(m["a"], Measurement::NotMeasuredThisTime));
        assert_eq!(
            m.values()
                .filter(|v| matches!(v, Measurement::Measured { .. }))
                .count(),
            3,
            "con plazo compartido, la lenta habría dejado a las otras sin presupuesto"
        );
        // The ceiling is PER PROBE: the slow one consumed its whole own without cutting anyone
        // else's.
        assert!(
            stub.elapsed_of("a") >= Duration::from_secs(PROBE_TIMEOUT_SECS),
            "la sonda lenta debe agotar SU techo completo, no una fracción compartida"
        );
        for fast in ["b", "c", "d"] {
            assert!(
                stub.elapsed_of(fast) < Duration::from_secs(PROBE_TIMEOUT_SECS),
                "{fast} no debe haber esperado a la lenta"
            );
        }
    }

    /// Structural regression: `probe_models` handles `ProbeSeat::NotProbeable` returned BY THE
    /// FACTORY, not only the shortcut for `kind.is_probeable() == false` from the line above.
    /// With `StubProbes::measuring`/`without_window`/`one_slow`/`counting` the factory ALWAYS
    /// builds `Ready`, so none of those doubles exercise this arm — only a factory whose notion
    /// of "measurable" differs from that of the `kind` does, which is exactly what the real one
    /// would do if someday `is_probeable()` and `probe_for` drift out of sync.
    #[tokio::test]
    async fn a_seat_reported_not_probeable_mid_stream_is_not_a_failure() {
        let m = probe_models(
            ProviderKind::Ollama,
            &test_endpoint(),
            &["m"],
            &StubProbes::always_not_probeable(),
        )
        .await;
        assert!(matches!(m["m"], Measurement::NotMeasurable));
    }

    /// Same for `ProbeSeat::Unbuildable`: it is the path the REAL factory takes when the `kind`
    /// is measurable but the URL cannot build a client — fixable, so it degrades to *not
    /// measured this time*, not to *not measurable*.
    #[tokio::test]
    async fn an_unbuildable_seat_degrades_to_not_measured_this_time() {
        let m = probe_models(
            ProviderKind::Ollama,
            &test_endpoint(),
            &["m"],
            &StubProbes::always_unbuildable(),
        )
        .await;
        assert!(matches!(m["m"], Measurement::NotMeasuredThisTime));
    }

    /// SC-A24g: the derived threshold is a FRACTION of the window, never the window itself, so the size
    /// warning stays reachable on a large-window model.
    /// SC-A24j: the threshold comes from the MINIMUM of the mages, not the main one.
    #[test]
    fn the_warn_threshold_comes_from_the_minimum_mage_window() {
        let mages = BTreeMap::from([
            (
                "melchior".to_string(),
                Measurement::Measured {
                    window: 1_000_000,
                    digest: None,
                },
            ),
            (
                "balthasar".to_string(),
                Measurement::Measured {
                    window: 128_000,
                    digest: None,
                },
            ),
            ("caspar".to_string(), Measurement::NotMeasuredThisTime),
        ]);
        let derived = derive_warn_tokens(&mages).expect("hay al menos un mage medido");
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let expected = (128_000.0 * WARN_WINDOW_FRACTION) as usize;
        assert_eq!(
            derived, expected,
            "manda el primero que deja de aceptar el payload"
        );
        assert!(derived < 128_000, "una fracción, nunca la ventana entera");
    }

    /// SC-A24j (regression): a non-measurable mage is OMITTED from the minimum, not lowered.
    #[test]
    fn an_unmeasurable_mage_is_omitted_from_the_minimum() {
        let none = BTreeMap::from([("m".to_string(), Measurement::NotMeasuredThisTime)]);
        assert_eq!(
            derive_warn_tokens(&none),
            None,
            "sin ninguno medible se cae al nivel siguiente"
        );
    }

    /// Task 5.2 (SC-A24i): `min_mage_window` is the RAW number, without the warning fraction —
    /// different from `derive_warn_tokens`, which applies `WARN_WINDOW_FRACTION` to it. The
    /// stale-composition notice needs to compare against the MEASURED window, not against an
    /// already-reduced threshold.
    #[test]
    fn min_mage_window_returns_the_raw_minimum_unmeasurable_omitted() {
        let mages = BTreeMap::from([
            (
                "melchior".to_string(),
                Measurement::Measured {
                    window: 1_000_000,
                    digest: None,
                },
            ),
            (
                "balthasar".to_string(),
                Measurement::Measured {
                    window: 128_000,
                    digest: None,
                },
            ),
            ("caspar".to_string(), Measurement::NotMeasurable),
        ]);
        assert_eq!(
            min_mage_window(&mages),
            Some(128_000),
            "el mínimo CRUDO, sin fracción — distinto del umbral derivado"
        );
    }

    /// Task 5.2 (edge): no measurable mage ⇒ `None`, just like `derive_warn_tokens`.
    #[test]
    fn min_mage_window_is_none_when_nothing_is_measured() {
        let mages = BTreeMap::from([
            ("a".to_string(), Measurement::NotMeasurable),
            ("b".to_string(), Measurement::NotMeasuredThisTime),
        ]);
        assert_eq!(min_mage_window(&mages), None);
    }

    /// Dedup: four models requested, two distinct ⇒ TWO probes built, TWO entries in the
    /// returned map (the map dedups by key; requesting `[a, a, b, a]` yields `{a, b}` — there
    /// is no way it can produce four entries for three distinct names).
    #[tokio::test]
    async fn identical_endpoint_and_model_are_probed_once_and_shared() {
        let counting = StubProbes::counting();
        let m = probe_models(
            ProviderKind::Ollama,
            &test_endpoint(),
            &["a", "a", "b", "a"],
            &counting,
        )
        .await;

        assert_eq!(counting.probes_built(), 2, "solo dos modelos distintos");
        assert_eq!(m.len(), 2, "el mapa dedup-ea por clave");
        // The three entries of "a" share the result of the ONLY probe of "a", and that is
        // checked against the count above — comparing `m["a"] == m["a"]` would be tautological.
        assert_ne!(
            m["a"], m["b"],
            "sondas de modelos distintos dan resultados distintos"
        );
        assert!(matches!(m["a"], Measurement::Measured { .. }));
    }

    /// D-A07/R-A02 regression: the REAL factory probes the ROOT of the daemon (`/api/show`),
    /// never under the `/v1` prefix of completions.
    ///
    /// `StubProbes` covers the behavior of `probe_models` and NOT the real construction of the
    /// probe: if `OllamaProbeFactory` started hitting `/v1/api/show`, no other test in this
    /// module would see it. `ProviderProbe` does not expose the URL it uses internally (by
    /// design — see `magi_core::providers::provider_url`), so the only honest way to pin down
    /// this property is to exercise it against a real server: if the mock registered at
    /// `/api/show` is never hit, `mock.assert_async()` makes the test fail instead of letting a
    /// wrong URL pass in silence.
    #[tokio::test]
    async fn the_real_factory_probes_the_daemon_root_not_the_v1_prefix() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/api/show")
            .with_status(200)
            .with_body(r#"{"model_info":{"x.context_length":128000}}"#)
            .create_async()
            .await;

        let base = resolved(&format!("{}/v1", server.url()));
        let seat = OllamaProbeFactory.probe_for(ProviderKind::Ollama, &base, "m");
        let ProbeSeat::Ready(probe) = seat else {
            panic!("ollama es medible, tenía que producir una sonda lista");
        };

        let window = probe.window().await.expect("el mock responde 200");
        mock.assert_async().await;
        assert_eq!(
            window,
            Some(128_000),
            "si hubiera pegado en /v1/api/show, el mock de /api/show nunca se habría \
             golpeado y `mock.assert_async()` ya habría hecho fallar este test antes de \
             llegar acá"
        );

        assert!(
            matches!(
                OllamaProbeFactory.probe_for(ProviderKind::Anthropic, &base, "m"),
                ProbeSeat::NotProbeable
            ),
            "un kind no medible no produce sonda"
        );
    }

    /// B11: when `OllamaProvider::new` rejects the URL (here, by scheme — `ftp` is not
    /// `http`/`https`), the real factory reports `Unbuildable` with a reason already passed
    /// through `redact_foreign_error`, never raw `e.to_string()`. There is no credential to
    /// leak in THIS particular case (magi-core's scheme rejection only cites the scheme name,
    /// never the full URL — verified against `providers/provider_url.rs`), but the property
    /// that matters here is PLUMBING: that the error path goes through the redactor and not
    /// through a direct `.to_string()`, so a future magi-core error that DOES interpolate a URL
    /// with credentials is covered by construction, not by vigilance.
    #[test]
    fn the_real_factory_redacts_the_reason_when_construction_fails() {
        let bad = resolved("ftp://host/v1");
        match OllamaProbeFactory.probe_for(ProviderKind::Ollama, &bad, "m") {
            ProbeSeat::Unbuildable(reason) => {
                assert!(
                    !reason.as_str().is_empty(),
                    "una razón vacía no es diagnosticable"
                );
            }
            ProbeSeat::Ready(_) => panic!("un esquema ftp no debería construir un cliente"),
            ProbeSeat::NotProbeable => {
                panic!("ollama es medible, esto es un fallo de construcción, no de capacidad")
            }
        }
    }
}
