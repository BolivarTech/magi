// Author: Julian Bolivar
// Version: 0.17.0
// Date: 2026-08-27

//! Measurement of context windows and model digest, composed over `ProviderProbe` (REQ-A24).
//!
//! # By composition, not by migration
//!
//! `ProviderProbe` is a trait **separate** from `LlmProvider`: the `OllamaProvider` this module
//! builds exists **only** to call `.window()` and `.digest()` on it, never to generate. Only
//! `ollama` is measurable (`ProviderKind::is_probeable`); `openai-compat` and `anthropic` offer
//! no introspection and degrade to [`Measurement::NotMeasurable`].
//!
//! **Since REQ-R30 the trio's `ollama` seats ALSO complete through an `OllamaProvider`** (D-R12
//! reverted D-A07), so the type is no longer probe-only in this crate. The two roles still use
//! **separate instances**: this module builds its own for measuring, and `build_native_provider`
//! builds the seat's. Sharing one is possible and was left as a choice for whoever needs it —
//! but the constructors are not interchangeable, and that is the part to keep straight. Here
//! `new` is deliberate and safe, because every probe call is wrapped in its own
//! [`PROBE_TIMEOUT_SECS`] ceiling, so the client's own 300 s never governs. A **seat** must use
//! `with_timeout` with the derived value, or it breaks the derived scale in silence.
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
//! # Operational note: a measurement is only ever refreshed by a RESTART (MAGI S3 re-gate,
//! Balthasar)
//!
//! The probe runs once per process, at startup (REQ-A24), and its result is a **final state
//! for that process** — there is no periodic re-check and no lazy re-probe. If an operator
//! switches the daemon's model to one with a smaller context window while magi-rs keeps
//! running, `input_warn_tokens` stays derived from the STALE, larger window until the process
//! is restarted, which can silently disable the size warning exactly when a smaller window
//! makes it matter more. This is an accepted trade-off (re-probing would reintroduce the
//! unpredictable latency the short probe timeout exists to avoid), not an oversight — but it is
//! an operational fact worth stating outside this doc comment too: **a model swap on the daemon
//! requires restarting magi-rs to pick up the new window.**
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
use crate::magi::{
    PROBE_TIMEOUT_SECS, PROBE_WINDOW_MAX, PROBE_WINDOW_MIN, WARN_POOL_TOLERANCE,
    WARN_WINDOW_FRACTION,
};
use crate::notices::Notice;
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
    /// The `base_url` parameter's type is `&ResolvedEndpoint`, not `&str`: it is the newtype
    /// whose only constructor is `EndpointTemplate::resolve`, so a `base_url` with placeholders
    /// left unreplaced cannot reach here by construction — resolution happens at startup, after
    /// opening the vault and before probing or building the trio.
    fn probe_for(&self, kind: ProviderKind, base_url: &ResolvedEndpoint, model: &str) -> ProbeSeat;
}

/// Production: builds an `OllamaProvider` **to measure with**, never to complete with.
///
/// Since REQ-R30 the trio's `ollama` seats complete through their own `OllamaProvider`, built in
/// `build_native_provider` — a **separate instance**, not this one. `new` is correct *here* and
/// wrong *there*: every call this factory's result receives is wrapped in its own
/// [`PROBE_TIMEOUT_SECS`] ceiling, so the client's 300 s default never governs, whereas a seat
/// has nothing outside it to cut the request short.
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
///
/// **Public so the two measurement paths share ONE definition.** It was private, and the
/// per-consult path in `CachedProbe` therefore persisted whatever the daemon answered while this
/// one filtered it — the same asymmetry that let an out-of-range window through, one field over
/// (S2 Loop 2, Caspar). Duplicating the predicate there would have closed the symptom and kept
/// the shape that produced it: two rules for one question, free to drift apart again.
pub fn validate_digest(raw: Option<String>) -> Option<String> {
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

/// Derives `input_warn_tokens` from the trio, letting only in-band pool candidates lower it, and
/// reports the ones that fall outside the band (REQ-R21, D-R09).
///
/// # Why the base is the TRIO and not the whole pool
///
/// Deriving from every candidate looks conservative and is not. A small-window entry at the end of
/// the pool — the candidate *least* likely to ever run — would drag the threshold of every run
/// down with it, and the size warning would fire on practically every real consult. **A warning
/// that always sounds is ignored**, which is strictly worse than one that occasionally does not
/// fire.
///
/// And the scenario that derivation was meant to protect against **cannot do harm anyway**: by
/// magi-core's condition #6 a candidate is only selected when the prompt fits its window, so a
/// small one is never chosen for a payload it could not hold. The real protection is that
/// condition; this threshold only ever warns.
///
/// # What the band buys
///
/// A candidate within [`WARN_POOL_TOLERANCE`] of the base enters the minimum because doing so
/// costs nothing when it barely moves the number — a free, marginal calibration improvement,
/// **documented as that and not as a safety mechanism**. Everything below the band is reported
/// instead, with the model, its window and the base, so the operator can replace it with one of
/// comparable size.
///
/// **Every out-of-band entry is named in ONE message** (SC-R32): whoever assembled an unbalanced
/// pool usually has more than one, and finding them one start at a time costs a start each.
///
/// # Returns
///
/// The threshold and the notices owed. An empty pool, or one entirely in band, reports nothing.
///
/// # Examples
///
/// ```
/// # use magi_rs::magi::probe::derive_input_warn_tokens;
/// let (threshold, notices) = derive_input_warn_tokens(&[128_000, 128_000, 128_000], &[]);
/// assert!(threshold < 128_000);
/// assert!(notices.is_empty());
/// ```
#[must_use]
pub fn derive_input_warn_tokens(trio: &[usize], pool: &[(&str, usize)]) -> (usize, Vec<Notice>) {
    let base = trio.iter().copied().min().unwrap_or(0);
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let floor = (base as f64 * (1.0 - WARN_POOL_TOLERANCE)) as usize;

    let (in_band, out_of_band): (Vec<_>, Vec<_>) =
        pool.iter().partition(|(_, window)| *window >= floor);

    let effective = in_band
        .iter()
        .map(|(_, window)| *window)
        .chain(std::iter::once(base))
        .min()
        .unwrap_or(base);

    let mut notices = Vec::new();
    if !out_of_band.is_empty() {
        let listed = out_of_band
            .iter()
            .map(|(model, window)| format!("`{model}` ({window} tokens)"))
            .collect::<Vec<_>>()
            .join(", ");
        notices.push(Notice::resolution(format!(
            "notice: fallback candidates far below the trio's window are NOT lowering the size \
             warning threshold, which stays on the trio base of {base} tokens: {listed}. \
             Replace them with candidates of comparable window if you want the threshold to \
             reflect them."
        )));
    }

    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let threshold = (effective as f64 * WARN_WINDOW_FRACTION) as usize;
    (threshold, notices)
}

/// What is known about one model's context window, in **three** states rather than two (REQ-R26).
///
/// A reported window can be a **measurement** or an **assumption**, and a report that does not
/// distinguish them is stating a supposition as a fact. That is the entire reason this type exists
/// instead of an `Option<usize>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowState {
    /// The probe answered for this model.
    Measured(usize),
    /// Nothing was measured for this model, so the smallest window measured **in this run** is
    /// credited to it. Derived from the run, never a property of the model — which is why it is
    /// never persisted.
    Assumed(usize),
    /// Nothing measured and nothing to assume from. Falls back to today's non-strict behaviour
    /// rather than inventing a number, because an invented one would be trusted.
    Unknown,
}

/// The window assumed for a model that could not be measured: the **smallest** measured in this
/// run, or `None` when nothing was measured at all.
///
/// The smallest and not an average or the largest: an assumption that overstates a window invites
/// a rotation into a candidate that cannot hold the prompt, while one that understates it only
/// costs a warning.
///
/// # Examples
///
/// ```
/// # use std::collections::BTreeMap;
/// # use magi_rs::magi::probe::assumed_window;
/// assert_eq!(assumed_window(&BTreeMap::new()), None);
/// ```
#[must_use]
pub fn assumed_window(measured: &BTreeMap<String, Measurement>) -> Option<usize> {
    min_mage_window(measured)
}

/// What is known about `model`'s window, given everything measured in this run.
///
/// **This never removes anything.** The assumption informs; the filtering stays entirely with
/// magi-core, whose condition #6 compares a candidate's window against a `min_window_tokens`
/// computed **per consult** from the prompt (`orchestrator.rs:1199`) — a number that does not exist
/// at the moment the pool is declared, which is why a static elegibility check here is not merely
/// expensive but unbuildable.
///
/// It is also why `CachedProbe::window` answers `None` on a miss rather than the assumed value:
/// if the probe returned an assumption, magi-core could not tell it from a measurement, and
/// REQ-R11's fail-safe — *"pass strict only if a candidate has a MEASURED window"* — would become
/// unverifiable, since every candidate would then have a number.
#[must_use]
pub fn window_state(measured: &BTreeMap<String, Measurement>, model: &str) -> WindowState {
    match measured.get(model) {
        Some(Measurement::Measured { window, .. }) => WindowState::Measured(*window),
        _ => assumed_window(measured).map_or(WindowState::Unknown, WindowState::Assumed),
    }
}

/// Startup notices for the pool candidates running on an assumed window (REQ-R26/SC-R51).
///
/// # The two conditions, and the case they exist for
///
/// A notice is emitted **only** when something in this run was actually measured. Without that,
/// a cold daemon — the ordinary first run of any fresh install — leaves every candidate
/// unmeasured and the warning fires over the whole pool: transient, noisy, and with no action the
/// operator could take. *"Not measured this time"* is not *"not measurable"*, and the same
/// argument the threshold derivation uses applies here — **a warning that always sounds is
/// ignored**, which costs more than the one it would have delivered.
///
/// An endpoint where nothing is measurable is silent for the second reason: that candidate is not
/// *worse*, it is on a protocol that does not expose the datum at all.
#[must_use]
pub fn assumed_window_notices(
    measured: &BTreeMap<String, Measurement>,
    pool: &[crate::magi::rotation_config::FallbackEntry],
) -> Vec<Notice> {
    let Some(assumed) = assumed_window(measured) else {
        return Vec::new();
    };
    pool.iter()
        // Routed through [`window_state`] rather than an ad-hoc "not Measured" test: the three
        // states are the point of that type, and having the only place that reports them compute
        // the distinction some other way is how the two drift apart.
        .filter(|candidate| {
            // NOT measurable is not the same as not measured, and only the second earns a notice
            // (SC-R51). `window_state` credits both with the assumption, which REQ-R26's headline
            // supports — the assumption is informational and applies to any unmeasured candidate.
            // The NOTICE is narrower: telling an operator that a candidate on a protocol with no
            // introspection endpoint "has no measured window" reads as a defect in their setup,
            // when it is a property of the protocol and there is no action to take.
            //
            // Today this cannot fire — one orchestrator construction has one endpoint and one
            // `kind`, so either everything is measurable or nothing is, and nothing-measurable
            // already returned above with no assumption to make. It is written out anyway
            // because that is an emergent invariant nothing enforces: per-seat endpoints would
            // make the mixed case real, and the failure would be a confusing notice rather than
            // anything that breaks (S2 Loop 2, Balthasar).
            !matches!(
                measured.get(&candidate.model),
                Some(Measurement::NotMeasurable)
            ) && matches!(
                window_state(measured, &candidate.model),
                WindowState::Assumed(_)
            )
        })
        .map(|candidate| {
            Notice::info(format!(
                "notice: fallback candidate `{}` has no measured window; it is credited with \
                 {assumed} tokens, the smallest measured this run. It stays eligible, but a \
                 rotation into it may fail for a prompt larger than that.",
                candidate.model
            ))
        })
        .collect()
}

/// Whether `strict_context_guard` may actually be handed to magi-core, and the notice owed when it
/// may not (REQ-R11).
///
/// # The predicate is over CANDIDATES, which is why the pool is a parameter
///
/// magi-core applies the guard as condition **#6** of candidate selection: a candidate whose window
/// is unknown is admitted while the guard is off and **rejected** while it is on. So a `true` handed
/// down when nothing among the candidates was measured rejects **every** one of them — rotation
/// switches off entirely, and nothing fails visibly. That is not a guard; it is a declared pool that
/// was never eligible.
///
/// `measured` carries everything this run measured, **the trio included**, so a predicate over the
/// whole map is wrong in a specific and silent way: the trio measures, no candidate does, the
/// predicate answers `true`, and the pool dies. The question is only ever *"did any CANDIDATE get a
/// window?"*.
///
/// # It is NOT tied to the transport kind
///
/// Since magi-core 3.2.0 a probe is declared apart from the completions provider, so the `kind` no
/// longer predicts whether anything was measured. Deciding on it produces both symmetric errors:
/// denying the guard to an `openai-compat` that measured through a declared probe, and granting it
/// to an `ollama` whose daemon was cold — which is exactly the case the `false` default exists to
/// protect, and the first run of any fresh install.
///
/// This is the same predicate magi-core computes internally as `strict_guard_is_inert`
/// (`rotation.rs:839`), which is `pub(crate)` and therefore uncallable from here; it is recomputed
/// from this crate's own measurements rather than approximated.
///
/// # Returns
///
/// The value to hand magi-core, and `Some(notice)` **only** when a declared `true` had to be
/// overridden. A declared `false` reports nothing — the operator asked for the default and got it —
/// and neither does an empty pool: with no candidates there is nothing to protect and no surprise.
///
/// # Examples
///
/// ```
/// # use std::collections::BTreeMap;
/// # use magi_rs::magi::probe::effective_strict_guard;
/// let (effective, notice) = effective_strict_guard(true, &BTreeMap::new(), &[]);
/// assert!(!effective);
/// assert!(notice.is_none());
/// ```
#[must_use]
pub fn effective_strict_guard(
    declared: bool,
    measured: &BTreeMap<String, Measurement>,
    pool: &[crate::magi::rotation_config::FallbackEntry],
) -> (bool, Option<Notice>) {
    if !declared {
        return (false, None);
    }
    let any_candidate_measured = pool.iter().any(|candidate| {
        matches!(
            measured.get(&candidate.model),
            Some(Measurement::Measured { .. })
        )
    });
    if any_candidate_measured {
        return (true, None);
    }
    // An empty pool is "no rotation", a deliberate choice — the same case magi-core excludes from
    // its own inert-guard warning. Overriding it is not a surprise worth a line at startup.
    if pool.is_empty() {
        return (false, None);
    }
    (
        false,
        Some(Notice::resolution(
            "notice: `strict_context_guard = true` was declared but is NOT being applied, \
             because no candidate has a measured window. Applying it would reject every \
             fallback candidate and switch rotation off entirely. It takes effect on its own \
             once a measurement succeeds."
                .to_owned(),
        )),
    )
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

/// The Ollama command that fetches a model manifest, named by [`missing_model_notices`].
///
/// It is not a chosen value: it is the literal CLI verb that makes a declared tag resolvable on
/// the daemon, and it is the ONLY action that fixes the case the notice is about. A constant so
/// that the notice and any future caller quote the same string (B4).
const OLLAMA_PULL_COMMAND: &str = "ollama pull";

/// Names, at STARTUP, each configured model that this endpoint could not measure **while it was
/// measuring others** (REQ-R27, second half / SC-R35).
///
/// # Why the notice exists at all
///
/// A fallback tag that was never `ollama pull`ed is not detected for free: building the provider
/// does not verify that the model exists, and the probe [fails open][probe_models]. With no
/// mechanism, the first symptom would be a **degraded run** the day a mage actually falls over —
/// the latest and worst possible moment. The probe this module already pays for is the signal:
/// an endpoint that answered for other models and not for this one is not *"not measurable"*, it
/// is a model that is probably not there.
///
/// # Two conditions, and both are contract — without them the notice is noise
///
/// 1. **Only where measurement is POSSIBLE.** [`Measurement::NotMeasurable`] means the protocol
///    exposes no introspection (`openai-compat`, `anthropic`): there is nothing to conclude
///    about that model, so it is never named. Only [`Measurement::NotMeasuredThisTime`] — the
///    endpoint *can* answer and this time did not — qualifies.
/// 2. **Only if at least one model of the SAME endpoint was measured.** That is exactly what
///    separates *"this model is not there"* from *"this endpoint is not answering"*. Without it
///    a cold daemon fires the notice over the whole pool on everyone's first run: transient,
///    noisy, and with no action a user can take.
///
/// A model listed in `configured` that has no entry in `measured` was never probed, so nothing
/// was learned about it either — it is not named, for the same reason as condition 1.
///
/// # KNOWN FALSE-POSITIVE MODE — declared, not discovered in production
///
/// The two conditions treat the failure of an INDIVIDUAL probe as informative, and with a cold
/// cache that assumption weakens: magi-core's preflight fans out up to
/// `MAX_PREFLIGHT_CONCURRENCY = 4` probes (`rotation.rs:721`, with no knob) **plus** the three
/// mages, against an endpoint whose concurrency cap is **3**. Under that saturation one probe
/// can fail **by contention, not by absence**, while others on the same endpoint answer fine —
/// and the notice would point at a model that is perfectly present.
///
/// It is bounded noise, not a failure: it blocks nothing and the suggested action is harmless.
/// Two things bound it, and both are why the text below is worded the way it is:
/// **the text never states absence as a FACT** — it reports a measurement that did not happen
/// and offers `ollama pull <tag>` conditionally — and **a warm cache removes the condition
/// entirely**, because a run with no misses emits no probes at all. A first run against a cold
/// daemon is not to be judged by its notices.
///
/// # Contract
///
/// - Returns one [`Notice`] per DISTINCT qualifying model, in the order of `configured` — a pool
///   that repeats a model does not make the user read the same line twice.
/// - Tier is `Resolution`, never `Info`: a declared model that is probably not where it was
///   declared is *"the config resolved differently from what the file looks like"*, not routine
///   diagnostics.
/// - Never fails and never panics: like the rest of this module, it fails open.
///
/// # Complexity
///
/// One `O(m)` scan of `measured` for condition 2, then one `O(n log n)` pass over `configured`
/// (a `BTreeSet` of names already emitted), with no nested pass.
#[must_use]
pub fn missing_model_notices(
    measured: &BTreeMap<String, Measurement>,
    configured: &[String],
) -> Vec<Notice> {
    // CONDITION 2, evaluated first and for the whole table: without at least one model of this
    // same endpoint measured, a single probe's failure says nothing about that model — it says
    // the endpoint did not answer. Returning early here is what keeps a cold daemon from firing
    // the notice over the entire pool on a user's first run.
    if !measured
        .values()
        .any(|m| matches!(m, Measurement::Measured { .. }))
    {
        return Vec::new();
    }

    let mut already_named: BTreeSet<&str> = BTreeSet::new();
    configured
        .iter()
        // CONDITION 1: ONLY `NotMeasuredThisTime`. `NotMeasurable` is a protocol that exposes no
        // introspection and a missing entry was never probed — in both cases nothing was
        // observed about the model, and naming it would be inventing a conclusion.
        .filter(|model| {
            matches!(
                measured.get(model.as_str()),
                Some(Measurement::NotMeasuredThisTime)
            )
        })
        .filter(|model| already_named.insert(model.as_str()))
        .map(|model| {
            Notice::resolution(format!(
                "`{model}` could not be measured, on an endpoint that measured other models: \
                 if that tag is not there, `{OLLAMA_PULL_COMMAND} {model}` fixes it. A probe can \
                 also fail under load, so this is not proof that the model is absent."
            ))
        })
        .collect()
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
            unreachable!("these tests do not write to the vault")
        }
        fn get(&mut self, name: &str) -> Result<Zeroizing<String>, VaultError> {
            Err(VaultError::SecretNotFound(name.to_string()))
        }
        fn remove(&mut self, _name: &str) -> Result<(), VaultError> {
            unreachable!("these tests do not remove from the vault")
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
        EndpointTemplate::parse(raw, Scope::Root)
            .expect("well-formed test fixture")
            .resolve(&mut NoSecrets, Scope::Root)
            .expect("test fixture with no placeholders")
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
        redact_foreign_error(&std::io::Error::other("stub: construction rejected"))
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
                .expect("the stub's lock does not get poisoned in these tests")
                .get(model)
                .unwrap_or_else(|| panic!("no timing was recorded for {model}"))
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
            "the test must still be probing the exact edge"
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
            "not a failure: it is a capability that endpoint does not offer"
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
                "clamping to the extreme would turn garbage data into something \
                 plausible (window {absurd})"
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
                "the range is closed: {edge} must be accepted without degrading \
                 (PROBE_WINDOW_MIN/MAX edge)"
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
                    "a digest that is not 64 hex is discarded, the window survives"
                );
            }
            other => panic!("expected Measured, got {other:?}"),
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
            "a real ProviderError in window() must fail open, just like a timeout"
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
                    "the successfully measured window must survive"
                );
                assert!(
                    digest.is_none(),
                    "a failing digest degrades alone, without dragging down the window"
                );
            }
            other => panic!("expected Measured, got {other:?}"),
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
            "they run in parallel: the worst case is ONE ceiling, not four"
        );
        assert!(matches!(m["a"], Measurement::NotMeasuredThisTime));
        assert_eq!(
            m.values()
                .filter(|v| matches!(v, Measurement::Measured { .. }))
                .count(),
            3,
            "with a shared deadline, the slow one would have left the others with no budget"
        );
        // The ceiling is PER PROBE: the slow one consumed its whole own without cutting anyone
        // else's.
        assert!(
            stub.elapsed_of("a") >= Duration::from_secs(PROBE_TIMEOUT_SECS),
            "the slow probe must exhaust ITS OWN full ceiling, not a shared fraction"
        );
        for fast in ["b", "c", "d"] {
            assert!(
                stub.elapsed_of(fast) < Duration::from_secs(PROBE_TIMEOUT_SECS),
                "{fast} must not have waited for the slow one"
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
        let derived = derive_warn_tokens(&mages).expect("at least one mage was measured");
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let expected = (128_000.0 * WARN_WINDOW_FRACTION) as usize;
        assert_eq!(
            derived, expected,
            "the first to stop accepting the payload wins"
        );
        assert!(derived < 128_000, "a fraction, never the whole window");
    }

    /// SC-A24j (regression): a non-measurable mage is OMITTED from the minimum, not lowered.
    #[test]
    fn an_unmeasurable_mage_is_omitted_from_the_minimum() {
        let none = BTreeMap::from([("m".to_string(), Measurement::NotMeasuredThisTime)]);
        assert_eq!(
            derive_warn_tokens(&none),
            None,
            "with none measurable it falls to the next level"
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
            "the RAW minimum, without fraction — different from the derived threshold"
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

        assert_eq!(counting.probes_built(), 2, "only two distinct models");
        assert_eq!(m.len(), 2, "the map dedups by key");
        // The three entries of "a" share the result of the ONLY probe of "a", and that is
        // checked against the count above — comparing `m["a"] == m["a"]` would be tautological.
        assert_ne!(
            m["a"], m["b"],
            "probes of distinct models give distinct results"
        );
        assert!(matches!(m["a"], Measurement::Measured { .. }));
    }

    /// The REAL factory probes the ROOT of the daemon (`/api/show`), never under the `/v1`
    /// prefix of completions. The two families are siblings, so a probe that drifted under `/v1`
    /// would 404 — and, because probing fails **open**, would degrade to "not measured" without
    /// anyone seeing an error.
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
            panic!("ollama is measurable, it should have produced a ready probe");
        };

        let window = probe.window().await.expect("the mock responds 200");
        mock.assert_async().await;
        assert_eq!(
            window,
            Some(128_000),
            "if it had hit /v1/api/show, the /api/show mock would never have been hit \
             and `mock.assert_async()` would already have failed this test before \
             getting here"
        );

        assert!(
            matches!(
                OllamaProbeFactory.probe_for(ProviderKind::Anthropic, &base, "m"),
                ProbeSeat::NotProbeable
            ),
            "a non-measurable kind produces no probe"
        );
    }

    /// Builds a measurement table the way ONE endpoint's run would leave it: `Some(window)` is a
    /// model the endpoint measured, `None` a model it could have measured and did not
    /// ([`Measurement::NotMeasuredThisTime`]).
    ///
    /// It deliberately cannot express [`Measurement::NotMeasurable`]: that state belongs to a
    /// whole non-introspectable endpoint, and the one test that needs it MIXED with a measured
    /// model builds its map by hand, so that the mixture is visible at the call site instead of
    /// hidden behind a fixture flag.
    fn measurements(rows: &[(&str, Option<usize>)]) -> BTreeMap<String, Measurement> {
        rows.iter()
            .map(|(model, window)| {
                let value = window.map_or(Measurement::NotMeasuredThisTime, |w| {
                    Measurement::Measured {
                        window: w,
                        digest: None,
                    }
                });
                ((*model).to_string(), value)
            })
            .collect()
    }

    /// The models a run declares, in the shape `missing_model_notices` consumes.
    fn configured(models: &[&str]) -> Vec<String> {
        models.iter().map(|m| (*m).to_string()).collect()
    }

    /// SC-R35 (happy): the endpoint measured OTHERS but not this one ⇒ probably a missing tag.
    /// Named at startup together with the command that fixes it.
    #[test]
    fn a_model_that_alone_failed_on_a_responding_endpoint_is_named_at_startup() {
        let measured = measurements(&[("a", Some(128_000)), ("b", None)]);
        let notices = missing_model_notices(&measured, &configured(&["a", "b"]));
        assert_eq!(notices.len(), 1, "only the unmeasured model is named");
        assert!(
            notices[0].text.contains("ollama pull b"),
            "the notice must carry the command that fixes it: {}",
            notices[0].text
        );
        assert!(
            !notices[0].text.contains("ollama pull a"),
            "the model that WAS measured must not be named: {}",
            notices[0].text
        );
    }

    /// SC-R35 (second condition): where NOTHING could be measured there is NO such notice — that
    /// is an endpoint not answering, not a missing model. This is the condition that separates
    /// the two, and it is what keeps a cold first start from firing over the whole pool.
    #[test]
    fn nothing_measured_produces_no_missing_model_notice() {
        let measured = measurements(&[("a", None), ("b", None)]);
        assert!(
            missing_model_notices(&measured, &configured(&["a", "b"])).is_empty(),
            "an endpoint that measured nothing says nothing about any single model"
        );
    }

    /// SC-R35 (first condition): a model the endpoint CANNOT introspect is never named, even
    /// when another model on that same endpoint was measured.
    ///
    /// The mixture is what makes this test a guardian instead of a duplicate of the one above: a
    /// map of only [`Measurement::NotMeasurable`] would also be caught by condition 2 (nothing
    /// measured), so it would pass with condition 1 deleted. With one measured model present,
    /// condition 2 is satisfied and ONLY the *not measurable* vs *not measured this time*
    /// distinction can keep the notice silent.
    #[test]
    fn a_model_the_endpoint_cannot_introspect_is_never_named() {
        let measured = BTreeMap::from([
            (
                "a".to_string(),
                Measurement::Measured {
                    window: 128_000,
                    digest: None,
                },
            ),
            ("b".to_string(), Measurement::NotMeasurable),
        ]);
        assert!(
            missing_model_notices(&measured, &configured(&["a", "b"])).is_empty(),
            "no introspection is not a failure and concludes nothing about the model"
        );
    }

    /// SC-R35 (declared false-positive mode): under saturation a probe can fail by CONTENTION,
    /// not by absence, so the text must report a measurement that did not happen and offer the
    /// command conditionally — never assert that the model is gone.
    #[test]
    fn the_notice_reports_a_failed_measurement_not_a_confirmed_absence() {
        let measured = measurements(&[("a", Some(128_000)), ("b", None)]);
        let notices = missing_model_notices(&measured, &configured(&["a", "b"]));
        let text = &notices
            .first()
            .expect("the qualifying model must produce a notice")
            .text;
        assert!(
            text.contains("could not be measured"),
            "it must name the measurement failure, which is the only thing observed: {text}"
        );
        assert!(
            text.contains("if"),
            "the remedy must be offered conditionally, not as a diagnosis: {text}"
        );
        for absolute in [
            "is missing",
            "is not present",
            "is not installed",
            "does not exist",
        ] {
            assert!(
                !text.contains(absolute),
                "the notice must not state absence as a fact (found {absolute:?}): {text}"
            );
        }
    }

    /// SC-R35: a declared model that could not be measured on an endpoint that measured others
    /// is probably not there at all, and `ollama pull` is the action — so it reaches the
    /// screen (row 1 of task 3.1's classification table).
    #[test]
    fn the_missing_model_notice_reaches_the_screen() {
        let measured = measurements(&[("a", Some(128_000)), ("b", None)]);
        let notices = missing_model_notices(&measured, &configured(&["a", "b"]));
        assert_eq!(
            notices
                .first()
                .expect("the qualifying model must produce a notice")
                .level,
            tracing::Level::WARN
        );
    }

    /// **Row 4 of task 3.1's classification table** — a normal startup state is diagnostic,
    /// not action, so it goes only to the file.
    ///
    /// Nothing became unavailable here and nothing performs worse: the threshold stays on the
    /// trio base because the tolerance band says it should, and the notice exists to explain
    /// why the operator's small candidates did not move it. Its sibling for the same
    /// candidates — `assumed_window_notices` — has always been `info` for the same reason.
    #[test]
    fn a_normal_startup_diagnostic_stays_off_the_screen() {
        let (_, notices) = derive_input_warn_tokens(&[128_000], &[("tiny", 8_000)]);
        assert_eq!(
            notices
                .first()
                .expect("an out-of-band candidate must produce a notice")
                .level,
            tracing::Level::INFO,
            "explaining a threshold that resolved as designed is not actionable"
        );
    }

    /// Edge: a configured model with no entry in the table was never probed, so nothing was
    /// learned about it — it is not named, for the same reason a non-introspectable one is not.
    #[test]
    fn a_configured_model_that_was_never_probed_is_not_named() {
        let measured = measurements(&[("a", Some(128_000))]);
        assert!(
            missing_model_notices(&measured, &configured(&["a", "ghost"])).is_empty(),
            "a model absent from the table was never measured NOR failed to measure"
        );
    }

    /// Edge: a pool that declares the same model twice produces ONE line, not two — the user
    /// gains nothing from reading the same remedy again.
    #[test]
    fn a_model_repeated_in_the_configured_pool_is_named_once() {
        let measured = measurements(&[("a", Some(128_000)), ("b", None)]);
        let notices = missing_model_notices(&measured, &configured(&["b", "a", "b"]));
        assert_eq!(notices.len(), 1, "one line per distinct model: {notices:?}");
    }

    /// Edge (B13): empty inputs neither panic nor invent a notice.
    #[test]
    fn empty_inputs_produce_no_notice() {
        assert!(missing_model_notices(&BTreeMap::new(), &[]).is_empty());
        assert!(missing_model_notices(&BTreeMap::new(), &configured(&["a"])).is_empty());
        assert!(missing_model_notices(&measurements(&[("a", Some(128_000))]), &[]).is_empty());
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
                    "an empty reason is not diagnosable"
                );
            }
            ProbeSeat::Ready(_) => panic!("an ftp scheme should not build a client"),
            ProbeSeat::NotProbeable => {
                panic!("ollama is measurable, this is a construction failure, not a capability one")
            }
        }
    }

    // ---------------------------------------------------------------------------------------
    // `effective_strict_guard` — REQ-R11, the fail-safe (Task 6.4).

    // `measurements` already exists above in this module (B3): reused, not re-declared.

    /// A pool from `(model, lineage)` pairs.
    ///
    /// Uses the real [`FallbackEntry`], not a stand-in: the predicate under test reads `model`
    /// off it, and a local shape would let the two drift.
    fn pool_of(entries: &[(&str, &str)]) -> Vec<crate::magi::rotation_config::FallbackEntry> {
        entries
            .iter()
            .map(
                |(model, lineage)| crate::magi::rotation_config::FallbackEntry {
                    model: (*model).to_owned(),
                    lineage: crate::magi::lineage::Lineage::parse(lineage).expect("valid lineage"),
                },
            )
            .collect()
    }

    /// SC-R23: a declared `true` with NO measured window is NOT passed down, a notice names THAT
    /// reason, and rotation keeps working — never the silent shutdown of every candidate.
    #[test]
    fn a_declared_guard_with_nothing_measured_is_not_passed_and_is_announced() {
        let (effective, notice) = effective_strict_guard(
            true,
            &measurements(&[("m2", None)]),
            &pool_of(&[("m2", "zhipu")]),
        );
        assert!(
            !effective,
            "passing true here would make EVERY candidate fail condition #6: rotation dies whole"
        );
        let n = notice.expect("a declared true that is not applied must be announced");
        assert!(
            n.text.contains("measured window"),
            "the notice must name the REAL reason: {}",
            n.text
        );
        assert!(
            !n.text.contains("kind"),
            "not a derived reason like the transport kind: {}",
            n.text
        );
    }

    /// SC-R38: the criterion is MEASUREMENT, not transport. Tying it to the declared `kind`
    /// produces both symmetric errors — denying the guard to an `openai-compat` that did measure
    /// through a declared probe, and granting it to a cold `ollama` that measured nothing.
    #[test]
    fn the_criterion_is_measurement_not_the_declared_kind() {
        let pool = pool_of(&[("m2", "zhipu")]);
        let (warm, _) =
            effective_strict_guard(true, &measurements(&[("m2", Some(128_000))]), &pool);
        assert!(warm, "a measured candidate must not be denied the guard");
        let (cold, _) = effective_strict_guard(true, &measurements(&[("m2", None)]), &pool);
        assert!(!cold, "a cold candidate must not be granted it");
    }

    /// THE CASE THAT MAKES THE `pool` PARAMETER NECESSARY, and it is silent without it: the TRIO
    /// measured and NO CANDIDATE did.
    ///
    /// `measured` carries everything measured, trio included, so a predicate over the whole map
    /// answers `true` here — and passing `strict = true` makes every candidate fail condition #6.
    /// Rotation dies entirely, with nothing failing visibly. The predicate is over CANDIDATES.
    #[test]
    fn a_measured_trio_with_no_measured_candidate_does_not_enable_the_guard() {
        let measured =
            measurements(&[("melchior-model", Some(262_144)), ("candidate-model", None)]);
        let (effective, notice) =
            effective_strict_guard(true, &measured, &pool_of(&[("candidate-model", "zhipu")]));
        assert!(
            !effective,
            "the predicate is over CANDIDATES, not over everything that was measured"
        );
        assert!(notice.is_some(), "and the override is announced");
    }

    /// SC-R15: the cold start takes candidates away from nobody, under EITHER declared value.
    #[test]
    fn a_cold_start_never_removes_candidates_regardless_of_the_declared_guard() {
        let pool = pool_of(&[("m2", "zhipu"), ("m3", "minimax")]);
        for declared in [false, true] {
            let (effective, _) = effective_strict_guard(
                declared,
                &measurements(&[("m2", None), ("m3", None)]),
                &pool,
            );
            assert!(
                !effective,
                "declared={declared}: a cold start must never end up filtering candidates"
            );
        }
    }

    /// A declared `false` is passed through untouched and says nothing: the operator asked for the
    /// default and got it, so there is no resolution to announce.
    #[test]
    fn a_declared_false_is_never_announced() {
        let pool = pool_of(&[("m2", "zhipu")]);
        let (effective, notice) =
            effective_strict_guard(false, &measurements(&[("m2", Some(128_000))]), &pool);
        assert!(!effective);
        assert!(
            notice.is_none(),
            "nothing was overridden, so there is nothing to report"
        );
    }

    /// An empty pool is "no rotation", a choice — and there is nothing for the guard to filter,
    /// so a declared `true` is still not passed, and silently: it is not an override of intent.
    #[test]
    fn an_empty_pool_disables_the_guard_without_a_notice() {
        let (effective, notice) = effective_strict_guard(true, &measurements(&[]), &[]);
        assert!(!effective);
        assert!(
            notice.is_none(),
            "with no pool there is no candidate to protect and no surprise to report"
        );
    }

    // ---------------------------------------------------------------------------------------
    // The assumed window as DIAGNOSTIC — REQ-R26 / D-R13 (Task 6.5).

    /// SC-R29: the smallest MEASURED window is assumed for a candidate that could not be
    /// measured, and the state says **assumed**, not measured. Reporting a supposition as a fact
    /// is worse than reporting nothing, which is the whole reason this distinction exists.
    #[test]
    fn an_unmeasurable_candidate_is_marked_assumed_not_measured() {
        let measured = measurements(&[("a", Some(128_000)), ("b", Some(32_000)), ("c", None)]);
        assert_eq!(window_state(&measured, "a"), WindowState::Measured(128_000));
        assert_eq!(
            window_state(&measured, "c"),
            WindowState::Assumed(32_000),
            "the assumption is the SMALLEST measured window, not the largest or an average"
        );
    }

    /// With NOTHING measured there is no assumption to make: the state is `Unknown` and the
    /// candidate falls back to today's non-strict behaviour, rather than being handed an invented
    /// number that would then be trusted.
    #[test]
    fn with_nothing_measured_there_is_no_assumption() {
        assert_eq!(
            window_state(&measurements(&[("c", None)]), "c"),
            WindowState::Unknown
        );
        assert_eq!(assumed_window(&measurements(&[("c", None)])), None);
    }

    /// A model ABSENT from the map is treated exactly like one that was probed and did not
    /// answer: assumed, not unknown.
    ///
    /// This corrects the expectation this test was first written with. The distinction between
    /// "never probed" and "probed without an answer" carries **no information for the consumer**:
    /// in both cases nothing is known about the model, and the honest credit is still the run's
    /// smallest measured window. `Unknown` is reserved for the case where there is nothing to
    /// assume FROM — which is a statement about the run, not about one model.
    #[test]
    fn a_model_absent_from_the_map_is_assumed_like_one_that_did_not_answer() {
        let measured = measurements(&[("a", Some(128_000))]);
        assert_eq!(
            window_state(&measured, "absent"),
            WindowState::Assumed(128_000)
        );
        // And `Unknown` really is about the run having measured nothing at all.
        assert_eq!(
            window_state(&measurements(&[("a", None)]), "absent"),
            WindowState::Unknown
        );
    }

    /// The notice names the candidate, its assumed window and the CONSEQUENCE, because a
    /// diagnostic the operator cannot act on is noise.
    #[test]
    fn the_assumed_window_notice_is_actionable() {
        // Distinctive names: an assertion that a message contains "c" cannot fail, because "c"
        // also occurs in "candidate" and "credited".
        let measured = measurements(&[("seat-model", Some(128_000)), ("unmeasured-cand", None)]);
        let notices = assumed_window_notices(&measured, &pool_of(&[("unmeasured-cand", "zhipu")]));
        let text = notices
            .first()
            .expect("an assumed candidate must be reported")
            .text
            .clone();
        assert!(
            text.contains("unmeasured-cand"),
            "it must name the candidate: {text}"
        );
        assert!(
            text.contains("128000") || text.contains("128,000"),
            "and the window it is being credited with: {text}"
        );
        assert!(
            text.contains("rotation"),
            "and what it means for rotation: {text}"
        );
    }

    /// REQ-R27's two conditions, and the case they exist for: a COLD START where nothing was
    /// measured must produce NO notice.
    ///
    /// Without them the warning fires over the entire pool on everyone's very first run —
    /// transient, noisy, and with no action the operator could take. "Not measured this time" is
    /// not "not measurable", and a warning that always sounds is ignored.
    #[test]
    fn a_cold_start_produces_no_assumed_window_notice() {
        let cold = measurements(&[("a", None), ("c", None)]);
        assert!(
            assumed_window_notices(&cold, &pool_of(&[("c", "zhipu")])).is_empty(),
            "a cold start has nothing to compare against and nothing to advise"
        );
    }

    /// Nor does an endpoint where nothing is MEASURABLE: that candidate is not "worse", it is on
    /// a protocol that does not expose the datum at all.
    #[test]
    fn an_unmeasurable_endpoint_produces_no_assumed_window_notice() {
        let mut unmeasurable = BTreeMap::new();
        unmeasurable.insert("a".to_owned(), Measurement::NotMeasurable);
        unmeasurable.insert("c".to_owned(), Measurement::NotMeasurable);
        assert!(assumed_window_notices(&unmeasurable, &pool_of(&[("c", "zhipu")])).is_empty());
    }

    /// A fully measured pool has nothing to assume and says nothing.
    #[test]
    fn a_fully_measured_pool_is_silent() {
        let measured = measurements(&[("a", Some(128_000)), ("c", Some(64_000))]);
        assert!(assumed_window_notices(&measured, &pool_of(&[("c", "zhipu")])).is_empty());
    }

    // ---------------------------------------------------------------------------------------
    // The warn threshold against the pool — REQ-R21 / D-R09 (Task 6.6).

    /// SC-R17: a pool entry more than [`WARN_POOL_TOLERANCE`] below the trio's base does NOT drag
    /// the threshold down — it is REPORTED.
    ///
    /// Deriving from the whole pool would let an 8 K entry at the END of the list — the LEAST
    /// likely candidate to ever run — pull every run's threshold down to ~6 K, firing the warning
    /// on practically every real consult. That is not conservative, it **destroys the signal**:
    /// a warning that always sounds is ignored, and then the one time it mattered is ignored too.
    #[test]
    fn an_out_of_band_pool_entry_is_reported_and_does_not_move_the_threshold() {
        let (threshold, notices) =
            derive_input_warn_tokens(&[128_000, 128_000, 128_000], &[("weak", 8_000)]);
        assert_eq!(
            threshold,
            (128_000f64 * WARN_WINDOW_FRACTION) as usize,
            "the base stays the trio's"
        );
        assert_eq!(notices.len(), 1);
        let t = &notices[0].text;
        assert!(t.contains("weak"), "the warning must name the model: {t}");
        assert!(
            t.contains("8000") && t.contains("128000"),
            "and both its window and the trio base, or it is not actionable: {t}"
        );
    }

    /// An entry INSIDE the band does enter the minimum, with no warning — a free, marginal
    /// calibration improvement. Documented as exactly that and NOT as a safety mechanism: the
    /// protection against a too-small candidate is magi-core's condition #6, not this threshold.
    #[test]
    fn an_in_band_pool_entry_enters_the_minimum_without_a_warning() {
        let (threshold, notices) =
            derive_input_warn_tokens(&[128_000, 128_000, 128_000], &[("close", 120_000)]);
        assert_eq!(threshold, (120_000f64 * WARN_WINDOW_FRACTION) as usize);
        assert!(notices.is_empty());
    }

    /// SC-R32: ALL out-of-band entries in ONE message. Whoever built an unbalanced pool probably
    /// has more than one, and discovering them one at a time costs a start each.
    #[test]
    fn every_out_of_band_entry_is_named_in_a_single_message() {
        let (_, notices) = derive_input_warn_tokens(&[128_000; 3], &[("a", 8_000), ("b", 16_000)]);
        assert_eq!(notices.len(), 1, "one message, not one per entry");
        let t = &notices[0].text;
        assert!(t.contains("a") && t.contains("b"), "both named: {t}");
        assert!(
            t.contains("8000") && t.contains("16000"),
            "with both windows: {t}"
        );
    }

    /// The band's edge belongs to the band: exactly `base * (1 - tolerance)` is IN, so a boundary
    /// value does not silently flip behaviour depending on floating-point luck.
    #[test]
    fn the_band_edge_is_inside_the_band() {
        let base = 100_000usize;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let edge = (base as f64 * (1.0 - WARN_POOL_TOLERANCE)) as usize;
        let (threshold, notices) = derive_input_warn_tokens(&[base, base, base], &[("edge", edge)]);
        assert_eq!(threshold, (edge as f64 * WARN_WINDOW_FRACTION) as usize);
        assert!(
            notices.is_empty(),
            "the edge is in band, so nothing to report"
        );
    }

    /// An empty pool changes nothing and says nothing.
    #[test]
    fn an_empty_pool_leaves_the_trio_base_untouched() {
        let (threshold, notices) = derive_input_warn_tokens(&[64_000, 128_000, 128_000], &[]);
        assert_eq!(
            threshold,
            (64_000f64 * WARN_WINDOW_FRACTION) as usize,
            "the base is the trio MINIMUM"
        );
        assert!(notices.is_empty());
    }
}
