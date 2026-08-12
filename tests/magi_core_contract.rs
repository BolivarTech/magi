// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-02

//! Guardian of the `magi-core 3.2.0` API surface (Task 0.0, MS2 Phase 0; extended in MS3 Phase 4).
//!
//! # What it tests, and what it does NOT
//!
//! **It does not test behavior.** It tests that every magi-core symbol MS2 consumes **exists
//! and types the way the plan assumes it does**. If magi-core renames it, changes its arity,
//! its argument order, or a field's type, this **fails to compile** — which is exactly the
//! wanted outcome, and in Phase 0 instead of Phase 4.
//!
//! # Why it exists
//!
//! The MS2 TDD plan assumed an API surface nobody had verified, and the first read of the
//! crate found **five** false assumptions in one pass: `with_client` does not exist on any
//! provider, `OllamaProvider` fixed a 300 s client timeout with no override, `RetryConfig` is
//! `#[non_exhaustive]`, `ClaudeProvider` takes `api_key` **first**, and `Mode` has no parsing
//! method at all. Five failures in a single pass is the measure of how much surface there is.
//!
//! **The second of those five stopped being true in 3.2.0**, which is the whole argument for this
//! file: `OllamaProvider::with_timeout` bounds both clients the type builds, so the impossibility
//! D-A07 rested on is gone and MS3 reverted it (REQ-R30). A reading of the crate goes stale in a
//! patch release — this one took **one day** — and the compiler does not.
//!
//! **The reading does not replace this file, it justifies it**: a reading goes stale the
//! moment magi-core publishes a new version; the compiler does not.
//!
//! # Relationship with `examples/ms2_contracts.rs`
//!
//! These are two files with two different lifetimes. The *example* cross-checks magi-rs's
//! **internal** contracts against each other and **gets deleted** when Phase 6 closes, once
//! the real implementation replaces it. This test covers the boundary with the **external
//! crate** and **outlives the milestone**: it is what makes a bump to magi-core 3.2.0 break
//! the suite instead of silently drifting.
//!
//! # How to read a failure here
//!
//! A compile error in this file is **not fixed by adjusting the test**. Look up the real name
//! in the crate, fix **every** occurrence in the plan, and record the difference in
//! `dev-docs/MS2-DECISIONS.md` with a date. If the symbol does not exist in any form — as
//! happened with `MagiReport::window_rejected` — it does **not** get invented: it is logged
//! as a missing capability and the requirement that depended on it is reworked around what
//! actually exists.

use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use magi_core::orchestrator::{MagiBuilder, MagiConfig as CoreMagiConfig};
use magi_core::provider::{LlmProvider, RetryConfig, RetryProvider};
use magi_core::providers::claude::ClaudeProvider;
use magi_core::providers::ollama::OllamaProvider;
use magi_core::providers::openai_compat::OpenAiCompatibleProvider;
use magi_core::reporting::{ExtractionFailure, InputSize, MagiReport};
use magi_core::rotation::{AgentRotation, FallbackPool, Lineage, ProviderProbe, RotationEvent};
use magi_core::schema::{AgentName, AgentOutput, Mode};
use magi_core::verdict_markers::{VERDICT_CLOSE, VERDICT_OPEN};
use magi_rs::magi::report_anchors::{CONTRACTUAL_ANCHORS, SECTION_ANCHORS};
use magi_rs::magi::PROBE_TIMEOUT_SECS;

/// Shared doubles for the integration tests (Task 0.7). Declared here because this is their
/// first consumer: a module under `tests/` that nobody declares **is not a build target**, so
/// neither `cargo check` nor `clippy --all-targets` would compile it.
mod support;

/// Syntactic endpoint for constructing providers. **Never contacted**: this file does no I/O,
/// only type-checking, and a provider is constructed without opening any connection.
const SYNTHETIC_BASE_URL: &str = "http://127.0.0.1:11434/v1";

/// Synthetic model, of the same nature as [`SYNTHETIC_BASE_URL`].
const SYNTHETIC_MODEL: &str = "guardian-model";

/// Shape of [`MagiReport`] that Phase 6 consumes, **with annotated types**.
///
/// The annotations are not ceremony: a `let _ = &r.field` proves only that the field exists
/// and nothing more, and all of Phase 6's telemetry **iterates** these structures. A shape
/// change — `Vec` to `BTreeMap`, `T` to `Option<T>` — would compile a loose binding and break
/// the whole phase. Existence and shape are two separate checks.
///
/// Never called: its body gets type-checked all the same, which is all that's needed.
/// `MagiReport` is `#[non_exhaustive]`, so there is no way to construct one from outside the
/// crate.
fn report_shape(r: &MagiReport) {
    let _: &str = &r.report;
    let _: bool = r.degraded;
    // PER SEAT and with a `Vec` inside. Without the type annotation, "naming the model that
    // did not adhere" (REQ-A09) seemed impossible from an `AgentName` key.
    let _: &BTreeMap<AgentName, Vec<ExtractionFailure>> = &r.extraction_failures;
    // **`Option`**, not a value. REQ-A11 requires the field to ALWAYS be present in magi-rs's
    // JSON, so the `None` is **mapped**, not omitted: it's our own translation, not a mirror
    // of the report.
    let _: &Option<InputSize> = &r.input_size;
    // The verified substitute for a `window_rejected` that does NOT exist on `MagiReport`
    // (it lives in `rotation.rs`, which is MS3). REQ-A11d and SC-A11g were reworked around
    // this.
    let _: &BTreeMap<AgentName, String> = &r.failed_agents;
    // Sustains SC-A11g: it being empty IS "zero valid verdicts", which is not a degraded
    // consensus but the absence of consensus.
    let _: &Vec<AgentOutput> = &r.agents;
    // MS3 (REQ-R07) pins the shape the placeholder deferred. It is a MAP KEYED BY SEAT and
    // populated for EVERY agent, rotated or not — its own rustdoc calls it "always present" —
    // so "did this mage rotate?" is `chain` being non-empty, NEVER the map's length. A test
    // written against `rotations.len() == 1` would be false for every possible run.
    let _: &BTreeMap<AgentName, AgentRotation> = &r.rotations;
}

/// [`AgentRotation`] fields that REQ-R06/R07/R08 surface, **with annotated types**.
///
/// `model_used` is the one that cannot be missing: a report naming the CONFIGURED model when the
/// fallback is what ran lies about its own evidence base. `ran_unmeasured` qualifies the verdict
/// (REQ-R08), and `chain` is the ordered list of hops.
fn agent_rotation_shape(a: &AgentRotation) {
    let _: &str = &a.model_configured;
    let _: &str = &a.model_used;
    let _: &Vec<RotationEvent> = &a.chain;
    let _: bool = a.ran_unmeasured;
}

/// The pool construction chain MS3 wires (Task 4.1), chained so each step must return `Self`.
///
/// `max_rotations` takes a `u32` and `0` is the kill-switch that must survive as a DECLARED
/// value: if this ever became `Option<u32>` or a `usize`, the collapse of `None` and `Some(0)`
/// would turn an explicit "no rotation" into "use the default" — the opposite instruction.
fn fallback_pool_surface(p: Arc<dyn LlmProvider>) -> FallbackPool {
    FallbackPool::builder()
        .push(p.clone(), Lineage::new("guardian-lineage"))
        // Pinned because PRODUCTION uses this one on every non-ephemeral run: whenever a
        // capability cache exists, candidates are pushed WITH their probe. Pinning only `push`
        // left the door production actually walks through unguarded, which defeats this file's
        // purpose of concentrating API drift in one place.
        .push_with_probe(p, Lineage::new("guardian-lineage"), guardian_probe())
        .max_rotations(2)
        .build()
}

/// A probe that answers nothing, for pinning signatures rather than behaviour.
///
/// `declared_model` is overridden deliberately: its trait default returns `None`, which silently
/// opts out of the preflight's correspondence check. A SEMANTIC change to that default would
/// compile everywhere and disable the check without a word, so overriding it here is what makes
/// the dependency visible.
struct GuardianProbe;

#[async_trait::async_trait]
impl ProviderProbe for GuardianProbe {
    async fn window(&self) -> Result<Option<usize>, magi_core::error::ProviderError> {
        Ok(None)
    }
    async fn digest(&self) -> Result<Option<String>, magi_core::error::ProviderError> {
        Ok(None)
    }
    fn declared_model(&self) -> Option<&str> {
        Some(SYNTHETIC_MODEL)
    }
}

/// See [`GuardianProbe`].
fn guardian_probe() -> Arc<dyn ProviderProbe> {
    Arc::new(GuardianProbe)
}

/// [`ExtractionFailure`] fields that REQ-A09 requires to surface.
///
/// `model` is the one that can't be missing: with rotation (MS3) the actionable question is
/// *which model* failed to adhere, not *which seat*.
fn extraction_failure_shape(f: &ExtractionFailure) {
    let _: &str = &f.model;
    let _: u8 = f.attempt;
    let _ = &f.cause;
}

/// [`InputSize`] fields, all three of which go into REQ-A11's JSON without omitting any.
fn input_size_shape(s: &InputSize) {
    let _: usize = s.estimated_tokens;
    let _: usize = s.warn_threshold;
    let _: bool = s.exceeded;
}

/// [`MagiBuilder`] methods that the trio wiring chains (Task 4.1).
///
/// Chained on purpose: each one must return `Self` by value. If any of them switched to
/// `&mut Self`, the chain would stop compiling here instead of in Phase 4.
fn builder_surface(b: MagiBuilder, p: Arc<dyn LlmProvider>) -> MagiBuilder {
    b.with_timeout(Duration::from_secs(90))
        // MS3 (REQ-R01): `with_agent`, not `with_provider` — the only door that carries the
        // rotation diversity key. `with_provider` still exists and still compiles, which is
        // exactly why the migration needed pinning: nothing about it fails on its own.
        .with_agent(
            AgentName::Melchior,
            p.clone(),
            Lineage::new("guardian-lineage"),
        )
        // Production's door whenever a cache exists, and the one whose ORDER is load-bearing: a
        // plain `with_agent` after it for the SAME seat discards the probe in silence.
        .with_agent_and_probe(
            AgentName::Balthasar,
            p.clone(),
            Lineage::new("guardian-lineage"),
            guardian_probe(),
        )
        .with_fallback_pool(fallback_pool_surface(p))
        .with_strict_context_guard(false)
        .with_input_warn_tokens(96_000)
        .with_retry_disabled()
}

/// That a concrete type satisfies [`LlmProvider`], not just that the trait exists.
fn assert_is_provider<P: LlmProvider + 'static>(_p: &P) {}

/// Same for [`ProviderProbe`], which is a **separate** trait — REQ-A24's composition depends
/// on being able to implement one without the other.
fn assert_is_probe<P: ProviderProbe + 'static>(_p: &P) {}

/// The `match` over [`Mode`] is exhaustive **with no `_` arm**.
///
/// magi-core documents the enum as deliberately closed: *"no `#[non_exhaustive]`: a new mode
/// should break exhaustive matches so consumers revisit their logic"*. Three MS2 functions
/// assume this (`GateThresholds::for_mode`, `CliMode::into_mode`, `normalize_label`), so
/// pinning it here is accepting the invitation: if 3.2.0 adds a mode, **this** is the first
/// thing that breaks, in Phase 0, instead of a `for_mode` returning the wrong threshold in
/// Phase 3.
fn mode_is_closed(m: Mode) -> &'static str {
    match m {
        Mode::CodeReview => "code-review",
        Mode::Design => "design",
        Mode::Analysis => "analysis",
    }
}

#[test]
fn magi_core_api_surface_is_what_the_plan_assumes() {
    // --- (1) The three seats and the three modes ------------------------------------------
    let _seats = [AgentName::Melchior, AgentName::Balthasar, AgentName::Caspar];
    assert_eq!(mode_is_closed(Mode::CodeReview), "code-review");
    assert_eq!(mode_is_closed(Mode::Design), "design");
    assert_eq!(mode_is_closed(Mode::Analysis), "analysis");

    // --- (2) TYPE properties the design hangs off of ---------------------------------------
    fn assert_clone<T: Clone>() {}
    fn assert_copy_eq<T: Copy + PartialEq>() {}
    // The three seats share a retry config and every `RetryProvider::with_config` consumes it
    // **by value**: without `Clone` it would have to be rebuilt per seat, and REQ-A04's
    // derived scale would stop being a single thing.
    assert_clone::<RetryConfig>();
    // `GateVerdict::Veto { mode: *mode }` and half a dozen `assert_eq!`s over modes.
    assert_copy_eq::<Mode>();
    // The probe's injection seam (Task 5.1) is `Arc<dyn ProviderProbe>`. If the trait stopped
    // being dyn-compatible, the whole factory would need rethinking — better to know here.
    let _: Option<Arc<dyn ProviderProbe>> = None;

    // --- (3) `RetryConfig` is `#[non_exhaustive]` -------------------------------------------
    // From outside the crate, neither the `RetryConfig { .. }` literal nor the functional
    // update `..default()` compile. The mandated pattern is a mutable `default()`, which is
    // what magi-core documents.
    let mut retry = RetryConfig::default();
    retry.operation_budget = Duration::from_secs(54);
    let _: Duration = retry.operation_budget;

    // --- (4) `Mode` has NO parsing method ----------------------------------------------------
    // What exists is `Display` + serde in kebab-case, which is why MS2 needs its own
    // `ModeExt::parse_config_value` (Task 1.0). That trait belongs to magi-rs and is born in
    // Phase 1: naming it here would make it non-compilable right at the spike whose job is to
    // prevent exactly that.
    let _: String = Mode::CodeReview.to_string();
    let parsed: Mode = serde_json::from_str(r#""code-review""#).expect("kebab-case");
    assert_eq!(parsed, Mode::CodeReview);

    // --- (5) Constructors, with their real ARGUMENT ORDER -----------------------------------
    // `api_key` FIRST on Claude. Both parameters are `impl Into<String>`, so swapping them
    // **compiles** and fails at runtime with a 401 — the kind of defect no review ever catches.
    let _ = ClaudeProvider::new("api-key", SYNTHETIC_MODEL);
    let _ = ClaudeProvider::with_timeout("api-key", SYNTHETIC_MODEL, Duration::from_secs(27));

    // `Option<String>` in the THIRD parameter; `None` is the Ollama case (keyless).
    let openai = OpenAiCompatibleProvider::new(SYNTHETIC_BASE_URL, SYNTHETIC_MODEL, None)
        .expect("valid synthetic base_url");
    let _ = OpenAiCompatibleProvider::with_timeout(
        SYNTHETIC_BASE_URL,
        SYNTHETIC_MODEL,
        None,
        Duration::from_secs(27),
    );
    assert_is_provider(&openai);

    // `OllamaProvider` serves BOTH roles as of v0.13.0, and `with_timeout` is the load-bearing
    // constructor — the one REQ-R30 requires and §7 of the spec names in the only remaining
    // prohibition on this type: never `new`, because it delegates with a 300 s default that
    // cannot satisfy `operation_budget + client_timeout <= ceiling` and does so while compiling
    // and running perfectly.
    //
    // This comment used to say the opposite — "ONLY as a probe: its sole constructor fixes a
    // 300 s client with no override" — which was the state before 3.2.0 added `with_timeout` and
    // before this milestone reverted D-A07 on the strength of it. The module header nine lines
    // into this file already said so, so the file contradicted itself (S4 Loop 2, Caspar).
    //
    // Both constructors are pinned. `new` because `src/magi/probe.rs` still uses it deliberately
    // and safely, under its own 5 s ceiling; `with_timeout` because everything that completes
    // through this type depends on it existing with this arity. Either one disappearing upstream
    // must break here rather than at a call site.
    let ollama =
        OllamaProvider::new(SYNTHETIC_BASE_URL, SYNTHETIC_MODEL).expect("valid synthetic base_url");
    let _ =
        OllamaProvider::with_timeout(SYNTHETIC_BASE_URL, SYNTHETIC_MODEL, Duration::from_secs(27));
    assert_is_provider(&ollama);
    assert_is_probe(&ollama);

    // --- (6) `RetryProvider` wraps an `Arc<dyn LlmProvider>` ---------------------------------
    // REQ-A03: `MagiBuilder::build()` does NOT wrap anything, so without this the trio loses
    // the retry it currently inherits from the adapter — a resilience regression.
    let inner: Arc<dyn LlmProvider> = Arc::new(
        OpenAiCompatibleProvider::new(SYNTHETIC_BASE_URL, SYNTHETIC_MODEL, None)
            .expect("valid synthetic base_url"),
    );
    let _ = RetryProvider::with_config(inner, RetryConfig::default());

    // --- (7) Orchestrator config: the two fields the derived scale reads ---------------------
    let _: Duration = CoreMagiConfig::default().timeout;
    let _: usize = CoreMagiConfig::default().max_input_len;

    // --- (8) Verdict markers (3.0.0 contract) -------------------------------------------------
    // Test doubles MUST emit the verdict between these markers: magi-core removed its search
    // parser, so a bare JSON no longer parses no matter how valid it is.
    assert!(!VERDICT_OPEN.is_empty());
    assert!(!VERDICT_CLOSE.is_empty());

    // --- (9) Shapes that can't be instantiated outside the crate ------------------------------
    // Referencing the item marks it as used; its body was already type-checked. There's no
    // need to call it, and no `#[allow(dead_code)]` to justify.
    let _ = report_shape;
    let _ = extraction_failure_shape;
    let _ = input_size_shape;
    let _ = builder_surface;
    // MS3 (Phase 4). `fallback_pool_surface` needs no entry: `builder_surface` calls it.
    let _ = agent_rotation_shape;
}

/// Content long enough for the orchestrator to dispatch all three seats.
///
/// magi-core's complexity gate vetoes trivial content, so a short payload would make these
/// guardians measure **zero calls** and pass for the wrong reason.
const DISPATCHABLE_CONTENT: &str =
    "Content that is more than long enough for the orchestrator to dispatch all three seats \
     instead of vetoing the query as trivial, which is what a short payload would trigger.";

/// Seats magi-core dispatches per query: the full trio.
const EXPECTED_SEATS: usize = 3;

/// Attempts per seat when facing an invalid schema — **measured**, not assumed.
///
/// A probe against magi-core 3.1.0 (2026-08-02) observed all three seats making exactly two
/// calls each. That's where the factor of 2 in the `--timeout` formula (REQ-A04) comes from,
/// and that's why the exact value is asserted instead of `>= 2`: that `>=` would pass with
/// just **one** seat retrying, i.e. it would not distinguish the healthy case from the
/// degraded one.
const ATTEMPTS_PER_SEAT: usize = 2;

/// SC-A04b, first half: a schema failure consumes **TWO** `timeout` windows.
///
/// This is where the factor of 2 in the headless `--timeout` formula (REQ-A04) comes from. If
/// magi-core stopped retrying on invalid schema, the scale would end up oversized and nobody
/// would notice — the consult would keep working, only the derived `--timeout` would cover
/// twice what's needed.
///
/// **The count is PER SEAT, and that's the difference between a guardian and window dressing.**
/// With a global counter, `total >= 2` passes with three mages each calling once — i.e. **even
/// if magi-core never retries at all**. The system prompt discriminates seats because each
/// mage receives its own (REQ-A02).
#[tokio::test]
async fn schema_retry_consumes_two_timeout_windows_per_seat() {
    let ceiling = Duration::from_secs(2);
    let provider = Arc::new(support::SchemaFailsOnceProvider::new(
        Duration::from_millis(100),
    ));
    let magi = MagiBuilder::new(provider.clone())
        .with_timeout(ceiling)
        .build()
        .expect("the builder accepts a single shared provider");

    let started = Instant::now();
    let _ = magi.analyze(&Mode::Analysis, DISPATCHABLE_CONTENT).await;
    let elapsed = started.elapsed();

    let by_seat = provider.calls_by_seat();
    // Only the COUNTS go into the messages: the keys are the full system prompts, i.e. ~30 KB
    // that would make any failure unreadable. The count is the data; the key is just the
    // medium.
    let counts: Vec<usize> = by_seat.values().copied().collect();

    assert_eq!(
        by_seat.len(),
        EXPECTED_SEATS,
        "expected {EXPECTED_SEATS} seats with distinct system prompts and got {}: either \
         magi-core stopped dispatching the full trio, or it stopped giving each mage its own \
         system prompt (REQ-A02) — which is what makes this count discriminating",
        by_seat.len(),
    );
    assert!(
        counts.iter().all(|n| *n == ATTEMPTS_PER_SEAT),
        "each seat must consume exactly {ATTEMPTS_PER_SEAT} attempts on invalid schema; \
         observed {counts:?}. Fewer ⇒ magi-core stopped retrying and REQ-A04's factor of 2 \
         oversizes the scale. More ⇒ the worst case is no longer 2x the ceiling and the \
         `--timeout` formula underestimates it",
    );
    assert!(
        elapsed < ceiling * 3,
        "the worst case exceeded 2x the ceiling ({elapsed:?} with ceiling {ceiling:?})",
    );
}

/// SC-A04b, second half: a **hanging** provider consumes only ONE window.
///
/// This is the asymmetry that makes the formula correct: a provider timeout does **not**
/// trigger the corrective schema retry, so that path costs 1x, not 2x. If magi-core started
/// retrying after a timeout too, the worst case per mage would jump from 2x to 4x and the
/// derived `--timeout` would start cutting off healthy consults.
#[tokio::test]
async fn a_hanging_provider_consumes_one_timeout_window() {
    let ceiling = Duration::from_millis(300);
    let provider = Arc::new(support::HangingProvider::default());
    let magi = MagiBuilder::new(Arc::clone(&provider) as Arc<dyn LlmProvider>)
        .with_timeout(ceiling)
        .build()
        .expect("the builder accepts a single shared provider");

    let started = Instant::now();
    let _ = magi.analyze(&Mode::Analysis, DISPATCHABLE_CONTENT).await;
    let elapsed = started.elapsed();

    // Presence before the bound (S4 Loop 2, Balthasar). The assertion below is an UPPER bound,
    // and an upper bound is satisfied perfectly by never dispatching at all: a gate that started
    // rejecting `DISPATCHABLE_CONTENT`, or a builder change that short-circuited, would leave
    // this test green while the property it guards went unexercised.
    assert!(
        provider.calls() > 0,
        "precondition: the provider was never entered, so nothing hung and the bound below \
         measures nothing"
    );
    assert!(
        elapsed >= ceiling,
        "a hang must consume its window: {elapsed:?} is under the {ceiling:?} ceiling, which \
         means the call returned by some path other than the timeout"
    );

    assert!(
        elapsed < ceiling * 2,
        "a hang consumed {elapsed:?} with ceiling {ceiling:?}: magi-core started retrying \
         after a timeout, and REQ-A04's worst case goes from 2x to 4x",
    );
}

/// How long it's tolerable to wait for the overlap double's rendezvous before concluding
/// dispatch is NOT concurrent (MAGI S3 re-gate, Caspar — replaces the first draft's fixed
/// 500 ms `OVERLAP_DWELL`).
///
/// **A generous ceiling, not a precise measurement** — the same discipline the rest of the
/// project requires for clock-dependent tests (`.config/nextest.toml` documents twice over the
/// cost of not following it): with genuinely concurrent dispatch, [`support::OverlapCountingProvider`]'s
/// rendezvous resolves in milliseconds even under the Argon2 load from the rest of the suite
/// (this test runs in the `default` group, not `heavy`); if magi-core regressed to serial
/// dispatch, the last seat's rendezvous would NEVER complete (nothing can "leave" until all
/// three "arrive"), so this ceiling is exactly where that defect turns into a clear failure
/// instead of a hung suite.
const OVERLAP_RENDEZVOUS_DEADLINE: Duration = Duration::from_secs(30);

/// Per-seat ceiling passed to the builder for THIS test. Just as generous as
/// [`OVERLAP_RENDEZVOUS_DEADLINE`] and for the same reason: if magi-core's internal ceiling
/// (REQ-A04) were tighter than what the rendezvous needs under real contention, a seat could
/// be aborted by timeout BEFORE reaching the rendezvous — a different failure (agent timeout)
/// this test does not exist to diagnose. It does not measure real wall-clock: the double
/// generates nothing, it just waits for the other two.
const OVERLAP_AGENT_TIMEOUT: Duration = OVERLAP_RENDEZVOUS_DEADLINE;

/// SC-A04e: the three mages execute **overlapped**, not serially.
///
/// This is what sustains the "**NOT** multiplied by 3" in the `--timeout` formula (REQ-A04):
/// with parallel dispatch, the worst case of a consult is that of the slowest mage, not the
/// sum of the three. If magi-core switched to serial dispatch, that worst case would jump from
/// 2x to 6x the ceiling and the derived `--timeout` would start cutting off perfectly healthy
/// consults — **without a single line of magi-rs changing**, which is exactly the silent
/// failure this guardian turns into a broken suite.
///
/// **The peak is asserted as {EXPECTED_SEATS}, not `>= 2`.** Two overlapping mages plus a
/// third running serially would already break the formula — the worst case becomes 4x — and
/// a `>= 2` would pass it as fine.
///
/// **It waits on a CONDITION (a rendezvous of three arrivals), not on a fixed duration** — see
/// [`support::OverlapCountingProvider`]'s doc comment for why: the previous version of this
/// test slept 500 ms per call and trusted the scheduler to dispatch all three within that
/// window, which is precisely the load-induced flakiness pattern this project has already
/// diagnosed twice.
#[tokio::test]
async fn the_three_mages_execute_concurrently() {
    let (provider, peak) = support::OverlapCountingProvider::new(EXPECTED_SEATS);

    let magi = MagiBuilder::new(provider)
        .with_timeout(OVERLAP_AGENT_TIMEOUT)
        .build()
        .expect("the builder accepts a single shared provider");

    let outcome = tokio::time::timeout(
        OVERLAP_RENDEZVOUS_DEADLINE,
        magi.analyze(&Mode::Analysis, DISPATCHABLE_CONTENT),
    )
    .await;
    assert!(
        outcome.is_ok(),
        "did not complete within {OVERLAP_RENDEZVOUS_DEADLINE:?}: likely SERIAL dispatch — the \
         last seat's rendezvous would never have reached {EXPECTED_SEATS} arrivals",
    );

    let observed = peak.load(Ordering::SeqCst);
    assert_eq!(
        observed, EXPECTED_SEATS,
        "concurrency peak = {observed}, expected {EXPECTED_SEATS}: magi-core stopped \
         dispatching all three seats in parallel. The `--timeout` formula (REQ-A04) assumes \
         full overlap and now underestimates the worst case",
    );
}

/// SPIKE for Task 0.6: what `MagiReport::report` exposes in a locatable way.
///
/// Decides the maximum reachable `TruncationLevel`, which Task 6.2 consumes. It runs in Phase
/// 0 rather than Phase 6 on purpose: structural truncation depends on being able to locate the
/// verdict and findings in markdown generated by **another crate**, and that's an assumption.
/// Discovering it can't be done with the milestone nearly finished would leave the requirement
/// with no way out.
#[tokio::test]
async fn report_shape_matches_what_the_truncation_design_assumes() {
    let provider = Arc::new(support::AdheringTrioProvider);
    let magi = MagiBuilder::new(provider)
        .build()
        .expect("the builder accepts a single shared provider");
    let report = magi
        .analyze(&Mode::CodeReview, DISPATCHABLE_CONTENT)
        .await
        .expect("all three adhere, so there is a report");

    assert!(
        !report.report.is_empty(),
        "an empty report would invalidate any truncation level",
    );

    // The anchors are IMPORTED from production, not redeclared here. Redeclaring them would
    // split the truth in two: the spike would measure one thing and `truncate_report` (Task
    // 6.2) would read another, and the disagreement would only surface once a report came out
    // badly truncated.
    let anchors = SECTION_ANCHORS.expect("the spike concluded that `Structural` is reachable");
    for anchor in [
        anchors.verdict_start,
        anchors.findings_start,
        anchors.findings_end,
    ] {
        // `contains("")` is ALWAYS true, so an anchor emptied by a refactor would satisfy every
        // check below and turn this whole guardian decorative — it would keep passing while
        // `Structural` truncation silently stopped being reachable, which is the one outcome it
        // exists to catch (S4 Loop 2, Balthasar).
        assert!(
            !anchor.is_empty(),
            "an empty section anchor makes every `contains` below vacuously true"
        );
        assert!(
            report.report.contains(anchor),
            "missing section anchor: {anchor:?}. magi-core changed its rendering and REQ-A11b's \
             `Structural` level is no longer reachable with these anchors — which is exactly \
             what this guardian exists to warn about before a user does.\n\
             Observed report:\n{}",
            report.report,
        );
    }
    for anchor in CONTRACTUAL_ANCHORS {
        assert!(
            !anchor.is_empty(),
            "an empty contractual anchor makes the check below vacuously true"
        );
        assert!(
            report.report.contains(anchor),
            "missing contractual anchor: {anchor:?}. These are the ones magi-core ALWAYS \
             emits, so their absence lowers the truncation ceiling to `Bytes`.\n\
             Observed report:\n{}",
            report.report,
        );
    }
}

/// Defensive cap on what the mock will actually write before giving up.
///
/// The body is meant to look **endless**, but a literally infinite loop turns a failed
/// guardian into a **hung** test, which is the worst outcome: it doesn't report, it blocks the
/// suite, and someone has to go hunt for it. 64 MiB is 64x magi-core's 1 MiB cap, so if the
/// reader cuts off — which is what's being verified — it's never reached.
const MOCK_ENDLESS_BODY_LIMIT: usize = 64 * 1024 * 1024;

/// Size of each chunk the mock writes.
const MOCK_CHUNK_BYTES: usize = 8 * 1024;

/// Ceiling of SERVED bytes the assertion tolerates before concluding magi-core stopped cutting
/// off (loop 1 fix round CE, F6).
///
/// **This is the number that makes the test discriminate.** The three assertions it had
/// before (mock hit, `None`/`Err` result, `elapsed` under the 5 s timeout) are ALL satisfied
/// just the same by a reader with no cap that swallows the full 64 MiB over loopback — that
/// takes well under 5 s — and fails to parse the JSON anyway because it deliberately never
/// closes, so it would end up in `Ok(None)`/`Err(_)` either way. None of those three measure
/// the one thing that separates "cut off while reading" from "read everything and only
/// measured afterward": the BYTES actually transferred before the connection got cut.
///
/// The value is 8x magi-core's 1 MiB `MAX_SHOW_BODY_BYTES` — generous enough not to couple to
/// the exact byte count of the `pub(crate)` constant that can't be named from here, plus margin
/// for one chunk in flight when the reader cuts off — and 8x below the mock's 64 MiB: a reader
/// with no cap crosses it immediately, one that cuts off at 1 MiB never comes close.
const MAX_BYTES_TOLERATED_BEFORE_CUT: usize = 8 * 1024 * 1024;

/// REQ-A16b / SC-A16c — **satisfied BY magi-core, verified here.**
///
/// `OllamaProvider::window()` does its own HTTP and magi-core already bounds the body:
/// `MAX_SHOW_BODY_BYTES = 1 MiB`, read via `read_probe_body`, whose rustdoc says the body is
/// *"untrusted"* and stays *"bounded"* — i.e. it cuts off **while reading**, which is exactly
/// what the requirement asks for.
///
/// **That's why magi-rs does NOT implement its own `read_capped`**: it would be
/// reimplementing an existing guard, which is what R-A02 forbids. What is warranted is a
/// guardian that the property keeps holding. The constant is `pub(crate)`, so it can't be
/// asserted on directly — the assertion is on the **behavior** instead, which is what matters
/// anyway.
///
/// The body is emitted in chunks rather than as one giant `String`: the point is that the
/// reader cuts off **during** the read, and a finite-but-large body doesn't distinguish "cut
/// off while reading" from "buffered it all and measured afterward" — **unless the actually
/// served bytes are counted**, which is what this test does (fix round CE, F6): the mock
/// increments a shared counter before EVERY `write_all`, so its final value is exactly how
/// much made it through before the reader cut the connection — the only signal that
/// distinguishes "cut off while reading" from "read everything and then still failed to parse
/// it".
///
/// If this test **fails**, magi-core stopped capping and REQ-A16b becomes magi-rs's
/// responsibility — and the guardian said so before a hostile endpoint said so in production.
#[tokio::test]
async fn magi_core_rejects_an_endless_probe_body_instead_of_accumulating_it() {
    let bytes_served = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = Arc::clone(&bytes_served);

    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/api/show")
        .with_status(200)
        .with_chunked_body(move |writer| {
            // A JSON that never closes: it starts out plausible and never ends.
            writer.write_all(br#"{"model_info":{"llama.context_length":"#)?;
            let chunk = [b'1'; MOCK_CHUNK_BYTES];
            let mut written = 0usize;
            while written < MOCK_ENDLESS_BODY_LIMIT {
                // Counted BEFORE writing: what matters is how much was ATTEMPTED to be
                // served, not only what confirmedly arrived — this way the counter can't
                // undercount due to a race between the increment and the connection cutting
                // off.
                counter.fetch_add(MOCK_CHUNK_BYTES, Ordering::SeqCst);
                // When the reader cuts off and drops the connection, this returns `Err` and
                // the callback ends on its own. That `?` IS the expected end of the mock.
                writer.write_all(&chunk)?;
                written += MOCK_CHUNK_BYTES;
            }
            Ok(())
        })
        .create_async()
        .await;

    let probe = OllamaProvider::new(server.url(), SYNTHETIC_MODEL).expect("mock's base_url");

    let started = Instant::now();
    let window = probe.window().await;
    let elapsed = started.elapsed();

    // FIRST, that the mock was actually hit, and this isn't ceremony: the assertion below
    // accepts `Err(_)`, so it would pass just as well if the request never arrived — a
    // malformed URL, a refused connection — and the guardian would be certifying a cap it
    // never exercised. It's the same vacuity that makes an `any(>= 2)` useless.
    mock.assert_async().await;

    assert!(
        matches!(window, Ok(None) | Err(_)),
        "an endless body must degrade to not-measured or to an error, never complete with a \
         value: magi-core stopped bounding the probe body and REQ-A16b became ours",
    );
    assert!(
        elapsed < Duration::from_secs(PROBE_TIMEOUT_SECS),
        "took {elapsed:?}: it must cut off BY SIZE while reading, not accumulate until a \
         timeout expires. Against a hostile endpoint the difference is between 1 MiB and \
         unbounded memory",
    );
    let served = bytes_served.load(Ordering::SeqCst);
    assert!(
        served < MAX_BYTES_TOLERATED_BEFORE_CUT,
        "the mock managed to serve {served} bytes before the reader cut the connection — \
         that's {}x magi-core's documented cap (1 MiB): it stopped cutting off while reading \
         and buffered the entire body instead",
        served / (1024 * 1024),
    );
}
