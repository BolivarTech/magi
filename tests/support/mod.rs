// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-02

//! Shared doubles for MS2's integration tests (Task 0.7).
//!
//! # Why this is a module with an owner and not a per-test detail
//!
//! The plan cites close to twenty doubles across twelve tasks and **did not budget for them
//! in any of them**. Leaving them implicit does two concrete kinds of damage, both already
//! seen in this project:
//!
//! 1. **It breaks the Red phase.** "Red for the right reason" requires that the only missing
//!    symbol be the one the task implements. An unwritten double makes the test **fail to
//!    compile**, which is a different kind of red and a violation of `CLAUDE.local.md` §3.
//! 2. **It hides cost where it doesn't show.** The type-check and imports of a shared module
//!    don't appear in the `use` block of the file that consumes it, so it's exactly the cost
//!    that gets underestimated when budgeting a task.
//!
//! # Built incrementally, alongside the phase that debuts it
//!
//! The twenty aren't all written here. Each phase adds its own in its own Step 1; what this
//! task fixes is the **owner and the location**, so they never again show up as names with no
//! file. Today it holds Phase 0's.
//!
//! **`MockEndpoint` is NOT here yet, and that's on purpose:** it needs `wiremock`, which only
//! enters as a dev-dependency in Task 0.5. Writing it earlier would leave this module unable
//! to compile, which is exactly the failure it's here to prevent. That task adds it, along
//! with its dependency.
//!
//! # Lazy imports
//!
//! Tasks across all seven phases consume this module, so it will end up bringing in types born
//! in Phase 5. Each double imports **its own things inside its own block**, never at the
//! header: otherwise a later task breaks the **entire** test collection and the failure
//! doesn't point back to the task that caused it.

// A module under `tests/` compiles ONCE PER TEST BINARY, and each binary uses a different
// subset. The `dead_code` that produces is structural to the layout, not a forgotten symbol:
// it's the pattern the Rust Book itself documents for `tests/common`. Without this, adding a
// double for one binary would fail `-D warnings` in every other one.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use magi_core::error::{ExternalErrorKind, ProviderError};
// `CompletionConfig` lives in `provider`, NOT in `orchestrator`: there it's only imported and
// is private. The plan had pasted the `orchestrator` path into all three doubles.
use magi_core::provider::{CompletionConfig, LlmProvider};
use magi_core::verdict_markers::{VERDICT_CLOSE, VERDICT_OPEN};

/// Name the doubles report via `LlmProvider::name`.
///
/// `LlmProvider` requires **three** methods, not just `complete`: `name` and `model` have no
/// default impl. It's telemetry — magi-core uses them to name the provider in a report — so a
/// double can return a fixed value, but it can't omit them.
const DOUBLE_PROVIDER_NAME: &str = "test-double";

/// Model the doubles report via `LlmProvider::model`. See [`DOUBLE_PROVIDER_NAME`].
const DOUBLE_MODEL_NAME: &str = "test-double-model";

/// A verdict that satisfies magi-core's schema.
///
/// Goes **between the markers** in every double that emits it: magi-core 3.0.0 removed its
/// search parser, so a bare JSON no longer parses no matter how valid it is.
#[must_use]
pub fn valid_verdict_json() -> String {
    r#"{"agent":"melchior","verdict":"approve","confidence":0.9,
        "summary":"ok","reasoning":"ok","findings":[],"recommendation":"ok"}"#
        .to_string()
}

/// Wraps a verdict in the markers magi-core requires to extract it.
#[must_use]
pub fn marked_verdict() -> String {
    format!("{VERDICT_OPEN}\n{}\n{VERDICT_CLOSE}", valid_verdict_json())
}

/// The three seats, lowercase, exactly as magi-core expects them in the `agent` field.
const SEAT_NAMES: [&str; 3] = ["melchior", "balthasar", "caspar"];

/// A verdict **with findings**, issued in the name of `agent`.
///
/// Two things this helper exists to solve, both discovered by running the tests:
///
/// 1. **The findings can't be empty.** [`valid_verdict_json`]'s are, and with an empty list
///    the report might not emit the findings section at all — meaning a spike on top of it
///    couldn't decide whether that section is locatable, which is exactly what Task 0.6 has
///    to decide.
/// 2. **The `agent` field MUST match the seat that asked.** magi-core validates that
///    correspondence and discards a verdict that doesn't satisfy it: a double that answers
///    `"melchior"` to all three leaves `succeeded: 1` and the orchestrator aborts with
///    `InsufficientAgents`. The retry guardian didn't notice because it only counts calls, so
///    the constraint stayed invisible until a test looked at the report.
#[must_use]
pub fn verdict_json_with_findings(agent: &str) -> String {
    format!(
        r#"{{"agent":"{agent}","verdict":"conditional","confidence":0.85,
        "summary":"one-line summary","reasoning":"the mage's reasoning",
        "findings":[
          {{"severity":"critical","title":"First finding","detail":"detail of the first",
           "file":"src/x.rs","line":42,"category":"logic-error"}},
          {{"severity":"warning","title":"Second finding","detail":"detail of the second",
           "file":"src/y.rs","line":7,"category":"performance"}}
        ],
        "recommendation":"what it recommends"}}"#
    )
}

/// Deduces the seat from the **first line** of a system prompt.
///
/// **Only the header, never the whole prompt, and this cost a whole cycle to discover:** the
/// prompts mention each other — Caspar's says *"Leave happy-path correctness analysis to
/// Melchior"* — so searching the name across the whole text would assign Caspar's verdict to
/// Melchior, magi-core rejects it with `agent identity mismatch`, and the seat is lost. The
/// first line is `# <Name> — <Role>`, which does discriminate correctly.
///
/// Falls back to `melchior` if it recognizes none: a double must not fail silently, but it
/// also must not panic inside an orchestrator task — the resulting `InsufficientAgents` gives
/// it away anyway, with a better message than a panic inside a `spawn`.
///
/// **Free function, not a method on a single double (B3, MAGI S3 re-gate fix):** it originally
/// lived only in `AdheringTrioProvider`, but `OverlapCountingProvider` has the SAME need — a
/// double that answers the same `"agent":"melchior"` to all three seats makes magi-core reject
/// and **retry** the two mismatched ones, multiplying `complete` calls beyond the number of
/// seats and breaking any guardian that counts exact calls (like `OverlapCountingProvider`'s
/// rendezvous). Sharing the function keeps a third double from reimplementing it wrong.
fn seat_from_prompt(system_prompt: &str) -> &'static str {
    let header = system_prompt
        .lines()
        .next()
        .unwrap_or_default()
        .to_lowercase();
    SEAT_NAMES
        .into_iter()
        .find(|seat| header.contains(seat))
        .unwrap_or("melchior")
}

/// Answers a valid verdict **in the name of the seat that asked**, with findings.
///
/// All three seats share the instance and the double discriminates them by their system
/// prompt, which is where the mage's name appears (REQ-A02). This way the report comes out
/// with all three adhering, which is the condition for the render to expose every section.
pub struct AdheringTrioProvider;

#[async_trait]
impl LlmProvider for AdheringTrioProvider {
    async fn complete(
        &self,
        system_prompt: &str,
        _user_prompt: &str,
        _config: &CompletionConfig,
    ) -> Result<String, ProviderError> {
        let seat = seat_from_prompt(system_prompt);
        Ok(format!(
            "{VERDICT_OPEN}\n{}\n{VERDICT_CLOSE}",
            verdict_json_with_findings(seat)
        ))
    }

    fn name(&self) -> &str {
        DOUBLE_PROVIDER_NAME
    }

    fn model(&self) -> &str {
        DOUBLE_MODEL_NAME
    }
}

/// Returns invalid schema on each seat's first attempt and valid on the second.
///
/// Sustains SC-A04b's guardian: a validation failure consumes **two** `timeout` windows, which
/// is where the factor of 2 in the `--timeout` formula (REQ-A04) comes from.
pub struct SchemaFailsOnceProvider {
    /// Calls **per seat**, discriminated by system prompt.
    ///
    /// `Mutex<BTreeMap>` and not `AtomicUsize`: a global counter can't distinguish "one seat
    /// retried" from "three seats each called once", and with three mages a `total >= 2`
    /// passes **even if magi-core never retries at all**. A guardian that can't fail is worse
    /// than none, because it also certifies.
    ///
    /// The system prompt serves as the key because each mage receives its own (REQ-A02), so
    /// the double discriminates seats without needing to know about `AgentName`.
    pub calls_by_seat: Mutex<BTreeMap<String, usize>>,
    /// Simulated latency of each call.
    pub per_call: Duration,
}

impl SchemaFailsOnceProvider {
    /// Builds the double with its count map empty.
    #[must_use]
    pub fn new(per_call: Duration) -> Self {
        Self {
            calls_by_seat: Mutex::new(BTreeMap::new()),
            per_call,
        }
    }

    /// Copy of the per-seat count, for asserting on it without holding the lock.
    ///
    /// # Panics
    ///
    /// If the `Mutex` was poisoned by a panic in another test.
    #[must_use]
    pub fn calls_by_seat(&self) -> BTreeMap<String, usize> {
        self.calls_by_seat.lock().expect("not poisoned").clone()
    }
}

#[async_trait]
impl LlmProvider for SchemaFailsOnceProvider {
    async fn complete(
        &self,
        system_prompt: &str,
        _user_prompt: &str,
        _config: &CompletionConfig,
    ) -> Result<String, ProviderError> {
        tokio::time::sleep(self.per_call).await;
        let previous = {
            let mut map = self.calls_by_seat.lock().expect("not poisoned");
            let counter = map.entry(system_prompt.to_string()).or_insert(0);
            *counter += 1;
            *counter - 1
        };
        if previous == 0 {
            Ok("not a verdict".to_string())
        } else {
            Ok(marked_verdict())
        }
    }

    fn name(&self) -> &str {
        DOUBLE_PROVIDER_NAME
    }

    fn model(&self) -> &str {
        DOUBLE_MODEL_NAME
    }
}

/// Never responds. Sustains the other half of SC-A04b: a hang consumes **one** window, because
/// a provider timeout does **not** trigger the corrective schema retry.
pub struct HangingProvider;

#[async_trait]
impl LlmProvider for HangingProvider {
    async fn complete(
        &self,
        _system_prompt: &str,
        _user_prompt: &str,
        _config: &CompletionConfig,
    ) -> Result<String, ProviderError> {
        std::future::pending::<()>().await;
        Err(ProviderError::external(
            "unreachable",
            ExternalErrorKind::Network,
        ))
    }

    fn name(&self) -> &str {
        DOUBLE_PROVIDER_NAME
    }

    fn model(&self) -> &str {
        DOUBLE_MODEL_NAME
    }
}

/// Records the highest number of simultaneous executions it saw.
///
/// Sustains SC-A04e: the three mages **overlap**. If magi-core switched to serial dispatch,
/// the worst case would jump from 2x to 6x the ceiling and the derived `--timeout` would start
/// cutting off healthy consults — without a single line of magi-rs changing.
///
/// **Rendezvous, not a fixed `sleep` (MAGI S3 re-gate, Caspar).** The previous version had
/// every call sleep a fixed `dwell` (500 ms) before returning, trusting that all three would
/// arrive WITHIN that window for the peak to reach 3 — exactly the flakiness pattern this repo
/// has already diagnosed twice (`.config/nextest.toml`): under the Argon2 load from the rest
/// of the suite (this test runs in the `default` group, not `heavy`), a seat delayed in
/// dispatching could return AFTER another had already slept out its whole window and returned,
/// dropping the observed peak to 2 with no real defect involved.
///
/// With a [`tokio::sync::Barrier`] sized `expected`, no call can "leave" (decrement `live`)
/// until `expected` of them have "arrived" — CPU contention only makes the rendezvous take
/// longer, never the observed peak lower. It's the same discipline the rest of the project
/// requires for clock-dependent tests: wait on a CONDITION, not on a duration.
///
/// **The verdict answers in the name of the seat that asked, via [`seat_from_prompt`] — and
/// this is NOT cosmetic for a fixed-size rendezvous.** The first version of this fix reused
/// `marked_verdict()` (always `"agent":"melchior"`), and the rendezvous hung: magi-core rejects
/// Balthasar's/Caspar's verdict with `agent identity mismatch` and **retries** those two seats,
/// so `complete` ends up being called more than `expected` times for a single `analyze`. With
/// a `Barrier` of size 3 that leaves stray arrivals that never complete a group of three —
/// exactly the finding [`AdheringTrioProvider`]'s doc comment documents, applied here where it
/// also breaks the synchronization, not just the count.
pub struct OverlapCountingProvider {
    /// Executions currently in flight.
    pub live: Arc<AtomicUsize>,
    /// Highest number of simultaneous executions observed.
    pub peak: Arc<AtomicUsize>,
    /// Rendezvous point: only releases once `expected` calls have arrived at once.
    barrier: Arc<tokio::sync::Barrier>,
}

impl OverlapCountingProvider {
    /// Builds the double along with the counters the test will read.
    ///
    /// `expected` is how many simultaneous calls the rendezvous should wait for before
    /// releasing all of them — typically [`EXPECTED_SEATS`] in `magi_core_contract.rs`, but
    /// the double doesn't hardcode that number: it's the caller who knows how many seats it's
    /// going to dispatch.
    #[must_use]
    pub fn new(expected: usize) -> (Arc<Self>, Arc<AtomicUsize>) {
        let live = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let provider = Arc::new(Self {
            live,
            peak: Arc::clone(&peak),
            barrier: Arc::new(tokio::sync::Barrier::new(expected)),
        });
        (provider, peak)
    }
}

#[async_trait]
impl LlmProvider for OverlapCountingProvider {
    async fn complete(
        &self,
        system_prompt: &str,
        _user_prompt: &str,
        _config: &CompletionConfig,
    ) -> Result<String, ProviderError> {
        let now = self.live.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        // Blocks until every expected call has arrived — see the struct doc for why this
        // replaces a fixed `sleep`. The caller wraps the whole `analyze` in a generous timeout
        // so a genuine regression to serial dispatch (this never resolving) fails clearly
        // instead of hanging the suite.
        self.barrier.wait().await;
        self.live.fetch_sub(1, Ordering::SeqCst);
        // NOT `marked_verdict()` — see the struct doc for why a shared, seat-blind verdict
        // (always `"agent":"melchior"`) breaks a fixed-size rendezvous: magi-core retries the
        // two mismatched seats, and their retry calls arrive after the barrier has already
        // moved past this generation.
        let seat = seat_from_prompt(system_prompt);
        Ok(format!(
            "{VERDICT_OPEN}\n{}\n{VERDICT_CLOSE}",
            verdict_json_with_findings(seat)
        ))
    }

    fn name(&self) -> &str {
        DOUBLE_PROVIDER_NAME
    }

    fn model(&self) -> &str {
        DOUBLE_MODEL_NAME
    }
}
