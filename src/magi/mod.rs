// Author: Julian Bolivar Version: 1.0.0 Date: 2026-08-02

//! MAGI subsystem of magi-rs: mode resolution, complexity gate, and probe.

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

pub mod endpoint;
pub mod gate;
pub mod kind;
pub mod mode;
pub mod probe;
pub mod report_anchors;

use std::time::Duration;

/// Bounds of the admissible per-mage ceiling range (§4.9 of the spec). `pub`, not private: they
/// are consumed by `validate_agent_timeout` from `config.rs` (bin) and the invariant sweep from
/// `tests/` — two distinct crates, so private ones would not compile in either. The §4.9 range
/// is contract, not internal detail.
pub const AGENT_TIMEOUT_MIN_SECS: u64 = 30;
/// See [`AGENT_TIMEOUT_MIN_SECS`].
pub const AGENT_TIMEOUT_MAX_SECS: u64 = 120;

/// Ceiling PER MAGE and PER ATTEMPT (REQ-A04, verified against `orchestrator.rs`).
///
/// 90 s: enough for a legitimate generation from a cloud model with cold-load, and leaves the
/// worst case per mage (2 attempts) at 180 s. The magi-core default (300) is too high: it makes
/// the retry chain unreachable.
pub const AGENT_TIMEOUT_SECS: u64 = 90;

/// Fraction of the ceiling for the total retry budget.
///
/// 0.6 + 0.3 = 0.9 < 1.0: leaves a 10 % margin so that abandonment is **TYPED**
/// (`OperationBudgetExhausted`) and not an opaque cut from the external ceiling.
const OPERATION_BUDGET_FRACTION: f64 = 0.6;
/// Fraction of the ceiling for the timeout of ONE HTTP request. See
/// [`OPERATION_BUDGET_FRACTION`].
const CLIENT_TIMEOUT_FRACTION: f64 = 0.3;

/// Absolute floors: below them, no real request completes.
const MIN_OPERATION_BUDGET: Duration = Duration::from_secs(10);
/// See [`MIN_OPERATION_BUDGET`].
const MIN_CLIENT_TIMEOUT: Duration = Duration::from_secs(5);

/// Smallest ceiling for which the derived scale **still satisfies** REQ-A04.
///
/// # The hole this constant closes
///
/// The two floors above are `.max()`, so they **win** when the fraction ends up below. With a
/// ceiling of 10 s the derivation yields `max(6,10) + max(3,5) = 15 > 10`: the invariant that
/// REQ-A04 declares *"impossible to break by construction"* ends up broken —
/// **and it was reachable from `magi.toml`**, because `agent_timeout_secs` was not validated.
///
/// Worse: the invariant sweep ran from 30 to 120, so it **never crossed the breaking point**.
/// The test did not fail because it was not looking. A guardian that only walks the happy range
/// certifies the happy range, not the invariant.
///
/// The sum of the floors IS the breaking point, so it is **derived** from them instead of
/// written by hand: moving a floor without moving this would silently reopen the hole.
///
/// `pub` for the same reason as [`AGENT_TIMEOUT_MIN_SECS`]: the rustdoc for
/// `MagiConfig::validate_agent_timeout` links to it from the **bin**, and an intra-doc link to
/// a private symbol from another crate does not resolve. It is the documented breaking point of
/// the derivation, i.e. contract of the module — not an internal detail.
pub const AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS: u64 =
    MIN_OPERATION_BUDGET.as_secs() + MIN_CLIENT_TIMEOUT.as_secs();

/// Total retry budget, DERIVED from the ceiling (REQ-A04).
///
/// The derivation is what makes it impossible to configure an invalid scale: no combination
/// exists that breaks `operation_budget + client_timeout <= techo`.
///
/// **Caller contract (documented, not type-enforced — MAGI S2 re-gate, Caspar):** the
/// "impossible to break by construction" claim holds only for
/// `ceiling_secs >= AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS`. `config.rs` upholds that by validating
/// `[magi].agent_timeout_secs` into the narrower `AGENT_TIMEOUT_MIN_SECS..=AGENT_TIMEOUT_MAX_SECS`
/// range before this ever runs. A hypothetical caller outside that validated path that invokes
/// this `pub` function directly with a ceiling below the absolute floor gets a `budget +
/// client_timeout` that legitimately exceeds `ceiling_secs` — the floors win over the fraction,
/// as `derived_scale_satisfies_invariant_across_the_whole_admissible_range` deliberately
/// exercises down to that exact floor to prove.
#[must_use]
pub fn derive_operation_budget(ceiling_secs: u64) -> Duration {
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let derived = Duration::from_secs((ceiling_secs as f64 * OPERATION_BUDGET_FRACTION) as u64);
    derived.max(MIN_OPERATION_BUDGET)
}

/// Timeout for ONE HTTP request, DERIVED from the ceiling. See [`derive_operation_budget`].
#[must_use]
pub fn derive_client_timeout(ceiling_secs: u64) -> Duration {
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let derived = Duration::from_secs((ceiling_secs as f64 * CLIENT_TIMEOUT_FRACTION) as u64);
    derived.max(MIN_CLIENT_TIMEOUT)
}

/// Ceiling for the classification call (REQ-A07c).
///
/// 6 s: it is ONE label, not a generation. A generous ceiling here cancels the benefit of the
/// cheap route. On slow providers it expires and falls back to `Analysis`/`Default` — it is
/// declared best-effort, and `default_mode` is the no-latency exit.
pub const CLASSIFY_TIMEOUT_SECS: u64 = 6;

/// Ceiling for ONE probe ping (REQ-A24). Per ping, NOT shared deadline.
///
/// 5 s: it is one HTTP request to an endpoint that is typically local, and startup does NOT
/// depend on its result. It is sized by how much extra wait is tolerable at startup, not by the
/// worst case. Well below the 30 s of `DEFAULT_PREFLIGHT_TIMEOUT`.
pub const PROBE_TIMEOUT_SECS: u64 = 5;

/// Input cap OF MAGI-RS, before magi-core (REQ-A11b).
///
/// 256 KiB. The criterion is COST, not capacity: the payload goes to the three mages, so you
/// pay for three. Hand-picked within the 256 KiB–1 MiB range, comfortably below the 4 MiB of
/// `max_input_len` so that the magi-core one never bites.
pub const MAX_QUERY_BYTES: usize = 256 * 1024;

/// Fraction of the measured window used to derive `input_warn_tokens` (REQ-A24b).
///
/// 0.75: the warning must arrive BEFORE getting close to the limit, so a threshold AT the limit
/// warns about nothing. Never 1.0 — it would disable the guardrail on large-context models.
///
/// **Confirmed intended (MAGI S3 re-gate, Caspar): the derived warning is largely vestigial for
/// 128K+-window mages, and that is fine.** With [`MAX_QUERY_BYTES`] at 256 KiB and
/// [`CHARS_PER_TOKEN_EST`], the maximum possible input is ~65K tokens; a mage measured at 128K
/// tokens derives a threshold of ~96K (`0.75 × 128K`), which the input can never reach — the
/// derived warning simply never fires for that mage. This is not a bug: [`MAX_QUERY_BYTES`] is
/// the hard guard that actually bounds what gets sent, and the derived warning's job is to
/// catch the case a large payload approaches a **small** measured window (well under the
/// generous hard cap), which is exactly where it still fires.
pub const WARN_WINDOW_FRACTION: f64 = 0.75;

/// Floor of the closed range of an acceptable probe window (REQ-A16b).
///
/// Out of range degrades to *unmeasured*, it is NEVER clipped to the extreme: a clipped value
/// is used as if it were real. The maximum covers the known large-context models.
pub const PROBE_WINDOW_MIN: usize = 2_048;
/// Ceiling of the range. See [`PROBE_WINDOW_MIN`].
pub const PROBE_WINDOW_MAX: usize = 2_000_000;

/// Ratio that triggers the composition staleness × window notice (SC-A24i).
///
/// 0.8: with the cap **converted to tokens** above 80 % of the measured window, the margin is
/// so small that switching to a smaller model crosses it and the size warning turns off by
/// itself.
pub const STALE_NOTICE_RATIO: f64 = 0.8;

// CALIBRATION, verified against the defaults we ship — not a lone number.
//
// The notice compares `bytes_to_tokens_est(MAX_QUERY_BYTES)` against the MEASURED window. With
// the previous value (512 KiB ⇒ ~131 k estimated tokens) and a mage with a 128 k window, the
// ratio was 1.0 and **the notice fired on EVERY startup of the default configuration** — a
// warning that always appears stops being read, which is worse than not having it.
//
// With 256 KiB (~65 k tokens) against 128 k the ratio is 0.50, comfortably below the threshold:
// the notice again means what it says, "this configuration is tight".
//
// 256 KiB remains inside the §4.9 range (256 KB – 1 MB), remains "a real review diff" and
// **makes the cap ×3 cheaper**, which is the cost criterion of REQ-A11b.

/// Characters per token of the project's shared estimator.
///
/// **It exists because `max_query_bytes` and the measured window are in DIFFERENT UNITS** —
/// bytes against tokens— and comparing them directly compares nothing: the SC-A24i notice would
/// fire or not by arithmetic accident. It is the same value that `[memory]` already uses, and
/// it is a **declared approximation**, not a measurement: the notice names it.
pub const CHARS_PER_TOKEN_EST: f64 = 4.0;

/// Converts a cap in bytes to estimated tokens, so it can be compared with a window.
///
/// Rounds **up**, which is the safe direction: overestimating the payload size makes the notice
/// fire more often, not less.
#[must_use]
pub fn bytes_to_tokens_est(bytes: usize) -> usize {
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let tokens = (bytes as f64 / CHARS_PER_TOKEN_EST).ceil() as usize;
    tokens
}

/// Slack for the headless `--timeout`, as a percentage of the larger term in the formula
/// (§4.9).
pub const HEADLESS_TIMEOUT_HOLGURA_PCT: u64 = 20;

/// Minimum wall-clock for a run that launches consult, **derived at RUNTIME from the CONFIGURED
/// ceiling** (REQ-A04).
///
/// **It is NOT a `const`, and that is the fix.** A constant is calculated over
/// [`AGENT_TIMEOUT_SECS`] —the built-in default— while `[magi].agent_timeout_secs` is
/// **configurable**: an operator raising it to 120 would leave the minimum calculated over 90
/// and
/// the relation would break **at runtime**, which is exactly the failure mode that REQ-A04
/// exists to eliminate. Deriving by construction is useless if it is derived from the wrong
/// value.
///
/// The effective value is provided by `MagiConfig::effective_agent_timeout_secs()` (contract),
/// which is what resolves precedence between the declared value and the default. The two inner
/// layers are DERIVED from that number, never configured (REQ-A04).
///
/// The formula is `classification + 2 × ceiling + slack`. **It is NOT multiplied by 3**: the
/// mages run in parallel (verified, SC-A04e), so the worst case is that of the slowest mage,
/// not the sum of the three.
#[must_use]
pub fn headless_consult_timeout_secs(configured_ceiling: u64) -> u64 {
    let dominant = 2 * configured_ceiling; // the formula's larger/dominant term
    let minimum = CLASSIFY_TIMEOUT_SECS + dominant;
    // §4.9: the slack is 10–30 % of the LARGER TERM, not of the total — over the total it
    // inflates proportionally to the small term, which is not the one that dominates the risk.
    minimum + dominant * HEADLESS_TIMEOUT_HOLGURA_PCT / 100
}

/// Wall-clock decision for a run, with its warning if applicable (SC-A04d).
pub struct TimeoutDecision {
    /// Effective seconds: what the operator asked for, or the derived default.
    pub effective_secs: u64,
    /// Warning when the requested value falls below the formula minimum.
    pub warning: Option<String>,
    /// Goes to the run JSON (REQ-A11d).
    pub below_formula: bool,
}

/// Resolves the run wall-clock. **It always obeys the explicit value**, and warns when that
/// value makes it impossible to complete a consult with schema retry.
#[must_use]
pub fn resolve_run_timeout(asked: Option<u64>, configured_ceiling: u64) -> TimeoutDecision {
    let minimum = headless_consult_timeout_secs(configured_ceiling);
    let Some(secs) = asked else {
        return TimeoutDecision {
            effective_secs: minimum,
            warning: None,
            below_formula: false,
        };
    };
    let below = secs < minimum;
    TimeoutDecision {
        effective_secs: secs,
        warning: below.then(|| {
            format!(
                "warning: --timeout {secs}s is below the {minimum}s the scale requires for \
                 `agent_timeout_secs = {configured_ceiling}`; a consult that needs its schema \
                 retry will NOT complete. Using the requested value anyway."
            )
        }),
        below_formula: below,
    }
}

/// Built-in gate threshold for `CodeReview` (REQ-A20).
///
/// 200 characters. Provenance: the example in the rustdoc of
/// `MagiBuilder::with_complexity_gate`. **It is NOT empirically calibrated** — it is the
/// library author's starting point, not a measurement. The telemetry from REQ-A20 allows tuning
/// it with data.
pub const GATE_CODE_REVIEW: usize = 200;
/// Built-in gate threshold for `Design`. See [`GATE_CODE_REVIEW`]: same provenance, same
/// calibration status.
pub const GATE_DESIGN: usize = 500;
/// Built-in gate threshold for `Analysis` (REQ-A20).
///
/// **It does NOT inherit the \"non-empty\" from magi-core's example, and that deviation is the
/// point:**
/// `Analysis` is the default for every modeless invocation, so a threshold of 1 would turn off
/// the gate on the most common autonomous path. The magi-core gate protects any consumer; ours
/// only sees autonomous routing, where vetoing is the job.
///
/// # Why 200 and not 150
///
/// The first version set 150, which made it **the lowest threshold of the three** — i.e. the
/// mode that is LEAST vetoed. That inverts the argument above: the reasoning says
/// *"it is the path that the gate most needs to cover"* and the number made it the most
/// permissive.
/// A threshold cannot contradict the rustdoc that justifies it.
///
/// It stays **equal to [`GATE_CODE_REVIEW`], not above**. Matching the strictest one would be
/// the other overcorrection: `Analysis` is the widest lens — every general question falls here
/// — and a threshold like [`GATE_DESIGN`] would veto legitimate queries for being short.
/// `Design` remains the highest because an architectural deliberation that can be posed in 300
/// characters almost never needs three perspectives.
///
/// Like the other two: **it is not empirically calibrated**. The telemetry from SC-A20h exists
/// so that the next choice has data instead of another guess.
pub const GATE_ANALYSIS: usize = 200;

/// The two line breaks that separate the mark from the preserved text.
///
/// A constant and not a loose `2` (B4): the number comes from the way the caller appends the
/// mark, and writing it by hand silently decouples it from that shape.
pub const TRUNCATION_SEPARATOR_LEN: usize = 2;

/// Output cap of the default report, on all three routes (REQ-A11b).
///
/// It is born in Phase 0 and not in Phase 6 for the same reason as the other three trimming
/// symbols: `effective_tool_result_cap` consumes it in **Phase 1**.
///
/// The criterion for the number is the same as for the input cap —COST— but the accounting is
/// reversed: the input is paid once for three mages, the output is paid once for each remaining
/// turn of the session, because it lives in the history.
pub const TOOL_RESULT_CAP_BYTES: usize = 64 * 1024;

/// Mark appended to a trimmed report. **A silent trim is indistinguishable from a complete
/// report**, and that is the entire reason it exists.
pub const TRUNCATION_MARK: &str = "[report truncated due to size limit]";

/// Bytes that the mark adds, so that each level deducts its own budget.
///
/// It is DERIVED from the constant instead of written by hand: changing the mark text without
/// moving this number would silently overflow the cap again, and only at the edge.
#[must_use]
pub fn mark_overhead() -> usize {
    TRUNCATION_MARK.len() + TRUNCATION_SEPARATOR_LEN
}

/// Minimum viable output cap: below it, not even the trim mark fits.
///
/// With a cap smaller than [`mark_overhead`], the three levels do `checked_sub` → `None` and
/// the trim **applies nothing**: the report comes out whole, meaning the configured cap is
/// silently ignored. A limit that stops applying when you tighten it is worse than not having
/// one.
#[must_use]
pub fn min_viable_output_cap() -> usize {
    mark_overhead() + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SC-A04 / REQ-A04: the scale is satisfied BY CONSTRUCTION for any ceiling in the range.
    #[test]
    fn derived_scale_satisfies_invariant_across_the_whole_admissible_range() {
        // It starts at the ABSOLUTE FLOOR, not at the §4.9 minimum: the breaking point is below
        // the configurable range, and a sweep that does not cross it proves nothing.
        //
        // Below `AGENT_TIMEOUT_MIN_SECS` (30s) the derived values are dominated by
        // `MIN_OPERATION_BUDGET`/`MIN_CLIENT_TIMEOUT` rather than the 0.6/0.3 fractions — the
        // practical, `config.rs`-validated range an operator can actually reach is 30-120s. The
        // sweep is deliberately WIDER than that: it exists specifically to prove the invariant
        // holds down at the absolute floor too, for any non-`config.rs` caller of
        // `derive_operation_budget`/`derive_client_timeout` (both `pub`) that is not gated by
        // that validation.
        for ceiling in AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS..=AGENT_TIMEOUT_MAX_SECS {
            let budget = derive_operation_budget(ceiling);
            let client = derive_client_timeout(ceiling);
            assert!(
                budget + client <= Duration::from_secs(ceiling),
                "ceiling {ceiling}s: {budget:?} + {client:?} exceeds the ceiling",
            );
            assert!(
                budget >= MIN_OPERATION_BUDGET,
                "ceiling {ceiling}s falls below the floor"
            );
            assert!(
                client >= MIN_CLIENT_TIMEOUT,
                "ceiling {ceiling}s falls below the floor"
            );
        }
    }

    /// REQ-A04: the headless --timeout covers classification + 2 attempts + slack.
    ///
    /// It is DERIVED, not hardcoded: a literal silently desynchronizes as soon as someone moves
    /// `AGENT_TIMEOUT_SECS`, and this test would only detect it in the next commit.
    #[test]
    fn headless_timeout_default_covers_classification_and_two_attempts() {
        let dominant = 2 * AGENT_TIMEOUT_SECS;
        let minimum = CLASSIFY_TIMEOUT_SECS + dominant;
        let holgura = dominant * HEADLESS_TIMEOUT_HOLGURA_PCT / 100;
        assert!(
            headless_consult_timeout_secs(AGENT_TIMEOUT_SECS) >= minimum + holgura,
            "the headless default does not cover {minimum}s + slack",
        );
    }

    /// SC-A04c — the headless `--timeout` respects the formula FOR THE CONFIGURED CEILING.
    ///
    /// It is the same bug that the replaced function had, named by its scenario: a `const` is
    /// tied to the built-in default, not to the value the operator put in `[magi]`.
    #[test]
    fn a_raised_ceiling_raises_the_headless_minimum_too() {
        for ceiling in AGENT_TIMEOUT_MIN_SECS..=AGENT_TIMEOUT_MAX_SECS {
            let dominant = 2 * ceiling;
            let minimum = CLASSIFY_TIMEOUT_SECS + dominant;
            let holgura = dominant * HEADLESS_TIMEOUT_HOLGURA_PCT / 100;
            assert!(
                headless_consult_timeout_secs(ceiling) >= minimum + holgura,
                "ceiling {ceiling}s: the headless minimum does not cover the formula"
            );
        }
        assert!(
            headless_consult_timeout_secs(120) > headless_consult_timeout_secs(90),
            "raising `agent_timeout_secs` MUST raise the minimum; a const would not"
        );
    }

    /// SC-A04d: an explicit `--timeout` below the minimum is OBEYED, with a warning.
    ///
    /// The flag is an operator wall-clock cap, not a safety invariant: whoever asks for
    /// `--timeout 5` wants to cut at 5 seconds, and forcing it to respect the formula would
    /// disobey a clear order. But a value below the minimum guarantees that **no consult with
    /// schema retry completes**, and that is not obvious from the command line.
    #[test]
    fn an_explicit_timeout_below_the_formula_is_obeyed_and_warned_about() {
        let asked = 5_u64;
        let decision = resolve_run_timeout(Some(asked), AGENT_TIMEOUT_SECS);
        assert_eq!(
            decision.effective_secs, asked,
            "the operator's order is obeyed"
        );
        let warning = decision
            .warning
            .expect("a value below the minimum must warn");
        assert!(
            warning.contains(&headless_consult_timeout_secs(AGENT_TIMEOUT_SECS).to_string()),
            "the warning names the minimum the formula required"
        );
        assert!(
            decision.below_formula,
            "and it travels to the JSON: whoever uses the flag runs in a pipeline, i.e. is less \
             likely to read stderr"
        );

        assert!(
            resolve_run_timeout(None, AGENT_TIMEOUT_SECS)
                .warning
                .is_none(),
            "the default does not warn about itself"
        );
        assert!(resolve_run_timeout(Some(1_000), AGENT_TIMEOUT_SECS)
            .warning
            .is_none());
    }

    /// §4.9: every value falls within its admissible range.
    #[test]
    fn plan_values_fall_inside_their_documented_ranges() {
        // The two halves of the defense, placed together so they read as one thing: load
        // validation prevents entering below the range, and the absolute floor sits comfortably
        // below that range — meaning the derivation never sees a ceiling where the floors win.
        //
        // `const` and not a loose `assert!`: the three comparisons in this test that are
        // between constants are evaluated AT COMPILE TIME, so violating them breaks the build
        // rather than a test. It is the strongest guarantee available and it is what clippy
        // asks for with `assertions_on_constants` — the alternative was a `#[allow]`, which
        // only silences the warning and leaves the check where it was.
        const {
            assert!(
                AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS < AGENT_TIMEOUT_MIN_SECS,
                "the configurable range MUST stay above the breaking point; if it ever \
                 did not, load validation would let through a ceiling that breaks the \
                 invariant and no other test would notice"
            );
        }
        assert!(
            (AGENT_TIMEOUT_MIN_SECS..=AGENT_TIMEOUT_MAX_SECS).contains(&AGENT_TIMEOUT_SECS),
            "the range comes from §4.9, not from repeated literals: with `30..=120` \
             written by hand here AND in the sweep above, moving the range leaves the \
             two disagreeing, and the one that fails is the one nobody watches"
        );
        assert!((3..=10).contains(&CLASSIFY_TIMEOUT_SECS));
        assert!((3..=10).contains(&PROBE_TIMEOUT_SECS));
        assert!((256 * 1024..=1024 * 1024).contains(&MAX_QUERY_BYTES));
        const {
            assert!(WARN_WINDOW_FRACTION > 0.0 && WARN_WINDOW_FRACTION < 1.0);
        }
        assert!((10..=30).contains(&HEADLESS_TIMEOUT_HOLGURA_PCT));
        const {
            assert!(PROBE_WINDOW_MIN < PROBE_WINDOW_MAX);
        }
        for t in [GATE_CODE_REVIEW, GATE_DESIGN, GATE_ANALYSIS] {
            assert!(
                t > 1,
                "a threshold of 1 turns off the gate for that mode (REQ-A20)"
            );
        }
    }

    /// SC-A24i: the estimator rounds UP, which is the safe direction.
    ///
    /// Overestimating the payload makes the notice fire more often, never less — and less is
    /// the failure mode that matters, because it silently turns off a warning.
    #[test]
    fn the_token_estimator_rounds_up_and_handles_the_empty_case() {
        assert_eq!(
            bytes_to_tokens_est(0),
            0,
            "an empty payload estimates no tokens"
        );
        assert_eq!(
            bytes_to_tokens_est(1),
            1,
            "a lone byte rounds up to one token, not zero"
        );
        assert_eq!(bytes_to_tokens_est(4), 1, "the exact case does not inflate");
        assert_eq!(
            bytes_to_tokens_est(5),
            2,
            "rounds up as soon as one byte is left over"
        );
    }

    /// The overhead is DERIVED from the mark text instead of written by hand.
    ///
    /// A number written by hand desynchronizes when editing the mark, and the resulting
    /// overflow appears only at the cap edge — i.e. almost never, and without a diagnosis.
    #[test]
    fn the_truncation_overhead_is_derived_from_the_mark_text() {
        assert_eq!(
            mark_overhead(),
            TRUNCATION_MARK.len() + TRUNCATION_SEPARATOR_LEN
        );
        assert!(
            mark_overhead() > TRUNCATION_SEPARATOR_LEN,
            "the mark contributes its own text"
        );
    }

    /// A cap below the overhead would make the trim stop applying SILENTLY.
    ///
    /// With `cap <= mark_overhead()` the three levels do `checked_sub` → `None` and the report
    /// comes out whole: a limit that is ignored when you tighten it is worse than none.
    #[test]
    fn the_minimum_viable_cap_leaves_room_for_the_mark_itself() {
        assert!(
            min_viable_output_cap() > mark_overhead(),
            "below the overhead the trim applies nothing and the cap is silently ignored"
        );
        assert!(
            TOOL_RESULT_CAP_BYTES > min_viable_output_cap(),
            "the built-in default must sit comfortably above the minimum viable value"
        );
    }
}
