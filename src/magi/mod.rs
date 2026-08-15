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
pub mod lineage;
pub mod mode;
pub mod probe;
pub mod report_anchors;
pub mod rotation_config;
pub mod rotation_report;

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
/// exists that breaks `operation_budget + client_timeout <= ceiling`.
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

/// How far below the trio's base a pool candidate's window may sit and still enter the
/// `input_warn_tokens` minimum (REQ-R21).
///
/// Lives here and not in `defaults.rs` for the same reason [`WARN_WINDOW_FRACTION`] does:
/// `defaults.rs` is bin-only, and the derivation that reads this is in the library.
///
/// **0.10 — CHOSEN by the project owner, not measured.** Same honesty as the complexity gate's
/// built-in thresholds: it is a starting point, and saying so is the difference between a number
/// someone can recalibrate and one they assume was derived from data.
///
/// The band exists because including the pool unconditionally would be the wrong kind of
/// conservative: one small-window entry at the END of the list — the candidate least likely to
/// ever run — would pull every run's threshold down and fire the size warning on practically every
/// real consult. **The protection against a candidate too small for the prompt is magi-core's
/// condition #6, never this threshold**, which only warns.
pub const WARN_POOL_TOLERANCE: f64 = 0.10;

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
pub const HEADLESS_TIMEOUT_SLACK_PCT: u64 = 20;

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
/// The effective value is resolved by the CALLER, in `main.rs`:
/// `cfg.magi().agent_timeout_secs.unwrap_or(AGENT_TIMEOUT_SECS)`. The two inner layers are
/// DERIVED from that number, never configured (REQ-A04).
///
/// This used to name a `MagiConfig::effective_agent_timeout_secs()` "contract" that does not
/// exist and never did (S1 Loop 2, Balthasar). The reason it could sit here unnoticed is worth
/// keeping: `MagiConfig` is **bin-only** and this module is in the **lib**, so the reference was
/// prose across a crate boundary — nothing the compiler or `cargo doc` could check, since an
/// intra-doc link to a type this crate cannot see would not have compiled in the first place.
/// Anywhere the lib describes a bin-side path, the name is unverified by construction; spell out
/// the expression rather than a symbol.
///
/// **It is NOT multiplied by 3**: the mages run in parallel (verified, SC-A04e), so the worst
/// case is that of the slowest mage, not the sum of the three.
///
/// # The four nested layers, and why only two of them multiply (REQ-R19/R20, D-R16)
///
/// ```text
/// rotation          1 + max_rotations models        ← multiplies
///  └─ attempt_model 1 ceiling per attempt, x2 if the corrective schema retry fires
///      └─ RetryProvider  operation_budget = ceiling x 0.6, max_retries = 3
///          └─ HTTP client client_timeout  = ceiling x 0.3
/// ```
///
/// **The retry does not multiply the wall-clock**, and that is the part worth stating because it
/// is counter-intuitive. `attempt_model` wraps `agent.execute_with(provider, …)` in ONE
/// `tokio::time::timeout` (`orchestrator.rs:1738`), and that `provider` **is** the seat already
/// wrapped in `RetryProvider` — so the whole retry chain spends budget *inside* a window this
/// formula has already counted. A factor for `max_retries` would over-dimension the timeout by an
/// order of magnitude.
///
/// **And it survives BECAUSE the scale is derived, not by luck.** `RetryConfig`'s own rustdoc
/// warns that its shipped defaults (`operation_budget + client_timeout = 600 + 300 = 900` against
/// a 300 s ceiling) leave every retry **inert** on a hang: the first attempt consumes the whole
/// budget and none of the retries run. magi-rs does not land there because it sets
/// `retry.operation_budget = derive_operation_budget(ceiling)`. Passing a `RetryConfig::default()`
/// without deriving would switch this layer off in silence — no failure, no warning, just no
/// retries.
///
/// # Why a hung model costs ONE ceiling and a schema failure costs TWO
///
/// The two `tokio::time::timeout`s in `attempt_model` are sequential, and the second is only
/// reached if the first **responded** and failed validation. A first attempt that expires returns
/// `Transport { kind: Timeout }` at once and rotates. Hence `retry_disabled` drops the multiplier
/// to one attempt per model rather than halving anything.
///
/// # Arguments
/// * `configured_ceiling` - `[magi].agent_timeout_secs`, resolved.
/// * `max_rotations` - `[magi].max_rotations`, resolved. `0` is the kill-switch and yields the
///   v0.12.0 value by construction, not by a special case.
/// * `retry_disabled` - `[magi].retry_disabled`, resolved.
#[must_use]
pub fn headless_consult_timeout_secs(
    configured_ceiling: u64,
    max_rotations: u32,
    retry_disabled: bool,
) -> u64 {
    // §4.9: the slack is 10-30 % of the LARGER TERM, not of the total — over the total it
    // inflates proportionally to the small term, which is not the one that dominates the risk.
    // `attempt_factor` already folds the slack in (it returns hundredths), so the division by
    // 100 below is what turns it back into seconds.
    //
    // Saturating: this function is `pub` and takes an arbitrary ceiling. The pre-E-B version
    // multiplied by a factor ~100x smaller, so plain arithmetic had far more headroom; folding the
    // hundredths into `attempt_factor` moved the overflow point a hundredfold closer. The
    // validated config path (30..=120) is nowhere near it, but a caller outside that path — and
    // SC-EB04 is one, sweeping up to 3600 — must not wrap.
    CLASSIFY_TIMEOUT_SECS.saturating_add(
        configured_ceiling.saturating_mul(attempt_factor(max_rotations, retry_disabled)) / 100,
    )
}

/// The multiplier both timeout directions share, scaled by 100.
///
/// Returns `attempts * models * (100 + HEADLESS_TIMEOUT_SLACK_PCT)`, i.e. the `x1.2` of the
/// forward formula expressed in integer hundredths so the inverse can divide by it exactly.
///
/// # Why this is one function and not two equivalent expressions
///
/// [`headless_consult_timeout_secs`] and [`derive_ceiling_from_timeout`] are inverses, and the
/// round-trip property that pins them is only true while they agree on this factor. Two
/// equivalent expressions in two places diverge at the first refactor that touches one of them,
/// and neither side's own tests notice — the forward test still passes, the inverse test still
/// passes, and only the round-trip between them is wrong. With one function, changing it moves
/// both directions at once.
///
/// The `100 + HEADLESS_TIMEOUT_SLACK_PCT` is **derived, never a literal `120`**: writing the
/// number by hand means raising the slack updates the forward direction and leaves the inverse
/// silently behind — the exact drift this extraction exists to prevent, reproduced one level down.
///
/// # Arguments
/// * `max_rotations` - `[magi].max_rotations`, **resolved** (effective, not the raw `Option`).
/// * `retry_disabled` - `[magi].retry_disabled`, resolved.
#[must_use]
pub fn attempt_factor(max_rotations: u32, retry_disabled: bool) -> u64 {
    let attempts_per_model: u64 = if retry_disabled { 1 } else { 2 };
    // Saturation point 1 of 3: the model count.
    let models_per_mage = u64::from(max_rotations).saturating_add(1);
    // Saturation point 2 of 3: the product. A saturated factor yields ceiling 0 in the inverse,
    // which floors — the correct degradation. Unreachable in the real range, but an overflow on
    // the path that computes a timeout produces a nonsensical ceiling, so the guard is cheap
    // insurance on the dangerous side.
    attempts_per_model
        .saturating_mul(models_per_mage)
        .saturating_mul(100 + HEADLESS_TIMEOUT_SLACK_PCT)
}

/// The derived ceiling BEFORE the floor is applied, from an already-computed factor.
///
/// Private, and factor-level rather than rotation-level, for two reasons: the floor is not
/// optional on any public path (see [`derive_ceiling_from_timeout`]), and taking the factor
/// directly lets the property tests sweep the whole factor space — covering every slack percentage
/// the constant could hold — without a parallel slack-parameterized copy of each public function.
fn raw_ceiling_from_factor(timeout_secs: u64, factor: u64) -> u64 {
    // `saturating_sub` is the load-bearing guard: with `timeout <= CLASSIFY_TIMEOUT_SECS` a plain
    // subtraction underflows in u64 and yields an ASTRONOMICAL ceiling instead of a minimal one —
    // the error pointing at the opposite of the safe side. The `* 100` undoes the hundredths
    // `attempt_factor` folds in, and overflows only above ~1.8e17 seconds.
    timeout_secs
        .saturating_sub(CLASSIFY_TIMEOUT_SECS)
        .saturating_mul(100)
        / factor
}

/// The smallest `--timeout` whose derived ceiling still reaches
/// [`AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS`], from an already-computed factor.
///
/// # Why this is not `headless_consult_timeout_secs(FLOOR, ..)`
///
/// That answers a **different question** — the timeout the floor ceiling *produces* — and the two
/// coincide only when the division is exact. `raw_ceiling_from_factor` truncates downward, so the
/// forward value can land one second short and feed back as `FLOOR - 1`: the number published as
/// "the `--timeout` that avoids the floor" would be the exact value that trips it. They coincide
/// at the shipped 20 % slack and diverge at 25 % or 33 %, so the bug would be latent, not visible.
///
/// # The `div_ceil` boundary, stated exactly
///
/// [`raw_ceiling_from_factor`] divides with truncation, so reaching a ceiling of `FLOOR` needs a
/// dividend of **at least** `FLOOR * factor` — any less truncates to `FLOOR - 1`. `div_ceil` is
/// what converts "at least" into the smallest integer that satisfies it; plain division would
/// answer the largest that does not.
///
/// The result is therefore both **sufficient** (the threshold does not floor) and **minimal**
/// (`threshold - 1` does), which is exactly the pair of assertions
/// `the_threshold_is_the_exact_boundary_of_floor_activation` makes for every factor. Neither half
/// is decorative: sufficiency alone would be satisfied by any large number, and minimality alone
/// by any small one.
fn threshold_from_factor(factor: u64) -> u64 {
    // `div_ceil`: we need the smallest dividend whose TRUNCATING division still reaches the floor.
    let needed = AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS
        .saturating_mul(factor)
        .div_ceil(100);
    CLASSIFY_TIMEOUT_SECS.saturating_add(needed)
}

/// The smallest `--timeout` that does **not** activate the ceiling floor, for this run's rotation
/// settings (REQ-EB02b).
///
/// It means "minimum to avoid the floor", **not** "minimum for the run to succeed": meeting it
/// guarantees the rotation arithmetic closes, never that the models answer in time.
///
/// The boundary is exact and tested as such: `threshold - 1` derives a ceiling below the floor and
/// `threshold` does not, for every factor — not merely for the slack percentage shipped today.
///
/// # Arguments
/// * `max_rotations` - `[magi].max_rotations`, **resolved** (effective, not the raw `Option`).
/// * `retry_disabled` - `[magi].retry_disabled`, resolved.
#[must_use]
pub fn floor_activation_threshold_secs(max_rotations: u32, retry_disabled: bool) -> u64 {
    threshold_from_factor(attempt_factor(max_rotations, retry_disabled))
}

/// The INVERSE of [`headless_consult_timeout_secs`]: the per-mage ceiling that fits inside an
/// explicit run deadline (REQ-EB01).
///
/// # Why this exists
///
/// `[magi].agent_timeout_secs` is validated into `30..=120`, so the per-attempt budget it derives
/// tops out at 72 s — regardless of how generous the run's `--timeout` is. A caller that grants
/// 1800 s of wall clock still gets 72 s per attempt, and a mage that needs more simply never
/// finishes. Deriving the ceiling from the deadline instead makes the budget scale with what the
/// operator actually granted.
///
/// # Why the floor is mandatory, not defensive
///
/// [`derive_operation_budget`]'s caller contract holds only for
/// `ceiling >= AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS`, and `config.rs` upholds it by validating
/// `agent_timeout_secs` before that function ever runs. **This function is precisely the caller
/// outside that validated path.** Without the floor, a small `--timeout` derives a ceiling below
/// 15 s, both inner layers hit their own minimums, and `operation_budget + client_timeout >
/// ceiling` — REQ-A04's invariant, broken. The floor is what keeps the guarantee true on a path
/// `config.rs` cannot see.
///
/// # Rounding, and why downward is the safe direction
///
/// Integer division truncates. A ceiling one second smaller can never make the run exceed its own
/// `--timeout`; rounding up could. The cost is at most 1 s of budget against a ceiling of
/// hundreds.
///
/// `--timeout 0` means **zero**, not "unbounded": it saturates to the floor like any other
/// insufficient deadline. The convention that 0 means infinite is common elsewhere, which is
/// exactly why this is stated rather than assumed.
///
/// # Arguments
/// * `timeout_secs` - the run's **RESOLVED** wall clock — `TimeoutDecision::effective_secs` from
///   [`resolve_run_timeout`], not the raw flag (REQ-EB03). The two are the same number whenever
///   the operator passed `--timeout`, since that function always obeys an explicit value; taking
///   the resolved one is what makes a future change there propagate here instead of silently
///   leaving the trio on a budget the run no longer uses.
///
///   The caller passes `Some` **only when the operator actually asked**. With no `--timeout`,
///   `resolve_run_timeout` returns the formula minimum, and feeding that back through this inverse
///   lands in `[c - 1, c]` — one second of drift that SC-EB01 forbids. The absent case keeps
///   `[magi].agent_timeout_secs` verbatim and never reaches this function.
/// * `max_rotations` - `[magi].max_rotations`, **resolved** (effective, not the raw `Option`).
/// * `retry_disabled` - `[magi].retry_disabled`, resolved.
#[must_use]
pub fn derive_ceiling_from_timeout(
    timeout_secs: u64,
    max_rotations: u32,
    retry_disabled: bool,
) -> u64 {
    // The floor is applied HERE and again in `BudgetTelemetry::derive`, and the duplication is
    // deliberate rather than an oversight (Checkpoint 2 loop 9, Caspar). `derive` cannot delegate
    // to this function, because it needs the RAW pre-floor value to answer `ceiling_floored` —
    // delegating would return the floored number and the flag could only ever report `false`.
    // So the two share the CONSTANT, not the expression: `AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS` has
    // one definition, and changing the floor's VALUE is still a one-line change in one place.
    // What is duplicated is a `.max()` against it, and the boundary that `.max()` produces is
    // pinned from both sides by `the_threshold_is_the_exact_boundary_of_floor_activation` and
    // `a_ceiling_landing_exactly_on_the_floor_is_not_reported_as_floored`.
    //
    // This is a weaker situation than `attempt_factor`'s and worth not conflating with it: there,
    // two copies of an ARITHMETIC EXPRESSION could drift while each side's tests passed. Here the
    // only thing that could drift is the policy "clamp up to the floor", which is one token.
    raw_ceiling_from_factor(timeout_secs, attempt_factor(max_rotations, retry_disabled))
        .max(AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS)
}

/// Wall-clock decision for a run, with its warning if applicable (SC-A04d).
///
/// `#[allow(clippy::manual_non_exhaustive)]`: clippy's suggested substitute, `#[non_exhaustive]`,
/// is not equivalent here and would silently weaken the guarantee. `#[non_exhaustive]` only blocks
/// struct-literal construction across a CRATE boundary — and `headless_runner.rs` (`main.rs`'s
/// binary crate, reaching this type through `use magi_rs::…`) already sits across one, so that
/// attribute would in fact have caught the specific construction site fixed alongside this seal.
/// What it would NOT catch is a sibling **library** module — `src/headless/`, for instance —
/// reaching in and forging a decision via the same crate's own field visibility. The private
/// `_resolved` field blocks that too, because it is scoped to this module, not to this crate. That
/// wider reach is the guarantee actually wanted, and it is what `#[non_exhaustive]` cannot give.
#[allow(clippy::manual_non_exhaustive)]
pub struct TimeoutDecision {
    /// Effective seconds: what the operator asked for, or the derived default.
    pub effective_secs: u64,
    /// Warning when the requested value falls below the formula minimum.
    pub warning: Option<String>,
    /// Goes to the run JSON (REQ-A11d).
    pub below_formula: bool,
    /// Private marker that seals the struct against a field-by-field literal outside this
    /// module. Only [`resolve_run_timeout`] and [`TimeoutDecision::obeyed`] may construct one —
    /// both live here, and a construction site anywhere else is exactly the accidental literal
    /// this field exists to block. How strong that guarantee is (and is not) is written out in
    /// Task 3 Step 4a.
    _resolved: (),
}

impl TimeoutDecision {
    /// Builds a decision for a value that is **already** the run's effective clock.
    ///
    /// For tests and the TUI path, which have exactly that: an already-resolved timeout with
    /// nothing left to compare against a formula minimum. Production headless code does not have
    /// this — it must go through [`resolve_run_timeout`], which knows the configured ceiling and
    /// can warn when the requested value falls short. Calling `obeyed` from that path would be a
    /// deliberate false statement about a value nobody has actually resolved.
    #[must_use]
    pub const fn obeyed(secs: u64) -> Self {
        Self {
            effective_secs: secs,
            warning: None,
            below_formula: false,
            _resolved: (),
        }
    }
}

/// Resolves the run wall-clock. **It always obeys the explicit value**, and warns when that
/// value makes it impossible to complete a consult with schema retry.
///
/// # A consumer derives the per-mage ceiling from `effective_secs`
///
/// `prepare_headless` in `main.rs` calls this **before** building the trio and feeds
/// `effective_secs` into `BudgetTelemetry::derive`, so this function's answer decides the budget
/// every mage runs under — not just the run's outer deadline (REQ-EB01/EB03).
///
/// That makes the ordering load-bearing: **the resolution must stay ahead of trio construction.**
/// Moving this call after it would put the trio back on a ceiling derived from a number the run
/// does not use, which is the defect the current ordering exists to make unreachable. Changing
/// *what* this function returns is fine and propagates correctly; changing *when* it is called
/// relative to the trio is not.
///
/// Documented here as well as at the call site on purpose: whoever breaks it will be editing this
/// function, not reading that one.
///
/// # Arguments
/// * `asked` - the explicit `--timeout`, if the operator gave one.
/// * `configured_ceiling` - `[magi].agent_timeout_secs`, resolved.
/// * `max_rotations` - `[magi].max_rotations`, resolved (REQ-R20).
/// * `retry_disabled` - `[magi].retry_disabled`, resolved.
#[must_use]
pub fn resolve_run_timeout(
    asked: Option<u64>,
    configured_ceiling: u64,
    max_rotations: u32,
    retry_disabled: bool,
) -> TimeoutDecision {
    let minimum = headless_consult_timeout_secs(configured_ceiling, max_rotations, retry_disabled);
    let Some(secs) = asked else {
        return TimeoutDecision {
            effective_secs: minimum,
            warning: None,
            below_formula: false,
            _resolved: (),
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
        _resolved: (),
    }
}

/// Derived ceiling above which a `--timeout` is more likely a typo than an intention.
///
/// 600 s. **Not a cap** — E-B exists to remove the cap, and this value clamps nothing. It marks
/// the point past which one extra digit is the likelier explanation: `--timeout 18000` typed for
/// `1800` buys ~1500 s per attempt, so a hung mage burns 25 minutes before giving up while the
/// operator believes they asked for 149 s. That is the single error this interface cannot
/// distinguish from a legitimate intention by arithmetic alone, so it is reported and obeyed.
///
/// **Chosen, not measured** — same honesty as the complexity gate's built-in thresholds. It sits
/// well above any ceiling a plausible `--timeout` derives (1800 s derives 249) and well below what
/// a fat-fingered order of magnitude produces.
///
/// **Operationally tunable.** It gates a warning and clamps nothing, so moving it changes only how
/// noisy the run is, never what the run does. If real deployments legitimately derive ceilings
/// above it, raise it — a warning that fires on correct configurations trains operators to ignore
/// warnings, which costs more than the typo it was meant to catch.
pub const CEILING_SANITY_SECS: u64 = 600;

/// A per-mage ceiling that **names where it came from** (REQ-EB01, R-EB03), paired with
/// [`BudgetTelemetry`] by [`BudgetTelemetry::derive`] so no caller can obtain a ceiling without
/// also seeing why it was chosen.
///
/// # Why a newtype instead of a `u64`
///
/// The correctness of E-B is temporal: the run's clock must be resolved *before* the trio is
/// built, or the mages get a budget derived from a number the run does not enforce. An earlier
/// design guarded that with a startup `assert!`, then with rustdoc at both ends. Both are checks
/// a refactor can walk past — the `assert!` because it only fires at runtime on a divergence that
/// cannot happen *yet*, the rustdoc because it lives in the file nobody is editing.
///
/// A `u64` in `TrioBuild` accepts any number from anywhere: the raw `--timeout`, a tier default,
/// a literal. This type does not. Every value is produced by one of exactly two routes, and
/// **each route states which path it represents**:
///
/// * [`BudgetTelemetry::derive`] with `Some(&TimeoutDecision)` — the derived path. The
///   `TimeoutDecision` is obtainable only from [`resolve_run_timeout`] or the explicitly-named
///   [`TimeoutDecision::obeyed`], so a bare `--timeout` cannot reach it.
/// * [`ResolvedCeiling::configured`] — the configured/TUI path, where **no clock is resolved at
///   all** and `[magi].agent_timeout_secs` is used verbatim. This is the same value
///   `derive(None, secs, …)` returns; it exists as a named constructor because otherwise every
///   unrelated test literal would have to call the deriver to obtain one.
///
/// # What this does and does not guarantee — read before strengthening the claim
///
/// It makes a ceiling **traceable**, not provably-resolved. `ResolvedCeiling::configured(raw)`
/// is one line and it compiles. What the type removes is the *silent* version: a bare integer
/// flowing into the trio with nothing at the call site saying which clock it came from. Anyone
/// bypassing it now has to write a constructor whose name asserts something false, in a function
/// whose rustdoc says so. That is a speed bump plus documentation — good, and worth having — but
/// it is **not** a proof.
///
/// It deliberately does **not** implement `From<u64>`, `Default` or `Deserialize`. `From<u64>`
/// would erase the provenance the two named routes exist to record, and `Deserialize` constructs
/// field-by-field without any constructor at all — the exact bypass `CLAUDE.md` records for
/// `MagiConfig`.
///
/// The inner value is private; call sites read it back with [`ResolvedCeiling::secs`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedCeiling(u64);

impl ResolvedCeiling {
    /// The resolved ceiling, in seconds.
    #[must_use]
    pub const fn secs(self) -> u64 {
        self.0
    }

    /// Builds a `ResolvedCeiling` directly from a value already known to be the effective
    /// ceiling — the construction route for callers (tests, the TUI path) that already have the
    /// number and only need the type, alongside [`BudgetTelemetry::derive`] for the derived path.
    #[must_use]
    pub const fn configured(secs: u64) -> Self {
        Self(secs)
    }
}

/// What the run's budget derivation decided, in a form a machine can read (REQ-EB02b).
///
/// # Why these five and not a boolean
///
/// A consumer that only learns *that* something degraded knows something is wrong and not what to
/// ask for. [`Self::floor_activation_threshold_secs`] lets it remediate itself — retry with that
/// value, or report it — without replicating our arithmetic on its side. A consumer forced to
/// replicate the formula is a consumer that will replicate it wrong; the review of this very
/// design found two arithmetic errors in that same calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
pub struct BudgetTelemetry {
    /// Per-attempt budget governing this run.
    pub operation_budget_secs: u64,
    /// The derived ceiling was raised to [`AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS`].
    ///
    /// **Strictly `derived < FLOOR`.** A ceiling that lands exactly on the floor was reached, not
    /// clamped. `<` and `<=` read identically and disagree in exactly the boundary case.
    pub ceiling_floored: bool,
    /// The smallest `--timeout` that does **not** activate the floor, under this run's rotation
    /// settings.
    ///
    /// **Present always**, not only when degraded: its most useful reading is how much margin
    /// remains *before* degrading, which a field that only appears after the fact cannot give.
    ///
    /// It means "minimum to avoid the floor", **not** "minimum for the run to succeed". Meeting it
    /// guarantees the rotation arithmetic closes, never that the models answer in time — a slow
    /// mage can still exhaust its budget far above this number, which is the very case that
    /// motivated E-B.
    pub floor_activation_threshold_secs: u64,
    /// The rotation count the formula actually used, so a consumer can evaluate the second lever
    /// (fewer rotations buy a larger budget at the same clock) without guessing our resolution.
    pub max_rotations_effective: u32,
    /// The derived ceiling exceeded [`CEILING_SANITY_SECS`] — a probable `--timeout` typo. The
    /// value is **not** clamped.
    pub ceiling_above_sanity: bool,
}

impl BudgetTelemetry {
    /// Resolves the effective ceiling for a run and describes the decision.
    ///
    /// Returns `(ResolvedCeiling, telemetry)`. **They are returned together on purpose:**
    /// the ceiling governs the run and the telemetry explains it, and a caller that could obtain
    /// one without the other could ship a budget nobody can see — which is the defect E-B was
    /// filed against.
    ///
    /// # Arguments
    /// * `run` - the run's **resolved** wall clock, or `None` for the configured/TUI path, which
    ///   keeps `configured_ceiling` untouched and is byte-identical to v0.14.3.
    ///   **It is a [`TimeoutDecision`], not a `u64`, and that is deliberate**: the raw `--timeout`
    ///   flag cannot reach the derived path directly, only the already-resolved decision can.
    /// * `configured_ceiling` - `[magi].agent_timeout_secs`, resolved.
    /// * `max_rotations` - resolved (effective) rotation count.
    /// * `retry_disabled` - `[magi].retry_disabled`, resolved.
    #[must_use]
    pub fn derive(
        run: Option<&TimeoutDecision>,
        configured_ceiling: u64,
        max_rotations: u32,
        retry_disabled: bool,
    ) -> (ResolvedCeiling, Self) {
        // The raw derivation runs ONCE, and the ceiling is the floored form of it. An earlier
        // draft called `derive_ceiling_from_timeout` and then recomputed the raw value separately,
        // running the same arithmetic twice — harmless numerically, but two call sites that must
        // agree is the shape of defect this whole change is about.
        //
        // Strictly `<` for `ceiling_floored`: a raw derivation landing exactly on the floor was
        // reached, not clamped.
        let raw = run
            .map(|d| {
                raw_ceiling_from_factor(
                    d.effective_secs,
                    attempt_factor(max_rotations, retry_disabled),
                )
            })
            .unwrap_or(configured_ceiling);
        // The floor applies to BOTH branches, and the `None` one is not defensive padding.
        // `config.rs` validates `agent_timeout_secs` into `30..=120`, so a below-floor configured
        // ceiling is unreachable through the loaded config — but this function is `pub`, and
        // `derive_operation_budget`'s own rustdoc already records that its "impossible by
        // construction" claim holds only above the floor. Leaving `None` unfloored would keep a
        // path on which REQ-A04's invariant is breakable, guarded by a precondition living in a
        // different module. One `.max()` makes it hold universally, and inside the validated
        // range it changes nothing at all.
        let ceiling = raw.max(AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS);
        (
            ResolvedCeiling(ceiling),
            Self {
                operation_budget_secs: derive_operation_budget(ceiling).as_secs(),
                ceiling_floored: raw < AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS,
                floor_activation_threshold_secs: floor_activation_threshold_secs(
                    max_rotations,
                    retry_disabled,
                ),
                max_rotations_effective: max_rotations,
                ceiling_above_sanity: ceiling > CEILING_SANITY_SECS,
            },
        )
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
        let slack = dominant * HEADLESS_TIMEOUT_SLACK_PCT / 100;
        assert!(
            headless_consult_timeout_secs(AGENT_TIMEOUT_SECS, 0, false) >= minimum + slack,
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
            let slack = dominant * HEADLESS_TIMEOUT_SLACK_PCT / 100;
            assert!(
                headless_consult_timeout_secs(ceiling, 0, false) >= minimum + slack,
                "ceiling {ceiling}s: the headless minimum does not cover the formula"
            );
        }
        assert!(
            headless_consult_timeout_secs(120, 0, false)
                > headless_consult_timeout_secs(90, 0, false),
            "raising `agent_timeout_secs` MUST raise the minimum; a const would not"
        );
    }

    /// SC-R31: the multiplier follows BOTH configuration keys, and **no new key is added** — the
    /// worst case is calculated from what the operator already declared in `magi.toml`.
    ///
    /// A `--timeout` that ignores either one cuts off healthy consults, and the symptom would
    /// appear **only when a rotation also happened**: very hard to reproduce and very easy to
    /// blame on the wrong model.
    #[test]
    fn the_multiplier_follows_both_config_keys() {
        // Defaults: 2 attempts x (1 + 2 rotations) models x 90 s = 540 s dominant.
        assert_eq!(
            headless_consult_timeout_secs(90, 2, false),
            CLASSIFY_TIMEOUT_SECS + 540 + 540 * HEADLESS_TIMEOUT_SLACK_PCT / 100
        );
        // `retry_disabled` ⇒ ONE attempt per model: magi-core returns the schema outcome
        // immediately instead of spending a second ceiling on the corrective retry.
        assert_eq!(
            headless_consult_timeout_secs(90, 2, true),
            CLASSIFY_TIMEOUT_SECS + 270 + 270 * HEADLESS_TIMEOUT_SLACK_PCT / 100
        );
    }

    /// SC-R18/SC-R04: the kill-switch restores the v0.12.0 value EXACTLY — **by construction**,
    /// not by a special case in the formula. `1 + max_rotations` with `max_rotations = 0` is one
    /// model, which is what the formula multiplied by before rotation existed.
    #[test]
    fn the_kill_switch_restores_the_v0_12_0_timeout_exactly() {
        assert_eq!(headless_consult_timeout_secs(90, 0, false), 222);
    }

    /// SC-R53: the retry chain lives INSIDE one ceiling per attempt (D-R16, verified against
    /// `orchestrator.rs:1738`), so the formula carries **no factor for `max_retries`**.
    ///
    /// `attempt_model` wraps `agent.execute_with(provider, …)` in ONE `tokio::time::timeout`, and
    /// that `provider` is the seat **already wrapped** in `RetryProvider` — so the retries spend
    /// budget inside a window the formula has already counted. Adding a factor for them would
    /// over-dimension the wall-clock by an order of magnitude.
    #[test]
    fn the_formula_carries_no_max_retries_factor() {
        let with_defaults = headless_consult_timeout_secs(90, 2, false);
        assert!(
            with_defaults < 1_000,
            "a max_retries factor would push this past 2000 s; got {with_defaults}"
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
        let decision = resolve_run_timeout(Some(asked), AGENT_TIMEOUT_SECS, 0, false);
        assert_eq!(
            decision.effective_secs, asked,
            "the operator's order is obeyed"
        );
        let warning = decision
            .warning
            .expect("a value below the minimum must warn");
        assert!(
            warning
                .contains(&headless_consult_timeout_secs(AGENT_TIMEOUT_SECS, 0, false).to_string()),
            "the warning names the minimum the formula required"
        );
        assert!(
            decision.below_formula,
            "and it travels to the JSON: whoever uses the flag runs in a pipeline, i.e. is less \
             likely to read stderr"
        );

        assert!(
            resolve_run_timeout(None, AGENT_TIMEOUT_SECS, 0, false)
                .warning
                .is_none(),
            "the default does not warn about itself"
        );
        assert!(
            resolve_run_timeout(Some(1_000), AGENT_TIMEOUT_SECS, 0, false)
                .warning
                .is_none()
        );
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
        assert!((10..=30).contains(&HEADLESS_TIMEOUT_SLACK_PCT));
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

    /// SC-EB02: a generous `--timeout` unlocks a budget far above the 72 s that the
    /// validated `agent_timeout_secs` range could ever reach. This is the requirement.
    #[test]
    fn a_generous_timeout_derives_a_ceiling_above_the_configured_maximum() {
        // 2 attempts x 3 models x 1.2 slack = 7.2; (1800 - 6) / 7.2 = 249.16 -> 249
        let ceiling = derive_ceiling_from_timeout(1800, 2, false);
        assert_eq!(
            ceiling, 249,
            "the inverse of the formula that produced 1800"
        );
        assert!(
            ceiling > AGENT_TIMEOUT_MAX_SECS,
            "the whole point of E-B: the derived path is NOT capped by the configured range"
        );
        assert_eq!(derive_operation_budget(ceiling).as_secs(), 149);
    }

    /// SC-EB04: round-trip, bounded to the range where it is TRUE. The inverse never
    /// returns MORE than the ceiling that produced the timeout — integer division
    /// truncates downward, which is the safe direction.
    ///
    /// **The upper bound is `u64::MAX / factor`, and it is not decorative.** Above it the
    /// forward direction's `saturating_mul` clamps, so the timeout no longer encodes the
    /// ceiling and the inverse cannot recover it. Stating the bound is the difference
    /// between a property and a claim that happens to hold on the samples chosen —
    /// `the_round_trip_breaks_above_its_stated_bound` pins the other side.
    #[test]
    fn the_round_trip_never_overshoots_the_ceiling_that_produced_it() {
        for ceiling in [15_u64, 16, 30, 45, 90, 120, 249, 600, 3600, 86_400] {
            for max_rotations in [0_u32, 1, 2, 5] {
                for retry_disabled in [false, true] {
                    let factor = attempt_factor(max_rotations, retry_disabled);
                    assert!(
                        ceiling <= u64::MAX / factor,
                        "sample outside the stated bound"
                    );
                    let t = headless_consult_timeout_secs(ceiling, max_rotations, retry_disabled);
                    let back = derive_ceiling_from_timeout(t, max_rotations, retry_disabled);
                    assert!(
                        back <= ceiling && back >= ceiling.saturating_sub(1),
                        "round-trip out of band: ceiling={ceiling} rotations={max_rotations} \
                         retry_disabled={retry_disabled} timeout={t} back={back}"
                    );
                }
            }
        }
    }

    /// And OUTSIDE the bound the property is false — asserted rather than left implied,
    /// because a bound that lives only in a doc comment drifts without failing anything.
    #[test]
    fn the_round_trip_breaks_above_its_stated_bound() {
        let factor = attempt_factor(2, false);
        let beyond = u64::MAX / factor + 1;
        let t = headless_consult_timeout_secs(beyond, 2, false);
        let back = derive_ceiling_from_timeout(t, 2, false);
        assert!(
            back < beyond,
            "saturation must LOSE information here; if this ever passes as an equality the \
             bound moved and the doc comments above are wrong"
        );
    }

    /// SC-EB04c: the complement of the round-trip. BELOW the floor the property is
    /// false by design — the floor wins — so that half of the domain gets its own
    /// assertion instead of being silently excluded.
    #[test]
    fn a_ceiling_below_the_absolute_floor_comes_back_as_exactly_the_floor() {
        for ceiling in [1_u64, 5, 10, 14] {
            let t = headless_consult_timeout_secs(ceiling, 2, false);
            assert_eq!(
                derive_ceiling_from_timeout(t, 2, false),
                AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS,
                "below the floor the round-trip is FALSE on purpose: REQ-A04's invariant \
                 outranks it"
            );
        }
    }

    /// SC-EB04b: effective vs configured `max_rotations` differ when the TOML declares
    /// nothing — the DEFAULT case, not a rare one. Both directions must be fed the same
    /// number or the round-trip breaks in the configuration almost everyone has.
    ///
    /// **The literal 2 is deliberate and must NOT be replaced by
    /// `crate::defaults::DEFAULT_MAX_ROTATIONS`.** `mod defaults` is declared only in
    /// `src/main.rs`, so it is bin-only and unreachable from this library module — the
    /// reference would not compile. Naming the coupling here is the substitute for the
    /// intra-crate link Rust cannot give us across that boundary.
    ///
    /// **A comment is not enough on its own, and the failure mode is silent:** if the default
    /// moves from 2 to 3, this test keeps passing — it simply stops testing the DEFAULT case
    /// and starts testing an arbitrary rotation count, while its name still claims otherwise
    /// (Checkpoint 2 loop 10, Caspar). The mechanical half of the guard therefore lives on the
    /// BIN side, where the constant is visible: see
    /// `the_lib_side_default_rotations_mirror_is_still_accurate` in `src/main.rs`, which fails
    /// if the two ever diverge. Neither test can enforce this alone; the pair can.
    #[test]
    fn both_directions_agree_when_max_rotations_comes_from_the_default() {
        // Mirrors `defaults::DEFAULT_MAX_ROTATIONS` (bin-only; see above).
        let effective = 2_u32;
        let t = headless_consult_timeout_secs(90, effective, false);
        assert_eq!(derive_ceiling_from_timeout(t, effective, false), 90);
    }

    /// SC-EB04d: the slack factor is DERIVED, never a literal. `attempt_factor` is the
    /// single place both directions read it from, so this test pins that they share it.
    #[test]
    fn the_slack_factor_is_shared_by_both_directions() {
        // attempt_factor returns attempts * models * (100 + SLACK_PCT).
        assert_eq!(
            attempt_factor(2, false),
            2 * 3 * (100 + HEADLESS_TIMEOUT_SLACK_PCT),
            "written as a literal 120, a change to HEADLESS_TIMEOUT_SLACK_PCT would move \
             the forward function and silently leave the inverse behind"
        );
        // 1 attempt x 1 model x slack factor: the identity multiplications are elided because
        // `clippy::identity_op` denies them under `-D warnings`; the value is unchanged.
        assert_eq!(attempt_factor(0, true), 100 + HEADLESS_TIMEOUT_SLACK_PCT);
    }

    /// SC-EB05c: the degenerate values. Without `saturating_sub`, `timeout - 6` underflows
    /// in u64 and the ceiling comes out ASTRONOMICAL instead of minimal — the error
    /// pointing at exactly the wrong side. `--timeout 0` means zero, never "unbounded".
    #[test]
    fn timeouts_at_or_below_the_classify_cost_produce_the_floor_and_never_underflow() {
        for timeout in [0_u64, 1, 5, 6, 7] {
            let ceiling = derive_ceiling_from_timeout(timeout, 2, false);
            assert_eq!(
                ceiling, AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS,
                "--timeout {timeout} must floor, not wrap"
            );
        }
    }

    /// SC-EB05: REQ-A04's invariant survives the derived path across the representable
    /// range. The bound is `u64::MAX / 100` because the formula multiplies by 100 before
    /// dividing — stating the bound rather than claiming a universality that is false.
    #[test]
    fn the_derived_scale_upholds_req_a04_across_the_representable_range() {
        let samples = [
            0_u64,
            1,
            6,
            7,
            60,
            90,
            600,
            1800,
            2166,
            86_400,
            1_000_000,
            u64::MAX / 100,
        ];
        for timeout in samples {
            for max_rotations in [0_u32, 1, 2, 7] {
                for retry_disabled in [false, true] {
                    let c = derive_ceiling_from_timeout(timeout, max_rotations, retry_disabled);
                    let sum =
                        derive_operation_budget(c).as_secs() + derive_client_timeout(c).as_secs();
                    assert!(
                        sum <= c,
                        "REQ-A04 broken: timeout={timeout} rotations={max_rotations} \
                         retry_disabled={retry_disabled} ceiling={c} sum={sum}"
                    );
                }
            }
        }
    }

    // -- Task 2: `BudgetTelemetry` — the five observable values ------------------------------

    /// The telemetry reports the ceiling the run ACTUALLY got, and reports it whether or
    /// not the derived path was taken.
    #[test]
    fn the_telemetry_reports_the_effective_ceiling_and_its_budget() {
        let (ceiling, t) =
            BudgetTelemetry::derive(Some(&TimeoutDecision::obeyed(1800)), 90, 2, false);
        assert_eq!(ceiling.secs(), 249);
        assert_eq!(t.operation_budget_secs, 149);
        assert!(!t.ceiling_floored);
        assert!(!t.ceiling_above_sanity);
        assert_eq!(t.max_rotations_effective, 2);
    }

    /// SC-EB01: with no `--timeout`, the ceiling is the configured one and the budget is
    /// byte-identical to v0.14.3. Verified by MUTATION: change the fallback to anything
    /// else and this test must go red.
    #[test]
    fn without_an_explicit_timeout_nothing_changes_at_all() {
        for configured in [30_u64, 45, 90, 120] {
            let (ceiling, t) = BudgetTelemetry::derive(None, configured, 2, false);
            assert_eq!(
                ceiling.secs(),
                configured,
                "the configured ceiling, untouched"
            );
            assert_eq!(
                t.operation_budget_secs,
                derive_operation_budget(configured).as_secs()
            );
            assert!(!t.ceiling_floored);
            assert!(!t.ceiling_above_sanity);
        }
    }

    /// REQ-A04 holds on the `None` path too, for a configured ceiling BELOW the floor.
    ///
    /// `config.rs` validates `agent_timeout_secs` into `30..=120`, so this is unreachable
    /// through the loaded config — but `derive` is `pub`, and a precondition enforced in
    /// another module is a precondition someone can walk around. The sweep above covers
    /// the validated range; this covers the hole underneath it.
    #[test]
    fn the_invariant_holds_for_a_below_floor_configured_ceiling() {
        for configured in [0_u64, 1, 10, 14, 15] {
            let (ceiling, t) = BudgetTelemetry::derive(None, configured, 2, false);
            assert!(ceiling.secs() >= AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS);
            let c = ceiling.secs();
            let sum = derive_operation_budget(c).as_secs() + derive_client_timeout(c).as_secs();
            assert!(sum <= c, "REQ-A04 broken at configured={configured}");
            // `ceiling_floored` means the same thing on BOTH paths: the value was RAISED to the
            // floor. It is not "the --timeout was too small" — it is "the ceiling did not reach
            // the floor on its own", whatever produced it. Strictly `<`, so 15 is not floored.
            assert_eq!(
                t.ceiling_floored,
                configured < AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS,
                "at configured={configured}"
            );
        }
    }

    /// An absurd `max_rotations` degrades to the floor rather than wrapping. `attempt_factor`
    /// saturates, the saturated divisor drives the raw ceiling to 0, and the floor catches it —
    /// so the "no upper bound on max_rotations" the gate noted is bounded in effect even though
    /// it is not bounded in type.
    #[test]
    fn an_absurd_rotation_count_degrades_to_the_floor() {
        for max_rotations in [1_000_u32, u32::MAX / 2, u32::MAX] {
            let (ceiling, t) = BudgetTelemetry::derive(
                Some(&TimeoutDecision::obeyed(1800)),
                90,
                max_rotations,
                false,
            );
            assert_eq!(ceiling.secs(), AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS);
            assert!(t.ceiling_floored, "and it is reported, not silent");
        }
    }

    /// SC-EB05d: what `floor_activation_threshold_secs` REPORTS when the rotation count makes the
    /// floor unavoidable — the case the test above leaves unexamined.
    ///
    /// **Why this is not a pedantic edge case** (Checkpoint 2 loop 13, Caspar). With a saturated
    /// factor the threshold comes out astronomically large. That number is *arithmetically right* —
    /// there genuinely is no `--timeout` that avoids the floor at this rotation count — but the
    /// field's whole promise is that a consumer can **auto-remediate without replicating our
    /// arithmetic**. A CI script that reads it and retries with that value retries with something
    /// unreachable, and then misdiagnoses the failure. Publishing a number that cannot be acted on
    /// contradicts the reason the field exists.
    ///
    /// **The resolution is guidance, not a clamp.** Clamping would publish a *false* threshold — a
    /// value that looks achievable and is not — which is worse than an obviously absurd one. What
    /// makes the honest number actionable is the SECOND lever, already published beside it:
    /// `max_rotations_effective`. When the threshold is unreachable the remediation is to lower
    /// rotations, never to raise `--timeout`, and the consumer has both numbers to see that.
    ///
    /// So this test pins the property and the co-reported signal, not a magic literal. **Record the
    /// observed value in a comment when writing it**, so the next reader sees the magnitude without
    /// re-deriving it.
    #[test]
    fn an_unreachable_threshold_is_reported_honestly_beside_the_lever_that_fixes_it() {
        for max_rotations in [u32::MAX / 2, u32::MAX] {
            let (_, t) = BudgetTelemetry::derive(
                Some(&TimeoutDecision::obeyed(1800)),
                90,
                max_rotations,
                false,
            );
            // Unreachable by construction: no u64 `--timeout` can satisfy it.
            assert!(
                t.floor_activation_threshold_secs > u64::from(u32::MAX),
                "at max_rotations={max_rotations} the threshold should be unreachable, not plausible"
            );
            // And the two signals a consumer needs to reach the RIGHT conclusion are both present.
            assert!(
                t.ceiling_floored,
                "the run IS degraded, so the threshold is not being reported about a healthy run"
            );
            assert_eq!(
                t.max_rotations_effective, max_rotations,
                "the lever that actually fixes this must be reported beside the threshold that cannot"
            );
        }
    }

    /// SC-EB03: a small `--timeout` LOWERS the ceiling, it never raises it. `max(configured,
    /// derived)` would let a 120 s ceiling run inside a 60 s deadline — killed by the outer
    /// clock with certainty.
    #[test]
    fn a_small_timeout_lowers_the_ceiling_instead_of_keeping_the_configured_one() {
        let (ceiling, t) =
            BudgetTelemetry::derive(Some(&TimeoutDecision::obeyed(60)), 120, 2, false);
        assert_eq!(ceiling.secs(), AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS);
        assert!(
            ceiling.secs() < 120,
            "the explicit deadline governs; it must not be ignored"
        );
        assert!(t.ceiling_floored);
    }

    /// `ceiling_floored` is `derived < FLOOR`, STRICT. A ceiling that lands exactly on the
    /// floor was reached, not clamped — `<` and `<=` are indistinguishable to read and
    /// produce different reports in precisely the boundary case.
    #[test]
    fn a_ceiling_landing_exactly_on_the_floor_is_not_reported_as_floored() {
        // The timeout whose exact inverse is the floor: 15 * 720 / 100 + 6 = 114.
        let exact = headless_consult_timeout_secs(AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS, 2, false);
        let (ceiling, t) =
            BudgetTelemetry::derive(Some(&TimeoutDecision::obeyed(exact)), 90, 2, false);
        assert_eq!(ceiling.secs(), AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS);
        assert!(!t.ceiling_floored, "reached, not clamped");

        let (_, below) =
            BudgetTelemetry::derive(Some(&TimeoutDecision::obeyed(exact - 1)), 90, 2, false);
        assert!(below.ceiling_floored, "one second under IS clamped");
    }

    /// `floor_activation_threshold_secs` is present ALWAYS, not only when degraded — its
    /// most useful reading is "how much margin do I have before I degrade".
    #[test]
    fn the_floor_activation_threshold_is_always_present() {
        for asked in [None, Some(60_u64), Some(1800)] {
            let decision = asked.map(TimeoutDecision::obeyed);
            let (_, t) = BudgetTelemetry::derive(decision.as_ref(), 90, 2, false);
            assert_eq!(
                t.floor_activation_threshold_secs,
                floor_activation_threshold_secs(2, false),
                "for asked={asked:?}"
            );
        }
    }

    /// **The threshold must actually WORK.** It is NOT
    /// `headless_consult_timeout_secs(FLOOR, ..)` — that is the timeout the floor ceiling
    /// *produces*, a different question from the smallest one that *avoids* the floor, and
    /// the two coincide only when the division is exact. At slack 25 % or 33 % they do not,
    /// and the field would publish the exact `--timeout` that trips the condition it exists
    /// to help operators avoid.
    ///
    /// Asserted directly rather than against a formula: one second below trips the floor,
    /// the threshold itself does not.
    ///
    /// **Swept over FACTORS, not over slack percentages.** `attempt_factor` is the only
    /// place `HEADLESS_TIMEOUT_SLACK_PCT` is read, so a factor sweep covers every slack
    /// the constant could ever hold — including 750 and 798, the values for 25 % and 33 %
    /// that break the old formulation — without a parallel slack-parameterized copy of
    /// every function.
    #[test]
    fn the_threshold_is_the_exact_boundary_of_floor_activation() {
        // 720 = the shipped 20 %; 750 and 798 are 25 % and 33 %, where the old
        // `headless_consult_timeout_secs(FLOOR, ..)` formulation returned a value that
        // floored. The rest sweep rotation counts and the retry-disabled multiplier.
        for factor in [110_u64, 120, 240, 360, 720, 750, 798, 1440, 4788] {
            let t = threshold_from_factor(factor);
            assert!(
                raw_ceiling_from_factor(t, factor) >= AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS,
                "the threshold itself must NOT floor: factor={factor} threshold={t}"
            );
            assert!(
                raw_ceiling_from_factor(t - 1, factor) < AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS,
                "and it must be the SMALLEST such value, or it overstates what is needed: \
                 factor={factor} threshold={t}"
            );
        }
    }

    /// SC-EB04c: the public threshold is composed, not hardcoded.
    ///
    /// **Asserted against a number derived OUTSIDE the function**, not against the function's own
    /// body. The previous version compared `floor_activation_threshold_secs(r, d)` to
    /// `threshold_from_factor(attempt_factor(r, d))` — which is that function's definition, so it
    /// held no matter what either helper computed, and it passed unchanged during the Red phase.
    ///
    /// 114 is the spec's documented boundary for the shipped configuration:
    /// `2 attempts x 3 models x 15 s floor x 1.2 slack + 6 s classify = 114`.
    #[test]
    fn the_public_threshold_composes_attempt_factor_with_the_boundary() {
        assert_eq!(
            floor_activation_threshold_secs(2, false),
            114,
            "the shipped configuration's floor-activation boundary"
        );
    }

    /// The sanity flag catches the likeliest error in this whole interface: one extra
    /// digit. `--timeout 18000` for `1800` buys ~1500 s per attempt, and a hung mage would
    /// burn 25 minutes before giving up. It WARNS and obeys — there is no upper clamp,
    /// that is the requirement.
    #[test]
    fn a_ceiling_above_the_sanity_threshold_is_flagged_but_never_clamped() {
        let (ceiling, t) =
            BudgetTelemetry::derive(Some(&TimeoutDecision::obeyed(18_000)), 90, 2, false);
        assert!(ceiling.secs() > CEILING_SANITY_SECS);
        assert!(t.ceiling_above_sanity, "a probable typo must be observable");
        assert_eq!(
            ceiling.secs(),
            derive_ceiling_from_timeout(18_000, 2, false),
            "flagged, NOT clamped: an upper bound is exactly what E-B removes"
        );
    }
}
