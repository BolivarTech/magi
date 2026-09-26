// Author: Julian Bolivar
// Version: 0.20.0
// Date: 2026-09-26

//! REQ-EE-5's per-activation clock-coverage check (S-9): whether the effective per-request
//! client timeout can deliver `[magi].max_tokens` at magi-core's measured reference generation
//! speed, and if not, which existing knob (`[magi].agent_timeout_secs` or `--timeout`) the
//! operator would need to raise, and to what, to cover it.
//!
//! This module is the PURE arithmetic only: deciding whether a clock covers a cap and
//! rendering the operator-facing line. Emitting it (through `tracing`, the process auditor, and
//! a surface's `NoticeSink` fallback, `crate::agent::mode_classifier::NoticeSink`, a bin-crate
//! type, so not linkable from this lib module) is `ConsultTool`'s `ClockCoverageAnnouncer`, in
//! the bin crate, because `NoticeSink` lives there.

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

/// Reference generation speed, tokens per second, for REQ-EE-5's coverage check.
///
/// Copied from magi-core 4.2.0's `MIN_MEASURED_TOK_S` (`src/provider.rs:1323`, measured
/// 2026-09-20, magi-core's own floor across its measured model families), which is
/// `pub(crate)` and `#[cfg(test)]`: it cannot be imported, so no compile-time coupling to the
/// upstream value is possible. The debt this leaves is held by
/// `tests/magi_core_contract.rs::the_reference_speed_is_magi_core_4_2_0s_measured_floor_and_the_pin_is_4_2_0`,
/// which re-states the value next to its upstream source and additionally pins the resolved
/// `magi-core` version from `Cargo.lock`, so a pin bump forces re-reading the upstream constant
/// before this line can be trusted again.
pub const COVERAGE_REFERENCE_TOK_PER_SEC: u64 = 55;

/// `tracing` target of the coverage WARN.
///
/// Its own target, separate from every other emitter in this crate, so a REQ-L30/L31 per-target
/// directive can raise or silence this one class of notice without touching any other.
pub const CLOCK_COVERAGE_TARGET: &str = "magi_rs::magi::clock_coverage";

/// Which setting the warning tells the operator to raise, and under what rotation settings.
///
/// A same-typed `u64` recommendation is ambiguous about which knob it names (the same hazard
/// `OpenAiSettings`/`ModeSources` exist to avoid elsewhere in this crate), so the lever is named
/// explicitly at every call site instead of inferred from context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoveringLever {
    /// The ceiling IS `[magi].agent_timeout_secs`, verbatim (the TUI path, and the headless path
    /// when the operator passed no `--timeout`).
    AgentTimeoutSecs,
    /// The ceiling is DERIVED from an explicit `--timeout`, under these rotation settings
    /// (`crate::magi::derive_ceiling_from_timeout`).
    Timeout {
        /// `[magi].max_rotations`, resolved.
        max_rotations: u32,
        /// `[magi].retry_disabled`, resolved.
        retry_disabled: bool,
    },
}

/// A clock that cannot cover the configured output cap at the reference speed (S-9).
///
/// Built only by [`Self::assess`], which returns `None` when the clock already covers the cap:
/// there is no way to construct one for a covered clock, so a caller cannot accidentally render
/// a warning nothing actually found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClockCoverageWarning {
    /// `[magi].max_tokens`, effective (`MagiConfig::effective_max_tokens`).
    cap_tokens: u32,
    /// The per-request client timeout the assessed ceiling derives, in seconds.
    client_timeout_secs: u64,
    /// How many tokens `client_timeout_secs` covers at [`COVERAGE_REFERENCE_TOK_PER_SEC`].
    covered_tokens: u64,
    /// Which knob the operator would raise.
    lever: CoveringLever,
    /// The smallest value of that knob that covers `cap_tokens`.
    covering_value_secs: u64,
}

impl ClockCoverageWarning {
    /// Assesses whether `ceiling`'s derived per-request client timeout can deliver `cap_tokens`
    /// at [`COVERAGE_REFERENCE_TOK_PER_SEC`], and if not, the smallest value of `lever`'s knob
    /// that would cover it.
    ///
    /// # Arguments
    /// * `cap_tokens` - `[magi].max_tokens`, effective.
    /// * `ceiling` - the per-mage ceiling this run actually uses.
    /// * `lever` - which knob a caller would raise to cover the cap, so the rendered line
    ///   recommends the one the operator can actually act on for this surface.
    ///
    /// # Returns
    /// `None` when the clock already covers `cap_tokens`; `Some` otherwise, carrying the exact
    /// covering value for `lever`.
    ///
    /// # Complexity
    /// `O(1)`; every arithmetic step saturates, so an absurd `cap_tokens`/`ceiling` pair
    /// degrades to a very large recommendation instead of overflowing.
    #[must_use]
    pub fn assess(
        cap_tokens: u32,
        ceiling: super::ResolvedCeiling,
        lever: CoveringLever,
    ) -> Option<Self> {
        let client_timeout_secs = super::derive_client_timeout(ceiling.secs()).as_secs();
        let cap = u64::from(cap_tokens);
        // The client timeout, in seconds, that would deliver `cap_tokens` at the reference
        // speed — `div_ceil` so a fractional second still counts as needed, never rounded away.
        let needed_client_timeout_secs = cap.div_ceil(COVERAGE_REFERENCE_TOK_PER_SEC);
        // Compared in the TIME domain, not by re-multiplying back into tokens: for positive
        // integers, `client_timeout_secs * REFERENCE >= cap` and `client_timeout_secs >=
        // needed_client_timeout_secs` are the same boundary (the standard `div_ceil` identity),
        // but the token-domain form can never land on an exact equality with `cap` unless `cap`
        // happens to be a multiple of the reference speed — which hides the boundary from a
        // test that swaps `>=` for `>`. The time domain hits that equality exactly at the
        // covering ceiling, so a wrong comparison operator here is observable.
        if client_timeout_secs >= needed_client_timeout_secs {
            return None;
        }
        let covered_tokens = client_timeout_secs.saturating_mul(COVERAGE_REFERENCE_TOK_PER_SEC);
        let covering_ceiling = super::min_ceiling_for_client_timeout(needed_client_timeout_secs);
        let covering_value_secs = match lever {
            // `AGENT_TIMEOUT_MIN_SECS` is the config's own validated floor: a ceiling below it
            // can never be loaded, so recommending one would name a value the operator cannot
            // actually set (S-9).
            CoveringLever::AgentTimeoutSecs => covering_ceiling.max(super::AGENT_TIMEOUT_MIN_SECS),
            CoveringLever::Timeout {
                max_rotations,
                retry_disabled,
            } => {
                super::min_timeout_deriving_ceiling(covering_ceiling, max_rotations, retry_disabled)
            }
        };
        Some(Self {
            cap_tokens,
            client_timeout_secs,
            covered_tokens,
            lever,
            covering_value_secs,
        })
    }

    /// The operator-facing line, naming the measured coverage and the knob that would fix it.
    ///
    /// Never mentions `--timeout` for [`CoveringLever::AgentTimeoutSecs`], nor
    /// `[magi].agent_timeout_secs` for [`CoveringLever::Timeout`]: a surface must not be handed
    /// advice for a knob it does not expose.
    ///
    /// # Complexity
    /// `O(1)`.
    #[must_use]
    pub fn render(&self) -> String {
        let lever_line = match self.lever {
            CoveringLever::AgentTimeoutSecs => format!(
                "set [magi].agent_timeout_secs to at least {} to cover it",
                self.covering_value_secs
            ),
            CoveringLever::Timeout { .. } => format!(
                "pass --timeout {} or more to cover it",
                self.covering_value_secs
            ),
        };
        format!(
            "magi: at {} tok/s the per-request clock ({}s) covers ~{} of the {}-token output \
             cap; {lever_line}. The consult proceeds.",
            COVERAGE_REFERENCE_TOK_PER_SEC,
            self.client_timeout_secs,
            self.covered_tokens,
            self.cap_tokens,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::magi::{derive_ceiling_from_timeout, ResolvedCeiling, AGENT_TIMEOUT_MIN_SECS};

    /// D-1: the default interactive consult (cap 16 384, ceiling 90 ⇒ client 27 s) is NOT
    /// covered; the warning states the coverage and the `agent_timeout_secs` that would cover
    /// it.
    #[test]
    fn the_default_interactive_clock_does_not_cover_the_cap_and_names_the_ceiling_that_would() {
        let w = ClockCoverageWarning::assess(
            16_384,
            ResolvedCeiling::configured(90),
            CoveringLever::AgentTimeoutSecs,
        )
        .expect("27 s at 55 tok/s cannot cover 16 384 tokens");
        assert_eq!(w.client_timeout_secs, 27);
        assert_eq!(w.covered_tokens, 1_485);
        assert_eq!(w.covering_value_secs, 994);
        let line = w.render();
        for needle in [
            "55 tok/s",
            "~1485",
            "16384",
            "[magi].agent_timeout_secs",
            "994",
        ] {
            assert!(line.contains(needle), "missing `{needle}`: {line}");
        }
        assert!(
            !line.contains("--timeout"),
            "the TUI has no --timeout: {line}"
        );
    }

    /// D-1, headless: `--timeout 1800` at `max_rotations = 2` derives 249 s ⇒ client 74 s; the
    /// warning names the `--timeout` that covers the cap, and it really does.
    #[test]
    fn the_default_headless_gate_clock_does_not_cover_the_cap_and_names_the_timeout_that_would() {
        let lever = CoveringLever::Timeout {
            max_rotations: 2,
            retry_disabled: false,
        };
        let ceiling = ResolvedCeiling::configured(derive_ceiling_from_timeout(1_800, 2, false));
        let w = ClockCoverageWarning::assess(16_384, ceiling, lever).expect("74 s cannot cover it");
        assert_eq!(w.client_timeout_secs, 74);
        assert_eq!(w.covered_tokens, 4_070);
        assert_eq!(w.covering_value_secs, 7_163);
        let line = w.render();
        for needle in ["~4070", "16384", "--timeout 7163"] {
            assert!(line.contains(needle), "missing `{needle}`: {line}");
        }
        let fixed = ResolvedCeiling::configured(derive_ceiling_from_timeout(7_163, 2, false));
        assert_eq!(
            ClockCoverageWarning::assess(16_384, fixed, lever),
            None,
            "the advice covers"
        );
    }

    /// S-9: a clock that covers the cap raises no warning — exactly at the boundary.
    #[test]
    fn a_clock_that_covers_the_cap_raises_no_warning_and_one_second_less_does() {
        let lever = CoveringLever::AgentTimeoutSecs;
        assert_eq!(
            ClockCoverageWarning::assess(16_384, ResolvedCeiling::configured(994), lever),
            None
        );
        assert!(
            ClockCoverageWarning::assess(16_384, ResolvedCeiling::configured(993), lever).is_some()
        );
    }

    /// The recommended ceiling is the exact boundary of coverage for every cap, and applying it
    /// silences the warning.
    ///
    /// **Deviation from the brief's literal `[56, 1_000, ...]` list (task 5, step 5a):** at
    /// `ResolvedCeiling::configured(AGENT_TIMEOUT_MIN_SECS)` (30 s), `derive_client_timeout`
    /// yields 9 s, which covers `9 * 55 = 495` tokens outright — so a cap of `56` is ALREADY
    /// covered at that ceiling and `assess` correctly returns `None`, making the brief's
    /// `unwrap_or_else(|| panic!("30 s cannot cover {cap}"))` panic on a mathematically
    /// impossible premise, not a defect in the implementation. `496` is the true boundary (the
    /// smallest cap NOT covered by 495 tokens) and keeps the sweep's intent — exercising a cap
    /// just above the covered range, in addition to the far larger ones — while being consistent
    /// with the same 55 tok/s arithmetic the other tests in this module pin.
    #[test]
    fn the_recommended_ceiling_is_the_exact_boundary_of_coverage() {
        let lever = CoveringLever::AgentTimeoutSecs;
        for cap in [496_u32, 1_000, 16_384, 65_536, 1_000_000, u32::MAX] {
            let w = ClockCoverageWarning::assess(
                cap,
                ResolvedCeiling::configured(AGENT_TIMEOUT_MIN_SECS),
                lever,
            )
            .unwrap_or_else(|| panic!("30 s cannot cover {cap}"));
            let c = w.covering_value_secs;
            assert_eq!(
                ClockCoverageWarning::assess(cap, ResolvedCeiling::configured(c), lever),
                None,
                "cap {cap}"
            );
            if c > AGENT_TIMEOUT_MIN_SECS {
                assert!(
                    ClockCoverageWarning::assess(cap, ResolvedCeiling::configured(c - 1), lever)
                        .is_some(),
                    "cap {cap}: {c} is not minimal"
                );
            }
        }
    }

    /// S-9: never a value the configuration rejects — the recommended `agent_timeout_secs` is
    /// never below the 30 s floor, even for a ceiling the config cannot produce (the derived
    /// floor, 15 s).
    #[test]
    fn the_warning_never_recommends_an_agent_timeout_below_the_floor() {
        let w = ClockCoverageWarning::assess(
            300,
            ResolvedCeiling::configured(15),
            CoveringLever::AgentTimeoutSecs,
        )
        .expect("4 s at 55 tok/s cannot cover 300 tokens");
        assert_eq!(w.covering_value_secs, AGENT_TIMEOUT_MIN_SECS);
    }

    /// The recommended `--timeout` covers the cap under every rotation setting, and one second
    /// less does not.
    #[test]
    fn the_recommended_timeout_derives_a_covering_ceiling_at_every_rotation_setting() {
        for max_rotations in 0..=3_u32 {
            for retry_disabled in [false, true] {
                let lever = CoveringLever::Timeout {
                    max_rotations,
                    retry_disabled,
                };
                for cap in [16_384_u32, 65_536] {
                    let w =
                        ClockCoverageWarning::assess(cap, ResolvedCeiling::configured(30), lever)
                            .expect("30 s cannot cover these caps");
                    let t = w.covering_value_secs;
                    let at = ResolvedCeiling::configured(derive_ceiling_from_timeout(
                        t,
                        max_rotations,
                        retry_disabled,
                    ));
                    let below = ResolvedCeiling::configured(derive_ceiling_from_timeout(
                        t - 1,
                        max_rotations,
                        retry_disabled,
                    ));
                    assert_eq!(
                        ClockCoverageWarning::assess(cap, at, lever),
                        None,
                        "r={max_rotations} d={retry_disabled} cap={cap}"
                    );
                    assert!(
                        ClockCoverageWarning::assess(cap, below, lever).is_some(),
                        "not minimal"
                    );
                }
            }
        }
    }

    /// Saturating end to end: the largest cap under an absurd rotation count never panics.
    #[test]
    fn the_assessment_saturates_instead_of_panicking() {
        let lever = CoveringLever::Timeout {
            max_rotations: u32::MAX,
            retry_disabled: false,
        };
        let w = ClockCoverageWarning::assess(u32::MAX, ResolvedCeiling::configured(30), lever)
            .expect("uncovered");
        assert!(!w.render().is_empty());
    }
}
