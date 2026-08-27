// Author: Julian Bolivar
// Version: 0.17.0
// Date: 2026-08-27

//! Complexity gate: pure predicate that decides whether an agent-**self-routed** consult merits
//! dispatching the three-perspective consensus (REQ-A20).
//!
//! Three properties that govern this module and are easy to get wrong:
//!
//! - **Only AUTONOMOUS routing is evaluated.** `/consult` in the TUI and explicit `magi consult` are NEVER vetoed — the gate sees the agent's decision to invoke the tool on its own, and nothing else. That distinction is about **call site** (where `evaluate` is invoked), not a flag this module can see: `gate.rs` does not know who called.
//! - **A veto is NOT an error.** [`GateVerdict::Veto`] is a normal result of the predicate, not an `Err`: the agent funnel translates it into a `ToolResult` explaining why it was not dispatched, and the turn continues.
//! - **Missing configuration does NOT turn the gate off.** Absent `[magi.complexity]` table ⇒ the built-ins of [`super::GATE_CODE_REVIEW`]/[`super::GATE_DESIGN`]/ [`super::GATE_ANALYSIS`] still apply. A mode declared at `0` is the only explicit way to turn **that** mode off — not the other two.

use magi_core::schema::Mode;

use super::{GATE_ANALYSIS, GATE_CODE_REVIEW, GATE_DESIGN};

/// Effective gate thresholds, one per mode (REQ-A20b).
///
/// Exists as its own type —and not as three loose `usize`s— because the point of per-mode
/// granularity is that **turning one off does not turn the others off**: with positional
/// parameters of the same type, a silent swap at a call site breaks exactly that property
/// without breaking compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GateThresholds {
    /// `code-review` threshold, in characters.
    pub code_review: usize,
    /// `design` threshold, in characters.
    pub design: usize,
    /// `analysis` threshold — the default mode, and therefore the one that CANNOT be
    /// effectively turned off by accident (SC-A20j).
    pub analysis: usize,
}

impl GateThresholds {
    /// The §4.9 built-ins, with no config in between.
    #[must_use]
    pub const fn builtin() -> Self {
        Self {
            code_review: GATE_CODE_REVIEW,
            design: GATE_DESIGN,
            analysis: GATE_ANALYSIS,
        }
    }

    /// Resolves the `[magi.complexity]` table against the built-ins.
    ///
    /// **Missing table ⇒ built-ins** (the gate stays alive: a security feature that
    /// turns off just by omitting a section is a turned-off feature). **Missing key INSIDE a
    /// present table ⇒ its built-in, not zero**: `Option::unwrap_or` per key, never `Default`
    /// over the whole struct, which would collapse all three to `0` and disable the entire gate
    /// just by declaring an empty `[magi.complexity]`.
    ///
    /// **Takes loose pieces, NOT `&ComplexityConfig`.** This module lives in the lib and
    /// `ComplexityConfig` in the bin (`src/config.rs`): taking the struct would tie a pure
    /// module to the shape of the TOML and make it uncompilable from the lib. Disassembling the
    /// table is the job of `config.rs`'s `gate_thresholds_from`, which already has it in hand.
    #[must_use]
    pub fn from_overrides(o: GateOverrides) -> Self {
        let GateOverrides {
            code_review,
            design,
            analysis,
        } = o;
        let b = Self::builtin();
        Self {
            code_review: code_review.unwrap_or(b.code_review),
            design: design.unwrap_or(b.design),
            analysis: analysis.unwrap_or(b.analysis),
        }
    }

    /// Threshold of the requested mode. `0` means "this mode is never vetoed" (interpreted by
    /// [`evaluate`], not this function).
    #[must_use]
    pub const fn for_mode(&self, mode: &Mode) -> usize {
        match mode {
            Mode::CodeReview => self.code_review,
            Mode::Design => self.design,
            Mode::Analysis => self.analysis,
        }
    }
}

/// Overrides of `[magi.complexity]`, with NAME per field.
///
/// Three positional arguments of the same type (`Option<usize>`, `Option<usize>`,
/// `Option<usize>`) are exactly the silent swap that the rustdoc of [`GateThresholds`]
/// condemns.
#[derive(Debug, Clone, Copy, Default)]
pub struct GateOverrides {
    /// Override of `code_review`; absent ⇒ its built-in.
    pub code_review: Option<usize>,
    /// Override of `design`; absent ⇒ its built-in.
    pub design: Option<usize>,
    /// Override of `analysis`; absent ⇒ its built-in.
    pub analysis: Option<usize>,
}

/// Result of evaluating the gate (REQ-A20).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateVerdict {
    /// The content merits the consensus: it is dispatched.
    Dispatch,
    /// Below its mode's threshold: NO model call is launched.
    Veto {
        /// Mode with which it was evaluated, for the telemetry log (SC-A20h).
        mode: Mode,
    },
}

/// Evaluates whether a **self-routed** consult merits being dispatched.
///
/// **100 % pure:** no async, no I/O, no model calls. It is evaluated in the funnel of the
/// agent, not inside `ConsultTool::execute` — see REQ-A20 for why
/// `MagiBuilder::with_complexity_gate` is not used.
///
/// It measures **characters**, not bytes (`content.chars().count()`, O(n) over the content,
/// without nested loops): a byte threshold would treat the same text differently in another
/// language.
#[must_use]
pub fn evaluate(content: &str, mode: &Mode, thresholds: &GateThresholds) -> GateVerdict {
    let threshold = thresholds.for_mode(mode);
    if threshold == 0 || content.chars().count() >= threshold {
        GateVerdict::Dispatch
    } else {
        GateVerdict::Veto { mode: *mode }
    }
}

/// Destination for gate telemetry (REQ-A20, SC-A20h).
///
/// **Separated from `RunObserver` on purpose.** The observer is optional by
/// design (`None` in the TUI, which is precisely the surface that self-routes the most
/// consults), so hanging a signal that SC-A20h requires *always* be logged from it would make
/// it conditional on the surface. Both implement this trait: the headless runner (bin) routing
/// it to its structured run log, the TUI (bin) to a bounded in-memory buffer. Only the trait
/// and [`NoGateTelemetry`] live here, in the lib — same boundary that separates
/// `ModeParseError` from `ConfigError`.
pub trait GateTelemetry: Send + Sync {
    /// Logs ONE evaluation: mode, content length, applied threshold, and whether it vetoed
    /// (SC-A20h). The applied threshold is ALWAYS carried, even in the line that dispatches:
    /// without the number on the passing side, calibrating the built-ins has nothing to compare
    /// against.
    fn on_gate_evaluation(&self, mode: &Mode, chars: usize, threshold: usize, vetoed: bool);
}

/// Null sink: zero logging, behavior identical to before this field existed. This is what the
/// `Default` of the agent run config (`AgentRunConfig`, in the binary) uses, so the field is
/// purely additive: no route that does not wire it explicitly changes behavior.
pub struct NoGateTelemetry;

impl GateTelemetry for NoGateTelemetry {
    fn on_gate_evaluation(&self, _mode: &Mode, _chars: usize, _threshold: usize, _vetoed: bool) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SC-A20b: threshold per mode, and the boundary is the documented one (inclusive).
    #[test]
    fn thresholds_are_per_mode_and_the_boundary_is_inclusive() {
        let t = GateThresholds::builtin();
        assert_eq!(
            evaluate(&"x".repeat(GATE_CODE_REVIEW), &Mode::CodeReview, &t),
            GateVerdict::Dispatch,
            "exactly AT the threshold, it passes"
        );
        assert_eq!(
            evaluate(&"x".repeat(GATE_CODE_REVIEW - 1), &Mode::CodeReview, &t),
            GateVerdict::Veto {
                mode: Mode::CodeReview
            }
        );
        assert_eq!(
            evaluate(&"x".repeat(GATE_CODE_REVIEW), &Mode::Design, &t),
            GateVerdict::Veto { mode: Mode::Design },
            "Design demands more"
        );
    }

    /// SC-A20j: the gate COVERS `Analysis`, the default mode for any invocation without one
    /// declared — a threshold of 1 would turn it off right on the most common autonomous path.
    #[test]
    fn the_gate_covers_analysis_the_default_mode() {
        let t = GateThresholds::builtin();
        // `const` and not a loose `assert!`: it is a comparison between two constants, so
        // clippy (`assertions_on_constants`) demands it be evaluated at compile time — same
        // form already used by `mod.rs`'s `plan_values_fall_inside_their_documented_ranges`.
        const {
            assert!(
                GATE_ANALYSIS > 1,
                "a threshold of 1 would turn off the gate right on the most common \
                 autonomous path"
            );
        }
        assert_eq!(
            evaluate("trivial", &Mode::Analysis, &t),
            GateVerdict::Veto {
                mode: Mode::Analysis
            }
        );
    }

    /// SC-A20d: without a table, the built-ins apply; `0` turns off ONLY that mode.
    #[test]
    fn an_absent_table_keeps_the_gate_alive_and_zero_disables_one_mode() {
        let t = GateThresholds::from_overrides(GateOverrides::default());
        assert_eq!(
            t,
            GateThresholds::builtin(),
            "table absent ⇒ built-ins: the gate does not turn off by omitting a section"
        );

        let t = GateThresholds::from_overrides(GateOverrides {
            code_review: Some(0),
            ..GateOverrides::default()
        });
        assert_eq!(evaluate("x", &Mode::CodeReview, &t), GateVerdict::Dispatch);
        assert_eq!(
            evaluate("x", &Mode::Design, &t),
            GateVerdict::Veto { mode: Mode::Design },
            "turning off one mode does not turn off the others: without granularity, \
             the only way out would be setting all three to zero, and a mechanism \
             that only knows how to turn itself off ends up off"
        );
    }

    // HONEST NOTE about `NoGateTelemetry`: there is no possible assertion against a no-op. An
    // earlier test here called `on_gate_evaluation` twice without asserting anything — it could
    // not fail under any change (same defect Task 3.1 found and fixed in its own simulation,
    // flagged by the review of this task: I2). The contract of `NoGateTelemetry` IS the empty
    // body of `on_gate_evaluation`, above — it is verified by reading it, not by running it.
    // What IS observable and IS tested in `agent/mod.rs` is that `AgentRunConfig::default()`
    // installs `NoGateTelemetry` and that a run without explicit `gate_telemetry` logs nothing
    // (`every_gate_evaluation_is_logged` tests the opposite case, with
    // `RecordingGateTelemetry`, which can fail).

    // HONEST NOTE (closed by Task 3.2): the REQ-A20 property "the gate sees AUTONOMOUS routing
    // (`ToolUse`) and NOT forced injection (`authorize_and_execute_tool`)" was DELIBERATELY
    // untestable from this module — the two call sites live in `agent/mod.rs`, not here;
    // `gate.rs` only exposes `evaluate`, which neither knows nor can know who invoked it. A
    // test written here could only SIMULATE the two call sites with literals, and a simulation
    // of that was exactly what made the earlier test false (deleted in Task 3.1 for that
    // reason). The real test, which fires `authorize_and_execute_tool` and the `ToolUse` loop
    // against a real `Agent`, now lives in `agent::tests::
    // a_forced_injection_bypasses_the_gate_while_a_model_choice_does_not`.
}
