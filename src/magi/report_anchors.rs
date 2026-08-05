// Author: Julian Bolivar Version: 1.0.0 Date: 2026-08-02

//! Anchors of the magi-core report — **single owner**, no copy on the test side.
//!
//! # Provenance: an OBSERVATION, not a published contract
//!
//! The format of `MagiReport::report` **is not public API** of magi-core: it is markdown that
//! the crate generates for human consumption. These anchors come from the Task 0.6 spike, run
//! against magi-core 3.1.0 on 2026-08-02, and were later verified against the crate's own
//! `src/reporting.rs` to learn **which ones are unconditional** and which depend on content.
//!
//! That is why they live in a module of their own, with their provenance written down: when
//! magi-core changes the rendering, **one** file is touched, and the guardian
//! `report_shape_matches_what_the_truncation_design_assumes` warns before a user does.
//!
//! # What was verified, and where
//!
//! `ReportFormatter` composes the report in this order (`reporting.rs:795-817`):
//!
//! | Section | Always present? |
//! |---|---|
//! | Verdict box (`MAGI SYSTEM -- VERDICT`) | **yes** |
//! | Estimation notes, extraction failures, input size | conditional |
//! | `## Key Findings` | only if there are findings |
//! | `## Dissenting Opinion` | only if there is dissent |
//! | `## Conditions for Approval` | only if there are conditions |
//! | `## Recommended Actions` | **yes**, and it is the last |
//!
//! From there comes the choice of `findings_end`: **not** `## Conditions for Approval`, which
//! is optional, but `## Recommended Actions`, which is always present and comes after
//! everything the truncation wants to keep.

/// SECTION anchors, **named**. Not a position-indexed `&[&str]`.
///
/// A list would leave the contract in the indices —`.first()`, `.get(1)`, `.get(2)`— and then a
/// spike that finds two anchors instead of three **compiles just the same** and lowers the
/// truncation ceiling **silently**, with `report_truncated` still saying `structural`. That is
/// the worst available failure mode: the consumer believes it has a guarantee that no longer
/// holds.
///
/// With named fields, a missing anchor is an `Option` that must be handled and an extra one has
/// nowhere to go: the mismatch between what was measured and what is assumed becomes a compile
/// error.
pub struct SectionAnchors {
    /// Where the verdict block starts. Without this there is no `Structural` level.
    ///
    /// It is the inner text of the ASCII box and not the `+===+` line, which repeats four times
    /// and does not distinguish beginning from end.
    pub verdict_start: &'static str,
    /// Where the findings section starts — the cut does NOT go here, it goes after.
    ///
    /// **May be missing**: magi-core omits the entire section when there are no findings. A
    /// consumer that assumes its presence treats "no findings" as "could not locate", which are
    /// two different things and only one is a degradation.
    pub findings_start: &'static str,
    /// Where the region that is preserved ends.
    ///
    /// `## Recommended Actions` and not `## Conditions for Approval`: the former is
    /// unconditional and goes last, so everything preservable —verdict, findings, dissent,
    /// conditions— stays on this side. Anchoring to an optional section would leave the end
    /// undefined precisely in reports that do not include it.
    pub findings_end: &'static str,
}

/// What the Task 0.6 spike measured. `None` ⇒ `Structural` is not reachable.
///
/// It is `Some`: the report exposes stable markdown headings, so the three truncation levels of
/// REQ-A11b are implementable.
pub const SECTION_ANCHORS: Option<SectionAnchors> = Some(SectionAnchors {
    verdict_start: "MAGI SYSTEM -- VERDICT",
    findings_start: "## Key Findings",
    findings_end: "## Recommended Actions",
});

/// CONTRACTUAL anchors: the subset that magi-core **always** emits.
///
/// Non-empty ⇒ at least the `Anchored` level is reachable, and that is its role: they are the
/// fallback for a report where `findings_start` does not exist because there were no findings.
/// With these two the verdict can still be delimited without falling back to byte counting.
pub const CONTRACTUAL_ANCHORS: &[&str] = &["MAGI SYSTEM -- VERDICT", "## Recommended Actions"];
