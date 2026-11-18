// Author: Julian Bolivar
// Version: 0.25.0
// Date: 2026-11-18

//! The clock of a review (REQ-RV06): how an explicit `--timeout` is split over the batches, the
//! budget check before the first trio call, and the worst case the TUI shows.

use super::error::{BudgetLimit, ReviewError};
use crate::magi::{
    attempt_factor, derive_ceiling_from_timeout, headless_consult_timeout_secs,
    raw_ceiling_from_factor, AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS, CALLS_PER_MODEL,
};

/// The wall clock of the next batch: what is left of the review, split over the batches still
/// pending.
///
/// # Arguments
///
/// * `remaining_secs` - What is left of the explicit `--timeout`.
/// * `pending` - The batches not run yet, the next one included.
///
/// # Returns
///
/// `remaining_secs / pending`; with no pending batch (a defensive case the pipeline never
/// reaches), `remaining_secs`.
///
/// # Examples
///
/// ```
/// use magi_rs::review::clock::batch_wall_clock_secs;
///
/// assert_eq!(batch_wall_clock_secs(3_000, 3), 1_000);
/// ```
pub fn batch_wall_clock_secs(remaining_secs: u64, pending: usize) -> u64 {
    match u64::try_from(pending) {
        Ok(0) => remaining_secs,
        Ok(n) => remaining_secs / n,
        Err(_) => 0,
    }
}

/// The budget check of an explicit `--timeout`, before any trio call (REQ-RV06).
///
/// The `--timeout` is split evenly over the batches; each batch needs at least the wall clock
/// [`headless_consult_timeout_secs`] gives at [`AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS`]. The check is
/// on the wall clock, not on the derived ceiling: [`derive_ceiling_from_timeout`] never returns
/// less than the floor, so a ceiling check could not fail.
///
/// # Arguments
///
/// * `total_secs` - The explicit `--timeout`.
/// * `batches` - The batches the review was split into.
/// * `max_rotations` - `[magi].max_rotations`, resolved.
/// * `retry_disabled` - `[magi].retry_disabled`, resolved.
///
/// # Returns
///
/// The ceiling derived from the initial per-batch wall clock.
///
/// # Errors
///
/// [`ReviewError::NothingToReview`] for zero batches and [`ReviewError::Budget`] when the wall
/// clock per batch is below what a batch needs at the floor, naming the batches, the per-batch
/// ceiling, the floor and the smallest `--timeout` that would be enough.
///
/// # Examples
///
/// ```
/// use magi_rs::review::clock::pre_budget;
///
/// assert!(pre_budget(60, 4, 2, false).is_err());
/// assert!(pre_budget(100_000, 4, 2, false).is_ok());
/// ```
pub fn pre_budget(
    total_secs: u64,
    batches: usize,
    max_rotations: u32,
    retry_disabled: bool,
) -> Result<u64, ReviewError> {
    // MAGI FIX (S6 loop 4): unreachable from `run_review`, whose planning fails first with the
    // omitted files named; kept so a direct caller with zero batches spends nothing.
    if batches == 0 {
        return Err(ReviewError::NothingToReview {
            skipped: Vec::new(),
        });
    }
    let per_batch = batch_wall_clock_secs(total_secs, batches);
    let min_per_batch = headless_consult_timeout_secs(
        AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS,
        max_rotations,
        retry_disabled,
    );
    if per_batch < min_per_batch {
        return Err(ReviewError::Budget {
            batches,
            per_batch_wall_clock_secs: per_batch,
            min_per_batch_secs: min_per_batch,
            per_batch_ceiling_secs: raw_ceiling_from_factor(
                per_batch,
                attempt_factor(max_rotations, retry_disabled),
            ),
            floor_secs: AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS,
            limit: BudgetLimit::Timeout,
            min_timeout_secs: u64::try_from(batches)
                .unwrap_or(u64::MAX)
                .saturating_mul(min_per_batch),
        });
    }
    Ok(derive_ceiling_from_timeout(
        per_batch,
        max_rotations,
        retry_disabled,
    ))
}

/// The worst case of a review in the TUI (REQ-RV18), a bound and never an expected duration:
/// every batch runs each seat's full chain, `(1 + max_rotations)` models with two calls each,
/// every call up to `agent_timeout_secs`. Like the spec's formula, it counts two calls whatever
/// `retry_disabled` says: a bound that can only be too high.
///
/// # Arguments
///
/// * `batches` - The batches the review was split into.
/// * `max_rotations` - `[magi].max_rotations`, resolved.
/// * `agent_timeout_secs` - `[magi].agent_timeout_secs`, resolved.
///
/// # Returns
///
/// `batches × (1 + max_rotations) × 2 × agent_timeout_secs`, saturating.
///
/// # Examples
///
/// ```
/// use magi_rs::review::clock::tui_worst_case_secs;
///
/// assert_eq!(tui_worst_case_secs(1, 2, 2_335), 14_010);
/// ```
pub fn tui_worst_case_secs(batches: usize, max_rotations: u32, agent_timeout_secs: u64) -> u64 {
    u64::try_from(batches)
        .unwrap_or(u64::MAX)
        .saturating_mul(u64::from(max_rotations).saturating_add(1))
        .saturating_mul(CALLS_PER_MODEL)
        .saturating_mul(agent_timeout_secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_wall_clock_is_the_remaining_time_split_over_pending_batches() {
        assert_eq!(batch_wall_clock_secs(3_000, 3), 1_000);
        assert_eq!(batch_wall_clock_secs(2_000, 2), 1_000);
        assert_eq!(batch_wall_clock_secs(10, 0), 10);
    }

    #[test]
    fn a_budget_below_the_floor_fails_naming_batches_ceiling_floor_and_minimum() {
        // [CP2 MS2 S2a2 loop 11] no batches is nothing to review, never a division by zero.
        assert!(matches!(
            pre_budget(60, 0, 2, false),
            Err(ReviewError::NothingToReview { .. })
        ));
        let err = pre_budget(60, 4, 2, false).expect_err("too little");
        match err {
            ReviewError::Budget {
                batches,
                floor_secs,
                min_timeout_secs,
                per_batch_wall_clock_secs,
                min_per_batch_secs,
                ..
            } => {
                assert_eq!(
                    (batches, floor_secs, per_batch_wall_clock_secs),
                    (4, 15, 15)
                );
                assert_eq!(
                    min_per_batch_secs,
                    crate::magi::headless_consult_timeout_secs(15, 2, false)
                );
                assert_eq!(
                    min_timeout_secs,
                    4 * crate::magi::headless_consult_timeout_secs(15, 2, false)
                );
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn the_tui_worst_case_is_the_full_chain_per_batch() {
        assert_eq!(tui_worst_case_secs(3, 2, 2335), 3 * 3 * 2 * 2335);
    }

    #[test]
    fn a_timeout_exactly_enough_passes_the_budget_and_one_second_less_fails() {
        // [MAGI S6 loop 13, C2-b5] the check is `<`: a `--timeout` that is exactly the minimum
        // runs. Only a value below the floor was tested, so `<=` would have refused a review
        // whose timeout is exactly sufficient with nothing turning red.
        let min = crate::magi::headless_consult_timeout_secs(15, 2, false);
        assert!(pre_budget(4 * min, 4, 2, false).is_ok());
        assert!(pre_budget(4 * min - 1, 4, 2, false).is_err());
    }

    #[test]
    fn the_tui_worst_case_saturates_instead_of_overflowing() {
        // [MAGI S6 loop 13, C2-b5] the saturating paths were unexercised.
        assert_eq!(tui_worst_case_secs(usize::MAX, 2, 2335), u64::MAX);
        assert_eq!(tui_worst_case_secs(1, u32::MAX, u64::MAX), u64::MAX);
        assert_eq!(tui_worst_case_secs(0, 2, u64::MAX), 0);
    }
}
