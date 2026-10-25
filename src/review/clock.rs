// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-24

//! The clock of a review (REQ-RV06): how an explicit `--timeout` is split over the batches, the
//! budget check before the first trio call, and the worst case the TUI shows.

use super::error::ReviewError;

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
pub fn batch_wall_clock_secs(remaining_secs: u64, pending: usize) -> u64 {
    let _ = (remaining_secs, pending);
    todo!()
}

/// The budget check of an explicit `--timeout`, before any trio call.
///
/// # Errors
///
/// [`ReviewError::NothingToReview`] for zero batches and [`ReviewError::Budget`] when the wall
/// clock per batch is below what a batch needs at the floor.
pub fn pre_budget(
    total_secs: u64,
    batches: usize,
    max_rotations: u32,
    retry_disabled: bool,
) -> Result<u64, ReviewError> {
    let _ = (total_secs, batches, max_rotations, retry_disabled);
    todo!()
}

/// The worst case of a review in the TUI.
pub fn tui_worst_case_secs(batches: usize, max_rotations: u32, agent_timeout_secs: u64) -> u64 {
    let _ = (batches, max_rotations, agent_timeout_secs);
    todo!()
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
            Err(ReviewError::NothingToReview)
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
}
