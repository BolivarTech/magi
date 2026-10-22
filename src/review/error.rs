// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! The typed errors of the `code-review` pipeline.

/// Every way a review can fail before or while it calls a model.
///
/// Each variant is specific so callers decide by the variant, never by the text of the message.
#[derive(Debug, thiserror::Error)]
pub enum ReviewError {
    /// A requested path resolves outside the workspace (or a worktree read would leave it).
    #[error("path is outside the workspace: {0}")]
    OutsideWorkspace(String),
    /// A git ref that is empty, starts with `-`, or holds a NUL, `..` or a space.
    #[error("invalid git ref: {0}")]
    InvalidRef(String),
    /// A git ref that does not name a commit.
    #[error("git ref not found: {0}")]
    RefNotFound(String),
    /// git could not run, or exited with an error (the message carries its bounded stderr).
    #[error("git failed: {0}")]
    Git(String),
    /// A file named explicitly that cannot be reviewed, with the reason.
    #[error("{path} is not reviewable: {cause}")]
    NotReviewable {
        /// The path as the caller named it.
        path: String,
        /// Why it cannot be reviewed.
        cause: &'static str,
    },
    /// The input holds no reviewable file.
    #[error("nothing to review: no reviewable file in the input")]
    NothingToReview,
    /// A hunk or unit that does not fit in one batch by itself.
    #[error("{what} does not fit in one batch ({tokens} tokens, limit {limit})")]
    TooLargeForBatch {
        /// What did not fit.
        what: String,
        /// Its estimated size in tokens.
        tokens: usize,
        /// The per-batch limit in tokens.
        limit: usize,
    },
    /// An explicit `--timeout` that leaves each batch less wall clock than the floor needs.
    // D22 compares the per-batch WALL CLOCK with the wall clock a batch needs at the floor; the
    // ceiling and the raw floor are context, never claimed to be "below" each other.
    #[error(
        "{batches} batches leave {per_batch_wall_clock_secs} s per batch, less than the \
         {min_per_batch_secs} s a batch needs at the {floor_secs} s floor (per-batch ceiling \
         {per_batch_ceiling_secs} s); pass --timeout {min_timeout_secs} or more"
    )]
    Budget {
        /// Batches the review was split into.
        batches: usize,
        /// Wall clock each batch would get.
        per_batch_wall_clock_secs: u64,
        /// Wall clock a batch needs at the floor.
        min_per_batch_secs: u64,
        /// The ceiling derived from the per-batch wall clock.
        per_batch_ceiling_secs: u64,
        /// The absolute floor of the ceiling.
        floor_secs: u64,
        /// The smallest `--timeout` that would be enough.
        min_timeout_secs: u64,
    },
    /// The batch reports could not be merged into one report.
    #[error("consolidation failed: {0}")]
    Consolidation(String),
    /// A local I/O error.
    #[error("I/O error: {0}")]
    Io(String),
    /// A git output that went past its byte cap; git was stopped.
    #[error("git output is larger than {limit} bytes")]
    GitOutputTooLarge {
        /// The cap in bytes.
        limit: u64,
    },
    /// A file over the per-file byte cap; its content is never returned.
    #[error("{path} is larger than {limit} bytes")]
    TooLarge {
        /// The file path.
        path: String,
        /// The cap in bytes.
        limit: u64,
    },
    /// The review reads more bytes in total than the per-review cap.
    #[error("the review reads more than {limit} bytes in total")]
    InputTooLarge {
        /// The cap in bytes.
        limit: u64,
    },
    /// The run's cancellation token fired mid-review.
    #[error("the review was cancelled")]
    Cancelled,
    /// A worktree path whose resolved target is a path a review never reads.
    #[error("{0} resolves to a path a review never reads")]
    Protected(String),
    /// A secret found in what was about to be sent.
    #[error(
        "{path} holds a secret at batching time: it changed during the review or was not filtered"
    )]
    EgressSecret {
        /// The file path.
        path: String,
    },
}
