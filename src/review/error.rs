// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! The typed errors of the `code-review` pipeline.

use super::filter::Skipped;
use super::secrets::{display_path, outward_text};

/// What set a review's total wall clock (REQ-RV06, RV17).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetLimit {
    /// An explicit `--timeout`.
    Timeout,
    /// What is left of the calling run (the agent tool's run deadline); no `--timeout` was set.
    RunDeadline,
}

/// The advice that ends a [`ReviewError::Budget`] message, naming the bound the user can move.
// MAGI FIX (S6 loop 4): a run deadline is not a `--timeout`; the message names what was set.
fn budget_advice(limit: BudgetLimit, min_secs: u64) -> String {
    match limit {
        BudgetLimit::Timeout => format!("pass --timeout {min_secs} or more"),
        BudgetLimit::RunDeadline => {
            format!("the review needs {min_secs} s or more left of the calling run")
        }
    }
}

/// Every way a review can fail before or while it calls a model.
///
/// Each variant is specific so callers decide by the variant, never by the text of the message.
///
/// The message ([`std::fmt::Display`]) is safe to send out of the process wherever it lands
/// (stderr, the run log, the JSON envelope, the TUI, a tool result): every path and ref passes
/// through [`display_path`] and every other foreign text (git's stderr, an I/O or consolidation
/// message) through [`outward_text`], so no secret the RV22 detector names and no control
/// character leaves in it. The fields keep the raw text for callers that match on it.
// MAGI FIX (S6 loop 3): one outward scrubber, at Display, for every surface at once.
#[derive(Debug, thiserror::Error)]
pub enum ReviewError {
    /// A requested path resolves outside the workspace (or a worktree read would leave it).
    #[error("path is outside the workspace: {}", display_path(.0))]
    OutsideWorkspace(String),
    /// A git ref that is empty, starts with `-`, or holds a NUL, `..` or a space.
    #[error("invalid git ref: {}", display_path(.0))]
    InvalidRef(String),
    /// A git ref that does not name a commit.
    #[error("git ref not found: {}", display_path(.0))]
    RefNotFound(String),
    /// git could not run, or exited with an error (the message carries its bounded stderr).
    #[error("git failed: {}", outward_text(.0))]
    Git(String),
    /// A file named explicitly that cannot be reviewed, with the reason.
    #[error("{} is not reviewable: {cause}", display_path(.path))]
    NotReviewable {
        /// The path as the caller named it.
        path: String,
        /// Why it cannot be reviewed.
        cause: &'static str,
    },
    /// The input holds no reviewable file; `skipped` lists every file left out, with its cause.
    #[error(
        "nothing to review: no reviewable file in the input{}",
        outward_text(&omissions(.skipped))
    )]
    NothingToReview {
        /// The files the input held and the review left out, with why (shown paths).
        skipped: Vec<Skipped>,
    },
    /// A hunk or unit that does not fit in one batch by itself.
    #[error(
        "{} does not fit in one batch ({tokens} tokens, limit {limit})",
        outward_text(.what)
    )]
    TooLargeForBatch {
        /// What did not fit.
        what: String,
        /// Its estimated size in tokens.
        tokens: usize,
        /// The per-batch limit in tokens.
        limit: usize,
    },
    /// An explicit `--timeout`, or what is left of a calling run, that leaves each batch less
    /// wall clock than the floor needs.
    // D22 compares the per-batch WALL CLOCK with the wall clock a batch needs at the floor; the
    // ceiling and the raw floor are context, never claimed to be "below" each other.
    #[error(
        "{batches} batches leave {per_batch_wall_clock_secs} s per batch, less than the \
         {min_per_batch_secs} s a batch needs at the {floor_secs} s floor (per-batch ceiling \
         {per_batch_ceiling_secs} s); {}",
        budget_advice(*.limit, *.min_timeout_secs)
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
        /// The smallest total that would be enough: a `--timeout`, or what is left of the run.
        min_timeout_secs: u64,
        /// Which bound set the total, so the message names the one the user can move.
        limit: BudgetLimit,
    },
    /// The batch reports could not be merged into one report.
    #[error("consolidation failed: {}", outward_text(.0))]
    Consolidation(String),
    /// A local I/O error.
    #[error("I/O error: {}", outward_text(.0))]
    Io(String),
    /// A git output that went past its byte cap; git was stopped.
    #[error("git output is larger than {limit} bytes")]
    GitOutputTooLarge {
        /// The cap in bytes.
        limit: u64,
    },
    /// A file over the per-file byte cap; its content is never returned.
    #[error("{} is larger than {limit} bytes", display_path(.path))]
    TooLarge {
        /// The file path.
        path: String,
        /// The cap in bytes.
        limit: u64,
    },
    /// The review reads more bytes in total than the per-review cap.
    #[error(
        "the review reads more than {limit} bytes in total ({held} held when {} was read)",
        display_path(path)
    )]
    InputTooLarge {
        /// The cap in bytes.
        limit: u64,
        /// The bytes already held when the read was refused.
        held: u64,
        /// The path whose read would have passed the cap.
        path: String,
    },
    /// The run's cancellation token fired mid-review.
    #[error("the review was cancelled")]
    Cancelled,
    /// A worktree path whose resolved target is a path a review never reads.
    #[error("{} resolves to a path a review never reads", display_path(.0))]
    Protected(String),
    /// A worktree path whose resolved target is not a regular file (a link to a directory, a FIFO,
    /// a socket or a device file).
    #[error("{} is not a regular file", display_path(.0))]
    NotRegularFile(String),
    /// The review no longer splits into the batches that were approved: the working tree moved
    /// between the approval and the run. Raised before any trio call.
    #[error(
        "the review changed since it was approved: {approved} batches were approved and it now \
         splits into {now}; ask for it again"
    )]
    PlanChanged {
        /// The batches the caller approved.
        approved: usize,
        /// The batches it splits into now.
        now: usize,
    },
    /// A secret found in what was about to be sent.
    #[error(
        "{} holds a secret at batching time: it changed during the review or was not filtered",
        display_path(.path)
    )]
    EgressSecret {
        /// The file path.
        path: String,
    },
}

/// How many omissions a [`ReviewError::NothingToReview`] message names before it counts the rest.
pub const MAX_OMISSIONS_SHOWN: usize = 20;

/// `; left out: a (cause), b (cause), and N more`, or nothing when no file was left out.
///
/// # Arguments
/// * `skipped` - The files left out.
///
/// # Returns
/// The suffix of the message; O(shown omissions).
fn omissions(skipped: &[Skipped]) -> String {
    if skipped.is_empty() {
        return String::new();
    }
    let shown: Vec<String> = skipped
        .iter()
        .take(MAX_OMISSIONS_SHOWN)
        .map(ToString::to_string)
        .collect();
    let rest = skipped.len().saturating_sub(MAX_OMISSIONS_SHOWN);
    let more = if rest > 0 {
        format!(", and {rest} more")
    } else {
        String::new()
    };
    format!("; left out: {}{more}", shown.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::review::secrets::find_secret;

    /// A secret-shaped token, split so the line holds no literal key.
    fn key() -> String {
        format!("sk-{}", "a".repeat(30))
    }

    /// Every variant that carries foreign text (git stderr, a path, a ref, a message).
    fn carrying(text: &str) -> Vec<ReviewError> {
        let t = text.to_string();
        vec![
            ReviewError::OutsideWorkspace(t.clone()),
            ReviewError::InvalidRef(t.clone()),
            ReviewError::RefNotFound(t.clone()),
            ReviewError::Git(t.clone()),
            ReviewError::NotReviewable {
                path: t.clone(),
                cause: "binary",
            },
            ReviewError::TooLargeForBatch {
                what: t.clone(),
                tokens: 1,
                limit: 1,
            },
            ReviewError::Consolidation(t.clone()),
            ReviewError::Io(t.clone()),
            ReviewError::TooLarge {
                path: t.clone(),
                limit: 1,
            },
            ReviewError::Protected(t.clone()),
            ReviewError::NotRegularFile(t.clone()),
            ReviewError::EgressSecret { path: t },
        ]
    }

    #[test]
    fn no_error_message_carries_a_secret_from_its_foreign_text() {
        // [MAGI S6 loop 3, S1] every outward error text passes the RV22 detector at Display.
        for e in carrying(&format!("fatal: bad object {}", key())) {
            let shown = e.to_string();
            assert_eq!(find_secret(&shown), None, "{shown}");
            assert!(shown.contains("redacted: openai-key>"), "{shown}");
        }
    }

    #[test]
    fn no_error_message_replays_a_control_character() {
        // [MAGI S6 loop 3, S1] a path or a git diagnostic never replays a terminal escape.
        for e in carrying("a\u{1b}[2Jb\nc") {
            let shown = e.to_string();
            assert!(!shown.chars().any(char::is_control), "{shown:?}");
            assert!(shown.contains("a\\u{1b}[2Jb\\nc"), "{shown}");
        }
    }

    #[test]
    fn a_long_omission_list_names_the_first_ones_and_counts_the_rest() {
        // [MAGI S6 loop 3, A2] a directory of thousands of binaries cannot flood stderr.
        let skipped = (0..MAX_OMISSIONS_SHOWN + 3)
            .map(|i| Skipped {
                path: format!("f{i}.bin"),
                cause: crate::review::filter::SkipCause::Binary,
            })
            .collect();
        let shown = ReviewError::NothingToReview { skipped }.to_string();
        assert!(shown.contains("f0.bin (binary)"), "{shown}");
        assert!(
            !shown.contains(&format!("f{MAX_OMISSIONS_SHOWN}.bin")),
            "{shown}"
        );
        assert!(shown.ends_with(", and 3 more"), "{shown}");
        let none = ReviewError::NothingToReview { skipped: vec![] }.to_string();
        assert_eq!(none, "nothing to review: no reviewable file in the input");
    }
}
