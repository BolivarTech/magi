// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-29

//! Server-grade logging for magi-rs: a daily file rotated in UTC, `.xz`
//! compression with retention, and an auditor that redacts secrets before they
//! reach any output.
//!
//! # The shape, and why it is this shape
//!
//! Everything decidable is a **pure function that returns a decision**;
//! a thin shim executes it. Rotation, retention, chunking and rendering never
//! touch the filesystem or read a clock. That is what makes "on day 8 it is
//! compressed and on day 31 it is deleted" testable with two dates instead of
//! thirty-one days of real files.
//!
//! # Lint policy, which is stricter here than in most of the crate
//!
//! This module is held to the same bar as `vault`: panicking constructs are
//! **denied**, not discouraged. The reason is specific rather than general —
//! this is the subsystem you read when everything else has already failed, so
//! a panic inside it takes the diagnostic channel down at the exact moment it
//! is needed. Fallible operations return `Result` and degrade to a documented
//! best effort; they never abort the process that was trying to log.
//!
//! The denials are lifted under `cfg(test)`, where `unwrap` on a literal date
//! is clarity rather than risk.

#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
#![cfg_attr(not(test), deny(clippy::panic))]
#![cfg_attr(not(test), deny(clippy::todo))]
#![cfg_attr(not(test), deny(clippy::unimplemented))]
#![deny(missing_docs)]

use std::path::PathBuf;

/// Everything this subsystem can fail at.
///
/// **Defined here because no task owned it.** Three tasks of the plan return
/// `Result<_, LoggingError>` — the compressor, the retention executor and
/// `init_logging` — and none declared the type. `mod.rs` is the subsystem's API
/// surface and is already one of the milestone's files, so putting it here
/// keeps the file count honest.
#[derive(Debug, thiserror::Error)]
pub enum LoggingError {
    /// The log directory could not be created.
    #[error("cannot create the log directory {path}: {source}")]
    DirCreate {
        /// Directory that could not be created.
        path: PathBuf,
        /// Underlying failure.
        source: std::io::Error,
    },
    /// A write, create or rename failed.
    #[error("cannot write {path}: {source}")]
    Write {
        /// File the operation targeted.
        path: PathBuf,
        /// Underlying failure.
        source: std::io::Error,
    },
    /// Compression, its read-back, or the comparison failed.
    #[error("cannot compress {path}: {source}")]
    Compress {
        /// File being compressed, or the staged temporary.
        path: PathBuf,
        /// Underlying failure.
        source: std::io::Error,
    },
    /// An operator-supplied filter directive could not be parsed.
    #[error("invalid filter directive {directive:?}: {reason}")]
    FilterInvalid {
        /// The directive as written.
        directive: String,
        /// Why it was rejected.
        reason: String,
    },
}

pub mod appender;
pub mod auditor;
pub mod chunk;
pub mod render;
pub mod retention;
pub mod rotation;
pub mod xz;

#[cfg(test)]
pub(crate) mod testutil;
