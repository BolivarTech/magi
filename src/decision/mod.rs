//! Typed decisions served by System One models (REQ-D00..D24).
//!
//! This module holds the System One request and response types, the input
//! validation shared by every surface that asks for a decision, and the
//! error and trace types a decision carries.
#![deny(missing_docs)]
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing,
        clippy::string_slice
    )
)]

pub mod client;
pub mod config;
pub mod endpoint;
pub mod envelope;
pub mod error;
pub mod protocol;

pub use client::{DecisionClient, DecisionSettings, SystemOneClient};
pub use config::{DecisionConfigError, DecisionFallback, DecisionSection, KeepAlive, ModelSpec};
pub use error::{Attempt, AttemptOutcome, Cause, Decision, DecisionError, ErrorKind, Trace, Usage};
pub use protocol::{Answer, DecisionRequest, NoulCriteria, Question, Questions};
