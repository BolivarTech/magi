// Author: Julian Bolivar
// Version: 0.18.0
// Date: 2026-08-31
#![deny(missing_docs)]
#![deny(clippy::missing_docs_in_private_items)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(clippy::missing_errors_doc, clippy::missing_panics_doc)]
// Panic/bounds-safety lints: ONLY in production. Tests use `unwrap`/`expect`/indexing
// idiomatically (a failure in a test IS the test failing, which is the correct behavior).
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

//! Headless: non-interactive surface of magi-rs (CI/CD mode + AI backend).
//!
//! MS1 hosts the domain errors ([`HeadlessError`]) and the shared types of the output contract
//! (module `types`, declared here to avoid forward-refs between TDD tasks). The parser,
//! formatting, logs, and exit codes arrive in later MS1 tasks. All modules are `pub`: the MS2
//! runner lives in the binary crate and can only reach `pub` API of the lib (a bin cannot reach
//! `pub(crate)`).

mod error;
pub mod exit;
pub mod input;
pub mod limits;
pub mod output;
pub mod policy;
pub mod resolution;
pub mod types;

#[cfg(test)]
pub(crate) mod test_support;

pub use error::HeadlessError;

/// Re-export of the sanitizer's fuzz entrypoint for the external target `fuzz_sanitize_error`
/// (Task 10 / REQ-H35). `#[doc(hidden)]`: it does not widen the documented public API, it only
/// makes it reachable from the `fuzz/` crate.
#[doc(hidden)]
pub use output::fuzz_sanitize_error_entrypoint;

/// Re-export of the tier-policy matrix's fuzz entrypoint for the external target `fuzz_policy`
/// (MS2 Task 10 / REQ-H35). `#[doc(hidden)]`: same convention — reachable from `fuzz/` without
/// widening the documented API.
#[doc(hidden)]
pub use policy::fuzz_policy_entrypoint;
