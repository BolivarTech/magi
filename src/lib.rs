// Author: Julian Bolivar
// Version: 0.25.0
// Date: 2026-11-08
#![forbid(unsafe_code)]
//! magi-rs library: exposes the `magi`, `vault`, and `headless` subsystems for fuzzing,
//! coverage, and tests, as well as the `main.rs` binary.
pub mod decision;
pub mod encoding;
pub mod endpoint_key;
pub mod headless;
pub mod logging;
pub mod magi;
pub mod notices;
pub mod redact;
pub mod review;
pub mod section_keys;
pub mod vault;
