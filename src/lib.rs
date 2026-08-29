// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-14
#![forbid(unsafe_code)]
//! magi-rs library: exposes the `magi`, `vault`, and `headless` subsystems for fuzzing,
//! coverage, and tests, as well as the `main.rs` binary.
pub mod encoding;
pub mod headless;
pub mod logging;
pub mod magi;
pub mod notices;
pub mod redact;
pub mod vault;
