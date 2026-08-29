// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-29

//! Server-grade logging for magi-rs: a daily file rotated in UTC, `.xz`
//! compression with retention, and an auditor that redacts secrets before they
//! reach any output.
//!
//! Everything decidable lives in pure modules that return decisions; a thin
//! shim executes them. That split is what makes "on day 8 it compresses and on
//! day 31 it is deleted" testable without fabricating 31 days of real files.

pub mod rotation;
