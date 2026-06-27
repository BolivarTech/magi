// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-06-26

//! Memory subsystem — tiered, consultable memory with forgetting.
//!
//! Organised as a set of cooperating sub-modules. This first task adds only the
//! configuration surface (`config`); subsequent tasks will add storage, retrieval,
//! decay/eviction, context assembly, and benchmarking.

pub mod config;
