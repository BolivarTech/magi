// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-14
#![forbid(unsafe_code)]
//! Biblioteca de magi-rs: expone los subsistemas `magi`, `vault` y `headless`
//! para fuzzing, cobertura y tests, además del binario `main.rs`.
pub mod headless;
pub mod magi;
pub mod redact;
pub mod vault;
