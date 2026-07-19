// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-18
#![deny(missing_docs)]
#![deny(clippy::missing_docs_in_private_items)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(clippy::missing_errors_doc, clippy::missing_panics_doc)]
// Lints de panic/bounds-safety: SOLO en producción. Los tests usan
// `unwrap`/`expect`/indexing idiomáticamente (un fallo en un test ES el test
// fallando, que es el comportamiento correcto).
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

//! Headless: superficie no-interactiva de magi-rs (modo CI/CD + backend de IA).
//!
//! MS1 aloja los errores de dominio ([`HeadlessError`]) y los tipos compartidos
//! del contrato de salida (módulo `types`, `pub(crate)`, declarados aquí para
//! evitar forward-refs entre tareas TDD). El parser, el formateo, los logs y los exit
//! codes llegan en las tareas posteriores de MS1; el cableado del `Agent` en MS2.

mod error;
pub(crate) mod exit;
pub mod input;
pub mod limits;
pub(crate) mod log;
pub(crate) mod output;
pub mod policy;
pub(crate) mod resolution;
pub(crate) mod types;

#[cfg(test)]
pub(crate) mod test_support;

pub use error::HeadlessError;

/// Re-export del fuzz entrypoint del sanitizer para el target externo
/// `fuzz_sanitize_error` (Task 10 / REQ-H35). `#[doc(hidden)]`: no ensancha la
/// API pública documentada, sólo la hace alcanzable desde el crate `fuzz/`.
#[doc(hidden)]
pub use output::fuzz_sanitize_error_entrypoint;
