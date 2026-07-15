// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-14
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(clippy::missing_errors_doc, clippy::missing_panics_doc)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![deny(clippy::todo, clippy::unimplemented)]
#![deny(clippy::indexing_slicing, clippy::string_slice)]

//! Vault: fundación criptográfica del agente (endurecimiento).
//!
//! MS1 aloja los errores de dominio y las primitivas de envelope; la
//! superficie de usuario (tabla `vault`, CLI) llega en MS2.

mod error;

pub use error::VaultError;
