// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-14
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

//! Vault: fundación criptográfica del agente (endurecimiento).
//!
//! MS1 aloja los errores de dominio y las primitivas de envelope; la
//! superficie de usuario (tabla `vault`, CLI) llega en MS2.

mod cli;
mod envelope;
mod error;
mod master;
mod memguard;
mod store;

pub use cli::{run_vault_cmd, TtyIo, VaultCmd, VaultIo};
pub use envelope::{bootstrap_envelope, fuzz_open_entrypoint, open_envelope, rekey_envelope};
pub use error::VaultError;
pub use master::{
    check_strength, create_passphrase, resolve_passphrase, PassphrasePrompt, TtyPrompt,
    MIN_PASSPHRASE_CHARS, PASSPHRASE_ENV,
};
pub use memguard::{harden_process, MaskedDek};
pub use store::{wire, SecretEntry, SecretStore, VaultStore};
