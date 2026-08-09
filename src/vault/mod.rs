// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-14
#![deny(missing_docs)]
#![deny(clippy::missing_docs_in_private_items)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(clippy::missing_errors_doc, clippy::missing_panics_doc)]
// Panic/bounds-safety lints: **ONLY** in production. Tests use `unwrap`/`expect`/indexing
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

//! Vault: cryptographic foundation of the agent (hardening).
//!
//! MS1 hosts the domain errors and the envelope primitives; the user surface (table `vault`,
//! CLI) arrives in MS2.

mod cli;
mod diagnose;
mod envelope;
mod error;
mod master;
mod memguard;
mod store;

pub use cli::{run_vault_cmd, TtyIo, VaultCmd, VaultIo};
pub use diagnose::{
    diagnose, format_diagnose_report, DiagnoseReport, DiagnoseVerdict, TableCounts,
};
pub use envelope::{bootstrap_envelope, fuzz_open_entrypoint, open_envelope, rekey_envelope};
pub use error::VaultError;
pub use master::{
    check_strength, create_passphrase, fuzz_passphrase_entrypoint, resolve_passphrase,
    strip_trailing_newline, PassphrasePrompt, TtyPrompt, MIN_PASSPHRASE_CHARS, PASSPHRASE_ENV,
};
pub use memguard::{harden_process, MaskedDek};
pub use store::{fuzz_value_roundtrip_entrypoint, wire, SecretEntry, SecretStore, VaultStore};
