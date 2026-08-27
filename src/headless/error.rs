// Author: Julian Bolivar
// Version: 0.17.0
// Date: 2026-08-27
//!
//! Follows the pattern of [`crate::vault::VaultError`]: foreign types are
//! **stringified** instead of being wrapped, except for the vault error
//! ([`VaultError`]) which is **wrapped** intact so that the exit-mapper (T9) can distinguish
//! its class.
//!
//! **Security invariant:** no error message ever contains a
//! secret — only its class or the stage that failed (inherited from [`VaultError`]).

use thiserror::Error;

use crate::vault::VaultError;

/// Headless subsystem domain errors.
///
/// Each variant names a failure class distinguishable by the edge (`main.rs`) so it can be
/// mapped to an actionable exit code (REQ-H23).
#[derive(Debug, Error)]
pub enum HeadlessError {
    /// **Operator input is invalid** — the misuse class, as opposed to a run that failed.
    ///
    /// Two families produce it. The prompt input (`-i`/stdin): non-UTF8, envelope without
    /// `prompt` under `--input-format json`, unknown field, duplicate key, or pathological
    /// nesting. And **paths named on the command line**: a `-w`/`--workdir` that is not an
    /// existing directory, or one whose chain contains a symlinked component
    /// (`ensure_raw_chain_symlink_free`).
    ///
    /// The second family is the reason this doc no longer says "the input (`-i`/stdin)" — a
    /// variant whose rustdoc enumerates its producers goes stale silently when a new call site
    /// picks it up, and the exit-code mapping is what makes the choice load bearing.
    ///
    /// The message **never** includes the raw prompt content (it could be sensitive); a path
    /// the operator typed is included, since naming it is the whole point. Maps to exit 2.
    #[error("invalid input: {0}")]
    InputInvalid(String),

    /// The input exceeds `MAX_INPUT_BYTES` (DoS bound, REQ-H29). Carries the limit in bytes,
    /// never the content. Maps to exit 2.
    #[error("input exceeds {0} bytes")]
    InputTooLarge(usize),

    /// I/O error reading/writing input or output. Carries the `io::Error` message, with no
    /// sensitive material. Maps to exit 1.
    #[error("I/O error: {0}")]
    Io(String),

    /// Storage-level failure (SQLite) outside the vault's typed corruption class. Maps to exit
    /// 1.
    #[error("storage error: {0}")]
    Storage(String),

    /// The operator (or interactive user) did not confirm an operation or canceled it. The edge
    /// exits with a non-zero code; it is not a system failure. Maps to exit 1.
    #[error("operation cancelled")]
    Aborted,

    /// There is no TTY and the passphrase was not provided via `-p`/`MAGI_PASSPHRASE`
    /// (REQ-H25/REQ-V40): headless **never** hangs waiting for a prompt it cannot read. Maps to
    /// exit 1.
    #[error("no passphrase: use -p or MAGI_PASSPHRASE in non-interactive environments")]
    PassphraseUnavailable,

    /// Error propagated from the Vault subsystem (wrong passphrase, corrupt meta, corrupt DB,
    /// etc.). It is **wrapped** intact —its `Display` is already sanitized— so that the exit-
    /// mapper (T9) can distinguish the concrete class.
    #[error(transparent)]
    Db(VaultError),
}

/// Translates a [`VaultError`] to its corresponding [`HeadlessError`].
///
/// The `match` is **exhaustive without a `_` wildcard** (MAGI CP2 run 1/2 Melchior/Caspar): a
/// new [`VaultError`] variant **breaks the build**, forcing an explicit mapping decision
/// instead of silent degradation. Variants with a direct equivalent in [`HeadlessError`]
/// (`PassphraseUnavailable`/`Aborted`/`Io`/`Storage`) are mapped to it; the rest are
/// **wrapped** in [`HeadlessError::Db`] so T9 can inspect their class.
impl From<VaultError> for HeadlessError {
    fn from(err: VaultError) -> Self {
        match err {
            VaultError::PassphraseUnavailable => HeadlessError::PassphraseUnavailable,
            VaultError::Aborted => HeadlessError::Aborted,
            VaultError::Io(msg) => HeadlessError::Io(msg),
            VaultError::Storage(msg) => HeadlessError::Storage(msg),
            wrapped @ (VaultError::WrongPassphrase
            | VaultError::VaultMetaCorrupt
            | VaultError::DbCorrupt { .. }
            | VaultError::SecretNotFound(_)
            | VaultError::Crypto(_)
            | VaultError::WeakPassphrase(_)
            | VaultError::ValueTooLarge(_)) => HeadlessError::Db(wrapped),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::HeadlessError;
    use crate::vault::VaultError;

    #[test]
    fn test_input_invalid_display_is_user_facing() {
        let e = HeadlessError::InputInvalid("missing prompt".into());
        assert!(e.to_string().contains("missing prompt"));
    }

    #[test]
    fn test_input_too_large_includes_the_limit_only() {
        let e = HeadlessError::InputTooLarge(10 * 1024 * 1024);
        assert!(e.to_string().contains("10485760"));
    }

    #[test]
    fn test_from_vault_passphrase_unavailable_maps_to_dedicated_variant() {
        let e: HeadlessError = VaultError::PassphraseUnavailable.into();
        assert!(matches!(e, HeadlessError::PassphraseUnavailable));
    }

    #[test]
    fn test_from_vault_wrong_passphrase_is_wrapped_in_db() {
        let e: HeadlessError = VaultError::WrongPassphrase.into();
        assert!(matches!(e, HeadlessError::Db(VaultError::WrongPassphrase)));
    }

    #[test]
    fn test_from_vault_db_corrupt_is_wrapped_and_leaks_no_secret() {
        let e: HeadlessError = VaultError::DbCorrupt {
            db_path: PathBuf::from("/tmp/.magi/.magi-rs-memory.db"),
            detail: "data present without envelope".into(),
        }
        .into();
        // Transparent Display forwards the (already-sanitized) VaultError message.
        let msg = e.to_string();
        assert!(msg.contains("data present without envelope"));
        assert!(matches!(e, HeadlessError::Db(VaultError::DbCorrupt { .. })));
    }

    #[test]
    fn test_from_vault_storage_maps_to_dedicated_variant() {
        let e: HeadlessError = VaultError::Storage("disk full".into()).into();
        assert!(matches!(e, HeadlessError::Storage(_)));
    }
}
