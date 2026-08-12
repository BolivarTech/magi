// Author: Julian Bolivar Version: 1.0.0 Date: 2026-07-14 Vault subsystem domain errors.
//!
//! Follows the pattern of [`crate::memory::error::MemoryError`]: foreign types (`cryptovault`,
//! `rusqlite`) are **stringified** instead of wrapped, to keep external types out of the
//! vault's public API and preserve `Send + Sync` without coupling to their versions.
//!
//! **Security invariant:** no error message ever contains the
//! *value* of a secret — only its *name* or the stage that failed.

use std::path::PathBuf;

use thiserror::Error;

/// Vault subsystem domain errors.
///
/// Each variant names a failure stage distinguishable by the caller; the underlying AEAD
/// prevents forgery regardless of which one is exposed.
#[derive(Debug, Clone, Error)]
pub enum VaultError {
    /// The DEK unwrap failed the AEAD tag after FEC correction — that is, the master key is
    /// **incorrect**.
    ///
    /// It is **retryable** and **never** triggers data deletion: see the never-delete policy
    /// (REQ-V35).
    #[error("incorrect passphrase")]
    WrongPassphrase,

    /// `vault_meta` is present but unrecoverable **even after** FEC correction (corruption
    /// beyond the codec's capacity).
    ///
    /// Requires **explicit** user action; the system never self-repairs by destroying data.
    #[error("vault metadata is corrupt and unrecoverable")]
    VaultMetaCorrupt,

    /// The `.magi/` DB is **corrupt**: encrypted data is present but
    /// **no** envelope (DEK) exists to decrypt them, or an expected table is missing
    /// from the schema (§2.1 / D-H10). **Absolute never-delete:** this state
    /// **never** triggers a deletion or a bootstrap on top — requires
    /// explicit user action (restore a backup or remove `.magi/` manually).
    ///
    /// The variant is **structured** so the edge can build actionable recovery text: `db_path`
    /// names which DB and `detail` why. The `Display` exposes **only** the path and the class —
    /// **never** a secret.
    #[error("database corrupt at {}: {detail}", .db_path.display())]
    DbCorrupt {
        /// Path of the affected `.magi/` DB (not sensitive material).
        db_path: PathBuf,
        /// Class of corruption (e.g. "data present without envelope" or "missing table
        /// `<name>`") — **never** contains a secret.
        detail: String,
    },

    /// No secret exists with the given name. The name is not sensitive material, so it may
    /// appear in the message.
    #[error("secret not found: {0}")]
    SecretNotFound(String),

    /// Cryptographic failure propagated from `cryptovault` (message already sanitized by the
    /// crate — no decode or timing oracles).
    #[error("crypto error: {0}")]
    Crypto(String),

    /// Failure at the SQLite storage level.
    #[error("storage error: {0}")]
    Storage(String),

    /// No TTY and the passphrase was not provided via `-p`/`MAGI_PASSPHRASE` (REQ-V40): the
    /// passphrase is **never** read from a pipe. Retryable.
    #[error("no passphrase: use -p or MAGI_PASSPHRASE in non-interactive environments")]
    PassphraseUnavailable,

    /// The passphrase does not reach the hard strength floor (REQ-V18). The message carries the
    /// reasons + tips, **never** the passphrase.
    #[error("passphrase rejected: {0}")]
    WeakPassphrase(String),

    /// Terminal I/O error (hidden prompt / echo). Message from the `io::Error`, no sensitive
    /// material.
    #[error("I/O error: {0}")]
    Io(String),

    /// The user did not confirm a destructive operation (REQ-V22). The CLI exits with a non-
    /// zero exit code so scripts detect it; it is not a system failure.
    #[error("operation cancelled")]
    Aborted,

    /// The value exceeds `cryptovault::MAX_PLAINTEXT_LEN` (MAGI run 4, Caspar). It carries the
    /// limit, never the value.
    #[error("value exceeds {0} bytes")]
    ValueTooLarge(usize),
}

#[cfg(test)]
mod tests {
    use super::VaultError;

    #[test]
    fn test_wrong_passphrase_display_is_user_facing_and_leaks_nothing() {
        let e = VaultError::WrongPassphrase;
        assert_eq!(e.to_string(), "incorrect passphrase");
    }

    #[test]
    fn test_vault_meta_corrupt_is_distinct_variant() {
        let e = VaultError::VaultMetaCorrupt;
        assert!(e.to_string().to_lowercase().contains("corrupt"));
    }

    #[test]
    fn test_secret_not_found_includes_name_only() {
        let e = VaultError::SecretNotFound("OPENAI_API_KEY".to_string());
        let msg = e.to_string();
        assert!(msg.contains("OPENAI_API_KEY"));
    }

    #[test]
    fn test_aborted_display_is_user_facing_and_stable() {
        assert_eq!(VaultError::Aborted.to_string(), "operation cancelled");
    }

    #[test]
    fn test_value_too_large_includes_the_limit_only() {
        let e = VaultError::ValueTooLarge(10 * 1024 * 1024);
        assert!(e.to_string().contains("10485760"));
    }

    #[test]
    fn test_db_corrupt_display_has_path_and_class_and_leaks_nothing() {
        let e = VaultError::DbCorrupt {
            db_path: std::path::PathBuf::from("/tmp/.magi/.magi-rs-memory.db"),
            detail: "data present without envelope".to_string(),
        };
        let msg = e.to_string();
        assert!(msg.contains(".magi-rs-memory.db"));
        assert!(msg.contains("data present without envelope"));
    }
}
