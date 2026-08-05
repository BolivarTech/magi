// Author: Julian Bolivar Version: 1.0.0 Date: 2026-07-14 DEK/KEK envelope primitives with
// `vault_meta` protected by FEC.
//!
//! # Model
//!
//! A random 32 B **DEK** (Data Encryption Key) encrypts all records; the master key derives a
//! **KEK** (Key Encryption Key) that *wraps* the DEK (`wrap_key`, salt-as-AAD). The wrapped DEK
//! and the salt live in `vault_meta`. Changing the master key only re-wraps the DEK (O(1)); the
//! data is not re-encrypted.
//!
//! # FEC Layers (why the crate blob is FEC-encoded)
//!
//! `cryptovault::CryptoVault::wrap_key` already applies `AEAD → FEC → base64`, but the
//! **base64 remains as the outer layer**: a rotten bit in the base64 text
//! stored breaks the `base64-decode` **before** the crate's internal FEC can correct it. That
//! is why this module applies **its own** [`ConcatenatedFec`] layer over the base64
//! representation on disk: it corrects the bit-rot of the *stored representation* so that the
//! crate's `base64-decode` never sees corrupt input.
//!
//! ## Exact scope of the correction (precision — code review Loop 1)
//!
//! FEC coverage **is not uniform over the whole blob**: [`fec_encode`] prepends a **raw**
//! length prefix (`u32` LE, [`LEN_PREFIX`] bytes) **outside** the FEC region. A rotten bit
//! **inside the prefix** (4 bytes out of ~600) is **not** self-corrected — but it **fails
//! safe**: it always produces [`VaultError::VaultMetaCorrupt`] (never a panic, never a wrong
//! DEK), because a corrupt `pre_len` is rejected by the RS block-count check or fails
//! downstream (invalid salt length / AEAD tag). Verified exhaustively (all 32 single-bit flips
//! of the prefix).
//! *The length cannot be put inside the FEC region:* `ConcatenatedFec::decode`
//! **requires** `pre_len` as a parameter to decode, so the prefix must go
//! outside. Hardening it (e.g., redundancy over the prefix) is a robustness follow-up, not a
//! correctness defect (`dev-docs/PENDING_IMPLEMENTATION.md`).
//!
//! Note: along this path the crate's **internal FEC** rarely corrects anything — if the
//! external `fec_decode` succeeds, it has already recovered the exact bytes, and
//! [`open_envelope`] only then calls `unwrap_key`; if it fails, it returns `VaultMetaCorrupt`
//! without reaching the internal FEC. The crate's FEC remains the defense for the payload in
//! transit over other channels; on disk, the external layer is the one that corrects.
//!
//! # Distinction: corruption vs. wrong key
//!
//! [`open_envelope`] evaluates the FEC **before** the AEAD, so that:
//! - a failed `fec_decode` ⇒ [`VaultError::VaultMetaCorrupt`] (corrupt data);
//! - a failed `unwrap_key` after a successful FEC (AEAD tag) ⇒ [`VaultError::WrongPassphrase`] (wrong key).

use std::sync::{Arc, Mutex};

use cryptovault::fec::{ConcatenatedFec, ErrorCorrection};
use cryptovault::{CryptoError, CryptoVault};
use rusqlite::{Connection, OptionalExtension};
use zeroize::Zeroizing;

use crate::vault::VaultError;

/// Width of the original length prefix (`u32` little-endian) that [`fec_encode`] prepends to
/// the blob so that [`fec_decode`] is self-describing.
const LEN_PREFIX: usize = 4;

/// Output of [`bootstrap_envelope`]: `(salt_fec, wrapped_dek_fec, dek)`.
///
/// The first two entries go to `vault_meta` (FEC-encoded); the third is the plaintext DEK to
/// cache in memory.
type Bootstrapped = (Vec<u8>, Vec<u8>, Zeroizing<Vec<u8>>);

/// Translates a [`cryptovault::CryptoError`] to the vault domain.
///
/// The mapping is unambiguous because each stage produces a distinct variant: `Cipher` is only
/// produced by the AEAD (tag failure = wrong key/AAD);
/// `ErrorCorrection`/`Encoding`/`InvalidInput` only by the FEC/framing layer.
fn map_crypto_err(e: CryptoError) -> VaultError {
    match e {
        CryptoError::Cipher(_) => VaultError::WrongPassphrase,
        CryptoError::ErrorCorrection(_)
        | CryptoError::Encoding(_)
        | CryptoError::InvalidInput(_) => VaultError::VaultMetaCorrupt,
        CryptoError::KeyDerivation(m) => VaultError::Crypto(m),
    }
}

/// Wraps `bytes` in a keyless [`ConcatenatedFec`] layer, with the original length prefixed
/// (`u32` LE) so that decode is self-describing.
///
/// FEC is a **keyless** codec: it corrects bit-rot of the on-disk representation without
/// needing the secret.
///
/// # Errors
///
/// [`VaultError::Crypto`] if `bytes.len()` does not fit in the `u32` prefix (never reachable
/// for `vault_meta` entries, ~48 B; reported as a typed error instead of silently truncating
/// the prefix — defense in release).
fn fec_encode(bytes: &[u8]) -> Result<Vec<u8>, VaultError> {
    let len = u32::try_from(bytes.len()).map_err(|_| {
        VaultError::Crypto("vault_meta entry exceeds the u32 length prefix".to_string())
    })?;
    // Encode first: `ConcatenatedFec` expands ~2.3x, so sizing by `bytes.len()` would force a
    // realloc. With `encoded.len()` the capacity is exact.
    let encoded = ConcatenatedFec::default().encode(bytes);
    let mut out = Vec::with_capacity(LEN_PREFIX + encoded.len());
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&encoded);
    Ok(out)
}

/// Recovers the original bytes from a blob produced by [`fec_encode`].
///
/// # Errors
///
/// [`VaultError::VaultMetaCorrupt`] if the length prefix is missing or
/// `ConcatenatedFec::decode` fails to correct the corruption.
fn fec_decode(blob: &[u8]) -> Result<Vec<u8>, VaultError> {
    // `split_first_chunk` yields the 4-byte prefix as a `&[u8; LEN_PREFIX]` and the remaining
    // payload in one bounds-safe step — no fallible `try_into` (whose error arm was unreachable
    // once the length was already guaranteed).
    let (len_arr, payload) = blob
        .split_first_chunk::<LEN_PREFIX>()
        .ok_or(VaultError::VaultMetaCorrupt)?;
    let pre_len = u32::from_le_bytes(*len_arr) as usize;
    ConcatenatedFec::default()
        .decode(payload, pre_len)
        .map_err(map_crypto_err)
}

/// Bootstraps a new envelope for a given master key.
///
/// Generates a random DEK and salt, derives the KEK from `master` + salt, and wraps the DEK.
/// Returns `(salt_fec, wrapped_dek_fec, dek)`: the first two entries go to `vault_meta` (FEC-
/// encoded), the DEK is cached in memory.
///
/// `master` is the user's current passphrase **as `&str`** (UTF-8 valid by construction; never
/// raw non-UTF8 bytes).
///
/// # Errors
///
/// [`VaultError::Crypto`] if the generation of random material, the KEK derivation, or the DEK
/// wrapping fail.
pub fn bootstrap_envelope(vault: &CryptoVault, master: &str) -> Result<Bootstrapped, VaultError> {
    // An RNG failure during bootstrap is NOT corruption of `vault_meta` (metadata does not
    // exist yet): it is reported as `Crypto`, not `VaultMetaCorrupt`.
    let salt = cryptovault::generate_salt()
        .map_err(|e| VaultError::Crypto(format!("salt generation failed: {e}")))?;
    let dek = cryptovault::generate_dek()
        .map_err(|e| VaultError::Crypto(format!("DEK generation failed: {e}")))?;
    let kek = vault.derive_key(master, &salt).map_err(map_crypto_err)?;
    let wrapped = vault.wrap_key(&kek, &salt, &dek).map_err(map_crypto_err)?;
    let salt_fec = fec_encode(&salt)?;
    let wrapped_fec = fec_encode(wrapped.as_bytes())?;
    Ok((salt_fec, wrapped_fec, dek))
}

/// Opens an existing envelope, recovering the DEK.
///
/// Corrects the bit-rot of `salt_fec`/`wrapped_dek_fec` (FEC) **before** unwrapping, and
/// distinguishes corruption from wrong key.
///
/// # Errors
///
/// - [`VaultError::VaultMetaCorrupt`] if the FEC cannot recover `salt` or the wrapped blob (corruption beyond its capacity), or if the blob is not valid base64.
/// - [`VaultError::WrongPassphrase`] if `master` is incorrect (the AEAD tag of the unwrap fails after a successful FEC). **Retryable; never deletes.**
/// - [`VaultError::Crypto`] on a key-derivation failure.
pub fn open_envelope(
    vault: &CryptoVault,
    master: &str,
    salt_fec: &[u8],
    wrapped_dek_fec: &[u8],
) -> Result<Zeroizing<Vec<u8>>, VaultError> {
    let salt = fec_decode(salt_fec)?;
    let wrapped_bytes = fec_decode(wrapped_dek_fec)?;
    let wrapped = String::from_utf8(wrapped_bytes).map_err(|_| VaultError::VaultMetaCorrupt)?;
    let kek = vault.derive_key(master, &salt).map_err(map_crypto_err)?;
    vault
        .unwrap_key(&kek, &salt, &wrapped)
        .map_err(map_crypto_err)
}

/// Read-only FEC-only check used by `magi vault diagnose` (REQ-H32): verifies that both
/// `vault_meta` blobs (`salt_fec`, `wrapped_dek_fec`) FEC-decode
/// **without ever attempting the AEAD unwrap** — no master passphrase is
/// accepted or needed, so this can run on a DB nobody can currently unlock.
///
/// `pub(super)` (not re-exported from [`crate::vault`]): this is an internal building block for
/// [`crate::vault::diagnose`], not part of the crate's public API surface.
///
/// # Errors
/// [`VaultError::VaultMetaCorrupt`] if either blob fails to FEC-decode (the same failure
/// [`open_envelope`] would report **before** it ever reaches the AEAD stage).
pub(super) fn check_meta_fec(salt_fec: &[u8], wrapped_dek_fec: &[u8]) -> Result<(), VaultError> {
    fec_decode(salt_fec)?;
    fec_decode(wrapped_dek_fec)?;
    Ok(())
}

/// Reads the FEC-encoded `salt` and `wrapped_dek` rows from `vault_meta`.
///
/// # Errors
/// [`VaultError::Storage`] on a SQL failure; [`VaultError::VaultMetaCorrupt`] if a row is
/// missing.
fn read_meta(guard: &Connection) -> Result<(Vec<u8>, Vec<u8>), VaultError> {
    let salt: Option<Vec<u8>> = guard
        .query_row("SELECT value FROM vault_meta WHERE key = 'salt'", [], |r| {
            r.get(0)
        })
        .optional()
        .map_err(|e| VaultError::Storage(e.to_string()))?;
    let wrapped: Option<Vec<u8>> = guard
        .query_row(
            "SELECT value FROM vault_meta WHERE key = 'wrapped_dek'",
            [],
            |r| r.get(0),
        )
        .optional()
        .map_err(|e| VaultError::Storage(e.to_string()))?;
    match (salt, wrapped) {
        (Some(s), Some(w)) => Ok((s, w)),
        _ => Err(VaultError::VaultMetaCorrupt),
    }
}

/// Re-keys the envelope: re-wraps the **same** DEK under a new passphrase (O(1) — no record is
/// re-encrypted). Implements `magi vault passwd` (REQ-V20).
///
/// Steps: (1) **verify `current`** by unwrapping the DEK — a wrong passphrase fails the AEAD
/// tag and returns [`VaultError::WrongPassphrase`], changing nothing (the lock that stops
/// `passwd` being a recovery path); (2) fresh salt → KEK_new → re-wrap the SAME DEK; (3) write
/// `{salt, wrapped_dek}` in ONE `BEGIN IMMEDIATE` transaction, crash-safe (the DB stays
/// openable with the old **or** new passphrase, never bricked — SC-V35). Argon2 runs OUTSIDE
/// the connection lock.
///
/// # Errors
/// [`VaultError::WrongPassphrase`] if `current` is wrong; [`VaultError::Storage`] on a SQL
/// failure or a detected concurrent re-wrap; [`VaultError::VaultMetaCorrupt`] or
/// [`VaultError::Crypto`] on corrupt metadata or a crypto failure.
pub fn rekey_envelope(
    vault: &CryptoVault,
    conn: &Arc<Mutex<Connection>>,
    current: &str,
    new: &str,
) -> Result<(), VaultError> {
    rekey_envelope_inner(vault, conn, current, new, || {})
}

/// Shared body of [`rekey_envelope`] with an injectable hook that runs **between** the initial
/// `vault_meta` read and the write transaction — the exact TOCTOU window the compare-and-abort
/// guards. Production calls it with a no-op closure.
fn rekey_envelope_inner(
    vault: &CryptoVault,
    conn: &Arc<Mutex<Connection>>,
    current: &str,
    new: &str,
    between_read_and_tx: impl FnOnce(),
) -> Result<(), VaultError> {
    // (1) Read current meta (short lock), then release before any Argon2.
    let (salt_fec, wrapped_fec) = {
        let guard = conn.lock().unwrap_or_else(|p| p.into_inner());
        read_meta(&guard)?
    };

    // (1b) Verify `current` and recover the DEK (Argon2 #1, off-lock). Wrong passphrase ⇒
    // WrongPassphrase, nothing written.
    let dek = open_envelope(vault, current, &salt_fec, &wrapped_fec)?;

    between_read_and_tx();

    // (2) Re-wrap the SAME DEK under a fresh salt/KEK (Argon2 #2, off-lock).
    let new_salt = cryptovault::generate_salt().map_err(map_crypto_err)?;
    let kek_new = vault.derive_key(new, &new_salt).map_err(map_crypto_err)?;
    let wrapped_new = vault
        .wrap_key(&kek_new, &new_salt, &dek)
        .map_err(map_crypto_err)?;
    let new_salt_fec = fec_encode(&new_salt)?;
    let new_wrapped_fec = fec_encode(wrapped_new.as_bytes())?;

    // (3) Atomic write with a compare-and-abort against a concurrent re-wrap.
    let mut guard = conn.lock().unwrap_or_else(|p| p.into_inner());
    let tx = guard
        .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .map_err(|e| VaultError::Storage(e.to_string()))?;
    let (_, current_wrapped) = read_meta(&tx)?;
    if current_wrapped != wrapped_fec {
        // Another process re-wrapped between our read and this transaction; abort rather than
        // clobber its envelope.
        return Err(VaultError::Storage(
            "concurrent rekey detected; aborted".to_string(),
        ));
    }
    tx.execute(
        "INSERT OR REPLACE INTO vault_meta (key, value) VALUES ('salt', ?1)",
        [&new_salt_fec],
    )
    .map_err(|e| VaultError::Storage(e.to_string()))?;
    tx.execute(
        "INSERT OR REPLACE INTO vault_meta (key, value) VALUES ('wrapped_dek', ?1)",
        [&new_wrapped_fec],
    )
    .map_err(|e| VaultError::Storage(e.to_string()))?;
    tx.commit()
        .map_err(|e| VaultError::Storage(e.to_string()))?;
    Ok(())
}

/// Test-only wrapper exposing the TOCTOU hook of [`rekey_envelope_inner`].
#[cfg(test)]
pub(crate) fn rekey_envelope_with_hook(
    vault: &CryptoVault,
    conn: &Arc<Mutex<Connection>>,
    current: &str,
    new: &str,
    between_read_and_tx: impl FnOnce(),
) -> Result<(), VaultError> {
    rekey_envelope_inner(vault, conn, current, new, between_read_and_tx)
}

/// Fuzzer entrypoint (`fuzz_vault_meta_decode`, Task 9).
///
/// Splits `data` into `(salt_fec, wrapped_fec)` deterministically and bounds-safely — first
/// `u16` LE = `len(salt_fec)` — and calls [`open_envelope`] with a `CryptoVault::default()` and
/// a fixed master, **discarding** the `Result`. The invariant the fuzzer verifies: *never panic
/// nor deletion*, whatever `data` is.
#[doc(hidden)]
pub fn fuzz_open_entrypoint(data: &[u8]) {
    const SPLIT_PREFIX: usize = 2;
    let Some(prefix) = data.get(0..SPLIT_PREFIX) else {
        return;
    };
    let Ok(len_arr) = <[u8; SPLIT_PREFIX]>::try_from(prefix) else {
        return;
    };
    let salt_len = u16::from_le_bytes(len_arr) as usize;
    let rest = data.get(SPLIT_PREFIX..).unwrap_or(&[]);
    let split = salt_len.min(rest.len());
    let (salt_fec, wrapped_fec) = rest.split_at(split);
    let vault = CryptoVault::default();
    let _ = open_envelope(&vault, "fuzz-master-key-fixed", salt_fec, wrapped_fec);
}

#[cfg(test)]
mod tests {
    use super::{bootstrap_envelope, open_envelope, rekey_envelope, rekey_envelope_with_hook};
    use crate::vault::VaultError;
    use rusqlite::Connection;
    use std::sync::{Arc, Mutex};

    /// Builds an in-memory DB with a `vault_meta` table bootstrapped under `master`. Returns
    /// the shared connection.
    fn meta_conn(master: &str) -> Arc<Mutex<Connection>> {
        let conn = Connection::open_in_memory().expect("mem");
        conn.execute(
            "CREATE TABLE vault_meta (key TEXT PRIMARY KEY, value BLOB NOT NULL)",
            [],
        )
        .expect("ddl");
        let vault = cryptovault::CryptoVault::default();
        let (salt_fec, wrapped_fec, _dek) = bootstrap_envelope(&vault, master).expect("boot");
        conn.execute(
            "INSERT INTO vault_meta (key, value) VALUES ('salt', ?1)",
            [&salt_fec],
        )
        .expect("s");
        conn.execute(
            "INSERT INTO vault_meta (key, value) VALUES ('wrapped_dek', ?1)",
            [&wrapped_fec],
        )
        .expect("w");
        Arc::new(Mutex::new(conn))
    }

    /// Reads back `(salt_fec, wrapped_fec)` from `vault_meta`.
    fn read_back(conn: &Arc<Mutex<Connection>>) -> (Vec<u8>, Vec<u8>) {
        let g = conn.lock().expect("lock");
        let s: Vec<u8> = g
            .query_row("SELECT value FROM vault_meta WHERE key='salt'", [], |r| {
                r.get(0)
            })
            .expect("s");
        let w: Vec<u8> = g
            .query_row(
                "SELECT value FROM vault_meta WHERE key='wrapped_dek'",
                [],
                |r| r.get(0),
            )
            .expect("w");
        (s, w)
    }

    #[test]
    fn test_rekey_opens_with_new_passphrase_and_same_dek_not_with_old() {
        let vault = cryptovault::CryptoVault::default();
        let conn = meta_conn("old-passphrase-long-enough");
        // Capture the DEK before, to prove it is unchanged.
        let (s0, w0) = read_back(&conn);
        let dek0 = open_envelope(&vault, "old-passphrase-long-enough", &s0, &w0).expect("open");

        rekey_envelope(
            &vault,
            &conn,
            "old-passphrase-long-enough",
            "new-passphrase-long-enough",
        )
        .expect("rekey");

        let (s1, w1) = read_back(&conn);
        let dek1 =
            open_envelope(&vault, "new-passphrase-long-enough", &s1, &w1).expect("new opens");
        assert_eq!(dek0.to_vec(), dek1.to_vec()); // SAME DEK (nothing re-encrypted)
        assert!(matches!(
            open_envelope(&vault, "old-passphrase-long-enough", &s1, &w1),
            Err(VaultError::WrongPassphrase)
        ));
    }

    #[test]
    fn test_rekey_with_wrong_current_changes_nothing() {
        let vault = cryptovault::CryptoVault::default();
        let conn = meta_conn("old-passphrase-long-enough");
        let before = read_back(&conn);
        let e =
            rekey_envelope(&vault, &conn, "WRONG", "new-passphrase-long-enough").expect_err("lock");
        assert!(matches!(e, VaultError::WrongPassphrase));
        assert_eq!(read_back(&conn), before); // untouched
    }

    #[test]
    fn test_rekey_detects_concurrent_rewrap_and_aborts_without_writing() {
        let vault = cryptovault::CryptoVault::default();
        let conn = meta_conn("old-passphrase-long-enough");
        let conn2 = conn.clone();
        // The hook runs between our read and our tx: a competitor re-keys first.
        let err = rekey_envelope_with_hook(
            &vault,
            &conn,
            "old-passphrase-long-enough",
            "our-new-passphrase-xyz",
            move || {
                let v = cryptovault::CryptoVault::default();
                rekey_envelope(
                    &v,
                    &conn2,
                    "old-passphrase-long-enough",
                    "winner-passphrase-xyz",
                )
                .expect("competitor rekey");
            },
        )
        .expect_err("must detect the concurrent rewrap");
        assert!(matches!(err, VaultError::Storage(_)));
        // The DB is openable with the WINNER's passphrase (never bricked — SC-V35).
        let (s, w) = read_back(&conn);
        open_envelope(&vault, "winner-passphrase-xyz", &s, &w).expect("winner opens");
    }

    // Test master = base64-like string (UTF-8 valid, like the keyring one).
    const M: &str = "bWFzdGVyLWtleS0zMi1ieXRlcy1iYXNlNjQtc3RyaW5n";

    #[test]
    fn test_envelope_bootstrap_then_open_recovers_same_dek() {
        let vault = cryptovault::CryptoVault::default();
        let (salt_fec, wrapped_fec, dek) = bootstrap_envelope(&vault, M).expect("bootstrap");
        let dek2 = open_envelope(&vault, M, &salt_fec, &wrapped_fec).expect("open");
        assert_eq!(&dek[..], &dek2[..]);
    }

    #[test]
    fn test_open_with_wrong_master_yields_wrong_passphrase_not_corrupt() {
        let vault = cryptovault::CryptoVault::default();
        let (salt_fec, wrapped_fec, _) = bootstrap_envelope(&vault, M).expect("bootstrap");
        let err = open_envelope(
            &vault,
            "d3JvbmctbWFzdGVyLWtleS1zdHJpbmc",
            &salt_fec,
            &wrapped_fec,
        )
        .expect_err("wrong master must fail");
        assert!(matches!(err, VaultError::WrongPassphrase));
    }

    #[test]
    fn test_open_with_fec_uncorrectable_wrapped_dek_yields_corrupt() {
        let vault = cryptovault::CryptoVault::default();
        let (salt_fec, mut wrapped_fec, _) = bootstrap_envelope(&vault, M).expect("bootstrap");
        for b in wrapped_fec.iter_mut() {
            *b ^= 0xFF; // daño masivo, más allá de la FEC
        }
        let err = open_envelope(&vault, M, &salt_fec, &wrapped_fec).expect_err("corrupt must fail");
        assert!(matches!(err, VaultError::VaultMetaCorrupt));
    }

    #[test]
    fn test_single_bit_flip_in_wrapped_dek_is_corrected_by_fec() {
        let vault = cryptovault::CryptoVault::default();
        let (salt_fec, mut wrapped_fec, dek) = bootstrap_envelope(&vault, M).expect("bootstrap");
        // Flip one bit in the FEC PAYLOAD (after the 4-byte length prefix).
        wrapped_fec[super::LEN_PREFIX] ^= 0x01;
        let dek2 = open_envelope(&vault, M, &salt_fec, &wrapped_fec).expect("bit-flip corregible");
        assert_eq!(&dek[..], &dek2[..]);
    }

    #[test]
    fn test_single_bit_flip_in_salt_is_corrected_by_fec() {
        let vault = cryptovault::CryptoVault::default();
        let (mut salt_fec, wrapped_fec, dek) = bootstrap_envelope(&vault, M).expect("bootstrap");
        salt_fec[super::LEN_PREFIX] ^= 0x01;
        let dek2 = open_envelope(&vault, M, &salt_fec, &wrapped_fec).expect("salt bit-flip");
        assert_eq!(&dek[..], &dek2[..]);
    }

    #[test]
    fn test_bit_flip_in_length_prefix_fails_safe_as_corrupt() {
        // The 4-byte length prefix sits OUTSIDE the FEC-protected region (see `fec_encode`), so
        // corruption there is not self-corrected — but it MUST fail safe: a typed
        // `VaultMetaCorrupt`, never a panic and never a wrong DEK (REQ-V35). Documents the
        // intentionally-uncorrected window.
        let vault = cryptovault::CryptoVault::default();
        let (salt_fec, wrapped_fec, _) = bootstrap_envelope(&vault, M).expect("bootstrap");

        // Multi-bit flip: invert the whole 4-byte prefix so `pre_len` is enormous.
        let mut multi = wrapped_fec.clone();
        for b in multi.iter_mut().take(super::LEN_PREFIX) {
            *b ^= 0xFF;
        }
        let err = open_envelope(&vault, M, &salt_fec, &multi).expect_err("prefix corruption fails");
        assert!(matches!(err, VaultError::VaultMetaCorrupt));

        // Single-bit flip in the most-significant prefix byte (LE) — a large `pre_len` the FEC
        // block-count check rejects.
        let mut single = wrapped_fec.clone();
        single[super::LEN_PREFIX - 1] ^= 0x80;
        let err2 = open_envelope(&vault, M, &salt_fec, &single).expect_err("prefix bit-flip fails");
        assert!(matches!(err2, VaultError::VaultMetaCorrupt));
    }

    #[test]
    fn test_check_meta_fec_succeeds_without_attempting_the_aead_unwrap() {
        // A bogus master never enters this check at all — it only takes the two FEC blobs, so a
        // passphrase-less caller (`vault diagnose`) can call it.
        let vault = cryptovault::CryptoVault::default();
        let (salt_fec, wrapped_fec, _dek) = bootstrap_envelope(&vault, M).expect("bootstrap");
        super::check_meta_fec(&salt_fec, &wrapped_fec).expect("both blobs FEC-decode");
    }

    #[test]
    fn test_check_meta_fec_reports_corrupt_on_an_uncorrectable_wrapped_blob() {
        let vault = cryptovault::CryptoVault::default();
        let (salt_fec, mut wrapped_fec, _dek) = bootstrap_envelope(&vault, M).expect("bootstrap");
        for b in wrapped_fec.iter_mut() {
            *b ^= 0xFF; // damage beyond the FEC's correction capacity
        }
        let err = super::check_meta_fec(&salt_fec, &wrapped_fec)
            .expect_err("uncorrectable blob must fail");
        assert!(matches!(err, VaultError::VaultMetaCorrupt));
    }

    #[test]
    fn test_fuzz_entrypoint_never_panics_on_arbitrary_input() {
        // Degenerate inputs: empty, short, random — never panic.
        for data in [
            &b""[..],
            &b"\x00"[..],
            &b"\x05\x00abcdefghij"[..],
            &[0xFFu8; 300],
        ] {
            super::fuzz_open_entrypoint(data);
        }
    }
}
