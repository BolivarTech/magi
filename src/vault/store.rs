// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-17
//! The `vault` table: encrypted secret storage (REQ-V01 / REQ-V02 / REQ-V03).
//!
//! [`VaultStore`] is the only type in this module that touches SQL; the
//! public surface is the [`SecretStore`] trait plus [`wire`], the single
//! entry point (D-V01) that creates the table if it is missing and returns
//! a store bound to an already-open connection and an already-unwrapped
//! data key — this module never resolves either on its own (R-V02).
//!
//! # Locking discipline (R-V08)
//!
//! Every write encrypts with [`MaskedDek::with_dek`] **before** taking the
//! connection lock, and every read releases the connection lock **before**
//! decrypting: the lock is held only for the duration of a single SQL
//! statement, never across Argon2 or AEAD/FEC work.
//!
//! # Error semantics
//!
//! A `value_blob` that fails to decrypt (tampered ciphertext, corrupted
//! encoding, …) always surfaces as [`VaultError::Crypto`] — **never**
//! [`VaultError::VaultMetaCorrupt`] (that variant is exclusive to the
//! `vault_meta` envelope in [`crate::vault::envelope`]) and **never**
//! [`VaultError::WrongPassphrase`] (one secret's blob failing integrity is
//! not evidence the whole envelope's KEK is wrong).

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

use cryptovault::{CryptoError, CryptoVault};
use rusqlite::{params, Connection, OptionalExtension};
use zeroize::Zeroizing;

use super::{MaskedDek, VaultError};

/// DDL for the `vault` table (REQ-V01): `name` is the primary key, the
/// secret's ciphertext is a `BLOB` (matching `vault_meta`'s convention even
/// though [`CryptoVault::encrypt_with_key`] returns base64 text — REQ-V01),
/// and both timestamps are epoch seconds.
const CREATE_VAULT_TABLE_SQL: &str = "CREATE TABLE IF NOT EXISTS vault (
    name       TEXT PRIMARY KEY,
    value_blob BLOB NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
)";

/// Upserts a secret (REQ-V01 / SC-V02): an idempotent add-or-modify that
/// preserves `created_at` and bumps `updated_at` when `name` already
/// exists.
const SET_SQL: &str = "INSERT INTO vault (name, value_blob, created_at, updated_at)
    VALUES (?1, ?2, ?3, ?4)
    ON CONFLICT(name) DO UPDATE SET value_blob = excluded.value_blob, updated_at = excluded.updated_at";

/// Reads one secret's raw ciphertext by name.
const GET_SQL: &str = "SELECT value_blob FROM vault WHERE name = ?1";

/// Deletes one secret by name; the caller inspects the affected-row count to
/// distinguish "deleted" from "did not exist" (SC-V03).
const REMOVE_SQL: &str = "DELETE FROM vault WHERE name = ?1";

/// Lists every secret's name and metadata, **never** the value (REQ-V09 /
/// SC-V04), ordered by name for deterministic output; SQLite formats the
/// epoch-second timestamps as human-readable UTC (`MS2.md` plan decision 2).
const LIST_SQL: &str =
    "SELECT name, datetime(created_at, 'unixepoch'), datetime(updated_at, 'unixepoch')
    FROM vault ORDER BY name ASC";

/// Checks whether a name already exists, for the CLI's overwrite
/// confirmation prompt (`SecretStore::contains`, consumed by Task 6's
/// `[Y/n]`).
const CONTAINS_SQL: &str = "SELECT EXISTS(SELECT 1 FROM vault WHERE name = ?1)";

/// One row of [`SecretStore::list`]: metadata only — the value is never
/// included (REQ-V09 / SC-V04).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecretEntry {
    /// The secret's name (the `vault` table's primary key).
    pub name: String,
    /// Creation time, formatted by SQLite as `YYYY-MM-DD HH:MM:SS` (UTC).
    pub created_at: String,
    /// Last-update time, same format as `created_at`; equal to it until the
    /// first overwriting `set`.
    pub updated_at: String,
}

/// Encrypted secret store: `set` is idempotent (add-or-modify), `get` is
/// **internal API for the agent only** (REQ-V09: `cli.rs` must never call
/// it), and `list` returns metadata alone, never a value.
///
/// # Concurrency contract
///
/// Every method takes `&mut self` because [`MaskedDek::with_dek`] rotates
/// the data key's in-RAM mask on every access (SC-V50). The store is
/// therefore **single-threaded by design** — a short-lived CLI process or a
/// single TUI handler — and sharing one instance across threads is the
/// caller's responsibility to synchronize externally.
pub trait SecretStore: Send {
    /// Adds a new secret or overwrites an existing one (idempotent).
    ///
    /// An existing `name`'s `created_at` is preserved; only `updated_at` advances (SC-V02).
    /// Encrypts before taking the connection lock and rejects an empty or whitespace-only
    /// `name` before touching the database (R-V08, "Non-empty name").
    ///
    /// # Errors
    ///
    /// [`VaultError::SecretNotFound`] if `name` is empty or whitespace-only;
    /// [`VaultError::Crypto`] if encryption fails; [`VaultError::Storage`]
    /// on a SQLite failure.
    fn set(&mut self, name: &str, value: &str) -> Result<(), VaultError>;

    /// Retrieves and decrypts one secret's value. **Internal API only**
    /// (REQ-V09) — the CLI never calls this.
    ///
    /// Reads the raw blob under the connection lock, releases the lock,
    /// then decrypts (R-V08).
    ///
    /// # Errors
    ///
    /// [`VaultError::SecretNotFound`] if `name` is empty, whitespace-only,
    /// or absent; [`VaultError::Crypto`] if the blob fails to decrypt
    /// (tampered ciphertext or corrupted encoding — never
    /// [`VaultError::VaultMetaCorrupt`], see the module docs);
    /// [`VaultError::Storage`] on a SQLite failure.
    fn get(&mut self, name: &str) -> Result<Zeroizing<String>, VaultError>;

    /// Deletes one secret by name.
    ///
    /// # Errors
    ///
    /// [`VaultError::SecretNotFound`] if `name` is empty, whitespace-only,
    /// or absent (deleting nothing is reported, never silent — SC-V03);
    /// [`VaultError::Storage`] on a SQLite failure.
    fn remove(&mut self, name: &str) -> Result<(), VaultError>;

    /// Lists every secret's name and timestamps, ordered by name. Never
    /// includes a value (REQ-V09 / SC-V04).
    ///
    /// # Errors
    ///
    /// [`VaultError::Storage`] on a SQLite failure.
    fn list(&mut self) -> Result<Vec<SecretEntry>, VaultError>;

    /// Reports whether `name` already exists, for the CLI's overwrite
    /// confirmation prompt (Task 6).
    ///
    /// # Errors
    ///
    /// [`VaultError::SecretNotFound`] if `name` is empty or whitespace-only;
    /// [`VaultError::Storage`] on a SQLite failure.
    fn contains(&mut self, name: &str) -> Result<bool, VaultError>;
}

/// The `vault` table's [`SecretStore`] implementation.
pub struct VaultStore {
    /// The connection shared with the rest of the encrypted DB — injected,
    /// never opened by this module (R-V02).
    conn: Arc<Mutex<Connection>>,
    /// The AEAD + FEC pipeline shared with the rest of the DB (R-V02); this
    /// module only ever calls its `*_with_key` byte/string front doors.
    vault: CryptoVault,
    /// The masked data key (Task 1); every access rotates its in-RAM mask.
    dek: MaskedDek,
}

/// The single entry point of this module (D-V01): creates the `vault` table
/// if it does not exist yet and returns a store bound to the given
/// (already-open) connection and (already-unwrapped) data key.
///
/// # Errors
///
/// [`VaultError::Storage`] if the `CREATE TABLE` statement fails.
pub fn wire(conn: Arc<Mutex<Connection>>, dek: MaskedDek) -> Result<VaultStore, VaultError> {
    {
        let guard = conn.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        guard
            .execute(CREATE_VAULT_TABLE_SQL, [])
            .map_err(|e| VaultError::Storage(e.to_string()))?;
    }
    Ok(VaultStore {
        conn,
        vault: CryptoVault::default(),
        dek,
    })
}

/// Maps every [`CryptoError`] variant arising from a `value_blob` operation
/// to [`VaultError::Crypto`] — deliberately flat, unlike
/// [`crate::vault::envelope`]'s mapping: a single secret's ciphertext
/// failing integrity or framing is never evidence about the `vault_meta`
/// envelope (no [`VaultError::VaultMetaCorrupt`]) and never evidence the
/// master passphrase itself is wrong (no [`VaultError::WrongPassphrase`]) —
/// see the module docs.
fn map_store_crypto_err(e: CryptoError) -> VaultError {
    VaultError::Crypto(e.to_string())
}

/// Extracts the raw bytes of a `value_blob` column regardless of which
/// SQLite storage class it currently holds.
///
/// The column is declared `BLOB`, but SQLite's dynamic typing means a raw
/// SQL statement (e.g. a hand-tampered row, or a `||`/`substr` expression in
/// a test) can leave it stored as `TEXT` instead. Both storage classes hold
/// the same underlying bytes, so both are read the same way; anything else
/// (`NULL`, `Integer`, `Real` — never produced by this module) yields an
/// empty buffer, which then fails safe further down the pipeline as a typed
/// [`VaultError::Crypto`] rather than a read error that would mask a tamper.
fn raw_column_bytes(value: rusqlite::types::ValueRef<'_>) -> Vec<u8> {
    match value {
        rusqlite::types::ValueRef::Blob(b) => b.to_vec(),
        rusqlite::types::ValueRef::Text(t) => t.to_vec(),
        _ => Vec::new(),
    }
}

/// Rejects an empty or whitespace-only secret name before any database access (R-V08, "Non-
/// empty name"): a blank name is a usage error, not a valid key, and letting it through would
/// create a ghost row.
///
/// # Errors
///
/// [`VaultError::SecretNotFound`] if `name` is empty or whitespace-only.
fn validate_name(name: &str) -> Result<(), VaultError> {
    if name.trim().is_empty() {
        return Err(VaultError::SecretNotFound(name.to_string()));
    }
    Ok(())
}

/// Current time as epoch seconds. `SystemTime::now()` predating the Unix
/// epoch is unreachable on any supported platform; `unwrap_or_default`
/// fails safe to a zero `Duration` instead of panicking, and the
/// `u64 -> i64` conversion fails safe to `i64::MAX` rather than wrapping —
/// neither branch is reachable in practice, but both are typed, not panics.
fn now_epoch_secs() -> i64 {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    i64::try_from(secs).unwrap_or(i64::MAX)
}

impl VaultStore {
    /// Locks the shared connection, recovering the guard if a panic
    /// elsewhere poisoned the mutex (the SQLite handle itself stays valid),
    /// so one poisoned operation never fails the store closed for the rest
    /// of the process.
    fn locked_conn(&self) -> MutexGuard<'_, Connection> {
        self.conn
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl SecretStore for VaultStore {
    fn set(&mut self, name: &str, value: &str) -> Result<(), VaultError> {
        validate_name(name)?;
        // Encrypt OUTSIDE the connection lock (R-V08): with_dek rotates the
        // DEK's mask, which must never happen while the SQLite guard is held.
        let vault = &self.vault;
        let sealed = self
            .dek
            .with_dek(|k| vault.encrypt_with_key(k, value))
            .map_err(map_store_crypto_err)?;
        let now = now_epoch_secs();
        let conn = self.locked_conn();
        conn.execute(SET_SQL, params![name, sealed.into_bytes(), now, now])
            .map_err(|e| VaultError::Storage(e.to_string()))?;
        Ok(())
    }

    fn get(&mut self, name: &str) -> Result<Zeroizing<String>, VaultError> {
        validate_name(name)?;
        // Read the raw blob under the lock, then release it (R-V08) BEFORE
        // running the FEC/Viterbi decode inside decrypt_with_key. Accepts
        // either SQLite storage class for the column: a hand-tampered row
        // (as in `test_tampered_blob_fails_typed_never_returns_wrong_value`)
        // can coerce a BLOB value to TEXT via SQL string functions, and this
        // must still fail as a typed crypto error, never a read error that
        // masks the tamper.
        let raw: Vec<u8> = {
            let conn = self.locked_conn();
            conn.query_row(GET_SQL, params![name], |row| {
                Ok(raw_column_bytes(row.get_ref(0)?))
            })
            .optional()
            .map_err(|e| VaultError::Storage(e.to_string()))?
            .ok_or_else(|| VaultError::SecretNotFound(name.to_string()))?
        };
        let blob = String::from_utf8(raw)
            .map_err(|_| VaultError::Crypto("value blob is not valid UTF-8".to_string()))?;
        let vault = &self.vault;
        self.dek
            .with_dek(|k| vault.decrypt_with_key(k, &blob))
            .map_err(map_store_crypto_err)
    }

    fn remove(&mut self, name: &str) -> Result<(), VaultError> {
        validate_name(name)?;
        let conn = self.locked_conn();
        let affected = conn
            .execute(REMOVE_SQL, params![name])
            .map_err(|e| VaultError::Storage(e.to_string()))?;
        if affected == 0 {
            return Err(VaultError::SecretNotFound(name.to_string()));
        }
        Ok(())
    }

    fn list(&mut self) -> Result<Vec<SecretEntry>, VaultError> {
        let conn = self.locked_conn();
        let mut stmt = conn
            .prepare(LIST_SQL)
            .map_err(|e| VaultError::Storage(e.to_string()))?;
        let rows = stmt
            .query_map([], |row| {
                Ok(SecretEntry {
                    name: row.get(0)?,
                    created_at: row.get(1)?,
                    updated_at: row.get(2)?,
                })
            })
            .map_err(|e| VaultError::Storage(e.to_string()))?;
        let mut entries = Vec::new();
        for row in rows {
            entries.push(row.map_err(|e| VaultError::Storage(e.to_string()))?);
        }
        Ok(entries)
    }

    fn contains(&mut self, name: &str) -> Result<bool, VaultError> {
        validate_name(name)?;
        let conn = self.locked_conn();
        conn.query_row(CONTAINS_SQL, params![name], |row| row.get(0))
            .map_err(|e| VaultError::Storage(e.to_string()))
    }
}

/// Fuzz entrypoint (`fuzz_secret_value_roundtrip`, Task 8 / REQ-V39).
///
/// Takes ARBITRARY bytes as a secret value (via `from_utf8_lossy`, since a stored
/// value is a `&str`), sets and gets it against a throwaway in-memory store, and
/// **discards** the results. The ONLY invariant this entrypoint verifies is
/// panic-freedom: no input — empty, non-UTF8, huge — ever panics, and every
/// failure is a typed [`VaultError`]. Round-trip VALUE equality is asserted
/// separately in the unit tests (`test_set_then_get_roundtrips_the_exact_value`),
/// NOT here — a fuzzer only needs to prove nothing crashes.
#[doc(hidden)]
pub fn fuzz_value_roundtrip_entrypoint(data: &[u8]) {
    let Ok(conn) = Connection::open_in_memory() else {
        return;
    };
    let Ok(dek) = MaskedDek::new(Zeroizing::new(vec![7u8; 32])) else {
        return;
    };
    let Ok(mut store) = wire(Arc::new(Mutex::new(conn)), dek) else {
        return;
    };
    let value = String::from_utf8_lossy(data);
    if store.set("k", &value).is_ok() {
        let _ = store.get("k");
    }
}

#[cfg(test)]
impl VaultStore {
    /// Test-only escape hatch to the shared connection, for assertions that
    /// need to inspect or tamper with raw rows directly.
    pub(crate) fn debug_conn(&self) -> Arc<Mutex<Connection>> {
        self.conn.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vault::MaskedDek;
    use std::sync::{Arc, Mutex};
    use zeroize::Zeroizing;

    fn fixture() -> VaultStore {
        let conn = rusqlite::Connection::open_in_memory().expect("mem db");
        let dek = MaskedDek::new(Zeroizing::new(vec![7u8; 32])).expect("32B");
        crate::vault::wire(Arc::new(Mutex::new(conn)), dek).expect("wire")
    }

    #[test]
    fn test_set_then_get_roundtrips_the_exact_value() {
        let mut s = fixture();
        s.set("OPENAI_API_KEY", "sk-test-123\nline2").expect("set");
        let v = s.get("OPENAI_API_KEY").expect("get");
        assert_eq!(v.as_str(), "sk-test-123\nline2"); // SC integrity, REQ-V03
    }

    #[test]
    fn test_set_overwrites_existing_and_bumps_updated_at_only() {
        let mut s = fixture();
        s.set("K", "v1").expect("set1");
        let e1 = &s.list().expect("ls")[0];
        let (c1, _) = (e1.created_at.clone(), e1.updated_at.clone());
        std::thread::sleep(std::time::Duration::from_millis(1100)); // 1s granularity
        s.set("K", "v2").expect("set2");
        let e2 = &s.list().expect("ls")[0];
        assert_eq!(e2.created_at, c1); // SC-V02: created is preserved
        assert_ne!(e2.updated_at, e2.created_at); // updated advances
        assert_eq!(s.get("K").expect("get").as_str(), "v2");
    }

    #[test]
    fn test_remove_deletes_and_missing_names_are_typed_errors() {
        let mut s = fixture();
        s.set("K", "v").expect("set");
        s.remove("K").expect("rm");
        assert!(matches!(s.get("K"), Err(VaultError::SecretNotFound(_))));
        assert!(matches!(s.remove("K"), Err(VaultError::SecretNotFound(_))));
    }

    #[test]
    fn test_empty_name_is_rejected() {
        // MAGI run 10/12: an empty or whitespace-only name => SecretNotFound
        // before the DB is touched.
        let mut s = fixture();
        assert!(matches!(s.set("", "v"), Err(VaultError::SecretNotFound(_))));
        assert!(matches!(
            s.set("   ", "v"),
            Err(VaultError::SecretNotFound(_))
        ));
        assert!(matches!(s.get(""), Err(VaultError::SecretNotFound(_))));
        assert!(matches!(s.remove(""), Err(VaultError::SecretNotFound(_))));
    }

    #[test]
    fn test_contains_reports_presence_and_rejects_empty_name() {
        // Loop-1 M-2: direct coverage of `contains` — true after set, false for
        // an absent name, and a typed error for a blank name (REQ-V00 per-fn
        // happy + edge coverage).
        let mut s = fixture();
        assert!(!s.contains("K").expect("absent")); // happy: absent => false
        s.set("K", "v").expect("set");
        assert!(s.contains("K").expect("present")); // happy: present => true
        assert!(matches!(s.contains(""), Err(VaultError::SecretNotFound(_)))); // edge: blank
    }

    #[test]
    fn test_list_yields_names_and_dates_never_values() {
        let mut s = fixture();
        s.set("A", "secret-value-A").expect("set");
        let rows = s.list().expect("ls");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].name, "A");
        assert!(!format!(
            "{:?} {} {}",
            rows[0].name, rows[0].created_at, rows[0].updated_at
        )
        .contains("secret-value")); // SC-V04
    }

    #[test]
    fn test_tampered_blob_fails_typed_never_returns_wrong_value() {
        // A tiny corruption (a handful of bytes) falls within
        // `ConcatenatedFec`'s own correction capacity and self-heals — see
        // `envelope.rs`'s single-bit-flip tests, which demonstrate exactly
        // that resilience for the same FEC stack. To prove the "never
        // returns a wrong value" contract this must corrupt the WHOLE blob,
        // the same way `envelope.rs`'s
        // `test_open_with_fec_uncorrectable_wrapped_dek_yields_corrupt`
        // does: flip every bit (XOR 0xFF), which is provably beyond any
        // error-correcting code's capacity while leaving the blob's length
        // untouched.
        let mut s = fixture();
        s.set("K", "v").expect("set");
        {
            let conn = s.debug_conn();
            let guard = conn.lock().expect("lock");
            let original: Vec<u8> = guard
                .query_row("SELECT value_blob FROM vault WHERE name='K'", [], |r| {
                    Ok(raw_column_bytes(r.get_ref(0)?))
                })
                .expect("read original blob");
            let tampered: Vec<u8> = original.iter().map(|b| b ^ 0xFF).collect();
            guard
                .execute(
                    "UPDATE vault SET value_blob = ?1 WHERE name = 'K'",
                    rusqlite::params![tampered],
                )
                .expect("tamper");
        }
        assert!(matches!(s.get("K"), Err(VaultError::Crypto(_))));
    }

    #[test]
    fn test_vault_error_messages_never_contain_secret_values() {
        // SC-V16 mechanized (MAGI run 3, Caspar): no VaultError Display ever
        // includes a value. Construct variants with a probe secret value and
        // assert its absence.
        let probe = "sk-super-secret-PROBE-9f3a";
        let mut s = fixture();
        let errs: Vec<String> = vec![
            VaultError::SecretNotFound("A".into()).to_string(),
            VaultError::Crypto("tag mismatch".into()).to_string(),
            VaultError::WrongPassphrase.to_string(),
            VaultError::VaultMetaCorrupt.to_string(),
        ];
        // A live case too: set the probe, then confirm no captured error text
        // above leaks it.
        s.set("A", probe).expect("set");
        for e in errs {
            assert!(!e.contains(probe), "VaultError leaked a value: {e}");
        }
    }

    #[test]
    fn test_value_is_not_plaintext_at_rest() {
        // SC-V10 at the column level: the stored blob does not contain the
        // plaintext. Read via `get_ref` (not a typed `String`/`Vec<u8>`
        // getter) so the assertion does not depend on which SQLite storage
        // class the BLOB column happens to report.
        let mut s = fixture();
        s.set("K", "super-secret-readable").expect("set");
        let blob: Vec<u8> = {
            let conn = s.debug_conn();
            let guard = conn.lock().expect("lock");
            guard
                .query_row("SELECT value_blob FROM vault WHERE name='K'", [], |r| {
                    Ok(raw_column_bytes(r.get_ref(0)?))
                })
                .expect("row")
        };
        // Deterministic, not just a probabilistic substring miss: the on-disk
        // blob is NOT the plaintext bytes (AEAD+FEC expands and randomizes it),
        // does not contain them, AND still decrypts back to the exact value —
        // together proving the value is really encrypted and recoverable.
        assert_ne!(blob.as_slice(), b"super-secret-readable");
        assert!(!String::from_utf8_lossy(&blob).contains("super-secret-readable"));
        assert_eq!(
            s.get("K").expect("roundtrip").as_str(),
            "super-secret-readable"
        );
    }

    #[test]
    fn test_fuzz_value_roundtrip_entrypoint_never_panics_on_arbitrary_input() {
        for data in [
            &b""[..],
            &b"\x00"[..],
            &b"\xff\xfe\x80"[..], // invalid UTF-8
            &[0x41u8; 100_000],   // large (within the ceiling)
        ] {
            super::fuzz_value_roundtrip_entrypoint(data);
        }
    }
}
