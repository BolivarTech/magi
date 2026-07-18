//! This module provides a persistent memory system based on SQLite with encryption.

use crate::agent::messages::Message;
use anyhow::Result;
use async_trait::async_trait;
use cryptovault::CryptoVault;
use magi_rs::vault::{bootstrap_envelope, open_envelope, MaskedDek, VaultError};
use rusqlite::{params, Connection, OptionalExtension};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use zeroize::Zeroizing;

/// Trait defining the behavior of the agent's memory.
#[async_trait]
pub trait MemoryStore: Send + Sync {
    /// Creates a new session and returns its ID.
    async fn create_session(&self, project_name: &str) -> Result<String>;

    /// Adds a message to a specific session.
    async fn add_message(&self, session_id: &str, message: &Message) -> Result<()>;

    /// Retrieves all messages for a session.
    async fn get_messages(&self, session_id: &str) -> Result<Vec<Message>>;

    /// Lists all sessions.
    async fn list_sessions(&self) -> Result<Vec<(String, String)>>; // (id, project_name)

    /// Stores a persistent fact about the project.
    async fn set_knowledge(&self, key: &str, value: &str) -> Result<()>;

    /// Retrieves a persistent fact.
    async fn get_knowledge(&self, key: &str) -> Result<Option<String>>;

    /// Lists all known project keys.
    async fn list_knowledge_keys(&self) -> Result<Vec<String>>;
}

/// A persistent memory store using SQLite and CryptoVault for encryption.
pub struct EncryptedSqliteMemory {
    conn: Arc<Mutex<Connection>>,
    vault: CryptoVault,
    /// Data key derived **once** from the per-DB salt + master password (B′), held
    /// **masked** in RAM ([`MaskedDek`], MS2 REQ-V42): never in the clear at rest,
    /// mask rotated on every access.
    ///
    /// The `Mutex` provides interior mutability for the `&mut self` mask rotation
    /// (the store is shared as `Arc<dyn MemoryStore + Send + Sync>`). **Lock
    /// discipline (R-V08): the DEK lock is NEVER held across the connection lock** —
    /// [`Self::seal`]/[`Self::unseal`] take and release it in a tight scope, always
    /// outside any `self.conn` guard.
    dek: std::sync::Mutex<MaskedDek>,
    /// `true` when construction discarded incompatible/corrupt on-disk content
    /// (the D6 reset fired on a non-empty DB). Surfaced to the user at startup (#11).
    was_reset: bool,
}

impl EncryptedSqliteMemory {
    /// Locks the connection, recovering the guard if the mutex was poisoned by a
    /// panic in another thread (the SQLite handle remains valid). Keeps
    /// persistence available instead of failing closed for the session (#8,
    /// supersedes the W11 error-on-poison behavior); the recovery is logged.
    fn locked_conn(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().unwrap_or_else(|poisoned| {
            use std::sync::atomic::{AtomicBool, Ordering};
            // Warn once per process: a persistently-poisoned mutex would otherwise
            // spam stderr on every op (and disrupt the TUI alternate screen).
            static POISON_WARNED: AtomicBool = AtomicBool::new(false);
            if !POISON_WARNED.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "WARNING: database connection mutex was poisoned by a panic in another \
                     thread; recovering the connection and continuing (further occurrences \
                     suppressed)."
                );
            }
            poisoned.into_inner()
        })
    }

    /// Collects raw `(role, blob)` rows for a session under the connection lock.
    ///
    /// The lock is held only for the duration of the SELECT and the iterator
    /// drain; it is released before any decryption happens (audit finding W12).
    fn collect_message_rows(&self, session_id: &str) -> Result<Vec<(String, String)>> {
        let conn = self.locked_conn();
        let mut stmt = conn.prepare(
            "SELECT role, content_blob FROM messages WHERE session_id = ? ORDER BY created_at ASC",
        )?;
        let mapped = stmt.query_map(params![session_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        let mut collected = Vec::new();
        for row in mapped {
            collected.push(row?);
        }
        Ok(collected)
    }

    /// Decrypts pre-collected `(role, blob)` rows into [`Message`]s.
    ///
    /// Holds **no** database lock: callers must collect rows and release the
    /// connection guard before invoking this, so per-row decryption (FEC/Viterbi
    /// decode under the cached data key) never serializes other DB callers
    /// (audit finding W12).
    fn decrypt_rows(&self, rows: Vec<(String, String)>) -> Result<Vec<Message>> {
        let mut messages = Vec::with_capacity(rows.len());
        for (role_str, blob) in rows {
            let decrypted = self.unseal(&blob)?;
            let content = serde_json::from_str(decrypted.as_str())?;
            let role = match role_str.as_str() {
                "User" => crate::agent::messages::Role::User,
                _ => crate::agent::messages::Role::Assistant,
            };
            messages.push(Message { role, content });
        }
        Ok(messages)
    }

    pub fn new(path: PathBuf, master_password: Zeroizing<String>) -> Result<Self> {
        Self::new_with_vault(path, master_password, CryptoVault::default())
    }

    /// Whether the on-disk DB had real content discarded (reset to fresh) during
    /// construction (#11). The caller surfaces this to the user at startup.
    pub fn was_reset(&self) -> bool {
        self.was_reset
    }

    /// Constructor that accepts a custom [`CryptoVault`] (e.g. a counting KDF in
    /// tests). Derives the data key **once** from the per-DB salt and caches it.
    pub(crate) fn new_with_vault(
        path: PathBuf,
        master_password: Zeroizing<String>,
        vault: CryptoVault,
    ) -> Result<Self> {
        let mut conn = Connection::open(path)?;

        // Set the busy timeout **FIRST**, before any pragma or table creation:
        // two openers racing on a brand-new DB contend for the WAL-header /
        // table-creation write lock, and without the timeout the loser fails
        // immediately with SQLITE_BUSY ("database is locked") instead of waiting
        // for the winner to finish bootstrapping. (Regression caught by the
        // concurrent-bootstrap test, REQ-V35 / SC-V51.)
        conn.busy_timeout(std::time::Duration::from_secs(5))?;

        // MAGI FIX: Enable WAL mode for high concurrency
        // We use query_row because execute fails for pragmas that return values in some drivers
        let _: String = conn.query_row("PRAGMA journal_mode = WAL", [], |row| row.get(0))?;
        conn.execute("PRAGMA synchronous = NORMAL", [])?;

        // Create every table of the schema (single source of truth — `init_schema`).
        init_schema(&conn)?;

        // Envelope open path (REQ-V29/V35): `vault_meta` holds `{salt, wrapped_dek}`,
        // each FEC-encoded. The master passphrase (UTF-8, from `-p`/env/prompt)
        // derives the KEK that unwraps the DEK; the DEK — held masked in `dek` —
        // encrypts every record.
        //
        // NEVER auto-delete on a crypto failure (REQ-V35): a wrong master or a
        // corrupt wrapped_dek fail the AEAD tag identically, so wiping here would
        // turn a typo into total data loss. The ONLY discard is the one-time format
        // migration below, gated on the `wrapped_dek` row being **absent** — a
        // deterministic schema check a wrong password can never trigger.
        // Already `Zeroizing<String>` (MS2: the passphrase never exists as a bare
        // `String`, closing the transient-copy window — REQ-V41).
        let password = master_password;
        let mut was_reset = false;

        let wrapped_dek: Option<Vec<u8>> = conn
            .query_row(
                "SELECT value FROM vault_meta WHERE key = 'wrapped_dek'",
                [],
                |r| r.get(0),
            )
            .optional()?;

        let derived_key = match wrapped_dek {
            // Existing envelope: open it. A failure propagates and NEVER deletes.
            Some(wrapped_fec) => {
                let salt_fec: Vec<u8> = conn
                    .query_row("SELECT value FROM vault_meta WHERE key = 'salt'", [], |r| {
                        r.get(0)
                    })
                    .optional()?
                    .ok_or_else(|| anyhow::anyhow!("vault_meta has wrapped_dek but no salt"))?;
                open_envelope(&vault, &password, &salt_fec, &wrapped_fec).map_err(map_open_err)?
            }
            // No envelope: brand-new DB, or a pre-envelope (old-format) DB whose
            // records are unreadable in the new format. Bootstrap under a write lock;
            // a racing opener ADOPTS the winner's envelope (no double DEK).
            None => {
                // Precompute a fresh envelope BEFORE taking the write lock, so the
                // expensive Argon2 KEK derivation never runs while the lock is held
                // (Caspar/concurrency: holding the write lock across Argon2 would
                // serialize a racing opener on the derivation). `bootstrap_envelope`
                // is pure — it generates salt+DEK and wraps, with no DB side effects,
                // so the work is simply discarded if a racing opener wins below.
                let (salt_mine, wrapped_mine, dek_mine) =
                    bootstrap_envelope(&vault, &password).map_err(map_open_err)?;

                // Under the write lock, do ONLY cheap SQL: re-check for a racing
                // bootstrap and either install our precomputed envelope or capture
                // the winner's for an out-of-lock unwrap.
                let tx =
                    conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                let raced: Option<Vec<u8>> = tx
                    .query_row(
                        "SELECT value FROM vault_meta WHERE key = 'wrapped_dek'",
                        [],
                        |r| r.get(0),
                    )
                    .optional()?;
                let adopted: Option<(Vec<u8>, Vec<u8>)> = match raced {
                    // A racing opener bootstrapped first between our read and the
                    // write lock: capture its envelope; unwrap it AFTER releasing
                    // the lock so its Argon2 derivation is off the hot lock too.
                    Some(wrapped_fec) => {
                        let salt_fec: Vec<u8> = tx.query_row(
                            "SELECT value FROM vault_meta WHERE key = 'salt'",
                            [],
                            |r| r.get(0),
                        )?;
                        Some((salt_fec, wrapped_fec))
                    }
                    // Fresh-start (REQ-V31): discard any old-format content
                    // (unreadable under the new crypto) and install our envelope.
                    None => {
                        let had_rows: i64 = tx.query_row(
                            "SELECT (SELECT COUNT(*) FROM sessions) \
                             + (SELECT COUNT(*) FROM messages) \
                             + (SELECT COUNT(*) FROM knowledge)",
                            [],
                            |r| r.get(0),
                        )?;
                        tx.execute("DELETE FROM messages", [])?;
                        tx.execute("DELETE FROM knowledge", [])?;
                        tx.execute("DELETE FROM sessions", [])?;
                        tx.execute(
                            "INSERT OR REPLACE INTO vault_meta (key, value) VALUES ('salt', ?1)",
                            params![salt_mine],
                        )?;
                        tx.execute(
                            "INSERT OR REPLACE INTO vault_meta (key, value) VALUES ('wrapped_dek', ?1)",
                            params![wrapped_mine],
                        )?;
                        was_reset = had_rows > 0;
                        None
                    }
                };
                tx.commit()?;
                if was_reset {
                    eprintln!(
                        "WARNING: existing on-disk history used an incompatible (pre-envelope) \
                         encryption format and has been reset (fresh start). This is expected \
                         after upgrading the storage format."
                    );
                }
                match adopted {
                    // A racing opener won: unwrap ITS envelope off the lock. A failure
                    // here propagates and NEVER deletes (REQ-V35).
                    Some((salt_fec, wrapped_fec)) => {
                        open_envelope(&vault, &password, &salt_fec, &wrapped_fec)
                            .map_err(map_open_err)?
                    }
                    // We installed our precomputed envelope.
                    None => dek_mine,
                }
            }
        };

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            vault,
            dek: std::sync::Mutex::new(MaskedDek::new(derived_key)?),
            was_reset,
        })
    }

    /// Encrypts `plaintext` with the masked DEK. **Never call while holding the
    /// `self.conn` lock** — takes the DEK lock internally (R-V08).
    fn seal(&self, plaintext: &str) -> Result<String> {
        let mut dek = self.dek.lock().unwrap_or_else(|p| p.into_inner());
        dek.with_dek(|k| self.vault.encrypt_with_key(k, plaintext))
            .map_err(|e| anyhow::anyhow!("Encryption failed: {e}"))
    }

    /// Decrypts `blob` with the masked DEK. Same lock contract as [`Self::seal`].
    fn unseal(&self, blob: &str) -> Result<Zeroizing<String>> {
        let mut dek = self.dek.lock().unwrap_or_else(|p| p.into_inner());
        dek.with_dek(|k| self.vault.decrypt_with_key(k, blob))
            .map_err(|e| anyhow::anyhow!("Decryption failed: {e}"))
    }
}

/// Maps a [`VaultError`] from the envelope open/bootstrap path into an
/// application-level [`anyhow::Error`], preserving the user-facing `Display`
/// message (`WrongPassphrase` ⇒ "incorrect passphrase"). **Never wipes data.**
///
/// Uses `.into()` (not `anyhow!("{e}")`) so the original [`VaultError`]
/// remains recoverable via [`anyhow::Error::downcast_ref`] — MS2's CLI
/// (`main.rs`) matches on the concrete variant to pick an exit code and to
/// drive the TUI's passphrase-retry loop (SC-V09), which a plain formatted
/// string would make impossible.
fn map_open_err(e: VaultError) -> anyhow::Error {
    e.into()
}

/// Creates every table of the magi-rs on-disk schema, idempotently.
///
/// Single source of truth for the schema shared by three call sites: the
/// encrypted store bootstrap ([`EncryptedSqliteMemory::new_with_vault`]), the
/// vector store ([`crate::memory::store::SqliteVectorStore::new`]), and the
/// headless `magi init` scaffold ([`crate::system::workspace::init`]). The five
/// tables — `sessions`, `messages`, `knowledge`, `vault_meta`, `memories` — are
/// exactly the set the bootstrap state machine (MS1 Task 3) row-counts; keeping
/// them in one place guarantees a freshly-`init`ed DB never self-reports
/// `DbCorrupt` for a missing table. All statements use `IF NOT EXISTS`, so
/// re-running against an initialized DB is a no-op. Does **not** set any
/// `PRAGMA` (WAL/synchronous) — those stay at the connection-open call sites.
///
/// # Errors
/// Returns the underlying [`rusqlite::Error`] if any `CREATE` statement fails.
pub(crate) fn init_schema(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            project_name TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content_blob TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        );
        CREATE TABLE IF NOT EXISTS knowledge (
            key TEXT PRIMARY KEY,
            value_blob TEXT NOT NULL,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS vault_meta (
            key TEXT PRIMARY KEY,
            value BLOB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            text_blob TEXT NOT NULL,
            embedding_blob TEXT NOT NULL,
            model_id TEXT NOT NULL,
            dim INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            salience REAL NOT NULL,
            access_count INTEGER NOT NULL DEFAULT 0,
            last_accessed_at INTEGER NOT NULL,
            superseded_by TEXT,
            evicted_at INTEGER,
            scope TEXT NOT NULL DEFAULT 'root',
            distilled_at INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope);",
    )
}

#[async_trait]
impl MemoryStore for EncryptedSqliteMemory {
    async fn create_session(&self, project_name: &str) -> Result<String> {
        let id = uuid::Uuid::new_v4().to_string();
        let conn = self.locked_conn();
        conn.execute(
            "INSERT INTO sessions (id, project_name) VALUES (?1, ?2)",
            params![id, project_name],
        )?;
        Ok(id)
    }

    async fn add_message(&self, session_id: &str, message: &Message) -> Result<()> {
        let json_content = serde_json::to_string(&message.content)?;
        let encrypted = self.seal(&json_content)?;

        let conn = self.locked_conn();
        conn.execute(
            "INSERT INTO messages (session_id, role, content_blob) VALUES (?1, ?2, ?3)",
            params![session_id, format!("{:?}", message.role), encrypted],
        )?;
        Ok(())
    }

    async fn get_messages(&self, session_id: &str) -> Result<Vec<Message>> {
        let raw_rows = self.collect_message_rows(session_id)?;
        self.decrypt_rows(raw_rows)
    }

    async fn list_sessions(&self) -> Result<Vec<(String, String)>> {
        let conn = self.locked_conn();
        let mut stmt =
            conn.prepare("SELECT id, project_name FROM sessions ORDER BY created_at DESC")?;
        let rows = stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;

        let mut sessions = Vec::new();
        for row in rows {
            sessions.push(row?);
        }
        Ok(sessions)
    }

    async fn set_knowledge(&self, key: &str, value: &str) -> Result<()> {
        let encrypted = self.seal(value)?;

        let conn = self.locked_conn();
        conn.execute(
            "INSERT OR REPLACE INTO knowledge (key, value_blob, updated_at) VALUES (?1, ?2, CURRENT_TIMESTAMP)",
            params![key, encrypted],
        )?;
        Ok(())
    }

    async fn get_knowledge(&self, key: &str) -> Result<Option<String>> {
        // Read the raw blob under the lock, then release it before decrypting:
        // `decrypt_with_key` runs FEC/Viterbi decode (~ms), and holding the
        // connection guard across it would serialize every other DB caller
        // (audit finding W12 — same two-phase split as `get_messages`).
        let blob: Option<String> = {
            let conn = self.locked_conn();
            let mut stmt = conn.prepare("SELECT value_blob FROM knowledge WHERE key = ?")?;
            stmt.query_row(params![key], |row| row.get::<_, String>(0))
                .optional()?
        };

        match blob {
            Some(blob) => {
                let decrypted = self.unseal(&blob)?;
                Ok(Some(decrypted.as_str().to_owned()))
            }
            None => Ok(None),
        }
    }

    async fn list_knowledge_keys(&self) -> Result<Vec<String>> {
        let conn = self.locked_conn();
        let mut stmt = conn.prepare("SELECT key FROM knowledge ORDER BY key ASC")?;
        let rows = stmt.query_map([], |row| row.get(0))?;

        let mut keys = Vec::new();
        for row in rows {
            keys.push(row?);
        }
        Ok(keys)
    }
}

impl EncryptedSqliteMemory {
    /// Returns an `Arc` clone of the shared SQLite connection for use by
    /// sibling stores (e.g. the tiered-memory vector store). The connection
    /// is already in WAL mode and has the busy timeout configured.
    // Narrow allow: called by SqliteVectorStore::new (Task 4) and wired in Task 12.
    #[allow(dead_code)]
    pub(crate) fn shared_conn(&self) -> Arc<Mutex<Connection>> {
        self.conn.clone()
    }

    /// Returns an independently-masked copy of the cached per-DB data key so sibling
    /// stores (e.g. the tiered-memory vector store, the vault) can encrypt / decrypt
    /// with the same AES-256-GCM-SIV key without running an additional Argon2
    /// derivation. `&self`, not `&mut self`: the `Mutex` gives interior mutability
    /// for the rotate-on-duplicate, so `main.rs` can call it on a shared value.
    ///
    /// # Errors
    ///
    /// [`VaultError::Crypto`] if generating the copy's fresh mask fails (a broken OS
    /// entropy source).
    pub(crate) fn data_key(&self) -> std::result::Result<MaskedDek, VaultError> {
        self.dek
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .duplicate()
    }
}

#[cfg(test)]
impl EncryptedSqliteMemory {
    pub(crate) fn conn_for_test(&self) -> &Arc<Mutex<Connection>> {
        &self.conn
    }

    pub(crate) fn collect_message_rows_for_test(
        &self,
        session_id: &str,
    ) -> Result<Vec<(String, String)>> {
        self.collect_message_rows(session_id)
    }

    /// Test-only: an independently-masked copy of the DEK, to assert the key is
    /// held via [`MaskedDek`] (the internal masking is unit-tested in `memguard`).
    pub(crate) fn data_key_for_test(&self) -> MaskedDek {
        self.data_key().expect("data_key")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cryptovault::cipher::Aes256GcmSivCipher;
    use cryptovault::fec::ConcatenatedFec;
    use cryptovault::kdf::{Argon2Kdf, KeyDerivation};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use tempfile::NamedTempFile;

    /// KDF that counts derivations and delegates to the real Argon2id.
    struct CountingKdf {
        inner: Argon2Kdf,
        calls: Arc<AtomicUsize>,
    }
    impl KeyDerivation for CountingKdf {
        fn derive_master(
            &self,
            password: &[u8],
            salt: &[u8],
        ) -> cryptovault::Result<Zeroizing<Vec<u8>>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.inner.derive_master(password, salt)
        }
    }

    /// Deterministic, **fast** KDF (SHA-256 of `password ‖ salt`) for tests that
    /// exercise concurrency, not the KDF itself. It yields the required 32-byte
    /// key and is deterministic per `(password, salt)` — so two openers of the
    /// same envelope derive the identical KEK — while avoiding the OWASP Argon2
    /// cost that would otherwise dominate a critical section under contention.
    struct FastKdf;
    impl KeyDerivation for FastKdf {
        fn derive_master(
            &self,
            password: &[u8],
            salt: &[u8],
        ) -> cryptovault::Result<Zeroizing<Vec<u8>>> {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(password);
            hasher.update(salt);
            Ok(Zeroizing::new(hasher.finalize().to_vec()))
        }
    }

    /// Builds a [`CryptoVault`] with [`FastKdf`] and the production cipher + FEC.
    fn fast_kdf_vault() -> CryptoVault {
        CryptoVault::new(
            Box::new(FastKdf),
            Box::new(Aes256GcmSivCipher),
            Box::new(ConcatenatedFec::default()),
        )
    }

    #[tokio::test]
    async fn test_key_is_derived_exactly_once_for_session_load() {
        // S-6 (load-bearing): construct + N adds + get_messages => 1 Argon2 call.
        let tmp = NamedTempFile::new().unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let vault = CryptoVault::new(
            Box::new(CountingKdf {
                inner: Argon2Kdf,
                calls: calls.clone(),
            }),
            Box::new(Aes256GcmSivCipher),
            Box::new(ConcatenatedFec::default()),
        );
        let memory = EncryptedSqliteMemory::new_with_vault(
            tmp.path().to_path_buf(),
            Zeroizing::new("pw".to_string()),
            vault,
        )
        .unwrap();
        let sid = memory.create_session("p").await.unwrap();
        for i in 0..5 {
            memory
                .add_message(&sid, &Message::user(&format!("m{i}")))
                .await
                .unwrap();
        }
        let msgs = memory.get_messages(&sid).await.unwrap();
        assert_eq!(msgs.len(), 5);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "Argon2 must run exactly once (envelope KEK derivation), not per record"
        );
    }

    #[tokio::test]
    async fn test_was_reset_flag_reflects_content_discard() {
        // S-1 (#11): a legacy DB (rows, no salt) that gets reset reports
        // was_reset() == true; a fresh DB reports false.
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute(
                "CREATE TABLE sessions (id TEXT PRIMARY KEY, project_name TEXT NOT NULL, \
                 created_at DATETIME DEFAULT CURRENT_TIMESTAMP)",
                [],
            )
            .unwrap();
            conn.execute(
                "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, \
                 role TEXT NOT NULL, content_blob TEXT NOT NULL, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)",
                [],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO sessions (id, project_name) VALUES ('old', 'legacy')",
                [],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO messages (session_id, role, content_blob) VALUES ('old', 'User', 'X')",
                [],
            )
            .unwrap();
        }
        let legacy = EncryptedSqliteMemory::new(path, Zeroizing::new("pw".to_string())).unwrap();
        assert!(
            legacy.was_reset(),
            "a legacy DB that discarded content must report was_reset()"
        );

        let tmp2 = NamedTempFile::new().unwrap();
        let fresh =
            EncryptedSqliteMemory::new(tmp2.path().to_path_buf(), Zeroizing::new("pw".to_string()))
                .unwrap();
        assert!(!fresh.was_reset(), "a fresh DB must not report was_reset()");
    }

    #[tokio::test]
    async fn test_legacy_db_without_salt_is_reset_on_open() {
        // S-7: a pre-B′ DB (rows present, no vault_meta salt) is wiped on open.
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute(
                "CREATE TABLE sessions (id TEXT PRIMARY KEY, project_name TEXT NOT NULL, \
                 created_at DATETIME DEFAULT CURRENT_TIMESTAMP)",
                [],
            )
            .unwrap();
            conn.execute(
                "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, \
                 role TEXT NOT NULL, content_blob TEXT NOT NULL, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)",
                [],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO sessions (id, project_name) VALUES ('old', 'legacy')",
                [],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO messages (session_id, role, content_blob) VALUES ('old', 'User', 'OLD_BLOB')",
                [],
            )
            .unwrap();
        }

        let memory = EncryptedSqliteMemory::new(path, Zeroizing::new("pw".to_string())).unwrap();
        assert!(
            memory.list_sessions().await.unwrap().is_empty(),
            "legacy rows must be wiped on open (D6 fresh-start)"
        );
        let sid = memory.create_session("fresh").await.unwrap();
        memory
            .add_message(&sid, &Message::user("new"))
            .await
            .unwrap();
        assert_eq!(memory.get_messages(&sid).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_salt_persists_across_reopen_same_password_roundtrips() {
        // S-8: salt persists => same password round-trips; different password fails.
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let sid;
        {
            let memory =
                EncryptedSqliteMemory::new(path.clone(), Zeroizing::new("P".to_string())).unwrap();
            sid = memory.create_session("p").await.unwrap();
            memory
                .add_message(&sid, &Message::user("persisted"))
                .await
                .unwrap();
        }
        {
            let memory =
                EncryptedSqliteMemory::new(path.clone(), Zeroizing::new("P".to_string())).unwrap();
            assert_eq!(
                memory.get_messages(&sid).await.unwrap(),
                vec![Message::user("persisted")]
            );
        }
        {
            // A different master password now fails to OPEN the envelope
            // (REQ-V35: the KEK-unwrap AEAD tag fails immediately), rather than
            // opening successfully and failing later on a per-record decrypt.
            let res = EncryptedSqliteMemory::new(path, Zeroizing::new("P-different".to_string()));
            assert!(res.is_err());
        }
    }

    #[tokio::test]
    async fn test_minor_salt_bitrot_is_corrected_and_history_survives() {
        // The persisted `salt` row is FEC-encoded by `magi_rs::vault::envelope`
        // as `[u32 LE length prefix][ConcatenatedFec::encode(salt)]` (see
        // `vault::envelope::fec_encode`); the crate's own
        // `test_single_bit_flip_in_salt_is_corrected_by_fec` demonstrates that a
        // single-bit flip in the FEC-protected region (i.e. at/after the 4-byte
        // length prefix) is within `ConcatenatedFec`'s correction capacity, so
        // the salt recovers exactly and prior history still decrypts. A flip
        // *inside* the unprotected 4-byte length prefix is a different,
        // out-of-scope failure mode (a corrupted length is not FEC-covered), so
        // this test targets byte index 4 specifically.
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let sid;
        {
            let memory =
                EncryptedSqliteMemory::new(path.clone(), Zeroizing::new("P".to_string())).unwrap();
            sid = memory.create_session("p").await.unwrap();
            memory
                .add_message(&sid, &Message::user("survives"))
                .await
                .unwrap();
        }
        // Flip a single bit just past the 4-byte length prefix, within the
        // FEC-protected region of the stored salt blob.
        {
            let conn = Connection::open(&path).unwrap();
            let mut blob: Vec<u8> = conn
                .query_row("SELECT value FROM vault_meta WHERE key = 'salt'", [], |r| {
                    r.get(0)
                })
                .unwrap();
            let idx = 4.min(blob.len().saturating_sub(1));
            blob[idx] ^= 0x01;
            conn.execute(
                "UPDATE vault_meta SET value = ?1 WHERE key = 'salt'",
                params![blob],
            )
            .unwrap();
        }
        // Reopen: FEC corrects the salt -> same key -> history survives. Per
        // REQ-V35, this must NEVER silently discard the data even if correction
        // failed (it would surface as a typed Err instead) — so an `unwrap()`
        // here is the correct, honest assertion of the never-wipe contract.
        let memory = EncryptedSqliteMemory::new(path, Zeroizing::new("P".to_string())).unwrap();
        assert_eq!(
            memory.get_messages(&sid).await.unwrap(),
            vec![Message::user("survives")],
            "a single-bit flip within FEC capacity must self-correct, preserving history"
        );
    }

    #[tokio::test]
    async fn test_encrypted_sqlite_memory() {
        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path().to_path_buf();
        let password = "master_key_123";

        let memory =
            EncryptedSqliteMemory::new(path, Zeroizing::new(password.to_string())).unwrap();
        let sid = memory.create_session("test_proj").await.unwrap();

        let msg = Message::user("Hello secure world");
        memory.add_message(&sid, &msg).await.unwrap();

        let msgs = memory.get_messages(&sid).await.unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0], msg);

        // Verify encryption (raw read)
        let conn = Connection::open(tmp_file.path()).unwrap();
        let blob: String = conn
            .query_row("SELECT content_blob FROM messages LIMIT 1", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert!(
            !blob.contains("Hello"),
            "Database should contain encrypted blob, not plaintext"
        );

        // Verify list_sessions (to clear dead code warning)
        let sessions = memory.list_sessions().await.unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].1, "test_proj");
    }

    #[tokio::test]
    async fn test_project_knowledge_persistence() {
        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path().to_path_buf();
        let password = "knowledge_key_123".to_string();

        let memory = EncryptedSqliteMemory::new(path, Zeroizing::new(password)).unwrap();

        memory
            .set_knowledge("architecture", "Clean hex with encrypted SQLite")
            .await
            .unwrap();

        let fact = memory.get_knowledge("architecture").await.unwrap();
        assert_eq!(fact.unwrap(), "Clean hex with encrypted SQLite");

        // Verify multiple keys
        memory.set_knowledge("port", "54545").await.unwrap();
        let keys = memory.list_knowledge_keys().await.unwrap();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"architecture".to_string()));
        assert!(keys.contains(&"port".to_string()));
    }

    #[tokio::test]
    async fn test_sqlite_concurrency_stress() {
        let tmp_file = tempfile::NamedTempFile::new().unwrap();
        let path = tmp_file.path().to_path_buf();
        let memory = Arc::new(
            EncryptedSqliteMemory::new(path, Zeroizing::new("stress_pass".to_string())).unwrap(),
        );

        let mut handles = vec![];
        for i in 0..20 {
            let mem_clone = memory.clone();
            handles.push(tokio::spawn(async move {
                let key = format!("key_{}", i);
                let val = format!("val_{}", i);
                mem_clone.set_knowledge(&key, &val).await
            }));
        }

        for h in handles {
            let res = h.await.unwrap();
            assert!(res.is_ok(), "Concurrent write failed: {:?}", res.err());
        }

        let keys = memory.list_knowledge_keys().await.unwrap();
        assert_eq!(keys.len(), 20);
    }

    #[tokio::test]
    async fn test_poisoned_lock_recovers_and_continues() {
        // A-S1 (#8, supersedes W11): a poisoned mutex is recovered (into_inner) so
        // persistence keeps working instead of failing closed for the session.
        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path().to_path_buf();
        let memory = EncryptedSqliteMemory::new(path, Zeroizing::new("pw".to_string())).unwrap();

        let conn = memory.conn_for_test().clone();
        let _ = std::thread::spawn(move || {
            let _guard = conn.lock().unwrap();
            panic!("intentional poison");
        })
        .join();

        // The lock is now poisoned; operations must recover and succeed.
        assert!(
            memory.list_sessions().await.is_ok(),
            "a poisoned lock must be recovered, not fail closed"
        );
        let sid = memory.create_session("after-poison").await.unwrap();
        assert!(!sid.is_empty());
        assert_eq!(
            memory.list_sessions().await.unwrap().len(),
            1,
            "persistence continues working after lock recovery"
        );
    }

    #[tokio::test]
    async fn test_get_messages_does_not_hold_lock_during_decrypt() {
        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path().to_path_buf();
        let memory =
            Arc::new(EncryptedSqliteMemory::new(path, Zeroizing::new("pw".to_string())).unwrap());
        let sid = memory.create_session("p").await.unwrap();

        for i in 0..4 {
            memory
                .add_message(&sid, &Message::user(&format!("message number {i}")))
                .await
                .unwrap();
        }

        let reader = {
            let m = memory.clone();
            let s = sid.clone();
            tokio::spawn(async move { m.get_messages(&s).await })
        };
        let writer = {
            let m = memory.clone();
            tokio::spawn(async move { m.create_session("concurrent").await })
        };

        let msgs = reader.await.unwrap().unwrap();
        let new_sid = writer.await.unwrap().unwrap();

        assert_eq!(
            msgs.len(),
            4,
            "all messages decrypt correctly after lock-drop refactor"
        );
        assert!(
            !new_sid.is_empty(),
            "a concurrent write completes; lock is not held across decrypt"
        );
        assert_eq!(msgs[0], Message::user("message number 0"));
    }

    #[tokio::test]
    async fn test_get_knowledge_does_not_hold_lock_during_decrypt() {
        // W12 / R-V08: get_knowledge must read the raw blob under the lock and
        // release it BEFORE decrypting, so a concurrent DB writer is not blocked
        // across the FEC/Viterbi decode. Both the read and the concurrent write
        // must complete, and the value must round-trip intact.
        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path().to_path_buf();
        let memory =
            Arc::new(EncryptedSqliteMemory::new(path, Zeroizing::new("pw".to_string())).unwrap());
        memory
            .set_knowledge("api-endpoint", "value-42")
            .await
            .unwrap();

        let reader = {
            let m = memory.clone();
            tokio::spawn(async move { m.get_knowledge("api-endpoint").await })
        };
        let writer = {
            let m = memory.clone();
            tokio::spawn(async move { m.create_session("concurrent").await })
        };

        let value = reader.await.unwrap().unwrap();
        let new_sid = writer.await.unwrap().unwrap();

        assert_eq!(
            value.as_deref(),
            Some("value-42"),
            "the secret decrypts correctly after the lock-drop refactor"
        );
        assert!(
            !new_sid.is_empty(),
            "a concurrent write completes; the lock is not held across decrypt"
        );
    }

    #[tokio::test]
    async fn test_decrypt_rows_runs_without_connection_lock() {
        let tmp_file = NamedTempFile::new().unwrap();
        let memory = EncryptedSqliteMemory::new(
            tmp_file.path().to_path_buf(),
            Zeroizing::new("pw".to_string()),
        )
        .unwrap();
        let sid = memory.create_session("p").await.unwrap();
        memory
            .add_message(&sid, &Message::user("hi"))
            .await
            .unwrap();

        let raw = memory.collect_message_rows_for_test(&sid).unwrap();
        let msgs = memory.decrypt_rows(raw).unwrap();
        assert_eq!(msgs, vec![Message::user("hi")]);
    }

    #[tokio::test]
    async fn test_derived_key_field_is_zeroizing_and_roundtrips() {
        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path().to_path_buf();

        let memory =
            EncryptedSqliteMemory::new(path, Zeroizing::new("zeroizing_pw".to_string())).unwrap();
        let sid = memory.create_session("p").await.unwrap();
        memory
            .add_message(&sid, &Message::user("secret payload"))
            .await
            .unwrap();

        // The DEK is held via MaskedDek (masking unit-tested in `memguard`); here we
        // assert it still yields a 32-byte key and the record round-trips.
        let mut dek = memory.data_key_for_test();
        assert_eq!(dek.with_dek(|k| k.len()), 32);

        let msgs = memory.get_messages(&sid).await.unwrap();
        assert_eq!(msgs, vec![Message::user("secret payload")]);
    }

    #[tokio::test]
    async fn test_wrong_master_key_does_not_wipe_database() {
        // REQ-V35: a wrong master password must fail to OPEN (Err) rather than
        // silently succeed and wipe or corrupt existing data; reopening with
        // the correct master afterwards must still see everything intact.
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let sid;
        {
            let memory = EncryptedSqliteMemory::new(
                path.clone(),
                Zeroizing::new("correcto-master-key-string".to_string()),
            )
            .unwrap();
            sid = memory.create_session("p").await.unwrap();
            memory
                .add_message(&sid, &Message::user("must survive"))
                .await
                .unwrap();
        }
        {
            let res = EncryptedSqliteMemory::new(
                path.clone(),
                Zeroizing::new("wrong-master-key-string".to_string()),
            );
            assert!(
                res.is_err(),
                "a wrong master password must fail to open, not silently succeed"
            );
        }
        {
            let memory = EncryptedSqliteMemory::new(
                path,
                Zeroizing::new("correcto-master-key-string".to_string()),
            )
            .unwrap();
            assert_eq!(
                memory.get_messages(&sid).await.unwrap(),
                vec![Message::user("must survive")],
                "the failed wrong-master open attempt must not have wiped or \
                 corrupted the data"
            );
        }
    }

    #[test]
    fn test_concurrent_bootstrap_on_fresh_db_yields_single_dek() {
        // Two openers race to bootstrap the envelope on the same brand-new DB.
        // `new_with_vault`'s `None` branch takes an `Immediate` write-lock
        // transaction before bootstrapping: only one thread wins that lock and
        // creates `vault_meta`; the other must re-check under the lock and
        // ADOPT the winner's envelope rather than creating a second,
        // incompatible DEK. If a second DEK were created, one thread's message
        // would be unreadable under the other's key after reopening — this test
        // asserts both are readable under one shared DEK.
        //
        // Each thread opens its own `Connection` (mirroring real concurrent
        // process/thread access) and drives its own single-threaded Tokio
        // runtime, since `EncryptedSqliteMemory` is constructed synchronously
        // but `MemoryStore` methods are async.
        //
        // A [`FastKdf`] vault is injected via `new_with_vault` (the same test
        // API `test_key_is_derived_exactly_once_for_session_load` uses): the
        // envelope bootstrap/adopt logic under test is identical regardless of
        // KDF cost, but the production OWASP Argon2 (~seconds) would run *inside*
        // the bootstrap write transaction, so the winner would hold the write
        // lock longer than the loser's 5 s `busy_timeout` and spuriously fail the
        // loser's open with `SQLITE_BUSY`. `FastKdf` keeps the critical section
        // short so the race exercises envelope adoption, not Argon2 latency.
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();

        // Pre-create the DB file in WAL mode WITH the full schema so the two
        // racing opens exercise the ENVELOPE bootstrap race (the invariant under
        // test) and not the unrelated DB-file-setup contention that precedes it.
        // Rationale: in production `new_with_vault` runs `PRAGMA journal_mode =
        // WAL` and the four `CREATE TABLE` statements *before* the Immediate
        // bootstrap transaction; under a hard barrier those setup writes on a
        // brand-new file contend and can trip a transient `SQLITE_BUSY`
        // ("database is locked") that is orthogonal to the envelope logic.
        // Seeding WAL (persisted in the DB header) + the tables here makes those
        // steps no-ops on both racing opens, leaving `vault_meta` empty so both
        // still find no `wrapped_dek` and still race the bootstrap — now the only
        // contended step, and one covered by the 5 s `busy_timeout`.
        {
            let seed = Connection::open(&path).unwrap();
            let _: String = seed
                .query_row("PRAGMA journal_mode = WAL", [], |r| r.get(0))
                .unwrap();
            seed.execute_batch(
                "CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    project_name TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content_blob TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );
                CREATE TABLE IF NOT EXISTS knowledge (
                    key TEXT PRIMARY KEY,
                    value_blob TEXT NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS vault_meta (
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL
                );",
            )
            .unwrap();
        }

        let barrier = Arc::new(Barrier::new(2));

        let spawn_opener = |label: &'static str| {
            let path = path.clone();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async {
                    let memory = EncryptedSqliteMemory::new_with_vault(
                        path,
                        Zeroizing::new("shared-master".to_string()),
                        fast_kdf_vault(),
                    )
                    .expect("open must not fail under a concurrent bootstrap race");
                    let sid = memory.create_session(label).await.unwrap();
                    memory
                        .add_message(&sid, &Message::user(&format!("from {label}")))
                        .await
                        .unwrap();
                });
            })
        };

        let t1 = spawn_opener("thread-a");
        let t2 = spawn_opener("thread-b");
        t1.join().expect("thread-a must not panic");
        t2.join().expect("thread-b must not panic");

        // Reopen once more (same FastKdf vault, so the same KEK unwraps the
        // stored DEK) and verify BOTH sessions' messages decrypt under one
        // shared DEK; a divergent DEK would surface here as a decrypt failure
        // for whichever session was written under the race loser's key.
        let memory = EncryptedSqliteMemory::new_with_vault(
            path,
            Zeroizing::new("shared-master".to_string()),
            fast_kdf_vault(),
        )
        .unwrap();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let sessions = rt.block_on(memory.list_sessions()).unwrap();
        assert_eq!(sessions.len(), 2, "both concurrent sessions were persisted");

        let mut total_messages = 0;
        for (sid, _project_name) in sessions {
            let msgs = rt.block_on(memory.get_messages(&sid)).unwrap();
            assert_eq!(
                msgs.len(),
                1,
                "each session's message must decrypt under the shared DEK"
            );
            total_messages += msgs.len();
        }
        assert_eq!(total_messages, 2);
    }
}
