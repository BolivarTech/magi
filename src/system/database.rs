//! This module provides a persistent memory system based on SQLite with encryption.

use crate::agent::messages::Message;
use crate::utils::crypto::{CryptoVault, ErrorCorrection, ReedSolomonCodec, SALT_LEN};
use anyhow::Result;
use async_trait::async_trait;
use rand::RngCore;
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
    /// Data key derived **once** from the per-DB salt + master password (B′).
    ///
    /// `Zeroizing<Vec<u8>>` overwrites the key on drop. Derived in
    /// [`Self::new_with_vault`]; reused by every record so no per-record Argon2
    /// runs on the hot path.
    derived_key: Zeroizing<Vec<u8>>,
}

impl EncryptedSqliteMemory {
    /// Collects raw `(role, blob)` rows for a session under the connection lock.
    ///
    /// The lock is held only for the duration of the SELECT and the iterator
    /// drain; it is released before any decryption happens (audit finding W12).
    fn collect_message_rows(&self, session_id: &str) -> Result<Vec<(String, String)>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| anyhow::anyhow!("DB lock poisoned: {e}"))?;
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
    /// connection guard before invoking this, so per-row Argon2 key derivation
    /// never serializes other DB callers (audit finding W12).
    fn decrypt_rows(&self, rows: Vec<(String, String)>) -> Result<Vec<Message>> {
        let mut messages = Vec::with_capacity(rows.len());
        for (role_str, blob) in rows {
            let decrypted = self
                .vault
                .decrypt_with_key(&self.derived_key, &blob)
                .map_err(|e| anyhow::anyhow!("Decryption failed: {}", e))?;
            let content = serde_json::from_str(&decrypted)?;
            let role = match role_str.as_str() {
                "User" => crate::agent::messages::Role::User,
                _ => crate::agent::messages::Role::Assistant,
            };
            messages.push(Message { role, content });
        }
        Ok(messages)
    }

    pub fn new(path: PathBuf, master_password: String) -> Result<Self> {
        Self::new_with_vault(path, master_password, CryptoVault::default())
    }

    /// Constructor that accepts a custom [`CryptoVault`] (e.g. a counting KDF in
    /// tests). Derives the data key **once** from the per-DB salt and caches it.
    pub(crate) fn new_with_vault(
        path: PathBuf,
        master_password: String,
        vault: CryptoVault,
    ) -> Result<Self> {
        let mut conn = Connection::open(path)?;

        // MAGI FIX: Enable WAL mode for high concurrency
        // We use query_row because execute fails for pragmas that return values in some drivers
        let _: String = conn.query_row("PRAGMA journal_mode = WAL", [], |row| row.get(0))?;
        conn.execute("PRAGMA synchronous = NORMAL", [])?;
        // Set a busy timeout to prevent "database is locked" errors during contention
        conn.busy_timeout(std::time::Duration::from_secs(5))?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                project_name TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content_blob TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )",
            [],
        )?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS knowledge (
                key TEXT PRIMARY KEY,
                value_blob TEXT NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS vault_meta (
                key TEXT PRIMARY KEY,
                value BLOB NOT NULL
            )",
            [],
        )?;

        // B′: one per-DB salt → one key derivation. The salt is **RS-encoded** on
        // disk (#10) so minor bit-rot is corrected on read, and a salt that is
        // absent OR fails to RS-decode to SALT_LEN bytes is treated as **absent**,
        // so D6 self-heals (reset + re-bootstrap) instead of bricking the DB.
        // NOTE: Reed-Solomon is error *correction*, not a cryptographic integrity
        // check — a valid-codeword corruption (e.g. an all-zero block) can still
        // mis-decode to a wrong-but-valid salt and bypass the self-heal (narrow
        // residual; a salt MAC would fully close it — see docs/FASE0-FOLLOWUPS.md #15).
        // Per D6 the reset wipes content in a single transaction (atomic) and warns
        // only when real (non-empty) history is discarded, so it is observable.
        let salt_codec = ReedSolomonCodec::default();
        let valid_salt: Option<Vec<u8>> = conn
            .query_row("SELECT value FROM vault_meta WHERE key = 'salt'", [], |r| {
                r.get::<_, Vec<u8>>(0)
            })
            .optional()?
            // `decode` already truncates to SALT_LEN on success, so the length
            // check is belt-and-suspenders. An `Err` (corrupt beyond RS recovery,
            // or a non-RS raw value from a pre-#10 DB) maps to `None` to
            // intentionally trigger the D6 self-heal below rather than deriving a
            // wrong key from a bad salt.
            .and_then(|blob| match salt_codec.decode(&blob, SALT_LEN) {
                Ok(s) if s.len() == SALT_LEN => Some(s),
                _ => None,
            });
        let salt: Vec<u8> = match valid_salt {
            Some(s) => s,
            None => {
                let had_rows: i64 = conn.query_row(
                    "SELECT (SELECT COUNT(*) FROM sessions) \
                     + (SELECT COUNT(*) FROM messages) \
                     + (SELECT COUNT(*) FROM knowledge)",
                    [],
                    |r| r.get(0),
                )?;

                let mut new_salt = vec![0u8; SALT_LEN];
                rand::rngs::OsRng.fill_bytes(&mut new_salt);
                let salt_blob = salt_codec.encode(&new_salt);

                let tx = conn.transaction()?;
                tx.execute("DELETE FROM messages", [])?;
                tx.execute("DELETE FROM knowledge", [])?;
                tx.execute("DELETE FROM sessions", [])?;
                // OR REPLACE: a self-heal reset fires when the salt row is absent
                // *or* present-but-invalid (corrupt), so the row may already exist.
                tx.execute(
                    "INSERT OR REPLACE INTO vault_meta (key, value) VALUES ('salt', ?1)",
                    params![salt_blob],
                )?;
                tx.commit()?;

                if had_rows > 0 {
                    eprintln!(
                        "WARNING: existing on-disk history used an incompatible or corrupt \
                         encryption salt and has been reset (fresh start). This is expected \
                         after upgrading the storage format."
                    );
                }
                new_salt
            }
        };

        // Derive the data key exactly once; the incoming password is zeroized
        // after derivation.
        let password = Zeroizing::new(master_password);
        let derived_key = vault
            .derive_key(&password, &salt)
            .map_err(|e| anyhow::anyhow!("Key derivation failed: {}", e))?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            vault,
            derived_key,
        })
    }
}

#[async_trait]
impl MemoryStore for EncryptedSqliteMemory {
    async fn create_session(&self, project_name: &str) -> Result<String> {
        let id = uuid::Uuid::new_v4().to_string();
        let conn = self
            .conn
            .lock()
            .map_err(|e| anyhow::anyhow!("DB lock poisoned: {e}"))?;
        conn.execute(
            "INSERT INTO sessions (id, project_name) VALUES (?1, ?2)",
            params![id, project_name],
        )?;
        Ok(id)
    }

    async fn add_message(&self, session_id: &str, message: &Message) -> Result<()> {
        let json_content = serde_json::to_string(&message.content)?;
        let encrypted = self
            .vault
            .encrypt_with_key(&self.derived_key, &json_content)
            .map_err(|e| anyhow::anyhow!("Encryption failed: {}", e))?;

        let conn = self
            .conn
            .lock()
            .map_err(|e| anyhow::anyhow!("DB lock poisoned: {e}"))?;
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
        let conn = self
            .conn
            .lock()
            .map_err(|e| anyhow::anyhow!("DB lock poisoned: {e}"))?;
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
        let encrypted = self
            .vault
            .encrypt_with_key(&self.derived_key, value)
            .map_err(|e| anyhow::anyhow!("Encryption failed: {}", e))?;

        let conn = self
            .conn
            .lock()
            .map_err(|e| anyhow::anyhow!("DB lock poisoned: {e}"))?;
        conn.execute(
            "INSERT OR REPLACE INTO knowledge (key, value_blob, updated_at) VALUES (?1, ?2, CURRENT_TIMESTAMP)",
            params![key, encrypted],
        )?;
        Ok(())
    }

    async fn get_knowledge(&self, key: &str) -> Result<Option<String>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| anyhow::anyhow!("DB lock poisoned: {e}"))?;
        let mut stmt = conn.prepare("SELECT value_blob FROM knowledge WHERE key = ?")?;

        let res = stmt.query_row(params![key], |row| row.get::<_, String>(0));

        match res {
            Ok(blob) => {
                let decrypted = self
                    .vault
                    .decrypt_with_key(&self.derived_key, &blob)
                    .map_err(|e| anyhow::anyhow!("Decryption failed: {}", e))?;
                Ok(Some(decrypted))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(anyhow::anyhow!("Database error: {}", e)),
        }
    }

    async fn list_knowledge_keys(&self) -> Result<Vec<String>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| anyhow::anyhow!("DB lock poisoned: {e}"))?;
        let mut stmt = conn.prepare("SELECT key FROM knowledge ORDER BY key ASC")?;
        let rows = stmt.query_map([], |row| row.get(0))?;

        let mut keys = Vec::new();
        for row in rows {
            keys.push(row?);
        }
        Ok(keys)
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

    pub(crate) fn derived_key_type_for_test(&self) -> &zeroize::Zeroizing<Vec<u8>> {
        &self.derived_key
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::crypto::{
        Aes256GcmSivCipher, Argon2Kdf, CryptoError, CryptoVault, KeyDerivation, ReedSolomonCodec,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    /// KDF that counts derivations and delegates to the real Argon2id.
    struct CountingKdf {
        inner: Argon2Kdf,
        calls: Arc<AtomicUsize>,
    }
    impl KeyDerivation for CountingKdf {
        fn derive_key(
            &self,
            password: &[u8],
            salt: &[u8],
            output_len: usize,
        ) -> std::result::Result<zeroize::Zeroizing<Vec<u8>>, CryptoError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.inner.derive_key(password, salt, output_len)
        }
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
            Box::new(ReedSolomonCodec::default()),
        );
        let memory = EncryptedSqliteMemory::new_with_vault(
            tmp.path().to_path_buf(),
            "pw".to_string(),
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
            "Argon2 must run exactly once (B′), not per record"
        );
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

        let memory = EncryptedSqliteMemory::new(path, "pw".to_string()).unwrap();
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
            let memory = EncryptedSqliteMemory::new(path.clone(), "P".to_string()).unwrap();
            sid = memory.create_session("p").await.unwrap();
            memory
                .add_message(&sid, &Message::user("persisted"))
                .await
                .unwrap();
        }
        {
            let memory = EncryptedSqliteMemory::new(path.clone(), "P".to_string()).unwrap();
            assert_eq!(
                memory.get_messages(&sid).await.unwrap(),
                vec![Message::user("persisted")]
            );
        }
        {
            let memory = EncryptedSqliteMemory::new(path, "P-different".to_string()).unwrap();
            let res = memory.get_messages(&sid).await;
            assert!(res.is_err());
            assert!(res.unwrap_err().to_string().contains("Decryption failed"));
        }
    }

    #[tokio::test]
    async fn test_corrupt_salt_triggers_self_heal_reset() {
        // S-2: an invalid/unrecoverable salt (here: a raw, non-RS-encoded value)
        // is treated as absent so D6 self-heals (reset) instead of bricking the DB.
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        {
            let memory = EncryptedSqliteMemory::new(path.clone(), "P".to_string()).unwrap();
            let sid = memory.create_session("p").await.unwrap();
            memory
                .add_message(&sid, &Message::user("orig"))
                .await
                .unwrap();
        }
        // Overwrite the salt with a raw, non-RS-encoded value (irrecoverable).
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute(
                "UPDATE vault_meta SET value = ?1 WHERE key = 'salt'",
                params![vec![0x42u8; SALT_LEN]],
            )
            .unwrap();
        }
        // Reopen: invalid salt -> treated absent -> D6 reset, no error.
        let memory = EncryptedSqliteMemory::new(path, "P".to_string()).unwrap();
        assert!(
            memory.list_sessions().await.unwrap().is_empty(),
            "an invalid/unrecoverable salt must self-heal via a D6 reset"
        );
        let fresh = memory.create_session("fresh").await.unwrap();
        memory
            .add_message(&fresh, &Message::user("new"))
            .await
            .unwrap();
        assert_eq!(memory.get_messages(&fresh).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_various_invalid_salt_shapes_self_heal() {
        // Strengthens S-2: every salt shape that RS-decode REJECTS (empty,
        // raw/non-RS-encoded, truncated below one block) must self-heal via a D6
        // reset, never validate to a wrong salt. A valid-codeword corruption such
        // as an all-zero block is a known RS mis-correction residual (see
        // docs/FASE0-FOLLOWUPS.md #15) and is intentionally NOT asserted here.
        for bad in [Vec::<u8>::new(), vec![0x42u8; SALT_LEN], vec![0x07u8; 30]] {
            let tmp = NamedTempFile::new().unwrap();
            let path = tmp.path().to_path_buf();
            {
                let m = EncryptedSqliteMemory::new(path.clone(), "P".to_string()).unwrap();
                let s = m.create_session("p").await.unwrap();
                m.add_message(&s, &Message::user("x")).await.unwrap();
            }
            {
                let conn = Connection::open(&path).unwrap();
                conn.execute(
                    "UPDATE vault_meta SET value = ?1 WHERE key = 'salt'",
                    params![bad],
                )
                .unwrap();
            }
            let m = EncryptedSqliteMemory::new(path, "P".to_string()).unwrap();
            assert!(
                m.list_sessions().await.unwrap().is_empty(),
                "an RS-rejected invalid salt must self-heal via a D6 reset"
            );
        }
    }

    #[tokio::test]
    async fn test_minor_salt_bitrot_is_corrected_and_history_survives() {
        // S-3: an RS-encoded salt with a few flipped bytes is corrected on read,
        // so the derived key is unchanged and prior history still decrypts.
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let sid;
        {
            let memory = EncryptedSqliteMemory::new(path.clone(), "P".to_string()).unwrap();
            sid = memory.create_session("p").await.unwrap();
            memory
                .add_message(&sid, &Message::user("survives"))
                .await
                .unwrap();
        }
        // Flip a few bytes of the stored salt blob (within RS correction capacity).
        {
            let conn = Connection::open(&path).unwrap();
            let mut blob: Vec<u8> = conn
                .query_row("SELECT value FROM vault_meta WHERE key = 'salt'", [], |r| {
                    r.get(0)
                })
                .unwrap();
            for b in blob.iter_mut().take(3) {
                *b ^= 0xAA;
            }
            conn.execute(
                "UPDATE vault_meta SET value = ?1 WHERE key = 'salt'",
                params![blob],
            )
            .unwrap();
        }
        // Reopen: RS corrects the salt -> same key -> history survives.
        let memory = EncryptedSqliteMemory::new(path, "P".to_string()).unwrap();
        assert_eq!(
            memory.get_messages(&sid).await.unwrap(),
            vec![Message::user("survives")],
            "RS-encoded salt must self-correct minor bit-rot, preserving history"
        );
    }

    #[tokio::test]
    async fn test_encrypted_sqlite_memory() {
        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path().to_path_buf();
        let password = "master_key_123";

        let memory = EncryptedSqliteMemory::new(path, password.to_string()).unwrap();
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

        let memory = EncryptedSqliteMemory::new(path, password).unwrap();

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
        let memory = Arc::new(EncryptedSqliteMemory::new(path, "stress_pass".to_string()).unwrap());

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
    async fn test_poisoned_lock_returns_error_not_panic() {
        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path().to_path_buf();
        let memory = EncryptedSqliteMemory::new(path, "pw".to_string()).unwrap();

        let conn = memory.conn_for_test().clone();
        let _ = std::thread::spawn(move || {
            let _guard = conn.lock().unwrap();
            panic!("intentional poison");
        })
        .join();

        let result = memory.list_sessions().await;
        assert!(result.is_err(), "a poisoned lock must yield Err, not panic");
        assert!(
            result.unwrap_err().to_string().contains("poisoned"),
            "error must identify the poisoned lock"
        );
    }

    #[tokio::test]
    async fn test_get_messages_does_not_hold_lock_during_decrypt() {
        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path().to_path_buf();
        let memory = Arc::new(EncryptedSqliteMemory::new(path, "pw".to_string()).unwrap());
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
    async fn test_decrypt_rows_runs_without_connection_lock() {
        let tmp_file = NamedTempFile::new().unwrap();
        let memory =
            EncryptedSqliteMemory::new(tmp_file.path().to_path_buf(), "pw".to_string()).unwrap();
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

        let memory = EncryptedSqliteMemory::new(path, "zeroizing_pw".to_string()).unwrap();
        let sid = memory.create_session("p").await.unwrap();
        memory
            .add_message(&sid, &Message::user("secret payload"))
            .await
            .unwrap();

        let _assert_type: &zeroize::Zeroizing<Vec<u8>> = memory.derived_key_type_for_test();

        let msgs = memory.get_messages(&sid).await.unwrap();
        assert_eq!(msgs, vec![Message::user("secret payload")]);
    }
}
