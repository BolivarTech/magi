//! This module provides a persistent memory system based on SQLite with encryption.

use crate::agent::messages::Message;
use anyhow::Result;
use async_trait::async_trait;
use cryptovault::CryptoVault;
use magi_rs::vault::{bootstrap_envelope, open_envelope, MaskedDek, VaultError};
use rusqlite::{params, Connection, OptionalExtension};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use zeroize::Zeroizing;

/// SQLite busy-timeout (seconds) applied at connection open. Two openers racing
/// on a brand-new DB contend for the WAL-header / table-creation write lock;
/// without this timeout the loser fails immediately with `SQLITE_BUSY` instead
/// of waiting for the winner to finish bootstrapping (REQ-V35 / SC-V51).
const BUSY_TIMEOUT_SECS: u64 = 5;

/// Data tables that must **all** exist and be **empty** for a DB that has no
/// envelope to be a legitimate bootstrap candidate (§2.1 / REQ-H20 / D-H10). A
/// missing one is a partial/foreign schema ([`VaultError::DbCorrupt`]); any
/// populated one is data with no key to read it ([`VaultError::DbCorrupt`]).
/// Iterated in this deterministic order so the reported corruption is stable.
const DATA_TABLES: [&str; 4] = ["sessions", "messages", "knowledge", "memories"];

/// `detail` for the §2.1 "no envelope, yet records present" corruption: the
/// encrypted rows cannot be read without the DEK and are **never** discarded.
const DETAIL_DATA_WITHOUT_ENVELOPE: &str = "data present without envelope";

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

    /// Constructor that accepts a custom [`CryptoVault`] (e.g. a counting KDF in
    /// tests). Opens a **raw path**, so it **creates the schema** ([`init_schema`])
    /// before applying the §2.1 state machine — the TUI/`vault`-CLI path that may
    /// point at a not-yet-initialized file. Derives the data key **once** from the
    /// per-DB salt and caches it (masked).
    ///
    /// Because [`init_schema`] guarantees every table exists, the "missing table"
    /// corruption arm of the state machine is unreachable from here; a
    /// data-without-envelope DB still surfaces as [`VaultError::DbCorrupt`] and is
    /// **never** wiped (never-delete absolute, REQ-H20 / D-H10). Use
    /// [`Self::open_with_state_machine`] to open an already-initialized `.magi/` DB
    /// without re-creating any schema.
    pub(crate) fn new_with_vault(
        path: PathBuf,
        master_password: Zeroizing<String>,
        vault: CryptoVault,
    ) -> Result<Self> {
        let mut conn = open_connection(&path).map_err(map_open_err)?;

        // Create every table of the schema (single source of truth — `init_schema`).
        init_schema(&conn)?;

        // Already `Zeroizing<String>` (MS2: the passphrase never exists as a bare
        // `String`, closing the transient-copy window — REQ-V41).
        let derived_key = open_or_bootstrap(&mut conn, &vault, master_password.as_str(), &path)
            .map_err(map_open_err)?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            vault,
            dek: std::sync::Mutex::new(MaskedDek::new(derived_key)?),
        })
    }

    /// Opens an **already-initialized** `.magi/` DB via the §2.1 never-delete
    /// bootstrap state machine, **without creating any schema**. This is the
    /// headless open path: `magi init` (Task 1/2) already created the schema, so a
    /// missing table here is corruption, never silently re-created.
    ///
    /// See [`open_or_bootstrap`] for the exact §2.1 evaluation order and the
    /// never-delete guarantee (REQ-H20 / D-H10 / SC-H21).
    ///
    /// # Errors
    ///
    /// - [`VaultError::DbCorrupt`] — a missing data table (partial/foreign schema)
    ///   or records present with no envelope. **Never wipes.**
    /// - [`VaultError::VaultMetaCorrupt`] — `vault_meta` present but FEC-uncorrectable.
    /// - [`VaultError::WrongPassphrase`] — envelope present, AEAD tag fails. Retryable.
    /// - [`VaultError::Crypto`] / [`VaultError::Storage`] — a crypto or SQL failure.
    pub(crate) fn open_with_state_machine(
        path: PathBuf,
        master_password: Zeroizing<String>,
    ) -> std::result::Result<Self, VaultError> {
        Self::open_with_state_machine_vault(path, master_password, CryptoVault::default())
    }

    /// [`Self::open_with_state_machine`] with an injectable [`CryptoVault`] (a fast
    /// deterministic KDF in tests). Same never-delete semantics and errors.
    ///
    /// # Errors
    ///
    /// Identical to [`Self::open_with_state_machine`].
    pub(crate) fn open_with_state_machine_vault(
        path: PathBuf,
        master_password: Zeroizing<String>,
        vault: CryptoVault,
    ) -> std::result::Result<Self, VaultError> {
        let mut conn = open_connection(&path)?;
        // NO init_schema: opening, not initializing. A missing table is corruption.
        let derived_key = open_or_bootstrap(&mut conn, &vault, master_password.as_str(), &path)?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            vault,
            dek: std::sync::Mutex::new(MaskedDek::new(derived_key)?),
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

/// Opens a SQLite connection with the standard pragmas (busy timeout, WAL,
/// `synchronous = NORMAL`). Shared by both entry points so the pragma order is
/// identical regardless of whether the schema is created afterwards.
///
/// The busy timeout is set **first**, before any pragma, so two openers racing
/// on a brand-new file wait for one another instead of failing `SQLITE_BUSY`.
///
/// # Errors
///
/// [`VaultError::Storage`] if the file cannot be opened or a pragma fails.
fn open_connection(path: &Path) -> std::result::Result<Connection, VaultError> {
    let conn = Connection::open(path).map_err(|e| VaultError::Storage(e.to_string()))?;
    conn.busy_timeout(std::time::Duration::from_secs(BUSY_TIMEOUT_SECS))
        .map_err(|e| VaultError::Storage(e.to_string()))?;
    // `query_row` (not `execute`): `journal_mode` returns the new mode, which
    // `execute` rejects on some driver builds.
    let _: String = conn
        .query_row("PRAGMA journal_mode = WAL", [], |row| row.get(0))
        .map_err(|e| VaultError::Storage(e.to_string()))?;
    conn.execute("PRAGMA synchronous = NORMAL", [])
        .map_err(|e| VaultError::Storage(e.to_string()))?;
    Ok(conn)
}

/// Maps a `rusqlite` error from a table read into a [`VaultError`], turning
/// SQLite's "no such table" into [`VaultError::DbCorrupt`] (a partial/foreign
/// schema is corruption, **never** a bootstrap candidate — §2.1). Any other
/// failure is [`VaultError::Storage`].
fn map_table_err(e: rusqlite::Error, table: &str, db_path: &Path) -> VaultError {
    // SQLite reports a missing table as "no such table: <name>". Matching the
    // message keeps this robust across rusqlite's error-struct shapes.
    if e.to_string().contains("no such table") {
        VaultError::DbCorrupt {
            db_path: db_path.to_path_buf(),
            detail: format!("missing table `{table}`"),
        }
    } else {
        VaultError::Storage(e.to_string())
    }
}

/// Counts the rows of `table`, mapping a missing table to
/// [`VaultError::DbCorrupt`] (§2.1). `table` is always a compile-time constant
/// from [`DATA_TABLES`], never caller input, so the formatted SQL carries no
/// injection risk.
///
/// # Errors
///
/// - [`VaultError::DbCorrupt`] if `table` is absent (`detail` names it).
/// - [`VaultError::Storage`] on any other SQLite failure.
fn count_rows(
    conn: &Connection,
    table: &str,
    db_path: &Path,
) -> std::result::Result<i64, VaultError> {
    conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))
        .map_err(|e| map_table_err(e, table, db_path))
}

/// Reads the FEC-encoded `wrapped_dek` row from `vault_meta` (absent ⇒ `None`).
///
/// # Errors
///
/// [`VaultError::DbCorrupt`] if `vault_meta` itself is missing (schema
/// corruption); [`VaultError::Storage`] on any other SQLite failure.
fn read_wrapped_dek(
    conn: &Connection,
    db_path: &Path,
) -> std::result::Result<Option<Vec<u8>>, VaultError> {
    conn.query_row(
        "SELECT value FROM vault_meta WHERE key = 'wrapped_dek'",
        [],
        |r| r.get(0),
    )
    .optional()
    .map_err(|e| map_table_err(e, "vault_meta", db_path))
}

/// Applies the §2.1 never-delete bootstrap state machine on an already-open
/// connection, returning the recovered or freshly-generated DEK. **Never
/// deletes data** (REQ-H20 / D-H10 / SC-H21).
///
/// Evaluation order:
/// 1. **Envelope row present** ⇒ [`open_envelope`], which FEC-decodes `vault_meta`
///    **before** the AEAD: unwrap OK ⇒ open; FEC-uncorrectable ⇒
///    [`VaultError::VaultMetaCorrupt`]; AEAD tag fails ⇒
///    [`VaultError::WrongPassphrase`] (retryable, never wipes).
/// 2. **No envelope row** ⇒ [`count_rows`] every table in [`DATA_TABLES`] (a
///    missing table ⇒ [`VaultError::DbCorrupt`]): **all empty** ⇒ bootstrap a
///    fresh envelope under a `BEGIN IMMEDIATE` write lock (adopt-winner on a
///    concurrent race); **any data** ⇒ [`VaultError::DbCorrupt`]
///    (`"data present without envelope"`) — the ciphertext is unreadable without
///    the DEK and is **never** discarded.
///
/// **R-V08:** the expensive KEK/Argon2 derivation (inside [`bootstrap_envelope`]
/// / [`open_envelope`]) runs **outside** the connection write lock; the
/// `BEGIN IMMEDIATE` transaction wraps only the cheap `{salt, wrapped_dek}`
/// INSERT.
///
/// # Errors
///
/// See the branches above; also [`VaultError::Storage`] on a SQL failure.
fn open_or_bootstrap(
    conn: &mut Connection,
    vault: &CryptoVault,
    password: &str,
    db_path: &Path,
) -> std::result::Result<Zeroizing<Vec<u8>>, VaultError> {
    match read_wrapped_dek(conn, db_path)? {
        Some(wrapped_fec) => open_existing_envelope(conn, vault, password, &wrapped_fec, db_path),
        None => bootstrap_fresh_envelope(conn, vault, password, db_path),
    }
}

/// Opens an existing envelope (`vault_meta` has a `wrapped_dek`). Delegates to
/// [`open_envelope`], which evaluates FEC **before** the AEAD (§2.1).
///
/// # Errors
///
/// - [`VaultError::VaultMetaCorrupt`] if the `salt` row is missing or the FEC is
///   uncorrectable.
/// - [`VaultError::WrongPassphrase`] if the master is wrong (AEAD tag fails).
/// - [`VaultError::Storage`] on a SQLite failure.
fn open_existing_envelope(
    conn: &Connection,
    vault: &CryptoVault,
    password: &str,
    wrapped_fec: &[u8],
    db_path: &Path,
) -> std::result::Result<Zeroizing<Vec<u8>>, VaultError> {
    let salt_fec: Vec<u8> = conn
        .query_row("SELECT value FROM vault_meta WHERE key = 'salt'", [], |r| {
            r.get(0)
        })
        .optional()
        .map_err(|e| map_table_err(e, "vault_meta", db_path))?
        // A `wrapped_dek` with no `salt` is corrupt metadata, not a wrong master.
        .ok_or(VaultError::VaultMetaCorrupt)?;
    open_envelope(vault, password, &salt_fec, wrapped_fec)
}

/// Bootstraps a fresh envelope for a DB that has no `wrapped_dek` row.
///
/// The **never-delete guard** runs first: every [`DATA_TABLES`] entry must exist
/// (a missing one ⇒ [`VaultError::DbCorrupt`]) and be empty (any data ⇒
/// [`VaultError::DbCorrupt`] `"data present without envelope"`). Only an
/// all-empty schema is bootstrapped; **nothing is ever deleted**.
///
/// The KEK derivation runs **before** the `BEGIN IMMEDIATE` write lock (R-V08);
/// under the lock a racing opener's envelope is **adopted** rather than
/// double-bootstrapped (SC-V51 / §2.2).
///
/// # Errors
///
/// - [`VaultError::DbCorrupt`] on a missing table or data-without-envelope.
/// - [`VaultError::WrongPassphrase`] if a concurrent winner's adopted envelope
///   does not open under this passphrase.
/// - [`VaultError::Crypto`] / [`VaultError::Storage`] on a crypto or SQL failure.
fn bootstrap_fresh_envelope(
    conn: &mut Connection,
    vault: &CryptoVault,
    password: &str,
    db_path: &Path,
) -> std::result::Result<Zeroizing<Vec<u8>>, VaultError> {
    // NEVER-DELETE guard (§2.1): a no-envelope DB is a bootstrap candidate ONLY
    // if every data table exists and is empty. Any present table with data ⇒
    // DbCorrupt; a missing table ⇒ DbCorrupt (via `count_rows`). Neither is EVER
    // wiped or bootstrapped over. Row counting is cheap and needs no write lock.
    let mut total: i64 = 0;
    for table in DATA_TABLES {
        total = total
            .checked_add(count_rows(conn, table, db_path)?)
            .ok_or_else(|| VaultError::Storage("row-count overflow".to_string()))?;
    }
    if total > 0 {
        return Err(VaultError::DbCorrupt {
            db_path: db_path.to_path_buf(),
            detail: DETAIL_DATA_WITHOUT_ENVELOPE.to_string(),
        });
    }

    // Precompute a fresh envelope BEFORE taking the write lock, so the expensive
    // Argon2 KEK derivation never runs while the lock is held (R-V08).
    // `bootstrap_envelope` is pure (no DB side effects), so the work is simply
    // discarded if a racing opener wins the lock below.
    let (salt_mine, wrapped_mine, dek_mine) = bootstrap_envelope(vault, password)?;

    // Under the write lock, do ONLY cheap SQL: re-check for a racing bootstrap
    // and either install our precomputed envelope or capture the winner's for an
    // out-of-lock unwrap.
    let tx = conn
        .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .map_err(|e| VaultError::Storage(e.to_string()))?;
    let raced: Option<Vec<u8>> = tx
        .query_row(
            "SELECT value FROM vault_meta WHERE key = 'wrapped_dek'",
            [],
            |r| r.get(0),
        )
        .optional()
        .map_err(|e| VaultError::Storage(e.to_string()))?;
    let adopted: Option<(Vec<u8>, Vec<u8>)> = match raced {
        // A racing opener bootstrapped first between our read and the write lock:
        // capture its envelope; unwrap it AFTER releasing the lock so its Argon2
        // derivation is off the hot lock too.
        Some(wrapped_fec) => {
            let salt_fec: Vec<u8> = tx
                .query_row("SELECT value FROM vault_meta WHERE key = 'salt'", [], |r| {
                    r.get(0)
                })
                .map_err(|e| VaultError::Storage(e.to_string()))?;
            Some((salt_fec, wrapped_fec))
        }
        // No racing envelope: install ours. `INSERT OR REPLACE` tolerates a stale
        // partial `salt` row from a crashed prior bootstrap (crash-safe). This is
        // the ONLY write on this path — there is NO `DELETE` (never-delete
        // absolute, REQ-H20 / D-H10).
        None => {
            tx.execute(
                "INSERT OR REPLACE INTO vault_meta (key, value) VALUES ('salt', ?1)",
                params![salt_mine],
            )
            .map_err(|e| VaultError::Storage(e.to_string()))?;
            tx.execute(
                "INSERT OR REPLACE INTO vault_meta (key, value) VALUES ('wrapped_dek', ?1)",
                params![wrapped_mine],
            )
            .map_err(|e| VaultError::Storage(e.to_string()))?;
            None
        }
    };
    tx.commit()
        .map_err(|e| VaultError::Storage(e.to_string()))?;

    match adopted {
        // A racing opener won: unwrap ITS envelope off the lock. A failure here
        // propagates and NEVER deletes.
        Some((salt_fec, wrapped_fec)) => open_envelope(vault, password, &salt_fec, &wrapped_fec),
        // We installed our precomputed envelope.
        None => Ok(dek_mine),
    }
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
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use tempfile::NamedTempFile;

    /// Fixed passphrase for the §2.1 state-machine tests.
    fn test_master() -> Zeroizing<String> {
        Zeroizing::new("state-machine-test-master-key".to_string())
    }

    /// Counts rows of `table` on a raw connection (test oracle for the
    /// never-delete "before == after" assertions).
    fn row_count(conn: &Connection, table: &str) -> i64 {
        conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))
            .unwrap()
    }

    /// Seeds a DB that has the full schema and a row in `messages` but **no**
    /// envelope row in `vault_meta` — the §2.1 "data present without envelope"
    /// corruption. Returns the live tempfile (keep it in scope so the path stays
    /// valid), a raw read connection, and the DB path.
    fn seed_db_with_messages_no_envelope() -> (NamedTempFile, Connection, PathBuf) {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let conn = Connection::open(&path).unwrap();
        init_schema(&conn).unwrap();
        conn.execute(
            "INSERT INTO sessions (id, project_name) VALUES ('s', 'p')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO messages (session_id, role, content_blob) VALUES ('s', 'User', 'ciphertext')",
            [],
        )
        .unwrap();
        // `vault_meta` intentionally left EMPTY (no envelope).
        (tmp, conn, path)
    }

    /// KDF that, at derivation time, probes whether a second connection can take
    /// an IMMEDIATE write lock — proving the bootstrap holds **no** write lock
    /// across the (expensive) KEK derivation (R-V08). Delegates to [`FastKdf`].
    struct LockProbeKdf {
        inner: FastKdf,
        db_path: PathBuf,
        lock_free_at_derivation: Arc<AtomicBool>,
    }
    impl KeyDerivation for LockProbeKdf {
        fn derive_master(
            &self,
            password: &[u8],
            salt: &[u8],
        ) -> cryptovault::Result<Zeroizing<Vec<u8>>> {
            // If a refactor moved the derivation inside the BEGIN IMMEDIATE, this
            // probe would block for the busy timeout and fail, flipping the flag.
            let mut probe = Connection::open(&self.db_path).unwrap();
            probe
                .busy_timeout(std::time::Duration::from_millis(200))
                .unwrap();
            let got_lock = probe
                .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
                .is_ok();
            self.lock_free_at_derivation
                .store(got_lock, Ordering::SeqCst);
            self.inner.derive_master(password, salt)
        }
    }

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

    #[test]
    fn test_init_schema_creates_exactly_the_guarded_data_tables() {
        // Drift guard (Fix): DATA_TABLES (the never-delete row-count set) and
        // `init_schema` (the DDL) are coupled by convention only. Adding a table to
        // one but not the other silently weakens never-delete. This test pins the
        // relationship: the *data* tables `init_schema` creates must be EXACTLY
        // DATA_TABLES, plus `vault_meta` — which is the envelope, NOT user data, and
        // is therefore intentionally absent from DATA_TABLES. A future schema/guard
        // drift (add to one, forget the other) fails here.
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn).unwrap();

        let mut stmt = conn
            .prepare("SELECT name FROM sqlite_master WHERE type = 'table'")
            .unwrap();
        // Filter SQLite's internal bookkeeping tables (e.g. `sqlite_sequence`,
        // created by the AUTOINCREMENT column on `messages`).
        let created: std::collections::BTreeSet<String> = stmt
            .query_map([], |r| r.get::<_, String>(0))
            .unwrap()
            .map(|r| r.unwrap())
            .filter(|name| !name.starts_with("sqlite_"))
            .collect();

        let mut expected: std::collections::BTreeSet<String> =
            DATA_TABLES.iter().map(|t| (*t).to_string()).collect();
        // The envelope table is created by `init_schema` but is never a DATA_TABLE.
        expected.insert("vault_meta".to_string());

        assert_eq!(
            created, expected,
            "init_schema must create exactly the DATA_TABLES plus vault_meta; a drift \
             between DATA_TABLES and init_schema is a silent never-delete weakening"
        );

        // Every DATA_TABLES entry is actually created by init_schema (no guard
        // entry without matching DDL).
        for table in DATA_TABLES {
            assert!(
                created.contains(table),
                "DATA_TABLES entry `{table}` must be created by init_schema"
            );
        }
        // `vault_meta` is the envelope and must NEVER be a never-delete DATA_TABLE
        // (row-counting it would misclassify a bootstrapped-but-empty DB).
        assert!(
            !DATA_TABLES.contains(&"vault_meta"),
            "vault_meta is the envelope, not user data — it must not be a DATA_TABLE"
        );
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

    // NOTE (MS1 Task 3, Step 8c): the former `test_was_reset_flag_reflects_content_discard`
    // was REMOVED. Under never-delete ABSOLUTE (REQ-H20 / D-H10) there is no reset
    // — the `was_reset` flag and its startup notice no longer exist. A DB with data
    // but no envelope now yields `DbCorrupt` and is never wiped; that behavior is
    // covered by `test_open_without_envelope_but_with_data_is_dbcorrupt_never_wipes`
    // and `test_legacy_db_without_salt_is_dbcorrupt_and_never_wiped` below.

    #[tokio::test]
    async fn test_legacy_db_without_salt_is_dbcorrupt_and_never_wiped() {
        // Step 8c rewrite of the former `test_legacy_db_without_salt_is_reset_on_open`.
        // A pre-envelope DB (rows present, no `vault_meta` envelope) is now
        // CORRUPTION, not a fresh-start: opening returns `DbCorrupt` and the data
        // is left completely intact (never-delete absolute, REQ-H20 / D-H10 / SC-H21).
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

        // `new` runs `init_schema` (adding the missing tables) then the state
        // machine: no envelope + data present ⇒ DbCorrupt (never wiped).
        let err = EncryptedSqliteMemory::new(path.clone(), Zeroizing::new("pw".to_string()))
            .err()
            .expect("data without an envelope must fail to open, not be wiped");
        assert!(
            matches!(
                err.downcast_ref::<VaultError>(),
                Some(VaultError::DbCorrupt { .. })
            ),
            "expected DbCorrupt, got {err:?}"
        );

        // The legacy rows survive untouched — never-delete absolute.
        let reopened = Connection::open(&path).unwrap();
        assert_eq!(
            row_count(&reopened, "sessions"),
            1,
            "never-delete: the legacy session row must survive the failed open"
        );
        assert_eq!(
            row_count(&reopened, "messages"),
            1,
            "never-delete: the legacy message row must survive the failed open"
        );
    }

    #[test]
    fn test_open_without_envelope_but_with_data_is_dbcorrupt_never_wipes() {
        // Step 1 (the MOST critical): a DB with records but no envelope ⇒
        // DbCorrupt, and the state machine NEVER wipes the data (SC-H21).
        let (_tmp, conn, path) = seed_db_with_messages_no_envelope();
        let before = row_count(&conn, "messages");
        assert!(before > 0, "the seed must actually contain data");

        let err = EncryptedSqliteMemory::open_with_state_machine(path.clone(), test_master())
            .err()
            .expect("data without an envelope must be DbCorrupt");
        assert!(
            matches!(err, VaultError::DbCorrupt { .. }),
            "expected DbCorrupt, got {err:?}"
        );

        // The DB is INTACT — never wiped.
        let reopened = Connection::open(&path).unwrap();
        let after = row_count(&reopened, "messages");
        assert_eq!(
            before, after,
            "never-delete: the state machine must not delete any row"
        );
    }

    #[tokio::test]
    async fn test_open_without_envelope_and_empty_bootstraps_cleanly() {
        // Step 5: a fully-initialized, EMPTY DB (all tables present, no envelope)
        // is the legitimate bootstrap candidate ⇒ the state machine creates the
        // envelope and opens (SC-H20). The envelope then persists across reopen.
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        {
            let conn = Connection::open(&path).unwrap();
            init_schema(&conn).unwrap();
        }

        let store = EncryptedSqliteMemory::open_with_state_machine_vault(
            path.clone(),
            test_master(),
            fast_kdf_vault(),
        )
        .expect("an empty initialized DB must bootstrap cleanly");
        let sid = store.create_session("p").await.unwrap();
        store.add_message(&sid, &Message::user("hi")).await.unwrap();
        assert_eq!(
            store.get_messages(&sid).await.unwrap(),
            vec![Message::user("hi")]
        );
        drop(store);

        // Reopen: the envelope now exists, so the same passphrase opens it and the
        // history is intact.
        let reopened = EncryptedSqliteMemory::open_with_state_machine_vault(
            path,
            test_master(),
            fast_kdf_vault(),
        )
        .expect("the bootstrapped envelope must reopen with the same passphrase");
        assert_eq!(
            reopened.get_messages(&sid).await.unwrap(),
            vec![Message::user("hi")]
        );
    }

    #[tokio::test]
    async fn test_wrong_passphrase_via_state_machine_is_wrong_passphrase_and_intact() {
        // Step 7: a wrong passphrase ⇒ WrongPassphrase (retryable), data intact —
        // the vault never-wipe invariant expressed through the state machine.
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        {
            let conn = Connection::open(&path).unwrap();
            init_schema(&conn).unwrap();
        }
        let right = || Zeroizing::new("right-master-alpha".to_string());
        let wrong = || Zeroizing::new("wrong-master-bravo".to_string());

        let sid;
        {
            let store = EncryptedSqliteMemory::open_with_state_machine_vault(
                path.clone(),
                right(),
                fast_kdf_vault(),
            )
            .unwrap();
            sid = store.create_session("p").await.unwrap();
            store
                .add_message(&sid, &Message::user("must survive"))
                .await
                .unwrap();
        }

        let err = EncryptedSqliteMemory::open_with_state_machine_vault(
            path.clone(),
            wrong(),
            fast_kdf_vault(),
        )
        .err()
        .expect("a wrong passphrase must fail to open");
        assert!(
            matches!(err, VaultError::WrongPassphrase),
            "expected WrongPassphrase, got {err:?}"
        );

        let store =
            EncryptedSqliteMemory::open_with_state_machine_vault(path, right(), fast_kdf_vault())
                .expect("the correct passphrase must still open the untouched DB");
        assert_eq!(
            store.get_messages(&sid).await.unwrap(),
            vec![Message::user("must survive")],
            "the failed wrong-passphrase open must not have wiped the data"
        );
    }

    #[test]
    fn test_fec_damaged_vault_meta_is_vault_meta_corrupt_before_aead() {
        // Step 8: an envelope present but FEC-uncorrectable ⇒ VaultMetaCorrupt,
        // evaluated BEFORE the AEAD (a mass bit-flip fails the FEC decode, so no
        // derivation/AEAD runs at all).
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        // Bootstrap a valid envelope on an empty initialized DB.
        {
            let conn = Connection::open(&path).unwrap();
            init_schema(&conn).unwrap();
        }
        EncryptedSqliteMemory::open_with_state_machine_vault(
            path.clone(),
            test_master(),
            fast_kdf_vault(),
        )
        .unwrap();

        // Corrupt the wrapped_dek FEC beyond correction (mass bit-flip).
        {
            let conn = Connection::open(&path).unwrap();
            let mut blob: Vec<u8> = conn
                .query_row(
                    "SELECT value FROM vault_meta WHERE key = 'wrapped_dek'",
                    [],
                    |r| r.get(0),
                )
                .unwrap();
            for b in blob.iter_mut() {
                *b ^= 0xFF;
            }
            conn.execute(
                "UPDATE vault_meta SET value = ?1 WHERE key = 'wrapped_dek'",
                params![blob],
            )
            .unwrap();
        }

        let err = EncryptedSqliteMemory::open_with_state_machine_vault(
            path,
            test_master(),
            fast_kdf_vault(),
        )
        .err()
        .expect("FEC-uncorrectable vault_meta must fail");
        assert!(
            matches!(err, VaultError::VaultMetaCorrupt),
            "FEC damage must be VaultMetaCorrupt (before the AEAD), got {err:?}"
        );
    }

    #[test]
    fn test_concurrent_bootstrap_different_passphrase_loser_gets_wrong_passphrase() {
        // Step 8b (adopt-winner, §2.2 / SC-V51): two fresh opens race to bootstrap
        // the SAME empty DB with DIFFERENT passphrases. Only one persists the
        // {salt, wrapped_dek}; the other ADOPTS it and, because its passphrase
        // differs, fails the AEAD tag ⇒ WrongPassphrase — never a second DEK,
        // never wiped, retryable.
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        // Pre-seed WAL + the full schema so the race exercises the ENVELOPE
        // bootstrap (the invariant under test), not DB-file-setup contention.
        {
            let seed = Connection::open(&path).unwrap();
            let _: String = seed
                .query_row("PRAGMA journal_mode = WAL", [], |r| r.get(0))
                .unwrap();
            init_schema(&seed).unwrap();
        }

        let barrier = Arc::new(Barrier::new(2));
        // The thread reduces its open to a `Send` outcome (`Ok(())` opened /
        // `Err(kind)` classified) so the store never crosses the thread boundary.
        let spawn_opener = |pass: &'static str| {
            let path = path.clone();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || -> std::result::Result<(), &'static str> {
                barrier.wait();
                match EncryptedSqliteMemory::new_with_vault(
                    path,
                    Zeroizing::new(pass.to_string()),
                    fast_kdf_vault(),
                ) {
                    Ok(_) => Ok(()),
                    Err(e) => match e.downcast_ref::<VaultError>() {
                        Some(VaultError::WrongPassphrase) => Err("WrongPassphrase"),
                        _ => Err("other"),
                    },
                }
            })
        };

        let t1 = spawn_opener("passphrase-alpha-1234567");
        let t2 = spawn_opener("passphrase-bravo-7654321");
        let r1 = t1.join().expect("thread-a must not panic");
        let r2 = t2.join().expect("thread-b must not panic");

        let oks = [&r1, &r2].iter().filter(|r| r.is_ok()).count();
        assert_eq!(
            oks, 1,
            "exactly one opener bootstraps the envelope; the other adopts it"
        );
        for r in [&r1, &r2] {
            if let Err(kind) = r {
                assert_eq!(
                    *kind, "WrongPassphrase",
                    "the loser adopts the winner's envelope and fails the AEAD tag"
                );
            }
        }
    }

    #[test]
    fn test_partial_schema_missing_table_is_dbcorrupt_and_intact() {
        // Step 8d: a partial/foreign schema (a data table missing) ⇒ DbCorrupt
        // naming the table; the surviving tables are left intact (never-delete).
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        {
            let conn = Connection::open(&path).unwrap();
            init_schema(&conn).unwrap();
            conn.execute("DROP TABLE messages", []).unwrap();
            // A surviving row proves nothing is wiped on the corruption path.
            conn.execute(
                "INSERT INTO sessions (id, project_name) VALUES ('s', 'p')",
                [],
            )
            .unwrap();
        }

        let err = EncryptedSqliteMemory::open_with_state_machine(path.clone(), test_master())
            .err()
            .expect("a missing data table is corruption");
        match err {
            VaultError::DbCorrupt { ref detail, .. } => assert!(
                detail.contains("messages"),
                "detail must name the missing table, got {detail:?}"
            ),
            other => panic!("expected DbCorrupt naming the table, got {other:?}"),
        }

        // Intact: the surviving `sessions` row is untouched.
        let reopened = Connection::open(&path).unwrap();
        assert_eq!(
            row_count(&reopened, "sessions"),
            1,
            "never-delete: a partial-schema corruption must not wipe surviving tables"
        );
    }

    #[test]
    fn test_kek_derivation_happens_before_the_bootstrap_write_lock() {
        // Step 8e (R-V08 lock-ordering regression): the expensive KEK derivation
        // must run BEFORE the BEGIN IMMEDIATE write lock. The probe KDF confirms a
        // second connection can take an IMMEDIATE lock AT derivation time — which
        // is only possible if the bootstrap holds no write lock across the KDF.
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        {
            let conn = Connection::open(&path).unwrap();
            init_schema(&conn).unwrap();
        }
        let lock_free = Arc::new(AtomicBool::new(false));
        let vault = CryptoVault::new(
            Box::new(LockProbeKdf {
                inner: FastKdf,
                db_path: path.clone(),
                lock_free_at_derivation: lock_free.clone(),
            }),
            Box::new(Aes256GcmSivCipher),
            Box::new(ConcatenatedFec::default()),
        );

        EncryptedSqliteMemory::open_with_state_machine_vault(path, test_master(), vault)
            .expect("bootstrap must succeed");
        assert!(
            lock_free.load(Ordering::SeqCst),
            "R-V08: the KEK derivation must run BEFORE the BEGIN IMMEDIATE write lock"
        );
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
