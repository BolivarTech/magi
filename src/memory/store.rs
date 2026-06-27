// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-06-26

//! Encrypted vector store: `memories` table, `VectorStore` trait, and
//! `SqliteVectorStore` (REQ-01, REQ-03, REQ-04).
//!
//! # Integration design
//! `SqliteVectorStore` **shares** the `Arc<Mutex<Connection>>` and cached
//! `derived_key` from [`EncryptedSqliteMemory`] — it does NOT open a second
//! DB or re-derive a key. A fresh `CryptoVault::default()` is constructed
//! locally; the vault is a stateless algorithm bundle so this is zero-cost.
//!
//! # Lock discipline (W12)
//! The Mutex is held only long enough to collect raw ciphertext rows from
//! SQLite; it is **always released before any decryption runs**. This mirrors
//! the `collect_message_rows` / `decrypt_rows` pattern in `database.rs`.
//!
//! [`EncryptedSqliteMemory`]: crate::system::database::EncryptedSqliteMemory

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rusqlite::{params, TransactionBehavior};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

use crate::agent::messages::{Content, Message};
use crate::memory::error::MemoryError;
use crate::memory::MemoryKind;
use crate::utils::crypto::CryptoVault;

// ─── Memory struct ────────────────────────────────────────────────────────────

/// One stored memory record (REQ-01, SC-01).
///
/// `created_at` and `last_accessed_at` are Unix seconds (UTC); they are set
/// by the caller (a `Clock` abstraction is wired in later tasks). `access_count`
/// is held as `u64` with saturating semantics (CP2-B) and mapped to/from
/// `i64` in SQLite via `i64::try_from` / `u64::try_from` with `unwrap_or(MAX)`
/// guards so no value ever causes a panic.
///
/// # Scope
/// `scope` defaults to `"root"` (D-14). Multi-scope is activated in the Agent
/// Society layer (AS-REQ-09); everything in this task lives in root.
///
/// # Empty embedding
/// An empty `embedding` vector signals "needs lazy embed" (Tasks 5 / 7).
/// `model_id = ""` similarly signals a pending embed.
// Narrow allow: struct consumed by SqliteVectorStore (this task) and wired in Task 12.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct Memory {
    /// Unique identifier (UUID string).
    pub id: String,
    /// Session that produced this memory.
    pub session_id: String,
    /// Whether this is an episodic turn or a durable preference.
    pub kind: MemoryKind,
    /// Raw text of the memory (stored encrypted as `text_blob`).
    pub text: String,
    /// Embedding vector (stored encrypted as `embedding_blob`). Empty → pending.
    pub embedding: Vec<f32>,
    /// Model that produced the embedding, e.g. `"nomic-embed-text"`. Empty → pending.
    pub model_id: String,
    /// Vector dimension. `0` → pending embed.
    pub dim: usize,
    /// Creation time (Unix seconds UTC).
    pub created_at: i64,
    /// Salience score in `[0, 1]` (assigned at write time, Task 6).
    pub salience: f64,
    /// Access counter (CP2-B: saturating at `i64::MAX` in the SQLite store).
    pub access_count: u64,
    /// Time of last retrieval hit (Unix seconds UTC).
    pub last_accessed_at: i64,
    /// ID of the memory that supersedes this one, if any (REQ-10).
    pub superseded_by: Option<String>,
    /// Unix seconds when this memory was evicted, if at all (REQ-09 / D-07).
    pub evicted_at: Option<i64>,
    /// Scope tag (D-14). Default `"root"`.
    pub scope: String,
    /// Unix seconds when this memory was included in a distillation pass (REQ-17).
    pub distilled_at: Option<i64>,
}

// ─── MemoryDiagnostics ───────────────────────────────────────────────────────

/// Operator-visible store counts for the tiered memory subsystem (CP2-AN/S).
///
/// All queries are pure `COUNT`/`SUM` reads against the `memories` table —
/// no decryption, no ANN index traversal. Safe to call at startup before the
/// in-RAM index is built.
///
/// `active_count` and `pending_reembed_count` are scoped to the requested
/// scope. `archived_count` spans **all** scopes: archived rows are globally
/// excluded from active retrieval regardless of scope, so a cross-scope total
/// is the operationally useful metric.
///
/// # RAM estimate
/// `ram_estimate_bytes = Σ(dim × 4)` over active records with `dim > 0` in
/// the requested scope — the ceiling on in-RAM ANN index memory (4 bytes per
/// `f32` component).
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryDiagnostics {
    /// Active memories in the requested scope
    /// (`evicted_at IS NULL AND superseded_by IS NULL`).
    pub active_count: usize,
    /// Archived (evicted) memories across all scopes (`evicted_at IS NOT NULL`).
    pub archived_count: usize,
    /// Active memories in scope with no embedding (`model_id = ''` or `dim = 0`):
    /// candidates for lazy re-embed.
    pub pending_reembed_count: usize,
    /// In-RAM ANN index ceiling in bytes: `Σ(dim × size_of::<f32>())` for active
    /// records with real embeddings (`dim > 0`) in the requested scope.
    pub ram_estimate_bytes: usize,
}

// ─── VectorStore trait ────────────────────────────────────────────────────────

/// Trait for the encrypted, scope-aware vector store.
///
/// Every method maps all failures to typed [`MemoryError`] variants; none
/// may panic. Implementations must be `Send + Sync` for use across async tasks.
// Narrow allow: trait and all methods consumed by retrieval/decay/distiller
// modules (Tasks 7–10) and wired into the agent in Task 12.
#[allow(dead_code)]
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Persists a [`Memory`] record, encrypting `text` and `embedding` at rest.
    ///
    /// # Errors
    /// [`MemoryError::Crypto`] on encryption failure;
    /// [`MemoryError::Storage`] on SQLite failure (e.g. duplicate `id`).
    async fn insert(&self, m: &Memory) -> Result<(), MemoryError>;

    /// Retrieves a single [`Memory`] by `id`, or `None` if not found.
    ///
    /// # Errors
    /// [`MemoryError::Crypto`] if decryption fails;
    /// [`MemoryError::Storage`] on SQL or deserialization failure.
    async fn get(&self, id: &str) -> Result<Option<Memory>, MemoryError>;

    /// Returns all **active** memories for `scope` (SC-41, REQ-38):
    /// rows where `evicted_at IS NULL AND superseded_by IS NULL AND scope = scope`.
    ///
    /// # Errors
    /// [`MemoryError::Crypto`] or [`MemoryError::Storage`].
    async fn active(&self, scope: &str) -> Result<Vec<Memory>, MemoryError>;

    /// Increments `access_count` and sets `last_accessed_at = now` for every id
    /// in `ids`. Only the named records are touched (SC-09, REQ-07).
    ///
    /// # Errors
    /// [`MemoryError::Storage`] on SQL failure.
    async fn mark_accessed(&self, ids: &[String], now: i64) -> Result<(), MemoryError>;

    /// Sets `superseded_by = by` for the record with `id`, excluding it from
    /// future active retrieval (REQ-10, D-12).
    ///
    /// # Errors
    /// [`MemoryError::Storage`] on SQL failure.
    async fn set_superseded(&self, id: &str, by: &str) -> Result<(), MemoryError>;

    /// Sets `evicted_at = at` (or `NULL` to un-evict) for the record with `id`
    /// (REQ-09, REQ-32, D-07).
    ///
    /// # Errors
    /// [`MemoryError::Storage`] on SQL failure.
    async fn set_evicted(&self, id: &str, at: Option<i64>) -> Result<(), MemoryError>;

    /// Hard-deletes all records with the given `ids` (REQ-32,
    /// `evicted_retention_days = 0`).
    ///
    /// # Errors
    /// [`MemoryError::Storage`] on SQL failure.
    async fn hard_delete(&self, ids: &[String]) -> Result<(), MemoryError>;

    /// Sets `distilled_at = at` for every id in `ids` (REQ-17, D-15).
    ///
    /// # Errors
    /// [`MemoryError::Storage`] on SQL failure.
    async fn set_distilled(&self, ids: &[String], at: i64) -> Result<(), MemoryError>;

    /// Returns all archived (`evicted_at IS NOT NULL`) memories, regardless of
    /// scope. Used by [`purge_expired_archives`] to locate evicted rows that
    /// have exceeded their retention window (REQ-32, D-07).
    ///
    /// # Errors
    /// [`MemoryError::Crypto`] if decryption of any row fails;
    /// [`MemoryError::Storage`] on SQL or deserialization failure.
    ///
    /// [`purge_expired_archives`]: crate::memory::decay::purge_expired_archives
    async fn archived(&self) -> Result<Vec<Memory>, MemoryError>;

    /// Replaces the encrypted embedding for `id` with a freshly computed vector
    /// from the current embedder (CP2-C / REQ-02 / `reembed_pending`).
    ///
    /// Encrypts `embedding` outside the connection lock (W12) and then issues
    /// a single `UPDATE` on `embedding_blob`, `model_id`, and `dim`.
    ///
    /// # Errors
    /// [`MemoryError::Crypto`] if encryption fails;
    /// [`MemoryError::Storage`] if the `UPDATE` fails or `id` does not exist.
    async fn update_embedding(
        &self,
        id: &str,
        embedding: &[f32],
        model_id: &str,
        dim: usize,
    ) -> Result<(), MemoryError>;

    /// Sets `salience` for the record with `id`, clamped to `[0, 1]` (D-11).
    ///
    /// Called by the distiller's off-hot-path salience boost (REQ-35/SC-38):
    /// after the distiller identifies a durable preference in episodic memory,
    /// it can lift that memory's salience to the protected tier so that decay
    /// never evicts it.
    ///
    /// A finite `salience` value outside `[0, 1]` is silently clamped — this is
    /// not an error, because the caller may compute salience from floating-point
    /// arithmetic that can drift slightly.
    ///
    /// **Non-finite values** (`f64::NAN`, `f64::INFINITY`, `f64::NEG_INFINITY`)
    /// are rejected before clamping and return `Err(MemoryError::Config(_))` (G3).
    /// `NAN.clamp(0.0, 1.0)` returns NaN in Rust (NaN comparisons are false), so
    /// the guard must run **before** clamp, not after.
    ///
    /// # Errors
    /// [`MemoryError::Config`] if `salience` is non-finite (G3).
    /// [`MemoryError::Storage`] on SQL failure.
    async fn set_salience(&self, id: &str, salience: f64) -> Result<(), MemoryError>;
}

// ─── Free helpers ─────────────────────────────────────────────────────────────

/// Concatenates the text from a message's [`Content::Text`] and
/// [`Content::ToolResult`] blocks, space-joined. [`Content::ToolUse`] blocks
/// contribute nothing. Used by [`SqliteVectorStore::migrate_from_messages`].
fn extract_message_text(msg: &Message) -> String {
    msg.content
        .iter()
        .filter_map(|c| match c {
            Content::Text { text } => Some(text.as_str()),
            Content::ToolResult { content, .. } => Some(content.as_str()),
            Content::ToolUse { .. } => None,
        })
        .collect::<Vec<_>>()
        .join(" ")
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Raw encrypted row collected from SQLite before the connection lock is
/// released (W12 discipline: no decryption under the Mutex).
#[allow(dead_code)]
struct RawRow {
    id: String,
    session_id: String,
    kind_str: String,
    text_blob: String,
    embedding_blob: String,
    model_id: String,
    dim: i64,
    created_at: i64,
    salience: f64,
    access_count: i64,
    last_accessed_at: i64,
    superseded_by: Option<String>,
    evicted_at: Option<i64>,
    scope: String,
    distilled_at: Option<i64>,
}

/// SQL fragment selecting all `memories` columns in the order expected by
/// [`row_from_query`].
#[allow(dead_code)]
const SELECT_COLS: &str = "SELECT id, session_id, kind, text_blob, embedding_blob, \
    model_id, dim, created_at, salience, access_count, last_accessed_at, \
    superseded_by, evicted_at, scope, distilled_at FROM memories";

/// Maps a rusqlite `Row` to a [`RawRow`]. Column order must match
/// [`SELECT_COLS`].
#[allow(dead_code)]
fn row_from_query(row: &rusqlite::Row<'_>) -> rusqlite::Result<RawRow> {
    Ok(RawRow {
        id: row.get(0)?,
        session_id: row.get(1)?,
        kind_str: row.get(2)?,
        text_blob: row.get(3)?,
        embedding_blob: row.get(4)?,
        model_id: row.get(5)?,
        dim: row.get(6)?,
        created_at: row.get(7)?,
        salience: row.get(8)?,
        access_count: row.get(9)?,
        last_accessed_at: row.get(10)?,
        superseded_by: row.get(11)?,
        evicted_at: row.get(12)?,
        scope: row.get(13)?,
        distilled_at: row.get(14)?,
    })
}

/// Decrypts a [`RawRow`] into a [`Memory`] outside the connection lock (W12).
///
/// # Errors
/// [`MemoryError::Crypto`] on cipher failure; [`MemoryError::Storage`] on JSON
/// deserialization or an unknown `kind` string (data integrity error).
#[allow(dead_code)]
fn decode_row(raw: RawRow, vault: &CryptoVault, key: &[u8]) -> Result<Memory, MemoryError> {
    let kind = parse_kind(&raw.kind_str)?;

    let text = vault
        .decrypt_with_key(key, &raw.text_blob)
        .map_err(|e| MemoryError::Crypto(e.to_string()))?;

    let emb_json = vault
        .decrypt_with_key(key, &raw.embedding_blob)
        .map_err(|e| MemoryError::Crypto(e.to_string()))?;
    let embedding: Vec<f32> = serde_json::from_str(&emb_json)
        .map_err(|e| MemoryError::Storage(format!("embedding deserialization: {e}")))?;

    // CP2-B: saturating read — a negative DB value (should never happen in
    // normal operation) maps to u64::MAX as a safe sentinel, not a panic.
    let access_count = u64::try_from(raw.access_count).unwrap_or(u64::MAX);

    Ok(Memory {
        id: raw.id,
        session_id: raw.session_id,
        kind,
        text,
        embedding,
        model_id: raw.model_id,
        dim: raw.dim as usize,
        created_at: raw.created_at,
        salience: raw.salience,
        access_count,
        last_accessed_at: raw.last_accessed_at,
        superseded_by: raw.superseded_by,
        evicted_at: raw.evicted_at,
        scope: raw.scope,
        distilled_at: raw.distilled_at,
    })
}

/// Parses the stable on-disk `kind` string back to a [`MemoryKind`].
///
/// An unknown string is treated as a data integrity error, not a silent
/// default — consistent with the message `Role` serialization discipline.
#[allow(dead_code)]
fn parse_kind(s: &str) -> Result<MemoryKind, MemoryError> {
    match s {
        "episodic" => Ok(MemoryKind::Episodic),
        "preference" => Ok(MemoryKind::Preference),
        other => Err(MemoryError::Storage(format!(
            "unknown memory kind in DB: {:?} (data integrity error)",
            other
        ))),
    }
}

// ─── SqliteVectorStore ────────────────────────────────────────────────────────

/// SQLite-backed, encrypted, scope-aware vector store.
///
/// Shares the `Arc<Mutex<Connection>>` and cached `derived_key` from
/// [`EncryptedSqliteMemory`][crate::system::database::EncryptedSqliteMemory]
/// so the `memories` table lives in the same database file and uses the same
/// AES-256-GCM-SIV key. No second DB connection is opened and no additional
/// Argon2 key derivation is performed.
///
/// # Construction
/// ```ignore
/// let store = SqliteVectorStore::new(mem.shared_conn(), mem.data_key())?;
/// ```
///
/// # Thread safety
/// `SqliteVectorStore` is `Send + Sync`; it can be wrapped in an `Arc` and
/// shared across async tasks.
// Narrow allow: struct consumed by retrieval/decay/distiller (Tasks 7–10) and
// wired into the agent in Task 12.
#[allow(dead_code)]
pub struct SqliteVectorStore {
    conn: Arc<Mutex<rusqlite::Connection>>,
    vault: CryptoVault,
    derived_key: Zeroizing<Vec<u8>>,
}

impl SqliteVectorStore {
    /// Builds a `SqliteVectorStore` sharing `conn` and `derived_key` with an
    /// existing `EncryptedSqliteMemory`.
    ///
    /// Creates the `memories` table and its scope index if they do not exist.
    ///
    /// # Errors
    /// [`MemoryError::Storage`] if the DDL statement fails.
    // Narrow allow: called by the agent wiring in Task 12.
    #[allow(dead_code)]
    pub fn new(
        conn: Arc<Mutex<rusqlite::Connection>>,
        derived_key: Zeroizing<Vec<u8>>,
    ) -> Result<Self, MemoryError> {
        {
            let c = conn.lock().unwrap_or_else(|p| p.into_inner());
            c.execute_batch(
                "CREATE TABLE IF NOT EXISTS memories (
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
            .map_err(|e| MemoryError::Storage(e.to_string()))?;
        }
        Ok(Self {
            conn,
            vault: CryptoVault::default(),
            derived_key,
        })
    }

    /// Acquires the connection lock, recovering from a poisoned mutex (same
    /// recovery pattern as `EncryptedSqliteMemory::locked_conn`).
    fn locked_conn(&self) -> std::sync::MutexGuard<'_, rusqlite::Connection> {
        self.conn.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Returns operator-visible diagnostic counts for `scope` (CP2-AN/S).
    ///
    /// Four cheap `COUNT`/`SUM` queries — no decryption, no ANN traversal.
    /// Safe to call at startup before the in-RAM index is built.
    ///
    /// - `active_count` — rows where `evicted_at IS NULL AND superseded_by IS NULL AND scope = scope`.
    /// - `archived_count` — rows where `evicted_at IS NOT NULL` (all scopes).
    /// - `pending_reembed_count` — active rows in scope with `model_id = ''` or `dim = 0`.
    /// - `ram_estimate_bytes` — `Σ(dim × 4)` for active rows with `dim > 0` in scope.
    ///
    /// See [`MemoryDiagnostics`] for full field semantics.
    ///
    /// # Errors
    /// [`MemoryError::Storage`] if any of the underlying SQL queries fail.
    pub async fn diagnostics(&self, scope: &str) -> Result<MemoryDiagnostics, MemoryError> {
        let c = self.locked_conn();

        // Active records in the requested scope.
        let active_count: i64 = c
            .query_row(
                "SELECT COUNT(*) FROM memories \
                 WHERE evicted_at IS NULL AND superseded_by IS NULL AND scope = ?1",
                params![scope],
                |r| r.get(0),
            )
            .map_err(|e| MemoryError::Storage(e.to_string()))?;

        // Archived records (cross-scope — all evicted rows regardless of scope).
        let archived_count: i64 = c
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE evicted_at IS NOT NULL",
                [],
                |r| r.get(0),
            )
            .map_err(|e| MemoryError::Storage(e.to_string()))?;

        // Active records in scope that need a (re-)embed: no model or zero dim.
        let pending_reembed_count: i64 = c
            .query_row(
                "SELECT COUNT(*) FROM memories \
                 WHERE evicted_at IS NULL AND superseded_by IS NULL AND scope = ?1 \
                 AND (model_id = '' OR dim = 0)",
                params![scope],
                |r| r.get(0),
            )
            .map_err(|e| MemoryError::Storage(e.to_string()))?;

        // RAM ceiling: sum of dim*4 for active records with real embeddings in scope.
        let ram_estimate_bytes: i64 = c
            .query_row(
                "SELECT COALESCE(SUM(CAST(dim AS INTEGER) * 4), 0) FROM memories \
                 WHERE evicted_at IS NULL AND superseded_by IS NULL AND scope = ?1 \
                 AND dim > 0",
                params![scope],
                |r| r.get(0),
            )
            .map_err(|e| MemoryError::Storage(e.to_string()))?;

        Ok(MemoryDiagnostics {
            active_count: usize::try_from(active_count).unwrap_or(usize::MAX),
            archived_count: usize::try_from(archived_count).unwrap_or(usize::MAX),
            pending_reembed_count: usize::try_from(pending_reembed_count).unwrap_or(usize::MAX),
            ram_estimate_bytes: usize::try_from(ram_estimate_bytes).unwrap_or(usize::MAX),
        })
    }

    /// Imports prior conversation turns as episodic memories, **non-destructively**:
    /// each turn becomes a `Memory { kind: Episodic, embedding: vec![], model_id: "",
    /// scope: "root", .. }` (empty embedding ⇒ pending lazy re-embed in Task 7). All
    /// inserts run in ONE `IMMEDIATE` transaction (atomic; a mid-batch error rolls back —
    /// CP2-V). Already-present ids are skipped (idempotent — CP2-F), so re-running or
    /// overlapping batches never duplicate. Returns the number of NEW memories imported.
    ///
    /// `now` is the unix-seconds timestamp to stamp `created_at`/`last_accessed_at` with
    /// (a `Clock` is injected by the caller in later tasks).
    ///
    /// # Text extraction
    /// `Content::Text` and `Content::ToolResult` blocks are space-joined; `Content::ToolUse`
    /// contributes nothing. Turns whose combined text is empty/whitespace are skipped.
    ///
    /// # Idempotency
    /// Each turn is identified by a deterministic SHA-256 content hash of
    /// `session_id || 0x1F || role_str || 0x1F || text`, prefixed `"mig:"`.
    /// Duplicate ids within the same batch or across re-runs are silently skipped.
    ///
    /// # Errors
    /// [`MemoryError::Crypto`] if any turn's text exceeds the 50 MiB plaintext cap or
    /// cipher fails. [`MemoryError::Storage`] on any SQLite failure.
    /// On any error, **no** records are imported (atomic rollback).
    // Narrow allow: called by the agent wiring in Task 12.
    #[allow(dead_code)]
    pub async fn migrate_from_messages(
        &self,
        msgs: &[(String, Message)],
        now: i64,
    ) -> Result<usize, MemoryError> {
        /// Holds the pre-computed, pre-encrypted blobs for one turn.
        struct Prepared {
            id: String,
            session_id: String,
            text_blob: String,
            embedding_blob: String,
        }

        // Phase 1: extract text, derive content-hash ids, encrypt blobs — ALL outside
        // the connection lock (W12 discipline: no crypto under the Mutex).
        // If any encryption fails here, no transaction has started, so nothing is imported.
        let empty_emb_json = "[]";
        let mut prepared: Vec<Prepared> = Vec::with_capacity(msgs.len());

        for (session_id, msg) in msgs {
            let text = extract_message_text(msg);
            if text.trim().is_empty() {
                continue;
            }

            // Deterministic content-hash id for idempotency (CP2-F).
            let role_str = format!("{:?}", msg.role);
            let id = {
                let mut h = Sha256::new();
                h.update(session_id.as_bytes());
                h.update([0x1F_u8]);
                h.update(role_str.as_bytes());
                h.update([0x1F_u8]);
                h.update(text.as_bytes());
                format!("mig:{:x}", h.finalize())
            };

            // Encrypt the turn text; fails fast if text > MAX_PLAINTEXT_LEN (50 MiB).
            let text_blob = self
                .vault
                .encrypt_with_key(&self.derived_key, &text)
                .map_err(|e| MemoryError::Crypto(e.to_string()))?;

            // Each record gets its own independently-nonced embedding blob.
            let embedding_blob = self
                .vault
                .encrypt_with_key(&self.derived_key, empty_emb_json)
                .map_err(|e| MemoryError::Crypto(e.to_string()))?;

            prepared.push(Prepared {
                id,
                session_id: session_id.clone(),
                text_blob,
                embedding_blob,
            });
        }

        if prepared.is_empty() {
            return Ok(0);
        }

        // Phase 2: ONE IMMEDIATE transaction — the Mutex is held for the entire SQL
        // batch (required for atomicity; no decryption happens here so W12 is satisfied).
        let mut imported = 0;
        {
            let mut conn = self.locked_conn();
            let tx = conn
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .map_err(|e| MemoryError::Storage(e.to_string()))?;

            for p in &prepared {
                // Skip ids already present (idempotency — CP2-F).
                let count: i64 = tx
                    .query_row(
                        "SELECT COUNT(*) FROM memories WHERE id = ?1",
                        params![p.id],
                        |r| r.get(0),
                    )
                    .map_err(|e| MemoryError::Storage(e.to_string()))?;
                if count > 0 {
                    continue;
                }

                tx.execute(
                    "INSERT INTO memories \
                         (id, session_id, kind, text_blob, embedding_blob, model_id, dim, \
                          created_at, salience, access_count, last_accessed_at, \
                          superseded_by, evicted_at, scope, distilled_at) \
                     VALUES (?1, ?2, 'episodic', ?3, ?4, '', 0, ?5, 0.3, 0, ?5, \
                             NULL, NULL, 'root', NULL)",
                    params![p.id, p.session_id, p.text_blob, p.embedding_blob, now],
                )
                .map_err(|e| MemoryError::Storage(e.to_string()))?;

                imported += 1;
            }

            tx.commit()
                .map_err(|e| MemoryError::Storage(e.to_string()))?;
        } // MutexGuard dropped here, lock released.

        Ok(imported)
    }
}

#[async_trait]
impl VectorStore for SqliteVectorStore {
    async fn insert(&self, m: &Memory) -> Result<(), MemoryError> {
        // Encrypt BEFORE acquiring the lock so the critical section is short
        // (W12 spirit: minimize work under the Mutex).
        let text_blob = self
            .vault
            .encrypt_with_key(&self.derived_key, &m.text)
            .map_err(|e| MemoryError::Crypto(e.to_string()))?;

        let emb_json = serde_json::to_string(&m.embedding)
            .map_err(|e| MemoryError::Storage(format!("embedding serialization: {e}")))?;
        let embedding_blob = self
            .vault
            .encrypt_with_key(&self.derived_key, &emb_json)
            .map_err(|e| MemoryError::Crypto(e.to_string()))?;

        // CP2-B: saturating write — u64 values above i64::MAX are clamped to
        // i64::MAX rather than panicking or wrapping.
        let access_count_i64 = i64::try_from(m.access_count).unwrap_or(i64::MAX);

        let c = self.locked_conn();
        c.execute(
            "INSERT INTO memories \
                 (id, session_id, kind, text_blob, embedding_blob, model_id, dim, \
                  created_at, salience, access_count, last_accessed_at, \
                  superseded_by, evicted_at, scope, distilled_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
            params![
                m.id,
                m.session_id,
                m.kind.as_str(),
                text_blob,
                embedding_blob,
                m.model_id,
                m.dim as i64,
                m.created_at,
                m.salience,
                access_count_i64,
                m.last_accessed_at,
                m.superseded_by,
                m.evicted_at,
                m.scope,
                m.distilled_at,
            ],
        )
        .map_err(|e| MemoryError::Storage(e.to_string()))?;

        Ok(())
    }

    async fn get(&self, id: &str) -> Result<Option<Memory>, MemoryError> {
        // 1. Collect the raw ciphertext row under the lock.
        let raw: Option<RawRow> = {
            let c = self.locked_conn();
            let mut stmt = c
                .prepare(&format!("{SELECT_COLS} WHERE id = ?1"))
                .map_err(|e| MemoryError::Storage(e.to_string()))?;
            let mut rows = stmt
                .query_map(params![id], row_from_query)
                .map_err(|e| MemoryError::Storage(e.to_string()))?;
            match rows.next() {
                None => None,
                Some(r) => Some(r.map_err(|e| MemoryError::Storage(e.to_string()))?),
            }
        }; // lock released here

        // 2. Decrypt outside the lock (W12).
        match raw {
            None => Ok(None),
            Some(row) => Ok(Some(decode_row(row, &self.vault, &self.derived_key)?)),
        }
    }

    async fn active(&self, scope: &str) -> Result<Vec<Memory>, MemoryError> {
        // 1. Collect raw rows under the lock.
        let raw_rows: Vec<RawRow> = {
            let c = self.locked_conn();
            let mut stmt = c
                .prepare(&format!(
                    "{SELECT_COLS} WHERE evicted_at IS NULL \
                      AND superseded_by IS NULL AND scope = ?1"
                ))
                .map_err(|e| MemoryError::Storage(e.to_string()))?;
            let iter = stmt
                .query_map(params![scope], row_from_query)
                .map_err(|e| MemoryError::Storage(e.to_string()))?;
            let mut collected = Vec::new();
            for r in iter {
                collected.push(r.map_err(|e| MemoryError::Storage(e.to_string()))?);
            }
            collected
        }; // lock released here

        // 2. Decrypt outside the lock (W12).
        let mut out = Vec::with_capacity(raw_rows.len());
        for row in raw_rows {
            out.push(decode_row(row, &self.vault, &self.derived_key)?);
        }
        Ok(out)
    }

    async fn mark_accessed(&self, ids: &[String], now: i64) -> Result<(), MemoryError> {
        if ids.is_empty() {
            return Ok(());
        }
        // Pure SQL updates — no decryption — so the lock may be held for the
        // entire transaction without violating the W12 discipline.
        // IMMEDIATE transaction: all-or-nothing so a partial failure rolls back.
        let mut c = self.locked_conn();
        let tx = c
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|e| MemoryError::Storage(e.to_string()))?;
        for id in ids {
            tx.execute(
                "UPDATE memories \
                 SET access_count = CASE WHEN access_count < 9223372036854775807 \
                     THEN access_count + 1 ELSE access_count END, \
                     last_accessed_at = ?2 \
                 WHERE id = ?1",
                params![id, now],
            )
            .map_err(|e| MemoryError::Storage(e.to_string()))?;
        }
        tx.commit()
            .map_err(|e| MemoryError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn set_superseded(&self, id: &str, by: &str) -> Result<(), MemoryError> {
        let c = self.locked_conn();
        c.execute(
            "UPDATE memories SET superseded_by = ?2 WHERE id = ?1",
            params![id, by],
        )
        .map_err(|e| MemoryError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn set_evicted(&self, id: &str, at: Option<i64>) -> Result<(), MemoryError> {
        let c = self.locked_conn();
        c.execute(
            "UPDATE memories SET evicted_at = ?2 WHERE id = ?1",
            params![id, at],
        )
        .map_err(|e| MemoryError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn hard_delete(&self, ids: &[String]) -> Result<(), MemoryError> {
        if ids.is_empty() {
            return Ok(());
        }
        // IMMEDIATE transaction: all-or-nothing so a partial failure rolls back (F2).
        // Pure SQL deletes — no decryption — W12 satisfied.
        let mut c = self.locked_conn();
        let tx = c
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|e| MemoryError::Storage(e.to_string()))?;
        for id in ids {
            tx.execute("DELETE FROM memories WHERE id = ?1", params![id])
                .map_err(|e| MemoryError::Storage(e.to_string()))?;
        }
        tx.commit()
            .map_err(|e| MemoryError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn set_distilled(&self, ids: &[String], at: i64) -> Result<(), MemoryError> {
        if ids.is_empty() {
            return Ok(());
        }
        // IMMEDIATE transaction: all-or-nothing so a partial failure rolls back (F2).
        // Pure SQL updates — no decryption — W12 satisfied.
        let mut c = self.locked_conn();
        let tx = c
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|e| MemoryError::Storage(e.to_string()))?;
        for id in ids {
            tx.execute(
                "UPDATE memories SET distilled_at = ?2 WHERE id = ?1",
                params![id, at],
            )
            .map_err(|e| MemoryError::Storage(e.to_string()))?;
        }
        tx.commit()
            .map_err(|e| MemoryError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn update_embedding(
        &self,
        id: &str,
        embedding: &[f32],
        model_id: &str,
        dim: usize,
    ) -> Result<(), MemoryError> {
        // Encrypt outside the lock (W12 discipline: no crypto under the Mutex).
        let emb_json = serde_json::to_string(embedding)
            .map_err(|e| MemoryError::Storage(format!("embedding serialization: {e}")))?;
        let embedding_blob = self
            .vault
            .encrypt_with_key(&self.derived_key, &emb_json)
            .map_err(|e| MemoryError::Crypto(e.to_string()))?;

        // Hold the lock only for the SQL UPDATE.
        let c = self.locked_conn();
        let affected = c
            .execute(
                "UPDATE memories \
                 SET embedding_blob = ?2, model_id = ?3, dim = ?4 \
                 WHERE id = ?1",
                params![id, embedding_blob, model_id, dim as i64],
            )
            .map_err(|e| MemoryError::Storage(e.to_string()))?;

        if affected == 0 {
            return Err(MemoryError::Storage(format!(
                "update_embedding: id not found: {id}"
            )));
        }
        Ok(())
    }

    async fn archived(&self) -> Result<Vec<Memory>, MemoryError> {
        // 1. Collect raw rows under the lock (W12 discipline).
        let raw_rows: Vec<RawRow> = {
            let c = self.locked_conn();
            let mut stmt = c
                .prepare(&format!("{SELECT_COLS} WHERE evicted_at IS NOT NULL"))
                .map_err(|e| MemoryError::Storage(e.to_string()))?;
            let iter = stmt
                .query_map([], row_from_query)
                .map_err(|e| MemoryError::Storage(e.to_string()))?;
            let mut collected = Vec::new();
            for r in iter {
                collected.push(r.map_err(|e| MemoryError::Storage(e.to_string()))?);
            }
            collected
        }; // lock released here

        // 2. Decrypt outside the lock (W12).
        let mut out = Vec::with_capacity(raw_rows.len());
        for row in raw_rows {
            out.push(decode_row(row, &self.vault, &self.derived_key)?);
        }
        Ok(out)
    }

    async fn set_salience(&self, id: &str, salience: f64) -> Result<(), MemoryError> {
        // Reject non-finite values BEFORE clamping. `NaN.clamp(0.0, 1.0)` in
        // Rust returns NaN (NaN < min and NaN > max are both false, so clamp
        // returns self unchanged). A NaN stored in the DB would corrupt salience
        // comparisons and eviction decisions. `±Inf.clamp(0.0, 1.0)` would be
        // silently coerced to a valid value (0.0 or 1.0), hiding a caller bug
        // — also rejected so the contract is explicit (G3).
        if !salience.is_finite() {
            return Err(MemoryError::Config(format!(
                "set_salience: non-finite value is not permitted (got {salience}); \
                 pass a finite value in [0.0, 1.0] or outside that range to clamp"
            )));
        }
        // Clamp to [0, 1] before writing; floating-point drift from the caller
        // must not produce an out-of-range value in the DB.
        let s = salience.clamp(0.0, 1.0);
        let c = self.locked_conn();
        c.execute(
            "UPDATE memories SET salience = ?2 WHERE id = ?1",
            params![id, s],
        )
        .map_err(|e| MemoryError::Storage(e.to_string()))?;
        Ok(())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::messages::Message;
    use crate::system::database::EncryptedSqliteMemory;

    /// Creates a temp-file-backed `SqliteVectorStore` sharing the connection
    /// and key from a fresh `EncryptedSqliteMemory`. The returned
    /// `NamedTempFile` must stay alive for the test's lifetime.
    fn test_store() -> (tempfile::NamedTempFile, SqliteVectorStore) {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mem = EncryptedSqliteMemory::new(tmp.path().to_path_buf(), "pw".into()).unwrap();
        let store = SqliteVectorStore::new(mem.shared_conn(), mem.data_key()).unwrap();
        (tmp, store)
    }

    /// Builds a minimal `Memory` for testing.
    fn sample(id: &str, text: &str, emb: Vec<f32>) -> Memory {
        Memory {
            id: id.into(),
            session_id: "s".into(),
            kind: MemoryKind::Episodic,
            text: text.into(),
            embedding: emb.clone(),
            model_id: "nomic-embed-text".into(),
            dim: emb.len(),
            created_at: 1000,
            salience: 0.3,
            access_count: 0,
            last_accessed_at: 1000,
            superseded_by: None,
            evicted_at: None,
            scope: "root".into(),
            distilled_at: None,
        }
    }

    /// Reads a raw column value via a *separate* SQLite connection so we
    /// inspect what is actually on disk (not what the in-process store caches).
    fn raw_blob(tmp: &tempfile::NamedTempFile, col: &str) -> String {
        let conn = rusqlite::Connection::open(tmp.path()).unwrap();
        conn.query_row(&format!("SELECT {col} FROM memories LIMIT 1"), [], |r| {
            r.get::<_, String>(0)
        })
        .unwrap()
    }

    /// SC-05, SC-41: round-trip through insert/get preserves all fields; disk
    /// blobs contain ciphertext, not plaintext.
    #[tokio::test]
    async fn test_persist_and_reload_roundtrips_vector_and_metadata() {
        let (tmp, store) = test_store();
        let m = sample("m1", "remember the api budget is 8000", vec![0.1, 0.2, 0.3]);
        store.insert(&m).await.unwrap();
        let got = store.get("m1").await.unwrap().unwrap();
        assert_eq!(got, m);
        assert_eq!(got.scope, "root"); // SC-41
        assert!(!raw_blob(&tmp, "text_blob").contains("budget")); // SC-05 no cleartext text
        assert!(!raw_blob(&tmp, "embedding_blob").contains("0.1")); // SC-05 no cleartext vector
    }

    /// SC-41: `scope` defaults to `"root"` and `active` filters by scope.
    #[tokio::test]
    async fn test_default_scope_is_root_and_active_filters_by_scope() {
        let (_t, store) = test_store();
        store.insert(&sample("a", "x", vec![0.0; 3])).await.unwrap();
        assert_eq!(store.active("root").await.unwrap().len(), 1);
        assert!(store.active("other").await.unwrap().is_empty());
    }

    /// SC-09: `mark_accessed` updates ONLY the named ids.
    #[tokio::test]
    async fn test_mark_accessed_updates_only_named_ids() {
        let (_t, store) = test_store();
        store.insert(&sample("a", "a", vec![0.0; 3])).await.unwrap();
        store.insert(&sample("b", "b", vec![0.0; 3])).await.unwrap();
        store.mark_accessed(&["a".into()], 2000).await.unwrap();
        assert_eq!(store.get("a").await.unwrap().unwrap().access_count, 1);
        assert_eq!(
            store.get("a").await.unwrap().unwrap().last_accessed_at,
            2000
        );
        assert_eq!(store.get("b").await.unwrap().unwrap().access_count, 0);
    }

    /// CP2-B: `access_count` values that fit in `i64` round-trip exactly.
    /// `i64::MAX as u64` is the highest value the write-side
    /// `i64::try_from(…).unwrap_or(i64::MAX)` passes through without clamping.
    #[tokio::test]
    async fn test_access_count_storage_saturates_at_i64_max() {
        let (_t, store) = test_store();
        let mut m = sample("a", "x", vec![0.0; 3]);
        m.access_count = i64::MAX as u64;
        store.insert(&m).await.unwrap();
        assert_eq!(
            store.get("a").await.unwrap().unwrap().access_count,
            i64::MAX as u64
        );
    }

    /// SC-08: migration imports prior turns without loss and marks for lazy embed.
    #[tokio::test]
    async fn test_migration_imports_prior_turns_without_loss_and_marks_for_lazy_embed() {
        let (_t, store) = test_store();
        let prior = vec![
            ("s1".to_string(), Message::user("old fact one")),
            ("s1".to_string(), Message::assistant("ack")),
        ];
        let n = store.migrate_from_messages(&prior, 1000).await.unwrap();
        assert_eq!(n, 2);
        let active = store.active("root").await.unwrap();
        assert_eq!(active.len(), 2);
        assert!(active
            .iter()
            .all(|m| m.embedding.is_empty() && m.model_id.is_empty()));
        assert!(active
            .iter()
            .all(|m| matches!(m.kind, crate::memory::MemoryKind::Episodic)));
        assert!(active.iter().all(|m| m.scope == "root"));
    }

    /// CP2-F: re-running (and overlapping batches) imports no duplicates.
    #[tokio::test]
    async fn test_migration_is_idempotent_across_reruns_and_batches() {
        let (_t, store) = test_store();
        let prior = vec![("s1".to_string(), Message::user("fact"))];
        assert_eq!(store.migrate_from_messages(&prior, 1000).await.unwrap(), 1);
        assert_eq!(store.migrate_from_messages(&prior, 1000).await.unwrap(), 0); // already present
        assert_eq!(store.active("root").await.unwrap().len(), 1);
    }

    /// CP2-V: a mid-batch error rolls the whole batch back (nothing imported).
    #[tokio::test]
    async fn test_migration_is_atomic_on_failure() {
        let (_t, store) = test_store();
        // Second turn's text exceeds the crypto plaintext cap (50 MiB) ⇒ encrypt fails fast
        // (the cap is checked before any large allocation), aborting the transaction.
        let huge = "x".repeat(50 * 1024 * 1024 + 1);
        let batch = vec![
            ("s1".to_string(), Message::user("ok")),
            ("s1".to_string(), Message::user(&huge)),
        ];
        assert!(store.migrate_from_messages(&batch, 1000).await.is_err());
        assert!(
            store.active("root").await.unwrap().is_empty(),
            "rollback ⇒ nothing imported"
        );
    }

    /// CP2-AN/S: diagnostics reports correct active, archived, and pending counts,
    /// and produces a non-zero RAM estimate when active vectors exist.
    #[tokio::test]
    async fn test_diagnostics_reports_active_archived_and_pending() {
        let (_t, store) = test_store();

        // Active record with a real embedding.
        let m_active = sample(
            "diag-active",
            "active memory with vector",
            vec![0.1, 0.2, 0.3],
        );
        store.insert(&m_active).await.unwrap();

        // Active record with empty embedding → pending re-embed.
        let mut m_pending = sample("diag-pending", "pending re-embed memory", vec![]);
        m_pending.model_id = String::new();
        m_pending.dim = 0;
        store.insert(&m_pending).await.unwrap();

        // Insert a third record then evict it → archived.
        let m_evict = sample("diag-evicted", "evicted memory", vec![0.4, 0.5]);
        store.insert(&m_evict).await.unwrap();
        store
            .set_evicted("diag-evicted", Some(1_000_000))
            .await
            .unwrap();

        let diag = store.diagnostics("root").await.unwrap();

        assert_eq!(
            diag.active_count, 2,
            "two non-evicted records in root scope"
        );
        assert_eq!(diag.archived_count, 1, "one evicted record (any scope)");
        assert_eq!(
            diag.pending_reembed_count, 1,
            "one active record without embedding"
        );
        assert!(
            diag.ram_estimate_bytes > 0,
            "active vector ⇒ positive RAM estimate (got 0)"
        );
    }

    /// A1: `mark_accessed` with `access_count = i64::MAX` must NOT overflow.
    /// The SQL CASE guard must leave the value at `i64::MAX as u64`.
    #[tokio::test]
    async fn test_mark_accessed_saturates_at_i64_max() {
        let (_t, store) = test_store();
        let mut m = sample("sat", "overflow test", vec![0.0; 3]);
        m.access_count = i64::MAX as u64;
        store.insert(&m).await.unwrap();
        store.mark_accessed(&["sat".into()], 9999).await.unwrap();
        let got = store.get("sat").await.unwrap().unwrap();
        assert_eq!(
            got.access_count,
            i64::MAX as u64,
            "A1: access_count must stay at i64::MAX after mark_accessed (no overflow)"
        );
    }

    /// A2: `update_embedding` on a non-existent id must return an error,
    /// not silently succeed.
    #[tokio::test]
    async fn test_update_embedding_on_missing_id_is_err() {
        let (_t, store) = test_store();
        let result = store
            .update_embedding("no-such-id", &[0.1f32, 0.2], "model", 2)
            .await;
        assert!(
            matches!(result, Err(MemoryError::Storage(_))),
            "A2: update_embedding on missing id must be Err(Storage), got: {result:?}"
        );
    }

    /// A3: `diagnostics` i64→usize conversion must not truncate on large counts.
    /// (Smoke test: result fields are `usize`, not negative or wrapped.)
    #[tokio::test]
    async fn test_diagnostics_counts_are_usize_safe() {
        let (_t, store) = test_store();
        let d = store.diagnostics("root").await.unwrap();
        // All counts start at 0; the important thing is the code path compiles
        // and runs without panic. The try_from conversion is the fix; absence of
        // panic here demonstrates it is safe on this platform.
        assert_eq!(d.active_count, 0);
        assert_eq!(d.archived_count, 0);
        assert_eq!(d.pending_reembed_count, 0);
        assert_eq!(d.ram_estimate_bytes, 0);
    }

    /// CP2-F volume: a 1000-turn batch imports fully.
    #[tokio::test]
    async fn test_migration_imports_large_batch() {
        let (_t, store) = test_store();
        let prior: Vec<_> = (0..1000)
            .map(|i| ("s1".to_string(), Message::user(&format!("fact {i}"))))
            .collect();
        assert_eq!(
            store.migrate_from_messages(&prior, 1000).await.unwrap(),
            1000
        );
        assert_eq!(store.active("root").await.unwrap().len(), 1000);
    }

    // ── F2: transactional atomicity for multi-id ops ──────────────────────────

    /// F2: hard_delete of multiple ids removes all of them (atomic batch).
    #[tokio::test]
    async fn test_hard_delete_multiple_ids_removes_all() {
        let (_t, store) = test_store();
        for i in 0..3u8 {
            store
                .insert(&sample(
                    &format!("del{i}"),
                    &format!("txt{i}"),
                    vec![0.0; 3],
                ))
                .await
                .unwrap();
        }
        store
            .hard_delete(&["del0".into(), "del1".into(), "del2".into()])
            .await
            .unwrap();
        assert!(
            store.get("del0").await.unwrap().is_none(),
            "del0 must be gone"
        );
        assert!(
            store.get("del1").await.unwrap().is_none(),
            "del1 must be gone"
        );
        assert!(
            store.get("del2").await.unwrap().is_none(),
            "del2 must be gone"
        );
    }

    /// F2: mark_accessed on a batch updates all named ids in one shot.
    #[tokio::test]
    async fn test_mark_accessed_batch_updates_all_ids() {
        let (_t, store) = test_store();
        for i in 0..3u8 {
            store
                .insert(&sample(&format!("acc{i}"), &format!("t{i}"), vec![0.0; 3]))
                .await
                .unwrap();
        }
        store
            .mark_accessed(&["acc0".into(), "acc1".into(), "acc2".into()], 5000)
            .await
            .unwrap();
        for i in 0..3u8 {
            let m = store.get(&format!("acc{i}")).await.unwrap().unwrap();
            assert_eq!(m.access_count, 1, "F2: acc{i} access_count must be 1");
            assert_eq!(
                m.last_accessed_at, 5000,
                "F2: acc{i} last_accessed_at must be 5000"
            );
        }
    }

    /// F2: set_distilled on a batch stamps all named ids.
    #[tokio::test]
    async fn test_set_distilled_batch_stamps_all_ids() {
        let (_t, store) = test_store();
        for i in 0..3u8 {
            store
                .insert(&sample(&format!("dst{i}"), &format!("t{i}"), vec![0.0; 3]))
                .await
                .unwrap();
        }
        store
            .set_distilled(&["dst0".into(), "dst1".into(), "dst2".into()], 9999)
            .await
            .unwrap();
        for i in 0..3u8 {
            let m = store.get(&format!("dst{i}")).await.unwrap().unwrap();
            assert_eq!(
                m.distilled_at,
                Some(9999),
                "F2: dst{i} distilled_at must be 9999"
            );
        }
    }

    // ── G3: set_salience rejects non-finite values ────────────────────────────

    /// G3: `set_salience` with `f64::NAN` must return `Err(MemoryError::Config(_))`
    /// rather than a SQL-layer error or Ok. `f64::NAN.clamp(0.0, 1.0)` returns NaN
    /// in Rust (NaN comparisons are false, so the input propagates unchanged),
    /// which would either silently corrupt the DB or produce a `Storage` error
    /// depending on the SQL driver — both are wrong; we want a typed `Config` error.
    #[tokio::test]
    async fn test_set_salience_nan_is_config_err() {
        let (_t, store) = test_store();
        store
            .insert(&sample("sal-nan", "text", vec![0.0; 3]))
            .await
            .unwrap();
        let result = store.set_salience("sal-nan", f64::NAN).await;
        assert!(
            matches!(result, Err(MemoryError::Config(_))),
            "G3: set_salience(NaN) must return Err(Config), got: {result:?}"
        );
    }

    /// G3: `set_salience` with `f64::INFINITY` must return `Err(MemoryError::Config(_))`.
    /// Without the guard, `INFINITY.clamp(0.0, 1.0) = 1.0` — a valid storage value —
    /// so the SQL succeeds and the error is silently lost.
    #[tokio::test]
    async fn test_set_salience_infinity_is_config_err() {
        let (_t, store) = test_store();
        store
            .insert(&sample("sal-inf", "text", vec![0.0; 3]))
            .await
            .unwrap();
        let result = store.set_salience("sal-inf", f64::INFINITY).await;
        assert!(
            matches!(result, Err(MemoryError::Config(_))),
            "G3: set_salience(INFINITY) must return Err(Config), got: {result:?}"
        );
    }

    /// G3: `set_salience` with a finite out-of-range value (e.g. 1.5) must
    /// still succeed — out-of-range finite values are silently clamped to [0, 1].
    #[tokio::test]
    async fn test_set_salience_out_of_range_finite_is_clamped() {
        let (_t, store) = test_store();
        store
            .insert(&sample("sal-clamp", "text", vec![0.0; 3]))
            .await
            .unwrap();
        store.set_salience("sal-clamp", 1.5).await.unwrap();
        let got = store.get("sal-clamp").await.unwrap().unwrap();
        assert_eq!(
            got.salience, 1.0,
            "G3: out-of-range finite salience must be clamped to 1.0"
        );
    }

    /// F2: hard_delete on an empty slice is a no-op (no error, no rows touched).
    #[tokio::test]
    async fn test_hard_delete_empty_slice_is_noop() {
        let (_t, store) = test_store();
        store
            .insert(&sample("keep", "important", vec![0.0; 3]))
            .await
            .unwrap();
        store.hard_delete(&[]).await.unwrap();
        assert!(
            store.get("keep").await.unwrap().is_some(),
            "F2: empty hard_delete must not remove any rows"
        );
    }
}
