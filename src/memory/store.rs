// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-06-26

//! Encrypted vector store: `memories` table, `VectorStore` trait, and
//! `SqliteVectorStore` (REQ-01, REQ-03, REQ-04).
//!
//! # Integration design
//! `SqliteVectorStore` shares the `Arc<Mutex<Connection>>` and cached
//! `derived_key` from `EncryptedSqliteMemory`. See [`SqliteVectorStore::new`].
//!
//! # Lock discipline (W12)
//! The Mutex is held only to collect raw ciphertext rows; it is released
//! before any decryption runs.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use zeroize::Zeroizing;

use crate::memory::error::MemoryError;
use crate::memory::MemoryKind;
use crate::utils::crypto::CryptoVault;

// ─── Memory struct ────────────────────────────────────────────────────────────

/// One stored memory record (REQ-01, SC-01).
#[derive(Debug, Clone, PartialEq)]
pub struct Memory {
    pub id: String,
    pub session_id: String,
    pub kind: MemoryKind,
    pub text: String,
    pub embedding: Vec<f32>,
    pub model_id: String,
    pub dim: usize,
    pub created_at: i64,
    pub salience: f64,
    pub access_count: u64,
    pub last_accessed_at: i64,
    pub superseded_by: Option<String>,
    pub evicted_at: Option<i64>,
    pub scope: String,
    pub distilled_at: Option<i64>,
}

// ─── VectorStore trait ────────────────────────────────────────────────────────

#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn insert(&self, m: &Memory) -> Result<(), MemoryError>;
    async fn get(&self, id: &str) -> Result<Option<Memory>, MemoryError>;
    async fn active(&self, scope: &str) -> Result<Vec<Memory>, MemoryError>;
    async fn mark_accessed(&self, ids: &[String], now: i64) -> Result<(), MemoryError>;
    async fn set_superseded(&self, id: &str, by: &str) -> Result<(), MemoryError>;
    async fn set_evicted(&self, id: &str, at: Option<i64>) -> Result<(), MemoryError>;
    async fn hard_delete(&self, ids: &[String]) -> Result<(), MemoryError>;
    async fn set_distilled(&self, ids: &[String], at: i64) -> Result<(), MemoryError>;
}

// ─── SqliteVectorStore (stub) ─────────────────────────────────────────────────

pub struct SqliteVectorStore {
    conn: Arc<Mutex<rusqlite::Connection>>,
    #[allow(dead_code)]
    vault: CryptoVault,
    #[allow(dead_code)]
    derived_key: Zeroizing<Vec<u8>>,
}

impl SqliteVectorStore {
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
}

#[async_trait]
impl VectorStore for SqliteVectorStore {
    async fn insert(&self, _m: &Memory) -> Result<(), MemoryError> {
        Err(MemoryError::Storage("not implemented".into()))
    }
    async fn get(&self, _id: &str) -> Result<Option<Memory>, MemoryError> {
        Err(MemoryError::Storage("not implemented".into()))
    }
    async fn active(&self, _scope: &str) -> Result<Vec<Memory>, MemoryError> {
        Err(MemoryError::Storage("not implemented".into()))
    }
    async fn mark_accessed(&self, _ids: &[String], _now: i64) -> Result<(), MemoryError> {
        Err(MemoryError::Storage("not implemented".into()))
    }
    async fn set_superseded(&self, _id: &str, _by: &str) -> Result<(), MemoryError> {
        Err(MemoryError::Storage("not implemented".into()))
    }
    async fn set_evicted(&self, _id: &str, _at: Option<i64>) -> Result<(), MemoryError> {
        Err(MemoryError::Storage("not implemented".into()))
    }
    async fn hard_delete(&self, _ids: &[String]) -> Result<(), MemoryError> {
        Err(MemoryError::Storage("not implemented".into()))
    }
    async fn set_distilled(&self, _ids: &[String], _at: i64) -> Result<(), MemoryError> {
        Err(MemoryError::Storage("not implemented".into()))
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::database::EncryptedSqliteMemory;

    fn test_store() -> (tempfile::NamedTempFile, SqliteVectorStore) {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mem =
            EncryptedSqliteMemory::new(tmp.path().to_path_buf(), "pw".into()).unwrap();
        let store = SqliteVectorStore::new(mem.shared_conn(), mem.data_key()).unwrap();
        (tmp, store)
    }

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

    fn raw_blob(tmp: &tempfile::NamedTempFile, col: &str) -> String {
        let conn = rusqlite::Connection::open(tmp.path()).unwrap();
        conn.query_row(
            &format!("SELECT {col} FROM memories LIMIT 1"),
            [],
            |r| r.get::<_, String>(0),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn test_persist_and_reload_roundtrips_vector_and_metadata() {
        let (tmp, store) = test_store();
        let m = sample("m1", "remember the api budget is 8000", vec![0.1, 0.2, 0.3]);
        store.insert(&m).await.unwrap();
        let got = store.get("m1").await.unwrap().unwrap();
        assert_eq!(got, m);
        assert_eq!(got.scope, "root");
        assert!(!raw_blob(&tmp, "text_blob").contains("budget"));
        assert!(!raw_blob(&tmp, "embedding_blob").contains("0.1"));
    }

    #[tokio::test]
    async fn test_default_scope_is_root_and_active_filters_by_scope() {
        let (_t, store) = test_store();
        store.insert(&sample("a", "x", vec![0.0; 3])).await.unwrap();
        assert_eq!(store.active("root").await.unwrap().len(), 1);
        assert!(store.active("other").await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_mark_accessed_updates_only_named_ids() {
        let (_t, store) = test_store();
        store.insert(&sample("a", "a", vec![0.0; 3])).await.unwrap();
        store.insert(&sample("b", "b", vec![0.0; 3])).await.unwrap();
        store.mark_accessed(&["a".into()], 2000).await.unwrap();
        assert_eq!(store.get("a").await.unwrap().unwrap().access_count, 1);
        assert_eq!(store.get("a").await.unwrap().unwrap().last_accessed_at, 2000);
        assert_eq!(store.get("b").await.unwrap().unwrap().access_count, 0);
    }

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
}
