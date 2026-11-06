// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-11-05

//! Encrypted index of project documentation for the `code-review` context (RV07, RV19, RV21).
//!
//! Every documentation source (a path plus the key of its content: a git blob id, or a hash of
//! the working-tree bytes) is indexed once. Its chunks keep their section, text and vector
//! encrypted with the data key, exactly like [`crate::memory::store::SqliteVectorStore`]; the
//! path, the source key, the chunk hash and the index identity stay in the clear so rows can be
//! selected without decrypting anything.
//!
//! # Lock discipline
//!
//! The connection lock is never held across an `.await` and never taken while the data-key lock
//! is held (R-V08): the embedder runs first, the database after, in short synchronous calls.
//!
//! # Examples
//!
//! ```ignore
//! let store = ReviewDocStore::new(conn, dek, doc_index_id(e.model_id(), e.dim()))?;
//! let todo = store.missing(&wanted)?;
//! // ... read and chunk only `todo`, then:
//! store.reindex(&chunked, &e).await?;
//! let hits = store.search(&query_vector, 5, &allowed)?;
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use rusqlite::{params, Connection, TransactionBehavior};
use zeroize::Zeroizing;

use crate::memory::embedding::EmbeddingProvider;
use crate::memory::error::MemoryError;
use crate::memory::index::cosine;
use cryptovault::CryptoVault;
use magi_rs::review::docs::{doc_index_id, DocChunk, DocSource, WORKTREE_KEY_PREFIX};
use magi_rs::review::secrets::{display_path, find_secret};
use magi_rs::vault::MaskedDek;

/// Texts per `embed` request: thousands of chunks never go in one request (a 5 179-chunk
/// repository was measured, and it would otherwise be a single request).
pub const EMBED_BATCH_ITEMS: usize = 64;

/// `(path, key)` pairs per `search` query: two parameters each plus two fixed ones stay under
/// the 999-parameter limit of older SQLite builds.
const ALLOWED_PAIRS_PER_QUERY: usize = 400;

/// What one [`ReviewDocStore::reindex`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReindexStats {
    /// Chunks sent to the embedder (each distinct chunk hash once).
    pub embedded: usize,
    /// Chunks whose stored vector was reused instead of embedding the text again.
    pub reused: usize,
    /// Sources written (those already indexed are skipped and not counted).
    pub sources: usize,
    /// The index held rows of another identity (model, dimension or chunker) and was wiped.
    pub rebuilt: bool,
}

/// Everything `reindex` decides under the connection lock, before it embeds anything.
struct Plan<'a> {
    /// The sources to write, each once, in input order.
    new_sources: Vec<&'a (DocSource, Vec<DocChunk>)>,
    /// Foreign rows exist: the index is wiped in the write transaction.
    rebuilt: bool,
    /// Encrypted vectors already stored under this identity, by chunk hash.
    reusable: BTreeMap<String, Vec<u8>>,
    /// Distinct chunks (by hash, first appearance) that still need a vector.
    to_embed: Vec<&'a DocChunk>,
}

/// One row of `code_review_docs`, already encrypted.
struct DocRow<'a> {
    source: &'a DocSource,
    hash: &'a str,
    section_blob: Vec<u8>,
    text_blob: Vec<u8>,
    embedding_blob: Vec<u8>,
    dim: i64,
}

/// Encrypted, per-source index of documentation chunks (RV19, RV21).
///
/// Shares the connection and the masked data key with `EncryptedSqliteMemory`.
pub struct ReviewDocStore {
    conn: Arc<Mutex<Connection>>,
    vault: CryptoVault,
    /// The data key, masked in RAM; the lock is never held across the connection lock (R-V08).
    dek: Mutex<MaskedDek>,
    /// `doc_index_id(model, dim)` of the vectors this store serves.
    index_id: String,
    /// Stored vectors decrypted so far: the seam that shows a review decrypts each once.
    vector_decrypts: AtomicUsize,
}

fn db_err(e: rusqlite::Error) -> MemoryError {
    MemoryError::Storage(e.to_string())
}

impl ReviewDocStore {
    /// The shared connection, for tests that read raw rows.
    #[cfg(test)]
    pub(crate) fn connection_for_test(&self) -> &Arc<Mutex<Connection>> {
        &self.conn
    }

    /// Opens the store over `conn`, making sure the schema exists.
    ///
    /// # Arguments
    /// * `conn` - the connection shared with `EncryptedSqliteMemory`.
    /// * `dek` - a masked duplicate of the data key.
    /// * `index_id` - `doc_index_id(e.model_id(), e.dim())` of the embedder later passed to
    ///   [`Self::reindex`]: the identity of the vectors this store serves.
    ///
    /// # Errors
    /// [`MemoryError::Storage`] if the schema DDL fails.
    pub fn new(
        conn: Arc<Mutex<Connection>>,
        dek: MaskedDek,
        index_id: String,
    ) -> Result<Self, MemoryError> {
        {
            let c = conn.lock().unwrap_or_else(|p| p.into_inner());
            crate::system::database::init_schema(&c).map_err(db_err)?;
        }
        Ok(Self {
            conn,
            vault: CryptoVault::default(),
            dek: Mutex::new(dek),
            index_id,
            vector_decrypts: AtomicUsize::new(0),
        })
    }

    /// How many stored vectors this store has decrypted, for tests that count them.
    #[cfg(test)]
    pub(crate) fn vector_decrypts(&self) -> usize {
        self.vector_decrypts.load(Ordering::SeqCst)
    }

    fn locked_conn(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Encrypts `plaintext` with the data key. Never call while holding the connection lock.
    fn seal(&self, plaintext: &str) -> Result<Vec<u8>, MemoryError> {
        let mut dek = self.dek.lock().unwrap_or_else(|p| p.into_inner());
        dek.with_dek(|k| self.vault.encrypt_with_key(k, plaintext))
            .map(String::into_bytes)
            .map_err(|e| MemoryError::Crypto(e.to_string()))
    }

    /// Decrypts a blob written by [`Self::seal`]. Same lock contract.
    fn unseal(&self, blob: &[u8]) -> Result<Zeroizing<String>, MemoryError> {
        let text = std::str::from_utf8(blob)
            .map_err(|e| MemoryError::Crypto(format!("stored blob is not text: {e}")))?;
        let mut dek = self.dek.lock().unwrap_or_else(|p| p.into_inner());
        dek.with_dek(|k| self.vault.decrypt_with_key(k, text))
            .map_err(|e| MemoryError::Crypto(e.to_string()))
    }

    fn seal_vector(&self, vector: &[f32]) -> Result<Vec<u8>, MemoryError> {
        let json = serde_json::to_string(vector)
            .map_err(|e| MemoryError::Storage(format!("embedding serialization: {e}")))?;
        self.seal(&json)
    }

    fn unseal_vector(&self, blob: &[u8]) -> Result<Vec<f32>, MemoryError> {
        self.vector_decrypts.fetch_add(1, Ordering::SeqCst);
        let json = self.unseal(blob)?;
        serde_json::from_str(json.as_str())
            .map_err(|e| MemoryError::Storage(format!("embedding deserialization: {e}")))
    }

    /// Whether any stored row (source or chunk) carries another index identity.
    fn has_foreign_rows(&self, c: &Connection) -> Result<bool, MemoryError> {
        c.query_row(
            "SELECT EXISTS(SELECT 1 FROM code_review_doc_sources WHERE index_id <> ?1) \
             OR EXISTS(SELECT 1 FROM code_review_docs WHERE index_id <> ?1)",
            [&self.index_id],
            |r| r.get(0),
        )
        .map_err(db_err)
    }

    /// The sources already indexed under this identity.
    fn indexed_sources(&self, c: &Connection) -> Result<BTreeSet<DocSource>, MemoryError> {
        let mut stmt = c
            .prepare("SELECT path, key FROM code_review_doc_sources WHERE index_id = ?1")
            .map_err(db_err)?;
        let rows = stmt
            .query_map([&self.index_id], |r| {
                Ok(DocSource {
                    path: r.get(0)?,
                    key: r.get(1)?,
                })
            })
            .map_err(db_err)?;
        rows.collect::<Result<_, _>>().map_err(db_err)
    }

    /// The sources of `wanted` not indexed yet, in `wanted` order.
    ///
    /// When any stored row carries another identity (another model, dimension or chunker),
    /// every source of `wanted` is missing: the stored vectors are not comparable, and the
    /// change must be noticed even when nothing is new.
    ///
    /// Complexity: O(stored sources + wanted log stored); the stored sources are read once, so
    /// no SQL parameter limit applies.
    ///
    /// # Errors
    /// [`MemoryError::Storage`] if a query fails.
    pub fn missing(&self, wanted: &[DocSource]) -> Result<Vec<DocSource>, MemoryError> {
        let indexed = {
            let c = self.locked_conn();
            if self.has_foreign_rows(&c)? {
                return Ok(wanted.to_vec());
            }
            self.indexed_sources(&c)?
        };
        Ok(wanted
            .iter()
            .filter(|s| !indexed.contains(*s))
            .cloned()
            .collect())
    }

    /// Indexes whole sources, each with its chunks (possibly none); a source already indexed is
    /// skipped.
    ///
    /// Order of work: check the embedder's identity, refuse chunks with secrets, read what is
    /// stored (lock released), embed the rest in requests of at most [`EMBED_BATCH_ITEMS`]
    /// texts (no lock held), then write everything in one transaction. All or nothing: any
    /// failure before the write leaves the index as it was. A chunk already stored under any
    /// source reuses its vector (re-encrypted under the new row), so an unchanged chunk is
    /// never embedded twice. Rows of another identity are wiped in the same transaction
    /// (`rebuilt`); nothing else is ever deleted here (see [`Self::gc`]).
    ///
    /// Complexity: O(chunks) hash lookups, plus the embedder's cost for the new chunks.
    ///
    /// # Errors
    /// [`MemoryError::Storage`] if the embedder's identity differs from the store's, a chunk
    /// carries a secret, the embedder answers a vector of another length than it declares, or
    /// a query fails; [`MemoryError::Embedding`] if the embedder fails; [`MemoryError::Crypto`]
    /// if encryption or decryption fails.
    pub async fn reindex(
        &self,
        sources: &[(DocSource, Vec<DocChunk>)],
        embedder: &dyn EmbeddingProvider,
    ) -> Result<ReindexStats, MemoryError> {
        if embedder.dim() == 0 {
            return Err(MemoryError::Storage(
                "the embedder has no known dimension yet; an index needs one".to_string(),
            ));
        }
        let embedder_id = doc_index_id(embedder.model_id(), embedder.dim());
        if embedder_id != self.index_id {
            return Err(MemoryError::Storage(format!(
                "embedder identity `{embedder_id}` differs from the index identity `{}`",
                self.index_id
            )));
        }
        refuse_secrets(sources)?;
        let plan = self.plan(sources)?;
        let fresh = embed_all(&plan.to_embed, embedder).await?;
        let stats = ReindexStats {
            embedded: plan.to_embed.len(),
            reused: plan.reusable.len(),
            sources: plan.new_sources.len(),
            rebuilt: plan.rebuilt,
        };
        let rows = self.encode_rows(&plan, &fresh)?;
        self.write(&plan, &rows)?;
        Ok(stats)
    }

    /// Reads, under one short lock, what `reindex` needs to decide: foreign rows, the sources
    /// to write, and the stored vectors the new chunks can reuse (raw, still encrypted).
    fn plan<'a>(&self, sources: &'a [(DocSource, Vec<DocChunk>)]) -> Result<Plan<'a>, MemoryError> {
        let c = self.locked_conn();
        let rebuilt = self.has_foreign_rows(&c)?;
        let indexed = if rebuilt {
            BTreeSet::new()
        } else {
            self.indexed_sources(&c)?
        };
        let mut seen: BTreeSet<&DocSource> = BTreeSet::new();
        let mut new_sources: Vec<&'a (DocSource, Vec<DocChunk>)> = Vec::new();
        for entry in sources {
            if !indexed.contains(&entry.0) && seen.insert(&entry.0) {
                new_sources.push(entry);
            }
        }
        let mut reusable = BTreeMap::new();
        let mut to_embed: Vec<&DocChunk> = Vec::new();
        let mut known: BTreeSet<&str> = BTreeSet::new();
        let mut lookup = c
            .prepare_cached(
                "SELECT embedding_blob FROM code_review_docs \
                 WHERE chunk_hash = ?1 AND index_id = ?2 LIMIT 1",
            )
            .map_err(db_err)?;
        for chunk in new_sources.iter().flat_map(|entry| entry.1.iter()) {
            if !known.insert(chunk.hash.as_str()) {
                continue;
            }
            let stored: Option<Vec<u8>> = if rebuilt {
                None
            } else {
                optional(lookup.query_row(params![chunk.hash, self.index_id], |r| r.get(0)))?
            };
            match stored {
                Some(blob) => {
                    reusable.insert(chunk.hash.clone(), blob);
                }
                None => to_embed.push(chunk),
            }
        }
        Ok(Plan {
            new_sources,
            rebuilt,
            reusable,
            to_embed,
        })
    }

    /// Encrypts every row of the new sources (off the connection lock).
    fn encode_rows<'a>(
        &self,
        plan: &Plan<'a>,
        fresh: &[Vec<f32>],
    ) -> Result<Vec<DocRow<'a>>, MemoryError> {
        let mut vectors: BTreeMap<&str, Vec<f32>> = BTreeMap::new();
        for (chunk, vector) in plan.to_embed.iter().zip(fresh) {
            vectors.insert(chunk.hash.as_str(), vector.clone());
        }
        for (hash, blob) in &plan.reusable {
            vectors.insert(hash.as_str(), self.unseal_vector(blob)?);
        }
        let mut rows = Vec::new();
        for entry in plan.new_sources.iter().copied() {
            let (source, chunks) = (&entry.0, &entry.1);
            let mut in_source: BTreeSet<&str> = BTreeSet::new();
            for chunk in chunks.iter().filter(|c| in_source.insert(c.hash.as_str())) {
                let vector = vectors.get(chunk.hash.as_str()).ok_or_else(|| {
                    MemoryError::Storage("a chunk has no vector after embedding".to_string())
                })?;
                rows.push(DocRow {
                    source,
                    hash: &chunk.hash,
                    section_blob: self.seal(&chunk.section)?,
                    text_blob: self.seal(&chunk.text)?,
                    embedding_blob: self.seal_vector(vector)?,
                    dim: i64::try_from(vector.len()).unwrap_or(i64::MAX),
                });
            }
        }
        Ok(rows)
    }

    /// Writes the wipe (when `rebuilt`), the rows and the source rows in one transaction.
    /// Inserts are `OR IGNORE` by key: two processes indexing the same source write the same
    /// thing.
    fn write(&self, plan: &Plan<'_>, rows: &[DocRow<'_>]) -> Result<(), MemoryError> {
        let mut c = self.locked_conn();
        let tx = c
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_err)?;
        if plan.rebuilt {
            tx.execute_batch("DELETE FROM code_review_docs; DELETE FROM code_review_doc_sources;")
                .map_err(db_err)?;
        }
        {
            let mut insert_row = tx
                .prepare_cached(
                    "INSERT OR IGNORE INTO code_review_docs \
                     (key, chunk_hash, path, section_blob, text_blob, embedding_blob, dim, \
                      index_id) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                )
                .map_err(db_err)?;
            for r in rows {
                insert_row
                    .execute(params![
                        r.source.key,
                        r.hash,
                        r.source.path,
                        r.section_blob,
                        r.text_blob,
                        r.embedding_blob,
                        r.dim,
                        self.index_id
                    ])
                    .map_err(db_err)?;
            }
            let mut insert_source = tx
                .prepare_cached(
                    "INSERT OR IGNORE INTO code_review_doc_sources (path, key, index_id) \
                     VALUES (?1, ?2, ?3)",
                )
                .map_err(db_err)?;
            for (s, _) in &plan.new_sources {
                insert_source
                    .execute(params![s.path, s.key, self.index_id])
                    .map_err(db_err)?;
            }
        }
        tx.commit().map_err(db_err)
    }

    /// The `k` chunks most similar to `query` among the rows whose `(path, key)` is in
    /// `allowed` and whose identity is this store's, best first.
    ///
    /// Rows of another dimension are skipped (vectors of two models are never compared). Only
    /// the vectors of allowed rows are decrypted, and the section and text only for the `k`
    /// returned. Rows are read raw under the lock and decrypted after it is released.
    ///
    /// Complexity: O(rows * d) to score plus O(r log r) to order the `r` allowed rows; one call
    /// per batch.
    ///
    /// # Errors
    /// [`MemoryError::Storage`] if a query fails; [`MemoryError::Crypto`] if decryption fails.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        allowed: &BTreeSet<DocSource>,
    ) -> Result<Vec<(DocChunk, f32)>, MemoryError> {
        if k == 0 || query.is_empty() {
            return Ok(Vec::new());
        }
        let candidates = self.allowed_vectors(query.len(), allowed)?;
        let mut scored = Vec::with_capacity(candidates.len());
        for (key, hash, blob) in candidates {
            let vector = self.unseal_vector(&blob)?;
            if vector.len() == query.len() {
                scored.push((cosine(query, &vector), key, hash));
            }
        }
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        scored.truncate(k);
        scored
            .into_iter()
            .map(|(score, key, hash)| -> Result<_, MemoryError> {
                Ok((self.load_chunk(&key, hash)?, score))
            })
            .collect()
    }

    /// `(key, chunk hash, encrypted vector)` of the allowed rows of this identity and `dim`.
    ///
    /// The `(path, key)` filter runs in SQL, in batches of [`ALLOWED_PAIRS_PER_QUERY`] pairs
    /// (so no query passes SQLite's 999-parameter limit), so the blobs of other revisions are
    /// never copied under the connection lock.
    fn allowed_vectors(
        &self,
        dim: usize,
        allowed: &BTreeSet<DocSource>,
    ) -> Result<Vec<(String, String, Vec<u8>)>, MemoryError> {
        let dim = i64::try_from(dim).unwrap_or(i64::MAX);
        let pairs: Vec<&DocSource> = allowed.iter().collect();
        let c = self.locked_conn();
        let mut out = Vec::new();
        for batch in pairs.chunks(ALLOWED_PAIRS_PER_QUERY) {
            let values = vec!["(?, ?)"; batch.len()].join(", ");
            let sql = format!(
                "SELECT key, chunk_hash, embedding_blob FROM code_review_docs \
                 WHERE index_id = ? AND dim = ? AND (path, key) IN (VALUES {values})"
            );
            let mut args: Vec<&dyn rusqlite::ToSql> = vec![&self.index_id, &dim];
            for s in batch {
                args.push(&s.path);
                args.push(&s.key);
            }
            let mut stmt = c.prepare(&sql).map_err(db_err)?;
            let rows = stmt
                .query_map(rusqlite::params_from_iter(args), |r| {
                    Ok((r.get(0)?, r.get(1)?, r.get(2)?))
                })
                .map_err(db_err)?;
            for row in rows {
                out.push(row.map_err(db_err)?);
            }
        }
        Ok(out)
    }
    /// Reads one chunk's row raw under the lock, then decrypts it off the lock.
    fn load_chunk(&self, key: &str, hash: String) -> Result<DocChunk, MemoryError> {
        let (path, section, text): (String, Vec<u8>, Vec<u8>) = self
            .locked_conn()
            .query_row(
                "SELECT path, section_blob, text_blob FROM code_review_docs \
                 WHERE key = ?1 AND chunk_hash = ?2",
                params![key, hash],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .map_err(db_err)?;
        Ok(DocChunk {
            path,
            section: self.unseal(&section)?.to_string(),
            text: self.unseal(&text)?.to_string(),
            hash,
        })
    }

    /// Deletes (`DELETE FROM`, both tables, one transaction) every source not in `live`; with
    /// `keep_worktree_keys`, a source whose key is a working-tree key stays. Returns how many
    /// sources it deleted.
    ///
    /// Caller contract: `live` holds every source a local ref reaches, plus the current
    /// working-tree set when the review listed it (then `keep_worktree_keys` is `false`, so
    /// older working-tree versions go); a review that did not list the working tree passes
    /// `true` and leaves those sources for the next one that does. Never `DROP TABLE`: a
    /// dropped table would make the next open read this database as one from an older version.
    ///
    /// Complexity: O(stored sources) with one pair of deletes per doomed source; `live` is
    /// only probed, so no SQL parameter limit applies.
    ///
    /// # Errors
    /// [`MemoryError::Storage`] if a statement fails (nothing is deleted then).
    pub fn gc(
        &self,
        live: &BTreeSet<DocSource>,
        keep_worktree_keys: bool,
    ) -> Result<usize, MemoryError> {
        let mut c = self.locked_conn();
        let tx = c
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_err)?;
        let stored: Vec<DocSource> = {
            let mut stmt = tx
                .prepare("SELECT path, key FROM code_review_doc_sources")
                .map_err(db_err)?;
            let rows = stmt
                .query_map([], |r| {
                    Ok(DocSource {
                        path: r.get(0)?,
                        key: r.get(1)?,
                    })
                })
                .map_err(db_err)?;
            rows.collect::<Result<_, _>>().map_err(db_err)?
        };
        let doomed: Vec<&DocSource> = stored
            .iter()
            .filter(|s| {
                !live.contains(*s)
                    && !(keep_worktree_keys && s.key.starts_with(WORKTREE_KEY_PREFIX))
            })
            .collect();
        for s in &doomed {
            tx.execute(
                "DELETE FROM code_review_docs WHERE path = ?1 AND key = ?2",
                params![s.path, s.key],
            )
            .map_err(db_err)?;
            tx.execute(
                "DELETE FROM code_review_doc_sources WHERE path = ?1 AND key = ?2",
                params![s.path, s.key],
            )
            .map_err(db_err)?;
        }
        tx.commit().map_err(db_err)?;
        Ok(doomed.len())
    }
}

/// `Ok(None)` for "no row", the error otherwise.
fn optional<T>(r: rusqlite::Result<T>) -> Result<Option<T>, MemoryError> {
    match r {
        Ok(v) => Ok(Some(v)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(db_err(e)),
    }
}

/// Defense in depth (RV22): a chunk whose section or text matches a secret pattern is refused
/// before anything is embedded. The message names the path, never the secret.
fn refuse_secrets(sources: &[(DocSource, Vec<DocChunk>)]) -> Result<(), MemoryError> {
    for chunk in sources.iter().flat_map(|entry| entry.1.iter()) {
        if find_secret(&chunk.text).is_some() || find_secret(&chunk.section).is_some() {
            return Err(MemoryError::Storage(format!(
                "documentation chunk of `{}` matches a secret pattern and was not embedded",
                display_path(&chunk.path)
            )));
        }
    }
    Ok(())
}

/// Embeds `chunks` (document prefix applied) in requests of at most [`EMBED_BATCH_ITEMS`]
/// texts, returning one vector per chunk, in order. No lock is held by this function.
///
/// # Errors
/// [`MemoryError::Embedding`] if the embedder fails; [`MemoryError::Storage`] if it answers
/// a different count than asked, or a vector of another length than `embedder.dim()`.
async fn embed_all(
    chunks: &[&DocChunk],
    embedder: &dyn EmbeddingProvider,
) -> Result<Vec<Vec<f32>>, MemoryError> {
    let dim = embedder.dim();
    let prefix = embedder.document_prefix();
    let mut out = Vec::with_capacity(chunks.len());
    for batch in chunks.chunks(EMBED_BATCH_ITEMS) {
        let texts: Vec<String> = batch
            .iter()
            .map(|c| format!("{prefix}{}", c.text))
            .collect();
        let vectors = embedder.embed(&texts).await?;
        if vectors.len() != batch.len() || vectors.iter().any(|v| v.len() != dim) {
            return Err(MemoryError::Storage(format!(
                "the embedder answered {} vectors for {} texts, or a vector not of the \
                 declared dimension {dim}",
                vectors.len(),
                batch.len()
            )));
        }
        out.extend(vectors);
    }
    Ok(out)
}

#[cfg(test)]
pub(crate) mod test_embedder {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{DocChunk, DocSource, EmbeddingProvider};
    use crate::memory::error::EmbeddingError;

    /// Counts embedded texts; a known text gets its fixed vector, anything else `[0.5; dim]`.
    /// `largest` = texts in the biggest single request. Model "fake" by default, like
    /// `store_for_test`'s index id: every fixture embedder passes the index-id gate unless a
    /// test means to fail it.
    pub(crate) struct FakeEmbedder {
        model: String,
        dim: usize,
        fixed: BTreeMap<String, Vec<f32>>,
        calls: AtomicUsize,
        largest: AtomicUsize,
    }

    impl FakeEmbedder {
        pub(crate) fn new(dim: usize) -> Self {
            Self {
                model: "fake".into(),
                dim,
                fixed: BTreeMap::new(),
                calls: AtomicUsize::new(0),
                largest: AtomicUsize::new(0),
            }
        }

        /// `dim` = the length of the first vector (the tests pass 3, the store's).
        pub(crate) fn with_vectors(v: &[(&str, Vec<f32>)]) -> Self {
            Self {
                fixed: v.iter().map(|(t, x)| (t.to_string(), x.clone())).collect(),
                ..Self::new(v.first().map_or(3, |(_, x)| x.len()))
            }
        }

        /// Another model name, for a store opened under `doc_index_id(model, dim)`.
        pub(crate) fn named(self, model: &str) -> Self {
            Self {
                model: model.into(),
                ..self
            }
        }

        pub(crate) fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }

        pub(crate) fn largest_request(&self) -> usize {
            self.largest.load(Ordering::SeqCst)
        }
    }

    #[async_trait::async_trait]
    impl EmbeddingProvider for FakeEmbedder {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            self.calls.fetch_add(texts.len(), Ordering::SeqCst);
            self.largest.fetch_max(texts.len(), Ordering::SeqCst);
            Ok(texts
                .iter()
                .map(|t| {
                    self.fixed
                        .get(t)
                        .cloned()
                        .unwrap_or_else(|| vec![0.5; self.dim])
                })
                .collect())
        }
        fn model_id(&self) -> &str {
            &self.model
        }
        fn dim(&self) -> usize {
            self.dim
        }
        fn query_prefix(&self) -> &str {
            ""
        }
        fn document_prefix(&self) -> &str {
            ""
        }
    }

    /// An embedder that always fails. Model "fake", dim 3 (the fixture stores'): it passes the
    /// index-id gate and fails in `embed()`, the path its tests guard.
    pub(crate) struct FailingEmbedder;

    #[async_trait::async_trait]
    impl EmbeddingProvider for FailingEmbedder {
        async fn embed(&self, _t: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            Err(EmbeddingError::Http("down".into()))
        }
        fn model_id(&self) -> &str {
            "fake"
        }
        fn dim(&self) -> usize {
            3
        }
        fn query_prefix(&self) -> &str {
            ""
        }
        fn document_prefix(&self) -> &str {
            ""
        }
    }

    /// A chunk of `path` with the hash `chunk_markdown` gives its section.
    pub(crate) fn chunk(path: &str, section: &str, text: &str) -> DocChunk {
        let hash = magi_rs::review::docs::chunk_markdown(path, &format!("# {section}\n{text}\n"))
            .into_iter()
            .next()
            .expect("one chunk")
            .hash;
        DocChunk {
            path: path.into(),
            section: section.into(),
            text: text.into(),
            hash,
        }
    }

    /// A documentation source: a path and the key of its content.
    pub(crate) fn src(path: &str, key: &str) -> DocSource {
        DocSource {
            path: path.into(),
            key: key.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_embedder::*;
    use super::*;
    use crate::memory::error::EmbeddingError;

    fn store_for_test() -> (ReviewDocStore, tempfile::NamedTempFile) {
        let tmp = tempfile::NamedTempFile::new().expect("tmp");
        let mem = crate::system::database::EncryptedSqliteMemory::new(
            tmp.path().to_path_buf(),
            zeroize::Zeroizing::new(crate::system::database::TEST_MASTER_PASSPHRASE.to_string()),
        )
        .expect("db");
        (
            ReviewDocStore::new(
                mem.shared_conn(),
                mem.data_key().expect("dek"),
                magi_rs::review::docs::doc_index_id("fake", 3),
            )
            .expect("store"),
            tmp,
        )
    }

    /// The `dim` column of every row.
    fn dims(store: &ReviewDocStore) -> Vec<i64> {
        store
            .connection_for_test()
            .lock()
            .expect("lock")
            .prepare("SELECT dim FROM code_review_docs ORDER BY key, chunk_hash")
            .expect("prep")
            .query_map([], |r| r.get(0))
            .expect("q")
            .collect::<Result<_, _>>()
            .expect("rows")
    }

    #[tokio::test]
    async fn an_indexed_source_is_never_read_or_embedded_again() {
        // `missing` is what the caller reads; a second reindex of the same source does nothing.
        let (store, _dir) = store_for_test();
        let e = FakeEmbedder::new(3);
        let wanted = [src("README.md", "k1"), src("docs/b.md", "k2")];
        assert_eq!(store.missing(&wanted).expect("missing").len(), 2);
        let s = store
            .reindex(
                &[(src("README.md", "k1"), vec![chunk("README.md", "A", "x")])],
                &e,
            )
            .await
            .expect("first");
        assert_eq!((s.embedded, s.sources), (1, 1));
        assert_eq!(
            store.missing(&wanted).expect("missing"),
            vec![src("docs/b.md", "k2")]
        );
        let again = store
            .reindex(
                &[(src("README.md", "k1"), vec![chunk("README.md", "A", "x")])],
                &e,
            )
            .await
            .expect("again");
        assert_eq!((again.embedded, again.sources, e.calls()), (0, 0, 1));
    }

    #[tokio::test]
    async fn a_source_without_doc_chunks_is_remembered_as_indexed() {
        // A file with no doc comment would otherwise be read again on every review.
        let (store, _dir) = store_for_test();
        let e = FakeEmbedder::new(3);
        store
            .reindex(&[(src("src/m.rs", "k3"), vec![])], &e)
            .await
            .expect("indexed");
        assert!(store
            .missing(&[src("src/m.rs", "k3")])
            .expect("missing")
            .is_empty());
        assert_eq!(e.calls(), 0);
    }

    #[tokio::test]
    async fn a_chunk_shared_by_two_revisions_is_embedded_once() {
        // RV-S37: the second README differs in one section only.
        let (store, _dir) = store_for_test();
        let e = FakeEmbedder::new(3);
        store
            .reindex(
                &[(
                    src("README.md", "k1"),
                    vec![chunk("README.md", "A", "x"), chunk("README.md", "B", "y")],
                )],
                &e,
            )
            .await
            .expect("first");
        let s = store
            .reindex(
                &[(
                    src("README.md", "k2"),
                    vec![chunk("README.md", "A", "x"), chunk("README.md", "B", "y2")],
                )],
                &e,
            )
            .await
            .expect("second");
        assert_eq!((s.embedded, s.reused), (1, 1));
        assert_eq!(e.calls(), 3);
        // The reused row is served under the second source (rows carry their source; the chunk
        // hash includes the path, so two paths never share a row).
        let hits = store
            .search(
                &[0.5, 0.5, 0.5],
                5,
                &BTreeSet::from([src("README.md", "k2")]),
            )
            .expect("search");
        assert_eq!(hits.len(), 2, "{hits:?}");
    }

    #[tokio::test]
    async fn search_only_considers_the_allowed_sources() {
        // RV-S36: the other revision's chunk is the more similar one, and it is never returned.
        let (store, _dir) = store_for_test();
        // Three dimensions, the store's: a 2-dim embedder was refused by the index-id gate.
        let e = FakeEmbedder::with_vectors(&[
            ("old docs", vec![1.0, 0.0, 0.0]),
            ("new docs", vec![0.0, 1.0, 0.0]),
        ]);
        store
            .reindex(
                &[
                    (
                        src("README.md", "old"),
                        vec![chunk("README.md", "U", "old docs")],
                    ),
                    (
                        src("README.md", "new"),
                        vec![chunk("README.md", "U", "new docs")],
                    ),
                ],
                &e,
            )
            .await
            .expect("ok");
        let hits = store
            .search(
                &[1.0, 0.0, 0.0],
                5,
                &BTreeSet::from([src("README.md", "new")]),
            )
            .expect("search");
        assert_eq!(
            hits.iter()
                .map(|(c, _)| c.text.as_str())
                .collect::<Vec<_>>(),
            vec!["new docs"]
        );
    }

    #[tokio::test]
    async fn search_returns_the_most_similar_chunk_first() {
        let (store, _dir) = store_for_test();
        let e = FakeEmbedder::with_vectors(&[
            ("about dogs", vec![1.0, 0.0, 0.0]),
            ("about cats", vec![0.0, 1.0, 0.0]),
        ]);
        store
            .reindex(
                &[(
                    src("d.md", "k"),
                    vec![
                        chunk("d.md", "dogs", "about dogs"),
                        chunk("d.md", "cats", "about cats"),
                    ],
                )],
                &e,
            )
            .await
            .expect("ok");
        let hits = store
            .search(&[0.9, 0.1, 0.0], 1, &BTreeSet::from([src("d.md", "k")]))
            .expect("search");
        assert_eq!(hits[0].0.section, "dogs");
    }

    #[tokio::test]
    async fn a_dimension_change_forces_a_reindex() {
        // The dimension is part of the index id (T3.2): `missing` sees the change before anything
        // is read, so a review indexes every source it wants instead of serving a subset.
        let tmp = tempfile::NamedTempFile::new().expect("tmp");
        let mem = crate::system::database::EncryptedSqliteMemory::new(
            tmp.path().to_path_buf(),
            zeroize::Zeroizing::new(crate::system::database::TEST_MASTER_PASSPHRASE.to_string()),
        )
        .expect("db");
        use magi_rs::review::docs::doc_index_id;
        let d3 = ReviewDocStore::new(
            mem.shared_conn(),
            mem.data_key().expect("dek"),
            doc_index_id("fake", 3),
        )
        .expect("d3");
        d3.reindex(
            &[(src("d.md", "k1"), vec![chunk("d.md", "A", "x")])],
            &FakeEmbedder::new(3),
        )
        .await
        .expect("dim 3");
        let d5 = ReviewDocStore::new(
            mem.shared_conn(),
            mem.data_key().expect("dek"),
            doc_index_id("fake", 5),
        )
        .expect("d5");
        assert_eq!(
            d5.missing(&[src("d.md", "k1")]).expect("missing"),
            vec![src("d.md", "k1")]
        );
        // The same chunk under another source: its stored vector has the old dimension, so it is
        // embedded again instead of reused, and the old source is gone with its rows.
        let s = d5
            .reindex(
                &[(
                    src("d.md", "k2"),
                    vec![chunk("d.md", "A", "x"), chunk("d.md", "B", "y")],
                )],
                &FakeEmbedder::new(5),
            )
            .await
            .expect("dim 5");
        assert_eq!((s.embedded, s.reused, s.rebuilt), (2, 0, true));
        assert_eq!(dims(&d5), vec![5, 5]);
    }

    #[tokio::test]
    async fn another_embedding_model_rebuilds_the_index_even_with_nothing_new() {
        // Same dimension, another model: the old vectors are not comparable. Every source is
        // missing again, so the next review re-indexes instead of serving no documentation,
        // review after review, in silence.
        let tmp = tempfile::NamedTempFile::new().expect("tmp");
        let mem = crate::system::database::EncryptedSqliteMemory::new(
            tmp.path().to_path_buf(),
            zeroize::Zeroizing::new(crate::system::database::TEST_MASTER_PASSPHRASE.to_string()),
        )
        .expect("db");
        use magi_rs::review::docs::doc_index_id;
        let old = ReviewDocStore::new(
            mem.shared_conn(),
            mem.data_key().expect("dek"),
            doc_index_id("m1", 3),
        )
        .expect("old");
        // Each store gets its own model's embedder (the gate refuses any other).
        old.reindex(
            &[(src("d.md", "k"), vec![chunk("d.md", "A", "x")])],
            &FakeEmbedder::new(3).named("m1"),
        )
        .await
        .expect("m1");
        let new = ReviewDocStore::new(
            mem.shared_conn(),
            mem.data_key().expect("dek"),
            doc_index_id("m2", 3),
        )
        .expect("new");
        let wanted = BTreeSet::from([src("d.md", "k")]);
        assert!(
            new.search(&[0.5, 0.5, 0.5], 5, &wanted)
                .expect("search")
                .is_empty(),
            "never compared across models"
        );
        assert_eq!(
            new.missing(&[src("d.md", "k")]).expect("missing"),
            vec![src("d.md", "k")]
        );
        let s = new
            .reindex(
                &[(src("d.md", "k"), vec![chunk("d.md", "A", "x")])],
                &FakeEmbedder::new(3).named("m2"),
            )
            .await
            .expect("m2");
        assert_eq!((s.embedded, s.reused, s.rebuilt), (1, 0, true));
        assert_eq!(
            new.search(&[0.5, 0.5, 0.5], 5, &wanted)
                .expect("search")
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn embedding_requests_are_bounded_and_written_once() {
        // Thousands of chunks never go in one request; all requests run before the one write.
        let (store, _dir) = store_for_test();
        let e = FakeEmbedder::new(3);
        let n = EMBED_BATCH_ITEMS * 2 + 1;
        let many: Vec<DocChunk> = (0..n)
            .map(|i| chunk("d.md", &format!("S{i}"), &format!("t{i}")))
            .collect();
        let s = store
            .reindex(&[(src("d.md", "k"), many)], &e)
            .await
            .expect("ok");
        assert_eq!((s.embedded, e.calls()), (n, n));
        assert_eq!(e.largest_request(), EMBED_BATCH_ITEMS);
        assert_eq!(dims(&store).len(), n);
    }

    #[tokio::test]
    async fn a_reindex_whose_embedder_fails_leaves_the_previous_index_whole() {
        // Embed first, write once.
        let (store, _dir) = store_for_test();
        store
            .reindex(
                &[(
                    src("d.md", "k1"),
                    vec![chunk("d.md", "A", "x"), chunk("d.md", "B", "y")],
                )],
                &FakeEmbedder::new(3),
            )
            .await
            .expect("dim 3");
        assert!(store
            .reindex(
                &[(src("d.md", "k2"), vec![chunk("d.md", "A", "x2")])],
                &FailingEmbedder
            )
            .await
            .is_err());
        assert_eq!(
            dims(&store),
            vec![3, 3],
            "the index before the failed reindex, whole"
        );
        assert_eq!(
            store.missing(&[src("d.md", "k2")]).expect("missing"),
            vec![src("d.md", "k2")],
            "nothing recorded"
        );
    }

    #[tokio::test]
    async fn search_accepts_more_allowed_sources_than_sqlite_parameters() {
        // `allowed` never goes as one `IN (...)` (999 parameters in older SQLite builds).
        let (store, _dir) = store_for_test();
        let all: Vec<(DocSource, Vec<DocChunk>)> = (0..1_200)
            .map(|i| {
                (
                    src(&format!("docs/{i}.md"), "k"),
                    vec![chunk(&format!("docs/{i}.md"), "S", &format!("t{i}"))],
                )
            })
            .collect();
        store
            .reindex(&all, &FakeEmbedder::new(3))
            .await
            .expect("index");
        let allowed: BTreeSet<DocSource> = all.iter().map(|(s, _)| s.clone()).collect();
        assert_eq!(
            store
                .search(&[0.5, 0.5, 0.5], 3, &allowed)
                .expect("search")
                .len(),
            3
        );
    }

    #[tokio::test]
    async fn the_clean_up_accepts_more_live_sources_than_sqlite_parameters() {
        // gc's `live` set never goes as one `IN (...)` either. Chunkless sources: no embed.
        let (store, _dir) = store_for_test();
        let live: Vec<(DocSource, Vec<DocChunk>)> = (0..1_200)
            .map(|i| (src(&format!("src/{i}.rs"), "k"), vec![]))
            .collect();
        let mut all = live.clone();
        all.push((src("src/gone.rs", "k"), vec![]));
        store
            .reindex(&all, &FakeEmbedder::new(3))
            .await
            .expect("index");
        let live_set: BTreeSet<DocSource> = live.iter().map(|(s, _)| s.clone()).collect();
        assert_eq!(
            store.gc(&live_set, false).expect("gc"),
            1,
            "only the source no ref reaches"
        );
        assert!(store
            .missing(&live.iter().map(|(s, _)| s.clone()).collect::<Vec<_>>())
            .expect("missing")
            .is_empty());
    }

    #[tokio::test]
    async fn a_source_without_chunks_is_asked_again_after_an_index_change() {
        // Its only trace is its source row: that row carries the index identity too.
        let tmp = tempfile::NamedTempFile::new().expect("tmp");
        let mem = crate::system::database::EncryptedSqliteMemory::new(
            tmp.path().to_path_buf(),
            zeroize::Zeroizing::new(crate::system::database::TEST_MASTER_PASSPHRASE.to_string()),
        )
        .expect("db");
        use magi_rs::review::docs::doc_index_id;
        let old = ReviewDocStore::new(
            mem.shared_conn(),
            mem.data_key().expect("dek"),
            doc_index_id("m1", 3),
        )
        .expect("old");
        old.reindex(
            &[(src("src/m.rs", "k"), vec![])],
            &FakeEmbedder::new(3).named("m1"),
        )
        .await
        .expect("m1");
        let new = ReviewDocStore::new(
            mem.shared_conn(),
            mem.data_key().expect("dek"),
            doc_index_id("m2", 3),
        )
        .expect("new");
        assert_eq!(
            new.missing(&[src("src/m.rs", "k")]).expect("missing"),
            vec![src("src/m.rs", "k")]
        );
    }

    #[tokio::test]
    async fn search_skips_rows_of_another_dimension() {
        // A query of another dimension skips every row, never compares them.
        let (store, _dir) = store_for_test();
        store
            .reindex(
                &[(src("d.md", "k"), vec![chunk("d.md", "A", "x")])],
                &FakeEmbedder::new(3),
            )
            .await
            .expect("dim 3");
        assert!(store
            .search(&[1.0, 0.0], 5, &BTreeSet::from([src("d.md", "k")]))
            .expect("search")
            .is_empty());
    }

    #[tokio::test]
    async fn the_clean_up_deletes_every_source_that_is_not_live() {
        // RV-S39: rows and source go together.
        let (store, _dir) = store_for_test();
        let e = FakeEmbedder::new(3);
        let wt = src("README.md", &magi_rs::review::docs::worktree_key(b"w"));
        store
            .reindex(
                &[
                    (src("README.md", "k1"), vec![chunk("README.md", "A", "x")]),
                    (src("README.md", "k2"), vec![chunk("README.md", "A", "y")]),
                    (wt.clone(), vec![chunk("README.md", "A", "z")]),
                ],
                &e,
            )
            .await
            .expect("index");
        assert_eq!(
            store
                .gc(&BTreeSet::from([src("README.md", "k2")]), false)
                .expect("gc"),
            2
        );
        assert_eq!(
            store
                .missing(&[src("README.md", "k1"), src("README.md", "k2"), wt.clone()])
                .expect("missing"),
            vec![src("README.md", "k1"), wt]
        );
        assert_eq!(dims(&store).len(), 1, "only k2's row is left");
    }

    #[tokio::test]
    async fn after_a_revision_review_the_clean_up_keeps_working_tree_sources() {
        // A revision review never listed the working tree, so it cannot judge those sources.
        let (store, _dir) = store_for_test();
        let e = FakeEmbedder::new(3);
        let wt = src("README.md", &magi_rs::review::docs::worktree_key(b"w"));
        store
            .reindex(
                &[
                    (src("README.md", "k1"), vec![chunk("README.md", "A", "x")]),
                    (wt.clone(), vec![]),
                ],
                &e,
            )
            .await
            .expect("index");
        assert_eq!(store.gc(&BTreeSet::new(), true).expect("gc"), 1);
        assert!(store.missing(&[wt]).expect("missing").is_empty());
    }

    #[tokio::test]
    async fn the_text_is_stored_encrypted() {
        let (store, _tmp) = store_for_test();
        store
            .reindex(
                &[(
                    src("d.md", "k"),
                    vec![chunk("d.md", "SECTION-MARKER-PT", "PLAINTEXT-MARKER")],
                )],
                &FakeEmbedder::new(3),
            )
            .await
            .expect("ok");
        let (text, section, vector): (Vec<u8>, Vec<u8>, Vec<u8>) = store
            .connection_for_test()
            .lock()
            .expect("lock")
            .query_row(
                "SELECT text_blob, section_blob, embedding_blob FROM code_review_docs",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .expect("row");
        assert!(
            !text.windows(16).any(|w| w == b"PLAINTEXT-MARKER"),
            "text stored in the clear"
        );
        // The section and the vector are encrypted too (FakeEmbedder answers 0.5 per dimension).
        assert!(
            !section.windows(17).any(|w| w == b"SECTION-MARKER-PT"),
            "section stored in the clear"
        );
        let clear: Vec<u8> = [0.5f32; 3].iter().flat_map(|f| f.to_le_bytes()).collect();
        assert!(
            !vector.windows(clear.len()).any(|w| w == clear.as_slice()),
            "embedding stored in the clear"
        );
        // The vector is serialized as JSON text before it is sealed: that text must not show
        // either (a regression that skipped the seal would leave exactly this).
        assert!(
            !vector.windows(11).any(|w| w == b"0.5,0.5,0.5"),
            "embedding JSON stored in the clear"
        );
        assert!(
            !vector.windows(4).any(|w| w == b"[0.5"),
            "embedding JSON stored in the clear"
        );
    }

    #[tokio::test]
    async fn a_secret_in_a_section_name_alone_is_refused_before_anything_is_embedded() {
        // refuse_secrets checks the section as well as the text; here only the section matches.
        let (store, _dir) = store_for_test();
        let e = FakeEmbedder::new(3);
        let section = "KEY=sk-ant-api03-AAAAAAAAAAAAAAAAAAAA"; // allow-secret-scan
        let r = store
            .reindex(
                &[(
                    src("leak.md", "k"),
                    vec![chunk("leak.md", section, "harmless text")],
                )],
                &e,
            )
            .await;
        assert!(
            matches!(&r, Err(MemoryError::Storage(m)) if m.contains("leak.md") && !m.contains("sk-ant-")),
            "{:?}",
            r.as_ref().err()
        );
        assert_eq!(e.calls(), 0);
    }

    #[tokio::test]
    async fn an_embedder_without_a_known_dimension_is_refused_even_with_nothing_to_embed() {
        // dim() == 0 (autodetect, nothing answered yet) would record sources under `#0#`.
        let tmp = tempfile::NamedTempFile::new().expect("tmp");
        let mem = crate::system::database::EncryptedSqliteMemory::new(
            tmp.path().to_path_buf(),
            zeroize::Zeroizing::new(crate::system::database::TEST_MASTER_PASSPHRASE.to_string()),
        )
        .expect("db");
        let store = ReviewDocStore::new(
            mem.shared_conn(),
            mem.data_key().expect("dek"),
            magi_rs::review::docs::doc_index_id("fake", 0),
        )
        .expect("store");
        let r = store
            .reindex(&[(src("src/m.rs", "k"), vec![])], &FakeEmbedder::new(0))
            .await;
        assert!(
            matches!(&r, Err(MemoryError::Storage(m)) if m.contains("dimension")),
            "{r:?}"
        );
        assert!(
            store
                .missing(&[src("src/m.rs", "k")])
                .expect("missing")
                .len()
                == 1
        );
    }

    #[tokio::test]
    async fn search_with_no_allowed_sources_returns_nothing() {
        let (store, _dir) = store_for_test();
        store
            .reindex(
                &[(src("d.md", "k"), vec![chunk("d.md", "A", "x")])],
                &FakeEmbedder::new(3),
            )
            .await
            .expect("index");
        assert!(store
            .search(&[0.5, 0.5, 0.5], 5, &BTreeSet::new())
            .expect("search")
            .is_empty());
    }

    #[tokio::test]
    async fn two_identical_chunks_in_one_reindex_are_stored_once() {
        // Repeated boilerplate sections hash alike, and (key, chunk_hash) is the primary key.
        let (store, _dir) = store_for_test();
        let e = FakeEmbedder::new(3);
        let s = store
            .reindex(
                &[(
                    src("d.md", "k"),
                    vec![chunk("d.md", "A", "x"), chunk("d.md", "A", "x")],
                )],
                &e,
            )
            .await
            .expect("not an error");
        assert_eq!((s.embedded, e.calls()), (1, 1));
    }

    #[tokio::test]
    async fn an_embedder_answering_mixed_dimensions_fails_the_reindex_whole() {
        // One stored dimension: a response mixing lengths writes nothing.
        let (store, _dir) = store_for_test();
        store
            .reindex(
                &[(src("d.md", "k1"), vec![chunk("d.md", "A", "x")])],
                &FakeEmbedder::new(3),
            )
            .await
            .expect("dim 3");
        // Declares 3 (first vector), the store's, so it passes the index-id gate; "z" is 2 long.
        let mixed =
            FakeEmbedder::with_vectors(&[("y", vec![1.0, 0.0, 0.0]), ("z", vec![1.0, 0.0])]);
        let all = vec![chunk("d.md", "B", "y"), chunk("d.md", "C", "z")];
        assert!(store
            .reindex(&[(src("d.md", "k2"), all)], &mixed)
            .await
            .is_err());
        assert_eq!(
            dims(&store),
            vec![3],
            "the index before the failed reindex, whole"
        );
    }

    /// Declares one dimension and answers another.
    struct MislabeledEmbedder;
    #[async_trait::async_trait]
    impl EmbeddingProvider for MislabeledEmbedder {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            Ok(texts.iter().map(|_| vec![0.5; 5]).collect())
        }
        fn model_id(&self) -> &str {
            "fake"
        }
        fn dim(&self) -> usize {
            3
        }
        fn query_prefix(&self) -> &str {
            ""
        }
        fn document_prefix(&self) -> &str {
            ""
        }
    }

    #[tokio::test]
    async fn an_embedder_answering_another_length_than_it_declares_fails_the_reindex() {
        // The index id carries dim(): a vector of another length would mislabel the whole index.
        let (store, _dir) = store_for_test();
        store
            .reindex(
                &[(src("d.md", "k1"), vec![chunk("d.md", "A", "x")])],
                &FakeEmbedder::new(3),
            )
            .await
            .expect("dim 3");
        assert!(store
            .reindex(
                &[(src("d.md", "k2"), vec![chunk("d.md", "B", "y")])],
                &MislabeledEmbedder
            )
            .await
            .is_err());
        assert_eq!(dims(&store), vec![3], "nothing written");
    }

    /// Another model with the same dimension as the store's.
    struct OtherModelEmbedder;
    #[async_trait::async_trait]
    impl EmbeddingProvider for OtherModelEmbedder {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            Ok(texts.iter().map(|_| vec![0.5; 3]).collect())
        }
        fn model_id(&self) -> &str {
            "other"
        }
        fn dim(&self) -> usize {
            3
        }
        fn query_prefix(&self) -> &str {
            ""
        }
        fn document_prefix(&self) -> &str {
            ""
        }
    }

    #[tokio::test]
    async fn an_embedder_of_another_model_is_refused_even_with_the_same_dimension() {
        // The store's index_id names a model: a same-length vector of another model never lands
        // under it.
        let (store, _dir) = store_for_test();
        assert!(store
            .reindex(
                &[(src("d.md", "k1"), vec![chunk("d.md", "A", "x")])],
                &OtherModelEmbedder
            )
            .await
            .is_err());
        assert!(dims(&store).is_empty(), "nothing written");
    }

    #[tokio::test]
    async fn a_chunk_with_a_secret_is_refused_before_anything_is_embedded() {
        // Defense in depth (RV22): a caller that skipped without_secrets fails here, before
        // embed().
        let (store, _dir) = store_for_test();
        let e = FakeEmbedder::new(3);
        let text = "export KEY=sk-ant-api03-AAAAAAAAAAAAAAAAAAAA"; // allow-secret-scan
        let r = store
            .reindex(
                &[
                    (src("d.md", "k1"), vec![chunk("d.md", "A", "x")]),
                    (src("leak.md", "k2"), vec![chunk("leak.md", "Setup", text)]),
                ],
                &e,
            )
            .await;
        assert!(
            matches!(&r, Err(MemoryError::Storage(m)) if m.contains("leak.md") && !m.contains("sk-ant-")),
            "{:?}",
            r.as_ref().err()
        );
        assert_eq!(e.calls(), 0);
    }

    /// Embeds like FakeEmbedder and records whether the DB lock was taken while it ran.
    struct LockProbe {
        conn: Arc<Mutex<rusqlite::Connection>>,
        calls: std::sync::atomic::AtomicUsize,
        held: std::sync::atomic::AtomicBool,
    }
    #[async_trait::async_trait]
    impl EmbeddingProvider for LockProbe {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if self.conn.try_lock().is_err() {
                self.held.store(true, std::sync::atomic::Ordering::SeqCst);
            }
            Ok(texts.iter().map(|_| vec![0.5; 3]).collect())
        }
        fn model_id(&self) -> &str {
            "fake" // passes the index-id gate, so embed() runs
        }
        fn dim(&self) -> usize {
            3
        }
        fn query_prefix(&self) -> &str {
            ""
        }
        fn document_prefix(&self) -> &str {
            ""
        }
    }

    #[tokio::test]
    async fn the_database_lock_is_free_while_the_embedder_runs() {
        // CLAUDE.md: never hold the DB lock across an await (it would block every other DB user).
        let (store, _dir) = store_for_test();
        let probe = LockProbe {
            conn: Arc::clone(store.connection_for_test()),
            calls: Default::default(),
            held: Default::default(),
        };
        store
            .reindex(&[(src("d.md", "k"), vec![chunk("d.md", "A", "x")])], &probe)
            .await
            .expect("ok");
        assert_eq!(probe.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert!(
            !probe.held.load(std::sync::atomic::Ordering::SeqCst),
            "the DB lock was held during embed()"
        );
    }
}
