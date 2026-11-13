// Author: Julian Bolivar
// Version: 0.25.0
// Date: 2026-11-08

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
//! The first half is asserted, not only written: `the_database_lock_is_free_while_the_embedder_runs`
//! probes the connection lock from inside the embedder (MS3 §6 loop12); the second half is
//! structural (the key lives in a `MaskedDek` read only through `with_dek`, which never calls out).
//!
//! # Examples
//!
//! ```ignore
//! let store = ReviewDocStore::new(conn, dek, doc_index_id(e.model_id(), e.dim()))?;
//! let todo = store.missing(&wanted)?;
//! // ... read and chunk only `todo`, then:
//! store.reindex(&chunked, &e).await?;
//! let vectors = store.load_vectors(query_vector.len(), &allowed)?; // once per review
//! let (hits, vanished) = store.search_loaded_counting(&vectors, &query_vector, 5)?; // per batch
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
use magi_rs::review::docs::{chunk_hash, doc_index_id, DocChunk, DocSource, WORKTREE_KEY_PREFIX};
use magi_rs::review::secrets::{display_path, find_secret};
use magi_rs::vault::MaskedDek;

/// Texts per `embed` request: thousands of chunks never go in one request (a 5 179-chunk
/// repository was measured, and it would otherwise be a single request).
pub const EMBED_BATCH_ITEMS: usize = 64;

/// `(path, key)` pairs per `search` query: two parameters each plus two fixed ones stay under
/// the 999-parameter limit of older SQLite builds.
const ALLOWED_PAIRS_PER_QUERY: usize = 400;

/// The `VALUES` list of `pairs` two-parameter rows, `(?, ?), (?, ?), ...`: the one builder of
/// the batched pair queries ([`ReviewDocStore::allowed_vectors`], [`ReviewDocStore::load_chunks`],
/// [`ReviewDocStore::gc`]), so they cannot drift apart. O(pairs).
fn pair_values(pairs: usize) -> String {
    vec!["(?, ?)"; pairs].join(", ")
}

/// What one [`ReviewDocStore::reindex`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReindexStats {
    /// Chunks sent to the embedder (each distinct chunk hash once).
    pub embedded: usize,
    /// Chunks whose stored vector was reused instead of embedding the text again.
    pub reused: usize,
    /// Sources written (those already indexed are skipped and not counted).
    pub sources: usize,
    /// The index held rows of another identity (model, dimension or chunker) and was wiped.
    pub rebuilt: bool,
    /// Sources left out for a secret pattern, by [`display_path`]: never embedded or written.
    pub refused: Vec<String>,
}

/// One allowed row read raw: key, chunk hash, encrypted vector and its `dim` column.
type RawVector = (String, String, Vec<u8>, i64);

/// Everything `reindex` decides under the connection lock, before it embeds anything.
struct Plan<'a> {
    /// The sources to write, each once, in input order.
    new_sources: Vec<&'a (DocSource, Vec<DocChunk>)>,
    /// Foreign rows exist: the index is wiped in the write transaction.
    rebuilt: bool,
    /// [`ReviewDocStore::foreign_fingerprint`] at plan time; the write requires it unchanged.
    foreign: [i64; 4],
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

/// One stored chunk read raw: its path and its still-encrypted section and text.
type RawChunk = (String, Vec<u8>, Vec<u8>);

/// The decrypted vectors of one review's allowed documentation rows, from
/// [`ReviewDocStore::load_vectors`]: `(source key, chunk hash, vector)` of the given dimension.
#[derive(Debug, Clone, Default)]
pub struct DocVectors {
    /// The dimension every vector has.
    dim: usize,
    /// `(source key, chunk hash, vector)` per allowed row.
    rows: Vec<(String, String, Vec<f32>)>,
}

impl DocVectors {
    /// The dimension the vectors were loaded for.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Encrypted, per-source index of documentation chunks (RV19, RV21).
///
/// Shares the connection and the masked data key with `EncryptedSqliteMemory`. One index
/// identity per database at a time: two embedding models used in turn rebuild the index on
/// every switch, a full re-embedding declared each time (`identities_alternate_by_rebuild`); pin
/// one `[embedding]` model per database.
///
/// MAGI FIX (MS3 §6 loop14), declared limitations (safe direction, each costs only a repeat
/// embedding or an omitted chunk, never a leak or a wrong chunk served): (1) the index identity
/// is the model id plus its dimension, not a digest of the weights, so a model re-published
/// under the same id with the same dimension keeps its old vectors until the database is purged
/// (`DELETE FROM`, README); (2) two processes indexing the same chunk at once may both embed it,
/// the second write replacing the first with an equal vector (writes are keyed by content);
/// (3) a reusable blob that no longer decrypts fails that indexing (`encode_rows`), so the
/// review runs without documentation and says so; the blob is not quarantined, and the failure
/// repeats for every review that would reuse it until the source is replaced or the index is
/// purged (`DELETE FROM`, README). It is never copied forward or served.
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
    /// A database written before v0.25.0 is refused here too, read-only and with the message
    /// that names the missing table (RV19, never a migration), whatever open produced `conn`:
    /// the schema call is `CREATE ... IF NOT EXISTS` and would otherwise upgrade it in place.
    ///
    /// # Arguments
    /// * `conn` - the connection shared with `EncryptedSqliteMemory`.
    /// * `dek` - a masked duplicate of the data key.
    /// * `index_id` - `doc_index_id(e.model_id(), e.dim())` of the embedder later passed to
    ///   [`Self::reindex`]: the identity of the vectors this store serves.
    ///
    /// # Errors
    /// [`MemoryError::UnsupportedSchema`] for a database older than v0.25.0;
    /// [`MemoryError::Storage`] if the schema DDL fails.
    pub fn new(
        conn: Arc<Mutex<Connection>>,
        dek: MaskedDek,
        index_id: String,
    ) -> Result<Self, MemoryError> {
        {
            let c = conn.lock().unwrap_or_else(|p| p.into_inner());
            crate::system::database::init_store_schema(&c)?;
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

    /// Count and highest rowid of the foreign rows of each table: any write or delete by another
    /// identity changes it (a delete of its highest row plus one insert would not; declared).
    fn foreign_fingerprint(&self, c: &Connection) -> Result<[i64; 4], MemoryError> {
        c.query_row(
            "SELECT (SELECT COUNT(*) FROM code_review_docs WHERE index_id <> ?1), \
             (SELECT COALESCE(MAX(rowid), 0) FROM code_review_docs WHERE index_id <> ?1), \
             (SELECT COUNT(*) FROM code_review_doc_sources WHERE index_id <> ?1), \
             (SELECT COALESCE(MAX(rowid), 0) FROM code_review_doc_sources WHERE index_id <> ?1)",
            [&self.index_id],
            |r| Ok([r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?]),
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

    /// Indexes whole sources, each with its chunks (possibly none); one already indexed is skipped.
    ///
    /// Checks the embedder's identity and the chunks' hashes, leaves out the sources with a
    /// secret ([`ReindexStats::refused`]), reads what is stored, embeds the rest in requests of at
    /// most [`EMBED_BATCH_ITEMS`] texts (no lock held), then writes in one transaction: all or
    /// nothing. A chunk stored under any source reuses its vector, so it is never embedded twice;
    /// rows of another identity are wiped in that transaction (`rebuilt`), nothing else.
    ///
    /// O(chunks) lookups plus the embedder. Memory: one call's sealed rows, mostly the sealed
    /// vectors (about 12 KB each at 768 dimensions); `prepare_docs` calls per batch of at most
    /// [`EMBED_BATCH_ITEMS`] chunks or one source (a source of many chunks costs more, declared).
    ///
    /// # Errors
    /// [`MemoryError::Storage`] for another identity, a chunk whose hash or path does not match
    /// it, a vector of another length than declared, or a failing query;
    /// [`MemoryError::Embedding`]; [`MemoryError::Crypto`].
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
        // MAGI FIX (MS3 §6 loop15): the secret refusal runs first, so a source left out anyway
        // cannot fail the others through its chunk hashes (R1 Melchior).
        let (safe, refused) = without_secret_sources(sources);
        refuse_inconsistent_chunks(&safe)?;
        let plan = self.plan(&safe)?;
        let fresh = embed_all(&plan.to_embed, embedder).await?;
        let rows = self.encode_rows(&plan, &fresh, embedder.dim())?;
        let rebuilt = self.write(&plan, &rows)?;
        Ok(ReindexStats {
            embedded: plan.to_embed.len(),
            reused: plan.reusable.len(),
            sources: plan.new_sources.len(),
            rebuilt,
            refused,
        })
    }

    /// Reads, under one short lock, what `reindex` needs to decide: foreign rows, the sources
    /// to write, and the stored vectors the new chunks can reuse (raw, still encrypted).
    fn plan<'a>(
        &self,
        sources: &[&'a (DocSource, Vec<DocChunk>)],
    ) -> Result<Plan<'a>, MemoryError> {
        let c = self.locked_conn();
        let foreign = self.foreign_fingerprint(&c)?;
        let rebuilt = foreign != [0; 4];
        let indexed = if rebuilt {
            BTreeSet::new()
        } else {
            self.indexed_sources(&c)?
        };
        let mut seen: BTreeSet<&DocSource> = BTreeSet::new();
        let mut new_sources: Vec<&'a (DocSource, Vec<DocChunk>)> = Vec::new();
        for &entry in sources {
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
            // MAGI FIX (MS3 §6 loop4): a rebuild wipes only foreign rows, so this identity's
            // stored vectors are still reusable (the lookup is filtered by index_id).
            let stored: Option<Vec<u8>> =
                optional(lookup.query_row(params![chunk.hash, self.index_id], |r| r.get(0)))?;
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
            foreign,
            reusable,
            to_embed,
        })
    }

    /// Encrypts every row of the new sources (off the connection lock).
    ///
    /// A reused vector is decrypted and sealed again, not copied as a blob (declared, MS3 §6
    /// loop12): O(reused) decrypt plus seal, at most [`magi_rs::review::docs::MAX_DOC_CHUNKS`]. The
    /// round trip is where a stored blob is validated (`unseal_vector` checks its shape and
    /// dimension) before it is written under a new row, so a damaged blob fails here and is never
    /// copied forward; a verbatim copy would skip the check for a saving the bound makes small.
    ///
    /// Peak memory is declared too: the call holds every sealed row at once (about 240 MB of
    /// embedding blobs at the cap and 768 dimensions, plus the decrypted vectors and the 32 MiB
    /// text bound), because the transaction needs them all before it locks; a streaming write
    /// would hold the connection across embedding batches, which R-V08 forbids.
    fn encode_rows<'a>(
        &self,
        plan: &Plan<'a>,
        fresh: &[Vec<f32>],
        dim: usize,
    ) -> Result<Vec<DocRow<'a>>, MemoryError> {
        let mut vectors: BTreeMap<&str, Vec<f32>> = BTreeMap::new();
        for (chunk, vector) in plan.to_embed.iter().zip(fresh) {
            vectors.insert(chunk.hash.as_str(), vector.clone());
        }
        for (hash, blob) in &plan.reusable {
            let vector = self.unseal_vector(blob)?;
            // MAGI FIX (MS3 §6 loop13): a reused vector is held to the declared dimension, as the
            // fresh ones are (`embed_all`): a damaged row is never written forward.
            if vector.len() != dim {
                return Err(MemoryError::Storage(format!(
                    "a stored vector has {} dimensions, the index declares {dim}",
                    vector.len()
                )));
            }
            vectors.insert(hash.as_str(), vector);
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

    /// Writes the wipe, the rows and the source rows in one transaction; answers whether it
    /// wiped. Foreign rows are wiped only when the plan saw them (`rebuilt`); any change to them
    /// by the time the transaction starts is a conflict (nothing written, the review runs without
    /// documentation and says so), so a wipe never erases a write made after the plan
    /// (`rebuilt_wipe_is_rechecked`). Inserts are `OR IGNORE`; a source another process's clean-up deleted after the plan is
    /// written back, stale until the next clean-up (declared, `reinserted_rows_are_gc_again`).
    ///
    /// # Errors
    /// [`MemoryError::Storage`] for the conflict or a failing statement.
    // MAGI FIX (MS3 §6 loop3): foreign rows written between the plan and the write were kept,
    // and `OR IGNORE` then dropped this store's rows with the same keys.
    // MAGI FIX (MS3 §6 loop6): they were then wiped, erasing the other identity's write; a race
    // is now a conflict.
    fn write(&self, plan: &Plan<'_>, rows: &[DocRow<'_>]) -> Result<bool, MemoryError> {
        let mut c = self.locked_conn();
        let tx = c
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_err)?;
        // MAGI FIX (MS3 §6 loop10): a rebuilt wipe was not re-checked.
        if self.foreign_fingerprint(&tx)? != plan.foreign {
            return Err(identity_conflict());
        }
        let wiped = plan.rebuilt;
        if wiped {
            tx.execute(
                "DELETE FROM code_review_docs WHERE index_id <> ?1",
                [&self.index_id],
            )
            .map_err(db_err)?;
            tx.execute(
                "DELETE FROM code_review_doc_sources WHERE index_id <> ?1",
                [&self.index_id],
            )
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
        tx.commit().map_err(db_err)?;
        Ok(wiped)
    }

    /// The `k` chunks most similar to `query` among the rows whose `(path, key)` is in
    /// `allowed` and whose identity is this store's, best first.
    ///
    /// A row of another dimension is an error (vectors of two models are never compared). Only
    /// the vectors of allowed rows are decrypted, and the section and text only for the `k`
    /// returned. Rows are read raw under the lock and decrypted after it is released.
    ///
    /// Complexity: O(r) decryptions, O(r * d) to score plus O(r log r) to order the `r` allowed
    /// rows. Test-only: a review loads once with [`Self::load_vectors`] and scores each batch
    /// with [`Self::search_loaded`] instead.
    ///
    /// # Errors
    /// [`MemoryError::Storage`] if a query fails; [`MemoryError::Crypto`] if decryption fails.
    #[cfg(test)]
    pub(crate) fn search(
        &self,
        query: &[f32],
        k: usize,
        allowed: &BTreeSet<DocSource>,
    ) -> Result<Vec<(DocChunk, f32)>, MemoryError> {
        if k == 0 || query.is_empty() {
            return Ok(Vec::new());
        }
        let vectors = self.load_vectors(query.len(), allowed)?;
        self.search_loaded(&vectors, query, k)
    }

    /// The decrypted vectors of the rows whose `(path, key)` is in `allowed`, of this identity:
    /// loaded once per review, scored by [`Self::search_loaded`] per batch. Read raw under the
    /// lock, decrypted after it (R-V08). A row not of `dim` (vector or `dim` column) is an error
    /// (`load_dim_mismatch_is_an_error`).
    ///
    /// With rows of another identity stored, a non-empty `allowed` is an error, never an empty
    /// answer (another identity's rebuild may have wiped this review's rows); the review declares
    /// its documentation unavailable; so is an allowed source no longer indexed
    /// (`vanished_source_is_an_error`). One snapshot. O(stored sources) plus O(r) decryptions.
    /// So the caller builds `allowed` from what it indexed: after `bounded`, without the sources
    /// `netted_chunks` dropped and those [`ReindexStats::refused`] names (`allowed_excludes_refused`).
    ///
    /// # Errors
    /// [`MemoryError::Storage`] if another identity holds the index, a row is not of `dim` or a
    /// query fails; [`MemoryError::Crypto`] if decryption fails.
    // MAGI FIX (MS3 §6 loop6): a wipe by another identity read as an empty index.
    // MAGI FIX (MS3 §6 loop7): the check and the read took two locks, each its own snapshot.
    pub fn load_vectors(
        &self,
        dim: usize,
        allowed: &BTreeSet<DocSource>,
    ) -> Result<DocVectors, MemoryError> {
        let candidates = {
            let mut c = self.locked_conn();
            let snapshot = c.transaction().map_err(db_err)?;
            if !allowed.is_empty() && self.has_foreign_rows(&snapshot)? {
                return Err(identity_conflict());
            }
            // MAGI FIX (MS3 §6 loop10): a source deleted after its reindex loaded as none.
            let indexed = self.indexed_sources(&snapshot)?;
            if !allowed.iter().all(|s| indexed.contains(s)) {
                return Err(MemoryError::Storage(
                    "a documentation source of this review is no longer indexed".to_string(),
                ));
            }
            self.allowed_vectors(&snapshot, allowed)?
        };
        let mut rows = Vec::with_capacity(candidates.len());
        for (key, hash, blob, stored_dim) in candidates {
            let vector = self.unseal_vector(&blob)?;
            // MAGI FIX (MS3 §6 loop9): a row of another dimension was dropped in silence.
            // MAGI FIX (MS3 §6 loop10): the `dim` column was never read back.
            if vector.len() != dim || usize::try_from(stored_dim).ok() != Some(dim) {
                return Err(MemoryError::Storage(format!(
                    "a stored vector of dimension {} (column {stored_dim}) where {dim} was asked",
                    vector.len()
                )));
            }
            rows.push((key, hash, vector));
        }
        Ok(DocVectors { dim, rows })
    }

    /// The `k` chunks of `vectors` most similar to `query`, best first; only their section and
    /// text are read and decrypted. A `query` of another dimension than `vectors`, or holding a
    /// value that is not finite, is an error (`dimension_mismatch_is_an_error`,
    /// `non_finite_query_is_an_error`), as the stored vectors are checked by `embed_all`; a hit
    /// whose row was deleted since `vectors` was loaded is skipped. A zero-norm vector, or one
    /// whose norm overflows, on either side scores 0 (`degenerate_query_scores_zero`).
    ///
    /// Complexity: O(r * d) to score plus O(r log r) to order the `r` loaded rows.
    ///
    /// # Errors
    /// [`MemoryError::Storage`] for such a query or a failing query;
    /// [`MemoryError::Crypto`] if decryption fails.
    #[cfg(test)]
    pub(crate) fn search_loaded(
        &self,
        vectors: &DocVectors,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<(DocChunk, f32)>, MemoryError> {
        self.search_loaded_counting(vectors, query, k)
            .map(|(hits, _)| hits)
    }

    /// [`Self::search_loaded`], also giving how many of the best `k` rows were deleted since
    /// `vectors` was loaded (another process's clean-up): the review declares that loss.
    ///
    /// # Returns
    /// The hits and the number of vanished rows.
    ///
    /// # Errors
    /// As [`Self::search_loaded`].
    // MAGI FIX (MS3 §6 loop15): a vanished row was one hit fewer with no signal (R1 Caspar).
    pub fn search_loaded_counting(
        &self,
        vectors: &DocVectors,
        query: &[f32],
        k: usize,
    ) -> Result<(Vec<(DocChunk, f32)>, usize), MemoryError> {
        // MAGI FIX (MS3 §6 loop8): a mismatch matched nothing in silence.
        // MAGI FIX (MS3 §6 loop9): a NaN query ranked arbitrary chunks first.
        if query.len() != vectors.dim || query.iter().any(|x| !x.is_finite()) {
            return Err(MemoryError::Storage(format!(
                "a query of dimension {} against vectors of dimension {}, or not finite",
                query.len(),
                vectors.dim
            )));
        }
        if k == 0 {
            return Ok((Vec::new(), 0));
        }
        let mut scored: Vec<(f32, &str, &str)> = vectors
            .rows
            .iter()
            // MAGI FIX (MS3 §6 loop10): an overflowing norm scored NaN; it now scores 0.
            .map(|(key, hash, v)| {
                let s = cosine(query, v);
                (
                    if s.is_finite() { s } else { 0.0 },
                    key.as_str(),
                    hash.as_str(),
                )
            })
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        scored.truncate(k);
        // MAGI FIX (MS3 §6 loop4): a row deleted since `load_vectors` (another process's
        // clean-up) failed the whole search; it is now one hit fewer. The hits are read in one
        // query per `ALLOWED_PAIRS_PER_QUERY` pairs, not one query each.
        let wanted: Vec<(&str, &str)> = scored.iter().map(|(_, k, h)| (*k, *h)).collect();
        let mut rows = self.load_chunks(&wanted)?;
        let mut hits = Vec::with_capacity(scored.len());
        let mut vanished = 0usize;
        for (score, key, hash) in scored {
            let Some((path, section, text)) = rows.remove(&(key.to_string(), hash.to_string()))
            else {
                vanished += 1;
                continue;
            };
            hits.push((
                DocChunk {
                    path,
                    section: self.unseal(&section)?.to_string(),
                    text: self.unseal(&text)?.to_string(),
                    hash: hash.to_string(),
                },
                score,
            ));
        }
        Ok((hits, vanished))
    }

    /// `(key, chunk hash, encrypted vector)` of the allowed rows of this identity.
    ///
    /// The `(path, key)` filter runs in SQL, in batches of [`ALLOWED_PAIRS_PER_QUERY`] pairs
    /// (so no query passes SQLite's 999-parameter limit), so the blobs of other revisions are
    /// never copied under the connection lock. Runs on the caller's locked connection `c`.
    fn allowed_vectors(
        &self,
        c: &Connection,
        allowed: &BTreeSet<DocSource>,
    ) -> Result<Vec<RawVector>, MemoryError> {
        let pairs: Vec<&DocSource> = allowed.iter().collect();
        let mut out = Vec::new();
        for batch in pairs.chunks(ALLOWED_PAIRS_PER_QUERY) {
            let values = pair_values(batch.len());
            let sql = format!(
                "SELECT key, chunk_hash, embedding_blob, dim FROM code_review_docs \
                 WHERE index_id = ? AND (path, key) IN (VALUES {values})"
            );
            let mut args: Vec<&dyn rusqlite::ToSql> = vec![&self.index_id];
            for s in batch {
                args.push(&s.path);
                args.push(&s.key);
            }
            let mut stmt = c.prepare(&sql).map_err(db_err)?;
            let rows = stmt
                .query_map(rusqlite::params_from_iter(args), |r| {
                    Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
                })
                .map_err(db_err)?;
            for row in rows {
                out.push(row.map_err(db_err)?);
            }
        }
        Ok(out)
    }
    /// The raw `(path, section blob, text blob)` of each `(key, chunk hash)` in `wanted` that
    /// still exists, read under one lock and decrypted by the caller off it (R-V08). A pair whose
    /// row was deleted is simply absent. No `index_id` filter, and none is needed: `chunk_hash`
    /// is injective over path, section and text, so a row of any identity with that pair holds
    /// the same bytes, and the vector scored is this review's own. The invariant is on the
    /// PLAINTEXT (MS3 §6 loop12): each blob is decrypted on its own with the store's key, so
    /// nothing relies on deterministic ciphertext (every seal draws a fresh nonce), and one
    /// shared key per database is what lets a row of another identity decrypt at all.
    ///
    /// Complexity: one query per [`ALLOWED_PAIRS_PER_QUERY`] pairs.
    fn load_chunks(
        &self,
        wanted: &[(&str, &str)],
    ) -> Result<BTreeMap<(String, String), RawChunk>, MemoryError> {
        let c = self.locked_conn();
        let mut out = BTreeMap::new();
        for batch in wanted.chunks(ALLOWED_PAIRS_PER_QUERY) {
            let values = pair_values(batch.len());
            let sql = format!(
                "SELECT key, chunk_hash, path, section_blob, text_blob FROM code_review_docs \
                 WHERE (key, chunk_hash) IN (VALUES {values})"
            );
            let args = batch.iter().flat_map(|(k, h)| [*k, *h]);
            let mut stmt = c.prepare(&sql).map_err(db_err)?;
            let rows = stmt
                .query_map(rusqlite::params_from_iter(args), |r| {
                    Ok(((r.get(0)?, r.get(1)?), (r.get(2)?, r.get(3)?, r.get(4)?)))
                })
                .map_err(db_err)?;
            for row in rows {
                let (k, v) = row.map_err(db_err)?;
                out.insert(k, v);
            }
        }
        Ok(out)
    }

    /// Deletes (`DELETE FROM`, both tables, one transaction) every source not in `live`; with
    /// `keep_worktree_keys`, a source whose key is a working-tree key stays. Returns how many
    /// sources it deleted.
    ///
    /// Caller contract: `live` holds every source a local ref reaches, plus the working-tree set
    /// when the review listed it (`keep_worktree_keys` `false`); a review that did not list it
    /// passes `true` and leaves those sources. Never `DROP TABLE` (the next open would read an
    /// older database). Every identity's rows count: `live` names sources by `(path, key)`.
    ///
    /// Complexity: O(stored sources), one pair of deletes per [`ALLOWED_PAIRS_PER_QUERY`] doomed
    /// sources; each delete by `(path, key)` scans `code_review_docs` (keyed `(key, chunk_hash)`),
    /// milliseconds at [`magi_rs::review::docs::MAX_DOC_CHUNKS`], so no second index is kept;
    /// raising that bound much needs a `(path, key)` index (declared).
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
        // MAGI FIX (MS3 §6 loop5): one pair of deletes per batch of doomed sources, not per
        // source.
        for batch in doomed.chunks(ALLOWED_PAIRS_PER_QUERY) {
            let values = pair_values(batch.len());
            for table in ["code_review_docs", "code_review_doc_sources"] {
                let args = batch.iter().flat_map(|s| [&s.path, &s.key]);
                tx.execute(
                    &format!("DELETE FROM {table} WHERE (path, key) IN (VALUES {values})"),
                    rusqlite::params_from_iter(args),
                )
                .map_err(db_err)?;
            }
        }
        tx.commit().map_err(db_err)?;
        Ok(doomed.len())
    }
}

/// The error of a documentation index held by another index identity at the same time
/// (another embedding model, dimension or chunker on the same database).
fn identity_conflict() -> MemoryError {
    MemoryError::Storage(
        "another index identity changed the documentation index while this review ran".to_string(),
    )
}

/// `Ok(None)` for "no row", the error otherwise.
fn optional<T>(r: rusqlite::Result<T>) -> Result<Option<T>, MemoryError> {
    match r {
        Ok(v) => Ok(Some(v)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(db_err(e)),
    }
}

/// Defense in depth (RV22): splits `sources` into the ones safe to index and the
/// [`display_path`] of each one left out because its path, or a chunk's path, section or text,
/// matches a secret pattern (a private-key block's headers included). A left-out source is never
/// embedded or written, and only it (`a_refused_source_spares_others`); `prepare_docs` nets first
/// and declares any refusal withheld. O(bytes of the chunks).
///
/// Limit, declared (MAGI FIX, MS3 loop13, R1 Caspar): a *path* that matches is left out, not
/// masked. The path is the source's identity and travels in clear with every chunk, so a masked
/// path would alias different files (and different secrets) to one source; "mask, do not omit"
/// (RV22 as amended) is about a file's content, which is masked before it reaches here. The
/// omission of a secret-shaped path is the documented behaviour end to end (smoke S49) and is
/// tested by `a_secret_in_a_source_path_is_refused_before_anything_is_embedded_or_written`.
// MAGI FIX (MS3 §6 loop3): a source path is stored in clear and travels with each chunk.
// MAGI FIX (MS3 §6 loop8): one match refused the whole reindex.
fn without_secret_sources(
    sources: &[(DocSource, Vec<DocChunk>)],
) -> (Vec<&(DocSource, Vec<DocChunk>)>, Vec<String>) {
    let leaks = |(s, chunks): &(DocSource, Vec<DocChunk>)| {
        find_secret(&s.path).is_some()
            || chunks.iter().any(|c| {
                [&c.path, &c.section, &c.text]
                    .iter()
                    .any(|t| find_secret(t).is_some())
            })
    };
    let (refused, safe): (Vec<_>, Vec<_>) = sources.iter().partition(|e| leaks(e));
    let names = refused.iter().map(|(s, _)| display_path(&s.path)).collect();
    (safe, names)
}

/// Refuses a chunk whose hash is not [`chunk_hash`] of its own path, section and text, or whose
/// path is not its source's path, before anything is read, embedded or written.
///
/// Vector reuse is keyed by the chunk hash, so a stale hash on changed text, or a chunk filed
/// under another source, would serve another text's vector unsaid; the chunkers agree, and this
/// checks it at the storage boundary. All or nothing (a caller bug, not one source's content).
/// O(total text). The message names the path through [`display_path`].
// MAGI FIX (MS3 §6 loop6): the hash was trusted as given.
fn refuse_inconsistent_chunks(sources: &[&(DocSource, Vec<DocChunk>)]) -> Result<(), MemoryError> {
    for (source, chunks) in sources.iter().copied() {
        let bad = chunks
            .iter()
            .any(|c| c.path != source.path || c.hash != chunk_hash(&c.path, &c.section, &c.text));
        if bad {
            return Err(MemoryError::Storage(format!(
                "documentation source `{}` has a chunk whose hash or path does not match it \
                 and was not indexed",
                display_path(&source.path)
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
/// a different count than asked, a vector of another length than `embedder.dim()`, or a vector
/// holding a value that is not finite.
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
        // MAGI FIX (MS3 §6 loop4): serde_json stores NaN and infinity as `null`, which never
        // reads back, so a non-finite vector would poison its source for every later review.
        if vectors.iter().flatten().any(|x| !x.is_finite()) {
            return Err(MemoryError::Storage(
                "the embedder answered a vector holding a value that is not finite".to_string(),
            ));
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

    /// A chunk of `path` carrying the hash of its own path, section and text.
    pub(crate) fn chunk(path: &str, section: &str, text: &str) -> DocChunk {
        let hash = magi_rs::review::docs::chunk_hash(path, section, text);
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
    async fn identities_alternate_by_rebuild() {
        // [MAGI S6 MS3 loop8, R1 Caspar] One identity at a time, by design: two models used in
        // turn on one database rebuild on every switch (a full re-embedding), declared each time
        // (`rebuilt`), never a comparison across models.
        let tmp = tempfile::NamedTempFile::new().expect("tmp");
        let mem = crate::system::database::EncryptedSqliteMemory::new(
            tmp.path().to_path_buf(),
            zeroize::Zeroizing::new(crate::system::database::TEST_MASTER_PASSPHRASE.to_string()),
        )
        .expect("db");
        let open = |m: &str| {
            let id = magi_rs::review::docs::doc_index_id(m, 3);
            ReviewDocStore::new(mem.shared_conn(), mem.data_key().expect("dek"), id).expect("s")
        };
        let sources = [(src("d.md", "k"), vec![chunk("d.md", "A", "x")])];
        for (round, m) in ["m1", "m2", "m1"].into_iter().enumerate() {
            let e = FakeEmbedder::new(3).named(m);
            let s = open(m).reindex(&sources, &e).await.expect("reindex");
            assert_eq!((s.embedded, s.rebuilt), (1, round > 0), "round {round}");
        }
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
    async fn a_rebuild_still_reuses_the_vectors_of_its_own_identity() {
        // [MAGI S6 MS3 loop4, R1 Balthasar] The wipe deletes only rows of another identity, so a
        // chunk already stored under this one keeps its vector and is not embedded again.
        let (store, _dir) = store_for_test();
        let e = FakeEmbedder::new(3);
        store
            .reindex(&[(src("d.md", "k1"), vec![chunk("d.md", "A", "x")])], &e)
            .await
            .expect("first");
        store
            .connection_for_test()
            .lock()
            .expect("lock")
            .execute(
                "INSERT INTO code_review_doc_sources (path, key, index_id) \
                 VALUES ('old.md', 'k0', 'other#3#1')",
                [],
            )
            .expect("foreign row");
        let s = store
            .reindex(&[(src("d.md", "k2"), vec![chunk("d.md", "A", "x")])], &e)
            .await
            .expect("rebuild");
        assert_eq!((s.embedded, s.reused, s.rebuilt), (0, 1, true));
        assert_eq!(e.calls(), 1, "only the first reindex embedded");
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
        // [MAGI S6 MS3 loop6] Before the rebuild another identity holds the index: an error,
        // never a comparison across models (`vectors_loaded_while_another_identity_...`).
        assert!(
            new.search(&[0.5, 0.5, 0.5], 5, &wanted)
                .map_or(true, |hits| hits.is_empty()),
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

    /// Runs `other`'s reindex inside its own first `embed`, i.e. between this store's plan and
    /// its write: the race two stores of different identities can run on one connection.
    struct RacingEmbedder<'a> {
        other: &'a ReviewDocStore,
        sources: Vec<(DocSource, Vec<DocChunk>)>,
        raced: std::sync::atomic::AtomicBool,
    }
    #[async_trait::async_trait]
    impl EmbeddingProvider for RacingEmbedder<'_> {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            if !self.raced.swap(true, Ordering::SeqCst) {
                self.other
                    .reindex(&self.sources, &FakeEmbedder::new(3).named("m2"))
                    .await
                    .map_err(|e| EmbeddingError::Http(e.to_string()))?;
            }
            Ok(texts.iter().map(|_| vec![0.5; 3]).collect())
        }
        fn model_id(&self) -> &str {
            "m1"
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
    async fn a_reindex_raced_by_another_identity_is_a_conflict_that_deletes_nothing() {
        // [MAGI S6 MS3 loop3, R1 Caspar] Plan and write are apart (the embedder runs unlocked).
        // Another identity's reindex that lands in between left this store's rows ignored by
        // `INSERT OR IGNORE` (the keys do not carry the identity): it reported success and served
        // nothing. The foreign rows are now checked again inside the write's transaction.
        // [MAGI S6 MS3 loop6, R1 Caspar] Wiping them there erased the other identity's successful
        // write in silence; a race is now a conflict: this reindex writes nothing and fails, and
        // the other identity keeps what it wrote.
        let tmp = tempfile::NamedTempFile::new().expect("tmp");
        let mem = crate::system::database::EncryptedSqliteMemory::new(
            tmp.path().to_path_buf(),
            zeroize::Zeroizing::new(crate::system::database::TEST_MASTER_PASSPHRASE.to_string()),
        )
        .expect("db");
        use magi_rs::review::docs::doc_index_id;
        let open = |model: &str| {
            ReviewDocStore::new(
                mem.shared_conn(),
                mem.data_key().expect("dek"),
                doc_index_id(model, 3),
            )
            .expect("store")
        };
        let (mine, other) = (open("m1"), open("m2"));
        let sources = vec![(src("d.md", "k"), vec![chunk("d.md", "A", "x")])];
        let racing = RacingEmbedder {
            other: &other,
            sources: sources.clone(),
            raced: Default::default(),
        };
        let r = mine.reindex(&sources, &racing).await;
        assert!(
            matches!(&r, Err(MemoryError::Storage(m)) if m.contains("another index identity")),
            "{r:?}"
        );
        let wanted = BTreeSet::from([src("d.md", "k")]);
        assert_eq!(
            other
                .search(&[0.5, 0.5, 0.5], 5, &wanted)
                .expect("search")
                .len(),
            1,
            "the other identity serves what it reported written"
        );
        assert!(other
            .missing(&[src("d.md", "k")])
            .expect("missing")
            .is_empty());
    }

    #[tokio::test]
    async fn the_index_identity_is_the_model_id_and_dimension_not_the_weights() {
        // [MAGI S6 MS3 loop15, R1; declared on `ReviewDocStore`] a model re-published under the
        // same id and dimension keeps its old vectors: the chunk is reused, never embedded again.
        let (store, _dir) = store_for_test();
        let first = FakeEmbedder::with_vectors(&[("x", vec![1.0, 0.0, 0.0])]);
        store
            .reindex(
                &[(src("a.md", "k1"), vec![chunk("a.md", "A", "x")])],
                &first,
            )
            .await
            .expect("first");
        let republished = FakeEmbedder::with_vectors(&[("x", vec![0.0, 1.0, 0.0])]);
        let stats = store
            .reindex(
                &[(src("a.md", "k2"), vec![chunk("a.md", "A", "x")])],
                &republished,
            )
            .await
            .expect("second");
        assert_eq!((stats.embedded, stats.reused), (0, 1));
        assert_eq!(republished.calls(), 0);
    }

    #[tokio::test]
    async fn an_undecryptable_reusable_blob_fails_that_indexing_and_writes_nothing() {
        // [MAGI S6 MS3 loop15, R1; declared on `ReviewDocStore`] a stored vector that no longer
        // decrypts is never copied forward or served: the indexing that would reuse it fails
        // (the review then runs without documentation, declared), and nothing is written.
        let (store, _dir) = store_for_test();
        let e = FakeEmbedder::new(3);
        store
            .reindex(&[(src("a.md", "k1"), vec![chunk("a.md", "A", "x")])], &e)
            .await
            .expect("first");
        store
            .connection_for_test()
            .lock()
            .expect("lock")
            .execute(
                "UPDATE code_review_docs SET embedding_blob = zeroblob(64)",
                [],
            )
            .expect("tamper");
        let b = src("a.md", "k2");
        let r = store
            .reindex(&[(b.clone(), vec![chunk("a.md", "A", "x")])], &e)
            .await;
        assert!(matches!(r, Err(MemoryError::Crypto(_))), "{r:?}");
        assert_eq!(
            store.missing(std::slice::from_ref(&b)).expect("missing"),
            vec![b]
        );
    }

    #[tokio::test]
    async fn dim_column_is_checked() {
        // [MAGI S6 MS3 loop10, R1 Caspar] The `dim` column was written and never read back.
        let (store, _dir) = store_for_test();
        let s = (src("d.md", "k"), vec![chunk("d.md", "A", "x")]);
        store
            .reindex(&[s], &FakeEmbedder::new(3))
            .await
            .expect("dim 3");
        store
            .connection_for_test()
            .lock()
            .expect("lock")
            .execute("UPDATE code_review_docs SET dim = 5", [])
            .expect("tamper");
        let allowed = BTreeSet::from([src("d.md", "k")]);
        assert!(matches!(
            store.load_vectors(3, &allowed),
            Err(MemoryError::Storage(_))
        ));
    }

    #[tokio::test]
    async fn reused_vector_length_is_checked() {
        // MAGI FIX (MS3 §6 loop13, R1 Caspar): a stored vector reused for a new source was only
        // deserialized, so one of another length (a damaged row) was written forward under the
        // declared dimension.
        let (store, _dir) = store_for_test();
        let first = (src("d.md", "k"), vec![chunk("d.md", "A", "x")]);
        store
            .reindex(&[first], &FakeEmbedder::new(3))
            .await
            .expect("dim 3");
        let short = store.seal_vector(&[1.0, 2.0]).expect("seal");
        store
            .connection_for_test()
            .lock()
            .expect("lock")
            .execute("UPDATE code_review_docs SET embedding_blob = ?1", [&short])
            .expect("tamper");
        // The same chunk under a new version of the file (another source key): its vector is reused.
        let second = (src("d.md", "k2"), vec![chunk("d.md", "A", "x")]);
        let err = store
            .reindex(&[second], &FakeEmbedder::new(3))
            .await
            .expect_err("a reused vector of the wrong length is refused");
        assert!(matches!(err, MemoryError::Storage(_)), "{err}");
    }

    #[tokio::test]
    async fn vanished_source_is_an_error() {
        // [MAGI S6 MS3 loop10, R1 Melchior] A source deleted after the reindex (a wipe, another
        // process's clean-up) loaded as "no documentation"; it is now the index unavailable.
        let (store, _dir) = store_for_test();
        let s = (src("d.md", "k"), vec![chunk("d.md", "A", "x")]);
        store
            .reindex(&[s], &FakeEmbedder::new(3))
            .await
            .expect("dim 3");
        let allowed = BTreeSet::from([src("d.md", "k")]);
        assert_eq!(store.load_vectors(3, &allowed).expect("load").rows.len(), 1);
        store.gc(&BTreeSet::new(), false).expect("gc");
        assert!(matches!(
            store.load_vectors(3, &allowed),
            Err(MemoryError::Storage(_))
        ));
    }

    #[tokio::test]
    async fn rebuilt_wipe_is_rechecked() {
        // [MAGI S6 MS3 loop10, R1 Caspar] A rebuild wiped what another identity wrote after the
        // plan; that write is now a conflict too, and the other identity keeps its rows.
        let tmp = tempfile::NamedTempFile::new().expect("tmp");
        let mem = crate::system::database::EncryptedSqliteMemory::new(
            tmp.path().to_path_buf(),
            zeroize::Zeroizing::new(crate::system::database::TEST_MASTER_PASSPHRASE.to_string()),
        )
        .expect("db");
        let open = |model: &str| {
            let id = magi_rs::review::docs::doc_index_id(model, 3);
            ReviewDocStore::new(mem.shared_conn(), mem.data_key().expect("dek"), id).expect("store")
        };
        let (mine, other) = (open("m1"), open("m2"));
        let old = (src("d.md", "k"), vec![chunk("d.md", "A", "x")]);
        let m2 = FakeEmbedder::new(3).named("m2");
        other
            .reindex(std::slice::from_ref(&old), &m2)
            .await
            .expect("m2");
        let racing = RacingEmbedder {
            other: &other,
            sources: vec![(src("e.md", "k2"), vec![chunk("e.md", "B", "y")])],
            raced: Default::default(),
        };
        let r = mine.reindex(&[old], &racing).await;
        assert!(matches!(&r, Err(MemoryError::Storage(_))), "{r:?}");
        let both = [src("d.md", "k"), src("e.md", "k2")];
        assert!(other.missing(&both).expect("missing").is_empty());
    }

    #[tokio::test]
    async fn vectors_loaded_while_another_identity_holds_the_index_are_an_error_not_silence() {
        // [MAGI S6 MS3 loop6, R1 Caspar] A wrote, then B (another identity) rebuilt and wiped A's
        // rows: A loading afterwards got zero vectors and served no documentation undeclared.
        // It is now an error, which the review declares as the index being unavailable.
        let tmp = tempfile::NamedTempFile::new().expect("tmp");
        let mem = crate::system::database::EncryptedSqliteMemory::new(
            tmp.path().to_path_buf(),
            zeroize::Zeroizing::new(crate::system::database::TEST_MASTER_PASSPHRASE.to_string()),
        )
        .expect("db");
        use magi_rs::review::docs::doc_index_id;
        let open = |model: &str| {
            ReviewDocStore::new(
                mem.shared_conn(),
                mem.data_key().expect("dek"),
                doc_index_id(model, 3),
            )
            .expect("store")
        };
        let (a, b) = (open("m1"), open("m2"));
        let sources = vec![(src("d.md", "k"), vec![chunk("d.md", "A", "x")])];
        a.reindex(&sources, &FakeEmbedder::new(3).named("m1"))
            .await
            .expect("a");
        b.reindex(&sources, &FakeEmbedder::new(3).named("m2"))
            .await
            .expect("b rebuilds");
        let wanted = BTreeSet::from([src("d.md", "k")]);
        assert!(
            matches!(a.load_vectors(3, &wanted), Err(MemoryError::Storage(_))),
            "a must not read the wipe as an empty index"
        );
        assert!(
            a.load_vectors(3, &BTreeSet::new())
                .expect("nothing asked")
                .rows
                .is_empty(),
            "asking for nothing is never an error"
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
    async fn dimension_mismatch_is_an_error() {
        // [MAGI S6 MS3 loop8, R1 Melchior] A query of another dimension than the loaded vectors
        // (a caller bug) matched nothing in silence, read as "no documentation hits".
        let (store, _dir) = store_for_test();
        let e = FakeEmbedder::new(3);
        store
            .reindex(&[(src("d.md", "k"), vec![chunk("d.md", "A", "x")])], &e)
            .await
            .expect("dim 3");
        let vectors = store
            .load_vectors(3, &BTreeSet::from([src("d.md", "k")]))
            .expect("load");
        let r = store.search_loaded(&vectors, &[1.0, 0.0], 5);
        assert!(matches!(r, Err(MemoryError::Storage(_))), "{r:?}");
    }

    #[tokio::test]
    async fn non_finite_query_is_an_error() {
        // [MAGI S6 MS3 loop9, R1 Caspar] A NaN or infinite query scored every row NaN and
        // `total_cmp` returned arbitrary chunks, with no error.
        let (store, _dir) = store_for_test();
        let e = FakeEmbedder::new(3);
        store
            .reindex(&[(src("d.md", "k"), vec![chunk("d.md", "A", "x")])], &e)
            .await
            .expect("dim 3");
        let vectors = store
            .load_vectors(3, &BTreeSet::from([src("d.md", "k")]))
            .expect("load");
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let r = store.search_loaded(&vectors, &[bad, 0.0, 0.0], 5);
            assert!(matches!(r, Err(MemoryError::Storage(_))), "{bad}: {r:?}");
        }
    }

    #[tokio::test]
    async fn degenerate_query_scores_zero() {
        // [MAGI S6 MS3 loop10, R1 Caspar] A zero query, or one whose norm overflows, scored NaN.
        let (store, _dir) = store_for_test();
        store
            .reindex(
                &[(src("d.md", "k"), vec![chunk("d.md", "A", "x")])],
                &FakeEmbedder::new(3),
            )
            .await
            .expect("dim 3");
        let vectors = store
            .load_vectors(3, &BTreeSet::from([src("d.md", "k")]))
            .expect("load");
        for q in [[0.0; 3], [f32::MAX; 3]] {
            let hits = store.search_loaded(&vectors, &q, 5).expect("search");
            assert_eq!(hits.len(), 1, "{q:?}");
            assert_eq!(hits[0].1, 0.0, "{q:?}");
        }
    }

    #[tokio::test]
    async fn load_dim_mismatch_is_an_error() {
        // [MAGI S6 MS3 loop9, R1 Melchior] Rows of another dimension than asked are never
        // compared; they were dropped in silence, read as "no documentation".
        let (store, _dir) = store_for_test();
        store
            .reindex(
                &[(src("d.md", "k"), vec![chunk("d.md", "A", "x")])],
                &FakeEmbedder::new(3),
            )
            .await
            .expect("dim 3");
        let allowed = BTreeSet::from([src("d.md", "k")]);
        let r = store.load_vectors(2, &allowed);
        assert!(matches!(r, Err(MemoryError::Storage(_))), "{r:?}");
        assert!(store.search(&[1.0, 0.0], 5, &allowed).is_err());
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
    async fn reinserted_rows_are_gc_again() {
        // [MAGI S6 MS3 loop9, R1 Caspar] Declared race: a reindex planned before another
        // process's clean-up writes its rows back (`OR IGNORE`); the next clean-up deletes them.
        let (store, _dir) = store_for_test();
        let e = FakeEmbedder::new(3);
        let doomed = [(src("d.md", "k"), vec![chunk("d.md", "A", "x")])];
        store.reindex(&doomed, &e).await.expect("index");
        assert_eq!(store.gc(&BTreeSet::new(), false).expect("gc"), 1);
        store.reindex(&doomed, &e).await.expect("written back");
        assert_eq!(dims(&store).len(), 1, "stale until the next clean-up");
        assert_eq!(store.gc(&BTreeSet::new(), false).expect("gc"), 1);
        assert!(dims(&store).is_empty());
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

    #[test]
    fn a_store_handed_an_older_database_refuses_it_and_creates_nothing() {
        // [MAGI S6 MS3 loop3, R1 Melchior] The refusal of a 0.24.x database (RV19) rested on the
        // caller's checked open: handed a raw connection, `new` created the v0.25.0 tables.
        let tmp = tempfile::NamedTempFile::new().expect("tmp");
        let dek = crate::system::database::EncryptedSqliteMemory::new(
            tmp.path().to_path_buf(),
            zeroize::Zeroizing::new(crate::system::database::TEST_MASTER_PASSPHRASE.to_string()),
        )
        .expect("db")
        .data_key()
        .expect("dek");
        let (conn, _old) = crate::system::database::pre_0_25_connection_for_test();
        let r = ReviewDocStore::new(
            Arc::clone(&conn),
            dek,
            magi_rs::review::docs::doc_index_id("fake", 3),
        );
        assert!(
            // [MAGI S6 MS3 loop4, R3 Balthasar] typed, so a caller can tell it from storage.
            matches!(&r, Err(e @ MemoryError::UnsupportedSchema { missing_table: "code_review_docs", .. }) if e.to_string().contains("0.24.x")),
            "{:?}",
            r.as_ref().err()
        );
        assert_eq!(crate::system::database::review_tables_for_test(&conn), 0);
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
            matches!(&r, Ok(s) if s.refused == ["leak.md"] && s.sources == 0),
            "{r:?}"
        );
        assert_eq!(e.calls(), 0);
    }

    #[tokio::test]
    async fn a_secret_in_a_source_path_is_refused_before_anything_is_embedded_or_written() {
        // [MAGI S6 MS3 loop3, R1 Caspar] RV22's tie-break: a path with a secret shape is never
        // masked and never leaves the process, so a file named like a key is omitted whole. The
        // path sits in a clear column and travels with every chunk, so the store refuses it
        // itself, with or without chunks, whatever its caller checked before.
        let (store, _dir) = store_for_test();
        let e = FakeEmbedder::new(3);
        let path = "docs/sk-ant-api03-AAAAAAAAAAAAAAAAAAAA.md"; // allow-secret-scan
        for chunks in [vec![chunk(path, "Usage", "harmless text")], Vec::new()] {
            let r = store.reindex(&[(src(path, "k"), chunks)], &e).await;
            assert!(
                matches!(&r, Ok(s) if s.refused.len() == 1 && !s.refused[0].contains("sk-ant-")),
                "{r:?}"
            );
        }
        assert_eq!(e.calls(), 0);
        assert_eq!(
            store.missing(&[src(path, "k")]).expect("missing"),
            vec![src(path, "k")],
            "nothing was written"
        );
    }

    #[tokio::test]
    async fn a_refused_source_cannot_fail_the_whole_reindex_through_its_chunk_hash() {
        // [MAGI S6 MS3 loop15, R1 Melchior] the consistency check ran before the secret refusal,
        // so a source that was going to be left out anyway failed every other source with it.
        let (store, _dir) = store_for_test();
        let e = FakeEmbedder::new(3);
        let path = "docs/sk-ant-api03-AAAAAAAAAAAAAAAAAAAA.md"; // allow-secret-scan
        let mut stale = chunk(path, "Usage", "harmless text");
        stale.hash = "stale".into();
        let r = store
            .reindex(
                &[
                    (src(path, "k"), vec![stale]),
                    (src("ok.md", "k2"), vec![chunk("ok.md", "A", "x")]),
                ],
                &e,
            )
            .await;
        assert!(
            matches!(&r, Ok(s) if s.refused.len() == 1 && s.sources == 1),
            "{r:?}"
        );
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

    #[tokio::test]
    async fn an_embedder_answering_a_non_finite_vector_fails_the_reindex_and_writes_nothing() {
        // [MAGI S6 MS3 loop4, R1 Melchior] serde_json writes NaN or infinity as `null`, so a
        // stored non-finite vector could never be read back: every later review of its source
        // would fail. The reindex refuses it up front instead, all or nothing.
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let (store, _dir) = store_for_test();
            let e = FakeEmbedder::with_vectors(&[("y", vec![1.0, bad, 0.0])]);
            let r = store
                .reindex(&[(src("d.md", "k"), vec![chunk("d.md", "B", "y")])], &e)
                .await;
            assert!(
                matches!(&r, Err(MemoryError::Storage(m)) if m.contains("finite")),
                "{bad}: {r:?}"
            );
            assert!(dims(&store).is_empty(), "{bad}: nothing written");
        }
    }

    #[tokio::test]
    async fn a_hit_whose_row_vanished_after_loading_is_skipped_not_a_failed_search() {
        // [MAGI S6 MS3 loop4, R1 Melchior] Another process may delete a row between
        // `load_vectors` and `search_loaded` (its clean-up); the other hits still serve.
        let (store, _dir) = store_for_test();
        let e =
            FakeEmbedder::with_vectors(&[("x", vec![1.0, 0.0, 0.0]), ("y", vec![0.9, 0.1, 0.0])]);
        store
            .reindex(
                &[
                    (src("a.md", "k1"), vec![chunk("a.md", "A", "x")]),
                    (src("b.md", "k2"), vec![chunk("b.md", "B", "y")]),
                ],
                &e,
            )
            .await
            .expect("index");
        let allowed: BTreeSet<DocSource> = [src("a.md", "k1"), src("b.md", "k2")].into();
        let vectors = store.load_vectors(3, &allowed).expect("load");
        store
            .connection_for_test()
            .lock()
            .expect("lock")
            .execute("DELETE FROM code_review_docs WHERE key = 'k1'", [])
            .expect("delete");
        let hits = store
            .search_loaded(&vectors, &[1.0, 0.0, 0.0], 5)
            .expect("the search survives a vanished row");
        let paths: Vec<&str> = hits.iter().map(|(c, _)| c.path.as_str()).collect();
        assert_eq!(paths, vec!["b.md"]);
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
                &[(src("leak.md", "k2"), vec![chunk("leak.md", "Setup", text)])],
                &e,
            )
            .await;
        assert!(matches!(&r, Ok(s) if s.refused == ["leak.md"]), "{r:?}");
        assert_eq!(e.calls(), 0);
    }

    #[tokio::test]
    async fn a_secret_split_by_a_hidden_character_never_reaches_the_embedder() {
        // [MAGI S6 MS3 loop16, R1 Caspar] The embedder request body is an egress point (RV22):
        // a token split by a zero-width space is invisible to `find_secret` and travelled whole.
        let (store, _dir) = store_for_test();
        let e = FakeEmbedder::new(3);
        let text = "token = ghp_AAAAAAAAAAAAAAAAAA\u{200b}AAAAAAAAAAAAAAAAAA"; // allow-secret-scan
        let r = store
            .reindex(
                &[(src("leak.md", "k3"), vec![chunk("leak.md", "Setup", text)])],
                &e,
            )
            .await;
        assert!(matches!(&r, Ok(s) if s.refused == ["leak.md"]), "{r:?}");
        assert_eq!(e.calls(), 0, "nothing embedded");
    }

    #[tokio::test]
    async fn a_refused_source_spares_others() {
        // [MAGI S6 MS3 loop8, R1 Caspar] One source with a secret refused the whole reindex, so
        // the revision lost all its documentation; now only that source is left out, by name.
        let (store, _dir) = store_for_test();
        let e = FakeEmbedder::new(3);
        let text = "export KEY=sk-ant-api03-AAAAAAAAAAAAAAAAAAAA"; // allow-secret-scan
        let stats = store
            .reindex(
                &[
                    (src("d.md", "k1"), vec![chunk("d.md", "A", "x")]),
                    (src("leak.md", "k2"), vec![chunk("leak.md", "Setup", text)]),
                ],
                &e,
            )
            .await
            .expect("the clean source is indexed");
        assert_eq!(
            (stats.sources, stats.refused.clone()),
            (1, vec!["leak.md".to_string()])
        );
        assert_eq!(e.calls(), 1, "only the clean chunk was embedded");
        assert_eq!(
            store
                .missing(&[src("d.md", "k1"), src("leak.md", "k2")])
                .expect("missing"),
            vec![src("leak.md", "k2")]
        );
    }

    #[tokio::test]
    async fn risk_based_heading_is_indexed() {
        // [MAGI S6 MS3 loop8, R1 Caspar] Text next to a pattern's opener (`sk-` in "Risk-based")
        // is no secret: the source is indexed, nothing refused.
        let (store, _dir) = store_for_test();
        let e = FakeEmbedder::new(3);
        let c = chunk(
            "d.md",
            "Risk-based testing",
            "Prefer risk-based, task-sized tests.",
        );
        let stats = store
            .reindex(&[(src("d.md", "k1"), vec![c])], &e)
            .await
            .expect("indexed");
        assert_eq!((stats.sources, stats.refused.len()), (1, 0));
    }

    #[tokio::test]
    async fn a_chunk_whose_hash_or_path_does_not_match_it_is_refused_before_any_reuse() {
        // [MAGI S6 MS3 loop6, R1 Caspar] Reuse is keyed by the chunk hash: a stale hash on a
        // changed text, or a chunk filed under another source's path, would serve the stored
        // vector of other text. The store recomputes the hash and checks the path first.
        let (store, _dir) = store_for_test();
        let e = FakeEmbedder::new(3);
        store
            .reindex(&[(src("d.md", "k1"), vec![chunk("d.md", "A", "old")])], &e)
            .await
            .expect("first");
        let mut stale = chunk("d.md", "A", "new text");
        stale.hash = chunk("d.md", "A", "old").hash;
        let other_path = chunk("other.md", "A", "x");
        for (case, bad) in [("stale hash", stale), ("other path", other_path)] {
            let r = store.reindex(&[(src("d.md", "k2"), vec![bad])], &e).await;
            assert!(
                matches!(&r, Err(MemoryError::Storage(m)) if m.contains("d.md")),
                "{case}: {r:?}"
            );
            assert_eq!(
                store.missing(&[src("d.md", "k2")]).expect("missing").len(),
                1,
                "{case}: nothing written"
            );
        }
        assert_eq!(e.calls(), 1, "only the first reindex embedded");
    }

    #[tokio::test]
    async fn a_chunk_holding_a_private_key_block_is_refused_before_anything_is_embedded() {
        // [MAGI S6 MS3 loop4, R1 Caspar] RV22's exception: a multi-line private-key block never
        // leaves the process. `refuse_secrets` refuses it by itself, whatever its caller netted
        // first: the PEM, OpenPGP and SSH2 headers are patterns of the same table `find_secret`
        // scans, in the text or in the section.
        let pem = concat!("-----", "BEGIN ", "PRIVATE KEY-----\nMIIEvQIBADANBg\n");
        let pgp = concat!("-----", "BEGIN PGP ", "PRIVATE KEY BLOCK-----\nlQOYBGZ\n");
        let ssh2 = concat!(
            "---- BEGIN SSH2 ",
            "ENCRYPTED PRIVATE KEY ----\nP2/56wAAA\n"
        );
        for block in [pem, pgp, ssh2] {
            for (section, text) in [("Setup", block), (block, "harmless text")] {
                let (store, _dir) = store_for_test();
                let e = FakeEmbedder::new(3);
                let r = store
                    .reindex(
                        &[(src("key.md", "k"), vec![chunk("key.md", section, text)])],
                        &e,
                    )
                    .await;
                assert!(
                    matches!(&r, Ok(s) if s.refused == ["key.md"]),
                    "{block:?} in {section:?}: {r:?}"
                );
                assert_eq!(e.calls(), 0, "{block:?} reached the embedder");
                assert!(dims(&store).is_empty(), "nothing written");
            }
        }
    }

    #[tokio::test]
    async fn a_source_path_shaped_like_a_secret_is_refused_without_quoting_it() {
        // [MAGI S6 MS3 loop5, R1 Melchior] The source-path arm is the one place where bytes that
        // match a pattern reach a message (one that can surface in the run log): it names the
        // path through `display_path`, which redacts it, and a source with no chunk is still
        // refused, so the path never reaches the store either.
        let token = concat!("ghp_", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"); // allow-secret-scan
        let path = format!("docs/{token}.md");
        let (store, _dir) = store_for_test();
        let e = FakeEmbedder::new(3);
        let r = store.reindex(&[(src(&path, "k"), vec![])], &e).await;
        match r {
            Ok(s) if s.refused.len() == 1 => {
                let m = &s.refused[0];
                assert!(!m.contains(token), "{m}");
                assert!(m.contains("<path redacted: "), "{m}");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
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
