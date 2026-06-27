// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-06-27

//! Public retrieval facade: composite reranker, the **`recall`** entry-point
//! (D-13 / B1 seam), and lazy re-embedding of pending vectors (CP2-C).
//!
//! # Design
//!
//! This module is the **stable public interface** between the vector store and
//! every consumer that needs ranked memories — the context assembler (Task 11),
//! the future MAGI deliberation layer (AS-REQ-11), and any other agent-society
//! consumer (D-13 / REQ-37 / B1). The function signature is:
//!
//! ```ignore
//! pub async fn recall(
//!     store:    &dyn VectorStore,
//!     embedder: &dyn EmbeddingProvider,
//!     clock:    &dyn Clock,
//!     cfg:      &MemoryConfig,
//!     query:    &str,
//!     budget:   usize,   // advisory; assembler enforces the hard token limit
//!     scope:    &str,
//! ) -> Result<Vec<RankedMemory>, MemoryError>
//! ```
//!
//! It is deliberately **independent of the context assembler** (Task 11 consumes
//! it, but this module does not know about the assembler) so L3 consumers can
//! call it directly without pulling in token-accounting logic.
//!
//! # Index selection (D-05)
//!
//! - Default path: [`BruteForceIndex`] — exact cosine, always available, deterministic.
//! - `--features ann` opt-in: `InstantDistanceIndex` (HNSW) when `cfg.index == "ann"`.
//!
//! # Determinism (R-06)
//!
//! All time-dependent quantities (`age_days`, `recency`) use the injected
//! [`Clock`] — `SystemTime::now()` is never called inside this module.

use std::cmp::Ordering;

use crate::memory::clock::Clock;
use crate::memory::config::MemoryConfig;
use crate::memory::embedding::EmbeddingProvider;
use crate::memory::error::{EmbeddingError, MemoryError};
use crate::memory::index::{BruteForceIndex, VectorIndex};
use crate::memory::store::{Memory, VectorStore};

#[cfg(feature = "ann")]
use crate::memory::index::InstantDistanceIndex;

// ─── Public types ─────────────────────────────────────────────────────────────

/// A [`Memory`] paired with its composite rerank score.
///
/// Returned by [`recall`] in descending score order. The score is a
/// weighted combination of cosine similarity, recency, and salience
/// (see the formula in [`recall`]'s documentation). Scores are in `[0, 1]`
/// under the default weight configuration but are not formally bounded.
///
/// This is the **stable output type of the B1 seam** (D-13 / REQ-37) and
/// MUST NOT be altered once downstream consumers (Task 11, Agent Society) are
/// wired.
// Narrow allow: consumed by the context assembler (Task 11) and wired into the
// agent in Task 12; no non-test caller exists yet.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct RankedMemory {
    /// The decoded memory record.
    pub memory: Memory,
    /// Composite rerank score (similarity × weight + recency × weight + salience × weight).
    pub score: f64,
}

// ─── recall ───────────────────────────────────────────────────────────────────

/// Embeds the query, retrieves active memories of the current model in `scope`,
/// reranks by similarity + recency + salience, updates access stats on the hits,
/// and returns the ranked list.
///
/// This is the **public, assembler-independent retrieval entry-point** (D-13 /
/// REQ-37 / B1). Consumers include the context assembler (Task 11) and future
/// Agent Society consumers (AS-REQ-11). Its contract MUST remain stable.
///
/// # Algorithm
///
/// 1. Embed `query` (with `embedder.query_prefix()` applied).
/// 2. Load all active memories for `scope` from `store`.
/// 3. Filter by `model_id == embedder.model_id()` AND `dim == embedder.dim()`
///    AND non-empty embedding (D-06).
/// 4. Build a [`BruteForceIndex`] (or `InstantDistanceIndex` when `--features ann`
///    and `cfg.index == "ann"`) seeded with `cfg.seed`.
/// 5. Retrieve `cfg.top_k` candidates.
/// 6. Rerank: `score = (w_sim·sim + w_rec·recency + w_sal·salience) / (w_sim + w_rec + w_sal)`,
///    where `recency = 0.5^(age_days / decay_half_life_days)` and `age_days` is
///    measured via the injected [`Clock`] (D-18).
/// 7. Sort by `score` descending; tie-break by `created_at` descending, then `id` ascending (R-06).
/// 8. Update access stats (`access_count`, `last_accessed_at`) for the returned IDs only (REQ-07).
///
/// # Parameters
///
/// - `budget`: advisory hint for the caller (token budget); `recall` itself does
///   NOT enforce it — the context assembler (Task 11) does.
///
/// # Errors
///
/// - [`MemoryError::Embedding`] if the embedder fails (auth, rate-limit, dim mismatch, …).
/// - [`MemoryError::Storage`] or [`MemoryError::Crypto`] on store failures.
// Narrow allow: consumed by the context assembler (Task 11); no non-test caller yet.
#[allow(dead_code)]
pub async fn recall(
    store: &dyn VectorStore,
    embedder: &dyn EmbeddingProvider,
    clock: &dyn Clock,
    cfg: &MemoryConfig,
    query: &str,
    _budget: usize,
    scope: &str,
) -> Result<Vec<RankedMemory>, MemoryError> {
    // Step 1: embed the query, applying the task prefix (D-04).
    let qv = embed_query(embedder, query).await?;

    // Step 2: load active memories.
    let mems = store.active(scope).await?;

    // Step 3: filter by current model / dim (D-06).
    let current_model = embedder.model_id();
    let current_dim = embedder.dim();

    let filtered: Vec<(usize, &Memory)> = mems
        .iter()
        .enumerate()
        .filter(|(_, m)| {
            m.model_id == current_model && m.dim == current_dim && !m.embedding.is_empty()
        })
        .collect();

    if filtered.is_empty() {
        return Ok(vec![]);
    }

    // Step 4: build the index over the filtered set.
    // id_index values are positions into `mems` so we can look up the Memory directly.
    let points: Vec<(usize, Vec<f32>)> = filtered
        .iter()
        .map(|(i, m)| (*i, m.embedding.clone()))
        .collect();

    let hits: Vec<(usize, f32)> = {
        #[cfg(feature = "ann")]
        if cfg.index == "ann" {
            let idx = InstantDistanceIndex::build(&points, cfg.seed);
            idx.search(&qv, cfg.top_k)
        } else {
            BruteForceIndex::build(&points, cfg.seed).search(&qv, cfg.top_k)
        }

        // Default (no-ann) path: always brute force.
        #[cfg(not(feature = "ann"))]
        {
            BruteForceIndex::build(&points, cfg.seed).search(&qv, cfg.top_k)
        }
    };

    // Step 5 & 6: rerank using similarity + recency (time-based, D-18) + salience.
    let now = clock.now();
    let w_sum = cfg.weight_similarity + cfg.weight_recency + cfg.weight_salience;
    let w_sum = if w_sum == 0.0 { 1.0 } else { w_sum };

    let mut ranked: Vec<RankedMemory> = hits
        .iter()
        .map(|(i, sim)| {
            let m = &mems[*i];
            let age_secs = (now - m.last_accessed_at).max(0);
            let age_days = age_secs as f64 / 86_400.0;
            let recency = 0.5f64.powf(age_days / cfg.decay_half_life_days);
            let score = (cfg.weight_similarity * f64::from(*sim)
                + cfg.weight_recency * recency
                + cfg.weight_salience * m.salience)
                / w_sum;
            RankedMemory {
                memory: m.clone(),
                score,
            }
        })
        .collect();

    // Step 7: sort — score desc, created_at desc, id asc (deterministic, R-06).
    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| b.memory.created_at.cmp(&a.memory.created_at))
            .then_with(|| a.memory.id.cmp(&b.memory.id))
    });

    // Step 8: update access stats for returned hits only (REQ-07).
    let hit_ids: Vec<String> = ranked.iter().map(|r| r.memory.id.clone()).collect();
    store.mark_accessed(&hit_ids, now).await?;

    Ok(ranked)
}

// ─── reembed_pending ─────────────────────────────────────────────────────────

/// Re-embeds memories whose embedding is empty OR whose `model_id` differs from
/// the embedder's current model (CP2-C / REQ-02).
///
/// Processes at most `cfg.reembed_batch_size` memories per call (throttled to
/// avoid overwhelming the embedder). On [`EmbeddingError::RateLimited`] the
/// function stops early and returns the count processed so far — the caller is
/// expected to retry after a backoff.
///
/// The document prefix ([`EmbeddingProvider::document_prefix`]) is applied
/// before embedding (D-04). New embeddings are written via
/// [`VectorStore::update_embedding`].
///
/// # Returns
///
/// The number of memories successfully re-embedded in this call.
///
/// # Errors
///
/// - [`MemoryError::Embedding`] for non-rate-limit embedding failures.
/// - [`MemoryError::Storage`] / [`MemoryError::Crypto`] for store write failures.
// Narrow allow: consumed by the background embedding runner (Task 12);
// no non-test caller yet.
#[allow(dead_code)]
pub async fn reembed_pending(
    store: &dyn VectorStore,
    embedder: &dyn EmbeddingProvider,
    cfg: &MemoryConfig,
    scope: &str,
) -> Result<usize, MemoryError> {
    let mems = store.active(scope).await?;
    let current_model = embedder.model_id();

    // Collect memories that need re-embedding.
    let pending: Vec<&Memory> = mems
        .iter()
        .filter(|m| m.embedding.is_empty() || m.model_id != current_model)
        .collect();

    if pending.is_empty() {
        return Ok(0);
    }

    let batch_size = cfg.reembed_batch_size.max(1);
    let doc_prefix = embedder.document_prefix();
    let mut re_embedded = 0usize;

    for chunk in pending.chunks(batch_size) {
        // Apply document prefix (D-04).
        let texts: Vec<String> = chunk
            .iter()
            .map(|m| {
                if doc_prefix.is_empty() {
                    m.text.clone()
                } else {
                    format!("{doc_prefix}{}", m.text)
                }
            })
            .collect();

        let embeddings = match embedder.embed(&texts).await {
            Ok(v) => v,
            Err(EmbeddingError::RateLimited) => {
                // Stop early; caller retries after backoff (CP2-C).
                return Ok(re_embedded);
            }
            Err(e) => return Err(MemoryError::Embedding(e)),
        };

        for (m, emb) in chunk.iter().zip(embeddings.iter()) {
            let dim = emb.len();
            store
                .update_embedding(&m.id, emb, current_model, dim)
                .await?;
        }

        re_embedded += chunk.len();
    }

    Ok(re_embedded)
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// Embeds `raw` using the embedder's `query_prefix` (D-04) and returns a single
/// vector.  Inlined because `embed_query` is not a trait method.
async fn embed_query(embedder: &dyn EmbeddingProvider, raw: &str) -> Result<Vec<f32>, MemoryError> {
    let prefix = embedder.query_prefix();
    let prefixed = if prefix.is_empty() {
        raw.to_string()
    } else {
        format!("{prefix}{raw}")
    };
    let mut vecs = embedder.embed(&[prefixed]).await?;
    vecs.pop().ok_or_else(|| {
        MemoryError::Embedding(EmbeddingError::Malformed("empty response for query".into()))
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::clock::FixedClock;
    use crate::memory::config::MemoryConfig;
    use crate::memory::error::EmbeddingError;
    use crate::memory::store::{Memory, SqliteVectorStore};
    use crate::memory::MemoryKind;
    use crate::system::database::EncryptedSqliteMemory;
    use async_trait::async_trait;

    // ── Deterministic test embedder ───────────────────────────────────────

    /// L2-normalised bag-of-words over a fixed-dim hash.
    ///
    /// Texts that share words produce vectors with high cosine similarity.
    /// Deterministic: same text + dim → same vector every time (R-06).
    fn bow(text: &str, dim: usize) -> Vec<f32> {
        let mut v = vec![0f32; dim];
        for w in text.to_lowercase().split_whitespace() {
            let h = w
                .bytes()
                .fold(0usize, |a, b| a.wrapping_mul(31).wrapping_add(b as usize))
                % dim;
            v[h] += 1.0;
        }
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 {
            for x in &mut v {
                *x /= n;
            }
        }
        v
    }

    /// Fake embedder that computes `bow` in-process — no HTTP, fully deterministic.
    struct FakeEmbedder {
        dim: usize,
        model: String,
    }

    #[async_trait]
    impl EmbeddingProvider for FakeEmbedder {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            Ok(texts.iter().map(|t| bow(t, self.dim)).collect())
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

    // ── Store helpers ─────────────────────────────────────────────────────

    fn make_test_store() -> (tempfile::NamedTempFile, SqliteVectorStore) {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mem = EncryptedSqliteMemory::new(tmp.path().to_path_buf(), "pw".into()).unwrap();
        let store = SqliteVectorStore::new(mem.shared_conn(), mem.data_key()).unwrap();
        (tmp, store)
    }

    /// Inserts a memory with a FakeEmbedder-produced embedding.
    async fn insert_mem(
        store: &SqliteVectorStore,
        id: &str,
        text: &str,
        embedder: &FakeEmbedder,
        created_at: i64,
        last_accessed_at: i64,
        salience: f64,
    ) {
        let emb = bow(text, embedder.dim);
        let m = Memory {
            id: id.into(),
            session_id: "s".into(),
            kind: MemoryKind::Episodic,
            text: text.into(),
            embedding: emb,
            model_id: embedder.model_id().into(),
            dim: embedder.dim(),
            created_at,
            salience,
            access_count: 0,
            last_accessed_at,
            superseded_by: None,
            evicted_at: None,
            scope: "root".into(),
            distilled_at: None,
        };
        store.insert(&m).await.unwrap();
    }

    // ── SC-06 ─────────────────────────────────────────────────────────────

    /// SC-06: a relevant memory planted among distractors surfaces in top-k.
    #[tokio::test]
    async fn test_relevant_memory_surfaces_above_distractors() {
        let (_tmp, store) = make_test_store();
        let emb = FakeEmbedder {
            dim: 32,
            model: "fake".into(),
        };
        let clock = FixedClock::new(1_000_000);
        let cfg = MemoryConfig {
            top_k: 5,
            ..MemoryConfig::default()
        };

        // Plant the target.
        insert_mem(
            &store,
            "target",
            "context budget is 8000 tokens",
            &emb,
            1000,
            1000,
            0.5,
        )
        .await;

        // 20 distractors with unrelated words.
        for i in 0..20 {
            insert_mem(
                &store,
                &format!("d{i}"),
                &format!("unrelated network latency distractor {i}"),
                &emb,
                900,
                900,
                0.3,
            )
            .await;
        }

        let results = recall(&store, &emb, &clock, &cfg, "what is the budget", 0, "root")
            .await
            .unwrap();

        assert!(
            !results.is_empty(),
            "recall must return at least one result"
        );
        assert!(
            results.iter().any(|r| r.memory.id == "target"),
            "the target memory should surface in top-{} (SC-06)",
            cfg.top_k
        );
    }

    // ── SC-07 ─────────────────────────────────────────────────────────────

    /// SC-07 + R-06: the more recent memory ranks higher when cosine similarity
    /// is equal; two consecutive calls return the same id order.
    #[tokio::test]
    async fn test_recency_breaks_ties_deterministically() {
        let (_tmp, store) = make_test_store();
        let emb = FakeEmbedder {
            dim: 32,
            model: "fake".into(),
        };
        // Use a fixed "current time" far in the future so ages are large and different.
        let now = 1_000_000i64;
        let clock = FixedClock::new(now);
        let cfg = MemoryConfig {
            top_k: 2,
            ..MemoryConfig::default()
        };

        // Identical text → same cosine with any query.
        // "new" was accessed more recently → lower age → higher recency score.
        insert_mem(&store, "old", "the budget policy", &emb, 500, 100, 0.5).await;
        insert_mem(&store, "new", "the budget policy", &emb, 1000, 900_000, 0.5).await;

        let r1 = recall(&store, &emb, &clock, &cfg, "budget policy", 0, "root")
            .await
            .unwrap();
        assert_eq!(r1.len(), 2, "both memories should be returned");
        assert_eq!(
            r1[0].memory.id, "new",
            "more recently accessed memory should rank first (SC-07)"
        );

        // After the first recall, mark_accessed sets both to now=1_000_000.
        // On the second call they tie on recency → tie-break by created_at desc
        // ("new" has created_at=1000 > 500) → same winner.
        let r2 = recall(&store, &emb, &clock, &cfg, "budget policy", 0, "root")
            .await
            .unwrap();
        assert_eq!(
            r2[0].memory.id, r1[0].memory.id,
            "R-06: deterministic ordering across consecutive calls"
        );
    }

    // ── SC-33 ─────────────────────────────────────────────────────────────

    /// SC-33: memories with a foreign model_id are excluded; only the current
    /// model's embeddings are returned.
    #[tokio::test]
    async fn test_model_dim_filter_excludes_foreign_vectors() {
        let (_tmp, store) = make_test_store();
        let emb_a = FakeEmbedder {
            dim: 32,
            model: "model-A".into(),
        };
        let emb_b = FakeEmbedder {
            dim: 32,
            model: "model-B".into(),
        };
        let clock = FixedClock::new(1_000_000);
        let cfg = MemoryConfig {
            top_k: 5,
            ..MemoryConfig::default()
        };

        insert_mem(&store, "m_a", "budget policy", &emb_a, 1000, 1000, 0.5).await;
        insert_mem(&store, "m_b", "budget policy", &emb_b, 1000, 1000, 0.5).await;

        // Recall with embedder_b: only model-B memories should appear (D-06, SC-33).
        let results = recall(&store, &emb_b, &clock, &cfg, "budget", 0, "root")
            .await
            .unwrap();

        assert!(
            results.iter().all(|r| r.memory.model_id == "model-B"),
            "all returned memories must have the current model_id"
        );
        assert!(
            results.iter().any(|r| r.memory.id == "m_b"),
            "model-B memory should appear in results"
        );
        assert!(
            !results.iter().any(|r| r.memory.id == "m_a"),
            "model-A memory must be excluded (D-06)"
        );
    }

    // ── SC-40 ─────────────────────────────────────────────────────────────

    /// SC-40: `recall` is callable directly without any assembler — it is the
    /// stable B1 seam (D-13 / REQ-37).
    #[tokio::test]
    async fn test_recall_is_public_and_independent_of_assembler() {
        let (_tmp, store) = make_test_store();
        let emb = FakeEmbedder {
            dim: 32,
            model: "fake".into(),
        };
        let clock = FixedClock::new(1_000_000);
        let cfg = MemoryConfig {
            top_k: 3,
            ..MemoryConfig::default()
        };

        insert_mem(
            &store,
            "m1",
            "the context budget is 8000 tokens",
            &emb,
            1000,
            1000,
            0.5,
        )
        .await;
        insert_mem(&store, "m2", "unrelated distractor", &emb, 1000, 1000, 0.3).await;

        // Call recall directly — no assembler involved (SC-40).
        let results = recall(&store, &emb, &clock, &cfg, "budget tokens", 0, "root")
            .await
            .unwrap();

        assert!(
            !results.is_empty(),
            "recall must return at least one result (SC-40)"
        );
        assert!(
            results[0].score > 0.0,
            "ranked memories must have a positive score"
        );
        assert_eq!(
            results[0].memory.id, "m1",
            "the relevant memory should be ranked first"
        );
    }

    // ── CP2-C / SC-08 ─────────────────────────────────────────────────────

    /// CP2-C: memories with an empty embedding (or foreign model) get re-embedded
    /// by `reembed_pending`; afterwards all active memories carry the current model.
    #[tokio::test]
    async fn test_pending_vectors_get_reembedded_with_current_model() {
        let (_tmp, store) = make_test_store();
        let emb = FakeEmbedder {
            dim: 32,
            model: "fake".into(),
        };
        let cfg = MemoryConfig {
            reembed_batch_size: 10,
            ..MemoryConfig::default()
        };

        // Insert a memory with no embedding (pending).
        let pending = Memory {
            id: "pending".into(),
            session_id: "s".into(),
            kind: MemoryKind::Episodic,
            text: "some text to re-embed".into(),
            embedding: vec![], // empty = pending (SC-08)
            model_id: "".into(),
            dim: 0,
            created_at: 1000,
            salience: 0.3,
            access_count: 0,
            last_accessed_at: 1000,
            superseded_by: None,
            evicted_at: None,
            scope: "root".into(),
            distilled_at: None,
        };
        store.insert(&pending).await.unwrap();

        let count = reembed_pending(&store, &emb, &cfg, "root").await.unwrap();
        assert_eq!(count, 1, "should re-embed 1 pending memory");

        let updated = store.get("pending").await.unwrap().unwrap();
        assert!(
            !updated.embedding.is_empty(),
            "embedding must be non-empty after reembed_pending"
        );
        assert_eq!(
            updated.model_id,
            emb.model_id(),
            "model_id must match the embedder after re-embedding"
        );
        assert_eq!(
            updated.dim,
            emb.dim(),
            "dim must match the embedder after re-embedding"
        );
    }

    // ── CP2-E ─────────────────────────────────────────────────────────────

    /// CP2-E: evicted and superseded memories are never returned by `recall`.
    #[tokio::test]
    async fn test_index_only_holds_active_set() {
        let (_tmp, store) = make_test_store();
        let emb = FakeEmbedder {
            dim: 32,
            model: "fake".into(),
        };
        let clock = FixedClock::new(1_000_000);
        let cfg = MemoryConfig {
            top_k: 10,
            ..MemoryConfig::default()
        };

        insert_mem(
            &store,
            "active",
            "the budget is 8000",
            &emb,
            1000,
            1000,
            0.5,
        )
        .await;

        // Insert + evict.
        insert_mem(
            &store,
            "evicted",
            "budget limit exceeded",
            &emb,
            1000,
            1000,
            0.5,
        )
        .await;
        store.set_evicted("evicted", Some(999)).await.unwrap();

        // Insert + supersede.
        insert_mem(&store, "old_fact", "budget was 6000", &emb, 900, 900, 0.5).await;
        insert_mem(
            &store,
            "new_fact",
            "budget is 8000 now",
            &emb,
            1000,
            1000,
            0.5,
        )
        .await;
        store.set_superseded("old_fact", "new_fact").await.unwrap();

        let results = recall(&store, &emb, &clock, &cfg, "what is the budget", 0, "root")
            .await
            .unwrap();

        assert!(
            !results.iter().any(|r| r.memory.id == "evicted"),
            "evicted memory must not appear (CP2-E)"
        );
        assert!(
            !results.iter().any(|r| r.memory.id == "old_fact"),
            "superseded memory must not appear (CP2-E)"
        );
        assert!(
            results.iter().any(|r| r.memory.id == "active"),
            "active memory must appear"
        );
        assert!(
            results.iter().any(|r| r.memory.id == "new_fact"),
            "the superseding memory (new_fact) must appear"
        );
    }
}
