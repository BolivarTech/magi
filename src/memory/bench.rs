// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-06-27

//! Benchmark harness for tiered-memory evaluation (T14, D-08, REQ-23'/24'/25').
//!
//! Provides a [`DeterministicEmbedder`] (no network, pure SHA-256 word-bag vectors)
//! and the two-arm harness ([`run_full_benchmark`]) that measures recall accuracy,
//! staleness rate, and mean context tokens for `load_all` vs `selective` mode.
//!
//! # Usage (tests)
//! ```
//! cargo nextest run memory::bench::tests
//! ```
//!
//! # Usage (binary)
//! ```
//! cargo run --bin bench_memory
//! ```

use async_trait::async_trait;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

use crate::memory::clock::Clock;
use crate::memory::config::MemoryConfig;
use crate::memory::embedding::EmbeddingProvider;
use crate::memory::error::{EmbeddingError, MemoryError};
use crate::memory::retrieval::recall;
use crate::memory::store::{Memory, VectorStore};
use crate::memory::tokens::estimate_tokens;
use crate::memory::MemoryKind;

// ─── Dataset types ─────────────────────────────────────────────────────────────

/// A single planted fact in the benchmark dataset (REQ-23').
#[derive(Debug, Clone)]
pub struct BenchFact {
    /// Stable identifier used in probe expectations.
    pub id: String,
    /// The fact's plain text, stored as a memory.
    pub text: String,
    /// If set, this fact supersedes the fact with the given id (REQ-10).
    pub supersedes: Option<String>,
}

/// A probe question for one benchmark query (REQ-23', SC-28).
#[derive(Debug, Clone)]
pub struct BenchProbe {
    /// Stable identifier matching [`BenchProbeResult::probe_id`].
    pub id: String,
    /// The natural-language query submitted to the retrieval system.
    pub query: String,
    /// Fact ids that MUST appear in the retrieved context for this probe to be
    /// counted as a hit (recall). The check is substring: fact text ⊆ context.
    pub expected_fact_ids: Vec<String>,
    /// Fact ids that are stale (superseded). Their presence in the context is
    /// counted as a staleness hit.
    pub staleness_fact_ids: Vec<String>,
}

/// Synthetic benchmark dataset (REQ-23', R-08).
#[derive(Debug)]
pub struct BenchmarkDataset {
    /// All planted facts (both current and superseded).
    pub facts: Vec<BenchFact>,
    /// Probe questions for session B evaluation.
    pub probes: Vec<BenchProbe>,
}

impl BenchmarkDataset {
    /// Returns the text for `fact_id`, or `""` if not found.
    // Narrow allow: utility method kept for future probes that verify text overlap.
    #[allow(dead_code)]
    fn fact_text(&self, fact_id: &str) -> &str {
        self.facts
            .iter()
            .find(|f| f.id == fact_id)
            .map(|f| f.text.as_str())
            .unwrap_or("")
    }
}

// ─── Result types ──────────────────────────────────────────────────────────────

/// Result for a single probe in one benchmark arm (REQ-25').
#[derive(Debug, Clone)]
pub struct BenchProbeResult {
    /// Matches [`BenchProbe::id`].
    pub probe_id: String,
    /// Fact ids whose text appears in the assembled context for this probe.
    pub retrieved_fact_ids: Vec<String>,
    /// Estimated token count of the assembled context (D-02/D-16).
    pub context_tokens: usize,
}

/// Per-arm aggregate metrics (REQ-25').
#[derive(Debug, Clone)]
pub struct ArmReport {
    /// Fraction of probes where all expected facts appear in context (`[0,1]`).
    pub recall_accuracy: f64,
    /// Fraction of probes where at least one stale (superseded) fact appears (`[0,1]`).
    pub staleness_rate: f64,
    /// Mean context tokens across all probes.
    pub mean_context_tokens: f64,
}

/// Full benchmark report comparing both arms (SC-27/SC-29).
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    /// Legacy `load_all` arm: all conversation history injected into context.
    pub load_all: ArmReport,
    /// Tiered `selective` arm: top-k recalled memories injected into context.
    pub selective: ArmReport,
}

// ─── DeterministicEmbedder ─────────────────────────────────────────────────────

/// Pure-SHA-256 word-bag embedder for use in benchmarks and tests (CP2-O, R-06).
///
/// Produces deterministic vectors without any network call. Each word in the
/// input text is hashed to a bucket index; the resulting bag-of-words vector is
/// L2-normalised. Semantically similar texts sharing keywords get high cosine
/// similarity — sufficient for the synthetic benchmark dataset.
pub struct DeterministicEmbedder {
    dim: usize,
    model: String,
}

impl DeterministicEmbedder {
    /// Creates a new embedder with the given vector dimension and model label.
    pub fn new(dim: usize, model: impl Into<String>) -> Self {
        Self {
            dim,
            model: model.into(),
        }
    }

    /// Produces a normalised word-bag vector for `text` using SHA-256 hashing.
    fn word_bag(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dim];
        for word in text.split_whitespace() {
            let hash = Sha256::digest(word.to_lowercase().as_bytes());
            // Use the first 8 bytes as a little-endian u64 index.
            let idx = u64::from_le_bytes(hash[..8].try_into().expect("sha256 ≥ 8 bytes")) as usize
                % self.dim;
            v[idx] += 1.0;
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-9 {
            v.iter_mut().for_each(|x| *x /= norm);
        }
        v
    }
}

#[async_trait]
impl EmbeddingProvider for DeterministicEmbedder {
    /// Returns word-bag vectors; no network call (CP2-O, R-06).
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        Ok(texts.iter().map(|t| self.word_bag(t)).collect())
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

// ─── Pure metric functions ─────────────────────────────────────────────────────

/// Fraction of probes where ALL `expected_fact_ids` appear in the retrieved context.
///
/// Returns `0.0` for an empty result slice (SC-28).
pub fn compute_recall_accuracy(dataset: &BenchmarkDataset, results: &[BenchProbeResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    let hits = results
        .iter()
        .filter(|r| {
            let Some(probe) = dataset.probes.iter().find(|p| p.id == r.probe_id) else {
                return false;
            };
            probe
                .expected_fact_ids
                .iter()
                .all(|eid| r.retrieved_fact_ids.contains(eid))
        })
        .count();
    hits as f64 / results.len() as f64
}

/// Fraction of probes where at least one `staleness_fact_id` appears in the retrieved context.
///
/// Returns `0.0` for an empty result slice (SC-28).
pub fn compute_staleness_rate(dataset: &BenchmarkDataset, results: &[BenchProbeResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    let stale = results
        .iter()
        .filter(|r| {
            let Some(probe) = dataset.probes.iter().find(|p| p.id == r.probe_id) else {
                return false;
            };
            probe
                .staleness_fact_ids
                .iter()
                .any(|sid| r.retrieved_fact_ids.contains(sid))
        })
        .count();
    stale as f64 / results.len() as f64
}

/// Mean context token count across all probe results.
///
/// Returns `0.0` for an empty result slice (SC-28).
pub fn compute_mean_context_tokens(results: &[BenchProbeResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    let total: usize = results.iter().map(|r| r.context_tokens).sum();
    total as f64 / results.len() as f64
}

// ─── Harness internals ─────────────────────────────────────────────────────────

/// Populates `store` with all `dataset` facts, applying supersession where declared.
async fn populate_store(
    dataset: &BenchmarkDataset,
    store: &dyn VectorStore,
    embedder: &dyn EmbeddingProvider,
    clock: &dyn Clock,
) -> Result<(), MemoryError> {
    let base_ts = clock.now();
    // Map fact_id → memory record id for supersession links.
    let mut id_map: HashMap<String, String> = HashMap::new();

    for (i, fact) in dataset.facts.iter().enumerate() {
        let mem_id = format!("bench-{}", fact.id);
        id_map.insert(fact.id.clone(), mem_id.clone());

        let doc_prefix = embedder.document_prefix().to_string();
        let prefixed = if doc_prefix.is_empty() {
            fact.text.clone()
        } else {
            format!("{doc_prefix}{}", fact.text)
        };
        let embedding = embedder
            .embed(&[prefixed])
            .await
            .map_err(MemoryError::Embedding)?
            .pop()
            .unwrap_or_default();

        let memory = Memory {
            id: mem_id.clone(),
            session_id: "bench-session".into(),
            kind: MemoryKind::Episodic,
            text: fact.text.clone(),
            embedding,
            model_id: embedder.model_id().to_string(),
            dim: embedder.dim(),
            created_at: base_ts + i as i64,
            salience: 0.5,
            access_count: 0,
            last_accessed_at: base_ts + i as i64,
            superseded_by: None,
            evicted_at: None,
            scope: "root".into(),
            distilled_at: None,
        };
        // Idempotent: skip silently if the memory id already exists
        // (e.g. on the second run of run_full_benchmark on the same store — SC-27).
        match store.insert(&memory).await {
            Ok(()) => {}
            Err(MemoryError::Storage(ref e)) if e.contains("UNIQUE constraint") => {}
            Err(e) => return Err(e),
        }
    }

    // Apply supersession in a second pass (so both records exist).
    for fact in &dataset.facts {
        if let Some(old_id) = &fact.supersedes {
            if let (Some(old_mem_id), Some(new_mem_id)) =
                (id_map.get(old_id.as_str()), id_map.get(&fact.id))
            {
                store.set_superseded(old_mem_id, new_mem_id).await?;
            }
        }
    }

    Ok(())
}

/// Simulates the `load_all` arm: every dataset fact is present in the context.
///
/// This mirrors the v0.6.0 behavior where `Agent::load_history` injects ALL
/// conversation turns — including superseded ones — into the prompt (REQ-15/SC-19).
fn run_load_all_arm(dataset: &BenchmarkDataset, cfg: &MemoryConfig) -> Vec<BenchProbeResult> {
    let all_texts: String = dataset
        .facts
        .iter()
        .map(|f| f.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let total_tokens = estimate_tokens(&all_texts, cfg.chars_per_token);

    dataset
        .probes
        .iter()
        .map(|probe| {
            let retrieved_fact_ids = dataset.facts.iter().map(|f| f.id.clone()).collect();
            BenchProbeResult {
                probe_id: probe.id.clone(),
                retrieved_fact_ids,
                context_tokens: total_tokens,
            }
        })
        .collect()
}

/// Runs the `selective` arm: top-k memories recalled per probe via ANN + reranker.
///
/// Only `active()` memories are considered (no superseded or evicted facts),
/// reducing staleness to zero when supersession is correctly applied (SC-12).
async fn run_selective_arm(
    dataset: &BenchmarkDataset,
    store: &dyn VectorStore,
    embedder: &dyn EmbeddingProvider,
    clock: &dyn Clock,
    cfg: &MemoryConfig,
) -> Result<Vec<BenchProbeResult>, MemoryError> {
    let mut results = Vec::new();
    for probe in &dataset.probes {
        // C2: pass token budget (assembler semantics) not top_k (count semantics).
        let ranked = recall(
            store,
            embedder,
            clock,
            cfg,
            &probe.query,
            cfg.context_budget_tokens,
            "root",
        )
        .await?;
        let context_text: String = ranked
            .iter()
            .map(|rm| rm.memory.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let context_tokens = estimate_tokens(&context_text, cfg.chars_per_token);

        // Map recalled memory texts back to fact ids.
        let retrieved_fact_ids: Vec<String> = ranked
            .iter()
            .filter_map(|rm| {
                dataset
                    .facts
                    .iter()
                    .find(|f| f.text == rm.memory.text)
                    .map(|f| f.id.clone())
            })
            .collect();

        results.push(BenchProbeResult {
            probe_id: probe.id.clone(),
            retrieved_fact_ids,
            context_tokens,
        });
    }
    Ok(results)
}

// ─── Public harness ────────────────────────────────────────────────────────────

/// Runs both benchmark arms against `dataset` and returns a [`BenchmarkReport`].
///
/// Populates `store` with all dataset facts (including supersession), then
/// evaluates both `load_all` and `selective` modes and computes the three
/// headline metrics (REQ-24', SC-27).
///
/// Uses `embedder` (typically [`DeterministicEmbedder`]) for vector operations
/// so no network calls occur during a benchmark run (CP2-O, R-06).
///
/// # Errors
/// [`MemoryError::Storage`] / [`MemoryError::Crypto`] from the store, or
/// [`MemoryError::Embedding`] if the embedder fails (unexpected in the
/// deterministic embedder path).
pub async fn run_full_benchmark(
    dataset: &BenchmarkDataset,
    store: &dyn VectorStore,
    embedder: &dyn EmbeddingProvider,
    clock: &dyn Clock,
    cfg: &MemoryConfig,
) -> Result<BenchmarkReport, MemoryError> {
    populate_store(dataset, store, embedder, clock).await?;

    let load_all_results = run_load_all_arm(dataset, cfg);
    let selective_results = run_selective_arm(dataset, store, embedder, clock, cfg).await?;

    Ok(BenchmarkReport {
        load_all: ArmReport {
            recall_accuracy: compute_recall_accuracy(dataset, &load_all_results),
            staleness_rate: compute_staleness_rate(dataset, &load_all_results),
            mean_context_tokens: compute_mean_context_tokens(&load_all_results),
        },
        selective: ArmReport {
            recall_accuracy: compute_recall_accuracy(dataset, &selective_results),
            staleness_rate: compute_staleness_rate(dataset, &selective_results),
            mean_context_tokens: compute_mean_context_tokens(&selective_results),
        },
    })
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    use crate::memory::clock::FixedClock;
    use crate::memory::config::MemoryConfig;
    use crate::memory::store::SqliteVectorStore;
    use crate::system::database::EncryptedSqliteMemory;

    // ── Dataset helpers ──────────────────────────────────────────────────────

    fn make_small_dataset() -> BenchmarkDataset {
        BenchmarkDataset {
            facts: vec![
                BenchFact {
                    id: "pref_rust".into(),
                    text: "I prefer Rust over Python for systems programming".into(),
                    supersedes: None,
                },
                BenchFact {
                    id: "pref_dark".into(),
                    text: "I use dark mode in all editors and terminals".into(),
                    supersedes: None,
                },
                BenchFact {
                    id: "editor_old".into(),
                    text: "I use vim as my primary editor".into(),
                    supersedes: None,
                },
                BenchFact {
                    id: "editor_new".into(),
                    text: "I switched from vim to neovim".into(),
                    supersedes: Some("editor_old".into()),
                },
                BenchFact {
                    id: "distractor_1".into(),
                    text: "The capital of France is Paris".into(),
                    supersedes: None,
                },
                BenchFact {
                    id: "distractor_2".into(),
                    text: "Water boils at one hundred degrees Celsius".into(),
                    supersedes: None,
                },
            ],
            probes: vec![
                BenchProbe {
                    id: "q_rust".into(),
                    query: "Rust Python systems programming".into(),
                    expected_fact_ids: vec!["pref_rust".into()],
                    staleness_fact_ids: vec![],
                },
                BenchProbe {
                    id: "q_editor".into(),
                    query: "editor vim neovim".into(),
                    expected_fact_ids: vec!["editor_new".into()],
                    staleness_fact_ids: vec!["editor_old".into()],
                },
                BenchProbe {
                    id: "q_dark".into(),
                    query: "dark mode editors terminals".into(),
                    expected_fact_ids: vec!["pref_dark".into()],
                    staleness_fact_ids: vec![],
                },
            ],
        }
    }

    fn make_small_cfg() -> MemoryConfig {
        MemoryConfig {
            top_k: 2,
            chars_per_token: 4.0,
            ..Default::default()
        }
    }

    /// Convenience to open a fresh in-memory store backed by a tempfile DB.
    fn open_store() -> (NamedTempFile, Arc<SqliteVectorStore>) {
        let tmp = NamedTempFile::new().unwrap();
        let mem = EncryptedSqliteMemory::new(
            tmp.path().to_path_buf(),
            zeroize::Zeroizing::new("benchpw".to_string()),
        )
        .unwrap();
        let store =
            Arc::new(SqliteVectorStore::new(mem.shared_conn(), mem.data_key().unwrap()).unwrap());
        (tmp, store)
    }

    // ── SC-28 — hand-calculable unit tests on pure metric functions ──────────

    /// `compute_recall_accuracy` returns the fraction of probes where ALL
    /// expected fact ids appear in `retrieved_fact_ids` (SC-28).
    #[test]
    fn test_recall_accuracy_hand_calculated() {
        let dataset = BenchmarkDataset {
            facts: vec![
                BenchFact {
                    id: "f1".into(),
                    text: "fact one".into(),
                    supersedes: None,
                },
                BenchFact {
                    id: "f2".into(),
                    text: "fact two".into(),
                    supersedes: None,
                },
            ],
            probes: vec![
                BenchProbe {
                    id: "q1".into(),
                    query: "one".into(),
                    expected_fact_ids: vec!["f1".into()],
                    staleness_fact_ids: vec![],
                },
                BenchProbe {
                    id: "q2".into(),
                    query: "two".into(),
                    expected_fact_ids: vec!["f2".into()],
                    staleness_fact_ids: vec![],
                },
            ],
        };
        let results = vec![
            BenchProbeResult {
                probe_id: "q1".into(),
                retrieved_fact_ids: vec!["f1".into()], // HIT
                context_tokens: 10,
            },
            BenchProbeResult {
                probe_id: "q2".into(),
                retrieved_fact_ids: vec![], // MISS
                context_tokens: 5,
            },
        ];
        let acc = compute_recall_accuracy(&dataset, &results);
        assert!(
            (acc - 0.5).abs() < 1e-9,
            "expected recall_accuracy=0.5 (1 hit / 2 probes), got {acc}"
        );
    }

    /// `compute_staleness_rate` returns the fraction of probes where any stale
    /// fact id appears in the retrieved context (SC-28).
    #[test]
    fn test_staleness_rate_hand_calculated() {
        let dataset = BenchmarkDataset {
            facts: vec![
                BenchFact {
                    id: "old".into(),
                    text: "old fact".into(),
                    supersedes: None,
                },
                BenchFact {
                    id: "new".into(),
                    text: "new fact".into(),
                    supersedes: Some("old".into()),
                },
            ],
            probes: vec![
                BenchProbe {
                    id: "q1".into(),
                    query: "fact".into(),
                    expected_fact_ids: vec!["new".into()],
                    staleness_fact_ids: vec!["old".into()],
                },
                BenchProbe {
                    id: "q2".into(),
                    query: "other".into(),
                    expected_fact_ids: vec![],
                    staleness_fact_ids: vec![],
                },
            ],
        };
        let results = vec![
            BenchProbeResult {
                probe_id: "q1".into(),
                retrieved_fact_ids: vec!["old".into()], // stale fact present
                context_tokens: 8,
            },
            BenchProbeResult {
                probe_id: "q2".into(),
                retrieved_fact_ids: vec![], // no stale
                context_tokens: 4,
            },
        ];
        let rate = compute_staleness_rate(&dataset, &results);
        assert!(
            (rate - 0.5).abs() < 1e-9,
            "expected staleness_rate=0.5 (1 stale / 2 probes), got {rate}"
        );
    }

    /// `compute_mean_context_tokens` averages the token counts (SC-28).
    #[test]
    fn test_mean_context_tokens_hand_calculated() {
        let results = vec![
            BenchProbeResult {
                probe_id: "q1".into(),
                retrieved_fact_ids: vec![],
                context_tokens: 10,
            },
            BenchProbeResult {
                probe_id: "q2".into(),
                retrieved_fact_ids: vec![],
                context_tokens: 20,
            },
        ];
        let mean = compute_mean_context_tokens(&results);
        assert!(
            (mean - 15.0).abs() < 1e-9,
            "expected mean_context_tokens=15.0, got {mean}"
        );
    }

    // ── CP2-O — DeterministicEmbedder makes no network calls ────────────────

    /// `DeterministicEmbedder` produces unit-length vectors without any
    /// network call; the embedder's model and dim are accessible (CP2-O, R-06).
    #[tokio::test]
    async fn test_benchmark_uses_deterministic_embedder() {
        let emb = DeterministicEmbedder::new(64, "det-test");
        assert_eq!(emb.model_id(), "det-test");
        assert_eq!(emb.dim(), 64);

        // embed() returns vectors without making any HTTP call.
        let texts = vec!["hello world".to_string(), "rust programming".to_string()];
        let vecs = emb
            .embed(&texts)
            .await
            .expect("embed must succeed without network");
        assert_eq!(vecs.len(), 2, "one vector per input text");
        assert_eq!(vecs[0].len(), 64, "vector length = dim");
        assert_eq!(vecs[1].len(), 64, "vector length = dim");

        // Vectors are normalised (L2-norm ≈ 1.0).
        let norm: f32 = vecs[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "word-bag vector must be L2-normalised; got norm={norm}"
        );

        // Same text → same vector (deterministic, R-06).
        let v1 = emb.embed(&["rust".to_string()]).await.unwrap().remove(0);
        let v2 = emb.embed(&["rust".to_string()]).await.unwrap().remove(0);
        assert_eq!(
            v1, v2,
            "embedding must be deterministic under the same input"
        );
    }

    // ── SC-27 — harness runs both arms and emits a report ───────────────────

    /// The harness runs both arms (`load_all` and `selective`) without error
    /// and produces a [`BenchmarkReport`] with valid metric fields (SC-27, REQ-24').
    #[tokio::test]
    async fn test_benchmark_runs_both_arms_and_emits_report() {
        let dataset = make_small_dataset();
        let cfg = make_small_cfg();
        let emb = Arc::new(DeterministicEmbedder::new(128, "det-bench"));
        let clk = Arc::new(FixedClock::new(1_000_000));
        let (_tmp, store) = open_store();

        let report = run_full_benchmark(&dataset, &*store, &*emb, &*clk, &cfg)
            .await
            .expect("run_full_benchmark must succeed with DeterministicEmbedder");

        // Both arms produce valid metrics in [0,1].
        assert!(
            report.load_all.recall_accuracy >= 0.0 && report.load_all.recall_accuracy <= 1.0,
            "load_all recall out of range: {}",
            report.load_all.recall_accuracy
        );
        assert!(
            report.selective.recall_accuracy >= 0.0 && report.selective.recall_accuracy <= 1.0,
            "selective recall out of range: {}",
            report.selective.recall_accuracy
        );
        assert!(
            report.load_all.staleness_rate >= 0.0 && report.load_all.staleness_rate <= 1.0,
            "load_all staleness out of range: {}",
            report.load_all.staleness_rate
        );
        assert!(
            report.selective.staleness_rate >= 0.0 && report.selective.staleness_rate <= 1.0,
            "selective staleness out of range: {}",
            report.selective.staleness_rate
        );
        assert!(
            report.load_all.mean_context_tokens > 0.0,
            "load_all must have non-zero context tokens (all facts injected)"
        );
    }

    // ── SC-27 — reproducibility under same store ─────────────────────────────

    /// Running the harness twice on the same populated store produces identical
    /// results (deterministic seed + embedder, R-06).
    #[tokio::test]
    async fn test_benchmark_is_reproducible_under_same_seed() {
        let dataset = make_small_dataset();
        let cfg = make_small_cfg();
        let emb = Arc::new(DeterministicEmbedder::new(128, "det-bench"));
        let clk = Arc::new(FixedClock::new(1_000_000));
        let (_tmp, store) = open_store();

        // First run: populate and measure.
        let r1 = run_full_benchmark(&dataset, &*store, &*emb, &*clk, &cfg)
            .await
            .unwrap();

        // Second run on the SAME store (duplicate inserts are silently skipped by
        // SQLite UNIQUE constraint on `id`; we re-run to verify metric stability).
        let r2 = run_full_benchmark(&dataset, &*store, &*emb, &*clk, &cfg)
            .await
            .unwrap();

        assert!(
            (r1.selective.recall_accuracy - r2.selective.recall_accuracy).abs() < 1e-9,
            "selective recall must be identical across runs: {} vs {}",
            r1.selective.recall_accuracy,
            r2.selective.recall_accuracy
        );
        assert!(
            (r1.selective.staleness_rate - r2.selective.staleness_rate).abs() < 1e-9,
            "selective staleness must be identical across runs"
        );
    }

    // ── SC-29 — selective ≥ load_all recall, bounded tokens, lower staleness ─

    /// On the synthetic dataset, `selective` mode achieves recall ≥ `load_all`,
    /// uses fewer context tokens, and has a lower staleness rate (SC-29, REQ-25').
    #[tokio::test]
    async fn test_selective_outperforms_load_all_on_dataset() {
        let dataset = make_small_dataset();
        let cfg = MemoryConfig {
            top_k: 2,
            chars_per_token: 4.0,
            ..Default::default()
        };
        let emb = Arc::new(DeterministicEmbedder::new(128, "det-bench"));
        let clk = Arc::new(FixedClock::new(1_000_000));
        let (_tmp, store) = open_store();

        let report = run_full_benchmark(&dataset, &*store, &*emb, &*clk, &cfg)
            .await
            .expect("run_full_benchmark must succeed");

        // SC-29 headline assertions.
        assert!(
            report.selective.recall_accuracy >= report.load_all.recall_accuracy,
            "selective recall ({}) must be ≥ load_all recall ({})",
            report.selective.recall_accuracy,
            report.load_all.recall_accuracy
        );
        assert!(
            report.selective.mean_context_tokens < report.load_all.mean_context_tokens,
            "selective context tokens ({:.1}) must be < load_all ({:.1}) — bounded window",
            report.selective.mean_context_tokens,
            report.load_all.mean_context_tokens
        );
        assert!(
            report.selective.staleness_rate <= report.load_all.staleness_rate,
            "selective staleness ({}) must be ≤ load_all ({}) — supersession excludes stale facts",
            report.selective.staleness_rate,
            report.load_all.staleness_rate
        );

        // Verify load_all has non-zero staleness (editor_old is stale in context).
        assert!(
            report.load_all.staleness_rate > 0.0,
            "load_all must exhibit staleness for the superseded editor fact; got 0.0"
        );
        // Verify selective has zero staleness (editor_old is superseded → excluded from active()).
        assert!(
            report.selective.staleness_rate == 0.0,
            "selective must have zero staleness since superseded facts are excluded from active(); \
             got {}",
            report.selective.staleness_rate
        );
    }
}
