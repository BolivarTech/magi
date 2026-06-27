// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-06-27

//! Preference distiller: promotes durable preferences from episodic memory to a
//! compact profile, manages hard supersession, and renders the always-injected
//! preference context (D-11/D-12/D-15/REQ-16/REQ-17).
//!
//! # Design
//! - The LLM judgment surface is isolated behind [`DistillJudge`] so that the
//!   non-determinism is mockable in tests (R-06).
//! - [`promote_to_profile`] is an internal reusable primitive (D-15): the distiller
//!   and future Agent Society consumers can call it without going through the full
//!   distillation pass.
//! - [`render_profile`] produces the bounded, always-injected preference string.

use async_trait::async_trait;
use sha2::{Digest, Sha256};

use crate::memory::clock::Clock;
use crate::memory::config::MemoryConfig;
use crate::memory::embedding::EmbeddingProvider;
use crate::memory::error::MemoryError;
use crate::memory::index::cosine;
use crate::memory::store::{Memory, VectorStore};
use crate::memory::tokens::{budget_after_margin, estimate_tokens};
use crate::memory::MemoryKind;

// ─── DistillJudge ─────────────────────────────────────────────────────────────

/// The LLM judgment surface, isolated so the non-determinism is mockable in
/// tests (R-06).
///
/// The real LLM-backed implementation (Task 13b) is injected at the call site;
/// tests use a [`SpyJudge`](tests::SpyJudge) that records calls and returns
/// canned values.
///
/// Both methods may be called with an empty `episodic` slice or empty strings
/// and must not panic in those cases.
// Narrow allow: trait consumed by the agent distiller trigger (Task 13b).
#[allow(dead_code)]
#[async_trait]
pub trait DistillJudge: Send + Sync {
    /// Summarize durable preferences from a batch of episodic memories.
    ///
    /// Returns a `Vec<String>` of preference statements (may be empty when no
    /// new durable preferences are found).
    ///
    /// # Errors
    /// [`MemoryError`] on any LLM failure. A failure here is **non-fatal**
    /// (CP2-Z): [`distill`] catches it, logs it, and leaves the batch
    /// undistilled so the next pass can retry it.
    async fn summarize_preferences(
        &self,
        episodic: &[Memory],
    ) -> Result<Vec<String>, MemoryError>;

    /// Returns `true` when memory `b` contradicts or supersedes memory `a`
    /// about the same subject.
    ///
    /// Both `a` and `b` are raw memory text from pairs whose embedding cosine
    /// similarity meets the `supersede_similarity_threshold` (D-12).
    ///
    /// # Errors
    /// [`MemoryError`] on any LLM failure. A failure here is **non-fatal**
    /// (CP2-Z): [`distill`] catches it per-pair, logs it, and continues with
    /// the next candidate pair.
    async fn contradicts(&self, a: &str, b: &str) -> Result<bool, MemoryError>;
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Runs the off-hot-path distillation pass (D-12/D-15).
///
/// **NO-OP** when `cfg.distill_enabled == false` (CP2-AB): zero judge calls
/// and zero egress.
///
/// Otherwise, operates on not-yet-distilled episodic memories
/// (`distilled_at IS NULL`) in deterministic FIFO order
/// (`created_at ASC, id ASC`) up to a `distill_max_batch_tokens` privacy cap
/// (CP2-L, guarded by the safety margin).
///
/// Steps:
/// 1. **Summarize** — call [`DistillJudge::summarize_preferences`] on the
///    batch. Each returned string is upserted as a `Preference` via
///    [`promote_to_profile`] (latest-wins dedup, salience lifted to the
///    protected tier — D-11/SC-38).
/// 2. **Hard supersession** (D-12) — among batch members that have embeddings
///    from the current `embedder`, for pairs with cosine similarity ≥
///    `supersede_similarity_threshold`, ask the judge [`contradicts`]; mark the
///    older `superseded_by` the newer. Capped at
///    `supersede_max_candidate_pairs` (CP2-U). Preferences use latest-wins
///    deterministically — no LLM needed.
/// 3. **Mark** the batch `distilled_at = now`.
///
/// Any judge error is **non-fatal** (CP2-Z): caught and logged; the affected
/// memories stay `distilled_at IS NULL` for the next pass to retry.
///
/// # Errors
/// [`MemoryError::Storage`] or [`MemoryError::Crypto`] from the store; never
/// from judge failures (those are caught).
// Narrow allow: wired into the agent in Task 13b.
#[allow(dead_code)]
pub async fn distill(
    _store: &dyn VectorStore,
    _judge: &dyn DistillJudge,
    _embedder: &dyn EmbeddingProvider,
    _clock: &dyn Clock,
    _cfg: &MemoryConfig,
    _scope: &str,
) -> Result<(), MemoryError> {
    unimplemented!("Task 13a GREEN: distill not yet implemented")
}

/// Renders the always-injected preference profile (REQ-16/SC-22).
///
/// Fetches all active `Preference`-kind memories in `scope`, sorts them by
/// priority (highest salience first, then newest first, then id ascending for
/// determinism), and emits a bullet-list string bounded to
/// `cfg.profile_max_tokens`. Entries are never split mid-line (CP2-AI).
///
/// Returns an empty `String` when there are no active preference memories.
///
/// # Errors
/// [`MemoryError::Crypto`] or [`MemoryError::Storage`] on store failure.
// Narrow allow: wired into the context assembler / agent in Task 13b.
#[allow(dead_code)]
pub async fn render_profile(
    _store: &dyn VectorStore,
    _cfg: &MemoryConfig,
    _scope: &str,
) -> Result<String, MemoryError> {
    unimplemented!("Task 13a GREEN: render_profile not yet implemented")
}

// ─── Internal reusable primitive ─────────────────────────────────────────────

/// Upserts a `Preference`-kind [`Memory`] for the preference string `pref`
/// (D-15, REQ-17, SC-38).
///
/// The memory id is deterministic:
/// ```text
/// id = format!("pref:{:x}", Sha256::digest(normalized))
/// ```
/// where `normalized = pref.to_lowercase().trim()`.
/// This gives **latest-wins upsert** semantics: `"Use Rust"` and `"use rust  "`
/// map to the same id, so the newer call always replaces the older one.
///
/// The stored `salience` is `cfg.preference_salience.clamp(0.0, 1.0)` — always
/// at the protected tier (`≥ protect_salience_threshold`), so this memory is
/// never evicted by decay (REQ-09/REQ-35/D-11).
///
/// The `embedding` is left empty (`vec![]`, `model_id = ""`, `dim = 0`) to be
/// populated lazily by `reembed_pending` later (SC-08/CP2-C).
///
/// # Errors
/// [`MemoryError::Storage`] on SQL failure; [`MemoryError::Crypto`] if text
/// encryption fails.
// Narrow allow: called by distill (this module) and future AS-REQ-10 consumers;
// not a public API.
#[allow(dead_code)]
async fn promote_to_profile(
    _store: &dyn VectorStore,
    _clock: &dyn Clock,
    _cfg: &MemoryConfig,
    _scope: &str,
    _pref: &str,
) -> Result<(), MemoryError> {
    unimplemented!("Task 13a GREEN: promote_to_profile not yet implemented")
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use sha2::{Digest, Sha256};

    use super::*;
    use crate::memory::clock::FixedClock;
    use crate::memory::config::MemoryConfig;
    use crate::memory::error::EmbeddingError;
    use crate::memory::store::{Memory, SqliteVectorStore};
    use crate::memory::MemoryKind;
    use crate::system::database::EncryptedSqliteMemory;

    // ── Bag-of-words helper (copied from retrieval tests, R-06) ──────────────

    /// L2-normalised bag-of-words over a fixed-dim hash. Texts sharing words
    /// produce vectors with high cosine similarity. Deterministic (R-06).
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

    // ── FakeEmbedder ─────────────────────────────────────────────────────────

    /// Fake embedder that computes `bow` in-process — no HTTP, fully
    /// deterministic (R-06). Mirrors the one in `retrieval::tests`.
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

    // ── SpyJudge ─────────────────────────────────────────────────────────────

    /// Spy `DistillJudge` that records call counts and returns canned values.
    ///
    /// `summarize_calls` and `contradicts_calls` are `Arc<Mutex<usize>>` so the
    /// counts can be read from the test after passing `&SpyJudge` to `distill`.
    pub(super) struct SpyJudge {
        /// Canned preferences to return from `summarize_preferences`.
        prefs: Vec<String>,
        /// If `true`, `summarize_preferences` returns `Err` (CP2-Z test).
        fail_summarize: bool,
        /// Canned return value for `contradicts` when `fail_contradicts = false`.
        contradicts_val: bool,
        /// If `true`, `contradicts` returns `Err` (CP2-Z test).
        fail_contradicts: bool,
        /// Counter of `summarize_preferences` calls.
        summarize_calls: Arc<Mutex<usize>>,
        /// Counter of `contradicts` calls.
        contradicts_calls: Arc<Mutex<usize>>,
    }

    #[async_trait]
    impl DistillJudge for SpyJudge {
        async fn summarize_preferences(
            &self,
            _episodic: &[Memory],
        ) -> Result<Vec<String>, MemoryError> {
            *self.summarize_calls.lock().unwrap() += 1;
            if self.fail_summarize {
                Err(MemoryError::Storage(
                    "spy: summarize_preferences forced error".into(),
                ))
            } else {
                Ok(self.prefs.clone())
            }
        }

        async fn contradicts(&self, _a: &str, _b: &str) -> Result<bool, MemoryError> {
            *self.contradicts_calls.lock().unwrap() += 1;
            if self.fail_contradicts {
                Err(MemoryError::Storage(
                    "spy: contradicts forced error".into(),
                ))
            } else {
                Ok(self.contradicts_val)
            }
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn make_test_store() -> (tempfile::NamedTempFile, SqliteVectorStore) {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mem = EncryptedSqliteMemory::new(tmp.path().to_path_buf(), "pw".into()).unwrap();
        let store = SqliteVectorStore::new(mem.shared_conn(), mem.data_key()).unwrap();
        (tmp, store)
    }

    async fn insert_episodic(
        store: &SqliteVectorStore,
        id: &str,
        text: &str,
        embedding: Vec<f32>,
        model_id: &str,
        dim: usize,
        created_at: i64,
    ) {
        let m = Memory {
            id: id.into(),
            session_id: "s".into(),
            kind: MemoryKind::Episodic,
            text: text.into(),
            embedding,
            model_id: model_id.into(),
            dim,
            created_at,
            salience: 0.3,
            access_count: 0,
            last_accessed_at: created_at,
            superseded_by: None,
            evicted_at: None,
            scope: "root".into(),
            distilled_at: None,
        };
        store.insert(&m).await.unwrap();
    }

    /// Builds a spy judge and returns it along with cloned counter handles.
    fn make_spy(
        prefs: Vec<String>,
        fail_summarize: bool,
        contradicts_val: bool,
        fail_contradicts: bool,
    ) -> (SpyJudge, Arc<Mutex<usize>>, Arc<Mutex<usize>>) {
        let summarize_calls = Arc::new(Mutex::new(0usize));
        let contradicts_calls = Arc::new(Mutex::new(0usize));
        let spy = SpyJudge {
            prefs,
            fail_summarize,
            contradicts_val,
            fail_contradicts,
            summarize_calls: Arc::clone(&summarize_calls),
            contradicts_calls: Arc::clone(&contradicts_calls),
        };
        (spy, summarize_calls, contradicts_calls)
    }

    // ── SC-20 ─────────────────────────────────────────────────────────────────

    /// SC-20: judge returns the same preference twice; `render_profile` contains
    /// it exactly once (latest-wins dedup via deterministic id).
    #[tokio::test]
    async fn test_distill_creates_one_deduped_profile_entry() {
        let (_tmp, store) = make_test_store();
        let emb = FakeEmbedder {
            dim: 8,
            model: "fake".into(),
        };
        let clock = FixedClock::new(1_000);
        let cfg = MemoryConfig {
            distill_enabled: true,
            ..MemoryConfig::default()
        };

        insert_episodic(&store, "e1", "some text", vec![], "", 0, 1_000).await;

        // Judge returns the same preference string twice → single deduped entry.
        let (spy, _, _) =
            make_spy(vec!["Use Rust".into(), "Use Rust".into()], false, false, false);

        distill(&store, &spy, &emb, &clock, &cfg, "root")
            .await
            .unwrap();

        let profile = render_profile(&store, &cfg, "root").await.unwrap();

        let occurrences = profile.matches("Use Rust").count();
        assert_eq!(
            occurrences, 1,
            "SC-20: 'Use Rust' should appear exactly once in the profile, got:\n{profile}"
        );
    }

    // ── SC-21 ─────────────────────────────────────────────────────────────────

    /// SC-21: two `promote_to_profile` calls with the same normalized text
    /// (different casing/whitespace) — the later one wins.
    #[tokio::test]
    async fn test_latest_wins_on_conflicting_preferences() {
        let (_tmp, store) = make_test_store();
        let clock1 = FixedClock::new(1_000);
        let clock2 = FixedClock::new(2_000);
        let cfg = MemoryConfig::default();

        // First call: "Use Rust" (normalized = "use rust").
        promote_to_profile(&store, &clock1, &cfg, "root", "Use Rust")
            .await
            .unwrap();
        // Second call: same normalized id, different casing — latest wins.
        promote_to_profile(&store, &clock2, &cfg, "root", "use rust  ")
            .await
            .unwrap();

        // Exactly one profile entry should exist.
        let profile = render_profile(&store, &cfg, "root").await.unwrap();
        let line_count = profile.lines().count();
        assert_eq!(
            line_count, 1,
            "SC-21: should be exactly one profile entry (dedup), got:\n{profile}"
        );
        // The stored text is the latest call's argument.
        assert!(
            profile.contains("use rust"),
            "SC-21: profile should contain the latest preference, got:\n{profile}"
        );
    }

    // ── SC-38 ─────────────────────────────────────────────────────────────────

    /// SC-38: a memory promoted via `promote_to_profile` has
    /// `salience ≥ protect_salience_threshold` and `kind = Preference`.
    #[tokio::test]
    async fn test_boost_lifts_preference_to_protected_tier() {
        let (_tmp, store) = make_test_store();
        let clock = FixedClock::new(1_000);
        let cfg = MemoryConfig::default();

        promote_to_profile(&store, &clock, &cfg, "root", "always use dark mode")
            .await
            .unwrap();

        // Reconstruct the deterministic id to look up the raw memory.
        let id = format!(
            "pref:{:x}",
            Sha256::digest("always use dark mode".as_bytes())
        );
        let mem = store
            .get(&id)
            .await
            .unwrap()
            .expect("promoted preference must exist in store");

        assert_eq!(
            mem.kind,
            MemoryKind::Preference,
            "SC-38: kind must be Preference"
        );
        assert!(
            mem.salience >= cfg.protect_salience_threshold,
            "SC-38: salience ({}) must be >= protect_salience_threshold ({}) (D-11)",
            mem.salience,
            cfg.protect_salience_threshold
        );
    }

    // ── SC-39 ─────────────────────────────────────────────────────────────────

    /// SC-39: F1 (older) and F2 (newer) have similar embeddings ≥ threshold;
    /// spy says contradicts=true; after distill, F1.superseded_by == "f2" and
    /// F1 is excluded from active retrieval.
    #[tokio::test]
    async fn test_hard_supersession_marks_older_superseded_by_newer() {
        let (_tmp, store) = make_test_store();
        let emb = FakeEmbedder {
            dim: 32,
            model: "fake".into(),
        };
        let clock = FixedClock::new(1_000);
        let cfg = MemoryConfig {
            supersede_similarity_threshold: 0.5, // low to survive hash collisions
            supersede_max_candidate_pairs: 50,
            distill_enabled: true,
            ..MemoryConfig::default()
        };

        // F1 and F2 share 9 of 10 words → cosine ≈ 0.9 >> 0.5 threshold.
        let f1_text = "user always prefers to use dark mode on all screens";
        let f2_text = "user always prefers to use light mode on all screens";
        insert_episodic(&store, "f1", f1_text, bow(f1_text, 32), "fake", 32, 1_000).await;
        insert_episodic(&store, "f2", f2_text, bow(f2_text, 32), "fake", 32, 2_000).await;

        let (spy, _, _) = make_spy(vec![], false, true /* contradicts=true */, false);

        distill(&store, &spy, &emb, &clock, &cfg, "root")
            .await
            .unwrap();

        // F1 (older) must be superseded_by F2 (newer).
        let f1 = store.get("f1").await.unwrap().expect("f1 must still exist");
        assert_eq!(
            f1.superseded_by,
            Some("f2".to_string()),
            "SC-39: f1 (older) must be superseded_by f2 (newer)"
        );

        // F1 must no longer appear in active retrieval.
        let active_ids: Vec<String> = store
            .active("root")
            .await
            .unwrap()
            .into_iter()
            .map(|m| m.id)
            .collect();
        assert!(
            !active_ids.contains(&"f1".to_string()),
            "SC-39: superseded f1 must not appear in active memories"
        );
    }

    // ── CP2-U ─────────────────────────────────────────────────────────────────

    /// CP2-U: `contradicts` is called at most `supersede_max_candidate_pairs`
    /// times even when more candidate pairs exceed the threshold.
    #[tokio::test]
    async fn test_candidate_pairs_are_capped() {
        let (_tmp, store) = make_test_store();
        let emb = FakeEmbedder {
            dim: 8,
            model: "fake".into(),
        };
        let clock = FixedClock::new(1_000);
        let cap = 2usize;
        let cfg = MemoryConfig {
            supersede_similarity_threshold: 0.01, // near-zero → all pairs qualify
            supersede_max_candidate_pairs: cap,
            distill_enabled: true,
            ..MemoryConfig::default()
        };

        // 4 memories with identical text → 6 candidate pairs (all qualify at threshold≈0).
        // With cap=2, only 2 contradicts calls should be made.
        let text = "context budget policy";
        for i in 0..4i64 {
            insert_episodic(
                &store,
                &format!("m{i}"),
                text,
                bow(text, 8),
                "fake",
                8,
                1_000 + i,
            )
            .await;
        }

        let (spy, _, contradicts_calls) = make_spy(vec![], false, false, false);

        distill(&store, &spy, &emb, &clock, &cfg, "root")
            .await
            .unwrap();

        let calls = *contradicts_calls.lock().unwrap();
        assert!(
            calls <= cap,
            "CP2-U: contradicts must be called at most {cap} times, got {calls}"
        );
        assert_eq!(
            calls, cap,
            "CP2-U: with 4 identical-text memories (6 pairs) and cap={cap}, \
             exactly {cap} contradicts calls expected"
        );
    }

    // ── CP2-Z ─────────────────────────────────────────────────────────────────

    /// CP2-Z: judge's `summarize_preferences` returns `Err`; `distill` returns
    /// `Ok(())`; the episodic memories remain `distilled_at IS NULL` for retry.
    #[tokio::test]
    async fn test_judge_failure_is_non_fatal_and_retried() {
        let (_tmp, store) = make_test_store();
        let emb = FakeEmbedder {
            dim: 8,
            model: "fake".into(),
        };
        let clock = FixedClock::new(1_000);
        let cfg = MemoryConfig {
            distill_enabled: true,
            ..MemoryConfig::default()
        };

        insert_episodic(&store, "e1", "some text", vec![], "", 0, 1_000).await;

        // Judge always fails on summarize.
        let (spy, _, _) = make_spy(vec![], true /* fail_summarize */, false, false);

        let result = distill(&store, &spy, &emb, &clock, &cfg, "root").await;
        assert!(
            result.is_ok(),
            "CP2-Z: distill must return Ok(()) on judge failure, got: {result:?}"
        );

        // e1 must stay undistilled so the next pass can retry.
        let e1 = store.get("e1").await.unwrap().expect("e1 must still exist");
        assert!(
            e1.distilled_at.is_none(),
            "CP2-Z: memory must remain undistilled (distilled_at IS NULL) when judge fails"
        );
    }

    // ── CP2-AB ────────────────────────────────────────────────────────────────

    /// CP2-AB: when `distill_enabled = false`, `distill` is a no-op and the
    /// spy judge receives zero calls.
    #[tokio::test]
    async fn test_disabled_distiller_makes_no_judge_calls() {
        let (_tmp, store) = make_test_store();
        let emb = FakeEmbedder {
            dim: 8,
            model: "fake".into(),
        };
        let clock = FixedClock::new(1_000);
        let cfg = MemoryConfig {
            distill_enabled: false, // master switch OFF
            ..MemoryConfig::default()
        };

        insert_episodic(&store, "e1", "some text", vec![], "", 0, 1_000).await;

        let (spy, summarize_calls, contradicts_calls) = make_spy(vec![], false, false, false);

        distill(&store, &spy, &emb, &clock, &cfg, "root")
            .await
            .unwrap();

        assert_eq!(
            *summarize_calls.lock().unwrap(),
            0,
            "CP2-AB: summarize_preferences must not be called when distill_enabled=false"
        );
        assert_eq!(
            *contradicts_calls.lock().unwrap(),
            0,
            "CP2-AB: contradicts must not be called when distill_enabled=false"
        );
    }

    // ── SC-42 ─────────────────────────────────────────────────────────────────

    /// SC-42: `distill` marks the processed batch `distilled_at`; a second
    /// pass processes nothing new and does not call the judge again.
    #[tokio::test]
    async fn test_distill_processes_only_undistilled_in_fifo() {
        let (_tmp, store) = make_test_store();
        let emb = FakeEmbedder {
            dim: 8,
            model: "fake".into(),
        };
        let clock = FixedClock::new(1_000);
        let cfg = MemoryConfig {
            distill_enabled: true,
            ..MemoryConfig::default()
        };

        insert_episodic(&store, "e1", "some text", vec![], "", 0, 1_000).await;

        let (spy, summarize_calls, _) = make_spy(vec![], false, false, false);

        // First pass: e1 is undistilled → processed.
        distill(&store, &spy, &emb, &clock, &cfg, "root")
            .await
            .unwrap();

        let e1 = store.get("e1").await.unwrap().expect("e1 must exist");
        assert!(
            e1.distilled_at.is_some(),
            "SC-42: first distill must set distilled_at"
        );
        let calls_after_first = *summarize_calls.lock().unwrap();
        assert_eq!(
            calls_after_first, 1,
            "SC-42: summarize must be called exactly once for the first pass"
        );

        // Second pass: e1 is already distilled → empty batch → no new judge calls.
        distill(&store, &spy, &emb, &clock, &cfg, "root")
            .await
            .unwrap();

        let calls_after_second = *summarize_calls.lock().unwrap();
        assert_eq!(
            calls_after_second, calls_after_first,
            "SC-42: second distill must not call summarize_preferences again \
             (no undistilled memories)"
        );
    }
}
