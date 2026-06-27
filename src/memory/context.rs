// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-06-27

//! Budget-aware context assembler (P3 / REQ-12/13/14/REQ-33, D-10/D-17).
//!
//! [`assemble_selective`] is the sole enforcer of the hard token cap. It calls
//! [`recall`] (the stable B1 seam, D-13) and greedily fills the remaining space
//! with highest-ranked memories. The profile and the current turn are always
//! present; the system prompt and profile together must fit — otherwise the
//! config is broken and the call returns `Err(BudgetUnsatisfiable)`.

use crate::agent::messages::Message;
use crate::memory::clock::Clock;
use crate::memory::config::MemoryConfig;
use crate::memory::embedding::EmbeddingProvider;
use crate::memory::error::MemoryError;
use crate::memory::store::VectorStore;

// ─── Public types ─────────────────────────────────────────────────────────────

/// Bounded context to send to the provider (produced by [`assemble_selective`]).
///
/// The [`messages`] slice is ready to pass to `Provider::stream_messages` or
/// `Provider::send_messages`. [`used_tokens`] is always `<= budget_after_margin(…)`.
/// [`notices`] is non-empty only when a degraded-mode policy fires (e.g. turn
/// truncation under `oversized_turn_policy = "truncate"`).
#[derive(Debug, Clone)]
pub struct AssembledContext {
    /// Assembled messages: at most `[User(preamble), User(current_turn)]`.
    ///
    /// The preamble message combines system, profile, and any kept recalls joined
    /// by `"\n\n"`. The current-turn message follows it (possibly truncated).
    pub messages: Vec<Message>,
    /// Estimated token count (`<= budget_after_margin(…)` guaranteed).
    pub used_tokens: usize,
    /// Diagnostics — non-empty only when a degraded-mode policy fired
    /// (e.g. `"current turn truncated to fit context budget"`).
    pub notices: Vec<String>,
}

// ─── assemble_selective ───────────────────────────────────────────────────────

/// Assembles a bounded context under a token budget (REQ-13, D-17).
///
/// # Priority
/// 1. `system` — always in the preamble (if non-empty).
/// 2. `profile` — always in the preamble (if non-empty). **Never dropped.**
/// 3. Ranked recalls — greedily fill the remaining recall space in score order
///    (highest first). Recalls that do not fit are excluded (SC-17).
/// 4. `current_turn` — always the last message. **Never silently dropped.**
///
/// # Oversized current turn (D-17)
/// If `system + profile` fit but adding the turn would exceed the budget:
/// - `oversized_turn_policy = "truncate"`: truncates the turn text to fit and
///   pushes a notice to [`AssembledContext::notices`].
/// - any other value (incl. `"error"`): returns `Err(BudgetUnsatisfiable)`.
///
/// If `system + profile` alone do not fit (broken config) →
/// `Err(BudgetUnsatisfiable)`.
///
/// # Determinism (R-06)
/// All time-dependent quantities come from the injected [`Clock`]. Token counts
/// use the deterministic heuristic from [`estimate_tokens`]. Results are fully
/// determined by the inputs + `cfg.seed`.
///
/// # Errors
/// - [`MemoryError::BudgetUnsatisfiable`] when the budget is too small to hold
///   even `system + profile`, or when `oversized_turn_policy = "error"` and the
///   turn overflows.
/// - [`MemoryError::Embedding`] / [`MemoryError::Storage`] /
///   [`MemoryError::Crypto`] on retrieval failures.
// Narrow allow: wired into `query_streaming` in Task 12; no non-test caller yet.
#[allow(dead_code)]
pub async fn assemble_selective(
    _store: &dyn VectorStore,
    _embedder: &dyn EmbeddingProvider,
    _clock: &dyn Clock,
    _cfg: &MemoryConfig,
    _system: &str,
    _profile: &str,
    _current_turn: &Message,
    _scope: &str,
) -> Result<AssembledContext, MemoryError> {
    // RED: stub — Green phase implements the full algorithm.
    Err(MemoryError::Config("not implemented".into()))
}

// ─── Private helpers (stubs for RED; implemented in GREEN) ────────────────────

/// Joins all `Content::Text` blocks in `msg` with a space.
#[allow(dead_code)]
fn extract_turn_text(_msg: &Message) -> String {
    String::new()
}

/// Truncates `text` so that `estimate_tokens(result, cpt) <= max_tokens`,
/// respecting UTF-8 scalar boundaries (`char_indices` / `is_char_boundary`).
#[allow(dead_code)]
fn truncate_to_tokens(_text: &str, _max_tokens: usize, _cpt: f64) -> String {
    String::new()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::messages::{Content, Role};
    use crate::memory::clock::FixedClock;
    use crate::memory::config::MemoryConfig;
    use crate::memory::embedding::EmbeddingProvider;
    use crate::memory::error::{EmbeddingError, MemoryError};
    use crate::memory::store::{Memory, SqliteVectorStore};
    use crate::memory::MemoryKind;
    use crate::system::database::EncryptedSqliteMemory;
    use async_trait::async_trait;

    // ── Deterministic test embedder (bag-of-words, dim-32) ────────────────

    /// L2-normalised bag-of-words over a fixed-dim hash.  Texts sharing words
    /// produce similar vectors; deterministic (same text + dim → same vector).
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

    // ── Text extraction helpers for assertions ────────────────────────────

    fn preamble_text(ctx: &AssembledContext) -> String {
        ctx.messages
            .first()
            .and_then(|m| {
                m.content.iter().find_map(|c| {
                    if let Content::Text { text } = c {
                        Some(text.clone())
                    } else {
                        None
                    }
                })
            })
            .unwrap_or_default()
    }

    fn last_turn_text(ctx: &AssembledContext) -> String {
        ctx.messages
            .last()
            .and_then(|m| {
                m.content.iter().find_map(|c| {
                    if let Content::Text { text } = c {
                        Some(text.clone())
                    } else {
                        None
                    }
                })
            })
            .unwrap_or_default()
    }

    // SC-16 ───────────────────────────────────────────────────────────────

    /// SC-16: budget smaller than the total candidates; `used_tokens <= budget`;
    /// the preamble contains the profile; the final message is the current turn.
    #[tokio::test]
    async fn test_context_never_exceeds_budget_and_keeps_profile_and_turn() {
        let (_tmp, store) = make_test_store();
        let emb = FakeEmbedder {
            dim: 32,
            model: "fake".into(),
        };
        let clock = FixedClock::new(1_000_000);
        let cfg = MemoryConfig {
            context_budget_tokens: 200,
            response_headroom_tokens: 0,
            safety_margin_ratio: 0.0,
            top_k: 20,
            ..MemoryConfig::default()
        };
        // 20 memories whose combined text greatly exceeds 200 tokens.
        for i in 0..20 {
            insert_mem(
                &store,
                &format!("m{i}"),
                "context budget token assembly recall memory relevant window",
                &emb,
                1000,
                1000,
                0.5,
            )
            .await;
        }
        let turn = Message::user("What is the context budget?");
        let result = assemble_selective(
            &store,
            &emb,
            &clock,
            &cfg,
            "system prompt",
            "user profile preferences",
            &turn,
            "root",
        )
        .await
        .unwrap();

        assert!(
            result.used_tokens <= 200,
            "used_tokens={} must not exceed budget=200 (SC-16)",
            result.used_tokens
        );
        let preamble = preamble_text(&result);
        assert!(
            preamble.contains("user profile preferences"),
            "preamble must include the profile (SC-16)"
        );
        assert_eq!(
            result.messages.last().unwrap().role,
            Role::User,
            "final message must be a user message (current turn, SC-16)"
        );
        let lt = last_turn_text(&result);
        assert!(
            lt.contains("context budget"),
            "final message must contain the current turn text (SC-16)"
        );
    }

    // SC-17 ───────────────────────────────────────────────────────────────

    /// SC-17: with room for only some recalls, the highest-scoring ones (sharing
    /// words with the query) are in the preamble; lower-scoring ones are excluded;
    /// profile and turn are never dropped.
    #[tokio::test]
    async fn test_higher_score_recalls_included_rest_excluded() {
        let (_tmp, store) = make_test_store();
        let emb = FakeEmbedder {
            dim: 32,
            model: "fake".into(),
        };
        let clock = FixedClock::new(1_000_000);
        // Budget: sys(1t) + profile(2t) + turn(4t) = 7t. With budget=20, recall_space=13t.
        // Each memory is ~10t, so only 1 can fit. The relevant ones rank above distractors.
        let cfg = MemoryConfig {
            context_budget_tokens: 20,
            response_headroom_tokens: 0,
            safety_margin_ratio: 0.0,
            top_k: 20,
            ..MemoryConfig::default()
        };
        // Relevant memories share keywords with the query "context budget".
        insert_mem(
            &store,
            "rel1",
            "context budget assembly relevant",
            &emb,
            1000,
            1000,
            0.5,
        )
        .await;
        insert_mem(
            &store,
            "rel2",
            "budget context memory relevant",
            &emb,
            1000,
            1000,
            0.5,
        )
        .await;
        // Distractors have zero overlap with the query.
        for i in 0..8 {
            insert_mem(
                &store,
                &format!("dist{i}"),
                &format!("unrelated cat dog fish bird {i}"),
                &emb,
                900,
                900,
                0.2,
            )
            .await;
        }
        let turn = Message::user("context budget");
        let result = assemble_selective(
            &store,
            &emb,
            &clock,
            &cfg,
            "sys",
            "profile",
            &turn,
            "root",
        )
        .await
        .unwrap();

        assert!(
            result.used_tokens <= 20,
            "used_tokens={} must not exceed budget=20 (SC-17)",
            result.used_tokens
        );
        let preamble = preamble_text(&result);
        assert!(preamble.contains("profile"), "profile must always be present (SC-17)");
        // The highest-score recall (rel1 or rel2) must appear; distractors must not
        // (only ~13 recall tokens available, rel1 takes 10 → no room for distractors).
        assert!(
            preamble.contains("assembly relevant") || preamble.contains("memory relevant"),
            "at least one relevant recall must be in the preamble (SC-17)"
        );
        assert!(
            !preamble.contains("unrelated cat dog fish"),
            "distractors must be excluded when budget is exhausted (SC-17)"
        );
        let lt = last_turn_text(&result);
        assert!(
            lt.contains("context budget"),
            "current turn must be present (SC-17)"
        );
    }

    // SC-18 ───────────────────────────────────────────────────────────────

    /// SC-18: assemble with 10, 100, 1000 stored memories; `used_tokens` stays
    /// within `[0, budget]` in all three cases and does not grow with store size.
    #[tokio::test]
    async fn test_bounded_growth_independent_of_history_size() {
        let emb = FakeEmbedder {
            dim: 32,
            model: "fake".into(),
        };
        let clock = FixedClock::new(1_000_000);
        let budget = 300usize;
        let cfg = MemoryConfig {
            context_budget_tokens: budget,
            response_headroom_tokens: 0,
            safety_margin_ratio: 0.0,
            top_k: 12,
            ..MemoryConfig::default()
        };
        let turn = Message::user("context budget assembly");
        let sizes = [10usize, 100, 1000];

        for &n in &sizes {
            let (_tmp, store) = make_test_store();
            for i in 0..n {
                insert_mem(
                    &store,
                    &format!("m{i}"),
                    "context budget token recall memory assembly",
                    &emb,
                    1000,
                    1000,
                    0.5,
                )
                .await;
            }
            let result = assemble_selective(
                &store,
                &emb,
                &clock,
                &cfg,
                "system",
                "profile",
                &turn,
                "root",
            )
            .await
            .unwrap();

            assert!(
                result.used_tokens <= budget,
                "used_tokens={} must be <= budget={} with {} memories (SC-18)",
                result.used_tokens,
                budget,
                n
            );
        }
    }

    // SC-35 (truncate) ────────────────────────────────────────────────────

    /// SC-35: a current turn whose text alone exceeds `budget - sys - profile`
    /// under `oversized_turn_policy = "truncate"` ⇒ the turn is truncated, a
    /// notice is present, system+profile are preserved, `used_tokens <= budget`.
    #[tokio::test]
    async fn test_oversized_turn_truncates_with_notice() {
        let (_tmp, store) = make_test_store();
        let emb = FakeEmbedder {
            dim: 32,
            model: "fake".into(),
        };
        let clock = FixedClock::new(1_000_000);
        // sys="sys"(3 chars, ~1t) + profile="profile"(7 chars, 2t) = 3t.
        // budget=30 → space=27t.  Turn ≈ 352 chars → 101t at cpt=3.5. → must truncate.
        let cfg = MemoryConfig {
            context_budget_tokens: 30,
            response_headroom_tokens: 0,
            safety_margin_ratio: 0.0,
            oversized_turn_policy: "truncate".into(),
            ..MemoryConfig::default()
        };
        let system = "sys";
        let profile = "profile";
        // ~352-char turn → ~101 tokens at default cpt=3.5.
        let long_turn: String = "abcdefghij ".repeat(32);
        assert!(long_turn.len() > 100, "turn must be large enough to trigger truncation");
        let turn = Message::user(&long_turn);

        let result = assemble_selective(
            &store,
            &emb,
            &clock,
            &cfg,
            system,
            profile,
            &turn,
            "root",
        )
        .await
        .unwrap();

        assert!(
            result.used_tokens <= 30,
            "used_tokens={} must not exceed budget=30 even with oversized turn (SC-35)",
            result.used_tokens
        );
        assert!(
            !result.notices.is_empty(),
            "a truncation notice must be present (SC-35)"
        );
        let preamble = preamble_text(&result);
        assert!(
            preamble.contains(profile),
            "profile must be preserved after truncation (SC-35)"
        );
        let lt = last_turn_text(&result);
        assert!(
            lt.len() < long_turn.len(),
            "turn must be shorter after truncation: got {} chars, expected < {} (SC-35)",
            lt.len(),
            long_turn.len()
        );
    }

    // SC-35 (unsatisfiable) ───────────────────────────────────────────────

    /// SC-35: when `system + profile` alone exceed the budget (broken config),
    /// `assemble_selective` returns `Err(BudgetUnsatisfiable)`.
    #[tokio::test]
    async fn test_system_plus_profile_unsatisfiable_errors() {
        let (_tmp, store) = make_test_store();
        let emb = FakeEmbedder {
            dim: 32,
            model: "fake".into(),
        };
        let clock = FixedClock::new(1_000_000);
        let cfg = MemoryConfig {
            context_budget_tokens: 1,
            response_headroom_tokens: 0,
            safety_margin_ratio: 0.0,
            ..MemoryConfig::default()
        };
        // System alone is >> 1 token.
        let system = "This is a system prompt that is definitely longer than one token here";
        let profile = "User profile data that also adds tokens";
        let turn = Message::user("hello");

        let result = assemble_selective(
            &store,
            &emb,
            &clock,
            &cfg,
            system,
            profile,
            &turn,
            "root",
        )
        .await;

        assert!(
            matches!(result, Err(MemoryError::BudgetUnsatisfiable)),
            "must return BudgetUnsatisfiable when system+profile don't fit (SC-35), got: {result:?}"
        );
    }
}
