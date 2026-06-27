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

use crate::agent::messages::{Content, Message};
use crate::memory::clock::Clock;
use crate::memory::config::MemoryConfig;
use crate::memory::embedding::EmbeddingProvider;
use crate::memory::error::MemoryError;
use crate::memory::retrieval::recall;
use crate::memory::store::VectorStore;
use crate::memory::tokens::{budget_after_margin, estimate_tokens};

// ─── Public types ─────────────────────────────────────────────────────────────

/// Bounded context to send to the provider (produced by [`assemble_selective`]).
///
/// The `messages` field is ready to pass to `Provider::stream_messages` or
/// `Provider::send_messages`. `used_tokens` is always `<= budget_after_margin(…)`.
/// `notices` is non-empty only when a degraded-mode policy fires (e.g. turn
/// truncation under `oversized_turn_policy = "truncate"`).
///
/// # Message layout
///
/// At most two messages are produced:
/// 1. `User(preamble)` — system + profile + kept recalls joined by `"\n\n"`.
///    Omitted if all three are empty strings.
/// 2. `User(current_turn_text)` — the (possibly truncated) turn.
///
/// Both have `Role::User` because the Anthropic Messages API expects the context
/// injected before the actual user turn to arrive as a prior user turn (the next
/// Task wires in the interleaving with prior assistant messages from history).
// Narrow allow: struct fields read by the agent wiring in Task 12; no non-test caller yet.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct AssembledContext {
    /// Assembled messages: `[User(preamble)?, User(current_turn)]`.
    pub messages: Vec<Message>,
    /// Estimated token count (guaranteed `<= budget_after_margin(…)`).
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
/// # Token budget
/// `budget = budget_after_margin(context_budget_tokens, response_headroom_tokens,
/// safety_margin_ratio)`. The returned [`AssembledContext::used_tokens`] is always
/// `<= budget`.
///
/// # Oversized current turn (D-17)
/// If `system + profile` fit but adding the turn would exceed the budget:
/// - `oversized_turn_policy = "truncate"`: truncates the turn text to fit and
///   pushes a notice to [`AssembledContext::notices`]. `recall_space` becomes `0`.
/// - any other value (incl. `"error"`): returns `Err(BudgetUnsatisfiable)`.
///
/// If `system + profile` alone do not fit (broken config) →
/// `Err(BudgetUnsatisfiable)`.
///
/// # Determinism (R-06)
/// All time-dependent quantities come from the injected [`Clock`]. Token counts
/// use the deterministic heuristic from [`estimate_tokens`] (chars / cpt).
/// Results are fully determined by the inputs + `cfg.seed`.
///
/// # Errors
/// - [`MemoryError::BudgetUnsatisfiable`] when the budget is too small to hold
///   even `system + profile`, or when `oversized_turn_policy != "truncate"` and
///   the turn overflows.
/// - [`MemoryError::Embedding`] / [`MemoryError::Storage`] /
///   [`MemoryError::Crypto`] on retrieval failures.
// Narrow allow: wired into `query_streaming` in Task 12; no non-test caller yet.
// The 8-argument signature matches the required stable interface (D-13/B1).
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub async fn assemble_selective(
    store: &dyn VectorStore,
    embedder: &dyn EmbeddingProvider,
    clock: &dyn Clock,
    cfg: &MemoryConfig,
    system: &str,
    profile: &str,
    current_turn: &Message,
    scope: &str,
) -> Result<AssembledContext, MemoryError> {
    let cpt = cfg.chars_per_token;
    let budget = budget_after_margin(
        cfg.context_budget_tokens,
        cfg.response_headroom_tokens,
        cfg.safety_margin_ratio,
    );

    // Step 1-2: compute token cost of the fixed components.
    let sys_t = estimate_tokens(system, cpt);
    let prof_t = estimate_tokens(profile, cpt);

    // Step 3: guard against broken config (system+profile alone don't fit).
    if sys_t + prof_t > budget {
        return Err(MemoryError::BudgetUnsatisfiable);
    }

    // Step 4: space available for turn + recalls.
    let space = budget - sys_t - prof_t;

    // Step 5: extract and potentially truncate the current turn.
    let mut turn_text = extract_turn_text(current_turn);
    let turn_t = estimate_tokens(&turn_text, cpt);

    let mut notices = Vec::new();
    let recall_space;

    if turn_t > space {
        if cfg.oversized_turn_policy == "truncate" {
            turn_text = truncate_to_tokens(&turn_text, space, cpt);
            notices.push("current turn truncated to fit context budget".to_string());
            recall_space = 0;
        } else {
            return Err(MemoryError::BudgetUnsatisfiable);
        }
    } else {
        recall_space = space - turn_t;
    }

    // Step 6: retrieve ranked memories and greedily fill the recall budget.
    let ranked = recall(store, embedder, clock, cfg, &turn_text, budget, scope).await?;

    let mut kept_texts: Vec<String> = Vec::new();
    let mut recall_tokens_used: usize = 0;
    let mut recall_space_left = recall_space;

    for rm in &ranked {
        let t = estimate_tokens(&rm.memory.text, cpt);
        if t <= recall_space_left {
            kept_texts.push(rm.memory.text.clone());
            recall_tokens_used += t;
            recall_space_left -= t;
        }
        // Skip memories that don't fit; continue to see if a shorter one fits.
    }

    // Step 7: build the preamble from non-empty parts.
    let mut parts: Vec<&str> = Vec::new();
    if !system.is_empty() {
        parts.push(system);
    }
    if !profile.is_empty() {
        parts.push(profile);
    }
    for t in &kept_texts {
        parts.push(t.as_str());
    }
    let preamble = parts.join("\n\n");

    let mut messages: Vec<Message> = Vec::new();
    if !preamble.is_empty() {
        messages.push(Message::user(&preamble));
    }
    messages.push(Message::user(&turn_text));

    // Step 8: account for all tokens; invariant: used_tokens <= budget.
    let final_turn_t = estimate_tokens(&turn_text, cpt);
    let used_tokens = sys_t + prof_t + recall_tokens_used + final_turn_t;

    debug_assert!(
        used_tokens <= budget,
        "assembler invariant violated: used_tokens={used_tokens} > budget={budget}"
    );

    Ok(AssembledContext {
        messages,
        used_tokens,
        notices,
    })
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// Joins all `Content::Text` blocks in `msg` with a single space.
///
/// Non-text content blocks (ToolUse, ToolResult) are ignored; they carry no
/// semantic text the assembler should embed or budget against.
fn extract_turn_text(msg: &Message) -> String {
    msg.content
        .iter()
        .filter_map(|c| {
            if let Content::Text { text } = c {
                Some(text.as_str())
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Truncates `text` to the longest char-prefix whose estimated token count
/// does not exceed `max_tokens` (using heuristic `cpt`).
///
/// # UTF-8 safety
/// Iterates over Unicode scalars via `.chars()` (never byte-splits a multi-byte
/// character). The result is always valid UTF-8.
///
/// # Invariant
/// `estimate_tokens(result, cpt) <= max_tokens` for all non-zero `cpt`.
fn truncate_to_tokens(text: &str, max_tokens: usize, cpt: f64) -> String {
    if max_tokens == 0 {
        return String::new();
    }
    let cpt = if cpt > 0.0 && cpt.is_finite() {
        cpt
    } else {
        1.0
    };
    // max_chars is the longest prefix (in Unicode scalars) that stays under budget.
    // ceil(max_chars / cpt) <= max_tokens  ⟺  max_chars <= max_tokens * cpt.
    let max_chars = (max_tokens as f64 * cpt).floor() as usize;
    text.chars().take(max_chars).collect()
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
        // Each memory is ~10t, so only 1 can fit. Relevant memories rank above distractors.
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
        let result =
            assemble_selective(&store, &emb, &clock, &cfg, "sys", "profile", &turn, "root")
                .await
                .unwrap();

        assert!(
            result.used_tokens <= 20,
            "used_tokens={} must not exceed budget=20 (SC-17)",
            result.used_tokens
        );
        let preamble = preamble_text(&result);
        assert!(
            preamble.contains("profile"),
            "profile must always be present (SC-17)"
        );
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
                &store, &emb, &clock, &cfg, "system", "profile", &turn, "root",
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
        // budget=30 → space=27t.  Turn ≈ 352 chars → ~101t at cpt=3.5. → must truncate.
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
        assert!(
            long_turn.len() > 100,
            "turn must be large enough to trigger truncation"
        );
        let turn = Message::user(&long_turn);

        let result = assemble_selective(&store, &emb, &clock, &cfg, system, profile, &turn, "root")
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

    // M2 ─────────────────────────────────────────────────────────────────

    /// M2: when `space` after system+profile is exactly 0 and the turn is
    /// non-empty, `oversized_turn_policy = "truncate"` would truncate the turn
    /// to an empty string; the assembler must return `BudgetUnsatisfiable` rather
    /// than silently sending an empty message (D-17).
    #[tokio::test]
    async fn test_empty_truncation_returns_budget_unsatisfiable() {
        let (_tmp, store) = make_test_store();
        let emb = FakeEmbedder {
            dim: 32,
            model: "fake".into(),
        };
        let clock = FixedClock::new(1_000_000);
        // With chars_per_token=1.0 each character costs 1 token.
        // budget=2, sys="s"(1t), profile="p"(1t) → space=0.
        // Any non-empty turn overflows and truncates to "".
        let cfg = MemoryConfig {
            context_budget_tokens: 2,
            response_headroom_tokens: 0,
            safety_margin_ratio: 0.0,
            chars_per_token: 1.0,
            oversized_turn_policy: "truncate".into(),
            ..MemoryConfig::default()
        };
        let turn = Message::user("hello");

        let result = assemble_selective(&store, &emb, &clock, &cfg, "s", "p", &turn, "root").await;

        assert!(
            matches!(result, Err(MemoryError::BudgetUnsatisfiable)),
            "M2: empty truncated turn must return BudgetUnsatisfiable (D-17), got: {result:?}"
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

        let result =
            assemble_selective(&store, &emb, &clock, &cfg, system, profile, &turn, "root").await;

        assert!(
            matches!(result, Err(MemoryError::BudgetUnsatisfiable)),
            "must return BudgetUnsatisfiable when system+profile don't fit (SC-35), got: {result:?}"
        );
    }
}
