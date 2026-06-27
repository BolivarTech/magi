// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-06-26

//! Configuration for the tiered-memory subsystem: the `[memory]` and
//! `[embedding]` sections of `magi.toml`.
//!
//! Both structs use `#[serde(deny_unknown_fields)]` so a typo — or, deliberately,
//! an `api_key` field — is a parse error rather than silent acceptance (API keys
//! never live in `magi.toml`). Every field defaults via a function in the private
//! [`d`] module, which is also what `Default` delegates to, so a bare config and a
//! partially-specified section both resolve to the same documented values.

use crate::memory::error::MemoryError;
use serde::Deserialize;

/// Runtime configuration for the tiered-memory subsystem (`[memory]` section).
///
/// Defaults are the Ollama-first, determinism-friendly profile; see each field.
/// All weights and thresholds feed deterministic retrieval/decay (seeded by
/// [`MemoryConfig::seed`]).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryConfig {
    /// `"selective"` (new tiered path) or `"load_all"` (v0.6.0 control). Default `"selective"`.
    #[serde(default = "d::mode")]
    pub mode: String,
    /// Token budget for the assembled context. Default `8000`.
    #[serde(default = "d::context_budget_tokens")]
    pub context_budget_tokens: usize,
    /// Tokens reserved for the model's response. Default `1024`.
    #[serde(default = "d::response_headroom_tokens")]
    pub response_headroom_tokens: usize,
    /// Safety margin as a fraction of the budget, guarding heuristic underestimation. Default `0.1`.
    #[serde(default = "d::safety_margin_ratio")]
    pub safety_margin_ratio: f64,
    /// Heuristic characters-per-token divisor. Conservative default `3.5` (Spanish-friendly).
    /// Tune lower for token-dense content: code ~3.0, CJK ~2.0.
    #[serde(default = "d::chars_per_token")]
    pub chars_per_token: f64,
    /// Policy when the current turn alone exceeds the budget: `"truncate"` or `"error"`. Default `"truncate"`.
    #[serde(default = "d::oversized_turn_policy")]
    pub oversized_turn_policy: String,
    /// Retrieval candidate count. Default `12`.
    #[serde(default = "d::top_k")]
    pub top_k: usize,
    /// Reranker weight on cosine similarity. Default `1.0`.
    #[serde(default = "d::weight_similarity")]
    pub weight_similarity: f64,
    /// Reranker weight on recency. Default `0.3`.
    #[serde(default = "d::weight_recency")]
    pub weight_recency: f64,
    /// Reranker weight on salience. Default `0.5`.
    #[serde(default = "d::weight_salience")]
    pub weight_salience: f64,
    /// Base salience assigned at write. Default `0.3`.
    #[serde(default = "d::default_salience")]
    pub default_salience: f64,
    /// Protected-floor salience for `kind=preference`. Default `1.0`.
    #[serde(default = "d::preference_salience")]
    pub preference_salience: f64,
    /// Salience at/above which a memory is never evicted by decay. Default `0.9`.
    #[serde(default = "d::protect_salience_threshold")]
    pub protect_salience_threshold: f64,
    /// Recency half-life in wall-clock days (decay via injected `Clock`). Default `30.0`.
    #[serde(default = "d::decay_half_life_days")]
    pub decay_half_life_days: f64,
    /// Cap on the access-reinforcement contribution (diminishing returns). Default `50`.
    #[serde(default = "d::access_saturation_cap")]
    pub access_saturation_cap: u64,
    /// Strength below which a memory is eligible for forgetting. Default `0.1`.
    #[serde(default = "d::forget_strength_threshold")]
    pub forget_strength_threshold: f64,
    /// Eviction retention: `-1` archive (never hard-delete), `0` immediate hard-delete,
    /// `N>0` hard-delete after N days. Default `-1`.
    #[serde(default = "d::evicted_retention_days")]
    pub evicted_retention_days: i64,
    /// Hard ceiling on active records (anti-DoS). `0` = explicit operator opt-out. Default `50_000`.
    #[serde(default = "d::max_records")]
    pub max_records: usize,
    /// Embedding-similarity threshold for "same subject" hard-supersession candidates. Default `0.85`.
    #[serde(default = "d::supersede_similarity_threshold")]
    pub supersede_similarity_threshold: f64,
    /// Run the distiller every N turns (`0` = on-demand/session-close only). Default `20`.
    #[serde(default = "d::distill_every_n_turns")]
    pub distill_every_n_turns: usize,
    /// Run the distiller on session close. Default `true`.
    #[serde(default = "d::distill_on_session_close")]
    pub distill_on_session_close: bool,
    /// Token bound on the always-injected preference profile. Default `1024`.
    #[serde(default = "d::profile_max_tokens")]
    pub profile_max_tokens: usize,
    /// Deterministic seed for retrieval/decay/benchmark. Default `42`.
    #[serde(default = "d::seed")]
    pub seed: u64,
    /// Substrings that lift a memory's salience at write (preference markers).
    #[serde(default = "d::salience_markers")]
    pub salience_markers: Vec<String>,
    /// Retrieval index: `"exact"` (deterministic brute-force, default) or `"ann"`
    /// (opt-in, requires the `ann` build feature). Default `"exact"`.
    #[serde(default = "d::index")]
    pub index: String,
    /// Token cap on the distiller's per-run LLM batch (privacy bound). Default `4000`.
    #[serde(default = "d::distill_max_batch_tokens")]
    pub distill_max_batch_tokens: usize,
    /// Cap on same-subject candidate pairs the distiller judges per run. Default `50`.
    #[serde(default = "d::supersede_max_candidate_pairs")]
    pub supersede_max_candidate_pairs: usize,
    /// Master switch for the LLM distiller (`false` = zero memory egress for distillation). Default `true`.
    #[serde(default = "d::distill_enabled")]
    pub distill_enabled: bool,
    /// Max memories re-embedded per lazy pass (throttle). Default `32`.
    #[serde(default = "d::reembed_batch_size")]
    pub reembed_batch_size: usize,
    /// Max evictions per forgetting pass (clock-jump guard). Default `1000`.
    #[serde(default = "d::max_evictions_per_pass")]
    pub max_evictions_per_pass: usize,
    /// Batch size for the throttled lazy migration. Default `256`.
    #[serde(default = "d::migration_throttle_batch")]
    pub migration_throttle_batch: usize,
}

impl MemoryConfig {
    /// Validates that runtime-sensitive fields have legal values.
    ///
    /// Intended to be called at startup (after loading `magi.toml`) so that
    /// a misconfigured value is surfaced as a startup notice rather than
    /// producing silent NaN or divide-by-zero at runtime (B1).
    ///
    /// # Errors
    /// Returns `Err(MemoryError::Config(_))` on the first invalid field found.
    pub fn validate(&self) -> Result<(), MemoryError> {
        if self.decay_half_life_days <= 0.0 {
            return Err(MemoryError::Config(format!(
                "decay_half_life_days must be > 0.0, got {}",
                self.decay_half_life_days
            )));
        }
        if self.chars_per_token <= 0.0 {
            return Err(MemoryError::Config(format!(
                "chars_per_token must be > 0.0, got {}",
                self.chars_per_token
            )));
        }
        if self.protect_salience_threshold <= 0.0 || self.protect_salience_threshold > 1.0 {
            return Err(MemoryError::Config(format!(
                "protect_salience_threshold must be in (0.0, 1.0], got {}",
                self.protect_salience_threshold
            )));
        }
        // safety_margin_ratio must be in [0.0, 1.0): a value ≥ 1.0 would reduce the
        // usable budget to 0 or negative, making the context assembler degenerate.
        if self.safety_margin_ratio < 0.0 || self.safety_margin_ratio >= 1.0 {
            return Err(MemoryError::Config(format!(
                "safety_margin_ratio must be in [0.0, 1.0), got {}",
                self.safety_margin_ratio
            )));
        }
        if self.context_budget_tokens == 0 {
            return Err(MemoryError::Config(
                "context_budget_tokens must be > 0".into(),
            ));
        }
        if self.top_k == 0 {
            return Err(MemoryError::Config("top_k must be > 0".into()));
        }
        // Individual weight negativity check before sum check so the error message
        // names the specific offending weight.
        if self.weight_similarity < 0.0 || self.weight_recency < 0.0 || self.weight_salience < 0.0 {
            return Err(MemoryError::Config(format!(
                "reranker weights must be >= 0.0; got similarity={}, recency={}, salience={}",
                self.weight_similarity, self.weight_recency, self.weight_salience
            )));
        }
        // All-zero weight set makes the reranker degenerate (every candidate scores 0).
        let weight_sum = self.weight_similarity + self.weight_recency + self.weight_salience;
        if weight_sum == 0.0 {
            return Err(MemoryError::Config(
                "reranker weights must not all be zero (sum must be > 0.0)".into(),
            ));
        }
        Ok(())
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            mode: d::mode(),
            context_budget_tokens: d::context_budget_tokens(),
            response_headroom_tokens: d::response_headroom_tokens(),
            safety_margin_ratio: d::safety_margin_ratio(),
            chars_per_token: d::chars_per_token(),
            oversized_turn_policy: d::oversized_turn_policy(),
            top_k: d::top_k(),
            weight_similarity: d::weight_similarity(),
            weight_recency: d::weight_recency(),
            weight_salience: d::weight_salience(),
            default_salience: d::default_salience(),
            preference_salience: d::preference_salience(),
            protect_salience_threshold: d::protect_salience_threshold(),
            decay_half_life_days: d::decay_half_life_days(),
            access_saturation_cap: d::access_saturation_cap(),
            forget_strength_threshold: d::forget_strength_threshold(),
            evicted_retention_days: d::evicted_retention_days(),
            max_records: d::max_records(),
            supersede_similarity_threshold: d::supersede_similarity_threshold(),
            distill_every_n_turns: d::distill_every_n_turns(),
            distill_on_session_close: d::distill_on_session_close(),
            profile_max_tokens: d::profile_max_tokens(),
            seed: d::seed(),
            salience_markers: d::salience_markers(),
            index: d::index(),
            distill_max_batch_tokens: d::distill_max_batch_tokens(),
            supersede_max_candidate_pairs: d::supersede_max_candidate_pairs(),
            distill_enabled: d::distill_enabled(),
            reembed_batch_size: d::reembed_batch_size(),
            max_evictions_per_pass: d::max_evictions_per_pass(),
            migration_throttle_batch: d::migration_throttle_batch(),
        }
    }
}

/// Embedding-provider configuration (`[embedding]` section). OpenAI-compatible;
/// the default targets local Ollama with the `nomic-embed-text` model.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingConfig {
    /// Provider kind; reuses the OpenAI-compatible path. Default `"openai"`.
    #[serde(default = "d::emb_provider")]
    pub provider: String,
    /// Endpoint base URL. Default local Ollama `"http://localhost:11434/v1"`.
    #[serde(default = "d::emb_base_url")]
    pub base_url: String,
    /// Embedding model id. Default `"nomic-embed-text"`.
    #[serde(default = "d::emb_model")]
    pub model: String,
    /// Vector dimension; `0` = autodetect from the first response. Default `768`.
    #[serde(default = "d::emb_dim")]
    pub dim: usize,
    /// Prefix applied to query text before embedding. Default `"search_query: "`.
    #[serde(default = "d::query_prefix")]
    pub query_prefix: String,
    /// Prefix applied to stored text before embedding. Default `"search_document: "`.
    #[serde(default = "d::document_prefix")]
    pub document_prefix: String,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: d::emb_provider(),
            base_url: d::emb_base_url(),
            model: d::emb_model(),
            dim: d::emb_dim(),
            query_prefix: d::query_prefix(),
            document_prefix: d::document_prefix(),
        }
    }
}

/// Default-value functions wrapping the documented constants. Shared by both
/// `serde(default = ...)` and `Default`, so a bare config and a partial section
/// resolve identically (single source of truth). Constants are centralized in
/// `crate::defaults` in the refactor pass.
mod d {
    pub fn mode() -> String {
        "selective".into()
    }
    pub fn context_budget_tokens() -> usize {
        8000
    }
    pub fn response_headroom_tokens() -> usize {
        1024
    }
    pub fn safety_margin_ratio() -> f64 {
        0.1
    }
    pub fn chars_per_token() -> f64 {
        3.5
    }
    pub fn oversized_turn_policy() -> String {
        "truncate".into()
    }
    pub fn top_k() -> usize {
        12
    }
    pub fn weight_similarity() -> f64 {
        1.0
    }
    pub fn weight_recency() -> f64 {
        0.3
    }
    pub fn weight_salience() -> f64 {
        0.5
    }
    pub fn default_salience() -> f64 {
        0.3
    }
    pub fn preference_salience() -> f64 {
        1.0
    }
    pub fn protect_salience_threshold() -> f64 {
        0.9
    }
    pub fn decay_half_life_days() -> f64 {
        30.0
    }
    pub fn access_saturation_cap() -> u64 {
        50
    }
    pub fn forget_strength_threshold() -> f64 {
        0.1
    }
    pub fn evicted_retention_days() -> i64 {
        -1
    }
    pub fn max_records() -> usize {
        50_000
    }
    pub fn supersede_similarity_threshold() -> f64 {
        0.85
    }
    pub fn distill_every_n_turns() -> usize {
        20
    }
    pub fn distill_on_session_close() -> bool {
        true
    }
    pub fn profile_max_tokens() -> usize {
        1024
    }
    pub fn seed() -> u64 {
        42
    }
    pub fn salience_markers() -> Vec<String> {
        ["prefer", "preference", "always", "never", "remember"]
            .iter()
            .map(|s| (*s).to_string())
            .collect()
    }
    pub fn index() -> String {
        "exact".into()
    }
    pub fn distill_max_batch_tokens() -> usize {
        4000
    }
    pub fn supersede_max_candidate_pairs() -> usize {
        50
    }
    pub fn distill_enabled() -> bool {
        true
    }
    pub fn reembed_batch_size() -> usize {
        32
    }
    pub fn max_evictions_per_pass() -> usize {
        1000
    }
    pub fn migration_throttle_batch() -> usize {
        256
    }
    pub fn emb_provider() -> String {
        "openai".into()
    }
    pub fn emb_base_url() -> String {
        "http://localhost:11434/v1".into()
    }
    pub fn emb_model() -> String {
        "nomic-embed-text".into()
    }
    pub fn emb_dim() -> usize {
        768
    }
    pub fn query_prefix() -> String {
        "search_query: ".into()
    }
    pub fn document_prefix() -> String {
        "search_document: ".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MagiConfig;

    // ── B1 validate() tests ───────────────────────────────────────────────────

    #[test]
    fn test_validate_accepts_defaults() {
        assert!(
            MemoryConfig::default().validate().is_ok(),
            "B1: defaults must pass validate()"
        );
    }

    #[test]
    fn test_validate_rejects_zero_half_life() {
        let cfg = MemoryConfig {
            decay_half_life_days: 0.0,
            ..MemoryConfig::default()
        };
        assert!(
            cfg.validate().is_err(),
            "B1: decay_half_life_days=0.0 must be rejected by validate()"
        );
    }

    #[test]
    fn test_validate_rejects_zero_chars_per_token() {
        let cfg = MemoryConfig {
            chars_per_token: 0.0,
            ..MemoryConfig::default()
        };
        assert!(
            cfg.validate().is_err(),
            "B1: chars_per_token=0.0 must be rejected by validate()"
        );
    }

    #[test]
    fn test_validate_rejects_invalid_protect_salience() {
        // 0.0 is out of range (must be > 0.0)
        assert!(
            MemoryConfig {
                protect_salience_threshold: 0.0,
                ..MemoryConfig::default()
            }
            .validate()
            .is_err(),
            "B1: protect_salience_threshold=0.0 must be rejected"
        );
        // 1.5 is out of range (must be <= 1.0)
        assert!(
            MemoryConfig {
                protect_salience_threshold: 1.5,
                ..MemoryConfig::default()
            }
            .validate()
            .is_err(),
            "B1: protect_salience_threshold=1.5 must be rejected"
        );
        // 1.0 is valid (boundary)
        assert!(
            MemoryConfig {
                protect_salience_threshold: 1.0,
                ..MemoryConfig::default()
            }
            .validate()
            .is_ok(),
            "B1: protect_salience_threshold=1.0 must be accepted (inclusive boundary)"
        );
    }

    // ── New range-check tests (Fix 1) ────────────────────────────────────────

    #[test]
    fn test_validate_rejects_safety_margin_ratio_ge_one() {
        assert!(
            MemoryConfig {
                safety_margin_ratio: 1.0,
                ..MemoryConfig::default()
            }
            .validate()
            .is_err(),
            "safety_margin_ratio=1.0 must be rejected (usable budget becomes 0)"
        );
    }

    #[test]
    fn test_validate_rejects_negative_safety_margin_ratio() {
        assert!(
            MemoryConfig {
                safety_margin_ratio: -0.1,
                ..MemoryConfig::default()
            }
            .validate()
            .is_err(),
            "safety_margin_ratio=-0.1 must be rejected"
        );
    }

    #[test]
    fn test_validate_accepts_zero_safety_margin_ratio() {
        assert!(
            MemoryConfig {
                safety_margin_ratio: 0.0,
                ..MemoryConfig::default()
            }
            .validate()
            .is_ok(),
            "safety_margin_ratio=0.0 must be accepted (valid lower bound)"
        );
    }

    #[test]
    fn test_validate_rejects_zero_context_budget_tokens() {
        assert!(
            MemoryConfig {
                context_budget_tokens: 0,
                ..MemoryConfig::default()
            }
            .validate()
            .is_err(),
            "context_budget_tokens=0 must be rejected"
        );
    }

    #[test]
    fn test_validate_rejects_zero_top_k() {
        assert!(
            MemoryConfig {
                top_k: 0,
                ..MemoryConfig::default()
            }
            .validate()
            .is_err(),
            "top_k=0 must be rejected"
        );
    }

    #[test]
    fn test_validate_rejects_negative_reranker_weight() {
        assert!(
            MemoryConfig {
                weight_similarity: -1.0,
                ..MemoryConfig::default()
            }
            .validate()
            .is_err(),
            "negative weight_similarity must be rejected"
        );
    }

    #[test]
    fn test_validate_rejects_all_zero_reranker_weights() {
        assert!(
            MemoryConfig {
                weight_similarity: 0.0,
                weight_recency: 0.0,
                weight_salience: 0.0,
                ..MemoryConfig::default()
            }
            .validate()
            .is_err(),
            "all-zero reranker weights must be rejected (degenerate reranker)"
        );
    }

    #[test]
    fn test_validate_accepts_max_records_zero() {
        assert!(
            MemoryConfig {
                max_records: 0,
                ..MemoryConfig::default()
            }
            .validate()
            .is_ok(),
            "max_records=0 is a valid opt-out and must not be rejected"
        );
    }

    // ── F3: non-finite f64 rejection ──────────────────────────────────────────

    #[test]
    fn test_validate_rejects_nan_decay_half_life_days() {
        // NaN: `NaN <= 0.0` is false so current check passes — must be caught.
        assert!(
            MemoryConfig {
                decay_half_life_days: f64::NAN,
                ..MemoryConfig::default()
            }
            .validate()
            .is_err(),
            "F3: decay_half_life_days=NaN must be rejected"
        );
    }

    #[test]
    fn test_validate_rejects_nan_chars_per_token() {
        assert!(
            MemoryConfig {
                chars_per_token: f64::NAN,
                ..MemoryConfig::default()
            }
            .validate()
            .is_err(),
            "F3: chars_per_token=NaN must be rejected"
        );
    }

    #[test]
    fn test_validate_rejects_nan_safety_margin_ratio() {
        // NaN: `NaN < 0.0` and `NaN >= 1.0` are both false so current check passes.
        assert!(
            MemoryConfig {
                safety_margin_ratio: f64::NAN,
                ..MemoryConfig::default()
            }
            .validate()
            .is_err(),
            "F3: safety_margin_ratio=NaN must be rejected"
        );
    }

    #[test]
    fn test_validate_rejects_nan_protect_salience_threshold() {
        assert!(
            MemoryConfig {
                protect_salience_threshold: f64::NAN,
                ..MemoryConfig::default()
            }
            .validate()
            .is_err(),
            "F3: protect_salience_threshold=NaN must be rejected"
        );
    }

    #[test]
    fn test_validate_rejects_inf_decay_half_life_days() {
        // Inf: `Inf <= 0.0` is false so current check passes — must be caught.
        assert!(
            MemoryConfig {
                decay_half_life_days: f64::INFINITY,
                ..MemoryConfig::default()
            }
            .validate()
            .is_err(),
            "F3: decay_half_life_days=Infinity must be rejected"
        );
    }

    #[test]
    fn test_validate_rejects_inf_weight_similarity() {
        // Inf weight passes individual negativity check; must still be caught.
        assert!(
            MemoryConfig {
                weight_similarity: f64::INFINITY,
                ..MemoryConfig::default()
            }
            .validate()
            .is_err(),
            "F3: weight_similarity=Infinity must be rejected"
        );
    }

    // ── F3: string-enum validation ────────────────────────────────────────────

    #[test]
    fn test_validate_rejects_bogus_mode() {
        assert!(
            MemoryConfig {
                mode: "bogus".into(),
                ..MemoryConfig::default()
            }
            .validate()
            .is_err(),
            "F3: mode='bogus' must be rejected (valid: selective, load_all)"
        );
    }

    #[test]
    fn test_validate_rejects_bogus_oversized_turn_policy() {
        assert!(
            MemoryConfig {
                oversized_turn_policy: "skip".into(),
                ..MemoryConfig::default()
            }
            .validate()
            .is_err(),
            "F3: oversized_turn_policy='skip' must be rejected (valid: truncate, error)"
        );
    }

    #[test]
    fn test_validate_rejects_bogus_index() {
        assert!(
            MemoryConfig {
                index: "hnsw".into(),
                ..MemoryConfig::default()
            }
            .validate()
            .is_err(),
            "F3: index='hnsw' must be rejected (valid: exact, ann)"
        );
    }

    #[test]
    fn test_validate_accepts_all_valid_string_enum_values() {
        for mode in &["selective", "load_all"] {
            assert!(
                MemoryConfig {
                    mode: (*mode).into(),
                    ..MemoryConfig::default()
                }
                .validate()
                .is_ok(),
                "F3: mode='{mode}' must be accepted"
            );
        }
        for policy in &["truncate", "error"] {
            assert!(
                MemoryConfig {
                    oversized_turn_policy: (*policy).into(),
                    ..MemoryConfig::default()
                }
                .validate()
                .is_ok(),
                "F3: oversized_turn_policy='{policy}' must be accepted"
            );
        }
        for idx in &["exact", "ann"] {
            assert!(
                MemoryConfig {
                    index: (*idx).into(),
                    ..MemoryConfig::default()
                }
                .validate()
                .is_ok(),
                "F3: index='{idx}' must be accepted"
            );
        }
    }

    #[test]
    fn test_absent_memory_section_uses_documented_defaults() {
        let c = MagiConfig::from_toml_str("provider = \"openai\"").unwrap();
        assert_eq!(c.memory.mode, "selective");
        assert_eq!(c.memory.chars_per_token, 3.5);
        assert_eq!(c.memory.safety_margin_ratio, 0.1);
        assert_eq!(c.memory.seed, 42);
        assert_eq!(c.memory.max_records, 50_000);
        assert_eq!(c.memory.index, "exact");
        assert!(c.memory.distill_enabled);
        assert_eq!(c.embedding.model, "nomic-embed-text");
        assert_eq!(c.embedding.dim, 768);
        assert_eq!(c.embedding.query_prefix, "search_query: ");
    }

    #[test]
    fn test_unknown_field_in_memory_or_embedding_is_err() {
        // deny_unknown_fields — a stray key (incl. api_key) is a parse error (REQ-21).
        assert!(MagiConfig::from_toml_str("[memory]\napi_key = \"x\"").is_err());
        assert!(MagiConfig::from_toml_str("[embedding]\napi_key = \"x\"").is_err());
    }

    #[test]
    fn test_parses_full_memory_and_embedding_sections() {
        // A present section parses its values; an omitted field still resolves to
        // its documented default (CP2-A index="ann" is accepted).
        let toml = "\
[memory]
mode = \"load_all\"
context_budget_tokens = 4000
evicted_retention_days = -1
index = \"ann\"
[embedding]
base_url = \"http://localhost:11434/v1\"
model = \"nomic-embed-text\"
dim = 768
";
        let c = MagiConfig::from_toml_str(toml).unwrap();
        assert_eq!(c.memory.mode, "load_all");
        assert_eq!(c.memory.context_budget_tokens, 4000);
        assert_eq!(c.memory.evicted_retention_days, -1);
        assert_eq!(c.memory.index, "ann");
        assert_eq!(c.memory.seed, 42); // omitted field → documented default
        assert_eq!(c.embedding.model, "nomic-embed-text");
    }
}
