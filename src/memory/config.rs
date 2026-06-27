// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-06-26

//! Configuration structs for the tiered-memory subsystem (stubs — RED phase).
//! Fields present; Default yields zero/empty values until GREEN adds documented defaults.

/// Memory-subsystem runtime configuration (`[memory]` section of `magi.toml`).
#[derive(Debug, Clone, PartialEq, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryConfig {
    #[serde(default)] pub mode: String,
    #[serde(default)] pub context_budget_tokens: usize,
    #[serde(default)] pub response_headroom_tokens: usize,
    #[serde(default)] pub safety_margin_ratio: f64,
    #[serde(default)] pub chars_per_token: f64,
    #[serde(default)] pub oversized_turn_policy: String,
    #[serde(default)] pub top_k: usize,
    #[serde(default)] pub weight_similarity: f64,
    #[serde(default)] pub weight_recency: f64,
    #[serde(default)] pub weight_salience: f64,
    #[serde(default)] pub default_salience: f64,
    #[serde(default)] pub preference_salience: f64,
    #[serde(default)] pub protect_salience_threshold: f64,
    #[serde(default)] pub decay_half_life_days: f64,
    #[serde(default)] pub access_saturation_cap: u64,
    #[serde(default)] pub forget_strength_threshold: f64,
    #[serde(default)] pub evicted_retention_days: i64,
    #[serde(default)] pub max_records: usize,
    #[serde(default)] pub supersede_similarity_threshold: f64,
    #[serde(default)] pub distill_every_n_turns: usize,
    #[serde(default)] pub distill_on_session_close: bool,
    #[serde(default)] pub profile_max_tokens: usize,
    #[serde(default)] pub seed: u64,
    #[serde(default)] pub salience_markers: Vec<String>,
    #[serde(default)] pub index: String,
    #[serde(default)] pub distill_max_batch_tokens: usize,
    #[serde(default)] pub supersede_max_candidate_pairs: usize,
    #[serde(default)] pub distill_enabled: bool,
    #[serde(default)] pub reembed_batch_size: usize,
    #[serde(default)] pub max_evictions_per_pass: usize,
    #[serde(default)] pub migration_throttle_batch: usize,
}

/// Embedding-provider configuration (`[embedding]` section of `magi.toml`).
#[derive(Debug, Clone, PartialEq, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingConfig {
    #[serde(default)] pub provider: String,
    #[serde(default)] pub base_url: String,
    #[serde(default)] pub model: String,
    #[serde(default)] pub dim: usize,
    #[serde(default)] pub query_prefix: String,
    #[serde(default)] pub document_prefix: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MagiConfig;

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
}
