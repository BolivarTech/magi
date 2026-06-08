// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-06-08

//! Pure decision logic that maps per-agent MAGI model overrides to adapter specs
//! and the static-path notice. Isolates the `magi_core::schema::AgentName`
//! coupling and the `"{backend}-{agent}"` naming so `main.rs` stays thin glue.

// consumed by main.rs Task 6 wiring; items unused until then
#![allow(dead_code)]

use crate::config::{resolve_magi_override, MagiModelsConfig};
use magi_core::schema::AgentName;

/// Env-var model overrides for the three MAGI agents. Named fields (no positional
/// same-type swap risk — mirrors the `OpenAiSettings` precedent).
#[derive(Debug, Default, Clone)]
pub struct MagiEnvModels {
    pub melchior: Option<String>,
    pub balthasar: Option<String>,
    pub caspar: Option<String>,
}

/// A resolved per-agent provider override: which agent, which model, and the
/// adapter display name (`"{backend}-{agent}"`) for magi-core's report/logs.
#[derive(Debug, Clone, PartialEq)]
pub struct MagiAdapterSpec {
    pub agent: AgentName,
    pub model: String,
    pub adapter_name: String,
}

/// Lowercase label for an agent (matches magi-core's lowercase serialization).
pub fn agent_label(agent: AgentName) -> &'static str {
    match agent {
        AgentName::Melchior => "melchior",
        AgentName::Balthasar => "balthasar",
        AgentName::Caspar => "caspar",
    }
}

/// Computes the per-agent adapter specs for agents with an effective override
/// (env > TOML > none). An empty result means "no overrides — use `Magi::new`".
///
/// # Arguments
/// * `backend_label` - `"openai"` or `"anthropic"` (the resolved principal backend).
/// * `cfg`           - The `[magi]` TOML config.
/// * `env`           - Env-var overrides per agent.
pub fn resolve_magi_adapter_specs(
    backend_label: &str,
    cfg: &MagiModelsConfig,
    env: &MagiEnvModels,
) -> Vec<MagiAdapterSpec> {
    [
        (
            AgentName::Melchior,
            cfg.melchior_model.as_deref(),
            env.melchior.as_deref(),
        ),
        (
            AgentName::Balthasar,
            cfg.balthasar_model.as_deref(),
            env.balthasar.as_deref(),
        ),
        (
            AgentName::Caspar,
            cfg.caspar_model.as_deref(),
            env.caspar.as_deref(),
        ),
    ]
    .into_iter()
    .filter_map(|(agent, toml, env_model)| {
        resolve_magi_override(toml, env_model).map(|model| MagiAdapterSpec {
            agent,
            model,
            adapter_name: format!("{backend_label}-{}", agent_label(agent)),
        })
    })
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(m: Option<&str>, b: Option<&str>, c: Option<&str>) -> MagiModelsConfig {
        MagiModelsConfig {
            melchior_model: m.map(str::to_string),
            balthasar_model: b.map(str::to_string),
            caspar_model: c.map(str::to_string),
        }
    }

    #[test]
    fn test_no_overrides_yields_empty_specs() {
        // S-5: trigger false
        let specs = resolve_magi_adapter_specs(
            "anthropic",
            &MagiModelsConfig::default(),
            &MagiEnvModels::default(),
        );
        assert!(specs.is_empty());
    }

    #[test]
    fn test_empty_string_override_yields_empty_specs() {
        // S-5: empty string is not an override
        let specs = resolve_magi_adapter_specs(
            "anthropic",
            &cfg(Some(""), None, None),
            &MagiEnvModels {
                melchior: Some("  ".into()),
                ..Default::default()
            },
        );
        assert!(specs.is_empty());
    }

    #[test]
    fn test_single_override_anthropic_naming() {
        // S-7
        let specs = resolve_magi_adapter_specs(
            "anthropic",
            &cfg(Some("qwen3:8b"), None, None),
            &MagiEnvModels::default(),
        );
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].agent, AgentName::Melchior);
        assert_eq!(specs[0].model, "qwen3:8b");
        assert_eq!(specs[0].adapter_name, "anthropic-melchior");
    }

    #[test]
    fn test_three_overrides() {
        // S-8
        let specs = resolve_magi_adapter_specs(
            "anthropic",
            &cfg(Some("a"), Some("b"), Some("c")),
            &MagiEnvModels::default(),
        );
        assert_eq!(specs.len(), 3);
        let names: Vec<&str> = specs.iter().map(|s| s.adapter_name.as_str()).collect();
        assert_eq!(
            names,
            [
                "anthropic-melchior",
                "anthropic-balthasar",
                "anthropic-caspar"
            ]
        );
    }

    #[test]
    fn test_openai_naming_and_env_precedence() {
        // S-9 + S-4 via the spec path: env beats TOML, openai backend label
        let specs = resolve_magi_adapter_specs(
            "openai",
            &cfg(None, None, Some("toml-deepseek")),
            &MagiEnvModels {
                caspar: Some("deepseek-r1:32b".into()),
                ..Default::default()
            },
        );
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].agent, AgentName::Caspar);
        assert_eq!(specs[0].model, "deepseek-r1:32b"); // env wins
        assert_eq!(specs[0].adapter_name, "openai-caspar");
    }

    #[test]
    fn test_static_with_overrides_emits_notice() {
        // S-13
        let n = static_override_notice(true, true);
        assert!(n.as_deref().unwrap().contains("per-agent models ignored"));
    }

    #[test]
    fn test_no_notice_when_not_static_or_no_overrides() {
        assert_eq!(static_override_notice(false, true), None);
        assert_eq!(static_override_notice(true, false), None);
        assert_eq!(static_override_notice(false, false), None);
    }
}
