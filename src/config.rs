// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-05-25

//! Persistent magi-rs configuration from `magi.toml`. NON-SECRET only — API keys
//! never live here (env/keyring/key.txt).

// Public API of this module is consumed by `main.rs` (Task 6 wiring) and by
// tests; no items here should be flagged dead_code under any cfg.

use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MagiConfig {
    pub provider: Option<String>,
    #[serde(default)]
    pub openai: OpenAiConfig,
    #[serde(default)]
    pub anthropic: AnthropicConfig,
    #[serde(default)]
    pub magi: MagiModelsConfig,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiConfig {
    pub base_url: Option<String>,
    pub model: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnthropicConfig {
    pub model: Option<String>,
}

/// Per-agent MAGI model overrides (`[magi]` section). All optional; absent ⇒ each
/// agent shares the principal provider's model (backward compatible with v0.4.0).
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MagiModelsConfig {
    /// Override model for Melchior (the Scientist). `None` ⇒ principal model.
    pub melchior_model: Option<String>,
    /// Override model for Balthasar (the Pragmatist). `None` ⇒ principal model.
    pub balthasar_model: Option<String>,
    /// Override model for Caspar (the Critic). `None` ⇒ principal model.
    pub caspar_model: Option<String>,
}

impl MagiConfig {
    /// Parse a `magi.toml` string. Malformed TOML or unknown fields -> `Err` (RF-1).
    pub fn from_toml_str(s: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(s)
    }

    /// Loads `<dir>/magi.toml`. Returns `(config, Option<warning>)`. Absent → defaults,
    /// no warning. Malformed/unknown-field → defaults + a warning string (main.rs
    /// surfaces it as a startup notice — no panic, no silent stderr-only loss).
    ///
    /// Joins `dir` with the literal filename `magi.toml`. Callers should pass a
    /// canonical directory (e.g., from `env::current_dir()?`) so the resolution
    /// is reproducible across the process lifetime. Relative paths are accepted
    /// but their meaning depends on the current working directory at call time;
    /// if the process later changes `cwd`, a relative `dir` will resolve
    /// against a different absolute location.
    ///
    /// # Arguments
    /// * `dir` - Directory in which to look for `magi.toml`. Recommended to be
    ///   canonical/absolute so subsequent code paths cannot drift.
    ///
    /// # Returns
    /// `(MagiConfig, Option<String>)` — the parsed config (or defaults on any
    /// error path) and an optional human-readable warning to surface in the UI.
    pub fn load(dir: &Path) -> (Self, Option<String>) {
        let path = dir.join("magi.toml");
        match std::fs::read_to_string(&path) {
            Ok(s) => match Self::from_toml_str(&s) {
                Ok(c) => (c, None),
                Err(e) => (
                    Self::default(),
                    Some(format!(
                        "Note: {} is invalid and was ignored ({e}); using defaults.",
                        path.display()
                    )),
                ),
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => (Self::default(), None),
            Err(e) => (
                Self::default(),
                Some(format!(
                    "Note: {} could not be read ({e}); using defaults.",
                    path.display()
                )),
            ),
        }
    }
}

/// env `MAGI_PROVIDER` > TOML `provider` > `"anthropic"` (RF-2).
///
/// # Arguments
/// * `config` - Parsed `MagiConfig` from `magi.toml` (may be default if file absent/invalid).
/// * `env_provider` - Value of `MAGI_PROVIDER` env var, if set.
///
/// # Returns
/// Resolved provider name: env overrides TOML; falls back to `"anthropic"`.
pub fn resolve_provider(config: &MagiConfig, env_provider: Option<&str>) -> String {
    env_provider
        .map(str::to_string)
        .or_else(|| config.provider.clone())
        .unwrap_or_else(|| "anthropic".into())
}

/// env `OPENAI_BASE_URL` > TOML `[openai].base_url` > `"https://api.openai.com/v1"` (RF-3).
///
/// # Arguments
/// * `config` - Parsed `MagiConfig`.
/// * `env_base_url` - Value of `OPENAI_BASE_URL` env var, if set.
///
/// # Returns
/// Resolved OpenAI-compatible base URL.
pub fn resolve_openai_base_url(config: &MagiConfig, env_base_url: Option<&str>) -> String {
    env_base_url
        .map(str::to_string)
        .or_else(|| config.openai.base_url.clone())
        .unwrap_or_else(|| "https://api.openai.com/v1".into())
}

/// env `OPENAI_MODEL` > TOML `[openai].model`; **required** — `Err` if absent in both (RF-3).
///
/// # Arguments
/// * `config` - Parsed `MagiConfig`.
/// * `env_model` - Value of `OPENAI_MODEL` env var, if set.
///
/// # Errors
/// Returns `Err` when neither the env var nor the TOML field is set, because the
/// OpenAI-compatible provider cannot operate without a model name.
pub fn resolve_openai_model(
    config: &MagiConfig,
    env_model: Option<&str>,
) -> anyhow::Result<String> {
    env_model
        .map(str::to_string)
        .or_else(|| config.openai.model.clone())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "provider 'openai' selected but no model set \
                 (OPENAI_MODEL env or [openai].model in magi.toml)"
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parses_full_config() {
        let c = MagiConfig::from_toml_str(
            "provider = \"openai\"\n[openai]\nbase_url = \"http://localhost:11434/v1\"\nmodel = \"phi4-mini\"\n[anthropic]\nmodel = \"claude-sonnet-4-6\"\n",
        ).unwrap();
        assert_eq!(c.provider.as_deref(), Some("openai"));
        assert_eq!(
            c.openai.base_url.as_deref(),
            Some("http://localhost:11434/v1")
        );
        assert_eq!(c.openai.model.as_deref(), Some("phi4-mini"));
        assert_eq!(c.anthropic.model.as_deref(), Some("claude-sonnet-4-6"));
    }

    #[test]
    fn test_empty_is_default() {
        assert_eq!(
            MagiConfig::from_toml_str("").unwrap(),
            MagiConfig::default()
        );
    }

    #[test]
    fn test_malformed_is_err() {
        assert!(MagiConfig::from_toml_str("provider = =bad").is_err());
    }

    #[test]
    fn test_unknown_field_is_err() {
        assert!(MagiConfig::from_toml_str("provdier = \"openai\"").is_err());
    }

    // -------------------------------------------------------------------------
    // Task 2: load + resolution tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_load_missing_file_is_default_no_warning() {
        let dir = tempfile::tempdir().unwrap();
        let (c, warn) = MagiConfig::load(dir.path());
        assert_eq!(c, MagiConfig::default());
        assert!(warn.is_none());
    }

    #[test]
    fn test_load_reads_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("magi.toml"), "provider = \"openai\"").unwrap();
        let (c, warn) = MagiConfig::load(dir.path());
        assert_eq!(c.provider.as_deref(), Some("openai"));
        assert!(warn.is_none());
    }

    #[test]
    fn test_load_malformed_yields_default_plus_warning() {
        // RF-1 + MAGI: malformed config does not crash; returns defaults AND a
        // human-facing warning (main.rs surfaces it as a TUI startup notice).
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("magi.toml"), "provdier = \"x\"").unwrap();
        let (c, warn) = MagiConfig::load(dir.path());
        assert_eq!(c, MagiConfig::default());
        assert!(warn.unwrap().contains("magi.toml"));
    }

    #[test]
    fn test_resolve_provider_precedence() {
        let c = MagiConfig {
            provider: Some("anthropic".into()),
            ..Default::default()
        };
        assert_eq!(resolve_provider(&c, Some("openai")), "openai"); // S-2 env wins
        assert_eq!(resolve_provider(&c, None), "anthropic");
        // S-3
        assert_eq!(resolve_provider(&MagiConfig::default(), None), "anthropic");
    }

    #[test]
    fn test_resolve_openai_base_url_precedence() {
        let c = MagiConfig {
            openai: OpenAiConfig {
                base_url: Some("http://toml/v1".into()),
                model: None,
            },
            ..Default::default()
        };
        assert_eq!(
            resolve_openai_base_url(&c, Some("http://env/v1")),
            "http://env/v1"
        );
        assert_eq!(resolve_openai_base_url(&c, None), "http://toml/v1");
        assert_eq!(
            resolve_openai_base_url(&MagiConfig::default(), None),
            "https://api.openai.com/v1"
        );
    }

    #[test]
    fn test_resolve_openai_model_required() {
        assert!(resolve_openai_model(&MagiConfig::default(), None).is_err());
        let c = MagiConfig {
            openai: OpenAiConfig {
                base_url: None,
                model: Some("phi4-mini".into()),
            },
            ..Default::default()
        };
        assert_eq!(resolve_openai_model(&c, None).unwrap(), "phi4-mini");
        assert_eq!(
            resolve_openai_model(&c, Some("gpt-4o-mini")).unwrap(),
            "gpt-4o-mini"
        );
    }

    #[test]
    fn test_load_unreadable_file_yields_default_plus_warning() {
        // A directory named `magi.toml` makes read_to_string fail with a
        // non-NotFound error → must surface a warning, not be treated as absent.
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("magi.toml")).unwrap();
        let (c, warn) = MagiConfig::load(dir.path());
        assert_eq!(c, MagiConfig::default());
        assert!(warn.unwrap().contains("magi.toml"));
    }

    // -------------------------------------------------------------------------
    // Task 1: MagiModelsConfig parsing tests (S-1, S-2, S-3)
    // -------------------------------------------------------------------------

    #[test]
    fn test_parses_magi_section() {
        // S-1
        let c = MagiConfig::from_toml_str(
            "[magi]\nmelchior_model = \"qwen3:8b\"\ncaspar_model = \"deepseek-r1:32b\"\n",
        )
        .unwrap();
        assert_eq!(c.magi.melchior_model.as_deref(), Some("qwen3:8b"));
        assert_eq!(c.magi.balthasar_model, None);
        assert_eq!(c.magi.caspar_model.as_deref(), Some("deepseek-r1:32b"));
    }

    #[test]
    fn test_absent_magi_section_is_default() {
        // S-2
        let c = MagiConfig::from_toml_str("provider = \"anthropic\"").unwrap();
        assert_eq!(c.magi, MagiModelsConfig::default());
    }

    #[test]
    fn test_unknown_field_in_magi_section_is_err() {
        // S-3
        assert!(MagiConfig::from_toml_str("[magi]\nunknown_field = \"x\"").is_err());
    }
}
