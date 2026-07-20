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
    #[serde(default)]
    pub memory: crate::memory::config::MemoryConfig,
    #[serde(default)]
    pub embedding: crate::memory::config::EmbeddingConfig,
    #[serde(default)]
    pub headless: HeadlessConfig,
}

/// `[headless]` section of `magi.toml` (spec §11). Every field is optional; an
/// unset field falls back to its built-in constant default (see
/// `main.rs::resolve_headless_limits`). Unknown keys are a parse error.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeadlessConfig {
    /// Cap on `-i`/stdin input bytes (REQ-H29). Overrides `MAX_INPUT_BYTES`.
    pub max_input_bytes: Option<usize>,
    /// Elevated tool-call cap under `--full-auto` (REQ-H08). Overrides `FULL_AUTO_MAX_TOOL_CALLS`.
    pub full_auto_max_tool_calls: Option<u32>,
    /// Keep at most the last N run logs (REQ-H34). Overrides `LOG_RETENTION_RUNS`.
    pub log_retention: Option<usize>,
    /// Total log-dir byte ceiling (REQ-H24). Overrides `LOG_MAX_BYTES`.
    pub log_max_bytes: Option<u64>,
    /// Cap on each tool result in the output (REQ-H14). Overrides `TOOL_RESULT_CAP`.
    pub tool_result_cap_bytes: Option<usize>,
    /// Default log level (REQ-H24): `error`|`warn`|`info`|`debug`. Overrides `"info"`.
    pub log_level: Option<String>,
    /// Default wall-clock timeout secs for tool-executing tiers (REQ-H36).
    /// Overrides `FULL_AUTO_TIMEOUT_SECS`.
    pub timeout_secs: Option<u64>,
    /// Whether the envelope may override the operator `system` prompt (REQ-H12b).
    /// Defaults to `false` (the envelope `system` is ignored unless enabled).
    pub allow_system_override: Option<bool>,
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
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MagiModelsConfig {
    /// Override model for Melchior (the Scientist). `None` ⇒ principal model.
    pub melchior_model: Option<String>,
    /// Override model for Balthasar (the Pragmatist). `None` ⇒ principal model.
    pub balthasar_model: Option<String>,
    /// Override model for Caspar (the Critic). `None` ⇒ principal model.
    pub caspar_model: Option<String>,
    /// Auto-approve autonomous MAGI (`consult`) launches when the main LLM
    /// self-routes to the `consult` tool in the agent tool loop. Default `false`
    /// — the agent asks before launching the 3-perspective consensus. `true`
    /// launches without asking, but announces it in the TUI (3 LLM calls take
    /// time). The explicit `/consult` TUI command is NEVER gated — it is always
    /// user-initiated and requires no approval regardless of this flag.
    #[serde(default = "default_auto_approve")]
    pub auto_approve: bool,
}

impl Default for MagiModelsConfig {
    fn default() -> Self {
        Self {
            melchior_model: None,
            balthasar_model: None,
            caspar_model: None,
            auto_approve: default_auto_approve(),
        }
    }
}

/// Default value for [`MagiModelsConfig::auto_approve`]: `false` (require
/// explicit approval before each autonomous MAGI consensus launch).
fn default_auto_approve() -> bool {
    false
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

/// env `MAGI_PROVIDER` > TOML `provider` > `DEFAULT_PROVIDER` (RF-1).
///
/// The no-config default is **Ollama-first** (`"openai"`); Anthropic is opt-in
/// via `provider="anthropic"` in `magi.toml` or `MAGI_PROVIDER=anthropic`.
///
/// # Arguments
/// * `config` - Parsed `MagiConfig` from `magi.toml` (may be default if file absent/invalid).
/// * `env_provider` - Value of `MAGI_PROVIDER` env var, if set.
///
/// # Returns
/// Resolved provider name: env overrides TOML; falls back to `DEFAULT_PROVIDER`.
pub fn resolve_provider(config: &MagiConfig, env_provider: Option<&str>) -> String {
    env_provider
        .map(str::to_string)
        .or_else(|| config.provider.clone())
        .unwrap_or_else(|| crate::defaults::DEFAULT_PROVIDER.into())
}

/// env `OPENAI_BASE_URL` > TOML `[openai].base_url` > `DEFAULT_OPENAI_BASE_URL` (RF-2).
///
/// The no-config default points at local Ollama (`http://localhost:11434/v1`).
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
        .unwrap_or_else(|| crate::defaults::DEFAULT_OPENAI_BASE_URL.into())
}

/// Resolves a per-agent MAGI model override. Precedence: env (non-empty) > TOML
/// (non-empty) > `None`. A blank/whitespace value (env or TOML) is treated as
/// unset and falls through to the next level. `None` means the agent uses the
/// principal provider's model (RF-2, S-4, S-5).
///
/// # Arguments
/// * `toml_model` - The `[magi].<agent>_model` value, if present.
/// * `env_model`  - The `MAGI_MODEL_<AGENT>` env value, if present.
///
/// # Returns
/// `Some(model)` when an effective override exists; `None` otherwise.
pub fn resolve_magi_override(toml_model: Option<&str>, env_model: Option<&str>) -> Option<String> {
    fn non_empty(s: Option<&str>) -> Option<String> {
        s.map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    }
    non_empty(env_model).or_else(|| non_empty(toml_model))
}

/// env `OPENAI_MODEL` > TOML `[openai].model` > `DEFAULT_OPENAI_MODEL` (RF-3).
/// No longer fallible: the openai path has a built-in default (Ollama-first).
///
/// # Arguments
/// * `config` - Parsed `MagiConfig`.
/// * `env_model` - Value of `OPENAI_MODEL` env var, if set.
///
/// # Returns
/// Resolved model name; env overrides TOML, both override the built-in default.
pub fn resolve_openai_model(config: &MagiConfig, env_model: Option<&str>) -> String {
    env_model
        .map(str::to_string)
        .or_else(|| config.openai.model.clone())
        .unwrap_or_else(|| crate::defaults::DEFAULT_OPENAI_MODEL.into())
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
        use crate::defaults::DEFAULT_PROVIDER;
        let c = MagiConfig {
            provider: Some("anthropic".into()),
            ..Default::default()
        };
        assert_eq!(resolve_provider(&c, Some("openai")), "openai"); // env wins
        assert_eq!(resolve_provider(&c, None), "anthropic"); // TOML
                                                             // S-1: no config → DEFAULT_PROVIDER ("openai")
        assert_eq!(
            resolve_provider(&MagiConfig::default(), None),
            DEFAULT_PROVIDER
        );
        assert_eq!(resolve_provider(&MagiConfig::default(), None), "openai");
    }

    #[test]
    fn test_resolve_openai_base_url_precedence() {
        use crate::defaults::DEFAULT_OPENAI_BASE_URL;
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
        // S-1: no config → Ollama
        assert_eq!(
            resolve_openai_base_url(&MagiConfig::default(), None),
            DEFAULT_OPENAI_BASE_URL
        );
        assert_eq!(
            resolve_openai_base_url(&MagiConfig::default(), None),
            "http://localhost:11434/v1"
        );
    }

    #[test]
    fn test_resolve_openai_model_defaults() {
        use crate::defaults::DEFAULT_OPENAI_MODEL;
        // S-2: no env, no TOML → DEFAULT_OPENAI_MODEL (was Err)
        assert_eq!(
            resolve_openai_model(&MagiConfig::default(), None),
            DEFAULT_OPENAI_MODEL
        );
        // S-3: env/TOML still win
        let c = MagiConfig {
            openai: OpenAiConfig {
                base_url: None,
                model: Some("phi4-mini".into()),
            },
            ..Default::default()
        };
        assert_eq!(resolve_openai_model(&c, None), "phi4-mini");
        assert_eq!(resolve_openai_model(&c, Some("gpt-4o-mini")), "gpt-4o-mini");
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

    // ── auto_approve field tests ──────────────────────────────────────────────

    /// Default `[magi]` section (absent or empty) must have `auto_approve = false`.
    ///
    /// RED: fails until `auto_approve` is added to `MagiModelsConfig`.
    #[test]
    fn test_magi_auto_approve_defaults_to_false() {
        let c = MagiConfig::from_toml_str("").unwrap();
        assert!(
            !c.magi.auto_approve,
            "auto_approve must default to false (opt-in, never silently enabled)"
        );
        // Also check that an explicit [magi] section without the field also defaults.
        let c2 = MagiConfig::from_toml_str("[magi]\nmelchior_model = \"qwen3:8b\"").unwrap();
        assert!(
            !c2.magi.auto_approve,
            "auto_approve must default to false even when [magi] section is present"
        );
    }

    /// `[magi] auto_approve = true` must parse to `true`.
    ///
    /// RED: fails until `auto_approve` is added to `MagiModelsConfig`.
    #[test]
    fn test_magi_auto_approve_true_parses() {
        let c = MagiConfig::from_toml_str("[magi]\nauto_approve = true").unwrap();
        assert!(
            c.magi.auto_approve,
            "auto_approve = true in [magi] must parse to true"
        );
    }

    /// `deny_unknown_fields` must still reject genuinely unknown fields even after
    /// adding `auto_approve` (regression guard — field name typos must not silently
    /// apply the default).
    #[test]
    fn test_magi_auto_approve_typo_is_still_rejected() {
        assert!(
            MagiConfig::from_toml_str("[magi]\nauto_approv = true").is_err(),
            "typo 'auto_approv' (missing 'e') must be rejected by deny_unknown_fields"
        );
    }

    // -------------------------------------------------------------------------
    // Task 2: resolve_magi_override precedence tests (S-4, S-5)
    // -------------------------------------------------------------------------

    #[test]
    fn test_resolve_magi_override_env_wins_over_toml() {
        // S-4: env > TOML
        assert_eq!(
            resolve_magi_override(Some("toml-model"), Some("env-model")),
            Some("env-model".to_string())
        );
    }

    #[test]
    fn test_resolve_magi_override_toml_when_no_env() {
        // S-4: TOML when env absent
        assert_eq!(
            resolve_magi_override(Some("toml-model"), None),
            Some("toml-model".to_string())
        );
    }

    #[test]
    fn test_resolve_magi_override_none_when_both_absent() {
        // S-4: none ⇒ principal model
        assert_eq!(resolve_magi_override(None, None), None);
    }

    #[test]
    fn test_resolve_magi_override_empty_string_is_unset() {
        // S-5: empty (env or TOML) is treated as unset, falls through precedence
        assert_eq!(
            resolve_magi_override(Some("toml"), Some("   ")),
            Some("toml".to_string())
        );
        assert_eq!(resolve_magi_override(Some(""), None), None);
        assert_eq!(resolve_magi_override(Some(""), Some("")), None);
    }

    // -------------------------------------------------------------------------
    // [headless] section tests (spec §11). `HeadlessConfig` already derives
    // `#[serde(deny_unknown_fields)]`, so these LOCK the existing parsing
    // contract (documenting, not driving it) rather than being a Red/Green
    // pair — they still fail if a future edit silently loosens the section.
    // -------------------------------------------------------------------------

    /// An unknown key inside `[headless]` (e.g. a typo) is a parse ERROR, not
    /// silent acceptance — `deny_unknown_fields` applies to this section like
    /// every other `MagiConfig` sub-table.
    #[test]
    fn test_headless_section_unknown_field_is_err() {
        assert!(MagiConfig::from_toml_str("[headless]\nmax_input_byte = 1024").is_err());
    }

    /// A `[headless]` block with several keys set parses into the matching
    /// `Option` fields; unset keys stay `None` (resolved to their built-in
    /// default elsewhere, `main.rs::resolve_headless_limits`/
    /// `resolve_log_level`/`resolve_allow_system_override`).
    #[test]
    fn test_headless_section_parses_configured_keys() {
        let c = MagiConfig::from_toml_str(
            "[headless]\n\
             max_input_bytes = 2048\n\
             full_auto_max_tool_calls = 30\n\
             log_retention = 7\n\
             log_max_bytes = 1048576\n\
             tool_result_cap_bytes = 4096\n\
             log_level = \"debug\"\n\
             timeout_secs = 120\n\
             allow_system_override = true\n",
        )
        .unwrap();

        assert_eq!(c.headless.max_input_bytes, Some(2048));
        assert_eq!(c.headless.full_auto_max_tool_calls, Some(30));
        assert_eq!(c.headless.log_retention, Some(7));
        assert_eq!(c.headless.log_max_bytes, Some(1_048_576));
        assert_eq!(c.headless.tool_result_cap_bytes, Some(4096));
        assert_eq!(c.headless.log_level.as_deref(), Some("debug"));
        assert_eq!(c.headless.timeout_secs, Some(120));
        assert_eq!(c.headless.allow_system_override, Some(true));
    }

    /// An absent `[headless]` section parses to all-`None` (every cap falls
    /// back to its built-in default).
    #[test]
    fn test_headless_section_absent_is_all_none() {
        let c = MagiConfig::from_toml_str("").unwrap();
        assert_eq!(c.headless, HeadlessConfig::default());
    }
}
