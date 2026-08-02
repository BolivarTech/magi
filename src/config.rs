// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-05-25

//! Persistent magi-rs configuration from `magi.toml`. NON-SECRET only — API keys
//! never live here (env/keyring/key.txt).

// Public API of this module is consumed by `main.rs` (Task 6 wiring) and by
// tests; no items here should be flagged dead_code under any cfg.

use magi_rs::magi::kind::ProviderKind;
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

/// env `ANTHROPIC_MODEL` > TOML `[anthropic].model` > `DEFAULT_ANTHROPIC_MODEL`.
///
/// Mirrors [`resolve_openai_model`]'s precedence exactly. Fixes a MAGI re-gate
/// WARNING: prior call sites in `main.rs` disagreed on precedence — the
/// headless path checked TOML before env (backwards), and the TUI/other path
/// (`discover_config`) read only env and ignored `[anthropic].model`
/// entirely. Both now route through this single resolver.
///
/// # Arguments
/// * `config` - Parsed `MagiConfig`.
/// * `env_model` - Value of `ANTHROPIC_MODEL` env var, if set.
///
/// # Returns
/// Resolved model name; env overrides TOML, both override the built-in default.
pub fn resolve_anthropic_model(config: &MagiConfig, env_model: Option<&str>) -> String {
    env_model
        .map(str::to_string)
        .or_else(|| config.anthropic.model.clone())
        .unwrap_or_else(|| crate::defaults::DEFAULT_ANTHROPIC_MODEL.into())
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
    fn test_resolve_anthropic_model_env_wins_over_toml() {
        // MAGI re-gate WARNING fix: env must win over TOML (not the other way
        // around, which was the bug in the pre-fix headless call site).
        let c = MagiConfig {
            anthropic: AnthropicConfig {
                model: Some("claude-toml-model".into()),
            },
            ..Default::default()
        };
        assert_eq!(
            resolve_anthropic_model(&c, Some("claude-env-model")),
            "claude-env-model"
        );
    }

    #[test]
    fn test_resolve_anthropic_model_toml_when_no_env() {
        let c = MagiConfig {
            anthropic: AnthropicConfig {
                model: Some("claude-toml-model".into()),
            },
            ..Default::default()
        };
        assert_eq!(resolve_anthropic_model(&c, None), "claude-toml-model");
    }

    #[test]
    fn test_resolve_anthropic_model_default_when_neither() {
        use crate::defaults::DEFAULT_ANTHROPIC_MODEL;
        assert_eq!(
            resolve_anthropic_model(&MagiConfig::default(), None),
            DEFAULT_ANTHROPIC_MODEL
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
    // -------------------------------------------------------------------------
    // Task 1.1: base_url to root + unified provider vocabulary (REQ-A01b, A12, A21)
    // -------------------------------------------------------------------------

    use magi_core::schema::Mode;

    #[test]
    fn provider_kind_accepts_the_three_values_and_rejects_the_rest() {
        assert_eq!(
            ProviderKind::parse("ollama").unwrap(),
            Some(ProviderKind::Ollama)
        );
        assert_eq!(
            ProviderKind::parse("openai-compat").unwrap(),
            Some(ProviderKind::OpenAiCompat)
        );
        assert_eq!(
            ProviderKind::parse("anthropic").unwrap(),
            Some(ProviderKind::Anthropic)
        );
        assert!(ProviderKind::parse("banana").is_err());
        assert!(
            ProviderKind::parse("openai").is_err(),
            "el valor viejo ya no es válido"
        );
    }

    /// REQ-A01b: un valor inválido NO se traga — ni siquiera pasando por los resolutores.
    ///
    /// Este test existe porque el test de `ProviderKind::parse` **no alcanza**: prueba la
    /// unidad, y el fallback silencioso vive en el LLAMADOR. Un resolutor con
    /// `.ok().flatten().unwrap_or(default)` deja pasar `"banana"` como `Ollama` mientras el
    /// test de la unidad sigue verde.
    ///
    /// **Corrección de la ruling del coordinador (2026-08-02).** El plan original probaba
    /// esto contra `MagiConfig::load(&path)`, pero `load` en Task 1.1 conserva su firma
    /// externa `(dir: &Path) -> (Self, Option<String>)` (se vuelve falible recién en Task
    /// 1.4, junto con el cableado de `main.rs`/`Workspace::config_path()` que esa tarea
    /// posee). La propiedad que este test defiende — que un valor inválido no se convierte
    /// en un fallback silencioso — se prueba contra `from_toml_str`, que es donde
    /// `validate_vocabulary()` corre de verdad; `load()` la ejercita indirectamente porque
    /// llama a `from_toml_str` sobre el contenido del archivo. Task 1.4 agrega la misma
    /// aserción contra `load()` cuando esa función se vuelve falible.
    #[test]
    fn an_invalid_vocabulary_value_is_rejected_at_parse_not_swallowed_by_a_resolver() {
        for (toml, what) in [
            ("provider = \"banana\"\n", "provider de raíz"),
            ("[magi]\nkind = \"banana\"\n", "[magi].kind"),
            ("[magi]\ndefault_mode = \"banana\"\n", "[magi].default_mode"),
        ] {
            assert!(
                MagiConfig::from_toml_str(toml).is_err(),
                "{what}: un valor no reconocido debe ser ERROR, nunca un fallback silencioso"
            );
        }
    }

    /// SC-A12g / REQ-A12: regla general — vacío o en blanco es AUSENTE, nunca inválido.
    #[test]
    fn blank_string_keys_are_absent_not_invalid() {
        assert_eq!(ProviderKind::parse("").unwrap(), None);
        assert_eq!(ProviderKind::parse("   ").unwrap(), None);
        let toml = "provider = \"\"\nbase_url = \"  \"\n";
        let cfg = MagiConfig::from_toml_str(toml).expect("vacío no debe romper el parseo");
        assert_eq!(
            cfg.effective_provider(),
            ProviderKind::Ollama,
            "cae al default built-in"
        );
        // `effective_base_url()` devuelve `Result<EndpointTemplate, _>` desde REQ-A16c, así
        // que el test compara el TEXTO de la plantilla.
        assert_eq!(
            cfg.effective_base_url().unwrap().as_str(),
            crate::defaults::DEFAULT_OPENAI_BASE_URL
        );
    }

    /// SC-A02b: `[magi].kind` HEREDA de la raíz cuando no se declara.
    #[test]
    fn magi_kind_inherits_from_root_provider_when_absent() {
        let toml = "provider = \"ollama\"\n[magi]\n";
        let cfg = MagiConfig::from_toml_str(toml).unwrap();
        assert_eq!(cfg.effective_magi_kind(), ProviderKind::Ollama);

        let toml = "provider = \"ollama\"\n[magi]\nkind = \"anthropic\"\n";
        let cfg = MagiConfig::from_toml_str(toml).unwrap();
        assert_eq!(cfg.effective_magi_kind(), ProviderKind::Anthropic);
        assert_eq!(
            cfg.effective_provider(),
            ProviderKind::Ollama,
            "el principal no cambia"
        );
    }

    /// SC-A21c: herencia y override del endpoint.
    #[test]
    fn base_url_inherits_from_root_and_sections_override_it() {
        let toml = "base_url = \"http://lan:11434/v1\"\n[magi]\n";
        let cfg = MagiConfig::from_toml_str(toml).unwrap();
        // Texto de la PLANTILLA: `effective_*_base_url()` devuelve `Result<EndpointTemplate,_>`
        // desde REQ-A16c.
        assert_eq!(
            cfg.effective_magi_base_url().unwrap().as_str(),
            "http://lan:11434/v1"
        );
        assert_eq!(
            cfg.effective_embedding_base_url().unwrap().as_str(),
            "http://lan:11434/v1"
        );

        let toml = "base_url = \"http://lan:11434/v1\"\n[magi]\nbase_url = \"http://otro/v1\"\n";
        let cfg = MagiConfig::from_toml_str(toml).unwrap();
        assert_eq!(
            cfg.effective_magi_base_url().unwrap().as_str(),
            "http://otro/v1"
        );
    }

    /// El campo viejo ya no existe: su presencia es campo desconocido.
    #[test]
    fn openai_section_no_longer_accepts_base_url() {
        let toml = "[openai]\nbase_url = \"http://x/v1\"\n";
        assert!(MagiConfig::from_toml_str(toml).is_err());
    }

    /// REQ-A14: las API keys NUNCA viven en `magi.toml`, y `deny_unknown_fields` lo hace
    /// **mecánico** en vez de una convención que alguien tiene que recordar.
    #[test]
    fn an_api_key_anywhere_in_the_toml_is_a_parse_error() {
        for toml in [
            "api_key = \"sk-secreto\"\n",
            "[openai]\napi_key = \"sk-secreto\"\n",
            "[anthropic]\napi_key = \"sk-secreto\"\n",
            "[magi]\napi_key = \"sk-secreto\"\n",
        ] {
            let err = MagiConfig::from_toml_str(toml)
                .expect_err("una api_key en el TOML debe ser ERROR, no aceptación silenciosa");
            assert!(
                !err.to_string().contains("sk-secreto"),
                "y el error NO puede repetir el secreto que se está rechazando"
            );
        }
    }

    /// REQ-A15: `default_mode` se resuelve con la misma regla vacío=ausente.
    ///
    /// **Devuelve `Option<Mode>`, NO `Result`**: la validación vive en `validate_vocabulary`,
    /// que corre en `load()`/`from_toml_str()`. Un resolutor que devuelve `Result` invita a
    /// que el llamador escriba `.ok()` — y eso ya pasó dos veces en este plan.
    #[test]
    fn effective_default_mode_follows_the_same_blank_is_absent_rule() {
        let cfg = MagiConfig::from_toml_str("[magi]\ndefault_mode = \"code-review\"\n").unwrap();
        assert_eq!(cfg.effective_default_mode(), Some(Mode::CodeReview));

        let cfg = MagiConfig::from_toml_str("[magi]\ndefault_mode = \"\"\n").unwrap();
        assert_eq!(cfg.effective_default_mode(), None);

        // El valor inválido no llega nunca a este resolutor: muere en el parseo.
        assert!(MagiConfig::from_toml_str("[magi]\ndefault_mode = \"banana\"\n").is_err());
    }

    // -------------------------------------------------------------------------
    // B13: cobertura de las funciones públicas restantes que Task 1.1 produce
    // (`Interfaces > Produces` del brief), sin test explícito en el Step 1.
    // -------------------------------------------------------------------------

    /// `seats()` resuelve cada mage a su modelo declarado, o al fallback del backend.
    #[test]
    fn seats_resolves_each_mage_to_its_override_or_the_backend_fallback() {
        let cfg = MagiSectionConfig {
            melchior_model: Some("custom-melchior".into()),
            ..MagiSectionConfig::default()
        };
        let seats = cfg.seats("backend-default");
        assert_eq!(seats.len(), 3);
        assert_eq!(
            seats[0],
            (AgentName::Melchior, "custom-melchior".to_string())
        );
        assert_eq!(
            seats[1],
            (AgentName::Balthasar, "backend-default".to_string())
        );
        assert_eq!(seats[2], (AgentName::Caspar, "backend-default".to_string()));
    }

    /// `fallback_model()` es el modelo del BACKEND, nunca el de un mage — elegir el de
    /// Melchior lo volvería el default por accidente (ver su rustdoc / Task 4.1).
    #[test]
    fn fallback_model_is_the_backend_model_not_any_seats_override() {
        let cfg = MagiSectionConfig {
            melchior_model: Some("should-not-win".into()),
            ..MagiSectionConfig::default()
        };
        assert_eq!(cfg.fallback_model("backend-default"), "backend-default");
    }

    /// `magi_endpoint_diverges()` es true si el trío declara `kind` o `base_url` propios,
    /// y blanco cuenta como NO declarado (REQ-A12).
    #[test]
    fn magi_endpoint_diverges_when_the_trio_declares_its_own_kind_or_base_url() {
        assert!(!MagiConfig::default().magi_endpoint_diverges());

        let own_kind = MagiConfig::from_toml_str("[magi]\nkind = \"anthropic\"\n").unwrap();
        assert!(own_kind.magi_endpoint_diverges());

        let own_url =
            MagiConfig::from_toml_str("[magi]\nbase_url = \"http://other/v1\"\n").unwrap();
        assert!(own_url.magi_endpoint_diverges());

        let blank = MagiConfig::from_toml_str("[magi]\nkind = \"\"\n").unwrap();
        assert!(!blank.magi_endpoint_diverges());
    }

    /// `effective_max_query_bytes()`: declarado gana, ausente cae al built-in.
    #[test]
    fn effective_max_query_bytes_falls_back_to_the_built_in_when_absent() {
        assert_eq!(
            MagiConfig::default().effective_max_query_bytes(),
            magi_rs::magi::MAX_QUERY_BYTES
        );

        let declared = MagiConfig::from_toml_str("[magi]\nmax_query_bytes = 999\n").unwrap();
        assert_eq!(declared.effective_max_query_bytes(), 999);
    }

    /// `effective_tool_result_cap()`: declarado gana, ausente cae al built-in.
    #[test]
    fn effective_tool_result_cap_falls_back_to_the_built_in_when_absent() {
        assert_eq!(
            MagiConfig::default().effective_tool_result_cap(),
            magi_rs::magi::TOOL_RESULT_CAP_BYTES
        );

        let above_min = magi_rs::magi::min_viable_output_cap() + 10;
        let declared = MagiConfig {
            tool_result_cap_bytes: Some(above_min),
            ..Default::default()
        };
        assert_eq!(declared.effective_tool_result_cap(), above_min);
    }
}
