// Author: Julian Bolivar Version: 1.0.0 Date: 2026-05-25

//! Persistent magi-rs configuration from `magi.toml`. NON-SECRET only — API keys never live
//! here (env/keyring/key.txt).

// Public API of this module is consumed by `main.rs` (Task 6 wiring) and by tests; no items
// here should be flagged dead_code under any cfg.

mod migrate;

use std::path::Path;

use magi_core::schema::{AgentName, Mode};
use magi_rs::magi::endpoint::{EndpointError, EndpointTemplate};
use magi_rs::magi::gate::{GateOverrides, GateThresholds};
use magi_rs::magi::kind::{ProviderKind, ProviderKindParseError};
use magi_rs::magi::mode::{ModeExt, ModeParseError};
use magi_rs::magi::{min_viable_output_cap, AGENT_TIMEOUT_MAX_SECS, AGENT_TIMEOUT_MIN_SECS};
use serde::Deserialize;

/// Configuration errors from `magi.toml` (Task 1.1, REQ-A01b/A04/A11b/A21b).
///
/// Lives in the **bin** (not in `magi_rs::magi`) because it is specific to the SHAPE of the
/// TOML; the pure vocabulary error types (`ProviderKindParseError`, `ModeParseError`) live in
/// the lib and are absorbed here with `From`, which is the correct dependency direction (the
/// lib cannot know a type from the bin).
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// `provider` or `[magi].kind` bring a present but unrecognized value.
    #[error("unknown provider: {got:?} (valid: {valid})")]
    UnknownProviderKind {
        /// What the file brought in.
        got: String,
        /// The three accepted values, so the error is actionable.
        valid: &'static str,
    },

    /// `[magi].default_mode` brings a present but unrecognized value.
    #[error("unknown mode: {got:?} (valid: {valid})")]
    UnknownMode {
        /// What the file brought in.
        got: String,
        /// The three accepted labels, so the error is actionable.
        valid: &'static str,
    },

    /// The file brings v0.11.0 migration patterns (REQ-A21b). The text is already rendered by
    /// [`migrate::render_migration_error`] — it names each incompatibility and its correction.
    #[error("{0}")]
    NeedsMigration(String),

    /// The TOML does not parse, or the file could not be read.
    #[error("{0}")]
    Parse(String),

    /// `[magi].agent_timeout_secs` falls outside the acceptable range of §4.9.
    #[error(
        "agent_timeout_secs = {got} out of range [{min}, {max}]: below {min}s a legitimate \
         generation does not fit; above {max}s a consult's worst case (2 attempts per mage) \
         exceeds 4 minutes. Not clamped to the extreme — rejected."
    )]
    AgentTimeoutOutOfRange {
        /// The declared value.
        got: u64,
        /// Floor of the acceptable range (§4.9).
        min: u64,
        /// Ceiling of the acceptable range (§4.9).
        max: u64,
    },

    /// `tool_result_cap_bytes` falls below the minimum viable (REQ-A11b).
    #[error(
        "tool_result_cap_bytes = {got} is below the minimum viable value ({min}): below that \
         threshold not even the truncation mark fits, and the configured cap is silently \
         ignored instead of applied."
    )]
    OutputCapTooSmall {
        /// The declared value.
        got: usize,
        /// The minimum viable ([`magi_rs::magi::min_viable_output_cap`]).
        min: usize,
    },

    /// A `base_url` (root, `[magi]` or `[embedding]`) is not a valid template: it carries a
    /// literal credential instead of the placeholders `[user]:[password]`, an unknown
    /// placeholder, or it could not be traversed (REQ-A16c, SC-A16d).
    ///
    /// The text never repeats the offending value — that is guaranteed by the `Display` of
    /// [`EndpointError`] on which it relies, not by this `#[error]`.
    #[error("{0}")]
    Endpoint(#[from] EndpointError),
}

impl From<ProviderKindParseError> for ConfigError {
    fn from(e: ProviderKindParseError) -> Self {
        Self::UnknownProviderKind {
            got: e.got,
            valid: e.valid,
        }
    }
}

impl From<ModeParseError> for ConfigError {
    fn from(e: ModeParseError) -> Self {
        match e {
            ModeParseError::Unknown { got, valid } => Self::UnknownMode { got, valid },
        }
    }
}

/// Extracts a parse error message **without the offending line** (B11/REQ-A16).
///
/// **`toml::de::Error`'s `Display` reproduces the RAW TEXT of the file around the
/// error** — for `api_key = "sk-secreto"\n` (rejected by `deny_unknown_fields`, REQ-A14) the
/// full `Display` is:
///
/// ```text
/// TOML parse error at line 1, column 1
///   |
/// 1 | api_key = "sk-secreto"
///   | ^^^^^^^
/// unknown field `api_key`, expected `provider`
/// ```
///
/// — the secret being rejected ends up printed in the error message itself. This was discovered
/// by running `an_api_key_anywhere_in_the_toml_is_a_parse_error`: the original `.map_err(|e|
/// ConfigError::Parse(e.to_string()))?` (copied from the body that step-by-step pastes this
/// task's brief) used the full `Display`, and the test failed against the very code the spec
/// asked to transcribe — B9/REQ-A00c demands rejecting that, not transcribing it.
/// `toml::de::Error::message()` gives the semantic message WITHOUT the source excerpt; for the
/// same case it is just `"unknown field \`api_key\`, expected \`provider\`"` — the NAME of the
/// rejected field, never its value, because `deny_unknown_fields` rejects the key before
/// looking at the value type.
///
/// **Position IS RECOVERED (fix round 2, I4) — only the excerpt was prohibited.**
/// `message()` also drops line/column, and SC-A21g requires that a syntax error name a
/// position. `toml::de::Error::span()` gives the range of BYTES of the error without any
/// content — `raw` is traversed counting line breaks up to that byte (it is never sliced nor
/// printed), so the position cannot reintroduce the leak this function exists to avoid.
///
/// Security remains one of the five categories that are never treated as residual, and a type
/// mismatch in a NON-secret field (e.g., `agent_timeout_secs = "x"`) could also, in principle,
/// echo a value in `message()` — closing the path at the root (never show the source excerpt)
/// remains more robust than enumerating "safe" fields; that does not change here.
fn safe_parse_error(e: &toml::de::Error, raw: &str) -> String {
    let message = e.message();
    match e.span() {
        Some(span) => {
            let (line, column) = line_col_of(raw, span.start);
            format!("{message} (line {line}, column {column})")
        }
        None => message.to_string(),
    }
}

/// 1-indexed `(line, column)` position of byte `offset` in `raw`.
///
/// **Never returns anything that is IN `raw`** — it only counts characters and line
/// breaks up to `offset`, so it cannot reintroduce the source excerpt that [`safe_parse_error`]
/// exists to suppress. Without `indexing_slicing`/`string_slice`: it traverses via
/// `char_indices()`, never slicing `raw` by position.
fn line_col_of(raw: &str, offset: usize) -> (usize, usize) {
    let mut line = 1usize;
    let mut column = 1usize;
    for (idx, ch) in raw.char_indices() {
        if idx >= offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            column = 1;
        } else {
            column += 1;
        }
    }
    (line, column)
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MagiConfig {
    pub provider: Option<String>,
    /// Default endpoint OF THE SYSTEM (REQ-A21): used by the main agent, the trio, and the
    /// embedder unless their own section overrides it. Absent ⇒ the built-in.
    /// **BREAKING**: up to v0.11.0 this key lived in `[openai].base_url`, which no longer
    /// exists — see [`ConfigError::NeedsMigration`].
    pub base_url: Option<String>,
    /// Report OUTPUT cap, on ALL THREE paths (TUI, `magi query`, headless consult — REQ-A11b).
    /// Absent ⇒ [`magi_rs::magi::TOOL_RESULT_CAP_BYTES`].
    pub tool_result_cap_bytes: Option<usize>,
    #[serde(default)]
    pub openai: OpenAiConfig,
    #[serde(default)]
    pub anthropic: AnthropicConfig,
    #[serde(default)]
    pub magi: MagiSectionConfig,
    #[serde(default)]
    pub memory: crate::memory::config::MemoryConfig,
    #[serde(default)]
    pub embedding: crate::memory::config::EmbeddingConfig,
    #[serde(default)]
    pub headless: HeadlessConfig,
}

/// `[headless]` section of `magi.toml` (spec §11). Every field is optional; an unset field
/// falls back to its built-in constant default (see `main.rs::resolve_headless_limits`).
/// Unknown keys are a parse error.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeadlessConfig {
    /// Cap on `-i`/stdin input bytes (REQ-H29). Overrides `MAX_INPUT_BYTES`.
    pub max_input_bytes: Option<usize>,
    /// Elevated tool-call cap under `--full-auto` (REQ-H08). Overrides
    /// `FULL_AUTO_MAX_TOOL_CALLS`.
    pub full_auto_max_tool_calls: Option<u32>,
    /// Keep at most the last N run logs (REQ-H34). Overrides `LOG_RETENTION_RUNS`.
    pub log_retention: Option<usize>,
    /// Total log-dir byte ceiling (REQ-H24). Overrides `LOG_MAX_BYTES`.
    pub log_max_bytes: Option<u64>,
    // `tool_result_cap_bytes` NO LONGER LIVES HERE (Task 1.3, third migration pattern of
    // REQ-A21b): it moved up to the root level because under `[headless]` it only covered batch
    // mode and left interactive mode loose, which is exactly where the report is re-sent on
    // every turn of a long session. A cap that protects the cheap case and not the expensive
    // one protects the wrong case. A file that still declares it here receives the guided
    // migration error, not a bare `unknown field` — see `detect_migrations`. Default log level
    // (REQ-H24): `error`|`warn`|`info`|`debug`. Overrides `"info"`.
    pub log_level: Option<String>,
    /// Default wall-clock timeout secs for tool-executing tiers (REQ-H36). Overrides
    /// `FULL_AUTO_TIMEOUT_SECS`.
    pub timeout_secs: Option<u64>,
    /// Whether the envelope may override the operator `system` prompt (REQ-H12b). Defaults to
    /// `false` (the envelope `system` is ignored unless enabled).
    pub allow_system_override: Option<bool>,
}

/// `[openai]` section. Shared by `provider = "ollama"` and `provider = "openai-compat"` — the
/// two providers speak the same Chat-Completions transport and are distinguished only by
/// capability (only `ollama` is probeable, REQ-A24), not by config shape.
///
/// **`base_url` no longer lives here** — it moved to the root of `MagiConfig` (REQ-A21).
/// A `magi.toml` that still declares `[openai].base_url` fails to parse
/// (`deny_unknown_fields`); see [`ConfigError::NeedsMigration`].
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiConfig {
    pub model: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnthropicConfig {
    pub model: Option<String>,
}

/// Sub-table `[magi.complexity]`. Absent ⇒ built-ins; a mode set to `0` ⇒ that mode is never
/// vetoed; a mode missing inside a present table ⇒ its built-in, **not zero** (REQ-A20b).
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComplexityConfig {
    /// Minimum length (characters) to dispatch a self-routed consult in `CodeReview`. Absent ⇒
    /// the library built-in.
    pub code_review: Option<usize>,
    /// See [`Self::code_review`], for `Design`.
    pub design: Option<usize>,
    /// See [`Self::code_review`], for `Analysis`. **It does not inherit the "non-empty" rule
    /// from the magi-core example** (REQ-A20): `Analysis` is the default for every modeless
    /// invocation, so a threshold of 1 would turn off the gate on the most common autonomous
    /// path.
    pub analysis: Option<usize>,
}

/// `[magi]` table. Renamed from `MagiModelsConfig` because it no longer contains only models —
/// Task 1.1 adds the rest of the trio vocabulary (kind, endpoint, mode, complexity gate,
/// timeouts) that the MS2 spec assigns to this section.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MagiSectionConfig {
    /// Override model for Melchior (the Scientist). `None` ⇒ principal model.
    pub melchior_model: Option<String>,
    /// Override model for Balthasar (the Pragmatist). `None` ⇒ principal model.
    pub balthasar_model: Option<String>,
    /// Override model for Caspar (the Critic). `None` ⇒ principal model.
    pub caspar_model: Option<String>,
    /// Auto-approve autonomous MAGI (`consult`) launches when the main LLM self-routes to the
    /// `consult` tool in the agent tool loop. Default `false` — the agent asks before launching
    /// the 3-perspective consensus. `true` launches without asking, but announces it in the TUI
    /// (3 LLM calls take time). The explicit `/consult` TUI command is NEVER gated — it is
    /// always user-initiated and requires no approval regardless of this flag.
    #[serde(default = "default_auto_approve")]
    pub auto_approve: bool,

    /// Provider of the trio; absent ⇒ **inherits** the root one (REQ-A01b). It is not a
    /// heuristic: inheritance reads a declared value, it does not guess an observed one.
    pub kind: Option<String>,
    /// Endpoint of the trio; absent ⇒ inherits the root one (REQ-A21).
    pub base_url: Option<String>,
    /// Fixed mode for every invocation without `--mode`; skips inference (REQ-A07).
    pub default_mode: Option<String>,
    /// Declares that the content under analysis is NOT trustworthy (REQ-A07d). With this
    /// active, omitting the mode is **error**, not inference.
    pub untrusted_content: Option<bool>,
    /// Input cap OF MAGI-RS, before magi-core (REQ-A11b). Absent ⇒
    /// [`magi_rs::magi::MAX_QUERY_BYTES`].
    pub max_query_bytes: Option<usize>,
    /// Ceiling per mage; the two internal retry layers are derived from it (REQ-A04/A15). Must
    /// fall within `[AGENT_TIMEOUT_MIN_SECS, AGENT_TIMEOUT_MAX_SECS]`.
    pub agent_timeout_secs: Option<u64>,
    /// Size warning threshold; absent ⇒ measured by probe, or the magi-core default
    /// (REQ-A15/A24b).
    pub input_warn_tokens: Option<usize>,
    /// Disables transport retry (REQ-A15).
    pub retry_disabled: Option<bool>,
    /// Complexity gate thresholds per mode; absent ⇒ built-ins (REQ-A20b).
    pub complexity: Option<ComplexityConfig>,
}

impl MagiSectionConfig {
    /// The three seats with their resolved model: the declared one, or the backend's.
    ///
    /// # Arguments
    /// * `fallback` - Backend model, used by any seat without an override.
    #[must_use]
    pub fn seats(&self, fallback: &str) -> Vec<(AgentName, String)> {
        vec![
            (
                AgentName::Melchior,
                self.melchior_model
                    .clone()
                    .unwrap_or_else(|| fallback.into()),
            ),
            (
                AgentName::Balthasar,
                self.balthasar_model
                    .clone()
                    .unwrap_or_else(|| fallback.into()),
            ),
            (
                AgentName::Caspar,
                self.caspar_model.clone().unwrap_or_else(|| fallback.into()),
            ),
        ]
    }

    /// Model of the **builder fallback** — the one magi-core would use for an agent without an
    /// override.
    ///
    /// It is the backend's, not any mage's: choosing Melchior's would accidentally make it the
    /// default. With all three seats overridden it is never used, and for that very reason it
    /// should be a written decision.
    ///
    /// # Arguments
    /// * `backend_model` - Resolved default backend model.
    #[must_use]
    pub fn fallback_model<'a>(&self, backend_model: &'a str) -> &'a str {
        backend_model
    }
}

impl Default for MagiSectionConfig {
    fn default() -> Self {
        Self {
            melchior_model: None,
            balthasar_model: None,
            caspar_model: None,
            auto_approve: default_auto_approve(),
            kind: None,
            base_url: None,
            default_mode: None,
            untrusted_content: None,
            max_query_bytes: None,
            agent_timeout_secs: None,
            input_warn_tokens: None,
            retry_disabled: None,
            complexity: None,
        }
    }
}

/// Default value for [`MagiSectionConfig::auto_approve`]: `false` (require explicit approval
/// before each autonomous MAGI consensus launch).
fn default_auto_approve() -> bool {
    false
}

impl MagiConfig {
    /// Parses a `magi.toml` from text, **validating the vocabulary** just like [`Self::load`]
    /// (REQ-A01b).
    ///
    /// **It validates on purpose, even if it is the helper most used by tests.** A
    /// `from_toml_str` that deserializes without validating would let tests build
    /// configurations that `load()` would never accept — the suite would exercise a path that
    /// production does not have, the same class of gap as a resolver `.ok().flatten()` that
    /// swallows an invalid value.
    ///
    /// # Errors
    /// [`ConfigError::NeedsMigration`] if the file brings v0.11.0 patterns (stub: until Task
    /// 1.3 it never occurs — see [`migrate`]); [`ConfigError::Parse`] if the TOML does not
    /// parse; [`ConfigError::UnknownProviderKind`] / [`ConfigError::UnknownMode`] if
    /// `provider`, `[magi].kind` or `[magi].default_mode` bring a present but unrecognized
    /// value; [`ConfigError::AgentTimeoutOutOfRange`] / [`ConfigError::OutputCapTooSmall`] if
    /// those numbers fall outside their range.
    pub fn from_toml_str(s: &str) -> Result<Self, ConfigError> {
        // The migration pass goes FIRST, just like in `load()` (Task 1.4). Without this, tests
        // would exercise an error path that production does not have: a v0.11.0 `magi.toml`
        // would give here serde's bare `unknown field`, while the real user receives the guided
        // message.
        let found = migrate::detect_migrations(s);
        if !found.is_empty() {
            return Err(ConfigError::NeedsMigration(
                migrate::render_migration_error(&found),
            ));
        }
        let cfg: Self =
            toml::from_str(s).map_err(|e| ConfigError::Parse(safe_parse_error(&e, s)))?;
        cfg.validate_vocabulary()?;
        Ok(cfg)
    }

    /// Validates ALL the vocabulary in the file and its numeric ranges. Called from
    /// [`Self::from_toml_str`] (and therefore from `load()`), **before** any `effective_*()`
    /// can swallow an invalid value and silently fall back to the default.
    ///
    /// # Errors
    /// See [`Self::from_toml_str`].
    ///
    /// **Why validation goes here and not in the resolvers.** A resolver that does
    /// `parse(s).ok().flatten().unwrap_or(default)` **swallows the error**: `provider =
    /// "banana"` would silently become `Ollama`, which is exactly the silent fallback that
    /// REQ-A01b forbids. By validating on load, by the time the resolvers run there is nothing
    /// invalid left to swallow.
    fn validate_vocabulary(&self) -> Result<(), ConfigError> {
        ProviderKind::parse(self.provider.as_deref().unwrap_or_default())?;
        ProviderKind::parse(self.magi.kind.as_deref().unwrap_or_default())?;
        <Mode as ModeExt>::parse_config_value(
            self.magi.default_mode.as_deref().unwrap_or_default(),
        )?;
        self.validate_agent_timeout()?;
        self.validate_output_cap()?;
        Ok(())
    }

    /// `agent_timeout_secs` outside the range of §4.9 is a **configuration error**.
    ///
    /// # Errors
    /// [`ConfigError::AgentTimeoutOutOfRange`] with the value, the range, and the why.
    ///
    /// **It is not clamped to the limit, it is rejected** — same criterion as the probe window
    /// (REQ-A16b): clamping turns a value the operator mistyped into a plausible one, and then
    /// the system behaves differently from what the file says.
    /// It exists because without this REQ-A04 would be **breakable from `magi.toml`**: with a
    /// ceiling below the absolute floor of the derivation, the internal floors win and the sum
    /// exceeds the ceiling. "Impossible by construction" is only true if the input range is
    /// bounded.
    ///
    /// Rejects an output cap below the minimum viable (REQ-A11b).
    fn validate_agent_timeout(&self) -> Result<(), ConfigError> {
        let Some(secs) = self.magi.agent_timeout_secs else {
            return Ok(()); // ausente ⇒ el default built-in, ya válido
        };
        if (AGENT_TIMEOUT_MIN_SECS..=AGENT_TIMEOUT_MAX_SECS).contains(&secs) {
            return Ok(());
        }
        Err(ConfigError::AgentTimeoutOutOfRange {
            got: secs,
            min: AGENT_TIMEOUT_MIN_SECS,
            max: AGENT_TIMEOUT_MAX_SECS,
        })
    }

    /// # Errors
    ///
    /// [`ConfigError::OutputCapTooSmall`] with the received value and the minimum.
    /// Effective provider of the main agent: root key, or the built-in default.
    fn validate_output_cap(&self) -> Result<(), ConfigError> {
        let Some(cap) = self.tool_result_cap_bytes else {
            return Ok(());
        };
        let min = min_viable_output_cap();
        if cap < min {
            return Err(ConfigError::OutputCapTooSmall { got: cap, min });
        }
        Ok(())
    }

    /// **Infallible by precondition:** [`Self::validate_vocabulary`] already ran in
    ///
    /// [`Self::from_toml_str`]/`load()`, so the only possible `None` is the absent-or-empty
    /// one.
    /// Task 4.1: consumed in production by `resolve_effective_provider_kind` (main agent
    /// backend) and by `build_magi_orchestrator`/`effective_magi_kind` (trio kind, via
    /// inheritance). Covered by `blank_string_keys_are_absent_not_invalid` and
    /// `magi_kind_inherits_from_root_provider_when_absent`.
    ///
    /// I5 (review round 2): restored. `MagiConfig`'s fields are `pub` and it derives `Default`,
    /// so `MagiConfig { provider: Some("banana".into()), ..Default::default() }` compiles and,
    /// without this, would silently return `Ollama` — the precondition this function's own doc
    /// calls "infallible by precondition" is exactly what this checks.
    #[must_use]
    pub fn effective_provider(&self) -> ProviderKind {
        // Mode declared in `[magi].default_mode`, or `None` if absent/empty (REQ-A15).
        debug_assert!(
            self.validate_vocabulary().is_ok(),
            "load() debe haber validado"
        );
        ProviderKind::parse(self.provider.as_deref().unwrap_or_default())
            .unwrap_or(None)
            .unwrap_or(ProviderKind::Ollama)
    }

    /// **Returns `Option`, not `Result`, on purpose.** With `Result`, every caller
    ///
    /// would end up writing `.ok().flatten()` and swallowing the `ConfigError` — turning a
    /// `default_mode = "banana"` into "no mode declared, infer it". An invalid value dies in
    /// `load()`/`from_toml_str()`; by the time anyone calls here, the only possible answer is
    /// already "yes, this mode" or "none at all".
    /// Same precondition as [`Self::effective_provider`].
    ///
    /// Consumed in production by `run_consult_subcommand` (`main.rs`, REQ-A15): the
    /// `Configured` level of `resolve_mode_guarded`'s five-level precedence. Covered by
    /// `effective_default_mode_follows_the_same_blank_is_absent_rule`.
    ///
    /// I5 (review round 2): restored — same precondition/rationale as `effective_provider`'s
    /// `debug_assert!`.
    #[must_use]
    pub fn effective_default_mode(&self) -> Option<Mode> {
        // Trio `kind`: declared, or **inherited** from the main one (REQ-A01b).
        debug_assert!(
            self.validate_vocabulary().is_ok(),
            "load() debe haber validado"
        );
        <Mode as ModeExt>::parse_config_value(self.magi.default_mode.as_deref().unwrap_or_default())
            .unwrap_or(None)
    }

    /// Inheritance is NOT a heuristic: a heuristic guesses from an observed datum (e.g., the
    /// port); inheritance reads a declared value. There is nothing to guess wrong.
    ///
    /// Same precondition as [`Self::effective_provider`].
    ///
    /// Task 4.1: consumed in production by `build_magi_orchestrator` (`main.rs`), AFTER the
    /// latter validates the raw `kind` on its own — the `.unwrap_or(None)` here swallows an
    /// unrecognized value just like `effective_provider`, so `build_magi_orchestrator` cannot
    /// depend on this accessor to report its typed invalid-kind error; it does so BEFORE, with
    /// its own `ProviderKind::parse`. Covered by
    /// `magi_kind_inherits_from_root_provider_when_absent`.
    ///
    /// `true` if the trio runs on a different endpoint or kind from the main one.
    #[must_use]
    pub fn effective_magi_kind(&self) -> ProviderKind {
        ProviderKind::parse(self.magi.kind.as_deref().unwrap_or_default())
            .unwrap_or(None)
            .unwrap_or_else(|| self.effective_provider())
    }

    /// **It is decided on what is DECLARED, not by comparing resolved URLs.** Two different
    ///
    /// templates can resolve to the same host — one with vault credentials and one without —
    /// and comparing the result would say "they do not diverge" for a configuration that does.
    /// What matters here is the operator's intention.
    /// Task 4.4: consumed in production by `divergence_notice` (`main.rs`, REQ-A07p) — the
    /// `#[allow(dead_code)]` it had was removed here because there is now a real caller.
    /// Covered by `magi_endpoint_diverges_when_the_trio_declares_its_own_kind_or_base_url`.
    ///
    /// Input cap OF magi-rs, before magi-core's `max_input_len` (REQ-A11b).
    #[must_use]
    pub fn magi_endpoint_diverges(&self) -> bool {
        let declara_url = self
            .magi
            .base_url
            .as_deref()
            .is_some_and(|s| !s.trim().is_empty());
        let declara_kind = self
            .magi
            .kind
            .as_deref()
            .is_some_and(|s| !s.trim().is_empty());
        declara_url || declara_kind
    }

    /// Rationale for the number: **cost, not capacity**. magi-core already skips models where
    /// the prompt does not fit, so this does not protect the model — it bounds the expense, and
    /// the payload is paid for three because it goes to the three mages.
    ///
    /// Consumed by `check_query_size` on ALL THREE entry paths (Task 6.2, REQ-A11b, SC-A11c):
    /// `ConsultTool::execute`, the direct headless path (`headless_runner::analyze_direct`) and
    /// the explicit `/consult` of the TUI.
    ///
    /// Report OUTPUT cap, on ALL THREE paths (REQ-A11b).
    #[must_use]
    pub fn effective_max_query_bytes(&self) -> usize {
        self.magi
            .max_query_bytes
            .unwrap_or(magi_rs::magi::MAX_QUERY_BYTES)
    }

    /// **It lives at the root and not in `[headless]`**: under `[headless]` it would only cover
    ///
    /// batch mode and leave interactive mode loose, which is exactly where the report is re-
    /// sent on every turn of a long session. A cap that protects the cheap case and not the
    /// expensive one protects the wrong case. The `allow(dead_code)` this had was removed in
    /// Task 1.3: `resolve_headless_limits` already consumes it. Task 6.2 closes the other two
    /// paths: `register_consult_tool_if_available` (main.rs) passes it to
    /// `ConsultTool::with_output_cap` for the TUI and `magi query` tool loop, and
    /// `TuiMagiRuntimeConfig::tool_result_cap` applies it to the explicit `/consult` of the TUI
    /// via `truncate_report`.
    /// System endpoint: declared root, or the built-in default.
    #[must_use]
    pub fn effective_tool_result_cap(&self) -> usize {
        self.tool_result_cap_bytes
            .unwrap_or(magi_rs::magi::TOOL_RESULT_CAP_BYTES)
    }

    /// Returns the TEMPLATE, not an already usable `&str`: resolving credentials requires the
    /// vault, and that is the only path to a usable endpoint (REQ-A16c).
    ///
    /// # Errors
    ///
    /// [`EndpointError`] if the declared value is not a valid template (literal credential,
    /// unknown placeholder, or untraversable URL). See
    /// [`magi_rs::magi::endpoint::EndpointTemplate::parse`].
    /// Resolves a section's override (`[magi].base_url`/`[embedding].base_url`) against the
    /// same effective system endpoint, or inherits if there is no override.
    pub fn effective_base_url(&self) -> Result<EndpointTemplate, EndpointError> {
        EndpointTemplate::parse(
            self.base_url
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .unwrap_or(crate::defaults::DEFAULT_OPENAI_BASE_URL),
        )
    }

    /// Shared by [`Self::effective_magi_base_url`] and [`Self::effective_embedding_base_url`]:
    /// both apply **exactly** the same rule ("own, blank-is-absent, else inheritance") and
    /// repeating it in each would be the kind of duplication that B3 forbids — desynchronizing
    /// them would change the rule in one section without anyone noticing in the other.
    ///
    /// # Errors
    ///
    /// See [`Self::effective_base_url`].
    /// Trio endpoint: override of `[magi].base_url`, or inheritance from the system.
    fn override_or_inherit_base_url(
        &self,
        own: Option<&str>,
    ) -> Result<EndpointTemplate, EndpointError> {
        match own.map(str::trim).filter(|s| !s.is_empty()) {
            Some(own) => EndpointTemplate::parse(own),
            None => self.effective_base_url(), // herencia, ya validada
        }
    }

    /// # Errors
    ///
    /// See [`Self::effective_base_url`].
    /// Consumed by [`Self::load`] (Task 1.4) to validate that the trio's template is usable
    /// BEFORE startup finishes — it also closes SC-A16d for `[magi].base_url`, not only for the
    /// root and the embedder. The actual native trio construction on this value remains Phase
    /// 4; here it is only validated.
    ///
    /// Embedder endpoint: override of `[embedding].base_url`, or inheritance from the system
    /// (REQ-A21 — behavior change from v0.11.0, see
    /// [`crate::memory::config::EmbeddingConfig::base_url`]).
    pub fn effective_magi_base_url(&self) -> Result<EndpointTemplate, EndpointError> {
        self.override_or_inherit_base_url(self.magi.base_url.as_deref())
    }

    /// # Errors
    ///
    /// See [`Self::effective_base_url`].
    /// Loads `magi.toml` from its **path** (not its directory) — Task 1.4 finally consumes
    /// `Workspace::config_path()` (REQ-A22b).
    pub fn effective_embedding_base_url(&self) -> Result<EndpointTemplate, EndpointError> {
        self.override_or_inherit_base_url(self.embedding.base_url.as_deref())
    }

    /// An **absent** file returns the built-in defaults, `Ok`, with no notices. An **empty or
    /// whitespace-only** file too: every root field is optional, so an empty TOML is a valid
    /// TOML that declares zero things (SC-A21f).
    ///
    /// **Behavior change from v0.11.0 (REQ-A23).** There a broken file
    ///
    /// produced *warning + defaults* — with `base_url` moving to root, that path would silently
    /// discard the whole file and the user would run with defaults believing their config
    /// applies. A **present** `magi.toml` that does not parse, that declares an unrecognized
    /// `provider`/`[magi].kind`/`[magi].default_mode`, or that declares a `base_url` (root,
    /// `[magi]` or `[embedding]`) with a literal credential instead of the REQ-A16c
    /// placeholders, **terminates the process** — it never silently degrades to defaults.
    /// # Errors
    ///
    /// - [`ConfigError::NeedsMigration`] if the file brings v0.11.0 patterns.
    /// - [`ConfigError::Parse`] if it exists and does not parse, or could not be read.
    /// - [`ConfigError::UnknownProviderKind`] / [`ConfigError::UnknownMode`] if
    /// `provider`, `[magi].kind` or `[magi].default_mode` bring a present but unrecognized
    /// value.
    /// - [`ConfigError::AgentTimeoutOutOfRange`] / [`ConfigError::OutputCapTooSmall`] if
    /// those numbers fall outside their range.
    /// - [`ConfigError::Endpoint`] if the root, `[magi]` or `[embedding]` `base_url` carries
    /// a literal credential, an unknown placeholder, or could not be traversed (SC-A16d) —
    /// before this, ONLY the embedder path (`main.rs::attach_persistent_memory`) saw this
    /// error, and degraded it to a notice + plain-text memory instead of stopping startup.
    /// # Arguments
    ///
    /// * `path` - Path to the `magi.toml` file. Recommended absolute/canonical (e.g.
    /// `Workspace::config_path()`) so resolution is reproducible.
    /// # Returns
    ///
    /// `(MagiConfig, Vec<String>)` — the parsed config and the notices from REQ-A12b/A12c about
    /// resolutions that did not come from what was written in the file.
    /// Delegates ALL shape+vocabulary validation to `from_toml_str` (migration, safe parsing
    /// via `safe_parse_error`, vocabulary, numeric ranges) — repeating it here would duplicate
    /// exactly the logic that function centralizes (B3) and, worse, would re-filter the
    /// offending line of a malformed TOML through the raw `Display` of `toml::de::Error` (see
    /// `safe_parse_error`'s doc).
    pub fn load(path: &Path) -> Result<(Self, Vec<String>), ConfigError> {
        let raw = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok((Self::default(), Vec::new()));
            }
            Err(e) => {
                return Err(ConfigError::Parse(format!(
                    "{}: could not be read ({e})",
                    path.display()
                )))
            }
        };

        // REQ-A16c/SC-A16d: a `base_url` with a literal credential, an unknown placeholder, or
        // an untraversable URL stops startup HERE, in ALL THREE sections that can declare it.
        // `from_toml_str` does NOT validate this on purpose: its own tests
        // (`resolve_effective_*_endpoint_propagates_a_malformed_ template_error`) build a
        // malformed template and observe the error at the RESOLUTION point, not at parse time —
        // moving it there would break them. Endpoint validation lives at the production
        // boundary, `load()`.
        let cfg = Self::from_toml_str(&raw).map_err(|e| attach_path(e, path))?;

        // Notices from REQ-A12b/A12c: every resolution that does not come from what was written
        // in the file is announced, so the user does not have to guess what ended up applying.
        // Called only from [`Self::load`], on a config that already passed
        // `validate_vocabulary` and endpoint validation — so its internal `effective_*()` can
        // never fail here.
        cfg.effective_base_url()?;
        cfg.effective_magi_base_url()?;
        cfg.effective_embedding_base_url()?;

        let notices = cfg.resolution_notices();
        Ok((cfg, notices))
    }

    /// `DEFAULT_PROVIDER` is already the REQ-A01b vocabulary value ("ollama"), the same one
    /// `effective_provider()` falls to when `provider` is absent/empty (Task 4.1 collapsed the
    /// separate legacy constant).
    fn resolution_notices(&self) -> Vec<String> {
        let mut out = Vec::new();

        if self
            .provider
            .as_deref()
            .is_some_and(|s| s.trim().is_empty())
        {
            out.push(format!(
                "notice: `provider` is empty; using the default `{}`",
                // The TEMPLATE is shown by its text and **not redacted**: by REQ-A16c it cannot
                // contain a secret (a literal credential is a config error, already rejected by
                // `load()` before reaching here), so running it through `redact_url` would be
                // redundant *and* wrongly typed.
                crate::defaults::DEFAULT_PROVIDER,
            ));
        }

        // m2 (fix round 2, coordinator, 2026-08-03): `if let`, NOT `let … else { return out }`.
        // Before, a failed `effective_base_url()` cut the WHOLE function — including the two
        // Anthropic checks below, which do not depend on this template — relying on "`load()`
        // already validated". That guarantee is real TODAY but lives in `load()`, a different
        // function: a future caller of `resolution_notices()`, or a reorder within `load()`,
        // would silently produce zero Anthropic notices. Restricting the `if let` to this
        // single block costs zero and removes the coupling — see
        // `a_failed_root_base_url_does_not_swallow_the_anthropic_notices`.
        //
        // REQ-A12c: with `anthropic`, the root `base_url` is NOT used for the main agent —
        // Anthropic has its own endpoint. There are TWO sub-cases and both warn, for different
        // reasons:
        if let Ok(root) = self.effective_base_url() {
            if root.as_str() != crate::defaults::DEFAULT_OPENAI_BASE_URL
                && self.embedding.base_url.is_none()
            {
                out.push(format!(
                    "notice: the embedder inherits `base_url = {}` from the root; declare it \
                     in [embedding] if you want a different one",
                    root.as_str(),
                ));
            }
        }

        // (a) the user DECLARED a base_url  => they think it is used, and it is not (b) the
        // Ollama default remained      => looks like a migration oversight
        //
        // Same inconsistency one level down: the Anthropic trio with its own declared base_url,
        // which is also not used.
        if self.effective_provider() == ProviderKind::Anthropic {
            let declared = self.base_url.is_some();
            out.push(if declared {
                "notice: with `provider = \"anthropic\"` the root `base_url` is NOT used for \
                 the main agent (Anthropic uses its own endpoint); it only applies to [magi] \
                 and [embedding] if they inherit it"
                    .to_string()
            } else {
                "notice: `provider = \"anthropic\"` with the default Ollama `base_url`. That \
                 value is NOT used for the main agent; if you wanted Ollama, fix `provider`"
                    .to_string()
            });
        }

        // Prefixes a [`ConfigError::Parse`] with the path of the offending file; the other
        // variants are already self-contained (they name the field, the value, or the range)
        // and do not need the path to be actionable.
        if self.effective_magi_kind() == ProviderKind::Anthropic && self.magi.base_url.is_some() {
            out.push(
                "notice: with `[magi].kind = \"anthropic\"` the `[magi].base_url` is NOT \
                 used: Anthropic uses its own endpoint"
                    .to_string(),
            );
        }

        out
    }
}

/// Effective backend of the main agent: env `MAGI_PROVIDER` > TOML `provider` >
/// `DEFAULT_PROVIDER` (RF-1, REQ-A01b).
fn attach_path(e: ConfigError, path: &Path) -> ConfigError {
    match e {
        ConfigError::Parse(msg) => ConfigError::Parse(format!("{}: {msg}", path.display())),
        other => other,
    }
}

/// Task 4.1: removes the `legacy_backend_label`/`resolve_provider` shim that normalized the new
/// vocabulary (`ollama`/`openai-compat`/`anthropic`) onto the legacy label `"openai"` so that
/// the `provider_kind == "openai"` chain in `main.rs` kept working without touching it. With
/// that chain migrated to `ProviderKind` (same task), there is nothing left to normalize: the
/// vocabulary is unique end to end.
///
/// **`MAGI_PROVIDER` receives the same treatment as `provider`/`[magi].kind` in the TOML**:
///
/// a present but unrecognized value is an explicit error (REQ-A01b), not a silent fallback —
/// unlike the removed shim, which let ANY old env-var value pass through unchecked. Empty or
/// blank is treated as absent (REQ-A12).
/// # Arguments
///
/// * `config` - Parsed `MagiConfig` from `magi.toml` (may be default if file absent/invalid).
/// * `env_provider` - Value of `MAGI_PROVIDER` env var, if set.
/// # Errors
///
/// [`ProviderKindParseError`] if `MAGI_PROVIDER` is present and not one of the three vocabulary
/// values.
/// Resolves a per-agent MAGI model override. Precedence: env (non-empty) > TOML (non-empty) >
/// `None`. A blank/whitespace value (env or TOML) is treated as unset and falls through to the
/// next level. `None` means the agent uses the backend's model (RF-2, S-4, S-5).
pub fn resolve_effective_provider_kind(
    config: &MagiConfig,
    env_provider: Option<&str>,
) -> Result<ProviderKind, ProviderKindParseError> {
    if let Some(raw) = env_provider {
        if let Some(kind) = ProviderKind::parse(raw)? {
            return Ok(kind);
        }
    }
    Ok(config.effective_provider())
}

/// Restored fix round 1 (coordinator, 2026-08-03): Task 4.1 deleted this along with
/// `agent::magi_wiring` (its only caller, the retired per-agent-adapter machinery) on the
/// reasoning that the native trio's `build_magi_orchestrator` had no env-override parameter in
/// the brief's pasted signature. That reasoning does not survive R-A03: "the only admitted
/// breakages are those declared in REQ-A21, REQ-A22 and REQ-A23" — three, not four — and
/// `MAGI_MODEL_*` appears nowhere in `spec-behavior.md` as an authorized removal. Silence plus
/// R-A03 means the capability stays. `main.rs`'s `build_magi_orchestrator` now takes an env-
/// override parameter and calls this for each seat, layered on top of
/// [`MagiSectionConfig::seats`]'s TOML-or-backend resolution — giving the full `env > TOML >
/// backend's model` chain.
///
/// # Arguments
///
/// * `toml_model` - The `[magi].<agent>_model` value, if present.
/// * `env_model`  - The `MAGI_MODEL_<AGENT>` env value, if present.
/// # Returns
///
/// `Some(model)` when an effective override exists; `None` otherwise.
/// env `OPENAI_MODEL` > TOML `[openai].model` > `DEFAULT_OPENAI_MODEL` (RF-3). No longer
/// fallible: the openai path has a built-in default (Ollama-first).
pub fn resolve_magi_override(toml_model: Option<&str>, env_model: Option<&str>) -> Option<String> {
    fn non_empty(s: Option<&str>) -> Option<String> {
        s.map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    }
    non_empty(env_model).or_else(|| non_empty(toml_model))
}

/// # Arguments
///
/// * `config` - Parsed `MagiConfig`.
/// * `env_model` - Value of `OPENAI_MODEL` env var, if set.
/// # Returns
///
/// Resolved model name; env overrides TOML, both override the built-in default.
/// env `ANTHROPIC_MODEL` > TOML `[anthropic].model` > `DEFAULT_ANTHROPIC_MODEL`.
pub fn resolve_openai_model(config: &MagiConfig, env_model: Option<&str>) -> String {
    env_model
        .map(str::to_string)
        .or_else(|| config.openai.model.clone())
        .unwrap_or_else(|| crate::defaults::DEFAULT_OPENAI_MODEL.into())
}

/// Mirrors [`resolve_openai_model`]'s precedence exactly. Fixes a MAGI re-gate WARNING: prior
/// call sites in `main.rs` disagreed on precedence — the headless path checked TOML before env
/// (backwards), and the TUI/other path (`discover_config`) read only env and ignored
/// `[anthropic].model` entirely. Both now route through this single resolver.
///
/// # Arguments
///
/// * `config` - Parsed `MagiConfig`.
/// * `env_model` - Value of `ANTHROPIC_MODEL` env var, if set.
/// # Returns
///
/// Resolved model name; env overrides TOML, both override the built-in default.
/// Builds the complexity gate thresholds from `[magi.complexity]` (REQ-A20b).
pub fn resolve_anthropic_model(config: &MagiConfig, env_model: Option<&str>) -> String {
    env_model
        .map(str::to_string)
        .or_else(|| config.anthropic.model.clone())
        .unwrap_or_else(|| crate::defaults::DEFAULT_ANTHROPIC_MODEL.into())
}

/// **It lives here, and not in `magi_rs::magi::gate` — moved from Task 1.1 (see
///
/// `.superpowers/sdd/claude-plan-tdd/ORDER-FIXES.md`, #1).** `gate.rs` lives in the lib and
/// cannot know the shape of the TOML; breaking `[magi.complexity]` into loose pieces
/// (`GateOverrides`) is this module's job, since it already has the table in hand.
/// Absent table ⇒ `GateOverrides::default()` ⇒ the three built-ins from
/// [`GateThresholds::builtin`] (the gate is not turned off by omitting the section). Narrow
/// allow: consumed by the TUI/`magi query`/`magi consult` autonomous-routing wiring in Tasks
/// 3.2/3.3, not this task. Covered by
/// `gate_thresholds_from_reads_the_complexity_table_and_falls_back_to_builtins`.
///
/// -------------------------------------------------------------------------
#[allow(dead_code)]
#[must_use]
pub fn gate_thresholds_from(config: &MagiConfig) -> GateThresholds {
    let overrides = config
        .magi
        .complexity
        .as_ref()
        .map(|c| GateOverrides {
            code_review: c.code_review,
            design: c.design,
            analysis: c.analysis,
        })
        .unwrap_or_default();
    GateThresholds::from_overrides(overrides)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Task 3.1: `gate_thresholds_from` — breaks `[magi.complexity]` into `GateThresholds`
    // (REQ-A20b). `gate.rs` lives in the lib and cannot know the shape of the TOML; this
    // function is the only one that breaks it apart.
    // -------------------------------------------------------------------------
    // The table populates the declared thresholds and inherits the built-in for those absent
    // WITHIN a present table; without a table, the three built-ins.

    /// `provider = "openai"` and `[openai].base_url` are both v0.11.0 shapes (Task 1.1 breaks
    /// both, REQ-A21/A01b) — the root-level `base_url` and the `ollama`/`openai-
    /// compat`/`anthropic` vocabulary replace them.
    #[test]
    fn gate_thresholds_from_reads_the_complexity_table_and_falls_back_to_builtins() {
        let with_table =
            MagiConfig::from_toml_str("[magi.complexity]\ncode_review = 50\nanalysis = 0\n")
                .unwrap();
        let t = gate_thresholds_from(&with_table);
        assert_eq!(t.code_review, 50, "declarado: se usa el valor del archivo");
        assert_eq!(
            t.design,
            GateThresholds::builtin().design,
            "ausente DENTRO de la tabla: su built-in, no cero"
        );
        assert_eq!(
            t.analysis, 0,
            "0 declarado se preserva: es la vía de apagar ESE modo"
        );

        let without_table = MagiConfig::default();
        assert_eq!(
            gate_thresholds_from(&without_table),
            GateThresholds::builtin(),
            "tabla ausente ⇒ built-ins: el gate no se apaga por omitir la sección"
        );
    }

    #[test]
    fn test_parses_full_config() {
        // -------------------------------------------------------------------------
        let c = MagiConfig::from_toml_str(
            "provider = \"ollama\"\nbase_url = \"http://localhost:11434/v1\"\n[openai]\nmodel = \"phi4-mini\"\n[anthropic]\nmodel = \"claude-sonnet-4-6\"\n",
        ).unwrap();
        assert_eq!(c.provider.as_deref(), Some("ollama"));
        assert_eq!(c.base_url.as_deref(), Some("http://localhost:11434/v1"));
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

    // Task 2: load + resolution tests
    // -------------------------------------------------------------------------
    // Task 1.4: `load` takes a FILE path now, not a directory, and returns `Result<(Self,
    // Vec<String>), ConfigError>`.

    #[test]
    fn test_load_missing_file_is_default_no_warning() {
        // Task 1.1: `"openai"` is no longer a valid `provider` value (REQ-A01b) — `"anthropic"`
        // exercises the same "a real value round-trips through load()" property without
        // depending on the retired vocabulary.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("magi.toml");
        let (c, notices) = MagiConfig::load(&path).unwrap();
        assert_eq!(c, MagiConfig::default());
        assert!(notices.is_empty());
    }

    #[test]
    fn test_load_reads_file() {
        // NOT asserting `notices.is_empty()`: `provider = "anthropic"` with no declared
        // `base_url` is exactly SC-A12d's sub-case (b) — the built-in Ollama default is still
        // sitting there, unused by the principal provider, and REQ-A12c requires a notice about
        // it. `silent_resolutions_are_announced_as_notices` and
        // `anthropic_flags_both_the_declared_and_the_defaulted_base_url` cover that notice
        // directly; this test's only job is the round-trip.
        //
        // Task 4.1: replaces `test_resolve_provider_precedence` (deleted along with the retired
        // `resolve_provider`/`legacy_backend_label` shim). Same env > TOML > default
        // precedence, expressed directly in the REQ-A01b vocabulary — no more legacy label to
        // normalize onto.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("magi.toml");
        std::fs::write(&path, "provider = \"anthropic\"").unwrap();
        let (c, _notices) = MagiConfig::load(&path).unwrap();
        assert_eq!(c.provider.as_deref(), Some("anthropic"));
    }

    /// S-1: no config, no env → the built-in default (Ollama-first).
    #[test]
    fn test_resolve_effective_provider_kind_precedence() {
        let c = MagiConfig {
            provider: Some("anthropic".into()),
            ..Default::default()
        };
        assert_eq!(
            resolve_effective_provider_kind(&c, Some("ollama")).unwrap(),
            ProviderKind::Ollama, // env wins
        );
        assert_eq!(
            resolve_effective_provider_kind(&c, None).unwrap(),
            ProviderKind::Anthropic, // TOML
        );
        // Task 4.1: `MAGI_PROVIDER` gets the SAME explicit-error treatment as `provider`/
        // `[magi].kind` in the TOML — an unrecognized value is never a silent fallback
        // (REQ-A01b). The retired shim used to let ANY old env-var value pass through
        // unchecked; that asymmetry does not survive the migration.
        assert_eq!(
            resolve_effective_provider_kind(&MagiConfig::default(), None).unwrap(),
            ProviderKind::Ollama,
        );
    }

    /// SC-A12g / REQ-A12: a blank `MAGI_PROVIDER` is treated as ABSENT, not invalid — falls
    /// through to the TOML/default the same as an unset env var.
    #[test]
    fn an_unrecognized_env_provider_is_a_configuration_error() {
        let err =
            resolve_effective_provider_kind(&MagiConfig::default(), Some("banana")).unwrap_err();
        assert!(err.to_string().contains("banana"));
    }

    /// `test_resolve_openai_base_url_precedence` removed (fix round 3, L1/L2/S1):
    /// `resolve_openai_base_url` — the function it tested — bypassed blank-is-absent and
    /// credential resolution entirely, and was removed in favor of `main.rs`'s
    /// `resolve_effective_principal_endpoint` (which reuses `MagiConfig::effective_base_url`
    /// and is covered where it lives, alongside `resolve_effective_embedding_endpoint`).
    #[test]
    fn a_blank_env_provider_falls_through_to_the_toml_default() {
        let c = MagiConfig {
            provider: Some("anthropic".into()),
            ..Default::default()
        };
        assert_eq!(
            resolve_effective_provider_kind(&c, Some("   ")).unwrap(),
            ProviderKind::Anthropic,
        );
    }

    // S-2: no env, no TOML → DEFAULT_OPENAI_MODEL (was Err)

    #[test]
    fn test_resolve_openai_model_defaults() {
        use crate::defaults::DEFAULT_OPENAI_MODEL;
        // S-3: env/TOML still win
        assert_eq!(
            resolve_openai_model(&MagiConfig::default(), None),
            DEFAULT_OPENAI_MODEL
        );
        // MAGI re-gate WARNING fix: env must win over TOML (not the other way around, which was
        // the bug in the pre-fix headless call site).
        let c = MagiConfig {
            openai: OpenAiConfig {
                model: Some("phi4-mini".into()),
            },
            ..Default::default()
        };
        assert_eq!(resolve_openai_model(&c, None), "phi4-mini");
        assert_eq!(resolve_openai_model(&c, Some("gpt-4o-mini")), "gpt-4o-mini");
    }

    #[test]
    fn test_resolve_anthropic_model_env_wins_over_toml() {
        // A directory named `magi.toml` makes read_to_string fail with a non-NotFound error →
        // REQ-A23: must be FATAL, never degrade to defaults + warning.
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
    fn test_load_unreadable_file_is_fatal() {
        // -------------------------------------------------------------------------
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("magi.toml");
        std::fs::create_dir(&path).unwrap();
        let err = MagiConfig::load(&path).expect_err("unreadable magi.toml must be fatal");
        assert!(err.to_string().contains("magi.toml"));
    }

    // Task 1: MagiSectionConfig parsing tests (S-1, S-2, S-3)
    // -------------------------------------------------------------------------
    // S-1

    #[test]
    fn test_parses_magi_section() {
        // S-2
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
        // S-3
        let c = MagiConfig::from_toml_str("provider = \"anthropic\"").unwrap();
        assert_eq!(c.magi, MagiSectionConfig::default());
    }

    #[test]
    fn test_unknown_field_in_magi_section_is_err() {
        // ── auto_approve field tests ──────────────────────────────────────────────
        assert!(MagiConfig::from_toml_str("[magi]\nunknown_field = \"x\"").is_err());
    }

    // Default `[magi]` section (absent or empty) must have `auto_approve = false`.

    /// Historical: added when `auto_approve` first landed on this section (now
    /// `MagiSectionConfig`).
    ///
    /// Also check that an explicit [magi] section without the field also defaults.
    #[test]
    fn test_magi_auto_approve_defaults_to_false() {
        let c = MagiConfig::from_toml_str("").unwrap();
        assert!(
            !c.magi.auto_approve,
            "auto_approve must default to false (opt-in, never silently enabled)"
        );
        // `[magi] auto_approve = true` must parse to `true`.
        let c2 = MagiConfig::from_toml_str("[magi]\nmelchior_model = \"qwen3:8b\"").unwrap();
        assert!(
            !c2.magi.auto_approve,
            "auto_approve must default to false even when [magi] section is present"
        );
    }

    /// Historical: added when `auto_approve` first landed on this section (now
    /// `MagiSectionConfig`).
    ///
    /// `deny_unknown_fields` must still reject genuinely unknown fields even after adding
    /// `auto_approve` (regression guard — field name typos must not silently apply the
    /// default).
    #[test]
    fn test_magi_auto_approve_true_parses() {
        let c = MagiConfig::from_toml_str("[magi]\nauto_approve = true").unwrap();
        assert!(
            c.magi.auto_approve,
            "auto_approve = true in [magi] must parse to true"
        );
    }

    /// -------------------------------------------------------------------------
    #[test]
    fn test_magi_auto_approve_typo_is_still_rejected() {
        assert!(
            MagiConfig::from_toml_str("[magi]\nauto_approv = true").is_err(),
            "typo 'auto_approv' (missing 'e') must be rejected by deny_unknown_fields"
        );
    }

    // Task 2 / restored fix round 1: resolve_magi_override precedence tests (S-4, S-5).
    // -------------------------------------------------------------------------
    // S-4: env > TOML

    #[test]
    fn test_resolve_magi_override_env_wins_over_toml() {
        // S-4: TOML when env absent
        assert_eq!(
            resolve_magi_override(Some("toml-model"), Some("env-model")),
            Some("env-model".to_string())
        );
    }

    #[test]
    fn test_resolve_magi_override_toml_when_no_env() {
        // S-4: none ⇒ principal model
        assert_eq!(
            resolve_magi_override(Some("toml-model"), None),
            Some("toml-model".to_string())
        );
    }

    #[test]
    fn test_resolve_magi_override_none_when_both_absent() {
        // S-5: empty (env or TOML) is treated as unset, falls through precedence
        assert_eq!(resolve_magi_override(None, None), None);
    }

    #[test]
    fn test_resolve_magi_override_empty_string_is_unset() {
        // -------------------------------------------------------------------------
        assert_eq!(
            resolve_magi_override(Some("toml"), Some("   ")),
            Some("toml".to_string())
        );
        assert_eq!(resolve_magi_override(Some(""), None), None);
        assert_eq!(resolve_magi_override(Some(""), Some("")), None);
    }

    // [headless] section tests (spec §11). `HeadlessConfig` already derives
    // `#[serde(deny_unknown_fields)]`, so these LOCK the existing parsing contract
    // (documenting, not driving it) rather than being a Red/Green pair — they still fail if a
    // future edit silently loosens the section.
    // -------------------------------------------------------------------------
    // An unknown key inside `[headless]` (e.g. a typo) is a parse ERROR, not silent acceptance
    // — `deny_unknown_fields` applies to this section like every other `MagiConfig` sub-table.

    /// A `[headless]` block with several keys set parses into the matching `Option` fields;
    /// unset keys stay `None` (resolved to their built-in default elsewhere,
    /// `main.rs::resolve_headless_limits`/
    /// `resolve_log_level`/`resolve_allow_system_override`).
    #[test]
    fn test_headless_section_unknown_field_is_err() {
        assert!(MagiConfig::from_toml_str("[headless]\nmax_input_byte = 1024").is_err());
    }

    /// An absent `[headless]` section parses to all-`None` (every cap falls back to its built-
    /// in default).
    #[test]
    fn test_headless_section_parses_configured_keys() {
        let c = MagiConfig::from_toml_str(
            "[headless]\n\
             max_input_bytes = 2048\n\
             full_auto_max_tool_calls = 30\n\
             log_retention = 7\n\
             log_max_bytes = 1048576\n\
             log_level = \"debug\"\n\
             timeout_secs = 120\n\
             allow_system_override = true\n",
        )
        .unwrap();

        assert_eq!(c.headless.max_input_bytes, Some(2048));
        assert_eq!(c.headless.full_auto_max_tool_calls, Some(30));
        assert_eq!(c.headless.log_retention, Some(7));
        assert_eq!(c.headless.log_max_bytes, Some(1_048_576));
        assert_eq!(c.headless.log_level.as_deref(), Some("debug"));
        assert_eq!(c.headless.timeout_secs, Some(120));
        assert_eq!(c.headless.allow_system_override, Some(true));
    }

    /// -------------------------------------------------------------------------
    #[test]
    fn test_headless_section_absent_is_all_none() {
        let c = MagiConfig::from_toml_str("").unwrap();
        assert_eq!(c.headless, HeadlessConfig::default());
    }

    // Task 1.1: base_url to root + unified provider vocabulary (REQ-A01b, A12, A21)
    // -------------------------------------------------------------------------
    // REQ-A01b: an invalid value is NOT swallowed — not even passing through the resolvers.

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

    /// This test exists because the `ProviderKind::parse` test is **not enough**: it tests the
    /// unit, and the silent fallback lives in the CALLER. A resolver with
    /// `.ok().flatten().unwrap_or(default)` lets `"banana"` through as `Ollama` while the unit
    /// test stays green.
    ///
    /// **Correction to the coordinator's ruling (2026-08-02).** The original plan tested
    ///
    /// this against `MagiConfig::load(&path)`, but `load` in Task 1.1 keeps its external
    /// signature `(dir: &Path) -> (Self, Option<String>)` (it only becomes fallible in Task
    /// 1.4, together with the `main.rs`/`Workspace::config_path()` wiring that that task owns).
    /// The property this test defends — that an invalid value does not become a silent fallback
    /// — is tested against `from_toml_str`, which is where `validate_vocabulary()` really runs;
    /// `load()` exercises it indirectly because it calls `from_toml_str` on the file contents.
    /// Task 1.4 adds the same assertion against `load()` when that function becomes fallible.
    /// SC-A12g / REQ-A12: general rule — empty or blank is ABSENT, never invalid.
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

    /// `effective_base_url()` returns `Result<EndpointTemplate, _>` since REQ-A16c, so the test
    /// compares the TEXT of the template.
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
        // SC-A02b: `[magi].kind` INHERITS from the root when not declared.
        assert_eq!(
            cfg.effective_base_url().unwrap().as_str(),
            crate::defaults::DEFAULT_OPENAI_BASE_URL
        );
    }

    /// SC-A21c: endpoint inheritance and override.
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

    /// Template TEXT: `effective_*_base_url()` returns `Result<EndpointTemplate,_>` since
    /// REQ-A16c.
    #[test]
    fn base_url_inherits_from_root_and_sections_override_it() {
        let toml = "base_url = \"http://lan:11434/v1\"\n[magi]\n";
        let cfg = MagiConfig::from_toml_str(toml).unwrap();
        // The old field no longer exists: its presence is an unknown field.
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

    /// SC-A12 / REQ-A14: an unknown field is a PARSE ERROR, not silent acceptance.
    #[test]
    fn openai_section_no_longer_accepts_base_url() {
        let toml = "[openai]\nbase_url = \"http://x/v1\"\n";
        assert!(MagiConfig::from_toml_str(toml).is_err());
    }

    /// API keys NEVER live in `magi.toml`, and `deny_unknown_fields` makes it **mechanical**
    /// instead of a convention someone has to remember. Closes SC-A12 with the case that
    /// matters most: the misspelled field that would also be a secret.
    ///
    /// REQ-A15: `default_mode` resolves with the same empty=absent rule.
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

    /// **Returns `Option<Mode>`, NOT `Result`**: validation lives in `validate_vocabulary`,
    ///
    /// which runs in `load()`/`from_toml_str()`. A resolver that returns `Result` invites the
    /// caller to write `.ok()` — and that already happened twice in this plan.
    /// The invalid value never reaches this resolver: it dies at parse time.
    #[test]
    fn effective_default_mode_follows_the_same_blank_is_absent_rule() {
        let cfg = MagiConfig::from_toml_str("[magi]\ndefault_mode = \"code-review\"\n").unwrap();
        assert_eq!(cfg.effective_default_mode(), Some(Mode::CodeReview));

        let cfg = MagiConfig::from_toml_str("[magi]\ndefault_mode = \"\"\n").unwrap();
        assert_eq!(cfg.effective_default_mode(), None);

        // -------------------------------------------------------------------------
        assert!(MagiConfig::from_toml_str("[magi]\ndefault_mode = \"banana\"\n").is_err());
    }

    // B13: coverage of the remaining public functions that Task 1.1 produces (`Interfaces >
    // Produces` from the brief), with no explicit test in Step 1.
    // -------------------------------------------------------------------------
    // `seats()` resolves each mage to its declared model, or to the backend fallback.

    /// `fallback_model()` is the BACKEND's model, never a mage's — choosing Melchior's would
    /// accidentally make it the default (see its rustdoc / Task 4.1).
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

    /// `magi_endpoint_diverges()` is true if the trio declares its own `kind` or `base_url`,
    /// and blank counts as NOT declared (REQ-A12).
    #[test]
    fn fallback_model_is_the_backend_model_not_any_seats_override() {
        let cfg = MagiSectionConfig {
            melchior_model: Some("should-not-win".into()),
            ..MagiSectionConfig::default()
        };
        assert_eq!(cfg.fallback_model("backend-default"), "backend-default");
    }

    /// SC-A02c ("absent" half): `kind = ""` is treated as absent and inherits — the `blank`
    /// below. The other half (`kind = "banana"` ⇒ trio not buildable) closes in Phase 4.
    ///
    /// `effective_max_query_bytes()`: declared wins, absent falls back to built-in.
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

    /// `effective_tool_result_cap()`: declared wins, absent falls back to built-in.
    #[test]
    fn effective_max_query_bytes_falls_back_to_the_built_in_when_absent() {
        assert_eq!(
            MagiConfig::default().effective_max_query_bytes(),
            magi_rs::magi::MAX_QUERY_BYTES
        );

        let declared = MagiConfig::from_toml_str("[magi]\nmax_query_bytes = 999\n").unwrap();
        assert_eq!(declared.effective_max_query_bytes(), 999);
    }

    /// -------------------------------------------------------------------------
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

    // Fix round 2 (coordinator review, 2026-08-02): I3/I4/I5/m8 — B13 coverage this task's own
    // new functions shipped without.
    // -------------------------------------------------------------------------
    // I3: both range boundaries of `agent_timeout_secs` (§4.9) are accepted; one step outside
    // either end is rejected. `validate_agent_timeout` shipped with zero tests and an
    // inclusive-both-ends range with nothing pinning the edge.

    /// I3: the output-cap floor (`min_viable_output_cap()`) itself is accepted; one byte below
    /// it is rejected. Same "zero tests on a boundary" gap as `validate_agent_timeout`.
    #[test]
    fn agent_timeout_secs_accepts_both_boundaries_and_rejects_one_step_outside() {
        for ok in [AGENT_TIMEOUT_MIN_SECS, AGENT_TIMEOUT_MAX_SECS] {
            let toml = format!("[magi]\nagent_timeout_secs = {ok}\n");
            assert!(
                MagiConfig::from_toml_str(&toml).is_ok(),
                "{ok}s is inside [{AGENT_TIMEOUT_MIN_SECS}, {AGENT_TIMEOUT_MAX_SECS}] and must be accepted"
            );
        }
        for bad in [AGENT_TIMEOUT_MIN_SECS - 1, AGENT_TIMEOUT_MAX_SECS + 1] {
            let toml = format!("[magi]\nagent_timeout_secs = {bad}\n");
            assert!(
                MagiConfig::from_toml_str(&toml).is_err(),
                "{bad}s is one step outside the range and must be rejected"
            );
        }
    }

    /// I4: `safe_parse_error` keeps line/column (SC-A21g requires a syntax error to name a
    /// position) but never the offending value — only the source EXCERPT needed suppressing to
    /// fix the `api_key` leak (see `safe_parse_error`'s own doc), not the position.
    #[test]
    fn tool_result_cap_bytes_accepts_the_minimum_viable_floor_and_rejects_one_byte_below() {
        let min = magi_rs::magi::min_viable_output_cap();
        let ok_toml = format!("tool_result_cap_bytes = {min}\n");
        assert!(
            MagiConfig::from_toml_str(&ok_toml).is_ok(),
            "the floor itself must be accepted"
        );
        let bad_toml = format!("tool_result_cap_bytes = {}\n", min - 1);
        assert!(
            MagiConfig::from_toml_str(&bad_toml).is_err(),
            "one byte below the floor must be rejected"
        );
    }

    /// Leading blank line: line 2 (not 1) pins that this is a real computed position, not a
    /// hardcoded "line 1, column 1".
    #[test]
    fn safe_parse_error_keeps_the_position_but_drops_the_offending_value() {
        // I5: `effective_provider` is documented "infallible by precondition" — that
        // precondition is `validate_vocabulary` having already run. `MagiConfig`'s fields are
        // `pub` and it derives `Default`, so nothing at the type level stops a caller from
        // skipping `from_toml_str`/`load()` and constructing an invalid config directly; the
        // `debug_assert!` is what turns that misuse into a loud debug-build panic instead of a
        // silent `Ollama` fallback.
        let toml = "\napi_key = \"sk-secreto\"\n";
        let err = MagiConfig::from_toml_str(toml).unwrap_err();
        let msg = err.to_string();
        assert!(!msg.contains("sk-secreto"), "leaked the secret: {msg}");
        assert!(
            msg.contains("line") && msg.contains("column"),
            "lost the position: {msg}"
        );
        assert!(msg.contains("line 2"), "wrong line: {msg}");
    }

    /// I5: same precondition, same gap, for `effective_default_mode`.
    #[test]
    #[should_panic(expected = "validado")]
    fn effective_provider_panics_in_debug_builds_when_validate_vocabulary_was_skipped() {
        let cfg = MagiConfig {
            provider: Some("banana".into()),
            ..Default::default()
        };
        let _ = cfg.effective_provider();
    }

    /// m8: `[magi].base_url = ""` is blank, not a declared override — it must inherit the root,
    /// not be treated as "the trio declared its own endpoint".
    #[test]
    #[should_panic(expected = "validado")]
    fn effective_default_mode_panics_in_debug_builds_when_validate_vocabulary_was_skipped() {
        let cfg = MagiConfig {
            magi: MagiSectionConfig {
                default_mode: Some("banana".into()),
                ..MagiSectionConfig::default()
            },
            ..Default::default()
        };
        let _ = cfg.effective_default_mode();
    }

    /// m8: a whitespace-only `default_mode` is blank, not a value — same blank-is-absent rule
    /// as every other vocabulary key.
    #[test]
    fn magi_base_url_blank_is_treated_as_absent() {
        let toml = "base_url = \"http://lan:11434/v1\"\n[magi]\nbase_url = \"\"\n";
        let cfg = MagiConfig::from_toml_str(toml).unwrap();
        assert_eq!(
            cfg.effective_magi_base_url().unwrap().as_str(),
            "http://lan:11434/v1",
            "blank must inherit the root, not stay blank"
        );
        assert!(
            !cfg.magi_endpoint_diverges(),
            "a blank override does not count as declaring one (REQ-A12)"
        );
    }

    /// m8: `[embedding].base_url`'s OVERRIDE winning over the root was untested — the existing
    /// coverage only proved inheritance, never that a declared embedding-specific endpoint
    /// takes precedence over it.
    #[test]
    fn default_mode_whitespace_only_is_treated_as_absent() {
        let cfg = MagiConfig::from_toml_str("[magi]\ndefault_mode = \"   \"\n").unwrap();
        assert_eq!(cfg.effective_default_mode(), None);
    }

    /// The root and [magi] are unaffected by the embedding-only override.
    #[test]
    fn embedding_base_url_override_wins_over_root_inheritance() {
        let toml = "base_url = \"http://lan:11434/v1\"\n\
                     [embedding]\n\
                     base_url = \"http://embedding-only:11434/v1\"\n";
        let cfg = MagiConfig::from_toml_str(toml).unwrap();
        assert_eq!(
            cfg.effective_embedding_base_url().unwrap().as_str(),
            "http://embedding-only:11434/v1"
        );
        // -------------------------------------------------------------------------
        assert_eq!(
            cfg.effective_base_url().unwrap().as_str(),
            "http://lan:11434/v1"
        );
    }

    // Task 1.4: `load` fallible, resolution notices, `--init-config` retirement
    // -------------------------------------------------------------------------
    // REQ-A01b through the PRODUCTION path, not just the parser.

    /// Mandatory complement to
    /// `an_invalid_vocabulary_value_is_rejected_at_parse_not_swallowed_by_a_resolver` (Task
    /// 1.1): the former tests `from_toml_str`, this one tests that `load()` no longer degrades
    /// the error to defaults-plus-warning — which is exactly what it did between the close of
    /// Task 1.1 and the close of this one (known intermediate gap, see ORDER-FIXES rupture #8).
    ///
    /// REQ-A23: present and does not parse ⇒ FATAL. Absent ⇒ silent default.
    #[test]
    fn an_invalid_vocabulary_value_is_fatal_through_load_not_degraded_to_defaults() {
        for (toml, what) in [
            ("provider = \"banana\"\n", "provider de raíz"),
            ("[magi]\nkind = \"banana\"\n", "[magi].kind"),
            ("[magi]\ndefault_mode = \"banana\"\n", "[magi].default_mode"),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("magi.toml");
            std::fs::write(&path, toml).unwrap();
            assert!(
                MagiConfig::load(&path).is_err(),
                "{what}: por load() también debe ser ERROR, nunca defaults + warning"
            );
        }
    }

    /// m6 (fix round 2, coordinator, 2026-08-03) / SC-A21f: a PRESENT file but empty or
    /// whitespace-only is also silent — every root field is optional, so a blank TOML is a
    /// valid TOML that declares zero things. `from_toml_str("")` and `detect_migrations("")`
    /// were already covered separately; this is the only thing missing: `load()` end-to-end
    /// against a real FILE, which is what its own rustdoc claims as covered.
    #[test]
    fn a_present_but_broken_config_is_fatal_while_an_absent_one_is_silent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("magi.toml");

        assert!(
            MagiConfig::load(&path).is_ok(),
            "ausente: default silencioso"
        );

        // SECOND OBLIGATION inherited from Task 1.1: SC-A16d, CLOSED failure on a literal
        // credential in `base_url` — in ALL THREE sections that can declare it, not only the
        // one the embedder path (`attach_persistent_memory`) already covered by degrading to a
        // notice + plain-text memory (which SC-A16d forbids).
        std::fs::write(&path, "   \n").unwrap();
        let (cfg, notices) = MagiConfig::load(&path).expect("en blanco: default silencioso");
        assert_eq!(cfg, MagiConfig::default());
        assert!(notices.is_empty());

        std::fs::write(&path, "provdier = \"x\"").unwrap();
        let err = MagiConfig::load(&path).expect_err("presente y roto: FATAL");
        assert!(err.to_string().contains("magi.toml"));
    }

    /// Affirms the TWO halves the requirement asks for: that startup fails, and that the
    /// message does not repeat the value of the found credential.
    ///
    /// m3 (fix round 2, coordinator, 2026-08-03): the password was already covered; the USER
    /// was not — and this test lives in `config.rs`, so a regression in
    /// `EndpointError::LiteralCredential` (which only carries `&'static str`, never the
    /// received value) would not be seen by the module the obligation names if we only pinned
    /// the half.
    #[test]
    fn a_literal_credential_in_any_base_url_scope_fails_closed_at_load() {
        for (toml, scope) in [
            ("base_url = \"https://alice:s3cr3t@host/v1\"\n", "raíz"),
            (
                "[magi]\nbase_url = \"https://alice:s3cr3t@host/v1\"\n",
                "[magi]",
            ),
            (
                "[embedding]\nbase_url = \"https://alice:s3cr3t@host/v1\"\n",
                "[embedding]",
            ),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("magi.toml");
            std::fs::write(&path, toml).unwrap();
            let err = match MagiConfig::load(&path) {
                Err(e) => e,
                Ok(_) => {
                    panic!("{scope}: load() debió fallar ante una credencial literal en base_url")
                }
            };
            let msg = err.to_string();
            // And "it does not leak" is not enough: the message has to be ACTIONABLE — name the
            // correct placeholder and the vault command, not just say "invalid credential".
            assert!(
                !msg.contains("s3cr3t"),
                "{scope}: filtró la contraseña: {msg}"
            );
            assert!(!msg.contains("alice"), "{scope}: filtró el usuario: {msg}");
            // REQ-A12b: silent resolutions are announced.
            assert!(
                msg.contains("[user]") && msg.contains("[password]"),
                "{scope}: no nombra el placeholder: {msg}"
            );
            assert!(
                msg.contains("magi-rs vault set"),
                "{scope}: no nombra el comando para arreglarlo: {msg}"
            );
        }
    }

    /// SC-A12d: inconsistent combination detected ON LOAD, in its TWO sub-cases.
    #[test]
    fn silent_resolutions_are_announced_as_notices() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("magi.toml");

        std::fs::write(&path, "provider = \"\"\n").unwrap();
        let (_, notices) = MagiConfig::load(&path).unwrap();
        assert!(notices.iter().any(|n| n.contains("provider")));

        std::fs::write(&path, "base_url = \"http://lan:11434/v1\"\n").unwrap();
        let (_, notices) = MagiConfig::load(&path).unwrap();
        assert!(
            notices.iter().any(|n| n.contains("embedder")),
            "el embedder hereda un base_url NO-default: hay que decirlo"
        );

        std::fs::write(&path, "base_url = \"http://localhost:11434/v1\"\n").unwrap();
        let (_, notices) = MagiConfig::load(&path).unwrap();
        assert!(
            !notices.iter().any(|n| n.contains("embedder")),
            "heredar el DEFAULT no es sorprendente: sería ruido en cada arranque"
        );
    }

    /// (b) Ollama default sitting there — the case an `is_some()` guard did NOT cover.
    #[test]
    fn anthropic_flags_both_the_declared_and_the_defaulted_base_url() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("magi.toml");

        // (a) declared explicitly — the user thinks it is used.
        std::fs::write(&path, "provider = \"anthropic\"\n").unwrap();
        let (_, notices) = MagiConfig::load(&path).unwrap();
        assert!(
            notices.iter().any(|n| n.contains("default Ollama")),
            "sin base_url declarado el default sigue ahí, y parece un olvido de migración"
        );

        // And the same case one level down, in the trio.
        std::fs::write(
            &path,
            "provider = \"anthropic\"\nbase_url = \"http://x/v1\"\n",
        )
        .unwrap();
        let (_, notices) = MagiConfig::load(&path).unwrap();
        assert!(notices.iter().any(|n| n.contains("NOT used")));

        // Without Anthropic there is nothing to warn about.
        std::fs::write(
            &path,
            "[magi]\nkind = \"anthropic\"\nbase_url = \"http://x/v1\"\n",
        )
        .unwrap();
        let (_, notices) = MagiConfig::load(&path).unwrap();
        assert!(notices.iter().any(|n| n.contains("[magi].base_url")));

        // Needle must match production's actual casing ("NOT used", line ~1902 above) — the
        // lowercase "not used" this assertion used before never matches, so it was vacuously
        // true regardless of whether the code was correct.
        std::fs::write(&path, "provider = \"ollama\"\n").unwrap();
        let (_, notices) = MagiConfig::load(&path).unwrap();
        // m2 (fix round 2, coordinator, 2026-08-03): a failed `effective_base_url()` must NOT
        // silence the Anthropic inconsistency notices that follow it.
        assert!(!notices.iter().any(|n| n.contains("NOT used")));
    }

    /// `resolution_notices()` only runs today inside `load()`, AFTER `load()` already validated
    /// the three templates — so in production `effective_base_url()` never fails here. But that
    /// guarantee lives in `load()`, a DIFFERENT function: if `resolution_notices()` is ever
    /// called from elsewhere, or if `load()` is reordered, the original `let Ok(root) = … else
    /// { return out }` would cut the WHOLE function at the first implicit `?` — including the
    /// two Anthropic checks that have nothing to do with `effective_base_url()` — with no
    /// signal that coverage was lost.
    ///
    /// `resolution_notices()` is called DIRECTLY (access from `mod tests` to the parent
    /// module's private), not through `load()`: it is the only way to put this function under
    /// the precondition that its own `else` says never occurs, without duplicating `load()`'s
    /// validation in the test.
    ///
    /// Test precondition: the root template DOES fail (literal credential), so
    /// `resolution_notices()` runs exactly under the condition its own comment said "infallible
    /// in practice".
    #[test]
    fn a_failed_root_base_url_does_not_swallow_the_anthropic_notices() {
        let cfg = MagiConfig::from_toml_str(
            "provider = \"anthropic\"\nbase_url = \"https://alice:s3cr3t@host/v1\"\n",
        )
        .unwrap();
        // Test precondition: the root template DOES fail (literal credential), so
        // `resolution_notices()` runs under exactly the condition that its own comment
        // called "infallible in practice".
        assert!(cfg.effective_base_url().is_err());

        let notices = cfg.resolution_notices();
        assert!(
            notices.iter().any(|n| n.contains("NOT used")),
            "el aviso de incoherencia de Anthropic no debe depender de que la \
             plantilla de raíz haya parseado: {notices:?}"
        );
    }
}
