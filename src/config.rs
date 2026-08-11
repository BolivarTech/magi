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

/// Construction boundary (REQ-R23, SC-R21): **every field is private**, so there is no
/// `MagiConfig { provider: Some("banana"), ..Default::default() }` to write. The two ways in are
/// [`Self::load`]/[`Self::from_toml_str`] — the production path — and [`MagiConfigBuilder`],
/// both of which run [`Self::validate_vocabulary`]. The vocabulary is therefore enforced at the
/// TYPE level and not merely asserted after the fact.
///
/// This closes the debt the MS2 §6 gate raised three times and the project owner accepted as a
/// documented residual on 2026-08-09. It was deferred then because it touches dozens of
/// construction sites in `main.rs` and `headless_runner.rs`, and doing that inside a quality gate
/// would have invalidated verdicts already earned for a change with no behavioural effect.
///
/// **`Default` and the `assert!`s stay, and that is not leftover caution.** `Deserialize` is
/// still a way to materialize this struct without passing through either constructor — a plain
/// `toml::from_str::<MagiConfig>` does exactly that — so the preconditions of
/// [`Self::effective_provider`] and [`Self::effective_default_mode`] remain reachable, and they
/// remain `assert!` (every build profile), not `debug_assert!`. Private fields removed the
/// *literal* bypass, which was the wide one; they did not remove the deserialization one.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MagiConfig {
    /// Backend of the main agent (REQ-A01b). Absent ⇒ the built-in default.
    provider: Option<String>,
    /// Default endpoint OF THE SYSTEM (REQ-A21): used by the main agent, the trio, and the
    /// embedder unless their own section overrides it. Absent ⇒ the built-in.
    /// **BREAKING**: up to v0.11.0 this key lived in `[openai].base_url`, which no longer
    /// exists — see [`ConfigError::NeedsMigration`].
    base_url: Option<String>,
    /// Report OUTPUT cap, on ALL THREE paths (TUI, `magi query`, headless consult — REQ-A11b).
    /// Absent ⇒ [`magi_rs::magi::TOOL_RESULT_CAP_BYTES`].
    tool_result_cap_bytes: Option<usize>,
    /// `[openai]` section — see [`OpenAiConfig`].
    #[serde(default)]
    openai: OpenAiConfig,
    /// `[anthropic]` section — see [`AnthropicConfig`].
    #[serde(default)]
    anthropic: AnthropicConfig,
    /// `[magi]` section — see [`MagiSectionConfig`].
    #[serde(default)]
    magi: MagiSectionConfig,
    /// `[memory]` section — see [`crate::memory::config::MemoryConfig`].
    #[serde(default)]
    memory: crate::memory::config::MemoryConfig,
    /// `[embedding]` section — see [`crate::memory::config::EmbeddingConfig`].
    #[serde(default)]
    embedding: crate::memory::config::EmbeddingConfig,
    /// `[headless]` section — see [`HeadlessConfig`].
    #[serde(default)]
    headless: HeadlessConfig,
}

/// Crate-internal builder: the ONLY way to obtain a [`MagiConfig`] other than `Deserialize`
/// (REQ-R23). It mirrors the shape `AutonomousRunConfig` already uses in this crate — private
/// fields, no public literal, a single fallible exit — for the same reason: a type whose invalid
/// state cannot be NAMED needs no assertion that it never occurs.
///
/// **Why `#[cfg(test)]`.** Production builds a configuration exactly one way,
/// [`MagiConfig::load`], so outside the test profile the builder has no caller and the linter is
/// right to say so. Compiling it only under `cfg(test)` states that fact instead of hiding it
/// behind an `#[allow(dead_code)]`, and it costs nothing: SC-R21 is enforced by the fields being
/// private, which holds in every profile. The first production caller removes this attribute
/// together with the code that needs it — the compiler will ask.
///
/// **Which setters exist, and why not one per field.** Only the six fields with a real call site
/// have a setter (`provider`, `base_url`, `tool_result_cap_bytes`, `openai`, `anthropic`,
/// `magi`). `[memory]`, `[embedding]` and `[headless]` are reached exclusively through
/// [`MagiConfig::from_toml_str`], so a setter for them would be dead code — and adding one to
/// round out the API would be fabricating a caller. Whichever task first needs one adds it
/// together with its consumer.
///
/// # Examples
///
/// ```ignore
/// let cfg = MagiConfig::builder()
///     .provider(Some("anthropic".to_string()))
///     .build()?;
/// ```
#[cfg(test)]
#[derive(Debug, Default)]
pub(crate) struct MagiConfigBuilder {
    /// Value under construction. It is never handed out unvalidated: [`MagiConfigBuilder::build`]
    /// is the only move-out, and it validates first.
    inner: MagiConfig,
}

#[cfg(test)]
impl From<MagiConfig> for MagiConfigBuilder {
    /// Reopens an already-built configuration for modification — the replacement for the
    /// functional-update syntax (`..base`) that private fields retire.
    ///
    /// It is not a hole in the invariant: the only exit remains [`MagiConfigBuilder::build`], so
    /// a derived configuration is validated exactly like an original one.
    fn from(inner: MagiConfig) -> Self {
        Self { inner }
    }
}

#[cfg(test)]
impl MagiConfigBuilder {
    /// Sets the root `provider` (REQ-A01b). `None` ⇒ absent, i.e. the built-in default.
    #[must_use]
    pub(crate) fn provider(mut self, v: Option<String>) -> Self {
        self.inner.provider = v;
        self
    }

    /// Sets the root `base_url` template (REQ-A21). `None` ⇒ absent, i.e. the built-in default.
    #[must_use]
    pub(crate) fn base_url(mut self, v: Option<String>) -> Self {
        self.inner.base_url = v;
        self
    }

    /// Sets the root `tool_result_cap_bytes` (REQ-A11b). `None` ⇒ absent, i.e. the built-in cap.
    #[must_use]
    pub(crate) fn tool_result_cap_bytes(mut self, v: Option<usize>) -> Self {
        self.inner.tool_result_cap_bytes = v;
        self
    }

    /// Sets the whole `[openai]` section.
    #[must_use]
    pub(crate) fn openai(mut self, v: OpenAiConfig) -> Self {
        self.inner.openai = v;
        self
    }

    /// Sets the whole `[anthropic]` section.
    #[must_use]
    pub(crate) fn anthropic(mut self, v: AnthropicConfig) -> Self {
        self.inner.anthropic = v;
        self
    }

    /// Sets the whole `[magi]` section.
    #[must_use]
    pub(crate) fn magi(mut self, v: MagiSectionConfig) -> Self {
        self.inner.magi = v;
        self
    }

    /// Validates the accumulated vocabulary and yields the [`MagiConfig`].
    ///
    /// It runs exactly the check [`MagiConfig::from_toml_str`] runs, so a builder-built config
    /// and a file-loaded one are subject to the same rules — a builder that validated less would
    /// let the suite exercise a state production cannot reach.
    ///
    /// # Errors
    ///
    /// Whatever [`MagiConfig::validate_vocabulary`] rejects: [`ConfigError::UnknownProviderKind`],
    /// [`ConfigError::UnknownMode`], [`ConfigError::AgentTimeoutOutOfRange`] or
    /// [`ConfigError::OutputCapTooSmall`].
    pub(crate) fn build(self) -> Result<MagiConfig, ConfigError> {
        self.inner.validate_vocabulary()?;
        Ok(self.inner)
    }

    /// Yields the [`MagiConfig`] **without validating**.
    ///
    /// It exists because a handful of tests must reach the `assert!` preconditions of
    /// [`MagiConfig::effective_provider`] / [`MagiConfig::effective_default_mode`], and those are
    /// only reachable from a config the validation never saw. `Deserialize` is the real remaining
    /// bypass those assertions defend against, so this reproduces that state directly instead of
    /// round-tripping through TOML to fake it. Every other caller wants
    /// [`MagiConfigBuilder::build`].
    #[must_use]
    pub(crate) fn build_unvalidated(self) -> MagiConfig {
        self.inner
    }
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
    /// Opens a [`MagiConfigBuilder`] — the only way to hand-build a config (REQ-R23).
    ///
    /// Production never calls this: it goes through [`Self::load`]. It exists so crate-internal
    /// callers get the same validation `load()` gets instead of a field literal that gets none.
    /// See [`MagiConfigBuilder`] for why it compiles only under `cfg(test)`.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn builder() -> MagiConfigBuilder {
        MagiConfigBuilder::default()
    }

    /// Root `provider` exactly as declared — the RAW value, blanks included. For the resolved
    /// backend, with inheritance and the built-in default applied, use
    /// [`Self::effective_provider`].
    ///
    /// `cfg(test)`: every production path wants the resolved value, not the raw one.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn provider(&self) -> Option<&str> {
        self.provider.as_deref()
    }

    /// `[openai]` section, as declared.
    #[must_use]
    pub(crate) fn openai(&self) -> &OpenAiConfig {
        &self.openai
    }

    /// `[anthropic]` section, as declared.
    #[must_use]
    pub(crate) fn anthropic(&self) -> &AnthropicConfig {
        &self.anthropic
    }

    /// `[magi]` section, as declared. Prefer the `effective_*` accessors where one exists —
    /// they apply the inheritance and blank-is-absent rules this raw view does not.
    #[must_use]
    pub(crate) fn magi(&self) -> &MagiSectionConfig {
        &self.magi
    }

    /// `[memory]` section, as declared.
    #[must_use]
    pub(crate) fn memory(&self) -> &crate::memory::config::MemoryConfig {
        &self.memory
    }

    /// `[embedding]` section, as declared.
    #[must_use]
    pub(crate) fn embedding(&self) -> &crate::memory::config::EmbeddingConfig {
        &self.embedding
    }

    /// `[headless]` section, as declared.
    #[must_use]
    pub(crate) fn headless(&self) -> &HeadlessConfig {
        &self.headless
    }

    /// Root `base_url` exactly as declared — the RAW template, blanks included. For the resolved
    /// one, with the built-in default applied, use [`Self::effective_base_url`].
    ///
    /// `cfg(test)`: every production path wants the resolved template, not the raw string.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn base_url(&self) -> Option<&str> {
        self.base_url.as_deref()
    }

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
    fn validate_agent_timeout(&self) -> Result<(), ConfigError> {
        let Some(secs) = self.magi.agent_timeout_secs else {
            return Ok(()); // absent ⇒ the built-in default, already valid
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

    /// Rejects an output cap below the minimum viable (REQ-A11b).
    ///
    /// # Errors
    ///
    /// [`ConfigError::OutputCapTooSmall`] with the received value and the minimum.
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

    /// Effective provider of the main agent: root key, or the built-in default.
    ///
    /// **Infallible by precondition:** [`Self::validate_vocabulary`] already ran in
    /// [`Self::from_toml_str`]/`load()`, so the only possible `None` is the absent-or-empty
    /// one.
    /// Task 4.1: consumed in production by `resolve_effective_provider_kind` (main agent
    /// backend) and by `build_magi_orchestrator`/`effective_magi_kind` (trio kind, via
    /// inheritance). Covered by `blank_string_keys_are_absent_not_invalid` and
    /// `magi_kind_inherits_from_root_provider_when_absent`.
    ///
    /// I5 (review round 2): restored, because without it an unvalidated config would silently
    /// return `Ollama` — the precondition this function's own doc calls "infallible by
    /// precondition" is exactly what this checks.
    ///
    /// Loop 2 fix (Melchior/Balthasar, S1): **`assert!`, not `debug_assert!`.** The original
    /// version only checked in debug builds, so a release binary had **no check at all** for a
    /// precondition documented as security-relevant (REQ-A01b: an invalid provider must never
    /// silently become a working default). `assert!` is never compiled out regardless of
    /// profile, so the panic this precondition guards is now present in every build, not just
    /// the one `cargo nextest` happens to run.
    ///
    /// **Why `assert!` and not a `Result`.** This function's signature (`-> ProviderKind`,
    /// consumed by every call site as infallible) is load-bearing across the bin — turning it
    /// fallible would ripple a `?`/`.unwrap()` decision through every caller. Reaching this
    /// precondition violation is a programmer bug, not a runtime input, and `assert!` is the
    /// idiomatic response to a violated contract rather than a recoverable error.
    ///
    /// **Why the check survives Task 1.1's private fields (REQ-R23).** Making the struct literal
    /// unconstructible removed the WIDE bypass, not every one: `Deserialize` still materializes a
    /// `MagiConfig` without either validating constructor (`toml::from_str::<MagiConfig>`, and
    /// `MagiConfigBuilder::build_unvalidated` in tests). This assertion is what those remaining
    /// paths run into, and it is still `assert!`, never `debug_assert!`.
    #[must_use]
    pub fn effective_provider(&self) -> ProviderKind {
        assert!(
            self.validate_vocabulary().is_ok(),
            "MagiConfig::effective_provider called on an unvalidated config — \
             construct it via from_toml_str()/load(), never by deserializing it directly"
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
    /// assertion.
    ///
    /// Loop 2 fix (Melchior/Balthasar, S1), and Task 1.1's private fields (REQ-R23): same
    /// `assert!`-not-`debug_assert!` reasoning, and the same reason the check outlives the
    /// literal it used to guard against — see [`Self::effective_provider`]'s rustdoc for the
    /// full explanation.
    #[must_use]
    pub fn effective_default_mode(&self) -> Option<Mode> {
        // Mode declared in `[magi].default_mode`, or `None` if absent/empty (REQ-A15).
        assert!(
            self.validate_vocabulary().is_ok(),
            "MagiConfig::effective_default_mode called on an unvalidated config — \
             construct it via from_toml_str()/load(), never by deserializing it directly"
        );
        <Mode as ModeExt>::parse_config_value(self.magi.default_mode.as_deref().unwrap_or_default())
            .unwrap_or(None)
    }

    /// Trio `kind`: declared, or **inherited** from the main one (REQ-A01b).
    ///
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
    #[must_use]
    pub fn effective_magi_kind(&self) -> ProviderKind {
        ProviderKind::parse(self.magi.kind.as_deref().unwrap_or_default())
            .unwrap_or(None)
            .unwrap_or_else(|| self.effective_provider())
    }

    /// `true` if the trio runs on a different endpoint or kind from the main one.
    ///
    /// **It is decided on what is DECLARED, not by comparing resolved URLs.** Two different
    /// templates can resolve to the same host — one with vault credentials and one without —
    /// and comparing the result would say "they do not diverge" for a configuration that does.
    /// What matters here is the operator's intention.
    /// Task 4.4: consumed in production by `divergence_notice` (`main.rs`, REQ-A07p) — the
    /// `#[allow(dead_code)]` it had was removed here because there is now a real caller.
    /// Covered by `magi_endpoint_diverges_when_the_trio_declares_its_own_kind_or_base_url`.
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

    /// Input cap OF magi-rs, before magi-core's `max_input_len` (REQ-A11b).
    ///
    /// Rationale for the number: **cost, not capacity**. magi-core already skips models where
    /// the prompt does not fit, so this does not protect the model — it bounds the expense, and
    /// the payload is paid for three because it goes to the three mages.
    ///
    /// Consumed by `check_query_size` on ALL THREE entry paths (Task 6.2, REQ-A11b, SC-A11c):
    /// `ConsultTool::execute`, the direct headless path (`headless_runner::analyze_direct`) and
    /// the explicit `/consult` of the TUI.
    #[must_use]
    pub fn effective_max_query_bytes(&self) -> usize {
        self.magi
            .max_query_bytes
            .unwrap_or(magi_rs::magi::MAX_QUERY_BYTES)
    }

    /// Report OUTPUT cap, on ALL THREE paths (REQ-A11b).
    ///
    /// **It lives at the root and not in `[headless]`**: under `[headless]` it would only cover
    /// batch mode and leave interactive mode loose, which is exactly where the report is re-
    /// sent on every turn of a long session. A cap that protects the cheap case and not the
    /// expensive one protects the wrong case. The `allow(dead_code)` this had was removed in
    /// Task 1.3: `resolve_headless_limits` already consumes it. Task 6.2 closes the other two
    /// paths: `register_consult_tool_if_available` (main.rs) passes it to
    /// `ConsultTool::with_output_cap` for the TUI and `magi query` tool loop, and
    /// `TuiMagiRuntimeConfig::tool_result_cap` applies it to the explicit `/consult` of the TUI
    /// via `truncate_report`.
    #[must_use]
    pub fn effective_tool_result_cap(&self) -> usize {
        self.tool_result_cap_bytes
            .unwrap_or(magi_rs::magi::TOOL_RESULT_CAP_BYTES)
    }

    /// System endpoint: declared root, or the built-in default.
    ///
    /// Returns the TEMPLATE, not an already usable `&str`: resolving credentials requires the
    /// vault, and that is the only path to a usable endpoint (REQ-A16c).
    ///
    /// # Errors
    ///
    /// [`EndpointError`] if the declared value is not a valid template (literal credential,
    /// unknown placeholder, or untraversable URL). See
    /// [`magi_rs::magi::endpoint::EndpointTemplate::parse`].
    pub fn effective_base_url(&self) -> Result<EndpointTemplate, EndpointError> {
        EndpointTemplate::parse(
            self.base_url
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .unwrap_or(crate::defaults::DEFAULT_OPENAI_BASE_URL),
            magi_rs::magi::endpoint::Scope::Root,
        )
    }

    /// Resolves a section's override (`[magi].base_url`/`[embedding].base_url`) against the
    /// same effective system endpoint, or inherits if there is no override.
    ///
    /// Shared by [`Self::effective_magi_base_url`] and [`Self::effective_embedding_base_url`]:
    /// both apply **exactly** the same rule ("own, blank-is-absent, else inheritance") and
    /// repeating it in each would be the kind of duplication that B3 forbids — desynchronizing
    /// them would change the rule in one section without anyone noticing in the other.
    ///
    /// # Errors
    ///
    /// See [`Self::effective_base_url`].
    fn override_or_inherit_base_url(
        &self,
        own: Option<&str>,
        scope: magi_rs::magi::endpoint::Scope,
    ) -> Result<EndpointTemplate, EndpointError> {
        match own.map(str::trim).filter(|s| !s.is_empty()) {
            Some(own) => EndpointTemplate::parse(own, scope),
            None => self.effective_base_url(), // herencia, ya validada
        }
    }

    /// Trio endpoint: override of `[magi].base_url`, or inheritance from the system.
    ///
    /// # Errors
    ///
    /// See [`Self::effective_base_url`].
    /// Consumed by [`Self::load`] (Task 1.4) to validate that the trio's template is usable
    /// BEFORE startup finishes — it also closes SC-A16d for `[magi].base_url`, not only for the
    /// root and the embedder. The actual native trio construction on this value remains Phase
    /// 4; here it is only validated.
    pub fn effective_magi_base_url(&self) -> Result<EndpointTemplate, EndpointError> {
        self.override_or_inherit_base_url(
            self.magi.base_url.as_deref(),
            magi_rs::magi::endpoint::Scope::Magi,
        )
    }

    /// Embedder endpoint: override of `[embedding].base_url`, or inheritance from the system
    /// (REQ-A21 — behavior change from v0.11.0, see
    /// [`crate::memory::config::EmbeddingConfig::base_url`]).
    ///
    /// # Errors
    ///
    /// See [`Self::effective_base_url`].
    pub fn effective_embedding_base_url(&self) -> Result<EndpointTemplate, EndpointError> {
        self.override_or_inherit_base_url(
            self.embedding.base_url.as_deref(),
            magi_rs::magi::endpoint::Scope::Embedding,
        )
    }

    /// Loads `magi.toml` from its **path** (not its directory) — Task 1.4 finally consumes
    /// `Workspace::config_path()` (REQ-A22b).
    ///
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
    /// - [`ConfigError::UnknownProviderKind`] / [`ConfigError::UnknownMode`] if `provider`, `[magi].kind` or `[magi].default_mode` bring a present but unrecognized value.
    /// - [`ConfigError::AgentTimeoutOutOfRange`] / [`ConfigError::OutputCapTooSmall`] if those numbers fall outside their range.
    /// - [`ConfigError::Endpoint`] if the root, `[magi]` or `[embedding]` `base_url` carries a literal credential, an unknown placeholder, or could not be traversed (SC-A16d) — before this, ONLY the embedder path (`main.rs::attach_persistent_memory`) saw this error, and degraded it to a notice + plain-text memory instead of stopping startup.
    /// # Arguments
    ///
    /// * `path` - Path to the `magi.toml` file. Recommended absolute/canonical (e.g. `Workspace::config_path()`) so resolution is reproducible.
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

        // REQ-A12c, same shape one level down: `[magi].kind = "anthropic"` with its own
        // declared `[magi].base_url` — that endpoint is not used either.
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

/// Prefixes a [`ConfigError::Parse`] with the path of the offending file; the other variants
/// are already self-contained (they name the field, the value, or the range) and do not need
/// the path to be actionable.
fn attach_path(e: ConfigError, path: &Path) -> ConfigError {
    match e {
        ConfigError::Parse(msg) => ConfigError::Parse(format!("{}: {msg}", path.display())),
        other => other,
    }
}

/// Effective backend of the main agent: env `MAGI_PROVIDER` > TOML `provider` >
/// `DEFAULT_PROVIDER` (RF-1, REQ-A01b).
///
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

/// Treats a blank or whitespace-only string as absent (REQ-A12): a value that is empty or
/// blank is ABSENT, never a literal value, for every text-valued key this module resolves.
/// An env var exported empty in a CI script (`OPENAI_MODEL=""`) is indistinguishable from
/// never having been set, so it must fall through to the next precedence level rather than
/// being forwarded as a literal empty string.
///
/// Shared by every resolver in this module — and by `main.rs`'s `MagiEnvModelOverrides`,
/// the only other reader of a `MAGI_MODEL_*`-shaped env var — that reads a text value from
/// env or TOML, so the predicate lives in exactly one place (B3) rather than being
/// re-implemented per call site — see [`resolve_magi_override`], [`resolve_openai_model`],
/// [`resolve_anthropic_model`]. [`ProviderKind::parse`] applies the same rule for
/// `provider`/`[magi].kind`, but cannot share this helper: it also validates the non-blank
/// remainder against a fixed vocabulary, which is a different, fallible operation this
/// helper does not perform.
///
/// `pub(crate)`, not private: `main.rs` reuses it for `MagiEnvModelOverrides::from_raw`
/// rather than writing a third copy of the same trim-and-filter predicate.
pub(crate) fn non_blank(s: Option<&str>) -> Option<&str> {
    s.map(str::trim).filter(|s| !s.is_empty())
}

/// Resolves a per-agent MAGI model override. Precedence: env (non-empty) > TOML (non-empty) >
/// `None`. A blank/whitespace value (env or TOML) is treated as unset and falls through to the
/// next level. `None` means the agent uses the backend's model (RF-2, S-4, S-5).
///
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
pub fn resolve_magi_override(toml_model: Option<&str>, env_model: Option<&str>) -> Option<String> {
    non_blank(env_model)
        .or_else(|| non_blank(toml_model))
        .map(str::to_string)
}

/// env `OPENAI_MODEL` > TOML `[openai].model` > `DEFAULT_OPENAI_MODEL` (RF-3). No longer
/// fallible: the openai path has a built-in default (Ollama-first).
///
/// **S1 gate re-review fix (Balthasar):** both `env_model` and `config.openai.model` go
/// through [`non_blank`] — a blank value at either level falls through to the next one
/// instead of being forwarded as a literal empty model name (REQ-A12), matching
/// [`resolve_magi_override`]'s already-correct handling of the sibling per-agent overrides.
///
/// # Arguments
///
/// * `config` - Parsed `MagiConfig`.
/// * `env_model` - Value of `OPENAI_MODEL` env var, if set.
/// # Returns
///
/// Resolved model name; env overrides TOML, both override the built-in default.
pub fn resolve_openai_model(config: &MagiConfig, env_model: Option<&str>) -> String {
    non_blank(env_model)
        .or_else(|| non_blank(config.openai.model.as_deref()))
        .map(str::to_string)
        .unwrap_or_else(|| crate::defaults::DEFAULT_OPENAI_MODEL.into())
}

/// env `ANTHROPIC_MODEL` > TOML `[anthropic].model` > `DEFAULT_ANTHROPIC_MODEL`.
///
/// Mirrors [`resolve_openai_model`]'s precedence exactly, including the blank-is-absent
/// handling on both levels (REQ-A12, S1 gate re-review fix). Fixes a MAGI re-gate WARNING:
/// prior call sites in `main.rs` disagreed on precedence — the headless path checked TOML
/// before env (backwards), and the TUI/other path (`discover_config`) read only env and
/// ignored `[anthropic].model` entirely. Both now route through this single resolver.
///
/// # Arguments
///
/// * `config` - Parsed `MagiConfig`.
/// * `env_model` - Value of `ANTHROPIC_MODEL` env var, if set.
/// # Returns
///
/// Resolved model name; env overrides TOML, both override the built-in default.
pub fn resolve_anthropic_model(config: &MagiConfig, env_model: Option<&str>) -> String {
    non_blank(env_model)
        .or_else(|| non_blank(config.anthropic.model.as_deref()))
        .map(str::to_string)
        .unwrap_or_else(|| crate::defaults::DEFAULT_ANTHROPIC_MODEL.into())
}

/// Builds the complexity gate thresholds from `[magi.complexity]` (REQ-A20b).
///
/// **It lives here, and not in `magi_rs::magi::gate` — moved from Task 1.1 (see
/// `.superpowers/sdd/claude-plan-tdd/ORDER-FIXES.md`, #1).** `gate.rs` lives in the lib and
/// cannot know the shape of the TOML; breaking `[magi.complexity]` into loose pieces
/// (`GateOverrides`) is this module's job, since it already has the table in hand.
/// Absent table ⇒ `GateOverrides::default()` ⇒ the three built-ins from
/// [`GateThresholds::builtin`] (the gate is not turned off by omitting the section).
///
/// Consumed in production by `AutonomousRunConfig::from_magi_config` (`main.rs`), which hands
/// the result to both autonomous surfaces. Covered by
/// `gate_thresholds_from_reads_the_complexity_table_and_falls_back_to_builtins`.
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
        assert_eq!(t.code_review, 50, "declared: the file's value is used");
        assert_eq!(
            t.design,
            GateThresholds::builtin().design,
            "absent WITHIN the table: its built-in, not zero"
        );
        assert_eq!(
            t.analysis, 0,
            "a declared 0 is preserved: it is the way to turn off THAT mode"
        );

        let without_table = MagiConfig::default();
        assert_eq!(
            gate_thresholds_from(&without_table),
            GateThresholds::builtin(),
            "table absent ⇒ built-ins: the gate does not turn off by omitting the section"
        );
    }

    /// MAGI S2 re-gate (Balthasar): the scaffolded example's `[magi.complexity]` table drifted
    /// from the built-ins (`analysis = 120` in the file vs. `GATE_ANALYSIS = 200` in code) with
    /// nothing to catch it. A one-off value fix does not prevent it from drifting again — this
    /// parses `docs/magi.toml.example` for real and asserts its thresholds resolve to exactly
    /// [`GateThresholds::builtin`], so any future edit to either side that breaks the match
    /// fails this test instead of silently shipping a misleading example.
    #[test]
    fn example_toml_complexity_table_matches_the_builtin_gate_thresholds() {
        let raw = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/docs/magi.toml.example"
        ))
        .expect("docs/magi.toml.example must be readable");
        let parsed = MagiConfig::from_toml_str(&raw).expect(
            "docs/magi.toml.example must parse as valid v0.12.0 TOML (commented lines inert)",
        );
        assert_eq!(
            gate_thresholds_from(&parsed),
            GateThresholds::builtin(),
            "docs/magi.toml.example's [magi.complexity] table must mirror the built-in \
             thresholds exactly — it is the first thing an operator copies, and a mismatch \
             there is a lie about what the gate actually does out of the box"
        );
    }

    /// MAGI S10 gate finding (third pass): the scaffolded example's `balthasar_model` and
    /// `caspar_model` had drifted to older tag names (`kimi-k2.6:cloud`, `glm-5.2:cloud`)
    /// that no longer matched `src/defaults.rs`'s `DEFAULT_MAGI_BALTHASAR`/`DEFAULT_MAGI_CASPAR`
    /// — the same class of defect `example_toml_complexity_table_matches_the_builtin_gate_thresholds`
    /// exists to catch, just for the `[magi]` per-mage models instead of the complexity table.
    /// Same fix shape: pin the example's parsed values against the single source of truth so a
    /// future edit to either side that breaks the match fails this test instead of shipping a
    /// stale example silently.
    #[test]
    fn example_toml_magi_models_match_the_builtin_defaults() {
        let raw = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/docs/magi.toml.example"
        ))
        .expect("docs/magi.toml.example must be readable");
        let parsed = MagiConfig::from_toml_str(&raw).expect(
            "docs/magi.toml.example must parse as valid v0.12.0 TOML (commented lines inert)",
        );
        assert_eq!(
            parsed.magi.melchior_model.as_deref(),
            Some(crate::defaults::DEFAULT_MAGI_MELCHIOR),
            "docs/magi.toml.example's melchior_model must mirror DEFAULT_MAGI_MELCHIOR"
        );
        assert_eq!(
            parsed.magi.balthasar_model.as_deref(),
            Some(crate::defaults::DEFAULT_MAGI_BALTHASAR),
            "docs/magi.toml.example's balthasar_model must mirror DEFAULT_MAGI_BALTHASAR"
        );
        assert_eq!(
            parsed.magi.caspar_model.as_deref(),
            Some(crate::defaults::DEFAULT_MAGI_CASPAR),
            "docs/magi.toml.example's caspar_model must mirror DEFAULT_MAGI_CASPAR"
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

    /// SC-A21b: an absent `magi.toml` stays a silent default — no error, no degradation.
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
        let c = MagiConfig::builder()
            .provider(Some("anthropic".into()))
            .build()
            .unwrap();
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

    /// SC-A12g / REQ-A12: a PRESENT but unrecognized `MAGI_PROVIDER` value is a configuration
    /// error, never a silent fallback — same rule as an invalid `provider`/`[magi].kind` in the
    /// TOML.
    #[test]
    fn an_unrecognized_env_provider_is_a_configuration_error() {
        let err =
            resolve_effective_provider_kind(&MagiConfig::default(), Some("banana")).unwrap_err();
        assert!(err.to_string().contains("banana"));
    }

    /// SC-A12g / REQ-A12: a blank `MAGI_PROVIDER` is treated as ABSENT, not invalid — falls
    /// through to the TOML/default the same as an unset env var.
    #[test]
    fn a_blank_env_provider_falls_through_to_the_toml_default() {
        let c = MagiConfig::builder()
            .provider(Some("anthropic".into()))
            .build()
            .unwrap();
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
        let c = MagiConfig::builder()
            .openai(OpenAiConfig {
                model: Some("phi4-mini".into()),
            })
            .build()
            .unwrap();
        assert_eq!(resolve_openai_model(&c, None), "phi4-mini");
        assert_eq!(resolve_openai_model(&c, Some("gpt-4o-mini")), "gpt-4o-mini");
    }

    #[test]
    fn test_resolve_anthropic_model_env_wins_over_toml() {
        // Mirrors test_resolve_openai_model_env_wins_over_toml just above: env must win over
        // TOML for the Anthropic model too, not the other way around.
        let c = MagiConfig::builder()
            .anthropic(AnthropicConfig {
                model: Some("claude-toml-model".into()),
            })
            .build()
            .unwrap();
        assert_eq!(
            resolve_anthropic_model(&c, Some("claude-env-model")),
            "claude-env-model"
        );
    }

    #[test]
    fn test_resolve_anthropic_model_toml_when_no_env() {
        let c = MagiConfig::builder()
            .anthropic(AnthropicConfig {
                model: Some("claude-toml-model".into()),
            })
            .build()
            .unwrap();
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

    // S1 gate re-review finding (Balthasar): `resolve_openai_model`/`resolve_anthropic_model`
    // must treat a blank/whitespace-only env value as absent (REQ-A12), the same rule
    // `resolve_effective_provider_kind`/`resolve_magi_override` already apply — an
    // `OPENAI_MODEL=""` exported empty in a CI script must fall through to TOML, then the
    // built-in default, not be forwarded as a literal empty model name.

    #[test]
    fn test_resolve_openai_model_blank_env_falls_through_to_toml() {
        let c = MagiConfig::builder()
            .openai(OpenAiConfig {
                model: Some("phi4-mini".into()),
            })
            .build()
            .unwrap();
        assert_eq!(resolve_openai_model(&c, Some("")), "phi4-mini");
        assert_eq!(resolve_openai_model(&c, Some("   ")), "phi4-mini");
    }

    #[test]
    fn test_resolve_openai_model_blank_env_falls_through_to_default_when_no_toml() {
        use crate::defaults::DEFAULT_OPENAI_MODEL;
        assert_eq!(
            resolve_openai_model(&MagiConfig::default(), Some("")),
            DEFAULT_OPENAI_MODEL
        );
    }

    #[test]
    fn test_resolve_anthropic_model_blank_env_falls_through_to_toml() {
        let c = MagiConfig::builder()
            .anthropic(AnthropicConfig {
                model: Some("claude-toml-model".into()),
            })
            .build()
            .unwrap();
        assert_eq!(resolve_anthropic_model(&c, Some("")), "claude-toml-model");
        assert_eq!(
            resolve_anthropic_model(&c, Some("   ")),
            "claude-toml-model"
        );
    }

    #[test]
    fn test_resolve_anthropic_model_blank_env_falls_through_to_default_when_no_toml() {
        use crate::defaults::DEFAULT_ANTHROPIC_MODEL;
        assert_eq!(
            resolve_anthropic_model(&MagiConfig::default(), Some("")),
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

    /// SC-A12b: `min_agents` is not configurable. `ConsensusConfig` is deliberately not
    /// exposed as a `magi.toml` key (REQ-A15) — accepting `min_agents` would let an operator
    /// declare a two-mage consensus as valid, which changes the GATE's semantics (a degraded
    /// run becomes approvable), not its performance. `deny_unknown_fields` on
    /// `MagiSectionConfig` is what enforces the hard floor; this test proves that "would
    /// reject" is actually "does reject" rather than an assumption nobody checks.
    #[test]
    fn min_agents_and_any_consensus_config_field_are_rejected_as_unknown() {
        assert!(
            MagiConfig::from_toml_str("[magi]\nmin_agents = 2").is_err(),
            "min_agents must never be accepted: it would let a two-mage consensus \
             pass as valid, turning a degraded run into an approvable one"
        );
        // `epsilon` is `ConsensusConfig`'s other field — same struct, same reasoning.
        assert!(
            MagiConfig::from_toml_str("[magi]\nepsilon = 0.001").is_err(),
            "no ConsensusConfig-shaped field is configurable, not just min_agents"
        );
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
            "the old value is no longer valid"
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
    #[test]
    fn an_invalid_vocabulary_value_is_rejected_at_parse_not_swallowed_by_a_resolver() {
        for (toml, what) in [
            ("provider = \"banana\"\n", "root provider"),
            ("[magi]\nkind = \"banana\"\n", "[magi].kind"),
            ("[magi]\ndefault_mode = \"banana\"\n", "[magi].default_mode"),
        ] {
            assert!(
                MagiConfig::from_toml_str(toml).is_err(),
                "{what}: an unrecognized value must be an ERROR, never a silent fallback"
            );
        }
    }

    /// SC-A12g / REQ-A12: general rule — empty or blank is ABSENT, never invalid.
    ///
    /// `effective_base_url()` returns `Result<EndpointTemplate, _>` since REQ-A16c, so the test
    /// compares the TEXT of the template.
    #[test]
    fn blank_string_keys_are_absent_not_invalid() {
        assert_eq!(ProviderKind::parse("").unwrap(), None);
        assert_eq!(ProviderKind::parse("   ").unwrap(), None);
        let toml = "provider = \"\"\nbase_url = \"  \"\n";
        let cfg = MagiConfig::from_toml_str(toml).expect("empty must not break parsing");
        assert_eq!(
            cfg.effective_provider(),
            ProviderKind::Ollama,
            "falls back to the built-in default"
        );
        // SC-A02b: `[magi].kind` INHERITS from the root when not declared.
        assert_eq!(
            cfg.effective_base_url().unwrap().as_str(),
            crate::defaults::DEFAULT_OPENAI_BASE_URL
        );
    }

    /// SC-A02b: `kind` inherits from the root when not declared, and does not change the
    /// principal's own provider when it is.
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
            "the principal does not change"
        );
    }

    /// SC-A21c: endpoint inheritance and override — `base_url` at the root is inherited by
    /// every section unless that section declares its own.
    ///
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
    #[test]
    fn an_api_key_anywhere_in_the_toml_is_a_parse_error() {
        for toml in [
            "api_key = \"sk-secreto\"\n",
            "[openai]\napi_key = \"sk-secreto\"\n",
            "[anthropic]\napi_key = \"sk-secreto\"\n",
            "[magi]\napi_key = \"sk-secreto\"\n",
        ] {
            let err = MagiConfig::from_toml_str(toml)
                .expect_err("an api_key in the TOML must be an ERROR, not silent acceptance");
            assert!(
                !err.to_string().contains("sk-secreto"),
                "and the error must NOT repeat the secret it is rejecting"
            );
        }
    }

    /// REQ-A15: `default_mode` resolves with the same empty=absent rule.
    ///
    /// **Returns `Option<Mode>`, NOT `Result`**: validation lives in `validate_vocabulary`,
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
        let declared = MagiConfig::builder()
            .tool_result_cap_bytes(Some(above_min))
            .build()
            .unwrap();
        assert_eq!(declared.effective_tool_result_cap(), above_min);
    }

    // Fix round 2 (coordinator review, 2026-08-02): I3/I4/I5/m8 — B13 coverage this task's own
    // new functions shipped without.
    // -------------------------------------------------------------------------
    // I3: both range boundaries of `agent_timeout_secs` (§4.9) are accepted; one step outside
    // either end is rejected. `validate_agent_timeout` shipped with zero tests and an
    // inclusive-both-ends range with nothing pinning the edge.

    /// I3: both range boundaries of `agent_timeout_secs` (§4.9) are accepted; one step outside
    /// either end is rejected. `validate_agent_timeout` shipped with zero tests and an
    /// inclusive-both-ends range with nothing pinning the edge.
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

    /// I3: the output-cap floor (`min_viable_output_cap()`) itself is accepted; one byte below
    /// it is rejected. Same "zero tests on a boundary" gap as `validate_agent_timeout`.
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

    /// I4: `safe_parse_error` keeps line/column (SC-A21g requires a syntax error to name a
    /// position) but never the offending value — only the source EXCERPT needed suppressing to
    /// fix the `api_key` leak (see `safe_parse_error`'s own doc), not the position.
    ///
    /// Leading blank line: line 2 (not 1) pins that this is a real computed position, not a
    /// hardcoded "line 1, column 1".
    #[test]
    fn safe_parse_error_keeps_the_position_but_drops_the_offending_value() {
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

    /// SC-A16g: a broken TOML carrying a `[user]:[password]`-style placeholder does not leak,
    /// because `safe_parse_error` (proven above to drop the offending value on ANY input) and
    /// `detect_migrations` (proven in `migrate.rs`'s
    /// `a_syntactically_broken_toml_gets_a_syntax_error_not_migration_advice` to require
    /// structural validity, not a textual match) hold TOGETHER on the real
    /// `from_toml_str` path — this is the combination the scenario actually describes, not
    /// either property in isolation. What sits on the offending line is `[password]`, never a
    /// secret, so a `line`-citing error is safe by construction.
    #[test]
    fn a_broken_toml_with_a_placeholder_still_only_cites_a_safe_position() {
        // Unterminated string on the base_url line: syntactically broken. It also textually
        // resembles the v0.11.0 `[openai].base_url` migration pattern PLUS carries the
        // placeholder — both would be red herrings if `detect_migrations` matched by grep
        // instead of requiring structural validity.
        let toml =
            "provider = \"openai\"\n[openai]\nbase_url = \"https://[user]:[password]@host/v1\n";
        let err = MagiConfig::from_toml_str(toml).unwrap_err();
        assert!(
            matches!(err, ConfigError::Parse(_)),
            "a syntactically broken file must get a syntax error, not migration advice: {err}"
        );
        let msg = err.to_string();
        assert!(
            !msg.contains("host"),
            "safe_parse_error must never echo the source excerpt, placeholder or not: {msg}"
        );
        assert!(
            msg.contains("line") && msg.contains("column"),
            "the position is still safe to cite — nothing sensitive sits there: {msg}"
        );
    }

    /// I5: `effective_provider` is documented "infallible by precondition" — that precondition
    /// is `validate_vocabulary` having already run. Task 1.1's private fields (REQ-R23) closed
    /// the struct-literal way of skipping it, but not `Deserialize` — reproduced here by
    /// `build_unvalidated`, the builder's test-only exit — so the precondition is still
    /// violable and still worth asserting.
    ///
    /// Loop 2 fix (Melchior/Balthasar, S1): the guard is now `assert!`, which panics in EVERY
    /// build profile, not only under `debug_assertions` — a release binary hitting this misuse
    /// used to silently fall back to `Ollama` with zero signal; now it panics the same way a
    /// debug build always did. `cargo nextest` always builds with `debug_assertions` on, so this
    /// test cannot itself distinguish `assert!` from the old `debug_assert!` at runtime — the
    /// property that changed (never compiled out) is a static one, verified by reading the
    /// macro used, not by a profile-dependent test run.
    #[test]
    #[should_panic(expected = "unvalidated config")]
    fn effective_provider_panics_when_validate_vocabulary_was_skipped() {
        let cfg = MagiConfig::builder()
            .provider(Some("banana".into()))
            .build_unvalidated();
        let _ = cfg.effective_provider();
    }

    /// I5: same precondition, same gap, for `effective_default_mode`. See
    /// `effective_provider_panics_when_validate_vocabulary_was_skipped` for the Loop 2 fix note.
    #[test]
    #[should_panic(expected = "unvalidated config")]
    fn effective_default_mode_panics_when_validate_vocabulary_was_skipped() {
        let cfg = MagiConfig::builder()
            .magi(MagiSectionConfig {
                default_mode: Some("banana".into()),
                ..MagiSectionConfig::default()
            })
            .build_unvalidated();
        let _ = cfg.effective_default_mode();
    }

    /// Sixth-pass gate finding (S1, Balthasar) — **rejected**: `effective_magi_kind` needs no
    /// `assert!` of its own, and this test is what pins that instead of leaving it as a claim.
    ///
    /// `effective_magi_kind` has two branches. When `[magi].kind` parses successfully, the
    /// returned value is a fresh, direct `ProviderKind::parse` of the CURRENT field — nothing
    /// is swallowed into a default, so there is nothing for an assert to catch. When it does
    /// NOT parse (absent, blank, or — as constructed here, bypassing `load()` — invalid), the
    /// function falls through to `self.effective_provider()`, which ALREADY carries the same
    /// `validate_vocabulary().is_ok()` precondition check that
    /// `effective_provider_panics_when_validate_vocabulary_was_skipped` pins — and
    /// `validate_vocabulary` checks `magi.kind` too (see its body), so the delegated call
    /// re-validates the very field this test corrupts. Adding a second `assert!` directly in
    /// `effective_magi_kind` would duplicate a check the call it already makes performs on the
    /// SAME field; this test demonstrates the existing delegation already catches the misuse.
    #[test]
    #[should_panic(expected = "unvalidated config")]
    fn effective_magi_kind_panics_via_the_delegated_effective_provider_check_when_invalid() {
        let cfg = MagiConfig::builder()
            .magi(MagiSectionConfig {
                kind: Some("banana".into()),
                ..MagiSectionConfig::default()
            })
            .build_unvalidated();
        let _ = cfg.effective_magi_kind();
    }

    /// m8: `[magi].base_url = ""` is blank, not a declared override — it must inherit the root,
    /// not be treated as "the trio declared its own endpoint".
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

    /// m8: a whitespace-only `default_mode` is blank, not a value — same blank-is-absent rule
    /// as every other vocabulary key.
    #[test]
    fn default_mode_whitespace_only_is_treated_as_absent() {
        let cfg = MagiConfig::from_toml_str("[magi]\ndefault_mode = \"   \"\n").unwrap();
        assert_eq!(cfg.effective_default_mode(), None);
    }

    /// m8: `[embedding].base_url`'s OVERRIDE winning over the root was untested — the existing
    /// coverage only proved inheritance, never that a declared embedding-specific endpoint
    /// takes precedence over it.
    ///
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
            ("provider = \"banana\"\n", "root provider"),
            ("[magi]\nkind = \"banana\"\n", "[magi].kind"),
            ("[magi]\ndefault_mode = \"banana\"\n", "[magi].default_mode"),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("magi.toml");
            std::fs::write(&path, toml).unwrap();
            assert!(
                MagiConfig::load(&path).is_err(),
                "{what}: through load() must also be ERROR, never defaults + warning"
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
        let err = MagiConfig::load(&path).expect_err("present and broken: FATAL");
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
        for (toml, scope, expected_entry) in [
            (
                "base_url = \"https://alice:s3cr3t@host/v1\"\n",
                "root",
                "BASE_URL_USER",
            ),
            (
                "[magi]\nbase_url = \"https://alice:s3cr3t@host/v1\"\n",
                "[magi]",
                "MAGI_BASE_URL_USER",
            ),
            (
                "[embedding]\nbase_url = \"https://alice:s3cr3t@host/v1\"\n",
                "[embedding]",
                "EMBEDDING_BASE_URL_USER",
            ),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("magi.toml");
            std::fs::write(&path, toml).unwrap();
            let err = match MagiConfig::load(&path) {
                Err(e) => e,
                Ok(_) => {
                    panic!(
                        "{scope}: load() should have failed given a literal credential in base_url"
                    )
                }
            };
            let msg = err.to_string();
            // And "it does not leak" is not enough: the message has to be ACTIONABLE — name the
            // correct placeholder and the vault command, not just say "invalid credential".
            assert!(
                !msg.contains("s3cr3t"),
                "{scope}: leaked the password: {msg}"
            );
            assert!(
                !msg.contains("alice"),
                "{scope}: leaked the username: {msg}"
            );
            // REQ-A12b: silent resolutions are announced.
            assert!(
                msg.contains("[user]") && msg.contains("[password]"),
                "{scope}: does not name the placeholder: {msg}"
            );
            assert!(
                msg.contains("magi-rs vault set"),
                "{scope}: does not name the command to fix it: {msg}"
            );
            // Loop 1 fix round CE, F22: not just AN entry name — the RIGHT one for this scope.
            // Before the fix every scope named the root's `BASE_URL_USER`/`BASE_URL_PASSWORD`,
            // which this assertion did not catch because it only checked for the placeholder
            // syntax, never the entry name itself.
            assert!(
                msg.contains(expected_entry),
                "{scope}: expected entry {expected_entry}, got: {msg}"
            );
        }
    }

    /// SC-A12c: what resolved silently gets said — an empty `provider` names the default used,
    /// and a non-default root `base_url` says where the embedder's endpoint came from.
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
            "the embedder inherits a NON-default base_url: it must be said"
        );

        std::fs::write(&path, "base_url = \"http://localhost:11434/v1\"\n").unwrap();
        let (_, notices) = MagiConfig::load(&path).unwrap();
        assert!(
            !notices.iter().any(|n| n.contains("embedder")),
            "inheriting the DEFAULT is not surprising: it would be noise on every startup"
        );
    }

    /// SC-A12d: an incoherent `provider`/`base_url` combination is flagged AT LOAD, not
    /// discovered later as a network error against `localhost`. Covers REQ-A12c's two
    /// sub-cases — (a) the user DECLARED a `base_url` and it is silently unused, and (b) the
    /// untouched Ollama default sitting there, which reads like a migration oversight — and the
    /// same shape one level down for `[magi].kind = "anthropic"`. (b) is the case an earlier
    /// `is_some()` guard did NOT cover.
    #[test]
    fn anthropic_flags_both_the_declared_and_the_defaulted_base_url() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("magi.toml");

        // (a) declared explicitly — the user thinks it is used.
        std::fs::write(&path, "provider = \"anthropic\"\n").unwrap();
        let (_, notices) = MagiConfig::load(&path).unwrap();
        assert!(
            notices.iter().any(|n| n.contains("default Ollama")),
            "with no base_url declared the default is still there, and it looks like a \
             migration oversight"
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
            "the Anthropic incoherence notice must not depend on the root template \
             having parsed: {notices:?}"
        );
    }

    // Task 1.1 (REQ-R23 / SC-R21): the vocabulary validation must be impossible to bypass by
    // building a `MagiConfig` as a field literal.
    // -------------------------------------------------------------------------

    /// SC-R21: a `MagiConfig` cannot be built bypassing validation.
    ///
    /// The literal path is gone, so the ONLY way in is the builder, which validates. This is the
    /// runtime half of the assertion: the builder REJECTS what the literal used to accept
    /// silently. The compile-time half is
    /// `no_magi_config_field_is_public_so_no_literal_can_skip_validation`.
    #[test]
    fn a_magi_config_with_an_unknown_provider_cannot_be_built() {
        let err = MagiConfig::builder()
            .provider(Some("banana".to_string()))
            .build()
            .expect_err("an unrecognized provider must not build");
        assert!(
            matches!(err, ConfigError::UnknownProviderKind { .. }),
            "expected UnknownProviderKind, got {err:?}"
        );
    }

    /// The builder validates the WHOLE vocabulary, not only the root `provider`: an unknown
    /// `[magi].default_mode` is rejected on the same call, so no second, unguarded axis is left
    /// reachable through it.
    #[test]
    fn a_magi_config_with_an_unknown_default_mode_cannot_be_built() {
        let err = MagiConfig::builder()
            .magi(MagiSectionConfig {
                default_mode: Some("banana".to_string()),
                ..MagiSectionConfig::default()
            })
            .build()
            .expect_err("an unrecognized default_mode must not build");
        assert!(
            matches!(err, ConfigError::UnknownMode { .. }),
            "expected UnknownMode, got {err:?}"
        );
    }

    /// The valid vocabulary still builds, so the guard is not simply rejecting everything.
    #[test]
    fn the_three_vocabulary_values_build() {
        for p in ["ollama", "openai-compat", "anthropic"] {
            MagiConfig::builder()
                .provider(Some(p.to_string()))
                .build()
                .unwrap_or_else(|e| panic!("{p} must build, got {e:?}"));
        }
    }

    /// Exact source line that opens the `MagiConfig` declaration, once trimmed.
    ///
    /// It is matched on the FULL trimmed line rather than by `contains`, so the `const` below
    /// (whose own source line embeds this same text) cannot match itself.
    const MAGI_CONFIG_STRUCT_HEADER: &str = "pub struct MagiConfig {";

    /// Line that closes a struct block written in the crate's `rustfmt` style: a lone brace at
    /// the item's own indentation, which for a top-level item is column zero.
    const STRUCT_BLOCK_END: &str = "}";

    /// Prefix of a field declared public. Field attributes (`#[serde(default)]`) sit on their
    /// own line in this crate's `rustfmt` style, so a field's visibility always opens its line.
    const PUBLIC_FIELD_PREFIX: &str = "pub ";

    /// SC-R21: **no `MagiConfig` field is public**, so `MagiConfig { provider: Some("banana"),
    /// ..Default::default() }` does not compile and the builder is the only way in.
    ///
    /// **Why this is a source-level assertion and not `trybuild`.** SC-R21 is a property of the
    /// TYPE, not of any value, so no ordinary runtime test can observe it — the plan is right
    /// about that. Its proposed fixture, however, cannot work here: `trybuild` compiles each
    /// case against the **library** target (`magi_rs`), and `config` is a **binary-only** module
    /// (`mod config;` in `main.rs`; `lib.rs` exports only `headless`, `magi`, `notices`,
    /// `redact` and `vault`). A case naming `magi_rs::config::MagiConfig` therefore fails to
    /// compile because the PATH does not resolve — and `trybuild` reports a case that fails to
    /// compile as a **pass**. Restoring `pub` on the fields would not change that outcome, which
    /// is precisely the mutation the guardian has to detect, so the `trybuild` version would be
    /// a guardian that guards nothing. The plan's fallback, a `compile_fail` doctest, fails for
    /// a second, independent reason: Cargo does not collect doctests from binary targets, and
    /// `cargo nextest` — this project's runner — does not run doctests at all, so it would never
    /// execute.
    ///
    /// This assertion has neither problem: it reads the declaration it guards and fails the
    /// moment a field becomes public again, which is the mutation B16 asks about.
    #[test]
    fn no_magi_config_field_is_public_so_no_literal_can_skip_validation() {
        let source = include_str!("config.rs");
        let mut after_header = source
            .lines()
            .skip_while(|line| line.trim() != MAGI_CONFIG_STRUCT_HEADER)
            .skip(1)
            .peekable();
        assert!(
            after_header.peek().is_some(),
            "the `{MAGI_CONFIG_STRUCT_HEADER}` declaration was not found in this file — \
             the guardian lost its subject and must be repointed, not deleted"
        );

        let public_fields: Vec<&str> = after_header
            .take_while(|line| line.trim() != STRUCT_BLOCK_END)
            .filter(|line| line.trim().starts_with(PUBLIC_FIELD_PREFIX))
            .collect();

        assert!(
            public_fields.is_empty(),
            "MagiConfig must have no public field, or a field literal can build a \
             configuration the vocabulary validation never saw; found: {public_fields:?}"
        );
    }
}
