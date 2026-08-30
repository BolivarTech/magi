// Author: Julian Bolivar
// Version: 0.17.0
// Date: 2026-08-27

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
use magi_rs::magi::lineage::{Lineage, LineageError};
use magi_rs::magi::mode::{ModeExt, ModeParseError};
use magi_rs::magi::{min_viable_output_cap, AGENT_TIMEOUT_MAX_SECS, AGENT_TIMEOUT_MIN_SECS};
use magi_rs::notices::Notice;
use serde::Deserialize;

/// Configuration errors from `magi.toml` (Task 1.1, REQ-A01b/A04/A11b/A21b).
///
/// Lives in the **bin** (not in `magi_rs::magi`) because it is specific to the SHAPE of the
/// TOML; the pure vocabulary error types (`ProviderKindParseError`, `ModeParseError`) live in
/// the lib and are absorbed here with `From`, which is the correct dependency direction (the
/// lib cannot know a type from the bin).
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// The declared lineages do not satisfy `enforce_diversity` (REQ-R29).
    ///
    /// A load error and not a notice because the evidence is **declarative**: three distinct seat
    /// lineages and pool coverage are free to check and true right now. The empirical half —
    /// two models sharing a weights digest — only ever warns, because that evidence can have
    /// aged.
    #[error("{0}")]
    Diversity(#[from] magi_rs::magi::rotation_config::DiversityError),

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

    /// The file brings incompatibilities from the previous generation (REQ-A21b/REQ-R22). The
    /// text is already rendered by [`migrate::render_migration_error`] — it names each
    /// incompatibility and its correction, all of them in ONE message.
    ///
    /// Reached today by the v0.12.0 → v0.13.0 break: a seat that declares a model and not its
    /// lineage ([`migrate::missing_seat_lineages`]). The pre-parse half of the pass
    /// ([`migrate::detect_migrations`]) declares no pattern since the v0.11.0 set retired, and
    /// feeds this same variant when it is reloaded.
    #[error("{0}")]
    NeedsMigration(String),

    /// The TOML does not parse, or the file could not be read.
    #[error("{0}")]
    Parse(String),

    /// `[magi].agent_timeout_secs` falls outside the acceptable range of §4.9.
    #[error(
        "agent_timeout_secs = {got} out of range [{min}, {max}]: below {min}s a legitimate \
         generation does not fit; above {max}s a consult's worst case — 2 attempts per mage on \
         each of `1 + max_rotations` models — reaches 12 minutes at the default two rotations, \
         and grows from there. Not clamped to the extreme — rejected."
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
    /// The text never repeats the offending value, and **this variant no longer takes that on
    /// trust** (S1 Loop 2, Balthasar). It used to render `{0}` directly, with a comment saying
    /// the guarantee lived in [`EndpointError`]'s `Display` "and not in this `#[error]`" — which
    /// is precisely the shape CLAUDE.md names as recurrent: a promise held by a type this variant
    /// does not own, enforced by nothing, and revocable by an edit in another module that would
    /// pass every gate. `EndpointError`'s own guarantee stays; this is the second layer, so a
    /// future message that embeds a URL is redacted here instead of printed.
    #[error("{}", magi_rs::redact::redact_foreign_error(_0))]
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
///
/// # Rule for any NEW production path
///
/// Obtain a `MagiConfig` from [`Self::load`] or [`Self::from_toml_str`]. Never from a bare
/// `toml::from_str::<MagiConfig>`, and never from [`MagiConfigBuilder::build_unvalidated`] —
/// both skip [`Self::validate_vocabulary`], and the `assert!`s above are a backstop that turns
/// the mistake into a panic, not a design that makes it safe. [`MagiConfigBuilder`] itself is
/// `#[cfg(test)]` today precisely because no production path needs it; if one ever does, the
/// builder is promoted and its `build` (the validating exit) is the one to use.
///
/// This is written here rather than only in the project runbook because the runbook is not
/// tracked in git, so it reaches neither a reviewer reading this file nor a developer who
/// clones the repository (asked for by S1 Loop 2, Balthasar).
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
    /// `[logging]` section — see [`LoggingSection`].
    #[serde(default)]
    logging: LoggingSection,
}

/// The `[logging]` section of `magi.toml`.
///
/// # What is deliberately NOT here
///
/// `format` and `rotation_tz`. Their absence is not an oversight: they arrive
/// with their functionality, because shipping an inert key promises something
/// the binary does not do.
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoggingSection {
    /// Where the daily files live. Absent ⇒ `.magi/logs`.
    log_dir: Option<String>,
    /// Level or per-target directive for the file branch.
    file_filter: Option<String>,
    /// Level or per-target directive for the screen branch. **MS2.**
    tui_filter: Option<String>,
    /// Whether rotated files are compressed.
    compress: Option<bool>,
    /// Age in days past which a file is compressed.
    compress_after_days: Option<i64>,
    /// Age in days past which a file is deleted.
    retain_days: Option<i64>,
    /// Ceiling on the total bytes retention may leave on disk.
    max_total_bytes: Option<u64>,
}

// There is no `DEFAULT_LOG_DIR` here on purpose: `Workspace::logs_dir()` already
// owns that path, and a second constant naming the same directory is two places
// to change it and one chance to change only one.
/// Where the log goes, and whether anyone asked for it.
///
/// The provenance travels WITH the path rather than being re-derived, because
/// D-L20 turns on it: a declared directory that cannot be created is a startup
/// error, a defaulted one degrades to a notice. Two functions reading the same
/// precedence chain to answer "which was it?" is how the two answers drift.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedLogDir {
    /// The directory to use.
    pub path: std::path::PathBuf,
    /// True when a flag, an environment variable or the file named it.
    pub declared: bool,
}

/// Environment variable that overrides the log directory.
pub const ENV_LOG_DIR: &str = "MAGI_LOG_DIR";
/// Environment variable that overrides the file filter.
pub const ENV_FILE_FILTER: &str = "MAGI_LOG_FILTER";

impl MagiConfig {
    /// Resolves the log directory by precedence.
    ///
    /// `--log-dir` > `MAGI_LOG_DIR` > `[logging].log_dir` > `.magi/logs`.
    ///
    /// # Parameters
    ///
    /// * `flag` — the `--log-dir` value, if given.
    /// * `env` — the environment variable's value, if set.
    /// * `default` — where logs go when nothing was declared, which the
    ///   workspace already knows (`Workspace::logs_dir()`).
    ///
    /// # A blank environment variable is ABSENT, never invalid
    ///
    /// An exported-but-unfilled variable is an everyday accident in a CI script.
    /// Treating it as invalid turns that accident into a hard failure — and,
    /// worse, into D-L20's fatal path, which applies only to a log directory
    /// that was actually **declared** and cannot be created.
    ///
    /// # Complexity
    ///
    /// `O(1)`.
    #[must_use]
    pub fn resolve_log_dir(
        &self,
        flag: Option<&str>,
        env: Option<&str>,
        default: &std::path::Path,
    ) -> ResolvedLogDir {
        let declared = non_blank(flag)
            .or_else(|| non_blank(env))
            .or_else(|| non_blank(self.logging.log_dir.as_deref()));
        ResolvedLogDir {
            // A BLANK value is absent, never declared. An exported-but-unfilled
            // MAGI_LOG_DIR is an everyday CI accident, and reading it as
            // "declared" turns that accident into the hard startup failure
            // D-L20 reserves for a path the operator actually asked for.
            declared: declared.is_some(),
            path: declared.map_or_else(|| default.to_path_buf(), std::path::PathBuf::from),
        }
    }

    /// Resolves the file branch's filter by the same precedence.
    ///
    /// # Complexity
    ///
    /// `O(1)`.
    #[must_use]
    pub fn resolve_file_filter(&self, flag: Option<&str>, env: Option<&str>) -> String {
        non_blank(flag)
            .or_else(|| non_blank(env))
            .or_else(|| non_blank(self.logging.file_filter.as_deref()))
            .unwrap_or("info")
            .to_string()
    }

    /// The retention settings, with the built-in defaults filled in.
    ///
    /// # Complexity
    ///
    /// `O(1)`.
    #[must_use]
    pub fn retention(&self) -> magi_rs::logging::retention::RetentionConfig {
        magi_rs::logging::retention::RetentionConfig {
            compress: self.logging.compress.unwrap_or(true),
            compress_after_days: self.logging.compress_after_days.unwrap_or(7),
            retain_days: self.logging.retain_days.unwrap_or(30),
            max_total_bytes: self.logging.max_total_bytes.unwrap_or(u64::MAX),
        }
    }
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
    /// It runs every check [`MagiConfig::from_toml_str`] applies **to a parsed config**, so a
    /// builder-built config and a file-loaded one are subject to the same rules — a builder that
    /// validated less would let the suite exercise a state production cannot reach, and it did:
    /// see `the_builder_rejects_a_seat_that_declares_a_model_without_its_lineage`.
    ///
    /// The one thing it cannot run is [`migrate::detect_migrations`], which matches **raw TOML
    /// text** and so has no input here. That is a difference in kind, not a gap in rigour: a
    /// builder never has a source file to carry a retired pattern.
    ///
    /// # Errors
    ///
    /// Whatever [`MagiConfig::validate_vocabulary`] rejects: [`ConfigError::NeedsMigration`] for
    /// a seat declaring a model without its lineage, [`ConfigError::UnknownProviderKind`],
    /// [`ConfigError::UnknownMode`], [`ConfigError::AgentTimeoutOutOfRange`],
    /// [`ConfigError::OutputCapTooSmall`] or [`ConfigError::Diversity`].
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
    // `log_retention`, `log_max_bytes` and `log_level` are GONE (REQ-L32), with
    // no guided migration and none coming: a file that still declares one gets
    // serde's bare `unknown field`, which is fatal. `src/config/migrate.rs` is
    // deliberately not extended for them.
    //
    // What the bare error cannot say is in the CHANGELOG (REQ-L33): retention
    // moved from RUNS to DAYS and there is no conversion. Without that, the
    // natural path — read the name in the error, find the replacement, copy the
    // number over — produces a config that starts fine and retains something
    // else entirely.
    // `tool_result_cap_bytes` NO LONGER LIVES HERE (Task 1.3, third migration pattern of
    // REQ-A21b): it moved up to the root level because under `[headless]` it only covered batch
    // mode and left interactive mode loose, which is exactly where the report is re-sent on
    // every turn of a long session. A cap that protects the cheap case and not the expensive
    // one protects the wrong case. A file that still declares it here received the guided
    // migration error until v0.13.0 retired that pattern set (REQ-R22); it now gets serde's bare
    // `unknown field` — see `detect_migrations`. Default log level
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

    /// Independent failure domain of [`Self::melchior_model`] (REQ-R02).
    ///
    /// **Mandatory for a seat that declares a model, and never inferred** (R-R03): the lineage is
    /// the label that decides all rotation eligibility, and it is a semantic choice of the
    /// operator — the same two models can legitimately be two lineages for one user and one for
    /// another. A seat that inherits the built-in model inherits the built-in lineage with it, so
    /// only a *declared* model obliges a declared lineage. Blank counts as absent, like every
    /// other text key.
    ///
    /// A `magi.toml` from v0.12.0 has none of these keys; that break is reported whole by
    /// [`migrate::missing_seat_lineages`], never one key per start.
    pub melchior_lineage: Option<String>,
    /// Independent failure domain of [`Self::balthasar_model`] — see [`Self::melchior_lineage`].
    pub balthasar_lineage: Option<String>,
    /// Independent failure domain of [`Self::caspar_model`] — see [`Self::melchior_lineage`].
    pub caspar_lineage: Option<String>,
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

    /// Fallback models a mage may rotate through; absent ⇒ [`DEFAULT_MAX_ROTATIONS`] (REQ-R05).
    ///
    /// **`0` is the kill-switch**, and it must survive as a declared value: collapsing `None` and
    /// `Some(0)` would turn an explicit "no rotation" into "use the default".
    pub max_rotations: Option<u32>,
    /// Refuse fallback candidates whose window could not be measured; absent ⇒
    /// [`DEFAULT_STRICT_CONTEXT_GUARD`] (REQ-R11).
    ///
    /// What the operator **declares**. Whether it is **applied** is another matter: magi-rs passes
    /// it down only when at least one candidate has a measured window, because a `true` with
    /// nothing measured would disqualify every candidate and switch rotation off in silence.
    pub strict_context_guard: Option<bool>,
    /// Require the three seats to declare distinct lineages; absent ⇒
    /// [`DEFAULT_ENFORCE_DIVERSITY`] (REQ-R29).
    ///
    /// **Exclusive to magi-rs** — never forwarded to magi-core, which treats the lineage as an
    /// opaque string.
    pub enforce_diversity: Option<bool>,

    /// The shared rotation pool, ordered strongest to weakest (REQ-R13).
    ///
    /// **Goes LAST in the file, not last in the `[magi]` block.** In TOML every loose key and
    /// sub-table must precede the first array of tables, so a rule phrased as "last in the block"
    /// would let a later addition to `[magi]` land after the array and parse into the wrong table.
    /// "Last in the file" leaves nowhere to get it wrong.
    #[serde(default)]
    pub fallback: Vec<magi_rs::magi::rotation_config::FallbackEntry>,
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

    /// The independent failure domain of one seat: the declared label, or the built-in one when
    /// the seat declares no model (REQ-R01/R02).
    ///
    /// # The trigger rule, and why it has two halves
    ///
    /// A seat that **declares a model** owes a lineage: the label decides every rotation
    /// eligibility question, it is a semantic choice of the operator, and there is no observable
    /// datum to derive it from without inventing one (R-R03). A seat that **declares no model**
    /// runs the built-in model, whose lineage is built in too — so it inherits both together and
    /// owes nothing. Without that second half the default configuration, belonging to whoever
    /// never touched `[magi]`, would register three seats with no lineage, and
    /// `MagiBuilder::build()` rejects a blank one outright.
    ///
    /// A declared lineage wins in both cases: the operator who bothered to label a seat is the
    /// authority on what its failure domain is.
    ///
    /// Blank counts as **absent**, never invalid — the rule every text key has followed since MS2.
    ///
    /// # Errors
    ///
    /// [`LineageError::Missing`] naming the configuration key, when a seat declares a model and no
    /// lineage. `MagiConfig::load` already rejects that file through
    /// [`migrate::missing_seat_lineages`], which reports all three seats at once; this is the same
    /// rule enforced at the point of use, for the crate-internal builder that bypasses `load`.
    #[must_use = "the resolved lineage is what the seat registers with"]
    pub fn lineage_of_seat(&self, seat: AgentName) -> Result<Lineage, LineageError> {
        let (declared, model, key, built_in) = match seat {
            AgentName::Melchior => (
                &self.melchior_lineage,
                &self.melchior_model,
                "melchior_lineage",
                crate::defaults::DEFAULT_MAGI_MELCHIOR_LINEAGE,
            ),
            AgentName::Balthasar => (
                &self.balthasar_lineage,
                &self.balthasar_model,
                "balthasar_lineage",
                crate::defaults::DEFAULT_MAGI_BALTHASAR_LINEAGE,
            ),
            AgentName::Caspar => (
                &self.caspar_lineage,
                &self.caspar_model,
                "caspar_lineage",
                crate::defaults::DEFAULT_MAGI_CASPAR_LINEAGE,
            ),
        };

        match declared.as_deref().map(Lineage::parse) {
            // Declared and non-blank: the operator's label wins, trimmed.
            Some(Ok(lineage)) => Ok(lineage),
            // Absent or blank. `Lineage::parse` cannot name the key that was empty — it is a value
            // parser — so the error is remapped here, where the key IS known.
            _ if model.as_deref().is_some_and(|m| !m.trim().is_empty()) => {
                Err(LineageError::Missing { key })
            }
            // No model declared either: the seat runs the built-in model and inherits its lineage.
            _ => Lineage::parse(built_in),
        }
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
            melchior_lineage: None,
            balthasar_lineage: None,
            caspar_lineage: None,
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
            max_rotations: None,
            strict_context_guard: None,
            enforce_diversity: None,
            fallback: Vec::new(),
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

    /// Fallback models a mage may rotate through (REQ-R05).
    ///
    /// A declared `0` is honoured as the kill-switch; only an **absent** key falls back to
    /// [`DEFAULT_MAX_ROTATIONS`].
    ///
    /// **No upper bound, and that is deliberate** (asked by S1 Loop 2, Caspar, noting that
    /// `agent_timeout_secs` and `tool_result_cap_bytes` both carry ranges). Three reasons, in
    /// order: the arithmetic cannot overflow — `u32::MAX + 1` models times two attempts times a
    /// ceiling capped at 120 is on the order of `1e12`, against a `u64` headroom of `1.8e19`;
    /// real rotation is bounded by the POOL, which is finite and declared, so a number past its
    /// length buys nothing; and REQ-R05 specifies a default and a kill-switch and no range, so
    /// inventing one here would be behaviour the spec does not define. If a bound is wanted it
    /// is a spec change, not a hardening patch.
    #[must_use]
    pub(crate) fn effective_max_rotations(&self) -> u32 {
        self.magi
            .max_rotations
            .unwrap_or(crate::defaults::DEFAULT_MAX_ROTATIONS)
    }

    /// What the operator **declared** for the context guard (REQ-R11).
    ///
    /// Deliberately named `declared_`, not `effective_`: whether the guard is actually applied
    /// depends on there being at least one measured candidate window, which this type cannot know.
    /// That decision belongs to the trio construction, and naming it here would invite a caller to
    /// pass this value straight to magi-core — which is the silent-rotation-shutdown bug.
    #[must_use]
    pub(crate) fn declared_strict_context_guard(&self) -> bool {
        self.magi
            .strict_context_guard
            .unwrap_or(crate::defaults::DEFAULT_STRICT_CONTEXT_GUARD)
    }

    /// Whether the three seats must declare distinct lineages (REQ-R29).
    #[must_use]
    pub(crate) fn effective_enforce_diversity(&self) -> bool {
        self.magi
            .enforce_diversity
            .unwrap_or(crate::defaults::DEFAULT_ENFORCE_DIVERSITY)
    }

    /// The shared rotation pool, in declared order (strongest to weakest).
    #[must_use]
    pub(crate) fn fallback_pool(&self) -> &[magi_rs::magi::rotation_config::FallbackEntry] {
        &self.magi.fallback
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
    /// [`ConfigError::NeedsMigration`] if the file brings a declared migration pattern — no set
    /// is declared today, so it never occurs (see [`migrate`]); [`ConfigError::Parse`] if the
    /// TOML does not
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
        // The v0.13.0 half of the same pass runs INSIDE `validate_vocabulary`, as its first
        // check — see the comment there for why it lives at that depth rather than here.
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
        // The v0.13.0 half of the migration pass, and it can only run POST-parse: a MISSING key
        // produces no serde complaint at all, so there is nothing for a pre-parse pass to get
        // ahead of, and the typed struct is what a rename would break the compile over (REQ-R22,
        // SC-R46/R47). It goes FIRST here for the same reason the pre-parse half goes before the
        // deserialize: a file one generation behind should be told to migrate, not audited for
        // vocabulary it is about to rewrite anyway.
        //
        // **Why inside this function and not in `from_toml_str`.** It used to sit in the caller,
        // which made the guarantee depend on every caller remembering to run two checks in the
        // right order — and `MagiConfigBuilder::build` did not, so a builder-built config with no
        // lineages passed BOTH this check (never called) and `validate_diversity_rules` (which
        // skips silently, on a premise only `from_toml_str` honoured). Moving it here makes the
        // premise locally true for every caller instead of a convention two of them share.
        let missing = migrate::missing_seat_lineages(&self.magi);
        if !missing.is_empty() {
            return Err(ConfigError::NeedsMigration(
                migrate::render_migration_error(&missing),
            ));
        }
        ProviderKind::parse(self.provider.as_deref().unwrap_or_default())?;
        ProviderKind::parse(self.magi.kind.as_deref().unwrap_or_default())?;
        <Mode as ModeExt>::parse_config_value(
            self.magi.default_mode.as_deref().unwrap_or_default(),
        )?;
        self.validate_agent_timeout()?;
        self.validate_output_cap()?;
        self.validate_diversity_rules()?;
        Ok(())
    }

    /// Enforces REQ-R29's **declarative** half at load time.
    ///
    /// # Why this lives here and not in the trio builder
    ///
    /// It is a property of the configuration, knowable the moment the file parses, and a load
    /// error is the only thing that reaches an operator *before* they depend on the safety net.
    /// Discovering it at rotation time means discovering it when a mage has already fallen —
    /// the worst moment, and the one the requirement exists to avoid.
    ///
    /// # Errors
    ///
    /// [`ConfigError::Diversity`] when `enforce_diversity` is on and either two seats share a
    /// lineage or a declared pool leaves a seat uncovered.
    fn validate_diversity_rules(&self) -> Result<(), ConfigError> {
        // A seat whose lineage cannot be resolved is already reported by
        // `migrate::missing_seat_lineages`, which names ALL of them at once. Failing here on the
        // first would replace a complete message with a partial one.
        //
        // **That premise is now local, not a convention between callers** (S1 Loop 2). The check
        // it defers to runs at the head of `validate_vocabulary`, which is the only function that
        // calls this one — so by the time control reaches here, a missing lineage has already
        // returned an error and this `Err` arm is unreachable from every validating path. It
        // stays because `build_unvalidated` exists and reaching it must degrade to silence
        // rather than panic, not because anything on the way in might still forget.
        let mut seats = Vec::new();
        for name in [AgentName::Melchior, AgentName::Balthasar, AgentName::Caspar] {
            match self.magi.lineage_of_seat(name) {
                Ok(lineage) => seats.push((name, lineage)),
                Err(_) => return Ok(()),
            }
        }
        magi_rs::magi::rotation_config::validate_diversity(
            &seats,
            &self.magi.fallback,
            self.effective_enforce_diversity(),
        )?;
        Ok(())
    }

    /// Every diversity notice this configuration owes, given the models the seats will ACTUALLY
    /// run.
    ///
    /// # It takes the resolved seats instead of re-deriving them
    ///
    /// Production resolves a seat's model as `env > TOML > backend`, through
    /// `seats_with_env_overrides`. Re-deriving here from `seats(backend_model)` would skip the env
    /// layer and make this a **third** resolver disagreeing with the two the project already
    /// unified — the same divergence closed once before, when the probe was moved onto that
    /// resolution so it could not measure a different model from the one the trio runs.
    ///
    /// It would also print statements that are simply false, in both directions: three distinct
    /// `MAGI_MODEL_*` over an undeclared trio would be told all three mages share one model, and
    /// a declared trio collapsed to one model BY those variables would hear nothing.
    ///
    /// # The two notices
    ///
    /// **Coverage** comes from [`validate_diversity`]'s soft path: under `enforce_diversity =
    /// false` an uncovered seat is a notice rather than an error, and dropping that vector would
    /// leave the mono-provider user — the one the requirement exists for — in silence.
    ///
    /// **Collapse** is the case labels cannot see. `seats()` falls an undeclared seat back to the
    /// backend model, so a configuration naming no trio runs one model under three distinct
    /// built-in labels: literally what a label-distinctness check approves and what SC-R44
    /// rejects. A notice and not an error, because an absent `magi.toml` must start silently and
    /// pointing at a single-model endpoint is a choice rather than a mistake.
    ///
    /// [`validate_diversity`]: magi_rs::magi::rotation_config::validate_diversity
    #[must_use]
    pub(crate) fn diversity_notices(&self, resolved_seats: &[(AgentName, String)]) -> Vec<Notice> {
        let mut notices = Vec::new();

        // Coverage. Under `enforce = true` this cannot error here — `load()` already validated and
        // would have refused the file — and since S1 Loop 2 the same is true of the builder's
        // validating exit, which now runs the seat-lineage check too. What remains reachable is
        // `build_unvalidated` alone, and it degrades to silence rather than a panic.
        let with_lineage: Vec<(AgentName, Lineage)> = resolved_seats
            .iter()
            .filter_map(|(seat, _)| self.magi.lineage_of_seat(*seat).ok().map(|l| (*seat, l)))
            .collect();
        if with_lineage.len() == resolved_seats.len() {
            if let Ok(coverage) = magi_rs::magi::rotation_config::validate_diversity(
                &with_lineage,
                &self.magi.fallback,
                self.effective_enforce_diversity(),
            ) {
                notices.extend(coverage);
            }
        }

        // Collapse.
        let distinct: std::collections::BTreeSet<&str> = resolved_seats
            .iter()
            .map(|(_, model)| model.as_str())
            .collect();
        if let Some((_, model)) = resolved_seats.first().filter(|_| distinct.len() == 1) {
            // BOTH levers are named. The collapse can come from the TOML *or* from three
            // identical `MAGI_MODEL_*`, and the environment wins — so telling someone who has
            // already declared a model per seat to go declare one is advice they have followed
            // and that cannot help. The sibling notice in `rotation_config.rs` had exactly this
            // defect one iteration ago.
            //
            // Binding the model through `first()` also retires a `map_or("<unknown>", …)` that
            // `distinct.len() == 1` had made unreachable: a fallback string describing a state
            // that cannot occur reads as though the state can.
            notices.push(Notice::resolution(format!(
                "notice: all three mages resolve to the same model (`{model}`), so their declared \
                 lineages describe one failure domain rather than three. The consensus still has \
                 three perspectives — those are structural — but a shared outage takes all three \
                 at once. Declare a model per seat in `[magi]`, or check whether `MAGI_MODEL_*` \
                 is overriding them — the environment wins over the file."
            )));
        }

        notices
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

    // S1 Loop 2 (Caspar, INFO) asked why this has no `assert!` of its own on the branch where
    // `[magi].kind` IS valid, unlike its two siblings. It is deliberate and stays: the assertion
    // guards "validate_vocabulary was skipped", and on that branch the answer this returns comes
    // from `magi.kind` alone and is correct regardless. When `magi.kind` is absent or invalid the
    // call delegates to `effective_provider`, whose assertion does fire — and `validate_vocabulary`
    // checks `magi.kind` too, so an invalid one cannot reach a caller either way. Adding a direct
    // assertion would buy no coverage and would change a property that
    // `effective_magi_kind_panics_via_the_delegated_effective_provider_check_when_invalid` names
    // and pins. Recorded here so the next reader does not file it a second time.

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

    /// SC-L122: a blank env var is ABSENT, and absence never reaches D-L20.
    #[test]
    fn a_declared_log_dir_is_marked_declared_and_a_defaulted_one_is_not() {
        // D-L20 turns on exactly this bit: declared fails the startup, defaulted
        // degrades to a notice. Getting it backwards either takes CI down over a
        // directory nobody asked for, or hands an operator who DID ask an empty
        // directory and no clue -- the complaint the feature exists to answer.
        let cfg = MagiConfig::default();
        let fallback = std::path::Path::new("/default/logs");

        assert!(
            cfg.resolve_log_dir(Some("/from/flag"), None, fallback)
                .declared,
            "a flag is a declaration"
        );
        assert!(
            cfg.resolve_log_dir(None, Some("/from/env"), fallback)
                .declared,
            "so is an environment variable"
        );
        assert!(
            !cfg.resolve_log_dir(None, None, fallback).declared,
            "and the fallback is not"
        );
    }

    #[test]
    fn a_blank_log_dir_is_absent_rather_than_declared() {
        // An exported-but-unfilled MAGI_LOG_DIR is an everyday CI accident.
        // Reading it as declared converts that accident into D-L20's hard
        // failure, which is the opposite of what the rule is for.
        let cfg = MagiConfig::default();
        let fallback = std::path::Path::new("/default/logs");
        for blank in ["", "   ", "\t"] {
            let r = cfg.resolve_log_dir(None, Some(blank), fallback);
            assert!(!r.declared, "{blank:?} must not count as declared");
            assert_eq!(r.path, fallback);
        }
    }

    #[test]
    fn a_blank_log_dir_env_var_does_not_break_startup() {
        let cfg = MagiConfig::default();
        let fallback = std::path::Path::new("/w/.magi/logs");

        // Exported and left unfilled is an everyday CI accident. Treating it as
        // invalid turns the accident into a hard failure — and into the fatal
        // path of D-L20, which applies only to a directory actually DECLARED
        // and impossible to create.
        for blank in [
            "", "   ", "	", "
  ",
        ] {
            assert_eq!(
                cfg.resolve_log_dir(None, Some(blank), fallback).path,
                fallback,
                "a blank env ({blank:?}) must fall through to the default"
            );
        }
    }

    #[test]
    fn the_log_dir_precedence_is_flag_then_env_then_file_then_default() {
        let cfg = MagiConfig::default();
        let fallback = std::path::Path::new("/w/.magi/logs");

        assert_eq!(
            cfg.resolve_log_dir(Some("/from/flag"), Some("/from/env"), fallback)
                .path,
            std::path::PathBuf::from("/from/flag"),
            "the flag wins over the env"
        );
        assert_eq!(
            cfg.resolve_log_dir(None, Some("/from/env"), fallback).path,
            std::path::PathBuf::from("/from/env"),
            "the env wins over the file and the default"
        );
        assert_eq!(cfg.resolve_log_dir(None, None, fallback).path, fallback);
    }

    #[test]
    fn an_unknown_key_in_the_logging_section_is_a_parse_error() {
        // `deny_unknown_fields`: a typo is rejected, never silently accepted.
        let toml = "[logging]
log_dirr = \"/typo\"
";
        assert!(
            toml::from_str::<MagiConfig>(toml).is_err(),
            "a misspelled key must not be swallowed"
        );
        // And the correctly spelled one parses, so the assertion above is not
        // passing because the whole section is rejected.
        let ok = "[logging]
log_dir = \"/right\"
";
        assert!(toml::from_str::<MagiConfig>(ok).is_ok());
    }

    #[test]
    fn format_and_rotation_tz_are_rejected_because_they_do_nothing_yet() {
        // Their absence is not an oversight: an inert key promises what the
        // binary does not do. `deny_unknown_fields` is what makes the promise
        // impossible to make by accident.
        for key in ["format", "rotation_tz"] {
            let toml = format!(
                "[logging]
{key} = \"whatever\"
"
            );
            assert!(
                toml::from_str::<MagiConfig>(&toml).is_err(),
                "{key} must not parse until it works"
            );
        }
    }
    use super::*;

    /// A commented-out `[embedding].base_url` INHERITS the root endpoint.
    ///
    /// This is the behaviour `magi init`'s scaffold now depends on: it emits the key commented
    /// so the embedder follows whatever the operator sets at the root, instead of pinning
    /// localhost forever. Before v0.14.0 the scaffold emitted it ACTIVE, so the inheritance
    /// path existed in `override_or_inherit_base_url` and the generated file never took it.
    ///
    /// Characterization test, stated plainly: it passed the moment it was written, because the
    /// resolver was always correct. What was wrong was the file feeding it. Its value is as a
    /// **lock** — the scaffold's decision to omit the key is only safe while omission means
    /// inheritance, and nothing else in the suite says so.
    #[test]
    fn an_absent_embedding_base_url_inherits_the_root_endpoint() {
        let toml = r#"
provider = "ollama"
base_url = "http://remote-host:11434/v1"

[embedding]
model = "nomic-embed-text-v2-moe:latest"
"#;
        let cfg = MagiConfig::from_toml_str(toml).expect("valid config");

        let root = cfg.effective_base_url().expect("root endpoint");
        let embedding = cfg
            .effective_embedding_base_url()
            .expect("embedding endpoint");
        assert_eq!(
            embedding,
            root,
            "an absent [embedding].base_url must resolve to the ROOT endpoint, not to the built-in localhost default"
        );
        assert!(
            root.as_str().contains("remote-host"),
            "precondition: the root must be the non-default value, or this test cannot tell inheritance from a coincidence with the built-in default"
        );
    }

    /// And a DECLARED `[embedding].base_url` still overrides, so the inheritance above is a
    /// consequence of absence rather than the key having stopped working.
    #[test]
    fn a_declared_embedding_base_url_still_overrides_the_root() {
        let toml = r#"
provider = "ollama"
base_url = "http://remote-host:11434/v1"

[embedding]
model = "nomic-embed-text-v2-moe:latest"
base_url = "http://embedder-host:11434/v1"
"#;
        let cfg = MagiConfig::from_toml_str(toml).expect("valid config");
        let embedding = cfg
            .effective_embedding_base_url()
            .expect("embedding endpoint");
        assert!(
            embedding.as_str().contains("embedder-host"),
            "a declared override must win over the root"
        );
    }

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
    /// The example's rotation pool must equal `DEFAULT_SCAFFOLD_POOL` — model AND lineage, in
    /// order.
    ///
    /// Its neighbours already mirror the trio's models and lineages and the complexity
    /// thresholds against `defaults.rs`; the pool was the largest set of literals left
    /// hand-synced, and v0.14.0 grew it from three pairs to five across two separate commits —
    /// one editing the generator, one editing the example. That manual mirror step is exactly
    /// what this file's neighbouring rustdoc warns about: the example still PARSING proves the
    /// labels are present, and says nothing about them still being the right ones.
    ///
    /// Order matters and is asserted: the pool is rotation PREFERENCE, strongest first, so a
    /// reordered example teaches a different fallback order than the tool generates.
    #[test]
    fn example_toml_fallback_pool_matches_the_builtin_scaffold_pool() {
        let raw = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/docs/magi.toml.example"
        ))
        .expect("docs/magi.toml.example must be readable");
        let parsed = MagiConfig::from_toml_str(&raw).expect("the example must parse");

        let actual: Vec<(String, String)> = parsed
            .fallback_pool()
            .iter()
            .map(|e| (e.model.clone(), e.lineage.to_string()))
            .collect();
        let expected: Vec<(String, String)> = crate::defaults::DEFAULT_SCAFFOLD_POOL
            .iter()
            .map(|(m, l)| ((*m).to_string(), (*l).to_string()))
            .collect();

        assert_eq!(
            actual, expected,
            "docs/magi.toml.example's [[magi.fallback]] entries must match              DEFAULT_SCAFFOLD_POOL exactly, in order"
        );
    }

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
        // Same drift, one release later: since v0.13.0 each declared model owes a declared
        // lineage, so the example now carries three more literals that can rot away from
        // `src/defaults.rs`. That the example still PARSES only proves the labels are present —
        // it says nothing about them still being the right ones.
        assert_eq!(
            parsed.magi.melchior_lineage.as_deref(),
            Some(crate::defaults::DEFAULT_MAGI_MELCHIOR_LINEAGE),
            "docs/magi.toml.example's melchior_lineage must mirror DEFAULT_MAGI_MELCHIOR_LINEAGE"
        );
        assert_eq!(
            parsed.magi.balthasar_lineage.as_deref(),
            Some(crate::defaults::DEFAULT_MAGI_BALTHASAR_LINEAGE),
            "docs/magi.toml.example's balthasar_lineage must mirror DEFAULT_MAGI_BALTHASAR_LINEAGE"
        );
        assert_eq!(
            parsed.magi.caspar_lineage.as_deref(),
            Some(crate::defaults::DEFAULT_MAGI_CASPAR_LINEAGE),
            "docs/magi.toml.example's caspar_lineage must mirror DEFAULT_MAGI_CASPAR_LINEAGE"
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
            // Both declared seats carry their lineage: since v0.13.0 a declared model without one
            // is a migration error, so a fixture that omitted it would be testing that error
            // rather than the section parsing this test is about.
            "[magi]\nmelchior_model = \"qwen3:8b\"\nmelchior_lineage = \"alibaba\"\n\
             caspar_model = \"deepseek-r1:32b\"\ncaspar_lineage = \"deepseek\"\n",
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
        let c2 = MagiConfig::from_toml_str(
            "[magi]\nmelchior_model = \"qwen3:8b\"\nmelchior_lineage = \"alibaba\"\n",
        )
        .unwrap();
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
             timeout_secs = 120\n\
             allow_system_override = true\n",
        )
        .unwrap();

        assert_eq!(c.headless.max_input_bytes, Some(2048));
        assert_eq!(c.headless.full_auto_max_tool_calls, Some(30));
        assert_eq!(c.headless.timeout_secs, Some(120));
        assert_eq!(c.headless.allow_system_override, Some(true));
    }

    /// REQ-L32: the three retired keys are a PARSE ERROR, with no guided
    /// migration and none coming.
    ///
    /// What the bare `unknown field` cannot say is in the CHANGELOG (REQ-L33):
    /// retention moved from RUNS to DAYS with no conversion. The natural path —
    /// read the name in the error, find the replacement, copy the number over —
    /// otherwise produces a config that starts fine and retains something else.
    #[test]
    fn the_retired_headless_log_keys_are_rejected_outright() {
        for key in [
            "log_retention = 7",
            "log_max_bytes = 1048576",
            "log_level = \"debug\"",
        ] {
            let toml = format!(
                "[headless]
{key}
"
            );
            assert!(
                MagiConfig::from_toml_str(&toml).is_err(),
                "{key} must be fatal, not silently ignored"
            );
        }
        // And a section without them still parses, so the assertions above are
        // not passing because `[headless]` itself broke.
        assert!(MagiConfig::from_toml_str(
            "[headless]
max_input_bytes = 2048
"
        )
        .is_ok());
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
            // S1 Loop 2 (Balthasar): asserting only that SOME error occurred and that it does not
            // leak is a guardian that does not guard. Drop `deny_unknown_fields` from the four
            // structs and the parse SUCCEEDS, the seat-lineage check errors instead, and that
            // error contains no `sk-secreto` either — so both assertions below used to hold while
            // an `api_key` was being silently accepted. Naming the field is what ties the test to
            // the rejection it exists to pin (B16).
            assert!(
                err.to_string().contains("api_key"),
                "and it must be the unknown-field rejection that NAMES the key: {err}"
            );
            assert!(
                !err.to_string().contains("sk-secreto"),
                "and the error must NOT repeat the secret it is rejecting"
            );
        }
    }

    /// S1 Loop 2 (Caspar and Balthasar, one root cause): [`MagiConfigBuilder::build`] promised
    /// parity with [`MagiConfig::from_toml_str`] and did not have it. The seat-lineage check ran
    /// in the CALLER, so `build` skipped it — and [`MagiConfig::validate_diversity_rules`] then
    /// skipped ITSELF, on the documented premise that the lineage check had already reported.
    /// A builder-built config declaring a model with no lineage therefore passed **both**, which
    /// is precisely the "state production cannot reach" the builder's rustdoc claims is
    /// impossible.
    ///
    /// **Mutation-verified (B16):** move `missing_seat_lineages` back out of
    /// `validate_vocabulary` and into `from_toml_str`, and this test goes green again because
    /// `build()` accepts the config.
    #[test]
    fn the_builder_rejects_a_seat_that_declares_a_model_without_its_lineage() {
        let magi = MagiSectionConfig {
            melchior_model: Some("qwen3.5:397b-cloud".into()),
            ..Default::default()
        };

        let err = MagiConfig::builder()
            .magi(magi)
            .build()
            .expect_err("a seat with a model and no lineage must not build");

        assert!(
            matches!(err, ConfigError::NeedsMigration(_)),
            "and it is the GUIDED migration error, not a bare vocabulary complaint: {err:?}"
        );
        assert!(
            err.to_string().contains("melchior_lineage"),
            "naming the key the operator has to add: {err}"
        );
    }

    /// S1 Loop 2 (Caspar): a `[[magi.fallback]]` entry with a blank `model` is rejected at parse
    /// time, the same way its sibling `lineage` field already was.
    ///
    /// The consequence of accepting it is worse than an entry that fails when a mage rotates: a
    /// blank candidate still counts toward **pool coverage**, so `seats_without_coverage` would
    /// certify a seat as covered by something that can never answer — the illusory safety net the
    /// coverage check exists to prevent.
    ///
    /// **Mutation-verified (B16):** drop the `deserialize_with` on `FallbackEntry::model` and
    /// this goes green, because the entry parses and joins the pool.
    #[test]
    fn a_fallback_entry_with_a_blank_model_is_rejected_at_parse_time() {
        let toml = "[magi]\n\
                    melchior_model = \"a\"\nmelchior_lineage = \"la\"\n\
                    balthasar_model = \"b\"\nbalthasar_lineage = \"lb\"\n\
                    caspar_model = \"c\"\ncaspar_lineage = \"lc\"\n\
                    \n[[magi.fallback]]\nmodel = \"   \"\nlineage = \"lz\"\n";

        let err = MagiConfig::from_toml_str(toml)
            .expect_err("a fallback entry with a blank model must not parse");

        assert!(
            err.to_string().contains("model"),
            "and the error must name the field the operator has to fill: {err}"
        );

        // And the surviving half of the same asymmetry: `lineage` is trimmed by `Lineage::parse`,
        // so `model` is too. Untrimmed, ` qwen ` reaches the endpoint with its spaces and 404s
        // for a reason that is invisible in the file.
        let padded = "[magi]\n\
                      melchior_model = \"a\"\nmelchior_lineage = \"la\"\n\
                      balthasar_model = \"b\"\nbalthasar_lineage = \"lb\"\n\
                      caspar_model = \"c\"\ncaspar_lineage = \"lc\"\n\
                      \n[[magi.fallback]]\nmodel = \"  qwen3.5:397b-cloud  \"\nlineage = \"lz\"\n";
        let cfg = MagiConfig::from_toml_str(padded).expect("a padded model tag still parses");
        assert_eq!(
            cfg.fallback_pool()
                .first()
                .expect("the pool has the declared entry")
                .model,
            "qwen3.5:397b-cloud",
            "the tag must be stored trimmed, the same way its sibling lineage is"
        );
    }

    /// S1 Loop 2 (Balthasar): `ConfigError::Endpoint` now redacts instead of trusting
    /// [`EndpointError`]'s `Display`, and this pins the half of that change which **can** be
    /// exercised.
    ///
    /// **Said plainly: the leak-blocking half is not testable today, by construction.** All four
    /// `EndpointError` variants carry fixed messages whose only fields are `&'static str` vault
    /// entry names, so none of them can embed a URL — there is nothing for the redaction to
    /// catch, and a test asserting otherwise would have to build the leaking string itself, which
    /// is the canary B16 calls useless. The layer is defence against a variant someone adds
    /// later.
    ///
    /// What IS at risk right now is the opposite failure, and it is the one CLAUDE.md warns
    /// about when it says the two redaction helpers are not interchangeable: the wrong one
    /// collapses a message to `***`. These errors are the operator's instructions for fixing
    /// their config, so a redaction that ate them would trade a hypothetical leak for a certain
    /// dead end.
    #[test]
    fn wrapping_an_endpoint_error_redacts_without_eating_the_instructions() {
        let err = ConfigError::from(EndpointError::MissingVaultEntry {
            entry: "BASE_URL_PASSWORD",
        });
        let text = err.to_string();

        assert!(
            text.contains("BASE_URL_PASSWORD") && text.contains("magi-rs vault set"),
            "the operator must still be told WHICH entry and HOW to create it: {text}"
        );
        assert!(
            !text.contains("***"),
            "and nothing here is a credential, so nothing may be blanked: {text}"
        );
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

    // -------------------------------------------------------------------------
    // `lineage_of_seat()` — REQ-R01/R02, SC-R05, SC-R19.

    /// SC-R05: a seat that DECLARES a model without its lineage is an explicit configuration
    /// error, never an inference from the model name. Guessing it would fabricate the label that
    /// decides all rotation eligibility, and the lineage is a SEMANTIC CHOICE of the operator
    /// (R-R03) — the same pair of models can legitimately be two lineages for one user and one
    /// for another, so there is no observable datum to derive it from without inventing one.
    #[test]
    fn a_seat_that_declares_a_model_without_its_lineage_is_an_error_not_an_inference() {
        let cfg = MagiSectionConfig {
            melchior_model: Some("qwen3.5:397b-cloud".into()),
            ..MagiSectionConfig::default()
        };
        let err = cfg
            .lineage_of_seat(AgentName::Melchior)
            .expect_err("a declared model without its lineage must error");
        assert!(
            matches!(
                err,
                LineageError::Missing {
                    key: "melchior_lineage"
                }
            ),
            "the error must name the key the operator has to add: {err:?}"
        );
    }

    /// The trigger rule established by Task 2.6, and the half that was NOT wired until now: a seat
    /// that declares no model runs the built-in one, so it inherits the built-in lineage with it.
    /// Without this, the default configuration — the one belonging to whoever never touched the
    /// trio — registers seats with no lineage and rotation has nothing to decide on.
    #[test]
    fn a_seat_that_declares_no_model_inherits_the_built_in_lineage() {
        let cfg = MagiSectionConfig::default();
        for (seat, expected) in [
            (
                AgentName::Melchior,
                crate::defaults::DEFAULT_MAGI_MELCHIOR_LINEAGE,
            ),
            (
                AgentName::Balthasar,
                crate::defaults::DEFAULT_MAGI_BALTHASAR_LINEAGE,
            ),
            (
                AgentName::Caspar,
                crate::defaults::DEFAULT_MAGI_CASPAR_LINEAGE,
            ),
        ] {
            let resolved = cfg
                .lineage_of_seat(seat)
                .unwrap_or_else(|e| panic!("{seat:?} must inherit a built-in lineage, got {e:?}"));
            assert_eq!(resolved.as_str(), expected, "wrong built-in for {seat:?}");
        }
    }

    /// SC-R19: blank is ABSENT, never invalid — the rule every text key has followed since MS2.
    /// An exported-but-unfilled variable in a CI script is an everyday accident, and answering it
    /// with an "invalid value" the operator cannot act on punishes the accident instead of
    /// guiding it. On a seat that declares a model, absent means the error above.
    #[test]
    fn a_blank_lineage_on_a_declaring_seat_is_absent_not_invalid() {
        let cfg = MagiSectionConfig {
            caspar_model: Some("deepseek-v4-pro:cloud".into()),
            caspar_lineage: Some("   ".into()),
            ..MagiSectionConfig::default()
        };
        let err = cfg
            .lineage_of_seat(AgentName::Caspar)
            .expect_err("blank must error as MISSING");
        assert!(
            matches!(
                err,
                LineageError::Missing {
                    key: "caspar_lineage"
                }
            ),
            "blank must be MISSING, not an invalid value: {err:?}"
        );
    }

    /// A declared lineage wins over the built-in one even when the seat takes the backend model:
    /// the operator who bothered to label a seat is the authority on what its failure domain is.
    #[test]
    fn a_declared_lineage_wins_over_the_built_in_one() {
        let cfg = MagiSectionConfig {
            balthasar_lineage: Some("  mi-linaje-raro  ".into()),
            ..MagiSectionConfig::default()
        };
        assert_eq!(
            cfg.lineage_of_seat(AgentName::Balthasar)
                .expect("a declared lineage must resolve")
                .as_str(),
            "mi-linaje-raro",
            "the declared label wins, trimmed"
        );
    }

    /// SC-R44 THROUGH THE REAL LOAD PATH: a single-lineage trio does not parse.
    ///
    /// This is the WIRING guardian, and it is the one that was missing. `validate_diversity` had
    /// unit tests from the day it was written and **no production caller for the whole
    /// milestone**, so the requirement was not delivered at all while its tests were green and
    /// the CHANGELOG announced it as a breaking change. A test that calls the function directly
    /// can never catch that; only one that drives `from_toml_str` can.
    #[test]
    fn a_single_lineage_trio_fails_to_load_under_the_default() {
        let err = MagiConfig::from_toml_str(
            "provider = \"ollama\"\n\
             [magi]\n\
             melchior_model    = \"m1\"\nmelchior_lineage  = \"anthropic\"\n\
             balthasar_model   = \"m2\"\nbalthasar_lineage = \"anthropic\"\n\
             caspar_model      = \"m3\"\ncaspar_lineage    = \"anthropic\"\n",
        )
        .expect_err("three seats on one lineage must not load under enforce_diversity = true");
        let msg = err.to_string();
        assert!(
            msg.contains("melchior") && msg.contains("balthasar") && msg.contains("caspar"),
            "the error must name the affected seats: {msg}"
        );
        assert!(
            msg.contains("enforce_diversity"),
            "and carry the ONE-LINE way out: {msg}"
        );
    }

    /// SC-R43 through the load path: a pool that leaves a seat uncovered is an error under the
    /// default, and a config that declares `enforce_diversity = false` loads.
    #[test]
    fn an_uncovered_seat_fails_under_the_default_and_loads_when_disabled() {
        let toml = |extra: &str| {
            format!(
                "provider = \"ollama\"\n\
                 [magi]\n\
                 melchior_model    = \"m1\"\nmelchior_lineage  = \"opus\"\n\
                 balthasar_model   = \"m2\"\nbalthasar_lineage = \"sonnet\"\n\
                 caspar_model      = \"m3\"\ncaspar_lineage    = \"haiku\"\n\
                 {extra}\
                 [[magi.fallback]]\n\
                 model   = \"rescue\"\nlineage = \"opus\"\n"
            )
        };
        assert!(
            MagiConfig::from_toml_str(&toml("")).is_err(),
            "an `opus` candidate covers only Melchior; the other two have nowhere to rotate"
        );
        assert!(
            MagiConfig::from_toml_str(&toml("enforce_diversity = false\n")).is_ok(),
            "the mono-provider exit must still load"
        );
    }

    /// The DEFAULT install still loads. Nothing about REQ-R29 may cost a fresh clone its start:
    /// an absent `magi.toml` is a silent default, and a user who declared no trio has asserted
    /// nothing about diversity to be wrong about.
    #[test]
    fn a_config_that_declares_no_trio_still_loads() {
        assert!(MagiConfig::from_toml_str("provider = \"ollama\"\n").is_ok());
        assert!(MagiConfig::from_toml_str("").is_ok());
    }

    /// The hand-off the plan left open, answered: three seats that resolve to the SAME model are
    /// reported, whatever their labels say.
    ///
    /// `seats()` falls an undeclared seat back to the BACKEND model, so a config naming no trio
    /// runs one model under three built-in labels — literally the case SC-R44 rejects, wearing
    /// three names. A distinctness check over labels approves it, which is why the resolved
    /// models are checked separately.
    ///
    /// It is a NOTICE and not an error on purpose: an absent file must start silently, and
    /// pointing at a single-model endpoint is a legitimate choice rather than a mistake.
    #[test]
    fn three_seats_resolving_to_one_model_are_reported_however_they_are_labelled() {
        let cfg = MagiConfig::from_toml_str("provider = \"ollama\"\n").expect("defaults load");
        let notices = cfg.diversity_notices(&cfg.magi().seats("one-backend-model"));
        assert_eq!(
            notices.len(),
            1,
            "the collapse must be reported: {notices:?}"
        );
        assert!(
            notices[0].text.contains("one-backend-model"),
            "and name the model they all resolve to: {}",
            notices[0].text
        );

        let declared = MagiConfig::from_toml_str(
            "provider = \"ollama\"\n\
             [magi]\n\
             melchior_model    = \"a\"\nmelchior_lineage  = \"la\"\n\
             balthasar_model   = \"b\"\nbalthasar_lineage = \"lb\"\n\
             caspar_model      = \"c\"\ncaspar_lineage    = \"lc\"\n",
        )
        .expect("a declared trio loads");
        assert!(
            declared
                .diversity_notices(&declared.magi().seats("unused"))
                .is_empty(),
            "three distinct models say nothing"
        );
    }

    /// SC-R37: a loose `[magi]` key placed AFTER the first `[[magi.fallback]]` does NOT parse
    /// silently into the wrong table — it lands inside the pool entry and `deny_unknown_fields`
    /// rejects it, naming the key.
    ///
    /// This is why the convention is "the pool goes last in the FILE" rather than "last in the
    /// `[magi]` block": in TOML every loose key must precede the first array of tables, so a key
    /// added to `[magi]` later could end up here.
    ///
    /// **What this does NOT claim.** A SUB-TABLE after the pool — `[embedding]`, say — is
    /// perfectly valid TOML and lands where it should: a table header closes the preceding array
    /// of tables. There is nothing to detect there, and demanding an error would mean rejecting a
    /// correct file. The convention still stands as guidance; only this half is enforceable.
    #[test]
    fn a_loose_magi_key_after_the_pool_is_rejected_by_name() {
        let err = MagiConfig::from_toml_str(
            "provider = \"ollama\"\n\
             [magi]\n\
             melchior_model    = \"m\"\nmelchior_lineage  = \"a\"\n\
             balthasar_model   = \"b\"\nbalthasar_lineage = \"b\"\n\
             caspar_model      = \"c\"\ncaspar_lineage    = \"c\"\n\
             [[magi.fallback]]\n\
             model   = \"x\"\nlineage = \"x\"\n\
             max_rotations = 2\n",
        )
        .expect_err("a loose key after the pool must not be accepted");
        assert!(
            err.to_string().contains("max_rotations"),
            "the error must name the misplaced key: {err}"
        );
    }

    /// And the half that must KEEP working: a sub-table after the pool is valid TOML and parses
    /// where it belongs.
    #[test]
    fn a_sub_table_after_the_pool_is_valid_and_parses_where_it_belongs() {
        let cfg = MagiConfig::from_toml_str(
            "provider = \"ollama\"\n\
             [magi]\n\
             melchior_model    = \"m\"\nmelchior_lineage  = \"a\"\n\
             balthasar_model   = \"b\"\nbalthasar_lineage = \"b\"\n\
             caspar_model      = \"c\"\ncaspar_lineage    = \"c\"\n\
             [[magi.fallback]]\n\
             model   = \"x\"\nlineage = \"x\"\n\
             [embedding]\n\
             model = \"nomic-embed-text\"\n",
        )
        .expect("a table header closes the array: this is correct TOML");
        assert_eq!(cfg.fallback_pool().len(), 1);
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
    /// `detect_migrations` (which reports nothing while no pattern set is declared, and when one
    /// is reloaded must keep requiring structural validity rather than a textual match — see
    /// `migrate.rs`'s module docs) hold TOGETHER on the real `from_toml_str` path — this is the
    /// combination the scenario actually describes, not either property in isolation. What sits
    /// on the offending line is `[password]`, never a secret, so a `line`-citing error is safe by
    /// construction.
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
    /// valid TOML that declares zero things. `from_toml_str("")` was already covered separately;
    /// this is the only thing missing: `load()` end-to-end
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

    /// REQ-R05 / D-R14: magi-rs ships the crate's own `DEFAULT_MAX_ROTATIONS`, not one of its own.
    #[test]
    fn max_rotations_defaults_to_two() {
        let cfg = MagiConfig::from_toml_str(
            "[magi]
",
        )
        .unwrap();
        assert_eq!(cfg.effective_max_rotations(), 2);
    }

    /// A declared `0` is the kill-switch and must stay REACHABLE. Collapsing it with
    /// `unwrap_or_default()` would turn an explicit "no rotation" into "use the default", which is
    /// the opposite of what the operator wrote.
    #[test]
    fn a_declared_zero_is_the_kill_switch_not_an_absent_value() {
        let cfg = MagiConfig::from_toml_str(
            "[magi]
max_rotations = 0
",
        )
        .unwrap();
        assert_eq!(cfg.effective_max_rotations(), 0);
    }

    /// REQ-R11: the guard defaults to FALSE, Ollama included. The case it bites is the cold start,
    /// which is transitory and is the first run anyone makes.
    #[test]
    fn strict_context_guard_defaults_to_false() {
        let cfg = MagiConfig::from_toml_str(
            "[magi]
",
        )
        .unwrap();
        assert!(!cfg.declared_strict_context_guard());
        let on = MagiConfig::from_toml_str(
            "[magi]
strict_context_guard = true
",
        )
        .unwrap();
        assert!(on.declared_strict_context_guard());
    }

    /// REQ-R29: diversity is REQUIRED by default. A pool without diversity that starts silently is
    /// a safety net the operator believes they have and do not.
    #[test]
    fn enforce_diversity_defaults_to_true() {
        let cfg = MagiConfig::from_toml_str(
            "[magi]
",
        )
        .unwrap();
        assert!(cfg.effective_enforce_diversity());
        let off = MagiConfig::from_toml_str(
            "[magi]
enforce_diversity = false
",
        )
        .unwrap();
        assert!(!off.effective_enforce_diversity());
    }

    /// REQ-R13: the pool parses from `[[magi.fallback]]`, in declared order, and is empty when the
    /// array is absent.
    #[test]
    fn the_fallback_pool_parses_in_declared_order() {
        let cfg = MagiConfig::from_toml_str(
            "[magi]
",
        )
        .unwrap();
        assert!(
            cfg.fallback_pool().is_empty(),
            "absent array means no rotation, not an error"
        );

        let cfg = MagiConfig::from_toml_str(concat!(
            "[magi]
",
            "[[magi.fallback]]
model = \"glm-5.2:cloud\"
lineage = \"zhipu\"
",
            "[[magi.fallback]]
model = \"minimax-m3:cloud\"
lineage = \"minimax\"
",
        ))
        .unwrap();
        let pool = cfg.fallback_pool();
        assert_eq!(pool.len(), 2);
        assert_eq!(
            pool[0].model, "glm-5.2:cloud",
            "order is rotation preference, strongest first"
        );
        assert_eq!(pool[1].lineage.as_str(), "minimax");
    }

    /// SC-R11: a misspelled key in the rotation section is an explicit parse error, never silently
    /// ignored. `deny_unknown_fields` gives this for free — the test PINS it, because the new keys
    /// were added to a struct where dropping the attribute would still compile.
    #[test]
    fn a_misspelled_rotation_key_is_a_parse_error() {
        let err = MagiConfig::from_toml_str(
            "[magi]
max_rotation = 2
",
        )
        .expect_err("an unknown key must be rejected, never ignored");
        assert!(
            err.to_string().contains("max_rotation"),
            "the error must name the key: {err}"
        );
    }
}
