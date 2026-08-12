#![forbid(unsafe_code)]

mod agent;
mod config;
mod defaults;
mod headless_runner;
mod memory;
mod services;
mod system;
mod task;
mod tools;
mod tui;

use crate::agent::provider::{build_openai_provider, AnthropicProvider, Provider, StaticProvider};
use crate::agent::Agent;
// NOTE: this `MagiConfig` is the magi-rs TOML config (`crate::config::MagiConfig`).
// It is DISTINCT from `magi_core::orchestrator::MagiConfig` — the latter is NEVER
// imported here, avoiding the name collision.
use crate::config::{
    non_blank, resolve_anthropic_model, resolve_effective_provider_kind, resolve_magi_override,
    resolve_openai_model, HeadlessConfig, MagiConfig,
};
use crate::headless_runner::{
    resolve_tier_timeout_default, run_consult, run_query, MagiRuntimeParams,
};
use crate::memory::clock::SystemClock;
use crate::memory::embedding::OpenAiCompatibleEmbedder;
use crate::memory::store::SqliteVectorStore;
use crate::system::cached_probe::CachedProbe;
use crate::system::database::{EncryptedSqliteMemory, MemoryStore};
use crate::system::fs::{FileSystem, RealFileSystem};
use crate::system::grep::RipGrep;
use crate::system::model_cache::ModelCapabilityCache;
use crate::system::workspace::Workspace;
use crate::tools::bash::BashTool;
use crate::tools::grep::GrepTool;
use crate::tools::knowledge::ProjectFactTool;
use crate::tools::ls::ListTool;
use crate::tools::read::FileReadTool;
use crate::tools::write::FileWriteTool;
use clap::Parser;
use cryptovault::CryptoVault;
use magi_core::error::ProviderError;
use magi_core::orchestrator::{Magi, MagiBuilder};
use magi_core::provider::{LlmProvider, RetryConfig, RetryProvider};
use magi_core::providers::claude::ClaudeProvider;
use magi_core::providers::ollama::OllamaProvider;
use magi_core::providers::openai_compat::OpenAiCompatibleProvider;
use magi_core::rotation::{FallbackPool, Lineage as CoreLineage, ProviderProbe};
use magi_core::schema::{AgentName, Mode};
use magi_rs::headless::exit::exit_code as headless_exit_code;
use magi_rs::headless::input::{parse_input, read_input_bounded, InputFormat};
use magi_rs::headless::limits::{HeadlessLimits, NORMAL_MAX_TOOL_CALLS};
use magi_rs::headless::log::{LogLevel, RunLog};
use magi_rs::headless::output::{write_json, write_text};
use magi_rs::headless::policy::{Policy, Tier};
use magi_rs::headless::resolution::{
    resolve as resolve_params, CliOverrides, ConfigDefaults, Resolved,
};
use magi_rs::headless::types::{ErrorKind, RunOutcome, StopReason};
use magi_rs::headless::HeadlessError;
use magi_rs::magi::endpoint::{EndpointTemplate, ResolvedEndpoint, Scope};
use magi_rs::magi::kind::{ProviderKind, ProviderKindParseError};
use magi_rs::magi::lineage::LineageError;
use magi_rs::magi::probe::{
    assumed_window_notices, derive_input_warn_tokens, derive_warn_tokens, effective_strict_guard,
    min_mage_window, missing_model_notices, probe_models, Measurement, OllamaProbeFactory,
    ProbeFactory,
};
use magi_rs::magi::rotation_config::corroborate_by_digest;
use magi_rs::magi::{
    bytes_to_tokens_est, derive_client_timeout, derive_operation_budget, AGENT_TIMEOUT_SECS,
    CHARS_PER_TOKEN_EST, STALE_NOTICE_RATIO,
};
use magi_rs::notices::{render_notices, Notice};
use magi_rs::redact::{redact_foreign_error, redact_foreign_text, redact_url, SafeErrorText};
use magi_rs::vault::{
    check_strength, create_passphrase, diagnose, format_diagnose_report, harden_process,
    rekey_envelope, resolve_passphrase, run_vault_cmd, strip_trailing_newline, wire,
    PassphrasePrompt, SecretEntry, SecretStore, TtyIo, TtyPrompt, VaultCmd, VaultError,
    PASSPHRASE_ENV,
};
use std::collections::BTreeMap;
use std::env;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use zeroize::Zeroizing;

/// A [`SecretStore`] shared between the passphrase-resolution/API-key
/// discovery code above and the TUI's `/login`/`/logout` handlers
/// (`tui::run_tui_ext`'s `secret_store` parameter).
///
/// `VaultStore` (the concrete type behind this trait object) is `Send` but
/// deliberately **not** `Sync` (its mask rotates on every access, MAGI run 8
/// — see `src/vault/store.rs`); the `Mutex` supplies the exclusion, which is
/// exactly what `Mutex<T>: Sync` requires of its `T: Send`.
type SharedSecretStore = Arc<Mutex<dyn SecretStore + Send>>;

/// A passphrase read from `-p`/`--passphrase`, wrapped in [`Zeroizing`] so the
/// plaintext never lingers in memory after use (secrets-separation invariant).
///
/// Its [`Debug`] is redacted so the secret can never leak through the derived
/// `Debug` of [`Args`]; it `Deref`s to `str` so existing accessors
/// (`args.passphrase.as_deref()`) keep working unchanged.
#[derive(Clone)]
struct SecretArg(Zeroizing<String>);

impl std::ops::Deref for SecretArg {
    type Target = str;
    fn deref(&self) -> &str {
        self.0.as_str()
    }
}

impl std::fmt::Debug for SecretArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SecretArg(<redacted>)")
    }
}

/// clap value-parser for [`SecretArg`]: wraps the raw CLI string in
/// [`Zeroizing`]. Infallible — any string is a syntactically valid passphrase.
fn parse_secret_arg(s: &str) -> Result<SecretArg, std::convert::Infallible> {
    Ok(SecretArg(Zeroizing::new(s.to_string())))
}

/// Message shown when `--init-config` is used (REQ-A22): the flag is retired, and this
/// names the replacement instead of doing anything else. Mirrors
/// `tui::init_config_retired_message` for the TUI `/init-config` slash command.
///
/// **Why this exists instead of a clap value-parser (fix round 2, coordinator,
/// 2026-08-03, m4/m5).** An earlier version kept `init_config` as an
/// always-failing-value-parser `Option<String>` specifically so clap itself would
/// reject the flag. That backfired: clap's own rejection renders `error: invalid
/// value 'retired' for '--init-config <INIT_CONFIG>': ...` — a message whose entire
/// purpose is to not make the user think, opening with a synthetic token (`retired`)
/// they never typed. `init_config` is a plain `bool` again; `run` checks it and prints
/// THIS message, before any other startup work, then exits.
fn init_config_retired_message() -> String {
    "`--init-config` was retired; run `magi init` instead.".to_string()
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Log out: removes ANTHROPIC_API_KEY from the vault.
    #[arg(short, long)]
    logout: bool,

    /// RETIRED (REQ-A22): `magi init` is the only scaffolder now. A plain, hidden
    /// `bool` — `run` checks it FIRST, before any other startup work, and prints
    /// [`init_config_retired_message`] instead of doing anything else. Not a clap
    /// `value_parser` trick (fix round 2, coordinator, 2026-08-03, m4/m5): that
    /// rendered clap's own synthetic `error: invalid value 'retired' for
    /// '--init-config <INIT_CONFIG>': ...`, defeating the point of a message meant to
    /// not make the user think. Still hidden from `--help` (`hide = true`) so the
    /// retired flag doesn't invite new use — the flag shipped for three releases and
    /// is documented in `CLAUDE.md`, so silently un-recognizing it would turn a
    /// one-line migration into a search.
    #[arg(long, hide = true)]
    init_config: bool,

    /// Master passphrase (precedence: -p > MAGI_PASSPHRASE > interactive
    /// prompt). Global: also applies to the `vault` subcommand (REQ-V04).
    #[arg(short = 'p', long, global = true, value_parser = parse_secret_arg)]
    passphrase: Option<SecretArg>,

    #[command(subcommand)]
    command: Option<TopCmd>,
}

/// Input format selector for `--input-format` (REQ-H04); maps to the library's
/// [`InputFormat`] (auto-detect when the flag is absent).
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum CliInputFormat {
    /// Treat the whole input as the prompt verbatim (never parsed as JSON).
    Text,
    /// Parse the input as a JSON envelope object.
    Json,
}

impl CliInputFormat {
    /// Maps this CLI selector to the library [`InputFormat`].
    fn into_lib(self) -> InputFormat {
        match self {
            CliInputFormat::Text => InputFormat::Text,
            CliInputFormat::Json => InputFormat::Json,
        }
    }
}

/// Output format selector for `--output-format` (REQ-H04); text (default) or a
/// single buffered JSON object.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum CliOutputFormat {
    /// The response text only (streamed to stdout / buffered to `-o`).
    Text,
    /// A single rich JSON object (buffered).
    Json,
}

/// Log-verbosity selector for `--log-level` (REQ-H24); maps to [`LogLevel`].
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum CliLogLevel {
    /// Only irrecoverable failures.
    Error,
    /// Recoverable but noteworthy conditions.
    Warn,
    /// Normal operational notices (default).
    Info,
    /// Maximum verbosity (redacted, capped tool inputs).
    Debug,
}

impl CliLogLevel {
    /// Maps this CLI selector to the library [`LogLevel`].
    fn into_lib(self) -> LogLevel {
        match self {
            CliLogLevel::Error => LogLevel::Error,
            CliLogLevel::Warn => LogLevel::Warn,
            CliLogLevel::Info => LogLevel::Info,
            CliLogLevel::Debug => LogLevel::Debug,
        }
    }
}

/// Mode selector for `--mode` (REQ-A07).
///
/// Defect #12 (registered plan debt): verified against `clap_derive-4.6.4`'s
/// `ValueEnum` derive (`DEFAULT_CASING = CasingStyle::Kebab`, `src/item.rs`)
/// rather than assumed — the default kebab-casing of these three variant
/// names already produces `code-review`/`design`/`analysis`, exactly the
/// vocabulary [`crate::magi::mode::normalize_label`] accepts, so no
/// `#[value(name = "...")]` override is needed. A test below
/// (`cli_mode_casing_matches_the_shared_mode_vocabulary`) pins this so a
/// future clap upgrade that changes the default can't silently desync the
/// two vocabularies.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum CliMode {
    /// Code review: correctness, security, edge cases.
    CodeReview,
    /// Design: architecture, approach selection.
    Design,
    /// General analysis (default).
    Analysis,
}

impl CliMode {
    /// Converts the clap variant to magi-core's `Mode`.
    ///
    /// Explicit conversion, not a generic `From` or `#[serde(into)]`: these are two
    /// enums with different owners — clap owns the CLI vocabulary, magi-core owns the
    /// domain one — and they can diverge one day. The exhaustive `match` is what turns
    /// that divergence into a compile error instead of a silent mistranslation.
    ///
    /// Consumed in production by `run_consult_subcommand` (Task 2.3, REQ-A07c): the
    /// direct `magi consult` path resolves its explicit `--mode` through this before
    /// falling back to classification. Also covered by
    /// `every_surface_accepts_an_explicit_mode` and
    /// `cli_mode_casing_matches_the_shared_mode_vocabulary`.
    #[must_use]
    fn into_mode(self) -> Mode {
        match self {
            Self::CodeReview => Mode::CodeReview,
            Self::Design => Mode::Design,
            Self::Analysis => Mode::Analysis,
        }
    }
}

/// Flags shared by the `query` and `consult` headless subcommands
/// (REQ-H03/H04/H05/H07/H08/H12/H36). `--consult` is only meaningful for
/// `query` (`consult` already forces the multi-perspective pass); on `consult`
/// it is inert.
#[derive(clap::Args, Debug)]
struct HeadlessArgs {
    /// Read the prompt/envelope from a file; omitted ⇒ stdin (REQ-H03).
    #[arg(short = 'i', long)]
    input: Option<PathBuf>,
    /// Write the output to a file (atomic); omitted ⇒ stdout (REQ-H03).
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,
    /// Force the input interpretation; omitted ⇒ auto-detect (REQ-H04).
    #[arg(long, value_enum)]
    input_format: Option<CliInputFormat>,
    /// Output format; omitted ⇒ text (REQ-H04).
    #[arg(long, value_enum)]
    output_format: Option<CliOutputFormat>,
    /// Working directory: file-tool sandbox root and `.magi/` walk-up base
    /// (default cwd, REQ-H05).
    #[arg(short = 'w', long)]
    workdir: Option<PathBuf>,
    /// Stateless: do not persist session/history/memories (REQ-H18).
    #[arg(long)]
    no_memory: bool,
    /// Auto-approve all registered tools, hard barriers intact (REQ-H07).
    #[arg(long)]
    auto: bool,
    /// Like `--auto` plus elevated caps and silenced soft guards (REQ-H08).
    #[arg(long)]
    full_auto: bool,
    /// Wall-clock ceiling in seconds for the whole run (REQ-H36).
    #[arg(long)]
    timeout: Option<u64>,
    /// Run-log verbosity; omitted ⇒ info (REQ-H24).
    #[arg(long, value_enum)]
    log_level: Option<CliLogLevel>,
    /// Override the run-log directory (default `.magi/logs`, REQ-H24).
    #[arg(long)]
    log_dir: Option<PathBuf>,
    /// Honor an envelope `system` override (default: ignore it, REQ-H12b).
    #[arg(long)]
    allow_system_override: bool,
    /// Refuse to overwrite the `-o` output file if it already exists (REQ-H03).
    #[arg(long)]
    no_clobber: bool,
    /// Force exactly one MAGI multi-perspective pass in `query` (REQ-H22).
    #[arg(long)]
    consult: bool,
    /// Per-request model override (wins over the envelope, REQ-H12).
    #[arg(long)]
    model: Option<String>,
    /// Per-request provider override (wins over the envelope, REQ-H12).
    #[arg(long)]
    provider: Option<String>,
    /// Per-request tool-call cap; the operator flag can RAISE the ceiling
    /// (REQ-H08/H12b).
    #[arg(long)]
    max_tool_calls: Option<u32>,
    /// Consult lens; omitted ⇒ INFERRED with an extra model call (REQ-A07c).
    ///
    /// Declaring it avoids that call. See also `[magi].default_mode`, which fixes
    /// the lens for every invocation without touching any call site.
    #[arg(long, value_enum)]
    mode: Option<CliMode>,
    /// Declares that the content under analysis is NOT trustworthy (REQ-A07d).
    ///
    /// With this flag set, omitting `--mode` is an ERROR instead of an inference:
    /// no path lets the content itself steer the lens it is reviewed under.
    #[arg(long)]
    untrusted_content: bool,
}

impl Args {
    /// Extracts the `--mode` declared on whichever headless subcommand this parse
    /// holds (`query` or `consult`); `None` if there is no subcommand, or the
    /// subcommand carries no `--mode` (REQ-A07).
    ///
    /// Consumed in production by `run()`, which reads it **before**
    /// `args.command.take()` (that `.take()` would otherwise empty `self.command`
    /// and make this always return `None`) and forwards the result into
    /// `run_consult_subcommand` as `explicit_mode`. Also covered by
    /// `every_surface_accepts_an_explicit_mode` and
    /// `cli_mode_casing_matches_the_shared_mode_vocabulary`.
    fn mode_of_consult(&self) -> Option<Mode> {
        match &self.command {
            Some(TopCmd::Query(h)) | Some(TopCmd::Consult(h)) => h.mode.map(CliMode::into_mode),
            _ => None,
        }
    }

    /// `true` if `--untrusted-content` was declared on whichever headless subcommand
    /// this parse holds; `false` if there is no subcommand (REQ-A07d).
    ///
    /// Production reads the flag straight off `HeadlessArgs.untrusted_content`
    /// once the subcommand is already destructured out of `args.command`
    /// (`run_consult_subcommand`, which OR-s it with the envelope's own
    /// `untrusted_content` field and `[magi].untrusted_content` — any one
    /// surface can raise the guard, REQ-A07d) — unlike `mode_of_consult`, there
    /// is no need to read this one before `.take()`. This top-level accessor
    /// stays as the CLI-parsing-level assertion surface: covered by
    /// `untrusted_content_is_declarable_where_the_threat_lives` and
    /// `mode_and_untrusted_content_are_absent_without_a_subcommand`.
    /// **`#[cfg(test)]`, not `#[allow(dead_code)]`** (S3 Loop 2, Balthasar). It carried the
    /// `allow` and a comment explaining that production reads the field directly — which is a
    /// tested-only accessor sitting in production code, telling the linter something untrue about
    /// the crate. §6.1.8 is explicit that an `#[allow]` must never be fabricated to quiet a gate.
    /// Compiling it only under `cfg(test)` says the same thing honestly: this exists for the two
    /// tests that pin the CLI-parsing property, and for nothing else.
    #[cfg(test)]
    fn untrusted_content(&self) -> bool {
        match &self.command {
            Some(TopCmd::Query(h)) | Some(TopCmd::Consult(h)) => h.untrusted_content,
            _ => false,
        }
    }
}

/// Top-level subcommands beyond the default TUI launch.
///
/// MS2 extends this same enum with `Query`/`Consult`; keep new variants
/// self-contained so that addition stays mechanical.
#[derive(clap::Subcommand, Debug)]
enum TopCmd {
    /// Encrypted, zero-knowledge secret store (`ls`/`set`/`rm`/`passwd`).
    #[command(subcommand)]
    Vault(VaultCmd),

    /// Scaffold a fresh `.magi/` state directory in the working directory
    /// (config, encrypted DB, logs); refuses to nest or overwrite (REQ-H01).
    Init,

    /// Run the agent headless over a prompt with structured I/O (REQ-H02).
    Query(HeadlessArgs),

    /// Force a MAGI multi-perspective analysis over the prompt (REQ-H02/H21).
    Consult(HeadlessArgs),
}

#[derive(Debug)]
struct Config {
    api_key: String,
    model: String,
    source: String,
}

/// Default Anthropic model when none is configured via `ANTHROPIC_MODEL`.
/// Single source of truth lives in `crate::defaults`; this alias keeps
/// existing call sites and tests working unchanged.
pub(crate) use crate::defaults::DEFAULT_ANTHROPIC_MODEL as DEFAULT_MODEL;

/// The four host strings `reqwest::Url::host_str` can return for a local
/// address (MAGI re-gate hardening, Caspar/Melchior): the bare loopback
/// forms and the bracketed IPv6 form the `url` crate actually emits for
/// `[::1]` hosts. Compared for an EXACT match, never a substring.
const LOCAL_HOSTS: [&str; 4] = ["localhost", "127.0.0.1", "::1", "[::1]"];

/// Returns `true` when `base_url` resolves to a local address (`localhost`,
/// `127.0.0.1`, or `::1`/`[::1]`) — used by the cloud-egress notice
/// (CP2-AG/AJ, Task 13b) to suppress the warning for on-device embedders.
///
/// Parses `base_url` and compares the **host** component exactly against
/// [`LOCAL_HOSTS`] (never a substring match — a prior substring-based
/// version false-matched a hostname like `notlocalhost.evil.com`, silently
/// suppressing the warning). A `base_url` whose host cannot be parsed is
/// treated as **non-local**: for a security-egress notice, erring toward
/// showing the warning is the safe direction (fail-safe).
fn is_localhost(base_url: &str) -> bool {
    reqwest::Url::parse(base_url)
        .ok()
        .and_then(|url| url.host_str().map(|host| LOCAL_HOSTS.contains(&host)))
        .unwrap_or(false)
}

/// Whether the discovered workspace has a `.magi/magi.toml` on disk — the
/// canonical config-file existence check feeding
/// `crate::defaults::should_emit_default_notice` (REQ-H16/H17). A `None`
/// workspace (no `.magi/` discovered at all) means no config file exists.
///
/// Fixes a MAGI re-gate finding: the prior call site checked the legacy
/// loose `<cwd>/magi.toml` path (D-H07 retired that layout) instead of the
/// canonical `.magi/magi.toml`, so the "no magi.toml — using Ollama
/// defaults" startup notice fired/suppressed based on the wrong file.
///
/// The `.exists()` here carries no path-traversal risk: `ws` is the workspace
/// already produced by the hardened `crate::system::workspace::discover`, which
/// rejects a symlinked/junction `.magi` component and confines to the operator's
/// `-w`/cwd tree (REQ-H30). `config_path()` merely joins the fixed
/// `magi.toml` leaf to that validated, operator-controlled `.magi/` dir — no
/// caller/attacker-supplied path component reaches this probe.
fn magi_toml_exists(workspace: Option<&Workspace>) -> bool {
    workspace.is_some_and(|ws| ws.config_path().exists())
}

/// Resolves the `ANTHROPIC_API_KEY` config: the **consumed environment value**
/// first, then the vault (REQ-V12/H12) — the OS keyring and `key.txt` are no
/// longer consulted at all (REQ-V37).
///
/// `env_key` is the `ANTHROPIC_API_KEY` value that was read out of the process
/// environment (and scrubbed) at startup by [`read_then_scrub_secret_env`]
/// (REQ-H37): sourcing it from there — rather than re-reading `env::var` — keeps
/// the `env > vault` precedence intact even after the live env var is gone.
/// `secret_store` is `None` for an ephemeral (no-persistence) session, in which
/// case only the consumed environment value is consulted.
///
/// The model is resolved via [`resolve_anthropic_model`] (`env > TOML >
/// default`, MAGI re-gate fix): `config` is the already-loaded `magi.toml`, so
/// an `[anthropic] model` there is honored instead of being silently ignored.
fn discover_config(
    config: &MagiConfig,
    env_key: Option<&str>,
    secret_store: Option<&SharedSecretStore>,
) -> Option<Config> {
    let model = resolve_anthropic_model(config, env::var("ANTHROPIC_MODEL").ok().as_deref());
    // REQ-A12 / MS2 gate S8 seventh-pass finding: a blank/whitespace-only env value is ABSENT,
    // never a literal key — an `ANTHROPIC_API_KEY=""` exported empty in a CI script must fall
    // through to the vault instead of short-circuiting it with an empty key that will 401.
    if let Some(key) = non_blank(env_key) {
        return Some(Config {
            api_key: key.trim().to_string(),
            model,
            source: "ENV".to_string(),
        });
    }
    let ss = secret_store?;
    let mut guard = ss.lock().unwrap_or_else(|p| p.into_inner());
    let key = guard.get("ANTHROPIC_API_KEY").ok()?;
    Some(Config {
        // Trim like the env path: a stored/exported key with stray whitespace or
        // a trailing newline would otherwise produce a malformed auth header (401).
        api_key: key.as_str().trim().to_string(),
        model,
        source: "vault".to_string(),
    })
}

/// Resolves the `OPENAI_API_KEY` used by the OpenAI-compatible chat provider
/// and the embedder: the **consumed environment value** first, then the vault
/// (REQ-V12/H12), mirroring [`discover_config`]'s precedence for the Anthropic
/// key.
///
/// `env_key` is the value read out of (and scrubbed from) the process
/// environment at startup ([`read_then_scrub_secret_env`], REQ-H37); consulting
/// it — rather than re-reading `env::var` — preserves `env > vault` after the
/// live env var is gone. Both sources are trimmed for the same reason
/// `discover_config` trims: a key with stray whitespace or a trailing newline (a
/// common `export KEY=$(cat f)` artifact) would otherwise produce a malformed
/// `Authorization` header (401).
fn resolve_openai_key(
    env_key: Option<&str>,
    secret_store: Option<&SharedSecretStore>,
) -> Option<String> {
    // REQ-A12 / MS2 gate S8 seventh-pass finding: same absent-not-invalid rule as
    // `discover_config` — a blank `OPENAI_API_KEY` falls through to the vault.
    if let Some(key) = non_blank(env_key) {
        return Some(key.trim().to_string());
    }
    let ss = secret_store?;
    let mut guard = ss.lock().unwrap_or_else(|p| p.into_inner());
    guard
        .get("OPENAI_API_KEY")
        .ok()
        .map(|z| z.as_str().trim().to_string())
}

/// Environment variable name holding the Anthropic API key (REQ-H37).
const ANTHROPIC_KEY_ENV: &str = "ANTHROPIC_API_KEY";

/// Environment variable name holding the OpenAI-compatible API key (REQ-H37).
const OPENAI_KEY_ENV: &str = "OPENAI_API_KEY";

/// Warning emitted at startup when loose legacy state files sit in the working
/// directory but no `.magi/` exists (REQ-H17/H31). The pre-headless layout is no
/// longer read or migrated — the user must run `magi init` to adopt the unified
/// `.magi/` state directory.
const LEGACY_LAYOUT_WARNING: &str =
    "warning: found a legacy .magi-rs-memory.db/magi.toml loose in this directory; \
     the pre-.magi/ layout is no longer used — run `magi init` to create a .magi/ \
     state directory (the legacy files are not read or migrated)";

/// Secrets read out of the process environment before it is scrubbed (REQ-H37).
///
/// The passphrase is wrapped in [`Zeroizing`] so its memory is wiped on drop
/// (the secrets-separation invariant); the API keys are plain `String`s,
/// matching how the rest of `main.rs` already carries them.
///
/// The fields are consumed by [`bootstrap_headless`]/[`run`]: after the scrub
/// the live env vars are gone, so config and passphrase resolution source these
/// captured values instead of `env::var` (REQ-H37).
struct ConsumedSecrets {
    /// The master passphrase (`MAGI_PASSPHRASE`), if it was set.
    passphrase: Option<Zeroizing<String>>,
    /// The Anthropic API key (`ANTHROPIC_API_KEY`), if it was set.
    anthropic_key: Option<String>,
    /// The OpenAI-compatible API key (`OPENAI_API_KEY`), if it was set.
    openai_key: Option<String>,
}

/// Reads the three secret env vars into a [`ConsumedSecrets`], then removes all
/// three from the process environment (REQ-H37).
///
/// The removal is **symmetric** — passphrase *and* both API keys — so that a
/// later in-workspace interpreter (`python`/`node`) cannot exfiltrate them by
/// reading `/proc/<pid>/environ`.
///
/// # Safety of call site
///
/// This calls [`std::env::remove_var`], which is **undefined behaviour** once
/// the process has spawned additional threads (a concurrent env read races the
/// removal). It **must** therefore be invoked single-threaded at startup,
/// **before** the multi-thread tokio runtime spawns any worker.
/// [`bootstrap_headless`] enforces that call site (it invokes this as its first
/// statement, before building the runtime); this function only provides the
/// mechanism.
fn read_then_scrub_secret_env() -> ConsumedSecrets {
    let passphrase = env::var(PASSPHRASE_ENV).ok().map(Zeroizing::new);
    let anthropic_key = env::var(ANTHROPIC_KEY_ENV).ok();
    let openai_key = env::var(OPENAI_KEY_ENV).ok();
    env::remove_var(PASSPHRASE_ENV);
    env::remove_var(ANTHROPIC_KEY_ENV);
    env::remove_var(OPENAI_KEY_ENV);
    ConsumedSecrets {
        passphrase,
        anthropic_key,
        openai_key,
    }
}

/// Outcome of resolving the vault passphrase and opening the encrypted store
/// for the TUI (`main.rs`'s resequencing, REQ-V04/V06/V17/V35).
enum MemoryAttachment {
    /// The passphrase resolved (or was freshly created) and the encrypted
    /// store opened successfully.
    Encrypted(EncryptedSqliteMemory),
    /// No usable passphrase (unavailable, aborted, or retries exhausted), or
    /// the store could not be opened for a reason other than a wrong
    /// passphrase. The session runs without persistence rather than ever
    /// falling back to a constant/synthesized passphrase (audit finding C4,
    /// preserved from the pre-vault design).
    Ephemeral,
}

/// Resolves the passphrase that opens (or bootstraps) the encrypted store.
///
/// `db_absent` selects between two policies: **present** ⇒
/// [`resolve_passphrase`] (`-p`/env-flag > prompt, single entry, REQ-V04);
/// **absent** (first run) ⇒ if `passphrase_flag` already supplies a value it is
/// used directly after [`check_strength`] (nothing to confirm it against);
/// otherwise, with a TTY, [`create_passphrase`] runs the double-entry +
/// zero-knowledge-warning flow (REQ-V17); without a TTY and without a flag,
/// fails closed with [`VaultError::PassphraseUnavailable`] rather than hanging
/// on a prompt that cannot be read (REQ-H25 / REQ-V40's fail-closed spirit,
/// applied to bootstrap).
///
/// `passphrase_flag` already folds the `-p` CLI flag and the (scrubbed,
/// consumed) `MAGI_PASSPHRASE` value together with `-p` winning (REQ-H37): after
/// the startup env scrub there is no live env var left to read here.
///
/// # Errors
/// [`VaultError::PassphraseUnavailable`] as described above;
/// [`VaultError::WeakPassphrase`] if a directly-supplied first-run value
/// does not meet the strength floor; [`VaultError::Io`] on a terminal read
/// failure.
fn resolve_master_passphrase(
    db_absent: bool,
    passphrase_flag: Option<Zeroizing<String>>,
    prompt: &mut dyn PassphrasePrompt,
) -> Result<Zeroizing<String>, VaultError> {
    if !db_absent {
        return resolve_passphrase(passphrase_flag, prompt);
    }
    if let Some(p) = passphrase_flag {
        // Normalize identically to unlock (resolve_passphrase) so a passphrase
        // created with a trailing newline can be reproduced on unlock.
        let p = strip_trailing_newline(p);
        check_strength(p.as_str())?;
        return Ok(p);
    }
    // First run with no `-p`/`MAGI_PASSPHRASE`: create interactively, or fail
    // closed when there is no TTY (never hang on an unreadable prompt).
    if !prompt.is_interactive() {
        return Err(VaultError::PassphraseUnavailable);
    }
    create_passphrase(prompt, false)
}

/// Whether an [`EncryptedSqliteMemory::new`] failure was specifically
/// [`VaultError::WrongPassphrase`] (vs. some other failure that should not
/// trigger a retry prompt). Relies on `database.rs::map_open_err` preserving
/// the original `VaultError` behind the `anyhow::Error` (via `.into()`, not
/// a formatted string) so it can be recovered with `downcast_ref`.
fn is_wrong_passphrase(e: &anyhow::Error) -> bool {
    matches!(
        e.downcast_ref::<VaultError>(),
        Some(VaultError::WrongPassphrase)
    )
}

/// Resolves the passphrase and opens the encrypted store for the TUI,
/// retrying on [`VaultError::WrongPassphrase`] (SC-V09) since a long-lived
/// interactive session makes retry the normal case.
///
/// After the *first* failed attempt, retries go straight to the interactive
/// prompt — never back through `-p`/env — or a wrong `-p`/`MAGI_PASSPHRASE`
/// value would loop forever with no way out. A non-interactive session (no
/// TTY to retry against) or an empty retry entry (user cancelled) degrades
/// to [`MemoryAttachment::Ephemeral`] with a `notices` entry explaining why
/// (decision of plan 1). Never wipes: no branch of this function deletes
/// `db_path` (REQ-V35) — a wrong passphrase only ever produces a retryable
/// error.
/// Wraps low-level warnings — [`harden_process`]'s best-effort hardening failures and
/// `MaskedDek::warnings`'s mlock diagnostics — as `Resolution` [`Notice`]s.
///
/// Never `Info`: an mlock/dump-suppression failure is a security-posture regression,
/// not diagnostic noise, so it must survive [`magi_rs::notices::NOTICE_MAX_INFO`]'s
/// cap unconditionally.
fn low_level_warning_notices(warnings: &[String]) -> Vec<Notice> {
    warnings
        .iter()
        .map(|w| Notice::resolution(format!("warning: {w}")))
        .collect()
}

/// Wraps the raw `String`s [`open_tui_memory`] and `attach_persistent_memory` push
/// into their `&mut Vec<String>` buffer as `Resolution` [`Notice`]s.
///
/// Both helpers keep the `Vec<String>` signature on purpose: they are shared with the
/// headless `query` path (REQ-H), which has no startup list to render a tier into —
/// see the `notices` module doc for why that is the correct scope boundary. `run()`
/// bridges the gap HERE, at the one place that does render a startup list.
///
/// The tier is always `Resolution`, never `Info` — deliberately conservative in ONE
/// direction. These are messages `run()` did not author (it cannot read the intent
/// behind an arbitrary string produced by a helper it calls), so a per-message
/// classification would be a guess, and the two ways to guess wrong are not
/// symmetric: defaulting to `Info` risks `NOTICE_MAX_INFO`'s cap silently dropping a
/// real warning (e.g. "running WITHOUT persistence") — exactly what the cap must
/// never do to a signal. Defaulting to `Resolution` risks a merely-diagnostic message
/// (e.g. the memory diagnostics summary) surviving the cap when it didn't need to —
/// noise, not a lost signal. Between the two failure directions, this picks the one
/// that costs nothing important.
fn wrap_helper_notices(texts: Vec<String>) -> Vec<Notice> {
    texts.into_iter().map(Notice::resolution).collect()
}

/// The "no persistence at all" warning shown when the TUI reaches the end of the
/// memory-attach sequence with no store to attach.
///
/// Extracted (rather than inlined at its one call site) so the test pinning that this
/// exact message tiers above `Info` exercises the real call site, not a copy of it.
fn no_persistence_notice() -> Notice {
    Notice::resolution(
        "WARNING: this session runs WITHOUT persistence — your conversation and \
         project knowledge will NOT be saved (any existing on-disk database is left \
         untouched). Provide the vault passphrase (-p, MAGI_PASSPHRASE, or the \
         interactive prompt) to restore persistence."
            .to_string(),
    )
}

fn open_tui_memory(
    db_path: &std::path::Path,
    passphrase_flag: Option<Zeroizing<String>>,
    prompt: &mut dyn PassphrasePrompt,
    notices: &mut Vec<String>,
) -> MemoryAttachment {
    let db_absent = !db_path.exists();
    let mut passphrase = match resolve_master_passphrase(db_absent, passphrase_flag, prompt) {
        Ok(p) => p,
        Err(e) => {
            // Surface the SPECIFIC reason (e.g. a rejected weak passphrase vs. no
            // passphrase available) so the user knows why the session degraded.
            // VaultError's Display never contains the passphrase (verified).
            notices.push(format!(
                "WARNING: {e}; running WITHOUT persistence for this session (any \
                 existing on-disk database is left untouched)."
            ));
            return MemoryAttachment::Ephemeral;
        }
    };

    loop {
        match EncryptedSqliteMemory::new(db_path.to_path_buf(), passphrase) {
            Ok(store) => return MemoryAttachment::Encrypted(store),
            Err(e) => {
                if !is_wrong_passphrase(&e) {
                    notices.push(format!(
                        "WARNING: could not open the encrypted database ({e}); \
                         running WITHOUT persistence for this session."
                    ));
                    return MemoryAttachment::Ephemeral;
                }
                if !prompt.is_interactive() {
                    notices.push(
                        "WARNING: incorrect passphrase and no interactive terminal to \
                         retry; running WITHOUT persistence for this session."
                            .to_string(),
                    );
                    return MemoryAttachment::Ephemeral;
                }
                let retry_msg = "Incorrect passphrase (if this DB predates v0.9.0's \
                    keyring, there is no migration: delete it manually to start \
                    fresh). Passphrase: ";
                match prompt.read_passphrase(retry_msg, false) {
                    Ok(p) if !p.is_empty() => passphrase = p,
                    _ => {
                        notices.push(
                            "WARNING: passphrase entry cancelled; running WITHOUT \
                             persistence for this session."
                                .to_string(),
                        );
                        return MemoryAttachment::Ephemeral;
                    }
                }
            }
        }
    }
}

/// Maps a [`VaultError`] to the CLI's process exit code.
///
/// Deliberately an exhaustive `match` with no `_` arm (MAGI run 6, Caspar):
/// adding a `VaultError` variant without assigning it an exit code here is a
/// compile error, so the mapping can never silently go stale.
fn vault_error_exit_code(e: &VaultError) -> i32 {
    match e {
        VaultError::Aborted
        | VaultError::WrongPassphrase
        | VaultError::PassphraseUnavailable
        | VaultError::WeakPassphrase(_)
        | VaultError::ValueTooLarge(_)
        | VaultError::SecretNotFound(_)
        // Data-corruption errors are runtime failures, not CLI misuse (REQ-H23
        // names `DbCorrupt` → 1 explicitly). The headless taxonomy already maps
        // both corruption variants to `HeadlessError::Db` → exit 1, so the same
        // corruption now exits the same code on the vault and headless surfaces
        // (the user action is restore/delete, never "you invoked me wrong").
        | VaultError::VaultMetaCorrupt
        | VaultError::DbCorrupt { .. } => 1,
        // Unexpected internal failures the caller cannot act on directly.
        VaultError::Crypto(_) | VaultError::Storage(_) | VaultError::Io(_) => 2,
    }
}

/// Extracts and reports the [`VaultError`] behind an
/// [`EncryptedSqliteMemory::new`] failure, printing a generic message for
/// the rare case (a raw schema anomaly) that is not one. Returns the
/// process exit code to use.
fn report_open_failure(e: &anyhow::Error) -> i32 {
    match e.downcast_ref::<VaultError>() {
        Some(ve) => {
            eprintln!("error: {ve}");
            vault_error_exit_code(ve)
        }
        None => {
            eprintln!("error: {e}");
            2
        }
    }
}

/// Discovers the unified `.magi/` state directory for a subcommand that
/// **requires** persistent state, mapping the two failure modes to a CLI exit
/// code: no `.magi/` in the ancestor chain ⇒ a clear "run `magi init`" refusal
/// (REQ-H17), a discovery error (e.g. a symlinked `.magi` component, REQ-H30) ⇒
/// its typed exit code. Legacy loose files in the cwd are never read (D-H07).
///
/// # Errors
/// Returns (via `Err`) the process exit code to use on absence (1) or on a
/// discovery error ([`headless_error_exit_code`]).
fn require_workspace(cwd: &std::path::Path) -> Result<crate::system::workspace::Workspace, i32> {
    match crate::system::workspace::discover(cwd) {
        Ok(Some(ws)) => Ok(ws),
        Ok(None) => {
            eprintln!(
                "error: no .magi/ state directory found in this directory or any \
                 parent; run `magi init` to create one"
            );
            Err(1)
        }
        Err(e) => {
            eprintln!("error: {e}");
            Err(headless_error_exit_code(&e))
        }
    }
}

/// Runs `magi-rs vault <cmd>` (short-lived process, never reaches the TUI):
/// discovers the `.magi/` state directory (REQ-H16/H17), resolves the
/// passphrase, opens the encrypted store, wires the vault, and drives
/// [`run_vault_cmd`]. Returns the process exit code.
fn run_vault_subcommand(
    cmd: VaultCmd,
    passphrase_flag: Option<Zeroizing<String>>,
    workspace_root: &std::path::Path,
    hardening_warnings: &[String],
) -> i32 {
    let ws = match require_workspace(workspace_root) {
        Ok(ws) => ws,
        Err(code) => return code,
    };
    let db_path = ws.db_path();
    let db_absent = !db_path.exists();
    let mut prompt = TtyPrompt;
    let passphrase = match resolve_master_passphrase(db_absent, passphrase_flag, &mut prompt) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            return vault_error_exit_code(&e);
        }
    };
    // A copy survives the move into `new` below; `rekey` (passwd) needs the
    // ALREADY-VERIFIED current passphrase, and `new`'s successful return is
    // exactly that verification (REQ-V20 step 1).
    let current_passphrase = Zeroizing::new(passphrase.as_str().to_string());

    let store = match EncryptedSqliteMemory::new(db_path, passphrase) {
        Ok(s) => s,
        Err(e) => return report_open_failure(&e),
    };

    let dek = match store.data_key() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: {e}");
            return vault_error_exit_code(&e);
        }
    };
    // Best-effort hardening/mlock warnings are visible here: the `vault`
    // branch never reaches the TUI's startup_notices, so stderr is the only
    // place SC-V48 can surface for the CLI.
    for w in hardening_warnings.iter().chain(dek.warnings().iter()) {
        eprintln!("warning: {w}");
    }

    let conn = store.shared_conn();
    let mut vault_store = match wire(conn.clone(), dek) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}");
            return vault_error_exit_code(&e);
        }
    };

    let mut rekey = |new: &str| -> Result<(), VaultError> {
        rekey_envelope(
            &CryptoVault::default(),
            &conn,
            current_passphrase.as_str(),
            new,
        )
    };

    let mut io = TtyIo::new();
    match run_vault_cmd(cmd, &mut vault_store, &mut io, &mut rekey) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("error: {e}");
            vault_error_exit_code(&e)
        }
    }
}

/// Resolves the DB path `magi vault diagnose` inspects: the `.magi/`
/// discovered by [`crate::system::workspace::discover`] (nearest ancestor,
/// REQ-H16) when one exists, otherwise the legacy `workspace_root`-relative
/// path [`run_vault_subcommand`] and the TUI still use. A discovery failure
/// (e.g. a symlinked `.magi` component, REQ-H30) is surfaced rather than
/// silently falling back, since that failure is itself security-relevant.
///
/// # Errors
/// Propagates [`magi_rs::headless::HeadlessError`] from `discover`.
fn resolve_diagnose_db_path(
    workspace_root: &std::path::Path,
) -> Result<std::path::PathBuf, magi_rs::headless::HeadlessError> {
    match crate::system::workspace::discover(workspace_root)? {
        Some(ws) => Ok(ws.db_path()),
        None => Ok(workspace_root.join(".magi-rs-memory.db")),
    }
}

/// Runs `magi-rs vault diagnose`: a **read-only** structural probe (REQ-H32)
/// that never unlocks the vault and never requires a passphrase — it is
/// intercepted in [`main`] BEFORE [`run_vault_subcommand`]'s passphrase
/// resolution ever runs. Returns the process exit code.
///
/// An absent DB is reported (not an error): there is nothing to diagnose yet.
fn run_vault_diagnose(workspace_root: &std::path::Path, names: bool) -> i32 {
    let db_path = match resolve_diagnose_db_path(workspace_root) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            return headless_error_exit_code(&e);
        }
    };
    if !db_path.exists() {
        println!("no database found at {}", db_path.display());
        return 0;
    }
    let conn = match rusqlite::Connection::open_with_flags(
        &db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    ) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: storage error: {e}");
            return vault_error_exit_code(&VaultError::Storage(e.to_string()));
        }
    };
    match diagnose(&conn, names) {
        Ok(report) => {
            for line in format_diagnose_report(&report) {
                println!("{line}");
            }
            0
        }
        Err(e) => {
            eprintln!("error: {e}");
            vault_error_exit_code(&e)
        }
    }
}

/// Runs `magi-rs --logout`: opens the vault and removes `ANTHROPIC_API_KEY`
/// (the CLI analogue of SC-V36/SC-V37). An absent DB, or an absent key, is
/// reported as "no stored session" rather than an error. Returns the
/// process exit code.
fn run_logout(passphrase_flag: Option<Zeroizing<String>>, workspace_root: &std::path::Path) -> i32 {
    // Route through `.magi/` discovery (REQ-H16/H17): no `.magi/` (or a `.magi/`
    // with no DB yet) means there is nothing to log out of — reported as "no
    // stored session", not an error. Legacy loose files are never read.
    let db_path = match crate::system::workspace::discover(workspace_root) {
        Ok(Some(ws)) => ws.db_path(),
        Ok(None) => {
            println!("no stored session");
            return 0;
        }
        Err(e) => {
            eprintln!("error: {e}");
            return headless_error_exit_code(&e);
        }
    };
    if !db_path.exists() {
        println!("no stored session");
        return 0;
    }
    let mut prompt = TtyPrompt;
    let passphrase = match resolve_passphrase(passphrase_flag, &mut prompt) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            return vault_error_exit_code(&e);
        }
    };
    let store = match EncryptedSqliteMemory::new(db_path, passphrase) {
        Ok(s) => s,
        Err(e) => return report_open_failure(&e),
    };
    let dek = match store.data_key() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: {e}");
            return vault_error_exit_code(&e);
        }
    };
    let mut vault_store = match wire(store.shared_conn(), dek) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}");
            return vault_error_exit_code(&e);
        }
    };
    match vault_store.remove("ANTHROPIC_API_KEY") {
        Ok(()) => {
            println!("Logged out successfully.");
            0
        }
        Err(VaultError::SecretNotFound(_)) => {
            println!("no stored session");
            0
        }
        Err(e) => {
            eprintln!("error: {e}");
            vault_error_exit_code(&e)
        }
    }
}

/// Maps a [`magi_rs::headless::HeadlessError`] to the CLI's process exit code,
/// mirroring the headless exit taxonomy (REQ-H23): input/misuse ⇒ 2, every
/// other class ⇒ 1.
///
/// The library's `headless::exit::exit_code` is `pub(crate)` and thus
/// unreachable from this bin crate, so the `magi init` edges map directly here;
/// the exhaustive `match` (no `_` arm) keeps the two taxonomies from drifting —
/// a new variant breaks the build instead of defaulting silently.
fn headless_error_exit_code(e: &magi_rs::headless::HeadlessError) -> i32 {
    use magi_rs::headless::HeadlessError;
    match e {
        HeadlessError::InputInvalid(_) | HeadlessError::InputTooLarge(_) => 2,
        HeadlessError::Io(_)
        | HeadlessError::Storage(_)
        | HeadlessError::Aborted
        | HeadlessError::PassphraseUnavailable
        | HeadlessError::Db(_) => 1,
    }
}

/// Resolves the passphrase used to bootstrap the vault envelope during
/// `magi init`, **never** prompting interactively (§2.2). `passphrase_flag`
/// already folds `-p` and the consumed `MAGI_PASSPHRASE` (`-p` wins, REQ-H37);
/// if it is absent the DB is left envelope-less (`Ok(None)`) — the documented
/// "first interactive run creates it" behavior, not an error. A supplied value
/// is normalized (trailing newline stripped, so a later unlock reproduces the
/// KEK) and must clear the strength floor ([`check_strength`], REQ-V17).
///
/// # Errors
/// [`VaultError::WeakPassphrase`] if a supplied passphrase is below the floor.
fn resolve_init_passphrase(
    passphrase_flag: Option<Zeroizing<String>>,
) -> Result<Option<Zeroizing<String>>, VaultError> {
    if let Some(p) = passphrase_flag {
        let p = strip_trailing_newline(p);
        check_strength(p.as_str())?;
        return Ok(Some(p));
    }
    Ok(None)
}

/// Runs `magi-rs init`: scaffolds a fresh `.magi/` state directory under `cwd`
/// and, when a passphrase is available non-interactively, bootstraps the empty
/// vault envelope so headless runs need no interaction (§2.2). Returns the
/// process exit code (0 on success; non-zero on refusal or error).
///
/// Refuses (never destroys state) in three cases: a discoverable `.magi/`
/// already exists in an ancestor (would nest — Step 4b), the walk-up aborts on
/// a symlinked component (propagated, never treated as "clean" — Step 4c), or a
/// `.magi/` already exists in `cwd` ([`workspace::init`] returns `Aborted`).
/// Diagnostics go to stderr; the created path is the sole stdout line.
fn run_init(cwd: &std::path::Path, passphrase_flag: Option<Zeroizing<String>>) -> i32 {
    // Nested-`.magi/` guard: handle ALL THREE `discover` branches explicitly —
    // a walk aborted by a symlink is NOT "no ancestor found" and must never
    // fall through to init (Step 4b/4c).
    match crate::system::workspace::discover(cwd) {
        Ok(Some(ws)) => {
            eprintln!(
                "error: an existing .magi/ was found at {}; use it, or run \
                 `magi init` from a different root (refusing to nest a second \
                 .magi/ inside it)",
                ws.magi_dir.display()
            );
            return 1;
        }
        Ok(None) => {}
        Err(e) => {
            eprintln!("error: {e}");
            return headless_error_exit_code(&e);
        }
    }

    // Resolve the bootstrap passphrase BEFORE touching the filesystem so a weak
    // `-p`/env value fails fast without leaving a half-created `.magi/`.
    let bootstrap_passphrase = match resolve_init_passphrase(passphrase_flag) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            return vault_error_exit_code(&e);
        }
    };

    let ws = match crate::system::workspace::init(cwd) {
        Ok(ws) => ws,
        Err(e) => {
            eprintln!("error: {e}");
            return headless_error_exit_code(&e);
        }
    };
    // stdout carries only the created state directory (diagnostics ⇒ stderr).
    println!("{}", ws.magi_dir.display());

    // §2.2: bootstrap the envelope when a passphrase was supplied; otherwise
    // leave the DB envelope-less for the first interactive run to create.
    if let Some(passphrase) = bootstrap_passphrase {
        if let Err(e) = crate::system::database::EncryptedSqliteMemory::open_with_state_machine(
            ws.db_path(),
            passphrase,
        ) {
            eprintln!("error: {e}");
            return vault_error_exit_code(&e);
        }
    }
    0
}

/// Converts a subcommand's legacy `i32` exit status (always `0..=3` for this
/// CLI) into a process [`ExitCode`], clamping any out-of-range value to a
/// generic failure so the conversion can never panic.
fn exit_code(code: i32) -> ExitCode {
    ExitCode::from(u8::try_from(code).unwrap_or(1))
}

/// Startup ordering harness (REQ-H37, anti-UB): reads and **scrubs** the secret
/// environment variables single-threaded — as its very first statement, before
/// any tokio runtime exists — then builds the multi-thread runtime and drives
/// `body` on it, handing the captured [`ConsumedSecrets`] in.
///
/// Encapsulating scrub → build → `block_on` here makes the ordering invariant
/// **structural**: [`std::env::remove_var`] is undefined behaviour once worker
/// threads exist, so the only way to move the scrub after the runtime is built
/// would be to rewrite this function's body. `main` therefore does nothing but
/// call this.
///
/// A runtime-build failure, or an error returned by `body`, is reported to
/// stderr and mapped to [`ExitCode::FAILURE`]; a successful `body` returns its
/// own [`ExitCode`].
fn bootstrap_headless<F, Fut>(body: F) -> ExitCode
where
    F: FnOnce(ConsumedSecrets) -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<ExitCode>>,
{
    // FIRST: capture + scrub the secret env single-threaded, before the runtime
    // spawns any worker (`remove_var` is UB under concurrency — REQ-H37).
    let secrets = read_then_scrub_secret_env();
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("error: failed to build the async runtime: {e}");
            return ExitCode::FAILURE;
        }
    };
    match runtime.block_on(body(secrets)) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn main() -> ExitCode {
    bootstrap_headless(run)
}

/// The async application body, driven on the multi-thread runtime built by
/// [`bootstrap_headless`]. `secrets` carries the passphrase and API keys read
/// out of (and removed from) the process environment at startup (REQ-H37):
/// every downstream resolution sources them from here, never from a now-scrubbed
/// `env::var`.
///
/// # Errors
/// Propagates any fatal I/O, configuration, or TUI error (mapped to
/// [`ExitCode::FAILURE`] by [`bootstrap_headless`]).
async fn run(secrets: ConsumedSecrets) -> anyhow::Result<ExitCode> {
    let ConsumedSecrets {
        passphrase: env_passphrase,
        anthropic_key,
        openai_key,
    } = secrets;

    let mut args = Args::parse();

    // REQ-A22 (fix round 2, coordinator, 2026-08-03): `--init-config` is retired.
    // Checked FIRST, before ANY other startup work (workspace/legacy-layout
    // detection, process hardening, subcommand dispatch, `-p`/`--logout`) — the
    // whole point of this message is to be the only thing the user sees.
    if args.init_config {
        eprintln!("{}", init_config_retired_message());
        return Ok(ExitCode::FAILURE);
    }

    // Fold the `-p` CLI flag and the consumed `MAGI_PASSPHRASE` into ONE flag,
    // `-p` winning (precedence `-p` > `MAGI_PASSPHRASE`, REQ-H37): after the env
    // scrub there is no live env var left, so the captured value carries the env
    // tier. An empty env value counts as absent (matches `resolve_passphrase`).
    // Own it as `Zeroizing` (REQ-V41): the only bare copy is clap's own field,
    // dropped right after this `.take()`. `-p` itself stays visible in `argv` by
    // design (REQ-V04) — only the *value* leaving argv unzeroized is in scope.
    let passphrase_flag: Option<Zeroizing<String>> = args
        .passphrase
        .take()
        .map(|s| s.0)
        .or_else(|| env_passphrase.filter(|p| !p.is_empty()));
    let workspace_root = env::current_dir()?;

    // REQ-H31: loose legacy state with no `.magi/` ⇒ a visible stderr warning
    // (detect only — never read or migrate the legacy files, D-H07).
    if crate::system::workspace::detect_legacy_files(&workspace_root) {
        eprintln!("{LEGACY_LAYOUT_WARNING}");
    }

    // REQ-V42: best-effort process hardening, once, before any secret
    // material exists.
    let hardening_warnings = harden_process();

    // Read BEFORE `args.command.take()` below empties `args.command` — after
    // that point `mode_of_consult()` would always answer `None` (REQ-A07c).
    let explicit_consult_mode = args.mode_of_consult();

    match args.command.take() {
        // REQ-H32: intercepted BEFORE `run_vault_subcommand` so a diagnose
        // never resolves a passphrase or opens/unlocks the vault.
        Some(TopCmd::Vault(VaultCmd::Diagnose { names })) => {
            return Ok(exit_code(run_vault_diagnose(&workspace_root, names)));
        }
        Some(TopCmd::Vault(cmd)) => {
            return Ok(exit_code(run_vault_subcommand(
                cmd,
                passphrase_flag,
                &workspace_root,
                &hardening_warnings,
            )));
        }
        Some(TopCmd::Init) => {
            return Ok(exit_code(run_init(&workspace_root, passphrase_flag)));
        }
        Some(TopCmd::Query(h)) => {
            return Ok(exit_code(
                run_query_subcommand(
                    h,
                    passphrase_flag,
                    &workspace_root,
                    anthropic_key,
                    openai_key,
                )
                .await,
            ));
        }
        Some(TopCmd::Consult(h)) => {
            return Ok(exit_code(
                run_consult_subcommand(
                    h,
                    explicit_consult_mode,
                    passphrase_flag,
                    &workspace_root,
                    anthropic_key,
                    openai_key,
                )
                .await,
            ));
        }
        // No subcommand ⇒ fall through to the TUI launch below.
        None => {}
    }

    if args.logout {
        return Ok(exit_code(run_logout(passphrase_flag, &workspace_root)));
    }

    // ── TUI path ─────────────────────────────────────────────────────────
    // Task 1.5: every source below pushes a `Notice`, never a bare `String` — the
    // tier decides both print ORDER (`Blocking` first) and whether `NOTICE_MAX_INFO`
    // can ever trim it. `open_tui_memory` and `attach_persistent_memory` keep their
    // `&mut Vec<String>` signature: they are shared with the headless `query` path
    // (REQ-H), which has no startup list to render a tier into (see `notices`
    // module doc for why that boundary is deliberate, not unfinished). `run()`
    // bridges the gap at the one place that DOES render a startup list, via
    // `wrap_helper_notices`.
    let mut startup_notices: Vec<Notice> = low_level_warning_notices(&hardening_warnings);

    // Discover the unified `.magi/` state directory (walk-up, nearest ancestor,
    // REQ-H16). A discovery error degrades to no-persistence with a notice
    // (never crashes the TUI); a missing `.magi/` runs ephemeral and hints
    // `magi init` (REQ-H17) — the legacy cwd-relative DB/config are never read.
    let workspace = match crate::system::workspace::discover(&workspace_root) {
        Ok(ws) => ws,
        Err(e) => {
            startup_notices.push(Notice::resolution(format!(
                "WARNING: could not resolve the .magi/ state directory ({e}); \
                 running WITHOUT persistence for this session."
            )));
            None
        }
    };

    let mut prompt = TtyPrompt;
    let mut open_memory_notices: Vec<String> = Vec::new();
    let attachment = match workspace.as_ref() {
        Some(ws) => open_tui_memory(
            &ws.db_path(),
            passphrase_flag,
            &mut prompt,
            &mut open_memory_notices,
        ),
        None => {
            startup_notices.push(Notice::resolution(
                "WARNING: no .magi/ state directory found — running WITHOUT \
                 persistence. Run `magi init` to create one and enable saved \
                 history (any existing on-disk database is left untouched)."
                    .to_string(),
            ));
            MemoryAttachment::Ephemeral
        }
    };
    startup_notices.extend(wrap_helper_notices(open_memory_notices));

    let (memory_store, secret_store): (Option<EncryptedSqliteMemory>, Option<SharedSecretStore>) =
        match attachment {
            MemoryAttachment::Encrypted(store) => match store.data_key() {
                Ok(dek) => {
                    startup_notices.extend(low_level_warning_notices(dek.warnings()));
                    match wire(store.shared_conn(), dek) {
                        Ok(vstore) => (
                            Some(store),
                            Some(Arc::new(Mutex::new(vstore)) as SharedSecretStore),
                        ),
                        Err(e) => {
                            startup_notices.push(Notice::resolution(format!(
                                "WARNING: could not open the secret vault ({e}); \
                                 ANTHROPIC_API_KEY/OPENAI_API_KEY must come from the \
                                 environment this session."
                            )));
                            (Some(store), None)
                        }
                    }
                }
                Err(e) => {
                    startup_notices.push(Notice::resolution(format!(
                        "WARNING: could not derive the vault key ({e}); \
                         ANTHROPIC_API_KEY/OPENAI_API_KEY must come from the \
                         environment this session."
                    )));
                    (Some(store), None)
                }
            },
            MemoryAttachment::Ephemeral => (None, None),
        };

    // REQ-R25: the capability cache rides on the SAME encrypted database as the conversation
    // history — its own connection would mean a second file and a second key. Absent on the
    // ephemeral path, and a failure to open it DEGRADES: measurements stop being remembered
    // between runs, which is slower, not wrong.
    let capability_cache: Option<Arc<ModelCapabilityCache>> =
        open_capability_cache(memory_store.as_ref(), &mut startup_notices);

    // Config lives in `.magi/magi.toml` (REQ-H16/H17): load it from the
    // discovered `.magi/`, or fall back to built-in defaults when none exists —
    // the legacy loose cwd `magi.toml` is never read. Loaded BEFORE
    // `discover_config` below so the resolved `[anthropic].model` can be honored
    // there (MAGI re-gate fix) instead of discover_config reading env-only.
    //
    // Task 1.4/REQ-A23: `load` is now fallible — a present-but-broken magi.toml (bad
    // parse, unknown vocabulary, or a literal credential in any `base_url`) TERMINATES
    // the process via `?` instead of silently degrading to defaults, which is exactly
    // what v0.11.0 did and what SC-A16d/REQ-A23 forbid now.
    let (magi_config, config_notices) = match workspace.as_ref() {
        Some(ws) => MagiConfig::load(&ws.config_path())?,
        None => (MagiConfig::default(), Vec::new()),
    };
    // REQ-V12/H12: API key discovery happens AFTER the vault is (possibly) open,
    // env tier sourced from the consumed (scrubbed) value.
    let config = discover_config(
        &magi_config,
        anthropic_key.as_deref(),
        secret_store.as_ref(),
    );
    // Task 4.1: replaces `resolve_provider`/`legacy_backend_label` — the vocabulary is
    // unified now, so there is nothing left to normalize a raw `ProviderKind` onto.
    let provider_kind =
        resolve_effective_provider_kind(&magi_config, env::var("MAGI_PROVIDER").ok().as_deref())
            .map_err(|e| anyhow::anyhow!("{e}"))?;

    // Task 4.1: no longer carries a `model_label` third element — that was only ever
    // read by the retired adapter-naming machinery (`MagiCoreProviderAdapter`'s display
    // name); the native trio resolves its OWN per-seat models from `magi_config`
    // directly (`build_magi_orchestrator`), independent of the principal's model.
    let (provider, provider_info): (Arc<dyn Provider>, String) = match provider_kind {
        // `Ollama` and `OpenAiCompat` share the `[openai]`-transport branch: they
        // speak the same Chat-Completions protocol and differ only in capability
        // (probeability), never in how the PRINCIPAL provider is built. REQ-R30
        // changed which type builds the TRIO's seats; this is the conversational
        // agent's provider and that milestone does not touch it.
        ProviderKind::Ollama | ProviderKind::OpenAiCompat => {
            // env > vault (REQ-V12); falls back to the local-Ollama dummy so a
            // real OpenAI/Groq/OpenRouter endpoint still fails loudly with 401
            // rather than silently defaulting to an insecure constant.
            let api_key = resolve_openai_key(openai_key.as_deref(), secret_store.as_ref())
                .unwrap_or_else(|| "ollama".to_string());
            // Fix round 3 (L1/L2/S1): resolves blank-is-absent + vault credentials,
            // never the raw template — see `resolve_effective_principal_endpoint`'s
            // own doc for what this replaces and why.
            let resolved_base_url = resolve_effective_principal_endpoint(
                &magi_config,
                env::var("OPENAI_BASE_URL").ok().as_deref(),
                secret_store.as_ref(),
            )
            .map_err(|e| anyhow::anyhow!("{e}"))?;
            let base_url = resolved_base_url.as_str().to_string();
            let model =
                resolve_openai_model(&magi_config, env::var("OPENAI_MODEL").ok().as_deref());
            let info = openai_provider_info(&base_url, &model);
            (build_openai_provider(&base_url, &api_key, &model), info)
        }
        ProviderKind::Anthropic => {
            if let Some(ref c) = config {
                (
                    Arc::new(AnthropicProvider::new(c.api_key.clone(), c.model.clone())),
                    format!("Magi API ({}) Model: {}", c.source, c.model),
                )
            } else {
                (
                    Arc::new(StaticProvider),
                    "Static Mode: no API key found. Set ANTHROPIC_API_KEY or run \
                     `magi-rs vault set ANTHROPIC_API_KEY` (recommended). /login \
                     (OAuth) is best-effort and may be rate-limited."
                        .to_string(),
                )
            }
        }
    };

    // Notices shown when the TUI starts — the provider banner plus any persistence,
    // reset, or vault warnings that would otherwise be lost to pre-TUI stderr.
    startup_notices.push(Notice::info(provider_info));
    // REQ-A12b/A12c: notices for resolutions that didn't come straight from what was
    // written in `magi.toml` (a blank `provider`, an inherited non-default embedder
    // endpoint, an Anthropic/base_url incoherence) — surfaced the same way the old
    // malformed-config warning used to be, but `load()` no longer needs a *malformed*
    // branch here: that path is now fatal and propagated by the `?` above.
    // `config_notices` stays `Vec<String>` (produced by `config::resolution_notices`,
    // out of this task's file list) — tiered `Resolution` here, which by name IS what
    // these are: "the config resolved differently than the file appears to say."
    startup_notices.extend(config_notices.into_iter().map(Notice::resolution));
    // B1: surface invalid memory-config values as a startup notice (never panic).
    if let Err(e) = magi_config.memory().validate() {
        startup_notices.push(Notice::resolution(format!("memory config warning: {e}")));
    }
    // H2: surface invalid embedding-config values alongside memory-config (never panic).
    if let Err(e) = magi_config.embedding().validate() {
        startup_notices.push(Notice::resolution(format!("embedding config warning: {e}")));
    }
    // RF-9: when there is no magi.toml at all, make the Ollama-first default visible
    // (never-silent). A present-but-minimal magi.toml does NOT trigger this.
    if crate::defaults::should_emit_default_notice(
        provider_kind,
        magi_toml_exists(workspace.as_ref()),
    ) {
        startup_notices.push(Notice::info(crate::defaults::no_config_notice()));
    }

    // Build the MAGI trio with magi-core's NATIVE providers (REQ-A01) — independent
    // of the principal provider's own availability: a trio using a keyless `ollama`
    // kind can be perfectly buildable even when the principal fell back to
    // `StaticProvider` for lack of an Anthropic key, and vice versa. Task 4.3 owns the
    // polished per-surface unavailable-trio behavior (REQ-A06, `trio_unavailable_message`,
    // conditional tool registration); this keeps the failure typed and non-silent (B9)
    // without pre-empting that task's contract.
    let endpoints = resolve_endpoints(
        &magi_config,
        env::var("OPENAI_BASE_URL").ok().as_deref(),
        secret_store.as_ref(),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    let creds = EnvVaultCredentials {
        magi_config: &magi_config,
        anthropic_env: anthropic_key.as_deref(),
        openai_env: openai_key.as_deref(),
        secret_store: secret_store.as_ref(),
    };

    // Read ONCE and shared by the probe AND the builder below (sixth-pass gate finding, S8):
    // the probe must measure exactly the models `build_magi_orchestrator` is about to run,
    // which requires the SAME `MagiEnvModelOverrides`, not two independently-constructed ones.
    let env_overrides = MagiEnvModelOverrides::from_env();

    // REQ-A24/A24b/A24c (Task 5.2): measure the principal and the trio BEFORE building it, so
    // `input_warn_tokens` can be derived from the MAGES window (REQ-A24b) and startup announces
    // the three measurement states (REQ-A24c). Never blocks or fails startup: each probe fails
    // open inside `probe_models`/`orchestrate_probes`.
    let (warn_tokens, measured) = probe_and_report(
        &magi_config,
        &endpoints,
        provider_kind,
        &OllamaProbeFactory,
        &env_overrides,
        &stateless_extra_models(&magi_config, capability_cache.as_ref()),
        &mut startup_notices,
    )
    .await;

    // Task 4.3 (REQ-A06/SC-A06b): the startup notice and the response a future `/consult` will
    // see share the SAME text, built once right here — `trio_unavailable_for_tui` is what makes
    // that equality verifiable instead of relying on this site and `run_tui_ext` constructing
    // the same `String` independently.
    let mut consult_unavailable_message: Option<String> = None;
    let consult_magi: Option<Arc<Magi>> = match build_magi_orchestrator(
        &TrioBuild {
            cfg: &magi_config,
            principal_kind: provider_kind,
            endpoints: &endpoints,
            creds: Some(&creds),
            warn_tokens,
            env_overrides: &env_overrides,
            capability_cache: capability_cache.as_ref(),
            measured: &measured,
        },
        &mut startup_notices,
    ) {
        Ok(magi) => Some(magi),
        Err(e) => {
            let (notice, msg) = trio_unavailable_for_tui(&e);
            startup_notices.push(notice);
            consult_unavailable_message = Some(msg);
            None
        }
    };

    // Cloned BEFORE `Agent::new(provider)` consumes it below — same pattern as
    // `run_consult_subcommand`'s own classifier construction. REQ-A07d: the TUI's
    // explicit `/consult` needs the same `resolve_mode_guarded` classifier/config
    // pair the direct headless `magi consult` path already has.
    let (tui_mode_classifier, tui_classifier_notices) =
        tui_mode_classifier_wiring(provider.clone());
    let tui_default_mode = magi_config.effective_default_mode();
    let tui_untrusted_content = magi_config.magi().untrusted_content.unwrap_or(false);
    // REQ-A07p/SC-A07p: `tui_default_mode.is_none()` IS "will this session infer the
    // mode" — the same signal `divergence_notice` needs, already computed above; reusing
    // it (instead of calling `effective_default_mode()` a second time) is B3, not just
    // convenience.
    push_divergence_notice(
        &magi_config,
        tui_default_mode.is_none(),
        &mut startup_notices,
    );

    let mut agent = Agent::new(provider);

    match memory_store {
        Some(concrete_store) => {
            // Wire persistence + the tiered-memory subsystem (shared with the
            // headless `query` path, DRY). The embedding key is resolved here
            // (env > vault) so the helper stays free of the secret-store plumbing.
            let embed_key = resolve_openai_key(openai_key.as_deref(), secret_store.as_ref());
            let mut attach_notices: Vec<String> = Vec::new();
            attach_persistent_memory(
                &mut agent,
                concrete_store,
                &magi_config,
                embed_key,
                secret_store.as_ref(),
                &mut attach_notices,
            )
            .await?;
            startup_notices.extend(wrap_helper_notices(attach_notices));
        }
        None => {
            // #7: surface the no-persistence state in the TUI, not just pre-TUI stderr.
            startup_notices.push(no_persistence_notice());
        }
    }

    let fs: Arc<dyn FileSystem> = Arc::new(RealFileSystem::new());
    agent.register_tool(Box::new(ListTool::new(fs.clone(), workspace_root.clone())?));
    agent.register_tool(Box::new(FileReadTool::new(
        fs.clone(),
        workspace_root.clone(),
    )?));
    agent.register_tool(Box::new(FileWriteTool::new(
        fs.clone(),
        workspace_root.clone(),
    )?));
    agent.register_tool(Box::new(GrepTool::new(
        Box::new(RipGrep::new("rg")),
        workspace_root.clone(),
    )?));
    agent.register_tool(Box::new(BashTool::new(workspace_root.clone())?));
    register_consult_tool_if_available(
        &mut agent,
        consult_magi.as_ref(),
        magi_config.magi().auto_approve,
        registered_magi_kind(&magi_config, provider_kind),
        magi_config.magi_endpoint_diverges(),
        magi_config.effective_max_query_bytes(),
        magi_config.effective_tool_result_cap(),
    );

    crate::tui::run_tui_ext(
        agent,
        render_notices(startup_notices),
        crate::tui::TuiConsultWiring {
            consult: consult_magi,
            consult_unavailable_message,
            magi_auto_approve: magi_config.magi().auto_approve,
            // M1 fix: threads `[magi].agent_timeout_secs` through to the
            // post-`/login` trio rebuild, which used to hardcode the
            // built-in default regardless of this config.
            agent_timeout_secs: magi_config.magi().agent_timeout_secs,
        },
        secret_store,
        crate::tui::TuiMagiRuntimeConfig {
            mode_classifier: tui_mode_classifier,
            classifier_notices: tui_classifier_notices,
            default_mode: tui_default_mode,
            untrusted_content: tui_untrusted_content,
            magi_kind: registered_magi_kind(&magi_config, provider_kind),
            max_query_bytes: magi_config.effective_max_query_bytes(),
            tool_result_cap: magi_config.effective_tool_result_cap(),
        },
        // The chat loop's SELF-ROUTED consults (REQ-A20/A07d) — a different surface from the
        // explicit `/consult` above, which is what `TuiMagiRuntimeConfig` serves.
        AutonomousRunConfig::from_magi_config(&magi_config),
    )
    .await?;
    Ok(ExitCode::SUCCESS)
}

/// Third-party credentials resolved `env > vault` (REQ-A12), reduced to what the native trio
/// needs: one API key per backend that requires it (`openai-compat`, `anthropic`). `ollama` is
/// keyless and never looks them up.
///
/// Separate from the endpoint and not redundant with it: [`ResolvedEndpoint`] can carry
/// `userinfo` (authentication of the proxy or the server serving the model), while the backend
/// API key goes in a header (`Authorization: Bearer` / `x-api-key`). Two credentials, two
/// destinations.
trait Credentials {
    /// The API key for the OpenAI-compat transport (`OPENAI_API_KEY`).
    fn openai(&self) -> Option<String>;
    /// The Anthropic API key (`ANTHROPIC_API_KEY`).
    fn anthropic(&self) -> Option<String>;
}

/// Bridge between the existing `env > vault` resolution ([`discover_config`],
/// [`resolve_openai_key`]) and the [`Credentials`] trait required by the native trio.
///
/// Reuses those two functions instead of reimplementing precedence a third time (B3):
/// `discover_config` also resolves the Anthropic model, which is discarded here — cheap, and
/// avoids yet another copy of the same four lines "trimmed env, or trimmed `vault.get(NAME)`".
struct EnvVaultCredentials<'a> {
    /// Config already loaded — `discover_config` needs it for the Anthropic model (discarded
    /// here), even though this view only asks for the key.
    magi_config: &'a MagiConfig,
    /// `ANTHROPIC_API_KEY` already read (and scrubbed) from the environment at startup.
    anthropic_env: Option<&'a str>,
    /// `OPENAI_API_KEY` already read (and scrubbed) from the environment at startup.
    openai_env: Option<&'a str>,
    /// The vault opened this session, if any.
    secret_store: Option<&'a SharedSecretStore>,
}

impl Credentials for EnvVaultCredentials<'_> {
    fn openai(&self) -> Option<String> {
        resolve_openai_key(self.openai_env, self.secret_store)
    }
    fn anthropic(&self) -> Option<String> {
        discover_config(self.magi_config, self.anthropic_env, self.secret_store).map(|c| c.api_key)
    }
}

/// The two endpoints that gate startup, resolved at once.
///
/// The symbol is born here (`main.rs`), not in `config.rs`: Task 4.1 is its first consumer
/// (ORDER-FIXES.md #7/#8 — a symbol is written in the task that first consumes it) and it needs
/// [`SharedSecretStore`]/[`NoVaultInScope`], which are already private to this file and this
/// file only — moving it to `config.rs` would force exporting them or reimplementing the same
/// "optional vault, unresolved template never reaches an HTTP client" pattern a second time.
///
/// **Deliberately does NOT include the embedding endpoint (S8 review round, finding 1).** An
/// earlier version resolved `[embedding].base_url` here too, fail-closed like the other two —
/// but that made a broken embedding config (a missing vault entry for its
/// `[user]:[password]` placeholder, say) abort the ENTIRE process via the `?` both call sites
/// (`run()`/`prepare_headless()`) apply to this function's result, even for a session that
/// never attaches persistent memory at all (an ephemeral TUI run, headless `--no-memory`).
/// Root and the trio are in play for EVERY session with a principal provider or a trio, so a
/// broken config for either is unavoidably a config problem for the current run; the embedder
/// is only in play once a vector store actually attaches, and that path
/// (`resolve_effective_embedding_endpoint`, called from `attach_persistent_memory`) already
/// resolves it and degrades gracefully to text-only persistence with a notice on failure
/// (REQ-29) — this struct must not duplicate that resolution with a stricter, unconditional
/// failure mode. See `resolve_endpoints_does_not_fail_closed_on_an_unresolvable_embedding_placeholder`.
struct ResolvedEndpoints {
    /// Root `base_url` — main agent.
    ///
    /// Task 4.1: the principal provider itself keeps resolving its own endpoint via
    /// `resolve_effective_principal_endpoint` (B3 remains pending a deliberately deferred
    /// unification). Task 5.2 adds a different real consumer to it: it is the endpoint against
    /// which `orchestrate_probes` probes the principal model (REQ-A24), so the
    /// `#[allow(dead_code)]` it had is removed here. It is resolved the same way, fail-closed,
    /// because `resolve_endpoints` is THE startup step for these endpoints at once — leaving
    /// this one out would make it two steps. Covered by
    /// `resolve_endpoints_resolves_the_two_fields_from_the_same_root_when_none_diverge`.
    root: ResolvedEndpoint,
    /// `[magi].base_url` or inheritance — the trio and its probe. The only field
    /// `build_magi_orchestrator` reads today.
    magi: ResolvedEndpoint,
}

/// The effective root template: `OPENAI_BASE_URL` (if non-blank) over what is
/// declared/inherited in `magi.toml`.
///
/// Extracted (fix round 2, I1) so that [`resolve_effective_principal_endpoint`] AND
/// [`resolve_endpoints`] apply EXACTLY the same env layer — before this fix, only the principal
/// saw it, so `OPENAI_BASE_URL` moved the conversational agent without moving the trio when
/// `[magi].base_url` was absent (inheriting).
///
/// # Errors
/// An `OPENAI_BASE_URL` or root `base_url` that is not a valid template.
fn effective_root_template(
    magi_config: &MagiConfig,
    env_base_url: Option<&str>,
) -> Result<EndpointTemplate, String> {
    match env_base_url.map(str::trim).filter(|s| !s.is_empty()) {
        Some(env_val) => EndpointTemplate::parse(env_val, Scope::Root).map_err(|e| {
            format!(
                "OPENAI_BASE_URL is invalid: {}",
                magi_rs::redact::redact_foreign_error(&e)
            )
        }),
        None => magi_config.effective_base_url().map_err(|e| {
            format!(
                "base_url is invalid: {}",
                magi_rs::redact::redact_foreign_error(&e)
            )
        }),
    }
}

/// The startup step: after opening the vault, BEFORE the probe and the trio.
///
/// Fails CLOSED on `root` and `magi`: a placeholder without an entry stops the process
/// naming the entry and the command (`magi-rs vault set …`), never substitutes empty
/// (SC-A16f) — it inherits that guarantee from [`resolve_template`].
///
/// `env_base_url` is `OPENAI_BASE_URL` — the SAME variable that already moved the principal
/// (see [`effective_root_template`]). The embedding endpoint is deliberately NOT resolved
/// here (S8 review round, finding 1) — see [`ResolvedEndpoints`]'s own doc for why: it has no
/// production consumer at this step, and its real, gracefully-degrading resolution already
/// lives in `resolve_effective_embedding_endpoint`/`attach_persistent_memory`.
///
/// # Errors
/// An already-readable message (see [`resolve_template`]) from the first unresolvable
/// `root`/`magi` endpoint.
fn resolve_endpoints(
    magi_config: &MagiConfig,
    env_base_url: Option<&str>,
    secret_store: Option<&SharedSecretStore>,
) -> Result<ResolvedEndpoints, String> {
    let root_tpl = effective_root_template(magi_config, env_base_url)?;

    // The trio inherits the SAME effective root (with its env layer) when it does not declare
    // its own — never bare `effective_magi_base_url()`, which only sees TOML.
    let magi_tpl = match magi_config
        .magi()
        .base_url
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        Some(own) => EndpointTemplate::parse(own, Scope::Magi).map_err(|e| {
            format!(
                "magi base_url is invalid: {}",
                magi_rs::redact::redact_foreign_error(&e)
            )
        })?,
        None => root_tpl.clone(),
    };

    Ok(ResolvedEndpoints {
        root: resolve_template(&root_tpl, Scope::Root, secret_store)?,
        magi: resolve_template(&magi_tpl, Scope::Magi, secret_store)?,
    })
}

/// Resolves the effective `kind` of the trio: declared, or the ALREADY-RESOLVED one from the
/// principal (`principal_kind`, NOT `cfg.effective_magi_kind()`/`cfg.effective_provider()` —
/// those two accessors are TOML-only and would ignore `MAGI_PROVIDER`).
///
/// Shared between [`build_magi_orchestrator`] (real construction) and [`orchestrate_probes`]
/// (probing, REQ-A24, Task 5.2) so that both ALWAYS see the same kind (B3): without this, a
/// `MAGI_PROVIDER` that moves the principal without declaring `[magi].kind` would make the
/// probe measure a different backend from the one the trio actually ends up using — exactly the
/// bug that the `principal_kind` parameter of `build_magi_orchestrator` already exists to avoid
/// in real construction.
///
/// # Errors
/// [`ProviderKindParseError`] if `[magi].kind` is present and not recognized.
fn resolve_magi_kind(
    cfg: &MagiConfig,
    principal_kind: ProviderKind,
) -> Result<ProviderKind, ProviderKindParseError> {
    Ok(
        ProviderKind::parse(cfg.magi().kind.as_deref().unwrap_or_default())?
            .unwrap_or(principal_kind),
    )
}

/// The trio `kind` every DOWNSTREAM CONSUMER must see (`ConsultTool::with_kind`,
/// `MagiRuntimeParams::kind`) — resolved the SAME way [`build_magi_orchestrator`]/
/// [`orchestrate_probes`] actually build the trio (MS2 gate S8 finding).
///
/// A thin wrapper over [`resolve_magi_kind`], NOT a second, divergent rule: every call site
/// that needs to report/consume the trio's active kind (the four in `run()`,
/// `run_query_subcommand`, `run_consult_subcommand`) goes through this one function instead of
/// each calling `cfg.effective_magi_kind()` directly — that accessor is TOML-only and ignores
/// `MAGI_PROVIDER`, so a `MAGI_PROVIDER` that moves the principal without declaring
/// `[magi].kind` used to make these call sites report a stale kind. That stale value reaches
/// `explain_magi_error`'s keyless-auth hint (REQ-A12c): a wrong `kind` there either adds the
/// hint when the real cause has nothing to do with keyless auth, or suppresses it when it
/// should have fired.
///
/// An unrecognized `[magi].kind` falls back to `principal_kind` instead of propagating
/// [`resolve_magi_kind`]'s typed error: a genuinely invalid `[magi].kind` already makes the
/// trio unbuildable upstream (`build_magi_orchestrator` returns `Err`), so `consult_magi` is
/// `None`/absent and no call site ever registers `ConsultTool` or builds `MagiRuntimeParams`
/// with this fallback value — it is unreachable in production, never silently wrong.
///
/// # Parameters
/// * `cfg` - the loaded `magi.toml`.
/// * `principal_kind` - the ALREADY env-resolved principal kind (`provider_kind` in `run()`,
///   `HeadlessContext::provider_kind` in the two dispatchers) — never `cfg.effective_provider()`.
///
/// # Returns
/// The kind every downstream consumer of the trio should report/use.
#[must_use]
fn registered_magi_kind(cfg: &MagiConfig, principal_kind: ProviderKind) -> ProviderKind {
    resolve_magi_kind(cfg, principal_kind).unwrap_or(principal_kind)
}

/// BACKEND model for `kind`: the one a trio seat inherits without its own override, AND the
/// model probed as the "principal model" (REQ-A24). `[openai]` serves both `ollama` AND
/// `openai-compat` because they share the completions protocol (REQ-A01b).
///
/// Shared between [`build_magi_orchestrator`] and [`orchestrate_probes`] (B3, Task 5.2) —
/// before this extraction, probing the correct model at startup would have required repeating
/// this same resolution at the call site.
fn resolve_backend_model(cfg: &MagiConfig, kind: ProviderKind) -> &str {
    match kind {
        ProviderKind::Ollama | ProviderKind::OpenAiCompat => cfg
            .openai()
            .model
            .as_deref()
            .unwrap_or(crate::defaults::DEFAULT_OPENAI_MODEL),
        ProviderKind::Anthropic => cfg
            .anthropic()
            .model
            .as_deref()
            .unwrap_or(crate::defaults::DEFAULT_ANTHROPIC_MODEL),
    }
}

/// Orchestrates the probes of the principal and the trio (REQ-A24, Task 5.2): one batch if they
/// share endpoint and kind, two in `join!` if they diverge — and the returned trio table is
/// ALWAYS re-projected so that the principal window never contaminates [`derive_warn_tokens`]
/// (SC-A24j): that function takes the minimum of what it receives, so passing it a table that
/// included the principal would let a small-window principal lower the threshold that REQ-A24b
/// defines over the MAGES.
///
/// **Resolves the principal's model and the trio's model SEPARATELY, each with its OWN
/// kind — fix round 1 (finding Logic+Structure).** The first version of this function received
/// `backend_model`/`trio_models` already resolved by the CALLER, and the two call sites
/// (`run()`/`prepare_headless()`) resolved them with `resolve_backend_model(cfg,
/// principal_kind)` — the PRINCIPAL's kind — for BOTH groups. That gives the right answer only
/// when the trio does not diverge (there `magi_kind == principal_kind` by trivial inheritance)
/// and breaks exactly when `[magi].kind` declares a kind DIFFERENT from the principal: a trio
/// seat without its own override ended up inheriting the model from the PRINCIPAL'S SECTION
/// (`[anthropic].model` with the principal on `anthropic`, for example) instead of the model
/// from ITS OWN section (`[openai].model` with the trio on `ollama`). The usual symptom is a
/// silent degradation to *not measured*; the worst case is that that name matches a real model
/// on the trio's endpoint and the probe measures the window of an ALIEN model, poisoning
/// `input_warn_tokens` with a number unrelated to what the trio actually runs. Resolving HERE
/// INSIDE, with the same `resolve_magi_kind` that the divergent branch already used for the
/// KIND, closes the hole by construction: there is no way for this function and
/// `build_magi_orchestrator` (which does exactly this same kind+model resolution) to end up
/// seeing a different model for the same config — the duplication between the two call sites is
/// precisely why the bug existed TWICE (B3).
///
/// **Never blocks or fails startup**: each individual probe fails open inside of
/// `probe_models` (REQ-A24), and an invalid `[magi].kind` here degrades the ENTIRE trio to *not
/// measured* instead of propagating an error — `build_magi_orchestrator`, called later with the
/// SAME config, is the one that reports that invalid `[magi].kind` with its typed error; this
/// probe only needs a best effort, never the final word.
///
/// The `kind` goes by GROUP, not global: with the trio on `ollama` and the principal on
/// `anthropic`, probing the principal with the trio's kind would ask `/api/show` of an endpoint
/// that does not have it.
///
/// Returns the PRINCIPAL's model in addition to its measurements — the caller
/// ([`probe_and_report`]) needs it to name the startup notice (REQ-A24c) without resolving it a
/// second time.
async fn orchestrate_probes(
    cfg: &MagiConfig,
    endpoints: &ResolvedEndpoints,
    principal_kind: ProviderKind,
    factory: &dyn ProbeFactory,
    // `MAGI_MODEL_<AGENT>` overrides, applied per seat via `seats_with_env_overrides` — the
    // SAME resolution `build_magi_orchestrator` performs when it actually constructs the trio
    // (sixth-pass gate finding, S8, Balthasar: before this parameter, the probe measured the
    // TOML/backend model while the trio ran the env-overridden one).
    env_overrides: &MagiEnvModelOverrides,
    // SC-R30: models measured ALONGSIDE the trio, on the trio's endpoint and kind. Used by the
    // stateless path to measure the first pool candidate, which nothing else would measure once
    // there is no cache to drive a lazy probe. They ride the SAME batch rather than a second one:
    // an extra batch would make startup cost two ceilings instead of one.
    extra_models: &[String],
) -> (String, Option<Measurement>, BTreeMap<String, Measurement>) {
    let principal_model = resolve_backend_model(cfg, principal_kind).to_string();

    if !cfg.magi_endpoint_diverges() {
        // Same endpoint and same kind (`magi_endpoint_diverges() == false` implies
        // `[magi].kind`/`[magi].base_url` absent, so the trio inherits `principal_kind`
        // trivially — the trio's fallback is EXACTLY the same model as the principal's, with no
        // possible ambiguity): ONE batch so as not to probe the same thing four times.
        let trio_seats = seats_with_env_overrides(cfg, &principal_model, env_overrides);
        let mut trio_models: Vec<&str> = trio_seats.iter().map(|(_, m)| m.as_str()).collect();
        trio_models.extend(extra_models.iter().map(String::as_str));
        let mut all = trio_models.clone();
        all.push(principal_model.as_str());
        let measured = probe_models(principal_kind, &endpoints.root, &all, factory).await;
        // Re-projects the TRIO's table from `measured`: one probe, two views — the principal
        // never enters what is returned as the trio's table.
        let trio_only: BTreeMap<String, Measurement> = trio_models
            .iter()
            .map(|m| {
                (
                    (*m).to_string(),
                    measured
                        .get(*m)
                        .cloned()
                        .unwrap_or(Measurement::NotMeasuredThisTime),
                )
            })
            .collect();
        let principal_measurement = measured.get(principal_model.as_str()).cloned();
        (principal_model, principal_measurement, trio_only)
    } else {
        match resolve_magi_kind(cfg, principal_kind) {
            Ok(magi_kind) => {
                // FIX round 1: the trio's fallback comes from `resolve_backend_model(cfg,
                // magi_kind)` — the TRIO's kind, already resolved above — NEVER from
                // `principal_kind`. An `[openai].model`/`[anthropic].model` is a property of
                // the SECTION, and the section is chosen by the kind of EACH group, not the
                // principal's.
                let trio_model = resolve_backend_model(cfg, magi_kind).to_string();
                let trio_seats = seats_with_env_overrides(cfg, &trio_model, env_overrides);
                let mut trio_models: Vec<&str> =
                    trio_seats.iter().map(|(_, m)| m.as_str()).collect();
                trio_models.extend(extra_models.iter().map(String::as_str));

                // `join!`, not two `.await`s in a row: in series the worst-case startup would
                // be TWO ceilings; the required property (SC-A24k, one level up, between
                // BATCHES rather than between probes in a batch) is that it still be ONE.
                //
                // `principal_models` bound to a variable (not an inline `&[...]`): the
                // temporary array of a slice literal does not live beyond the expression that
                // creates it, and `tokio::join!` expands its two branches into a single `match`
                // that keeps them alive beyond that expression — E0716 without this `let`.
                let principal_models = [principal_model.as_str()];
                let (principal, trio) = tokio::join!(
                    probe_models(principal_kind, &endpoints.root, &principal_models, factory),
                    probe_models(magi_kind, &endpoints.magi, &trio_models, factory),
                );
                let principal_measurement = principal.get(principal_model.as_str()).cloned();
                (principal_model, principal_measurement, trio)
            }
            Err(_) => {
                // Invalid `[magi].kind`: `build_magi_orchestrator` reports it with its own
                // typed error when it builds the real trio. Here there is no valid
                // `ProviderKind` with which to resolve either the kind or the model of the
                // trio, so it degrades the WHOLE trio to *not measured* without guessing either
                // — the principal is probed alone, because its kind and model are valid by
                // construction (`principal_kind` already arrives resolved).
                let principal_models = [principal_model.as_str()];
                let principal =
                    probe_models(principal_kind, &endpoints.root, &principal_models, factory).await;
                // The THREE seats, named with the PRINCIPAL's model (env-overridden, same as
                // the other two branches, purely for naming consistency) so the returned table
                // has three plausible keys — it is never probed with that name here, so the
                // name cannot poison anything: the three values are `NotMeasuredThisTime` by
                // construction, not by probing.
                let trio_seats = seats_with_env_overrides(cfg, &principal_model, env_overrides);
                let trio = trio_seats
                    .into_iter()
                    .map(|(_, m)| (m, Measurement::NotMeasuredThisTime))
                    .collect();
                let principal_measurement = principal.get(principal_model.as_str()).cloned();
                (principal_model, principal_measurement, trio)
            }
        }
    }
}

/// Probes the principal and the trio, pushes the resulting notices to `notices`, and derives
/// `input_warn_tokens` (REQ-A24b/SC-A24e: what is declared in `[magi].input_warn_tokens` wins
/// over what is measured).
///
/// **The COMPLETE block that Task 5.2 had duplicated between `run()` and `prepare_headless()`
/// — fix round 1, B3.** The duplication is precisely why the Logic+Structure finding of this
/// round existed TWICE instead of once: each call site had its own copy of the
/// `backend_model`/`trio_models` resolution, and only one of the two copies needed to be wrong
/// for the bug to appear. With a single function that does the probing, assembles the notices
/// and derives the threshold, the two call sites are reduced to one call — and a test against
/// this function exercises EXACTLY what the two real call sites invoke, closing the gap that
/// let the original finding through (the round-0 tests built `trio_models` by hand instead of
/// going through real resolution).
/// The models the STATELESS path must measure alongside the trio (SC-R30).
///
/// With a cache, measurement is lazy: a candidate gets measured the moment rotation reaches for it,
/// and the answer is remembered. **Without one there is no lazy path at all** — no `CachedProbe` is
/// built — so a candidate would never be measured, and the strict guard's fail-safe would have
/// nothing to work with on every run.
///
/// It measures **one** candidate, not the pool: the list is ordered strongest to weakest, so the
/// first entry is the most likely rotation destination, and measuring the rest would pay for
/// candidates that probably never run — the very cost the cache exists to avoid.
fn stateless_extra_models(
    cfg: &MagiConfig,
    capability_cache: Option<&Arc<ModelCapabilityCache>>,
) -> Vec<String> {
    if capability_cache.is_some() {
        return Vec::new();
    }
    cfg.fallback_pool()
        .first()
        .map(|entry| vec![entry.model.clone()])
        .into_iter()
        .flatten()
        .collect()
}

async fn probe_and_report(
    cfg: &MagiConfig,
    endpoints: &ResolvedEndpoints,
    principal_kind: ProviderKind,
    factory: &dyn ProbeFactory,
    // Threaded straight through to `orchestrate_probes` — see its own doc (sixth-pass gate
    // finding, S8).
    env_overrides: &MagiEnvModelOverrides,
    // SC-R30: see `orchestrate_probes`. Empty when a cache is available, since measurement is
    // lazy then and a candidate is measured the moment rotation reaches for it.
    extra_models: &[String],
    notices: &mut Vec<Notice>,
    // REQ-R11 (Task 6.4): the trio measurements come back with the threshold because the guard's
    // fail-safe needs them, and re-probing to recover what this call already learned would pay
    // the cost twice for the same answer.
) -> (Option<usize>, BTreeMap<String, Measurement>) {
    let (principal_model, principal_measurement, trio) = orchestrate_probes(
        cfg,
        endpoints,
        principal_kind,
        factory,
        env_overrides,
        extra_models,
    )
    .await;
    notices.push(Notice::info(format!(
        "{principal_model}: {}",
        probe_notice(&principal_measurement.unwrap_or(Measurement::NotMeasuredThisTime))
    )));
    // MAGI S3 re-gate (Caspar): the principal's own notice above only ever reports the
    // PRINCIPAL's measurement — on a cold daemon the trio's models (usually different from
    // the principal's, per REQ-A05) can fail to measure independently, silently falling
    // `input_warn_tokens` back to magi-core's built-in default with nothing telling the user
    // their small-window mage's size warning is not the one actually in effect.
    //
    // **Two INDEPENDENT `if`s, not `if`/`else if` (S8 gate re-review fix).** The previous
    // `else if` only reached `trio_probe_incomplete_notice` when `min_mage_window` returned
    // `None` — i.e. when EVERY mage was cold. But `min_mage_window` returns `Some` as soon as
    // ONE mage measures, so a PARTIALLY cold trio (the common cold-daemon case: some mages
    // warm up faster than others) took the `Some` branch and never reached the incomplete-
    // measurement notice at all — exactly backwards from what the notice exists to catch,
    // since `derive_warn_tokens` below takes the minimum of whichever mages happened to
    // measure and a cold mage with the smallest window silently vanishes from that minimum.
    // The two conditions are not mutually exclusive: a trio can simultaneously have a
    // measured-but-stale minimum window AND an unmeasured seat, and both notices are
    // independently actionable, so both must be free to fire.
    if let Some(min_window) = min_mage_window(&trio) {
        if let Some(n) = stale_composition_notice(min_window, cfg.effective_max_query_bytes()) {
            notices.push(Notice::resolution(n));
        }
    }
    if let Some(n) = trio_probe_incomplete_notice(&trio, cfg.magi().input_warn_tokens) {
        notices.push(Notice::resolution(n));
    }
    // REQ-A24b/SC-A24e: the explicit (`[magi].input_warn_tokens`) wins over the measured.
    let warn = cfg
        .magi()
        .input_warn_tokens
        .or_else(|| derive_warn_tokens(&trio));
    (warn, trio)
}

/// Builds the notice for when at least one mage of the trio is `NotMeasuredThisTime` (cold) —
/// the gap `probe_and_report`'s per-principal notice leaves open (see its call site).
/// Independent of whether `min_mage_window` derived a threshold at all: a trio can be
/// PARTIALLY cold and still have a `Some` minimum from the mages that did measure, and that
/// case is exactly what this notice exists to catch (S8 gate re-review fix).
///
/// `None` (no notice) in two cases where firing one would be noise, not signal:
/// - **`declared` is `Some`**: `[magi].input_warn_tokens` already wins over anything derived
///   (REQ-A24b/SC-A24e), so a failed derivation changes nothing observable — there is no
///   surprise to report.
/// - **Every mage is [`Measurement::NotMeasurable`]**, never
///   [`Measurement::NotMeasuredThisTime`]: that is the expected, non-actionable case of a
///   `kind` that offers no introspection (`openai-compat`/`anthropic`, SC-A24b) — not a failure,
///   and [`probe_notice`] already covers "not measurable" wording for the principal; repeating
///   it here for the trio would be the same non-news twice.
fn trio_probe_incomplete_notice(
    trio: &BTreeMap<String, Measurement>,
    declared: Option<usize>,
) -> Option<String> {
    if declared.is_some() {
        return None;
    }
    let any_cold = trio
        .values()
        .any(|m| matches!(m, Measurement::NotMeasuredThisTime));
    any_cold.then(|| {
        "input_warn_tokens could not be derived this startup: at least one of the trio's \
         models was not measured (the daemon may be cold); the size warning falls back to \
         magi-core's built-in default until a later startup measures it — set \
         `[magi].input_warn_tokens` to fix it at a known value regardless"
            .to_string()
    })
}

/// Digest characters shown in the startup notice (REQ-A24c).
///
/// 12: enough to distinguish manifests without being noise — the full digest is 64 hex
/// (`DIGEST_HEX_LEN` in `magi::probe`, already validated there before reaching here), and
/// showing it whole in a startup line is noise: it is an identifier, not a secret worth seeing
/// in full.
///
/// It is trimmed with `chars().take(..)`, NEVER with `&d[..N]`: the digest is already validated
/// as 64 hex ASCII (REQ-A16b) and a byte slice would be safe today, but the project's invariant
/// is "byte-indexing without verification is forbidden", with no exceptions for convenience —
/// an exception justified today is the one someone copies tomorrow to a non-ASCII field.
const DIGEST_PREVIEW_LEN: usize = 12;

/// Renders the probe's startup notice (REQ-A24c). Three states, not two — see [`Measurement`]:
/// *measured*, *not measurable* (the endpoint does not offer introspection, not a failure) and
/// *not measured this time* (the common case of a cold daemon on first startup).
fn probe_notice(m: &Measurement) -> String {
    match m {
        Measurement::Measured { window, digest } => {
            let d = digest.as_deref().map_or_else(
                || "digest not resolved".to_string(),
                |d| {
                    format!(
                        "digest {}…",
                        d.chars().take(DIGEST_PREVIEW_LEN).collect::<String>()
                    )
                },
            );
            format!("probe: window {window} tokens, {d}")
        }
        Measurement::NotMeasurable => {
            "probe: this endpoint does not offer model introspection (not a failure)".into()
        }
        Measurement::NotMeasuredThisTime => {
            "probe: not measured this time (the daemon may be cold); the next startup will \
             likely measure it"
                .into()
        }
    }
}

/// Warns when `max_query_bytes` is CLOSE to the measured window of the MAGES (SC-A24i/REQ-A24)
/// — never the principal's, which does not receive that payload.
///
/// Compares in TOKENS, not bytes against tokens: `max_query_bytes` is in bytes and
/// `window_tokens` is in tokens, and contrasting them directly would make the notice fire or
/// not by arithmetic accident. The message names the estimator so the reader knows the
/// converted number is an approximation, not a measurement.
fn stale_composition_notice(window_tokens: usize, max_query_bytes: usize) -> Option<String> {
    let cap_tokens = bytes_to_tokens_est(max_query_bytes);
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let threshold = (window_tokens as f64 * STALE_NOTICE_RATIO) as usize;
    (cap_tokens > threshold).then(|| {
        format!(
            "notice: `max_query_bytes` ({max_query_bytes} B ≈ {cap_tokens} tokens at \
             {CHARS_PER_TOKEN_EST} chars/token) is close to the measured window \
             ({window_tokens} tokens); if you switch to a model with a smaller window, the \
             size warning may stop firing — restart magi-rs after the change"
        )
    })
}

/// Why a trio seat could not be built — typed, not `String` (REQ-A05b): the caller reports the
/// three fallen seats at once and needs to distinguish missing-credential from transport
/// failure without parsing text.
///
/// **It NO LONGER has an `Http` variant.** It briefly had one (Task 4.4, round 1),
/// anticipating that `build_native_provider` would capture a real `ProviderError::Http` on a
/// seat's first use. Verified that this never happens:
/// `OpenAiCompatibleProvider::with_timeout`/`from_authority` (magi-core 3.1.0) do not make any
/// HTTP request during construction — their only failure mode is `ProviderError::Network`, via
/// `client_build_error`. The variant remained permanently without a production constructor, so
/// it was removed in round 2 instead of dragging a `#[allow(dead_code)]` that did not protect
/// anything real. The translation of keyless 401/403 (REQ-A12c) now operates on the ALREADY-
/// RENDERED cause of `MagiReport::failed_agents` — see
/// `tools::consult::keyless_auth_explanation` — which is where a real 401 IS reachable; the
/// details and their genuine scope are in this task's report.
#[derive(Debug, thiserror::Error)]
enum SeatError {
    /// The kind requires a credential and none is resolved.
    #[error("missing credential {var} for this backend")]
    MissingCredential {
        /// Name of the expected vault variable/entry.
        var: &'static str,
    },
    /// The HTTP client could not be built. `SafeErrorText`, not `String`: a foreign error's
    /// text may carry the URL with credentials, and this type is only built by going through
    /// [`redact_foreign_error`].
    #[error("could not build the HTTP client: {0}")]
    Transport(SafeErrorText),
    /// The seat declares a model but not the lineage that model belongs to (REQ-R02).
    ///
    /// A `SeatError` rather than its own `TrioError` variant so it joins the mechanism that
    /// reports **every** failing seat at once: an operator who forgot one lineage has usually
    /// forgotten more than one, and discovering them one start at a time costs a start each.
    /// `MagiConfig::load` rejects such a file earlier and more completely; this arm is the same
    /// rule enforced at the point of use, for the crate-internal builder that bypasses `load`.
    #[error("missing lineage: declare [magi].{key} for this seat's model")]
    MissingLineage {
        /// Configuration key the operator has to add.
        key: &'static str,
    },
}

/// Renders ONE `(seat, cause)` as `"Melchior: missing credential …"`.
///
/// Single shared formatting primitive between the `Display` of [`TrioError::SeatUnbuildable`]
/// (below) and [`trio_unavailable_message`] (Task 4.3, B3): before this function there were two
/// independent wordings of the same information — the `Display` derived by `thiserror` reduced
/// `seats` to a count (`seats.len()`) while the actionable startup message did name seat and
/// cause. Any FUTURE `{e}`/`.to_string()` on a `TrioError` — not just the three sites Task 4.3
/// audits by hand — would silently inherit the poor version. `cause` uses its `Display`
/// (`thiserror`), which already goes through [`redact_foreign_error`] where appropriate
/// (`SeatError::Transport`), so this function does not need its own wording.
fn format_seat_failure(seat: &AgentName, cause: &SeatError) -> String {
    format!("{seat:?}: {cause}")
}

/// Why the trio could not be built (REQ-A06).
#[derive(Debug, thiserror::Error)]
enum TrioError {
    /// One or more declared seats failed. They are listed **all**, not just the first: the
    /// three share credential and endpoint, so when one fails due to configuration the normal
    /// thing is for all three to fail — reporting one at a time forces three startups to
    /// discover a single problem.
    ///
    /// The `Display` names EACH seat and its cause (fix round, Task 4.3 review of 4.1): a
    /// `#[error("…", seats.len())]` that only counts ("3") is exactly the defect that motivated
    /// this task — a user without `OPENAI_API_KEY` literally saw "unbuildable seats: 3",
    /// without saying which seat or why.
    #[error(
        "unbuildable seats: {}",
        seats.iter().map(|(s, c)| format_seat_failure(s, c)).collect::<Vec<_>>().join("; ")
    )]
    SeatUnbuildable {
        /// `Seat and cause, one per failure.
        seats: Vec<(AgentName, SeatError)>,
    },
    /// `[magi].kind` brings a value that is not in the vocabulary.
    #[error("unrecognized `[magi].kind`: {0}")]
    UnknownKind(String),
    /// No seat was declared. Different from `SeatUnbuildable`: none failed here, there simply
    /// was none to build.
    #[error("no seats declared for the trio")]
    NoSeats,
    /// `MagiBuilder::build()` rejected the configuration. `SafeErrorText`, not `String`: the
    /// message comes from magi-core, which does not know our redaction rule and may quote
    /// `base_url` with credentials.
    #[error("magi-core rejected the construction: {0}")]
    Builder(SafeErrorText),
}

/// Single actionable message for the three surfaces (REQ-A06, SC-A05b/SC-A05c).
///
/// Only one so the startup notice, the TUI response and the headless error say **the same
/// thing**: if they diverge, the user believes they face three different problems. Reuses
/// [`format_seat_failure`] (B3) instead of re-deriving its own `seats` summary — the only
/// difference from the `Display` of `TrioError` is the separator (one per line here, for human
/// reading in a list of seats; `"; "` in the technical `Display`, intended for a single
/// log/chaining line).
#[must_use]
fn trio_unavailable_message(err: &TrioError) -> String {
    match err {
        TrioError::SeatUnbuildable { seats } => {
            let detail = seats
                .iter()
                .map(|(seat, cause)| format!("  {}", format_seat_failure(seat, cause)))
                .collect::<Vec<_>>()
                .join("\n");
            format!(
                "MAGI consensus is not available — these seats could not be built:\n{detail}\n\n\
                 Check the credential for the backend declared in `[magi]`, or store it with \
                 `magi-rs vault set`."
            )
        }
        TrioError::UnknownKind(k) => format!(
            "MAGI consensus is not available: `[magi].kind = \"{k}\"` is not recognized. \
             Valid values: ollama, openai-compat, anthropic."
        ),
        TrioError::NoSeats | TrioError::Builder(_) => {
            "MAGI consensus is not available: the trio could not be built.".to_string()
        }
    }
}

/// Per-seat model overrides via `MAGI_MODEL_MELCHIOR`/`BALTHASAR`/`CASPAR`.
///
/// Restored, fix round 1 (coordinator, 2026-08-03): removed by mistake in Task 4.1 along with
/// `agent::magi_wiring` (its only production caller, part of the removed adapter machinery) —
/// but R-A03 only admits the three declared breakages in REQ-A21/A22/A23, and this capability
/// was never one of them. Silence plus R-A03 means the capability stays.
#[derive(Debug, Clone, Default)]
struct MagiEnvModelOverrides {
    /// `MAGI_MODEL_MELCHIOR`.
    melchior: Option<String>,
    /// `MAGI_MODEL_BALTHASAR`.
    balthasar: Option<String>,
    /// `MAGI_MODEL_CASPAR`.
    caspar: Option<String>,
}

impl MagiEnvModelOverrides {
    /// The override for THIS process for `seat`, if `MAGI_MODEL_<AGENT>` is set.
    fn for_seat(&self, seat: AgentName) -> Option<&str> {
        match seat {
            AgentName::Melchior => self.melchior.as_deref(),
            AgentName::Balthasar => self.balthasar.as_deref(),
            AgentName::Caspar => self.caspar.as_deref(),
        }
    }

    /// Builds a set of overrides from three ALREADY-READ raw env values (REQ-A12, S8 gate
    /// re-review finding), applying [`crate::config::non_blank`] to each — a blank
    /// `MAGI_MODEL_<AGENT>` (common in CI, where an empty variable is exported rather than
    /// left undeclared) becomes `None` here instead of an active empty-string override.
    ///
    /// **Verified redundant, kept anyway.** `for_seat`'s only production caller
    /// (`build_magi_orchestrator`) already passes its result through
    /// [`resolve_magi_override`], which independently applies the same blank-is-absent
    /// predicate to its `env_model` parameter — see
    /// `a_blank_magi_model_env_override_falls_through_to_the_toml_or_backend_model`, which
    /// pins that end-to-end property. Filtering it here too removes the inconsistency a
    /// reviewer would otherwise trip over reading this struct in isolation, and closes the
    /// gap for any FUTURE caller of `for_seat` that does not route through
    /// `resolve_magi_override`.
    ///
    /// Extracted from [`from_env`] so the filtering is testable without mutating
    /// process-global environment variables (test isolation, B13) — `env::var` reads are not
    /// otherwise injectable in this struct.
    fn from_raw(melchior: Option<&str>, balthasar: Option<&str>, caspar: Option<&str>) -> Self {
        Self {
            melchior: crate::config::non_blank(melchior).map(str::to_string),
            balthasar: crate::config::non_blank(balthasar).map(str::to_string),
            caspar: crate::config::non_blank(caspar).map(str::to_string),
        }
    }

    /// Reads the three environment variables ONCE, at startup (same moment as the rest of this
    /// file's `env > TOML > default` resolution).
    fn from_env() -> Self {
        Self::from_raw(
            env::var("MAGI_MODEL_MELCHIOR").ok().as_deref(),
            env::var("MAGI_MODEL_BALTHASAR").ok().as_deref(),
            env::var("MAGI_MODEL_CASPAR").ok().as_deref(),
        )
    }
}

/// Applies `MAGI_MODEL_<AGENT>` env overrides on top of [`MagiSectionConfig::seats`]'s
/// TOML-or-backend resolution — the SAME `env > TOML > backend` chain
/// [`build_magi_orchestrator`] applies when it actually constructs each seat's provider.
///
/// Extracted (sixth-pass gate finding, S8, Balthasar) so [`orchestrate_probes`] and
/// [`build_magi_orchestrator`] cannot see a different model for the same seat: before this, the
/// probe read `cfg.magi.seats(...)` directly and never consulted `env_overrides` at all, so an
/// operator setting `MAGI_MODEL_MELCHIOR` had the probe measure one model's window while the
/// trio actually ran a different one — silently poisoning `input_warn_tokens` (REQ-A24b, whose
/// value is the MINIMUM across the mages) with a number that describes a model nobody is
/// running. One function computing the resolved seat models, used by both, makes that
/// divergence structurally impossible instead of a discipline to remember at each call site
/// (B3 — the same reasoning that already produced [`resolve_magi_kind`]/[`resolve_backend_model`]
/// for the kind and the backend model).
fn seats_with_env_overrides(
    cfg: &MagiConfig,
    backend_model: &str,
    env_overrides: &MagiEnvModelOverrides,
) -> Vec<(AgentName, String)> {
    cfg.magi()
        .seats(backend_model)
        .into_iter()
        .map(|(seat, toml_or_backend_model)| {
            let env_model = env_overrides.for_seat(seat);
            let model = resolve_magi_override(Some(&toml_or_backend_model), env_model)
                .unwrap_or(toml_or_backend_model);
            (seat, model)
        })
        .collect()
}

/// Formats a `base_url` resolution result for display in a startup notice: the endpoint
/// TEMPLATE on success, or a redacted rendering of the error's `Display` on failure.
///
/// The success branch never needs [`redact_url`]: an [`EndpointTemplate`] cannot contain a
/// secret by construction (REQ-A16c — `EndpointTemplate::parse` rejects a literal credential,
/// so `as_str()` can only ever carry the `[user]:[password]` placeholders). Running it through
/// redaction anyway would blank that informative placeholder text by position, which is exactly
/// what the `endpoint_display_text_leaves_a_valid_template_untouched` test guards against.
///
/// Generic over the error type — rather than hard-coded to `EndpointError` — precisely so a
/// test can exercise the failure branch with a fabricated error whose `Display` embeds a
/// credential. `EndpointError`'s own variants cannot do that today: every string-carrying
/// field is `&'static str` (a vault entry name or fixed text), never text derived from the
/// received value — see `src/magi/endpoint.rs`, where `EndpointError` is defined in THIS crate,
/// not a genuinely external, separately-versioned dependency like `magi-core` (whose
/// `#[non_exhaustive]` error types are what [`redact_foreign_error`] exists to guard against a
/// *future* release changing under us). That distinction does not make the failure branch safe
/// to leave unguarded, though (sixth-pass gate finding, S8, Balthasar): "no current variant
/// leaks the value" was true only by inspection, not by anything the compiler enforces, so
/// routing it through [`redact_foreign_text`] anyway makes the property hold structurally
/// instead of by convention — a no-op today, and still correct if a future variant ever does
/// interpolate raw text.
#[must_use]
fn endpoint_display_text<E: std::fmt::Display>(result: &Result<EndpointTemplate, E>) -> String {
    match result {
        Ok(template) => template.as_str().to_string(),
        Err(e) => redact_foreign_text(&e.to_string()).as_str().to_string(),
    }
}

/// Announces that content goes through the principal provider BEFORE the trio
/// (REQ-A07c/REQ-A07p, SC-A07p), when that is effectively what will happen.
///
/// It fires **only** when the trio diverges from the principal (`cfg.magi_endpoint_diverges()`)
/// **and** `inference_active` is `true`: with everything on the same endpoint there is no
/// divergence
/// to report, and with inference inactive (`[magi].default_mode` declared) content never goes
/// to the principal to be classified — the notice would be noise in both cases.
///
/// **Divergence from Step 3 of this task's brief — proven by the very
/// test the brief delivered, not just argued.** The original pseudocode RECALCULATED
/// `will_attempt_classification` inside (`cfg.effective_default_mode().is_none()`), COMPLETELY
/// IGNORING the `inference_active` parameter. With identical `cfg` in the last two assertions
/// of `endpoint_divergence_is_announced_only_ when_it_actually_diverges` — the only difference
/// is `true` vs. `false` in the second argument — an internal recalculation would have yielded
/// the SAME result in both calls, contradicting the third assertion (`divergence_notice(&cfg,
/// false).is_none()`). The parameter must be the ONLY source for that side of the gate;
/// recalculating it inside is not a style variation, it is a bug that the brief's own test
/// makes visible as soon as it runs.
///
/// # Parameters
/// * `cfg` - the configuration already loaded (post [`MagiConfig::load`]); see the note about infallibility below on why its two `effective_*_base_url()` do not fail in production.
/// * `inference_active` - `true` when THIS session may end up classifying the mode by content — the caller already knows it (it needs it for other decisions in the same run, such as whether it is worth warning about the cost of REQ-A07c) and it is received instead of being re-derived here, precisely so this function does not have a second opinion about something the caller already resolved.
///
/// # Returns
/// `Some(Notice)` (tier `Resolution`) when divergence and inference coincide; `None` in any
/// other case.
#[must_use]
fn divergence_notice(cfg: &MagiConfig, inference_active: bool) -> Option<Notice> {
    if !(cfg.magi_endpoint_diverges() && inference_active) {
        return None;
    }

    // INFALLIBLE BY PRECONDITION, same pattern as `MagiConfig::effective_provider`/
    // `effective_default_mode`: `MagiConfig::load()` already called
    // `effective_base_url()?`/`effective_magi_base_url()?` before returning this `cfg` (see
    // `config.rs::load`), so an `Err` here can only happen if someone built `MagiConfig` by
    // hand skipping `load()` — a caller bug, not a user input. The `debug_assert!` turns it
    // into a loud panic in debug/test.
    //
    // But it is NOT propagated with `.ok()?` (the pattern this task's brief already marks as
    // [CRITICAL] once in this gate): in a RELEASE build, without `debug_assertions`, that would
    // silently swallow the error and make this PRIVACY notice disappear exactly when something
    // has already gone wrong. Instead, if resolution ever failed despite the precondition, the
    // notice is emitted ANYWAY, with the error text in place of the endpoint — the property
    // that matters is that the EMISSION of this notice never silently depends on whether
    // parsing succeeded.
    // `debug_assert!` and NOT `assert!`, deliberately — asked again by S3 Loop 2 (Melchior), who
    // read it as the "precondition that matters in release" pattern CLAUDE.md warns about. It is
    // the opposite case: this precondition does NOT matter in release, because the paragraph above
    // already handles its violation. `endpoint_display_text` renders the `Err` as redacted text
    // and the notice is emitted regardless, so the property worth protecting — that a PRIVACY
    // notice never silently disappears — holds in every build profile by construction rather than
    // by an assertion. An `assert!` here would trade a degraded-but-present notice for a panic,
    // which is strictly worse for the user it exists to warn.
    //
    // The suggested alternative, returning `None` on an invalid template, is the one thing that
    // must not happen: that is the silent disappearance, spelled differently.
    //
    // Note for whoever tries to test the release path: they cannot, from the default profile.
    // `cfg(debug_assertions)` is on under `cargo test`, so these fire before the fallback is
    // reached — which is why the fallback's RENDERING is pinned one level down instead, by
    // `endpoint_display_text_redacts_a_credential_a_future_error_variant_might_embed` and
    // `endpoint_display_text_leaves_a_valid_template_untouched`.
    let magi_url = cfg.effective_magi_base_url();
    let root_url = cfg.effective_base_url();
    debug_assert!(magi_url.is_ok(), "load() must have validated");
    debug_assert!(root_url.is_ok(), "load() must have validated");

    // See `endpoint_display_text`'s own doc for why the success branch never needs
    // `redact_url` and the failure branch is routed through `redact_foreign_text` anyway
    // (sixth-pass gate finding, S8).
    let magi_text = endpoint_display_text(&magi_url);
    let root_text = endpoint_display_text(&root_url);

    Some(Notice::resolution(format!(
        "notice: the trio runs on {magi_text} but mode inference sends the content to the \
         main provider FIRST ({root_text}). Declare `[magi].default_mode` to avoid that \
         step."
    )))
}

/// Pushes the notice from [`divergence_notice`] into `notices` when it applies (SC-A07p,
/// wiring).
///
/// Factored apart from `divergence_notice` to give the WRITE itself —not just the predicate— a
/// point a test can invoke directly: `run()`, the real owner of `startup_notices`, opens the
/// vault, discovers the real workspace and uses a real TTY, so it cannot be handled from a unit
/// test (same limitation that `MagiConfig::resolution_notices`'s own test in `config.rs`
/// already documents and resolves by calling the function directly). This is the ONLY line
/// `run()` executes for this, so confirming the diff invokes it there is a one-line review, not
/// a review of all of `run()` — the failure mode this exists to close (`divergence_notice`
/// correct but never called) already happened once in this plan (Task 4.3).
fn push_divergence_notice(cfg: &MagiConfig, inference_active: bool, notices: &mut Vec<Notice>) {
    if let Some(n) = divergence_notice(cfg, inference_active) {
        notices.push(n);
    }
}

/// Normalizes an Ollama root to the OpenAI-compat shape (`…/v1`), idempotent, **and warns when
/// it had to touch something**.
///
/// **It is no longer needed to make the URL work, and is kept anyway.** It was written when
/// `OllamaProvider` was out of the path (D-A07) and nothing else normalized, so a `base_url =
/// "http://localhost:11434"` hit `/chat/completions` at the root and 404'd on first use. Since
/// REQ-R30 the provider accepts both spellings itself — but this function is the only thing that
/// still knows **which one arrived**, and the notice exists to tell the operator to write down
/// what actually happens. Dropping it because the URL now works either way would be a silent
/// behaviour change (R-R04), which is the same objection that made it return the notice instead
/// of normalizing quietly in the first place.
///
/// **The returned `root` and the notice text do NOT share the same URL** (fix round 2,
/// C1, REQ-A16c path #2): `base_url` here is already the RESOLVED endpoint — post placeholder
/// substitution — so it may carry a real credential. The `root` needs it intact (it is what
/// builds the HTTP client); the notice is text that ends up in the TUI startup list and in
/// headless stderr, so it goes through [`redact_url`] before being interpolated. Two uses, two
/// rules — hence the function builds the notice from `normalized` but redacts a COPY for the
/// text, instead of redacting `normalized` in place.
fn openai_compat_root(base_url: &str) -> (String, Option<String>) {
    let trimmed = base_url.trim_end_matches('/');
    if trimmed.rsplit('/').next() == Some("v1") {
        (trimmed.to_string(), None)
    } else {
        let normalized = format!("{trimmed}/v1");
        let notice = format!(
            "notice: Ollama `base_url` without a `/v1` suffix; using `{}` for \
             completions. Declare it explicitly so the configuration says what \
             actually happens.",
            redact_url(&normalized)
        );
        (normalized, Some(notice))
    }
}

/// Builds ONE native provider from magi-core according to the declared `kind` (REQ-A01b).
///
/// **`ollama` completes through magi-core's `OllamaProvider`** (REQ-R30, which reverts D-A07).
/// D-A07 was decided on an *impossibility* — the type's only constructor pinned a 300 s client
/// timeout with no override, incompatible with REQ-A04 (`operation_budget + client_timeout <=
/// ceiling`) — and magi-core 3.2.0 removed it by adding `with_timeout`, which bounds **both**
/// HTTP clients the type builds. So the question of which transport to use was answered on its
/// merits for the first time, and legibility decided it: `kind = "ollama"` wired to
/// `OllamaProvider` is what anyone opening this function expects.
///
/// It is **not** a protocol change and therefore does not violate R-R04: `OllamaProvider::complete`
/// delegates to an internal `OpenAiCompatibleProvider`, so the transport is the same one this arm
/// used before — only its constructor moved.
///
/// **What survives D-A07's reversal is the narrower rule: never `new`.** It delegates to
/// `with_timeout(..., DEFAULT_CLIENT_TIMEOUT)` = 300 s, and getting it wrong compiles, runs, and
/// breaks the derived scale silently.
///
/// **`ollama` is keyless**: an authenticated `base_url` under this kind does not fail here —
/// it fails on first use with a 401, which Task 4.4 translates.
///
/// # Errors
/// [`SeatError::MissingCredential`] if the kind requires a credential and none is resolved;
/// [`SeatError::Transport`] if the HTTP client could not be built.
fn build_native_provider(
    kind: ProviderKind,
    base_url: &ResolvedEndpoint,
    model: &str,
    creds: Option<&dyn Credentials>,
    client_timeout: Duration,
    notices: &mut Vec<Notice>,
) -> Result<Arc<dyn LlmProvider>, SeatError> {
    // `redact_foreign_error`, NOT `to_string()`: magi-core assembles the message, which does
    // not know our redaction rule and may quote `base_url`.
    let to_seat = |e: ProviderError| SeatError::Transport(redact_foreign_error(&e));

    Ok(match kind {
        // `api_key = None` ⇒ no `Authorization` header, which is what Ollama expects.
        ProviderKind::Ollama => {
            // The NOTICE is kept even though `OllamaProvider` accepts both spellings itself: it
            // does not exist to make the URL work, it exists to tell the operator to write down
            // what actually happens. Losing it would be a silent behaviour change (R-R04), and
            // `openai_compat_root` is the only thing that still knows which spelling arrived.
            //
            // `.as_str()`, not `.to_string()` (Melchior, loop 32): `base_url` is a newtype and
            // `with_timeout` takes `impl Into<String>` — `&str` already satisfies it without
            // the intermediate step.
            let (root, notice) = openai_compat_root(base_url.as_str());
            if let Some(n) = notice {
                notices.push(Notice::resolution(n));
            }
            // NEVER `new` (REQ-R30): it delegates to `with_timeout(..., DEFAULT_CLIENT_TIMEOUT)`
            // = 300 s, which cannot satisfy `operation_budget + client_timeout <= ceiling`.
            // Picking the wrong constructor compiles, runs, and breaks the derived scale in
            // silence — see `the_ollama_seat_honours_the_client_timeout_it_was_given`.
            Arc::new(OllamaProvider::with_timeout(root, model, client_timeout).map_err(to_seat)?)
        }
        ProviderKind::OpenAiCompat => {
            let key = creds
                .and_then(|c| c.openai())
                .ok_or(SeatError::MissingCredential {
                    var: "OPENAI_API_KEY",
                })?;
            Arc::new(
                OpenAiCompatibleProvider::with_timeout(
                    base_url.as_str(),
                    model,
                    Some(key),
                    client_timeout,
                )
                .map_err(to_seat)?,
            )
        }
        ProviderKind::Anthropic => {
            let key = creds
                .and_then(|c| c.anthropic())
                .ok_or(SeatError::MissingCredential {
                    var: "ANTHROPIC_API_KEY",
                })?;
            Arc::new(ClaudeProvider::with_timeout(key, model, client_timeout).map_err(to_seat)?)
        }
    })
}

// Test-only (I2, fix round 2): trace of what the LAST call to `build_magi_orchestrator` wired
// IN THIS THREAD — (seat, resolved model, a NEW Arc was allocated around the built provider).
// Exists so a test can assert against the REAL function instead of reconstructing its wiring
// logic in a custom `MagiBuilder` (which is exactly what let the production wrapper disappear
// unnoticed). Does not change `build_magi_orchestrator`'s signature — each test in this file
// runs in its OWN thread (the `#[test]` harness spawns them that way by design), so the
// thread-local isolates one call from another without extra coordination.
//
// THE THIRD FIELD IS MEASURED, NOT ASSERTED — and it was not, until Task 3.1 ran the mutation
// (B16). It used to push the literal `true`, under a comment claiming that removing the
// production wrapper "without touching the trace" would break the count this test verifies.
// That claim was false: replacing `Arc::new(RetryProvider::with_config(p, retry))` with a bare
// `p` leaves `seats.push` running and the literal saying `true`, so the guardian went on
// passing through the exact regression it existed to catch. It now compares the resulting
// allocation's address against the one it was handed: same address ⇒ nothing wrapped it.
//
// What that does and does not prove: it proves a new `Arc` was allocated around `p`, not that
// the wrapper is specifically a `RetryProvider`. `LlmProvider` is a foreign trait with no
// `Any` (R-A01 forbids touching magi-core), so there is no downcast to do better — but a
// measurement that can be wrong for a narrow reason beats a literal that cannot be wrong at
// all. Both address reads are `cfg(test)`, so production pays nothing and keeps its move.
#[cfg(test)]
thread_local! {
    static SEAT_WIRING_TRACE: std::cell::RefCell<Vec<(AgentName, String, bool)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Test-only: the trace left by the LAST call to `build_magi_orchestrator` in this thread. See
/// [`SEAT_WIRING_TRACE`].
#[cfg(test)]
fn seat_wiring_trace() -> Vec<(AgentName, String, bool)> {
    SEAT_WIRING_TRACE.with(|t| t.borrow().clone())
}

// Test-only (Task 3.1): the lineage each seat was REGISTERED with — recorded in the loop that
// calls `with_agent`, not in the one that builds the providers. The two facts have two different
// sites and each guardian is anchored at its own: the retry wrap can only disappear where the
// wrapping happens, and the lineage can only be wrong where the registration happens. Reading it
// from the built `Magi` is not an option — `MagiBuilder::agent_lineages` is private and the
// orchestrator exposes no reader (R-A01 forbids touching that crate).
#[cfg(test)]
thread_local! {
    static SEAT_LINEAGE_TRACE: std::cell::RefCell<Vec<(AgentName, String)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Test-only: the lineages the LAST call to `build_magi_orchestrator` registered in this thread.
/// See [`SEAT_LINEAGE_TRACE`].
#[cfg(test)]
fn seat_lineage_trace() -> Vec<(AgentName, String)> {
    SEAT_LINEAGE_TRACE.with(|t| t.borrow().clone())
}

/// Test-only (Task 4.1): what the LAST call to `build_magi_orchestrator` handed magi-core as a
/// fallback pool.
///
/// Every field is READ BACK from the values that were actually passed, never restated as a
/// literal — the mistake `SEAT_WIRING_TRACE` made and that Task 3.1's mutation exposed.
#[cfg(test)]
#[derive(Debug, Clone)]
struct PoolWiring {
    /// `(model, lineage)` per candidate, in the order they were pushed.
    candidates: Vec<(String, String)>,
    /// Rotation ceiling handed to the pool. `0` is the declared kill-switch.
    max_rotations: u32,
    /// Whether the strict context guard was handed to the builder as `true`.
    strict_guard: bool,
}

// Test-only: the pool of the LAST call in this thread. `None` means no pool was declared, which
// is a different fact from an empty pool and the tests distinguish them.
#[cfg(test)]
thread_local! {
    static POOL_WIRING_TRACE: std::cell::RefCell<Option<PoolWiring>> =
        const { std::cell::RefCell::new(None) };
}

/// Test-only: see [`POOL_WIRING_TRACE`].
#[cfg(test)]
fn pool_wiring_trace() -> Option<PoolWiring> {
    POOL_WIRING_TRACE.with(|t| t.borrow().clone())
}

/// Opens the capability cache over an already-open encrypted database (REQ-R25).
///
/// `None` means measurements will not be remembered between runs, which is **slower, not wrong**:
/// every consumer of the cache degrades to measuring. Two distinct paths produce it and both are
/// legitimate: the ephemeral run has no database at all, and an unreadable or unwritable table is
/// a degradation the whole measurement subsystem is built to absorb (Task 6.8).
///
/// An **absent or empty** table is NOT this case: `init_schema` creates it and the first run fills
/// it. Confusing the two would mean never persisting anything on exactly the clean start where the
/// cache pays off most.
fn open_capability_cache(
    store: Option<&EncryptedSqliteMemory>,
    notices: &mut Vec<Notice>,
) -> Option<Arc<ModelCapabilityCache>> {
    let store = store?;
    let dek = match store.data_key() {
        Ok(d) => d,
        Err(e) => {
            notices.push(Notice::resolution(format!(
                "notice: model measurements will not be remembered between runs \
                 (could not derive the key: {e})."
            )));
            return None;
        }
    };
    match ModelCapabilityCache::new(store.shared_conn(), dek) {
        Ok(c) => Some(Arc::new(c)),
        Err(e) => {
            notices.push(Notice::resolution(format!(
                "notice: model measurements will not be remembered between runs ({e})."
            )));
            None
        }
    }
}

/// Everything `build_magi_orchestrator` needs, bundled.
///
/// Seven positional parameters had accumulated and the eighth is what forced the question, but
/// arity was never the real problem: `&MagiConfig`, `&ResolvedEndpoints` and
/// `&MagiEnvModelOverrides` are three references a caller passes in a row, and a transposition
/// among them changes which decision wins with **nothing to catch it** — the same hazard that made
/// `OpenAiSettings` and `ModeSources` named structs instead of argument lists.
///
/// `notices` stays a separate `&mut` parameter on purpose: it is the function's OUTPUT channel,
/// and folding an out-param into a struct of inputs reads as though it were one.
struct TrioBuild<'a> {
    /// Loaded configuration.
    cfg: &'a MagiConfig,
    /// The ALREADY-RESOLVED `ProviderKind` of the principal (env `MAGI_PROVIDER` > TOML >
    /// default), not `cfg.effective_provider()`: before, an absent `[magi].kind` inherited by
    /// re-reading `provider` from TOML on its own, so `MAGI_PROVIDER` moved the principal without
    /// moving the trio. This field is what makes inheritance see the SAME decision.
    principal_kind: ProviderKind,
    /// ALREADY-RESOLVED endpoints. The builder does not know the vault: resolution is a named step
    /// of `main.rs`, and `resolve_endpoints` is the sole producer of `ResolvedEndpoints`.
    endpoints: &'a ResolvedEndpoints,
    /// Credentials for the kinds that need one; `ollama` is keyless.
    creds: Option<&'a dyn Credentials>,
    /// Warning threshold produced by the probe, or `None` to keep magi-core's default.
    warn_tokens: Option<usize>,
    /// `MAGI_MODEL_*`, layered ON TOP of `seats(backend_model)` so the chain is
    /// `env > TOML > backend` without duplicating that resolution.
    env_overrides: &'a MagiEnvModelOverrides,
    /// REQ-R10/R25/R28. `None` is the ephemeral path (Task 6.8): no encrypted database means
    /// nothing to remember measurements in, which degrades measurement — never the run.
    capability_cache: Option<&'a Arc<ModelCapabilityCache>>,
    /// What the startup probe measured, keyed by model (REQ-R11). Feeds the strict-guard
    /// fail-safe, which must know whether any CANDIDATE got a window.
    measured: &'a BTreeMap<String, Measurement>,
}

/// Builds the MAGI trio with the NATIVE providers of magi-core (REQ-A01).
///
/// The adapter disappears and with it the system-prompt doubling: each mage receives its prompt
/// through the provider's own channel.
///
/// `notices` receives the non-fatal construction notices (e.g. the normalization of an Ollama
/// `base_url` without `/v1`). It is passed by parameter and not returned separately so that a
/// notice emitted on the error path **also** reaches the user: a seat failure and a strange URL
/// are usually the same problem seen from two sides.
///
/// `warn_tokens` enters by PARAMETER and is not resolved inside: it is produced by the probe
/// (`orchestrate_probes`/`derive_warn_tokens`, Task 5.2), called by the call site BEFORE this
/// function. With `None` it falls back to magi-core's default — the v0.11.0 behavior, which
/// remains the result when the probe measured nothing measurable.
///
/// # Errors
/// - [`TrioError::UnknownKind`] if `[magi].kind` brings an unrecognized value. It is validated HERE with its own `ProviderKind::parse`, not via `cfg.effective_magi_kind()`: that accessor assumes `validate_vocabulary` already ran and swallows an unrecognized value falling back to inheritance — a correct precondition for its other callers, but exactly the one this point must NOT assume in order to report the error.
/// - [`TrioError::SeatUnbuildable`] with **all** the seats that could not be built and their cause.
fn build_magi_orchestrator(
    b: &TrioBuild<'_>,
    notices: &mut Vec<Notice>,
) -> Result<Arc<Magi>, TrioError> {
    let &TrioBuild {
        cfg,
        principal_kind,
        endpoints,
        creds,
        warn_tokens,
        env_overrides,
        capability_cache,
        measured,
    } = b;
    // Absent/empty `[magi].kind` inherits `principal_kind` — the ALREADY-RESOLVED one, not
    // `cfg.effective_provider()` (TOML-only). Present-but-unrecognized remains a typed error.
    // Task 5.2: extracted to `resolve_magi_kind`, shared with `orchestrate_probes` (B3) —
    // before, each had its own copy of this same rule, with the risk that a probe measured a
    // different kind from the one the trio actually ends up using.
    let kind = resolve_magi_kind(cfg, principal_kind).map_err(|e| TrioError::UnknownKind(e.got))?;
    // The trio uses the ALREADY-RESOLVED endpoint that `main.rs` produced — it does not re-
    // resolve or read the template.
    let base = &endpoints.magi;

    // BACKEND model: the one a seat inherits without its own model, and the builder fallback's.
    // Task 5.2: extracted to `resolve_backend_model`, shared with `orchestrate_probes` (B3).
    let backend_model: &str = resolve_backend_model(cfg, kind);
    let ceiling = Duration::from_secs(cfg.magi().agent_timeout_secs.unwrap_or(AGENT_TIMEOUT_SECS));

    // `RetryConfig` is `#[non_exhaustive]`: outside the crate there is NO literal nor
    // `..default()` — it is built with `default()` and adjusted field by field.
    let mut retry = RetryConfig::default();
    retry.operation_budget = derive_operation_budget(ceiling.as_secs());
    let client_timeout = derive_client_timeout(ceiling.as_secs());

    // The THREE seats are built first, so that ALL that fail can be reported.
    let mut failures: Vec<(AgentName, SeatError)> = Vec::new();
    let mut seats: Vec<(AgentName, Arc<dyn LlmProvider>, CoreLineage, String)> = Vec::new();

    // Test-only (I2, fix round 2): clears the trace of the PREVIOUS call in this thread before
    // starting — a test's `seat_wiring_trace()` must see ONLY what THIS call wired.
    #[cfg(test)]
    SEAT_WIRING_TRACE.with(|t| t.borrow_mut().clear());
    #[cfg(test)]
    SEAT_LINEAGE_TRACE.with(|t| t.borrow_mut().clear());
    // The pool trace needs the same clear, and for a sharper reason than the other two: it is
    // written only INSIDE the `if` that wires a declared pool, so a call with no pool left the
    // previous call's value standing and a test would read a pool this call never wired — the
    // failure mode being that it passes (S3 Loop 2, Balthasar).
    #[cfg(test)]
    POOL_WIRING_TRACE.with(|t| *t.borrow_mut() = None);

    // `env > TOML > backend`, via `seats_with_env_overrides` — the SAME resolution
    // `orchestrate_probes` now applies (B3, sixth-pass gate finding S8), so the two cannot see
    // a different model for the same seat.
    for (seat, model) in seats_with_env_overrides(cfg, backend_model, env_overrides) {
        // REQ-R01/R02: resolved BEFORE the provider, so a seat missing its lineage joins the same
        // "report every failing seat at once" pass instead of aborting on the first one.
        let lineage = match cfg.magi().lineage_of_seat(seat) {
            Ok(l) => CoreLineage::from(l),
            Err(LineageError::Missing { key }) => {
                failures.push((seat, SeatError::MissingLineage { key }));
                continue;
            }
        };
        match build_native_provider(kind, base, &model, creds, client_timeout, notices) {
            // REQ-A03: `MagiBuilder::build()` does NOT wrap anything, so without this the retry
            // the trio inherited from the adapter is lost.
            Ok(p) => {
                // Read BEFORE the move, so the trace below can compare against it. `*const ()`
                // takes the data address of what may be a fat pointer; `cfg(test)` only, so
                // production keeps the move and pays nothing.
                #[cfg(test)]
                let unwrapped_addr = Arc::as_ptr(&p) as *const () as usize;
                let wrapped = Arc::new(RetryProvider::with_config(p, retry.clone()));
                // Recorded ON THE SAME branch that does the real wrap (I2, fix round 2), and
                // MEASURED rather than asserted — see the note on `SEAT_WIRING_TRACE` for the
                // mutation that proved the previous literal `true` guarded nothing. A different
                // address means something allocated around the provider; the same one means the
                // wrap is gone, which is precisely the silent regression this guards.
                #[cfg(test)]
                SEAT_WIRING_TRACE.with(|t| {
                    let wrapped_addr = Arc::as_ptr(&wrapped) as *const () as usize;
                    t.borrow_mut()
                        .push((seat, model.clone(), wrapped_addr != unwrapped_addr));
                });
                seats.push((seat, wrapped, lineage, model.clone()));
            }
            Err(cause) => failures.push((seat, cause)),
        }
    }

    if !failures.is_empty() {
        return Err(TrioError::SeatUnbuildable { seats: failures });
    }
    if seats.is_empty() {
        // Unreachable today — `seats()` always returns the three — but the variant exists for
        // Task 4.3's exhaustive `match` and does not depend on that internal invariant to be
        // correct.
        return Err(TrioError::NoSeats);
    }

    // The builder's FALLBACK is built SEPARATELY, and the name matters: it is NOT magi-rs's
    // "principal provider" (the conversational agent's, which this milestone does not touch).
    // With all three seats overridden this provider is never used, and that is precisely why it
    // is useful for it to be a written decision.
    //
    // `&mut sink` discardable: the normalization notice already went out in the seat loop with
    // the SAME `base_url`. Pushing it again would duplicate it on screen (and `render_notices`
    // would dedupe it anyway, but it is not worth building it twice).
    let mut sink: Vec<Notice> = Vec::new();
    let fallback_provider = build_native_provider(
        kind,
        base,
        cfg.magi().fallback_model(backend_model),
        creds,
        client_timeout,
        &mut sink,
    )
    .map_err(|e| TrioError::Builder(redact_foreign_error(&e)))?;

    let mut builder = MagiBuilder::new(Arc::new(RetryProvider::with_config(
        fallback_provider,
        retry.clone(),
    )))
    .with_timeout(ceiling);

    // REQ-A15: the OTHER TWO exposed keys are also wired. Declaring them in TOML without
    // connecting them would make them decorative.
    if cfg.magi().retry_disabled.unwrap_or(false) {
        builder = builder.with_retry_disabled();
    }

    // REQ-R01: `with_agent`, not `with_provider` — the only door that carries the rotation
    // diversity key. Note that `with_agent` also does `primary_probes.remove(&agent)` ("a plain
    // primary declares no probe", `orchestrator.rs:309-320`), so once Phase 6 registers probes
    // through `with_agent_and_probe`, a plain `with_agent` for the SAME seat afterwards would
    // discard that probe in silence. Order is load-bearing here.
    // The RESOLVED url dies here: everything downstream of this line — the cache key, the probe,
    // any notice — sees only the redacted form (REQ-A16c/SC-R58). The encrypted database protects
    // the file; it does not make writing a credential into it acceptable.
    let endpoint_redacted = redact_url(base.as_str());
    let declared_pool = cfg.fallback_pool();

    // REQ-R11: the declared value is NOT what reaches magi-core. A `true` with no measured
    // candidate rejects every one of them at condition #6 and switches rotation off whole, so
    // the fail-safe resolves it and announces the override with its REAL reason.
    let (effective_guard, guard_notice) =
        effective_strict_guard(cfg.declared_strict_context_guard(), measured, declared_pool);
    if let Some(n) = guard_notice {
        notices.push(n);
    }

    // REQ-R21/D-R09: the threshold refines here and not in `probe_and_report`, because at startup
    // the POOL is not measured — measurement is lazy — so its windows exist only in the cache, and
    // the builder is the one place that holds both it and the trio's measurements.
    //
    // An explicitly declared `[magi].input_warn_tokens` is never touched (REQ-A24b/SC-A24e): the
    // operator's number wins over anything derived, and refining it would be overriding an
    // instruction rather than filling a gap.
    let warn_tokens = if cfg.magi().input_warn_tokens.is_some() {
        warn_tokens
    } else {
        // Filtered BY SEAT, not "every value in the map": the map may also carry pool candidates
        // (the stateless path measures one), and folding a candidate into the trio base would
        // defeat the band this derivation exists to enforce — the small candidate would lower the
        // very base it is supposed to be compared against.
        let trio_windows: Vec<usize> = seats
            .iter()
            .filter_map(|(_, _, _, model)| match measured.get(model) {
                Some(Measurement::Measured { window, .. }) => Some(*window),
                _ => None,
            })
            .collect();
        if trio_windows.is_empty() {
            warn_tokens
        } else {
            let pool_windows: Vec<(&str, usize)> = capability_cache
                .map(|cache| {
                    declared_pool
                        .iter()
                        .filter_map(|entry| {
                            cache
                                .get(&endpoint_redacted, &entry.model)
                                .ok()
                                .flatten()
                                .map(|row| (entry.model.as_str(), row.window))
                        })
                        .collect()
                })
                .unwrap_or_default();
            let (threshold, band_notices) = derive_input_warn_tokens(&trio_windows, &pool_windows);
            notices.extend(band_notices);
            Some(threshold)
        }
    };

    // REQ-R29, second half: corroborate the DECLARED lineages against the cached digests. Only
    // under `enforce_diversity`, and only ever as a warning — the declarative half already ran at
    // load time and errored there if it had to. Reads the cache; never probes (SC-R42).
    if cfg.effective_enforce_diversity() {
        if let Some(cache) = capability_cache {
            let mut rows: Vec<(String, magi_rs::magi::lineage::Lineage, Option<String>)> =
                Vec::new();
            for (seat, _, _, model) in &seats {
                if let Ok(lineage) = cfg.magi().lineage_of_seat(*seat) {
                    let digest = cache
                        .get(&endpoint_redacted, model)
                        .ok()
                        .flatten()
                        .and_then(|row| row.digest);
                    rows.push((model.clone(), lineage, digest));
                }
            }
            for entry in declared_pool {
                let digest = cache
                    .get(&endpoint_redacted, &entry.model)
                    .ok()
                    .flatten()
                    .and_then(|row| row.digest);
                rows.push((entry.model.clone(), entry.lineage.clone(), digest));
            }
            notices.extend(corroborate_by_digest(&rows));
        }
    }

    // REQ-R29's resolved-model half. `seats()` falls an undeclared seat back to the backend
    // model, so a config naming no trio runs ONE model under three built-in labels — the case a
    // label-distinctness check approves while it is literally what SC-R44 rejects. The load-time
    // check cannot error on it without denying a fresh clone its first start, so it is reported
    // here instead, where the resolved backend model is known.
    // The seats ALREADY resolved through `env > TOML > backend`, not a third derivation: a
    // separate one would disagree with the trio and the probe, and print statements that are
    // false in both directions.
    let resolved_seats: Vec<(AgentName, String)> = seats
        .iter()
        .map(|(seat, _, _, model)| (*seat, model.clone()))
        .collect();
    notices.extend(cfg.diversity_notices(&resolved_seats));

    // REQ-R27, second half (HAND-OFF from Task 2.8, which implemented this and left it with no
    // production caller): name, at STARTUP, each configured model this endpoint could not measure
    // WHILE it was measuring others. Until this line existed, a user whose fallback tag was never
    // `ollama pull`ed saw nothing at startup and found out when a mage fell — the worst possible
    // moment, and literally what the requirement exists to prevent.
    //
    // Its two conditions live inside the function: only where measurement is possible, and only if
    // at least one model of the SAME endpoint was measured. Those are what keep a cold daemon from
    // firing it over the whole pool on a first run.
    {
        let configured: Vec<String> = seats
            .iter()
            .map(|(_, _, _, model)| model.clone())
            .chain(declared_pool.iter().map(|entry| entry.model.clone()))
            .collect();
        notices.extend(missing_model_notices(measured, &configured));
    }

    // REQ-R26/SC-R51: candidates running on an ASSUMED window are announced — and nothing here
    // removes them. The assumption informs; the filtering stays entirely with magi-core, whose
    // condition #6 needs a per-consult prompt size that does not exist at this point.
    notices.extend(assumed_window_notices(measured, declared_pool));

    // REQ-R25: a model that LEFT the configuration stops being referenced. Done ONCE, here, with
    // this run's configured set. It DEGRADES rather than aborts: a failed prune leaves extra rows,
    // which is inert — the key is set membership, so an orphan row is simply never read.
    if let Some(cache) = capability_cache {
        let configured: Vec<(String, String)> = seats
            .iter()
            .map(|(_, _, _, m)| m)
            .chain(declared_pool.iter().map(|e| &e.model))
            .map(|m| (endpoint_redacted.clone(), m.clone()))
            .collect();
        let _ = cache.prune_absent(&configured);
    }

    // REQ-R28/D-R15: ONE probe per DISTINCT model. magi-core indexes capabilities by `model_id`
    // and knows nothing about endpoints, so two probes for one model are the failure mode being
    // closed — it keeps whichever answered last, in nondeterministic order. Deduplicating at
    // registration makes that unreachable instead of merely forbidden on paper.
    //
    // The map is used ONLY as a dedup set and is never iterated: a `BTreeMap` orders
    // alphabetically, so building the pool from it would silently reorder rotation preference.
    // The order that is load-bearing comes from `declared_pool` below.
    let mut probes: BTreeMap<String, Arc<CachedProbe>> = BTreeMap::new();
    if let Some(cache) = capability_cache {
        for model in seats
            .iter()
            .map(|(_, _, _, m)| m)
            .chain(declared_pool.iter().map(|e| &e.model))
        {
            probes.entry(model.clone()).or_insert_with(|| {
                // `OllamaProvider::new` is CORRECT here and wrong for a seat: every call this
                // probe makes is wrapped in `CachedProbe`'s own ceiling, so the type's 300 s
                // default never governs. A seat has nothing outside it to cut the request short.
                let source: Option<Arc<dyn ProviderProbe>> = if kind.is_probeable() {
                    OllamaProvider::new(base.as_str(), model.clone())
                        .ok()
                        .map(|p| Arc::new(p) as Arc<dyn ProviderProbe>)
                } else {
                    None
                };
                Arc::new(CachedProbe::new(
                    Arc::clone(cache),
                    endpoint_redacted.clone(),
                    model.clone(),
                    source,
                ))
            });
        }
    }

    // SC-R52, second half: a pool entry that repeats a model a SEAT already declares is
    // announced. It is NOT pruned — `used` is per mage (`rotation.rs:213`), so another seat can
    // still rotate into it, and dropping it would take a usable candidate away from the other
    // two. But it is almost certainly a copy-paste, and left silent it surfaces only as an
    // unexpectedly short rotation chain during a real incident.
    {
        let seat_models: std::collections::BTreeSet<&str> =
            seats.iter().map(|(_, _, _, m)| m.as_str()).collect();
        for entry in declared_pool {
            if seat_models.contains(entry.model.as_str()) {
                notices.push(Notice::info(format!(
                    "notice: the fallback candidate `{}` repeats a model one of the mages already \
                     runs. It stays in the pool — another seat can still rotate into it — but it \
                     buys that seat nothing.",
                    entry.model
                )));
            }
        }
    }

    // REQ-R01 + SC-R55. Exactly ONE registration call per seat: `with_agent` after
    // `with_agent_and_probe` for the same seat would DISCARD the probe in silence
    // (`orchestrator.rs:309-320`), and nothing about that failure is visible until a rotation
    // needs the measurement — the worst possible moment.
    for (seat, provider, lineage, model) in seats {
        // REQ-R28/SC-R56: the probe and the completions provider are declared APART, so keeping
        // them pointed at the same model is the caller's job. magi-core checks it too and warns —
        // but through `tracing::warn!`, and magi-rs has no subscriber, so that event is emitted
        // into the void. The comparison is ours to make and costs nothing: both names are in hand
        // here, and it is a string equality with no I/O.
        //
        // It NEVER rejects, for the same reason the crate does not: a probe is not authoritative
        // over which model a provider serves. But a mis-pointed probe files the window under
        // another model's name AND feeds the digest collision check, which is the one fail-closed
        // direction in this subsystem — it can reject a candidate that was healthy.
        if let Some(probe) = probes.get(&model) {
            if probe.declared_model() != Some(provider.model()) {
                notices.push(Notice::resolution(format!(
                    "notice: the probe registered for {seat:?} measures `{}` while its provider \
                     serves `{}`. The measurement will be filed under the wrong model; rotation \
                     still runs.",
                    probe.declared_model().unwrap_or("<unknown>"),
                    provider.model()
                )));
            }
        }
        // Recorded ON THE SAME loop that registers, for the same reason the wrap trace is
        // recorded on the branch that wraps: a guardian anchored anywhere else stops guarding.
        #[cfg(test)]
        SEAT_LINEAGE_TRACE.with(|t| t.borrow_mut().push((seat, lineage.as_str().to_owned())));
        builder = match probes.get(&model) {
            Some(probe) => builder.with_agent_and_probe(
                seat,
                provider,
                lineage,
                Arc::clone(probe) as Arc<dyn ProviderProbe>,
            ),
            None => builder.with_agent(seat, provider, lineage),
        };
    }

    // REQ-R03/R05: the SHARED rotation pool. Shared and not per-seat (D-R02) because the property
    // consensus needs — that a rotation lands on a lineage the other two seats do not hold — is a
    // property of the pool's DIVERSITY, not of who owns the list. Three lists would triple what has
    // to be declared for the same guarantee, and would add a failure mode a single pool does not
    // have: one list drying up while the other two still hold candidates.
    if !declared_pool.is_empty() {
        let mut pool = FallbackPool::builder().max_rotations(cfg.effective_max_rotations());
        #[cfg(test)]
        let mut wired: Vec<(String, String)> = Vec::new();

        for entry in declared_pool {
            // A candidate that cannot be BUILT is dropped with a notice, never fatal. Same shape
            // as REQ-R27's missing-model notice, and for the same reason: rotation is a safety
            // net, and refusing to start because the net is one strand short would deny the
            // operator the run their seats can perfectly well serve. In practice this is near
            // unreachable — candidates share endpoint, kind and client timeout with the seats, so
            // whatever breaks one breaks all three seats first, and THAT is fatal.
            let mut sink: Vec<Notice> = Vec::new();
            match build_native_provider(kind, base, &entry.model, creds, client_timeout, &mut sink)
            {
                Ok(candidate) => {
                    #[cfg(test)]
                    wired.push((entry.model.clone(), entry.lineage.as_str().to_owned()));
                    // Wrapped like the seats: `MagiBuilder::build()` wraps nothing, so a candidate
                    // pushed raw would silently be the one seat in the run without transport retry.
                    let wrapped: Arc<dyn LlmProvider> =
                        Arc::new(RetryProvider::with_config(candidate, retry.clone()));
                    let lineage = CoreLineage::from(entry.lineage.clone());
                    // Order comes from `declared_pool`, NEVER from the dedup map: it is the
                    // rotation preference, strongest to weakest.
                    pool = match probes.get(&entry.model) {
                        Some(probe) => pool.push_with_probe(
                            wrapped,
                            lineage,
                            Arc::clone(probe) as Arc<dyn ProviderProbe>,
                        ),
                        None => pool.push(wrapped, lineage),
                    };
                }
                Err(cause) => notices.push(Notice::resolution(format!(
                    "notice: fallback candidate `{}` could not be built ({cause}); \
                     rotation will not be able to use it.",
                    entry.model
                ))),
            }
        }

        #[cfg(test)]
        POOL_WIRING_TRACE.with(|t| {
            *t.borrow_mut() = Some(PoolWiring {
                candidates: wired,
                max_rotations: cfg.effective_max_rotations(),
                strict_guard: effective_guard,
            });
        });

        builder = builder
            .with_fallback_pool(pool.build())
            .with_strict_context_guard(effective_guard);
    }

    // Applied HERE, after the pool has had its say (REQ-R21): the threshold is only final once
    // the in-band candidates have been folded in, and a builder method applied earlier would
    // have captured the trio-only value.
    if let Some(warn) = warn_tokens {
        builder = builder.with_input_warn_tokens(warn);
    }

    builder
        .build()
        .map(Arc::new)
        .map_err(|e| TrioError::Builder(redact_foreign_error(&e)))
}

/// Registers the `consult` tool on `agent` ONLY IF the trio was built (REQ-A06, SC-A06a).
///
/// Shared between the TUI (`run`) and `magi query` (`run_query_subcommand`, B3): before this
/// function each had its own copy of the same `if let Some(...) { register_tool(...) }`, two
/// places that could diverge over time with nothing preventing it.
///
/// **When the trio is not buildable, the tool is NOT registered** — never halfway, and
/// never with an `execute` that fails on first use: that would waste a turn of the tool loop
/// (and a model call) to discover something already known at startup, besides inviting the
/// principal model to route to something that cannot run. `kind` - the `ProviderKind` under
/// which the trio runs (REQ-A12c): construction-time, via `ConsultTool::with_kind`, so
/// `ConsultTool::execute` does not have to resolve it again on each call. Determines whether a
/// 401/403 from `MagiReport::failed_agents` is explained as keyless configuration — see
/// `tools::consult::keyless_auth_explanation`. `magi_endpoint_diverges` -
/// `MagiConfig::magi_endpoint_diverges()`, resolved ONCE here (fix round 1, Finding 1) and
/// passed to `ConsultTool::with_magi_endpoint_diverges` — same pattern as `kind`, same reason:
/// `ConsultTool::execute` does not re-resolve it per call. `max_query_bytes` -
/// `MagiConfig::effective_max_query_bytes()` (REQ-A11b), passed to
/// `ConsultTool::with_max_query_bytes` — it is the same cap applied by the direct headless path
/// and the TUI's explicit `/consult` (SC-A11c), resolved here once. `output_cap` -
/// `MagiConfig::effective_tool_result_cap()` (REQ-A11b), passed to
/// `ConsultTool::with_output_cap` — bounds the `ToolResult` that re-enters the conversation
/// history (TUI auto-routed and `magi query`'s tool loop, the two routes that share this call
/// site).
fn register_consult_tool_if_available(
    agent: &mut Agent,
    consult_magi: Option<&Arc<Magi>>,
    auto_approve: bool,
    kind: ProviderKind,
    magi_endpoint_diverges: bool,
    max_query_bytes: usize,
    output_cap: usize,
) {
    if let Some(magi) = consult_magi {
        agent.register_tool(Box::new(
            crate::tools::consult::ConsultTool::new(magi.clone(), auto_approve)
                .with_kind(kind)
                .with_magi_endpoint_diverges(magi_endpoint_diverges)
                .with_max_query_bytes(max_query_bytes)
                .with_output_cap(output_cap),
        ));
    }
}

/// Upper bound on the gate-evaluation lines [`BufferedGateTelemetry`] keeps in memory.
///
/// **A bound, not a budget.** The sink lives for the whole process, and the TUI — the surface
/// that self-routes the most consults (`magi_rs::magi::gate`'s own module doc) — can run for
/// hours: an unbounded `Vec` there is a slow leak fed by model behaviour, which is exactly the
/// kind of growth an attacker-adjacent input should never drive. 256 is chosen so the buffer
/// costs at most a few tens of KiB while still holding far more evaluations than any single
/// session produces in practice (one line per autonomous `consult` request, capped per turn by
/// `DEFAULT_MAX_TOOL_CALLS`). Once full, further lines are DROPPED rather than rotating the
/// oldest out: the point of the sample is calibrating thresholds, and the first N evaluations
/// of a session answer that as well as the last N — while dropping is the only variant that
/// cannot silently rewrite what an earlier read already reported.
const MAX_GATE_TELEMETRY_LINES: usize = 256;

/// The production [`GateTelemetry`] sink (REQ-A20, SC-A20h).
///
/// **What the recorded sample can and cannot answer.** It only sees what the agent *chose to
/// route* to `consult`, so it answers *"of what the agent wanted to consult, how much did we
/// stop?"* — the calibration question — and it does **not** answer *"how many valuable consults
/// are we missing?"*, because a consult the agent never routed leaves no trace here. Reading it
/// as the second question is how thresholds get lowered on evidence that does not exist.
///
/// Buffers rather than writing through, because its two consumers write to destinations this
/// type must not know about and cannot hold: the headless runner appends to a `RunLog` it owns
/// mutably (and which is borrowed elsewhere while the agent future is in flight), and the TUI
/// can only write to stderr *after* leaving raw mode. Both drain it once the run is over.
struct BufferedGateTelemetry {
    /// Rendered lines, capped at [`MAX_GATE_TELEMETRY_LINES`].
    lines: Mutex<Vec<String>>,
}

impl BufferedGateTelemetry {
    /// A fresh, empty sink.
    fn new() -> Self {
        Self {
            lines: Mutex::new(Vec::new()),
        }
    }

    /// Takes every line recorded so far, leaving the sink empty.
    ///
    /// A poisoned lock is recovered from rather than propagated (same pattern as the vault's
    /// `PoisonError::into_inner` recovery): telemetry must never be the thing that fails a run.
    fn drain(&self) -> Vec<String> {
        let mut guard = self
            .lines
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::mem::take(&mut guard)
    }
}

impl magi_rs::magi::gate::GateTelemetry for BufferedGateTelemetry {
    fn on_gate_evaluation(&self, mode: &Mode, chars: usize, threshold: usize, vetoed: bool) {
        let mut guard = self
            .lines
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if guard.len() >= MAX_GATE_TELEMETRY_LINES {
            return;
        }
        // The APPLIED threshold is on BOTH sides on purpose (SC-A20h): "vetoed at 40 chars"
        // does not say whether the bar was 50 or 500, and calibrating is comparing the two.
        guard.push(format!(
            "gate: mode={mode} chars={chars} threshold={threshold} result={}",
            if vetoed { "veto" } else { "dispatch" }
        ));
    }
}

/// The operator configuration the agent's **autonomous** consult funnel needs, resolved once
/// from `magi.toml` and handed to every surface that runs an agent tool loop (REQ-A07/A07d,
/// REQ-A20/A20b, SC-A20h).
///
/// **Why it is a type and not three arguments.** `[magi.complexity]`, `[magi].default_mode`,
/// `[magi].untrusted_content` and the gate's telemetry sink only ever travel together, from
/// here to `AgentRunConfig`; three loose values would have to be threaded through two already
/// wide signatures, and the `untrusted_content` flag among them is a **security** control whose
/// omission at one call site is exactly the defect this type exists to make unrepresentable.
///
/// **Why its fields are private and it has no `Default`.** `AgentRunConfig::default()` must
/// keep meaning "interactive semantics, byte-for-byte" — that is pinned by regression tests and
/// is not negotiable — so a caller that forgets the operator's configuration silently gets a
/// safe-*looking* run with the gate on built-ins, `untrusted_content` off and no telemetry.
/// The omission is made visible one level up instead: this value cannot be conjured (no
/// `Default`, no public field literal), the only way to obtain one is
/// [`AutonomousRunConfig::from_magi_config`], which demands the loaded `MagiConfig`, and both
/// autonomous surfaces **require** one in their signature. A new surface therefore cannot
/// forget it by omission — it would have to actively construct one, and there is no
/// constructor that yields a neutered value.
#[derive(Clone)]
struct AutonomousRunConfig {
    /// `[magi.complexity]` resolved against the built-ins (REQ-A20b).
    gate_thresholds: magi_rs::magi::gate::GateThresholds,
    /// `[magi].default_mode` + `[magi].untrusted_content` (REQ-A07/A07d).
    mode_config: magi_rs::magi::mode::ModeConfig,
    /// Where every gate evaluation is recorded (SC-A20h).
    telemetry: Arc<BufferedGateTelemetry>,
}

impl AutonomousRunConfig {
    /// Resolves the operator's autonomous-consult configuration from `magi.toml`.
    ///
    /// # Parameters
    /// * `config` - the loaded, already vocabulary-validated `magi.toml`.
    ///
    /// # Returns
    /// A value carrying the effective gate thresholds, the effective mode configuration, and a
    /// fresh, empty telemetry sink.
    #[must_use]
    fn from_magi_config(config: &MagiConfig) -> Self {
        Self {
            // `config.rs` owns the disassembly of `[magi.complexity]` — the lib-side gate
            // cannot know the shape of the TOML, which is why the function lives there.
            gate_thresholds: crate::config::gate_thresholds_from(config),
            mode_config: magi_rs::magi::mode::ModeConfig {
                default_mode: config.effective_default_mode(),
                untrusted_content: config.magi().untrusted_content.unwrap_or(false),
            },
            telemetry: Arc::new(BufferedGateTelemetry::new()),
        }
    }

    /// Overlays this configuration onto `base`, which supplies every field that is a property
    /// of the **surface** (tool-call cap, observer, cancellation, system prompt) rather than of
    /// the operator's `magi.toml`.
    ///
    /// # Parameters
    /// * `base` - the surface's own run configuration, typically `AgentRunConfig::default()` for the TUI or the tier-resolved one for headless.
    ///
    /// # Returns
    /// `base` with `gate_thresholds`, `mode_config` and `gate_telemetry` replaced.
    #[must_use]
    fn apply(&self, base: crate::agent::AgentRunConfig) -> crate::agent::AgentRunConfig {
        crate::agent::AgentRunConfig {
            gate_thresholds: self.gate_thresholds,
            mode_config: self.mode_config,
            gate_telemetry: Arc::clone(&self.telemetry)
                as Arc<dyn magi_rs::magi::gate::GateTelemetry>,
            ..base
        }
    }

    /// Takes every gate evaluation recorded so far, leaving the sink empty (SC-A20h).
    ///
    /// # Returns
    /// One rendered line per evaluation, in the order they occurred.
    #[must_use]
    fn drain_telemetry(&self) -> Vec<String> {
        self.telemetry.drain()
    }
}

/// Builds the TUI's mode classifier together with the sink its one-time notices go to
/// (REQ-A07c).
///
/// **The two are returned as a pair because they must be the SAME instance.** The classifier
/// emits its cost/expiry notices through whatever sink it was constructed with; the TUI can
/// only reroute them away from stderr — where they land on top of the ratatui frame — by
/// attaching the response channel to that exact sink once it exists. Handing back two
/// independent values would compile, run, and silently keep writing over the frame, which is
/// the defect this pair exists to make unrepresentable.
///
/// # Parameters
/// * `provider` - the principal provider; the classification is one label, so it is paid at the principal's price and never at the trio's.
///
/// # Returns
/// The classifier for `TuiMagiRuntimeConfig::mode_classifier`, and the sink for its
/// `classifier_notices`.
fn tui_mode_classifier_wiring(
    provider: Arc<dyn Provider>,
) -> (
    Arc<dyn magi_rs::magi::mode::ModeClassifier>,
    Arc<crate::tui::TuiNoticeSink>,
) {
    let notices = Arc::new(crate::tui::TuiNoticeSink::new());
    let classifier: Arc<dyn magi_rs::magi::mode::ModeClassifier> =
        Arc::new(crate::agent::mode_classifier::ProviderClassifier::new(
            provider,
            Arc::clone(&notices) as Arc<dyn crate::agent::mode_classifier::NoticeSink>,
        ));
    (classifier, notices)
}

/// Builds the pair (startup notice, `/consult` message) for the TUI when the trio is not
/// buildable (REQ-A06, SC-A06b).
///
/// Exists so that the property "the startup notice and the `/consult` response say EXACTLY the
/// same thing" is testable instead of depending on two places in `run()` building the same
/// `String` on their own and staying in sync by hand.
fn trio_unavailable_for_tui(err: &TrioError) -> (Notice, String) {
    let msg = trio_unavailable_message(err);
    (Notice::blocking(msg.clone()), msg)
}

/// A [`SecretStore`] that always reports "not found" — every method fails or
/// returns empty (review round 2, C3).
///
/// Used by [`resolve_effective_embedding_endpoint`] when no real vault handle is
/// in scope. `EndpointTemplate::resolve` only ever calls `get()` when the template
/// actually declares `[user]:[password]` placeholders — a plain URL with no
/// credentials resolves without touching the vault at all (see its own doc
/// comment: "the common case does not pay even a lookup"). So substituting this stub
/// real vault is exactly the right defense: the common, credential-free case is
/// unaffected, and a template that genuinely needs a vault entry fails LOUDLY
/// (a typed [`EndpointError::MissingVaultEntry`](magi_rs::magi::endpoint::EndpointError)
/// instead of baking unresolved placeholder text into an HTTP client.
struct NoVaultInScope;

impl SecretStore for NoVaultInScope {
    fn set(&mut self, name: &str, _value: &str) -> Result<(), VaultError> {
        Err(VaultError::SecretNotFound(name.to_string()))
    }
    fn get(&mut self, name: &str) -> Result<Zeroizing<String>, VaultError> {
        Err(VaultError::SecretNotFound(name.to_string()))
    }
    fn remove(&mut self, name: &str) -> Result<(), VaultError> {
        Err(VaultError::SecretNotFound(name.to_string()))
    }
    fn list(&mut self) -> Result<Vec<SecretEntry>, VaultError> {
        Ok(Vec::new())
    }
    fn contains(&mut self, _name: &str) -> Result<bool, VaultError> {
        Ok(false)
    }
}

/// Resolves an already-parsed [`EndpointTemplate`] into a real, credential-
/// substituted [`ResolvedEndpoint`] against `secret_store` (REQ-A16c) — shared
/// by every `base_url` consumer (fix round 3, L2/C3; B3: this used to be
/// duplicated between the embedding path and the principal-provider path).
///
/// Resolves against `secret_store` when one is in scope; when it is not,
/// resolution is attempted against [`NoVaultInScope`], which fails only when
/// the template genuinely needs a vault entry — so a credential-free URL still
/// succeeds without a vault, and a template that DOES need one fails LOUDLY
/// instead of silently using unresolved placeholder text.
///
/// # Errors
/// A human-readable message naming the vault entry the template needs that the
/// current session cannot supply.
fn resolve_template(
    template: &EndpointTemplate,
    scope: Scope,
    secret_store: Option<&SharedSecretStore>,
) -> Result<ResolvedEndpoint, String> {
    let resolution = match secret_store {
        Some(ss) => {
            let mut guard = ss.lock().unwrap_or_else(|p| p.into_inner());
            template.resolve(&mut *guard, scope)
        }
        None => template.resolve(&mut NoVaultInScope, scope),
    };
    resolution.map_err(|e| {
        if secret_store.is_some() {
            format!(
                "base_url credential error: {}",
                magi_rs::redact::redact_foreign_error(&e)
            )
        } else {
            format!(
                "base_url needs vault-stored credentials, but no vault is open this \
                 session: {e}"
            )
        }
    })
}

/// Resolves the embedder's EFFECTIVE endpoint — declared, or inherited from the
/// root `base_url` — into a URL an HTTP client can use directly.
///
/// Review round 2 (C1/C2/C3): the code this replaces (a) gated inheritance on
/// `base_url.is_none()`, so a blank `Some("")` skipped it and reached the embedder
/// as an empty URL (C1); (b) discarded `effective_embedding_base_url`'s `Err` via
/// `if let Ok`, so a malformed template (e.g. a literal credential) silently fell
/// back to the Ollama default with no error and no notice (C2); (c) baked the
/// unresolved TEMPLATE text — including a literal `[user]:[password]` placeholder
/// — into the HTTP client instead of a resolved endpoint (C3). This function
/// propagates every error instead of swallowing one.
///
/// Returns a plain `String`, not a [`ResolvedEndpoint`] — this is the one
/// remaining boundary of that shape in the codebase (parked, review round 3);
/// [`resolve_effective_principal_endpoint`] keeps the redacting type instead.
///
/// # Errors
/// A human-readable message naming the problem: an invalid template
/// (`effective_embedding_base_url`'s own [`EndpointError`](magi_rs::magi::endpoint::EndpointError)),
/// or a vault entry the template needs that the current session cannot supply.
fn resolve_effective_embedding_endpoint(
    magi_config: &MagiConfig,
    secret_store: Option<&SharedSecretStore>,
) -> Result<String, String> {
    let template = magi_config.effective_embedding_base_url().map_err(|e| {
        format!(
            "embedding base_url is invalid: {}",
            magi_rs::redact::redact_foreign_error(&e)
        )
    })?;
    resolve_template(&template, Scope::Embedding, secret_store).map(|r| r.as_str().to_string())
}

/// Resolves the PRINCIPAL provider's effective endpoint: `OPENAI_BASE_URL` (if
/// non-blank) overrides the root `base_url` (declared or inherited-to-default via
/// [`MagiConfig::effective_base_url`]), then substitutes any `[user]:[password]`
/// credentials from the vault — mirroring [`resolve_effective_embedding_endpoint`]
/// for the OTHER `base_url` consumer (fix round 3, L1/L2/S1).
///
/// This replaces the old `resolve_openai_base_url` (`config.rs`, removed), which
/// had three defects this function does not: (L1) `env_base_url.map(str::to_string)`
/// returned an env value UNCONDITIONALLY, so `OPENAI_BASE_URL=""` (an
/// exported-but-unfilled CI variable) short-circuited past the TOML/default
/// fallback instead of being treated as absent (REQ-A12); (L2) it returned the
/// raw, unresolved template text — a `base_url` with `[user]:[password]`
/// placeholders reached `build_openai_provider` and `oai_creds` verbatim,
/// exactly like the embedder did before C3; (S1) callers formatted that raw
/// text directly into a user-visible notice, so a LITERAL credential (not just
/// an unresolved placeholder) printed to the TUI and stderr unredacted. This
/// function's blank-is-absent check runs on the env value BEFORE it can win the
/// precedence, and its `Result<ResolvedEndpoint, _>` return keeps the
/// redacting-capable type all the way to the call site — callers redact with
/// [`redact_url`] before ever putting the value in a notice (S1), and use
/// [`ResolvedEndpoint::as_str`] only for the actual HTTP client / `oai_creds`.
///
/// # Errors
/// Ver [`resolve_template`]; additionally, an invalid `OPENAI_BASE_URL` or root
/// `base_url` template (a literal credential, an unknown placeholder, or an
/// unparseable URL).
fn resolve_effective_principal_endpoint(
    magi_config: &MagiConfig,
    env_base_url: Option<&str>,
    secret_store: Option<&SharedSecretStore>,
) -> Result<ResolvedEndpoint, String> {
    let template = effective_root_template(magi_config, env_base_url)?;
    resolve_template(&template, Scope::Root, secret_store)
}

/// Builds the OpenAI-compatible provider's user-visible startup notice, with
/// `base_url` REDACTED (S1, fix round 3): the value handed to this function may
/// be a fully vault-resolved endpoint containing a real credential.
///
/// Pure and separately testable — `run()` (the TUI entry point) is not easily
/// unit-testable end to end, and "does the notice ever contain the secret" is
/// exactly the property that needs a test, not just a manual read of the code.
#[must_use]
fn openai_provider_info(base_url: &str, model: &str) -> String {
    format!(
        "OpenAI-compatible ({}) Model: {model}",
        redact_url(base_url)
    )
}

/// Wires an opened encrypted store into `agent` as persistent message memory
/// plus the tiered-memory subsystem (Task 12), appending any non-fatal notices
/// to `notices`. Shared by the TUI launch and the headless `query` path so the
/// memory wiring lives in exactly one place (DRY).
///
/// `embed_key` is the already-resolved `OPENAI_API_KEY` for the embedder
/// (`env > vault`); the caller resolves it so this helper stays free of most
/// secret-store plumbing. `secret_store` (review round 2, C3) is still needed
/// here — not just the resolved key — because [`resolve_effective_embedding_endpoint`]
/// may need to substitute `[user]:[password]` credentials into the embedding
/// `base_url` itself, which is a different secret than the API key.
///
/// # Errors
/// Propagates a fatal session-store error from `list_sessions`/`create_session`;
/// a failed vector-store or embedder init degrades to text-only persistence
/// with a notice, never an error (REQ-29).
async fn attach_persistent_memory(
    agent: &mut Agent,
    concrete_store: EncryptedSqliteMemory,
    magi_config: &MagiConfig,
    embed_key: Option<String>,
    secret_store: Option<&SharedSecretStore>,
    notices: &mut Vec<String>,
) -> anyhow::Result<()> {
    // Build the vector store from the shared connection + masked DEK. Errors
    // here are non-fatal: fall through without the tiered-memory subsystem
    // rather than refusing to start (REQ-29).
    let vstore_result = concrete_store
        .data_key()
        .map_err(|e| crate::memory::error::MemoryError::Crypto(e.to_string()))
        .and_then(|dek| SqliteVectorStore::new(concrete_store.shared_conn(), dek));

    // Never-delete absolute (REQ-H20 / D-H10): on-disk content is NEVER
    // auto-reset — a data-without-envelope DB fails to open with a typed
    // `DbCorrupt` instead of being wiped — so there is no reset notice.
    let memory: Arc<dyn MemoryStore> = Arc::new(concrete_store);
    let sessions = memory.list_sessions().await?;
    let session_id = if let Some((id, _)) = sessions.first() {
        id.clone()
    } else {
        memory.create_session("default").await?
    };
    agent.set_memory(memory.clone(), session_id);
    let _ = agent.load_history().await;

    // Wire the tiered-memory subsystem when the vector store initialised
    // successfully. The embedding key may be the dummy `"ollama"` for the local
    // Ollama server (it ignores auth).
    if let Ok(vstore) = vstore_result {
        // Task 1.1 (REQ-A21): `[embedding].base_url` is now optional and, when
        // absent, inherits the root `base_url` — a value `OpenAiCompatibleEmbedder`
        // cannot see on its own (it only receives `&EmbeddingConfig`, not
        // `&MagiConfig`). Resolve the EFFECTIVE, CREDENTIAL-RESOLVED endpoint here
        // (review round 2, C1/C2/C3 — see `resolve_effective_embedding_endpoint`'s
        // own doc) and bake it into the config handed to the embedder, so the
        // actual HTTP client and the "sending to X" notice below never disagree
        // about where data goes. Any error — a malformed template or a vault
        // entry the endpoint needs but this session can't supply — degrades to
        // text-only persistence with a notice, same as an embedder construction
        // failure (REQ-29): it is never silently swallowed.
        let embedder_result = resolve_effective_embedding_endpoint(magi_config, secret_store)
            .and_then(|url| {
                let mut cfg = magi_config.embedding().clone();
                cfg.base_url = Some(url);
                OpenAiCompatibleEmbedder::new(&cfg, embed_key).map_err(|e| e.to_string())
            });
        match embedder_result {
            Err(err) => {
                notices.push(format!(
                    "embedding client init failed ({err}); \
                     memory subsystem disabled (text-only persistence)"
                ));
            }
            Ok(embedder_inner) => {
                let embedder = Arc::new(embedder_inner);
                let clock = Arc::new(SystemClock);
                let vstore = Arc::new(vstore);
                let vstore_diag = Arc::clone(&vstore);
                agent.set_memory_subsystem(vstore, embedder, clock, magi_config.memory().clone());
                agent.on_session_open().await.ok();

                // CP2-AN/S: one-line diagnostics summary — never fail startup on error.
                if let Ok(d) = vstore_diag.diagnostics("root").await {
                    notices.push(format!(
                        "memory: {} active, {} archived, {} pending re-embed (~{} KB index)",
                        d.active_count,
                        d.archived_count,
                        d.pending_reembed_count,
                        d.ram_estimate_bytes / 1024,
                    ));
                }

                // CP2-AG/AJ: warn when the distiller will send memory batches to a
                // cloud embedding endpoint (non-localhost).
                //
                // Task 1.1 (REQ-A21): `embedding.base_url` is now `Option<String>` and
                // may be absent (inheriting the root `base_url`), so the EFFECTIVE
                // endpoint — declared or inherited — comes from
                // `MagiConfig::effective_embedding_base_url`, not the raw field. A
                // template parse error (REQ-A16c: a literal credential instead of the
                // `[user]:[password]` placeholders) degrades to skipping this notice —
                // it is informational, and `load()` does not yet fail closed on a bad
                // template (that validation is not part of Task 1.1's scope).
                if let Ok(embedding_url) = magi_config.effective_embedding_base_url() {
                    if magi_config.memory().distill_enabled && !is_localhost(embedding_url.as_str())
                    {
                        notices.push(format!(
                            "Memory distiller will send bounded memory batches \
                             (≤ {} tokens) to {} — set distill_enabled = false \
                             in [memory] for zero cloud memory egress.",
                            magi_config.memory().distill_max_batch_tokens,
                            embedding_url.as_str(),
                        ));
                    }
                }
            }
        }
    }

    // ProjectFactTool needs the same store; register it on the encrypted path only.
    agent.register_tool(Box::new(ProjectFactTool::new(memory.clone())));
    Ok(())
}

/// Opens the encrypted store at `db_path` with `passphrase` for a headless run,
/// mapping the vault failure to a typed [`HeadlessError`] (never wiping — a
/// wrong passphrase is retryable, REQ-H20/H23).
///
/// # Errors
/// The mapped [`HeadlessError`] behind an `EncryptedSqliteMemory::new` failure
/// (`WrongPassphrase`, `DbCorrupt`, storage, …).
fn open_headless_memory(
    db_path: &Path,
    passphrase: Zeroizing<String>,
) -> Result<Option<EncryptedSqliteMemory>, HeadlessError> {
    match EncryptedSqliteMemory::new(db_path.to_path_buf(), passphrase) {
        Ok(store) => Ok(Some(store)),
        Err(e) => Err(match e.downcast::<VaultError>() {
            Ok(ve) => HeadlessError::from(ve),
            Err(other) => HeadlessError::Storage(other.to_string()),
        }),
    }
}

/// Reads the headless input from `-i <file>` or stdin, bounded to the
/// EFFECTIVE `max_input_bytes` cap (REQ-H29, anti-DoS; spec §11 — the operator
/// may lower it below `MAX_INPUT_BYTES` via `[headless] max_input_bytes`).
///
/// # Errors
/// [`HeadlessError::Io`] on a file open or read failure, or
/// [`HeadlessError::InputTooLarge`] when the source exceeds the cap.
fn read_headless_input(
    input: Option<&Path>,
    max_input_bytes: usize,
) -> Result<Vec<u8>, HeadlessError> {
    match input {
        Some(path) => {
            let file = std::fs::File::open(path).map_err(|e| HeadlessError::Io(e.to_string()))?;
            read_input_bounded(file, max_input_bytes)
        }
        None => read_input_bounded(std::io::stdin().lock(), max_input_bytes),
    }
}

/// Maps a [`RunOutcome`] error class to a class-equivalent [`HeadlessError`] so
/// the shared headless exit taxonomy ([`headless_exit_code`]) assigns the exit
/// code (REQ-H14/H23). Only the error *class* matters here (the message is
/// already rendered), so a runtime-class kind maps to a placeholder `Io`.
fn headless_error_for_exit(kind: ErrorKind) -> HeadlessError {
    match kind {
        ErrorKind::InputInvalid => HeadlessError::InputInvalid(String::new()),
        ErrorKind::PassphraseUnavailable => HeadlessError::PassphraseUnavailable,
        // `TierDenied` is exit-mapped upstream in `exit_code_for_outcome`
        // (`EXIT_TIER_DENIED` = 3) before this function is reached; `HeadlessError`
        // has no tier-denied representation, so this explicit arm exists to make
        // the invariant visible — a tier denial must never fall into the generic
        // runtime bucket below and silently degrade to exit 1.
        ErrorKind::TierDenied => HeadlessError::Io(String::new()),
        ErrorKind::DbCorrupt
        | ErrorKind::WrongPassphrase
        | ErrorKind::Provider
        | ErrorKind::Timeout
        | ErrorKind::Runtime => HeadlessError::Io(String::new()),
    }
}

/// Process exit code when the authorization tier blocked the task (REQ-H14/H23b).
/// Mirrors the library's tier-denied code so a `TierDenied` error payload maps to
/// the same value the `stop_reason == Denied` path already yields.
const EXIT_TIER_DENIED: i32 = 3;

/// Computes the process exit code of a finished headless run through the shared
/// headless taxonomy (REQ-H23/H23b): a typed error dominates; otherwise a
/// task-blocking tier denial (empty response + `stop_reason == Denied`) ⇒ 3;
/// else 0.
fn exit_code_for_outcome(outcome: &RunOutcome) -> i32 {
    // A `TierDenied` error payload maps to the dedicated tier-denied exit code
    // (REQ-H14 taxonomy). This payload is currently unreachable — tier denials
    // flow via `stop_reason == Denied` with `error == None`, handled below — but
    // the error-payload→exit mapping is kept consistent with the taxonomy so a
    // future `TierDenied` payload never silently degrades to a generic runtime 1.
    if outcome
        .error
        .as_ref()
        .is_some_and(|e| e.kind == ErrorKind::TierDenied)
    {
        return EXIT_TIER_DENIED;
    }
    let exit_err = outcome
        .error
        .as_ref()
        .map(|e| headless_error_for_exit(e.kind));
    let response_empty = outcome.response.as_deref().is_none_or(str::is_empty);
    let tier_denied = outcome.stop_reason == StopReason::Denied;
    headless_exit_code(
        exit_err.as_ref(),
        outcome.stop_reason,
        response_empty,
        tier_denied,
    )
}

/// Finishes a `--no-clobber` write: on `write_result` failure (disk full, IO
/// error mid-`write_all`), removes the just-created `path` before surfacing
/// the error, so a failed write never leaves a PARTIAL target file on disk
/// (MAGI re-gate WARNING — mirrors the tmp+rename branch's own
/// cleanup-on-rename-failure in [`write_output_atomic`]). The `--no-clobber`
/// no-TOCTOU guarantee is unaffected: this only runs after the exclusive
/// `O_CREAT|O_EXCL` create already succeeded.
///
/// The `remove_file` result is deliberately ignored — the write error is
/// already the primary failure to report, and a cleanup that races with
/// another process/fails is best-effort, not a new error condition.
fn finish_no_clobber_write(
    path: &Path,
    write_result: std::io::Result<()>,
) -> Result<(), HeadlessError> {
    write_result.map_err(|e| {
        // A partial write must not leave a truncated target file on disk —
        // mirror the tmp+rename branch's own cleanup-on-rename-failure. The
        // `remove_file` result is intentionally ignored: the write error is
        // already the primary failure to report, and this cleanup racing
        // with another process (or itself failing) is best-effort, not a
        // new error condition.
        let _ = std::fs::remove_file(path);
        HeadlessError::Io(e.to_string())
    })
}

/// Writes `contents` to `path` atomically (REQ-H03): with `--no-clobber` an
/// atomic `O_CREAT|O_EXCL` create that refuses an existing file; otherwise a
/// temp-file + rename that overwrites in place. The parent directory must exist.
///
/// # Errors
/// [`HeadlessError::InputInvalid`] (→ exit 2) if the parent is missing or the
/// destination exists under `--no-clobber`; [`HeadlessError::Io`] on any other
/// filesystem error.
fn write_output_atomic(
    path: &Path,
    contents: &[u8],
    no_clobber: bool,
) -> Result<(), HeadlessError> {
    let parent = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    if !parent.exists() {
        return Err(HeadlessError::InputInvalid(format!(
            "output parent directory does not exist: {}",
            parent.display()
        )));
    }
    if no_clobber {
        // O_CREAT|O_EXCL atomic create — no TOCTOU check-then-write.
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
        {
            Ok(mut f) => finish_no_clobber_write(path, f.write_all(contents)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                Err(HeadlessError::InputInvalid(format!(
                    "output file already exists (--no-clobber): {}",
                    path.display()
                )))
            }
            Err(e) => Err(HeadlessError::Io(e.to_string())),
        }
    } else {
        let tmp = parent.join(format!(".magi-out.tmp.{:016x}", rand::random::<u64>()));
        std::fs::write(&tmp, contents).map_err(|e| HeadlessError::Io(e.to_string()))?;
        std::fs::rename(&tmp, path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            HeadlessError::Io(e.to_string())
        })
    }
}

/// Serializes `outcome` in the requested format to the destination selected by
/// `h` — a buffered atomic `-o <file>` (REQ-H03), or stdout (JSON buffered, text
/// streamed; clamp notices always go to stderr, REQ-H13/H14).
///
/// `tool_result_cap` is the EFFECTIVE per-tool-result byte cap (spec §11) — an
/// operator-lowered `[headless] tool_result_cap_bytes` reaches the JSON
/// truncation from here.
///
/// # Errors
/// [`HeadlessError`] on a serialization or output-write failure.
fn write_headless_output(
    h: &HeadlessArgs,
    outcome: &RunOutcome,
    out_json: bool,
    tool_result_cap: usize,
) -> Result<(), HeadlessError> {
    if let Some(path) = &h.output {
        // With `-o` the output is BUFFERED (never streamed to a file) and then
        // written atomically (REQ-H13 reconciled with REQ-H03).
        let mut buf: Vec<u8> = Vec::new();
        if out_json {
            write_json(&mut buf, outcome, tool_result_cap)?;
        } else {
            write_text(&mut buf, &mut std::io::stderr(), outcome)?;
        }
        write_output_atomic(path, &buf, h.no_clobber)
    } else {
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        if out_json {
            write_json(&mut out, outcome, tool_result_cap)?;
        } else {
            write_text(&mut out, &mut std::io::stderr(), outcome)?;
        }
        out.flush().map_err(|e| HeadlessError::Io(e.to_string()))
    }
}

/// Emits `outcome` in the requested format and returns the process exit code
/// (shared by `query` and `consult`). An output-write failure is reported to
/// stderr and mapped to its own exit code.
///
/// `tool_result_cap` is the EFFECTIVE per-tool-result byte cap (spec §11),
/// forwarded to [`write_headless_output`].
fn finish_headless(h: &HeadlessArgs, outcome: &RunOutcome, tool_result_cap: usize) -> i32 {
    let out_json = matches!(h.output_format, Some(CliOutputFormat::Json));
    if let Err(e) = write_headless_output(h, outcome, out_json, tool_result_cap) {
        eprintln!("error: {e}");
        return headless_error_exit_code(&e);
    }
    exit_code_for_outcome(outcome)
}

/// Resolves the effective run-log verbosity (REQ-H24, spec §11). Precedence:
/// the `--log-level` CLI flag wins; else the `[headless] log_level` config
/// string is parsed; else the default (`info`).
///
/// # Errors
/// [`HeadlessError::InputInvalid`] if `cfg` is set but is not one of
/// `error`/`warn`/`info`/`debug` — an unrecognized value is a clear typed
/// error, never a silent fallback to the default.
fn resolve_log_level(
    cli: Option<CliLogLevel>,
    cfg: Option<&str>,
) -> Result<LogLevel, HeadlessError> {
    if let Some(l) = cli {
        return Ok(l.into_lib());
    }
    match cfg {
        Some(s) => s.parse(),
        None => Ok(LogLevel::Info),
    }
}

/// Starts the JSONL run log for a headless run, or returns `None` when logging
/// is disabled (`--no-memory` without an explicit `--log-dir`, REQ-H24). A
/// start failure degrades to no logging with a stderr warning (best-effort).
///
/// `limits` supplies the EFFECTIVE `log_retention`/`log_max_bytes` caps (spec
/// §11) so an operator-lowered `[headless]` override actually governs pruning.
///
/// # Errors
/// [`HeadlessError::InputInvalid`] if `--log-level`/`[headless] log_level`
/// resolve to an invalid verbosity string (see [`resolve_log_level`]) — this
/// is a config/usage error, distinct from the best-effort log-file-open
/// degradation below.
fn build_run_log(
    h: &HeadlessArgs,
    workspace: Option<&Workspace>,
    limits: &HeadlessLimits,
    log_level_cfg: Option<&str>,
) -> Result<Option<RunLog>, HeadlessError> {
    let level = resolve_log_level(h.log_level, log_level_cfg)?;
    let logs_dir = if let Some(d) = &h.log_dir {
        Some(d.clone())
    } else if h.no_memory {
        None
    } else {
        workspace.map(Workspace::logs_dir)
    };
    let Some(dir) = logs_dir else {
        return Ok(None);
    };
    match RunLog::start(
        &dir,
        level,
        limits.log_retention_runs,
        limits.log_max_bytes,
        limits.tool_result_cap,
    ) {
        Ok(log) => Ok(Some(log)),
        Err(e) => {
            eprintln!("warning: could not start the run log ({e}); continuing without it");
            Ok(None)
        }
    }
}

/// Maps the `--auto`/`--full-auto` flags to an authorization [`Tier`]:
/// `--full-auto` wins when both are set; neither ⇒ the read-only `Default`
/// (REQ-H07/H08).
fn tier_from_flags(auto: bool, full_auto: bool) -> Tier {
    if full_auto {
        Tier::FullAuto
    } else if auto {
        Tier::Auto
    } else {
        Tier::Default
    }
}

/// The resolved run context shared by the headless `query` and `consult`
/// dispatchers: state discovery, an (optionally) opened store, the resolved
/// parameters, and a ready-built principal provider (REQ-H02).
struct HeadlessContext {
    /// The `.magi/` walk-up base and file-tool sandbox root (`-w`/cwd).
    workdir: PathBuf,
    /// The opened encrypted store (persistence + vault), if one was unlocked.
    memory: Option<EncryptedSqliteMemory>,
    /// The loaded `magi.toml` config from the discovered `.magi/`.
    magi_config: MagiConfig,
    /// The resolved principal LLM provider.
    provider: Arc<dyn Provider>,
    /// The resolved principal provider kind (Task 4.1: `ProviderKind`, not the retired legacy
    /// `"openai"`/`"anthropic"` label — the vocabulary is unified now).
    ///
    /// Read by both dispatchers' production code (MS2 gate S8 finding): `run_query_subcommand`
    /// and `run_consult_subcommand` pass it into `registered_magi_kind` alongside `magi_config`
    /// to resolve the trio's `kind` for `ConsultTool`/`MagiRuntimeParams`, the SAME way
    /// `build_magi_orchestrator` resolves it — `magi_config.effective_magi_kind()` alone is
    /// TOML-only and would ignore `MAGI_PROVIDER`. It also still verifies a property
    /// `ctx.resolved.provider` alone cannot: that the raw string actually PARSED into the
    /// REQ-A01b vocabulary, not just that it equals some literal —
    /// `test_prepare_headless_cli_provider_override_normalizes_the_new_vocabulary` and its
    /// envelope-field sibling below assert on it directly.
    provider_kind: ProviderKind,
    /// The MAGI trio built with native providers (REQ-A01), or why it couldn't be
    /// (REQ-A06). Built ONCE here — both `run_query_subcommand` and
    /// `run_consult_subcommand` need the exact same result, so this is the shared
    /// prelude's job, same as the provider/config/vault resolution above it. Task 4.3
    /// owns the polished per-surface behavior over the `Err` case.
    consult_magi: Result<Arc<Magi>, TrioError>,
    /// The resolved effective run parameters (model/provider/caps/consult).
    resolved: Resolved,
    /// The resolved user prompt.
    prompt: String,
    /// The authorization tier for this run.
    tier: Tier,
    /// The already-resolved embedding key (`env > vault`), for persistence.
    embed_key: Option<String>,
    /// The vault handle wired over `memory`, if one unlocked (review round 2,
    /// C3): `attach_persistent_memory` needs the LIVE handle, not just a
    /// resolved key, to substitute `[user]:[password]` credentials into the
    /// embedding `base_url` itself.
    secret_store: Option<SharedSecretStore>,
    /// The started run log, if logging is enabled.
    run_log: Option<RunLog>,
    /// Effective headless numeric caps for this run (spec §11), resolved once in
    /// `prepare_headless` and reused by both dispatchers.
    limits: HeadlessLimits,
    /// The envelope's own `mode` field, already resolved to a [`Mode`] (REQ-A07c).
    /// Extracted from `envelope` here, before `resolve_params` consumes it by
    /// value below — `run_query_subcommand` ignores this field (`..`); only
    /// `run_consult_subcommand` merges it with the CLI-level `--mode`.
    env_mode: Option<Mode>,
    /// The envelope's own `untrusted_content` field (REQ-A07d), defaulting to
    /// `false` when absent. Same extraction-order note as [`Self::env_mode`].
    env_untrusted_content: bool,
    /// The REQ-A07p/SC-A07p endpoint-divergence notice for THIS run, if it applies —
    /// fix round 4. Already `eprintln!`'d to stderr by `prepare_headless` (same
    /// immediate-print convention as `cfg_notices`/`trio_notices` just above it), and
    /// ALSO kept here because `prepare_headless` cannot be driven from a unit test in any way
    /// that captures its stderr (this is a real process resource, global and not
    /// parallel-test-safe to redirect), so a test asserts on this field directly
    /// instead — see `test_prepare_headless_carries_the_divergence_notice_when_it_applies`.
    ///
    /// `#[allow(dead_code)]`: no dispatcher's PRODUCTION code reads it back off `ctx` (both
    /// destructure `HeadlessContext` with `..` for this field) — it exists purely so a test can
    /// assert against the real `prepare_headless` output instead of a hand-rolled stand-in.
    #[allow(dead_code)]
    divergence_notice: Option<Notice>,
}

/// Resolves the effective `allow_system_override` gate (REQ-H12b, spec §11):
/// whether the envelope `system` override is honored. Precedence: the
/// `--allow-system-override` CLI flag OR the `[headless] allow_system_override`
/// config value — **never weaker** than either source alone, since this is a
/// SECURITY gate (a caller-controlled `system` override is a prompt-injection
/// vector unless the operator explicitly opts in via either surface).
fn resolve_allow_system_override(flag: bool, cfg: Option<bool>) -> bool {
    flag || cfg.unwrap_or(false)
}

/// Ceiling for the effective `max_input_bytes` cap (MAGI re-gate finding,
/// REQ-H29): one below `usize::MAX`. `headless::input::read_input_bounded`
/// widens the cap to `u64` and adds `1` (`max_input_bytes as u64 + 1`) to
/// distinguish "exactly at the cap" from "over the cap" without reading
/// past it; if an operator set `[headless] max_input_bytes = usize::MAX`
/// that `+1` would overflow. Clamped here — the single call site that
/// builds the effective cap from operator config — rather than in
/// `input.rs`, which trusts its caller to have already bounded the value.
const MAX_SAFE_INPUT_BYTES: usize = usize::MAX - 1;

/// Resolves the effective headless numeric caps for this run by applying the
/// `[headless]` `magi.toml` overrides over the built-in constant defaults
/// (spec §11). Each unset `[headless]` key keeps its constant default.
/// `max_input_bytes` is additionally clamped to [`MAX_SAFE_INPUT_BYTES`] so
/// the downstream `+1` in `read_input_bounded` can never overflow.
fn resolve_headless_limits(cfg: &HeadlessConfig, tool_result_cap: usize) -> HeadlessLimits {
    let d = HeadlessLimits::default();
    HeadlessLimits {
        max_input_bytes: cfg
            .max_input_bytes
            .unwrap_or(d.max_input_bytes)
            .min(MAX_SAFE_INPUT_BYTES),
        full_auto_max_tool_calls: cfg
            .full_auto_max_tool_calls
            .unwrap_or(d.full_auto_max_tool_calls),
        log_retention_runs: cfg.log_retention.unwrap_or(d.log_retention_runs),
        log_max_bytes: cfg.log_max_bytes.unwrap_or(d.log_max_bytes),
        // Passed as a parameter rather than read from `cfg`: the key moved up from `[headless]`
        // to the root level (Task 1.3, third pattern of REQ-A21b), so `MagiConfig` resolves it
        // and this function can no longer read it from its own section.
        tool_result_cap,
        full_auto_timeout_secs: cfg.timeout_secs.unwrap_or(d.full_auto_timeout_secs),
    }
}

/// Discovers state, unlocks the vault (fail-closed), loads config, reads and
/// parses the input, resolves the effective parameters, and builds the
/// principal provider — the shared prelude of `query` and `consult` (REQ-H02).
///
/// # Errors
/// Returns (via `Err`) the process exit code after reporting the failure to
/// stderr (input/discovery/passphrase/vault errors), so the dispatcher can
/// return it directly.
async fn prepare_headless(
    h: &HeadlessArgs,
    mut passphrase_flag: Option<Zeroizing<String>>,
    cwd: &Path,
    anthropic_key: Option<String>,
    openai_key: Option<String>,
) -> Result<HeadlessContext, i32> {
    let workdir = h.workdir.clone().unwrap_or_else(|| cwd.to_path_buf());

    // `.magi/` discovery (walk-up, nearest ancestor, hardened, REQ-H16/H30).
    // A discovery `Err` is security-relevant — it is the REQ-H30 rejection of a
    // symlinked/junction `.magi` component (or an IO fault during the hardened
    // walk), NOT a benign "no `.magi/` here". It is surfaced even under
    // `--no-memory`: a stateless run must still refuse to operate under a
    // symlinked `.magi` in its ancestry (there is no `--no-memory` carve-out in
    // REQ-H30). A benign ABSENCE is `Ok(None)`, handled just below — that path
    // is where `--no-memory` legitimately proceeds env-only (Docker-ephemeral,
    // SC-H24).
    let workspace = match crate::system::workspace::discover(&workdir) {
        Ok(ws) => ws,
        Err(e) => {
            eprintln!("error: {e}");
            return Err(headless_error_exit_code(&e));
        }
    };
    // A run that requires persistent state fails clearly without a `.magi/`
    // (REQ-H17); a stateless `--no-memory` run may proceed env-only.
    if workspace.is_none() && !h.no_memory {
        eprintln!(
            "error: no .magi/ state directory found in this directory or any \
             parent; run `magi init` to create one"
        );
        return Err(1);
    }

    // Open the vault only when a DB exists AND a passphrase is available
    // (REQ-H19). If persistence is requested but no passphrase can unlock the
    // existing DB, fail closed (REQ-H25) rather than silently dropping it.
    let db_path = workspace
        .as_ref()
        .map(Workspace::db_path)
        .filter(|p| p.exists());
    let memory = if let Some(db) = &db_path {
        if let Some(pass) = passphrase_flag.take() {
            match open_headless_memory(db, pass) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("error: {e}");
                    return Err(headless_error_exit_code(&e));
                }
            }
        } else if !h.no_memory {
            let e = HeadlessError::PassphraseUnavailable;
            eprintln!("error: {e}");
            return Err(headless_error_exit_code(&e));
        } else {
            None
        }
    } else {
        None
    };

    // Wire the vault over the opened store (env > vault secret resolution).
    let secret_store: Option<SharedSecretStore> = memory.as_ref().and_then(|store| {
        let dek = store.data_key().ok()?;
        let vstore = wire(store.shared_conn(), dek).ok()?;
        Some(Arc::new(Mutex::new(vstore)) as SharedSecretStore)
    });

    // Config lives in `.magi/magi.toml`; absent ⇒ built-in defaults. Task 1.4/REQ-A23:
    // `load` is fallible now — a present-but-broken magi.toml (bad parse, unknown
    // vocabulary, or a literal credential in any `base_url`) fails this run closed
    // instead of silently degrading to defaults.
    let (magi_config, cfg_notices) = match workspace.as_ref() {
        Some(ws) => match MagiConfig::load(&ws.config_path()) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("error: {e}");
                return Err(1);
            }
        },
        None => (MagiConfig::default(), Vec::new()),
    };
    for n in &cfg_notices {
        eprintln!("{n}");
    }

    // Resolved BEFORE reading input so the effective `max_input_bytes` (an
    // operator-lowered `[headless]` cap, spec §11) governs the read itself
    // rather than only the later ceiling — never the module constant alone.
    let limits = resolve_headless_limits(
        magi_config.headless(),
        magi_config.effective_tool_result_cap(),
    );

    // Read + parse the (bounded) input into an envelope (REQ-H03/H10/H29).
    let bytes = match read_headless_input(h.input.as_deref(), limits.max_input_bytes) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: {e}");
            return Err(headless_error_exit_code(&e));
        }
    };
    let forced_fmt = h.input_format.map(CliInputFormat::into_lib);
    let envelope = match parse_input(&bytes, forced_fmt) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("error: {e}");
            return Err(headless_error_exit_code(&e));
        }
    };
    let prompt = envelope.prompt.clone();
    // Extracted BEFORE `resolve_params(envelope, ...)` below consumes `envelope` by value — an
    // invalid `mode` string here is present-but-unrecognized (REQ-A12), so it fails this run
    // closed rather than being silently dropped (`resolved_mode()` never got a production
    // caller before this, so an invalid envelope `mode` was previously accepted and ignored).
    let env_mode = match envelope.resolved_mode() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            return Err(2);
        }
    };
    let env_untrusted_content = envelope.untrusted_content.unwrap_or(false);

    // Per-field defaults (toml/built-in), then resolution with the CLI
    // overrides and the operator cost ceiling (REQ-H12/H12b).
    //
    // Task 4.1: replaces `resolve_provider`/`legacy_backend_label` — the vocabulary is
    // unified now (REQ-A01b), so `MAGI_PROVIDER` gets the same explicit-error
    // treatment as `provider`/`[magi].kind` in the TOML instead of a permissive
    // pass-through onto a legacy label.
    let default_provider_kind = match resolve_effective_provider_kind(
        &magi_config,
        env::var("MAGI_PROVIDER").ok().as_deref(),
    ) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("error: {e}");
            return Err(1);
        }
    };
    let default_provider = default_provider_kind.to_string();
    // MAGI re-gate WARNING (Caspar): `default_model` must be computed for the
    // EFFECTIVE provider, not the config-default one. `resolve()` below
    // applies the envelope's `provider` and `model` overrides INDEPENDENTLY
    // (each field falls back to `defaults` on its own), so an envelope that
    // overrides only `provider` (e.g. `{"provider":"anthropic"}`, no `model`)
    // would otherwise inherit `default_model` computed for the *old* default
    // provider — a cross-provider mismatch (an Anthropic provider built with
    // an Ollama/OpenAI model name, or vice versa). Peeking `h.provider` (CLI
    // flag) and `envelope.provider` (envelope) here — with the SAME
    // precedence `resolve()` itself uses (override > envelope > default) —
    // makes `defaults.model` consistent with whichever provider will
    // actually win, without changing that precedence: an explicit envelope
    // `model` still overrides `defaults.model` unconditionally inside
    // `resolve()`, and a neither-set envelope still falls back to the
    // config-default provider's default model exactly as before.
    //
    // Task 4.1: `h.provider`/`envelope.provider` are validated against the SAME
    // vocabulary as `MAGI_PROVIDER` above — a garbage `--provider` value now fails
    // closed here instead of silently reaching `provider_kind == "openai"` unmapped
    // (the gap `legacy_backend_label`'s idempotent pass-through used to paper over).
    let effective_provider_kind = match h.provider.as_deref().or(envelope.provider.as_deref()) {
        Some(raw) => match ProviderKind::parse(raw) {
            Ok(Some(k)) => k,
            Ok(None) => default_provider_kind, // blank ⇒ absent, falls to the default
            Err(e) => {
                eprintln!("error: {e}");
                return Err(1);
            }
        },
        None => default_provider_kind,
    };
    let default_model = match effective_provider_kind {
        ProviderKind::Ollama | ProviderKind::OpenAiCompat => {
            resolve_openai_model(&magi_config, env::var("OPENAI_MODEL").ok().as_deref())
        }
        ProviderKind::Anthropic => {
            resolve_anthropic_model(&magi_config, env::var("ANTHROPIC_MODEL").ok().as_deref())
        }
    };
    let defaults = ConfigDefaults {
        model: default_model,
        provider: default_provider,
        // No `[headless]` cost/consult defaults yet; the Agent injects no system
        // message on any path, so the operator system prompt is empty.
        max_tool_calls: None,
        consult: None,
        system: String::new(),
    };
    let overrides = CliOverrides {
        model: h.model.clone(),
        provider: h.provider.clone(),
        max_tool_calls: h.max_tool_calls,
        consult: if h.consult { Some(true) } else { None },
    };
    let tier = tier_from_flags(h.auto, h.full_auto);
    // Operator ceiling: an explicit flag can RAISE it; else the `--full-auto`
    // elevation; else the toml value; else the normal cap (REQ-H08/H12b).
    // (`limits` was already resolved above, before reading the input.)
    let operator_ceiling = h
        .max_tool_calls
        .or(if h.full_auto {
            Some(limits.full_auto_max_tool_calls)
        } else {
            None
        })
        .or(defaults.max_tool_calls)
        .unwrap_or(NORMAL_MAX_TOOL_CALLS);
    let allow_system_override = resolve_allow_system_override(
        h.allow_system_override,
        magi_config.headless().allow_system_override,
    );
    let resolved = resolve_params(
        envelope,
        &defaults,
        &overrides,
        operator_ceiling,
        allow_system_override,
    );
    // Task 4.1: no more `legacy_backend_label` mutation — `resolved.provider` already
    // carries the unified REQ-A01b vocabulary end to end (whatever `h.provider`/
    // `envelope.provider`/`defaults.provider` actually said), and `provider_kind`
    // (below, used to build the ACTUAL provider) is the peek computed above from the
    // SAME override>envelope>default precedence, so the two never disagree about
    // which backend wins.
    let provider_kind = effective_provider_kind;

    // Build the principal provider from the resolved model/provider + keys.
    let provider: Arc<dyn Provider> = match provider_kind {
        ProviderKind::Ollama | ProviderKind::OpenAiCompat => {
            let api_key = resolve_openai_key(openai_key.as_deref(), secret_store.as_ref())
                .unwrap_or_else(|| "ollama".to_string());
            // Fix round 3 (L1/L2/S1): see `resolve_effective_principal_endpoint`'s doc.
            let resolved_base_url = match resolve_effective_principal_endpoint(
                &magi_config,
                env::var("OPENAI_BASE_URL").ok().as_deref(),
                secret_store.as_ref(),
            ) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("error: {e}");
                    return Err(1);
                }
            };
            let base_url = resolved_base_url.as_str().to_string();
            build_openai_provider(&base_url, &api_key, &resolved.model)
        }
        ProviderKind::Anthropic => {
            match discover_config(
                &magi_config,
                anthropic_key.as_deref(),
                secret_store.as_ref(),
            ) {
                Some(cfg) => Arc::new(AnthropicProvider::new(cfg.api_key, resolved.model.clone())),
                None => Arc::new(StaticProvider),
            }
        }
    };

    // Build the MAGI trio with native providers (REQ-A01), independent of the
    // principal provider's own availability — see the TUI `run()` block's own
    // comment for why. Built ONCE here (the shared prelude), not by each dispatcher:
    // `run_query_subcommand` and `run_consult_subcommand` need the identical result.
    let endpoints = match resolve_endpoints(
        &magi_config,
        env::var("OPENAI_BASE_URL").ok().as_deref(),
        secret_store.as_ref(),
    ) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("error: {e}");
            return Err(1);
        }
    };
    let creds = EnvVaultCredentials {
        magi_config: &magi_config,
        anthropic_env: anthropic_key.as_deref(),
        openai_env: openai_key.as_deref(),
        secret_store: secret_store.as_ref(),
    };

    // Read ONCE and shared by the probe AND the builder below — see `run()`'s comment for why
    // (sixth-pass gate finding, S8).
    let env_overrides = MagiEnvModelOverrides::from_env();

    // REQ-A24/A24b/A24c (Task 5.2): same polling as the TUI, see `run()`'s comment — never
    // blocks or fails headless startup.
    let mut trio_notices: Vec<Notice> = Vec::new();
    // Same wiring as the TUI, through the same opener (B3).
    let capability_cache = open_capability_cache(memory.as_ref(), &mut trio_notices);
    let (warn_tokens, measured) = probe_and_report(
        &magi_config,
        &endpoints,
        provider_kind,
        &OllamaProbeFactory,
        &env_overrides,
        &stateless_extra_models(&magi_config, capability_cache.as_ref()),
        &mut trio_notices,
    )
    .await;
    let consult_magi = build_magi_orchestrator(
        &TrioBuild {
            cfg: &magi_config,
            principal_kind: provider_kind,
            endpoints: &endpoints,
            creds: Some(&creds),
            warn_tokens,
            env_overrides: &env_overrides,
            capability_cache: capability_cache.as_ref(),
            measured: &measured,
        },
        &mut trio_notices,
    );
    // REQ-A07p/SC-A07p (fix round 4, finding 1): headless is the surface this notice
    // matters MOST for, not least — REQ-A07c/SC-A07f describe exactly this path (a
    // scripted `magi consult` without `--mode` pays the classification call this
    // notice warns about), and there is no human watching a TUI here to notice it
    // otherwise. Computed via the SAME `divergence_notice` `push_divergence_notice`
    // calls for the TUI (B3 — one predicate, two surfaces, never a second copy),
    // pushed into `trio_notices` so it renders through the same tier/dedup pass as
    // everything else printed below, and kept on `HeadlessContext` separately so
    // `test_prepare_headless_carries_the_divergence_notice_when_it_applies` can
    // assert against it without capturing stderr.
    let headless_divergence_notice =
        divergence_notice(&magi_config, magi_config.effective_default_mode().is_none());
    if let Some(n) = headless_divergence_notice.clone() {
        trio_notices.push(n);
    }
    for n in render_notices(trio_notices) {
        eprintln!("{n}");
    }

    let embed_key = resolve_openai_key(openai_key.as_deref(), secret_store.as_ref());
    let run_log = match build_run_log(
        h,
        workspace.as_ref(),
        &limits,
        magi_config.headless().log_level.as_deref(),
    ) {
        Ok(log) => log,
        Err(e) => {
            eprintln!("error: {e}");
            return Err(headless_error_exit_code(&e));
        }
    };

    Ok(HeadlessContext {
        workdir,
        memory,
        magi_config,
        provider,
        provider_kind,
        consult_magi,
        resolved,
        prompt,
        tier,
        embed_key,
        secret_store,
        run_log,
        limits,
        env_mode,
        env_untrusted_content,
        divergence_notice: headless_divergence_notice,
    })
}

/// Registers the sandboxed file tools on `agent` with `workdir` as the
/// `PathGuard` root (shared by the headless `query` path).
///
/// # Errors
/// Propagates a tool-construction failure (e.g. an un-canonicalizable root).
fn register_headless_tools(agent: &mut Agent, workdir: &Path) -> anyhow::Result<()> {
    let fs: Arc<dyn FileSystem> = Arc::new(RealFileSystem::new());
    agent.register_tool(Box::new(ListTool::new(fs.clone(), workdir.to_path_buf())?));
    agent.register_tool(Box::new(FileReadTool::new(
        fs.clone(),
        workdir.to_path_buf(),
    )?));
    agent.register_tool(Box::new(FileWriteTool::new(
        fs.clone(),
        workdir.to_path_buf(),
    )?));
    agent.register_tool(Box::new(GrepTool::new(
        Box::new(RipGrep::new("rg")),
        workdir.to_path_buf(),
    )?));
    agent.register_tool(Box::new(BashTool::new(workdir.to_path_buf())?));
    Ok(())
}

/// Runs `magi query`: builds the agent (provider + tools + optional persistence
/// and MAGI), drives the tier-gated tool loop through [`run_query`], and emits
/// the structured outcome. Returns the process exit code (REQ-H02/H13/H14/H23).
async fn run_query_subcommand(
    h: HeadlessArgs,
    passphrase_flag: Option<Zeroizing<String>>,
    cwd: &Path,
    anthropic_key: Option<String>,
    openai_key: Option<String>,
) -> i32 {
    let ctx = match prepare_headless(&h, passphrase_flag, cwd, anthropic_key, openai_key).await {
        Ok(c) => c,
        Err(code) => return code,
    };
    let HeadlessContext {
        workdir,
        magi_config,
        provider,
        provider_kind,
        consult_magi,
        resolved,
        prompt,
        tier,
        embed_key,
        secret_store,
        memory,
        mut run_log,
        limits,
        ..
    } = ctx;

    // Task 4.1: the trio is built ONCE, in `prepare_headless` (the shared prelude) —
    // this dispatcher only converts its `Result` to the `Option` the tool-registration
    // call below expects. Task 4.3 (REQ-A06): `trio_unavailable_message` renders the
    // SAME actionable text the TUI notice/`/consult` reply use, naming every failed
    // seat and its cause — not the bare failure count a plain `{e}` would print.
    let consult_magi: Option<Arc<Magi>> = match consult_magi {
        Ok(m) => Some(m),
        Err(e) => {
            eprintln!("note: {}", trio_unavailable_message(&e));
            None
        }
    };

    let mut agent = Agent::new(provider);

    // Persistence unless `--no-memory` (REQ-H18): notices go to stderr.
    if !h.no_memory {
        if let Some(store) = memory {
            let mut notices = Vec::new();
            if let Err(e) = attach_persistent_memory(
                &mut agent,
                store,
                &magi_config,
                embed_key,
                secret_store.as_ref(),
                &mut notices,
            )
            .await
            {
                eprintln!("error: {e}");
                return 1;
            }
            for n in notices {
                eprintln!("note: {n}");
            }
        }
    }

    if let Err(e) = register_headless_tools(&mut agent, &workdir) {
        eprintln!("error: {e}");
        return 1;
    }
    register_consult_tool_if_available(
        &mut agent,
        consult_magi.as_ref(),
        magi_config.magi().auto_approve,
        registered_magi_kind(&magi_config, provider_kind),
        magi_config.magi_endpoint_diverges(),
        magi_config.effective_max_query_bytes(),
        magi_config.effective_tool_result_cap(),
    );

    let policy = Policy::new(tier, resolved.max_tool_calls, h.timeout);
    let timeout = resolve_tier_timeout_default(&policy, limits.full_auto_timeout_secs);
    // SC-A04c/d: this route shares its single deadline with a forced/proactive consult
    // (REQ-H22), so REQ-A04's minimum applies here too. The deadline is obeyed either way;
    // what the check adds is the heads-up on stderr and the flag in the JSON.
    let timeout_decision = query_timeout_decision(timeout, consult_magi.is_some(), &magi_config);
    if let Some(w) = timeout_decision.as_ref().and_then(|d| d.warning.as_ref()) {
        eprintln!("{w}");
    }
    let wiring = crate::headless_runner::RunWiring {
        timeout,
        autonomous: AutonomousRunConfig::from_magi_config(&magi_config),
        timeout_below_formula: timeout_decision.is_some_and(|d| d.below_formula),
    };
    let outcome = run_query(
        resolved,
        policy,
        &mut agent,
        &prompt,
        &wiring,
        run_log.as_mut(),
    )
    .await;
    // SC-A20h: the run's gate evaluations reach the structured run log. Drained here rather
    // than inside `run_query` because the log is borrowed mutably by the run itself while the
    // agent future is in flight.
    if let Some(log) = run_log.as_mut() {
        for line in wiring.autonomous.drain_telemetry() {
            let _ = log.event(&magi_rs::headless::log::LogEvent::Message {
                level: LogLevel::Info,
                text: &line,
            });
        }
    }
    finish_headless(&h, &outcome, limits.tool_result_cap)
}

/// REQ-A04's coherence check for a `magi query` run that can dispatch a consult (SC-A04c/d).
///
/// **Why `magi query` needs it at all.** A forced `--consult`, or a proactive one under
/// `--auto`, runs inside the tool loop and therefore shares the run's single wall-clock
/// deadline (REQ-H22). That deadline came from the tier default or `[headless] timeout_secs`,
/// neither of which has any relation to `agent_timeout_secs` — so a run configured below
/// `classification_ceiling + 2 × agent_timeout_secs + slack` could not complete a consult whose
/// first attempt failed schema validation, and said so only as an opaque `error.kind = timeout`.
/// The identical misconfiguration on `magi consult` warned on both channels.
///
/// **The value is never overridden.** A wall-clock cap is an operator instruction, not a safety
/// invariant: someone asking for `--timeout 30` in a pipeline wants a cut at 30 seconds. What
/// was missing is the heads-up that this particular cut has a structural consequence.
///
/// # Parameters
/// * `deadline` - the deadline the tier policy resolved; `None` means unbounded, which cannot be too short.
/// * `consult_capable` - whether this run can dispatch a consult at all (a trio was built). With no trio there is no consult to be too short for, and warning would be noise.
/// * `ceiling` - the effective `[magi].agent_timeout_secs`.
///
/// # Returns
/// `None` when the check does not apply; otherwise the decision, whose `warning` names the
/// computed minimum and whose `below_formula` feeds the run's JSON.
/// The three configuration values REQ-R20's formula needs, resolved ONCE.
///
/// They travel together because they are read together at every site that asks *"how long may
/// this run take?"*, and resolving them separately at each site is exactly how two call sites end
/// up disagreeing about the same run's worst case — the failure mode being that a `--timeout`
/// computed from a stale worst case cuts off a healthy consult, and only when a rotation also
/// happened, which is very hard to reproduce.
///
/// A tuple and not a struct: the three types are distinct, so the compiler catches a
/// transposition, and the values are destructured at the single point of use.
fn timeout_scale(cfg: &MagiConfig) -> (u64, u32, bool) {
    (
        cfg.magi()
            .agent_timeout_secs
            .unwrap_or(magi_rs::magi::AGENT_TIMEOUT_SECS),
        cfg.effective_max_rotations(),
        cfg.magi().retry_disabled.unwrap_or(false),
    )
}

#[must_use]
fn query_timeout_decision(
    deadline: Option<Duration>,
    consult_capable: bool,
    cfg: &MagiConfig,
) -> Option<magi_rs::magi::TimeoutDecision> {
    if !consult_capable {
        return None;
    }
    // An unbounded run cannot be below any minimum, so there is nothing to check and nothing
    // to warn about.
    let secs = deadline?.as_secs();
    // The resolved deadline is fed in as the "asked" value whatever knob produced it — an
    // explicit `--timeout`, `[headless] timeout_secs`, or the tier default. All three are
    // operator declarations, so all three are obeyed and all three deserve the same heads-up
    // when they are structurally too short.
    let (ceiling, max_rotations, retry_disabled) = timeout_scale(cfg);
    Some(magi_rs::magi::resolve_run_timeout(
        Some(secs),
        ceiling,
        max_rotations,
        retry_disabled,
    ))
}

/// Resolves the wall-clock deadline actually enforced for a `magi consult` run
/// (SC-A04d's behavioral half, Task 6.2 Step 3c). An explicit `--timeout` is
/// obeyed verbatim — even below the formula's minimum, with `decision.warning`
/// carrying the heads-up (already wired by Task 6.1); its ABSENCE now falls back
/// to the formula-derived minimum instead of leaving the run unbounded.
///
/// `magi_rs::magi::resolve_run_timeout` already resolves and exhaustively tests
/// `effective_secs` for both cases — this is a one-line seam so the call site
/// reads as a single, auditable step instead of inlining `Duration::from_secs`
/// at the point of use.
#[must_use]
fn consult_deadline(decision: &magi_rs::magi::TimeoutDecision) -> Duration {
    Duration::from_secs(decision.effective_secs)
}

/// Runs `magi consult`: forces a direct MAGI multi-perspective analysis over the
/// prompt (no agent tool-loop) via [`run_consult`], then emits the structured
/// outcome. Returns the process exit code (REQ-H02/H21/H33).
///
/// `explicit_mode` is the caller's `Args::mode_of_consult()` (REQ-A07c): it must
/// be read from the top-level `Args` **before** `args.command.take()` empties
/// `self.command`, which is why it arrives as an already-resolved parameter
/// instead of being re-derived from `h` in here. It is merged with the
/// envelope's own `mode` field (`ctx.env_mode`), CLI winning ties — the same
/// override-over-envelope precedence `resolve_params` already uses for
/// `model`/`provider` above.
async fn run_consult_subcommand(
    h: HeadlessArgs,
    explicit_mode: Option<Mode>,
    passphrase_flag: Option<Zeroizing<String>>,
    cwd: &Path,
    anthropic_key: Option<String>,
    openai_key: Option<String>,
) -> i32 {
    let ctx = match prepare_headless(&h, passphrase_flag, cwd, anthropic_key, openai_key).await {
        Ok(c) => c,
        Err(code) => return code,
    };
    let HeadlessContext {
        magi_config,
        provider,
        provider_kind,
        consult_magi,
        resolved,
        prompt,
        mut run_log,
        limits,
        env_mode,
        env_untrusted_content,
        ..
    } = ctx;

    // Explicit `--mode` (already resolved by the caller via
    // `Args::mode_of_consult()`), or the envelope's own `mode` field — wins at
    // zero cost; its absence classifies over the PRINCIPAL provider (REQ-A07c).
    let explicit_mode = explicit_mode.or(env_mode);
    // `[magi].default_mode` (REQ-A15): fixes the lens for every invocation that
    // does not declare one, without touching this call site again.
    let configured_mode = magi_config.effective_default_mode();
    // Three surfaces can raise the guard (REQ-A07d): the CLI flag, the
    // envelope field, and the operator's `magi.toml` — any one activates it.
    let untrusted_content = h.untrusted_content
        || env_untrusted_content
        || magi_config.magi().untrusted_content.unwrap_or(false);
    // Fix round 2 (SC-A04d): ONE process-level notice sink, shared by the mode
    // classifier's own notices AND the --timeout-below-formula warning below —
    // one stderr output path, not two, and dedup is per-key so sharing cannot
    // suppress one notice because the other already fired.
    let notice_sink: Arc<dyn crate::agent::mode_classifier::NoticeSink> =
        Arc::new(crate::agent::mode_classifier::ProcessNoticeSink::default());
    let classifier =
        crate::agent::mode_classifier::ProviderClassifier::new(provider, Arc::clone(&notice_sink));
    // Task 4.1: the trio is built ONCE, in `prepare_headless` (the shared prelude); a
    // forced `magi consult` needs a LIVE trio unconditionally, so an unbuildable one
    // fails this run closed exactly as it did before (REQ-A06's polished per-surface
    // message is Task 4.3's own contract).
    // REQ-A06/SC-A06c: a forced `magi consult` with no buildable trio fails CLOSED —
    // returned BEFORE `run_consult`/`magi.analyze()` is ever reached, so no MAGI
    // object and no verdict-shaped output is ever produced for this run.
    let magi = match consult_magi {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {}", trio_unavailable_message(&e));
            return 1;
        }
    };

    // The consult path has no tier tool-gate; only its wall-clock deadline
    // bounds it (an over-cap prompt is rejected inside `run_consult`, REQ-H33).
    // `TimeoutDecision` resolves BOTH halves of SC-A04d now (Task 6.2, Step 3c):
    // `.effective_secs` is what actually gets enforced below — an explicit
    // `--timeout` is obeyed verbatim, and its ABSENCE no longer parks the run
    // forever (the pre-6.2 gap: `h.timeout.map(...)` produced `None` and
    // `analyze_direct`'s deadline arm never fired). `.below_formula`/`.warning`
    // — the JSON telemetry and the stderr notice, emitted by `analyze_direct`
    // via `runtime.notice_sink` — were already wired by Task 6.1.
    let (ceiling, max_rotations, retry_disabled) = timeout_scale(&magi_config);
    let timeout_decision =
        magi_rs::magi::resolve_run_timeout(h.timeout, ceiling, max_rotations, retry_disabled);
    let timeout = Some(consult_deadline(&timeout_decision));
    let runtime = MagiRuntimeParams {
        kind: registered_magi_kind(&magi_config, provider_kind),
        classifier: &classifier,
        configured_mode,
        untrusted_content,
        magi_config: &magi_config,
        timeout_decision,
        notice_sink: notice_sink.as_ref(),
    };
    let outcome = run_consult(
        resolved,
        magi,
        &prompt,
        timeout,
        explicit_mode,
        &runtime,
        run_log.as_mut(),
    )
    .await;
    finish_headless(&h, &outcome, limits.tool_result_cap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::messages::Message;
    use magi_rs::notices::NoticeTier;
    use magi_rs::vault::MaskedDek;

    /// Task 2.2 — `--mode` and `--untrusted-content` across the four surfaces
    /// (REQ-A07/A07b/A07c/A07d). Named `mode_surfaces` so
    /// `cargo nextest run mode_surfaces` selects exactly this group.
    ///
    /// Registered plan debt: the task brief's Step 1 code block pasted nine test
    /// bodies under this heading, but Steps 2/4 both say `PASS (3 tests)`. Four of
    /// the other six (`the_agent_that_decides_to_consult_also_picks_the_lens`,
    /// `untrusted_content_does_not_take_the_lens_away_from_the_agent`,
    /// `the_agent_alone_does_not_satisfy_the_untrusted_guard`,
    /// `configured_default_mode_beats_the_agents_choice`) call helper functions
    /// that exist nowhere in the repo and exercise behavior that needs
    /// `resolve_mode_guarded`/`ModeResolution` (Task 2.4) — still out of scope
    /// here; `src/tools/consult.rs`'s `execute` still hardcodes `Mode::Analysis`,
    /// confirming no live path can produce `ModeSource::AgentChosen` yet.
    ///
    /// The other two — `omitting_the_mode_costs_one_call_and_declaring_it_costs_none`
    /// and `the_consult_help_names_the_extra_call_and_how_to_avoid_it` — are
    /// Task 2.3's own: the direct `magi consult` path (`headless_runner::
    /// resolve_direct_mode`) now performs exactly the classification call REQ-A07c
    /// describes, so they are implemented below with the `run_consult_cli` /
    /// `render_help` helpers this heading used to say did not exist. The
    /// coordinator confirmed this reassignment; see the task report for the full
    /// six-way mapping to Task 2.3/2.4.
    mod mode_surfaces {
        use super::*;

        /// SC-A07b: the explicit wins across the four surfaces.
        ///
        /// m1 fix: the THREE labels for the THREE surfaces (CLI, envelope, TUI), not just
        /// `"design"`. With only one label covered, `"code-review"` could diverge between the
        /// CLI and `normalize_label` without any test in this group noticing — the same bug
        /// would be accepted in `magi.toml` and rejected on the command line, or vice versa.
        #[test]
        fn every_surface_accepts_an_explicit_mode() {
            use clap::Parser;

            for (label, expected) in [
                ("code-review", Mode::CodeReview),
                ("design", Mode::Design),
                ("analysis", Mode::Analysis),
            ] {
                let a = Args::parse_from(["magi-rs", "consult", "--mode", label]);
                assert_eq!(
                    a.mode_of_consult(),
                    Some(expected),
                    "CLI surface, label {label:?}"
                );

                let env_json = format!(r#"{{"prompt":"x","mode":"{label}"}}"#);
                let env = parse_input(env_json.as_bytes(), None).expect("valid envelope");
                assert_eq!(
                    env.resolved_mode().unwrap(),
                    Some(expected),
                    "envelope surface, label {label:?}"
                );

                let tui_line = format!("/consult --mode {label} this or that?");
                assert_eq!(
                    crate::tui::parse_tui_consult(&tui_line).unwrap().mode,
                    Some(expected),
                    "TUI surface, label {label:?}"
                );
            }
        }

        /// SC-A07q: an invalid `default_mode` is a configuration error.
        #[test]
        fn an_invalid_default_mode_is_a_config_error() {
            // The invalid value dies at PARSE time. `effective_default_mode` returns `Option`,
            // not `Result`, precisely so no caller can write `.ok()` (B9) — and therefore the
            // test cannot chain it with `.and_then` either.
            assert!(MagiConfig::from_toml_str("[magi]\ndefault_mode = \"banana\"\n").is_err());

            let cfg = MagiConfig::from_toml_str("[magi]\ndefault_mode = \"\"\n").unwrap();
            assert_eq!(
                cfg.effective_default_mode(),
                None,
                "empty is ABSENT, not invalid"
            );
        }

        /// SC-A07t: `untrusted_content` on three surfaces; the TUI does not have it.
        #[test]
        fn untrusted_content_is_declarable_where_the_threat_lives() {
            use clap::Parser;

            assert!(
                Args::parse_from(["magi-rs", "consult", "--untrusted-content"]).untrusted_content()
            );

            let env = parse_input(br#"{"prompt":"x","untrusted_content":true}"#, None)
                .expect("valid envelope");
            assert_eq!(
                env.untrusted_content,
                Some(true),
                "the envelope is the consumer of an automated gate: it cannot be missing"
            );

            assert!(
                crate::tui::parse_tui_consult("/consult --untrusted-content x").is_err(),
                "the TUI does not expose the flag: there is a human there who chose the content"
            );
        }

        /// Defect #12 (registered plan debt) + m1 fix: the earlier version compared clap's
        /// kebab-casing against HARDCODED literals typed by hand, which catches a clap-derive
        /// casing regression but would NOT catch `normalize_label`'s accepted vocabulary
        /// drifting away from those same three strings — the two sides could still diverge
        /// without this test noticing. The property that actually matters, and that survives
        /// either side changing, is "whatever clap emits, the shared vocabulary accepts": this
        /// reads the value straight from `ValueEnum::to_possible_value` (clap's own answer for
        /// what a variant parses from) and feeds THAT into `normalize_label`, never a literal.
        #[test]
        fn cli_mode_casing_matches_the_shared_mode_vocabulary() {
            use clap::ValueEnum;

            for variant in CliMode::value_variants() {
                let clap_emitted = variant
                    .to_possible_value()
                    .expect("a derived ValueEnum always has a possible value")
                    .get_name()
                    .to_string();
                assert_eq!(
                    magi_rs::magi::mode::normalize_label(&clap_emitted),
                    Some(variant.into_mode()),
                    "clap's own emitted value {clap_emitted:?} must be accepted by the shared \
                     vocabulary"
                );
            }
        }

        /// Edge case: no subcommand at all ⇒ neither accessor panics or fabricates a
        /// value; both report the "nothing declared" answer.
        #[test]
        fn mode_and_untrusted_content_are_absent_without_a_subcommand() {
            use clap::Parser;

            let a = Args::parse_from(["magi-rs"]);
            assert_eq!(a.mode_of_consult(), None);
            assert!(!a.untrusted_content());
        }

        /// Double of [`magi_rs::magi::mode::ModeClassifier`] that counts how many times it is
        /// invoked and always returns `label` — for SC-A07f/g, where what matters is the COUNT,
        /// not the content of the mocked response.
        struct CountingClassifier {
            /// Accumulated invocations of `classify`.
            calls: std::sync::atomic::AtomicUsize,
            /// Label that this invocation always "classifies".
            label: Mode,
        }

        impl CountingClassifier {
            /// Creates a counter at zero that will classify as `label`.
            fn new(label: Mode) -> Self {
                Self {
                    calls: std::sync::atomic::AtomicUsize::new(0),
                    label,
                }
            }

            /// How many times `classify` has been invoked so far.
            fn calls(&self) -> usize {
                self.calls.load(std::sync::atomic::Ordering::SeqCst)
            }
        }

        #[async_trait::async_trait]
        impl magi_rs::magi::mode::ModeClassifier for CountingClassifier {
            async fn classify(&self, _content: &str) -> Option<Mode> {
                self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Some(self.label)
            }
        }

        /// Parses `args` as if they were argv, and resolves the mode of the DIRECT `consult`
        /// exactly as `run_consult_subcommand` does in production: the explicit (`--mode`) wins
        /// at no cost; its absence goes through `classifier` (REQ-A07c,
        /// `headless_runner::resolve_direct_mode`).
        ///
        /// Does not build a real `Arc<Magi>`: SC-A07f/g only need the count of classification
        /// calls and the resolved mode, not a full MAGI report — raising the three mages to
        /// observe a counter would be paying the cost the gate itself exists to avoid.
        async fn run_consult_cli(
            args: &[&str],
            classifier: &dyn magi_rs::magi::mode::ModeClassifier,
            content: &str,
        ) -> Mode {
            use clap::Parser;

            let a = Args::parse_from(args);
            let explicit = a.mode_of_consult();
            crate::headless_runner::resolve_direct_mode(explicit, classifier, content).await
        }

        /// Renders the long `--help` of a headless subcommand (`"query"`/ `"consult"`), to
        /// verify that the help text documents the cost of omitting `--mode` (REQ-A19, SC-A07i)
        /// without having to launch the binary.
        fn render_help(subcommand: &str) -> String {
            use clap::CommandFactory;

            let mut cmd = Args::command();
            let sub = cmd
                .find_subcommand_mut(subcommand)
                .unwrap_or_else(|| panic!("no such subcommand: {subcommand}"));
            sub.render_long_help().to_string()
        }

        /// Task 2.3 (reassigned from 2.2, see the module header note) — SC-A07f/g: omitting
        /// `--mode` in the DIRECT `consult` costs EXACTLY one classification call; declaring it
        /// costs ZERO.
        #[tokio::test]
        async fn omitting_the_mode_costs_one_call_and_declaring_it_costs_none() {
            let counting = CountingClassifier::new(Mode::CodeReview);
            let mode = run_consult_cli(&["magi-rs", "consult"], &counting, "algo").await;
            assert_eq!(
                mode,
                Mode::CodeReview,
                "without --mode, what the classification returned is used"
            );
            assert_eq!(
                counting.calls(),
                1,
                "without --mode, classification happens exactly once"
            );

            let counting = CountingClassifier::new(Mode::CodeReview);
            let mode = run_consult_cli(
                &["magi-rs", "consult", "--mode", "design"],
                &counting,
                "algo",
            )
            .await;
            assert_eq!(mode, Mode::Design, "the explicit value is used as-is");
            assert_eq!(counting.calls(), 0, "declared ⇒ zero classification calls");
        }

        /// Task 2.3 (reassigned from 2.2) — SC-A07i: the `consult` `--help` says that omitting
        /// `--mode` adds a call to the model, and how to avoid it.
        ///
        /// A help that did not say it would be documenting a lie until this task made the cost
        /// it describes true — which is why this test could not exist before Task 2.3.
        #[test]
        fn the_consult_help_names_the_extra_call_and_how_to_avoid_it() {
            let help = render_help("consult");
            assert!(
                help.contains("extra model call"),
                "the --help must name the cost of omitting --mode: {help}"
            );
            assert!(
                help.contains("default_mode"),
                "and how to avoid it, via [magi].default_mode: {help}"
            );
        }

        // -------------------------------------------------------------------
        // Task 2.4 — `resolve_mode_guarded` and the `untrusted_content` guard, reassigned from
        // Task 2.2 (see this module's header note).
        // -------------------------------------------------------------------

        /// What is observable from resolving the mode of an AUTOROUTED `consult` — enough for
        /// the four inherited tests (SC-A07d/u/v/w).
        ///
        /// **Does not spin up a real `Agent`/`ConsultTool`.** Exercises
        /// `resolve_mode_guarded` — the production piece a real dispatch will use — with the
        /// SAME combination of parameters that dispatch would pass it for an agent-autorouted
        /// `consult` (no human-declared mode; the autonomous path does not have that level).
        /// Wiring the whole tool loop (injecting the resolved mode into `ConsultTool::execute`,
        /// the gate's veto counter) is Task 3.2's job — its own plan block states there, not
        /// here, that `ConsultTool::execute` will receive the already-resolved `(Mode,
        /// ModeSource)` pair. Building that machinery here would duplicate that task's work and
        /// risk breaking the ~15 existing `tools::consult` tests that call `execute` without
        /// injection.
        struct AgentTurnOutcome {
            /// The resolved effective mode.
            mode: Mode,
            /// Which level it came from.
            mode_source: magi_rs::magi::mode::ModeSource,
            /// Invocations of the classifier during this resolution.
            classification_calls: usize,
            /// `true` if the resolution was `Ok` (nothing blocked it).
            consult_ran: bool,
        }

        /// SC-A07d: the agent that decides to consult also decides the lens, via the `mode` in
        /// its own `input_schema` — zero classification calls.
        ///
        /// # Errors
        /// Never, in this case: `untrusted = false`.
        async fn run_turn_with_agent_chosen_mode(
            chosen: Mode,
        ) -> Result<AgentTurnOutcome, magi_rs::magi::mode::ModeError> {
            // Decoy label: if `Design` ends up being the resolved mode, the test discovers that
            // classification ran when it should not have.
            let counting = CountingClassifier::new(Mode::Design);
            let res = magi_rs::magi::mode::resolve_mode_guarded(
                magi_rs::magi::mode::ModeSources {
                    agent_chosen: Some(chosen),
                    ..magi_rs::magi::mode::ModeSources::default()
                },
                false,
                Some(&counting),
                "test content",
            )
            .await?;
            Ok(AgentTurnOutcome {
                mode: res.mode,
                mode_source: res.source,
                classification_calls: counting.calls(),
                consult_ran: true,
            })
        }

        /// SC-A07u: with `untrusted_content` active, the agent's choice still reaches — the
        /// flag blocks CLASSIFICATION (level 4), not the agent's choice (level 3).
        ///
        /// # Errors
        /// Never, in this case: the agent already chose, so the guard does not fire.
        async fn run_turn_with_untrusted_and_agent_chosen_mode(
            chosen: Mode,
        ) -> Result<AgentTurnOutcome, magi_rs::magi::mode::ModeError> {
            let counting = CountingClassifier::new(Mode::Design);
            let res = magi_rs::magi::mode::resolve_mode_guarded(
                magi_rs::magi::mode::ModeSources {
                    agent_chosen: Some(chosen),
                    ..magi_rs::magi::mode::ModeSources::default()
                },
                true,
                Some(&counting),
                "test content",
            )
            .await?;
            Ok(AgentTurnOutcome {
                mode: res.mode,
                mode_source: res.source,
                classification_calls: counting.calls(),
                consult_ran: true,
            })
        }

        /// SC-A07v: without an agent-chosen mode and without any other declaration, the flag
        /// fails closed — absent `AgentChosen` is not `Explicit`.
        ///
        /// # Errors
        /// [`magi_rs::magi::mode::ModeError::UntrustedContentRequiresExplicitMode`] always:
        /// that is exactly what this test verifies.
        async fn run_turn_with_untrusted_and_no_mode_at_all(
        ) -> Result<AgentTurnOutcome, magi_rs::magi::mode::ModeError> {
            let counting = CountingClassifier::new(Mode::Design);
            let res = magi_rs::magi::mode::resolve_mode_guarded(
                magi_rs::magi::mode::ModeSources::default(),
                true,
                Some(&counting),
                "test content",
            )
            .await?;
            Ok(AgentTurnOutcome {
                mode: res.mode,
                mode_source: res.source,
                classification_calls: counting.calls(),
                consult_ran: true,
            })
        }

        /// SC-A07w: `default_mode` beats the agent — the operator's knob for setting the lens
        /// above what the agent would choose.
        ///
        /// # Errors
        /// Never, in this case: `untrusted = false`.
        async fn run_turn_with_default_mode_and_agent_choice(
            configured: Mode,
            agent_choice: Mode,
        ) -> Result<AgentTurnOutcome, magi_rs::magi::mode::ModeError> {
            let counting = CountingClassifier::new(Mode::Design);
            let res = magi_rs::magi::mode::resolve_mode_guarded(
                magi_rs::magi::mode::ModeSources {
                    configured: Some(configured),
                    agent_chosen: Some(agent_choice),
                    ..magi_rs::magi::mode::ModeSources::default()
                },
                false,
                Some(&counting),
                "test content",
            )
            .await?;
            Ok(AgentTurnOutcome {
                mode: res.mode,
                mode_source: res.source,
                classification_calls: counting.calls(),
                consult_ran: true,
            })
        }

        /// SC-A07d — Task 2.4 (reassigned from 2.2). Verifies that the agent that decides to
        /// consult also decides the lens — the mode comes from `input_schema` (REQ-A07b), with
        /// no classification call.
        #[tokio::test]
        async fn the_agent_that_decides_to_consult_also_picks_the_lens() {
            let out = run_turn_with_agent_chosen_mode(Mode::CodeReview)
                .await
                .unwrap();
            assert_eq!(out.mode, Mode::CodeReview);
            // `AgentChosen`, NOT `Inferred`: as long as they shared the label, the
            // `untrusted_content` guard could not distinguish "the agent chose it" from "the
            // content said it", and ended up blocking both.
            assert_eq!(
                out.mode_source,
                magi_rs::magi::mode::ModeSource::AgentChosen,
                "the agent chose it, not a default"
            );
            assert_eq!(
                out.classification_calls, 0,
                "via the tool's schema: zero extra calls"
            );
        }

        /// SC-A07u — Task 2.4 (reassigned from 2.2). The flag does NOT take the lens away from
        /// the agent — it blocks level 4, not level 3.
        #[tokio::test]
        async fn untrusted_content_does_not_take_the_lens_away_from_the_agent() {
            let out = run_turn_with_untrusted_and_agent_chosen_mode(Mode::CodeReview)
                .await
                .unwrap();
            assert!(
                out.consult_ran,
                "the agent chose the lens: there is no classification to block"
            );
            assert_eq!(
                out.mode_source,
                magi_rs::magi::mode::ModeSource::AgentChosen
            );
            assert_eq!(out.classification_calls, 0);
        }

        /// SC-A07v — Task 2.4 (reassigned from 2.2). But the agent does NOT satisfy the guard
        /// on its own.
        #[tokio::test]
        async fn the_agent_alone_does_not_satisfy_the_untrusted_guard() {
            // Without an agent-chosen mode and without a declaration: the only outlet would be
            // classification, which is what the flag blocks.
            let out = run_turn_with_untrusted_and_no_mode_at_all().await;
            assert!(out.is_err(), "`AgentChosen` absent is not `Explicit`");
        }

        /// SC-A07w — Task 2.4 (reassigned from 2.2). `default_mode` beats the agent — the
        /// operator's knob for setting the lens.
        #[tokio::test]
        async fn configured_default_mode_beats_the_agents_choice() {
            let out = run_turn_with_default_mode_and_agent_choice(Mode::CodeReview, Mode::Design)
                .await
                .unwrap();
            assert_eq!(out.mode, Mode::CodeReview, "the config wins, not the agent");
            assert_eq!(out.mode_source, magi_rs::magi::mode::ModeSource::Configured);
        }

        /// SC-A07t: the JSON envelope declares the flag — it is the consumer of an automated
        /// gate, the surface where it matters most (REQ-A07d/A19). Without this surface the
        /// protection would not exist where the threat lives.
        #[tokio::test]
        async fn the_json_envelope_carries_the_flag() {
            let env = parse_input(br#"{"prompt":"x","untrusted_content":true}"#, None)
                .expect("valid envelope");
            assert_eq!(env.untrusted_content, Some(true));

            // Same resolution that `run_consult_subcommand` applies in production: the
            // envelope's `mode` as explicit, its `untrusted_content` as the flag — without a
            // declared mode, the flag must fail closed before classifying.
            let explicit = env
                .resolved_mode()
                .expect("there is no mode label to reject in this envelope");
            let untrusted = env.untrusted_content.unwrap_or(false);
            let err = magi_rs::magi::mode::resolve_mode_guarded(
                magi_rs::magi::mode::ModeSources {
                    explicit,
                    ..magi_rs::magi::mode::ModeSources::default()
                },
                untrusted,
                None,
                &env.prompt,
            )
            .await
            .expect_err("with no mode declared, the envelope's flag must fail closed");
            assert!(matches!(
                err,
                magi_rs::magi::mode::ModeError::UntrustedContentRequiresExplicitMode
            ));
        }
    }

    /// MAGI re-gate finding (Caspar/Melchior): a substring match on
    /// `"localhost"`/`"127.0.0.1"` false-matches a hostile hostname that
    /// merely contains the substring, silently suppressing the cloud-egress
    /// warning for a non-local backend.
    #[test]
    fn test_is_localhost_rejects_hostname_containing_localhost_substring() {
        assert!(!is_localhost("https://notlocalhost.evil.com"));
        assert!(!is_localhost("http://127.0.0.1.evil.com/v1"));
    }

    /// The three canonical local hosts are still recognized (exact host
    /// match, not substring).
    #[test]
    fn test_is_localhost_accepts_canonical_local_hosts() {
        assert!(is_localhost("http://localhost:11434/v1"));
        assert!(is_localhost("http://127.0.0.1:11434/v1"));
        assert!(is_localhost("http://[::1]:11434/v1"));
    }

    /// Fail-safe direction for a security-egress notice: a `base_url` whose
    /// host cannot be parsed must be treated as non-local (so the warning is
    /// shown), never as local (which would silently suppress it).
    #[test]
    fn test_is_localhost_treats_unparseable_url_as_non_local() {
        assert!(!is_localhost("not a url"));
        assert!(!is_localhost(""));
    }

    /// A `[headless]` config that sets a value overrides the constant default
    /// for every numeric cap (REQ-H08/H12b/H14/H24/H29/H34/H36, spec §11).
    #[test]
    fn test_resolve_headless_limits_applies_config_overrides() {
        let cfg = HeadlessConfig {
            max_input_bytes: Some(2048),
            full_auto_max_tool_calls: Some(30),
            log_retention: Some(7),
            log_max_bytes: Some(1024),
            timeout_secs: Some(120),
            ..Default::default()
        };
        let limits = resolve_headless_limits(&cfg, 4096);
        assert_eq!(limits.max_input_bytes, 2048);
        assert_eq!(limits.full_auto_max_tool_calls, 30);
        assert_eq!(limits.log_retention_runs, 7);
        assert_eq!(limits.log_max_bytes, 1024);
        assert_eq!(limits.tool_result_cap, 4096);
        assert_eq!(limits.full_auto_timeout_secs, 120);
    }

    /// An empty `[headless]` config keeps every built-in constant default.
    #[test]
    fn test_resolve_headless_limits_defaults_when_unset() {
        let limits = resolve_headless_limits(
            &HeadlessConfig::default(),
            HeadlessLimits::default().tool_result_cap,
        );
        assert_eq!(limits, HeadlessLimits::default());
    }

    /// MAGI re-gate finding (Balthasar): `headless::input::read_input_bounded`
    /// computes `max_input_bytes as u64 + 1` downstream; an operator-set
    /// `[headless] max_input_bytes = usize::MAX` must never reach that call
    /// site unclamped, or the `+1` overflows. `resolve_headless_limits` is the
    /// single place the effective cap is built from operator config, so the
    /// clamp lives here: `usize::MAX` is capped down to `usize::MAX - 1`.
    #[test]
    fn test_resolve_headless_limits_clamps_max_input_bytes_below_usize_max() {
        let cfg = HeadlessConfig {
            max_input_bytes: Some(usize::MAX),
            ..Default::default()
        };
        let limits = resolve_headless_limits(&cfg, HeadlessLimits::default().tool_result_cap);
        assert_eq!(
            limits.max_input_bytes,
            usize::MAX - 1,
            "usize::MAX must be clamped so the downstream +1 cannot overflow"
        );
    }

    /// A normal operator value passes through the clamp unchanged.
    #[test]
    fn test_resolve_headless_limits_leaves_normal_max_input_bytes_unclamped() {
        let cfg = HeadlessConfig {
            max_input_bytes: Some(2048),
            ..Default::default()
        };
        let limits = resolve_headless_limits(&cfg, HeadlessLimits::default().tool_result_cap);
        assert_eq!(limits.max_input_bytes, 2048);
    }

    /// `read_headless_input` enforces the EFFECTIVE cap passed in, not the
    /// `MAX_INPUT_BYTES` module constant — an operator-lowered `[headless]
    /// max_input_bytes` must reject an oversized `-i` file well below the
    /// 10 MiB default (REQ-H29, spec §11).
    #[test]
    fn test_read_headless_input_enforces_custom_effective_cap_from_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("prompt.txt");
        std::fs::write(&path, vec![b'x'; 11]).expect("write fixture");

        let small_cap = 10usize;
        let err = read_headless_input(Some(&path), small_cap)
            .expect_err("11 bytes must exceed the custom 10-byte cap");
        assert!(matches!(err, HeadlessError::InputTooLarge(limit) if limit == small_cap));

        // Exactly at the (smaller) cap is accepted.
        let path_ok = dir.path().join("prompt_ok.txt");
        std::fs::write(&path_ok, vec![b'x'; small_cap]).expect("write fixture");
        let bytes = read_headless_input(Some(&path_ok), small_cap).expect("must fit exactly");
        assert_eq!(bytes.len(), small_cap);
    }

    /// REQ-H24, spec §11: with no `--log-level` CLI flag, the
    /// `[headless] log_level` config string must resolve the run-log
    /// verbosity, not silently fall back to the `info` default.
    #[test]
    fn test_resolve_log_level_config_wins_over_default_without_cli_flag() {
        let level = resolve_log_level(None, Some("debug"))
            .expect("a valid config string must resolve, not error");
        assert_eq!(
            level,
            LogLevel::Debug,
            "the [headless] log_level config value must take effect, not the info default"
        );
    }

    /// The `--log-level` CLI flag wins over a conflicting config value.
    #[test]
    fn test_resolve_log_level_cli_flag_wins_over_config() {
        let level = resolve_log_level(Some(CliLogLevel::Error), Some("debug"))
            .expect("must resolve with both sources present");
        assert_eq!(level, LogLevel::Error);
    }

    /// No CLI flag and no config ⇒ the `info` default.
    #[test]
    fn test_resolve_log_level_defaults_to_info_when_unset() {
        assert_eq!(
            resolve_log_level(None, None).expect("must resolve"),
            LogLevel::Info
        );
    }

    /// An invalid `[headless] log_level` string is a clear typed error, never
    /// a silent fallback.
    #[test]
    fn test_resolve_log_level_rejects_invalid_config_string() {
        assert!(matches!(
            resolve_log_level(None, Some("verbose")),
            Err(HeadlessError::InputInvalid(_))
        ));
    }

    /// REQ-H12b, spec §11: SECURITY gate — `[headless] allow_system_override`
    /// must be able to enable the envelope `system` override even without the
    /// `--allow-system-override` CLI flag (the effective value is
    /// `flag OR config`, never weaker than either source alone).
    #[test]
    fn test_resolve_allow_system_override_config_enables_without_cli_flag() {
        assert!(
            resolve_allow_system_override(false, Some(true)),
            "a true [headless] allow_system_override must enable the gate on its own"
        );
    }

    /// The CLI flag alone (no config) still enables the gate.
    #[test]
    fn test_resolve_allow_system_override_cli_flag_alone_enables() {
        assert!(resolve_allow_system_override(true, None));
        assert!(resolve_allow_system_override(true, Some(false)));
    }

    /// Neither source set ⇒ the gate stays closed (secure default, REQ-H12b).
    #[test]
    fn test_resolve_allow_system_override_defaults_closed() {
        assert!(!resolve_allow_system_override(false, None));
        assert!(!resolve_allow_system_override(false, Some(false)));
    }

    /// MAGI re-gate finding (Caspar): the "no magi.toml" default-notice check
    /// must consult the canonical `.magi/magi.toml` (REQ-H16/H17), not a loose
    /// `<cwd>/magi.toml`. A workspace whose `.magi/magi.toml` exists reports
    /// `true`; no discovered workspace at all reports `false`.
    #[test]
    fn test_magi_toml_exists_checks_canonical_dot_magi_path() {
        let tmp = tempfile::tempdir().unwrap();
        let cwd = dunce::canonicalize(tmp.path()).unwrap();

        assert!(!magi_toml_exists(None), "no workspace ⇒ no config file");

        let magi_dir = cwd.join(".magi");
        std::fs::create_dir(&magi_dir).unwrap();
        std::fs::write(magi_dir.join("magi.toml"), "provider = \"anthropic\"\n").unwrap();
        let ws = Workspace {
            root: cwd.clone(),
            magi_dir: magi_dir.clone(),
        };
        assert!(magi_toml_exists(Some(&ws)));
    }

    /// A loose `magi.toml` sitting directly in cwd (the legacy pre-headless
    /// layout, REQ-H17) must NOT count, even though `.magi/` exists without
    /// its own config file — regression test for the bug where the call site
    /// checked `workspace_root.join("magi.toml")` instead of
    /// `ws.config_path()`.
    #[test]
    fn test_magi_toml_exists_ignores_legacy_loose_cwd_file() {
        let tmp = tempfile::tempdir().unwrap();
        let cwd = dunce::canonicalize(tmp.path()).unwrap();
        std::fs::write(cwd.join("magi.toml"), "provider = \"anthropic\"\n").unwrap();

        let magi_dir = cwd.join(".magi");
        std::fs::create_dir(&magi_dir).unwrap();
        let ws = Workspace {
            root: cwd.clone(),
            magi_dir,
        };
        assert!(!magi_toml_exists(Some(&ws)));
    }

    /// RAII env-var guard (no `temp_env` dep); restores the prior value on
    /// drop. Mirrors `vault::master`'s test helper — needed here too because
    /// `resolve_master_passphrase`/`discover_config`/`resolve_openai_key`
    /// read process-global env vars, and tests within this same binary run
    /// in parallel by default.
    fn with_var<R>(key: &str, val: Option<&str>, f: impl FnOnce() -> R) -> R {
        struct Guard {
            key: String,
            prev: Option<String>,
        }
        impl Drop for Guard {
            fn drop(&mut self) {
                match &self.prev {
                    Some(v) => std::env::set_var(&self.key, v),
                    None => std::env::remove_var(&self.key),
                }
            }
        }
        let _g = Guard {
            key: key.to_string(),
            prev: std::env::var(key).ok(),
        };
        match val {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
        f()
    }

    /// Scripted [`PassphrasePrompt`] double (R-V07: no real TTY in tests).
    struct FakePrompt {
        interactive: bool,
        answers: Vec<String>,
        reads: usize,
    }
    impl FakePrompt {
        fn interactive(answers: Vec<&str>) -> Self {
            Self {
                interactive: true,
                answers: answers.into_iter().map(str::to_string).collect(),
                reads: 0,
            }
        }
        fn non_interactive() -> Self {
            Self {
                interactive: false,
                answers: Vec::new(),
                reads: 0,
            }
        }
    }
    impl PassphrasePrompt for FakePrompt {
        fn is_interactive(&self) -> bool {
            self.interactive
        }
        fn read_passphrase(
            &mut self,
            _msg: &str,
            _show: bool,
        ) -> Result<Zeroizing<String>, VaultError> {
            let i = self.reads;
            self.reads += 1;
            Ok(Zeroizing::new(
                self.answers.get(i).cloned().unwrap_or_default(),
            ))
        }
    }

    /// An in-memory [`SharedSecretStore`] fixture for the discovery tests.
    fn vault_fixture() -> SharedSecretStore {
        let conn = rusqlite::Connection::open_in_memory().expect("mem db");
        let dek = MaskedDek::new(Zeroizing::new(vec![3u8; 32])).expect("32B");
        let store = wire(Arc::new(Mutex::new(conn)), dek).expect("wire");
        Arc::new(Mutex::new(store)) as SharedSecretStore
    }

    /// Wiring smoke test (Task 6, updated Task 4.1): env > TOML > default
    /// (Ollama-first). Pure resolution; no side effects. `resolve_provider` (the
    /// legacy-label-normalizing shim this test used to exercise) is retired — the
    /// unified `resolve_effective_provider_kind` is covered directly in
    /// `config.rs`'s own tests; this one just pins that `main.rs` sees the SAME
    /// resolution its callers depend on.
    #[test]
    fn test_resolve_effective_provider_kind_wiring() {
        assert_eq!(
            resolve_effective_provider_kind(
                &MagiConfig::builder()
                    .provider(Some("anthropic".into()))
                    .build()
                    .unwrap(),
                Some("ollama")
            )
            .unwrap(),
            ProviderKind::Ollama
        );
        assert_eq!(
            resolve_effective_provider_kind(&MagiConfig::default(), None).unwrap(),
            ProviderKind::Ollama
        );
    }

    /// SC-A22: `--init-config` was retired (REQ-A22). Fix round 2 (coordinator,
    /// 2026-08-03, m4/m5): the flag must parse CLEANLY as a plain bool — no clap-level
    /// rejection — because a clap `value_parser` error over a synthetic
    /// `default_missing_value` renders `error: invalid value 'retired' for
    /// '--init-config <INIT_CONFIG>': ...`, a message whose entire purpose is to not
    /// make the user think, opening with a token they never typed. The retirement
    /// message is `run`'s job now, printed at runtime before anything else — this test
    /// pins the EXACT text the user sees, not just that "magi init" appears somewhere
    /// in whatever clap happened to render.
    #[test]
    fn init_config_flag_shows_a_clean_retirement_message_not_a_synthetic_clap_error() {
        use clap::Parser;
        let args = Args::try_parse_from(["magi-rs", "--init-config"])
            .expect("the flag must still parse — clap-level rejection is what this fixes");
        assert!(args.init_config);
        assert_eq!(
            init_config_retired_message(),
            "`--init-config` was retired; run `magi init` instead."
        );
        // The default (flag absent) path is unaffected.
        let default_args = Args::try_parse_from(["magi-rs"]).expect("default parse");
        assert!(!default_args.init_config);
    }

    #[test]
    fn test_first_run_strips_trailing_newline_from_flag_and_env_like_unlock() {
        // Loop-2 S1 (Balthasar): first-run creation must normalize -p/env the
        // SAME way unlock (resolve_passphrase) does. Otherwise
        // `MAGI_PASSPHRASE=$(cat f)` (trailing \n) creates a KEK from "...\n"
        // that unlock — which strips — can never reproduce => permanent lockout.
        let mut prompt = FakePrompt::non_interactive();
        let flag = Some(Zeroizing::new("correct horse battery staple\n".to_string()));
        let got = resolve_master_passphrase(true, flag, &mut prompt).expect("first-run flag");
        assert_eq!(got.as_str(), "correct horse battery staple");
    }

    #[test]
    fn test_args_parses_passphrase_flag_and_vault_subcommand() {
        use clap::Parser;
        let a = Args::parse_from(["magi-rs", "-p", "hunter2"]);
        assert_eq!(a.passphrase.as_deref(), Some("hunter2"));
        assert!(a.command.is_none());

        let b = Args::parse_from(["magi-rs", "vault", "ls"]);
        assert!(matches!(b.command, Some(TopCmd::Vault(VaultCmd::Ls))));

        // `-p` is global: it also parses ahead of the `vault` subcommand.
        let c = Args::parse_from(["magi-rs", "-p", "hunter2", "vault", "ls"]);
        assert_eq!(c.passphrase.as_deref(), Some("hunter2"));
        assert!(matches!(c.command, Some(TopCmd::Vault(VaultCmd::Ls))));
    }

    #[test]
    fn test_vault_error_exit_code_assigns_one_or_two_to_every_variant() {
        // Sanity check for the exhaustive match: runtime errors (incl. data
        // corruption, REQ-H23) -> 1; unexpected internal failures -> 2. Guards
        // against a silent flip during refactors (the compiler already guards
        // against a MISSING variant).
        let exit_one = [
            VaultError::Aborted,
            VaultError::WrongPassphrase,
            VaultError::PassphraseUnavailable,
            VaultError::WeakPassphrase("x".into()),
            VaultError::ValueTooLarge(1),
            VaultError::SecretNotFound("x".into()),
            VaultError::VaultMetaCorrupt,
            VaultError::DbCorrupt {
                db_path: std::path::PathBuf::from("x"),
                detail: "data present without envelope".into(),
            },
        ];
        for e in &exit_one {
            assert_eq!(vault_error_exit_code(e), 1, "{e}");
        }
        let exit_two = [
            VaultError::Crypto("x".into()),
            VaultError::Storage("x".into()),
            VaultError::Io("x".into()),
        ];
        for e in &exit_two {
            assert_eq!(vault_error_exit_code(e), 2, "{e}");
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_api_key_resolution_prefers_env_over_vault() {
        // SC-V12 / REQ-H12: the consumed env value wins; else the vault; else
        // None (StaticProvider). The env tier is now a value threaded from
        // `ConsumedSecrets`, not a live env var — so it is passed explicitly.
        // `ANTHROPIC_MODEL` is still read from the environment by
        // `discover_config`, so it is pinned absent for determinism.
        with_var("ANTHROPIC_MODEL", None, || {
            let ss = vault_fixture();
            {
                let mut guard = ss.lock().unwrap();
                guard.set("ANTHROPIC_API_KEY", "sk-from-vault").unwrap();
            }
            let config = MagiConfig::default();

            // Neither consumed env value nor vault: None.
            assert!(discover_config(&config, None, None).is_none());

            // Vault only (no consumed env value): vault wins.
            let cfg = discover_config(&config, None, Some(&ss)).expect("vault key");
            assert_eq!(cfg.api_key, "sk-from-vault");
            assert_eq!(cfg.source, "vault");

            // Both present: the consumed env value wins.
            let cfg = discover_config(&config, Some("sk-from-env"), Some(&ss)).expect("env key");
            assert_eq!(cfg.api_key, "sk-from-env");
            assert_eq!(cfg.source, "ENV");
        });
    }

    #[test]
    #[serial_test::serial]
    fn test_discover_config_falls_through_to_vault_on_a_blank_env_value() {
        // MS2 gate S8 seventh-pass finding (Balthasar): an `ANTHROPIC_API_KEY=""` exported
        // empty must be treated as ABSENT (REQ-A12) and fall through to the vault, mirroring
        // `test_resolve_openai_key_falls_through_to_vault_on_a_blank_env_value`.
        with_var("ANTHROPIC_MODEL", None, || {
            let ss = vault_fixture();
            {
                let mut guard = ss.lock().unwrap();
                guard.set("ANTHROPIC_API_KEY", "sk-from-vault").unwrap();
            }
            let config = MagiConfig::default();

            let cfg = discover_config(&config, Some(""), Some(&ss)).expect("vault key");
            assert_eq!(cfg.api_key, "sk-from-vault");
            assert_eq!(cfg.source, "vault");

            let cfg = discover_config(&config, Some("   "), Some(&ss)).expect("vault key");
            assert_eq!(cfg.api_key, "sk-from-vault");
            assert_eq!(cfg.source, "vault");
        });
    }

    #[test]
    #[serial_test::serial]
    fn test_discover_config_model_prefers_env_over_toml_then_falls_back_to_toml() {
        // MAGI re-gate WARNING fix: `discover_config` previously read only
        // `ANTHROPIC_MODEL` from the environment and never consulted
        // `[anthropic].model` at all. It must now resolve via
        // `resolve_anthropic_model` (env > TOML > default), matching the
        // precedence `resolve_openai_model` already uses.
        let ss = vault_fixture();
        {
            let mut guard = ss.lock().unwrap();
            guard.set("ANTHROPIC_API_KEY", "sk-from-vault").unwrap();
        }
        let config = MagiConfig::builder()
            .anthropic(crate::config::AnthropicConfig {
                model: Some("claude-toml-model".into()),
            })
            .build()
            .unwrap();

        // No env: the TOML model is honored (was previously ignored entirely).
        with_var("ANTHROPIC_MODEL", None, || {
            let cfg = discover_config(&config, None, Some(&ss)).expect("vault key");
            assert_eq!(cfg.model, "claude-toml-model");
        });

        // Env set: env wins over TOML.
        with_var("ANTHROPIC_MODEL", Some("claude-env-model"), || {
            let cfg = discover_config(&config, None, Some(&ss)).expect("vault key");
            assert_eq!(cfg.model, "claude-env-model");
        });
    }

    #[test]
    #[serial_test::serial]
    fn test_api_keys_are_trimmed_of_surrounding_whitespace() {
        // Loop-2 S3 (Balthasar): a key with a trailing newline (a common
        // `export KEY=$(cat f)` artifact) or stray whitespace must be trimmed,
        // else the auth header is malformed (401). Both keys, both paths — the
        // env tier is now the consumed value passed explicitly (REQ-H12/H37).
        // `ANTHROPIC_MODEL` is still read from the environment, pinned absent.
        with_var("ANTHROPIC_MODEL", None, || {
            let ss = vault_fixture();
            {
                let mut guard = ss.lock().unwrap();
                guard
                    .set("ANTHROPIC_API_KEY", "  sk-vault-anthropic\n")
                    .unwrap();
                guard.set("OPENAI_API_KEY", "sk-vault-openai\t").unwrap();
            }
            let config = MagiConfig::default();
            // Vault paths trim.
            assert_eq!(
                discover_config(&config, None, Some(&ss))
                    .expect("a")
                    .api_key,
                "sk-vault-anthropic"
            );
            assert_eq!(
                resolve_openai_key(None, Some(&ss)).as_deref(),
                Some("sk-vault-openai")
            );
            // Consumed env values trim (and win over the vault).
            assert_eq!(
                resolve_openai_key(Some("sk-env-openai\n"), Some(&ss)).as_deref(),
                Some("sk-env-openai")
            );
            assert_eq!(
                discover_config(&config, Some(" sk-env-anthropic "), Some(&ss))
                    .expect("b")
                    .api_key,
                "sk-env-anthropic"
            );
        });
    }

    #[test]
    fn test_resolve_openai_key_prefers_env_over_vault() {
        // REQ-H12: the consumed env value wins; else the vault; else None. The
        // resolver no longer reads a live env var, so this is fully isolated.
        let ss = vault_fixture();
        assert!(resolve_openai_key(None, None).is_none());
        assert!(resolve_openai_key(None, Some(&ss)).is_none());
        {
            let mut guard = ss.lock().unwrap();
            guard.set("OPENAI_API_KEY", "sk-oai-vault").unwrap();
        }
        assert_eq!(
            resolve_openai_key(None, Some(&ss)).as_deref(),
            Some("sk-oai-vault")
        );
        // The consumed env value wins over the vault entry.
        assert_eq!(
            resolve_openai_key(Some("sk-oai-env"), Some(&ss)).as_deref(),
            Some("sk-oai-env")
        );
    }

    #[test]
    fn test_resolve_openai_key_falls_through_to_vault_on_a_blank_env_value() {
        // MS2 gate S8 seventh-pass finding (Balthasar): an `OPENAI_API_KEY=""` exported empty
        // in a CI script must be treated as ABSENT (REQ-A12), the same rule `non_blank` already
        // applies to `effective_root_template` and `MagiEnvModelOverrides::from_raw` — not as a
        // literal empty key that short-circuits the vault lookup and produces an authentication
        // failure while a perfectly good stored credential sits unused.
        let ss = vault_fixture();
        {
            let mut guard = ss.lock().unwrap();
            guard.set("OPENAI_API_KEY", "sk-oai-vault").unwrap();
        }
        // Empty string and whitespace-only both fall through to the vault.
        assert_eq!(
            resolve_openai_key(Some(""), Some(&ss)).as_deref(),
            Some("sk-oai-vault")
        );
        assert_eq!(
            resolve_openai_key(Some("   "), Some(&ss)).as_deref(),
            Some("sk-oai-vault")
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_scrub_removes_passphrase_and_api_keys_from_process_env() {
        // REQ-H37: the three secrets are captured into the struct AND removed
        // from the process env (symmetric) so `/proc/<pid>/environ` cannot leak
        // them to a later in-workspace interpreter. `with_var` restores the
        // pre-test env on drop, so the scrub leaves no state between tests.
        with_var(PASSPHRASE_ENV, Some("pw-secret"), || {
            with_var(ANTHROPIC_KEY_ENV, Some("sk-anthropic"), || {
                with_var(OPENAI_KEY_ENV, Some("sk-openai"), || {
                    let consumed = read_then_scrub_secret_env();
                    // Values were captured before removal.
                    assert_eq!(
                        consumed.passphrase.as_deref().map(String::as_str),
                        Some("pw-secret")
                    );
                    assert_eq!(consumed.anthropic_key.as_deref(), Some("sk-anthropic"));
                    assert_eq!(consumed.openai_key.as_deref(), Some("sk-openai"));
                    // All three are now gone from the environment.
                    assert!(env::var(PASSPHRASE_ENV).is_err());
                    assert!(env::var(ANTHROPIC_KEY_ENV).is_err());
                    assert!(env::var(OPENAI_KEY_ENV).is_err());
                });
            });
        });
    }

    #[test]
    #[serial_test::serial]
    fn test_tool_child_env_excludes_secrets_and_arbitrary_lc() {
        // REQ-H37: the child env is a whole-name-equality allowlist, not a prefix
        // match — so secrets and an attacker-chosen `LC_EVIL` are excluded while
        // an allowlisted `LC_ALL` and `PATH` pass through. `PATH` is pinned so
        // the assertion never depends on the host's ambient environment. The
        // allowlist now lives in `crate::tools::proc_group` (single source).
        //
        // Lookups here are CASE-INSENSITIVE: Windows does not guarantee the case
        // of the env-var name it returns (a set `PATH` may come back as `Path`),
        // and the allowlist itself matches case-insensitively on Windows — so a
        // case-sensitive `get("PATH")` would spuriously miss the passed-through
        // entry on some Windows hosts (a real CI flake). Exclusion is checked
        // under any casing too, which only strengthens the secret guarantee.
        with_var(PASSPHRASE_ENV, Some("pw-secret"), || {
            with_var(ANTHROPIC_KEY_ENV, Some("sk-anthropic"), || {
                with_var(OPENAI_KEY_ENV, Some("sk-openai"), || {
                    with_var("LC_EVIL", Some("evil"), || {
                        with_var("LC_ALL", Some("C.UTF-8"), || {
                            with_var("PATH", Some("/usr/bin"), || {
                                let child = crate::tools::proc_group::tool_child_env();
                                let lookup = |key: &str| -> Option<&str> {
                                    child
                                        .iter()
                                        .find(|(k, _)| k.eq_ignore_ascii_case(key))
                                        .map(|(_, v)| v.as_str())
                                };
                                // Secrets and arbitrary `LC_*` are excluded (under any casing).
                                assert!(lookup(PASSPHRASE_ENV).is_none());
                                assert!(lookup(ANTHROPIC_KEY_ENV).is_none());
                                assert!(lookup(OPENAI_KEY_ENV).is_none());
                                assert!(lookup("LC_EVIL").is_none());
                                // Allowlisted names pass through.
                                assert_eq!(lookup("PATH"), Some("/usr/bin"));
                                assert_eq!(lookup("LC_ALL"), Some("C.UTF-8"));
                            });
                        });
                    });
                });
            });
        });
    }

    #[test]
    #[serial_test::serial]
    fn test_open_tui_memory_degrades_to_ephemeral_when_no_tty_and_no_passphrase_on_first_run() {
        with_var(PASSPHRASE_ENV, None, || {
            let tmp = tempfile::tempdir().unwrap();
            let db_path = tmp.path().join("absent.db");
            let mut prompt = FakePrompt::non_interactive();
            let mut notices = Vec::new();
            let attachment = open_tui_memory(&db_path, None, &mut prompt, &mut notices);
            assert!(matches!(attachment, MemoryAttachment::Ephemeral));
            assert!(!db_path.exists(), "must never create a DB it cannot open");
            assert!(notices.iter().any(|n| n.contains("WARNING")));
        });
    }

    #[test]
    #[serial_test::serial]
    fn test_open_tui_memory_creates_db_on_first_run_with_passphrase_flag() {
        with_var(PASSPHRASE_ENV, None, || {
            let tmp = tempfile::tempdir().unwrap();
            let db_path = tmp.path().join("fresh.db");
            let flag = Some(Zeroizing::new("correct horse battery staple".to_string()));
            let mut prompt = FakePrompt::non_interactive();
            let mut notices = Vec::new();
            let attachment = open_tui_memory(&db_path, flag, &mut prompt, &mut notices);
            assert!(matches!(attachment, MemoryAttachment::Encrypted(_)));
            assert!(db_path.exists());
        });
    }

    #[test]
    #[serial_test::serial]
    fn test_wrong_passphrase_is_retryable_and_never_wipes() {
        // Adaptation surface for SC-V09/SC-V27: an existing DB, wrong
        // passphrase then the right one via the interactive prompt, and the
        // prior history survives untouched.
        with_var(PASSPHRASE_ENV, None, || {
            let tmp = tempfile::tempdir().unwrap();
            let db_path = tmp.path().join("existing.db");
            let rt = tokio::runtime::Runtime::new().unwrap();
            {
                let store = EncryptedSqliteMemory::new(
                    db_path.clone(),
                    Zeroizing::new("right-passphrase-0123456".to_string()),
                )
                .unwrap();
                let sid = rt.block_on(store.create_session("p")).unwrap();
                rt.block_on(store.add_message(&sid, &Message::user("keep me")))
                    .unwrap();
            }

            let mut prompt =
                FakePrompt::interactive(vec!["wrong-passphrase-xyz", "right-passphrase-0123456"]);
            let mut notices = Vec::new();
            let attachment = open_tui_memory(&db_path, None, &mut prompt, &mut notices);
            match attachment {
                MemoryAttachment::Encrypted(store) => {
                    let sessions = rt.block_on(store.list_sessions()).unwrap();
                    assert_eq!(sessions.len(), 1, "prior history must survive intact");
                }
                MemoryAttachment::Ephemeral => {
                    panic!("the second (correct) attempt must recover the store")
                }
            }
        });
    }

    /// Addition 2 (coordinator ruling, 2026-08-03, Task 1.5): the module-level
    /// `notices` tests already pin that `render_notices` never trims `Resolution`/
    /// `Blocking` under the cap — what they can't see is whether `run()` itself
    /// files a genuinely important message under `Info` in the first place. Names
    /// the concrete, known-critical messages instead of re-testing the tiering
    /// machinery: a future edit that reclassifies one of these downward to `Info`
    /// breaks THIS test, not just a generic property.
    ///
    /// Covers the hardening/vault family (REQ-V42 mlock diagnostics — never
    /// diagnostic noise, always a security-posture regression) and the fixed
    /// "no persistence at all" warning.
    #[test]
    fn known_critical_startup_messages_tier_above_info() {
        let hardening =
            low_level_warning_notices(&["could not set RLIMIT_CORE=0: EPERM".to_string()]);
        assert_eq!(hardening.len(), 1);
        assert_ne!(
            hardening[0].tier,
            NoticeTier::Info,
            "a hardening warning must never degrade to Info: {}",
            hardening[0].text
        );
        assert!(hardening[0].text.contains("RLIMIT_CORE"));

        let np = no_persistence_notice();
        assert_ne!(
            np.tier,
            NoticeTier::Info,
            "the no-persistence warning must never degrade to Info: {}",
            np.text
        );
        assert!(np.text.contains("WITHOUT persistence"));
    }

    /// Same guarantee for the family of messages [`open_tui_memory`] produces — the
    /// concrete scenario mirrors
    /// `test_open_tui_memory_degrades_to_ephemeral_when_no_tty_and_no_passphrase_on_first_run`
    /// (no TTY, no passphrase, first run), fed through the SAME wrapper `run()`
    /// uses so the two can never disagree.
    #[test]
    #[serial_test::serial]
    fn open_tui_memory_notices_never_degrade_to_info() {
        with_var(PASSPHRASE_ENV, None, || {
            let tmp = tempfile::tempdir().unwrap();
            let db_path = tmp.path().join("absent.db");
            let mut prompt = FakePrompt::non_interactive();
            let mut texts = Vec::new();
            let attachment = open_tui_memory(&db_path, None, &mut prompt, &mut texts);
            assert!(matches!(attachment, MemoryAttachment::Ephemeral));

            let notices = wrap_helper_notices(texts);
            assert!(!notices.is_empty());
            for n in &notices {
                assert_ne!(
                    n.tier,
                    NoticeTier::Info,
                    "helper-produced notice degraded to Info: {}",
                    n.text
                );
            }
            assert!(notices
                .iter()
                .any(|n| n.text.contains("WITHOUT persistence")));
        });
    }

    #[tokio::test]
    async fn test_rotating_anthropic_key_in_vault_never_invalidates_the_db() {
        // Adaptation (never deletion) of the pre-vault
        // `test_agent_history_resilience_to_key_rotation`: the invariant now
        // maps onto the vault — rotating the ANTHROPIC_API_KEY stored
        // *inside* the vault table never touches the DB's own master
        // passphrase, so message history stays fully readable throughout.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let store = EncryptedSqliteMemory::new(
            tmp.path().to_path_buf(),
            Zeroizing::new("db-master-passphrase-1".to_string()),
        )
        .unwrap();
        let sid = store.create_session("p").await.unwrap();
        store
            .add_message(&sid, &Message::user("keep me"))
            .await
            .unwrap();

        let dek = store.data_key().unwrap();
        let mut vault_store = wire(store.shared_conn(), dek).unwrap();
        vault_store.set("ANTHROPIC_API_KEY", "sk-old").unwrap();
        vault_store.set("ANTHROPIC_API_KEY", "sk-rotated").unwrap(); // rotation

        let msgs = store.get_messages(&sid).await.unwrap();
        assert_eq!(msgs, vec![Message::user("keep me")]);
        assert_eq!(
            vault_store.get("ANTHROPIC_API_KEY").unwrap().as_str(),
            "sk-rotated"
        );
    }

    #[tokio::test]
    async fn test_db_moved_to_another_machine_opens_with_passphrase_alone() {
        // SC-V14: create a DB in tempdir A with a passphrase, COPY the file
        // to tempdir B, and open the copy with the SAME passphrase alone —
        // no keyring, no env, nothing else needed.
        let dir_a = tempfile::tempdir().unwrap();
        let path_a = dir_a.path().join("a.db");
        let passphrase = Zeroizing::new("portable-passphrase-xyz-9".to_string());
        {
            let store = EncryptedSqliteMemory::new(path_a.clone(), passphrase.clone()).unwrap();
            let sid = store.create_session("p").await.unwrap();
            store.add_message(&sid, &Message::user("hi")).await.unwrap();
            // Force WAL contents into the main file so copying just the
            // `.db` file (no `-wal`/`-shm` sidecars) is sufficient.
            let conn = store.shared_conn();
            let guard = conn.lock().unwrap();
            guard
                .execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")
                .unwrap();
        }

        let dir_b = tempfile::tempdir().unwrap();
        let path_b = dir_b.path().join("copy.db");
        std::fs::copy(&path_a, &path_b).unwrap();

        let store_b = EncryptedSqliteMemory::new(path_b, passphrase).unwrap();
        let sessions = store_b.list_sessions().await.unwrap();
        assert_eq!(sessions.len(), 1);
        let msgs = store_b.get_messages(&sessions[0].0).await.unwrap();
        assert_eq!(msgs, vec![Message::user("hi")]);
    }

    #[test]
    fn test_run_logout_reports_no_stored_session_when_db_absent() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path().to_path_buf();
        assert_eq!(run_logout(None, &workspace), 0);
    }

    #[test]
    fn test_args_parses_init_subcommand() {
        use clap::Parser;
        let a = Args::parse_from(["magi-rs", "init"]);
        assert!(matches!(a.command, Some(TopCmd::Init)));
        // `-p` remains global ahead of `init` (mirrors the vault subcommand).
        let b = Args::parse_from(["magi-rs", "-p", "hunter2", "init"]);
        assert_eq!(b.passphrase.as_deref(), Some("hunter2"));
        assert!(matches!(b.command, Some(TopCmd::Init)));
    }

    /// Returns how many envelope rows (`wrapped_dek`) a freshly-`init`ed DB has:
    /// `0` ⇒ envelope-less, `1` ⇒ bootstrapped. Test-only helper.
    fn envelope_row_count(db_path: &std::path::Path) -> i64 {
        let conn = rusqlite::Connection::open(db_path).expect("open db");
        conn.query_row(
            "SELECT COUNT(*) FROM vault_meta WHERE key = 'wrapped_dek'",
            [],
            |r| r.get(0),
        )
        .expect("query envelope")
    }

    #[test]
    #[serial_test::serial]
    fn test_run_init_creates_magi_dir_and_refuses_second_run() {
        // Step 5 e2e: in-process dispatch in a tempdir cwd — first init succeeds
        // and creates the full `.magi/`; a second refuses (exit != 0).
        with_var(PASSPHRASE_ENV, None, || {
            let tmp = tempfile::tempdir().unwrap();
            let cwd = dunce::canonicalize(tmp.path()).unwrap();

            assert_eq!(run_init(&cwd, None), 0, "first init must succeed");
            assert!(cwd.join(".magi").is_dir(), ".magi/ must exist after init");
            assert!(cwd.join(".magi/.magi-rs-memory.db").exists());
            assert!(cwd.join(".magi/magi.toml").exists());
            assert!(cwd.join(".magi/logs").is_dir());

            assert_ne!(
                run_init(&cwd, None),
                0,
                "a second init must refuse (nested/existing .magi/)"
            );
        });
    }

    #[test]
    #[serial_test::serial]
    fn test_run_init_without_passphrase_leaves_no_envelope() {
        // §2.2: no `-p`/env ⇒ DB left envelope-less (first interactive run
        // creates it), and it is NOT an error.
        with_var(PASSPHRASE_ENV, None, || {
            let tmp = tempfile::tempdir().unwrap();
            let cwd = dunce::canonicalize(tmp.path()).unwrap();
            assert_eq!(run_init(&cwd, None), 0);
            assert_eq!(
                envelope_row_count(&cwd.join(".magi/.magi-rs-memory.db")),
                0,
                "no passphrase must leave the DB without an envelope"
            );
        });
    }

    #[test]
    #[serial_test::serial]
    fn test_run_init_with_passphrase_bootstraps_the_envelope() {
        // §2.2: a supplied passphrase (via `-p`) bootstraps the empty vault
        // envelope so the DB is headless-ready without interaction.
        with_var(PASSPHRASE_ENV, None, || {
            let tmp = tempfile::tempdir().unwrap();
            let cwd = dunce::canonicalize(tmp.path()).unwrap();
            let pass = Some(Zeroizing::new("correct horse battery staple".to_string()));
            assert_eq!(run_init(&cwd, pass), 0);
            assert_eq!(
                envelope_row_count(&cwd.join(".magi/.magi-rs-memory.db")),
                1,
                "a supplied passphrase must bootstrap the envelope"
            );
        });
    }

    #[test]
    fn test_args_parses_vault_diagnose_subcommand() {
        use clap::Parser;
        let a = Args::parse_from(["magi-rs", "vault", "diagnose"]);
        assert!(matches!(
            a.command,
            Some(TopCmd::Vault(VaultCmd::Diagnose { names: false }))
        ));
        let b = Args::parse_from(["magi-rs", "vault", "diagnose", "--names"]);
        assert!(matches!(
            b.command,
            Some(TopCmd::Vault(VaultCmd::Diagnose { names: true }))
        ));
    }

    #[test]
    fn test_run_vault_diagnose_reports_no_database_when_absent() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace = dunce::canonicalize(tmp.path()).unwrap();
        assert_eq!(run_vault_diagnose(&workspace, false), 0);
    }

    #[test]
    #[serial_test::serial]
    fn test_run_vault_diagnose_never_needs_a_passphrase_and_finds_magi_dir() {
        // REQ-H32: diagnose must succeed WITHOUT `-p`/`MAGI_PASSPHRASE` and
        // WITHOUT ever unlocking the vault, using the discovered `.magi/` DB
        // (not the legacy cwd-relative path) once `magi init` created one.
        with_var(PASSPHRASE_ENV, None, || {
            let tmp = tempfile::tempdir().unwrap();
            let cwd = dunce::canonicalize(tmp.path()).unwrap();
            assert_eq!(run_init(&cwd, None), 0, "init must succeed");

            // No envelope yet (init ran without -p): a fresh DB.
            assert_eq!(run_vault_diagnose(&cwd, false), 0);
        });
    }

    #[test]
    #[serial_test::serial]
    fn test_run_vault_diagnose_never_mutates_or_unlocks_a_bootstrapped_db() {
        with_var(PASSPHRASE_ENV, None, || {
            let tmp = tempfile::tempdir().unwrap();
            let cwd = dunce::canonicalize(tmp.path()).unwrap();
            let pass = Some(Zeroizing::new("correct horse battery staple".to_string()));
            assert_eq!(run_init(&cwd, pass), 0, "init -p must succeed");

            let db_path = cwd.join(".magi/.magi-rs-memory.db");
            let envelope_rows_before = envelope_row_count(&db_path);
            assert_eq!(envelope_rows_before, 1, "init -p bootstraps an envelope");

            // Diagnose with a completely WRONG passphrase available in the
            // environment must still succeed (it never even looks at it).
            assert_eq!(run_vault_diagnose(&cwd, true), 0);
            assert_eq!(
                envelope_row_count(&db_path),
                envelope_rows_before,
                "diagnose must never mutate vault_meta"
            );
        });
    }

    #[test]
    #[serial_test::serial]
    fn test_run_init_refuses_to_nest_inside_ancestor_magi_dir() {
        // Step 4b: an ancestor already has a discoverable `.magi/` ⇒ refuse,
        // and create nothing under the nested cwd.
        with_var(PASSPHRASE_ENV, None, || {
            let tmp = tempfile::tempdir().unwrap();
            let root = dunce::canonicalize(tmp.path()).unwrap();
            std::fs::create_dir_all(root.join(".magi")).unwrap();
            let sub = root.join("a/b");
            std::fs::create_dir_all(&sub).unwrap();

            assert_ne!(run_init(&sub, None), 0, "must refuse to nest a .magi/");
            assert!(
                !sub.join(".magi").exists(),
                "a refused nested init must create no .magi/"
            );
        });
    }

    #[test]
    #[cfg(unix)]
    #[serial_test::serial]
    fn test_run_init_fails_and_creates_nothing_when_ancestor_is_symlink() {
        // Step 4c: an ancestor is a symlink ⇒ `discover` returns `Err`; the
        // command must FAIL (not treat it as a clean tree) and create no `.magi/`.
        with_var(PASSPHRASE_ENV, None, || {
            let tmp = tempfile::tempdir().unwrap();
            let root = dunce::canonicalize(tmp.path()).unwrap();
            let real = root.join("real");
            std::fs::create_dir_all(&real).unwrap();
            let link = root.join("link");
            std::os::unix::fs::symlink(&real, &link).unwrap();
            let sub = link.join("sub");
            std::fs::create_dir_all(&sub).unwrap();

            assert_ne!(
                run_init(&sub, None),
                0,
                "a walk aborted by a symlink must fail, not init"
            );
            assert!(
                !sub.join(".magi").exists(),
                "an aborted walk must create no .magi/"
            );
        });
    }

    // ── T7: headless query/consult dispatch ────────────────────────────────

    /// A [`HeadlessArgs`] with every field at its inert default, for tests.
    fn base_hargs() -> HeadlessArgs {
        HeadlessArgs {
            input: None,
            output: None,
            input_format: None,
            output_format: None,
            workdir: None,
            no_memory: false,
            auto: false,
            full_auto: false,
            timeout: None,
            log_level: None,
            log_dir: None,
            allow_system_override: false,
            no_clobber: false,
            consult: false,
            model: None,
            provider: None,
            max_tool_calls: None,
            mode: None,
            untrusted_content: false,
        }
    }

    /// A finished [`RunOutcome`] with the given stop reason and optional error,
    /// enough to exercise the exit-code mapping.
    fn outcome_with(
        stop_reason: StopReason,
        response: Option<&str>,
        error_kind: Option<ErrorKind>,
    ) -> RunOutcome {
        use magi_rs::headless::types::{AppliedCaps, ErrorPayload, Timings, Usage};
        RunOutcome {
            response: response.map(str::to_string),
            model: "m".to_string(),
            provider: "p".to_string(),
            usage: Usage {
                input_tokens: 0,
                output_tokens: 0,
            },
            timings: Timings {
                total_ms: 1,
                ttfb_ms: None,
                per_turn_ms: Vec::new(),
            },
            stop_reason,
            tool_calls: Vec::new(),
            transcript: Vec::new(),
            consult: None,
            applied_caps: AppliedCaps {
                max_tool_calls: 15,
                max_tool_calls_clamped: false,
                timeout_secs: None,
                system_override_applied: false,
            },
            error: error_kind.map(|kind| ErrorPayload {
                message: String::new(),
                kind,
            }),
        }
    }

    #[test]
    fn test_tier_from_flags_full_auto_wins() {
        assert!(matches!(tier_from_flags(false, false), Tier::Default));
        assert!(matches!(tier_from_flags(true, false), Tier::Auto));
        assert!(matches!(tier_from_flags(false, true), Tier::FullAuto));
        assert!(matches!(tier_from_flags(true, true), Tier::FullAuto));
    }

    #[test]
    fn test_exit_code_for_outcome_taxonomy() {
        // Success ⇒ 0.
        assert_eq!(
            exit_code_for_outcome(&outcome_with(StopReason::Done, Some("hi"), None)),
            0
        );
        // Input-invalid error ⇒ 2 (misuse).
        assert_eq!(
            exit_code_for_outcome(&outcome_with(
                StopReason::Error,
                None,
                Some(ErrorKind::InputInvalid)
            )),
            2
        );
        // Tier denial that blocked the task (empty response + Denied) ⇒ 3.
        assert_eq!(
            exit_code_for_outcome(&outcome_with(StopReason::Denied, Some(""), None)),
            3
        );
        // A runtime-class error (timeout/provider/runtime) ⇒ 1.
        assert_eq!(
            exit_code_for_outcome(&outcome_with(
                StopReason::Error,
                None,
                Some(ErrorKind::Timeout)
            )),
            1
        );
        // A TierDenied error payload ⇒ 3 (taxonomy consistency, REQ-H14). Though
        // currently unreachable, the payload→exit mapping must not degrade to 1.
        assert_eq!(
            exit_code_for_outcome(&outcome_with(
                StopReason::Error,
                None,
                Some(ErrorKind::TierDenied)
            )),
            3
        );
    }

    #[test]
    fn test_write_output_atomic_overwrites_by_default() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("out.txt");
        std::fs::write(&path, b"old").unwrap();
        write_output_atomic(&path, b"new", false).expect("overwrite must succeed");
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
    }

    #[test]
    fn test_write_output_atomic_no_clobber_refuses_existing() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("out.txt");
        std::fs::write(&path, b"keep").unwrap();
        let err = write_output_atomic(&path, b"new", true).expect_err("no-clobber must refuse");
        assert!(matches!(err, HeadlessError::InputInvalid(_)));
        assert_eq!(
            std::fs::read(&path).unwrap(),
            b"keep",
            "existing file unchanged"
        );
        assert_eq!(
            headless_error_exit_code(&err),
            2,
            "no-clobber refusal is misuse"
        );
    }

    #[test]
    fn test_write_output_atomic_no_clobber_creates_when_absent() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("fresh.txt");
        write_output_atomic(&path, b"data", true).expect("create must succeed");
        assert_eq!(std::fs::read(&path).unwrap(), b"data");
    }

    #[test]
    fn test_finish_no_clobber_write_removes_partial_file_on_write_failure() {
        // MAGI re-gate WARNING (Caspar): a `write_all` failure mid-write under
        // `--no-clobber` must not leave a truncated target file behind. This
        // exercises the EXACT cleanup function `write_output_atomic`'s
        // `no_clobber` branch calls on a `write_all` error — the same code
        // path production hits, driven with a simulated I/O failure since a
        // genuine OS-level write failure (disk full / EIO) is not portably
        // inducible in a unit test. The pass-through wiring from the real
        // `f.write_all(contents)` call to this function is a one-line call
        // (`finish_no_clobber_write(path, f.write_all(contents))`), verified
        // by inspection.
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("out.txt");
        // Simulates the O_CREAT|O_EXCL create having already produced a
        // (partially written) target file before `write_all` failed.
        std::fs::write(&path, b"partial").unwrap();
        assert!(path.exists(), "precondition: the partial file exists");

        let simulated_failure = Err(std::io::Error::other("simulated disk-full mid-write"));
        let err = finish_no_clobber_write(&path, simulated_failure)
            .expect_err("a write failure must surface as an error");

        assert!(matches!(err, HeadlessError::Io(_)));
        assert!(
            !path.exists(),
            "a failed write must leave NO partial file on disk"
        );
    }

    #[test]
    fn test_finish_no_clobber_write_leaves_file_intact_on_success() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("out.txt");
        std::fs::write(&path, b"data").unwrap();

        finish_no_clobber_write(&path, Ok(())).expect("a successful write must not error");

        assert_eq!(
            std::fs::read(&path).unwrap(),
            b"data",
            "a successful write must leave the file untouched by cleanup"
        );
    }

    #[test]
    fn test_args_parses_query_and_consult_subcommands() {
        use clap::Parser;
        let q = Args::parse_from(["magi-rs", "query", "--auto", "-i", "q.txt"]);
        match q.command {
            Some(TopCmd::Query(h)) => {
                assert!(h.auto);
                assert_eq!(h.input.as_deref(), Some(std::path::Path::new("q.txt")));
            }
            _ => panic!("expected the query subcommand"),
        }
        let c = Args::parse_from([
            "magi-rs",
            "consult",
            "--full-auto",
            "--output-format",
            "json",
        ]);
        match c.command {
            Some(TopCmd::Consult(h)) => {
                assert!(h.full_auto);
                assert!(matches!(h.output_format, Some(CliOutputFormat::Json)));
            }
            _ => panic!("expected the consult subcommand"),
        }
    }

    /// Builds a `.magi/` under a fresh tempdir cwd with `provider = "anthropic"`
    /// so a keyless run resolves to the canned `StaticProvider` (deterministic,
    /// no network). Returns the temp guard (kept alive) and the canonical cwd.
    fn init_static_workspace() -> (tempfile::TempDir, std::path::PathBuf) {
        let tmp = tempfile::tempdir().unwrap();
        let cwd = dunce::canonicalize(tmp.path()).unwrap();
        crate::system::workspace::init(&cwd).expect("init .magi/");
        std::fs::write(cwd.join(".magi/magi.toml"), "provider = \"anthropic\"\n").unwrap();
        (tmp, cwd)
    }

    /// Builds a `.magi/` under a fresh tempdir cwd with NO provider override —
    /// `resolve_provider` falls through to `DEFAULT_PROVIDER` (`"openai"`), so
    /// tests here observe an ENVELOPE `provider` override crossing away from
    /// that config-default. Returns the temp guard and the canonical cwd.
    fn init_default_workspace() -> (tempfile::TempDir, std::path::PathBuf) {
        let tmp = tempfile::tempdir().unwrap();
        let cwd = dunce::canonicalize(tmp.path()).unwrap();
        crate::system::workspace::init(&cwd).expect("init .magi/");
        (tmp, cwd)
    }

    /// Writes a JSON envelope `body` to `<cwd>/<name>` and returns its path.
    fn write_envelope(cwd: &Path, name: &str, body: &str) -> std::path::PathBuf {
        let path = cwd.join(name);
        std::fs::write(&path, body).unwrap();
        path
    }

    #[test]
    #[serial_test::serial]
    fn test_prepare_headless_provider_only_envelope_resolves_matching_default_model() {
        // MAGI re-gate WARNING (Caspar): `resolve()` (headless/resolution.rs)
        // applies the envelope's `provider` and `model` overrides
        // INDEPENDENTLY. Before the fix, `default_model` was computed for the
        // config-DEFAULT provider ("openai" here — no toml override), so an
        // envelope overriding only `provider` to "anthropic" (no `model`) fed
        // an Ollama/OpenAI model name into `resolved.model` while
        // `resolved.provider` said "anthropic" — a cross-provider mismatch.
        with_var("MAGI_PROVIDER", None, || {
            with_var("ANTHROPIC_MODEL", None, || {
                with_var("OPENAI_MODEL", None, || {
                    let (_tmp, cwd) = init_default_workspace();
                    let input = write_envelope(
                        &cwd,
                        "env.json",
                        r#"{"prompt":"hi","provider":"anthropic"}"#,
                    );

                    let mut h = base_hargs();
                    h.input = Some(input);
                    h.workdir = Some(cwd.clone());
                    h.no_memory = true; // stateless ⇒ env-only, no passphrase needed

                    let rt = tokio::runtime::Runtime::new().unwrap();
                    let ctx = rt
                        .block_on(prepare_headless(&h, None, &cwd, None, None))
                        .expect("prepare_headless must succeed");

                    assert_eq!(ctx.resolved.provider, "anthropic");
                    assert_eq!(
                        ctx.resolved.model,
                        crate::defaults::DEFAULT_ANTHROPIC_MODEL,
                        "provider-without-model must resolve the default MODEL for the \
                         EFFECTIVE (envelope-overridden) provider, not the config-default \
                         provider's model"
                    );
                });
            });
        });
    }

    #[test]
    #[serial_test::serial]
    fn test_prepare_headless_explicit_envelope_model_still_wins_over_provider_default() {
        // The envelope's explicit `model` must still take precedence over
        // whatever default the effective-provider peek would otherwise supply
        // (precedence unchanged by the fix).
        with_var("MAGI_PROVIDER", None, || {
            with_var("ANTHROPIC_MODEL", None, || {
                with_var("OPENAI_MODEL", None, || {
                    let (_tmp, cwd) = init_default_workspace();
                    let input = write_envelope(
                        &cwd,
                        "env.json",
                        r#"{"prompt":"hi","provider":"anthropic","model":"custom-model"}"#,
                    );

                    let mut h = base_hargs();
                    h.input = Some(input);
                    h.workdir = Some(cwd.clone());
                    h.no_memory = true;

                    let rt = tokio::runtime::Runtime::new().unwrap();
                    let ctx = rt
                        .block_on(prepare_headless(&h, None, &cwd, None, None))
                        .expect("prepare_headless must succeed");

                    assert_eq!(ctx.resolved.provider, "anthropic");
                    assert_eq!(
                        ctx.resolved.model, "custom-model",
                        "an explicit envelope model must still win"
                    );
                });
            });
        });
    }

    #[test]
    #[serial_test::serial]
    fn test_prepare_headless_neither_provider_nor_model_set_uses_config_default() {
        // With neither the envelope nor a CLI flag setting `provider`, the
        // effective-provider peek must fall back to the SAME config-default
        // provider `resolve()` already used — no behavior change for the
        // common case.
        //
        // `init_default_workspace()` scaffolds `.magi/magi.toml` via
        // `render_default_magi_toml()`, which declares `provider = "ollama"` (REQ-A01b).
        // Task 4.1 retired the `resolve_provider`/`legacy_backend_label` shim that used
        // to normalize this onto the legacy `"openai"` string — with the vocabulary
        // unified end to end, `resolved.provider` now shows the value the file
        // actually says, `"ollama"`, not a translation of it. The ROUTING stays
        // Ollama-first either way; only the string representation changed.
        with_var("MAGI_PROVIDER", None, || {
            with_var("ANTHROPIC_MODEL", None, || {
                with_var("OPENAI_MODEL", None, || {
                    let (_tmp, cwd) = init_default_workspace();
                    let input = write_envelope(&cwd, "env.json", r#"{"prompt":"hi"}"#);

                    let mut h = base_hargs();
                    h.input = Some(input);
                    h.workdir = Some(cwd.clone());
                    h.no_memory = true;

                    let rt = tokio::runtime::Runtime::new().unwrap();
                    let ctx = rt
                        .block_on(prepare_headless(&h, None, &cwd, None, None))
                        .expect("prepare_headless must succeed");

                    assert_eq!(ctx.resolved.provider, "ollama");
                    assert_eq!(ctx.provider_kind, ProviderKind::Ollama);
                    assert_eq!(ctx.resolved.model, crate::defaults::DEFAULT_OPENAI_MODEL);
                });
            });
        });
    }

    /// I2 (review round 2), updated Task 4.1: the retired shim only normalized
    /// `default_provider` (the `env > TOML > default` winner) — `h.provider` (the CLI
    /// `--provider` flag) reached `effective_provider`/`resolved.provider`
    /// unnormalized. `magi query --provider ollama` must still route to the Ollama
    /// backend, now via `ProviderKind::parse` validating the SAME value directly
    /// instead of a legacy-label pass-through.
    #[test]
    fn test_prepare_headless_cli_provider_override_normalizes_the_new_vocabulary() {
        with_var("MAGI_PROVIDER", None, || {
            with_var("ANTHROPIC_MODEL", None, || {
                with_var("OPENAI_MODEL", None, || {
                    let (_tmp, cwd) = init_default_workspace();
                    let input = write_envelope(&cwd, "env.json", r#"{"prompt":"hi"}"#);

                    let mut h = base_hargs();
                    h.input = Some(input);
                    h.workdir = Some(cwd.clone());
                    h.no_memory = true;
                    h.provider = Some("ollama".to_string());

                    let rt = tokio::runtime::Runtime::new().unwrap();
                    let ctx = rt
                        .block_on(prepare_headless(&h, None, &cwd, None, None))
                        .expect("prepare_headless must succeed");

                    assert_eq!(ctx.resolved.provider, "ollama");
                    assert_eq!(ctx.provider_kind, ProviderKind::Ollama);
                });
            });
        });
    }

    /// I2, updated Task 4.1: same gap, via the envelope's `provider` field instead of
    /// the CLI flag.
    #[test]
    fn test_prepare_headless_envelope_provider_override_normalizes_the_new_vocabulary() {
        with_var("MAGI_PROVIDER", None, || {
            with_var("ANTHROPIC_MODEL", None, || {
                with_var("OPENAI_MODEL", None, || {
                    let (_tmp, cwd) = init_default_workspace();
                    let input =
                        write_envelope(&cwd, "env.json", r#"{"prompt":"hi","provider":"ollama"}"#);

                    let mut h = base_hargs();
                    h.input = Some(input);
                    h.workdir = Some(cwd.clone());
                    h.no_memory = true;

                    let rt = tokio::runtime::Runtime::new().unwrap();
                    let ctx = rt
                        .block_on(prepare_headless(&h, None, &cwd, None, None))
                        .expect("prepare_headless must succeed");

                    assert_eq!(ctx.resolved.provider, "ollama");
                    assert_eq!(ctx.provider_kind, ProviderKind::Ollama);
                });
            });
        });
    }

    // -------------------------------------------------------------------------
    // Fix round 2 (coordinator review, 2026-08-02): C1/C2/C3 —
    // `resolve_effective_embedding_endpoint` (the same four lines
    // `attach_persistent_memory` used to inline).
    // -------------------------------------------------------------------------

    /// C1: `[embedding].base_url = ""` used to skip inheritance entirely — the
    /// old code gated on `base_url.is_none()`, and `Some("")` is `Some`, so it
    /// reached the embedder as an empty URL instead of inheriting the root.
    /// `effective_embedding_base_url()`'s own blank-is-absent handling must
    /// decide this unconditionally.
    #[test]
    fn resolve_effective_embedding_endpoint_treats_blank_as_absent_and_inherits_root() {
        let cfg = MagiConfig::from_toml_str(
            "base_url = \"http://lan:11434/v1\"\n[embedding]\nbase_url = \"\"\n",
        )
        .unwrap();
        let resolved = resolve_effective_embedding_endpoint(&cfg, None).unwrap();
        assert_eq!(resolved, "http://lan:11434/v1");
    }

    /// C2: a malformed template (a literal credential instead of the
    /// `[user]:[password]` placeholders, REQ-A16c) must be a loud, propagated
    /// error — never a silent fall-back to the Ollama default that leaves the
    /// user believing their declared endpoint is in effect.
    #[test]
    fn resolve_effective_embedding_endpoint_propagates_a_malformed_template_error() {
        let cfg =
            MagiConfig::from_toml_str("[embedding]\nbase_url = \"https://user:hunter2@host/v1\"\n")
                .unwrap();
        let err = resolve_effective_embedding_endpoint(&cfg, None).unwrap_err();
        assert!(!err.contains("hunter2"), "leaked the credential: {err}");
    }

    /// C3 (positive): a template with real placeholders resolves to the ACTUAL
    /// credentialed endpoint — never the literal `[user]:[password]` text baked
    /// into an HTTP client.
    #[test]
    fn resolve_effective_embedding_endpoint_resolves_placeholders_against_the_vault() {
        let ss = vault_fixture();
        {
            let mut guard = ss.lock().unwrap();
            guard.set("EMBEDDING_BASE_URL_USER", "alice").unwrap();
            guard.set("EMBEDDING_BASE_URL_PASSWORD", "s3cr3t").unwrap();
        }
        let cfg = MagiConfig::from_toml_str(
            "[embedding]\nbase_url = \"https://[user]:[password]@host/v1\"\n",
        )
        .unwrap();
        let resolved = resolve_effective_embedding_endpoint(&cfg, Some(&ss)).unwrap();
        assert_eq!(resolved, "https://alice:s3cr3t@host/v1");
    }

    /// C3 (negative): with no vault handle in scope, a template that NEEDS one
    /// fails loudly — it must never bake the unresolved `[user]:[password]`
    /// placeholder text into the returned endpoint.
    #[test]
    fn resolve_effective_embedding_endpoint_fails_loudly_without_a_vault_for_placeholders() {
        let cfg = MagiConfig::from_toml_str(
            "[embedding]\nbase_url = \"https://[user]:[password]@host/v1\"\n",
        )
        .unwrap();
        let err = resolve_effective_embedding_endpoint(&cfg, None).unwrap_err();
        assert!(
            !err.contains("[user]") && !err.contains("[password]"),
            "leaked the unresolved template: {err}"
        );
    }

    /// Happy path / regression guard: nothing declared anywhere resolves to the
    /// built-in Ollama default, same as before this fix.
    #[test]
    fn resolve_effective_embedding_endpoint_uses_the_default_when_nothing_declared() {
        let cfg = MagiConfig::default();
        let resolved = resolve_effective_embedding_endpoint(&cfg, None).unwrap();
        assert_eq!(resolved, crate::defaults::DEFAULT_OPENAI_BASE_URL);
    }

    // -------------------------------------------------------------------------
    // Fix round 3 (coordinator review, 2026-08-02): L1/L2/S1 —
    // `resolve_effective_principal_endpoint`, the SAME C1/C2/C3 cluster on the
    // principal-provider path (the old `resolve_openai_base_url`, removed).
    // -------------------------------------------------------------------------

    /// L1: a blank root `base_url` is absent, not a value — falls to the
    /// built-in default, same rule as everywhere else (REQ-A12).
    #[test]
    fn resolve_effective_principal_endpoint_treats_blank_root_base_url_as_absent() {
        let cfg = MagiConfig::from_toml_str("base_url = \"\"\n").unwrap();
        let resolved = resolve_effective_principal_endpoint(&cfg, None, None).unwrap();
        assert_eq!(resolved.as_str(), crate::defaults::DEFAULT_OPENAI_BASE_URL);
    }

    /// L1: a blank `OPENAI_BASE_URL` env var is ALSO absent, not a value — the
    /// old `resolve_openai_base_url` returned it unconditionally
    /// (`env_base_url.map(str::to_string)`, no blank check), so an
    /// exported-but-unfilled CI variable short-circuited past the TOML/default
    /// fallback and the principal provider was built with an empty base URL.
    #[test]
    fn resolve_effective_principal_endpoint_treats_blank_env_override_as_absent() {
        let cfg = MagiConfig::from_toml_str("base_url = \"http://lan:11434/v1\"\n").unwrap();
        let resolved = resolve_effective_principal_endpoint(&cfg, Some(""), None).unwrap();
        assert_eq!(
            resolved.as_str(),
            "http://lan:11434/v1",
            "blank env must fall through to the TOML value, not short-circuit past it"
        );
    }

    /// L2/C2: a malformed template (a literal credential instead of the
    /// `[user]:[password]` placeholders, REQ-A16c) is a loud, propagated error —
    /// never a silent fall-back, and the error never repeats the credential.
    #[test]
    fn resolve_effective_principal_endpoint_propagates_a_malformed_template_error() {
        let cfg =
            MagiConfig::from_toml_str("base_url = \"https://user:hunter2@host/v1\"\n").unwrap();
        let err = resolve_effective_principal_endpoint(&cfg, None, None).unwrap_err();
        assert!(!err.contains("hunter2"), "leaked the credential: {err}");
    }

    /// L2/C3 (positive): a template with real placeholders resolves to the
    /// ACTUAL credentialed endpoint via the ROOT vault scope
    /// (`BASE_URL_USER`/`BASE_URL_PASSWORD`) — never the literal
    /// `[user]:[password]` text reaching `build_openai_provider`.
    #[test]
    fn resolve_effective_principal_endpoint_resolves_placeholders_against_the_vault() {
        let ss = vault_fixture();
        {
            let mut guard = ss.lock().unwrap();
            guard.set("BASE_URL_USER", "alice").unwrap();
            guard.set("BASE_URL_PASSWORD", "s3cr3t").unwrap();
        }
        let cfg = MagiConfig::from_toml_str("base_url = \"https://[user]:[password]@host/v1\"\n")
            .unwrap();
        let resolved = resolve_effective_principal_endpoint(&cfg, None, Some(&ss)).unwrap();
        assert_eq!(resolved.as_str(), "https://alice:s3cr3t@host/v1");
    }

    /// L2/C3 (negative): with no vault handle in scope, a template that NEEDS
    /// one fails loudly — it must never bake the unresolved `[user]:[password]`
    /// placeholder text into the returned endpoint.
    #[test]
    fn resolve_effective_principal_endpoint_fails_loudly_without_a_vault_for_placeholders() {
        let cfg = MagiConfig::from_toml_str("base_url = \"https://[user]:[password]@host/v1\"\n")
            .unwrap();
        let err = resolve_effective_principal_endpoint(&cfg, None, None).unwrap_err();
        assert!(
            !err.contains("[user]") && !err.contains("[password]"),
            "leaked the unresolved template: {err}"
        );
    }

    /// S1: `openai_provider_info` — the TUI/stderr-visible startup notice — never
    /// echoes a credential, even when handed a fully vault-resolved URL that
    /// contains a real one. `redact_url` (Task 1.2) is what makes this true;
    /// this test proves the call site actually uses it, not just that the
    /// redactor itself works (that's covered in `src/redact.rs`).
    #[test]
    fn openai_provider_info_never_echoes_a_credential() {
        let info = openai_provider_info("https://alice:s3cr3t@host/v1", "some-model");
        assert!(!info.contains("s3cr3t"), "leaked the credential: {info}");
        assert!(info.contains("host"), "the host must stay: {info}");
        assert!(info.contains("some-model"));
    }

    #[test]
    #[serial_test::serial]
    fn test_headless_query_static_provider_returns_response_exit_0() {
        with_var("MAGI_PROVIDER", None, || {
            let (_tmp, cwd) = init_static_workspace();
            let prompt = cwd.join("prompt.txt");
            std::fs::write(&prompt, b"hello").unwrap();
            let out = cwd.join("out.txt");

            let mut h = base_hargs();
            h.input = Some(prompt);
            h.output = Some(out.clone());
            h.workdir = Some(cwd.clone());
            h.no_memory = true; // stateless ⇒ env-only, no passphrase needed

            let rt = tokio::runtime::Runtime::new().unwrap();
            let code = rt.block_on(run_query_subcommand(h, None, &cwd, None, None));
            assert_eq!(code, 0, "a static-provider query must succeed");
            let body = std::fs::read_to_string(&out).unwrap();
            assert!(!body.is_empty(), "the response must be written to -o");
        });
    }

    #[test]
    #[serial_test::serial]
    fn test_headless_query_json_output_has_schema_version_and_stop_reason() {
        with_var("MAGI_PROVIDER", None, || {
            let (_tmp, cwd) = init_static_workspace();
            let prompt = cwd.join("q.txt");
            std::fs::write(&prompt, b"hello json").unwrap();
            let out = cwd.join("out.json");

            let mut h = base_hargs();
            h.input = Some(prompt);
            h.output = Some(out.clone());
            h.workdir = Some(cwd.clone());
            h.no_memory = true;
            h.output_format = Some(CliOutputFormat::Json);

            let rt = tokio::runtime::Runtime::new().unwrap();
            let code = rt.block_on(run_query_subcommand(h, None, &cwd, None, None));
            assert_eq!(code, 0);
            let v: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
            assert_eq!(v["schema_version"], 1);
            assert!(v.get("response").is_some());
            assert!(v.get("stop_reason").is_some());
            assert!(v.get("tool_calls").is_some());
        });
    }

    #[test]
    #[serial_test::serial]
    fn test_headless_query_no_clobber_on_existing_output_exits_2() {
        with_var("MAGI_PROVIDER", None, || {
            let (_tmp, cwd) = init_static_workspace();
            let prompt = cwd.join("q.txt");
            std::fs::write(&prompt, b"hi").unwrap();
            let out = cwd.join("exists.txt");
            std::fs::write(&out, b"PRESERVE").unwrap();

            let mut h = base_hargs();
            h.input = Some(prompt);
            h.output = Some(out.clone());
            h.workdir = Some(cwd.clone());
            h.no_memory = true;
            h.no_clobber = true;

            let rt = tokio::runtime::Runtime::new().unwrap();
            let code = rt.block_on(run_query_subcommand(h, None, &cwd, None, None));
            assert_eq!(code, 2, "--no-clobber on an existing -o file ⇒ exit 2");
            assert_eq!(
                std::fs::read(&out).unwrap(),
                b"PRESERVE",
                "the existing file must be untouched"
            );
        });
    }

    #[test]
    #[serial_test::serial]
    fn test_headless_consult_over_max_query_bytes_exits_2() {
        with_var("MAGI_PROVIDER", None, || {
            let (_tmp, cwd) = init_static_workspace();
            let prompt = cwd.join("big.txt");
            // REQ-A11b raised the default cap from the retired 8 KiB `MAX_QUERY_LEN`
            // to `magi_rs::magi::MAX_QUERY_BYTES` (256 KiB, SC-A11) — 9000 bytes no
            // longer exceeds it, so the fixture must genuinely be over the new cap.
            std::fs::write(&prompt, "x".repeat(magi_rs::magi::MAX_QUERY_BYTES + 1)).unwrap();
            let out = cwd.join("out.txt");

            let mut h = base_hargs();
            h.input = Some(prompt);
            h.output = Some(out);
            h.workdir = Some(cwd.clone());
            h.no_memory = true;

            let rt = tokio::runtime::Runtime::new().unwrap();
            // Task 4.1: `run_consult_subcommand` needs a LIVE MAGI trio unconditionally
            // (a forced consult has nowhere else to go) — `init_static_workspace`'s
            // `provider = "anthropic"` with NO key used to still "build" a trio under
            // the retired adapter (it wrapped whatever principal provider existed,
            // `StaticProvider` included, without checking credentials at all). The
            // native trio checks its OWN credentials for real (REQ-A05b), so it
            // correctly refuses to build here without one. A fake key is enough — it
            // is only ever used to satisfy `ClaudeProvider::with_timeout`'s
            // constructor (which never makes a network call); `analyze_direct`
            // (`src/headless_runner.rs`) rejects the oversized prompt BEFORE
            // `magi.analyze()` is ever reached, so this stays real-network-free
            // (R-A04).
            let code = rt.block_on(run_consult_subcommand(
                h,
                None,
                None,
                &cwd,
                Some("sk-ant-test-fake-key".to_string()),
                None,
            ));
            assert_eq!(
                code, 2,
                "an over-cap consult prompt ⇒ exit 2 (rejected, not truncated)"
            );
        });
    }

    /// SC-A04d (the half Task 6.1 deferred, Task 6.2 Step 3c): an ABSENT
    /// `--timeout` no longer leaves a `magi consult` run unbounded — it falls
    /// back to the formula-derived minimum instead of `None` (which, threaded
    /// into `analyze_direct`'s `deadline` arm, parks forever).
    ///
    /// A true end-to-end wall-clock test of this specific fix is impractical:
    /// even the LOWEST configurable `agent_timeout_secs` (30, §4.9's floor)
    /// yields a formula minimum of `headless_consult_timeout_secs(30)` ≈ 78s, so
    /// actually waiting for the deadline to fire would make this test far slower
    /// than the suite's budget tolerates. `resolve_run_timeout` itself is already
    /// exhaustively unit-tested in `src/magi/mod.rs` (the arithmetic); this pins
    /// the ONE-LINE wiring seam at the `magi consult` call site instead —
    /// `consult_deadline` is exactly the expression `run_consult_subcommand` now
    /// uses to build `timeout`, so a regression back to `h.timeout.map(...)`
    /// (which silently drops this branch) fails this test, not a 78s-plus wait.
    #[test]
    fn consult_deadline_falls_back_to_the_formula_minimum_when_absent() {
        let ceiling = magi_rs::magi::AGENT_TIMEOUT_SECS;
        let decision = magi_rs::magi::resolve_run_timeout(None, ceiling, 0, false);
        let deadline = consult_deadline(&decision);
        assert_eq!(
            deadline,
            Duration::from_secs(magi_rs::magi::headless_consult_timeout_secs(
                ceiling, 0, false
            )),
            "an absent --timeout must fall back to the formula-derived minimum"
        );
        assert!(
            deadline.as_secs() > 0,
            "must never resolve to an unbounded (None) run"
        );
    }

    /// SC-A04d: an explicit `--timeout` is obeyed VERBATIM, even below the
    /// formula's minimum — the flag is the operator's own wall-clock cap, not an
    /// invariant `consult_deadline` is entitled to override (the warning that
    /// this choice may starve a schema retry is Task 6.1's `.warning`, already
    /// wired to both stderr and the JSON).
    #[test]
    fn consult_deadline_obeys_an_explicit_timeout_even_below_the_formula() {
        let ceiling = magi_rs::magi::AGENT_TIMEOUT_SECS;
        let decision = magi_rs::magi::resolve_run_timeout(Some(1), ceiling, 0, false);
        assert_eq!(
            consult_deadline(&decision),
            Duration::from_secs(1),
            "the operator's explicit value must be obeyed even under the formula's minimum"
        );
    }

    /// Task 4.1 — native provider trio construction.
    ///
    /// SC-A01, SC-A02, SC-A03, SC-A05, SC-A05b, SC-A05c closed here. SC-A04 is already closed
    /// by `magi::mod::derived_scale_satisfies_invariant_across_the_whole_admissible_range`
    /// (Phase 0/1) — not this task's territory, so it is not duplicated. SC-A06 (surface
    /// behavior when the trio is unbuildable: actionable message in the TUI, `consult` absent
    /// from the registry, headless closed) remains **UNTESTED here**: it is the full contract
    /// of Task 4.3 (`trio_unavailable_message`), which does not yet exist — see this task's
    /// report.
    mod trio_construction {
        use super::*;
        use magi_core::error::ExternalErrorKind;
        use magi_core::provider::CompletionConfig;
        use magi_core::test_support::valid_verdict_for_current_agent;
        use std::time::Instant;

        /// Test endpoint: a flat `base_url` with no placeholders, so resolving it does not need
        /// a real vault — `NoVaultInScope` (already in production) is enough.
        fn test_endpoints() -> ResolvedEndpoints {
            let tpl = EndpointTemplate::parse("http://localhost:11434/v1", Scope::Root).unwrap();
            ResolvedEndpoints {
                root: tpl.resolve(&mut NoVaultInScope, Scope::Root).unwrap(),
                magi: tpl.resolve(&mut NoVaultInScope, Scope::Magi).unwrap(),
            }
        }

        /// Resolves an arbitrary flat `base_url` into a [`ResolvedEndpoint`], for the tests that
        /// need a real listening socket rather than the fixed `localhost:11434` above.
        fn endpoint_at(base_url: &str) -> ResolvedEndpoint {
            EndpointTemplate::parse(base_url, Scope::Root)
                .expect("a flat test URL must parse")
                .resolve(&mut NoVaultInScope, Scope::Magi)
                .expect("a URL with no placeholders needs no vault")
        }

        /// A local listener that accepts one connection and **answers nothing**, so a request
        /// against it can only end by the CLIENT's timeout.
        ///
        /// `127.0.0.1:0` picks a free port and the whole exchange stays on the machine — no real
        /// network, which R-R05 forbids. The returned guard keeps the task alive; dropping it
        /// closes the listener.
        async fn silent_listener() -> (String, tokio::task::JoinHandle<()>) {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("binding a loopback port must succeed");
            let addr = listener
                .local_addr()
                .expect("a bound listener has an address");
            let guard = tokio::spawn(async move {
                if let Ok((socket, _)) = listener.accept().await {
                    // Held open, unanswered, for as long as the test runs.
                    let _held = socket;
                    std::future::pending::<()>().await;
                }
            });
            (format!("http://{addr}"), guard)
        }

        /// A local listener that records the FIRST request line it receives, answers 404 so the
        /// client stops waiting, and hands that line back.
        ///
        /// This is what makes "both spellings reach the same endpoint" an observation instead of
        /// a restatement of the code: the assertion reads the path the provider actually put on
        /// the wire.
        async fn recording_listener() -> (String, tokio::task::JoinHandle<String>) {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("binding a loopback port must succeed");
            let addr = listener
                .local_addr()
                .expect("a bound listener has an address");
            let handle = tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await.expect("the client must connect");
                let mut buf = vec![0u8; 1024];
                let read = socket.read(&mut buf).await.unwrap_or(0);
                let text = String::from_utf8_lossy(&buf[..read]).into_owned();
                let _ = socket
                    .write_all(b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\n\r\n")
                    .await;
                text.lines().next().unwrap_or_default().to_owned()
            });
            (format!("http://{addr}"), handle)
        }

        /// SC-R48: the client timeout a seat is built with is the one it HONOURS — never the
        /// crate's 300 s default.
        ///
        /// `OllamaProvider::new` delegates to `with_timeout(..., DEFAULT_CLIENT_TIMEOUT)` = 300 s,
        /// which breaks `operation_budget + client_timeout <= agent_timeout_secs`. Picking the
        /// wrong constructor COMPILES, RUNS, and breaks the derived scale SILENTLY — the exact
        /// defect D-A07 existed to prevent, which survives its reversal (D-R12).
        ///
        /// Observed through BEHAVIOUR, not through the value: neither `reqwest::Client` nor
        /// `OllamaProvider` exposes its timeout, so a test that re-reads the argument it just
        /// passed would assert nothing. This one points the seat at a socket that never answers
        /// and requires the request to end anyway.
        ///
        /// The discriminating property is **"not the crate's 300 s"**, not "under 400 ms", so the
        /// deadline is generous on purpose: that keeps the test meaningful AND immune to load
        /// (R-R05 — wait on conditions, never on durations).
        #[tokio::test]
        async fn the_ollama_seat_honours_the_client_timeout_it_was_given() {
            let (base, _guard) = silent_listener().await;
            let mut notices = Vec::new();
            let provider = build_native_provider(
                ProviderKind::Ollama,
                &endpoint_at(&base),
                "any-model",
                None,
                Duration::from_millis(400),
                &mut notices,
            )
            .expect("ollama is keyless: it must build with no credentials");

            let outcome = tokio::time::timeout(
                Duration::from_secs(30),
                provider.complete("s", "u", &CompletionConfig::default()),
            )
            .await;

            let ended = outcome.expect(
                "the request must end by the client timeout the seat was BUILT with; \
                 hitting this deadline means the 300 s crate default reached the seat",
            );
            assert!(
                ended.is_err(),
                "a server that never answers must surface as an error, not a completion"
            );
        }

        /// SC-R49: both `base_url` spellings reach the same completions endpoint. Under
        /// `kind = "ollama"` a URL without `/v1` used to end in 404.
        #[tokio::test]
        async fn both_base_url_spellings_reach_the_same_completions_endpoint() {
            for suffix in ["", "/v1"] {
                let (host, recorded) = recording_listener().await;
                let mut notices = Vec::new();
                let provider = build_native_provider(
                    ProviderKind::Ollama,
                    &endpoint_at(&format!("{host}{suffix}")),
                    "any-model",
                    None,
                    Duration::from_secs(10),
                    &mut notices,
                )
                .expect("ollama is keyless");
                // The 404 is expected; what matters is the path that reached the wire.
                let _ = provider
                    .complete("s", "u", &CompletionConfig::default())
                    .await;
                let line = recorded.await.expect("the listener task must finish");
                assert!(
                    line.starts_with("POST /v1/chat/completions"),
                    "base_url ending in {suffix:?} put {line:?} on the wire"
                );
            }
        }

        /// SC-R50: `openai-compat` does NOT change transport, even pointing at an Ollama daemon.
        /// The partition is by DECLARED kind, never by what happens to be on the other side.
        ///
        /// Also pins the observable REQ-R30 declares changed, and which the CHANGELOG has to
        /// name: under `kind = "ollama"` the provider now identifies itself as `"ollama"` in
        /// errors and reports, where it used to say `"openai-compat"`.
        #[test]
        fn the_declared_kind_decides_the_transport_not_the_daemon_behind_it() {
            let mut notices = Vec::new();
            let creds = FixedCreds {
                openai: Some("test-key".to_string()),
                anthropic: None,
            };
            let compat = build_native_provider(
                ProviderKind::OpenAiCompat,
                &test_endpoints().magi,
                "any-model",
                Some(&creds),
                Duration::from_secs(10),
                &mut notices,
            )
            .expect("openai-compat with a key must build");
            assert_eq!(
                compat.name(),
                "openai-compat",
                "an Ollama daemon behind the URL must not change the declared transport"
            );

            let ollama = build_native_provider(
                ProviderKind::Ollama,
                &test_endpoints().magi,
                "any-model",
                None,
                Duration::from_secs(10),
                &mut notices,
            )
            .expect("ollama is keyless");
            assert_eq!(
                ollama.name(),
                "ollama",
                "REQ-R30: the ollama kind completes through OllamaProvider now"
            );
        }

        /// Body of an OpenAI-compatible response carrying a valid verdict for `agent`.
        ///
        /// Built with `serde_json` rather than by string interpolation: the verdict itself is
        /// JSON **inside** a JSON string field, and hand-escaping that is how a mock ends up
        /// serving something the parser rejects for a reason unrelated to the test.
        fn verdict_body(agent: &str) -> String {
            use magi_core::verdict_markers::{VERDICT_CLOSE, VERDICT_OPEN};
            let verdict = format!(
                "{VERDICT_OPEN}\n{{\"agent\":\"{agent}\",\"verdict\":\"approve\",\
                 \"confidence\":0.9,\"summary\":\"ok\",\"reasoning\":\"r\",\
                 \"recommendation\":\"go\",\"findings\":[]}}\n{VERDICT_CLOSE}"
            );
            serde_json::json!({ "choices": [ { "message": { "content": verdict } } ] }).to_string()
        }

        /// A `magi.toml` declaring the three seats, their lineages and one pool candidate.
        fn cfg_with_pool(max_rotations: u32) -> MagiConfig {
            MagiConfig::from_toml_str(&format!(
                "provider = \"ollama\"\n\
                 [magi]\n\
                 melchior_model    = \"ok-model\"\nmelchior_lineage  = \"lin-melchior\"\n\
                 balthasar_model   = \"ok-model\"\nbalthasar_lineage = \"lin-balthasar\"\n\
                 caspar_model \
                  = \"down-model\"\ncaspar_lineage    = \"lin-caspar\"\n\
                 agent_timeout_secs = 30\n\
                 max_rotations = {max_rotations}\n\
                 [[magi.fallback]]\n\
                 model   = \"rescue-model\"\nlineage = \"lin-rescue\"\n"
            ))
            .expect("the rotation config must parse")
        }

        /// SC-R23/REQ-R11: a **declared** `strict_context_guard = true` reaches magi-core as
        /// `false` when no candidate has a measured window.
        ///
        /// This is the case the fail-safe exists for and the only one that discriminates. Its
        /// neighbour below asserts `!wired.strict_guard` on a config that never declares the key,
        /// where declared and effective are both `false` — so it holds whether or not the
        /// fail-safe works, and the trace it reads recorded
        /// `cfg.declared_strict_context_guard()` rather than the value actually handed to the
        /// builder. Two halves of one blind spot: a trace reporting the wrong number, and the
        /// only test of it unable to tell (S3 Loop 2, Balthasar).
        ///
        /// What is at stake is not cosmetic. A `true` with nothing measured makes **every**
        /// candidate fail magi-core's condition #6, so rotation switches off whole — silently,
        /// on the cold start that is any fresh install's first run.
        ///
        /// **Mutation-verified (B16):** put `cfg.declared_strict_context_guard()` back in the
        /// trace and this goes red; the neighbour below stays green.
        #[test]
        fn a_declared_strict_guard_reaches_magi_core_as_false_when_nothing_is_measured() {
            let cfg = MagiConfig::from_toml_str(
                "provider = \"ollama\"\n\
                 [magi]\n\
                 melchior_model    = \"ok-model\"\nmelchior_lineage  = \"lin-melchior\"\n\
                 balthasar_model   = \"ok-model\"\nbalthasar_lineage = \"lin-balthasar\"\n\
                 caspar_model      = \"down-model\"\ncaspar_lineage    = \"lin-caspar\"\n\
                 agent_timeout_secs = 30\n\
                 max_rotations = 2\n\
                 strict_context_guard = true\n\
                 [[magi.fallback]]\n\
                 model   = \"rescue-model\"\nlineage = \"lin-rescue\"\n",
            )
            .expect("the rotation config must parse");
            assert!(
                cfg.declared_strict_context_guard(),
                "precondition: the operator DID declare it, or this test proves nothing"
            );

            let mut notices = Vec::new();
            let magi = build_magi_orchestrator(
                &TrioBuild {
                    cfg: &cfg,
                    principal_kind: ProviderKind::Ollama,
                    endpoints: &test_endpoints(),
                    creds: None,
                    warn_tokens: None,
                    env_overrides: &MagiEnvModelOverrides::default(),
                    capability_cache: None,
                    // Nothing measured: the cold start.
                    measured: &BTreeMap::new(),
                },
                &mut notices,
            )
            .expect("ollama is keyless");
            drop(magi);

            let wired = pool_wiring_trace().expect("a declared pool must be wired");
            assert!(
                !wired.strict_guard,
                "declared true, but no candidate measured — magi-core must receive false or the \
                 pool it was given was never eligible"
            );
            assert!(
                notices
                    .iter()
                    .any(|n| n.text.contains("strict_context_guard")),
                "and the override must be announced: a setting the operator wrote and the system \
                 did not apply is exactly what the notice rule exists for"
            );
        }

        /// SC-R01/REQ-R03: the pool declared in `[[magi.fallback]]` reaches magi-core with each
        /// candidate's model AND its lineage, in the declared order (strongest first).
        ///
        /// Read from `pool_wiring_trace()`, recorded in the same place the pool is handed to the
        /// builder: `MagiBuilder` keeps `fallback_pool` private and `Magi` exposes no reader, so
        /// from outside the crate there is nothing else to observe.
        #[test]
        fn the_declared_pool_reaches_magi_core_with_its_lineages() {
            let mut notices = Vec::new();
            let magi = build_magi_orchestrator(
                &TrioBuild {
                    cfg: &cfg_with_pool(2),
                    principal_kind: ProviderKind::Ollama,
                    endpoints: &test_endpoints(),
                    creds: None,
                    warn_tokens: None,
                    env_overrides: &MagiEnvModelOverrides::default(),
                    capability_cache: None,
                    measured: &BTreeMap::new(),
                },
                &mut notices,
            )
            .expect("ollama is keyless");
            drop(magi);

            let wired = pool_wiring_trace().expect("a declared pool must be wired");
            assert_eq!(
                wired.candidates,
                vec![("rescue-model".to_string(), "lin-rescue".to_string())],
                "the candidate must carry its declared lineage, not an inferred one"
            );
            assert_eq!(wired.max_rotations, 2);
            assert!(
                !wired.strict_guard,
                "an undeclared strict_context_guard reaches magi-core as false (REQ-R11): the \
                 case it would bite is the COLD START, where nothing measured yet means every \
                 candidate fails the window condition and rotation switches off in silence"
            );
        }

        /// SC-R52/D-R15: a model declared TWICE registers exactly ONE probe.
        ///
        /// magi-core indexes capabilities by `model_id` and knows nothing about endpoints, so two
        /// probes for one model leave it keeping whichever answered last, **in nondeterministic
        /// order**. Deduplicating at registration makes that unreachable instead of merely
        /// forbidden on paper — and two seats on the same model stay LEGAL, as they were in
        /// v0.12.0, since prohibiting them would be a configuration break bought for nothing.
        #[test]
        fn a_model_declared_twice_registers_a_single_probe() {
            let cfg = MagiConfig::from_toml_str(
                "provider = \"ollama\"\n\
                 [magi]\n\
                 melchior_model    = \"shared\"\nmelchior_lineage  = \"lin-a\"\n\
                 balthasar_model   = \"shared\"\nbalthasar_lineage = \"lin-b\"\n\
                 caspar_model \
                  = \"other\"\ncaspar_lineage    = \"lin-c\"\n\
                 [[magi.fallback]]\n\
                 model   = \"shared\"\nlineage = \"lin-rescue\"\n",
            )
            .expect("two seats on one model is legal");

            let cache = {
                let conn = std::sync::Mutex::new(
                    rusqlite::Connection::open_in_memory().expect("in-memory sqlite"),
                );
                let dek = magi_rs::vault::MaskedDek::new(zeroize::Zeroizing::new(vec![5u8; 32]))
                    .expect("32 bytes");
                Arc::new(ModelCapabilityCache::new(Arc::new(conn), dek).expect("schema"))
            };
            let mut notices = Vec::new();
            let magi = build_magi_orchestrator(
                &TrioBuild {
                    cfg: &cfg,
                    principal_kind: ProviderKind::Ollama,
                    endpoints: &test_endpoints(),
                    creds: None,
                    warn_tokens: None,
                    env_overrides: &MagiEnvModelOverrides::default(),
                    capability_cache: Some(&cache),
                    measured: &BTreeMap::new(),
                },
                &mut notices,
            )
            .expect("a repeated model must not fail the build");
            drop(magi);

            let wired = pool_wiring_trace().expect("the pool is declared");
            assert!(
                wired.candidates.iter().any(|(model, _)| model == "shared"),
                "the pool entry repeating a seat's model is KEPT, not pruned: `used` is per mage, \
                 so another seat can still rotate into it: {wired:?}"
            );
            // SC-R52's other half, and the reason this assertion is here rather than in a test of
            // its own: without it the whole notice block can be deleted and the suite stays green.
            assert!(
                notices
                    .iter()
                    .any(|n| n.text.contains("shared") && n.text.contains("repeats a model")),
                "the repetition must be announced — silent, it surfaces only as an unexpectedly \
                 short rotation chain during a real incident: {notices:?}"
            );
        }

        /// The env layer is what the collapse notice must see, and the ONLY thing that pins it.
        ///
        /// The fix for this gate's first CRITICAL moved correctness from inside
        /// `diversity_notices` — where it could not be got wrong — out to the CALLER, which is
        /// `pub(crate)` and can be handed anything. Reverting the call site to
        /// `cfg.magi().seats(backend_model)` compiles, and every other test here uses
        /// `MagiEnvModelOverrides::default()`, so the env layer is never in play and the whole
        /// regression comes back green. This test is the one that goes red.
        ///
        /// Both directions, because the fix claimed both: a DECLARED, distinct trio collapsed by
        /// three identical overrides must be reported, and an undeclared trio pulled apart by
        /// three distinct ones must NOT be.
        #[test]
        fn the_collapse_notice_reads_the_env_overrides_not_the_declared_models() {
            let declared_distinct = MagiConfig::from_toml_str(
                "provider = \"ollama\"\n\
                 [magi]\n\
                 melchior_model    = \"m-a\"\nmelchior_lineage  = \"la\"\n\
                 balthasar_model   = \"m-b\"\nbalthasar_lineage = \"lb\"\n\
                 caspar_model      = \"m-c\"\ncaspar_lineage    = \"lc\"\n",
            )
            .expect("a declared, distinct trio loads");

            let collapsed_by_env = MagiEnvModelOverrides {
                melchior: Some("one-model".to_string()),
                balthasar: Some("one-model".to_string()),
                caspar: Some("one-model".to_string()),
            };
            let mut notices = Vec::new();
            let magi = build_magi_orchestrator(
                &TrioBuild {
                    cfg: &declared_distinct,
                    principal_kind: ProviderKind::Ollama,
                    endpoints: &test_endpoints(),
                    creds: None,
                    warn_tokens: None,
                    env_overrides: &collapsed_by_env,
                    capability_cache: None,
                    measured: &BTreeMap::new(),
                },
                &mut notices,
            )
            .expect("ollama is keyless");
            drop(magi);
            assert!(
                notices
                    .iter()
                    .any(|n| n.text.contains("resolve to the same model")),
                "the TOML says three models; the env says one, and the env is what runs: \
                 {notices:?}"
            );

            // The mirror. Without this half, a notice that fired unconditionally would pass.
            let pulled_apart_by_env = MagiEnvModelOverrides {
                melchior: Some("m-1".to_string()),
                balthasar: Some("m-2".to_string()),
                caspar: Some("m-3".to_string()),
            };
            let undeclared =
                MagiConfig::from_toml_str("provider = \"ollama\"\n").expect("defaults load");
            let mut notices = Vec::new();
            let magi = build_magi_orchestrator(
                &TrioBuild {
                    cfg: &undeclared,
                    principal_kind: ProviderKind::Ollama,
                    endpoints: &test_endpoints(),
                    creds: None,
                    warn_tokens: None,
                    env_overrides: &pulled_apart_by_env,
                    capability_cache: None,
                    measured: &BTreeMap::new(),
                },
                &mut notices,
            )
            .expect("ollama is keyless");
            drop(magi);
            assert!(
                !notices
                    .iter()
                    .any(|n| n.text.contains("resolve to the same model")),
                "three distinct overrides ARE three models; saying otherwise is a false \
                 statement about the user's own configuration: {notices:?}"
            );
        }

        /// REQ-R15/SC-R22's soft path, driven through the builder.
        ///
        /// `validate_diversity` returns coverage gaps as NOTICES when `enforce_diversity` is
        /// false, and that vector was being discarded. The unit tests of `diversity_notices`
        /// cannot catch its return: they use configs with no `[[magi.fallback]]`, so the coverage
        /// branch is skipped entirely and their notice count is satisfied by the collapse notice
        /// alone. Only a config with a pool that covers nobody exercises it.
        #[test]
        fn an_uncovered_seat_is_announced_by_the_builder_when_enforcement_is_off() {
            let cfg = MagiConfig::from_toml_str(
                "provider = \"ollama\"\n\
                 [magi]\n\
                 melchior_model    = \"m-a\"\nmelchior_lineage  = \"opus\"\n\
                 balthasar_model   = \"m-b\"\nbalthasar_lineage = \"sonnet\"\n\
                 caspar_model      = \"m-c\"\ncaspar_lineage    = \"haiku\"\n\
                 enforce_diversity = false\n\
                 [[magi.fallback]]\n\
                 model   = \"rescue\"\nlineage = \"opus\"\n",
            )
            .expect("the mono-provider exit must load");

            let mut notices = Vec::new();
            let magi = build_magi_orchestrator(
                &TrioBuild {
                    cfg: &cfg,
                    principal_kind: ProviderKind::Ollama,
                    endpoints: &test_endpoints(),
                    creds: None,
                    warn_tokens: None,
                    env_overrides: &MagiEnvModelOverrides::default(),
                    capability_cache: None,
                    measured: &BTreeMap::new(),
                },
                &mut notices,
            )
            .expect("ollama is keyless");
            drop(magi);

            let text = notices
                .iter()
                .map(|n| n.text.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            assert!(
                text.contains("no fallback coverage"),
                "an `opus` candidate covers only Melchior; the other two must be named: {text}"
            );
            assert!(
                text.contains("balthasar") && text.contains("caspar"),
                "and both named: {text}"
            );
            // "All of them in ONE message" is asserted HERE as a count, not implied by the
            // message above: `text` is the join of every notice, so a version that emitted one
            // per seat would read identically. The stronger `notices.len() == 1` would be wrong
            // at this layer — other notices are legitimate in a builder run — so the count is
            // scoped to the ones this property is about.
            assert_eq!(
                notices
                    .iter()
                    .filter(|n| n.text.contains("no fallback coverage"))
                    .count(),
                1,
                "one message for all uncovered seats, not one per seat: {text}"
            );
        }

        /// REQ-R29's collapse notice, driven through `build_magi_orchestrator`.
        ///
        /// The unit test for `diversity_notices` calls the function directly, which is exactly the
        /// shape that let the CRITICAL of this gate's first iteration through — and the commit
        /// that fixed it stated the rule it then failed to apply here: *only a test that goes
        /// through the real path can catch a missing wire.*
        #[test]
        fn three_seats_on_one_model_are_announced_by_the_builder() {
            let cfg = MagiConfig::from_toml_str("provider = \"ollama\"\n")
                .expect("an absent trio must still load");
            let mut notices = Vec::new();
            let magi = build_magi_orchestrator(
                &TrioBuild {
                    cfg: &cfg,
                    principal_kind: ProviderKind::Ollama,
                    endpoints: &test_endpoints(),
                    creds: None,
                    warn_tokens: None,
                    env_overrides: &MagiEnvModelOverrides::default(),
                    capability_cache: None,
                    measured: &BTreeMap::new(),
                },
                &mut notices,
            )
            .expect("the default configuration must still build a trio");
            drop(magi);

            assert!(
                notices
                    .iter()
                    .any(|n| n.text.contains("resolve to the same model")),
                "with no trio declared all three seats fall back to the backend model, and their \
                 three built-in labels describe one failure domain: {notices:?}"
            );
        }

        /// SC-R24: rotation does NOT depend on the probe. With nothing measured at all, a failing
        /// mage still rotates — measurement improves the decision, it does not enable it.
        #[tokio::test]
        async fn rotation_works_with_no_measurement_at_all() {
            use mockito::Matcher;
            let mut server = mockito::Server::new_async().await;
            let _down = server
                .mock("POST", "/v1/chat/completions")
                .match_body(Matcher::AllOf(vec![
                    Matcher::Regex("(?i)caspar".into()),
                    Matcher::Regex("down-model".into()),
                ]))
                .with_status(400)
                .with_body("{\"error\":\"unavailable\"}")
                .create_async()
                .await;
            for (seat, model, agent) in [
                ("(?i)caspar", "rescue-model", "caspar"),
                ("(?i)melchior", "ok-model", "melchior"),
                ("(?i)balthasar", "ok-model", "balthasar"),
            ] {
                server
                    .mock("POST", "/v1/chat/completions")
                    .match_body(Matcher::AllOf(vec![
                        Matcher::Regex(seat.into()),
                        Matcher::Regex(model.into()),
                    ]))
                    .with_status(200)
                    .with_body(verdict_body(agent))
                    .create_async()
                    .await;
            }
            let endpoints = ResolvedEndpoints {
                root: endpoint_at(&server.url()),
                magi: endpoint_at(&server.url()),
            };
            let mut notices = Vec::new();
            let magi = build_magi_orchestrator(
                &TrioBuild {
                    cfg: &cfg_with_pool(2),
                    principal_kind: ProviderKind::Ollama,
                    endpoints: &endpoints,
                    creds: None,
                    warn_tokens: None,
                    env_overrides: &MagiEnvModelOverrides::default(),
                    capability_cache: None,
                    // NOTHING measured — the cold-start state.
                    measured: &BTreeMap::new(),
                },
                &mut notices,
            )
            .expect("ollama is keyless");

            let report = magi
                .analyze(
                    &Mode::Analysis,
                    "a question long enough to be a real consult",
                )
                .await
                .expect("the consult must complete");
            assert!(
                !report.rotations[&AgentName::Caspar].chain.is_empty(),
                "rotation must work with no measurement: the probe improves the decision, it does \
                 not enable it"
            );
            assert!(!report.degraded, "and the run is not degraded");
        }

        /// SC-R02: exhausting the chain WITHOUT a verdict does degrade the run — the other half of
        /// SC-R01, and the one that keeps "a rotation is not a degradation" from quietly becoming
        /// "nothing degrades any more".
        #[tokio::test]
        async fn exhausting_the_rotation_chain_degrades_the_run() {
            use mockito::Matcher;
            let mut server = mockito::Server::new_async().await;
            // EVERY model Caspar can reach fails, primary and candidate alike.
            let _all_down = server
                .mock("POST", "/v1/chat/completions")
                .match_body(Matcher::Regex("(?i)caspar".into()))
                .with_status(400)
                .with_body("{\"error\":\"unavailable\"}")
                .create_async()
                .await;
            for (seat, agent) in [("(?i)melchior", "melchior"), ("(?i)balthasar", "balthasar")] {
                server
                    .mock("POST", "/v1/chat/completions")
                    .match_body(Matcher::Regex(seat.into()))
                    .with_status(200)
                    .with_body(verdict_body(agent))
                    .create_async()
                    .await;
            }
            let endpoints = ResolvedEndpoints {
                root: endpoint_at(&server.url()),
                magi: endpoint_at(&server.url()),
            };
            let mut notices = Vec::new();
            let magi = build_magi_orchestrator(
                &TrioBuild {
                    cfg: &cfg_with_pool(2),
                    principal_kind: ProviderKind::Ollama,
                    endpoints: &endpoints,
                    creds: None,
                    warn_tokens: None,
                    env_overrides: &MagiEnvModelOverrides::default(),
                    capability_cache: None,
                    measured: &BTreeMap::new(),
                },
                &mut notices,
            )
            .expect("ollama is keyless");

            let report = magi
                .analyze(
                    &Mode::Analysis,
                    "a question long enough to be a real consult",
                )
                .await
                .expect("the consult must return even when a seat is lost");
            assert!(
                report.degraded,
                "no verdict after the whole chain IS degradation"
            );
            assert!(
                report.failed_agents.contains_key(&AgentName::Caspar),
                "and the lost seat is named: {:?}",
                report.failed_agents
            );
        }

        /// SC-R30: with NO cache the stateless path measures a FOURTH model — the first pool
        /// candidate — and with a cache it does not.
        ///
        /// Both halves are asserted, and the second is what makes the first mean anything: a test
        /// that only checked "four without a cache" would pass just as well if the extra candidate
        /// were measured unconditionally, which would pay on every start for a model the lazy path
        /// already covers.
        ///
        /// The FIRST candidate and not the pool: the list is ordered strongest to weakest, so it
        /// is the most likely rotation destination, and measuring the rest would spend on
        /// candidates that probably never run.
        #[test]
        fn the_stateless_path_measures_the_first_candidate_and_the_cached_one_does_not() {
            let cfg = cfg_with_pool(2);
            let stateless = stateless_extra_models(&cfg, None);
            assert_eq!(
                stateless,
                vec!["rescue-model".to_string()],
                "without a cache nothing else would ever measure a candidate: no CachedProbe is \
                 built, so there is no lazy path at all"
            );

            let conn = std::sync::Mutex::new(
                rusqlite::Connection::open_in_memory().expect("in-memory sqlite"),
            );
            let dek = magi_rs::vault::MaskedDek::new(zeroize::Zeroizing::new(vec![3u8; 32]))
                .expect("32 bytes");
            let cache = Arc::new(ModelCapabilityCache::new(Arc::new(conn), dek).expect("schema"));
            assert!(
                stateless_extra_models(&cfg, Some(&cache)).is_empty(),
                "with a cache the measurement is lazy: paying for the candidate at every start \
                 would be the cost the cache exists to avoid"
            );
        }

        /// REQ-R27 END TO END: a configured model this endpoint could not measure **while it was
        /// measuring others** is named at STARTUP, with the command that fixes it.
        ///
        /// This is the guardian for the wiring, not for the function: `missing_model_notices` was
        /// implemented and tested in Phase 2 and had **no production caller** until Phase 6, which
        /// meant the requirement was not delivered at all — the user saw nothing at startup and
        /// found out when a mage fell. A unit test of the function could never have caught that.
        #[test]
        fn a_configured_model_the_endpoint_could_not_measure_is_named_at_startup() {
            let mut notices = Vec::new();
            let magi = build_magi_orchestrator(
                &TrioBuild {
                    cfg: &cfg_with_pool(2),
                    principal_kind: ProviderKind::Ollama,
                    endpoints: &test_endpoints(),
                    creds: None,
                    warn_tokens: None,
                    env_overrides: &MagiEnvModelOverrides::default(),
                    capability_cache: None,
                    // The endpoint clearly measures — it answered for `ok-model` — and did not
                    // answer for the fallback. That is what separates "this model is not there"
                    // from "this endpoint is not answering".
                    measured: &[
                        (
                            "ok-model".to_string(),
                            magi_rs::magi::probe::Measurement::Measured {
                                window: 128_000,
                                digest: None,
                            },
                        ),
                        (
                            "rescue-model".to_string(),
                            magi_rs::magi::probe::Measurement::NotMeasuredThisTime,
                        ),
                    ]
                    .into_iter()
                    .collect(),
                },
                &mut notices,
            )
            .expect("ollama is keyless");
            drop(magi);

            assert!(
                notices
                    .iter()
                    .any(|n| n.text.contains("rescue-model") && n.text.contains("ollama pull")),
                "the startup notice must name the model AND the command that fixes it: {notices:?}"
            );
        }

        /// SC-R33: the warning threshold does NOT change inside the process — two consults in one
        /// session keep the same criterion, even when a mage rotated to a smaller-window model in
        /// between.
        ///
        /// Re-deriving would be structurally impossible anyway (`with_input_warn_tokens` is a
        /// BUILDER method and magi-rs builds the orchestrator once per process), and that is
        /// exactly why this is worth pinning rather than asserting in prose: the property is
        /// currently free, so nothing would complain if a later change started rebuilding the
        /// orchestrator per consult and quietly made the criterion drift between two identical
        /// questions.
        #[tokio::test]
        async fn the_warn_threshold_is_the_same_before_and_after_a_rotation() {
            use mockito::Matcher;
            let mut server = mockito::Server::new_async().await;
            let _down = server
                .mock("POST", "/v1/chat/completions")
                .match_body(Matcher::AllOf(vec![
                    Matcher::Regex("(?i)caspar".into()),
                    Matcher::Regex("down-model".into()),
                ]))
                .with_status(400)
                .with_body("{\"error\":\"model unavailable\"}")
                .create_async()
                .await;
            for (seat, model, agent) in [
                ("(?i)caspar", "rescue-model", "caspar"),
                ("(?i)melchior", "ok-model", "melchior"),
                ("(?i)balthasar", "ok-model", "balthasar"),
            ] {
                server
                    .mock("POST", "/v1/chat/completions")
                    .match_body(Matcher::AllOf(vec![
                        Matcher::Regex(seat.into()),
                        Matcher::Regex(model.into()),
                    ]))
                    .with_status(200)
                    .with_body(verdict_body(agent))
                    .expect_at_least(1)
                    .create_async()
                    .await;
            }

            let endpoints = ResolvedEndpoints {
                root: endpoint_at(&server.url()),
                magi: endpoint_at(&server.url()),
            };
            let mut notices = Vec::new();
            let magi = build_magi_orchestrator(
                &TrioBuild {
                    cfg: &cfg_with_pool(2),
                    principal_kind: ProviderKind::Ollama,
                    endpoints: &endpoints,
                    creds: None,
                    warn_tokens: Some(96_000),
                    env_overrides: &MagiEnvModelOverrides::default(),
                    capability_cache: None,
                    measured: &BTreeMap::new(),
                },
                &mut notices,
            )
            .expect("ollama is keyless");

            let prompt = "a question long enough to be a real consult";
            let first = magi
                .analyze(&Mode::Analysis, prompt)
                .await
                .expect("first consult");
            let second = magi
                .analyze(&Mode::Analysis, prompt)
                .await
                .expect("second consult");

            assert!(
                !second.rotations[&AgentName::Caspar].chain.is_empty(),
                "the second consult must actually have rotated, or this proves nothing"
            );
            // Both halves are read as VALUES, not as options: `assert_eq!(None, None)` would pass
            // just as well if the report never carried a threshold at all, which is the vacuous
            // shape this milestone has already found five times.
            let first_threshold = first
                .input_size
                .as_ref()
                .map(|s| s.warn_threshold)
                .expect("the report must carry a threshold, or there is nothing to compare");
            let second_threshold = second
                .input_size
                .as_ref()
                .map(|s| s.warn_threshold)
                .expect("same for the second consult");
            assert_eq!(
                first_threshold, second_threshold,
                "the criterion must not drift between two identical questions in one session"
            );
        }

        /// SC-R51, the half no unit test can cover: an unmeasured candidate is **announced and
        /// KEPT**.
        ///
        /// A test that only checked the notice exists would pass just as well if magi-rs were
        /// quietly dropping the candidate — the operator would see a warning about a candidate
        /// that is no longer there. Reading the pool magi-core actually received is what
        /// separates "informed" from "filtered".
        #[test]
        fn an_unmeasured_candidate_is_reported_and_still_reaches_the_pool() {
            let mut notices = Vec::new();
            let magi = build_magi_orchestrator(
                &TrioBuild {
                    cfg: &cfg_with_pool(2),
                    principal_kind: ProviderKind::Ollama,
                    endpoints: &test_endpoints(),
                    creds: None,
                    warn_tokens: None,
                    env_overrides: &MagiEnvModelOverrides::default(),
                    capability_cache: None,
                    // The trio measured; the candidate did not — exactly the shape that would
                    // tempt an implementation to "protect" the run by dropping it.
                    measured: &[
                        (
                            "ok-model".to_string(),
                            magi_rs::magi::probe::Measurement::Measured {
                                window: 128_000,
                                digest: None,
                            },
                        ),
                        (
                            "rescue-model".to_string(),
                            magi_rs::magi::probe::Measurement::NotMeasuredThisTime,
                        ),
                    ]
                    .into_iter()
                    .collect(),
                },
                &mut notices,
            )
            .expect("ollama is keyless");
            drop(magi);

            assert!(
                notices
                    .iter()
                    .any(|n| n.text.contains("rescue-model")
                        && n.text.contains("no measured window")),
                "the assumption must be announced: {notices:?}"
            );
            let wired = pool_wiring_trace().expect("the pool is declared");
            assert!(
                wired
                    .candidates
                    .iter()
                    .any(|(model, _)| model == "rescue-model"),
                "and the candidate must STILL be in the pool magi-core received: {wired:?}"
            );
        }

        /// SC-R04/REQ-R05: `max_rotations = 0` is a kill-switch, and it survives as a DECLARED
        /// value — collapsing `None` and `Some(0)` would turn an explicit "no rotation" into
        /// "use the default", which is the opposite instruction.
        #[test]
        fn max_rotations_zero_is_wired_as_a_declared_kill_switch() {
            let mut notices = Vec::new();
            let magi = build_magi_orchestrator(
                &TrioBuild {
                    cfg: &cfg_with_pool(0),
                    principal_kind: ProviderKind::Ollama,
                    endpoints: &test_endpoints(),
                    creds: None,
                    warn_tokens: None,
                    env_overrides: &MagiEnvModelOverrides::default(),
                    capability_cache: None,
                    measured: &BTreeMap::new(),
                },
                &mut notices,
            )
            .expect("ollama is keyless");
            drop(magi);

            let wired = pool_wiring_trace().expect("the pool is still declared");
            assert_eq!(
                wired.max_rotations, 0,
                "0 must reach magi-core as 0, never as the default"
            );
        }

        /// SC-R01 END TO END: a mage whose model fails **rotates to the declared candidate** and
        /// still emits a verdict, driven through the REAL `build_magi_orchestrator` rather than a
        /// hand-built `MagiBuilder`.
        ///
        /// This is the only test in the milestone that exercises the whole chain — config →
        /// pool → provider → rotation → report — so it is the one that would catch a wiring
        /// mistake the trace-based tests above cannot see, such as a pool built from the right
        /// models against the wrong endpoint.
        ///
        /// The server discriminates by **model**, which is what the request body carries, and the
        /// matchers are mutually exclusive so the result does not depend on mockito's matching
        /// order. `down-model` answers **400**: a non-retryable status, so `RetryProvider` gives
        /// up at once and the rotation is what is being measured, not the retry backoff.
        #[tokio::test]
        async fn a_mage_whose_model_fails_rotates_to_the_declared_candidate() {
            use mockito::Matcher;
            let mut server = mockito::Server::new_async().await;

            let _down = server
                .mock("POST", "/v1/chat/completions")
                .match_body(Matcher::AllOf(vec![
                    Matcher::Regex("(?i)caspar".into()),
                    Matcher::Regex("down-model".into()),
                ]))
                .with_status(400)
                .with_body("{\"error\":\"model unavailable\"}")
                .expect_at_least(1)
                .create_async()
                .await;
            let _rescue = server
                .mock("POST", "/v1/chat/completions")
                .match_body(Matcher::AllOf(vec![
                    Matcher::Regex("(?i)caspar".into()),
                    Matcher::Regex("rescue-model".into()),
                ]))
                .with_status(200)
                .with_body(verdict_body("caspar"))
                .expect_at_least(1)
                .create_async()
                .await;
            let _melchior = server
                .mock("POST", "/v1/chat/completions")
                .match_body(Matcher::Regex("(?i)melchior".into()))
                .with_status(200)
                .with_body(verdict_body("melchior"))
                .create_async()
                .await;
            let _balthasar = server
                .mock("POST", "/v1/chat/completions")
                .match_body(Matcher::Regex("(?i)balthasar".into()))
                .with_status(200)
                .with_body(verdict_body("balthasar"))
                .create_async()
                .await;

            let endpoint = endpoint_at(&server.url());
            let endpoints = ResolvedEndpoints {
                root: endpoint_at(&server.url()),
                magi: endpoint,
            };
            let mut notices = Vec::new();
            let magi = build_magi_orchestrator(
                &TrioBuild {
                    cfg: &cfg_with_pool(2),
                    principal_kind: ProviderKind::Ollama,
                    endpoints: &endpoints,
                    creds: None,
                    warn_tokens: None,
                    env_overrides: &MagiEnvModelOverrides::default(),
                    capability_cache: None,
                    measured: &BTreeMap::new(),
                },
                &mut notices,
            )
            .expect("ollama is keyless");

            let report = magi
                .analyze(
                    &Mode::Analysis,
                    "a question long enough to be a real consult",
                )
                .await
                .expect("the consult must complete");

            assert_eq!(
                report.agents.len(),
                3,
                "the consensus must still have three"
            );
            assert!(!report.degraded, "a rotation is NOT degradation (REQ-R04)");

            // `rotations` is populated for EVERY agent, rotated or not — its rustdoc calls it
            // "always present". What distinguishes the one that rotated is a non-empty `chain`.
            let caspar = report
                .rotations
                .get(&AgentName::Caspar)
                .expect("rotations is always present, for every agent");
            assert!(
                !caspar.chain.is_empty(),
                "Caspar's model was down: it must have hopped"
            );
            assert_eq!(
                caspar.model_used, "rescue-model",
                "REQ-R06: the report must name the model that ACTUALLY produced the verdict"
            );
            assert!(
                report.rotations[&AgentName::Melchior].chain.is_empty(),
                "nobody else had a reason to rotate"
            );
        }

        /// SC-R12/REQ-R16: with a credential-bearing endpoint and a mage that **really fails and
        /// rotates**, no output surface carries the credential.
        ///
        /// # Why it drives the real path instead of building the error
        ///
        /// Every previous no-leak assertion for rotation could have been written by handing
        /// `redact_foreign_text` a string with a credential in it — and that guardian proves only
        /// that the redactor redacts, which its own unit tests already prove. It says nothing
        /// about whether the **composition** calls it. This one goes through
        /// `build_magi_orchestrator`, a real 400, a real rotation, and the real renderers.
        ///
        /// # The honest limit, stated rather than discovered later
        ///
        /// Whether the credential ever reaches a `RotationEvent::detail` is **magi-core's**
        /// choice, and reqwest already strips userinfo from the URL it echoes. So the canary may
        /// well be absent for reasons that have nothing to do with our redaction — which is
        /// exactly how a no-leak test becomes a guardian of nothing. That is why the assertions
        /// come in pairs: each surface must be **non-trivially rendered** before its absence of
        /// the canary means anything. An empty string contains no secret either.
        #[tokio::test]
        async fn no_output_carries_the_credential_when_a_mage_really_fails_and_rotates() {
            use magi_rs::magi::rotation_report::{render_rotations, rotation_lines};
            use mockito::Matcher;
            let canary = super::divergence_and_keyless_auth::SEAT_CANARY;
            let mut server = mockito::Server::new_async().await;

            let _down = server
                .mock("POST", "/v1/chat/completions")
                .match_body(Matcher::AllOf(vec![
                    Matcher::Regex("(?i)caspar".into()),
                    Matcher::Regex("down-model".into()),
                ]))
                .with_status(400)
                .with_body("{\"error\":\"model unavailable\"}")
                .create_async()
                .await;
            for (seat, model) in [
                ("(?i)caspar", "rescue-model"),
                ("(?i)melchior", "melchior"),
                ("(?i)balthasar", "balthasar"),
            ] {
                let agent = if model == "rescue-model" {
                    "caspar"
                } else {
                    model
                };
                server
                    .mock("POST", "/v1/chat/completions")
                    .match_body(Matcher::AllOf(vec![
                        Matcher::Regex(seat.into()),
                        Matcher::Regex(
                            if model == "rescue-model" {
                                "rescue-model"
                            } else {
                                "ok-model"
                            }
                            .into(),
                        ),
                    ]))
                    .with_status(200)
                    .with_body(verdict_body(agent))
                    .create_async()
                    .await;
            }

            // The ONLY way to obtain a credential-bearing endpoint: REQ-A16c rejects a literal
            // one at parse time, so it has to come from placeholders resolved against a vault.
            let host = server.url().replace("http://", "");
            let endpoint = super::divergence_and_keyless_auth::credentialed_endpoint(&format!(
                "http://[user]:[password]@{host}/v1"
            ));
            let endpoints = ResolvedEndpoints {
                root: super::divergence_and_keyless_auth::credentialed_endpoint(&format!(
                    "http://[user]:[password]@{host}/v1"
                )),
                magi: endpoint,
            };

            let mut notices = Vec::new();
            let magi = build_magi_orchestrator(
                &TrioBuild {
                    cfg: &cfg_with_pool(2),
                    principal_kind: ProviderKind::Ollama,
                    endpoints: &endpoints,
                    creds: None,
                    warn_tokens: None,
                    env_overrides: &MagiEnvModelOverrides::default(),
                    capability_cache: None,
                    measured: &BTreeMap::new(),
                },
                &mut notices,
            )
            .expect("ollama is keyless: the credential rides in the URL, not in a header we set");

            let report = magi
                .analyze(
                    &Mode::Analysis,
                    "a question long enough to be a real consult",
                )
                .await
                .expect("the consult must complete");

            // FIRST HALF: the telemetry really rendered. Without this, every assertion below
            // would pass against a run that produced no rotation output at all.
            assert!(
                !report.rotations[&AgentName::Caspar].chain.is_empty(),
                "the canary means nothing unless a rotation actually happened"
            );
            let json = render_rotations(&report.rotations).to_string();
            let lines = rotation_lines(&report.rotations).join("\n");
            let annotated =
                crate::tools::consult::annotate_report_text(&report, ProviderKind::Ollama);
            assert!(
                json.contains("rescue-model") && lines.contains("rescue-model"),
                "both renderers must have produced real telemetry: {json} / {lines}"
            );
            assert!(
                annotated.contains("Model rotations"),
                "the text surface must carry the rotation section: {annotated}"
            );

            // SECOND HALF: and none of it carries the credential.
            for surface in [&json, &lines, &annotated] {
                assert!(
                    !surface.contains(canary),
                    "the credential reached an output surface: {surface}"
                );
                assert!(
                    !surface.contains("alice"),
                    "the username reached an output surface: {surface}"
                );
            }
        }

        /// Fixed test credentials, no env or vault behind them.
        struct FixedCreds {
            openai: Option<String>,
            anthropic: Option<String>,
        }
        impl Credentials for FixedCreds {
            fn openai(&self) -> Option<String> {
                self.openai.clone()
            }
            fn anthropic(&self) -> Option<String> {
                self.anthropic.clone()
            }
        }
        fn creds() -> FixedCreds {
            FixedCreds {
                openai: Some("test-openai-key".to_string()),
                anthropic: Some("claude-test-key".to_string()),
            }
        }

        fn cfg_openai_compat_without_credentials() -> MagiConfig {
            MagiConfig::from_toml_str("provider = \"openai-compat\"\n").unwrap()
        }

        /// Only Caspar has a model that does not resolve to a valid Claude alias; the other two
        /// inherit the backend model (`[anthropic].model`, valid). Making ONE seat fail without
        /// touching the other two needs an axis that varies by seat — an invalid model is the
        /// only one `build_native_provider` has (credential and endpoint are shared across all
        /// three).
        ///
        /// The string MUST NOT contain the `"claude-"` substring anywhere:
        /// `resolve_claude_alias` (magi-core) accepts ANY model containing it as pass-through —
        /// `"not-a-real-claude-alias"` contains it (`…real-`**`claude-`**`alias`) and therefore
        /// resolved OK, so it was not the broken fixture this test needed.
        fn cfg_with_only_caspar_unbuildable() -> MagiConfig {
            MagiConfig::from_toml_str(
                "provider = \"anthropic\"\n\
                 [anthropic]\n\
                 model = \"claude-sonnet-4-6\"\n\
                 [magi]\n\
                 caspar_model = \"totally-bogus-alias\"\n\
                 caspar_lineage = \"bogus\"\n",
            )
            .unwrap()
        }

        /// Hand-built, deliberately skipping `from_toml_str`'s validation
        /// (`validate_vocabulary`): an invalid `kind` never reaches `build_magi_orchestrator`
        /// through the real production path (`load()`/`from_toml_str()` already reject it
        /// earlier), but the function must still report it — defense in depth, see the rustdoc
        /// of `build_magi_orchestrator` on why it does NOT use `cfg.effective_magi_kind()` for
        /// this.
        fn cfg_with_kind(kind: &str) -> MagiConfig {
            // `build_unvalidated`, not `build`: this fixture's whole point is a `kind` the
            // validation would reject, so the validating exit cannot be the one used here.
            MagiConfig::builder()
                .provider(Some("ollama".to_string()))
                .magi(crate::config::MagiSectionConfig {
                    kind: Some(kind.to_string()),
                    ..crate::config::MagiSectionConfig::default()
                })
                .build_unvalidated()
        }

        /// Content above any magi-core internal minimum — not the REQ-A20 complexity gate
        /// (`Magi::analyze` does not use it here: `[NO usar
        /// MagiBuilder::with_complexity_gate]`, spec decision), just "non-empty and realistic".
        fn content_above_gate() -> String {
            "x".repeat(300)
        }

        /// Valid verdict ATTRIBUTED to the correct agent — magi-core's
        /// `parse_validate_and_check` rejects a verdict whose `"agent"` field does not match
        /// who it was dispatched to (`AgentIdentity`), so `valid_verdict_for_current_agent()`
        /// outside a `CURRENT_AGENT_IDENTITY` scope is not enough; that task-local is private
        /// to magi-core and only active DURING real dispatch, not when pre-building the
        /// response for a `RoutingMockProvider`.
        fn verdict_for(agent: AgentName) -> String {
            let name = serde_json::to_value(agent)
                .ok()
                .and_then(|v| v.as_str().map(str::to_owned))
                .unwrap_or_else(|| "melchior".to_string());
            format!(
                "{}\n{{\"agent\":\"{name}\",\"verdict\":\"approve\",\"confidence\":0.9,\
                 \"summary\":\"ok\",\"reasoning\":\"r\",\"recommendation\":\"go\",\
                 \"findings\":[]}}\n{}",
                magi_core::verdict_markers::VERDICT_OPEN,
                magi_core::verdict_markers::VERDICT_CLOSE,
            )
        }

        /// Test provider that records the system/user prompt it receives. Each seat of the trio
        /// gets its OWN instance (not shared): magi-core routes by assignment
        /// (`MagiBuilder::with_provider`), not by task identity, so reading
        /// `CURRENT_AGENT_IDENTITY` (private to magi-core) is not needed to know "for whom"
        /// this call is — the instance ALREADY knows, by construction.
        struct CapturingProvider {
            captured: std::sync::Mutex<Option<(String, String)>>,
        }
        impl CapturingProvider {
            fn new() -> Arc<Self> {
                Arc::new(Self {
                    captured: std::sync::Mutex::new(None),
                })
            }
            fn system_prompt(&self) -> String {
                self.captured
                    .lock()
                    .unwrap()
                    .clone()
                    .map(|(s, _)| s)
                    .unwrap_or_default()
            }
            fn user_prompt(&self) -> String {
                self.captured
                    .lock()
                    .unwrap()
                    .clone()
                    .map(|(_, u)| u)
                    .unwrap_or_default()
            }
        }
        #[async_trait::async_trait]
        impl LlmProvider for CapturingProvider {
            async fn complete(
                &self,
                system_prompt: &str,
                user_prompt: &str,
                _config: &CompletionConfig,
            ) -> Result<String, ProviderError> {
                *self.captured.lock().unwrap() =
                    Some((system_prompt.to_string(), user_prompt.to_string()));
                Ok(valid_verdict_for_current_agent())
            }
            fn name(&self) -> &str {
                "capturing"
            }
            fn model(&self) -> &str {
                "capturing"
            }
        }

        struct CapturedTrio {
            melchior: Arc<CapturingProvider>,
            balthasar: Arc<CapturingProvider>,
            caspar: Arc<CapturingProvider>,
        }
        impl CapturedTrio {
            fn system_prompt_of(&self, seat: AgentName) -> String {
                match seat {
                    AgentName::Melchior => self.melchior.system_prompt(),
                    AgentName::Balthasar => self.balthasar.system_prompt(),
                    AgentName::Caspar => self.caspar.system_prompt(),
                }
            }
            fn user_prompt_of(&self, seat: AgentName) -> String {
                match seat {
                    AgentName::Melchior => self.melchior.user_prompt(),
                    AgentName::Balthasar => self.balthasar.user_prompt(),
                    AgentName::Caspar => self.caspar.user_prompt(),
                }
            }
        }

        /// Builds a trio EXACTLY in the shape `build_magi_orchestrator` builds its — each seat
        /// with its OWN provider via `MagiBuilder::with_provider()`, with no adapter folding
        /// the prompt — to verify SC-A01. It does not go through `build_magi_orchestrator`
        /// itself: that one builds real HTTP providers
        /// (`OpenAiCompatibleProvider`/`ClaudeProvider`) for which there is no test-double
        /// injection point, and calling it would hit the network (forbidden, R-A04). This
        /// function tests the SAME construction shape with test providers in their place —
        /// together with SC-A02 (which fixes that no folding adapter exists in production),
        /// they close the property end to end.
        async fn build_trio_with_capturing_providers() -> CapturedTrio {
            let melchior = CapturingProvider::new();
            let balthasar = CapturingProvider::new();
            let caspar = CapturingProvider::new();
            let magi = MagiBuilder::new(melchior.clone() as Arc<dyn LlmProvider>)
                .with_provider(AgentName::Melchior, melchior.clone())
                .with_provider(AgentName::Balthasar, balthasar.clone())
                .with_provider(AgentName::Caspar, caspar.clone())
                .build()
                .expect("test trio should build");
            let _ = magi.analyze(&Mode::Analysis, &content_above_gate()).await;
            CapturedTrio {
                melchior,
                balthasar,
                caspar,
            }
        }

        /// SC-A01: the system prompt arrives INTACT through the provider channel, never folded
        /// inside the user turn. It is the property that stopped holding when the trio went
        /// through `MagiCoreProviderAdapter` (retired this same task): that adapter
        /// concatenated `"{system}\n\n{user}"` before reaching a single-channel magi-rs
        /// `Provider`. Nothing in `build_native_provider`/`build_magi_orchestrator` does that —
        /// it returns the native provider DIRECTLY, with no wrapper.
        ///
        /// **Honest note (fix round 2, I2)**: this test does NOT call
        /// `build_magi_orchestrator` — it cannot: that function always builds real HTTP
        /// providers (`OpenAiCompatibleProvider`/`ClaudeProvider`), with no injection point for
        /// a double, so invoking it would hit the network (forbidden, R-A04). And even if it
        /// could, `build_magi_orchestrator` does NOT determine the CONTENT of the system prompt
        /// — that is decided by `agent_factory.create_agents_with_prompts` in magi-core, from
        /// `AgentName` and `Mode`, after the trio is already built. Introspecting that function
        /// cannot prove anything about prompt distinctness: it is structurally the wrong
        /// function for this property. What this function DOES prove (SC-A03, below) is that
        /// `build_magi_orchestrator` wired three DISTINCT seats via `.with_provider()` — the
        /// exact pattern this test also uses, and the one that makes magi-core deliver a
        /// different persona to each one.
        #[tokio::test]
        async fn each_mage_receives_its_system_prompt_in_the_providers_own_channel() {
            let captured = build_trio_with_capturing_providers().await;
            let mut system_prompts = Vec::new();
            for seat in [AgentName::Melchior, AgentName::Balthasar, AgentName::Caspar] {
                let system = captured.system_prompt_of(seat);
                let user = captured.user_prompt_of(seat);
                assert!(
                    !system.is_empty(),
                    "{seat:?}: did not receive a system prompt"
                );
                assert!(
                    !user.contains(&system),
                    "{seat:?}: the system prompt was folded inside the user turn"
                );
                system_prompts.push(system);
            }
            // I2 (fix round 2): the property this task exists to restore is DISTINCTNESS
            // between seats, not just "non-empty" — three identical prompts would pass the
            // assertions above without saying anything false.
            assert_ne!(
                system_prompts[0], system_prompts[1],
                "Melchior and Balthasar received the SAME system prompt"
            );
            assert_ne!(
                system_prompts[0], system_prompts[2],
                "Melchior and Caspar received the SAME system prompt"
            );
            assert_ne!(
                system_prompts[1], system_prompts[2],
                "Balthasar and Caspar received the SAME system prompt"
            );
        }

        /// I2 (fix round 2, IMPORTANT): SC-A03 and SC-A05 (below) asserted against a
        /// `MagiBuilder` of their own, wrapped in their OWN `RetryProvider` — removing
        /// `build_magi_orchestrator`'s real wrapper left them green anyway, while their comment
        /// said "without the wrapper, this test turns red". That assertion was false.
        ///
        /// This test calls the REAL function, with `ollama` (keyless, no credential or network
        /// needed to BUILD — it only builds the HTTP client, not use it), and reads
        /// `seat_wiring_trace()` — the trace `build_magi_orchestrator` leaves ONLY in test
        /// builds, in the SAME branch that does the real wrapping.
        ///
        /// # It did not guard until Task 3.1 measured it (B16)
        ///
        /// The trace's wrapped flag used to be the literal `true`, and this rustdoc used to
        /// claim that removing `RetryProvider::with_config(...)` from production "without
        /// touching the trace" would break the count. The mutation says otherwise: replacing the
        /// wrap with a bare `p` left `seats.push` running, the literal unchanged, and this test
        /// **green** — through the exact regression it exists to catch. The flag is now the
        /// comparison of two allocation addresses, so the same mutation turns it red.
        ///
        /// It is still not runtime downcasting — `LlmProvider` is a foreign trait from magi-core
        /// with no `Any` (R-A01 forbids touching that crate) — so what this proves is that
        /// *something* wrapped the provider, not that the something is a `RetryProvider`.
        #[test]
        fn build_magi_orchestrator_wires_three_distinct_seats_each_wrapped_in_retry() {
            let cfg = MagiConfig::from_toml_str("provider = \"ollama\"\n").unwrap();
            let mut notices = Vec::new();
            let magi = build_magi_orchestrator(
                &TrioBuild {
                    cfg: &cfg,
                    principal_kind: ProviderKind::Ollama,
                    endpoints: &test_endpoints(),
                    creds: None,
                    warn_tokens: None,
                    env_overrides: &MagiEnvModelOverrides::default(),
                    capability_cache: None,
                    measured: &BTreeMap::new(),
                },
                &mut notices,
            )
            .expect("ollama is keyless: it must build with no credentials or network");
            drop(magi);

            let trace = seat_wiring_trace();
            assert_eq!(trace.len(), 3, "all three seats must end up wired");
            let seats: std::collections::HashSet<AgentName> =
                trace.iter().map(|(s, _, _)| *s).collect();
            assert_eq!(
                seats.len(),
                3,
                "the three wired seats must be DISTINCT from each other"
            );
            for expected in [AgentName::Melchior, AgentName::Balthasar, AgentName::Caspar] {
                assert!(seats.contains(&expected), "missing seat {expected:?}");
            }
            assert!(
                trace.iter().all(|(_, _, wrapped)| *wrapped),
                "all three seats must end up wrapped in RetryProvider (REQ-A03): {trace:?}"
            );
        }

        /// REQ-R01: each seat is registered with its DECLARED lineage. The registration migrates
        /// from `with_provider(seat, provider)` — what magi-rs used through v0.12.x — to
        /// `with_agent(seat, provider, lineage)`, which is the only door that carries the rotation
        /// diversity key.
        ///
        /// Asserted against `seat_lineage_trace()`, recorded in the SAME loop that calls
        /// `with_agent`, because `MagiBuilder::agent_lineages` is private and `Magi` exposes no
        /// reader for it: from outside the crate there is no other way to see what was registered.
        ///
        /// The RETRY wrap surviving this migration is pinned by
        /// `build_magi_orchestrator_wires_three_distinct_seats_each_wrapped_in_retry` above, which
        /// already reads the trace left on the branch that does the real wrapping — no second copy
        /// of that assertion is written here.
        #[test]
        fn each_seat_is_registered_with_its_declared_lineage() {
            let cfg = MagiConfig::from_toml_str(
                "provider = \"ollama\"\n\
                 [magi]\n\
                 melchior_model    = \"m-model\"\nmelchior_lineage  = \"declared-melchior\"\n\
                 balthasar_model   = \"b-model\"\nbalthasar_lineage = \"declared-balthasar\"\n\
                 caspar_model \
                  = \"c-model\"\ncaspar_lineage    = \"declared-caspar\"\n",
            )
            .expect("a fully declared trio must parse");
            let mut notices = Vec::new();
            let magi = build_magi_orchestrator(
                &TrioBuild {
                    cfg: &cfg,
                    principal_kind: ProviderKind::Ollama,
                    endpoints: &test_endpoints(),
                    creds: None,
                    warn_tokens: None,
                    env_overrides: &MagiEnvModelOverrides::default(),
                    capability_cache: None,
                    measured: &BTreeMap::new(),
                },
                &mut notices,
            )
            .expect("ollama is keyless: it must build with no credentials or network");
            drop(magi);

            let registered: std::collections::BTreeMap<AgentName, String> =
                seat_lineage_trace().into_iter().collect();
            assert_eq!(
                registered.len(),
                3,
                "all three seats must declare a lineage"
            );
            for (seat, expected) in [
                (AgentName::Melchior, "declared-melchior"),
                (AgentName::Balthasar, "declared-balthasar"),
                (AgentName::Caspar, "declared-caspar"),
            ] {
                assert_eq!(
                    registered.get(&seat).map(String::as_str),
                    Some(expected),
                    "{seat:?} must register its OWN declared lineage: {registered:?}"
                );
            }
        }

        /// The half of the trigger rule that keeps the DEFAULT configuration usable: a seat that
        /// declares no model runs the built-in one and inherits the built-in lineage with it.
        ///
        /// Without this, whoever never touched `[magi]` gets three seats registered with no
        /// lineage — and `MagiBuilder::build()` rejects a blank one outright
        /// (`orchestrator.rs:644`), so the trio would not build at all.
        #[test]
        fn seats_that_declare_no_model_register_the_built_in_lineages() {
            let cfg = MagiConfig::from_toml_str("provider = \"ollama\"\n").unwrap();
            let mut notices = Vec::new();
            let magi = build_magi_orchestrator(
                &TrioBuild {
                    cfg: &cfg,
                    principal_kind: ProviderKind::Ollama,
                    endpoints: &test_endpoints(),
                    creds: None,
                    warn_tokens: None,
                    env_overrides: &MagiEnvModelOverrides::default(),
                    capability_cache: None,
                    measured: &BTreeMap::new(),
                },
                &mut notices,
            )
            .expect("the default configuration must still build a trio");
            drop(magi);

            let registered: std::collections::BTreeMap<AgentName, String> =
                seat_lineage_trace().into_iter().collect();
            for (seat, expected) in [
                (AgentName::Melchior, defaults::DEFAULT_MAGI_MELCHIOR_LINEAGE),
                (
                    AgentName::Balthasar,
                    defaults::DEFAULT_MAGI_BALTHASAR_LINEAGE,
                ),
                (AgentName::Caspar, defaults::DEFAULT_MAGI_CASPAR_LINEAGE),
            ] {
                assert_eq!(
                    registered.get(&seat).map(String::as_str),
                    Some(expected),
                    "{seat:?} must inherit its built-in lineage: {registered:?}"
                );
            }
        }

        /// SC-A02: no production path implements `LlmProvider` through an adapter that folds
        /// the prompt.
        ///
        /// NO raw `"impl LlmProvider for"` grep across ALL of `src/`: the test doubles in THIS
        /// same file (`CapturingProvider` above) also implement it, so that grep would have a
        /// designed-in false positive as soon as a single test double exists — which is exactly
        /// what this task needs to test SC-A01/SC-A03/SC-A05 without real network (R-A04). What
        /// SC-A02 actually asks — "prompt adaptation does not survive" — is verified by the
        /// ABSENCE of a CONSTRUCTION of the retired concrete type (its constructor call,
        /// `::new`), not by a blind trait grep nor by the absence of the NAME — the name still
        /// appears, deliberately, in comments documenting what was retired and why (this same
        /// file, `agent/messages.rs`, `tui/mod.rs`).
        #[test]
        fn no_production_path_implements_llm_provider_via_a_folding_adapter() {
            // Built at RUNTIME, in two pieces: written as one literal it would match
            // grep's OWN needle on THIS line, in THIS file — a self-referential false
            // positive that would make the test permanently red the moment it exists.
            let needle = format!("{}{}", "MagiCoreProviderAdapter", "::new");
            let out = std::process::Command::new("grep")
                .args(["-rl", &needle, "src/"])
                .current_dir(env!("CARGO_MANIFEST_DIR"))
                .output()
                .expect("grep must run");
            assert!(
                String::from_utf8_lossy(&out.stdout).trim().is_empty(),
                "the retired adapter type must never be CONSTRUCTED again in src/"
            );
        }

        /// SC-A02c: invalid `kind` ⇒ trio unbuildable; empty ⇒ inherited.
        #[test]
        fn an_unknown_kind_makes_the_trio_unbuildable_while_a_blank_one_inherits() {
            let mut notices = Vec::new();
            assert!(
                matches!(
                    build_magi_orchestrator(
                        &TrioBuild {
                            cfg: &cfg_with_kind("banana"),
                            principal_kind: ProviderKind::Ollama,
                            endpoints: &test_endpoints(),
                            creds: None,
                            warn_tokens: None,
                            env_overrides: &MagiEnvModelOverrides::default(),
                            capability_cache: None,
                            measured: &BTreeMap::new(),
                        },
                        &mut notices,
                    ),
                    Err(TrioError::UnknownKind(_))
                ),
                "an unrecognized kind must be reported typed, never guessed"
            );

            let c = creds();
            let mut notices = Vec::new();
            assert!(
                build_magi_orchestrator(
                    &TrioBuild {
                        cfg: &cfg_with_kind(""),
                        principal_kind: ProviderKind::Ollama,
                        endpoints: &test_endpoints(),
                        creds: Some(&c),
                        warn_tokens: None,
                        env_overrides: &MagiEnvModelOverrides::default(),
                        capability_cache: None,
                        measured: &BTreeMap::new(),
                    },
                    &mut notices,
                )
                .is_ok(),
                "an empty kind inherits from the principal instead of failing"
            );
        }

        /// I1 (fix round 2, IMPORTANT): an absent `[magi].kind` must inherit the ALREADY-
        /// RESOLVED principal `ProviderKind` (`principal_kind`, which already saw
        /// `MAGI_PROVIDER`) — not re-read `provider` from TOML on its own via
        /// `cfg.effective_provider()`. Otherwise `MAGI_PROVIDER=anthropic` moves the
        /// conversational agent but leaves the trio on whatever `provider` the file says.
        ///
        /// Observable signal: the TOML says `provider = "ollama"` (keyless — ANY credential,
        /// including none, builds), but the passed `principal_kind` is `Anthropic` and no
        /// credential is given. If inheritance re-reads TOML (the bug), the trio builds anyway
        /// because Ollama demands nothing; if it truly inherits `principal_kind`, it fails
        /// asking for `ANTHROPIC_API_KEY`.
        #[test]
        fn a_blank_magi_kind_inherits_the_resolved_principal_kind_not_a_toml_only_read() {
            let cfg = MagiConfig::from_toml_str("provider = \"ollama\"\n").unwrap();
            let mut notices = Vec::new();
            let err = match build_magi_orchestrator(
                &TrioBuild {
                    cfg: &cfg,
                    principal_kind: ProviderKind::Anthropic,
                    endpoints: &test_endpoints(),
                    creds: None,
                    warn_tokens: None,
                    env_overrides: &MagiEnvModelOverrides::default(),
                    capability_cache: None,
                    measured: &BTreeMap::new(),
                },
                &mut notices,
            ) {
                Ok(_) => panic!(
                    "the trio inherited \"ollama\" from TOML instead of the already-resolved \
                     principal's ProviderKind::Anthropic"
                ),
                Err(e) => e,
            };
            match err {
                TrioError::SeatUnbuildable { seats } => {
                    assert_eq!(seats.len(), 3);
                    assert!(seats.iter().all(|(_, cause)| matches!(
                        cause,
                        SeatError::MissingCredential {
                            var: "ANTHROPIC_API_KEY"
                        }
                    )));
                }
                other => {
                    panic!("expected SeatUnbuildable (Anthropic without credential), got {other:?}")
                }
            }
        }

        /// SC-A05b / SC-A05c: fallen seats are named, and ALL of them.
        ///
        /// The fixture deliberately uses `kind = "openai-compat"`: `ollama` is keyless, so it
        /// never produces `MissingCredential` and the test would prove nothing.
        #[test]
        fn unbuildable_seats_are_named_all_at_once() {
            let mut notices = Vec::new();
            // `.expect_err()` needs the `Ok` side (`Arc<Magi>`) to be `Debug`, which
            // it is not — `match` avoids that bound entirely.
            let err = match build_magi_orchestrator(
                &TrioBuild {
                    cfg: &cfg_openai_compat_without_credentials(),
                    principal_kind: ProviderKind::OpenAiCompat,
                    endpoints: &test_endpoints(),
                    creds: None,
                    warn_tokens: None,
                    env_overrides: &MagiEnvModelOverrides::default(),
                    capability_cache: None,
                    measured: &BTreeMap::new(),
                },
                &mut notices,
            ) {
                Ok(_) => panic!("without credential the trio is not buildable"),
                Err(e) => e,
            };
            match err {
                TrioError::SeatUnbuildable { seats } => {
                    assert_eq!(
                        seats.len(),
                        3,
                        "the three share a credential: reporting one at a time forces \
                         three startups"
                    );
                    assert!(seats
                        .iter()
                        .all(|(_, cause)| matches!(cause, SeatError::MissingCredential { .. })));
                }
                other => panic!("expected SeatUnbuildable, got {other:?}"),
            }
        }

        /// Partial seats: 1 of 3 fallen is also reported complete, and ONLY that one.
        #[test]
        fn a_partial_seat_failure_names_exactly_the_seats_that_failed() {
            let c = creds();
            let mut notices = Vec::new();
            let err = match build_magi_orchestrator(
                &TrioBuild {
                    cfg: &cfg_with_only_caspar_unbuildable(),
                    principal_kind: ProviderKind::Anthropic,
                    endpoints: &test_endpoints(),
                    creds: Some(&c),
                    warn_tokens: None,
                    env_overrides: &MagiEnvModelOverrides::default(),
                    capability_cache: None,
                    measured: &BTreeMap::new(),
                },
                &mut notices,
            ) {
                Ok(_) => panic!("one fallen seat is enough for the trio to be unbuildable"),
                Err(e) => e,
            };
            match err {
                TrioError::SeatUnbuildable { seats } => {
                    assert_eq!(seats.len(), 1);
                    assert_eq!(seats[0].0, AgentName::Caspar);
                }
                other => panic!("expected SeatUnbuildable, got {other:?}"),
            }
        }

        /// SC-A03: a transient failure retries and the mage responds.
        ///
        /// **Honest correction (fix round 2, I2)**: the previous assertion of this
        /// comment — "without the wrapper, this test turns red" — was FALSE. This test builds
        /// its OWN `RetryProvider` over a double, so removing the
        /// `RetryProvider::with_config(...)` real wrapper from `build_magi_orchestrator` does
        /// not affect it at all: there is no way to inject a double INSIDE that function (it
        /// always builds real `OpenAiCompatibleProvider`/`ClaudeProvider`, with no injection
        /// point), so testing the DYNAMIC behavior of the retry (that it actually retries and
        /// the mage responds) without real network requires a double outside that function.
        /// What the real function DOES prove —that it actually wraps each seat in
        /// `RetryProvider`— is tested by
        /// `build_magi_orchestrator_wires_three_distinct_seats_each_wrapped_in_retry` (above),
        /// via the trace that function leaves in test. The two tests together close REQ-A03:
        /// one confirms the WRAP exists in the real function, the other confirms THAT WRAP
        /// (same shape, same derived `RetryConfig`) actually retries and survives a transient
        /// failure.
        ///
        /// Uses magi-core's `RoutingMockProvider` (feature `test-utils`, already enabled)
        /// instead of a self-owned counter: it routes by seat via `CURRENT_AGENT_IDENTITY`, so
        /// a SINGLE instance shared across the three seats does not confuse one seat's
        /// responses with another's under magi-core's REAL parallel dispatch (verified,
        /// SC-A04e) — a naive shared counter would, and that is why this test does not assert a
        /// total attempt count: with three seats dispatched in parallel, each with its own
        /// `RetryProvider`, the total number of calls depends on the exact interleaving and is
        /// not the property REQ-A03 promises. What it does promise — and what this test asserts
        /// — is that all three seats SURVIVE their initial failure and the consensus becomes
        /// complete.
        #[tokio::test]
        async fn a_transient_failure_is_retried_and_the_mage_answers() {
            let transient = || Err(ProviderError::external("boom", ExternalErrorKind::Network));
            let shared = Arc::new(
                magi_core::test_support::RoutingMockProvider::new()
                    .with_agent_responses(
                        AgentName::Melchior,
                        vec![transient(), Ok(verdict_for(AgentName::Melchior))],
                    )
                    .with_agent_responses(
                        AgentName::Balthasar,
                        vec![transient(), Ok(verdict_for(AgentName::Balthasar))],
                    )
                    .with_agent_responses(
                        AgentName::Caspar,
                        vec![transient(), Ok(verdict_for(AgentName::Caspar))],
                    ),
            );
            let retry = RetryConfig::default();
            let wrap = || {
                Arc::new(RetryProvider::with_config(
                    shared.clone() as Arc<dyn LlmProvider>,
                    retry.clone(),
                )) as Arc<dyn LlmProvider>
            };
            let magi = MagiBuilder::new(wrap())
                .with_provider(AgentName::Melchior, wrap())
                .with_provider(AgentName::Balthasar, wrap())
                .with_provider(AgentName::Caspar, wrap())
                .build()
                .expect("test trio should build");

            let report = magi
                .analyze(&Mode::Analysis, &content_above_gate())
                .await
                .expect("responds despite each seat's transient failure");
            assert!(!report.degraded, "and the consensus stayed complete");
        }

        /// Provider that exhausts its retry budget by ALWAYS failing — "hangs" in the sense of
        /// REQ-A05: never produces a usable verdict, so the ONLY way out is for `RetryProvider`
        /// to give up by exhausted budget, never for the provider to "resolve itself".
        struct AlwaysFailingProvider {
            calls: std::sync::atomic::AtomicUsize,
        }
        #[async_trait::async_trait]
        impl LlmProvider for AlwaysFailingProvider {
            async fn complete(
                &self,
                _s: &str,
                _u: &str,
                _c: &CompletionConfig,
            ) -> Result<String, ProviderError> {
                self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(20)).await;
                Err(ProviderError::external("hangs", ExternalErrorKind::Timeout))
            }
            fn name(&self) -> &str {
                "always-failing"
            }
            fn model(&self) -> &str {
                "always-failing"
            }
        }

        /// SC-A05: a provider that never produces a verdict gives up with a TYPED reason, and
        /// does so well BEFORE exhausting the per-mage ceiling — a blunt cut from the external
        /// ceiling does not distinguish "hung" from "slow". It uses a small `operation_budget`
        /// instead of the one derived from `AGENT_TIMEOUT_SECS` (90 s): the property under test
        /// is the SHAPE of the abandonment (early, typed), not the exact value of the derived
        /// budget — that is already tested by
        /// `derived_scale_satisfies_invariant_across_the_whole_admissible_range` in
        /// `magi/mod.rs`, exhaustively, without spending real wall-clock seconds. A high
        /// `max_retries` (50) is what makes the signal unambiguous: if the budget were NOT
        /// capping the abandonment, exhausting 50 retries at 20 ms each would take ~1 s — far
        /// above the margin this test tolerates.
        ///
        /// **Honest note (fix round 2, I2)**: as with SC-A03, this test builds its
        /// OWN `RetryProvider` over a double — there is no way to inject a double inside
        /// `build_magi_orchestrator` (it always builds real HTTP providers). The DYNAMIC
        /// behavior of the abandonment is tested here, against a `RetryConfig` with the SAME
        /// shape the real function derives; that the real function actually applies that shape
        /// (wraps each seat) is tested by
        /// `build_magi_orchestrator_wires_three_distinct_seats_each_wrapped_in_retry`, above.
        #[tokio::test]
        async fn a_hanging_provider_abandons_before_the_ceiling() {
            let inner = Arc::new(AlwaysFailingProvider {
                calls: std::sync::atomic::AtomicUsize::new(0),
            });
            // `RetryConfig` is `#[non_exhaustive]`: build from `default()` and adjust
            // per field, same as `build_magi_orchestrator` itself.
            let mut retry = RetryConfig::default();
            retry.operation_budget = Duration::from_millis(60);
            retry.base_delay = Duration::ZERO;
            retry.max_retries = 50;
            let provider = RetryProvider::with_config(inner as Arc<dyn LlmProvider>, retry);

            let started = Instant::now();
            let err = provider
                .complete("s", "u", &CompletionConfig::default())
                .await
                .expect_err("must abandon");
            let elapsed = started.elapsed();

            assert!(
                matches!(err, ProviderError::RetryAbandoned { .. }),
                "the abandonment must name its cause (RetryAbandoned), not be a silent \
                 cutoff: {err}"
            );
            assert!(
                elapsed < Duration::from_millis(500),
                "abandoned well before what 50 real retries would take: {elapsed:?}"
            );
        }

        /// `resolve_endpoints`: the two fields it fails closed on (`root`, `magi`) are
        /// resolved at once — covers the `root` that `build_magi_orchestrator` does not
        /// touch (it only uses `.magi`), which would otherwise go unread. `embedding` is
        /// deliberately NOT part of this step (S8 review round, finding 1) — see
        /// `resolve_endpoints_does_not_fail_closed_on_an_unresolvable_embedding_placeholder`.
        #[test]
        fn resolve_endpoints_resolves_the_two_fields_from_the_same_root_when_none_diverge() {
            let cfg = MagiConfig::default();
            let resolved = resolve_endpoints(&cfg, None, None).expect("no placeholders, no vault");
            assert_eq!(
                resolved.root.as_str(),
                crate::defaults::DEFAULT_OPENAI_BASE_URL
            );
            assert_eq!(
                resolved.magi.as_str(),
                crate::defaults::DEFAULT_OPENAI_BASE_URL
            );
        }

        /// The trio may diverge from the root endpoint.
        #[test]
        fn resolve_endpoints_lets_the_trio_diverge_from_the_root() {
            let cfg = MagiConfig::from_toml_str(
                "base_url = \"http://root:11434/v1\"\n\
                 [magi]\n\
                 base_url = \"http://trio:11434/v1\"\n",
            )
            .unwrap();
            let resolved = resolve_endpoints(&cfg, None, None).unwrap();
            assert_eq!(resolved.root.as_str(), "http://root:11434/v1");
            assert_eq!(resolved.magi.as_str(), "http://trio:11434/v1");
        }

        /// **S8 review round, finding 1.** `resolve_endpoints` is the startup step that
        /// aborts the ENTIRE process (`?`-propagated by both `run()` and
        /// `prepare_headless()`) on any endpoint it cannot resolve. Before this fix it
        /// also resolved `[embedding].base_url` there, so a broken embedding placeholder
        /// — a missing vault entry, here simulated by having no vault open at all —
        /// aborted startup even for a session that will NEVER attach persistent memory
        /// (an ephemeral TUI run, or headless `--no-memory`): a config problem in a
        /// feature the user is not using stopped the program from starting at all.
        ///
        /// The embedding endpoint's real, authoritative resolution already lives in
        /// `resolve_effective_embedding_endpoint`, called from `attach_persistent_memory`
        /// ONLY when a vector store actually attaches — and it already degrades
        /// gracefully to text-only persistence with a notice on failure (REQ-29), never
        /// aborting the process. Root and the trio stay fail-closed here on purpose:
        /// unlike the embedder, they are in play for every session with a principal
        /// provider or a trio, so a broken config for either IS a config problem the
        /// current session cannot avoid paying for.
        #[test]
        fn resolve_endpoints_does_not_fail_closed_on_an_unresolvable_embedding_placeholder() {
            let cfg = MagiConfig::from_toml_str(
                "[embedding]\nbase_url = \"https://[user]:[password]@host/v1\"\n",
            )
            .unwrap();
            // No vault open (`secret_store = None`): the embedding placeholder cannot be
            // substituted. Root and the trio do not declare placeholders, so they are
            // unaffected — startup must proceed regardless.
            let resolved = resolve_endpoints(&cfg, None, None)
                .expect("a broken, unused embedding endpoint must never abort the whole process");
            assert_eq!(
                resolved.root.as_str(),
                crate::defaults::DEFAULT_OPENAI_BASE_URL
            );
            assert_eq!(
                resolved.magi.as_str(),
                crate::defaults::DEFAULT_OPENAI_BASE_URL
            );
        }

        /// I1 (fix round 2, IMPORTANT): `resolve_endpoints` must see the SAME `OPENAI_BASE_URL`
        /// layer that already applied to `resolve_effective_principal_endpoint` — otherwise the
        /// env var moves the conversational agent but leaves the trio pointing at the
        /// TOML/default `base_url` when `[magi].base_url` is absent (inheriting).
        #[test]
        fn resolve_endpoints_honors_openai_base_url_for_the_inherited_trio_endpoint() {
            let cfg = MagiConfig::default(); // no own base_url, no [magi].base_url
            let resolved = resolve_endpoints(&cfg, Some("http://otherhost:9999/v1"), None).unwrap();
            assert_eq!(resolved.root.as_str(), "http://otherhost:9999/v1");
            assert_eq!(
                resolved.magi.as_str(),
                "http://otherhost:9999/v1",
                "the trio inherits the root ALREADY resolved with its env layer, not one \
                 recalculated from TOML alone"
            );
        }

        /// An OWN `[magi].base_url` still wins over the root env var — the env var only fills
        /// the inheritance gap, not override an explicit declaration.
        #[test]
        fn resolve_endpoints_lets_a_declared_trio_base_url_win_over_the_root_env_override() {
            let cfg =
                MagiConfig::from_toml_str("[magi]\nbase_url = \"http://trio:11434/v1\"\n").unwrap();
            let resolved = resolve_endpoints(&cfg, Some("http://otherhost:9999/v1"), None).unwrap();
            assert_eq!(resolved.root.as_str(), "http://otherhost:9999/v1");
            assert_eq!(resolved.magi.as_str(), "http://trio:11434/v1");
        }

        #[test]
        fn seat_error_variants_have_a_readable_display() {
            assert!(SeatError::MissingCredential {
                var: "OPENAI_API_KEY"
            }
            .to_string()
            .contains("OPENAI_API_KEY"));
            let transport = SeatError::Transport(redact_foreign_error(&std::io::Error::other(
                "connection refused",
            )));
            assert!(transport.to_string().contains("connection refused"));
        }

        #[test]
        fn trio_error_no_seats_has_a_readable_display() {
            assert!(!TrioError::NoSeats.to_string().is_empty());
        }

        #[test]
        fn openai_compat_root_normalizes_a_missing_v1_suffix_and_says_so() {
            let (root, notice) = openai_compat_root("http://localhost:11434");
            assert_eq!(root, "http://localhost:11434/v1");
            assert!(notice.is_some());

            let (root, notice) = openai_compat_root("http://localhost:11434/v1");
            assert_eq!(root, "http://localhost:11434/v1");
            assert!(notice.is_none(), "already normalized: no notice");

            let (root, _) = openai_compat_root("http://localhost:11434/v1/");
            assert_eq!(
                root, "http://localhost:11434/v1",
                "idempotent against a trailing slash"
            );
        }

        /// C1 (fix round 2, CRITICAL, Security): the normalization notice interpolates the
        /// ALREADY-RESOLVED endpoint (post-placeholder substitution REQ-A16c) — if it carries
        /// credentials, they must reach the notice TEXT redacted, even though the `root`
        /// returned for the real provider still carries them intact (the provider DOES need
        /// them: `api_key = None` for Ollama does not cover `userinfo` in the URL).
        #[test]
        fn openai_compat_root_redacts_credentials_in_the_notice_but_not_in_the_root() {
            let (root, notice) = openai_compat_root("https://realuser:realpass@ollama.lan:11434");
            assert_eq!(
                root, "https://realuser:realpass@ollama.lan:11434/v1",
                "the real ROOT is what the provider needs to authenticate"
            );
            let notice = notice.expect("without /v1, must warn");
            assert!(
                !notice.contains("realuser") && !notice.contains("realpass"),
                "the credential must not reach the notice: {notice}"
            );
            assert!(
                notice.contains("ollama.lan"),
                "the host must stay visible: {notice}"
            );
        }

        #[test]
        fn env_vault_credentials_resolves_openai_and_anthropic_from_env() {
            let cfg = MagiConfig::default();
            let c = EnvVaultCredentials {
                magi_config: &cfg,
                anthropic_env: Some("sk-ant-test"),
                openai_env: Some("sk-oai-test"),
                secret_store: None,
            };
            assert_eq!(c.openai().as_deref(), Some("sk-oai-test"));
            assert_eq!(c.anthropic().as_deref(), Some("sk-ant-test"));
        }

        #[test]
        fn env_vault_credentials_is_none_without_env_or_vault() {
            let cfg = MagiConfig::default();
            let c = EnvVaultCredentials {
                magi_config: &cfg,
                anthropic_env: None,
                openai_env: None,
                secret_store: None,
            };
            assert_eq!(c.openai(), None);
            assert_eq!(c.anthropic(), None);
        }

        /// Restores the precedence coverage the trio always had for per-seat overrides:
        /// `MAGI_MODEL_<AGENT>` (env) > `[magi].<agent>_model` (TOML) > backend model. Fix
        /// round 1 (coordinator, 2026-08-03): R-A03 only admits the three declared breakages in
        /// REQ-A21/A22/A23, and `MAGI_MODEL_*` is none of them — removing the capability when
        /// retiring the adapter was an undeclared breakage, so it is restored here.
        ///
        /// Uses Caspar alias validity as the observable signal — same trick as
        /// `cfg_with_only_caspar_unbuildable` — instead of inspecting internal state: an
        /// invalid model makes THAT seat fail to build, so "which model won?" is read from
        /// whether the trio builds or not, without needing real network.
        #[test]
        fn env_model_override_wins_over_toml_which_wins_over_the_backend_model() {
            let backend_only = MagiConfig::from_toml_str(
                "provider = \"anthropic\"\n[anthropic]\nmodel = \"claude-sonnet-4-6\"\n",
            )
            .unwrap();
            let toml_override_invalid = cfg_with_only_caspar_unbuildable();
            let c = creds();
            let endpoints = test_endpoints();

            // Neither TOML nor env: the BACKEND model (valid) is enough.
            let mut notices = Vec::new();
            assert!(
                build_magi_orchestrator(
                    &TrioBuild {
                        cfg: &backend_only,
                        principal_kind: ProviderKind::Anthropic,
                        endpoints: &endpoints,
                        creds: Some(&c),
                        warn_tokens: None,
                        env_overrides: &MagiEnvModelOverrides::default(),
                        capability_cache: None,
                        measured: &BTreeMap::new(),
                    },
                    &mut notices,
                )
                .is_ok(),
                "without overrides, the backend model must be enough"
            );

            // Invalid TOML override, NO env: the TOML is actually applied (and thus fails) — it
            // is not "silently ignored".
            let mut notices = Vec::new();
            assert!(
                build_magi_orchestrator(
                    &TrioBuild {
                        cfg: &toml_override_invalid,
                        principal_kind: ProviderKind::Anthropic,
                        endpoints: &endpoints,
                        creds: Some(&c),
                        warn_tokens: None,
                        env_overrides: &MagiEnvModelOverrides::default(),
                        capability_cache: None,
                        measured: &BTreeMap::new(),
                    },
                    &mut notices,
                )
                .is_err(),
                "the TOML override must be applied, even if invalid"
            );

            // The SAME invalid TOML, but with a VALID env override: env wins.
            let env_overrides = MagiEnvModelOverrides {
                caspar: Some("claude-opus-4-7".to_string()),
                ..MagiEnvModelOverrides::default()
            };
            let mut notices = Vec::new();
            assert!(
                build_magi_orchestrator(
                    &TrioBuild {
                        cfg: &toml_override_invalid,
                        principal_kind: ProviderKind::Anthropic,
                        endpoints: &endpoints,
                        creds: Some(&c),
                        warn_tokens: None,
                        env_overrides: &env_overrides,
                        capability_cache: None,
                        measured: &BTreeMap::new(),
                    },
                    &mut notices,
                )
                .is_ok(),
                "env must win over an invalid TOML override"
            );
        }

        // -------------------------------------------------------------------
        // S8 gate re-review finding (Balthasar): `MagiEnvModelOverrides` reads
        // MAGI_MODEL_* raw, with no blank-is-absent filtering of its own.
        // -------------------------------------------------------------------

        /// Full-chain regression (the reviewer's actual worry): a blank `MAGI_MODEL_MELCHIOR`
        /// must not reach `build_native_provider` as a literal empty model name. Reads the
        /// model `build_magi_orchestrator` actually wired via `seat_wiring_trace()` — the
        /// same trace `env_model_override_wins_over_toml_which_wins_over_the_backend_model`
        /// uses — rather than only asserting `.is_ok()`, which an empty-but-still-buildable
        /// model string could satisfy without proving which model was used.
        ///
        /// **Verified false positive for THIS call site, pinned rather than "fixed":**
        /// `for_seat`'s only production caller (`build_magi_orchestrator`, below) already
        /// passes its result through [`resolve_magi_override`], which independently applies
        /// the same blank-is-absent predicate to its `env_model` parameter — see
        /// `test_resolve_magi_override_empty_string_is_unset` in `config.rs`. This test
        /// constructs `MagiEnvModelOverrides` directly (not through `from_env`, which cannot
        /// be driven without mutating process-global env vars) to exercise exactly that
        /// downstream filtering.
        ///
        /// The refactor below (`from_raw`) additionally filters blanks at the struct's own
        /// boundary, which this test would still pass without: the two levels of defense are
        /// independently verified — this one at the `resolve_magi_override` boundary, the two
        /// below at `from_raw`'s own.
        #[test]
        fn a_blank_magi_model_env_override_falls_through_to_the_toml_or_backend_model() {
            let cfg = MagiConfig::from_toml_str(
                "provider = \"ollama\"\n[magi]\nmelchior_model = \"toml-melchior-model\"\n\
                 melchior_lineage = \"toml-melchior-lineage\"\n",
            )
            .unwrap();
            let env_overrides = MagiEnvModelOverrides {
                melchior: Some(String::new()),
                balthasar: Some("   ".to_string()),
                caspar: None,
            };
            let mut notices = Vec::new();

            let magi = build_magi_orchestrator(
                &TrioBuild {
                    cfg: &cfg,
                    principal_kind: ProviderKind::Ollama,
                    endpoints: &test_endpoints(),
                    creds: None,
                    warn_tokens: None,
                    env_overrides: &env_overrides,
                    capability_cache: None,
                    measured: &BTreeMap::new(),
                },
                &mut notices,
            )
            .expect("ollama is keyless: blank overrides must not break construction");
            drop(magi);

            let trace = seat_wiring_trace();
            let melchior_model = trace
                .iter()
                .find(|(seat, _, _)| *seat == AgentName::Melchior)
                .map(|(_, model, _)| model.as_str());
            assert_eq!(
                melchior_model,
                Some("toml-melchior-model"),
                "a blank MAGI_MODEL_MELCHIOR must fall through to [magi].melchior_model, \
                 never wire an empty model name: {trace:?}"
            );
        }

        /// `MagiEnvModelOverrides::from_raw` — the constructor `from_env` delegates to after
        /// reading the three raw env values — applies the SAME blank-is-absent rule
        /// (REQ-A12) as every other resolver in this file: a blank or whitespace-only value
        /// becomes `None`, never an active empty-string override.
        #[test]
        fn magi_env_model_overrides_from_raw_treats_blank_as_absent() {
            let overrides =
                MagiEnvModelOverrides::from_raw(Some(""), Some("   "), Some("caspar-model"));
            assert_eq!(overrides.melchior, None, "empty string must become absent");
            assert_eq!(
                overrides.balthasar, None,
                "whitespace-only must become absent"
            );
            assert_eq!(
                overrides.caspar,
                Some("caspar-model".to_string()),
                "a real value must survive untouched"
            );
        }

        /// Negative control: `None` (the var was never set) stays `None` — filtering blanks
        /// must not be confused with treating "unset" and "set-to-empty" differently in the
        /// other direction.
        #[test]
        fn magi_env_model_overrides_from_raw_leaves_absent_as_absent() {
            let overrides = MagiEnvModelOverrides::from_raw(None, None, None);
            assert_eq!(overrides.melchior, None);
            assert_eq!(overrides.balthasar, None);
            assert_eq!(overrides.caspar, None);
        }
    }

    /// Task 4.3 — REQ-A06/SC-A06: surface behavior when the trio is unbuildable.
    /// `trio_construction` (above) already covers that `build_magi_orchestrator` reports ALL
    /// fallen seats (SC-A05b/SC-A05c); this module covers what Task 4.3 adds — that this
    /// information REALLY reaches the user on every surface, and not just that the type carries
    /// it.
    mod trio_unavailable_surfaces {
        use super::*;
        use magi_core::test_support::RoutingMockProvider;

        fn seat_unbuildable(seats: Vec<(AgentName, SeatError)>) -> TrioError {
            TrioError::SeatUnbuildable { seats }
        }

        /// The shared formatting primitive names the seat AND the cause — not a count. Reused
        /// by both `Display` and `trio_unavailable_message`.
        #[test]
        fn format_seat_failure_names_the_seat_and_the_cause() {
            let text = format_seat_failure(
                &AgentName::Melchior,
                &SeatError::MissingCredential {
                    var: "OPENAI_API_KEY",
                },
            );
            assert!(text.contains("Melchior"), "{text}");
            assert!(text.contains("OPENAI_API_KEY"), "{text}");
        }

        /// R2 (inherited obligation from Task 4.1): the `Display` of `SeatUnbuildable` does not
        /// stop at a count — a future `{e}`/`.to_string()` that does not go through
        /// `trio_unavailable_message` still names each seat and its cause.
        #[test]
        fn seat_unbuildable_display_names_every_seat_not_just_a_count() {
            let err = seat_unbuildable(vec![
                (
                    AgentName::Melchior,
                    SeatError::MissingCredential {
                        var: "OPENAI_API_KEY",
                    },
                ),
                (
                    AgentName::Caspar,
                    SeatError::Transport(redact_foreign_error(&std::io::Error::other(
                        "connection refused",
                    ))),
                ),
            ]);
            let text = err.to_string();
            assert!(
                text.contains("Melchior") && text.contains("OPENAI_API_KEY"),
                "the first seat must be named with its cause: {text}"
            );
            assert!(
                text.contains("Caspar") && text.contains("connection refused"),
                "the second seat must be named with its cause: {text}"
            );
        }

        /// SC-A05b + SC-A05c together: the single message names EACH fallen seat, its cause,
        /// and reports ALL of them in a single run (not just the first one).
        #[test]
        fn trio_unavailable_message_names_every_failed_seat_and_its_cause() {
            let err = seat_unbuildable(vec![
                (
                    AgentName::Melchior,
                    SeatError::MissingCredential {
                        var: "OPENAI_API_KEY",
                    },
                ),
                (
                    AgentName::Balthasar,
                    SeatError::MissingCredential {
                        var: "OPENAI_API_KEY",
                    },
                ),
                (
                    AgentName::Caspar,
                    SeatError::Transport(redact_foreign_error(&std::io::Error::other(
                        "connection refused",
                    ))),
                ),
            ]);
            let msg = trio_unavailable_message(&err);
            assert!(msg.contains("Melchior"), "{msg}");
            assert!(msg.contains("Balthasar"), "{msg}");
            assert!(msg.contains("Caspar"), "{msg}");
            assert!(
                msg.matches("OPENAI_API_KEY").count() >= 2,
                "both credential causes must appear, not collapse: {msg}"
            );
            assert!(msg.contains("connection refused"), "{msg}");
            assert!(
                msg.contains("vault set"),
                "must say HOW to enable it: {msg}"
            );
        }

        /// `UnknownKind` names the invalid value AND the valid vocabulary.
        #[test]
        fn trio_unavailable_message_unknown_kind_names_the_bad_value_and_the_vocabulary() {
            let msg = trio_unavailable_message(&TrioError::UnknownKind("banana".to_string()));
            assert!(msg.contains("banana"), "{msg}");
            assert!(
                msg.contains("ollama")
                    && msg.contains("openai-compat")
                    && msg.contains("anthropic"),
                "{msg}"
            );
        }

        /// `NoSeats` and `Builder` share the same generic text — neither is reachable by the
        /// real production path today (see their own rustdocs), but
        /// `trio_unavailable_message`'s exhaustive `match` covers them anyway, and both must
        /// produce non-empty, actionable text.
        #[test]
        fn trio_unavailable_message_no_seats_and_builder_share_the_same_generic_text() {
            let no_seats_msg = trio_unavailable_message(&TrioError::NoSeats);
            let cause = redact_foreign_error(&std::io::Error::other("boom"));
            let builder_msg = trio_unavailable_message(&TrioError::Builder(cause));
            assert_eq!(no_seats_msg, builder_msg);
            assert!(!no_seats_msg.is_empty());
        }

        /// SC-A06b, the central invariant: the startup notice and the response a future
        /// `/consult` gives are the SAME string — not two independent wordings that could
        /// diverge.
        #[test]
        fn trio_unavailable_for_tui_notice_and_reply_are_the_same_text_and_blocking_tier() {
            let err = seat_unbuildable(vec![(
                AgentName::Melchior,
                SeatError::MissingCredential {
                    var: "OPENAI_API_KEY",
                },
            )]);
            let (notice, msg) = trio_unavailable_for_tui(&err);
            assert_eq!(notice.text, msg, "notice and reply must be the SAME text");
            assert_eq!(
                notice.tier,
                NoticeTier::Blocking,
                "an unbuildable trio demands action — it is not a Resolution or an Info"
            );
        }

        /// SC-A06a: a consult that cannot run is NOT registered — neither with the trio absent
        /// (it invites the model to route toward something destined to fail) nor,
        /// symmetrically, is it OMITTED when the trio did build (regression: the shared helper
        /// between the TUI and `magi query` must not disable the tool in the happy case).
        #[test]
        fn register_consult_tool_if_available_registers_when_buildable_and_omits_it_otherwise() {
            let magi: Arc<Magi> = Arc::new(Magi::new(Arc::new(RoutingMockProvider::new())));

            let mut agent_with_trio = Agent::new(Arc::new(StaticProvider));
            register_consult_tool_if_available(
                &mut agent_with_trio,
                Some(&magi),
                false,
                ProviderKind::Ollama,
                false,
                magi_rs::magi::MAX_QUERY_BYTES,
                magi_rs::magi::TOOL_RESULT_CAP_BYTES,
            );
            assert!(
                agent_with_trio.has_tool("consult"),
                "a buildable trio must register the tool"
            );

            let mut agent_without_trio = Agent::new(Arc::new(StaticProvider));
            register_consult_tool_if_available(
                &mut agent_without_trio,
                None,
                false,
                ProviderKind::Ollama,
                false,
                magi_rs::magi::MAX_QUERY_BYTES,
                magi_rs::magi::TOOL_RESULT_CAP_BYTES,
            );
            assert!(
                !agent_without_trio.has_tool("consult"),
                "an unbuildable trio must never register a tool the model could \
                 route to and only then discover it cannot run (SC-A06a)"
            );
        }

        /// SC-A06c: a forced `magi consult` with an unbuildable trio fails CLOSED — non-zero
        /// exit code, and writes NO output file (the run returns before a `Magi` exists to call
        /// `analyze` on, so no verdict can be fabricated).
        #[test]
        #[serial_test::serial]
        fn a_forced_consult_fails_closed_when_the_trio_is_unbuildable() {
            with_var("MAGI_PROVIDER", None, || {
                with_var("OPENAI_API_KEY", None, || {
                    let tmp = tempfile::tempdir().unwrap();
                    let cwd = dunce::canonicalize(tmp.path()).unwrap();
                    crate::system::workspace::init(&cwd).expect("init .magi/");
                    // openai-compat requires OPENAI_API_KEY for the trio (never for the
                    // principal agent, which falls back to the dummy "ollama") — without the
                    // variable, all THREE seats fail by missing credential.
                    std::fs::write(
                        cwd.join(".magi/magi.toml"),
                        "provider = \"openai-compat\"\n",
                    )
                    .unwrap();

                    let prompt = cwd.join("q.txt");
                    std::fs::write(&prompt, b"should we do X or Y given these constraints?")
                        .unwrap();
                    let out = cwd.join("out.json");

                    let mut h = base_hargs();
                    h.input = Some(prompt);
                    h.output = Some(out.clone());
                    h.workdir = Some(cwd.clone());
                    h.no_memory = true;

                    let rt = tokio::runtime::Runtime::new().unwrap();
                    let code = rt.block_on(run_consult_subcommand(h, None, None, &cwd, None, None));

                    assert_ne!(
                        code, 0,
                        "an unbuildable trio must fail a forced consult closed, never exit 0"
                    );
                    assert!(
                        !out.exists(),
                        "no output was ever written — the run returns BEFORE any \
                         MAGI object exists, so no verdict-shaped report can ever \
                         be fabricated (SC-A06, no_surface_ever_fabricates_a_verdict)"
                    );
                });
            });
        }
    }

    /// Task 4.4 — REQ-A07p/SC-A07p (endpoint divergence notice) and REQ-A12c/SC-A12f (keyless
    /// 401 translation).
    mod divergence_and_keyless_auth {
        use super::*;
        use magi_rs::magi::endpoint::EndpointError;

        /// Builds a `MagiConfig` with a root `base_url` and, optionally, an own
        /// `[magi].base_url` override — the only pair of fields that `magi_endpoint_diverges()`
        /// looks at.
        fn cfg_with_endpoints(root: &str, magi_override: Option<&str>) -> MagiConfig {
            MagiConfig::builder()
                .base_url(Some(root.to_string()))
                .magi(crate::config::MagiSectionConfig {
                    base_url: magi_override.map(str::to_string),
                    ..crate::config::MagiSectionConfig::default()
                })
                .build()
                .unwrap()
        }

        /// SC-A07p: endpoint divergence is warned, and ONLY when there is divergence AND
        /// inference is active.
        ///
        /// **Divergence from the brief's pseudocode, and why it is tested
        /// here and not only argued in `divergence_notice`'s rustdoc.** The brief recalculated
        /// `will_attempt_classification` internally, ignoring the second parameter. With the
        /// IDENTICAL `cfg` in the last two assertions — differing only in `true`/`false` — an
        /// internal recalculation would have yielded the same result in both, and the third
        /// assertion of this test (`divergence_notice(&cfg, false).is_none()`) would have
        /// failed. This test is the executable evidence of why the implementation uses the
        /// parameter as-is, without recalculating it.
        #[test]
        fn endpoint_divergence_is_announced_only_when_it_actually_diverges_and_inference_is_active()
        {
            let cfg = cfg_with_endpoints("http://a/v1", Some("http://b/v1"));
            let n =
                divergence_notice(&cfg, true).expect("there is divergence with inference active");
            assert!(
                n.text.contains("main provider"),
                "must say where the content passes through first: {}",
                n.text
            );
            assert_eq!(n.tier, NoticeTier::Resolution);

            assert!(
                divergence_notice(&cfg_with_endpoints("http://a/v1", None), true).is_none(),
                "same endpoint (the trio inherits): no divergence to announce"
            );
            assert!(
                divergence_notice(&cfg, false).is_none(),
                "without active inference the content does not pass through the principal"
            );
        }

        /// SC-A07p (wiring, TUI surface): the notice is not only PRODUCED, it is EMITTED — it
        /// reaches the vector that `run()` (the TUI) actually prints.
        ///
        /// **Declared scope, deliberately (fix round 4): this covers ONLY the TUI.**
        /// The brief for this task originally asked for "the vector that the TUI **and
        /// headless** print" (`task-4.4-brief.md:41`); a review found that this test (and its
        /// assertion message) had silently trimmed that coverage to only the TUI without any
        /// round report saying so — a requirement shrunk in silence inside the text of a test,
        /// which is exactly how a gap becomes invisible. The headless NO LONGER shares this
        /// test: it has its own,
        /// `test_prepare_headless_carries_the_divergence_notice_when_it_applies`, below,
        /// because `prepare_headless` cannot be tested by calling `push_divergence_notice`
        /// directly (that function is not its only caller; `prepare_headless` has its OWN call
        /// site — see that test for the real proof that THAT call site exists).
        ///
        /// Separate from the previous test on purpose, as this task's brief asks: that one
        /// verifies the PREDICATE; this one verifies the PUSH to the vector. `run()` (real
        /// owner of `startup_notices`) is not unit-testable — it opens the vault, discovers the
        /// workspace, and uses real TTY — so this test calls `push_divergence_notice` directly:
        /// it is the SAME function, and the ONLY one, that `run()` invokes for this (a one-line
        /// call, trivial to audit against the diff). A correct function that nobody calls would
        /// pass the previous test and leave the user without the notice — this is the exact
        /// failure mode of a "defined but not wired" that already happened once in this plan
        /// (Task 4.3).
        #[test]
        fn the_divergence_notice_reaches_the_tui_startup_notices() {
            let cfg = cfg_with_endpoints("http://a/v1", Some("http://b/v1"));
            let mut notices: Vec<Notice> = Vec::new();
            push_divergence_notice(&cfg, true, &mut notices);
            assert!(
                notices.iter().any(|n| n.text.contains("main provider")),
                "the notice must be in the vector the TUI prints (TUI surface only — see \
                 test_prepare_headless_carries_the_divergence_notice_when_it_applies for \
                 the headless surface): {notices:?}"
            );
        }

        /// SC-A07p (wiring, HEADLESS surface) — fix round 4, finding 1.
        ///
        /// **This is what was missing, with no test covering it.**
        /// `push_divergence_notice` only had ONE production call site, inside `run()` (the
        /// TUI); `prepare_headless` —the shared prelude of `magi query` and `magi consult`—
        /// never invoked it. REQ-A07c is explicitly about the headless path: a pipeline with
        /// `magi consult` without `--mode` is SC-A07f, and that pipeline has no TUI in which
        /// the notice could appear. Wiring only the interactive surface —where there is a human
        /// watching— and ignoring the automated one inverted the priority the spec itself sets.
        ///
        /// **Why this test drives `MagiConfig` by hand, not `push_divergence_
        /// notice` directly.** `prepare_headless` is a real function, with real I/O (discovered
        /// `.magi/`, `magi.toml` read from disk), so unlike the TUI test above —which cannot
        /// avoid calling `push_divergence_notice` DIRECTLY because `run()` is not testable at
        /// all— here the real function can be driven end to end:
        /// `init_default_workspace`/`write_envelope`/`base_hargs` (already used by
        /// `test_prepare_headless_cli_provider_override_normalizes_the_new_vocabulary` above)
        /// are exactly the harness that makes this possible with `--no-memory`, without a real
        /// vault.
        ///
        /// `HeadlessContext::divergence_notice` exists ONLY so this test can assert against the
        /// result without capturing stderr from the process (a global resource, unsafe for a
        /// parallel test suite) — same rationale that already justifies the `provider_kind`
        /// field on the same struct.
        #[test]
        fn test_prepare_headless_carries_the_divergence_notice_when_it_applies() {
            with_var("MAGI_PROVIDER", None, || {
                with_var("ANTHROPIC_MODEL", None, || {
                    with_var("OPENAI_MODEL", None, || {
                        let tmp = tempfile::tempdir().unwrap();
                        let cwd = dunce::canonicalize(tmp.path()).unwrap();
                        crate::system::workspace::init(&cwd).expect("init .magi/");
                        // Diverges (different root vs. [magi].base_url) and does NOT declare
                        // `default_mode` ⇒ inference active: the two conditions of
                        // `divergence_notice` (SC-A07p).
                        std::fs::write(
                            cwd.join(".magi/magi.toml"),
                            "base_url = \"http://a:11434/v1\"\n\
                             [magi]\n\
                             base_url = \"http://b:11434/v1\"\n",
                        )
                        .unwrap();
                        let input = write_envelope(&cwd, "env.json", r#"{"prompt":"hi"}"#);

                        let mut h = base_hargs();
                        h.input = Some(input);
                        h.workdir = Some(cwd.clone());
                        h.no_memory = true;

                        let rt = tokio::runtime::Runtime::new().unwrap();
                        let ctx = rt
                            .block_on(prepare_headless(&h, None, &cwd, None, None))
                            .expect("prepare_headless must succeed");

                        let notice = ctx.divergence_notice.expect(
                            "diverging config + active inference ⇒ Some — the headless \
                             prelude must call divergence_notice, same as run() does \
                             for the TUI (REQ-A07c/SC-A07f: headless is the surface \
                             this notice matters most for)",
                        );
                        assert!(notice.text.contains("main provider"), "{}", notice.text);
                    });
                });
            });
        }

        /// Precondition of `divergence_notice`: `load()` already validated both templates
        /// before returning a `MagiConfig`, so in production
        /// `effective_magi_base_url()`/`effective_base_url()` never fail here. Same pattern as
        /// `MagiConfig::effective_provider`/`effective_default_mode` (`config.rs`):
        /// constructing `MagiConfig` by hand, bypassing `load()`, is the only thing that can
        /// violate this precondition, and the `debug_assert!` turns it into a loud panic
        /// instead of a silent `Ollama`/`None`.
        #[test]
        #[should_panic(expected = "validated")]
        fn divergence_notice_panics_in_debug_builds_when_the_endpoint_template_is_invalid() {
            // Literal credential: `EndpointTemplate::parse` rejects it (REQ-A16c), so
            // `effective_magi_base_url()` fails — the precondition that `load()` normally
            // guarantees, deliberately violated.
            let cfg = MagiConfig::builder()
                .magi(crate::config::MagiSectionConfig {
                    base_url: Some("https://user:pass@host/v1".to_string()),
                    ..crate::config::MagiSectionConfig::default()
                })
                .build()
                .unwrap();
            let _ = divergence_notice(&cfg, true);
        }

        /// Sixth-pass gate finding (S8, Balthasar): `divergence_notice`'s error branch
        /// formatted `EndpointError::to_string()` verbatim, trusting a comment that no
        /// current variant embeds the received value. `EndpointError`'s own fields are all
        /// `&'static str` (see `src/magi/endpoint.rs`), so it cannot be mutated into leaking a
        /// credential today — which is exactly why `endpoint_display_text` is generic over the
        /// error type: this fabricated error stands in for a hypothetical future variant that
        /// DOES interpolate raw text, and proves the redaction wrap actually fires rather than
        /// relying on the same kind of by-inspection promise the finding objected to.
        struct FakeCredentialLeak;

        impl std::fmt::Display for FakeCredentialLeak {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(
                    f,
                    "connection failed: https://secretuser:secretpass@host.example:1234/v1"
                )
            }
        }

        #[test]
        fn endpoint_display_text_redacts_a_credential_a_future_error_variant_might_embed() {
            let result: Result<EndpointTemplate, FakeCredentialLeak> = Err(FakeCredentialLeak);

            let out = endpoint_display_text(&result);

            assert!(!out.contains("secretuser"), "credential leaked: {out}");
            assert!(!out.contains("secretpass"), "credential leaked: {out}");
            assert!(
                out.contains("host.example"),
                "the useful part (host) must survive redaction: {out}"
            );
        }

        /// Counterpart of the test above: the SUCCESS branch must stay untouched by
        /// redaction. `EndpointTemplate::as_str()` can only ever carry the `[user]:[password]`
        /// placeholders, never a real secret (REQ-A16c) — running it through
        /// `redact_foreign_text` too would blank that informative placeholder text by
        /// position, the exact regression this test guards against.
        #[test]
        fn endpoint_display_text_leaves_a_valid_template_untouched() {
            let template =
                EndpointTemplate::parse("https://[user]:[password]@host:1234/v1", Scope::Root)
                    .expect("placeholders are a valid template");
            let result: Result<EndpointTemplate, EndpointError> = Ok(template);

            let out = endpoint_display_text(&result);

            assert_eq!(out, "https://[user]:[password]@host:1234/v1");
        }

        /// R6 (Task 1.2b, `planning/claude-plan-tdd.md` ~L3160): closing the credential paths
        /// that plan marked as being born with the native trio in Phase 4.
        ///
        /// **It is not the same test the plan describes, and it cannot be.** The plan
        /// imagined a single canary in `src/magi/endpoint.rs` (lib) covering the five paths
        /// from a single place. But `MagiConfig` and `SeatError` are BIN-crate types (`mod
        /// config;`/`main.rs`) — neither appears in the `pub mod` list of `src/lib.rs`
        /// (`headless, magi, notices, redact, vault`), so a lib-crate test cannot name them,
        /// and `divergence_notice` could not live in `src/magi/mode.rs` for the same reason
        /// (see this task's report).
        ///
        /// **Path 4 moved**, and with it its coverage: round 2 of this task
        /// retired `explain_keyless_auth_failure(&SeatError, ProviderKind)` — it never had a
        /// real production caller — and replaced it with
        /// `tools::consult::keyless_auth_explanation(&str, ProviderKind)`, which operates on
        /// the ALREADY-RENDERED cause from `MagiReport::failed_agents` and lives in
        /// `src/tools/consult.rs`, where its own no-leak coverage
        /// (`keyless_auth_explanation_never_echoes_the_raw_cause`) also lives. The paths that
        /// DO remain in this file are tested here: 1 (`divergence_notice`) and 3
        /// (`trio_unavailable_message`). Path 2 (`openai_compat_root`) and 5 (the Anthropic-
        /// incoherence notice in `resolution_notices()`) already have their own coverage from
        /// Task 1.2b/4.1.
        #[test]
        fn no_notice_or_error_path_in_this_file_leaks_a_credential() {
            const CANARY: &str = "c4n4ry-s3cr3t";

            // Path 1 — `divergence_notice`: operates on the TEMPLATE
            // (`EndpointTemplate::as_str()`), which by construction (REQ-A16c) cannot contain a
            // secret — a literal there is rejected at parse time, never accepted and displayed.
            // A placeholder is used, not a literal canary, precisely because a literal canary
            // could not exist in this field (proven by the test above).
            let cfg = cfg_with_endpoints("http://a/v1", Some("https://[user]:[password]@b/v1"));
            let notice = divergence_notice(&cfg, true).expect("diverges with inference active");
            // Presence BEFORE absence (S3 Loop 2, Balthasar). `!x.contains(CANARY)` holds
            // trivially for an empty string, so a notice whose text stopped being produced —
            // or a redaction that collapsed the whole message — would leave this test green
            // while the surface it guards had vanished. Pin that the message is still the one
            // being scanned, then that the secret is not in it.
            assert!(
                notice.text.contains("the trio runs on") && notice.text.contains("[user]"),
                "precondition: the scanned notice must still be the divergence message, with \
                 the endpoint rendered in it: {}",
                notice.text
            );
            assert!(!notice.text.contains(CANARY));

            // Path 3 — `trio_unavailable_message`: the foreign cause goes through
            // `redact_foreign_error` BEFORE becoming `SeatError::Transport` (see
            // `build_native_provider::to_seat`); here the SAME composition is exercised,
            // directly on the type.
            let foreign =
                std::io::Error::other(format!("connect to https://alice:{CANARY}@host/v1"));
            let err = TrioError::SeatUnbuildable {
                seats: vec![(
                    AgentName::Melchior,
                    SeatError::Transport(redact_foreign_error(&foreign)),
                )],
            };
            let message = trio_unavailable_message(&err);
            assert!(
                message.contains("melchior") || message.contains("Melchior"),
                "precondition: the message must still name the seat that failed, or there is \
                 nothing here for the canary check to be about: {message}"
            );
            assert!(
                message.contains("host/v1"),
                "and it must still carry the redacted endpoint — a redaction that ate the whole \
                 cause would pass the canary check while destroying the diagnostic: {message}"
            );
            assert!(!message.contains(CANARY));
        }

        /// The canary value substituted into a template's `[password]`. Distinctive enough
        /// that a substring search cannot match it by accident.
        pub(super) const SEAT_CANARY: &str = "c4n4ry-s3cr3t";

        /// A [`SecretStore`] over a fixed map, so a credentialed [`ResolvedEndpoint`] can be
        /// built without standing up a real vault.
        ///
        /// A real one would work, and would also pay Argon2id at `p = 4` — four lanes on a
        /// six-core box — for a test that has nothing to do with key derivation. That is the
        /// cost `.config/nextest.toml` caps the `heavy` group to bound; not incurring it is
        /// better than being capped for it.
        struct FixedVault(std::collections::BTreeMap<&'static str, &'static str>);

        impl SecretStore for FixedVault {
            fn set(&mut self, _name: &str, _value: &str) -> Result<(), VaultError> {
                Ok(())
            }
            fn get(&mut self, name: &str) -> Result<Zeroizing<String>, VaultError> {
                self.0
                    .get(name)
                    .map(|v| Zeroizing::new((*v).to_string()))
                    .ok_or_else(|| VaultError::SecretNotFound(name.to_string()))
            }
            fn remove(&mut self, _name: &str) -> Result<(), VaultError> {
                Ok(())
            }
            fn list(&mut self) -> Result<Vec<SecretEntry>, VaultError> {
                Ok(Vec::new())
            }
            fn contains(&mut self, name: &str) -> Result<bool, VaultError> {
                Ok(self.0.contains_key(name))
            }
        }

        /// Resolves `template` against a vault holding [`SEAT_CANARY`] as the root password —
        /// the only way to obtain a credential-bearing endpoint, since REQ-A16c rejects a
        /// literal one at parse time.
        pub(super) fn credentialed_endpoint(template: &str) -> ResolvedEndpoint {
            let mut vault = FixedVault(
                [
                    ("BASE_URL_USER", "alice"),
                    ("BASE_URL_PASSWORD", SEAT_CANARY),
                ]
                .into_iter()
                .collect(),
            );
            EndpointTemplate::parse(template, Scope::Root)
                .expect("a placeholder template parses")
                .resolve(&mut vault, Scope::Root)
                .expect("the vault holds both entries")
        }

        /// Loop 1, F20 — the no-leak guarantee is asserted against the REAL composition in
        /// [`build_native_provider`], not a hand-rolled stand-in.
        ///
        /// Every previous no-leak assertion in this file built `SeatError::Transport` by
        /// calling `redact_foreign_error` itself, so the one production site that maps a
        /// `ProviderError` into it — the `to_seat` closure — had zero test call sites, and so
        /// did `build_native_provider`. This drives both.
        ///
        /// **The `Ok` half is the one that can actually fail today, and that is deliberate.**
        /// `openai_compat_root` interpolates the RESOLVED endpoint into its normalization
        /// notice, which reaches the TUI startup list and headless stderr; drop its
        /// `redact_url` and this assertion goes red immediately.
        ///
        /// **The survival check targets the interpolated value, not the template.** An earlier
        /// version of this test asserted only `.contains("/v1")` as proof the notice fired —
        /// but that literal is *also* static text in `openai_compat_root`'s own message
        /// ("without a `/v1` suffix"), so it is satisfied no matter what gets interpolated in
        /// its place. Mutation-verified: swapping `redact_url(&normalized)` for the opaque
        /// literal `"***"` at that call site left the old assertion green. The fixture below
        /// uses a host:port the template cannot produce by coincidence, so only the REAL
        /// redacted URL — `"***@" + host + ":" + port + "/v1"` — can satisfy the check.
        #[test]
        fn build_native_provider_never_leaks_a_resolved_credential() {
            // Distinctive enough that neither the message template nor an opaque `"***"`
            // literal could match it by chance — only interpolating the actual resolved,
            // redacted URL produces this host:port pair.
            const CANARY_HOST: &str = "zzq-mutation-canary.example";
            const CANARY_PORT: &str = "48213";

            // No `/v1` suffix ⇒ `openai_compat_root` normalizes AND emits its notice, which is
            // the path that embeds the URL in user-visible text.
            let endpoint = credentialed_endpoint(&format!(
                "http://[user]:[password]@{CANARY_HOST}:{CANARY_PORT}"
            ));
            let mut notices: Vec<Notice> = Vec::new();
            let built = build_native_provider(
                ProviderKind::Ollama,
                &endpoint,
                "some-model",
                None,
                Duration::from_secs(1),
                &mut notices,
            );
            assert!(built.is_ok(), "keyless ollama over http builds fine");

            // Only `redact_url`'s designed output over the resolved URL contains this exact
            // substring: the userinfo replaced in place, host and port left visible. A
            // template-only match (or a bare `"***"` standing in for the whole credential)
            // cannot carry the host/port, which is what makes this load-bearing rather than
            // cosmetic — see the mutation note above.
            let expected_redacted = format!("***@{CANARY_HOST}:{CANARY_PORT}/v1");
            assert!(
                notices.iter().any(|n| n.text.contains(&expected_redacted)),
                "test setup: the normalization notice must carry the resolved, redacted host, \
                 or this asserts nothing; notices: {notices:?}"
            );

            for n in &notices {
                assert!(
                    !n.text.contains(SEAT_CANARY),
                    "a construction notice leaked the resolved credential: {}",
                    n.text
                );
                assert!(
                    !n.text.contains("alice"),
                    "userinfo is redacted by position, so the user part goes too: {}",
                    n.text
                );
            }
        }

        /// The `Err` half: a real `ProviderError` from magi-core, mapped through the production
        /// `to_seat` closure and rendered by `trio_unavailable_message`.
        ///
        /// **Honest note on what this proves.** magi-core 3.1.0's `ProviderUrl::parse`
        /// deliberately never echoes its input (its own module doc says so, and the scheme
        /// branch interpolates only `parsed.scheme()`), so this message would be canary-free
        /// even if `to_seat` dropped its redaction today. What the test buys is the future:
        /// `ProviderError` is `#[non_exhaustive]`, a later version can add a variant that
        /// interpolates free text, and from that day this assertion is the thing standing
        /// between it and the terminal. It also closes the "zero test call sites" gap that let
        /// the composition go unexercised at all.
        #[test]
        fn a_real_seat_construction_failure_is_rendered_without_the_credential() {
            // A non-http(s) scheme is rejected by magi-core's own URL validation, so this is a
            // genuine `ProviderError` from the real constructor — no network, no mock.
            let endpoint = credentialed_endpoint("ftp://[user]:[password]@host/v1");
            let mut notices: Vec<Notice> = Vec::new();
            // `Arc<dyn LlmProvider>` has no `Debug`, so the `Ok` side cannot be unwrapped with
            // `expect_err`; matching keeps the assertion on the variant instead.
            let Err(err) = build_native_provider(
                ProviderKind::Ollama,
                &endpoint,
                "some-model",
                None,
                Duration::from_secs(1),
                &mut notices,
            ) else {
                panic!("a non-http scheme cannot build a provider");
            };
            assert!(
                matches!(err, SeatError::Transport(_)),
                "the failure must arrive through `to_seat`, the site under test: {err:?}"
            );

            let rendered = trio_unavailable_message(&TrioError::SeatUnbuildable {
                seats: vec![(AgentName::Melchior, err)],
            });
            assert!(
                !rendered.contains(SEAT_CANARY) && !rendered.contains("alice"),
                "the seat failure reached the user with a credential in it: {rendered}"
            );
        }
    }

    /// Loop 1, F7 / SC-A04c-d — REQ-A04's coherence check on `magi query`'s consult route.
    ///
    /// The formula, its warning and its JSON telemetry existed only for the direct
    /// `magi consult` path; `magi query`'s forced/proactive consult shares the run's single
    /// deadline and had none of them, so the same misconfiguration surfaced as an opaque
    /// `error.kind = timeout`.
    mod query_timeout_coherence {
        use super::*;

        /// The effective ceiling used across these cases; distinguishable from the built-in so
        /// a check that ignored it would not accidentally agree.
        const CEILING: u64 = 60;

        /// A config declaring [`CEILING`], so the check reads its scale where production reads it.
        fn cfg_with_ceiling() -> MagiConfig {
            MagiConfig::from_toml_str(&format!(
                "[magi]
agent_timeout_secs = {CEILING}
"
            ))
            .expect("a ceiling inside the admissible range must parse")
        }

        /// The formula minimum for that config, resolved THROUGH `timeout_scale` — the same
        /// resolution production uses.
        ///
        /// Deriving it rather than writing a literal is deliberate here and would be wrong in
        /// `magi::tests`: there the arithmetic itself is under test and is pinned against literal
        /// values; here what is under test is the WIRING — that the decision consults the config
        /// at all — so an expectation computed any other way would just be a second, drifting
        /// copy of the formula.
        fn formula_minimum() -> u64 {
            let (ceiling, max_rotations, retry_disabled) = timeout_scale(&cfg_with_ceiling());
            magi_rs::magi::headless_consult_timeout_secs(ceiling, max_rotations, retry_disabled)
        }

        /// A deadline below `classification + 2 × ceiling + slack` is reported, and the warning
        /// names the computed minimum so the operator can act on it.
        #[test]
        fn a_deadline_below_the_formula_is_reported_with_its_minimum() {
            let minimum = formula_minimum();
            let decision = query_timeout_decision(
                Some(Duration::from_secs(minimum - 1)),
                true,
                &cfg_with_ceiling(),
            )
            .expect("a bounded, consult-capable run is exactly the checked case");

            assert!(decision.below_formula);
            let warning = decision.warning.expect("below the formula ⇒ a warning");
            assert!(
                warning.contains(&minimum.to_string()),
                "the warning must name the computed minimum, or it is not actionable: {warning}"
            );
        }

        /// REQ-R20: the run's minimum FOLLOWS `[magi].max_rotations`, so a deadline that was
        /// generous before rotation existed can become too short once a pool is declared.
        ///
        /// Without this, `query_timeout_decision` could read the ceiling and ignore the rotation
        /// ceiling entirely and every other test here would still pass — they all compare against
        /// a minimum derived the same way. This one compares two DIFFERENT configs against each
        /// other, which is the only shape that can catch it.
        #[test]
        fn the_minimum_grows_with_the_declared_rotation_ceiling() {
            let no_rotation = MagiConfig::from_toml_str(&format!(
                "[magi]\nagent_timeout_secs = {CEILING}\nmax_rotations = 0\n"
            ))
            .expect("valid");
            let two_rotations = MagiConfig::from_toml_str(&format!(
                "[magi]\nagent_timeout_secs = {CEILING}\nmax_rotations = 2\n"
            ))
            .expect("valid");

            let minimum_of = |cfg: &MagiConfig| {
                // A deadline of 1 s is below any minimum, so the decision always carries one.
                query_timeout_decision(Some(Duration::from_secs(1)), true, cfg)
                    .expect("bounded and consult-capable")
                    .warning
                    .expect("below the formula ⇒ a warning naming the minimum")
            };

            assert_ne!(
                minimum_of(&no_rotation),
                minimum_of(&two_rotations),
                "declaring a rotation ceiling MUST move the minimum; if these agree, the \
                 decision is ignoring max_rotations and a healthy consult that rotates will be \
                 cut off — a symptom that only appears WHEN a rotation also happened"
            );
        }

        /// The operator's value is never overridden: a wall-clock cap is an instruction, not a
        /// safety invariant.
        #[test]
        fn the_requested_deadline_is_obeyed_even_when_it_is_too_short() {
            let minimum = formula_minimum();
            let asked = minimum - 1;
            let decision =
                query_timeout_decision(Some(Duration::from_secs(asked)), true, &cfg_with_ceiling())
                    .unwrap();
            assert_eq!(
                decision.effective_secs, asked,
                "obeying the request is the point; the check only adds the heads-up"
            );
        }

        /// A generous deadline warns about nothing.
        #[test]
        fn a_deadline_above_the_formula_is_silent() {
            let minimum = formula_minimum();
            let decision = query_timeout_decision(
                Some(Duration::from_secs(minimum + 1)),
                true,
                &cfg_with_ceiling(),
            )
            .unwrap();
            assert!(!decision.below_formula);
            assert!(decision.warning.is_none());
        }

        /// Edge cases where the check does not apply at all: an unbounded run cannot be too
        /// short, and a run with no trio has no consult to be too short for.
        #[test]
        fn the_check_does_not_apply_without_a_deadline_or_without_a_trio() {
            assert!(
                query_timeout_decision(None, true, &cfg_with_ceiling()).is_none(),
                "an unbounded run cannot be below any minimum"
            );
            assert!(
                query_timeout_decision(Some(Duration::from_secs(1)), false, &cfg_with_ceiling())
                    .is_none(),
                "with no trio built there is no consult to warn about — warning would be noise"
            );
        }
    }

    /// Loop 1, F3 — the mode classifier's notices must not reach stderr while the TUI owns the
    /// screen.
    ///
    /// `ProcessNoticeSink::once` calls `eprintln!` unconditionally, and `run_tui_ext` enters
    /// raw mode plus the alternate screen before the event loop starts, so a `/consult` with
    /// no `--mode` under the default scaffold — the very path inference exists to serve — wrote
    /// raw text over the ratatui frame. The fix is a sink, not a hardcoded destination: the
    /// same classifier runs headless, where stderr IS correct.
    mod tui_classifier_notices {
        use super::*;
        use crate::agent::mode_classifier::NoticeSink;
        use crate::tui::AgentResponse;

        /// The classifier the TUI is handed writes its notices to the sink the TUI attaches its
        /// channel to. Driven through a real `classify()` call, so a wiring that hands back two
        /// unrelated instances fails here instead of silently writing over the frame.
        #[tokio::test]
        async fn the_tui_classifier_emits_its_notices_through_the_attached_channel() {
            let (classifier, notices) = tui_mode_classifier_wiring(Arc::new(StaticProvider));
            let (tx, mut rx) = tokio::sync::mpsc::channel::<AgentResponse>(8);
            notices.attach(tx);

            // `StaticProvider`'s canned text is not one of the three labels, so this
            // classification fails — which is fine: the COST notice fires before the call,
            // unconditionally, and that is the one that used to hit the frame first.
            let _ = classifier
                .classify("some content with no declared mode")
                .await;

            let mut seen = Vec::new();
            while let Ok(resp) = rx.try_recv() {
                if let AgentResponse::Notice(text) = resp {
                    seen.push(text);
                }
            }
            assert!(
                seen.iter().any(|t| t.contains("--mode")),
                "the classifier's cost notice must arrive as a TUI Notice, not on stderr: \
                 {seen:?}"
            );
        }

        /// Before the channel exists the sink falls back to stderr — correct, because raw mode
        /// has not been entered yet — and it never drops a notice silently (B9).
        #[test]
        fn an_unattached_sink_still_deduplicates_by_key() {
            let sink = crate::tui::TuiNoticeSink::new();
            sink.once("k", "first");
            sink.once("k", "second");

            let (tx, mut rx) = tokio::sync::mpsc::channel::<AgentResponse>(8);
            sink.attach(tx);
            sink.once("k", "third");
            assert!(
                rx.try_recv().is_err(),
                "a key already emitted before attachment must stay emitted: \"once\" is per \
                 key and per process, not per destination"
            );

            sink.once("other", "fresh key");
            assert!(
                matches!(rx.try_recv(), Ok(AgentResponse::Notice(t)) if t == "fresh key"),
                "a key not yet seen must reach the channel once attached"
            );
        }

        /// MAGI S7 fix round, finding 1: a FULL channel (the TUI is alive and just
        /// momentarily saturated) must never fall back to `eprintln!` the way a
        /// closed/unattached one does — that was the actual bug, since a full channel is
        /// precisely the case where the frame is at risk. This fills the channel to
        /// capacity first, then calls `once` and proves the message still arrives via the
        /// channel (once room frees up) instead of being lost or printed.
        #[tokio::test]
        async fn a_full_channel_waits_for_room_instead_of_falling_back_to_stderr() {
            let (tx, mut rx) = tokio::sync::mpsc::channel::<AgentResponse>(1);
            let sink = crate::tui::TuiNoticeSink::new();
            // Clone before `attach` moves `tx` in, so this test keeps a handle able to
            // saturate the channel's single slot ahead of the notice under test.
            sink.attach(tx.clone());
            tx.try_send(AgentResponse::Notice("filler".to_string()))
                .expect("the fresh channel has room for the filler");

            // `once` now hits `TrySendError::Full` and must queue rather than print.
            sink.once("full-channel-key", "queued while full");

            // Poll for the condition instead of sleeping a fixed duration
            // (CLAUDE.local.md: "wait on conditions, never on durations"): drain the
            // filler first (frees the slot the spawned background send is waiting on),
            // then look for the queued notice.
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            let mut delivered = None;
            while std::time::Instant::now() < deadline {
                if let Ok(AgentResponse::Notice(text)) = rx.try_recv() {
                    if text == "filler" {
                        continue;
                    }
                    delivered = Some(text);
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
            assert_eq!(
                delivered.as_deref(),
                Some("queued while full"),
                "a notice queued while the channel was full must still be delivered \
                 through the channel, not dropped or printed to stderr"
            );
        }

        /// MS2 gate S7-f finding (Caspar): a `Closed` channel used to fall back to
        /// `eprintln!` unconditionally, on the theory that "receiver gone" always means
        /// "teardown, stderr is safe" — but `response_rx` drops as soon as `run_app`
        /// returns, which is BEFORE `run_tui_ext` calls `LeaveAlternateScreen`. A notice
        /// racing that window must defer instead of printing immediately: `flush()` (not
        /// `once()` itself) is what surfaces it, simulating `run_tui_ext`'s post-teardown
        /// call.
        #[test]
        fn a_closed_channel_defers_instead_of_printing_before_flush() {
            let (tx, rx) = tokio::sync::mpsc::channel::<AgentResponse>(8);
            let sink = crate::tui::TuiNoticeSink::new();
            sink.attach(tx);
            // Simulates `run_app` having returned and dropped `response_rx` — teardown has
            // STARTED but `LeaveAlternateScreen` has not run yet.
            drop(rx);

            sink.once("closed-key", "deferred message");
            assert_eq!(
                sink.pending_len(),
                1,
                "a notice hitting a closed channel before flush() must be buffered, not \
                 printed inline — printing here would race LeaveAlternateScreen"
            );

            let deferred = sink.flush();
            assert_eq!(
                deferred,
                vec!["deferred message".to_string()],
                "flush() must hand back exactly what was deferred, so run_tui_ext can print \
                 it only after the alternate screen is confirmed gone"
            );
        }

        /// Companion to the test above: once `flush()` has run — meaning `run_tui_ext` has
        /// already left the alternate screen — a LATER notice on the same (still closed)
        /// channel is safe to print immediately and must not be re-buffered, or it would
        /// never reach the user.
        #[test]
        fn a_notice_after_flush_is_not_rebuffered() {
            let (tx, rx) = tokio::sync::mpsc::channel::<AgentResponse>(8);
            let sink = crate::tui::TuiNoticeSink::new();
            sink.attach(tx);
            drop(rx);

            sink.once("k1", "first");
            assert_eq!(sink.flush(), vec!["first".to_string()]);

            sink.once("k2", "second");
            assert_eq!(
                sink.pending_len(),
                0,
                "once flushed, a later fallback must go straight to stderr instead of \
                 accumulating in the buffer, since the terminal is already known to be \
                 restored by then"
            );
            assert!(
                sink.flush().is_empty(),
                "a second flush after the sink is already Flushed must be a no-op, not \
                 re-deliver 'second' out of band"
            );
        }

        /// MS2 gate S7-f finding (Caspar), the `Full` branch's half: a background task
        /// queued while the channel was momentarily full can find it CLOSED by the time
        /// room would have freed up (the run ended mid-wait). That fallback must land in
        /// the same deferred buffer as the direct `Closed` case — not print for itself from
        /// inside the spawned task, which is exactly the race the fix closes.
        #[tokio::test]
        async fn a_full_channel_that_closes_before_room_frees_up_defers_too() {
            let (tx, rx) = tokio::sync::mpsc::channel::<AgentResponse>(1);
            let sink = crate::tui::TuiNoticeSink::new();
            sink.attach(tx.clone());
            tx.try_send(AgentResponse::Notice("filler".to_string()))
                .expect("the fresh channel has room for the filler");

            // Hits `TrySendError::Full` and spawns a background task waiting for room.
            sink.once("full-then-closed", "should defer, not print");

            // Drop the receiver while that background task is still waiting — simulates
            // `run_app` returning (and dropping `response_rx`) before room ever freed.
            drop(rx);

            // Poll a non-destructive peek (never `flush()`, which would itself flip the
            // sink to `Flushed` and hide the very race this proves closed) until the
            // spawned task has resolved and deferred its message.
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while std::time::Instant::now() < deadline && sink.pending_len() == 0 {
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
            assert_eq!(
                sink.flush(),
                vec!["should defer, not print".to_string()],
                "a full-then-closed notice must surface only through flush(), never by \
                 printing itself from the spawned task"
            );
        }
    }

    /// Loop 1, F1 — the operator's autonomous-consult configuration reaching `AgentRunConfig`.
    ///
    /// `[magi.complexity]`, `[magi].default_mode` and `[magi].untrusted_content` were parsed,
    /// validated and then consumed by nobody: no production caller ever set
    /// `AgentRunConfig`'s `gate_thresholds`/`mode_config`/`gate_telemetry`, so the two
    /// highest-traffic autonomous surfaces (the TUI chat loop and `magi query`'s tool loop) ran
    /// on built-ins with the `untrusted_content` guard permanently off. These tests pin the
    /// resolution step; the two surfaces' own tests pin that they consume it.
    mod autonomous_run_config {
        use super::*;
        use magi_rs::magi::gate::GateThresholds;

        /// REQ-A20b/REQ-A07/REQ-A07d: every operator key reaches the run configuration —
        /// including the per-mode `0` off-switch, which is the one value a "fall back to the
        /// built-in" bug would silently swallow.
        #[test]
        fn operator_thresholds_and_mode_config_reach_the_agent_run_config() {
            let cfg = MagiConfig::from_toml_str(
                "[magi]\n\
                 default_mode = \"code-review\"\n\
                 untrusted_content = true\n\
                 [magi.complexity]\n\
                 code_review = 7\n\
                 analysis = 0\n",
            )
            .expect("fixture parses");

            let run = AutonomousRunConfig::from_magi_config(&cfg).apply(Default::default());

            assert_eq!(
                run.gate_thresholds.code_review, 7,
                "a declared threshold must reach the gate, not the built-in"
            );
            assert_eq!(
                run.gate_thresholds.design,
                GateThresholds::builtin().design,
                "a key absent INSIDE a present table keeps its built-in, never zero"
            );
            assert_eq!(
                run.gate_thresholds.analysis, 0,
                "0 is the documented per-mode off-switch and must survive the trip"
            );
            assert_eq!(
                run.mode_config.default_mode,
                Some(Mode::CodeReview),
                "[magi].default_mode is level 2 of the five-level precedence"
            );
            assert!(
                run.mode_config.untrusted_content,
                "the untrusted-content guard must reach the funnel; off here means SC-A07r \
                 cannot fail closed on any autonomous surface"
            );
        }

        /// An empty `magi.toml` still leaves the gate ACTIVE on the built-ins (REQ-A20b): the
        /// absence of configuration is not an off-switch.
        #[test]
        fn an_unconfigured_file_keeps_the_builtin_gate_and_no_guard() {
            let run = AutonomousRunConfig::from_magi_config(&MagiConfig::default())
                .apply(Default::default());
            assert_eq!(run.gate_thresholds, GateThresholds::builtin());
            assert_eq!(run.mode_config.default_mode, None);
            assert!(!run.mode_config.untrusted_content);
        }

        /// SC-A20h: the sink installed by `apply` is the one `drain_telemetry` reads, so an
        /// evaluation recorded through `AgentRunConfig` is observable afterwards.
        #[test]
        fn the_installed_sink_is_the_one_that_can_be_drained() {
            let autonomous = AutonomousRunConfig::from_magi_config(&MagiConfig::default());
            let run = autonomous.apply(Default::default());

            run.gate_telemetry
                .on_gate_evaluation(&Mode::Analysis, 3, 200, true);
            run.gate_telemetry
                .on_gate_evaluation(&Mode::Design, 900, 500, false);

            let lines = autonomous.drain_telemetry();
            assert_eq!(lines.len(), 2, "both sides of the gate are recorded");
            assert!(
                lines[0].contains("analysis") && lines[0].contains("200"),
                "the APPLIED threshold travels with the line: {:?}",
                lines[0]
            );
            assert!(
                lines[1].contains("design") && lines[1].contains("500"),
                "the dispatching side carries its threshold too: {:?}",
                lines[1]
            );
            assert!(
                autonomous.drain_telemetry().is_empty(),
                "draining is a take, not a peek"
            );
        }

        /// Edge case: the buffer is bounded, so a long-lived TUI session cannot grow it without
        /// limit on model-driven input.
        #[test]
        fn the_telemetry_buffer_is_bounded() {
            let autonomous = AutonomousRunConfig::from_magi_config(&MagiConfig::default());
            let run = autonomous.apply(Default::default());
            for _ in 0..(MAX_GATE_TELEMETRY_LINES + 10) {
                run.gate_telemetry
                    .on_gate_evaluation(&Mode::Analysis, 1, 200, true);
            }
            assert_eq!(autonomous.drain_telemetry().len(), MAX_GATE_TELEMETRY_LINES);
        }
    }

    /// Task 5.2 — probe notices (REQ-A24c), the staleness-of-composition notice (SC-A24i), and
    /// `orchestrate_probes` (REQ-A24/SC-A24j/SC-A24k), the function that decides how many probe
    /// batches to launch and guarantees the trio table never includes the principal.
    mod probe_orchestration {
        use super::*;
        use async_trait::async_trait;
        use magi_core::rotation::ProviderProbe;
        use magi_rs::magi::probe::ProbeSeat;
        use std::sync::atomic::{AtomicUsize, Ordering};

        /// Probe that ALWAYS returns the same window, no digest, no real I/O.
        struct FixedWindowProbe {
            window: usize,
        }

        #[async_trait]
        impl ProviderProbe for FixedWindowProbe {
            async fn window(&self) -> Result<Option<usize>, ProviderError> {
                Ok(Some(self.window))
            }
            async fn digest(&self) -> Result<Option<String>, ProviderError> {
                Ok(None)
            }
        }

        /// `ProbeFactory` double with a FIXED window per model (map `model -> window`). A model
        /// ABSENT from the map degrades to `Unbuildable` — never panics, never invents a
        /// window. Does not reuse the private doubles from `magi::probe::tests` (they live in
        /// another module and are not exported) — R-A04 requires the same injection seam here.
        struct MappedProbeFactory {
            windows: BTreeMap<&'static str, usize>,
            /// How many times `probe_for` was called — to pin SC-A24h: re-reading an already-
            /// captured snapshot must never touch the factory again.
            calls: AtomicUsize,
        }

        impl MappedProbeFactory {
            fn new(pairs: &[(&'static str, usize)]) -> Self {
                Self {
                    windows: pairs.iter().copied().collect(),
                    calls: AtomicUsize::new(0),
                }
            }

            fn calls(&self) -> usize {
                self.calls.load(Ordering::SeqCst)
            }
        }

        impl ProbeFactory for MappedProbeFactory {
            fn probe_for(
                &self,
                kind: ProviderKind,
                _base_url: &ResolvedEndpoint,
                model: &str,
            ) -> ProbeSeat {
                self.calls.fetch_add(1, Ordering::SeqCst);
                if !kind.is_probeable() {
                    return ProbeSeat::NotProbeable;
                }
                match self.windows.get(model) {
                    Some(&window) => ProbeSeat::Ready(Arc::new(FixedWindowProbe { window })),
                    None => ProbeSeat::Unbuildable(redact_foreign_error(&std::io::Error::other(
                        "model not mapped in the test double",
                    ))),
                }
            }
        }

        /// Shared test endpoint: a flat `base_url` with no placeholders, so resolving it does
        /// not need a real vault (same pattern as `trio_construction`).
        fn test_endpoints() -> ResolvedEndpoints {
            let tpl = EndpointTemplate::parse("http://localhost:11434/v1", Scope::Root).unwrap();
            ResolvedEndpoints {
                root: tpl.resolve(&mut NoVaultInScope, Scope::Root).unwrap(),
                magi: tpl.resolve(&mut NoVaultInScope, Scope::Magi).unwrap(),
            }
        }

        /// DIVERGENT endpoints: the trio on a different host than the principal, to exercise
        /// the `join!` branch of `orchestrate_probes`.
        fn diverging_endpoints() -> ResolvedEndpoints {
            let root = EndpointTemplate::parse("http://root-host:11434/v1", Scope::Root)
                .unwrap()
                .resolve(&mut NoVaultInScope, Scope::Root)
                .unwrap();
            let magi = EndpointTemplate::parse("http://magi-host:11434/v1", Scope::Magi)
                .unwrap()
                .resolve(&mut NoVaultInScope, Scope::Magi)
                .unwrap();
            ResolvedEndpoints { root, magi }
        }

        /// `MagiConfig` whose `[magi]` declares its own `base_url` (and optionally `kind`) —
        /// the pair of fields that `magi_endpoint_diverges()` looks at. Without own section
        /// model: whoever needs one uses [`cfg_diverging_with_models`].
        ///
        /// `build_unvalidated`, not `build`: callers pass an unrecognized `kind` on purpose, to
        /// pin that the trio degrades instead of guessing, so the validating exit would reject
        /// exactly the cases this fixture exists to produce.
        fn cfg_diverging(kind: Option<&str>) -> MagiConfig {
            MagiConfig::builder()
                .magi(crate::config::MagiSectionConfig {
                    base_url: Some("http://magi-host:11434/v1".to_string()),
                    kind: kind.map(str::to_string),
                    ..crate::config::MagiSectionConfig::default()
                })
                .build_unvalidated()
        }

        /// `MagiConfig` with DISTINCT, nameable sections (`[openai].model`,
        /// `[anthropic].model`), with no per-seat override (`melchior_model` etc. — all three
        /// inherit the fallback). This is the fix-round-1 finding fixture: without own,
        /// controllable section names, there is no way to distinguish "the trio probed ITS
        /// model" from "the trio probed the principal's" — the two cases look identical if the
        /// two sections share the same name.
        fn cfg_with_distinct_section_models(
            openai_model: &str,
            anthropic_model: &str,
        ) -> MagiConfig {
            MagiConfig::builder()
                .openai(crate::config::OpenAiConfig {
                    model: Some(openai_model.to_string()),
                })
                .anthropic(crate::config::AnthropicConfig {
                    model: Some(anthropic_model.to_string()),
                })
                .build()
                .unwrap()
        }

        /// Like [`cfg_diverging`], but with BOTH nameable sections too — the fixture that
        /// exercises the fix-round-1 finding with the trio on a different endpoint AND a
        /// different kind than the principal at the same time. Unvalidated for the same reason
        /// as [`cfg_diverging`]: one caller passes an unrecognized `kind` deliberately.
        fn cfg_diverging_with_models(
            kind: Option<&str>,
            openai_model: &str,
            anthropic_model: &str,
        ) -> MagiConfig {
            crate::config::MagiConfigBuilder::from(cfg_with_distinct_section_models(
                openai_model,
                anthropic_model,
            ))
            .magi(crate::config::MagiSectionConfig {
                base_url: Some("http://magi-host:11434/v1".to_string()),
                kind: kind.map(str::to_string),
                ..crate::config::MagiSectionConfig::default()
            })
            .build_unvalidated()
        }

        // ---- probe_notice / stale_composition_notice (brief contract, Step 1) -----

        /// SC-A24f: cold start is explained, not confused with a failure.
        #[test]
        fn the_notice_distinguishes_the_three_measurement_states() {
            assert!(probe_notice(&Measurement::Measured {
                window: 128_000,
                digest: Some("ab".repeat(32)),
            })
            .contains("128000"));
            assert!(probe_notice(&Measurement::NotMeasurable).contains("does not offer"));
            let cold = probe_notice(&Measurement::NotMeasuredThisTime);
            assert!(
                cold.contains("this time") && cold.contains("next"),
                "must anticipate that the next startup will probably measure"
            );
        }

        /// The digest is displayed truncated: it is an identifier, not a secret, but 64 hex is
        /// noise.
        #[test]
        fn the_digest_is_shown_truncated() {
            let n = probe_notice(&Measurement::Measured {
                window: 1000,
                digest: Some("ab".repeat(32)),
            });
            assert!(!n.contains(&"ab".repeat(32)));
        }

        /// Edge: a window measured WITHOUT a digest (e.g. `/api/tags` did not resolve it) still
        /// reports the window — information that is available is not lost.
        #[test]
        fn a_measured_window_without_a_digest_still_reports_the_window() {
            let n = probe_notice(&Measurement::Measured {
                window: 128_000,
                digest: None,
            });
            assert!(n.contains("128000"));
            assert!(n.contains("digest not resolved"));
        }

        /// SC-A24i: it is warned, and the comparison is IN TOKENS — not bytes versus tokens.
        #[test]
        fn a_max_query_close_to_the_measured_window_is_flagged_after_unit_conversion() {
            let window_tokens = 100_000_usize;
            // A cap in BYTES that, when converted, lands just above the 80 % of the window.
            let close_bytes =
                ((window_tokens as f64 * STALE_NOTICE_RATIO * CHARS_PER_TOKEN_EST) as usize) + 8;
            let n = stale_composition_notice(window_tokens, close_bytes).expect("must warn");
            assert!(
                n.contains("tokens") && n.contains("chars/token"),
                "the notice must name the estimator: it is an approximation, not a measurement"
            );

            assert!(
                stale_composition_notice(window_tokens, close_bytes / 10).is_none(),
                "with wide slack there is no risk to announce"
            );
        }

        /// The bug this pair of functions exists to avoid: comparing bytes against tokens.
        #[test]
        fn comparing_raw_bytes_against_a_token_window_would_be_meaningless() {
            let window_tokens = 128_000_usize;
            // Not a loose literal: the test follows the real value.
            let cap_bytes = magi_rs::magi::MAX_QUERY_BYTES;
            assert!(
                cap_bytes > window_tokens,
                "raw, the cap 'exceeds' the window..."
            );
            assert!(
                bytes_to_tokens_est(cap_bytes) < window_tokens,
                "...but converted it does NOT exceed it: without conversion the notice \
                 would always fire"
            );
        }

        // ---- trio_probe_incomplete_notice (MAGI S3 re-gate, Caspar) --------------

        /// MAGI S3 re-gate (Caspar): the principal's own probe notice can succeed while the
        /// trio's cold-start failure — which is what actually drives `input_warn_tokens` — goes
        /// unreported. A mage stuck at `NotMeasuredThisTime`, with no declared
        /// `[magi].input_warn_tokens` to make the derivation failure moot, must surface a
        /// notice.
        #[test]
        fn a_cold_mage_with_no_declared_fallback_is_reported() {
            let trio = BTreeMap::from([
                (
                    "melchior-model".to_string(),
                    Measurement::NotMeasuredThisTime,
                ),
                (
                    "balthasar-model".to_string(),
                    Measurement::Measured {
                        window: 128_000,
                        digest: None,
                    },
                ),
            ]);
            let n = trio_probe_incomplete_notice(&trio, None)
                .expect("a cold mage with no declared fallback must warn");
            assert!(
                n.contains("input_warn_tokens") && n.contains("cold"),
                "must name the key and the cause: {n}"
            );
        }

        /// A declared `[magi].input_warn_tokens` already wins over anything derived
        /// (REQ-A24b/SC-A24e): the derivation failing changes nothing observable, so there is
        /// nothing to warn about.
        #[test]
        fn a_declared_fallback_makes_the_derivation_failure_moot() {
            let trio = BTreeMap::from([(
                "melchior-model".to_string(),
                Measurement::NotMeasuredThisTime,
            )]);
            assert!(
                trio_probe_incomplete_notice(&trio, Some(150_000)).is_none(),
                "with the key declared, the derivation failing changes nothing observable"
            );
        }

        /// Every mage `NotMeasurable` (not `NotMeasuredThisTime`) is the EXPECTED,
        /// non-actionable case of a `kind` with no introspection (SC-A24b) — not a cold-start
        /// failure, so no notice: `probe_notice`'s "does not offer introspection" wording
        /// already covers this for the principal, and repeating it for the trio would be the
        /// same non-news twice.
        #[test]
        fn every_mage_not_measurable_is_not_reported_as_a_cold_start() {
            let trio = BTreeMap::from([
                ("melchior-model".to_string(), Measurement::NotMeasurable),
                ("balthasar-model".to_string(), Measurement::NotMeasurable),
            ]);
            assert!(
                trio_probe_incomplete_notice(&trio, None).is_none(),
                "NotMeasurable is not a transient failure: it does not warrant this notice"
            );
        }

        /// An empty trio is the degenerate case: no mage, nothing cold, nothing to report.
        /// (`probe_and_report` also calls this unconditionally now — S8 gate re-review fix —
        /// so it is no longer gated behind `min_mage_window` returning `None`.)
        #[test]
        fn an_empty_trio_reports_nothing() {
            assert!(trio_probe_incomplete_notice(&BTreeMap::new(), None).is_none());
        }

        /// `MagiConfig` with the trio on the SAME endpoint/kind as the principal (shared branch
        /// of `orchestrate_probes`), but with all FOUR names — principal + three mages —
        /// distinct and test-controllable.
        ///
        /// **The three lineages are not decoration.** A seat that declares a model must declare
        /// its lineage (REQ-R02), and the three must differ while `enforce_diversity` is on
        /// (REQ-R29) — which is its default. This helper used to omit them and `build()` accepted
        /// it, because the seat-lineage check ran in `from_toml_str` rather than in
        /// `validate_vocabulary`; these four probe tests were therefore running against a config
        /// no operator could ever load. S1 Loop 2 closed that hole, and the full suite is what
        /// surfaced these — the per-commit scoped run does not reach this module.
        fn cfg_with_four_distinct_models(
            principal: &str,
            melchior: &str,
            balthasar: &str,
            caspar: &str,
        ) -> MagiConfig {
            MagiConfig::builder()
                .openai(crate::config::OpenAiConfig {
                    model: Some(principal.to_string()),
                })
                .magi(crate::config::MagiSectionConfig {
                    melchior_model: Some(melchior.to_string()),
                    melchior_lineage: Some("lineage-m".to_string()),
                    balthasar_model: Some(balthasar.to_string()),
                    balthasar_lineage: Some("lineage-b".to_string()),
                    caspar_model: Some(caspar.to_string()),
                    caspar_lineage: Some("lineage-c".to_string()),
                    ..crate::config::MagiSectionConfig::default()
                })
                .build()
                .unwrap()
        }

        // ---- orchestrate_probes: shared branch --------------------------------------

        /// SC-A24 / REQ-A24: shared endpoint and kind ⇒ ONE batch (one probe per unique model,
        /// principal included), and the returned trio table NEVER includes the principal.
        #[tokio::test]
        async fn shared_endpoint_probes_once_and_the_trio_table_excludes_the_principal() {
            let factory = MappedProbeFactory::new(&[
                ("principal", 4_096),
                ("melchior", 128_000),
                ("balthasar", 200_000),
                ("caspar", 256_000),
            ]);
            let cfg = cfg_with_four_distinct_models("principal", "melchior", "balthasar", "caspar");
            let (principal_model, principal, trio) = orchestrate_probes(
                &cfg,
                &test_endpoints(),
                ProviderKind::Ollama,
                &factory,
                &MagiEnvModelOverrides::default(),
                &[],
            )
            .await;

            assert_eq!(principal_model, "principal");
            assert!(matches!(
                principal,
                Some(Measurement::Measured { window: 4_096, .. })
            ));
            assert_eq!(trio.len(), 3, "only the THREE mages, never the principal");
            assert!(!trio.contains_key("principal"));
            assert!(matches!(
                trio["melchior"],
                Measurement::Measured {
                    window: 128_000,
                    ..
                }
            ));
            assert_eq!(
                factory.calls(),
                4,
                "one batch: all 4 probes (principal + 3 mages) are requested together"
            );
        }

        /// SC-A24j — the central property of this task: a small-window principal does NOT lower
        /// the derived threshold for the mages, because `derive_warn_tokens` never sees its
        /// measurement — `orchestrate_probes` excludes it from the trio table by construction,
        /// not by convention in the caller.
        #[tokio::test]
        async fn a_small_principal_never_lowers_the_mage_derived_threshold() {
            let factory = MappedProbeFactory::new(&[
                ("principal", 2_048), // the SMALLEST window in the whole process
                ("melchior", 1_000_000),
                ("balthasar", 512_000),
                ("caspar", 256_000),
            ]);
            let cfg = cfg_with_four_distinct_models("principal", "melchior", "balthasar", "caspar");
            let (_principal_model, _principal, trio) = orchestrate_probes(
                &cfg,
                &test_endpoints(),
                ProviderKind::Ollama,
                &factory,
                &MagiEnvModelOverrides::default(),
                &[],
            )
            .await;

            let derived = derive_warn_tokens(&trio).expect("all three mages measured");
            #[allow(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss
            )]
            let expected_from_caspar = (256_000.0 * magi_rs::magi::WARN_WINDOW_FRACTION) as usize;
            assert_eq!(
                derived, expected_from_caspar,
                "the MAGES' minimum (Caspar, 256k) wins — NEVER the principal's (2k), \
                 which would be an absurdly low threshold if it had leaked through"
            );
        }

        /// SC-A24h: the threshold derived from a startup snapshot is STABLE. The spec's full
        /// guarantee ("the probe runs ONCE per process") is structural — a single call site in
        /// `run()`/`prepare_headless()`, before the turn loop starts — and that is not
        /// something a unit test can exercise without starting the whole process. What IS
        /// verifiable here, and is the half of the property a unit test can pin: re-reading the
        /// SAME already-captured snapshot is a pure, deterministic operation that never touches
        /// the probe factory again — if something in the process "refreshed" the threshold just
        /// in case, this would expose it by a call count that rises alone.
        #[tokio::test]
        async fn the_probe_runs_once_and_the_threshold_stays_put() {
            let factory = MappedProbeFactory::new(&[("principal", 128_000), ("m", 256_000)]);
            let cfg = cfg_with_four_distinct_models("principal", "m", "m", "m");
            let (_principal_model, _principal, trio) = orchestrate_probes(
                &cfg,
                &test_endpoints(),
                ProviderKind::Ollama,
                &factory,
                &MagiEnvModelOverrides::default(),
                &[],
            )
            .await;
            let calls_after_the_startup_probe = factory.calls();
            assert!(
                calls_after_the_startup_probe > 0,
                "the probe must have run at least once at startup"
            );

            // Two "reads" of the same snapshot, like two successive queries inside the SAME
            // session — nothing here invokes `orchestrate_probes` again.
            let warn_at_startup = derive_warn_tokens(&trio);
            let warn_for_a_later_query = derive_warn_tokens(&trio);
            assert_eq!(
                warn_at_startup, warn_for_a_later_query,
                "the threshold derived from the startup snapshot does not change on its own"
            );
            assert_eq!(
                factory.calls(),
                calls_after_the_startup_probe,
                "deriving the threshold from an already-captured snapshot NEVER touches \
                 the probe again"
            );
        }

        /// Sixth-pass gate finding (S8, Balthasar): `orchestrate_probes` read
        /// `cfg.magi.seats(...)` raw, ignoring `MAGI_MODEL_<AGENT>` env overrides entirely —
        /// so with an override set, the probe measured the TOML/backend model's window while
        /// `build_magi_orchestrator` (which DOES apply the override via
        /// `resolve_magi_override`) actually ran the trio on a DIFFERENT model.
        /// `input_warn_tokens` derives from the probe's measurement (the MINIMUM across the
        /// mages, REQ-A24b) — measuring the wrong model here silently poisons that threshold
        /// for the model that is actually running, and per SC-A24j the dangerous direction is
        /// a threshold that comes out TOO HIGH (a guard-rail that switched itself off).
        #[tokio::test]
        async fn env_overrides_change_which_model_the_probe_measures() {
            let factory = MappedProbeFactory::new(&[
                ("principal", 4_096),
                ("toml-melchior", 1_000), // the TOML model — must NOT be what gets probed
                ("env-melchior", 128_000), // env-overridden — must be probed INSTEAD
                ("balthasar", 200_000),
                ("caspar", 256_000),
            ]);
            let cfg =
                cfg_with_four_distinct_models("principal", "toml-melchior", "balthasar", "caspar");
            let env_overrides = MagiEnvModelOverrides::from_raw(Some("env-melchior"), None, None);

            let (_principal_model, _principal, trio) = orchestrate_probes(
                &cfg,
                &test_endpoints(),
                ProviderKind::Ollama,
                &factory,
                &env_overrides,
                &[],
            )
            .await;

            assert!(
                trio.contains_key("env-melchior"),
                "the probe must measure the ENV-overridden model, the one the trio actually \
                 runs: {trio:?}"
            );
            assert!(
                !trio.contains_key("toml-melchior"),
                "the TOML model must not be probed once an env override wins for that seat: \
                 {trio:?}"
            );
        }

        // ---- orchestrate_probes: divergent branch --------------------------------------

        /// SC-A24k (one level up, between BATCHES): divergent endpoint ⇒ TWO independent calls,
        /// each with its own kind and endpoint. Principal AND trio on `ollama` (the ONLY
        /// measurable kind) on purpose, with the three trio seats overridden to their own model
        /// — this is the only way for the principal and the trio to measure DIFFERENT windows
        /// while both are measurable: if they shared kind AND no seat had an override, they
        /// would resolve to the SAME section by design (see
        /// `a_diverging_trio_kind_probes_its_own_section_model_not_the_principals`, which does
        /// cross sections).
        #[tokio::test]
        async fn diverging_endpoint_probes_the_trio_separately_with_its_own_kind() {
            let factory = MappedProbeFactory::new(&[("principal", 64_000), ("m", 128_000)]);
            let cfg = MagiConfig::builder()
                .openai(crate::config::OpenAiConfig {
                    model: Some("principal".to_string()),
                })
                .magi(crate::config::MagiSectionConfig {
                    base_url: Some("http://magi-host:11434/v1".to_string()),
                    kind: Some("ollama".to_string()),
                    // One model across the three seats, three distinct lineages: the seats share
                    // a model deliberately (probe dedup), and diversity is a property of the
                    // seats, not of the models (REQ-R29, SC-R52).
                    melchior_model: Some("m".to_string()),
                    melchior_lineage: Some("lineage-m".to_string()),
                    balthasar_model: Some("m".to_string()),
                    balthasar_lineage: Some("lineage-b".to_string()),
                    caspar_model: Some("m".to_string()),
                    caspar_lineage: Some("lineage-c".to_string()),
                    ..crate::config::MagiSectionConfig::default()
                })
                .build()
                .unwrap();
            let (principal_model, principal, trio) = orchestrate_probes(
                &cfg,
                &diverging_endpoints(),
                ProviderKind::Ollama,
                &factory,
                &MagiEnvModelOverrides::default(),
                &[],
            )
            .await;
            assert_eq!(principal_model, "principal");
            assert!(matches!(
                principal,
                Some(Measurement::Measured { window: 64_000, .. })
            ));
            assert!(
                trio.values().all(|m| matches!(
                    m,
                    Measurement::Measured {
                        window: 128_000,
                        ..
                    }
                )),
                "the three seats overridden to \"m\" must measure \"m\"'s window"
            );
        }

        /// **Rejected finding (S8 review round, finding 2) — pinned with a regression test.**
        /// The finding claimed `orchestrate_probes` "assumes a shared endpoint implies a
        /// shared kind": that a literally-shared `base_url` between the principal and the
        /// trio routes both into the single-batch branch regardless of `kind`, silently
        /// attributing one model's window to the other. That is not what the code does:
        /// `MagiConfig::magi_endpoint_diverges()` (`config.rs`) is `declara_url ||
        /// declara_kind` — declaring `[magi].kind` ALONE routes to the kind-aware `join!`
        /// branch, with NO dependency on whether `[magi].base_url` is also declared. The
        /// single-batch branch is only taken when NEITHER is declared, in which case the
        /// trio's kind is not merely "probably the same" — it is PROVABLY identical to the
        /// principal's by construction (`resolve_magi_kind` falls back to `principal_kind`
        /// under the exact same absence), never by comparing resolved URL strings.
        ///
        /// This reproduces the finding's own scenario directly: `[magi].kind` declared
        /// (`ollama`, diverging from the principal's `anthropic`) with `[magi].base_url`
        /// ABSENT, so the trio's resolved endpoint is the LITERAL SAME STRING as the
        /// principal's (asserted below as a precondition). If the code really reused one
        /// measurement off endpoint equality alone, the principal's non-probeable
        /// `anthropic` kind would either poison the trio's result or the trio would never
        /// be probed under its own (probeable) kind at all. Neither happens: the principal
        /// degrades to `NotMeasurable` (anthropic has no probe endpoint) while the trio, on
        /// the SAME endpoint string, is measured under its own `ollama` kind.
        #[tokio::test]
        async fn a_shared_endpoint_with_a_declared_diverging_kind_probes_with_its_own_kind() {
            let endpoints = test_endpoints();
            assert_eq!(
                endpoints.root.as_str(),
                endpoints.magi.as_str(),
                "precondition: the trio's endpoint is literally the same string as the \
                 principal's — no `[magi].base_url` override is declared below"
            );

            let factory = MappedProbeFactory::new(&[("m", 128_000)]);
            let cfg = MagiConfig::builder()
                .magi(crate::config::MagiSectionConfig {
                    // NOTE: `kind` diverges from the principal; `base_url` does NOT.
                    kind: Some("ollama".to_string()),
                    // The three seats share ONE model on purpose — this pins that the probe
                    // batch dedupes by model. Their LINEAGES still have to differ, because
                    // `enforce_diversity` defaults on and it is a property of the seats, not of
                    // the models they happen to point at (REQ-R29, SC-R52).
                    melchior_model: Some("m".to_string()),
                    melchior_lineage: Some("lineage-m".to_string()),
                    balthasar_model: Some("m".to_string()),
                    balthasar_lineage: Some("lineage-b".to_string()),
                    caspar_model: Some("m".to_string()),
                    caspar_lineage: Some("lineage-c".to_string()),
                    ..crate::config::MagiSectionConfig::default()
                })
                .build()
                .unwrap();

            let (_principal_model, principal, trio) = orchestrate_probes(
                &cfg,
                &endpoints,
                ProviderKind::Anthropic,
                &factory,
                &MagiEnvModelOverrides::default(),
                &[],
            )
            .await;

            assert!(
                matches!(principal, Some(Measurement::NotMeasurable)),
                "the principal (anthropic, not probeable) must not inherit a measurement \
                 from the trio's shared-string endpoint: {principal:?}"
            );
            assert!(
                trio.values().all(|m| matches!(
                    m,
                    Measurement::Measured {
                        window: 128_000,
                        ..
                    }
                )),
                "the trio (ollama, probeable) must still be measured under its OWN kind \
                 even though its endpoint string is identical to the principal's: {trio:?}"
            );
        }

        /// **Fix round 1 — Logic+Structure finding.** Reproduces the EXACT reported bug:
        /// principal on `anthropic` (reads `[anthropic].model`), trio on `ollama` (declared
        /// `[magi].kind`, diverging — reads `[openai].model`), and NO seat with its own
        /// override (`melchior_model`/`balthasar_model`/`caspar_model` absent), so all three
        /// inherit the fallback. The correct fallback is `[openai].model` — the trio's KIND
        /// section — never `[anthropic].model`, the principal's section.
        ///
        /// Before this fix, the two call sites (`run()`/`prepare_headless()`) resolved the
        /// trio's fallback with `resolve_backend_model(cfg, principal_kind)` — the principal's
        /// KIND, not the trio's — so a trio on `ollama` with the principal on `anthropic`
        /// attempted to probe the NAME of `[anthropic].model` against the trio's endpoint.
        ///
        /// The two models map to DIFFERENT windows (`claude-test` → 999 999, `qwen-test` → 128
        /// 000) so that, if the bug reappears, it shows up as a WRONG NUMBER — poisoning `input_warn_tokens` with a foreign model's window — rather than just a degradation to "not measured", which would be easier to overlook in a cursory review.
        #[tokio::test]
        async fn a_diverging_trio_kind_probes_its_own_section_model_not_the_principals() {
            let factory = MappedProbeFactory::new(&[
                ("claude-test", 999_999), // [anthropic].model — the PRINCIPAL's section
                ("qwen-test", 128_000),   // [openai].model — the trio's REAL section
            ]);
            let cfg = cfg_diverging_with_models(Some("ollama"), "qwen-test", "claude-test");

            let (principal_model, _principal, trio) = orchestrate_probes(
                &cfg,
                &diverging_endpoints(),
                ProviderKind::Anthropic,
                &factory,
                &MagiEnvModelOverrides::default(),
                &[],
            )
            .await;

            assert_eq!(
                principal_model, "claude-test",
                "the principal MUST resolve its own section — this is not what fails"
            );
            assert!(
                trio.values().all(|m| matches!(
                    m,
                    Measurement::Measured {
                        window: 128_000,
                        ..
                    }
                )),
                "the trio must probe qwen-test (ITS section, [openai].model under kind \
                 ollama) — never claude-test (the principal's): otherwise it would measure \
                 999999 (a foreign model's window) or degrade to NotMeasuredThisTime if \
                 claude-test did not exist on the real endpoint, and in no case would the \
                 derived threshold relate to the trio. Trio: {trio:?}"
            );
        }

        /// An invalid `[magi].kind` does not propagate error nor panic: it degrades the ENTIRE
        /// trio to
        /// *unmeasured*, without guessing a kind — `build_magi_orchestrator` is the one that
        /// reports
        /// the typed error when it builds the real trio with the SAME config.
        #[tokio::test]
        async fn an_invalid_magi_kind_degrades_the_trio_without_guessing() {
            let factory = MappedProbeFactory::new(&[("principal", 64_000)]);
            let cfg = cfg_diverging_with_models(Some("banana"), "principal", "irrelevant");
            let (principal_model, principal, trio) = orchestrate_probes(
                &cfg,
                &diverging_endpoints(),
                ProviderKind::Ollama,
                &factory,
                &MagiEnvModelOverrides::default(),
                &[],
            )
            .await;
            assert_eq!(principal_model, "principal");
            assert!(
                matches!(
                    principal,
                    Some(Measurement::Measured { window: 64_000, .. })
                ),
                "the principal IS probed: its kind is valid by construction"
            );
            assert!(
                trio.values()
                    .all(|m| matches!(m, Measurement::NotMeasuredThisTime)),
                "all three seats degrade without guessing any model"
            );
        }

        // ---- probe_and_report: consolidation of the duplicated block (fix round 1, B3) ---

        /// SC-A24e: what is DECLARED in `[magi].input_warn_tokens` wins over what is MEASURED —
        /// previously with no own test (an inline `Option::or_else` at each call site); now
        /// that the block is a shared function, it is a cheap assertion.
        #[tokio::test]
        async fn declared_input_warn_tokens_beats_the_measured_threshold() {
            let factory = MappedProbeFactory::new(&[
                ("principal", 128_000),
                ("melchior", 128_000),
                ("balthasar", 128_000),
                ("caspar", 128_000),
            ]);
            let base =
                cfg_with_four_distinct_models("principal", "melchior", "balthasar", "caspar");
            // `input_warn_tokens` can no longer be poked in after construction (REQ-R23), so the
            // fixture is reopened and the value declared before the (validating) build.
            let magi = crate::config::MagiSectionConfig {
                input_warn_tokens: Some(999),
                ..base.magi().clone()
            };
            let cfg = crate::config::MagiConfigBuilder::from(base)
                .magi(magi)
                .build()
                .unwrap();
            let mut notices = Vec::new();
            let (warn_tokens, _measured) = probe_and_report(
                &cfg,
                &test_endpoints(),
                ProviderKind::Ollama,
                &factory,
                &MagiEnvModelOverrides::default(),
                &[],
                &mut notices,
            )
            .await;
            assert_eq!(
                warn_tokens,
                Some(999),
                "the declared value wins even though the probe DID measure something different"
            );
        }

        /// SC-A24e (the other side): with nothing declared, the threshold comes from what is
        /// MEASURED.
        #[tokio::test]
        async fn absent_input_warn_tokens_falls_back_to_the_measured_threshold() {
            let factory = MappedProbeFactory::new(&[
                ("principal", 4_096),
                ("melchior", 128_000),
                ("balthasar", 128_000),
                ("caspar", 128_000),
            ]);
            let cfg = cfg_with_four_distinct_models("principal", "melchior", "balthasar", "caspar");
            let mut notices = Vec::new();
            let (warn_tokens, _measured) = probe_and_report(
                &cfg,
                &test_endpoints(),
                ProviderKind::Ollama,
                &factory,
                &MagiEnvModelOverrides::default(),
                &[],
                &mut notices,
            )
            .await;
            #[allow(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss
            )]
            let expected = (128_000.0 * magi_rs::magi::WARN_WINDOW_FRACTION) as usize;
            assert_eq!(warn_tokens, Some(expected));
            assert!(
                notices.iter().any(|n| n.text.contains("principal")),
                "the principal's notice was pushed into the shared list"
            );
        }

        /// S8 gate re-review finding (Caspar): `min_mage_window` returns `Some` as soon as
        /// ONE mage measures, so a branch that only checks `trio_probe_incomplete_notice`
        /// when `min_mage_window` is `None` skips it whenever the trio is PARTIALLY cold —
        /// exactly the case where it matters most. `derive_warn_tokens` takes the minimum of
        /// whichever mages happened to measure; if the cold one would have had the smallest
        /// window, the derived threshold comes out too high and the size warning stops firing
        /// silently for that mage.
        #[tokio::test]
        async fn a_partially_cold_trio_still_reports_the_incomplete_measurement_notice() {
            // Two of three mages measured; "caspar" is absent from the map, so `probe_for`
            // returns `Unbuildable` -> `Measurement::NotMeasuredThisTime` (the cold case) —
            // `min_mage_window` still returns `Some(128_000)` from the other two.
            let factory = MappedProbeFactory::new(&[
                ("principal", 128_000),
                ("melchior", 128_000),
                ("balthasar", 128_000),
            ]);
            let cfg = cfg_with_four_distinct_models("principal", "melchior", "balthasar", "caspar");
            let mut notices = Vec::new();
            let (warn_tokens, _measured) = probe_and_report(
                &cfg,
                &test_endpoints(),
                ProviderKind::Ollama,
                &factory,
                &MagiEnvModelOverrides::default(),
                &[],
                &mut notices,
            )
            .await;

            assert!(
                warn_tokens.is_some(),
                "two of three mages measured: the derivation itself still succeeds — this is \
                 not the bug"
            );
            assert!(
                notices
                    .iter()
                    .any(|n| n.text.contains("input_warn_tokens") && n.text.contains("cold")),
                "a partially cold trio must still surface the incomplete-measurement notice, \
                 not just a fully cold one: {notices:?}"
            );
        }

        // ---- resolve_magi_kind ---------------------------------------------------------

        /// Absent `[magi].kind` inherits the ALREADY-RESOLVED principal one — not
        /// `cfg.effective_provider()`/`cfg.effective_magi_kind()` (TOML-only), which would
        /// ignore `MAGI_PROVIDER`.
        #[test]
        fn resolve_magi_kind_inherits_the_resolved_principal_when_absent() {
            let cfg = MagiConfig::default();
            assert_eq!(
                resolve_magi_kind(&cfg, ProviderKind::Anthropic).unwrap(),
                ProviderKind::Anthropic,
                "inherits the ALREADY RESOLVED value, not cfg.effective_provider() \
                 (which would give Ollama)"
            );
        }

        /// Declared `[magi].kind` wins over the principal.
        #[test]
        fn resolve_magi_kind_prefers_the_declared_value_over_the_principal() {
            let cfg = cfg_diverging(Some("anthropic"));
            assert_eq!(
                resolve_magi_kind(&cfg, ProviderKind::Ollama).unwrap(),
                ProviderKind::Anthropic
            );
        }

        /// Unrecognized `[magi].kind` is a TYPED error, not a silent fallback.
        #[test]
        fn resolve_magi_kind_rejects_an_unknown_value() {
            let cfg = cfg_diverging(Some("banana"));
            let err = resolve_magi_kind(&cfg, ProviderKind::Ollama).unwrap_err();
            assert_eq!(err.got, "banana");
        }

        // ---- registered_magi_kind -------------------------------------------------------

        /// MS2 gate S8 finding: the value reported to `ConsultTool`/`MagiRuntimeParams` (and
        /// therefore to `explain_magi_error`'s keyless-auth hint, REQ-A12c) must follow the
        /// ALREADY-RESOLVED principal kind, exactly like `resolve_magi_kind` — never
        /// `MagiConfig::effective_magi_kind()` directly, which is TOML-only and ignores
        /// `MAGI_PROVIDER`. This is the same divergence `resolve_magi_kind_inherits_the_
        /// resolved_principal_when_absent` proves for the trio's own construction; this test
        /// proves the SAME value is what every downstream consumer sees too — the four call
        /// sites (`register_consult_tool_if_available` in `run()`/`run_query_subcommand`,
        /// `TuiMagiRuntimeConfig` in `run()`, `MagiRuntimeParams` in `run_consult_subcommand`)
        /// all resolve through this one function rather than each re-deriving it.
        #[test]
        fn registered_magi_kind_follows_the_resolved_principal_not_the_toml_only_accessor() {
            // `[magi].kind` absent; TOML root `provider` absent too, so the TOML-only
            // accessor falls to the built-in default (Ollama) — see the existing
            // `resolve_magi_kind_inherits_the_resolved_principal_when_absent` comment.
            let cfg = MagiConfig::default();
            assert_eq!(
                cfg.effective_magi_kind(),
                ProviderKind::Ollama,
                "sanity: this fixture only proves something if the TOML-only accessor \
                 actually disagrees with the env-resolved principal below"
            );

            // `MAGI_PROVIDER=openai-compat` moved the PRINCIPAL without declaring
            // `[magi].kind` — exactly the scenario the env-override gap reopens.
            assert_eq!(
                registered_magi_kind(&cfg, ProviderKind::OpenAiCompat),
                ProviderKind::OpenAiCompat,
                "must follow the resolved principal kind (matching resolve_magi_kind), \
                 not the TOML-only accessor"
            );
        }

        /// A declared `[magi].kind` still wins over the principal through this function too
        /// (it is a thin wrapper over `resolve_magi_kind`, not a second, divergent rule).
        #[test]
        fn registered_magi_kind_prefers_the_declared_value_over_the_principal() {
            let cfg = cfg_diverging(Some("anthropic"));
            assert_eq!(
                registered_magi_kind(&cfg, ProviderKind::Ollama),
                ProviderKind::Anthropic
            );
        }

        /// An unrecognized `[magi].kind` falls back to the principal here instead of
        /// propagating `resolve_magi_kind`'s typed error — safe because a genuinely invalid
        /// `[magi].kind` already made the trio unbuildable upstream (`build_magi_orchestrator`
        /// validates it and returns `Err`, so `consult_magi` is `None` and no call site ever
        /// registers `ConsultTool`/builds `MagiRuntimeParams` with this fallback value).
        #[test]
        fn registered_magi_kind_falls_back_to_the_principal_on_an_unrecognized_value() {
            let cfg = cfg_diverging(Some("banana"));
            assert_eq!(
                registered_magi_kind(&cfg, ProviderKind::Ollama),
                ProviderKind::Ollama
            );
        }

        // ---- resolve_backend_model ------------------------------------------------------

        /// `[openai].model` for `ollama`/`openai-compat`, `[anthropic].model` for `anthropic` —
        /// `[openai]` serves the first two because they share the protocol.
        #[test]
        fn resolve_backend_model_picks_the_section_matching_the_kind() {
            let cfg = MagiConfig::builder()
                .openai(crate::config::OpenAiConfig {
                    model: Some("qwen-test".to_string()),
                })
                .anthropic(crate::config::AnthropicConfig {
                    model: Some("claude-test".to_string()),
                })
                .build()
                .unwrap();
            assert_eq!(
                resolve_backend_model(&cfg, ProviderKind::Ollama),
                "qwen-test"
            );
            assert_eq!(
                resolve_backend_model(&cfg, ProviderKind::OpenAiCompat),
                "qwen-test"
            );
            assert_eq!(
                resolve_backend_model(&cfg, ProviderKind::Anthropic),
                "claude-test"
            );
        }

        /// Edge: both absent ⇒ the crate's built-in defaults, no `panic`/`unwrap`.
        #[test]
        fn resolve_backend_model_falls_back_to_the_built_in_defaults_when_absent() {
            let cfg = MagiConfig::default();
            assert_eq!(
                resolve_backend_model(&cfg, ProviderKind::Ollama),
                crate::defaults::DEFAULT_OPENAI_MODEL
            );
            assert_eq!(
                resolve_backend_model(&cfg, ProviderKind::Anthropic),
                crate::defaults::DEFAULT_ANTHROPIC_MODEL
            );
        }
    }
}
