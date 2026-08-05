#![forbid(unsafe_code)]

mod agent;
mod config;
mod defaults;
mod headless_runner;
mod memory;
mod services;
mod system;
mod tools;
mod tui;

use crate::agent::provider::{build_openai_provider, AnthropicProvider, Provider, StaticProvider};
use crate::agent::Agent;
// NOTE: this `MagiConfig` is the magi-rs TOML config (`crate::config::MagiConfig`).
// It is DISTINCT from `magi_core::orchestrator::MagiConfig` — the latter is NEVER
// imported here, avoiding the name collision.
use crate::config::{
    resolve_anthropic_model, resolve_effective_provider_kind, resolve_magi_override,
    resolve_openai_model, HeadlessConfig, MagiConfig,
};
use crate::headless_runner::{resolve_run_timeout, run_consult, run_query, MagiRuntimeParams};
use crate::memory::clock::SystemClock;
use crate::memory::embedding::OpenAiCompatibleEmbedder;
use crate::memory::store::SqliteVectorStore;
use crate::system::database::{EncryptedSqliteMemory, MemoryStore};
use crate::system::fs::{FileSystem, RealFileSystem};
use crate::system::grep::RipGrep;
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
use magi_core::providers::openai_compat::OpenAiCompatibleProvider;
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
use magi_rs::magi::probe::{
    derive_warn_tokens, min_mage_window, probe_models, Measurement, OllamaProbeFactory,
    ProbeFactory,
};
use magi_rs::magi::{
    bytes_to_tokens_est, derive_client_timeout, derive_operation_budget, AGENT_TIMEOUT_SECS,
    CHARS_PER_TOKEN_EST, STALE_NOTICE_RATIO,
};
use magi_rs::notices::{render_notices, Notice};
use magi_rs::redact::{redact_foreign_error, redact_url, SafeErrorText};
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
    #[allow(dead_code)]
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
    if let Some(key) = env_key {
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
    if let Some(key) = env_key {
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
        // (probeability), never in how the PRINCIPAL provider is built (D-A07 is
        // about the native trio, not this untouched path).
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
    if let Err(e) = magi_config.memory.validate() {
        startup_notices.push(Notice::resolution(format!("memory config warning: {e}")));
    }
    // H2: surface invalid embedding-config values alongside memory-config (never panic).
    if let Err(e) = magi_config.embedding.validate() {
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

    // REQ-A24/A24b/A24c (Task 5.2): mide el principal y el trío ANTES de construirlo, así
    // `input_warn_tokens` puede derivarse de la ventana de los MAGES (REQ-A24b) y el
    // arranque anuncia los tres estados de medición (REQ-A24c). Nunca bloquea ni falla el
    // arranque: cada sonda falla abierta dentro de `probe_models`/`orchestrate_probes`.
    let warn_tokens = probe_and_report(
        &magi_config,
        &endpoints,
        provider_kind,
        &OllamaProbeFactory,
        &mut startup_notices,
    )
    .await;

    // Task 4.3 (REQ-A06/SC-A06b): el notice de arranque y la respuesta que un futuro
    // `/consult` verá comparten el MISMO texto, construido una sola vez acá —
    // `trio_unavailable_for_tui` es lo que hace esa igualdad verificable en vez de
    // depender de que este sitio y `run_tui_ext` construyan el mismo `String` por su
    // cuenta.
    let mut consult_unavailable_message: Option<String> = None;
    let consult_magi: Option<Arc<Magi>> = match build_magi_orchestrator(
        &magi_config,
        provider_kind,
        &endpoints,
        Some(&creds),
        warn_tokens,
        &MagiEnvModelOverrides::from_env(),
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
    let tui_mode_classifier: Arc<dyn magi_rs::magi::mode::ModeClassifier> =
        Arc::new(crate::agent::mode_classifier::ProviderClassifier::new(
            provider.clone(),
            Arc::new(crate::agent::mode_classifier::ProcessNoticeSink::default()),
        ));
    let tui_default_mode = magi_config.effective_default_mode();
    let tui_untrusted_content = magi_config.magi.untrusted_content.unwrap_or(false);
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
        magi_config.magi.auto_approve,
        magi_config.effective_magi_kind(),
        magi_config.magi_endpoint_diverges(),
    );

    crate::tui::run_tui_ext(
        agent,
        render_notices(startup_notices),
        crate::tui::TuiConsultWiring {
            consult: consult_magi,
            consult_unavailable_message,
            magi_auto_approve: magi_config.magi.auto_approve,
            // M1 fix: threads `[magi].agent_timeout_secs` through to the
            // post-`/login` trio rebuild, which used to hardcode the
            // built-in default regardless of this config.
            agent_timeout_secs: magi_config.magi.agent_timeout_secs,
        },
        secret_store,
        crate::tui::TuiMagiRuntimeConfig {
            mode_classifier: tui_mode_classifier,
            default_mode: tui_default_mode,
            untrusted_content: tui_untrusted_content,
            magi_kind: magi_config.effective_magi_kind(),
        },
    )
    .await?;
    Ok(ExitCode::SUCCESS)
}

/// Credenciales de terceros resueltas `env > vault` (REQ-A12), reducidas a lo que el
/// trío nativo necesita: una API key por backend que la exige (`openai-compat`,
/// `anthropic`). `ollama` es keyless y nunca las consulta.
///
/// Aparte del endpoint y no redundante con él: [`ResolvedEndpoint`] puede traer
/// `userinfo` (autenticación del proxy o del servidor que sirve el modelo), mientras
/// que la API key del backend va en un header (`Authorization: Bearer` / `x-api-key`).
/// Dos credenciales, dos destinos.
trait Credentials {
    /// La API key para el transporte OpenAI-compat (`OPENAI_API_KEY`).
    fn openai(&self) -> Option<String>;
    /// La API key de Anthropic (`ANTHROPIC_API_KEY`).
    fn anthropic(&self) -> Option<String>;
}

/// Puente entre la resolución `env > vault` ya existente ([`discover_config`],
/// [`resolve_openai_key`]) y el trait [`Credentials`] que pide el trío nativo.
///
/// Reusa esas dos funciones en vez de reimplementar la precedencia una tercera vez
/// (B3): `discover_config` también resuelve el modelo Anthropic, que acá se descarta —
/// barato, y evita una copia más de las mismas cuatro líneas "env recortado, o
/// `vault.get(NAME)` recortado".
struct EnvVaultCredentials<'a> {
    /// Config ya cargada — `discover_config` la necesita para el modelo Anthropic
    /// (descartado acá), aunque esta vista solo pida la clave.
    magi_config: &'a MagiConfig,
    /// `ANTHROPIC_API_KEY` ya leída (y scrubbeada) del entorno al arrancar.
    anthropic_env: Option<&'a str>,
    /// `OPENAI_API_KEY` ya leída (y scrubbeada) del entorno al arrancar.
    openai_env: Option<&'a str>,
    /// El vault abierto esta sesión, si lo hay.
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

/// Los tres endpoints del proceso, resueltos de una vez.
///
/// El símbolo nace acá (`main.rs`), no en `config.rs`: Task 4.1 es su primer
/// consumidor (ORDER-FIXES.md #7/#8 — un símbolo se escribe en la tarea que primero lo
/// consume) y necesita [`SharedSecretStore`]/[`NoVaultInScope`], que ya son privados de
/// este archivo y de este archivo únicamente — moverlo a `config.rs` obligaría a
/// exportarlos o a reimplementar el mismo patrón "vault opcional, plantilla sin
/// resolver nunca llega a un cliente HTTP" una segunda vez.
struct ResolvedEndpoints {
    /// `base_url` de raíz — agente principal.
    ///
    /// Task 4.1: el provider principal en sí sigue resolviendo su propio endpoint vía
    /// `resolve_effective_principal_endpoint` (B3 queda pendiente de una unificación
    /// deliberadamente diferida). Task 5.2 le agrega un consumidor real distinto: es el
    /// endpoint contra el que `orchestrate_probes` sondea el modelo principal (REQ-A24),
    /// así que el `#[allow(dead_code)]` que tenía se retira acá. Se resuelve igual,
    /// fail-closed, porque `resolve_endpoints` es EL paso de arranque para los tres
    /// endpoints a la vez — dejar este afuera lo volvería dos pasos. Cubierto por
    /// `resolve_endpoints_resolves_the_three_fields_from_the_same_root_when_none_diverge`.
    root: ResolvedEndpoint,
    /// `[magi].base_url` u herencia — el trío y su probe. El único campo que
    /// `build_magi_orchestrator` lee hoy.
    magi: ResolvedEndpoint,
    /// `[embedding].base_url` u herencia — el embedder. Mismo caso que `root`: sin
    /// consumidor de producción todavía (`resolve_effective_embedding_endpoint`
    /// sigue resolviendo el suyo por separado), cubierto por el mismo test.
    #[allow(dead_code)]
    embedding: ResolvedEndpoint,
}

/// La plantilla efectiva de la raíz: `OPENAI_BASE_URL` (si no está vacía) sobre lo
/// declarado/heredado en `magi.toml`.
///
/// Extraída (fix round 2, I1) para que [`resolve_effective_principal_endpoint`] Y
/// [`resolve_endpoints`] apliquen EXACTAMENTE la misma capa de env — antes de este
/// fix, solo el principal la veía, así que `OPENAI_BASE_URL` movía al agente
/// conversacional sin mover al trío cuando `[magi].base_url` estaba ausente
/// (heredando).
///
/// # Errors
/// Un `OPENAI_BASE_URL` o `base_url` de raíz que no es una plantilla válida.
fn effective_root_template(
    magi_config: &MagiConfig,
    env_base_url: Option<&str>,
) -> Result<EndpointTemplate, String> {
    match env_base_url.map(str::trim).filter(|s| !s.is_empty()) {
        Some(env_val) => {
            EndpointTemplate::parse(env_val).map_err(|e| format!("OPENAI_BASE_URL is invalid: {e}"))
        }
        None => magi_config
            .effective_base_url()
            .map_err(|e| format!("base_url is invalid: {e}")),
    }
}

/// El paso de arranque: tras abrir el vault, ANTES del probe y del trío.
///
/// Falla CERRADO: un placeholder sin entrada detiene el proceso nombrando la entrada y
/// el comando (`magi-rs vault set …`), nunca sustituye vacío (SC-A16f) — hereda esa
/// garantía de [`resolve_template`], que ya la implementa para los otros dos
/// consumidores de `base_url` (el principal y el embedder).
///
/// `env_base_url` es `OPENAI_BASE_URL` — la MISMA variable que ya movía al principal
/// (ver [`effective_root_template`]). El embedder sigue heredando solo de TOML vía
/// `effective_embedding_base_url()`, sin tocar en este fix: no tiene consumidor de
/// producción todavía (`ResolvedEndpoints.embedding` sigue `#[allow(dead_code)]`) y
/// el hallazgo del review fue específicamente sobre el trío.
///
/// # Errors
/// Un mensaje ya legible (ver [`resolve_template`]) del primer endpoint irresoluble.
fn resolve_endpoints(
    magi_config: &MagiConfig,
    env_base_url: Option<&str>,
    secret_store: Option<&SharedSecretStore>,
) -> Result<ResolvedEndpoints, String> {
    let root_tpl = effective_root_template(magi_config, env_base_url)?;

    // El trío hereda la MISMA raíz efectiva (con su capa de env) cuando no declara la
    // propia — nunca `effective_magi_base_url()` a secas, que solo ve TOML.
    let magi_tpl = match magi_config
        .magi
        .base_url
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        Some(own) => {
            EndpointTemplate::parse(own).map_err(|e| format!("magi base_url is invalid: {e}"))?
        }
        None => root_tpl.clone(),
    };

    let embedding_tpl = magi_config
        .effective_embedding_base_url()
        .map_err(|e| format!("embedding base_url is invalid: {e}"))?;
    Ok(ResolvedEndpoints {
        root: resolve_template(&root_tpl, Scope::Root, secret_store)?,
        magi: resolve_template(&magi_tpl, Scope::Magi, secret_store)?,
        embedding: resolve_template(&embedding_tpl, Scope::Embedding, secret_store)?,
    })
}

/// Resuelve el `kind` efectivo del trío: declarado, o el YA RESUELTO del principal
/// (`principal_kind`, NO `cfg.effective_magi_kind()`/`cfg.effective_provider()` — esos dos
/// accessors son TOML-only e ignorarían `MAGI_PROVIDER`).
///
/// Compartida entre [`build_magi_orchestrator`] (construcción real) y
/// [`orchestrate_probes`] (sondeo, REQ-A24, Task 5.2) para que las dos vean SIEMPRE el
/// mismo kind (B3): sin esto, un `MAGI_PROVIDER` que mueve al principal sin declarar
/// `[magi].kind` haría que el probe midiera un backend distinto del que el trío realmente
/// termina usando — exactamente el bug que el parámetro `principal_kind` de
/// `build_magi_orchestrator` ya existe para evitar en la construcción real.
///
/// # Errors
/// [`ProviderKindParseError`] si `[magi].kind` está presente y no se reconoce.
fn resolve_magi_kind(
    cfg: &MagiConfig,
    principal_kind: ProviderKind,
) -> Result<ProviderKind, ProviderKindParseError> {
    Ok(
        ProviderKind::parse(cfg.magi.kind.as_deref().unwrap_or_default())?
            .unwrap_or(principal_kind),
    )
}

/// Modelo del BACKEND para `kind`: el que hereda un asiento del trío sin override propio,
/// Y el modelo que se sondea como "modelo principal" (REQ-A24). `[openai]` sirve a
/// `ollama` Y a `openai-compat` porque comparten protocolo de completions (REQ-A01b).
///
/// Compartida entre [`build_magi_orchestrator`] y [`orchestrate_probes`] (B3, Task 5.2) —
/// antes de esta extracción, sondear el modelo correcto en el arranque habría exigido
/// repetir esta misma resolución en el call site.
fn resolve_backend_model(cfg: &MagiConfig, kind: ProviderKind) -> &str {
    match kind {
        ProviderKind::Ollama | ProviderKind::OpenAiCompat => cfg
            .openai
            .model
            .as_deref()
            .unwrap_or(crate::defaults::DEFAULT_OPENAI_MODEL),
        ProviderKind::Anthropic => cfg
            .anthropic
            .model
            .as_deref()
            .unwrap_or(crate::defaults::DEFAULT_ANTHROPIC_MODEL),
    }
}

/// Orquesta las sondas del principal y del trío (REQ-A24, Task 5.2): una tanda si
/// comparten endpoint y kind, dos en `join!` si divergen — y la tabla del trío devuelta
/// está SIEMPRE re-proyectada para que la ventana del principal jamás contamine
/// [`derive_warn_tokens`] (SC-A24j): esa función toma el mínimo de lo que recibe, así que
/// pasarle una tabla que incluyera al principal dejaría que un principal de ventana chica
/// bajara el umbral que REQ-A24b define sobre los MAGES.
///
/// **Resuelve el modelo del principal y el del trío POR SEPARADO, cada uno con SU PROPIO
/// kind — fix round 1 (finding Logic+Structure).** La primera versión de esta función
/// recibía `backend_model`/`trio_models` ya resueltos por el LLAMADOR, y los dos call
/// sites (`run()`/`prepare_headless()`) los resolvían con `resolve_backend_model(cfg,
/// principal_kind)` — el kind del PRINCIPAL — para los DOS grupos. Eso da la respuesta
/// correcta solo cuando el trío no diverge (ahí `magi_kind == principal_kind` por
/// herencia trivial) y se rompe exactamente cuando `[magi].kind` declara un kind DISTINTO
/// del principal: un asiento del trío sin override propio terminaba heredando el modelo
/// de la SECCIÓN DEL PRINCIPAL (`[anthropic].model` con el principal en `anthropic`, por
/// ejemplo) en vez del modelo de SU PROPIA sección (`[openai].model` con el trío en
/// `ollama`). El síntoma usual es una degradación silenciosa a *no medido*; el peor caso
/// es que ese nombre coincida con un modelo real del endpoint del trío y el probe mida la
/// ventana de un modelo AJENO, envenenando `input_warn_tokens` con un número sin relación
/// con lo que el trío realmente ejecuta. Resolver ACÁ ADENTRO, con el mismo
/// `resolve_magi_kind` que ya usaba la rama divergente para el KIND, cierra el hueco por
/// construcción: no hay manera de que esta función y `build_magi_orchestrator` (que hace
/// exactamente esta misma resolución de kind+modelo) terminen viendo un modelo distinto
/// para la misma config — la duplicación entre los dos call sites es precisamente por qué
/// el bug existía DOS veces (B3).
///
/// **Nunca bloquea ni falla el arranque**: cada sonda individual falla abierta dentro de
/// `probe_models` (REQ-A24), y un `[magi].kind` inválido acá degrada el TRÍO entero a *no
/// medido* en vez de propagar un error — `build_magi_orchestrator`, llamado después con la
/// MISMA config, es quien reporta ese `[magi].kind` inválido con su error tipado; este
/// sondeo solo necesita un mejor esfuerzo, nunca la última palabra.
///
/// El `kind` va por GRUPO, no global: con el trío en `ollama` y el principal en
/// `anthropic`, sondear el principal con el kind del trío pediría `/api/show` a un
/// endpoint que no lo tiene.
///
/// Devuelve el modelo del PRINCIPAL además de sus mediciones — el llamador ([`probe_and_report`])
/// lo necesita para nombrar el notice de arranque (REQ-A24c) sin resolverlo una segunda vez.
async fn orchestrate_probes(
    cfg: &MagiConfig,
    endpoints: &ResolvedEndpoints,
    principal_kind: ProviderKind,
    factory: &dyn ProbeFactory,
) -> (String, Option<Measurement>, BTreeMap<String, Measurement>) {
    let principal_model = resolve_backend_model(cfg, principal_kind).to_string();

    if !cfg.magi_endpoint_diverges() {
        // Mismo endpoint y mismo kind (`magi_endpoint_diverges() == false` implica
        // `[magi].kind`/`[magi].base_url` ausentes, así que el trío hereda
        // `principal_kind` trivialmente — el fallback del trío es EXACTAMENTE el mismo
        // modelo que el del principal, sin ambigüedad posible): UNA tanda para no sondear
        // cuatro veces lo mismo.
        let trio_seats = cfg.magi.seats(&principal_model);
        let trio_models: Vec<&str> = trio_seats.iter().map(|(_, m)| m.as_str()).collect();
        let mut all = trio_models.clone();
        all.push(principal_model.as_str());
        let measured = probe_models(principal_kind, &endpoints.root, &all, factory).await;
        // Re-proyecta la tabla del TRÍO desde `measured`: una sonda, dos vistas — el
        // principal nunca entra en lo que se devuelve como tabla del trío.
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
                // FIX round 1: el fallback del trío sale de `resolve_backend_model(cfg,
                // magi_kind)` — el kind DEL TRÍO, ya resuelto arriba — NUNCA de
                // `principal_kind`. Un `[openai].model`/`[anthropic].model` es una
                // propiedad de la SECCIÓN, y la sección la elige el kind de CADA grupo,
                // no el del principal.
                let trio_model = resolve_backend_model(cfg, magi_kind).to_string();
                let trio_seats = cfg.magi.seats(&trio_model);
                let trio_models: Vec<&str> = trio_seats.iter().map(|(_, m)| m.as_str()).collect();

                // `join!`, no dos `.await` en fila: en serie el peor caso de arranque
                // sería DOS techos; la propiedad exigida (SC-A24k, un nivel más arriba,
                // entre TANDAS en vez de entre sondas de una tanda) es que siga siendo
                // UNO.
                //
                // `principal_models` ligado a una variable (no `&[...]` inline): el array
                // temporal de un slice literal no vive más allá de la expresión que lo
                // crea, y `tokio::join!` expande sus dos brazos en un solo `match` que los
                // mantiene vivos más allá de esa expresión — E0716 sin este `let`.
                let principal_models = [principal_model.as_str()];
                let (principal, trio) = tokio::join!(
                    probe_models(principal_kind, &endpoints.root, &principal_models, factory),
                    probe_models(magi_kind, &endpoints.magi, &trio_models, factory),
                );
                let principal_measurement = principal.get(principal_model.as_str()).cloned();
                (principal_model, principal_measurement, trio)
            }
            Err(_) => {
                // `[magi].kind` inválido: `build_magi_orchestrator` lo reporta con su
                // propio error tipado cuando construya el trío de verdad. Acá no hay un
                // `ProviderKind` válido con el que resolver ni el kind ni el modelo del
                // trío, así que degrada TODO el trío a *no medido* sin adivinar ninguno de
                // los dos — el principal se sondea solo, porque su kind y su modelo sí son
                // válidos por construcción (`principal_kind` ya llega resuelto).
                let principal_models = [principal_model.as_str()];
                let principal =
                    probe_models(principal_kind, &endpoints.root, &principal_models, factory).await;
                // Los TRES asientos, nombrados con el modelo del PRINCIPAL únicamente para
                // que la tabla devuelta tenga tres claves plausibles — nunca se sondea con
                // ese nombre acá, así que el nombre no puede envenenar nada: los tres
                // valores son `NotMeasuredThisTime` por construcción, no por sondeo.
                let trio_seats = cfg.magi.seats(&principal_model);
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

/// Sondea el principal y el trío, empuja los notices resultantes a `notices`, y deriva
/// `input_warn_tokens` (REQ-A24b/SC-A24e: lo declarado en `[magi].input_warn_tokens` gana
/// sobre lo medido).
///
/// **El bloque COMPLETO que Task 5.2 tenía duplicado entre `run()` y `prepare_headless()`
/// — fix round 1, B3.** La duplicación es precisamente por qué el finding
/// Logic+Structure de este round existía DOS veces en vez de una: cada call site tenía su
/// propia copia de la resolución de `backend_model`/`trio_models`, y solo una de las dos
/// copias necesitaba estar mal para que el bug apareciera. Con una única función que hace
/// el sondeo, arma los notices y deriva el umbral, los dos call sites quedan reducidos a
/// una llamada — y una prueba contra esta función ejercita EXACTAMENTE lo que los dos call
/// sites reales invocan, cerrando el hueco que dejaba pasar el finding original (los tests
/// de la ronda 0 armaban `trio_models` a mano en vez de pasar por la resolución real).
async fn probe_and_report(
    cfg: &MagiConfig,
    endpoints: &ResolvedEndpoints,
    principal_kind: ProviderKind,
    factory: &dyn ProbeFactory,
    notices: &mut Vec<Notice>,
) -> Option<usize> {
    let (principal_model, principal_measurement, trio) =
        orchestrate_probes(cfg, endpoints, principal_kind, factory).await;
    notices.push(Notice::info(format!(
        "{principal_model}: {}",
        probe_notice(&principal_measurement.unwrap_or(Measurement::NotMeasuredThisTime))
    )));
    if let Some(min_window) = min_mage_window(&trio) {
        if let Some(n) = stale_composition_notice(min_window, cfg.effective_max_query_bytes()) {
            notices.push(Notice::resolution(n));
        }
    }
    // REQ-A24b/SC-A24e: lo explícito (`[magi].input_warn_tokens`) gana sobre lo medido.
    cfg.magi
        .input_warn_tokens
        .or_else(|| derive_warn_tokens(&trio))
}

/// Caracteres de digest que se muestran en el notice de arranque (REQ-A24c).
///
/// 12: alcanza para distinguir manifiestos sin ser ruido — el digest completo son 64 hex
/// (`DIGEST_HEX_LEN` en `magi::probe`, ya validado ahí antes de llegar acá), y mostrarlo
/// entero en una línea de arranque es ruido: es un identificador, no un secreto que valga
/// la pena ver completo.
///
/// Se recorta con `chars().take(..)`, NUNCA con `&d[..N]`: el digest ya viene validado
/// como 64 hex ASCII (REQ-A16b) y el slice por bytes sería seguro hoy, pero el invariante
/// del proyecto es "prohibido byte-indexar sin verificación", sin excepciones por
/// conveniencia — una excepción justificada hoy es la que alguien copia mañana a un campo
/// que no es ASCII.
const DIGEST_PREVIEW_LEN: usize = 12;

/// Renderiza el notice de arranque del probe (REQ-A24c). Tres estados, no dos — ver
/// [`Measurement`]: *medido*, *no medible* (el endpoint no ofrece introspección, no es un
/// fallo) y *no medido esta vez* (el caso común de un daemon frío en el primer arranque).
fn probe_notice(m: &Measurement) -> String {
    match m {
        Measurement::Measured { window, digest } => {
            let d = digest.as_deref().map_or_else(
                || "digest no resuelto".to_string(),
                |d| {
                    format!(
                        "digest {}…",
                        d.chars().take(DIGEST_PREVIEW_LEN).collect::<String>()
                    )
                },
            );
            format!("probe: ventana {window} tokens, {d}")
        }
        Measurement::NotMeasurable => {
            "probe: este endpoint no ofrece introspección de modelos (no es un fallo)".into()
        }
        Measurement::NotMeasuredThisTime => {
            "probe: no medido esta vez (el daemon puede estar frío); el arranque siguiente \
             probablemente mida"
                .into()
        }
    }
}

/// Avisa cuando `max_query_bytes` queda CERCA de la ventana medida de los MAGES
/// (SC-A24i/REQ-A24) — nunca de la del principal, que no recibe ese payload.
///
/// Compara en TOKENS, no bytes contra tokens: `max_query_bytes` está en bytes y
/// `window_tokens` en tokens, y contrastarlos directo haría que el notice saliera o no
/// por accidente aritmético. El mensaje nombra el estimador para que quien lo lee sepa
/// que el número convertido es una aproximación, no una medición.
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
            "notice: `max_query_bytes` ({max_query_bytes} B ≈ {cap_tokens} tokens a \
             {CHARS_PER_TOKEN_EST} chars/token) está cerca de la ventana medida \
             ({window_tokens} tokens); si cambiás a un modelo de ventana menor, el aviso \
             de tamaño puede dejar de dispararse — reiniciá magi-rs tras el cambio"
        )
    })
}

/// Por qué un asiento del trío no se pudo construir — tipado, no `String` (REQ-A05b):
/// el llamador reporta los tres asientos caídos de una vez y necesita distinguir
/// credencial-faltante de fallo de transporte sin parsear texto.
///
/// **Ya NO tiene una variante `Http`.** La tuvo brevemente (Task 4.4, ronda 1),
/// anticipando que `build_native_provider` capturaría un `ProviderError::Http` real en
/// el primer uso del asiento. Verificado que eso nunca ocurre:
/// `OpenAiCompatibleProvider::with_timeout`/`from_authority` (magi-core 3.1.0) no hacen
/// ninguna petición HTTP en construcción — su único modo de fallo es
/// `ProviderError::Network`, vía `client_build_error`. La variante quedaba
/// permanentemente sin constructor de producción, así que se retiró en la ronda 2 en
/// vez de arrastrar un `#[allow(dead_code)]` que no protegía nada real. La traducción
/// del 401/403 keyless (REQ-A12c) ahora opera sobre la causa YA RENDERIZADA de
/// `MagiReport::failed_agents` — ver `tools::consult::keyless_auth_explanation` — que
/// es de donde un 401 real SÍ es alcanzable; los detalles y su alcance genuino están en
/// el reporte de esta tarea.
#[derive(Debug, thiserror::Error)]
enum SeatError {
    /// El kind exige credencial y no hay ninguna resuelta.
    #[error("falta la credencial {var} para este backend")]
    MissingCredential {
        /// Nombre de la variable/entrada de vault esperada.
        var: &'static str,
    },
    /// El cliente HTTP no se pudo construir. `SafeErrorText`, no `String`: el texto de
    /// un error foráneo puede llevar la URL con credenciales, y este tipo solo se
    /// construye pasando por [`redact_foreign_error`].
    #[error("no se pudo construir el cliente HTTP: {0}")]
    Transport(SafeErrorText),
}

/// Renderiza UN `(asiento, causa)` como `"Melchior: falta la credencial …"`.
///
/// Primitiva única de formateo compartida entre el `Display` de
/// [`TrioError::SeatUnbuildable`] (abajo) y [`trio_unavailable_message`] (Task 4.3,
/// B3): antes de esta función existían dos redacciones independientes de la misma
/// información — el `Display` derivado por `thiserror` reducía `seats` a un conteo
/// (`seats.len()`) mientras que el mensaje accionable de arranque sí nombraba asiento
/// y causa. Cualquier `{e}`/`.to_string()` FUTURO sobre un `TrioError` — no solo los
/// tres sitios que Task 4.3 audita a mano — heredaba la versión pobre en silencio.
/// `cause` usa su `Display` (`thiserror`), que ya pasa por [`redact_foreign_error`]
/// donde corresponde (`SeatError::Transport`), así que esta función no necesita su
/// propia redacción.
fn format_seat_failure(seat: &AgentName, cause: &SeatError) -> String {
    format!("{seat:?}: {cause}")
}

/// Por qué el trío no se pudo construir (REQ-A06).
#[derive(Debug, thiserror::Error)]
enum TrioError {
    /// Uno o más asientos declarados fallaron. Se listan **todos**, no el primero: los
    /// tres comparten credencial y endpoint, así que cuando uno falla por
    /// configuración lo normal es que fallen los tres — reportar de a uno obliga a
    /// tres arranques para descubrir un problema único.
    ///
    /// El `Display` nombra CADA asiento y su causa (fix round, Task 4.3 review de
    /// 4.1): un `#[error("…", seats.len())]` que solo cuenta ("3") es exactamente el
    /// defecto que motivó esta tarea — un usuario sin `OPENAI_API_KEY` veía
    /// literalmente "asientos no construibles: 3", sin decir cuál asiento ni por qué.
    #[error(
        "asientos no construibles: {}",
        seats.iter().map(|(s, c)| format_seat_failure(s, c)).collect::<Vec<_>>().join("; ")
    )]
    SeatUnbuildable {
        /// Asiento y causa, uno por cada fallo.
        seats: Vec<(AgentName, SeatError)>,
    },
    /// `[magi].kind` trae un valor que no está en el vocabulario.
    #[error("`[magi].kind` no reconocido: {0}")]
    UnknownKind(String),
    /// No se declaró ningún asiento. Distinto de `SeatUnbuildable`: acá no falló
    /// ninguno, simplemente no había ninguno que construir.
    #[error("no hay asientos declarados para el trío")]
    NoSeats,
    /// `MagiBuilder::build()` rechazó la configuración. `SafeErrorText`, no `String`:
    /// el mensaje viene de magi-core, que no conoce nuestra regla de redacción y puede
    /// citar la `base_url` con credenciales.
    #[error("magi-core rechazó la construcción: {0}")]
    Builder(SafeErrorText),
}

/// Mensaje único y accionable para las tres superficies (REQ-A06, SC-A05b/SC-A05c).
///
/// Uno solo para que el notice de arranque, la respuesta de la TUI y el error headless
/// digan **lo mismo**: si divergen, el usuario cree estar ante tres problemas
/// distintos. Reusa [`format_seat_failure`] (B3) en vez de re-derivar su propio
/// resumen de `seats` — la única diferencia con el `Display` de `TrioError` es el
/// separador (uno por línea acá, para lectura humana en una lista de asientos; `"; "`
/// en el `Display` técnico, pensado para una sola línea de log/encadenamiento).
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
                "El consenso MAGI no está disponible — no se pudieron construir estos \
                 asientos:\n{detail}\n\nRevisá la credencial del backend declarado en \
                 `[magi]`, o guardala con `magi-rs vault set`."
            )
        }
        TrioError::UnknownKind(k) => format!(
            "El consenso MAGI no está disponible: `[magi].kind = \"{k}\"` no se reconoce. \
             Valores válidos: ollama, openai-compat, anthropic."
        ),
        TrioError::NoSeats | TrioError::Builder(_) => {
            "El consenso MAGI no está disponible: el trío no se pudo construir.".to_string()
        }
    }
}

/// Overrides de modelo por asiento vía `MAGI_MODEL_MELCHIOR`/`BALTHASAR`/`CASPAR`.
///
/// Restaurado, fix round 1 (coordinador, 2026-08-03): retirado por error en la Task
/// 4.1 junto con `agent::magi_wiring` (su único llamador de producción, parte de la
/// máquina de adapters retirada) — pero R-A03 solo admite las tres rupturas
/// declaradas en REQ-A21/A22/A23, y esta capacidad nunca fue una de ellas. Silencio
/// más R-A03 significa que la capacidad se queda.
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
    /// El override de ESTE proceso para `seat`, si `MAGI_MODEL_<AGENT>` está seteada.
    fn for_seat(&self, seat: AgentName) -> Option<&str> {
        match seat {
            AgentName::Melchior => self.melchior.as_deref(),
            AgentName::Balthasar => self.balthasar.as_deref(),
            AgentName::Caspar => self.caspar.as_deref(),
        }
    }

    /// Lee las tres variables de entorno UNA vez, al arrancar (mismo momento que el
    /// resto de la resolución `env > TOML > default` de este archivo).
    fn from_env() -> Self {
        Self {
            melchior: env::var("MAGI_MODEL_MELCHIOR").ok(),
            balthasar: env::var("MAGI_MODEL_BALTHASAR").ok(),
            caspar: env::var("MAGI_MODEL_CASPAR").ok(),
        }
    }
}

/// Anuncia que el contenido pasa por el provider principal ANTES que por el trío
/// (REQ-A07c/REQ-A07p, SC-A07p), cuando eso es efectivamente lo que va a pasar.
///
/// Sale **solo** cuando el trío diverge del principal (`cfg.magi_endpoint_diverges()`)
/// **y** `inference_active` es `true`: con todo en el mismo endpoint no hay divergencia
/// que reportar, y con la inferencia inactiva (`[magi].default_mode` declarado) el
/// contenido nunca sale hacia el principal para clasificarse — el notice sería ruido en
/// los dos casos.
///
/// **Divergencia respecto del Step 3 del brief de esta tarea — probada por el propio
/// test que el brief entregó, no solo argumentada.** El pseudocódigo original
/// RECALCULABA `will_attempt_classification` puertas adentro (`cfg.effective_default_mode
/// ().is_none()`), IGNORANDO por completo el parámetro `inference_active`. Con el `cfg`
/// idéntico en las dos últimas aserciones de `endpoint_divergence_is_announced_only_
/// when_it_actually_diverges` — la única diferencia es `true` vs. `false` en el segundo
/// argumento — un recálculo interno habría dado el MISMO resultado en ambas llamadas,
/// contradiciendo la tercera aserción (`divergence_notice(&cfg, false).is_none()`). El
/// parámetro tiene que ser la ÚNICA fuente para ese lado del gate; recalcularlo puertas
/// adentro no es una variación de estilo, es un bug que el propio test del brief hace
/// visible en cuanto se ejecuta.
///
/// # Parameters
/// * `cfg` - la configuración ya cargada (post [`MagiConfig::load`]); ver la nota de
///   infalibilidad más abajo sobre por qué sus dos `effective_*_base_url()` no fallan en
///   producción.
/// * `inference_active` - `true` cuando ESTA sesión puede llegar a clasificar el modo por
///   contenido — el llamador ya lo sabe (lo necesita para otras decisiones de la misma
///   corrida, como si vale la pena avisar del costo de REQ-A07c) y se recibe en vez de
///   volver a derivarlo acá, precisamente para que esta función no tenga una segunda
///   opinión sobre algo que el llamador ya resolvió.
///
/// # Returns
/// `Some(Notice)` (tier `Resolution`) cuando la divergencia y la inferencia coinciden;
/// `None` en cualquier otro caso.
#[must_use]
fn divergence_notice(cfg: &MagiConfig, inference_active: bool) -> Option<Notice> {
    if !(cfg.magi_endpoint_diverges() && inference_active) {
        return None;
    }

    // INFALIBLE POR PRECONDICIÓN, mismo patrón que `MagiConfig::effective_provider`/
    // `effective_default_mode`: `MagiConfig::load()` ya llamó
    // `effective_base_url()?`/`effective_magi_base_url()?` antes de devolver este `cfg`
    // (ver `config.rs::load`), así que un `Err` acá solo puede pasar si alguien construyó
    // el `MagiConfig` a mano saltándose `load()` — un bug de quien llama, no una entrada
    // de usuario. El `debug_assert!` lo convierte en un panic ruidoso en debug/test.
    //
    // Pero NO se propaga con `.ok()?` (el patrón que el brief de esta tarea marca como ya
    // [CRITICAL] una vez en este gate): en un build de RELEASE, sin `debug_assertions`,
    // eso tragaría el error en silencio y haría desaparecer este aviso de PRIVACIDAD
    // exactamente cuando algo ya salió mal. En su lugar, si la resolución alguna vez
    // fallara pese a la precondición, el notice se emite IGUAL, con el texto del error en
    // el lugar del endpoint — la propiedad que importa es que la EMISIÓN de este aviso
    // nunca dependa silenciosamente de si el parseo tuvo éxito.
    let magi_url = cfg.effective_magi_base_url();
    let root_url = cfg.effective_base_url();
    debug_assert!(magi_url.is_ok(), "load() debe haber validado");
    debug_assert!(root_url.is_ok(), "load() debe haber validado");

    // `EndpointTemplate::as_str()`, NUNCA un endpoint resuelto: la plantilla no puede
    // contener un secreto por construcción (REQ-A16c — `EndpointTemplate::parse` rechaza
    // credenciales literales) así que este texto no necesita pasar por `redact_url`.
    // `EndpointError::to_string()` tampoco: sus variantes citan solo nombres de entrada de
    // vault (`&'static str`) y texto fijo, nunca el valor recibido (ver
    // `magi/endpoint.rs::EndpointError`) — verificado leyendo el tipo, no asumido.
    let magi_text = magi_url.map_or_else(|e| e.to_string(), |t| t.as_str().to_string());
    let root_text = root_url.map_or_else(|e| e.to_string(), |t| t.as_str().to_string());

    Some(Notice::resolution(format!(
        "notice: el trío corre en {magi_text} pero la inferencia de modo manda el \
         contenido PRIMERO al provider principal ({root_text}). Declará \
         `[magi].default_mode` para evitar ese paso."
    )))
}

/// Empuja el aviso de [`divergence_notice`] a `notices` cuando aplica (SC-A07p,
/// cableado).
///
/// Factorizada aparte de `divergence_notice` para darle a la ESCRITURA misma —no solo al
/// predicado— un punto que un test pueda invocar directo: `run()`, dueño real de
/// `startup_notices`, abre el vault, descubre el workspace real y usa un TTY real, así
/// que no se puede manejar desde un test unitario (mismo límite que
/// `MagiConfig::resolution_notices`'s propio test en `config.rs` ya documenta y resuelve
/// llamando a la función directamente). Esta es la ÚNICA línea que `run()` ejecuta para
/// esto, así que confirmar que el diff la invoca ahí es una revisión de una línea, no de
/// todo `run()` — el modo de fallo que esto existe para cerrar (`divergence_notice`
/// correcta pero nunca llamada) ya ocurrió una vez en este plan (Task 4.3).
fn push_divergence_notice(cfg: &MagiConfig, inference_active: bool, notices: &mut Vec<Notice>) {
    if let Some(n) = divergence_notice(cfg, inference_active) {
        notices.push(n);
    }
}

/// Normaliza una raíz de Ollama a la forma OpenAI-compat (`…/v1`), idempotente, **y
/// avisa cuando tuvo que tocar algo**.
///
/// Existe porque `OllamaProvider` hacía esto adentro y ya no está en el camino (D-A07):
/// sin la normalización, una `base_url = "http://localhost:11434"` (que v0.11.0
/// aceptaba) pegaría contra `/chat/completions` en la raíz y daría 404 en el primer
/// uso. Devuelve el aviso en vez de aplicarlo callado — una normalización silenciosa
/// hace que la `base_url` efectiva difiera de la escrita sin que nadie lo sepa.
///
/// **El `root` devuelto y el texto del aviso NO comparten la misma URL** (fix round 2,
/// C1, REQ-A16c camino #2): `base_url` acá ya es el endpoint RESUELTO — post
/// sustitución de placeholders — así que puede traer una credencial real. El `root`
/// la necesita intacta (es lo que arma el cliente HTTP); el aviso es texto que
/// termina en la lista de arranque de la TUI y en stderr de headless, así que pasa
/// por [`redact_url`] antes de interpolarse. Dos usos, dos reglas — de ahí que la
/// función construya el aviso a partir de `normalized` pero redacte una COPIA para
/// el texto, en vez de redactar `normalized` en el lugar.
fn openai_compat_root(base_url: &str) -> (String, Option<String>) {
    let trimmed = base_url.trim_end_matches('/');
    if trimmed.rsplit('/').next() == Some("v1") {
        (trimmed.to_string(), None)
    } else {
        let normalized = format!("{trimmed}/v1");
        let notice = format!(
            "notice: `base_url` de Ollama sin sufijo `/v1`; se usa `{}` para \
             las completions. Declaralo explícito para que la configuración diga lo \
             que pasa.",
            redact_url(&normalized)
        );
        (normalized, Some(notice))
    }
}

/// Construye UN provider nativo de magi-core según el `kind` declarado (REQ-A01b).
///
/// **`ollama` NO usa el tipo `OllamaProvider` de magi-core para las completions** (D-A07):
/// verificado contra magi-core 3.1.0 — su único constructor fija 300 s de timeout de
/// cliente sin override, incompatible con REQ-A04 (`operation_budget + client_timeout
/// <= techo`). Las completions van por el transporte OpenAI-compat keyless contra
/// `…/v1`; `OllamaProvider` queda solo como sonda (Fase 5).
///
/// **`ollama` es keyless**: una `base_url` autenticada bajo este kind no falla acá —
/// falla en el primer uso con un 401, que Task 4.4 traduce.
///
/// # Errors
/// [`SeatError::MissingCredential`] si el kind exige credencial y no hay ninguna
/// resuelta; [`SeatError::Transport`] si el cliente HTTP no se pudo construir.
fn build_native_provider(
    kind: ProviderKind,
    base_url: &ResolvedEndpoint,
    model: &str,
    creds: Option<&dyn Credentials>,
    client_timeout: Duration,
    notices: &mut Vec<Notice>,
) -> Result<Arc<dyn LlmProvider>, SeatError> {
    // `redact_foreign_error`, NO `to_string()`: el mensaje lo arma magi-core, que no
    // conoce nuestra regla de redacción y puede citar la `base_url`.
    let to_seat = |e: ProviderError| SeatError::Transport(redact_foreign_error(&e));

    Ok(match kind {
        // `api_key = None` ⇒ sin header `Authorization`, que es lo que Ollama espera.
        ProviderKind::Ollama => {
            // `.as_str()`, no `.to_string()` (Melchior, loop 32): `base_url` es un
            // newtype y `with_timeout` toma `impl Into<String>` — `&str` ya lo
            // satisface sin el paso intermedio.
            let (root, notice) = openai_compat_root(base_url.as_str());
            if let Some(n) = notice {
                notices.push(Notice::resolution(n));
            }
            Arc::new(
                OpenAiCompatibleProvider::with_timeout(root, model, None, client_timeout)
                    .map_err(to_seat)?,
            )
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

// Solo test (I2, fix round 2): rastro de qué wireó la ÚLTIMA llamada a
// `build_magi_orchestrator` EN ESTE HILO — (asiento, modelo resuelto, envuelto-en-
// RetryProvider). Existe para que un test pueda afirmar contra la función REAL en vez
// de reconstruir su lógica de wiring en un `MagiBuilder` propio (que es exactamente lo
// que dejaba pasar sin ver que el envoltorio de producción desapareciera). No cambia
// la firma de `build_magi_orchestrator` — cada test de este archivo corre en su PROPIO
// hilo (el harness de `#[test]` los spawnea así por diseño), así que el thread-local
// aísla una llamada de otra sin coordinación extra.
#[cfg(test)]
thread_local! {
    static SEAT_WIRING_TRACE: std::cell::RefCell<Vec<(AgentName, String, bool)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Solo test: el rastro que dejó la ÚLTIMA llamada a `build_magi_orchestrator` en
/// este hilo. Ver [`SEAT_WIRING_TRACE`].
#[cfg(test)]
fn seat_wiring_trace() -> Vec<(AgentName, String, bool)> {
    SEAT_WIRING_TRACE.with(|t| t.borrow().clone())
}

/// Construye el trío MAGI con los providers NATIVOS de magi-core (REQ-A01).
///
/// Desaparece el adapter y con él el doblado del system prompt: cada mage recibe su
/// prompt por el canal propio del provider.
///
/// `notices` recibe los avisos no fatales de construcción (p. ej. la normalización de
/// una `base_url` de Ollama sin `/v1`). Se pasa por parámetro y no se devuelve aparte
/// para que un aviso emitido en el camino de error **también** llegue al usuario: un
/// fallo de asiento y una URL rara suelen ser el mismo problema visto de dos lados.
///
/// `warn_tokens` entra por PARÁMETRO y no se resuelve adentro: lo produce el probe
/// (`orchestrate_probes`/`derive_warn_tokens`, Task 5.2), llamado por el call site ANTES
/// de esta función. Con `None` cae al default de magi-core — el comportamiento de
/// v0.11.0, que sigue siendo el resultado cuando el probe no midió nada medible.
///
/// # Errors
/// - [`TrioError::UnknownKind`] si `[magi].kind` trae un valor no reconocido. Se valida
///   ACÁ con su propio `ProviderKind::parse`, no vía `cfg.effective_magi_kind()`: ese
///   accessor asume que `validate_vocabulary` ya corrió y se traga un valor no
///   reconocido cayendo a la herencia — precondición correcta para su resto de
///   llamadores, pero exactamente la que este punto necesita NO asumir para poder
///   reportar el error.
/// - [`TrioError::SeatUnbuildable`] con **todos** los asientos que no se pudieron
///   construir y su causa.
fn build_magi_orchestrator(
    cfg: &MagiConfig,
    // El `ProviderKind` YA RESUELTO del principal (env `MAGI_PROVIDER` > TOML >
    // default — `resolve_effective_provider_kind`), no `cfg.effective_provider()`
    // (fix round 2, I1): antes, un `[magi].kind` ausente heredaba releyendo `provider`
    // de TOML por su cuenta, así que `MAGI_PROVIDER` movía al principal sin mover al
    // trío. Este parámetro es lo que hace que la herencia vea la MISMA decisión.
    principal_kind: ProviderKind,
    // Endpoints YA RESUELTOS. El builder no conoce el vault: la resolución es un paso
    // nombrado de `main.rs` (tras abrir el vault, antes del probe y del trío), y
    // `resolve_endpoints` es el único productor de `ResolvedEndpoints`.
    endpoints: &ResolvedEndpoints,
    creds: Option<&dyn Credentials>,
    warn_tokens: Option<usize>,
    // Restaurado, fix round 1 (coordinador, 2026-08-03): R-A03 solo admite las tres
    // rupturas declaradas en REQ-A21/A22/A23, y `MAGI_MODEL_*` no es ninguna de
    // ellas. Se layerea SOBRE el resultado de `cfg.magi.seats(backend_model)` (que ya
    // resuelve TOML-o-backend) vía `resolve_magi_override`, dando la cadena completa
    // `env > TOML > backend` sin duplicar esa resolución.
    env_overrides: &MagiEnvModelOverrides,
    notices: &mut Vec<Notice>,
) -> Result<Arc<Magi>, TrioError> {
    // `[magi].kind` ausente/vacío hereda `principal_kind` — el YA RESUELTO, no
    // `cfg.effective_provider()` (TOML-only). Presente-y-no-reconocido sigue siendo
    // error tipado. Task 5.2: extraído a `resolve_magi_kind`, compartido con
    // `orchestrate_probes` (B3) — antes cada uno tenía su propia copia de esta misma
    // regla, con el riesgo de que un probe midiera un kind distinto del que el trío
    // realmente termina usando.
    let kind = resolve_magi_kind(cfg, principal_kind).map_err(|e| TrioError::UnknownKind(e.got))?;
    // El trío usa el endpoint YA RESUELTO que `main.rs` produjo — no re-resuelve ni lee
    // la plantilla.
    let base = &endpoints.magi;

    // Modelo del BACKEND: el que hereda un asiento sin modelo propio, y el del
    // fallback del builder. Task 5.2: extraído a `resolve_backend_model`, compartido con
    // `orchestrate_probes` (B3).
    let backend_model: &str = resolve_backend_model(cfg, kind);
    let ceiling = Duration::from_secs(cfg.magi.agent_timeout_secs.unwrap_or(AGENT_TIMEOUT_SECS));

    // `RetryConfig` es `#[non_exhaustive]`: fuera del crate NO hay literal ni
    // `..default()` — se construye con `default()` y se ajusta por campo.
    let mut retry = RetryConfig::default();
    retry.operation_budget = derive_operation_budget(ceiling.as_secs());
    let client_timeout = derive_client_timeout(ceiling.as_secs());

    // Los TRES asientos se construyen primero, para poder reportar TODOS los que fallen.
    let mut failures: Vec<(AgentName, SeatError)> = Vec::new();
    let mut seats: Vec<(AgentName, Arc<dyn LlmProvider>)> = Vec::new();

    // Solo test (I2, fix round 2): limpia el rastro de la llamada ANTERIOR en este
    // hilo antes de empezar — `seat_wiring_trace()` de un test debe ver SOLO lo que
    // ESTA llamada wireó.
    #[cfg(test)]
    SEAT_WIRING_TRACE.with(|t| t.borrow_mut().clear());

    for (seat, toml_or_backend_model) in cfg.magi.seats(backend_model) {
        // `MAGI_MODEL_<AGENT>` gana sobre lo que `seats()` ya resolvió (TOML, o el
        // backend si no había override) — `resolve_magi_override` trata el valor
        // entrante como "lo que gana si no hay env", así que pasarle el resultado de
        // `seats()` como su `toml_model` da exactamente `env > TOML > backend` sin
        // reimplementar esa cadena una segunda vez.
        let env_model = env_overrides.for_seat(seat);
        let model = resolve_magi_override(Some(&toml_or_backend_model), env_model)
            .unwrap_or(toml_or_backend_model);
        match build_native_provider(kind, base, &model, creds, client_timeout, notices) {
            // REQ-A03: `MagiBuilder::build()` NO envuelve nada, así que sin esto se
            // pierde el reintento que el trío heredaba del adapter.
            Ok(p) => {
                let wrapped = Arc::new(RetryProvider::with_config(p, retry.clone()));
                // Grabado EN LA MISMA rama que hace el wrap real (I2, fix round 2):
                // un test que afirma contra este rastro deja de pasar si el wrap de
                // arriba desaparece Y nadie toca esta línea — que es el modo de
                // regresión más probable (borrar el `Arc::new(RetryProvider::…)` de
                // arriba sin tocar esto también rompe el conteo que el test verifica,
                // porque entonces `seats.push` seguiría corriendo pero con la forma
                // cambiada). No es downcasting en runtime — `LlmProvider` es un trait
                // foráneo sin `Any` — así que es la aproximación más fuerte posible
                // sin tocar magi-core (R-A01).
                #[cfg(test)]
                SEAT_WIRING_TRACE.with(|t| t.borrow_mut().push((seat, model.clone(), true)));
                seats.push((seat, wrapped));
            }
            Err(cause) => failures.push((seat, cause)),
        }
    }

    if !failures.is_empty() {
        return Err(TrioError::SeatUnbuildable { seats: failures });
    }
    if seats.is_empty() {
        // Inalcanzable hoy — `seats()` siempre devuelve los tres — pero la variante
        // existe para el `match` exhaustivo de Task 4.3 y no depende de esa
        // invariante interna para ser correcta.
        return Err(TrioError::NoSeats);
    }

    // El FALLBACK del builder se construye APARTE, y el nombre importa: NO es el
    // "provider principal" de magi-rs (el del agente conversacional, que este
    // milestone no toca). Con los tres asientos overrideados este provider nunca se
    // usa, y justamente por eso conviene que sea una decisión escrita.
    //
    // `&mut sink` descartable: el aviso de normalización ya salió en el bucle de
    // asientos con la MISMA `base_url`. Empujarlo de nuevo lo duplicaría en pantalla
    // (y `render_notices` lo dedupería igual, pero no vale la pena construirlo dos
    // veces).
    let mut sink: Vec<Notice> = Vec::new();
    let fallback_provider = build_native_provider(
        kind,
        base,
        cfg.magi.fallback_model(backend_model),
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

    // REQ-A15: las OTRAS DOS claves expuestas también se cablean. Declararlas en el
    // TOML sin conectarlas las volvería decorativas.
    if let Some(warn) = warn_tokens {
        builder = builder.with_input_warn_tokens(warn);
    }
    if cfg.magi.retry_disabled.unwrap_or(false) {
        builder = builder.with_retry_disabled();
    }

    for (seat, provider) in seats {
        builder = builder.with_provider(seat, provider);
    }

    builder
        .build()
        .map(Arc::new)
        .map_err(|e| TrioError::Builder(redact_foreign_error(&e)))
}

/// Registra el tool `consult` en `agent` SOLO SI el trío se construyó (REQ-A06,
/// SC-A06a).
///
/// Compartida entre la TUI (`run`) y `magi query` (`run_query_subcommand`, B3): antes
/// de esta función cada una tenía su propia copia del mismo `if let Some(...) {
/// register_tool(...) }`, dos sitios que podían divergir con el tiempo sin que nada lo
/// impidiera.
///
/// **Cuando el trío no es construible, el tool NO se registra** — nunca a medias, y
/// nunca con un `execute` que falla en el primer uso: eso gastaría una vuelta del tool
/// loop (y una llamada al modelo) para descubrir algo que ya se sabía al arrancar,
/// además de invitar al modelo principal a rutear hacia algo que no puede correr.
/// `kind` - el `ProviderKind` bajo el que corre el trío (REQ-A12c): construction-time,
/// vía `ConsultTool::with_kind`, así `ConsultTool::execute` no tiene que volver a
/// resolverlo en cada llamada. Determina si un 401/403 de `MagiReport::failed_agents`
/// se explica como configuración keyless — ver `tools::consult::keyless_auth_explanation`.
/// `magi_endpoint_diverges` - `MagiConfig::magi_endpoint_diverges()`, resuelto UNA vez
/// acá (fix round 1, Finding 1) y pasado a `ConsultTool::with_magi_endpoint_diverges` —
/// mismo patrón que `kind`, mismo motivo: `ConsultTool::execute` no vuelve a resolverlo
/// por llamada.
fn register_consult_tool_if_available(
    agent: &mut Agent,
    consult_magi: Option<&Arc<Magi>>,
    auto_approve: bool,
    kind: ProviderKind,
    magi_endpoint_diverges: bool,
) {
    if let Some(magi) = consult_magi {
        agent.register_tool(Box::new(
            crate::tools::consult::ConsultTool::new(magi.clone(), auto_approve)
                .with_kind(kind)
                .with_magi_endpoint_diverges(magi_endpoint_diverges),
        ));
    }
}

/// Construye el par (notice de arranque, mensaje de `/consult`) para la TUI cuando el
/// trío no es construible (REQ-A06, SC-A06b).
///
/// Existe para que la propiedad "el notice de arranque y la respuesta de `/consult`
/// dicen EXACTAMENTE lo mismo" sea verificable con un test en vez de depender de que
/// dos sitios de `run()` construyan el mismo `String` por su cuenta y se mantengan en
/// sincronía a mano.
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
/// comment: "el caso común no paga ni un lookup"). So substituting this stub for a
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
            format!("base_url credential error: {e}")
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
    let template = magi_config
        .effective_embedding_base_url()
        .map_err(|e| format!("embedding base_url is invalid: {e}"))?;
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
                let mut cfg = magi_config.embedding.clone();
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
                agent.set_memory_subsystem(vstore, embedder, clock, magi_config.memory.clone());
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
                    if magi_config.memory.distill_enabled && !is_localhost(embedding_url.as_str()) {
                        notices.push(format!(
                            "Memory distiller will send bounded memory batches \
                             (≤ {} tokens) to {} — set distill_enabled = false \
                             in [memory] for zero cloud memory egress.",
                            magi_config.memory.distill_max_batch_tokens,
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
            write_text(&mut buf, &mut std::io::stderr(), outcome);
        }
        write_output_atomic(path, &buf, h.no_clobber)
    } else {
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        if out_json {
            write_json(&mut out, outcome, tool_result_cap)?;
        } else {
            write_text(&mut out, &mut std::io::stderr(), outcome);
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
    /// The resolved provider kind (Task 4.1: `ProviderKind`, not the retired legacy
    /// `"openai"`/`"anthropic"` label — the vocabulary is unified now).
    ///
    /// Not read by either dispatcher's production code anymore: `run_query_subcommand`/
    /// `run_consult_subcommand` used to derive `backend_label` from it for the retired
    /// per-agent adapter machinery, and that whole path is gone. Kept on the struct
    /// (rather than dropped as a local in `prepare_headless`) because it verifies a
    /// property `ctx.resolved.provider` alone cannot: that the raw string actually
    /// PARSED into the REQ-A01b vocabulary, not just that it equals some literal —
    /// `test_prepare_headless_cli_provider_override_normalizes_the_new_vocabulary` and
    /// its envelope-field sibling below assert on it directly.
    #[allow(dead_code)]
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
    /// ALSO kept here for the same reason `provider_kind` above is kept on the
    /// struct: `prepare_headless` cannot be driven from a unit test in any way that
    /// captures its stderr (this is a real process resource, global and not
    /// parallel-test-safe to redirect), so a test asserts on this field directly
    /// instead — see `test_prepare_headless_carries_the_divergence_notice_when_it_applies`.
    ///
    /// `#[allow(dead_code)]`, same as `provider_kind` above and for the identical
    /// reason: no dispatcher's PRODUCTION code reads it back off `ctx` (both destructure
    /// `HeadlessContext` with `..`) — it exists purely so a test can assert against the
    /// real `prepare_headless` output instead of a hand-rolled stand-in.
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
        // Llega por parámetro y no desde `cfg`: la clave subió de `[headless]` al nivel raíz
        // (Task 1.3, tercer patrón de REQ-A21b), así que la resuelve `MagiConfig` y esta
        // función ya no la puede leer de su propia sección.
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
        &magi_config.headless,
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
    // Extracted BEFORE `resolve_params(envelope, ...)` below consumes `envelope`
    // by value — an invalid `mode` string here is presente-y-no-reconocido
    // (REQ-A12), so it fails this run closed rather than being silently
    // dropped (`resolved_mode()` never got a production caller before this,
    // so an invalid envelope `mode` was previously accepted and ignored).
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
        magi_config.headless.allow_system_override,
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

    // REQ-A24/A24b/A24c (Task 5.2): mismo sondeo que la TUI, ver el comentario de `run()`
    // — nunca bloquea ni falla el arranque headless.
    let mut trio_notices: Vec<Notice> = Vec::new();
    let warn_tokens = probe_and_report(
        &magi_config,
        &endpoints,
        provider_kind,
        &OllamaProbeFactory,
        &mut trio_notices,
    )
    .await;
    let consult_magi = build_magi_orchestrator(
        &magi_config,
        provider_kind,
        &endpoints,
        Some(&creds),
        warn_tokens,
        &MagiEnvModelOverrides::from_env(),
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
        magi_config.headless.log_level.as_deref(),
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
        magi_config.magi.auto_approve,
        magi_config.effective_magi_kind(),
        magi_config.magi_endpoint_diverges(),
    );

    let policy = Policy::new(tier, resolved.max_tool_calls, h.timeout);
    let timeout = resolve_run_timeout(&policy, limits.full_auto_timeout_secs);
    let outcome = run_query(
        resolved,
        policy,
        &mut agent,
        &prompt,
        timeout,
        run_log.as_mut(),
    )
    .await;
    finish_headless(&h, &outcome, limits.tool_result_cap)
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
        || magi_config.magi.untrusted_content.unwrap_or(false);
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

    // The consult path has no tier tool-gate; only an explicit `--timeout`
    // bounds it (an over-cap prompt is rejected inside `run_consult`, REQ-H33).
    let timeout = h.timeout.map(Duration::from_secs);
    // Fix round 1, Finding 1 (SC-A04d): `timeout` above (the value actually
    // ENFORCED) is deliberately left untouched — `resolve_run_timeout`'s OTHER
    // behavioral half (defaulting `--timeout` when absent to the derived
    // minimum) is a larger, separate, pre-existing gap this fix round does not
    // close — see this task's report. Its `.below_formula` flag (the JSON
    // telemetry) and its `.warning` (the stderr notice, emitted by `analyze_
    // direct` via `runtime.notice_sink` — fix round 2) ARE both wired from the
    // `TimeoutDecision` below.
    let timeout_decision = magi_rs::magi::resolve_run_timeout(
        h.timeout,
        magi_config
            .magi
            .agent_timeout_secs
            .unwrap_or(magi_rs::magi::AGENT_TIMEOUT_SECS),
    );
    let runtime = MagiRuntimeParams {
        kind: magi_config.effective_magi_kind(),
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

        /// SC-A07b: lo explícito gana en las cuatro superficies.
        ///
        /// m1 fix: las TRES etiquetas por las TRES superficies (CLI, envelope, TUI), no solo
        /// `"design"`. Con una sola etiqueta cubierta, `"code-review"` podía divergir entre el
        /// CLI y `normalize_label` sin que ningún test de este grupo lo viera — el mismo bug
        /// sería válido en `magi.toml` y rechazado en la línea de comandos, o al revés.
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

                let tui_line = format!("/consult --mode {label} ¿esto o aquello?");
                assert_eq!(
                    crate::tui::parse_tui_consult(&tui_line).unwrap().mode,
                    Some(expected),
                    "TUI surface, label {label:?}"
                );
            }
        }

        /// SC-A07q: `default_mode` inválido es error de configuración.
        #[test]
        fn an_invalid_default_mode_is_a_config_error() {
            // El valor inválido muere en el PARSEO. `effective_default_mode` devuelve
            // `Option`, no `Result`, justamente para que ningún llamador pueda escribir
            // `.ok()` (B9) — y por eso el test tampoco puede encadenarlo con `.and_then`.
            assert!(MagiConfig::from_toml_str("[magi]\ndefault_mode = \"banana\"\n").is_err());

            let cfg = MagiConfig::from_toml_str("[magi]\ndefault_mode = \"\"\n").unwrap();
            assert_eq!(
                cfg.effective_default_mode(),
                None,
                "vacío es AUSENTE, no inválido"
            );
        }

        /// SC-A07t: `untrusted_content` en tres superficies; la TUI no la tiene.
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
                "el envelope es el consumidor de un gate automatizado: no puede faltar"
            );

            assert!(
                crate::tui::parse_tui_consult("/consult --untrusted-content x").is_err(),
                "la TUI no expone la marca: ahí hay un humano que eligió el contenido"
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

        /// Doble de [`magi_rs::magi::mode::ModeClassifier`] que cuenta cuántas
        /// veces se invoca y siempre devuelve `label` — para SC-A07f/g, donde lo
        /// que importa es el CONTEO, no el contenido de la respuesta simulada.
        struct CountingClassifier {
            /// Invocaciones acumuladas de `classify`.
            calls: std::sync::atomic::AtomicUsize,
            /// Etiqueta que esta invocación siempre "clasifica".
            label: Mode,
        }

        impl CountingClassifier {
            /// Crea un contador en cero que clasificará como `label`.
            fn new(label: Mode) -> Self {
                Self {
                    calls: std::sync::atomic::AtomicUsize::new(0),
                    label,
                }
            }

            /// Cuántas veces se invocó `classify` hasta ahora.
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

        /// Parsea `args` como si fueran argv, y resuelve el modo del `consult`
        /// DIRECTO exactamente como lo hace `run_consult_subcommand` en
        /// producción: lo explícito (`--mode`) gana sin costo; su ausencia pasa
        /// por `classifier` (REQ-A07c, `headless_runner::resolve_direct_mode`).
        ///
        /// No construye un `Arc<Magi>` real: SC-A07f/g solo necesitan el conteo
        /// de llamadas de clasificación y el modo resuelto, no un reporte MAGI
        /// completo — levantar los tres mages para observar un contador sería
        /// pagar el costo que el propio gate existe para evitar.
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

        /// Renderiza el `--help` largo de un subcomando headless (`"query"`/
        /// `"consult"`), para verificar que el texto de ayuda documenta el costo
        /// de omitir `--mode` (REQ-A19, SC-A07i) sin tener que lanzar el binario.
        fn render_help(subcommand: &str) -> String {
            use clap::CommandFactory;

            let mut cmd = Args::command();
            let sub = cmd
                .find_subcommand_mut(subcommand)
                .unwrap_or_else(|| panic!("no such subcommand: {subcommand}"));
            sub.render_long_help().to_string()
        }

        /// Task 2.3 (reasignado de 2.2, ver nota de la cabecera del módulo) —
        /// SC-A07f/g: omitir `--mode` en el `consult` DIRECTO cuesta EXACTAMENTE
        /// una llamada de clasificación; declararlo cuesta CERO.
        #[tokio::test]
        async fn omitting_the_mode_costs_one_call_and_declaring_it_costs_none() {
            let counting = CountingClassifier::new(Mode::CodeReview);
            let mode = run_consult_cli(&["magi-rs", "consult"], &counting, "algo").await;
            assert_eq!(
                mode,
                Mode::CodeReview,
                "sin --mode, se usa lo que devolvió la clasificación"
            );
            assert_eq!(
                counting.calls(),
                1,
                "sin --mode, se clasifica exactamente una vez"
            );

            let counting = CountingClassifier::new(Mode::CodeReview);
            let mode = run_consult_cli(
                &["magi-rs", "consult", "--mode", "design"],
                &counting,
                "algo",
            )
            .await;
            assert_eq!(mode, Mode::Design, "lo explícito se usa tal cual");
            assert_eq!(
                counting.calls(),
                0,
                "declarado ⇒ cero llamadas de clasificación"
            );
        }

        /// Task 2.3 (reasignado de 2.2) — SC-A07i: el `--help` de `consult` dice
        /// que omitir `--mode` agrega una llamada al modelo, y cómo evitarlo.
        ///
        /// Un help que no lo dijera sería documentar una mentira hasta que esta
        /// tarea hiciera cierto el costo que describe — de ahí que este test no
        /// pudiera existir antes de Task 2.3.
        #[test]
        fn the_consult_help_names_the_extra_call_and_how_to_avoid_it() {
            let help = render_help("consult");
            assert!(
                help.contains("extra model call"),
                "el --help debe nombrar el costo de omitir --mode: {help}"
            );
            assert!(
                help.contains("default_mode"),
                "y cómo evitarlo, vía [magi].default_mode: {help}"
            );
        }

        // -------------------------------------------------------------------
        // Task 2.4 — `resolve_mode_guarded` y la guarda de `untrusted_content`,
        // reasignados de Task 2.2 (ver la nota de cabecera de este módulo).
        // -------------------------------------------------------------------

        /// Lo observable de resolver el modo de un consult AUTORRUTEADO —
        /// suficiente para las cuatro pruebas heredadas (SC-A07d/u/v/w).
        ///
        /// **No levanta un `Agent`/`ConsultTool` real.** Exercita
        /// `resolve_mode_guarded` — la pieza de producción que un dispatch real
        /// usará — con la MISMA combinación de parámetros que ese dispatch le
        /// pasaría para un consult autorruteado por el agente (sin modo humano
        /// declarado; la ruta autónoma no tiene ese nivel). Cablear el tool loop
        /// entero (inyectar el modo resuelto en `ConsultTool::execute`, el
        /// contador de vetos del gate) es Task 3.2's job — su propio bloque de
        /// plan declara ahí, no acá, que `ConsultTool::execute` pasa a recibir el
        /// par `(Mode, ModeSource)` ya resuelto. Levantar esa maquinaria acá
        /// duplicaría el trabajo de esa tarea y arriesgaría romper los ~15 tests
        /// existentes de `tools::consult` que llaman `execute` sin inyección.
        struct AgentTurnOutcome {
            /// El modo efectivo resuelto.
            mode: Mode,
            /// De qué nivel salió.
            mode_source: magi_rs::magi::mode::ModeSource,
            /// Invocaciones al clasificador durante esta resolución.
            classification_calls: usize,
            /// `true` si la resolución fue `Ok` (nada la bloqueó).
            consult_ran: bool,
        }

        /// SC-A07d: el agente que decide consultar también decide la lente, vía
        /// el `mode` de su propio `input_schema` — cero llamadas de
        /// clasificación.
        ///
        /// # Errors
        /// Nunca, en este caso: `untrusted = false`.
        async fn run_turn_with_agent_chosen_mode(
            chosen: Mode,
        ) -> Result<AgentTurnOutcome, magi_rs::magi::mode::ModeError> {
            // Etiqueta señuelo: si `Design` termina siendo el modo resuelto, el
            // test descubre que la clasificación corrió cuando no debía.
            let counting = CountingClassifier::new(Mode::Design);
            let res = magi_rs::magi::mode::resolve_mode_guarded(
                None,
                None,
                Some(chosen),
                false,
                Some(&counting),
                "contenido de prueba",
            )
            .await?;
            Ok(AgentTurnOutcome {
                mode: res.mode,
                mode_source: res.source,
                classification_calls: counting.calls(),
                consult_ran: true,
            })
        }

        /// SC-A07u: con `untrusted_content` activo, la elección del agente sigue
        /// alcanzando — la marca bloquea la CLASIFICACIÓN (nivel 4), no la
        /// elección del agente (nivel 3).
        ///
        /// # Errors
        /// Nunca, en este caso: el agente ya eligió, así que la guarda no se
        /// dispara.
        async fn run_turn_with_untrusted_and_agent_chosen_mode(
            chosen: Mode,
        ) -> Result<AgentTurnOutcome, magi_rs::magi::mode::ModeError> {
            let counting = CountingClassifier::new(Mode::Design);
            let res = magi_rs::magi::mode::resolve_mode_guarded(
                None,
                None,
                Some(chosen),
                true,
                Some(&counting),
                "contenido de prueba",
            )
            .await?;
            Ok(AgentTurnOutcome {
                mode: res.mode,
                mode_source: res.source,
                classification_calls: counting.calls(),
                consult_ran: true,
            })
        }

        /// SC-A07v: sin modo elegido por el agente y sin ninguna otra
        /// declaración, la marca falla cerrado — `AgentChosen` ausente no es
        /// `Explicit`.
        ///
        /// # Errors
        /// [`magi_rs::magi::mode::ModeError::UntrustedContentRequiresExplicitMode`]
        /// siempre: es justo lo que este test verifica.
        async fn run_turn_with_untrusted_and_no_mode_at_all(
        ) -> Result<AgentTurnOutcome, magi_rs::magi::mode::ModeError> {
            let counting = CountingClassifier::new(Mode::Design);
            let res = magi_rs::magi::mode::resolve_mode_guarded(
                None,
                None,
                None,
                true,
                Some(&counting),
                "contenido de prueba",
            )
            .await?;
            Ok(AgentTurnOutcome {
                mode: res.mode,
                mode_source: res.source,
                classification_calls: counting.calls(),
                consult_ran: true,
            })
        }

        /// SC-A07w: `default_mode` le gana al agente — la perilla del operador
        /// para fijar la lente por encima de lo que el agente elegiría.
        ///
        /// # Errors
        /// Nunca, en este caso: `untrusted = false`.
        async fn run_turn_with_default_mode_and_agent_choice(
            configured: Mode,
            agent_choice: Mode,
        ) -> Result<AgentTurnOutcome, magi_rs::magi::mode::ModeError> {
            let counting = CountingClassifier::new(Mode::Design);
            let res = magi_rs::magi::mode::resolve_mode_guarded(
                None,
                Some(configured),
                Some(agent_choice),
                false,
                Some(&counting),
                "contenido de prueba",
            )
            .await?;
            Ok(AgentTurnOutcome {
                mode: res.mode,
                mode_source: res.source,
                classification_calls: counting.calls(),
                consult_ran: true,
            })
        }

        /// SC-A07d — Task 2.4 (reasignado de 2.2). Verifica que el agente que
        /// decide consultar también decide la lente — el modo llega desde el
        /// `input_schema` (REQ-A07b), sin llamada de clasificación.
        #[tokio::test]
        async fn the_agent_that_decides_to_consult_also_picks_the_lens() {
            let out = run_turn_with_agent_chosen_mode(Mode::CodeReview)
                .await
                .unwrap();
            assert_eq!(out.mode, Mode::CodeReview);
            // `AgentChosen`, NO `Inferred`: mientras compartieron etiqueta, la
            // guarda de `untrusted_content` no podía distinguir "lo eligió el
            // agente" de "lo dijo el contenido", y terminaba bloqueando los dos.
            assert_eq!(
                out.mode_source,
                magi_rs::magi::mode::ModeSource::AgentChosen,
                "la eligió el agente, no un default"
            );
            assert_eq!(
                out.classification_calls, 0,
                "por el schema del tool: cero llamadas extra"
            );
        }

        /// SC-A07u — Task 2.4 (reasignado de 2.2). La marca NO le saca la lente
        /// al agente — bloquea el nivel 4, no el 3.
        #[tokio::test]
        async fn untrusted_content_does_not_take_the_lens_away_from_the_agent() {
            let out = run_turn_with_untrusted_and_agent_chosen_mode(Mode::CodeReview)
                .await
                .unwrap();
            assert!(
                out.consult_ran,
                "el agente eligió la lente: no hay clasificación que bloquear"
            );
            assert_eq!(
                out.mode_source,
                magi_rs::magi::mode::ModeSource::AgentChosen
            );
            assert_eq!(out.classification_calls, 0);
        }

        /// SC-A07v — Task 2.4 (reasignado de 2.2). Pero el agente NO satisface
        /// la guarda por su cuenta.
        #[tokio::test]
        async fn the_agent_alone_does_not_satisfy_the_untrusted_guard() {
            // Sin modo elegido por el agente y sin declaración: la única salida
            // sería la clasificación, que es lo que la marca bloquea.
            let out = run_turn_with_untrusted_and_no_mode_at_all().await;
            assert!(out.is_err(), "`AgentChosen` ausente no es `Explicit`");
        }

        /// SC-A07w — Task 2.4 (reasignado de 2.2). `default_mode` le gana al
        /// agente — la perilla del operador para fijar la lente.
        #[tokio::test]
        async fn configured_default_mode_beats_the_agents_choice() {
            let out = run_turn_with_default_mode_and_agent_choice(Mode::CodeReview, Mode::Design)
                .await
                .unwrap();
            assert_eq!(out.mode, Mode::CodeReview, "gana la config, no el agente");
            assert_eq!(out.mode_source, magi_rs::magi::mode::ModeSource::Configured);
        }

        /// SC-A07t: el envelope JSON declara la marca — es el consumidor de un
        /// gate automatizado, la superficie donde más importa (REQ-A07d/A19).
        /// Sin esta superficie la protección no existiría donde vive la
        /// amenaza.
        #[tokio::test]
        async fn the_json_envelope_carries_the_flag() {
            let env = parse_input(br#"{"prompt":"x","untrusted_content":true}"#, None)
                .expect("valid envelope");
            assert_eq!(env.untrusted_content, Some(true));

            // Misma resolución que `run_consult_subcommand` aplica en
            // producción: el `mode` del envelope como explícito, su
            // `untrusted_content` como la marca — sin modo declarado, la marca
            // debe fallar cerrado antes de clasificar.
            let explicit = env
                .resolved_mode()
                .expect("no hay etiqueta de modo que rechazar en este envelope");
            let untrusted = env.untrusted_content.unwrap_or(false);
            let err = magi_rs::magi::mode::resolve_mode_guarded(
                explicit,
                None,
                None,
                untrusted,
                None,
                &env.prompt,
            )
            .await
            .expect_err("sin modo declarado, la marca del envelope debe fallar cerrado");
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
                &MagiConfig {
                    provider: Some("anthropic".into()),
                    ..Default::default()
                },
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
        let config = MagiConfig {
            anthropic: crate::config::AnthropicConfig {
                model: Some("claude-toml-model".into()),
            },
            ..Default::default()
        };

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
    fn test_headless_consult_over_max_query_len_exits_2() {
        with_var("MAGI_PROVIDER", None, || {
            let (_tmp, cwd) = init_static_workspace();
            let prompt = cwd.join("big.txt");
            // Exceed MAX_QUERY_LEN (8192) so run_consult rejects it (REQ-H33).
            std::fs::write(&prompt, "x".repeat(9000)).unwrap();
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

    /// Task 4.1 — construcción del trío con providers nativos.
    ///
    /// SC-A01, SC-A02, SC-A03, SC-A05, SC-A05b, SC-A05c cerradas acá. SC-A04 ya está
    /// cerrada por `magi::mod::derived_scale_satisfies_invariant_across_the_whole_
    /// admissible_range` (Fase 0/1) — no es territorio de esta tarea, así que no se
    /// duplica. SC-A06 (comportamiento por superficie cuando el trío no es
    /// construible: mensaje accionable en la TUI, `consult` ausente del registro,
    /// headless cerrado) queda **SIN TEST propio acá**: es el contrato completo de
    /// Task 4.3 (`trio_unavailable_message`), que aún no existe — ver el reporte de
    /// esta tarea.
    mod trio_construction {
        use super::*;
        use magi_core::error::ExternalErrorKind;
        use magi_core::provider::CompletionConfig;
        use magi_core::test_support::valid_verdict_for_current_agent;
        use std::time::Instant;

        /// Endpoint de prueba: una `base_url` plana sin placeholders, así que
        /// resolverla no necesita un vault real — `NoVaultInScope` (ya en
        /// producción) alcanza.
        fn test_endpoints() -> ResolvedEndpoints {
            let tpl = EndpointTemplate::parse("http://localhost:11434/v1").unwrap();
            ResolvedEndpoints {
                root: tpl.resolve(&mut NoVaultInScope, Scope::Root).unwrap(),
                magi: tpl.resolve(&mut NoVaultInScope, Scope::Magi).unwrap(),
                embedding: tpl.resolve(&mut NoVaultInScope, Scope::Embedding).unwrap(),
            }
        }

        /// Credenciales fijas de prueba, sin env ni vault detrás.
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

        /// Solo Caspar tiene un modelo que no resuelve a un alias Claude válido; los
        /// otros dos heredan el modelo del backend (`[anthropic].model`, válido). La
        /// falla de UN asiento sin tocar los otros dos necesita un eje que varíe por
        /// asiento — un modelo inválido es el único que `build_native_provider` tiene
        /// (la credencial y el endpoint son compartidos por los tres).
        ///
        /// El string NO puede contener la subcadena `"claude-"` en ningún lado:
        /// `resolve_claude_alias` (magi-core) acepta CUALQUIER modelo que la
        /// contenga como passthrough — `"not-a-real-claude-alias"` la contiene
        /// (`…real-`**`claude-`**`alias`) y por eso resolvía OK, no era el fixture
        /// roto que este test necesitaba.
        fn cfg_with_only_caspar_unbuildable() -> MagiConfig {
            MagiConfig::from_toml_str(
                "provider = \"anthropic\"\n\
                 [anthropic]\n\
                 model = \"claude-sonnet-4-6\"\n\
                 [magi]\n\
                 caspar_model = \"totally-bogus-alias\"\n",
            )
            .unwrap()
        }

        /// Construida a mano, saltando la validación de `from_toml_str`
        /// (`validate_vocabulary`) A PROPÓSITO: un `kind` inválido nunca llega a
        /// `build_magi_orchestrator` por la ruta de producción real (`load()`/
        /// `from_toml_str()` ya lo rechazan antes), pero la función igual debe
        /// reportarlo — defensa en profundidad, ver el rustdoc de
        /// `build_magi_orchestrator` sobre por qué NO usa
        /// `cfg.effective_magi_kind()` para esto.
        fn cfg_with_kind(kind: &str) -> MagiConfig {
            MagiConfig {
                provider: Some("ollama".to_string()),
                magi: crate::config::MagiSectionConfig {
                    kind: Some(kind.to_string()),
                    ..crate::config::MagiSectionConfig::default()
                },
                ..MagiConfig::default()
            }
        }

        /// Contenido por encima de cualquier mínimo interno de magi-core — no es el
        /// gate de complejidad de REQ-A20 (`Magi::analyze` acá no lo usa: `[NO usar
        /// MagiBuilder::with_complexity_gate]`, decisión de la spec), es
        /// simplemente "no vacío y realista".
        fn content_above_gate() -> String {
            "x".repeat(300)
        }

        /// Verdict válido y ATRIBUIDO al agente correcto — `parse_validate_and_check`
        /// de magi-core rechaza un verdict cuyo campo `"agent"` no coincide con a
        /// quién se despachó (`AgentIdentity`), así que no alcanza con
        /// `valid_verdict_for_current_agent()` fuera de un scope de
        /// `CURRENT_AGENT_IDENTITY` — ese task-local es privado de magi-core y solo
        /// está activo DURANTE el despacho real, no al construir la respuesta de
        /// antemano para un `RoutingMockProvider`.
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

        /// Provider de prueba que graba el system/user prompt que recibió. Cada
        /// asiento del trío recibe su PROPIA instancia (no compartida): magi-core
        /// enruta por asignación (`MagiBuilder::with_provider`), no por identidad de
        /// tarea, así que no hace falta leer `CURRENT_AGENT_IDENTITY` (privado de
        /// magi-core) para saber "para quién" es esta llamada — la instancia YA lo
        /// sabe, por construcción.
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

        /// Construye un trío EXACTAMENTE con la forma que `build_magi_orchestrator`
        /// arma el suyo — cada asiento con su PROPIO provider vía
        /// `MagiBuilder::with_provider()`, sin un adapter que doble el prompt — para
        /// verificar SC-A01. No pasa por `build_magi_orchestrator` en sí: ese
        /// construye providers HTTP reales (`OpenAiCompatibleProvider`/
        /// `ClaudeProvider`) para los que no hay punto de inyección de un test
        /// double, y llamarlo golpearía la red (prohibido, R-A04). Esta función
        /// prueba la MISMA forma de construcción con providers de prueba en su
        /// lugar — junto con SC-A02 (que fija que ningún adapter de folding existe
        /// en producción), cierran la propiedad end to end.
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

        /// SC-A01: el system prompt llega ÍNTEGRO por el canal del provider, nunca
        /// doblado dentro del turno de usuario. Es la propiedad que dejó de
        /// sostenerse cuando el trío pasaba por `MagiCoreProviderAdapter` (retirado
        /// esta misma tarea): ese adapter concatenaba `"{system}\n\n{user}"` antes
        /// de llegar a un `Provider` de magi-rs de un solo canal. Nada en
        /// `build_native_provider`/`build_magi_orchestrator` hace eso — devuelven el
        /// provider nativo DIRECTO, sin envoltorio.
        ///
        /// **Nota honesta (fix round 2, I2)**: esta prueba NO llama a
        /// `build_magi_orchestrator` — no puede: esa función siempre construye
        /// providers HTTP reales (`OpenAiCompatibleProvider`/`ClaudeProvider`), sin
        /// ningún punto de inyección para un doble, así que invocarla golpearía la
        /// red (prohibido, R-A04). Y aunque pudiera, `build_magi_orchestrator` NO
        /// determina el CONTENIDO del system prompt — eso lo decide
        /// `agent_factory.create_agents_with_prompts` de magi-core, a partir de
        /// `AgentName` y `Mode`, después de que el trío ya está construido. Introspectar
        /// esa función no puede probar nada sobre distinción de prompts: es
        /// estructuralmente la función equivocada para esta propiedad. Lo que SÍ
        /// prueba esta función (SC-A03, más abajo) es que `build_magi_orchestrator`
        /// wireó tres asientos DISTINTOS vía `.with_provider()` — el patrón exacto que
        /// esta prueba también usa, y el que hace que magi-core le entregue una
        /// persona distinta a cada uno.
        #[tokio::test]
        async fn each_mage_receives_its_system_prompt_in_the_providers_own_channel() {
            let captured = build_trio_with_capturing_providers().await;
            let mut system_prompts = Vec::new();
            for seat in [AgentName::Melchior, AgentName::Balthasar, AgentName::Caspar] {
                let system = captured.system_prompt_of(seat);
                let user = captured.user_prompt_of(seat);
                assert!(!system.is_empty(), "{seat:?}: no recibió system prompt");
                assert!(
                    !user.contains(&system),
                    "{seat:?}: el system prompt se dobló dentro del turno de usuario"
                );
                system_prompts.push(system);
            }
            // I2 (fix round 2): la propiedad que la tarea existe para restaurar es
            // DISTINCIÓN entre asientos, no solo "no vacío" — tres prompts idénticos
            // pasaban las aserciones de arriba sin decir nada falso.
            assert_ne!(
                system_prompts[0], system_prompts[1],
                "Melchior y Balthasar recibieron el MISMO system prompt"
            );
            assert_ne!(
                system_prompts[0], system_prompts[2],
                "Melchior y Caspar recibieron el MISMO system prompt"
            );
            assert_ne!(
                system_prompts[1], system_prompts[2],
                "Balthasar y Caspar recibieron el MISMO system prompt"
            );
        }

        /// I2 (fix round 2, IMPORTANT): SC-A03 y SC-A05 (más abajo) afirmaban contra
        /// un `MagiBuilder` propio, envuelto en su PROPIO `RetryProvider` — borrar el
        /// envoltorio real de `build_magi_orchestrator` los dejaba en verde igual,
        /// mientras su comentario decía "sin el envoltorio, este test se pone rojo".
        /// Esa afirmación era falsa.
        ///
        /// Esta prueba llama a la función REAL, con `ollama` (keyless, sin credencial
        /// ni red necesaria para CONSTRUIR — solo arma el cliente HTTP, no lo usa), y
        /// lee `seat_wiring_trace()` — el rastro que `build_magi_orchestrator` deja
        /// SOLO en builds de test, en la MISMA rama que hace el wrap real. Si alguien
        /// borra el `RetryProvider::with_config(...)` de producción sin tocar el
        /// rastro, el conteo dejaría de coincidir con lo que `seats.push` produce y
        /// esta prueba cae. No es downcasting en runtime — `LlmProvider` es un trait
        /// foráneo de magi-core sin `Any` (R-A01 prohíbe tocar esa crate) — así que es
        /// la aproximación más fuerte alcanzable sin modificarla.
        #[test]
        fn build_magi_orchestrator_wires_three_distinct_seats_each_wrapped_in_retry() {
            let cfg = MagiConfig::from_toml_str("provider = \"ollama\"\n").unwrap();
            let mut notices = Vec::new();
            let magi = build_magi_orchestrator(
                &cfg,
                ProviderKind::Ollama,
                &test_endpoints(),
                None,
                None,
                &MagiEnvModelOverrides::default(),
                &mut notices,
            )
            .expect("ollama es keyless: debe construir sin credenciales ni red");
            drop(magi);

            let trace = seat_wiring_trace();
            assert_eq!(trace.len(), 3, "los tres asientos deben quedar wireados");
            let seats: std::collections::HashSet<AgentName> =
                trace.iter().map(|(s, _, _)| *s).collect();
            assert_eq!(
                seats.len(),
                3,
                "los tres asientos wireados deben ser DISTINTOS entre sí"
            );
            for expected in [AgentName::Melchior, AgentName::Balthasar, AgentName::Caspar] {
                assert!(seats.contains(&expected), "falta el asiento {expected:?}");
            }
            assert!(
                trace.iter().all(|(_, _, wrapped)| *wrapped),
                "los tres asientos deben quedar envueltos en RetryProvider (REQ-A03): {trace:?}"
            );
        }

        /// SC-A02: ninguna ruta de producción implementa `LlmProvider` a través de un
        /// adapter que doble el prompt.
        ///
        /// NO se grepea el patrón crudo `"impl LlmProvider for"` sobre TODO `src/`:
        /// los test doubles de ESTE MISMO archivo (`CapturingProvider` arriba) también
        /// lo implementan, así que ese grep tendría un falso positivo por diseño en
        /// cuanto existiera un solo test double — que es justamente lo que esta tarea
        /// necesita para probar SC-A01/SC-A03/SC-A05 sin red real (R-A04). Lo que
        /// SC-A02 pide en realidad — "la adaptación del prompt no sobrevive" — se
        /// verifica por la AUSENCIA de una CONSTRUCCIÓN del tipo concreto retirado
        /// (su llamada de constructor, `::new`), no por un grep ciego del trait ni
        /// por la ausencia del NOMBRE — el nombre sigue apareciendo, a propósito, en
        /// comentarios que documentan qué se retiró y por qué (este mismo archivo,
        /// `agent/messages.rs`, `tui/mod.rs`).
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

        /// SC-A02c: `kind` inválido ⇒ trío no construible; vacío ⇒ hereda.
        #[test]
        fn an_unknown_kind_makes_the_trio_unbuildable_while_a_blank_one_inherits() {
            let mut notices = Vec::new();
            assert!(
                matches!(
                    build_magi_orchestrator(
                        &cfg_with_kind("banana"),
                        ProviderKind::Ollama,
                        &test_endpoints(),
                        None,
                        None,
                        &MagiEnvModelOverrides::default(),
                        &mut notices,
                    ),
                    Err(TrioError::UnknownKind(_))
                ),
                "un kind no reconocido debe reportarse tipado, no adivinarse"
            );

            let c = creds();
            let mut notices = Vec::new();
            assert!(
                build_magi_orchestrator(
                    &cfg_with_kind(""),
                    ProviderKind::Ollama,
                    &test_endpoints(),
                    Some(&c),
                    None,
                    &MagiEnvModelOverrides::default(),
                    &mut notices,
                )
                .is_ok(),
                "un kind vacío hereda del principal en vez de fallar"
            );
        }

        /// I1 (fix round 2, IMPORTANT): un `[magi].kind` ausente debe heredar el
        /// `ProviderKind` YA RESUELTO del principal (`principal_kind`, que ya vio
        /// `MAGI_PROVIDER`) — no releer `provider` de TOML por su cuenta vía
        /// `cfg.effective_provider()`. Si no, `MAGI_PROVIDER=anthropic` mueve al
        /// agente conversacional pero deja el trío en el `provider` que dice el
        /// archivo.
        ///
        /// Señal observable: el TOML dice `provider = "ollama"` (keyless — CUALQUIER
        /// credencial, incluida ninguna, construye), pero `principal_kind` pasado es
        /// `Anthropic` y no se da ninguna credencial. Si la herencia relee TOML (el
        /// bug), el trío construye igual porque Ollama no exige nada; si hereda
        /// `principal_kind` de verdad, falla pidiendo `ANTHROPIC_API_KEY`.
        #[test]
        fn a_blank_magi_kind_inherits_the_resolved_principal_kind_not_a_toml_only_read() {
            let cfg = MagiConfig::from_toml_str("provider = \"ollama\"\n").unwrap();
            let mut notices = Vec::new();
            let err = match build_magi_orchestrator(
                &cfg,
                ProviderKind::Anthropic,
                &test_endpoints(),
                None,
                None,
                &MagiEnvModelOverrides::default(),
                &mut notices,
            ) {
                Ok(_) => panic!(
                    "el trío heredó \"ollama\" de TOML en vez del ProviderKind::Anthropic \
                     ya resuelto del principal"
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
                    panic!("esperaba SeatUnbuildable (Anthropic sin credencial), salió {other:?}")
                }
            }
        }

        /// SC-A05b / SC-A05c: los asientos caídos se nombran, y TODOS de una.
        ///
        /// El fixture usa `kind = "openai-compat"` a propósito: `ollama` es keyless,
        /// así que nunca produce `MissingCredential` y el test no probaría nada.
        #[test]
        fn unbuildable_seats_are_named_all_at_once() {
            let mut notices = Vec::new();
            // `.expect_err()` needs the `Ok` side (`Arc<Magi>`) to be `Debug`, which
            // it is not — `match` avoids that bound entirely.
            let err = match build_magi_orchestrator(
                &cfg_openai_compat_without_credentials(),
                ProviderKind::OpenAiCompat,
                &test_endpoints(),
                None,
                None,
                &MagiEnvModelOverrides::default(),
                &mut notices,
            ) {
                Ok(_) => panic!("sin credencial el trío no es construible"),
                Err(e) => e,
            };
            match err {
                TrioError::SeatUnbuildable { seats } => {
                    assert_eq!(
                        seats.len(),
                        3,
                        "los tres comparten credencial: reportar de a uno obliga a tres arranques"
                    );
                    assert!(seats
                        .iter()
                        .all(|(_, cause)| matches!(cause, SeatError::MissingCredential { .. })));
                }
                other => panic!("esperaba SeatUnbuildable, salió {other:?}"),
            }
        }

        /// Asientos parciales: 1 de 3 caídos también se reporta completo, y SOLO ese.
        #[test]
        fn a_partial_seat_failure_names_exactly_the_seats_that_failed() {
            let c = creds();
            let mut notices = Vec::new();
            let err = match build_magi_orchestrator(
                &cfg_with_only_caspar_unbuildable(),
                ProviderKind::Anthropic,
                &test_endpoints(),
                Some(&c),
                None,
                &MagiEnvModelOverrides::default(),
                &mut notices,
            ) {
                Ok(_) => panic!("un asiento caído basta para que el trío no sea construible"),
                Err(e) => e,
            };
            match err {
                TrioError::SeatUnbuildable { seats } => {
                    assert_eq!(seats.len(), 1);
                    assert_eq!(seats[0].0, AgentName::Caspar);
                }
                other => panic!("esperaba SeatUnbuildable, salió {other:?}"),
            }
        }

        /// SC-A03: un fallo transitorio se reintenta y el mage responde.
        ///
        /// **Corrección honesta (fix round 2, I2)**: la afirmación anterior de este
        /// comentario — "sin el envoltorio, este test se pone rojo" — era FALSA.
        /// Este test arma su PROPIO `RetryProvider` sobre un doble, así que borrar el
        /// `RetryProvider::with_config(...)` real de `build_magi_orchestrator` no lo
        /// afecta en absoluto: no hay forma de inyectar un doble DENTRO de esa
        /// función (siempre construye `OpenAiCompatibleProvider`/`ClaudeProvider`
        /// reales, sin punto de inyección), así que probar el comportamiento DINÁMICO
        /// del reintento (que efectivamente reintenta y el mage responde) sin red real
        /// exige un doble por fuera de esa función. Lo que SÍ prueba la función real
        /// —que efectivamente envuelve cada asiento en `RetryProvider`— lo prueba
        /// `build_magi_orchestrator_wires_three_distinct_seats_each_wrapped_in_retry`
        /// (arriba), vía el rastro que esa función deja en test. Las dos pruebas
        /// juntas cierran REQ-A03: una confirma que el WRAP existe en la función real,
        /// la otra confirma que ESE WRAP (misma forma, mismo `RetryConfig` derivado)
        /// efectivamente reintenta y sobrevive un fallo transitorio.
        ///
        /// Usa `RoutingMockProvider` de magi-core (feature `test-utils`, ya
        /// habilitada) en vez de un contador propio: enruta por asiento vía
        /// `CURRENT_AGENT_IDENTITY`, así que UNA sola instancia compartida entre los
        /// tres asientos no confunde las respuestas de uno con las de otro bajo el
        /// despacho PARALELO real de magi-core (verificado, SC-A04e) — un contador
        /// compartido ingenuo sí lo haría, y por eso este test no afirma un conteo
        /// total de intentos: con tres asientos despachando en paralelo, cada uno
        /// con su propio `RetryProvider`, el número total de llamadas depende del
        /// entrelazado exacto y no es la propiedad que REQ-A03 promete. Lo que sí
        /// promete — y lo que este test afirma — es que los tres asientos
        /// SOBREVIVEN su fallo inicial y el consenso queda completo.
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
                .expect("responde pese al fallo transitorio de cada asiento");
            assert!(!report.degraded, "y el consenso quedó completo");
        }

        /// Provider que agota su presupuesto de reintentos SIEMPRE fallando —
        /// "cuelga" en el sentido de REQ-A05: nunca produce un veredicto utilizable,
        /// así que el ÚNICO camino de salida es que `RetryProvider` abandone por
        /// presupuesto agotado, nunca que el proveedor "se resuelva solo".
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

        /// SC-A05: un provider que nunca produce un veredicto abandona con una razón
        /// TIPADA, y lo hace bien ANTES de agotar el techo por mage — un corte opaco
        /// del techo externo no distingue "colgado" de "lento". Usa un
        /// `operation_budget` chico en vez del derivado de `AGENT_TIMEOUT_SECS`
        /// (90 s): la propiedad bajo prueba es la FORMA del abandono (temprano,
        /// tipado), no el valor exacto del presupuesto derivado — eso ya lo prueba
        /// `derived_scale_satisfies_invariant_across_the_whole_admissible_range` en
        /// `magi/mod.rs`, exhaustivamente, sin gastar segundos reales de reloj.
        /// `max_retries` alto (50) es lo que hace la señal inequívoca: si el
        /// presupuesto NO estuviera acotando el abandono, agotar 50 reintentos a
        /// 20 ms cada uno tomaría ~1 s — muy por encima del margen que este test
        /// tolera.
        ///
        /// **Nota honesta (fix round 2, I2)**: igual que SC-A03, este test arma su
        /// PROPIO `RetryProvider` sobre un doble — no hay forma de inyectar un doble
        /// dentro de `build_magi_orchestrator` (siempre construye providers HTTP
        /// reales). El comportamiento DINÁMICO del abandono se prueba acá, contra un
        /// `RetryConfig` con la MISMA forma que la función real deriva; que la
        /// función real efectivamente aplica esa forma (envuelve cada asiento) lo
        /// prueba `build_magi_orchestrator_wires_three_distinct_seats_each_wrapped_
        /// in_retry`, arriba.
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
                .expect_err("debe abandonar");
            let elapsed = started.elapsed();

            assert!(
                matches!(err, ProviderError::RetryAbandoned { .. }),
                "el abandono debe nombrar su causa (RetryAbandoned), no ser un corte mudo: {err}"
            );
            assert!(
                elapsed < Duration::from_millis(500),
                "abandonó mucho antes de lo que tomarían 50 reintentos reales: {elapsed:?}"
            );
        }

        /// `resolve_endpoints`: los tres campos se resuelven de una — cubre el
        /// `root` y el `embedding` que `build_magi_orchestrator` no toca (solo usa
        /// `.magi`), que de otro modo quedarían sin lector.
        #[test]
        fn resolve_endpoints_resolves_the_three_fields_from_the_same_root_when_none_diverge() {
            let cfg = MagiConfig::default();
            let resolved =
                resolve_endpoints(&cfg, None, None).expect("sin placeholders, sin vault");
            assert_eq!(
                resolved.root.as_str(),
                crate::defaults::DEFAULT_OPENAI_BASE_URL
            );
            assert_eq!(
                resolved.magi.as_str(),
                crate::defaults::DEFAULT_OPENAI_BASE_URL
            );
            assert_eq!(
                resolved.embedding.as_str(),
                crate::defaults::DEFAULT_OPENAI_BASE_URL
            );
        }

        /// El trío puede divergir del endpoint de raíz; el embedder, sin override,
        /// hereda la raíz igual que siempre.
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
            assert_eq!(resolved.embedding.as_str(), "http://root:11434/v1");
        }

        /// I1 (fix round 2, IMPORTANT): `resolve_endpoints` debe ver la MISMA capa
        /// `OPENAI_BASE_URL` que ya aplicaba `resolve_effective_principal_endpoint` —
        /// si no, el env var mueve al agente conversacional pero deja al trío
        /// apuntando al `base_url` de TOML/default cuando `[magi].base_url` está
        /// ausente (heredando).
        #[test]
        fn resolve_endpoints_honors_openai_base_url_for_the_inherited_trio_endpoint() {
            let cfg = MagiConfig::default(); // sin base_url propio, sin [magi].base_url
            let resolved = resolve_endpoints(&cfg, Some("http://otherhost:9999/v1"), None).unwrap();
            assert_eq!(resolved.root.as_str(), "http://otherhost:9999/v1");
            assert_eq!(
                resolved.magi.as_str(),
                "http://otherhost:9999/v1",
                "el trío hereda la raíz YA resuelta con su capa de env, no una \
                 recalculada solo de TOML"
            );
        }

        /// Un `[magi].base_url` PROPIO sigue ganándole al env var de la raíz — el env
        /// solo llena el hueco de la herencia, no pisa una declaración explícita.
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
            assert!(notice.is_none(), "ya normalizado: sin aviso");

            let (root, _) = openai_compat_root("http://localhost:11434/v1/");
            assert_eq!(
                root, "http://localhost:11434/v1",
                "idempotente ante una barra final"
            );
        }

        /// C1 (fix round 2, CRITICAL, Security): el aviso de normalización interpola el
        /// endpoint YA RESUELTO (post-sustitución de placeholders REQ-A16c) — si trae
        /// credenciales, deben llegar redactadas al TEXTO del aviso, aunque el `root`
        /// devuelto para el provider real las siga llevando intactas (el provider SÍ
        /// las necesita: `api_key = None` para Ollama no cubre `userinfo` en la URL).
        #[test]
        fn openai_compat_root_redacts_credentials_in_the_notice_but_not_in_the_root() {
            let (root, notice) = openai_compat_root("https://realuser:realpass@ollama.lan:11434");
            assert_eq!(
                root, "https://realuser:realpass@ollama.lan:11434/v1",
                "el ROOT real es lo que el provider necesita para autenticar"
            );
            let notice = notice.expect("sin /v1, debe avisar");
            assert!(
                !notice.contains("realuser") && !notice.contains("realpass"),
                "la credencial no debe llegar al aviso: {notice}"
            );
            assert!(
                notice.contains("ollama.lan"),
                "el host sí debe seguir visible: {notice}"
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

        /// Restaura la cobertura de precedencia que el trío siempre tuvo para los
        /// overrides por asiento: `MAGI_MODEL_<AGENT>` (env) > `[magi].<agent>_model`
        /// (TOML) > el modelo del backend. Fix round 1 (coordinador, 2026-08-03):
        /// R-A03 solo admite las tres rupturas declaradas en REQ-A21/A22/A23, y
        /// `MAGI_MODEL_*` no es ninguna de ellas — quitar la capacidad al retirar el
        /// adapter fue una ruptura NO declarada, así que se restaura acá.
        ///
        /// Usa la validez del alias de Caspar como señal observable — mismo truco que
        /// `cfg_with_only_caspar_unbuildable` — en vez de inspeccionar estado interno:
        /// un modelo inválido hace que ESE asiento falle a construir, así que "¿cuál
        /// modelo ganó?" se lee de si el trío construye o no, sin necesitar red real.
        #[test]
        fn env_model_override_wins_over_toml_which_wins_over_the_backend_model() {
            let backend_only = MagiConfig::from_toml_str(
                "provider = \"anthropic\"\n[anthropic]\nmodel = \"claude-sonnet-4-6\"\n",
            )
            .unwrap();
            let toml_override_invalid = cfg_with_only_caspar_unbuildable();
            let c = creds();
            let endpoints = test_endpoints();

            // Ni TOML ni env: el modelo del BACKEND (válido) alcanza.
            let mut notices = Vec::new();
            assert!(
                build_magi_orchestrator(
                    &backend_only,
                    ProviderKind::Anthropic,
                    &endpoints,
                    Some(&c),
                    None,
                    &MagiEnvModelOverrides::default(),
                    &mut notices,
                )
                .is_ok(),
                "sin overrides, el modelo del backend debe alcanzar"
            );

            // Override de TOML inválido, SIN env: el TOML se aplica de verdad (y por
            // eso falla) — no "se ignora silenciosamente".
            let mut notices = Vec::new();
            assert!(
                build_magi_orchestrator(
                    &toml_override_invalid,
                    ProviderKind::Anthropic,
                    &endpoints,
                    Some(&c),
                    None,
                    &MagiEnvModelOverrides::default(),
                    &mut notices,
                )
                .is_err(),
                "el override de TOML debe aplicarse, aunque sea inválido"
            );

            // El MISMO TOML inválido, pero con env override VÁLIDO: env gana.
            let env_overrides = MagiEnvModelOverrides {
                caspar: Some("claude-opus-4-7".to_string()),
                ..MagiEnvModelOverrides::default()
            };
            let mut notices = Vec::new();
            assert!(
                build_magi_orchestrator(
                    &toml_override_invalid,
                    ProviderKind::Anthropic,
                    &endpoints,
                    Some(&c),
                    None,
                    &env_overrides,
                    &mut notices,
                )
                .is_ok(),
                "env debe ganarle a un override de TOML inválido"
            );
        }
    }

    /// Task 4.3 — REQ-A06/SC-A06: comportamiento por superficie cuando el trío no es
    /// construible. `trio_construction` (arriba) ya cubre que
    /// `build_magi_orchestrator` reporte TODOS los asientos caídos (SC-A05b/SC-A05c);
    /// este módulo cubre lo que Task 4.3 agrega — que esa información REALMENTE
    /// llegue al usuario en cada superficie, y no solo que el tipo la lleve.
    mod trio_unavailable_surfaces {
        use super::*;
        use magi_core::test_support::RoutingMockProvider;

        fn seat_unbuildable(seats: Vec<(AgentName, SeatError)>) -> TrioError {
            TrioError::SeatUnbuildable { seats }
        }

        /// La primitiva de formateo compartida nombra el asiento Y la causa — no un
        /// conteo. Reusada tanto por `Display` como por `trio_unavailable_message`.
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

        /// R2 (obligación heredada de Task 4.1): el `Display` de `SeatUnbuildable` NO
        /// se queda en un conteo — un `{e}`/`.to_string()` futuro que no pase por
        /// `trio_unavailable_message` sigue nombrando cada asiento y su causa.
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
                "el primer asiento debe nombrarse con su causa: {text}"
            );
            assert!(
                text.contains("Caspar") && text.contains("connection refused"),
                "el segundo asiento debe nombrarse con su causa: {text}"
            );
        }

        /// SC-A05b + SC-A05c juntas: el mensaje único nombra CADA asiento caído, su
        /// causa, y los reporta TODOS en una sola corrida (no solo el primero).
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
                "las dos causas de credencial deben aparecer, no colapsarse: {msg}"
            );
            assert!(msg.contains("connection refused"), "{msg}");
            assert!(
                msg.contains("vault set"),
                "debe decir CÓMO habilitarlo: {msg}"
            );
        }

        /// `UnknownKind` nombra el valor inválido Y el vocabulario válido.
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

        /// `NoSeats` y `Builder` comparten el mismo texto genérico — ninguno de los
        /// dos es alcanzable por la ruta de producción real hoy (ver sus propios
        /// rustdocs), pero el `match` exhaustivo de `trio_unavailable_message` los
        /// cubre igual, y ambos deben producir texto no vacío y accionable.
        #[test]
        fn trio_unavailable_message_no_seats_and_builder_share_the_same_generic_text() {
            let no_seats_msg = trio_unavailable_message(&TrioError::NoSeats);
            let cause = redact_foreign_error(&std::io::Error::other("boom"));
            let builder_msg = trio_unavailable_message(&TrioError::Builder(cause));
            assert_eq!(no_seats_msg, builder_msg);
            assert!(!no_seats_msg.is_empty());
        }

        /// SC-A06b, el invariante central: el notice de arranque y la respuesta que
        /// un futuro `/consult` da son el MISMO string — no dos redacciones
        /// independientes que puedan divergir.
        #[test]
        fn trio_unavailable_for_tui_notice_and_reply_are_the_same_text_and_blocking_tier() {
            let err = seat_unbuildable(vec![(
                AgentName::Melchior,
                SeatError::MissingCredential {
                    var: "OPENAI_API_KEY",
                },
            )]);
            let (notice, msg) = trio_unavailable_for_tui(&err);
            assert_eq!(
                notice.text, msg,
                "notice y respuesta deben ser el MISMO texto"
            );
            assert_eq!(
                notice.tier,
                NoticeTier::Blocking,
                "el trío no construible exige acción — no es un Resolution ni un Info"
            );
        }

        /// SC-A06a: un consult que no puede correr NO se registra — ni con el trío
        /// ausente (invita al modelo a rutear hacia algo destinado a fallar) ni,
        /// simétricamente, se OMITE cuando el trío sí construyó (regresión: el
        /// helper compartido entre la TUI y `magi query` no debe apagar el tool en
        /// el caso feliz).
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
            );
            assert!(
                !agent_without_trio.has_tool("consult"),
                "an unbuildable trio must never register a tool the model could \
                 route to and only then discover it cannot run (SC-A06a)"
            );
        }

        /// SC-A06c: un `magi consult` forzado con un trío no construible falla
        /// CERRADO — código de salida distinto de cero, y SIN escribir ningún
        /// archivo de salida (la corrida vuelve antes de que exista un `Magi` con
        /// el que llamar a `analyze`, así que ningún veredicto puede fabricarse).
        #[test]
        #[serial_test::serial]
        fn a_forced_consult_fails_closed_when_the_trio_is_unbuildable() {
            with_var("MAGI_PROVIDER", None, || {
                with_var("OPENAI_API_KEY", None, || {
                    let tmp = tempfile::tempdir().unwrap();
                    let cwd = dunce::canonicalize(tmp.path()).unwrap();
                    crate::system::workspace::init(&cwd).expect("init .magi/");
                    // openai-compat exige OPENAI_API_KEY para el trío (nunca para el
                    // agente principal, que cae al dummy "ollama") — sin la variable,
                    // los TRES asientos fallan por credencial faltante.
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

    /// Task 4.4 — REQ-A07p/SC-A07p (aviso de divergencia de endpoint) y
    /// REQ-A12c/SC-A12f (traducción del 401 keyless).
    mod divergence_and_keyless_auth {
        use super::*;

        /// Construye un `MagiConfig` con `base_url` de raíz y, opcionalmente, un
        /// override propio de `[magi].base_url` — el único par de campos que
        /// `magi_endpoint_diverges()` mira.
        fn cfg_with_endpoints(root: &str, magi_override: Option<&str>) -> MagiConfig {
            MagiConfig {
                base_url: Some(root.to_string()),
                magi: crate::config::MagiSectionConfig {
                    base_url: magi_override.map(str::to_string),
                    ..crate::config::MagiSectionConfig::default()
                },
                ..MagiConfig::default()
            }
        }

        /// SC-A07p: la divergencia de endpoint se avisa, y SOLO cuando hay
        /// divergencia Y la inferencia está activa.
        ///
        /// **Divergencia respecto del pseudocódigo del brief, y por qué está probada
        /// acá y no solo argumentada en el rustdoc de `divergence_notice`.** El brief
        /// recalculaba `will_attempt_classification` puertas adentro de la función,
        /// ignorando el segundo parámetro. Con el `cfg` IDÉNTICO en las últimas dos
        /// aserciones —difieren solo en `true`/`false`— un recálculo interno habría
        /// dado el mismo resultado en las dos, y la tercera aserción de este test
        /// (`divergence_notice(&cfg, false).is_none()`) habría fallado. Este test es
        /// la evidencia ejecutable de por qué la implementación usa el parámetro
        /// tal cual, sin recalcularlo.
        #[test]
        fn endpoint_divergence_is_announced_only_when_it_actually_diverges_and_inference_is_active()
        {
            let cfg = cfg_with_endpoints("http://a/v1", Some("http://b/v1"));
            let n = divergence_notice(&cfg, true).expect("hay divergencia con inferencia activa");
            assert!(
                n.text.contains("provider principal"),
                "debe decir por dónde pasa el contenido primero: {}",
                n.text
            );
            assert_eq!(n.tier, NoticeTier::Resolution);

            assert!(
                divergence_notice(&cfg_with_endpoints("http://a/v1", None), true).is_none(),
                "mismo endpoint (el trío hereda): no hay divergencia que anunciar"
            );
            assert!(
                divergence_notice(&cfg, false).is_none(),
                "sin inferencia activa el contenido no pasa por el principal"
            );
        }

        /// SC-A07p (cableado, superficie TUI): el aviso no solo se PRODUCE, se EMITE
        /// — llega al vector que `run()` (la TUI) efectivamente imprime.
        ///
        /// **Alcance declarado, a propósito (fix round 4): esto cubre SOLO la TUI.**
        /// El brief original de esta tarea pedía "el vector que la TUI **y el
        /// headless** imprimen" (`task-4.4-brief.md:41`); una revisión encontró que
        /// este test (y su mensaje de aserción) habían recortado esa cobertura a
        /// solo la TUI sin que ningún reporte de ronda lo dijera — un requisito
        /// achicado en silencio dentro del texto de un test, que es exactamente el
        /// modo en que un hueco se vuelve invisible. El headless YA NO comparte este
        /// test: tiene el suyo propio,
        /// `test_prepare_headless_carries_the_divergence_notice_when_it_applies`,
        /// más abajo, porque `prepare_headless` no puede probarse llamando a
        /// `push_divergence_notice` directo (esa función no es su única llamadora;
        /// `prepare_headless` tiene su PROPIO call site — ver ese test para la
        /// prueba real de que ESE call site existe).
        ///
        /// Va aparte del test anterior a propósito, como pide el brief de esta
        /// tarea: aquel verifica el PREDICADO; este verifica el EMPUJE al vector.
        /// `run()` (dueño real de `startup_notices`) no es unit-testeable — abre el
        /// vault, descubre el workspace y usa un TTY reales — así que este test
        /// llama a `push_divergence_notice` directamente: es la MISMA función, y la
        /// ÚNICA, que `run()` invoca para esto (una línea, trivial de auditar contra
        /// el diff). Una función correcta que nadie llama pasaría el test anterior y
        /// dejaría al usuario sin el aviso — es el modo de fallo exacto de un
        /// "definido pero no cableado" que ya ocurrió una vez en este plan (Task 4.3).
        #[test]
        fn the_divergence_notice_reaches_the_tui_startup_notices() {
            let cfg = cfg_with_endpoints("http://a/v1", Some("http://b/v1"));
            let mut notices: Vec<Notice> = Vec::new();
            push_divergence_notice(&cfg, true, &mut notices);
            assert!(
                notices
                    .iter()
                    .any(|n| n.text.contains("provider principal")),
                "el aviso tiene que estar en el vector que la TUI imprime \
                 (superficie TUI únicamente — ver \
                 test_prepare_headless_carries_the_divergence_notice_when_it_applies \
                 para la superficie headless): {notices:?}"
            );
        }

        /// SC-A07p (cableado, superficie HEADLESS) — fix round 4, finding 1.
        ///
        /// **Esto es lo que faltaba, sin ningún test que lo cubriera.**
        /// `push_divergence_notice` solo tenía UN call site de producción, dentro de
        /// `run()` (la TUI); `prepare_headless` —el preludio compartido de
        /// `magi query` y `magi consult`— nunca lo invocaba. REQ-A07c es
        /// explícitamente sobre la ruta headless: un pipeline con `magi consult` sin
        /// `--mode` es SC-A07f, y ese pipeline no tiene una TUI en la que el aviso
        /// pudiera aparecer. Cablear solo la superficie interactiva —donde hay un
        /// humano mirando— e ignorar la automatizada invertía la prioridad que la
        /// propia spec fija.
        ///
        /// **Por qué este test dirige el `MagiConfig` a mano, no `push_divergence_
        /// notice` directo.** `prepare_headless` es una función real, con I/O real
        /// (`.magi/` descubierto, `magi.toml` leído del disco), así que a diferencia
        /// del test de la TUI de arriba —que no puede evitar llamar a
        /// `push_divergence_notice` DIRECTO porque `run()` no es testeable en
        /// absoluto— ACÁ sí se puede manejar la función real de punta a punta:
        /// `init_default_workspace`/`write_envelope`/`base_hargs` (ya usados por
        /// `test_prepare_headless_cli_provider_override_normalizes_the_new_vocabulary`
        /// arriba) son exactamente el arnés que hace esto posible con `--no-memory`,
        /// sin vault real.
        ///
        /// `HeadlessContext::divergence_notice` existe SOLO para que este test pueda
        /// afirmar contra el resultado sin capturar stderr del proceso (un recurso
        /// global, no seguro para una suite de tests en paralelo) — mismo motivo que
        /// ya justifica el campo `provider_kind` de la misma struct.
        #[test]
        fn test_prepare_headless_carries_the_divergence_notice_when_it_applies() {
            with_var("MAGI_PROVIDER", None, || {
                with_var("ANTHROPIC_MODEL", None, || {
                    with_var("OPENAI_MODEL", None, || {
                        let tmp = tempfile::tempdir().unwrap();
                        let cwd = dunce::canonicalize(tmp.path()).unwrap();
                        crate::system::workspace::init(&cwd).expect("init .magi/");
                        // Diverge (root vs. [magi].base_url distintos) y NO declara
                        // `default_mode` ⇒ inferencia activa: las dos condiciones de
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
                        assert!(
                            notice.text.contains("provider principal"),
                            "{}",
                            notice.text
                        );
                    });
                });
            });
        }

        /// Precondición de `divergence_notice`: `load()` ya validó las dos
        /// plantillas antes de devolver un `MagiConfig`, así que en producción
        /// `effective_magi_base_url()`/`effective_base_url()` nunca fallan acá.
        /// Mismo patrón que `MagiConfig::effective_provider`/`effective_default_mode`
        /// (`config.rs`): construir el `MagiConfig` a mano, saltándose `load()`, es
        /// lo único que puede violar esa precondición, y el `debug_assert!` lo
        /// convierte en un panic ruidoso en vez de un `Ollama`/`None` silencioso.
        #[test]
        #[should_panic(expected = "validado")]
        fn divergence_notice_panics_in_debug_builds_when_the_endpoint_template_is_invalid() {
            // Credencial literal: `EndpointTemplate::parse` la rechaza (REQ-A16c), así
            // que `effective_magi_base_url()` falla — la precondición que `load()`
            // normalmente garantiza, violada a propósito.
            let cfg = MagiConfig {
                magi: crate::config::MagiSectionConfig {
                    base_url: Some("https://user:pass@host/v1".to_string()),
                    ..crate::config::MagiSectionConfig::default()
                },
                ..MagiConfig::default()
            };
            let _ = divergence_notice(&cfg, true);
        }

        /// R6 (Task 1.2b, `planning/claude-plan-tdd.md` ~L3160): cierre de los caminos
        /// de credencial que ese plan marcó como naciendo con el trío nativo en Fase 4.
        ///
        /// **No es el mismo test que el plan describe, y no puede serlo.** El plan
        /// imaginaba un único canario en `src/magi/endpoint.rs` (lib) cubriendo los
        /// cinco caminos desde un solo lugar. Pero `MagiConfig` y `SeatError` son
        /// tipos del crate BIN (`mod config;`/`main.rs`) — ninguno de los dos
        /// aparece en la lista `pub mod` de `src/lib.rs` (`headless, magi, notices,
        /// redact, vault`), así que un test del crate LIB no puede nombrarlos, y
        /// `divergence_notice` tampoco pudo vivir en `src/magi/mode.rs` por la misma
        /// razón (ver el reporte de esta tarea).
        ///
        /// **Camino 4 se movió**, y con él su cobertura: la ronda 2 de esta tarea
        /// retiró `explain_keyless_auth_failure(&SeatError, ProviderKind)` — nunca
        /// tuvo un llamador de producción real — y la reemplazó por
        /// `tools::consult::keyless_auth_explanation(&str, ProviderKind)`, que opera
        /// sobre la causa YA RENDERIZADA de `MagiReport::failed_agents` y vive en
        /// `src/tools/consult.rs`, donde también está su propia cobertura de no-leak
        /// (`keyless_auth_explanation_never_echoes_the_raw_cause`). Acá se prueban
        /// los caminos que SÍ siguen siendo de este archivo: 1
        /// (`divergence_notice`) y 3 (`trio_unavailable_message`). El camino 2
        /// (`openai_compat_root`) y el 5 (el aviso de incoherencia de Anthropic en
        /// `resolution_notices()`) ya tienen su propia cobertura desde Task
        /// 1.2b/4.1.
        #[test]
        fn no_notice_or_error_path_in_this_file_leaks_a_credential() {
            const CANARY: &str = "c4n4ry-s3cr3t";

            // Camino 1 — `divergence_notice`: opera sobre la PLANTILLA
            // (`EndpointTemplate::as_str()`), que por construcción (REQ-A16c) no
            // puede contener un secreto — un literal ahí es rechazado al parsear,
            // nunca aceptado y mostrado. Se usa el placeholder, no un canario
            // literal, precisamente porque un canario literal no podría existir en
            // este campo (probado por el test de arriba).
            let cfg = cfg_with_endpoints("http://a/v1", Some("https://[user]:[password]@b/v1"));
            let notice = divergence_notice(&cfg, true).expect("diverge con inferencia activa");
            assert!(!notice.text.contains(CANARY));

            // Camino 3 — `trio_unavailable_message`: la causa foránea pasa por
            // `redact_foreign_error` ANTES de convertirse en `SeatError::Transport`
            // (ver `build_native_provider::to_seat`); acá se ejercita la MISMA
            // composición, directo sobre el tipo.
            let foreign =
                std::io::Error::other(format!("connect to https://alice:{CANARY}@host/v1"));
            let err = TrioError::SeatUnbuildable {
                seats: vec![(
                    AgentName::Melchior,
                    SeatError::Transport(redact_foreign_error(&foreign)),
                )],
            };
            assert!(!trio_unavailable_message(&err).contains(CANARY));
        }
    }

    /// Task 5.2 — notices del probe (REQ-A24c), el aviso de composición staleness
    /// (SC-A24i), y `orchestrate_probes` (REQ-A24/SC-A24j/SC-A24k), la función que decide
    /// cuántas tandas de sondeo lanzar y garantiza que la tabla del trío nunca incluye al
    /// principal.
    mod probe_orchestration {
        use super::*;
        use async_trait::async_trait;
        use magi_core::rotation::ProviderProbe;
        use magi_rs::magi::probe::ProbeSeat;
        use std::sync::atomic::{AtomicUsize, Ordering};

        /// Sonda que devuelve SIEMPRE la misma ventana, sin digest, sin I/O real.
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

        /// Doble de `ProbeFactory` con ventana FIJA por modelo (mapa `modelo -> ventana`).
        /// Un modelo AUSENTE del mapa degrada a `Unbuildable` — nunca panica, nunca inventa
        /// una ventana. No reusa los dobles privados de `magi::probe::tests` (viven en otro
        /// módulo y no se exportan) — R-A04 exige la misma costura de inyección acá.
        struct MappedProbeFactory {
            windows: BTreeMap<&'static str, usize>,
            /// Cuántas veces se llamó a `probe_for` — para pinnear SC-A24h: releer un
            /// snapshot ya capturado nunca debe volver a tocar la fábrica.
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
                        "modelo no mapeado en el doble de test",
                    ))),
                }
            }
        }

        /// Endpoint de prueba compartido: una `base_url` plana sin placeholders, así que
        /// resolverla no necesita un vault real (mismo patrón que `trio_construction`).
        fn test_endpoints() -> ResolvedEndpoints {
            let tpl = EndpointTemplate::parse("http://localhost:11434/v1").unwrap();
            ResolvedEndpoints {
                root: tpl.resolve(&mut NoVaultInScope, Scope::Root).unwrap(),
                magi: tpl.resolve(&mut NoVaultInScope, Scope::Magi).unwrap(),
                embedding: tpl.resolve(&mut NoVaultInScope, Scope::Embedding).unwrap(),
            }
        }

        /// Endpoints DIVERGENTES: el trío en un host distinto del principal, para ejercitar
        /// la rama `join!` de `orchestrate_probes`.
        fn diverging_endpoints() -> ResolvedEndpoints {
            let root = EndpointTemplate::parse("http://root-host:11434/v1")
                .unwrap()
                .resolve(&mut NoVaultInScope, Scope::Root)
                .unwrap();
            let magi = EndpointTemplate::parse("http://magi-host:11434/v1")
                .unwrap()
                .resolve(&mut NoVaultInScope, Scope::Magi)
                .unwrap();
            let embedding = EndpointTemplate::parse("http://localhost:11434/v1")
                .unwrap()
                .resolve(&mut NoVaultInScope, Scope::Embedding)
                .unwrap();
            ResolvedEndpoints {
                root,
                magi,
                embedding,
            }
        }

        /// `MagiConfig` cuyo `[magi]` declara `base_url` propio (y opcionalmente `kind`) —
        /// el par de campos que `magi_endpoint_diverges()` mira. Sin modelo de sección
        /// propio: quien necesite uno usa [`cfg_diverging_with_models`].
        fn cfg_diverging(kind: Option<&str>) -> MagiConfig {
            MagiConfig {
                magi: crate::config::MagiSectionConfig {
                    base_url: Some("http://magi-host:11434/v1".to_string()),
                    kind: kind.map(str::to_string),
                    ..crate::config::MagiSectionConfig::default()
                },
                ..MagiConfig::default()
            }
        }

        /// `MagiConfig` con SECCIONES distintas y nombrables (`[openai].model`,
        /// `[anthropic].model`), sin ningún override por asiento (`melchior_model` etc. —
        /// los tres heredan el fallback). Es la fixture del finding de fix round 1: sin
        /// nombres de sección propios y controlables, no hay forma de distinguir "el trío
        /// sondeó SU modelo" de "el trío sondeó el del principal" — los dos casos se ven
        /// idénticos si las dos secciones comparten el mismo nombre.
        fn cfg_with_distinct_section_models(
            openai_model: &str,
            anthropic_model: &str,
        ) -> MagiConfig {
            MagiConfig {
                openai: crate::config::OpenAiConfig {
                    model: Some(openai_model.to_string()),
                },
                anthropic: crate::config::AnthropicConfig {
                    model: Some(anthropic_model.to_string()),
                },
                ..MagiConfig::default()
            }
        }

        /// Como [`cfg_diverging`], pero con las DOS secciones nombrables también — la
        /// fixture que ejercita el finding de fix round 1 con el trío en un endpoint Y un
        /// kind distintos del principal a la vez.
        fn cfg_diverging_with_models(
            kind: Option<&str>,
            openai_model: &str,
            anthropic_model: &str,
        ) -> MagiConfig {
            MagiConfig {
                magi: crate::config::MagiSectionConfig {
                    base_url: Some("http://magi-host:11434/v1".to_string()),
                    kind: kind.map(str::to_string),
                    ..crate::config::MagiSectionConfig::default()
                },
                ..cfg_with_distinct_section_models(openai_model, anthropic_model)
            }
        }

        // ---- probe_notice / stale_composition_notice (contrato del brief, Step 1) -----

        /// SC-A24f: el arranque en frío se explica, no se confunde con una falla.
        #[test]
        fn the_notice_distinguishes_the_three_measurement_states() {
            assert!(probe_notice(&Measurement::Measured {
                window: 128_000,
                digest: Some("ab".repeat(32)),
            })
            .contains("128000"));
            assert!(probe_notice(&Measurement::NotMeasurable).contains("no ofrece"));
            let cold = probe_notice(&Measurement::NotMeasuredThisTime);
            assert!(
                cold.contains("esta vez") && cold.contains("siguiente"),
                "debe anticipar que el próximo arranque probablemente mida"
            );
        }

        /// El digest se muestra truncado: es un identificador, no un secreto, pero 64 hex
        /// es ruido.
        #[test]
        fn the_digest_is_shown_truncated() {
            let n = probe_notice(&Measurement::Measured {
                window: 1000,
                digest: Some("ab".repeat(32)),
            });
            assert!(!n.contains(&"ab".repeat(32)));
        }

        /// Borde: una ventana medida SIN digest (p. ej. `/api/tags` no lo resolvió) sigue
        /// reportando la ventana — no se pierde información que sí se tiene.
        #[test]
        fn a_measured_window_without_a_digest_still_reports_the_window() {
            let n = probe_notice(&Measurement::Measured {
                window: 128_000,
                digest: None,
            });
            assert!(n.contains("128000"));
            assert!(n.contains("digest no resuelto"));
        }

        /// SC-A24i: se avisa, y la comparación es EN TOKENS — no bytes contra tokens.
        #[test]
        fn a_max_query_close_to_the_measured_window_is_flagged_after_unit_conversion() {
            let window_tokens = 100_000_usize;
            // Un cap en BYTES que, convertido, queda justo por encima del 80 % de la
            // ventana.
            let close_bytes =
                ((window_tokens as f64 * STALE_NOTICE_RATIO * CHARS_PER_TOKEN_EST) as usize) + 8;
            let n = stale_composition_notice(window_tokens, close_bytes).expect("debe avisar");
            assert!(
                n.contains("tokens") && n.contains("chars/token"),
                "el notice debe nombrar el estimador: es una aproximación, no una medición"
            );

            assert!(
                stale_composition_notice(window_tokens, close_bytes / 10).is_none(),
                "con holgura amplia no hay riesgo que anunciar"
            );
        }

        /// El bug que este par de funciones existe para evitar: comparar bytes contra
        /// tokens.
        #[test]
        fn comparing_raw_bytes_against_a_token_window_would_be_meaningless() {
            let window_tokens = 128_000_usize;
            // No un literal suelto: el test sigue al valor real.
            let cap_bytes = magi_rs::magi::MAX_QUERY_BYTES;
            assert!(
                cap_bytes > window_tokens,
                "en crudo el cap 'supera' la ventana..."
            );
            assert!(
                bytes_to_tokens_est(cap_bytes) < window_tokens,
                "...pero convertido NO la supera: sin conversión el notice saldría siempre"
            );
        }

        /// `MagiConfig` con el trío en el MISMO endpoint/kind que el principal (rama
        /// compartida de `orchestrate_probes`), pero con los CUATRO nombres — principal +
        /// tres mages — distintos y controlables por el test.
        fn cfg_with_four_distinct_models(
            principal: &str,
            melchior: &str,
            balthasar: &str,
            caspar: &str,
        ) -> MagiConfig {
            MagiConfig {
                openai: crate::config::OpenAiConfig {
                    model: Some(principal.to_string()),
                },
                magi: crate::config::MagiSectionConfig {
                    melchior_model: Some(melchior.to_string()),
                    balthasar_model: Some(balthasar.to_string()),
                    caspar_model: Some(caspar.to_string()),
                    ..crate::config::MagiSectionConfig::default()
                },
                ..MagiConfig::default()
            }
        }

        // ---- orchestrate_probes: rama compartida --------------------------------------

        /// SC-A24 / REQ-A24: endpoint y kind compartidos ⇒ UNA tanda (una sonda por
        /// modelo único, principal incluido), y la tabla del trío devuelta NUNCA incluye
        /// al principal.
        #[tokio::test]
        async fn shared_endpoint_probes_once_and_the_trio_table_excludes_the_principal() {
            let factory = MappedProbeFactory::new(&[
                ("principal", 4_096),
                ("melchior", 128_000),
                ("balthasar", 200_000),
                ("caspar", 256_000),
            ]);
            let cfg = cfg_with_four_distinct_models("principal", "melchior", "balthasar", "caspar");
            let (principal_model, principal, trio) =
                orchestrate_probes(&cfg, &test_endpoints(), ProviderKind::Ollama, &factory).await;

            assert_eq!(principal_model, "principal");
            assert!(matches!(
                principal,
                Some(Measurement::Measured { window: 4_096, .. })
            ));
            assert_eq!(trio.len(), 3, "solo los TRES mages, nunca el principal");
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
                "una tanda: las 4 sondas (principal + 3 mages) se piden juntas"
            );
        }

        /// SC-A24j — la propiedad central de esta tarea: un principal de ventana CHICA no
        /// baja el umbral derivado de los mages, porque `derive_warn_tokens` nunca ve su
        /// medición — `orchestrate_probes` la excluye de la tabla del trío por
        /// construcción, no por convención en el llamador.
        #[tokio::test]
        async fn a_small_principal_never_lowers_the_mage_derived_threshold() {
            let factory = MappedProbeFactory::new(&[
                ("principal", 2_048), // la ventana MÁS CHICA de todo el proceso
                ("melchior", 1_000_000),
                ("balthasar", 512_000),
                ("caspar", 256_000),
            ]);
            let cfg = cfg_with_four_distinct_models("principal", "melchior", "balthasar", "caspar");
            let (_principal_model, _principal, trio) =
                orchestrate_probes(&cfg, &test_endpoints(), ProviderKind::Ollama, &factory).await;

            let derived = derive_warn_tokens(&trio).expect("los tres mages midieron");
            #[allow(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss
            )]
            let expected_from_caspar = (256_000.0 * magi_rs::magi::WARN_WINDOW_FRACTION) as usize;
            assert_eq!(
                derived, expected_from_caspar,
                "el mínimo de los MAGES (Caspar, 256k) manda — NUNCA el del principal (2k), \
                 que sería un umbral absurdamente bajo si se hubiera colado"
            );
        }

        /// SC-A24h: el umbral derivado de un snapshot de arranque es ESTABLE. La garantía
        /// completa de la spec ("el probe corre UNA vez por proceso") es estructural — un
        /// único call site en `run()`/`prepare_headless()`, antes de que arranque el loop
        /// de turnos — y eso no es algo que un test de unidad pueda ejercitar sin levantar
        /// el proceso entero. Lo que SÍ es verificable acá, y es la mitad de la propiedad
        /// que un test de unidad puede pinnear: releer el MISMO snapshot ya capturado es
        /// una operación pura y determinista que nunca vuelve a tocar la fábrica de
        /// sondas — si algo en el proceso "refrescara" el umbral por las dudas, esto lo
        /// delataría por un conteo de llamadas que sube solo.
        #[tokio::test]
        async fn the_probe_runs_once_and_the_threshold_stays_put() {
            let factory = MappedProbeFactory::new(&[("principal", 128_000), ("m", 256_000)]);
            let cfg = cfg_with_four_distinct_models("principal", "m", "m", "m");
            let (_principal_model, _principal, trio) =
                orchestrate_probes(&cfg, &test_endpoints(), ProviderKind::Ollama, &factory).await;
            let calls_after_the_startup_probe = factory.calls();
            assert!(
                calls_after_the_startup_probe > 0,
                "la sonda debió correr al menos una vez en el arranque"
            );

            // Dos "lecturas" del mismo snapshot, como dos consultas sucesivas dentro de la
            // MISMA sesión — nada acá vuelve a invocar `orchestrate_probes`.
            let warn_at_startup = derive_warn_tokens(&trio);
            let warn_for_a_later_query = derive_warn_tokens(&trio);
            assert_eq!(
                warn_at_startup, warn_for_a_later_query,
                "el umbral derivado del snapshot de arranque no cambia solo"
            );
            assert_eq!(
                factory.calls(),
                calls_after_the_startup_probe,
                "derivar el umbral de un snapshot ya capturado NUNCA vuelve a tocar la sonda"
            );
        }

        // ---- orchestrate_probes: rama divergente --------------------------------------

        /// SC-A24k (un nivel más arriba, entre TANDAS): endpoint divergente ⇒ DOS
        /// llamadas independientes, cada una con su kind y su endpoint propios. Principal
        /// Y trío en `ollama` (el ÚNICO kind medible) a propósito, con los tres asientos
        /// del trío overrideados a un modelo propio — es la única forma de que el
        /// principal y el trío midan ventanas DISTINTAS mientras los dos son medibles: si
        /// compartieran kind Y ningún asiento tuviera override, resolverían a la MISMA
        /// sección por diseño (ver `a_diverging_trio_kind_probes_its_own_section_model_
        /// not_the_principals`, que sí cruza de sección).
        #[tokio::test]
        async fn diverging_endpoint_probes_the_trio_separately_with_its_own_kind() {
            let factory = MappedProbeFactory::new(&[("principal", 64_000), ("m", 128_000)]);
            let cfg = MagiConfig {
                openai: crate::config::OpenAiConfig {
                    model: Some("principal".to_string()),
                },
                magi: crate::config::MagiSectionConfig {
                    base_url: Some("http://magi-host:11434/v1".to_string()),
                    kind: Some("ollama".to_string()),
                    melchior_model: Some("m".to_string()),
                    balthasar_model: Some("m".to_string()),
                    caspar_model: Some("m".to_string()),
                    ..crate::config::MagiSectionConfig::default()
                },
                ..MagiConfig::default()
            };
            let (principal_model, principal, trio) =
                orchestrate_probes(&cfg, &diverging_endpoints(), ProviderKind::Ollama, &factory)
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
                "los tres asientos overrideados a \"m\" deben medir la ventana de \"m\""
            );
        }

        /// **Fix round 1 — finding Logic+Structure.** Reproduce el bug EXACTO reportado:
        /// principal en `anthropic` (lee `[anthropic].model`), trío en `ollama`
        /// (`[magi].kind` declarado, divergiendo — lee `[openai].model`), y NINGÚN asiento
        /// con override propio (`melchior_model`/`balthasar_model`/`caspar_model`
        /// ausentes), así que los tres heredan el fallback. El fallback correcto es
        /// `[openai].model` — la sección DEL KIND DEL TRÍO — nunca `[anthropic].model`,
        /// que es la sección del PRINCIPAL.
        ///
        /// Antes de este fix, los dos call sites (`run()`/`prepare_headless()`) resolvían
        /// el fallback del trío con `resolve_backend_model(cfg, principal_kind)` — el
        /// kind del PRINCIPAL, no el del trío — así que un trío en `ollama` con el
        /// principal en `anthropic` intentaba sondear el NOMBRE de `[anthropic].model`
        /// contra el endpoint del trío.
        ///
        /// Los dos modelos se mapean a ventanas DISTINTAS (`claude-test` → 999 999,
        /// `qwen-test` → 128 000) para que el bug, si reaparece, se vea como un NÚMERO
        /// EQUIVOCADO — envenenando `input_warn_tokens` con la ventana de un modelo ajeno
        /// — en vez de solo una degradación a "no medido", que sería más fácil de pasar
        /// por alto en una revisión superficial.
        #[tokio::test]
        async fn a_diverging_trio_kind_probes_its_own_section_model_not_the_principals() {
            let factory = MappedProbeFactory::new(&[
                ("claude-test", 999_999), // [anthropic].model — la sección del PRINCIPAL
                ("qwen-test", 128_000),   // [openai].model — la sección REAL del trío
            ]);
            let cfg = cfg_diverging_with_models(Some("ollama"), "qwen-test", "claude-test");

            let (principal_model, _principal, trio) = orchestrate_probes(
                &cfg,
                &diverging_endpoints(),
                ProviderKind::Anthropic,
                &factory,
            )
            .await;

            assert_eq!(
                principal_model, "claude-test",
                "el principal SÍ debe resolver su propia sección — esto no es lo que falla"
            );
            assert!(
                trio.values().all(|m| matches!(
                    m,
                    Measurement::Measured {
                        window: 128_000,
                        ..
                    }
                )),
                "el trío debe sondear qwen-test (SU sección, [openai].model bajo kind \
                 ollama) — nunca claude-test (la del principal): de lo contrario mediría \
                 999999 (la ventana de un modelo ajeno) o degradaría a \
                 NotMeasuredThisTime si claude-test no existiera en el endpoint real, y en \
                 ningún caso el umbral derivado tendría relación con el trío. Trío: {trio:?}"
            );
        }

        /// Un `[magi].kind` inválido no propaga error ni panica: degrada el TRÍO entero a
        /// *no medido*, sin adivinar un kind — `build_magi_orchestrator` es quien reporta
        /// el error tipado cuando construya el trío de verdad con la MISMA config.
        #[tokio::test]
        async fn an_invalid_magi_kind_degrades_the_trio_without_guessing() {
            let factory = MappedProbeFactory::new(&[("principal", 64_000)]);
            let cfg = cfg_diverging_with_models(Some("banana"), "principal", "irrelevant");
            let (principal_model, principal, trio) =
                orchestrate_probes(&cfg, &diverging_endpoints(), ProviderKind::Ollama, &factory)
                    .await;
            assert_eq!(principal_model, "principal");
            assert!(
                matches!(
                    principal,
                    Some(Measurement::Measured { window: 64_000, .. })
                ),
                "el principal SÍ se sondea: su kind es válido por construcción"
            );
            assert!(
                trio.values()
                    .all(|m| matches!(m, Measurement::NotMeasuredThisTime)),
                "los tres asientos degradan sin adivinar ningún modelo"
            );
        }

        // ---- probe_and_report: consolidación del bloque duplicado (fix round 1, B3) ---

        /// SC-A24e: lo DECLARADO en `[magi].input_warn_tokens` gana sobre lo MEDIDO —
        /// antes sin test propio (un `Option::or_else` inline en cada call site); ahora
        /// que el bloque es una función compartida, es una aserción barata.
        #[tokio::test]
        async fn declared_input_warn_tokens_beats_the_measured_threshold() {
            let factory = MappedProbeFactory::new(&[
                ("principal", 128_000),
                ("melchior", 128_000),
                ("balthasar", 128_000),
                ("caspar", 128_000),
            ]);
            let mut cfg =
                cfg_with_four_distinct_models("principal", "melchior", "balthasar", "caspar");
            cfg.magi.input_warn_tokens = Some(999);
            let mut notices = Vec::new();
            let warn_tokens = probe_and_report(
                &cfg,
                &test_endpoints(),
                ProviderKind::Ollama,
                &factory,
                &mut notices,
            )
            .await;
            assert_eq!(
                warn_tokens,
                Some(999),
                "lo declarado gana aunque el probe SÍ midió algo distinto"
            );
        }

        /// SC-A24e (el otro lado): sin declarar nada, el umbral sale de lo MEDIDO.
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
            let warn_tokens = probe_and_report(
                &cfg,
                &test_endpoints(),
                ProviderKind::Ollama,
                &factory,
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
                "el notice del principal se empujó a la lista compartida"
            );
        }

        // ---- resolve_magi_kind ---------------------------------------------------------

        /// `[magi].kind` ausente hereda el YA RESUELTO del principal — no
        /// `cfg.effective_provider()`/`cfg.effective_magi_kind()` (TOML-only), que
        /// ignorarían `MAGI_PROVIDER`.
        #[test]
        fn resolve_magi_kind_inherits_the_resolved_principal_when_absent() {
            let cfg = MagiConfig::default();
            assert_eq!(
                resolve_magi_kind(&cfg, ProviderKind::Anthropic).unwrap(),
                ProviderKind::Anthropic,
                "hereda lo YA RESUELTO, no cfg.effective_provider() (que daría Ollama)"
            );
        }

        /// `[magi].kind` declarado gana sobre el principal.
        #[test]
        fn resolve_magi_kind_prefers_the_declared_value_over_the_principal() {
            let cfg = cfg_diverging(Some("anthropic"));
            assert_eq!(
                resolve_magi_kind(&cfg, ProviderKind::Ollama).unwrap(),
                ProviderKind::Anthropic
            );
        }

        /// `[magi].kind` no reconocido es error TIPADO, no un fallback silencioso.
        #[test]
        fn resolve_magi_kind_rejects_an_unknown_value() {
            let cfg = cfg_diverging(Some("banana"));
            let err = resolve_magi_kind(&cfg, ProviderKind::Ollama).unwrap_err();
            assert_eq!(err.got, "banana");
        }

        // ---- resolve_backend_model ------------------------------------------------------

        /// `[openai].model` para `ollama`/`openai-compat`, `[anthropic].model` para
        /// `anthropic` — `[openai]` sirve a los dos primeros porque comparten protocolo.
        #[test]
        fn resolve_backend_model_picks_the_section_matching_the_kind() {
            let cfg = MagiConfig {
                openai: crate::config::OpenAiConfig {
                    model: Some("qwen-test".to_string()),
                },
                anthropic: crate::config::AnthropicConfig {
                    model: Some("claude-test".to_string()),
                },
                ..MagiConfig::default()
            };
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

        /// Borde: ambos ausentes ⇒ los defaults built-in del crate, no un `panic`/`unwrap`.
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
