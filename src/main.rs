#![forbid(unsafe_code)]

mod agent;
mod config;
mod defaults;
mod memory;
mod services;
mod system;
mod tools;
mod tui;

use crate::agent::magi_wiring::{
    resolve_magi_adapter_specs, static_override_notice, MagiEnvModels,
};
use crate::agent::provider::{build_openai_provider, AnthropicProvider, Provider, StaticProvider};
use crate::agent::Agent;
// NOTE: this `MagiConfig` is the magi-rs TOML config (`crate::config::MagiConfig`).
// It is DISTINCT from `magi_core::orchestrator::MagiConfig` — the latter is NEVER
// imported here, avoiding the name collision.
use crate::config::{resolve_openai_base_url, resolve_openai_model, resolve_provider, MagiConfig};
use crate::memory::clock::SystemClock;
use crate::memory::embedding::OpenAiCompatibleEmbedder;
use crate::memory::store::SqliteVectorStore;
use crate::system::database::{EncryptedSqliteMemory, MemoryStore};
use crate::system::fs::{FileSystem, RealFileSystem};
use crate::system::grep::RipGrep;
use crate::tools::bash::BashTool;
use crate::tools::grep::GrepTool;
use crate::tools::knowledge::ProjectFactTool;
use crate::tools::ls::ListTool;
use crate::tools::read::FileReadTool;
use crate::tools::write::FileWriteTool;
use clap::Parser;
use cryptovault::CryptoVault;
use magi_core::orchestrator::{Magi, MagiBuilder};
use magi_rs::vault::{
    check_strength, create_passphrase, harden_process, rekey_envelope, resolve_passphrase,
    run_vault_cmd, strip_trailing_newline, wire, PassphrasePrompt, SecretStore, TtyIo, TtyPrompt,
    VaultCmd, VaultError, PASSPHRASE_ENV,
};
use std::env;
use std::sync::{Arc, Mutex};
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

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Log out: removes ANTHROPIC_API_KEY from the vault.
    #[arg(short, long)]
    logout: bool,

    /// Write a default magi.toml to the workspace and exit (refuses to overwrite).
    #[arg(long)]
    init_config: bool,

    /// Master passphrase (precedence: -p > MAGI_PASSPHRASE > interactive
    /// prompt). Global: also applies to the `vault` subcommand (REQ-V04).
    #[arg(short = 'p', long, global = true)]
    passphrase: Option<String>,

    #[command(subcommand)]
    command: Option<TopCmd>,
}

/// Top-level subcommands beyond the default TUI launch.
#[derive(clap::Subcommand, Debug)]
enum TopCmd {
    /// Encrypted, zero-knowledge secret store (`ls`/`set`/`rm`/`passwd`).
    #[command(subcommand)]
    Vault(VaultCmd),
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

/// Returns `true` when `base_url` resolves to a local address
/// (`localhost`, `127.0.0.1`, or `[::1]`) — used by the cloud-egress notice
/// (CP2-AG/AJ, Task 13b) to suppress the warning for on-device embedders.
fn is_localhost(base_url: &str) -> bool {
    let lower = base_url.to_lowercase();
    lower.contains("localhost") || lower.contains("127.0.0.1") || lower.contains("[::1]")
}

/// Resolves the `ANTHROPIC_API_KEY` config: environment first, then the
/// vault (REQ-V12) — the OS keyring and `key.txt` are no longer consulted at
/// all (REQ-V37). `secret_store` is `None` for an ephemeral (no-persistence)
/// session, in which case only the environment is consulted.
fn discover_config(secret_store: Option<&SharedSecretStore>) -> Option<Config> {
    if let Ok(key) = env::var("ANTHROPIC_API_KEY") {
        let model = env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
        return Some(Config {
            api_key: key.trim().to_string(),
            model,
            source: "ENV".to_string(),
        });
    }
    let ss = secret_store?;
    let mut guard = ss.lock().unwrap_or_else(|p| p.into_inner());
    let key = guard.get("ANTHROPIC_API_KEY").ok()?;
    let model = env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    Some(Config {
        // Trim like the env path: a stored/exported key with stray whitespace or
        // a trailing newline would otherwise produce a malformed auth header (401).
        api_key: key.as_str().trim().to_string(),
        model,
        source: "vault".to_string(),
    })
}

/// Resolves the `OPENAI_API_KEY` used by the OpenAI-compatible chat provider
/// and the embedder: environment first, then the vault (REQ-V12), mirroring
/// [`discover_config`]'s precedence for the Anthropic key.
///
/// Both sources are trimmed for the same reason `discover_config` trims: a key
/// with stray whitespace or a trailing newline (a common `export KEY=$(cat f)`
/// artifact) would otherwise produce a malformed `Authorization` header (401).
fn resolve_openai_key(secret_store: Option<&SharedSecretStore>) -> Option<String> {
    if let Ok(key) = env::var("OPENAI_API_KEY") {
        return Some(key.trim().to_string());
    }
    let ss = secret_store?;
    let mut guard = ss.lock().unwrap_or_else(|p| p.into_inner());
    guard
        .get("OPENAI_API_KEY")
        .ok()
        .map(|z| z.as_str().trim().to_string())
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
/// [`resolve_passphrase`] (`-p` > env > prompt, single entry, REQ-V04);
/// **absent** (first run) ⇒ if `-p` or `MAGI_PASSPHRASE` already supply a
/// value it is used directly after [`check_strength`] (nothing to confirm
/// it against); otherwise, with a TTY, [`create_passphrase`] runs the
/// double-entry + zero-knowledge-warning flow (REQ-V17); without a TTY and
/// without `-p`/env, fails closed with [`VaultError::PassphraseUnavailable`]
/// rather than hanging on a prompt that cannot be read (mirrors REQ-V40's
/// fail-closed spirit, applied to bootstrap).
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
    match env::var(PASSPHRASE_ENV) {
        Ok(v) if !v.is_empty() => {
            let z = strip_trailing_newline(Zeroizing::new(v));
            check_strength(z.as_str())?;
            Ok(z)
        }
        _ => {
            if !prompt.is_interactive() {
                return Err(VaultError::PassphraseUnavailable);
            }
            create_passphrase(prompt, false)
        }
    }
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
        | VaultError::SecretNotFound(_) => 1,
        VaultError::Crypto(_)
        | VaultError::Storage(_)
        | VaultError::VaultMetaCorrupt
        | VaultError::Io(_) => 2,
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

/// Runs `magi-rs vault <cmd>` (short-lived process, never reaches the TUI):
/// resolves the passphrase, opens the encrypted store, wires the vault, and
/// drives [`run_vault_cmd`]. Returns the process exit code.
fn run_vault_subcommand(
    cmd: VaultCmd,
    passphrase_flag: Option<Zeroizing<String>>,
    workspace_root: &std::path::Path,
    hardening_warnings: &[String],
) -> i32 {
    let db_path = workspace_root.join(".magi-rs-memory.db");
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

/// Runs `magi-rs --logout`: opens the vault and removes `ANTHROPIC_API_KEY`
/// (the CLI analogue of SC-V36/SC-V37). An absent DB, or an absent key, is
/// reported as "no stored session" rather than an error. Returns the
/// process exit code.
fn run_logout(passphrase_flag: Option<Zeroizing<String>>, workspace_root: &std::path::Path) -> i32 {
    let db_path = workspace_root.join(".magi-rs-memory.db");
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut args = Args::parse();
    // Own the passphrase as `Zeroizing` immediately (REQ-V41): the only
    // remaining bare copy is clap's own field, dropped right after this
    // `.take()`. `-p` itself stays visible in `argv`/process listings by
    // design (REQ-V04, decision of plan 4) — only the *value* leaving argv
    // unzeroized is in scope here.
    let passphrase_flag = args.passphrase.take().map(Zeroizing::new);
    let workspace_root = env::current_dir()?;

    // REQ-V42: best-effort process hardening, once, before any secret
    // material exists.
    let hardening_warnings = harden_process();

    if let Some(TopCmd::Vault(cmd)) = args.command.take() {
        std::process::exit(run_vault_subcommand(
            cmd,
            passphrase_flag,
            &workspace_root,
            &hardening_warnings,
        ));
    }

    if args.logout {
        std::process::exit(run_logout(passphrase_flag, &workspace_root));
    }

    if args.init_config {
        match crate::defaults::write_default_config(&workspace_root) {
            Ok(path) => {
                println!("Wrote default config to {}", path.display());
                return Ok(());
            }
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(1);
            }
        }
    }

    // ── TUI path ─────────────────────────────────────────────────────────
    let mut startup_notices: Vec<String> = hardening_warnings
        .iter()
        .map(|w| format!("warning: {w}"))
        .collect();

    let db_path = workspace_root.join(".magi-rs-memory.db");
    let mut prompt = TtyPrompt;
    let attachment = open_tui_memory(&db_path, passphrase_flag, &mut prompt, &mut startup_notices);

    let (memory_store, secret_store): (Option<EncryptedSqliteMemory>, Option<SharedSecretStore>) =
        match attachment {
            MemoryAttachment::Encrypted(store) => match store.data_key() {
                Ok(dek) => {
                    for w in dek.warnings() {
                        startup_notices.push(format!("warning: {w}"));
                    }
                    match wire(store.shared_conn(), dek) {
                        Ok(vstore) => (
                            Some(store),
                            Some(Arc::new(Mutex::new(vstore)) as SharedSecretStore),
                        ),
                        Err(e) => {
                            startup_notices.push(format!(
                                "WARNING: could not open the secret vault ({e}); \
                                 ANTHROPIC_API_KEY/OPENAI_API_KEY must come from the \
                                 environment this session."
                            ));
                            (Some(store), None)
                        }
                    }
                }
                Err(e) => {
                    startup_notices.push(format!(
                        "WARNING: could not derive the vault key ({e}); \
                         ANTHROPIC_API_KEY/OPENAI_API_KEY must come from the \
                         environment this session."
                    ));
                    (Some(store), None)
                }
            },
            MemoryAttachment::Ephemeral => (None, None),
        };

    // REQ-V12: API key discovery happens AFTER the vault is (possibly) open.
    let config = discover_config(secret_store.as_ref());
    let (magi_config, config_warning) = MagiConfig::load(&workspace_root);
    let provider_kind = resolve_provider(&magi_config, env::var("MAGI_PROVIDER").ok().as_deref());

    // Credentials needed to build per-agent sibling providers (same backend, different
    // model) for MAGI per-agent overrides. Set inside the openai branch.
    let mut oai_creds: Option<(String, String)> = None; // (base_url, api_key)

    let (provider, provider_info, model_label): (Arc<dyn Provider>, String, String) =
        if provider_kind == "openai" {
            // env > vault (REQ-V12); falls back to the local-Ollama dummy so a
            // real OpenAI/Groq/OpenRouter endpoint still fails loudly with 401
            // rather than silently defaulting to an insecure constant.
            let api_key =
                resolve_openai_key(secret_store.as_ref()).unwrap_or_else(|| "ollama".to_string());
            let base_url =
                resolve_openai_base_url(&magi_config, env::var("OPENAI_BASE_URL").ok().as_deref());
            let model =
                resolve_openai_model(&magi_config, env::var("OPENAI_MODEL").ok().as_deref());
            let info = format!("OpenAI-compatible ({base_url}) Model: {model}");
            let model_label = model.clone();
            oai_creds = Some((base_url.clone(), api_key.clone()));
            (
                build_openai_provider(&base_url, &api_key, &model),
                info,
                model_label,
            )
        } else if let Some(ref c) = config {
            (
                Arc::new(AnthropicProvider::new(c.api_key.clone(), c.model.clone())),
                format!("Magi API ({}) Model: {}", c.source, c.model),
                c.model.clone(),
            )
        } else {
            (
                Arc::new(StaticProvider),
                "Static Mode: no API key found. Set ANTHROPIC_API_KEY or run \
                 `magi-rs vault set ANTHROPIC_API_KEY` (recommended). /login \
                 (OAuth) is best-effort and may be rate-limited."
                    .to_string(),
                "static".to_string(),
            )
        };

    // Notices shown when the TUI starts — the provider banner plus any persistence,
    // reset, or vault warnings that would otherwise be lost to pre-TUI stderr.
    startup_notices.push(provider_info);
    // MAGI fix f: surface malformed/unreadable magi.toml in the TUI rather than
    // losing it to pre-TUI stderr — same path as the persistence/reset notices.
    if let Some(w) = config_warning {
        startup_notices.push(w);
    }
    // B1: surface invalid memory-config values as a startup notice (never panic).
    if let Err(e) = magi_config.memory.validate() {
        startup_notices.push(format!("memory config warning: {e}"));
    }
    // H2: surface invalid embedding-config values alongside memory-config (never panic).
    if let Err(e) = magi_config.embedding.validate() {
        startup_notices.push(format!("embedding config warning: {e}"));
    }
    // RF-9: when there is no magi.toml at all, make the Ollama-first default visible
    // (never-silent). A present-but-minimal magi.toml does NOT trigger this.
    if crate::defaults::should_emit_default_notice(
        &provider_kind,
        workspace_root.join("magi.toml").exists(),
    ) {
        startup_notices.push(crate::defaults::no_config_notice());
    }

    // Build the MAGI orchestrator over the resolved backend. With no per-agent
    // overrides this is the v0.4.0 path (`Magi::new`, single shared adapter).
    // With overrides, build one adapter per overridden agent (same backend
    // creds, different model) via `MagiBuilder::with_provider`.
    let backend_label = if provider_kind == "openai" {
        "openai"
    } else {
        "anthropic"
    };
    let env_models = MagiEnvModels {
        melchior: env::var("MAGI_MODEL_MELCHIOR").ok(),
        balthasar: env::var("MAGI_MODEL_BALTHASAR").ok(),
        caspar: env::var("MAGI_MODEL_CASPAR").ok(),
    };
    let specs = resolve_magi_adapter_specs(backend_label, &magi_config.magi, &env_models);

    // Builds a sibling provider on the SAME backend with a different model.
    // Mirrors the principal provider resolution above: `"openai"` uses the captured
    // OpenAI creds; any OTHER backend uses the discovered Anthropic credentials —
    // the SAME `config.api_key` source as the principal (no second credential path).
    // On this non-static branch the relevant source is always `Some`, so a per-agent
    // override is never silently dropped (a malformed `provider_kind` still maps to
    // the Anthropic path, matching how the principal itself was built).
    let build_sibling = |model: &str| -> Option<Arc<dyn Provider>> {
        if provider_kind == "openai" {
            debug_assert!(
                oai_creds.is_some(),
                "openai backend must have captured oai_creds before building siblings"
            );
            oai_creds
                .as_ref()
                .map(|(b, k)| build_openai_provider(b, k, model))
        } else {
            config.as_ref().map(|c| {
                Arc::new(AnthropicProvider::new(c.api_key.clone(), model.to_string()))
                    as Arc<dyn Provider>
            })
        }
    };

    let consult_magi: Option<Arc<Magi>> = if provider.is_static() {
        // No backend to build adapters; surface a non-silent notice if the user
        // configured [magi] overrides anyway (RF-10, S-13).
        if let Some(notice) = static_override_notice(true, !specs.is_empty()) {
            startup_notices.push(notice);
        }
        None
    } else {
        let default_adapter = crate::agent::magi_adapter::MagiCoreProviderAdapter::new(
            provider.clone(),
            backend_label,
            model_label.clone(),
        );
        if specs.is_empty() {
            // v0.4.0 path — unchanged (S-6).
            Some(Arc::new(Magi::new(Arc::new(default_adapter))))
        } else {
            let mut builder = MagiBuilder::new(Arc::new(default_adapter));
            for spec in &specs {
                if let Some(sibling) = build_sibling(&spec.model) {
                    let adapter = crate::agent::magi_adapter::MagiCoreProviderAdapter::new(
                        sibling,
                        spec.adapter_name.clone(),
                        spec.model.clone(),
                    );
                    builder = builder.with_provider(spec.agent, Arc::new(adapter));
                }
            }
            // build() is fallible (MagiError); propagate to surface at startup (RF-6).
            Some(Arc::new(
                builder
                    .build()
                    .map_err(|e| anyhow::anyhow!("MAGI builder failed: {e}"))?,
            ))
        }
    };

    let mut agent = Agent::new(provider);

    match memory_store {
        Some(concrete_store) => {
            // Build the vector store from the shared connection + masked DEK.
            // Errors here are non-fatal: fall through without the tiered-memory
            // subsystem rather than refusing to start (REQ-29).
            let vstore_result = concrete_store
                .data_key()
                .map_err(|e| crate::memory::error::MemoryError::Crypto(e.to_string()))
                .and_then(|dek| SqliteVectorStore::new(concrete_store.shared_conn(), dek));

            // #11: surface a one-time reset notice if incompatible content was discarded.
            if concrete_store.was_reset() {
                startup_notices.push(
                    "Note: existing on-disk history used an incompatible/corrupt format and \
                     has been reset (fresh start)."
                        .to_string(),
                );
            }
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
            // successfully. `OPENAI_API_KEY` is the embedding API key (env > vault,
            // REQ-V12; may be the dummy `"ollama"` for the local Ollama server —
            // it ignores auth).
            if let Ok(vstore) = vstore_result {
                // W1: new() now returns Result; propagate failure as a startup notice
                // and degrade gracefully (text-only persistence; REQ-29).
                match OpenAiCompatibleEmbedder::new(
                    &magi_config.embedding,
                    resolve_openai_key(secret_store.as_ref()),
                ) {
                    Err(err) => {
                        startup_notices.push(format!(
                            "embedding client init failed ({err}); \
                         memory subsystem disabled (text-only persistence)"
                        ));
                    }
                    Ok(embedder_inner) => {
                        let embedder = Arc::new(embedder_inner);
                        let clock = Arc::new(SystemClock);
                        // Keep a reference for the startup diagnostics line (CP2-AN/S).
                        let vstore = Arc::new(vstore);
                        let vstore_diag = Arc::clone(&vstore);
                        agent.set_memory_subsystem(
                            vstore,
                            embedder,
                            clock,
                            magi_config.memory.clone(),
                        );
                        agent.on_session_open().await.ok();

                        // CP2-AN/S: one-line diagnostics summary — never fail startup on error.
                        if let Ok(d) = vstore_diag.diagnostics("root").await {
                            startup_notices.push(format!(
                        "memory: {} active, {} archived, {} pending re-embed (~{} KB index)",
                        d.active_count,
                        d.archived_count,
                        d.pending_reembed_count,
                        d.ram_estimate_bytes / 1024,
                    ));
                        }

                        // CP2-AG/AJ: warn the user when the distiller will send memory batches
                        // to a cloud embedding endpoint (non-localhost).
                        if magi_config.memory.distill_enabled
                            && !is_localhost(&magi_config.embedding.base_url)
                        {
                            startup_notices.push(format!(
                                "Memory distiller will send bounded memory batches \
                         (≤ {} tokens) to {} — set distill_enabled = false \
                         in [memory] for zero cloud memory egress.",
                                magi_config.memory.distill_max_batch_tokens,
                                magi_config.embedding.base_url,
                            ));
                        }
                    } // Ok(embedder_inner) arm
                } // match OpenAiCompatibleEmbedder::new
            } // if let Ok(vstore)

            // ProjectFactTool needs the same store; register it on the encrypted path only.
            agent.register_tool(Box::new(ProjectFactTool::new(memory.clone())));
        }
        None => {
            // #7: surface the no-persistence state in the TUI, not just pre-TUI stderr.
            startup_notices.push(
                "WARNING: this session runs WITHOUT persistence — your conversation and \
                 project knowledge will NOT be saved (any existing on-disk database is left \
                 untouched). Provide the vault passphrase (-p, MAGI_PASSPHRASE, or the \
                 interactive prompt) to restore persistence."
                    .to_string(),
            );
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
    if let Some(ref magi) = consult_magi {
        agent.register_tool(Box::new(crate::tools::consult::ConsultTool::new(
            magi.clone(),
            magi_config.magi.auto_approve,
        )));
    }

    crate::tui::run_tui_ext(
        agent,
        startup_notices,
        consult_magi,
        workspace_root,
        magi_config.magi.auto_approve,
        secret_store,
    )
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::messages::Message;
    use magi_rs::vault::MaskedDek;

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

    #[test]
    fn test_resolve_provider_wiring() {
        // Wiring smoke test (Task 6): env > TOML > default "openai" (Ollama-first).
        // Pure resolution; no side effects. The real branching in main() is
        // covered by integration with this same helper.
        use crate::config::{resolve_provider, MagiConfig};
        assert_eq!(
            resolve_provider(
                &MagiConfig {
                    provider: Some("anthropic".into()),
                    ..Default::default()
                },
                Some("openai")
            ),
            "openai"
        );
        assert_eq!(resolve_provider(&MagiConfig::default(), None), "openai");
    }

    #[test]
    fn test_args_parses_init_config_flag() {
        use clap::Parser;
        let a = Args::parse_from(["magi-rs", "--init-config"]);
        assert!(a.init_config);
        let b = Args::parse_from(["magi-rs"]);
        assert!(!b.init_config);
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
        // Sanity check for the exhaustive match: user/operation errors -> 1,
        // system/data errors -> 2. Guards against a silent flip during
        // refactors (the compiler already guards against a MISSING variant).
        let user_errors = [
            VaultError::Aborted,
            VaultError::WrongPassphrase,
            VaultError::PassphraseUnavailable,
            VaultError::WeakPassphrase("x".into()),
            VaultError::ValueTooLarge(1),
            VaultError::SecretNotFound("x".into()),
        ];
        for e in &user_errors {
            assert_eq!(vault_error_exit_code(e), 1, "{e}");
        }
        let system_errors = [
            VaultError::Crypto("x".into()),
            VaultError::Storage("x".into()),
            VaultError::VaultMetaCorrupt,
            VaultError::Io("x".into()),
        ];
        for e in &system_errors {
            assert_eq!(vault_error_exit_code(e), 2, "{e}");
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_api_key_resolution_prefers_env_over_vault() {
        // SC-V12: env set => env wins; env unset + vault has the key => vault
        // wins; neither => None (StaticProvider).
        with_var("ANTHROPIC_API_KEY", None, || {
            with_var("ANTHROPIC_MODEL", None, || {
                let ss = vault_fixture();
                {
                    let mut guard = ss.lock().unwrap();
                    guard.set("ANTHROPIC_API_KEY", "sk-from-vault").unwrap();
                }

                // Neither env nor vault (fresh fixture, unset env): None.
                assert!(discover_config(None).is_none());

                // Vault only: vault wins.
                let cfg = discover_config(Some(&ss)).expect("vault key");
                assert_eq!(cfg.api_key, "sk-from-vault");
                assert_eq!(cfg.source, "vault");

                // Both present: env wins.
                with_var("ANTHROPIC_API_KEY", Some("sk-from-env"), || {
                    let cfg = discover_config(Some(&ss)).expect("env key");
                    assert_eq!(cfg.api_key, "sk-from-env");
                    assert_eq!(cfg.source, "ENV");
                });
            });
        });
    }

    #[test]
    #[serial_test::serial]
    fn test_api_keys_are_trimmed_of_surrounding_whitespace() {
        // Loop-2 S3 (Balthasar): a key with a trailing newline (a common
        // `export KEY=$(cat f)` artifact) or stray whitespace must be trimmed,
        // else the auth header is malformed (401). Both keys, both paths.
        with_var("ANTHROPIC_API_KEY", None, || {
            with_var("ANTHROPIC_MODEL", None, || {
                with_var("OPENAI_API_KEY", None, || {
                    let ss = vault_fixture();
                    {
                        let mut guard = ss.lock().unwrap();
                        guard
                            .set("ANTHROPIC_API_KEY", "  sk-vault-anthropic\n")
                            .unwrap();
                        guard.set("OPENAI_API_KEY", "sk-vault-openai\t").unwrap();
                    }
                    // Vault paths trim.
                    assert_eq!(
                        discover_config(Some(&ss)).expect("a").api_key,
                        "sk-vault-anthropic"
                    );
                    assert_eq!(
                        resolve_openai_key(Some(&ss)).as_deref(),
                        Some("sk-vault-openai")
                    );
                    // Env paths trim (and win over the vault).
                    with_var("OPENAI_API_KEY", Some("sk-env-openai\n"), || {
                        assert_eq!(
                            resolve_openai_key(Some(&ss)).as_deref(),
                            Some("sk-env-openai")
                        );
                    });
                    with_var("ANTHROPIC_API_KEY", Some(" sk-env-anthropic "), || {
                        assert_eq!(
                            discover_config(Some(&ss)).expect("b").api_key,
                            "sk-env-anthropic"
                        );
                    });
                });
            });
        });
    }

    #[test]
    #[serial_test::serial]
    fn test_resolve_openai_key_prefers_env_over_vault() {
        with_var("OPENAI_API_KEY", None, || {
            let ss = vault_fixture();
            assert!(resolve_openai_key(Some(&ss)).is_none());
            {
                let mut guard = ss.lock().unwrap();
                guard.set("OPENAI_API_KEY", "sk-oai-vault").unwrap();
            }
            assert_eq!(
                resolve_openai_key(Some(&ss)).as_deref(),
                Some("sk-oai-vault")
            );
            with_var("OPENAI_API_KEY", Some("sk-oai-env"), || {
                assert_eq!(resolve_openai_key(Some(&ss)).as_deref(), Some("sk-oai-env"));
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
}
