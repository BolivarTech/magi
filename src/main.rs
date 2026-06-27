mod agent;
mod config;
mod defaults;
mod memory;
mod services;
mod system;
mod tools;
mod tui;
mod utils;

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
use crate::system::secrets::{KeyringStore, SecretStore};
use crate::tools::bash::BashTool;
use crate::tools::grep::GrepTool;
use crate::tools::knowledge::ProjectFactTool;
use crate::tools::ls::ListTool;
use crate::tools::read::FileReadTool;
use crate::tools::write::FileWriteTool;
use clap::Parser;
use magi_core::orchestrator::{Magi, MagiBuilder};
use std::env;
use std::fs;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Log out and clear stored API keys.
    #[arg(short, long)]
    logout: bool,

    /// Write a default magi.toml to the workspace and exit (refuses to overwrite).
    #[arg(long)]
    init_config: bool,
}

#[derive(Debug)]
struct Config {
    api_key: String,
    model: String,
    source: String,
}

/// Default Anthropic model when none is configured via `ANTHROPIC_MODEL` or
/// `key.txt`. Single source of truth lives in `crate::defaults`; this alias keeps
/// existing call sites (parse_key_file, discover_config_ext, the TUI `/login`
/// handler) and tests working unchanged.
pub(crate) use crate::defaults::DEFAULT_ANTHROPIC_MODEL as DEFAULT_MODEL;

/// Returns `true` when `base_url` resolves to a local address
/// (`localhost`, `127.0.0.1`, or `[::1]`) — used by the cloud-egress notice
/// (CP2-AG/AJ, Task 13b) to suppress the warning for on-device embedders.
fn is_localhost(base_url: &str) -> bool {
    let lower = base_url.to_lowercase();
    lower.contains("localhost") || lower.contains("127.0.0.1") || lower.contains("[::1]")
}

/// Parses `key.txt`-style content: line 1 = API key, line 2 = optional model.
///
/// Returns `(api_key, model)`. A blank, whitespace-only, or absent model line
/// falls back to [`DEFAULT_MODEL`]. Returns `None` when there is no non-empty
/// key line.
fn parse_key_file(content: &str) -> Option<(String, String)> {
    let lines: Vec<&str> = content.lines().collect();
    let key = lines.first().map(|s| s.trim()).filter(|s| !s.is_empty())?;
    let model = lines
        .get(1)
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .unwrap_or(DEFAULT_MODEL)
        .to_string();
    Some((key.to_string(), model))
}

async fn discover_config_ext(file_path: &str) -> Option<Config> {
    if let Ok(key) = env::var("ANTHROPIC_API_KEY") {
        let model = env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
        return Some(Config {
            api_key: key.trim().to_string(),
            model,
            source: "ENV".to_string(),
        });
    }

    let primary_service = "magi-rs";
    let legacy_service = "magi-rust";
    let primary_store = KeyringStore::new(primary_service);
    let legacy_store = KeyringStore::new(legacy_service);

    if let Ok(Some(key)) = primary_store.get_secret("ANTHROPIC_API_KEY").await {
        let model = env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
        return Some(Config {
            api_key: key,
            model,
            source: format!("Keyring ({})", primary_service),
        });
    }

    if let Ok(Some(key)) = legacy_store.get_secret("ANTHROPIC_API_KEY").await {
        if primary_store
            .set_secret("ANTHROPIC_API_KEY", &key)
            .await
            .is_ok()
        {
            let _ = legacy_store.delete_secret("ANTHROPIC_API_KEY").await;
        }
        let model = env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
        return Some(Config {
            api_key: key,
            model,
            source: format!("Keyring (Migrated from {})", legacy_service),
        });
    }

    if let Ok(content) = fs::read_to_string(file_path) {
        if let Some((api_key, model)) = parse_key_file(&content) {
            return Some(Config {
                api_key,
                model,
                source: file_path.to_string(),
            });
        }
    }
    None
}

async fn discover_or_create_master_key() -> anyhow::Result<String> {
    let primary_store = KeyringStore::new("magi-rs-internal");
    if let Ok(Some(key)) = primary_store.get_secret("DB_MASTER_KEY").await {
        return Ok(key);
    }

    let legacy_store = KeyringStore::new("magi-rust-internal");
    if let Ok(Some(key)) = legacy_store.get_secret("DB_MASTER_KEY").await {
        if primary_store
            .set_secret("DB_MASTER_KEY", &key)
            .await
            .is_ok()
        {
            let _ = legacy_store.delete_secret("DB_MASTER_KEY").await;
        }
        return Ok(key);
    }

    use base64::{engine::general_purpose::STANDARD, Engine as _};
    use rand::{thread_rng, RngCore};
    let mut key_bytes = [0u8; 32];
    thread_rng().fill_bytes(&mut key_bytes);
    let new_key = STANDARD.encode(key_bytes);
    primary_store.set_secret("DB_MASTER_KEY", &new_key).await?;
    Ok(new_key)
}

/// Decision on whether to attach encrypted persistent memory.
///
/// Maps the result of [`discover_or_create_master_key`] to an attachment mode.
/// On `Ok`, the recovered master password is used to attach the encrypted
/// SQLite store. On `Err` (e.g. an inaccessible OS keyring), the agent runs
/// **ephemerally** — no persistence — rather than ever falling back to a
/// constant passphrase, which would silently weaken encryption of every
/// future record (audit finding C4).
#[derive(Debug)]
enum MemoryAttachment {
    /// Attach encrypted memory using the recovered master password.
    Encrypted(String),
    /// Run without persistence (in-memory history only).
    Ephemeral,
}

/// Decides the memory-attachment mode from the master-key discovery result.
///
/// # Parameters
/// - `key_result`: the outcome of `discover_or_create_master_key().await`.
///
/// # Returns
/// `MemoryAttachment::Encrypted(pwd)` when a key was recovered, otherwise
/// `MemoryAttachment::Ephemeral`. Never returns a synthesized/constant key.
fn decide_memory_attachment(key_result: anyhow::Result<String>) -> MemoryAttachment {
    match key_result {
        Ok(master_pwd) => MemoryAttachment::Encrypted(master_pwd),
        Err(_) => MemoryAttachment::Ephemeral,
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let workspace_root = env::current_dir()?;

    if args.logout {
        let _ = KeyringStore::new("magi-rs")
            .delete_secret("ANTHROPIC_API_KEY")
            .await;
        let _ = KeyringStore::new("magi-rust")
            .delete_secret("ANTHROPIC_API_KEY")
            .await;
        println!("Logged out successfully.");
        return Ok(());
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

    let config = discover_config_ext("key.txt").await;
    let (magi_config, config_warning) = MagiConfig::load(&workspace_root);
    let provider_kind = resolve_provider(&magi_config, env::var("MAGI_PROVIDER").ok().as_deref());

    // Credentials needed to build per-agent sibling providers (same backend, different
    // model) for MAGI per-agent overrides. Set inside the openai branch.
    let mut oai_creds: Option<(String, String)> = None; // (base_url, api_key)

    let (provider, provider_info, model_label): (Arc<dyn Provider>, String, String) =
        if provider_kind == "openai" {
            // MAGI: OPENAI_API_KEY is sourced from env ONLY — NEVER from magi.toml
            // (security invariant: secrets do not live in plain-text config). The
            // "ollama" fallback is a dummy accepted by local Ollama which ignores
            // the Authorization header; real OpenAI/Groq/OpenRouter requests will
            // fail loudly with 401 if the env var is unset, which is the correct
            // behavior — no silent insecure default.
            let api_key = env::var("OPENAI_API_KEY").unwrap_or_else(|_| "ollama".to_string());
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
                "Static Mode: no API key found. Set ANTHROPIC_API_KEY or use key.txt \
                 (recommended). /login (OAuth) is best-effort and may be rate-limited."
                    .to_string(),
                "static".to_string(),
            )
        };

    // Notices shown when the TUI starts — the provider banner plus any persistence
    // or reset warnings that would otherwise be lost to pre-TUI stderr (#7/#11).
    let mut startup_notices = vec![provider_info];
    // MAGI fix f: surface malformed/unreadable magi.toml in the TUI rather than
    // losing it to pre-TUI stderr — same path as the persistence/reset notices.
    if let Some(w) = config_warning {
        startup_notices.push(w);
    }
    // B1: surface invalid memory-config values as a startup notice (never panic).
    if let Err(e) = magi_config.memory.validate() {
        startup_notices.push(format!("memory config warning: {e}"));
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
    let db_path = workspace_root.join(".magi-rs-memory.db");
    match decide_memory_attachment(discover_or_create_master_key().await) {
        MemoryAttachment::Encrypted(master_pwd) => {
            // Use a concrete binding first so we can call the pub(crate) accessors
            // (`shared_conn`, `data_key`) before the type is erased by
            // `Arc<dyn MemoryStore>`.  This is the Task-12 wiring described in
            // the brief: both stores share the same on-disk SQLite file.
            let concrete_store = EncryptedSqliteMemory::new(db_path, master_pwd)?;

            // Build the vector store from the shared connection + derived key.
            // Errors here are non-fatal: fall through without the tiered-memory
            // subsystem rather than refusing to start (REQ-29).
            let vstore_result =
                SqliteVectorStore::new(concrete_store.shared_conn(), concrete_store.data_key());

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
            // successfully.  `OPENAI_API_KEY` is the embedding API key (may be the
            // dummy `"ollama"` for the local Ollama server — it ignores auth).
            if let Ok(vstore) = vstore_result {
                let embedder = Arc::new(OpenAiCompatibleEmbedder::new(
                    &magi_config.embedding,
                    env::var("OPENAI_API_KEY").ok(),
                ));
                let clock = Arc::new(SystemClock);
                // Keep a reference for the startup diagnostics line (CP2-AN/S).
                let vstore = Arc::new(vstore);
                let vstore_diag = Arc::clone(&vstore);
                agent.set_memory_subsystem(vstore, embedder, clock, magi_config.memory.clone());
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
                        magi_config.memory.distill_max_batch_tokens, magi_config.embedding.base_url,
                    ));
                }
            }

            // ProjectFactTool needs the same store; register it on the encrypted path only.
            agent.register_tool(Box::new(ProjectFactTool::new(memory.clone())));
        }
        MemoryAttachment::Ephemeral => {
            // #7: surface the no-persistence state in the TUI, not just pre-TUI stderr.
            startup_notices.push(
                "WARNING: the OS keyring is unavailable, so this session runs WITHOUT \
                 persistence — your conversation and project knowledge will NOT be saved \
                 (any existing on-disk database is left untouched). Run /login or check \
                 your OS keyring to restore persistence."
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
        )));
    }

    crate::tui::run_tui_ext(agent, startup_notices, consult_magi, workspace_root).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_master_key_present_attaches_encrypted_memory() {
        let outcome = decide_memory_attachment(Ok("real-master-key".to_string()));
        match outcome {
            MemoryAttachment::Encrypted(pwd) => assert_eq!(pwd, "real-master-key"),
            MemoryAttachment::Ephemeral => {
                panic!("expected encrypted attachment when key is present")
            }
        }
    }

    #[test]
    fn test_master_key_error_degrades_to_ephemeral_without_constant() {
        let outcome = decide_memory_attachment(Err(anyhow::anyhow!("keyring inaccessible")));
        assert!(
            matches!(outcome, MemoryAttachment::Ephemeral),
            "a keyring failure must degrade to an ephemeral session, never to a constant key"
        );
        if let MemoryAttachment::Encrypted(pwd) =
            decide_memory_attachment(Err(anyhow::anyhow!("x")))
        {
            panic!("error path produced a passphrase: {pwd}");
        }
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
    fn test_parse_key_file_falls_back_to_default_model_on_blank_line() {
        // Line 2 present with a model -> that model is used.
        assert_eq!(
            parse_key_file("sk-ant-xyz\nclaude-opus-4-7\n"),
            Some(("sk-ant-xyz".to_string(), "claude-opus-4-7".to_string()))
        );
        // Line 2 blank / whitespace / absent -> DEFAULT_MODEL (the bug fix).
        for content in [
            "sk-ant-xyz\n\n",
            "sk-ant-xyz\n   \n",
            "sk-ant-xyz\n",
            "sk-ant-xyz",
        ] {
            assert_eq!(
                parse_key_file(content),
                Some(("sk-ant-xyz".to_string(), DEFAULT_MODEL.to_string())),
                "blank/absent model line must fall back to DEFAULT_MODEL (content: {content:?})"
            );
        }
        // No usable key line -> None.
        assert_eq!(parse_key_file(""), None);
        assert_eq!(parse_key_file("   \n"), None);
        // The configured default.
        assert_eq!(DEFAULT_MODEL, "claude-sonnet-4-6");
    }

    #[test]
    fn test_args_parses_init_config_flag() {
        use clap::Parser;
        let a = Args::parse_from(["magi-rs", "--init-config"]);
        assert!(a.init_config);
        let b = Args::parse_from(["magi-rs"]);
        assert!(!b.init_config);
    }
}
