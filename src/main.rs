mod agent;
mod config;
mod services;
mod system;
mod tools;
mod tui;
mod utils;

use crate::agent::provider::{
    AnthropicProvider, OpenAiCompatibleProvider, OpenAiSettings, Provider, StaticProvider,
};
use crate::agent::Agent;
use crate::config::{resolve_openai_base_url, resolve_openai_model, resolve_provider, MagiConfig};
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
use std::env;
use std::fs;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Log out and clear stored API keys.
    #[arg(short, long)]
    logout: bool,
}

#[derive(Debug)]
struct Config {
    api_key: String,
    model: String,
    source: String,
}

/// Default model when none is configured via `ANTHROPIC_MODEL` or `key.txt`.
/// Single source of truth — bump here to change the default everywhere.
/// `pub(crate)` so the TUI `/login` handler reuses it when rebuilding the
/// provider in-session (#9).
pub(crate) const DEFAULT_MODEL: &str = "claude-sonnet-4-6";

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

    let config = discover_config_ext("key.txt").await;
    let (magi_config, config_warning) = MagiConfig::load(&workspace_root);
    let provider_kind = resolve_provider(&magi_config, env::var("MAGI_PROVIDER").ok().as_deref());

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
                resolve_openai_model(&magi_config, env::var("OPENAI_MODEL").ok().as_deref())?;
            let info = format!("OpenAI-compatible ({base_url}) Model: {model}");
            let model_label = model.clone();
            (
                Arc::new(OpenAiCompatibleProvider::new(OpenAiSettings {
                    base_url,
                    api_key,
                    model,
                })),
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

    // Build the MAGI orchestrator over the SAME resolved backend (shared Arc) so
    // the consult tool reuses the resolved credentials. Skip on the StaticProvider
    // path: magi-core cannot parse canned text, so a consult would deterministically
    // fail with InsufficientAgents.
    let consult_magi: Option<std::sync::Arc<magi_core::orchestrator::Magi>> =
        if provider.is_static() {
            None
        } else {
            let provider_label = if provider_kind == "openai" {
                "openai"
            } else {
                "anthropic"
            };
            let adapter = crate::agent::magi_adapter::MagiCoreProviderAdapter::new(
                provider.clone(),
                provider_label,
                model_label,
            );
            Some(std::sync::Arc::new(magi_core::orchestrator::Magi::new(
                std::sync::Arc::new(adapter),
            )))
        };

    let mut agent = Agent::new(provider);
    let db_path = workspace_root.join(".magi-rs-memory.db");
    // Notices shown when the TUI starts — the provider banner plus any persistence
    // or reset warnings that would otherwise be lost to pre-TUI stderr (#7/#11).
    let mut startup_notices = vec![provider_info];
    // MAGI fix f: surface malformed/unreadable magi.toml in the TUI rather than
    // losing it to pre-TUI stderr — same path as the persistence/reset notices.
    if let Some(w) = config_warning {
        startup_notices.push(w);
    }
    match decide_memory_attachment(discover_or_create_master_key().await) {
        MemoryAttachment::Encrypted(master_pwd) => {
            let store = EncryptedSqliteMemory::new(db_path, master_pwd)?;
            // #11: surface a one-time reset notice if incompatible content was discarded.
            if store.was_reset() {
                startup_notices.push(
                    "Note: existing on-disk history used an incompatible/corrupt format and \
                     has been reset (fresh start)."
                        .to_string(),
                );
            }
            let memory: Arc<dyn MemoryStore> = Arc::new(store);
            let sessions = memory.list_sessions().await?;
            let session_id = if let Some((id, _)) = sessions.first() {
                id.clone()
            } else {
                memory.create_session("default").await?
            };
            agent.set_memory(memory.clone(), session_id);
            let _ = agent.load_history().await;

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

    crate::tui::run_tui_ext(agent, startup_notices).await?;
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
        // Wiring smoke test (Task 6): env > TOML > default "anthropic".
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
        assert_eq!(resolve_provider(&MagiConfig::default(), None), "anthropic");
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
}
