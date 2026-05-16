mod tools;
mod system;
mod agent;
mod tui;
mod services;
mod utils;

use std::sync::Arc;
use std::env;
use std::fs;
use clap::Parser;
use crate::agent::Agent;
use crate::agent::provider::{StaticProvider, AnthropicProvider, Provider};
use crate::system::fs::{FileSystem, RealFileSystem};
use crate::system::grep::RipGrep;
use crate::system::secrets::{SecretStore, KeyringStore};
use crate::system::database::{MemoryStore, EncryptedSqliteMemory};
use crate::tools::ls::ListTool;
use crate::tools::read::FileReadTool;
use crate::tools::write::FileWriteTool;
use crate::tools::grep::GrepTool;
use crate::tools::bash::BashTool;
use crate::tools::knowledge::ProjectFactTool;

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

async fn discover_config_ext(file_path: &str) -> Option<Config> {
    if let Ok(key) = env::var("ANTHROPIC_API_KEY") {
        let model = env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-sonnet-4-6".to_string());
        return Some(Config { api_key: key.trim().to_string(), model, source: "ENV".to_string() });
    }

    let primary_service = "magi-rs";
    let legacy_service = "magi-rust";
    let primary_store = KeyringStore::new(primary_service);
    let legacy_store = KeyringStore::new(legacy_service);

    if let Ok(Some(key)) = primary_store.get_secret("ANTHROPIC_API_KEY").await {
        let model = env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-sonnet-4-6".to_string());
        return Some(Config { api_key: key, model, source: format!("Keyring ({})", primary_service) });
    }

    if let Ok(Some(key)) = legacy_store.get_secret("ANTHROPIC_API_KEY").await {
        if primary_store.set_secret("ANTHROPIC_API_KEY", &key).await.is_ok() {
            let _ = legacy_store.delete_secret("ANTHROPIC_API_KEY").await;
        }
        let model = env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-sonnet-4-6".to_string());
        return Some(Config { api_key: key, model, source: format!("Keyring (Migrated from {})", legacy_service) });
    }

    if let Ok(content) = fs::read_to_string(file_path) {
        let lines: Vec<&str> = content.lines().collect();
        if let Some(key) = lines.get(0).map(|s| s.trim()).filter(|s| !s.is_empty()) {
            let model = lines.get(1).unwrap_or(&"claude-sonnet-4-6").trim().to_string();
            return Some(Config { api_key: key.to_string(), model, source: file_path.to_string() });
        }
    }
    None
}

async fn discover_or_create_master_key() -> anyhow::Result<String> {
    let primary_store = KeyringStore::new("magi-rs-internal");
    if let Ok(Some(key)) = primary_store.get_secret("DB_MASTER_KEY").await { return Ok(key); }

    let legacy_store = KeyringStore::new("magi-rust-internal");
    if let Ok(Some(key)) = legacy_store.get_secret("DB_MASTER_KEY").await {
        if primary_store.set_secret("DB_MASTER_KEY", &key).await.is_ok() {
            let _ = legacy_store.delete_secret("DB_MASTER_KEY").await;
        }
        return Ok(key);
    }

    use rand::{RngCore, thread_rng};
    use base64::{engine::general_purpose::STANDARD, Engine as _};
    let mut key_bytes = [0u8; 32];
    thread_rng().fill_bytes(&mut key_bytes);
    let new_key = STANDARD.encode(key_bytes);
    primary_store.set_secret("DB_MASTER_KEY", &new_key).await?;
    Ok(new_key)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let workspace_root = env::current_dir()?;

    if args.logout {
        let _ = KeyringStore::new("magi-rs").delete_secret("ANTHROPIC_API_KEY").await;
        let _ = KeyringStore::new("magi-rust").delete_secret("ANTHROPIC_API_KEY").await;
        println!("Logged out successfully.");
        return Ok(());
    }

    let config = discover_config_ext("key.txt").await;
    
    let (provider, provider_info): (Arc<dyn Provider>, String) = if let Some(ref c) = config {
        (
            Arc::new(AnthropicProvider::new(c.api_key.clone(), c.model.clone())),
            format!("Magi API ({}) Model: {}", c.source, c.model)
        )
    } else {
        (
            Arc::new(StaticProvider),
            "Static Mode (No API Key found. Please run /login or use ANTHROPIC_API_KEY)".to_string()
        )
    };

    let mut agent = Agent::new(provider);
    let db_path = workspace_root.join(".magi-rs-memory.db");
    let master_pwd = discover_or_create_master_key().await.unwrap_or_else(|_| "emergency-key".to_string());
    
    let memory: Arc<dyn MemoryStore> = Arc::new(EncryptedSqliteMemory::new(db_path, master_pwd)?);
    let sessions = memory.list_sessions().await?;
    let session_id = if let Some((id, _)) = sessions.first() { id.clone() } else { memory.create_session("default").await? };

    agent.set_memory(memory.clone(), session_id);
    let _ = agent.load_history().await;

    let fs: Arc<dyn FileSystem> = Arc::new(RealFileSystem::new());
    agent.register_tool(Box::new(ListTool::new(fs.clone(), workspace_root.clone())?));
    agent.register_tool(Box::new(FileReadTool::new(fs.clone(), workspace_root.clone())?));
    agent.register_tool(Box::new(FileWriteTool::new(fs.clone(), workspace_root.clone())?));
    agent.register_tool(Box::new(GrepTool::new(Box::new(RipGrep::new("rg")), workspace_root.clone())?));
    agent.register_tool(Box::new(BashTool::new(workspace_root.clone())?));
    agent.register_tool(Box::new(ProjectFactTool::new(memory.clone())));

    crate::tui::run_tui_ext(agent, Some(provider_info)).await?;
    Ok(())
}
