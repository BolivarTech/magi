//! This module provides a persistent memory system based on SQLite with encryption.

use async_trait::async_trait;
use anyhow::Result;
use crate::agent::messages::Message;
use std::path::PathBuf;
use rusqlite::{params, Connection};
use crate::utils::crypto::CryptoVault;
use std::sync::{Arc, Mutex};

/// Trait defining the behavior of the agent's memory.
#[async_trait]
pub trait MemoryStore: Send + Sync {
    /// Creates a new session and returns its ID.
    async fn create_session(&self, project_name: &str) -> Result<String>;
    
    /// Adds a message to a specific session.
    async fn add_message(&self, session_id: &str, message: &Message) -> Result<()>;
    
    /// Retrieves all messages for a session.
    async fn get_messages(&self, session_id: &str) -> Result<Vec<Message>>;
    
    /// Lists all sessions.
    async fn list_sessions(&self) -> Result<Vec<(String, String)>>; // (id, project_name)

    /// Stores a persistent fact about the project.
    async fn set_knowledge(&self, key: &str, value: &str) -> Result<()>;

    /// Retrieves a persistent fact.
    async fn get_knowledge(&self, key: &str) -> Result<Option<String>>;

    /// Lists all known project keys.
    async fn list_knowledge_keys(&self) -> Result<Vec<String>>;
}

/// A persistent memory store using SQLite and CryptoVault for encryption.
pub struct EncryptedSqliteMemory {
    conn: Arc<Mutex<Connection>>,
    vault: CryptoVault,
    master_password: String,
}

impl EncryptedSqliteMemory {
    pub fn new(path: PathBuf, master_password: String) -> Result<Self> {
        let conn = Connection::open(path)?;
        
        // MAGI FIX: Enable WAL mode for high concurrency
        // We use query_row because execute fails for pragmas that return values in some drivers
        let _: String = conn.query_row("PRAGMA journal_mode = WAL", [], |row| row.get(0))?;
        conn.execute("PRAGMA synchronous = NORMAL", [])?;
        // Set a busy timeout to prevent "database is locked" errors during contention
        conn.busy_timeout(std::time::Duration::from_secs(5))?;

        // Initialize Schema
        conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                project_name TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;
        
        conn.execute(
            "CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content_blob TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS knowledge (
                key TEXT PRIMARY KEY,
                value_blob TEXT NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            vault: CryptoVault::default(),
            master_password,
        })
    }
}

#[async_trait]
impl MemoryStore for EncryptedSqliteMemory {
    async fn create_session(&self, project_name: &str) -> Result<String> {
        let id = uuid::Uuid::new_v4().to_string();
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO sessions (id, project_name) VALUES (?1, ?2)",
            params![id, project_name],
        )?;
        Ok(id)
    }

    async fn add_message(&self, session_id: &str, message: &Message) -> Result<()> {
        let json_content = serde_json::to_string(&message.content)?;
        let encrypted = self.vault.encrypt(&self.master_password, &json_content)
            .map_err(|e| anyhow::anyhow!("Encryption failed: {}", e))?;
        
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO messages (session_id, role, content_blob) VALUES (?1, ?2, ?3)",
            params![session_id, format!("{:?}", message.role), encrypted],
        )?;
        Ok(())
    }

    async fn get_messages(&self, session_id: &str) -> Result<Vec<Message>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT role, content_blob FROM messages WHERE session_id = ? ORDER BY created_at ASC")?;
        
        let rows = stmt.query_map(params![session_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
            ))
        })?;

        let mut messages = Vec::new();
        for row in rows {
            let (role_str, blob) = row?;
            let decrypted = self.vault.decrypt(&self.master_password, &blob)
                .map_err(|e| anyhow::anyhow!("Decryption failed: {}", e))?;
            
            let content = serde_json::from_str(&decrypted)?;
            let role = match role_str.as_str() {
                "User" => crate::agent::messages::Role::User,
                _ => crate::agent::messages::Role::Assistant,
            };
            
            messages.push(Message { role, content });
        }
        
        Ok(messages)
    }

    async fn list_sessions(&self) -> Result<Vec<(String, String)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT id, project_name FROM sessions ORDER BY created_at DESC")?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?;
        
        let mut sessions = Vec::new();
        for row in rows {
            sessions.push(row?);
        }
        Ok(sessions)
    }

    async fn set_knowledge(&self, key: &str, value: &str) -> Result<()> {
        let encrypted = self.vault.encrypt(&self.master_password, value)
            .map_err(|e| anyhow::anyhow!("Encryption failed: {}", e))?;
        
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO knowledge (key, value_blob, updated_at) VALUES (?1, ?2, CURRENT_TIMESTAMP)",
            params![key, encrypted],
        )?;
        Ok(())
    }

    async fn get_knowledge(&self, key: &str) -> Result<Option<String>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT value_blob FROM knowledge WHERE key = ?")?;
        
        let res = stmt.query_row(params![key], |row| row.get::<_, String>(0));
        
        match res {
            Ok(blob) => {
                let decrypted = self.vault.decrypt(&self.master_password, &blob)
                    .map_err(|e| anyhow::anyhow!("Decryption failed: {}", e))?;
                Ok(Some(decrypted))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(anyhow::anyhow!("Database error: {}", e)),
        }
    }

    async fn list_knowledge_keys(&self) -> Result<Vec<String>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT key FROM knowledge ORDER BY key ASC")?;
        let rows = stmt.query_map([], |row| row.get(0))?;
        
        let mut keys = Vec::new();
        for row in rows {
            keys.push(row?);
        }
        Ok(keys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_encrypted_sqlite_memory() {
        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path().to_path_buf();
        let password = "master_key_123";
        
        let memory = EncryptedSqliteMemory::new(path, password.to_string()).unwrap();
        let sid = memory.create_session("test_proj").await.unwrap();
        
        let msg = Message::user("Hello secure world");
        memory.add_message(&sid, &msg).await.unwrap();
        
        let msgs = memory.get_messages(&sid).await.unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0], msg);
        
        // Verify encryption (raw read)
        let conn = Connection::open(tmp_file.path()).unwrap();
        let blob: String = conn.query_row("SELECT content_blob FROM messages LIMIT 1", [], |r| r.get(0)).unwrap();
        assert!(!blob.contains("Hello"), "Database should contain encrypted blob, not plaintext");

        // Verify list_sessions (to clear dead code warning)
        let sessions = memory.list_sessions().await.unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].1, "test_proj");
    }

    #[tokio::test]
    async fn test_project_knowledge_persistence() {
        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path().to_path_buf();
        let password = "knowledge_key_123".to_string();
        
        let memory = EncryptedSqliteMemory::new(path, password).unwrap();
        
        memory.set_knowledge("architecture", "Clean hex with encrypted SQLite").await.unwrap();
        
        let fact = memory.get_knowledge("architecture").await.unwrap();
        assert_eq!(fact.unwrap(), "Clean hex with encrypted SQLite");
        
        // Verify multiple keys
        memory.set_knowledge("port", "54545").await.unwrap();
        let keys = memory.list_knowledge_keys().await.unwrap();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"architecture".to_string()));
        assert!(keys.contains(&"port".to_string()));
    }

    #[tokio::test]
    async fn test_sqlite_concurrency_stress() {
        let tmp_file = tempfile::NamedTempFile::new().unwrap();
        let path = tmp_file.path().to_path_buf();
        let memory = Arc::new(EncryptedSqliteMemory::new(path, "stress_pass".to_string()).unwrap());

        let mut handles = vec![];
        for i in 0..20 {
            let mem_clone = memory.clone();
            handles.push(tokio::spawn(async move {
                let key = format!("key_{}", i);
                let val = format!("val_{}", i);
                mem_clone.set_knowledge(&key, &val).await
            }));
        }

        for h in handles {
            let res = h.await.unwrap();
            assert!(res.is_ok(), "Concurrent write failed: {:?}", res.err());
        }
        
        let keys = memory.list_knowledge_keys().await.unwrap();
        assert_eq!(keys.len(), 20);
    }
}
