//! This module provides a secure abstraction for storing secrets.

use async_trait::async_trait;
use anyhow::Result;
use keyring::Entry;

#[cfg(test)]
use mockall::automock;

/// Trait defining the behavior of a secure secret store.
#[async_trait]
#[cfg_attr(test, automock)]
pub trait SecretStore: Send + Sync {
    /// Stores a secret value under a specific key.
    async fn set_secret(&self, key: &str, value: &str) -> Result<()>;
    
    /// Retrieves a secret value by its key.
    async fn get_secret(&self, key: &str) -> Result<Option<String>>;
    
    /// Removes a secret from the store.
    async fn delete_secret(&self, key: &str) -> Result<()>;
}

/// A real implementation of SecretStore using the OS keyring.
pub struct KeyringStore {
    service_name: String,
}

impl KeyringStore {
    pub fn new(service_name: &str) -> Self {
        Self {
            service_name: service_name.to_string(),
        }
    }
}

#[async_trait]
impl SecretStore for KeyringStore {
    async fn set_secret(&self, key: &str, value: &str) -> Result<()> {
        let entry = Entry::new(&self.service_name, key)
            .map_err(|e| anyhow::anyhow!("Failed to create keyring entry: {}", e))?;
        entry.set_password(value)
            .map_err(|e| anyhow::anyhow!("Failed to store secret: {}", e))?;
        Ok(())
    }

    async fn get_secret(&self, key: &str) -> Result<Option<String>> {
        let entry = Entry::new(&self.service_name, key)
            .map_err(|e| anyhow::anyhow!("Failed to create keyring entry: {}", e))?;
        match entry.get_password() {
            Ok(p) => Ok(Some(p)),
            Err(keyring::Error::NoEntry) => Ok(None),
            Err(e) => Err(anyhow::anyhow!("Failed to retrieve secret: {}", e)),
        }
    }

    async fn delete_secret(&self, key: &str) -> Result<()> {
        let entry = Entry::new(&self.service_name, key)
            .map_err(|e| anyhow::anyhow!("Failed to create keyring entry: {}", e))?;
        match entry.delete_password() {
            Ok(_) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(e) => Err(anyhow::anyhow!("Failed to delete secret: {}", e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_keyring_store_basic_ops() {
        // Use a very specific service name for testing to avoid conflicts
        let store = KeyringStore::new("magi-rs-test-suite");
        let key = "test_key_123";
        let value = "super_secret_value_abc";

        // Clean up before starting
        let _ = store.delete_secret(key).await;

        // Set
        store.set_secret(key, value).await.unwrap();
        
        // Get
        let retrieved = store.get_secret(key).await.unwrap();
        assert_eq!(retrieved, Some(value.to_string()));
        
        // Delete
        store.delete_secret(key).await.unwrap();
        let deleted = store.get_secret(key).await.unwrap();
        assert!(deleted.is_none());
    }
}
