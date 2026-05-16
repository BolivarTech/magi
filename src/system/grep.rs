//! This module provides an abstraction for the Grep tool.

use async_trait::async_trait;
use std::path::Path;
use anyhow::Result;

#[cfg(test)]
use mockall::automock;

/// Trait defining the behavior of a pattern searcher.
#[async_trait]
#[cfg_attr(test, automock)]
pub trait Grep: Send + Sync {
    /// Searches for a pattern in a path.
    async fn search(&self, pattern: &str, path: &Path) -> Result<Vec<String>>;
}

/// A real implementation of Grep using RipGrep.
#[allow(dead_code)]
pub struct RipGrep {
    binary_path: String,
}

impl RipGrep {
    /// Creates a new `RipGrep` instance.
    pub fn new(binary_path: &str) -> Self {
        Self { binary_path: binary_path.to_string() }
    }
}

#[async_trait]
impl Grep for RipGrep {
    async fn search(&self, pattern: &str, path: &Path) -> Result<Vec<String>> {
        use tokio::process::Command;
        
        let output = Command::new(&self.binary_path)
            .arg(pattern)
            .arg(path)
            .output()
            .await?;

        if !output.status.success() && output.status.code() != Some(1) {
            return Err(anyhow::anyhow!("RipGrep failed: {}", String::from_utf8_lossy(&output.stderr)));
        }

        let results = String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(|s| s.to_string())
            .collect();

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_grep_streaming() {
        let mut mock = MockGrep::new();
        mock.expect_search()
            .times(1)
            .returning(|_, _| Box::pin(async move { Ok(vec!["match".to_string()]) }));
        
        let res = mock.search("test", Path::new(".")).await.unwrap();
        assert_eq!(res[0], "match");
    }
}
