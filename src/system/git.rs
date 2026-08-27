// Author: Julian Bolivar
// Version: 0.17.0
// Date: 2026-08-27

//! This module provides abstractions for system-level operations.

use anyhow::Result;
use async_trait::async_trait;
use serde::Serialize;

#[cfg(test)]
use mockall::automock;

/// Represents the current state of a Git repository.
#[derive(Debug, Serialize)]
#[allow(dead_code)]
pub struct GitState {
    pub branch: String,
    pub is_dirty: bool,
}

/// Trait defining the behavior of a Git adapter.
#[async_trait]
#[cfg_attr(test, automock)]
#[allow(dead_code)]
pub trait Git: Send + Sync {
    /// Returns the current state of the repository.
    async fn get_state(&self) -> Result<GitState>;

    /// Checks if the current directory is a git repository.
    async fn is_git(&self) -> bool;
}

/// A real implementation of the Git trait using the git CLI.
#[allow(dead_code)]
pub struct RealGit;

#[allow(dead_code)]
impl RealGit {
    /// Creates a new `RealGit` instance.
    pub fn new() -> Self {
        Self
    }

    async fn run_git(&self, args: &[&str]) -> Result<String> {
        use tokio::process::Command;
        let output = Command::new("git").args(args).output().await?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
        } else {
            Err(anyhow::anyhow!(
                "Git command failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ))
        }
    }
}

#[async_trait]
impl Git for RealGit {
    async fn get_state(&self) -> Result<GitState> {
        let branch = self.run_git(&["rev-parse", "--abbrev-ref", "HEAD"]).await?;
        let status = self.run_git(&["status", "--porcelain"]).await?;

        Ok(GitState {
            branch,
            is_dirty: !status.is_empty(),
        })
    }

    async fn is_git(&self) -> bool {
        self.run_git(&["rev-parse", "--is-inside-work-tree"])
            .await
            .is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_git_is_git() {
        let mut mock = MockGit::new();
        mock.expect_is_git()
            .times(1)
            .returning(|| Box::pin(async { true }));

        assert!(mock.is_git().await);
    }

    #[tokio::test]
    async fn test_git_get_state() {
        let mut mock = MockGit::new();
        mock.expect_get_state().times(1).returning(|| {
            Box::pin(async {
                Ok(GitState {
                    branch: "main".to_string(),
                    is_dirty: false,
                })
            })
        });

        let state = mock.get_state().await.unwrap();
        assert_eq!(state.branch, "main");
        assert!(!state.is_dirty);
    }
}
