//! This module provides a security layer for path validation and sandboxing.

use anyhow::{anyhow, Result};
use std::io;
use std::path::{Component, Path, PathBuf};

/// Utility to ensure paths are safe and stay within the workspace boundary.
#[allow(dead_code)]
pub struct PathGuard {
    workspace_root: PathBuf,
}

#[allow(dead_code)]
impl PathGuard {
    /// Returns a reference to the workspace root.
    pub fn workspace_root(&self) -> &Path {
        &self.workspace_root
    }

    /// Creates a new `PathGuard` with the given workspace root.
    pub fn new(root: PathBuf) -> Result<Self> {
        let root = Self::canonicalize_robust(&root)
            .map_err(|e| anyhow!("Invalid workspace root: {}", e))?;
        Ok(Self {
            workspace_root: root,
        })
    }

    /// Validates and canonicalizes a path, ensuring it is within the workspace.
    /// Returns the fully canonicalized path to avoid TOCTOU races in the caller.
    pub fn validate(&self, input_path: &Path) -> Result<PathBuf> {
        let path_str = input_path.to_string_lossy();

        // Security: Disallow null bytes which can be used to bypass filename checks
        if path_str.contains('\0') {
            return Err(anyhow!("Security violation: null bytes in path"));
        }

        let absolute_path = if input_path.is_absolute() {
            input_path.to_path_buf()
        } else {
            self.workspace_root.join(input_path)
        };

        let normalized = self.lexical_normalize(&absolute_path);

        let mut current = normalized.as_path();
        let mut uncanonicalized_components = Vec::new();
        let mut closest_existing_ancestor = None;

        // Iteratively find the closest existing ancestor
        while closest_existing_ancestor.is_none() {
            if current.exists() {
                closest_existing_ancestor = Some(current);
            } else {
                if let Some(file_name) = current.file_name() {
                    uncanonicalized_components.push(file_name);
                }
                match current.parent() {
                    Some(p) => current = p,
                    None => break, // Reached volume root
                }
            }
        }

        if let Some(ancestor) = closest_existing_ancestor {
            let canonical_ancestor = Self::canonicalize_robust(ancestor)
                .map_err(|e| anyhow!("Security check failed: unreachable ancestor: {}", e))?;

            // Critical check: Ensure canonical ancestor is within workspace root
            if !canonical_ancestor.starts_with(&self.workspace_root) {
                return Err(anyhow!(
                    "Security violation: path escapes sandbox via traversal"
                ));
            }

            // Reconstruct the final path ensuring all parts are anchored safely
            let mut final_path = canonical_ancestor;
            for comp in uncanonicalized_components.into_iter().rev() {
                final_path.push(comp);
            }

            // Final check on reconstructed path
            if !final_path.starts_with(&self.workspace_root) {
                return Err(anyhow!("Security violation: final path escapes sandbox"));
            }

            Ok(final_path)
        } else {
            Err(anyhow!(
                "Security violation: path has no valid anchor in workspace"
            ))
        }
    }

    /// Purely lexical normalization of a path.
    /// Preserves RootDir and Prefix to prevent popping out of root boundaries.
    fn lexical_normalize(&self, path: &Path) -> PathBuf {
        let mut components = Vec::new();
        for component in path.components() {
            match component {
                Component::ParentDir => match components.last() {
                    Some(Component::Normal(_)) => {
                        components.pop();
                    }
                    Some(Component::RootDir) | Some(Component::Prefix(_)) => {}
                    _ => {
                        components.push(Component::ParentDir);
                    }
                },
                Component::CurDir => {}
                c => {
                    components.push(c);
                }
            }
        }
        components.iter().collect()
    }

    /// Robust canonicalization that handles Windows verbatim paths (UNC prefixes) consistently.
    fn canonicalize_robust(path: &Path) -> io::Result<PathBuf> {
        let canonical = path.canonicalize()?;

        #[cfg(windows)]
        {
            let path_str = canonical.to_string_lossy();
            if path_str.starts_with(r"\\?\") {
                return Ok(PathBuf::from(&path_str[4..]));
            }
        }

        Ok(canonical)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_path_validation() {
        let dir = tempdir().expect("Failed to create temp dir");
        let root = dir.path().to_path_buf();
        let guard = PathGuard::new(root.clone()).expect("Failed to create guard");

        // Safe path
        let safe_file = root.join("safe.txt");
        fs::write(&safe_file, "data").unwrap();
        assert!(guard.validate(&safe_file).is_ok());

        // Unsafe path (traversal)
        let unsafe_path = root.join("dir/../../outside.txt");
        assert!(guard.validate(&unsafe_path).is_err());

        // Null byte attack
        let null_path = Path::new("test\0file.txt");
        assert!(guard.validate(null_path).is_err());

        // Non-existent safe path with multiple levels
        let new_file = root.join("a/b/c/new_file.txt");
        assert!(guard.validate(&new_file).is_ok());
    }
}
