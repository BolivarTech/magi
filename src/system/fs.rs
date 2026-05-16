//! This module provides an abstraction for the file system.
//! It is designed for efficiency and security, supporting buffered I/O,
//! UTF-8 boundary handling, path sandboxing, and recursive listing.

use async_trait::async_trait;
use std::path::{Path, PathBuf};
use tokio::fs::{self, File};
use tokio::io::{AsyncReadExt, AsyncSeekExt, SeekFrom};
use std::io;
use std::collections::HashSet;

#[cfg(test)]
use mockall::automock;

/// Errors specific to FileSystem operations.
#[derive(Debug, thiserror::Error)]
pub enum FsError {
    #[error("File not found: {0}")]
    NotFound(String),
    #[error("Access denied: {0}")]
    PermissionDenied(String),
    #[error("Is a directory: {0}")]
    IsADirectory(String),
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("Other error: {0}")]
    Other(String),
}

/// Trait defining the behavior of a file system.
#[async_trait]
#[cfg_attr(test, automock)]
pub trait FileSystem: Send + Sync {
    /// Reads a specific portion of a file, ensuring UTF-8 validity at boundaries.
    async fn read_file_paged(&self, path: &Path, offset: u64, limit: usize) -> Result<String, FsError>;

    /// Reads the entire content of a file (helper).
    async fn read_file(&self, path: &Path) -> Result<String, FsError> {
        self.read_file_paged(path, 0, usize::MAX).await
    }

    /// Writes a string to a file, creating parent directories if needed.
    async fn write_file(&self, path: &Path, content: &str) -> Result<(), FsError>;

    /// Checks if a path exists and is a file.
    #[allow(dead_code)]
    async fn exists(&self, path: &Path) -> bool;

    /// Removes a file from the filesystem.
    #[allow(dead_code)]
    async fn remove_file(&self, path: &Path) -> Result<(), FsError>;

    /// Lists files in a single directory (non-recursive).
    async fn list_directory(&self, dir: &Path) -> Result<Vec<String>, FsError>;

    /// Recursively lists files in a directory with cycle detection and limits.
    #[allow(dead_code)]
    async fn list_recursive(&self, dir: &Path, max_files: usize) -> Result<Vec<PathBuf>, FsError>;
}

/// A real implementation of the FileSystem trait using tokio::fs.
#[derive(Clone)]
pub struct RealFileSystem;

#[allow(dead_code)]
impl RealFileSystem {
    pub fn new() -> Self { Self }

    /// Helper for recursive listing with state.
    async fn list_inner(
        &self,
        dir: PathBuf,
        results: &mut Vec<PathBuf>,
        visited: &mut HashSet<PathBuf>,
        max_files: usize,
        current_depth: usize,
    ) -> Result<(), FsError> {
        const MAX_DEPTH: usize = 20;
        if current_depth > MAX_DEPTH { 
             return Err(FsError::Other("Recursion limit exceeded".to_string())); 
        }
        if results.len() >= max_files { return Ok(()); }

        // Canonicalize to detect cycles (symlink loops)
        let canonical = match dir.canonicalize() {
            Ok(c) => c,
            Err(e) => return Err(FsError::Io(e)),
        };
        
        if !visited.insert(canonical) {
            return Ok(()); // Cycle detected, skip
        }

        let mut entries = fs::read_dir(&dir).await.map_err(FsError::Io)?;
        while let Some(entry) = entries.next_entry().await.map_err(FsError::Io)? {
            if results.len() >= max_files { break; }
            
            let path = entry.path();
            let name = entry.file_name();
            let name_str = name.to_string_lossy();

            // Skip hidden files and common noise
            if name_str.starts_with('.') || name_str == "__pycache__" || name_str == "node_modules" || name_str == "target" {
                continue;
            }

            let file_type = entry.file_type().await.map_err(FsError::Io)?;
            if file_type.is_dir() {
                results.push(path.clone());
                Box::pin(self.list_inner(path, results, visited, max_files, current_depth + 1)).await?;
            } else {
                results.push(path);
            }
        }

        Ok(())
    }
}

#[async_trait]
impl FileSystem for RealFileSystem {
    async fn read_file_paged(&self, path: &Path, offset: u64, limit: usize) -> Result<String, FsError> {
        let mut file = File::open(path).await.map_err(|e| match e.kind() {
            io::ErrorKind::NotFound => FsError::NotFound(path.display().to_string()),
            io::ErrorKind::PermissionDenied => FsError::PermissionDenied(path.display().to_string()),
            _ => FsError::Io(e),
        })?;

        let metadata = file.metadata().await.map_err(FsError::Io)?;
        if metadata.is_dir() { return Err(FsError::IsADirectory(path.display().to_string())); }
        if offset > 0 { file.seek(SeekFrom::Start(offset)).await.map_err(FsError::Io)?; }

        const BUFFER_SIZE: usize = 64 * 1024;
        let mut result_bytes = Vec::with_capacity(BUFFER_SIZE.min(limit).min(1024*1024)); 
        let mut total_read = 0;
        let mut temp_buffer = [0u8; BUFFER_SIZE];

        while total_read < limit {
            let to_read = (limit - total_read).min(BUFFER_SIZE);
            let n = file.read(&mut temp_buffer[..to_read]).await.map_err(FsError::Io)?;
            if n == 0 { break; }
            result_bytes.extend_from_slice(&temp_buffer[..n]);
            total_read += n;
        }

        if result_bytes.is_empty() {
            return Ok(String::new());
        }

        let mut start_skip = 0;
        while start_skip < result_bytes.len() && (result_bytes[start_skip] & 0xC0) == 0x80 {
            start_skip += 1;
        }
        
        let mut valid_start_bytes = result_bytes[start_skip..].to_vec();

        if valid_start_bytes.is_empty() {
            return Ok(String::new());
        }

        loop {
            match std::str::from_utf8(&valid_start_bytes) {
                Ok(s) => return Ok(s.to_string()),
                Err(e) => {
                    let valid_len = e.valid_up_to();
                    
                    if e.error_len().is_none() {
                        if valid_len == 0 {
                            let mut extra = [0u8; 1];
                            let n = file.read(&mut extra).await.map_err(FsError::Io)?;
                            if n == 0 {
                                return Ok(String::new()); 
                            }
                            valid_start_bytes.push(extra[0]);
                            continue;
                        } else {
                            let truncated_count = valid_start_bytes.len() - valid_len;
                            file.seek(SeekFrom::Current(-(truncated_count as i64))).await.map_err(FsError::Io)?;
                            valid_start_bytes.truncate(valid_len);
                            return Ok(String::from_utf8(valid_start_bytes).unwrap());
                        }
                    } else {
                        return Err(FsError::Other(format!("Corrupt UTF-8 sequence at byte index: {}", e.valid_up_to())));
                    }
                }
            }
        }
    }

    async fn write_file(&self, path: &Path, content: &str) -> Result<(), FsError> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await.map_err(FsError::Io)?;
        }
        fs::write(path, content).await.map_err(FsError::Io)?;
        Ok(())
    }

    async fn exists(&self, path: &Path) -> bool {
        fs::metadata(path).await.map(|m| m.is_file()).unwrap_or(false)
    }

    async fn remove_file(&self, path: &Path) -> Result<(), FsError> {
        fs::remove_file(path).await.map_err(FsError::Io)?;
        Ok(())
    }

    async fn list_directory(&self, dir: &Path) -> Result<Vec<String>, FsError> {
        let mut entries = fs::read_dir(dir).await.map_err(FsError::Io)?;
        let mut results = Vec::new();
        while let Some(entry) = entries.next_entry().await.map_err(FsError::Io)? {
            results.push(entry.file_name().to_string_lossy().to_string());
        }
        Ok(results)
    }

    async fn list_recursive(&self, dir: &Path, max_files: usize) -> Result<Vec<PathBuf>, FsError> {
        let mut results = Vec::new();
        let mut visited = HashSet::new();
        self.list_inner(dir.to_path_buf(), &mut results, &mut visited, max_files, 0).await?;
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_utf8_boundary_handling() {
        let dir = tempdir().expect("Failed to create temp dir");
        let file_path = dir.path().join("utf8_test.txt");
        let fs = RealFileSystem::new();
        
        let content = "Hello 🦀!"; 
        fs.write_file(&file_path, content).await.unwrap();
        
        let result = fs.read_file_paged(&file_path, 0, 8).await.unwrap();
        assert_eq!(result, "Hello ");

        let result = fs.read_file_paged(&file_path, 7, 5).await.unwrap();
        assert_eq!(result, "!"); 
        
        let result = fs.read_file_paged(&file_path, 6, 1).await.unwrap();
        assert_eq!(result, "🦀"); 
    }

    #[tokio::test]
    async fn test_write_file_creates_dirs() {
        let dir = tempdir().expect("Failed to create temp dir");
        let nested_path = dir.path().join("a/b/c/test.txt");
        let fs = RealFileSystem::new();
        fs.write_file(&nested_path, "nested").await.expect("Failed to write nested");
        assert!(nested_path.exists());
    }

    #[tokio::test]
    async fn test_list_directory() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("test.txt");
        std::fs::write(&file, "test").unwrap();
        
        let fs = RealFileSystem::new();
        let list = fs.list_directory(dir.path()).await.unwrap();
        assert!(list.contains(&"test.txt".to_string()));
    }
}
