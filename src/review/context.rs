// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! File context for the review pipeline.
//!
//! Carries the structured metadata a semantic unit keeps about its source file
//! ([`FileContext`]) and detects a file's language from its extension
//! ([`detect_language`]).

use serde::Serialize;

/// Structured metadata about a source file, kept by each semantic unit.
///
/// # Examples
///
/// ```
/// use magi_rs::review::context::{detect_language, FileContext};
///
/// let ctx = FileContext {
///     language: detect_language("src/main.rs"),
///     ..FileContext::default()
/// };
/// assert_eq!(ctx.language, "rust");
/// ```
#[derive(Debug, Clone, Default, Serialize)]
pub struct FileContext {
    /// Programming language (e.g., "rust", "c", "python", "unknown").
    pub language: String,
    /// Whether the file is entirely new in this PR.
    pub is_new_file: bool,
    /// Whether a corresponding test file likely exists.
    pub has_test_file: bool,
    /// Number of added/modified lines in this file.
    pub lines_changed: u32,
    /// Detected framework/library hints from imports in the diff.
    pub framework_hints: Vec<String>,
}

/// Detect programming language from file extension.
///
/// # Arguments
///
/// * `file_path` - Path to the source file (e.g., "src/main.rs").
///
/// # Returns
///
/// Language identifier string. Returns "unknown" for unrecognized extensions.
///
/// # Examples
///
/// ```
/// use magi_rs::review::context::detect_language;
///
/// assert_eq!(detect_language("app.py"), "python");
/// assert_eq!(detect_language("Makefile"), "unknown");
/// ```
pub fn detect_language(file_path: &str) -> String {
    let ext = file_path.rsplit('.').next().unwrap_or("");
    match ext {
        "rs" => "rust",
        "c" | "h" => "c",
        "cpp" | "cxx" | "cc" | "hpp" | "hxx" => "cpp",
        "py" | "pyi" => "python",
        "js" | "jsx" | "mjs" => "javascript",
        "ts" | "tsx" | "mts" => "typescript",
        "go" => "go",
        "java" => "java",
        "cs" => "csharp",
        "swift" => "swift",
        "kt" | "kts" => "kotlin",
        "rb" => "ruby",
        "php" => "php",
        "sh" | "bash" => "shell",
        "ps1" => "powershell",
        "toml" => "toml",
        "yaml" | "yml" => "yaml",
        "json" => "json",
        "md" | "markdown" => "markdown",
        "xml" => "xml",
        "html" | "htm" => "html",
        "css" | "scss" | "sass" => "css",
        "sql" => "sql",
        _ => "unknown",
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_language_rust() {
        assert_eq!(detect_language("src/main.rs"), "rust");
    }

    #[test]
    fn detect_language_python() {
        assert_eq!(detect_language("app.py"), "python");
    }

    #[test]
    fn detect_language_unknown() {
        assert_eq!(detect_language("Makefile"), "unknown");
    }
}
