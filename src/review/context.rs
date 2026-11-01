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

/// Detect programming language from file extension, ignoring its ASCII case.
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
    // MAGI FIX (S6 loop 4): `MAIN.RS` is Rust too; the extension is matched case-insensitively.
    let ext = file_path
        .rsplit('.')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "rs" => "rust",
        "c" => "c",
        // MAGI FIX (S6 loop 15): `.h` is read with the C++ rules, so its label says `cpp` too.
        "cpp" | "cxx" | "cc" | "h" | "hpp" | "hxx" => "cpp",
        "py" | "pyi" => "python",
        // MAGI FIX (S6 loop 16): `.cjs` and `.cts` have an extractor, so they have a label too.
        "js" | "jsx" | "mjs" | "cjs" => "javascript",
        "ts" | "tsx" | "mts" | "cts" => "typescript",
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
    fn an_upper_case_extension_names_the_same_language() {
        // [MAGI S6 loop 4, B1] `FOO.RS` was `unknown`, so it lost its extractor.
        assert_eq!(detect_language("src/MAIN.RS"), "rust");
        assert_eq!(detect_language("App.Py"), "python");
        assert_eq!(detect_language("x.Cpp"), "cpp");
    }

    #[test]
    fn a_dot_h_header_is_labelled_cpp_like_the_extractor_that_reads_it() {
        // [MAGI S6 loop 15, B1] `.h` was labelled `c` while the C++ extractor read it, so the
        // prompt's code fence and the extractor disagreed on the header's language.
        assert_eq!(detect_language("include/a.h"), "cpp");
        assert_eq!(detect_language("A.H"), "cpp");
        assert_eq!(detect_language("a.c"), "c");
    }

    #[test]
    fn every_registry_extension_is_labelled_with_its_extractor_language() {
        // [MAGI S6 loop 16, B1-6] `detect_language` and the registry are two extension maps;
        // a registry extension missing here (`.cjs`, `.cts`) gave an `unknown` file header to a
        // file whose units the extractor still read.
        use crate::review::languages::{LanguageExtractor, LanguageRegistry};
        let registry = LanguageRegistry::new();
        let mut checked = 0;
        for ext in registry.extensions() {
            let path = format!("a.{ext}");
            let extractor = registry.get(&path).map(LanguageExtractor::language_id);
            assert_eq!(Some(detect_language(&path).as_str()), extractor, "{path}");
            checked += 1;
        }
        assert!(checked > 0, "the registry lists its extensions");
    }

    #[test]
    fn detect_language_unknown() {
        assert_eq!(detect_language("Makefile"), "unknown");
    }
}
