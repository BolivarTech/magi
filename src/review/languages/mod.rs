// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! Language-aware semantic extraction for source code.
//!
//! This module provides extractors that recognize syntactic boundaries
//! (functions, classes, structs, declaration groups) in source code
//! and return the units that contain changed lines. Used by the
//! review pipeline to send complete code units to the trio.
//!
//! # Architecture
//!
//! - [`LanguageExtractor`] trait defines the extraction interface.
//! - [`ExtractorKind`] enum provides zero-cost dispatch to concrete extractors.
//! - [`LanguageRegistry`] maps file extensions to extractors.

pub mod brace_family;
pub mod python;
pub mod scanner;

use std::collections::{HashMap, HashSet};

use crate::review::context::FileContext;

/// Kind of semantic code unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitKind {
    /// A standalone function (Rust `fn`, C function, Python `def`, JS function).
    Function,
    /// A method inside a class or impl block.
    Method,
    /// A class definition (Python `class`, Java/C#/JS class).
    Class,
    /// A struct definition (Rust, C, Go).
    Struct,
    /// An enum definition.
    Enum,
    /// A trait or interface definition.
    Trait,
    /// An impl block (Rust-specific).
    Impl,
    /// A group of related top-level declarations: constants, statics,
    /// type aliases, imports, macros, global variables.
    DeclarationGroup,
    /// Fallback: changed lines not inside any recognized construct.
    /// Context is lines around the change.
    TopLevel,
}

/// A semantic unit of code extracted from a source file.
#[derive(Debug, Clone)]
pub struct SemanticUnit {
    /// Kind of code block.
    pub kind: UnitKind,
    /// Name of the construct (function name, struct name, or
    /// descriptive label like "imports" or "constants").
    pub name: String,
    /// Source file path.
    pub file: String,
    /// Start line (1-indexed, inclusive).
    pub start_line: u32,
    /// End line (1-indexed, inclusive).
    pub end_line: u32,
    /// Complete source code of the unit (including doc comments
    /// and attributes above the declaration).
    pub content: String,
    /// Lines within this unit that were modified in the PR.
    pub changed_lines: Vec<u32>,
    /// File context metadata for this unit's file.
    pub context: FileContext,
}

/// Language-specific code extractor.
///
/// Implementors recognize syntactic boundaries (functions, classes,
/// declaration groups) in source code and return the units that
/// contain changed lines.
///
/// Concrete types: [`brace_family::BraceFamilyExtractor`],
/// [`python::PythonExtractor`].
/// Dispatched via [`ExtractorKind`] enum.
pub trait LanguageExtractor {
    /// Language identifier (e.g., "rust", "c", "python").
    fn language_id(&self) -> &str;
    /// File extensions this extractor handles (without dot).
    fn extensions(&self) -> &[&str];
    /// Extract semantic units from full file content, filtering
    /// to only those that overlap with `changed_lines`.
    fn extract_units(
        &self,
        content: &str,
        file_path: &str,
        changed_lines: &HashSet<u32>,
    ) -> Vec<SemanticUnit>;
}

/// Enum dispatch for language extractors.
///
/// Each variant wraps a concrete extractor. The enum delegates
/// [`LanguageExtractor`] methods to the inner type via match arms.
/// This avoids `Box<dyn>` and is trivially [`Clone`]-able.
#[derive(Clone)]
pub enum ExtractorKind {
    /// Brace-delimited languages (Rust, C, C++, Go, Java, JS, TS).
    BraceFamily(brace_family::BraceFamilyExtractor),
    /// Indentation-delimited (Python).
    Python(python::PythonExtractor),
}

impl LanguageExtractor for ExtractorKind {
    fn language_id(&self) -> &str {
        match self {
            Self::BraceFamily(e) => e.language_id(),
            Self::Python(e) => e.language_id(),
        }
    }

    fn extensions(&self) -> &[&str] {
        match self {
            // BraceFamilyExtractor stores extensions as Vec<String> (non-trait method).
            // The registry has already registered by extension, so this is unused at runtime.
            Self::BraceFamily(_) => &[],
            Self::Python(e) => e.extensions(),
        }
    }

    fn extract_units(
        &self,
        content: &str,
        file_path: &str,
        changed_lines: &HashSet<u32>,
    ) -> Vec<SemanticUnit> {
        match self {
            Self::BraceFamily(e) => e.extract_units(content, file_path, changed_lines),
            Self::Python(e) => e.extract_units(content, file_path, changed_lines),
        }
    }
}

/// Registry of language extractors, keyed by file extension.
///
/// # Examples
///
/// ```
/// use magi_rs::review::languages::LanguageRegistry;
///
/// let registry = LanguageRegistry::new();
/// assert!(registry.get("src/main.rs").is_some());
/// assert!(registry.get("unknown.xyz").is_none());
/// ```
pub struct LanguageRegistry {
    #[allow(dead_code)] // Red stub
    extractors: HashMap<String, ExtractorKind>,
}

impl LanguageRegistry {
    /// Create a registry with all built-in extractors.
    pub fn new() -> Self {
        todo!() // Red stub
    }

    /// Get the extractor for a file path, or `None` for unknown extensions.
    pub fn get(&self, _file_path: &str) -> Option<&ExtractorKind> {
        None // Red stub
    }
}

impl Default for LanguageRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a fallback semantic unit for changed lines in unknown file types.
///
/// Captures ±20 lines around each changed line as a `TopLevel` unit.
///
/// # Arguments
///
/// * `content` - Full file content.
/// * `file_path` - Path to the file.
/// * `changed_lines` - Set of changed line numbers.
/// * `context` - File context metadata.
pub fn fallback_extract(
    _content: &str,
    _file_path: &str,
    _changed_lines: &HashSet<u32>,
    _context: &FileContext,
) -> Vec<SemanticUnit> {
    Vec::new() // Red stub
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Test 37: registry_dispatches_by_extension ---
    #[test]
    fn registry_dispatches_by_extension() {
        let registry = LanguageRegistry::new();
        let ext = registry.get("src/main.rs");
        assert!(ext.is_some(), ".rs should be registered");
        assert!(
            matches!(ext.unwrap(), ExtractorKind::BraceFamily(_)),
            ".rs should use BraceFamily extractor"
        );
    }

    // --- Test 38: registry_returns_none_for_unknown ---
    #[test]
    fn registry_returns_none_for_unknown() {
        let registry = LanguageRegistry::new();
        assert!(
            registry.get("data.xyz").is_none(),
            "Unknown extension should return None"
        );
    }

    #[test]
    fn no_extractor_panics_on_any_prefix_of_hostile_source() {
        // Release is panic=abort: hostile input must never abort an extractor.
        let samples = [
            "fn a() { let s = \"}\"; '\\'' /* { */ }\n#[x] fn b(",
            "def f(x):\n    '''\n  @d\nclass C:\n\tpass\"\"\"",
            "/* open\n} } { 'a' \"\\\" `t ${x}` r#\"x\"# \u{e9}\u{1F600}",
        ];
        let registry = LanguageRegistry::new();
        let every: HashSet<u32> = (1..=20).collect();
        for path in [
            "a.rs", "a.py", "a.c", "a.cpp", "a.js", "a.ts", "a.go", "a.java",
        ] {
            let ex = registry.get(path).expect("registered");
            for s in samples {
                for end in s.char_indices().map(|(i, _)| i).chain([s.len()]) {
                    for lines in [&every, &HashSet::new()] {
                        let _ = ex.extract_units(&s[..end], path, lines);
                    }
                }
            }
        }
    }

    #[test]
    fn the_default_registry_has_the_ported_extractors_and_none_for_config_files() {
        // `.rs` has an extractor and `rust` as its language; config and unknown files have
        // none (config_files is not ported).
        let registry = LanguageRegistry::new();
        assert!(registry.get("src/a.rs").is_some());
        for path in [
            "Cargo.toml",
            "notes.txt",
            "x.unknownlang",
            "a.yaml",
            "README.md",
        ] {
            assert!(registry.get(path).is_none(), "{path}");
        }
        assert_eq!(crate::review::context::detect_language("src/a.rs"), "rust");
    }
}
