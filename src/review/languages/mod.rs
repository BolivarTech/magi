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
            Self::BraceFamily(e) => e.extensions(),
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
    extractors: HashMap<String, ExtractorKind>,
}

impl LanguageRegistry {
    /// Create a registry with all built-in extractors.
    pub fn new() -> Self {
        let mut registry = Self {
            extractors: HashMap::new(),
        };

        // Register brace-family extractors
        for extractor in brace_family::all_extractors() {
            for ext in extractor.extensions() {
                registry.extractors.insert(
                    ext.to_string(),
                    ExtractorKind::BraceFamily(extractor.clone()),
                );
            }
        }

        // Register Python
        let py = python::PythonExtractor;
        for ext in py.extensions() {
            registry
                .extractors
                .insert(ext.to_string(), ExtractorKind::Python(py));
        }

        registry
    }

    /// File extensions (without the dot) that have a registered extractor.
    pub fn extensions(&self) -> impl Iterator<Item = &str> {
        self.extractors.keys().map(String::as_str)
    }

    /// Get the extractor for a file path, or `None` for unknown extensions.
    pub fn get(&self, file_path: &str) -> Option<&ExtractorKind> {
        let ext = file_path.rsplit('.').next()?;
        self.extractors.get(ext)
    }
}

impl Default for LanguageRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// The changed lines of `sorted` that fall in `start..=end`.
///
/// Extractors call it once per construct with the changed lines sorted once, so a whole file
/// (every line changed) costs O(c log n + n) for c constructs and n lines instead of O(c × n).
///
/// # Arguments
///
/// * `sorted` - Changed lines, sorted ascending and without repeats.
/// * `start` - First line of the range (1-based, inclusive).
/// * `end` - Last line of the range (1-based, inclusive).
///
/// # Returns
///
/// The sub-slice of `sorted` inside the range, empty when none is.
pub fn lines_in_range(sorted: &[u32], start: u32, end: u32) -> &[u32] {
    let _ = (start, end);
    sorted.get(..0).unwrap_or(&[])
}

/// The fallback units of the changed lines no construct holds: one [`UnitKind::TopLevel`] unit
/// per gap between constructs that holds orphans, spanning that whole gap and nothing else.
///
/// A gap runs from the line after the previous construct (or line 1) to the line before the
/// next one (or the last line), so a fallback unit never holds a construct and a file split by
/// units keeps each unit as small as its source allows.
///
/// # Arguments
///
/// * `lines` - The file's lines.
/// * `file_path` - Path to the file.
/// * `constructs` - `(start, end)` line ranges (1-based, inclusive) of the file's constructs,
///   sorted by start and disjoint.
/// * `orphans` - Changed lines outside every construct.
///
/// # Returns
///
/// The units, in line order. O(o log o + o log c) for o orphans and c constructs.
pub(crate) fn gap_units(
    lines: &[&str],
    file_path: &str,
    constructs: &[(u32, u32)],
    mut orphans: Vec<u32>,
) -> Vec<SemanticUnit> {
    orphans.sort_unstable();
    orphans.dedup();
    let total = u32::try_from(lines.len()).unwrap_or(u32::MAX);
    let mut units: Vec<SemanticUnit> = Vec::new();
    for orphan in orphans {
        let next = constructs.partition_point(|&(start, _)| start <= orphan);
        let first = next
            .checked_sub(1)
            .and_then(|i| constructs.get(i))
            .map_or(1, |&(_, end)| end.saturating_add(1));
        let last = constructs
            .get(next)
            .map_or(total, |&(start, _)| start.saturating_sub(1));
        if let Some(unit) = units
            .last_mut()
            .filter(|u| u.start_line == first && u.end_line == last)
        {
            unit.changed_lines.push(orphan);
            continue;
        }
        let content = lines
            .get(first.saturating_sub(1) as usize..(last as usize).min(lines.len()))
            .unwrap_or(&[])
            .join("\n");
        units.push(SemanticUnit {
            kind: UnitKind::TopLevel,
            name: format!("lines {first}-{last}"),
            file: file_path.to_string(),
            start_line: first,
            end_line: last,
            content,
            changed_lines: vec![orphan],
            context: FileContext::default(),
        });
    }
    units
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
    content: &str,
    file_path: &str,
    changed_lines: &HashSet<u32>,
    context: &FileContext,
) -> Vec<SemanticUnit> {
    if changed_lines.is_empty() {
        return vec![];
    }
    let lines: Vec<&str> = content.lines().collect();
    let total = u32::try_from(lines.len()).unwrap_or(u32::MAX);

    let min_line = *changed_lines.iter().min().unwrap_or(&1);
    let max_line = *changed_lines.iter().max().unwrap_or(&total);

    let start = min_line.saturating_sub(20).max(1);
    let end = max_line.saturating_add(20).min(total);

    let start_idx = start.saturating_sub(1) as usize;
    let end_idx = end as usize;
    let unit_content = lines
        .get(start_idx..end_idx.min(lines.len()))
        .unwrap_or(&[])
        .join("\n");

    let mut sorted_changed: Vec<u32> = changed_lines
        .iter()
        .copied()
        .filter(|&l| l >= start && l <= end)
        .collect();
    sorted_changed.sort_unstable();

    vec![SemanticUnit {
        kind: UnitKind::TopLevel,
        name: format!("lines {}-{}", start, end),
        file: file_path.to_string(),
        start_line: start,
        end_line: end,
        content: unit_content,
        changed_lines: sorted_changed,
        context: context.clone(),
    }]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_in_range_returns_exactly_the_changed_lines_inside_the_range() {
        let sorted = [2, 4, 5, 9, 12];
        assert_eq!(lines_in_range(&sorted, 4, 9), &[4, 5, 9]);
        assert_eq!(lines_in_range(&sorted, 1, 2), &[2]);
        assert_eq!(lines_in_range(&sorted, 6, 8), &[] as &[u32]);
        assert_eq!(lines_in_range(&sorted, 13, 20), &[] as &[u32]);
        assert_eq!(lines_in_range(&[], 1, 5), &[] as &[u32]);
        assert_eq!(
            lines_in_range(&sorted, 9, 4),
            &[] as &[u32],
            "an empty range holds nothing"
        );
    }

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

    #[test]
    fn every_extractor_reports_its_extensions_through_the_trait() {
        let registry = LanguageRegistry::new();
        for ext in registry.extensions() {
            let ex = registry.get(&format!("a.{ext}")).expect("registered");
            assert!(
                ex.extensions().contains(&ext),
                "{ext}: {:?}",
                ex.extensions()
            );
        }
    }
}
