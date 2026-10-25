// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! Brace-delimited language extractor (Rust, C, C++, Go, Java, JS, TS).
//!
//! Uses [`CodeScanner`] for lexical-aware
//! brace counting and keyword detection to extract complete semantic
//! units (functions, structs, classes, declaration groups).

use std::collections::HashSet;

use super::scanner::{find_closing_brace, CodeScanner, LexicalRules, ScannedLine};
use super::{LanguageExtractor, SemanticUnit, UnitKind};
use crate::review::context::FileContext;

/// Configuration for a brace-delimited language extractor.
///
/// Pre-configured constructors exist for 7 languages: Rust, C, C++,
/// Go, Java, JavaScript, TypeScript.
#[derive(Clone)]
pub struct BraceFamilyExtractor {
    language: String,
    exts: Vec<String>,
    /// Keywords that start a function/method declaration.
    function_keywords: Vec<String>,
    /// Keywords that start a block declaration (struct, class, enum, etc.).
    block_keywords: Vec<String>,
    /// Keywords that start a single-line declaration (const, type, use, etc.).
    declaration_keywords: Vec<String>,
    /// Line prefixes for doc comments (e.g., "///", "//!").
    doc_comment_prefixes: Vec<String>,
    /// Line prefixes for attributes (e.g., "#[").
    attribute_prefixes: Vec<String>,
    /// Lexical rules for the CodeScanner.
    lexical_rules: LexicalRules,
}

impl LanguageExtractor for BraceFamilyExtractor {
    fn language_id(&self) -> &str {
        &self.language
    }

    fn extensions(&self) -> &[&str] {
        // The trait wants `&[&str]` but the extensions are owned `String`s, so this is empty.
        // The registry reads the inherent `BraceFamilyExtractor::extensions` instead.
        &[]
    }

    fn extract_units(
        &self,
        content: &str,
        file_path: &str,
        changed_lines: &HashSet<u32>,
    ) -> Vec<SemanticUnit> {
        if changed_lines.is_empty() {
            return vec![];
        }

        let lines: Vec<&str> = content.lines().collect();
        let mut scanner = CodeScanner::new(self.lexical_rules.clone());
        let scanned = scanner.scan_all(content);

        // Detect all constructs
        let constructs = self.detect_constructs(&lines, &scanned);

        // Filter to constructs that overlap with changed lines
        let mut units: Vec<SemanticUnit> = Vec::new();
        let mut covered_lines: HashSet<u32> = HashSet::new();

        for construct in &constructs {
            let overlaps = changed_lines
                .iter()
                .any(|&l| l >= construct.start_line && l <= construct.end_line);
            if overlaps {
                let changed_in_unit: Vec<u32> = changed_lines
                    .iter()
                    .copied()
                    .filter(|&l| l >= construct.start_line && l <= construct.end_line)
                    .collect();
                for &l in &changed_in_unit {
                    covered_lines.insert(l);
                }
                let start_idx = construct.start_line.saturating_sub(1) as usize;
                let end_idx = construct.end_line as usize;
                let unit_content = lines
                    .get(start_idx..end_idx.min(lines.len()))
                    .unwrap_or(&[])
                    .join("\n");

                units.push(SemanticUnit {
                    kind: construct.kind,
                    name: construct.name.clone(),
                    file: file_path.to_string(),
                    start_line: construct.start_line,
                    end_line: construct.end_line,
                    content: unit_content,
                    changed_lines: changed_in_unit,
                    context: FileContext::default(),
                });
            }
        }

        // Orphan fallback: changed lines not covered by any construct
        let orphans: Vec<u32> = changed_lines
            .iter()
            .copied()
            .filter(|l| !covered_lines.contains(l))
            .collect();

        if !orphans.is_empty() {
            let total_lines = u32::try_from(lines.len()).unwrap_or(u32::MAX);
            let min_orphan = *orphans.iter().min().unwrap_or(&1);
            let max_orphan = *orphans.iter().max().unwrap_or(&total_lines);

            // Find surrounding construct boundaries
            let start = self.find_gap_start(&constructs, min_orphan);
            let end = self.find_gap_end(&constructs, max_orphan, total_lines);

            let start_idx = start.saturating_sub(1);
            let end_idx = end as usize;
            let unit_content = lines
                .get(start_idx..end_idx.min(lines.len()))
                .unwrap_or(&[])
                .join("\n");

            let mut sorted_orphans = orphans;
            sorted_orphans.sort_unstable();

            let start_u32 = u32::try_from(start).unwrap_or(1);
            units.push(SemanticUnit {
                kind: UnitKind::TopLevel,
                name: format!("lines {}-{}", start_u32, end),
                file: file_path.to_string(),
                start_line: start_u32,
                end_line: end,
                content: unit_content,
                changed_lines: sorted_orphans,
                context: FileContext::default(),
            });
        }

        units
    }
}

/// Intermediate representation of a detected construct.
struct Construct {
    kind: UnitKind,
    name: String,
    start_line: u32,
    end_line: u32,
}

impl BraceFamilyExtractor {
    /// Detect all constructs in the file.
    fn detect_constructs(&self, lines: &[&str], scanned: &[ScannedLine]) -> Vec<Construct> {
        let mut constructs = Vec::new();
        let mut i = 0;

        while let Some(line) = lines.get(i) {
            if scanned.get(i).is_some_and(|s| s.is_non_code) {
                i += 1;
                continue;
            }

            let trimmed = line.trim();
            let line_num = u32::try_from(i + 1).unwrap_or(u32::MAX);

            // Check function keywords, then block keywords
            let keyword = self
                .match_function_keyword(trimmed)
                .map(|kind| (kind, &self.function_keywords))
                .or_else(|| {
                    self.match_block_keyword(trimmed)
                        .map(|kind| (kind, &self.block_keywords))
                });
            if let Some((kind, keywords)) = keyword {
                let name = extract_name_from_line(trimmed, keywords);
                let start = self.scan_back_for_docs(lines, i);
                let end = brace_block_end(scanned, i, lines.len(), line_num);
                constructs.push(Construct {
                    kind,
                    name,
                    start_line: u32::try_from(start + 1).unwrap_or(1),
                    end_line: end,
                });
                // Skip past this construct
                i = (end as usize).max(i + 1);
                continue;
            }

            // Check declaration keywords
            if self.match_declaration_keyword(trimmed) {
                let group_start = i;
                let mut group_end = i;
                // Scan forward for contiguous declarations of the same kind
                while group_end + 1 < lines.len() {
                    let next = lines.get(group_end + 1).map_or("", |l| l.trim());
                    if self.match_declaration_keyword(next) {
                        group_end += 1;
                    } else if next.is_empty() {
                        // Allow blank lines within groups
                        if lines
                            .get(group_end + 2)
                            .is_some_and(|l| self.match_declaration_keyword(l.trim()))
                        {
                            group_end += 2;
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                let doc_start = self.scan_back_for_docs(lines, group_start);
                let name = self.declaration_group_name(trimmed);
                constructs.push(Construct {
                    kind: UnitKind::DeclarationGroup,
                    name,
                    start_line: u32::try_from(doc_start + 1).unwrap_or(1),
                    end_line: u32::try_from(group_end + 1).unwrap_or(1),
                });
                i = group_end + 1;
                continue;
            }

            i += 1;
        }

        constructs
    }

    /// Check if the line starts a function declaration.
    fn match_function_keyword(&self, trimmed: &str) -> Option<UnitKind> {
        for kw in &self.function_keywords {
            if trimmed.starts_with(kw.as_str()) {
                return Some(UnitKind::Function);
            }
        }
        None
    }

    /// Check if the line starts a block declaration.
    fn match_block_keyword(&self, trimmed: &str) -> Option<UnitKind> {
        for kw in &self.block_keywords {
            if trimmed.starts_with(kw.as_str()) {
                return match kw.trim() {
                    "struct" => Some(UnitKind::Struct),
                    "pub struct" => Some(UnitKind::Struct),
                    "pub(crate) struct" => Some(UnitKind::Struct),
                    "enum" | "pub enum" | "pub(crate) enum" => Some(UnitKind::Enum),
                    "trait" | "pub trait" | "pub(crate) trait" => Some(UnitKind::Trait),
                    "impl" => Some(UnitKind::Impl),
                    "class" | "pub class" | "export class" => Some(UnitKind::Class),
                    "mod" | "pub mod" | "pub(crate) mod" => Some(UnitKind::Function), // module as function-like
                    "interface" | "pub interface" | "export interface" => Some(UnitKind::Trait),
                    "namespace" => Some(UnitKind::Class),
                    "union" | "type" => Some(UnitKind::Struct),
                    _ => Some(UnitKind::Class),
                };
            }
        }
        None
    }

    /// Check if the line starts a declaration.
    fn match_declaration_keyword(&self, trimmed: &str) -> bool {
        for kw in &self.declaration_keywords {
            if trimmed.starts_with(kw.as_str()) {
                return true;
            }
        }
        false
    }

    /// Scan backward from `idx` to include doc comments and attributes.
    fn scan_back_for_docs(&self, lines: &[&str], idx: usize) -> usize {
        let mut start = idx;
        while start > 0 {
            let Some(prev) = lines.get(start - 1).map(|l| l.trim()) else {
                break;
            };
            let is_doc = self
                .doc_comment_prefixes
                .iter()
                .any(|p| prev.starts_with(p.as_str()));
            let is_attr = self
                .attribute_prefixes
                .iter()
                .any(|p| prev.starts_with(p.as_str()));
            if is_doc || is_attr {
                start -= 1;
            } else {
                break;
            }
        }
        start
    }

    /// Generate a descriptive name for a declaration group.
    fn declaration_group_name(&self, first_line: &str) -> String {
        let trimmed = first_line.trim();
        if trimmed.starts_with("use ")
            || trimmed.starts_with("pub use ")
            || trimmed.starts_with("import ")
            || trimmed.starts_with("from ")
        {
            "imports".to_string()
        } else if trimmed.starts_with("const ") || trimmed.starts_with("pub const ") {
            "constants".to_string()
        } else if trimmed.starts_with("static ") || trimmed.starts_with("pub static ") {
            "statics".to_string()
        } else if trimmed.starts_with("#define ") {
            "defines".to_string()
        } else if trimmed.starts_with("typedef ") {
            "typedefs".to_string()
        } else if trimmed.starts_with("extern ") {
            "externs".to_string()
        } else if trimmed.starts_with("type ") || trimmed.starts_with("pub type ") {
            "type aliases".to_string()
        } else if trimmed.starts_with("let ") || trimmed.starts_with("var ") {
            "variables".to_string()
        } else if trimmed.starts_with("export ") {
            "exports".to_string()
        } else if trimmed.starts_with("package ") {
            "package".to_string()
        } else {
            "declarations".to_string()
        }
    }

    /// Find the start of the gap before `line` (end of previous construct or 1).
    fn find_gap_start(&self, constructs: &[Construct], line: u32) -> usize {
        let mut start = 1u32;
        for c in constructs {
            if c.end_line < line {
                start = start.max(c.end_line.saturating_add(1));
            }
        }
        start as usize
    }

    /// Find the end of the gap after `line` (start of next construct or total).
    fn find_gap_end(&self, constructs: &[Construct], line: u32, total: u32) -> u32 {
        let mut end = total;
        for c in constructs {
            if c.start_line > line && c.start_line < end {
                end = c.start_line.saturating_sub(1);
            }
        }
        end
    }
}

/// Find the 1-indexed line where the construct starting at line index `i` ends.
///
/// The opening brace is on line `i` or, failing that, within the next two lines. A construct
/// with no brace in that window (a single-line declaration) ends on `line_num`.
fn brace_block_end(scanned: &[ScannedLine], i: usize, line_count: usize, line_num: u32) -> u32 {
    let opens = |idx: usize| scanned.get(idx).is_some_and(|s| s.brace_delta > 0);
    let brace_line = if opens(i) {
        Some(i)
    } else {
        // Opening brace might be on the next line
        ((i + 1)..line_count.min(i + 3)).find(|&j| opens(j))
    };
    match brace_line {
        Some(open_idx) => {
            let close_idx = find_closing_brace(scanned, open_idx);
            u32::try_from(close_idx + 1).unwrap_or(line_num)
        }
        None => line_num,
    }
}

/// Extract a name from a line given a set of keywords.
fn extract_name_from_line(trimmed: &str, keywords: &[String]) -> String {
    for kw in keywords {
        if let Some(rest) = trimmed.strip_prefix(kw.as_str()) {
            // Take the first identifier-like token
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                return name;
            }
        }
    }
    // Fallback: take first word-like token
    trimmed
        .split_whitespace()
        .nth(1)
        .unwrap_or("unnamed")
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect::<String>()
}

/// Return all pre-configured brace-family extractors.
pub fn all_extractors() -> Vec<BraceFamilyExtractor> {
    vec![
        BraceFamilyExtractor::rust(),
        BraceFamilyExtractor::c(),
        BraceFamilyExtractor::cpp(),
        BraceFamilyExtractor::go(),
        BraceFamilyExtractor::java(),
        BraceFamilyExtractor::javascript(),
        BraceFamilyExtractor::typescript(),
    ]
}

impl BraceFamilyExtractor {
    /// Rust extractor.
    pub fn rust() -> Self {
        Self {
            language: "rust".to_string(),
            exts: vec!["rs".to_string()],
            function_keywords: vec![
                "pub async fn ".to_string(),
                "pub(crate) async fn ".to_string(),
                "async fn ".to_string(),
                "pub(crate) fn ".to_string(),
                "pub fn ".to_string(),
                "fn ".to_string(),
            ],
            block_keywords: vec![
                "pub(crate) struct ".to_string(),
                "pub struct ".to_string(),
                "struct ".to_string(),
                "pub(crate) enum ".to_string(),
                "pub enum ".to_string(),
                "enum ".to_string(),
                "pub(crate) trait ".to_string(),
                "pub trait ".to_string(),
                "trait ".to_string(),
                "impl ".to_string(),
                "pub(crate) mod ".to_string(),
                "pub mod ".to_string(),
                "mod ".to_string(),
            ],
            declaration_keywords: vec![
                "pub(crate) const ".to_string(),
                "pub const ".to_string(),
                "const ".to_string(),
                "pub(crate) static ".to_string(),
                "pub static ".to_string(),
                "static ".to_string(),
                "pub(crate) type ".to_string(),
                "pub type ".to_string(),
                "type ".to_string(),
                "pub use ".to_string(),
                "use ".to_string(),
            ],
            doc_comment_prefixes: vec!["///".to_string(), "//!".to_string()],
            attribute_prefixes: vec!["#[".to_string()],
            lexical_rules: LexicalRules::rust(),
        }
    }

    /// C extractor.
    pub fn c() -> Self {
        Self {
            language: "c".to_string(),
            exts: vec!["c".to_string(), "h".to_string()],
            function_keywords: vec![],
            block_keywords: vec![
                "struct ".to_string(),
                "union ".to_string(),
                "enum ".to_string(),
            ],
            declaration_keywords: vec![
                "#define ".to_string(),
                "typedef ".to_string(),
                "extern ".to_string(),
                "#include ".to_string(),
            ],
            doc_comment_prefixes: vec!["/**".to_string(), "///".to_string()],
            attribute_prefixes: vec![],
            lexical_rules: LexicalRules::c(),
        }
    }

    /// C++ extractor.
    pub fn cpp() -> Self {
        let mut ext = Self::c();
        ext.language = "cpp".to_string();
        ext.exts = vec![
            "cpp".to_string(),
            "cxx".to_string(),
            "cc".to_string(),
            "hpp".to_string(),
            "hxx".to_string(),
        ];
        ext.block_keywords
            .extend(["class ".to_string(), "namespace ".to_string()]);
        ext.lexical_rules = LexicalRules::cpp();
        ext
    }

    /// Go extractor.
    pub fn go() -> Self {
        Self {
            language: "go".to_string(),
            exts: vec!["go".to_string()],
            function_keywords: vec!["func ".to_string()],
            block_keywords: vec!["type ".to_string()],
            declaration_keywords: vec![
                "import ".to_string(),
                "var ".to_string(),
                "const ".to_string(),
            ],
            doc_comment_prefixes: vec!["//".to_string()],
            attribute_prefixes: vec![],
            lexical_rules: LexicalRules::go(),
        }
    }

    /// Java extractor.
    pub fn java() -> Self {
        Self {
            language: "java".to_string(),
            exts: vec!["java".to_string()],
            function_keywords: vec![
                "public ".to_string(),
                "private ".to_string(),
                "protected ".to_string(),
                "static ".to_string(),
            ],
            block_keywords: vec![
                "class ".to_string(),
                "interface ".to_string(),
                "enum ".to_string(),
                "public class ".to_string(),
                "public interface ".to_string(),
                "public enum ".to_string(),
            ],
            declaration_keywords: vec!["import ".to_string(), "package ".to_string()],
            doc_comment_prefixes: vec!["/**".to_string(), "///".to_string()],
            attribute_prefixes: vec!["@".to_string()],
            lexical_rules: LexicalRules::java(),
        }
    }

    /// JavaScript extractor.
    pub fn javascript() -> Self {
        Self {
            language: "javascript".to_string(),
            exts: vec!["js".to_string(), "jsx".to_string(), "mjs".to_string()],
            function_keywords: vec![
                "function ".to_string(),
                "async function ".to_string(),
                "export function ".to_string(),
                "export async function ".to_string(),
                "export default function ".to_string(),
            ],
            block_keywords: vec![
                "class ".to_string(),
                "export class ".to_string(),
                "export default class ".to_string(),
            ],
            declaration_keywords: vec![
                "import ".to_string(),
                "export ".to_string(),
                "const ".to_string(),
                "let ".to_string(),
                "var ".to_string(),
            ],
            doc_comment_prefixes: vec!["/**".to_string(), "///".to_string()],
            attribute_prefixes: vec!["@".to_string()],
            lexical_rules: LexicalRules::javascript(),
        }
    }

    /// TypeScript extractor.
    pub fn typescript() -> Self {
        let mut ext = Self::javascript();
        ext.language = "typescript".to_string();
        ext.exts = vec!["ts".to_string(), "tsx".to_string(), "mts".to_string()];
        ext.block_keywords.push("interface ".to_string());
        ext.block_keywords.push("export interface ".to_string());
        ext.declaration_keywords.push("type ".to_string());
        ext.declaration_keywords.push("export type ".to_string());
        ext.lexical_rules = LexicalRules::typescript();
        ext
    }

    /// Return the file extensions (used by the registry).
    pub fn extensions(&self) -> &[String] {
        &self.exts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rust_extractor() -> BraceFamilyExtractor {
        BraceFamilyExtractor::rust()
    }

    fn c_extractor() -> BraceFamilyExtractor {
        BraceFamilyExtractor::c()
    }

    fn extract(ext: &BraceFamilyExtractor, code: &str, changed: &[u32]) -> Vec<SemanticUnit> {
        let changed_set: HashSet<u32> = changed.iter().copied().collect();
        ext.extract_units(code, "test.rs", &changed_set)
    }

    // --- Test 19: rust_extracts_function_with_changed_line ---
    #[test]
    fn rust_extracts_function_with_changed_line() {
        let code = "fn foo() {\n    let x = 1;\n    let y = 2;\n}";
        let units = extract(&rust_extractor(), code, &[2]);
        assert_eq!(units.len(), 1, "Should extract one function");
        assert_eq!(units[0].kind, UnitKind::Function);
        assert_eq!(units[0].name, "foo");
        assert!(units[0].content.contains("fn foo()"));
    }

    // --- Test 20: rust_extracts_struct_with_changed_field ---
    #[test]
    fn rust_extracts_struct_with_changed_field() {
        let code = "pub struct Config {\n    pub model: String,\n    pub backend: String,\n}";
        let units = extract(&rust_extractor(), code, &[2]);
        assert_eq!(units.len(), 1, "Should extract one struct");
        assert_eq!(units[0].kind, UnitKind::Struct);
        assert_eq!(units[0].name, "Config");
    }

    // --- Test 21: rust_extracts_impl_block ---
    #[test]
    fn rust_extracts_impl_block() {
        let code = "impl Config {\n    fn new() -> Self {\n        Self {}\n    }\n}";
        let units = extract(&rust_extractor(), code, &[2]);
        assert_eq!(units.len(), 1, "Should extract one impl block");
        assert_eq!(units[0].kind, UnitKind::Impl);
        assert!(units[0].name.starts_with("Config"));
    }

    // --- Test 22: rust_extracts_const_declaration_group ---
    #[test]
    fn rust_extracts_const_declaration_group() {
        let code = "const MAX: u32 = 100;\nconst MIN: u32 = 0;\nconst DEFAULT: u32 = 50;";
        let units = extract(&rust_extractor(), code, &[2]);
        assert_eq!(units.len(), 1, "Should extract one declaration group");
        assert_eq!(units[0].kind, UnitKind::DeclarationGroup);
        assert_eq!(units[0].name, "constants");
    }

    // --- Test 23: rust_extracts_use_import_group ---
    #[test]
    fn rust_extracts_use_import_group() {
        let code =
            "use std::collections::HashMap;\nuse std::collections::HashSet;\nuse std::path::Path;";
        let units = extract(&rust_extractor(), code, &[2]);
        assert_eq!(units.len(), 1, "Should extract one import group");
        assert_eq!(units[0].kind, UnitKind::DeclarationGroup);
        assert_eq!(units[0].name, "imports");
    }

    // --- Test 24: rust_includes_doc_comments_and_attributes ---
    #[test]
    fn rust_includes_doc_comments_and_attributes() {
        let code =
            "/// A config struct.\n#[derive(Debug)]\npub struct Config {\n    pub x: u32,\n}";
        let units = extract(&rust_extractor(), code, &[4]);
        assert_eq!(units.len(), 1);
        assert!(
            units[0].content.contains("/// A config struct."),
            "Should include doc comment"
        );
        assert!(
            units[0].content.contains("#[derive(Debug)]"),
            "Should include attribute"
        );
    }

    // --- Test 25: rust_skips_unchanged_function ---
    #[test]
    fn rust_skips_unchanged_function() {
        let code = "fn foo() {\n    let x = 1;\n}\nfn bar() {\n    let y = 2;\n}";
        // Only change in bar (line 5), not in foo
        let units = extract(&rust_extractor(), code, &[5]);
        assert_eq!(units.len(), 1, "Should only extract changed function");
        assert_eq!(units[0].name, "bar");
    }

    // --- Test 26: rust_orphan_line_between_functions ---
    #[test]
    fn rust_orphan_line_between_functions() {
        let code = "fn foo() {\n    bar();\n}\n\n// orphan comment\n\nfn baz() {\n    qux();\n}";
        // Changed line 5 (orphan comment) — between fn foo and fn baz
        let units = extract(&rust_extractor(), code, &[5]);
        assert!(
            units.iter().any(|u| u.kind == UnitKind::TopLevel),
            "Orphan line should produce a TopLevel unit"
        );
    }

    // --- Test 27: c_extracts_global_variable ---
    #[test]
    fn c_extracts_global_variable() {
        // C global variables are at file scope. They look like declarations.
        // We use a declaration keyword approach — extern is captured.
        let code = "extern int count;\nextern int total;";
        let units = extract(&c_extractor(), code, &[1]);
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].kind, UnitKind::DeclarationGroup);
    }

    // --- Test 28: c_extracts_define_group ---
    #[test]
    fn c_extracts_define_group() {
        let code = "#define MAX_SIZE 100\n#define MIN_SIZE 10\n#define DEFAULT_SIZE 50";
        let units = extract(&c_extractor(), code, &[2]);
        assert_eq!(units.len(), 1, "Should extract one define group");
        assert_eq!(units[0].kind, UnitKind::DeclarationGroup);
        assert_eq!(units[0].name, "defines");
        assert!(
            units[0].content.contains("#define MAX_SIZE"),
            "Group should include all consecutive defines"
        );
        assert!(
            units[0].content.contains("#define DEFAULT_SIZE"),
            "Group should include all consecutive defines"
        );
    }

    /// Asserts each line `1..=total` lies in exactly one unit.
    fn assert_each_line_in_one_unit(units: &[SemanticUnit], total: u32) {
        for line in 1..=total {
            let holders: Vec<&str> = units
                .iter()
                .filter(|u| u.start_line <= line && line <= u.end_line)
                .map(|u| u.name.as_str())
                .collect();
            assert_eq!(holders.len(), 1, "line {line} is in {holders:?}");
        }
    }

    #[test]
    fn every_gap_between_constructs_is_its_own_fallback_unit() {
        // Blank lines and comments between functions are orphans; one
        // fallback unit spanning the first to the last of them swallowed the functions between.
        let code = "use a::b;

fn one() {
    x();
}
// note

fn two() {
    y();
}

fn three() {
    z();
}
";
        let every: Vec<u32> = (1..=14).collect();
        let units = extract(&rust_extractor(), code, &every);
        assert_each_line_in_one_unit(&units, 14);
        let gaps: Vec<(u32, u32)> = units
            .iter()
            .filter(|u| u.kind == UnitKind::TopLevel)
            .map(|u| (u.start_line, u.end_line))
            .collect();
        assert_eq!(gaps, vec![(2, 2), (6, 7), (11, 11)]);
    }

    #[test]
    fn a_gap_before_a_construct_on_the_last_line_stops_before_it() {
        let code = "fn a() {}

fn b() {}";
        let units = extract(&rust_extractor(), code, &[1, 2, 3]);
        assert_each_line_in_one_unit(&units, 3);
    }
}
