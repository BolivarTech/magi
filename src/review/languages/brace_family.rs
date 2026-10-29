// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! Brace-delimited language extractor (Rust, C, C++, Go, Java, JS, TS).
//!
//! Uses [`CodeScanner`] for lexical-aware
//! brace counting and keyword detection to extract complete semantic
//! units (functions, structs, classes, declaration groups).

use super::scanner::{find_closing_brace, CodeScanner, LexicalRules, ScannedLine};
use super::{
    gap_units, split_into_members, ChangedLines, LanguageExtractor, Member, SemanticUnit, UnitKind,
};
use crate::review::context::FileContext;

/// Configuration for a brace-delimited language extractor.
///
/// Pre-configured constructors exist for 7 languages: Rust, C, C++,
/// Go, Java, JavaScript, TypeScript.
#[derive(Clone)]
pub struct BraceFamilyExtractor {
    language: String,
    exts: &'static [&'static str],
    /// Keywords that start a function/method declaration.
    function_keywords: Vec<String>,
    /// Keywords that start a block declaration (struct, class, enum, etc.).
    block_keywords: Vec<String>,
    /// Keywords that start a single-line declaration (const, type, use, etc.).
    declaration_keywords: Vec<String>,
    /// The modifier words a declaration may open with, stripped before the keyword tables are
    /// matched ([`Self::strip_modifiers`]).
    modifiers: &'static Modifiers,
    /// Line prefixes for doc comments (e.g., "///", "//!").
    doc_comment_prefixes: Vec<String>,
    /// Line prefixes for attributes (e.g., "#[").
    attribute_prefixes: Vec<String>,
    /// Lexical rules for the CodeScanner.
    lexical_rules: LexicalRules,
    /// Whether an unindented signature line (`int main(void) {`) starts a function: C and C++
    /// have no function keyword.
    signature_functions: bool,
    /// Whether a function unit is named by the identifier just before its `(`: Java's function
    /// keywords are modifiers (`public static void run(`), so the token after them is a modifier
    /// or a type, not the name.
    name_before_paren: bool,
}

impl LanguageExtractor for BraceFamilyExtractor {
    fn language_id(&self) -> &str {
        &self.language
    }

    fn extensions(&self) -> &[&str] {
        self.exts
    }

    fn extract_units<'a>(
        &self,
        content: &str,
        file_path: &str,
        changed_lines: impl Into<ChangedLines<'a>>,
    ) -> Vec<SemanticUnit> {
        let changed_lines = changed_lines.into();
        if changed_lines.is_empty() {
            return vec![];
        }

        let lines: Vec<&str> = content.lines().collect();
        let mut scanner = CodeScanner::new(self.lexical_rules.clone());
        let scanned = scanner.scan_all(content);

        // Detect all constructs
        let constructs = self.detect_constructs(&lines, &scanned);
        let ranges: Vec<(u32, u32)> = constructs
            .iter()
            .map(|c| (c.start_line, c.end_line))
            .collect();
        // MAGI FIX (S6 loop 3): a whole file hands out its lines by range, without a set of all.
        let total = u32::try_from(lines.len()).unwrap_or(u32::MAX);
        let (shares, orphans) = changed_lines.assign(&ranges, total);

        // Filter to constructs that overlap with changed lines
        let mut units: Vec<SemanticUnit> = Vec::new();
        for (construct, changed_in_unit) in constructs.iter().zip(shares) {
            if !changed_in_unit.is_empty() {
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

        // Orphan fallback: one unit per gap between constructs that holds changed lines.
        units.extend(gap_units(&lines, file_path, &ranges, orphans));

        units
    }

    /// Splits an `impl`, class, trait, interface or namespace at its members: the constructs
    /// between the block's opening brace and its matching closing brace, found as at the top
    /// level (indented functions included). O(n) in the block's lines.
    fn split_block(&self, unit: &SemanticUnit) -> Vec<SemanticUnit> {
        // MAGI FIX (S6 loop 3): a block too large for one batch is reviewed method by method.
        if !matches!(
            unit.kind,
            UnitKind::Impl | UnitKind::Class | UnitKind::Trait
        ) {
            return Vec::new();
        }
        let lines: Vec<&str> = unit.content.lines().collect();
        let scanned = CodeScanner::new(self.lexical_rules.clone()).scan_all(&unit.content);
        let Some(open) = scanned
            .iter()
            .position(|s| !s.is_non_code && s.brace_delta > 0)
        else {
            return Vec::new();
        };
        let close = find_closing_brace(&scanned, open);
        let inner = (open + 1)..close.max(open + 1);
        let (Some(inner_lines), Some(inner_scanned)) =
            (lines.get(inner.clone()), scanned.get(inner))
        else {
            return Vec::new();
        };
        let offset = u32::try_from(open + 1).unwrap_or(u32::MAX);
        let members: Vec<Member> = self
            .detect_constructs(inner_lines, inner_scanned)
            .into_iter()
            .map(|c| {
                (
                    c.kind,
                    c.name,
                    c.start_line.saturating_add(offset),
                    c.end_line.saturating_add(offset),
                )
            })
            .collect();
        split_into_members(unit, members)
    }
}

/// The modifier words of a language, stripped before its keyword tables are matched.
struct Modifiers {
    /// Visibility words that may open any declaration (`pub`, `pub(crate)`, `public`,
    /// `export`): stripped, in any order and number, before every keyword table is matched.
    visibility: &'static [&'static str],
    /// Qualifier words that may open a block or function declaration (`const`, `async`,
    /// `unsafe`, `extern "C"`, `final`): stripped, after the visibility, before the block and
    /// function tables are matched, never the declaration table (Rust's `const X` is a
    /// constant, `const fn` a function).
    qualifiers: &'static [&'static str],
}

/// A language whose declarations open with their keyword.
const NO_MODIFIERS: Modifiers = Modifiers {
    visibility: &[],
    qualifiers: &[],
};

/// Rust's visibility (`pub`, `pub(crate)`, `pub(in path)`) and function qualifiers.
const RUST_MODIFIERS: Modifiers = Modifiers {
    visibility: &["pub"],
    qualifiers: &["const", "async", "unsafe", "extern", "default"],
};

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

            let stops = |j: usize| self.starts_construct(lines, scanned, j);
            if let Some((kind, keywords, matched)) = self.construct_keyword(trimmed) {
                // MAGI FIX (S6 loop 3): a Java method is named by the identifier before `(`.
                let name = Some(kind)
                    .filter(|k| self.name_before_paren && *k == UnitKind::Function)
                    .and_then(|_| identifier_before_paren(trimmed))
                    .unwrap_or_else(|| extract_name_from_line(matched, keywords));
                let start = self.scan_back_for_docs(lines, i);
                let end = brace_block_end(lines, scanned, i, line_num, &stops);
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
            if let Some(name) = self.signature_name(line, scanned.get(i)) {
                let opens = open_brace_line(lines, scanned, i, &stops).is_some();
                if opens || trimmed.ends_with('}') {
                    let start = self.scan_back_for_docs(lines, i);
                    let end = brace_block_end(lines, scanned, i, line_num, &stops);
                    constructs.push(Construct {
                        kind: UnitKind::Function,
                        name,
                        start_line: u32::try_from(start + 1).unwrap_or(1),
                        end_line: end,
                    });
                    i = (end as usize).max(i + 1);
                    continue;
                }
            }
            // Check declaration keywords
            if self.is_declaration(trimmed) {
                let group_start = i;
                let mut group_end = i;
                // Scan forward for contiguous declarations of the same kind
                while group_end + 1 < lines.len() {
                    let next = lines.get(group_end + 1).map_or("", |l| l.trim());
                    if self.is_declaration(next) {
                        group_end += 1;
                    } else if next.is_empty() {
                        // Allow blank lines within groups
                        if lines
                            .get(group_end + 2)
                            .is_some_and(|l| self.is_declaration(l.trim()))
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
                let name = self.declaration_group_name(self.strip_modifiers(trimmed, false));
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

    /// The keyword that opens a block or function construct on a trimmed line, its table, and
    /// the text it matched: the line with its visibility and qualifiers stripped
    /// ([`Self::strip_modifiers`]) or, failing that, the line as written. Block keywords are
    /// tried before function keywords: Java's `public class` would otherwise read as a function
    /// named `class`. O(k + n) for k keywords and a line of n characters.
    fn construct_keyword<'t>(&self, trimmed: &'t str) -> Option<(UnitKind, &[String], &'t str)> {
        // MAGI FIX (S6 loop 4): qualifiers combine freely (`pub(super) async unsafe fn`); the
        // tables of fixed prefixes missed most combinations.
        let candidates = [self.strip_modifiers(trimmed, true), trimmed];
        candidates
            .iter()
            .find_map(|&text| {
                self.match_block_keyword(text)
                    .map(|kind| (kind, self.block_keywords.as_slice(), text))
            })
            .or_else(|| {
                candidates.iter().find_map(|&text| {
                    self.match_function_keyword(text)
                        .map(|kind| (kind, self.function_keywords.as_slice(), text))
                })
            })
    }

    /// Whether a trimmed line opens a declaration: a declaration keyword starts the line as
    /// written or with its visibility stripped (`pub(crate) use`). O(k + n).
    fn is_declaration(&self, trimmed: &str) -> bool {
        self.match_declaration_keyword(trimmed)
            || self.match_declaration_keyword(self.strip_modifiers(trimmed, false))
    }

    /// `trimmed` without the visibility words it opens with and, when `qualifiers`, the
    /// qualifier words after them, in any order and number. A word counts only when a blank or
    /// a `(` follows it (`public`, not `publicity`); a `(...)` right after a word is part of it
    /// (`pub(in crate::a)`), and a quoted string after a word is too (`extern "C"`). O(n) in
    /// the line.
    ///
    /// # Arguments
    ///
    /// * `trimmed` - A trimmed line.
    /// * `qualifiers` - Whether the qualifier words are stripped too.
    ///
    /// # Returns
    ///
    /// The rest of the line, trimmed at its start; `trimmed` when it opens with no such word.
    fn strip_modifiers<'t>(&self, trimmed: &'t str, qualifiers: bool) -> &'t str {
        let words = self
            .modifiers
            .visibility
            .iter()
            .chain(self.modifiers.qualifiers.iter().filter(|_| qualifiers));
        let mut rest = trimmed;
        loop {
            let Some(after) = words.clone().find_map(|w| strip_modifier(rest, w)) else {
                return rest;
            };
            rest = after;
        }
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

    /// Whether line `j` (0-based) is code that opens a construct: a function, block or
    /// declaration keyword, or a signature where [`Self::signature_name`] applies. O(k) in the
    /// keywords.
    fn starts_construct(&self, lines: &[&str], scanned: &[ScannedLine], j: usize) -> bool {
        if scanned.get(j).is_none_or(|s| s.is_non_code) {
            return false;
        }
        let Some(line) = lines.get(j) else {
            return false;
        };
        let trimmed = line.trim();
        self.construct_keyword(trimmed).is_some()
            || self.is_declaration(trimmed)
            || self.signature_name(line, scanned.get(j)).is_some()
    }

    /// The function name of a C-family signature line, for extractors with
    /// `signature_functions`: a line that holds `(`, whose code does not end in `;` (a prototype;
    /// with `scanned`, a comment after the code does not count), does not
    /// open with `#`, a comment or `}`, has no `=` before its `(`, and names an identifier, not a
    /// control keyword, just before the `(` (`int *parse(` gives `parse`, `void Foo::bar(` gives
    /// `bar`). The line may be indented: a function inside a namespace, a class or an
    /// `extern "C"` block is a function too. Only lines outside every construct are asked, so a
    /// call inside a recognised function body is never taken for one. O(n) in the line.
    ///
    /// # Arguments
    ///
    /// * `line` - The line, untrimmed.
    /// * `scanned` - The scanner's record of the line; `None` falls back to the trimmed text.
    ///
    /// # Returns
    ///
    /// The function name, or `None` when the line is no signature.
    fn signature_name(&self, line: &str, scanned: Option<&ScannedLine>) -> Option<String> {
        // MAGI FIX (S6 loop 3): indentation no longer disqualifies a signature.
        if !self.signature_functions {
            return None;
        }
        let trimmed = line.trim();
        // MAGI FIX (S6 loop 4): a trailing comment ending in `;` is not a prototype's `;`.
        let prototype = scanned.map_or(trimmed.ends_with(';'), |s| s.ends_statement);
        if prototype
            || ["#", "//", "/*", "*", "}"]
                .iter()
                .any(|p| trimmed.starts_with(p))
        {
            return None;
        }
        let name = identifier_before_paren(trimmed)?;
        let control = [
            "if", "for", "while", "switch", "return", "sizeof", "else", "do",
        ];
        if control.contains(&name.as_str()) {
            return None;
        }
        Some(name)
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
}

/// The 0-based line holding the opening brace of the construct that starts at line `i`: line
/// `i` when it opens one, else the first later line that does, so a signature wrapped over any
/// number of lines (rustfmt's `fn f(\n    a: u8,\n) -> u8 {`) finds its brace.
///
/// The search ends without a brace when line `i` already ends the declaration (its code ends in
/// `;`), and before the first later line that is blank, is code ending in `;` (a statement: the
/// declaration had no block; a comment ending in `;` does not count), or where `stops` holds (the next construct's start); comment lines
/// are passed over. A line whose braces balance (`fn a() { x(); }`) opens nothing, so a one-line
/// construct never takes the brace of the construct after it. O(k) for the k lines scanned,
/// which never pass the next construct.
fn open_brace_line(
    lines: &[&str],
    scanned: &[ScannedLine],
    i: usize,
    stops: &dyn Fn(usize) -> bool,
) -> Option<usize> {
    let opens = |idx: usize| scanned.get(idx).is_some_and(|s| s.brace_delta > 0);
    let is_code = |idx: usize| scanned.get(idx).is_some_and(|s| !s.is_non_code);
    // MAGI FIX (S6 loop 4): a comment ending in `;` (`// keeps state;`) is not a statement.
    let ends_statement =
        |idx: usize| is_code(idx) && scanned.get(idx).is_some_and(|s| s.ends_statement);
    if opens(i) {
        return Some(i);
    }
    if ends_statement(i) {
        return None;
    }
    // MAGI FIX (S6 loop 3): no fixed two-line cap; the declaration's own end bounds the search.
    ((i + 1)..lines.len())
        .take_while(|&j| {
            !stops(j)
                && !lines.get(j).is_some_and(|l| l.trim().is_empty())
                && (opens(j) || !ends_statement(j))
        })
        .find(|&j| opens(j))
}

/// Find the 1-indexed line where the construct starting at line index `i` ends.
///
/// The opening brace is the one [`open_brace_line`] finds. A construct with no brace (a
/// single-line declaration) ends on `line_num`.
fn brace_block_end(
    lines: &[&str],
    scanned: &[ScannedLine],
    i: usize,
    line_num: u32,
    stops: &dyn Fn(usize) -> bool,
) -> u32 {
    match open_brace_line(lines, scanned, i, stops) {
        Some(open_idx) => {
            let close_idx = find_closing_brace(scanned, open_idx);
            u32::try_from(close_idx + 1).unwrap_or(line_num)
        }
        None => line_num,
    }
}

/// The identifier just before the first `(` of `line` (`int *parse(` gives `parse`,
/// `public static <T> List<T> wrap(` gives `wrap`).
///
/// # Arguments
///
/// * `line` - A signature line.
///
/// # Returns
///
/// `None` when the line has no `(`, holds `=` before it (an assignment or initializer), or no
/// identifier ends right before it. O(n) in the line.
fn identifier_before_paren(line: &str) -> Option<String> {
    let (before, _) = line.split_once('(')?;
    if before.contains('=') {
        return None;
    }
    let name: String = before
        .trim_end()
        .chars()
        .rev()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    Some(name).filter(|n| !n.is_empty())
}

/// `text` without the modifier `word` it opens with, and what belongs to the word: a `(...)`
/// right after it (`pub(crate)`) or, after blanks, a quoted string (`extern "C"`); the rest is
/// trimmed at its start. `None` when `text` does not open with `word` followed by a blank or a
/// `(`. O(n) in the text.
fn strip_modifier<'t>(text: &'t str, word: &str) -> Option<&'t str> {
    let rest = text.strip_prefix(word)?;
    if let Some(inner) = rest.strip_prefix('(') {
        let (_, after) = inner.split_once(')')?;
        return Some(after.trim_start());
    }
    if !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let rest = rest.trim_start();
    match rest.strip_prefix('"').and_then(|r| r.split_once('"')) {
        Some((_, after)) => Some(after.trim_start()),
        None => Some(rest),
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
            exts: &["rs"],
            function_keywords: vec!["fn ".to_string()],
            block_keywords: vec![
                "struct ".to_string(),
                "enum ".to_string(),
                "trait ".to_string(),
                "impl ".to_string(),
                "mod ".to_string(),
            ],
            declaration_keywords: vec![
                "const ".to_string(),
                "static ".to_string(),
                "type ".to_string(),
                "use ".to_string(),
            ],
            modifiers: &RUST_MODIFIERS,
            doc_comment_prefixes: vec!["///".to_string(), "//!".to_string()],
            attribute_prefixes: vec!["#[".to_string()],
            lexical_rules: LexicalRules::rust(),
            signature_functions: false,
            name_before_paren: false,
        }
    }

    /// C extractor.
    pub fn c() -> Self {
        Self {
            language: "c".to_string(),
            exts: &["c", "h"],
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
            modifiers: &NO_MODIFIERS,
            lexical_rules: LexicalRules::c(),
            signature_functions: true,
            name_before_paren: false,
        }
    }

    /// C++ extractor.
    pub fn cpp() -> Self {
        let mut ext = Self::c();
        ext.language = "cpp".to_string();
        ext.exts = &["cpp", "cxx", "cc", "hpp", "hxx"];
        ext.block_keywords
            .extend(["class ".to_string(), "namespace ".to_string()]);
        ext.lexical_rules = LexicalRules::cpp();
        ext
    }

    /// Go extractor.
    pub fn go() -> Self {
        Self {
            language: "go".to_string(),
            exts: &["go"],
            function_keywords: vec!["func ".to_string()],
            block_keywords: vec!["type ".to_string()],
            declaration_keywords: vec![
                "import ".to_string(),
                "var ".to_string(),
                "const ".to_string(),
            ],
            doc_comment_prefixes: vec!["//".to_string()],
            attribute_prefixes: vec![],
            modifiers: &NO_MODIFIERS,
            lexical_rules: LexicalRules::go(),
            signature_functions: false,
            name_before_paren: false,
        }
    }

    /// Java extractor.
    pub fn java() -> Self {
        Self {
            language: "java".to_string(),
            exts: &["java"],
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
            modifiers: &NO_MODIFIERS,
            lexical_rules: LexicalRules::java(),
            signature_functions: false,
            name_before_paren: true,
        }
    }

    /// JavaScript extractor.
    pub fn javascript() -> Self {
        Self {
            language: "javascript".to_string(),
            exts: &["js", "jsx", "mjs"],
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
            modifiers: &NO_MODIFIERS,
            lexical_rules: LexicalRules::javascript(),
            signature_functions: false,
            name_before_paren: false,
        }
    }

    /// TypeScript extractor.
    pub fn typescript() -> Self {
        let mut ext = Self::javascript();
        ext.language = "typescript".to_string();
        ext.exts = &["ts", "tsx", "mts"];
        ext.block_keywords.push("interface ".to_string());
        ext.block_keywords.push("export interface ".to_string());
        ext.declaration_keywords.push("type ".to_string());
        ext.declaration_keywords.push("export type ".to_string());
        ext.lexical_rules = LexicalRules::typescript();
        ext
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

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
        let code = "use a::b;\n\nfn one() {\n    x();\n}\n// note\n\nfn two() {\n    y();\n}\n\nfn three() {\n    z();\n}\n";
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
        let code = "fn a() {}\n\nfn b() {}";
        let units = extract(&rust_extractor(), code, &[1, 2, 3]);
        assert_each_line_in_one_unit(&units, 3);
    }

    #[test]
    fn c_extracts_a_function_by_its_signature() {
        // C has no function keyword: a definition is an unindented signature whose block
        // opens on its line or the next two; a prototype ends in `;` and is not one.
        let code = "#include <stdio.h>\n\nint main(int argc, char *argv[]) {\n    return 0;\n}\n\nint validate(const char *input)\n{\n    return input != NULL;\n}\n\nint prototype(void);\n";
        let units = extract(&c_extractor(), code, &[4, 9, 12]);
        let found: Vec<(&str, UnitKind, u32, u32)> = units
            .iter()
            .map(|u| (u.name.as_str(), u.kind, u.start_line, u.end_line))
            .collect();
        assert!(
            found.contains(&("main", UnitKind::Function, 3, 5)),
            "{found:?}"
        );
        assert!(
            found.contains(&("validate", UnitKind::Function, 7, 10)),
            "{found:?}"
        );
        assert!(
            !found
                .iter()
                .any(|(n, k, ..)| *n == "prototype" && *k == UnitKind::Function),
            "{found:?}"
        );
    }

    #[test]
    fn a_comment_ending_in_a_semicolon_does_not_end_a_signature() {
        // [MAGI S6 loop 4, B2] a line comment ending in `;` read as a statement: the search for
        // the brace stopped there (or the signature read as a prototype) and the body became a
        // gap.
        let between = "int f(void)
// keeps state;
{
    return 0;
}
";
        let units = extract(&c_extractor(), between, &[4]);
        assert_eq!(spans(&units), vec![("f", UnitKind::Function, 1, 5)]);
        let trailing = "int g(void) // returns 0;
{
    return 0;
}
";
        let units = extract(&c_extractor(), trailing, &[3]);
        assert_eq!(spans(&units), vec![("g", UnitKind::Function, 1, 4)]);
        let rust = "pub fn h(
    a: u8, // the count;
) -> u8 {
    a
}
";
        let units = extract(&rust_extractor(), rust, &[4]);
        assert_eq!(spans(&units), vec![("h", UnitKind::Function, 1, 5)]);
    }

    #[test]
    fn a_function_opening_two_blocks_on_one_line_ends_at_its_own_brace() {
        let code = "fn f() { if x {\n        a();\n    }\n    b();\n}\n\nfn g() {\n    c();\n}\n";
        let units = extract(&rust_extractor(), code, &[4]);
        let f = units
            .iter()
            .find(|u| u.name == "f")
            .expect("f holds line 4");
        assert_eq!((f.start_line, f.end_line), (1, 5));
    }

    #[test]
    fn a_one_line_function_does_not_take_the_next_construct() {
        let code = "fn a() { x(); }\nfn b() {\n    y();\n}\n";
        let units = extract(&rust_extractor(), code, &[3]);
        assert_eq!(units.len(), 1, "{units:?}");
        assert_eq!(units[0].name, "b");
        assert_eq!((units[0].start_line, units[0].end_line), (2, 4));
    }

    #[test]
    fn a_java_class_is_a_class_not_a_function() {
        let code = "public class Foo {\n    public void run() {\n    }\n}\n";
        let units = extract(&BraceFamilyExtractor::java(), code, &[3]);
        assert_eq!(units.len(), 1, "{units:?}");
        assert_eq!(
            (units[0].kind, units[0].name.as_str()),
            (UnitKind::Class, "Foo")
        );
    }

    /// `(name, kind, start, end)` of each unit.
    fn spans(units: &[SemanticUnit]) -> Vec<(&str, UnitKind, u32, u32)> {
        units
            .iter()
            .map(|u| (u.name.as_str(), u.kind, u.start_line, u.end_line))
            .collect()
    }

    #[test]
    fn a_signature_wrapped_over_several_lines_finds_its_brace() {
        // [MAGI S6 loop 3, B3] the brace was sought only two lines after the keyword, so a
        // rustfmt-wrapped signature collapsed to its first line and the body became a gap.
        let rust = "pub fn plan(\n    a: u8,\n    b: u8,\n    c: u8,\n) -> u8 {\n    a\n}\nfn g() {\n    x();\n}\n";
        let units = extract(&rust_extractor(), rust, &(1..=10).collect::<Vec<_>>());
        assert_eq!(
            spans(&units),
            vec![
                ("plan", UnitKind::Function, 1, 7),
                ("g", UnitKind::Function, 8, 10)
            ]
        );
        let c = "int f(int a,\n      int b,\n      int c)\n{\n    return a;\n}\n";
        let units = extract(&c_extractor(), c, &[5]);
        assert_eq!(spans(&units), vec![("f", UnitKind::Function, 1, 6)]);
    }

    #[test]
    fn a_declaration_ended_by_a_semicolon_or_a_blank_line_takes_no_later_brace() {
        // [MAGI S6 loop 3, B3] with the lookahead unbounded, a complete declaration must not
        // reach the brace of the code after it.
        let java = "static final int X = 1;\nvoid run() {\n    a();\n}\n";
        let units = extract(&BraceFamilyExtractor::java(), java, &[1]);
        assert_eq!(
            units
                .iter()
                .map(|u| (u.start_line, u.end_line))
                .collect::<Vec<_>>(),
            vec![(1, 1)],
            "{:?}",
            spans(&units)
        );
        let rust = "struct A;\n\nmacro_rules! m {\n    () => {};\n}\n";
        let units = extract(&rust_extractor(), rust, &[1]);
        assert_eq!(spans(&units), vec![("A", UnitKind::Struct, 1, 1)]);
        let rust = "struct B(u8)\n\nmacro_rules! m {\n    () => {};\n}\n";
        let units = extract(&rust_extractor(), rust, &[1]);
        assert_eq!(spans(&units), vec![("B", UnitKind::Struct, 1, 1)]);
    }

    /// The one unit of `kind` that `code` gives when line `changed` changes, split one level.
    fn split(
        ext: &BraceFamilyExtractor,
        code: &str,
        changed: u32,
        kind: UnitKind,
    ) -> Vec<SemanticUnit> {
        let every: HashSet<u32> =
            (1..=u32::try_from(code.lines().count()).expect("small")).collect();
        let block = ext
            .extract_units(code, "test.rs", &every)
            .into_iter()
            .find(|u| u.kind == kind && u.start_line <= changed && changed <= u.end_line)
            .expect("the block");
        ext.split_block(&block)
    }

    #[test]
    fn an_impl_block_splits_into_its_methods_and_keeps_its_header() {
        // [MAGI S6 loop 3, B4] an impl or class was one unit: one larger than the batch failed
        // the review although each of its methods fits.
        let code = "/// Doc.\nimpl Foo {\n    const N: u8 = 1;\n\n    /// A.\n    pub fn a(&self) {\n        x();\n    }\n\n    fn b(\n        &self,\n    ) -> u8 {\n        1\n    }\n}\n";
        let parts = split(&rust_extractor(), code, 7, UnitKind::Impl);
        assert_eq!(
            spans(&parts),
            vec![
                ("Foo", UnitKind::Impl, 1, 2),
                ("constants", UnitKind::DeclarationGroup, 3, 3),
                ("Foo", UnitKind::Impl, 4, 4),
                ("a", UnitKind::Method, 5, 8),
                ("Foo", UnitKind::Impl, 9, 9),
                ("b", UnitKind::Method, 10, 14),
                ("Foo", UnitKind::Impl, 15, 15),
            ]
        );
        assert_eq!(parts[0].content, "/// Doc.\nimpl Foo {");
        assert_eq!(
            parts[3].content,
            "    /// A.\n    pub fn a(&self) {\n        x();\n    }"
        );
        assert_eq!(parts[3].changed_lines, vec![5, 6, 7, 8]);
        assert_each_line_in_one_unit(&parts, 15);
    }

    #[test]
    fn a_java_class_and_a_cpp_namespace_split_into_their_members() {
        let java = "public class Foo {\n    private int n;\n\n    public void run() {\n        a();\n    }\n}\n";
        let parts = split(&BraceFamilyExtractor::java(), java, 5, UnitKind::Class);
        assert!(
            spans(&parts).contains(&("run", UnitKind::Method, 4, 6)),
            "{:?}",
            spans(&parts)
        );
        assert_each_line_in_one_unit(&parts, 7);
        let cpp = "namespace ns {\n    int f(int a) {\n        return a;\n    }\n}\n";
        let parts = split(&BraceFamilyExtractor::cpp(), cpp, 3, UnitKind::Class);
        assert_eq!(
            spans(&parts),
            vec![
                ("ns", UnitKind::Class, 1, 1),
                ("f", UnitKind::Method, 2, 4),
                ("ns", UnitKind::Class, 5, 5)
            ]
        );
    }

    #[test]
    fn a_function_or_a_block_without_members_does_not_split() {
        let code = "fn f() {\n    x();\n}\nstruct S {\n    a: u8,\n}\n";
        assert!(split(&rust_extractor(), code, 2, UnitKind::Function).is_empty());
        assert!(split(&rust_extractor(), code, 5, UnitKind::Struct).is_empty());
    }

    #[test]
    fn an_indented_c_function_is_a_function() {
        // [MAGI S6 loop 3, B5] a signature was a function only unindented, so the functions of
        // an `extern "C"` block, a namespace or a class were never units.
        let code = "extern \"C\" {\n    int add(int a, int b) {\n        return a + b;\n    }\n\n    static void log_it(void)\n    {\n        puts(\"x\");\n    }\n}\n";
        for ext in [c_extractor(), BraceFamilyExtractor::cpp()] {
            let units = extract(&ext, code, &[3, 8]);
            let found = spans(&units);
            assert!(
                found.contains(&("add", UnitKind::Function, 2, 4)),
                "{found:?}"
            );
            assert!(
                found.contains(&("log_it", UnitKind::Function, 6, 9)),
                "{found:?}"
            );
        }
    }

    #[test]
    fn a_java_method_is_named_by_the_identifier_before_its_parameters() {
        // [MAGI S6 loop 3, B8] the modifier keyword left `static` or `void` as the name.
        let code = "public static void run() {\n    a();\n}\nprivate int count(String s) {\n    return 0;\n}\npublic static <T> List<T> wrap(T t) {\n    return null;\n}\nprotected Foo(int x) {\n    b();\n}\n";
        let units = extract(&BraceFamilyExtractor::java(), code, &[2, 5, 8, 11]);
        let names: Vec<&str> = units.iter().map(|u| u.name.as_str()).collect();
        assert_eq!(names, vec!["run", "count", "wrap", "Foo"]);
    }

    #[test]
    fn rust_qualifiers_in_any_combination_are_recognised() {
        // [MAGI S6 loop 4, B2] the keyword table listed fixed forms, so `pub async unsafe fn`
        // or `pub(super) fn` matched nothing and the function became a gap.
        for (code, kind, name) in [
            (
                "pub async unsafe fn f() {\n    x();\n}\n",
                UnitKind::Function,
                "f",
            ),
            (
                "pub(crate) extern \"C\" fn cb() {\n    x();\n}\n",
                UnitKind::Function,
                "cb",
            ),
            (
                "pub(super) fn g() {\n    x();\n}\n",
                UnitKind::Function,
                "g",
            ),
            (
                "pub(in crate::a) const fn k() -> u8 {\n    1\n}\n",
                UnitKind::Function,
                "k",
            ),
            (
                "async unsafe fn h() {\n    x();\n}\n",
                UnitKind::Function,
                "h",
            ),
            ("extern fn e() {\n    x();\n}\n", UnitKind::Function, "e"),
            (
                "pub(super) struct S {\n    a: u8,\n}\n",
                UnitKind::Struct,
                "S",
            ),
            (
                "unsafe impl Send for S {\n    // none\n}\n",
                UnitKind::Impl,
                "Send",
            ),
            (
                "pub(crate) unsafe trait T {\n    fn t();\n}\n",
                UnitKind::Trait,
                "T",
            ),
        ] {
            let units = extract(&rust_extractor(), code, &[2]);
            assert_eq!(spans(&units), vec![(name, kind, 1, 3)], "{code}");
        }
        let uses = "pub(crate) use a::b;\npub(super) use c::d;\n";
        let units = extract(&rust_extractor(), uses, &[1, 2]);
        assert_eq!(
            spans(&units),
            vec![("imports", UnitKind::DeclarationGroup, 1, 2)]
        );
        let consts = "pub(super) const A: u8 = 1;\n";
        let units = extract(&rust_extractor(), consts, &[1]);
        assert_eq!(
            spans(&units),
            vec![("constants", UnitKind::DeclarationGroup, 1, 1)]
        );
    }

    #[test]
    fn java_modifiers_in_any_combination_name_the_declaration() {
        // [MAGI S6 loop 4, B2] `public final class` matched the function keyword `public ` and
        // became a function named `final`, which cannot split.
        for (code, kind, name) in [
            (
                "public final class C {\n    int a;\n}\n",
                UnitKind::Class,
                "C",
            ),
            ("abstract class A {\n    int a;\n}\n", UnitKind::Class, "A"),
            (
                "public sealed interface I {\n    int a();\n}\n",
                UnitKind::Trait,
                "I",
            ),
            ("record R(int x) {\n    int y;\n}\n", UnitKind::Class, "R"),
            (
                "public static enum E {\n    A, B;\n}\n",
                UnitKind::Enum,
                "E",
            ),
        ] {
            let units = extract(&BraceFamilyExtractor::java(), code, &[2]);
            assert_eq!(spans(&units), vec![(name, kind, 1, 3)], "{code}");
        }
    }

    #[test]
    fn qualified_rust_functions_are_functions() {
        for (code, name) in [
            ("pub const fn f() -> u8 {\n    1\n}\n", "f"),
            ("const fn f() -> u8 {\n    1\n}\n", "f"),
            ("unsafe fn g() {\n    h();\n}\n", "g"),
            ("pub unsafe fn g() {\n    h();\n}\n", "g"),
            ("extern \"C\" fn cb() {\n    h();\n}\n", "cb"),
        ] {
            let units = extract(&rust_extractor(), code, &[2]);
            assert_eq!(units.len(), 1, "{code}: {units:?}");
            assert_eq!(
                (units[0].kind, units[0].name.as_str(), units[0].end_line),
                (UnitKind::Function, name, 3),
                "{code}"
            );
        }
    }
}
