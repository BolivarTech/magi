// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! Lexical-aware source code scanner for brace counting.
//!
//! The [`CodeScanner`] is a single-pass state machine that tracks lexical
//! context (code, string, comment, raw string, template literal) and counts
//! braces only in code context. This avoids false brace matches inside
//! strings, comments, and character literals.
//!
//! # Examples
//!
//! ```
//! use magi_rs::review::languages::scanner::{CodeScanner, LexicalRules};
//!
//! let mut scanner = CodeScanner::new(LexicalRules::rust());
//! let results = scanner.scan_all("fn main() {\n    let s = \"{\";\n}\n");
//! assert_eq!(results[0].brace_delta, 1);  // fn main() {
//! assert_eq!(results[1].brace_delta, 0);  // brace in string — not counted
//! assert_eq!(results[2].brace_delta, -1); // closing }
//! ```

/// Raw string syntax variants across languages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawStringSyntax {
    /// No raw string support.
    None,
    /// Rust-style: `r#"..."#` with variable hash count.
    RustHashes,
    /// Python-style: `"""..."""` or `'''...'''`.
    TripleQuote,
    /// Go-style: backtick raw strings (`` `...` ``).
    GoBacktick,
}

/// Per-language lexical rules that configure the scanner.
///
/// Each brace-family language provides a `LexicalRules` instance
/// that tells the scanner which syntactic constructs exist.
#[derive(Debug, Clone)]
pub struct LexicalRules {
    /// Line comment start token (e.g., `"//"`). Empty string = none.
    pub line_comment: &'static str,
    /// Whether `/* */` block comments exist.
    pub block_comments: bool,
    /// Whether block comments can nest (`/* /* */ */` — Rust only).
    pub nested_comments: bool,
    /// Whether the language has character literals with `'` (C, Rust, Java).
    /// `false` for Python (single-quote strings) and JS (single-quote strings).
    pub char_literals: bool,
    /// Raw string syntax variant.
    pub raw_strings: RawStringSyntax,
    /// Whether backtick template literals exist (JS/TS only).
    pub template_literals: bool,
}

impl LexicalRules {
    /// Rust: `//` line comments, `/* */` nested block comments,
    /// `'` char literals, `r#"..."#` raw strings.
    pub fn rust() -> Self {
        Self {
            line_comment: "//",
            block_comments: true,
            nested_comments: true,
            char_literals: true,
            raw_strings: RawStringSyntax::RustHashes,
            template_literals: false,
        }
    }

    /// C: `//` line comments, `/* */` non-nested block comments,
    /// `'` char literals.
    pub fn c() -> Self {
        Self {
            line_comment: "//",
            block_comments: true,
            nested_comments: false,
            char_literals: true,
            raw_strings: RawStringSyntax::None,
            template_literals: false,
        }
    }

    /// C++: Same as C. (C++11 raw strings `R"(...)"` are rare
    /// enough to ignore in heuristic parsing.)
    pub fn cpp() -> Self {
        Self::c()
    }

    /// Go: `//` + `/* */`, char literals, backtick raw strings.
    pub fn go() -> Self {
        Self {
            line_comment: "//",
            block_comments: true,
            nested_comments: false,
            char_literals: true,
            raw_strings: RawStringSyntax::GoBacktick,
            template_literals: false,
        }
    }

    /// Java: `//` + `/* */`, char literals.
    pub fn java() -> Self {
        Self {
            line_comment: "//",
            block_comments: true,
            nested_comments: false,
            char_literals: true,
            raw_strings: RawStringSyntax::None,
            template_literals: false,
        }
    }

    /// JavaScript: `//` + `/* */`, backtick template literals.
    /// Single-quote `'` is a string delimiter (NOT char literal).
    pub fn javascript() -> Self {
        Self {
            line_comment: "//",
            block_comments: true,
            nested_comments: false,
            char_literals: false,
            raw_strings: RawStringSyntax::None,
            template_literals: true,
        }
    }

    /// TypeScript: same lexical rules as JavaScript.
    pub fn typescript() -> Self {
        Self::javascript()
    }

    /// Python: `#` line comments, no block comments, no char literals,
    /// triple-quote raw strings.
    pub fn python() -> Self {
        Self {
            line_comment: "#",
            block_comments: false,
            nested_comments: false,
            char_literals: false,
            raw_strings: RawStringSyntax::TripleQuote,
            template_literals: false,
        }
    }
}

/// Result of scanning a single source line.
#[derive(Debug, Clone)]
pub struct ScannedLine {
    /// 1-indexed line number.
    pub line_number: u32,
    /// Net brace depth change on this line (code context only).
    /// Positive = net opening braces, negative = net closing braces.
    pub brace_delta: i32,
    /// Cumulative brace depth AFTER this line (saturates at 0).
    pub depth_after: u32,
    /// `true` if the entire line is inside a comment or string
    /// (no code tokens on this line). Used to skip keyword detection.
    pub is_non_code: bool,
}

/// Context-aware source code scanner.
///
/// Tracks lexical context (strings, comments, raw strings) across
/// lines to provide accurate brace depth counting. The scanner is
/// stateful: call [`scan_line`](Self::scan_line) in order, or use
/// [`scan_all`](Self::scan_all) for the entire file.
///
/// # Examples
///
/// ```
/// use magi_rs::review::languages::scanner::{CodeScanner, LexicalRules};
///
/// let mut scanner = CodeScanner::new(LexicalRules::rust());
/// let results = scanner.scan_all("fn main() {\n    let s = \"{\";\n}\n");
/// assert_eq!(results[0].brace_delta, 1);  // fn main() {
/// assert_eq!(results[1].brace_delta, 0);  // brace in string — not counted
/// assert_eq!(results[2].brace_delta, -1); // closing }
/// ```
pub struct CodeScanner {
    #[allow(dead_code)] // Red stub
    rules: LexicalRules,
}

impl CodeScanner {
    /// Create a new scanner with language-specific lexical rules.
    pub fn new(_rules: LexicalRules) -> Self {
        todo!() // Red stub
    }

    /// Scan a single line, updating internal state.
    ///
    /// Must be called in order (line 1, line 2, ...) because the
    /// scanner carries context across lines (e.g., multi-line strings,
    /// block comments).
    pub fn scan_line(&mut self, _line_number: u32, _line: &str) -> ScannedLine {
        todo!() // Red stub
    }

    /// Scan all lines of a source file.
    ///
    /// Convenience method that splits on `\n` and calls
    /// [`scan_line`](Self::scan_line) for each line.
    pub fn scan_all(&mut self, _content: &str) -> Vec<ScannedLine> {
        Vec::new() // Red stub
    }

    /// Current cumulative brace depth.
    pub fn current_depth(&self) -> u32 {
        0 // Red stub
    }
}

/// Find the closing brace for a construct that opens at `open_idx`.
///
/// Uses pre-scanned brace depths instead of naive counting.
/// The closing brace is the first line after `open_idx` where
/// `depth_after` returns to the level it was at before the
/// opening brace.
///
/// # Arguments
///
/// * `scanned` - Pre-scanned line results from [`CodeScanner::scan_all`].
/// * `open_idx` - Index into `scanned` of the line containing the opening brace.
///
/// # Returns
///
/// Index into `scanned` of the line containing the matching closing brace.
/// If no closing brace is found, returns the last line index (graceful degradation).
pub fn find_closing_brace(_scanned: &[ScannedLine], _open_idx: usize) -> usize {
    0 // Red stub
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Test 1: scanner_counts_braces_in_code ---
    #[test]
    fn scanner_counts_braces_in_code() {
        let mut scanner = CodeScanner::new(LexicalRules::rust());
        let results = scanner.scan_all("fn foo() {\n    bar();\n}");
        assert_eq!(results[0].brace_delta, 1, "Opening brace on line 1");
        assert_eq!(results[1].brace_delta, 0, "No braces on line 2");
        assert_eq!(results[2].brace_delta, -1, "Closing brace on line 3");
        assert_eq!(results[0].depth_after, 1);
        assert_eq!(results[2].depth_after, 0);
    }

    // --- Test 2: scanner_ignores_braces_in_string_literal ---
    #[test]
    fn scanner_ignores_braces_in_string_literal() {
        let mut scanner = CodeScanner::new(LexicalRules::rust());
        let results = scanner.scan_all("let s = \"{\";");
        assert_eq!(
            results[0].brace_delta, 0,
            "Brace inside string should not be counted"
        );
    }

    // --- Test 3: scanner_ignores_braces_in_line_comment ---
    #[test]
    fn scanner_ignores_braces_in_line_comment() {
        let mut scanner = CodeScanner::new(LexicalRules::rust());
        let results = scanner.scan_all("// { not a brace");
        assert_eq!(
            results[0].brace_delta, 0,
            "Brace in line comment should not be counted"
        );
    }

    // --- Test 4: scanner_ignores_braces_in_block_comment ---
    #[test]
    fn scanner_ignores_braces_in_block_comment() {
        let mut scanner = CodeScanner::new(LexicalRules::rust());
        let results = scanner.scan_all("/* { } */");
        assert_eq!(
            results[0].brace_delta, 0,
            "Braces in block comment should not be counted"
        );
    }

    // --- Test 5: scanner_ignores_braces_in_char_literal ---
    #[test]
    fn scanner_ignores_braces_in_char_literal() {
        let mut scanner = CodeScanner::new(LexicalRules::rust());
        let results = scanner.scan_all("let c = '{';");
        assert_eq!(
            results[0].brace_delta, 0,
            "Brace in char literal should not be counted"
        );
    }

    // --- Test 6: scanner_handles_rust_raw_string ---
    #[test]
    fn scanner_handles_rust_raw_string() {
        let mut scanner = CodeScanner::new(LexicalRules::rust());
        let results = scanner.scan_all("let s = r#\"{ }\"#;");
        assert_eq!(
            results[0].brace_delta, 0,
            "Braces in raw string should not be counted"
        );
    }

    // --- Test 7: scanner_handles_nested_block_comments_rust ---
    #[test]
    fn scanner_handles_nested_block_comments_rust() {
        let mut scanner = CodeScanner::new(LexicalRules::rust());
        let results = scanner.scan_all("/* outer /* inner { } */ still comment { } */\nlet x = 1;");
        assert_eq!(
            results[0].brace_delta, 0,
            "Braces in nested block comment should not be counted"
        );
        assert!(
            results[0].is_non_code || results[0].brace_delta == 0,
            "Line should be non-code or have 0 delta"
        );
        assert!(!results[1].is_non_code, "Line after comment is code");
    }

    // --- Test 8: scanner_non_nested_block_comments_c ---
    #[test]
    fn scanner_non_nested_block_comments_c() {
        let mut scanner = CodeScanner::new(LexicalRules::c());
        // In C, block comments don't nest: /* start /* not nested */ ends here
        // After first */, we're back in code
        let results = scanner.scan_all("/* start /* not nested */ {");
        assert_eq!(
            results[0].brace_delta, 1,
            "After non-nested block comment ends, brace in code should be counted"
        );
    }

    // --- Test 9: scanner_handles_escape_in_string ---
    #[test]
    fn scanner_handles_escape_in_string() {
        let mut scanner = CodeScanner::new(LexicalRules::rust());
        let results = scanner.scan_all("let s = \"escaped \\\" still string {\";");
        assert_eq!(
            results[0].brace_delta, 0,
            "Brace after escaped quote in string should not be counted"
        );
    }

    // --- Test 10: scanner_handles_multiline_string ---
    #[test]
    fn scanner_handles_multiline_string() {
        let mut scanner = CodeScanner::new(LexicalRules::rust());
        // Three lines: line 1 opens a string, line 2 is entirely inside,
        // line 3 closes the string. The scanner doesn't enforce language
        // rules about line breaks in strings — it just tracks state.
        let results = scanner.scan_all("let s = \"hello\nworld { }\n\";");
        assert_eq!(results.len(), 3);
        assert_eq!(
            results[1].brace_delta, 0,
            "Braces on second line of multi-line string should not be counted"
        );
        assert!(
            results[1].is_non_code,
            "Second line should be marked as non-code (inside string)"
        );
    }

    // --- Test 11: scanner_handles_js_template_literal ---
    #[test]
    fn scanner_handles_js_template_literal() {
        let mut scanner = CodeScanner::new(LexicalRules::javascript());
        let results = scanner.scan_all("let s = `template { literal`;");
        assert_eq!(
            results[0].brace_delta, 0,
            "Brace in template literal should not be counted"
        );
    }

    // --- Test 12: scanner_handles_go_backtick_raw_string ---
    #[test]
    fn scanner_handles_go_backtick_raw_string() {
        let mut scanner = CodeScanner::new(LexicalRules::go());
        let results = scanner.scan_all("var s = `raw { string`");
        assert_eq!(
            results[0].brace_delta, 0,
            "Brace in Go backtick raw string should not be counted"
        );
    }

    // --- Test 13: scanner_lifetime_not_char_literal ---
    #[test]
    fn scanner_lifetime_not_char_literal() {
        let mut scanner = CodeScanner::new(LexicalRules::rust());
        let results = scanner.scan_all("fn foo<'a>(x: &'a str) {");
        assert_eq!(
            results[0].brace_delta, 1,
            "Opening brace should be counted; 'a is a lifetime, not char literal"
        );
    }

    // --- Test 14: scanner_mixed_code_and_string_on_one_line ---
    #[test]
    fn scanner_mixed_code_and_string_on_one_line() {
        let mut scanner = CodeScanner::new(LexicalRules::rust());
        let results = scanner.scan_all("let x = \"{\"; let y = 1;");
        assert!(
            !results[0].is_non_code,
            "Line with code and string should not be marked as non-code"
        );
        assert_eq!(
            results[0].brace_delta, 0,
            "Brace in string should not be counted"
        );
    }

    // --- Test 15: scanner_depth_tracks_across_lines ---
    #[test]
    fn scanner_depth_tracks_across_lines() {
        let mut scanner = CodeScanner::new(LexicalRules::rust());
        let code = "fn foo() {\n    if true {\n        bar();\n    }\n}";
        let results = scanner.scan_all(code);
        assert_eq!(results[0].depth_after, 1, "After fn foo() {{");
        assert_eq!(results[1].depth_after, 2, "After if true {{");
        assert_eq!(results[2].depth_after, 2, "bar() stays at depth 2");
        assert_eq!(results[3].depth_after, 1, "After inner }}");
        assert_eq!(results[4].depth_after, 0, "After outer }}");
    }

    // --- Test 16: scanner_depth_saturates_at_zero ---
    #[test]
    fn scanner_depth_saturates_at_zero() {
        let mut scanner = CodeScanner::new(LexicalRules::rust());
        let results = scanner.scan_all("}\n}");
        assert_eq!(
            results[0].depth_after, 0,
            "Extra closing brace should saturate at 0"
        );
        assert_eq!(
            results[1].depth_after, 0,
            "Second extra closing brace should still be 0"
        );
    }

    // --- Test 17: find_closing_brace_uses_scanned_depth ---
    #[test]
    fn find_closing_brace_uses_scanned_depth() {
        let mut scanner = CodeScanner::new(LexicalRules::rust());
        let code = "fn foo() {\n    if true {\n        bar();\n    }\n}";
        let scanned = scanner.scan_all(code);
        // find_closing_brace for line 0 (fn foo() {) should find line 4 (outer })
        let close = find_closing_brace(&scanned, 0);
        assert_eq!(close, 4, "Closing brace for fn foo() should be at line 5");
        // find_closing_brace for line 1 (if true {) should find line 3 (inner })
        let close = find_closing_brace(&scanned, 1);
        assert_eq!(close, 3, "Closing brace for if true should be at line 4");
    }

    // --- Test 18: find_closing_brace_skips_braces_in_strings ---
    #[test]
    fn find_closing_brace_skips_braces_in_strings() {
        let mut scanner = CodeScanner::new(LexicalRules::rust());
        let code = "fn foo() {\n    let s = \"}\";\n}";
        let scanned = scanner.scan_all(code);
        let close = find_closing_brace(&scanned, 0);
        assert_eq!(
            close, 2,
            "Should skip brace in string and find actual closing brace"
        );
    }
}
