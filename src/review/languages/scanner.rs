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

/// Character at `i`, or NUL past the end of the line.
///
/// Reads never index: release builds abort on panic (`panic = "abort"`), so the scanner reads
/// through `.get()`. Every call site is guarded by the loop bounds, so the NUL fallback is
/// never observed on a path that previously indexed in range.
fn char_at(chars: &[char], i: usize) -> char {
    chars.get(i).copied().unwrap_or('\0')
}

/// Lexical context states for the code scanner.
#[derive(Debug, Clone, PartialEq, Eq)]
enum LexicalState {
    /// Normal code — braces are structural.
    Code,
    /// Inside a line comment (until end of line).
    LineComment,
    /// Inside a block comment (`/* ... */`).
    /// `depth` tracks nesting (Rust supports nested block comments).
    BlockComment { depth: u32 },
    /// Inside a string literal (regular or byte string).
    /// `quote` is the delimiter (`"` or `'` or `` ` ``).
    /// `escapable` is `false` for raw strings.
    StringLiteral { quote: char, escapable: bool },
    /// Inside a Rust raw string: `r###"..."###`.
    /// `hashes` is the number of `#` delimiters.
    RawString { hashes: u32 },
    /// Inside a Python triple-quoted string (`"""` or `'''`).
    TripleQuote { quote: char },
    /// Inside a character literal (`'...'` in C/Rust/Java).
    CharLiteral,
    /// Inside a JS/TS template literal (`` `...` ``).
    TemplateLiteral,
}

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
    /// `true` when the line starts inside a multi-line comment or string and never returns to
    /// code. A line that starts in code is `false` even when it is all comment (`// note`):
    /// keyword detection does not need to skip it, since no keyword opens with a comment.
    pub is_non_code: bool,
    /// `true` when the line starts inside a multi-line comment or string, whether or not it
    /// returns to code: its leading whitespace is literal text, not indentation.
    pub starts_in_literal: bool,
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
    rules: LexicalRules,
    state: LexicalState,
    depth: u32,
}

impl CodeScanner {
    /// Create a new scanner with language-specific lexical rules.
    pub fn new(rules: LexicalRules) -> Self {
        Self {
            rules,
            state: LexicalState::Code,
            depth: 0,
        }
    }

    /// Scan a single line, updating internal state.
    ///
    /// Must be called in order (line 1, line 2, ...) because the
    /// scanner carries context across lines (e.g., multi-line strings,
    /// block comments).
    pub fn scan_line(&mut self, line_number: u32, line: &str) -> ScannedLine {
        let chars: Vec<char> = line.chars().collect();
        let len = chars.len();
        let mut i = 0;
        let mut delta: i32 = 0;
        let was_code_at_start = self.state == LexicalState::Code;
        let mut entered_code = false;

        while i < len {
            match &self.state {
                LexicalState::Code => {
                    entered_code = true;

                    // Check for line comment
                    if !self.rules.line_comment.is_empty() {
                        let lc = self.rules.line_comment;
                        if lc == "//"
                            && i + 1 < len
                            && char_at(&chars, i) == '/'
                            && char_at(&chars, i + 1) == '/'
                        {
                            self.state = LexicalState::LineComment;
                            break; // rest of line is comment
                        }
                        if lc == "#" && char_at(&chars, i) == '#' {
                            self.state = LexicalState::LineComment;
                            break;
                        }
                    }

                    // Check for block comment
                    if self.rules.block_comments
                        && i + 1 < len
                        && char_at(&chars, i) == '/'
                        && char_at(&chars, i + 1) == '*'
                    {
                        self.state = LexicalState::BlockComment { depth: 1 };
                        i += 2;
                        continue;
                    }

                    // Check for Rust raw string: r#"..."# or br#"..."#
                    if self.rules.raw_strings == RawStringSyntax::RustHashes {
                        if let Some(hashes) = self.try_rust_raw_string(&chars, i) {
                            self.state = LexicalState::RawString { hashes };
                            // Skip past r[b]###"
                            let mut skip = if char_at(&chars, i) == 'b' { 2 } else { 1 }; // b?r
                            skip += hashes as usize; // #'s
                            skip += 1; // opening "
                            i += skip;
                            continue;
                        }
                    }

                    // Check for triple-quote strings (Python)
                    if self.rules.raw_strings == RawStringSyntax::TripleQuote
                        && i + 2 < len
                        && (char_at(&chars, i) == '"' || char_at(&chars, i) == '\'')
                        && char_at(&chars, i + 1) == char_at(&chars, i)
                        && char_at(&chars, i + 2) == char_at(&chars, i)
                    {
                        let q = char_at(&chars, i);
                        self.state = LexicalState::TripleQuote { quote: q };
                        i += 3;
                        continue;
                    }

                    // Check for Go backtick raw string
                    if self.rules.raw_strings == RawStringSyntax::GoBacktick
                        && char_at(&chars, i) == '`'
                    {
                        self.state = LexicalState::StringLiteral {
                            quote: '`',
                            escapable: false,
                        };
                        i += 1;
                        continue;
                    }

                    // Check for template literal (JS/TS)
                    if self.rules.template_literals && char_at(&chars, i) == '`' {
                        self.state = LexicalState::TemplateLiteral;
                        i += 1;
                        continue;
                    }

                    // Check for character literal (Rust/C/Java)
                    if self.rules.char_literals && char_at(&chars, i) == '\'' {
                        if self.is_char_literal(&chars, i) {
                            self.state = LexicalState::CharLiteral;
                            i += 1; // skip opening quote, content handled in CharLiteral state
                            continue;
                        }
                        // Not a char literal (e.g., Rust lifetime) — stay in Code
                        i += 1;
                        continue;
                    }

                    // Check for string literal with "
                    if char_at(&chars, i) == '"' {
                        self.state = LexicalState::StringLiteral {
                            quote: '"',
                            escapable: true,
                        };
                        i += 1;
                        continue;
                    }

                    // Check for single-quote string (non-char-literal languages like JS/Python)
                    if !self.rules.char_literals && char_at(&chars, i) == '\'' {
                        self.state = LexicalState::StringLiteral {
                            quote: '\'',
                            escapable: true,
                        };
                        i += 1;
                        continue;
                    }

                    // Check for byte string: b"..."
                    if char_at(&chars, i) == 'b' && i + 1 < len && char_at(&chars, i + 1) == '"' {
                        self.state = LexicalState::StringLiteral {
                            quote: '"',
                            escapable: true,
                        };
                        i += 2;
                        continue;
                    }

                    // Count braces
                    if char_at(&chars, i) == '{' {
                        delta += 1;
                        self.depth = self.depth.saturating_add(1);
                    } else if char_at(&chars, i) == '}' {
                        delta -= 1;
                        self.depth = self.depth.saturating_sub(1);
                    }

                    i += 1;
                }

                LexicalState::LineComment => {
                    // Consume rest of line (handled by break above)
                    break;
                }

                LexicalState::BlockComment { depth: d } => {
                    let d = *d;
                    if i + 1 < len && char_at(&chars, i) == '*' && char_at(&chars, i + 1) == '/' {
                        if d <= 1 {
                            self.state = LexicalState::Code;
                        } else {
                            self.state = LexicalState::BlockComment {
                                depth: d.saturating_sub(1),
                            };
                        }
                        i += 2;
                        continue;
                    }
                    if self.rules.nested_comments
                        && i + 1 < len
                        && char_at(&chars, i) == '/'
                        && char_at(&chars, i + 1) == '*'
                    {
                        self.state = LexicalState::BlockComment {
                            depth: d.saturating_add(1),
                        };
                        i += 2;
                        continue;
                    }
                    i += 1;
                }

                LexicalState::StringLiteral { quote, escapable } => {
                    let q = *quote;
                    let esc = *escapable;
                    if esc && char_at(&chars, i) == '\\' {
                        i += 2; // skip escaped char
                        continue;
                    }
                    if char_at(&chars, i) == q {
                        self.state = LexicalState::Code;
                        i += 1;
                        continue;
                    }
                    i += 1;
                }

                LexicalState::RawString { hashes } => {
                    let h = *hashes;
                    // Look for closing " followed by h #'s
                    if char_at(&chars, i) == '"' {
                        let mut count = 0u32;
                        let mut j = i + 1;
                        while j < len && char_at(&chars, j) == '#' && count < h {
                            count += 1;
                            j += 1;
                        }
                        if count == h {
                            self.state = LexicalState::Code;
                            i = j;
                            continue;
                        }
                    }
                    i += 1;
                }

                LexicalState::TripleQuote { quote } => {
                    let q = *quote;
                    if i + 2 < len
                        && char_at(&chars, i) == q
                        && char_at(&chars, i + 1) == q
                        && char_at(&chars, i + 2) == q
                    {
                        self.state = LexicalState::Code;
                        i += 3;
                        continue;
                    }
                    i += 1;
                }

                LexicalState::CharLiteral => {
                    if char_at(&chars, i) == '\\' {
                        i += 2; // skip escaped char
                        continue;
                    }
                    if char_at(&chars, i) == '\'' {
                        self.state = LexicalState::Code;
                        i += 1;
                        continue;
                    }
                    i += 1;
                }

                LexicalState::TemplateLiteral => {
                    if char_at(&chars, i) == '\\' {
                        i += 2;
                        continue;
                    }
                    if char_at(&chars, i) == '`' {
                        self.state = LexicalState::Code;
                        i += 1;
                        continue;
                    }
                    i += 1;
                }
            }
        }

        // Reset line comment at end of line
        if self.state == LexicalState::LineComment {
            self.state = LexicalState::Code;
        }

        let is_non_code = !was_code_at_start && !entered_code;

        ScannedLine {
            line_number,
            brace_delta: delta,
            depth_after: self.depth,
            is_non_code,
            starts_in_literal: !was_code_at_start,
        }
    }

    /// Scan all lines of a source file.
    ///
    /// Convenience method that splits on `\n` and calls
    /// [`scan_line`](Self::scan_line) for each line.
    pub fn scan_all(&mut self, content: &str) -> Vec<ScannedLine> {
        content
            .lines()
            .enumerate()
            .map(|(i, line)| self.scan_line(u32::try_from(i + 1).unwrap_or(u32::MAX), line))
            .collect()
    }

    /// Current cumulative brace depth.
    pub fn current_depth(&self) -> u32 {
        self.depth
    }

    /// Check if `char_at(chars, i)` starts a Rust raw string (`r"`, `r#"`, `br"`, `br#"`).
    /// Returns the hash count if it does.
    fn try_rust_raw_string(&self, chars: &[char], i: usize) -> Option<u32> {
        let len = chars.len();
        let mut pos = i;

        // Optional 'b' prefix
        if pos < len && char_at(chars, pos) == 'b' {
            pos += 1;
        }

        // Must have 'r'
        if pos >= len || char_at(chars, pos) != 'r' {
            return None;
        }
        pos += 1;

        // Count '#' characters
        let mut hashes = 0u32;
        while pos < len && char_at(chars, pos) == '#' {
            hashes = hashes.saturating_add(1);
            pos += 1;
        }

        // Must end with '"'
        if pos < len && char_at(chars, pos) == '"' {
            // Disambiguation: r"..." is a raw string (0 hashes).
            // But we need at least 'r' followed by '"' or '#'s then '"'.
            Some(hashes)
        } else {
            None
        }
    }

    /// Heuristic to distinguish character literal `'x'` from Rust lifetime `'a`.
    ///
    /// Returns `true` if this looks like a char literal, `false` for lifetime.
    fn is_char_literal(&self, chars: &[char], i: usize) -> bool {
        let len = chars.len();
        // Need at least 3 chars: 'x' (quote, char, quote)
        if i + 2 >= len {
            return false;
        }

        let next = char_at(chars, i + 1);

        // Escaped char: '\n', '\\'
        if next == '\\' && i + 3 < len && char_at(chars, i + 3) == '\'' {
            return true;
        }
        // Simple char: 'a'
        if char_at(chars, i + 2) == '\'' {
            return true;
        }

        // Otherwise it's a lifetime or label
        false
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
pub fn find_closing_brace(scanned: &[ScannedLine], open_idx: usize) -> usize {
    let Some(open_line) = scanned.get(open_idx) else {
        return scanned.len().saturating_sub(1);
    };

    // The depth before the line: a line that opens several blocks (`fn f() { if x {`) closes
    // when the depth falls back to it, not after its innermost block.
    let opened = u32::try_from(open_line.brace_delta.max(0)).unwrap_or(0);
    let target_depth = open_line.depth_after.saturating_sub(opened);
    for (i, line) in scanned.iter().enumerate().skip(open_idx + 1) {
        if line.depth_after <= target_depth && line.brace_delta < 0 {
            return i;
        }
    }
    // Unclosed brace — return last line (graceful degradation)
    scanned.len().saturating_sub(1)
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

    #[test]
    fn a_char_literal_with_a_multi_char_escape_closes_on_its_own_quote() {
        // [MAGI S6 loop 3, B7] `'\x00'`, `'\u{1F600}'` and C's `'\012'` are one char each: read
        // as a lifetime, their closing quote paired with the next one and the `'{'` after it
        // counted as code.
        for (rules, code) in [
            (LexicalRules::rust(), r"let v = ['\x00','{'];"),
            (LexicalRules::rust(), r"let v = ['\u{1F600}','{'];"),
            (LexicalRules::c(), r"char v[] = {'\012','{'};"),
        ] {
            let r = CodeScanner::new(rules).scan_all(code);
            assert_eq!(r[0].brace_delta, 0, "{code}");
            assert_eq!(r[0].depth_after, 0, "{code}");
        }
    }

    #[test]
    fn balanced_braces_inside_a_template_expression_leave_the_depth_unchanged() {
        let mut scanner = CodeScanner::new(LexicalRules::javascript());
        let r = scanner.scan_all("const s = `x ${ {a: 1} } y`;\nfunction f() {\n}\n");
        assert_eq!(r[0].depth_after, 0);
        assert_eq!(r[1].depth_after, 1);
        assert_eq!(r[2].depth_after, 0);
    }
}
