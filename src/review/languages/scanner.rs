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

/// Characters after the backslash of the longest escaped char literal, Rust's `'\u{10FFFF}'`:
/// `u`, the two braces and six hex digits, plus the closing quote.
const MAX_CHAR_ESCAPE_CHARS: usize = 10;

/// The longest delimiter a C++ raw string allows between `"` and `(` (C++11 [lex.string]).
const MAX_CPP_RAW_DELIMITER_CHARS: usize = 16;

/// The encoding prefixes a C++ raw string may carry before its `R`, longest first.
const CPP_RAW_PREFIXES: [&str; 5] = ["u8R", "uR", "UR", "LR", "R"];

/// What [`ScannedLine::ends_statement`] records for a literal read in code context: any
/// character but `;` would do, since a literal is an operand.
const STRING_MARK: char = '"';

/// What `last_code` records for the `)` that closes the condition of an `if`, `while`, `for`
/// or `with`: a statement follows it, not an operator, so a `/` there opens a regex literal.
/// Any character that is neither an operand nor `;` would do; a control character never
/// appears in code.
const CONDITION_END_MARK: char = '\u{1}';

/// Keywords whose parenthesized condition is followed by a statement (`if (x) /re/.test(s)`).
const CONDITION_KEYWORDS: [&str; 4] = ["if", "while", "for", "with"];

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
    /// Inside a C++ raw string: `R"delim(...)delim"`; `delimiter` may be empty.
    CppRawString { delimiter: Vec<char> },
    /// Inside a Python triple-quoted string (`"""` or `'''`).
    TripleQuote { quote: char },
    /// Inside a character literal (`'...'` in C/Rust/Java).
    CharLiteral,
    /// Inside the text of a JS/TS template literal (`` `...` ``). A `${` returns to code for the
    /// expression, lexed like any code (its strings, regexes, comments and nested templates
    /// included), until the `}` that balances it brings the text back; the scanner's stack of
    /// open expressions records where each one ends. The braces of an expression are not
    /// counted: the template is one operand.
    TemplateLiteral,
    /// Inside a JS/TS regular expression literal (`/.../`); `in_class` is `true` inside a
    /// character class (`[...]`), where `/` does not close it.
    RegexLiteral { in_class: bool },
}

/// Raw string syntax variants across languages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawStringSyntax {
    /// No raw string support.
    None,
    /// Rust-style: `r#"..."#` with variable hash count.
    RustHashes,
    /// Python-style `"""..."""` or `'''...'''`, and Java's `"""` text blocks; a backslash
    /// escapes the next character.
    TripleQuote,
    /// Go-style: backtick raw strings (`` `...` ``).
    GoBacktick,
    /// C++11-style: `R"delim(...)delim"`, with an optional `u8`, `u`, `U` or `L` prefix; no
    /// escape is read inside, and only `)delim"` closes it.
    CppDelimited,
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
    /// Whether `/.../` regular expression literals exist (JS/TS only).
    pub regex_literals: bool,
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
            regex_literals: false,
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
            regex_literals: false,
        }
    }

    /// C++: same as C, plus C++11 raw strings (`R"(...)"`).
    pub fn cpp() -> Self {
        Self {
            // MAGI FIX (S6 loop 6): a raw string read as ordinary strings, so the braces and
            // the unbalanced quotes of `R"({"a": 1} ")"` moved the depth.
            raw_strings: RawStringSyntax::CppDelimited,
            ..Self::c()
        }
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
            regex_literals: false,
        }
    }

    /// Java: `//` + `/* */`, char literals, `"""` text blocks.
    pub fn java() -> Self {
        Self {
            line_comment: "//",
            block_comments: true,
            nested_comments: false,
            char_literals: true,
            // MAGI FIX (S6 loop 5): a text block read as three plain quotes, so a `"` inside it
            // returned to code and its braces moved the depth.
            raw_strings: RawStringSyntax::TripleQuote,
            template_literals: false,
            regex_literals: false,
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
            regex_literals: true,
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
            regex_literals: false,
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
    /// The lowest running net brace count reached within the line, from 0 at its start (so
    /// never positive): `} else {` gives -1 although its `brace_delta` is 0.
    pub min_delta: i32,
    /// Whether the last non-blank character the line has in code context is `;`: a comment or
    /// string after it does not count, so `int f(void) // note;` does not end a statement.
    pub ends_statement: bool,
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
    /// For each open `(` of a regex-literal language, whether it opens the condition of a
    /// [`CONDITION_KEYWORDS`] keyword. It lives across lines, as a condition may span them.
    parens: Vec<bool>,
    /// For each open `${...}` template expression, innermost last, how many `{` it holds still
    /// open: its `}` with none open returns to the template text.
    templates: Vec<u32>,
}

impl CodeScanner {
    /// Create a new scanner with language-specific lexical rules.
    pub fn new(rules: LexicalRules) -> Self {
        Self {
            rules,
            state: LexicalState::Code,
            depth: 0,
            parens: Vec::new(),
            templates: Vec::new(),
        }
    }

    /// Scan a single line, updating internal state.
    ///
    /// Must be called in order (line 1, line 2, ...) because the
    /// scanner carries context across lines (e.g., multi-line strings,
    /// block comments).
    pub fn scan_line(&mut self, line_number: u32, line: &str) -> ScannedLine {
        let chars: Vec<char> = line.chars().collect();
        let line_comment: Vec<char> = self.rules.line_comment.chars().collect();
        let len = chars.len();
        let mut i = 0;
        let mut delta: i32 = 0;
        let mut min_delta: i32 = 0;
        // The last non-blank character read in code context (a string counts as its quote).
        let mut last_code: Option<char> = None;
        // A line that starts inside a template expression starts inside the template literal.
        let was_code_at_start = self.state == LexicalState::Code && self.templates.is_empty();
        let mut entered_code = false;

        while i < len {
            match &self.state {
                LexicalState::Code => {
                    entered_code |= self.templates.is_empty();

                    // MAGI FIX (S6 loop 6): only `//` and `#` were matched, so any other
                    // configured token was ignored and the braces after it counted.
                    if !line_comment.is_empty()
                        && chars
                            .get(i..)
                            .is_some_and(|rest| rest.starts_with(&line_comment))
                    {
                        self.state = LexicalState::LineComment;
                        break; // rest of line is comment
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

                    // MAGI FIX (S6 loop 4): a regex literal's braces (`/}/`) are not structure.
                    if self.rules.regex_literals
                        && char_at(&chars, i) == '/'
                        && regex_may_start(&chars, i, last_code)
                    {
                        last_code = Some(STRING_MARK);
                        self.state = LexicalState::RegexLiteral { in_class: false };
                        i += 1;
                        continue;
                    }

                    // Check for Rust raw string: r#"..."# or br#"..."#
                    if self.rules.raw_strings == RawStringSyntax::RustHashes {
                        if let Some(hashes) = self.try_rust_raw_string(&chars, i) {
                            last_code = Some(STRING_MARK);
                            self.state = LexicalState::RawString { hashes };
                            // Skip past r[b]###"
                            let mut skip = if char_at(&chars, i) == 'b' { 2 } else { 1 }; // b?r
                            skip += hashes as usize; // #'s
                            skip += 1; // opening "
                            i += skip;
                            continue;
                        }
                    }

                    if self.rules.raw_strings == RawStringSyntax::CppDelimited {
                        if let Some((delimiter, skip)) = cpp_raw_string_start(&chars, i) {
                            last_code = Some(STRING_MARK);
                            self.state = LexicalState::CppRawString { delimiter };
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
                        last_code = Some(STRING_MARK);
                        self.state = LexicalState::TripleQuote { quote: q };
                        i += 3;
                        continue;
                    }

                    // Check for Go backtick raw string
                    if self.rules.raw_strings == RawStringSyntax::GoBacktick
                        && char_at(&chars, i) == '`'
                    {
                        last_code = Some(STRING_MARK);
                        self.state = LexicalState::StringLiteral {
                            quote: '`',
                            escapable: false,
                        };
                        i += 1;
                        continue;
                    }

                    // Check for template literal (JS/TS)
                    if self.rules.template_literals && char_at(&chars, i) == '`' {
                        last_code = Some(STRING_MARK);
                        self.state = LexicalState::TemplateLiteral;
                        i += 1;
                        continue;
                    }

                    // Check for character literal (Rust/C/Java)
                    if self.rules.char_literals && char_at(&chars, i) == '\'' {
                        if self.is_char_literal(&chars, i) {
                            last_code = Some(STRING_MARK);
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
                        last_code = Some(STRING_MARK);
                        self.state = LexicalState::StringLiteral {
                            quote: '"',
                            escapable: true,
                        };
                        i += 1;
                        continue;
                    }

                    // Check for single-quote string (non-char-literal languages like JS/Python)
                    if !self.rules.char_literals && char_at(&chars, i) == '\'' {
                        last_code = Some(STRING_MARK);
                        self.state = LexicalState::StringLiteral {
                            quote: '\'',
                            escapable: true,
                        };
                        i += 1;
                        continue;
                    }

                    // Check for byte string: b"..."
                    if char_at(&chars, i) == 'b' && i + 1 < len && char_at(&chars, i + 1) == '"' {
                        last_code = Some(STRING_MARK);
                        self.state = LexicalState::StringLiteral {
                            quote: '"',
                            escapable: true,
                        };
                        i += 2;
                        continue;
                    }

                    let c = char_at(&chars, i);
                    if !c.is_whitespace() {
                        last_code = Some(c);
                    }
                    // MAGI FIX (S6 loop 6): `if (ok) /}/.test(s)` read the `/` as a division,
                    // so the regex's brace closed the enclosing block.
                    if self.rules.regex_literals {
                        if c == '(' {
                            let word = word_before(&chars, i);
                            self.parens
                                .push(CONDITION_KEYWORDS.contains(&word.as_str()));
                        } else if c == ')' && self.parens.pop() == Some(true) {
                            last_code = Some(CONDITION_END_MARK);
                        }
                    }

                    // Count braces; those of a template expression are not structure.
                    let open_in_template = self.templates.last().copied();
                    if c == '{' {
                        if let Some(open) = self.templates.last_mut() {
                            *open = open.saturating_add(1);
                        } else {
                            delta += 1;
                            self.depth = self.depth.saturating_add(1);
                        }
                    } else if c == '}' {
                        match open_in_template {
                            Some(0) => {
                                self.templates.pop();
                                last_code = Some(STRING_MARK);
                                self.state = LexicalState::TemplateLiteral;
                            }
                            Some(_) => {
                                if let Some(open) = self.templates.last_mut() {
                                    *open = open.saturating_sub(1);
                                }
                            }
                            None => {
                                delta -= 1;
                                min_delta = min_delta.min(delta);
                                self.depth = self.depth.saturating_sub(1);
                            }
                        }
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

                LexicalState::CppRawString { delimiter } => {
                    let end = i + 1 + delimiter.len();
                    if char_at(&chars, i) == ')'
                        && chars.get(i + 1..end) == Some(delimiter.as_slice())
                        && char_at(&chars, end) == '"'
                    {
                        self.state = LexicalState::Code;
                        i = end + 1;
                        continue;
                    }
                    i += 1;
                }

                LexicalState::TripleQuote { quote } => {
                    let q = *quote;
                    // MAGI FIX (S6 loop 5): `\"""` closed the string on its escaped quote. A
                    // backslash escapes the next character in a raw string too, as far as where
                    // the string ends is concerned.
                    if char_at(&chars, i) == '\\' {
                        i += 2;
                        continue;
                    }
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

                LexicalState::RegexLiteral { in_class } => {
                    let in_class = *in_class;
                    match char_at(&chars, i) {
                        '\\' => {
                            i += 2;
                            continue;
                        }
                        '[' => self.state = LexicalState::RegexLiteral { in_class: true },
                        ']' if in_class => {
                            self.state = LexicalState::RegexLiteral { in_class: false };
                        }
                        '/' if !in_class => self.state = LexicalState::Code,
                        _ => {}
                    }
                    i += 1;
                }

                LexicalState::TemplateLiteral => {
                    if char_at(&chars, i) == '\\' {
                        i += 2;
                        continue;
                    }
                    // MAGI FIX (S6 loop 6): the expression was read as template text, so a
                    // backtick in one of its strings or regexes closed the template.
                    if char_at(&chars, i) == '$' && char_at(&chars, i + 1) == '{' {
                        self.templates.push(0);
                        // An expression starts like an operand: a `/` there opens a regex.
                        last_code = Some('{');
                        self.state = LexicalState::Code;
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

        // Reset line comment at end of line; a regex literal cannot span lines either, so an
        // unclosed one (a `/` misread as one) ends with its line.
        if matches!(
            self.state,
            LexicalState::LineComment | LexicalState::RegexLiteral { .. }
        ) {
            self.state = LexicalState::Code;
        }

        let is_non_code = !was_code_at_start && !entered_code;

        ScannedLine {
            line_number,
            brace_delta: delta,
            depth_after: self.depth,
            min_delta,
            ends_statement: last_code == Some(';'),
            is_non_code,
            starts_in_literal: !was_code_at_start,
        }
    }

    /// Scan all lines of a source file.
    ///
    /// Convenience method that splits `content` with [`str::lines`] and calls
    /// [`scan_line`](Self::scan_line) for each line: a trailing `\r` is dropped from each line
    /// and content ending in a newline has no empty last line, the same splitting every
    /// extractor applies before it maps scanned indices onto its own lines.
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

        // Escaped char: '\n', '\\', '\'', and the multi-char escapes '\x00', '\012', '\u{1F600}'.
        // MAGI FIX (S6 loop 3): the closing quote may sit up to MAX_CHAR_ESCAPE_CHARS after the
        // backslash, with only escape-body chars between; a lifetime never follows a backslash.
        if next == '\\' {
            return ((i + 3)..len.min(i + 3 + MAX_CHAR_ESCAPE_CHARS))
                .map(|k| char_at(chars, k))
                .take_while(|&c| c == '\'' || c.is_ascii_alphanumeric() || c == '{' || c == '}')
                .any(|c| c == '\'');
        }
        // Simple char: 'a'
        if char_at(chars, i + 2) == '\'' {
            return true;
        }

        // Otherwise it's a lifetime or label
        false
    }
}

/// Keywords after which a `/` starts a regular expression, not a division (`return /x/`).
const REGEX_PRECEDING_KEYWORDS: [&str; 13] = [
    "return",
    "typeof",
    "instanceof",
    "in",
    "of",
    "new",
    "delete",
    "void",
    "throw",
    "case",
    "do",
    "else",
    "yield",
];

/// Whether the `/` at `i` (not `//` nor `/*`) starts a JS regular expression literal, by what
/// precedes it on the line (`last_code`, the last non-blank code character): nothing, an
/// operator or an opening bracket (`= /x/`, `(/x/`, `, /x/`), or a keyword of
/// [`REGEX_PRECEDING_KEYWORDS`], or the `)` closing an `if`/`while`/`for`/`with` condition
/// ([`CONDITION_END_MARK`]), or a `}`. After an operand (an identifier, a number, a literal,
/// any other `)`, `]`, `++`, `--`) it is a division.
///
/// A heuristic, declared: a `}` reads as the end of a block, so `/` after it opens a regex
/// (`} /x/.test(s)`); dividing an object literal (`{a: 1} / 2`) is the case it misreads, and it
/// yields `NaN`, never meaningful code. A `)` that closes a call or a group, and `]`, read as
/// an operand's end. A condition keyword is recognized only on the line of its `(`. An unclosed
/// `/` misread as a regex ends with its line, so a misreading never spreads past it.
/// O(k) for the k characters before `i`.
fn regex_may_start(chars: &[char], i: usize, last_code: Option<char>) -> bool {
    let before = chars.get(..i).unwrap_or(&[]);
    match last_code {
        None => true,
        Some(c) if c == STRING_MARK || matches!(c, ')' | ']') => false,
        Some(c) if is_js_word_char(c) => {
            REGEX_PRECEDING_KEYWORDS.contains(&word_before(chars, i).as_str())
        }
        // `a++ / 2` divides; a single `+` or `-` is an operator, so a regex may follow it.
        Some(c @ ('+' | '-')) => !before
            .iter()
            .rev()
            .filter(|p| !p.is_whitespace())
            .take(2)
            .all(|&p| p == c),
        Some(_) => true,
    }
}

/// Whether a C++ raw string (`R"delim(`, with an optional encoding prefix) starts at `i`.
///
/// # Returns
///
/// The delimiter and the number of characters up to and including its `(`, or `None` when no
/// raw string starts there: the prefix is the tail of a longer identifier (`FOOR"x"`), or no
/// valid delimiter of at most [`MAX_CPP_RAW_DELIMITER_CHARS`] characters reaches a `(` on the
/// line. O(1): the prefix and the delimiter are bounded.
fn cpp_raw_string_start(chars: &[char], i: usize) -> Option<(Vec<char>, usize)> {
    if i > 0 && is_js_word_char(char_at(chars, i - 1)) {
        return None;
    }
    let rest = chars.get(i..)?;
    let prefix = CPP_RAW_PREFIXES.iter().find(|p| {
        let p: Vec<char> = p.chars().collect();
        rest.starts_with(&p) && rest.get(p.len()) == Some(&'"')
    })?;
    let open = prefix.chars().count() + 1;
    let delimiter: Vec<char> = rest
        .get(open..)?
        .iter()
        .take(MAX_CPP_RAW_DELIMITER_CHARS + 1)
        .take_while(|&&c| c != '(')
        .copied()
        .collect();
    let valid = delimiter.len() <= MAX_CPP_RAW_DELIMITER_CHARS
        && rest.get(open + delimiter.len()) == Some(&'(')
        && !delimiter
            .iter()
            .any(|&c| c.is_whitespace() || matches!(c, ')' | '\\' | '"'));
    valid.then(|| {
        let skip = open + delimiter.len() + 1;
        (delimiter, skip)
    })
}

/// Whether `c` can be part of a JS identifier or keyword (`$` included).
fn is_js_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '$'
}

/// The identifier or keyword that ends right before `i`, blanks skipped (`if (` gives `if`);
/// empty when a non-word character precedes. O(k) for the k characters before `i`.
fn word_before(chars: &[char], i: usize) -> String {
    let before = chars.get(..i).unwrap_or(&[]);
    let mut word: Vec<char> = before
        .iter()
        .rev()
        .skip_while(|c| c.is_whitespace())
        .take_while(|&&c| is_js_word_char(c))
        .copied()
        .collect();
    word.reverse();
    word.into_iter().collect()
}

/// Find the closing brace for a construct that opens at `open_idx`.
///
/// Uses pre-scanned brace depths instead of naive counting.
/// The closing brace is the first line after `open_idx` where the
/// depth, at any point within the line, returns to the level it was
/// at before the opening brace: `} else {` closes the block it ends
/// although it opens another one.
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
///
/// # Precondition
///
/// `scanned[open_idx]` opens a block (`brace_delta > 0`), as every caller guarantees: the
/// extractors pass the line [`ScannedLine::brace_delta`] says opens one. For a line that opens
/// none, the target depth is the depth after it, so the result is the first later line that
/// closes below that depth, or the last line; never a panic. It is not asserted: a
/// `debug_assert!` would hold the precondition only in debug builds.
pub fn find_closing_brace(scanned: &[ScannedLine], open_idx: usize) -> usize {
    let Some(open_line) = scanned.get(open_idx) else {
        return scanned.len().saturating_sub(1);
    };

    // The depth before the line: a line that opens several blocks (`fn f() { if x {`) closes
    // when the depth falls back to it, not after its innermost block.
    let opened = u32::try_from(open_line.brace_delta.max(0)).unwrap_or(0);
    let target_depth = open_line.depth_after.saturating_sub(opened);
    // MAGI FIX (S6 loop 4): a line that closes and reopens (`} else {`) nets zero; the lowest
    // depth within the line, not the depth after it, says whether it closed the block.
    let mut start_depth = open_line.depth_after;
    for (i, line) in scanned.iter().enumerate().skip(open_idx + 1) {
        let lowest = i64::from(start_depth) + i64::from(line.min_delta);
        if line.min_delta < 0 && lowest <= i64::from(target_depth) {
            return i;
        }
        start_depth = line.depth_after;
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
    fn a_block_closed_by_a_line_that_reopens_ends_on_that_line() {
        // [MAGI S6 loop 4, B2] `} else {` closes the `if` block but nets zero, so the search
        // passed it and ended on the `else` block's brace.
        let mut scanner = CodeScanner::new(LexicalRules::rust());
        let code = "fn f() {
    if a {
        x();
    } else {
        y();
    }
}";
        let scanned = scanner.scan_all(code);
        assert_eq!(find_closing_brace(&scanned, 1), 3, "the `if` block");
        assert_eq!(find_closing_brace(&scanned, 0), 6, "the function");
    }

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
    fn braces_inside_a_javascript_regex_literal_are_not_counted() {
        // [MAGI S6 loop 4, B2] `/}/` lowered the depth and `/\{/` raised it, so a function
        // holding one ended early or absorbed the code after it.
        let cases = [
            ("const r = /}/;", 0),
            (r"const r = /\{/g;", 0),
            ("const r = /[/}]/;", 0),
            ("if (/[}]/.test(s)) {", 1),
            ("return /{/.test(s);", 0),
            ("f(a, /}/);", 0),
            ("x = a / b / c; {", 1),
            ("x = (a) / 2; }", -1),
            ("y = a++ / 2; {", 1),
        ];
        for (line, delta) in cases {
            let mut scanner = CodeScanner::new(LexicalRules::javascript());
            assert_eq!(scanner.scan_line(1, line).brace_delta, delta, "{line}");
        }
        let mut rust = CodeScanner::new(LexicalRules::rust());
        assert_eq!(
            rust.scan_line(1, "let q = a /{").brace_delta,
            1,
            "not JS: no regex"
        );
    }

    #[test]
    fn a_regex_after_a_control_condition_or_a_block_close_is_not_counted() {
        // [MAGI S6 loop 6, B2a-b2/c1] `)` and `}` always read as the end of an operand, so a
        // regex opening a statement after `if (...)` or after a block counted its braces.
        let cases = [
            ("if (ok) /}/.test(s);", 0),
            ("if (a) /{/.test(s); {", 1),
            ("while (x) /}/.test(s);", 0),
            ("for (;;) /{/.exec(s);", 0),
            ("function f() {} /{/.test(s);", 0),
            ("if (f(a)) /}/.test(s);", 0),
            // Divisions keep dividing: a call, a group or an index is an operand.
            ("x = f(a) / 2; }", -1),
            ("x = a[0] / b[1]; }", -1),
            ("if (a / b) {", 1),
        ];
        for (line, delta) in cases {
            let mut scanner = CodeScanner::new(LexicalRules::javascript());
            assert_eq!(scanner.scan_line(1, line).brace_delta, delta, "{line}");
        }
        let code = "function f() {
  if (a &&
      b) /}/.test(s);
  g();
}
function h() {
}
";
        let results = CodeScanner::new(LexicalRules::javascript()).scan_all(code);
        assert_eq!(results[2].brace_delta, 0, "a condition spanning lines");
        assert_eq!(
            find_closing_brace(&results, 0),
            4,
            "f ends on its own brace"
        );
        assert_eq!(
            find_closing_brace(&results, 5),
            6,
            "h is still found after f"
        );
    }

    #[test]
    fn a_cpp_raw_string_keeps_its_quotes_and_braces_as_text() {
        // [MAGI S6 loop 6, B2a-b1/c2] C++ had no raw-string state: `R"({"a": 1})"` read as
        // ordinary strings, so its braces and its unbalanced quotes moved the depth.
        let code = r#"void f() {
    auto j = R"({"a": 1} " })";
    auto k = u8R"x(say ")" } )x";
    auto m = R"(
{ " {
)";
    g();
}
void h() {
}
"#;
        let results = CodeScanner::new(LexicalRules::cpp()).scan_all(code);
        for idx in [1, 2, 3, 4, 5] {
            assert_eq!(results[idx].brace_delta, 0, "line {}", idx + 1);
        }
        assert!(results[4].starts_in_literal, "line 5 is raw-string text");
        assert!(
            !results[6].starts_in_literal,
            "the raw string closed on line 6"
        );
        assert_eq!(
            find_closing_brace(&results, 0),
            7,
            "f ends on its own brace"
        );
        assert_eq!(
            find_closing_brace(&results, 8),
            9,
            "h is still found after f"
        );
        let mut c = CodeScanner::new(LexicalRules::cpp());
        assert_eq!(
            c.scan_line(1, r#"int FOOR = 1; s = "{"; {"#).brace_delta,
            1,
            "an identifier ending in R before a string is no raw string"
        );
    }

    #[test]
    fn a_backtick_inside_a_template_expression_string_or_regex_is_text() {
        // [MAGI S6 loop 6, B2a-m1] the template state ignored the lexical context of `${...}`,
        // so a backtick in a string or a regex there closed the template and the scanner lost
        // its place for the rest of the file.
        for template in [
            r#"const a = `x ${"`"} y`;"#,
            "const b = `x ${/`/.source} y`;",
            "const c = `x ${'`' + `n${'}'}`} y`;",
            "const d = `x ${ {k: '`'}.k } y`;",
        ] {
            let code = format!("{template}\nfunction f() {{\n}}\n");
            let r = CodeScanner::new(LexicalRules::javascript()).scan_all(&code);
            assert_eq!(
                r.iter().map(|l| l.depth_after).collect::<Vec<_>>(),
                vec![0, 1, 0],
                "{template}"
            );
            assert!(!r[1].starts_in_literal, "{template}");
        }
        let code = "const e = `x ${\n  \"`\"\n} y`;\nfunction f() {\n}\n";
        let r = CodeScanner::new(LexicalRules::javascript()).scan_all(code);
        assert_eq!(
            r.iter().map(|l| l.depth_after).collect::<Vec<_>>(),
            vec![0, 0, 0, 1, 0],
            "an expression spanning lines"
        );
    }

    #[test]
    fn any_configured_line_comment_token_hides_its_braces() {
        // [MAGI S6 loop 6, B2a-b3] only `//` and `#` were recognized, so another configured
        // token was ignored and the braces after it counted.
        let rules = LexicalRules {
            line_comment: "--",
            ..LexicalRules::c()
        };
        let mut scanner = CodeScanner::new(rules);
        assert_eq!(scanner.scan_line(1, "x = 1; -- { note").brace_delta, 0);
        assert_eq!(scanner.scan_line(2, "y = a - b; {").brace_delta, 1);
    }

    #[test]
    fn a_template_nested_in_a_template_expression_leaves_the_depth_unchanged() {
        // [MAGI S6 loop 4, B2] a nested template inside `${...}` (lexed as code since loop 6)
        // must not drift the depth of the code after it.
        let mut scanner = CodeScanner::new(LexicalRules::javascript());
        let r = scanner.scan_all("const s = `a${ `b${ {k: 1}.k }` }c`;\nfunction f() {\n}\n");
        assert_eq!(
            r.iter().map(|l| l.depth_after).collect::<Vec<_>>(),
            vec![0, 1, 0]
        );
    }

    #[test]
    fn balanced_braces_inside_a_template_expression_leave_the_depth_unchanged() {
        let mut scanner = CodeScanner::new(LexicalRules::javascript());
        let r = scanner.scan_all("const s = `x ${ {a: 1} } y`;\nfunction f() {\n}\n");
        assert_eq!(r[0].depth_after, 0);
        assert_eq!(r[1].depth_after, 1);
        assert_eq!(r[2].depth_after, 0);
    }

    #[test]
    fn an_escaped_quote_does_not_close_a_triple_quoted_string() {
        // [MAGI S6 loop 5, B2-c4] the triple-quote state skipped no escape, so `\"""` closed the
        // string on the escaped quote and the rest of it was scanned as code.
        let mut scanner = CodeScanner::new(LexicalRules::python());
        let results = scanner.scan_all("s = \"\"\"a \\\"\"\" b {\"\"\"\nx = {\n");
        assert_eq!(results[0].brace_delta, 0, "the brace is string text");
        assert!(!results[1].starts_in_literal, "the string closed on line 1");
        assert_eq!(results[1].brace_delta, 1);
    }

    #[test]
    fn a_java_text_block_keeps_its_quotes_and_braces_as_text() {
        // [MAGI S6 loop 5, B2-m5] Java had no text-block state: `"""` read as three ordinary
        // quotes, so a `"` inside the block returned the scanner to code and the braces after it
        // moved the depth.
        let code = "class A {\n    String s = \"\"\"\n        say \" { hi \\\"\"\" }\n        \"\"\";\n    void f() {\n        x();\n    }\n}\n";
        let results = CodeScanner::new(LexicalRules::java()).scan_all(code);
        assert_eq!(results[2].brace_delta, 0, "line 3 is text-block text");
        assert_eq!(
            results[3].depth_after, 1,
            "only the class is open after the block"
        );
        assert_eq!(
            find_closing_brace(&results, 0),
            7,
            "the class closes on its own brace"
        );
    }
}
