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

use super::python_strings::PythonStrings;

/// Character at `i`, or NUL past the end of the line.
///
/// Reads never index: release builds abort on panic (`panic = "abort"`), so the scanner reads
/// through `.get()`. Every call site is guarded by the loop bounds, so the NUL fallback is
/// never observed on a path that previously indexed in range.
fn char_at(chars: &[char], i: usize) -> char {
    chars.get(i).copied().unwrap_or('\0')
}

/// Characters after the backslash of the longest escaped char literal, Rust's `'\u{10FFFF}'`:
/// `u`, the two braces and six hex digits, plus the closing quote. Underscores between the
/// digits (`'\u{1_0}'`) count against it, so an escape with six digits and an underscore reads
/// as no char literal, declared: a heuristic window, not Rust's grammar.
const MAX_CHAR_ESCAPE_CHARS: usize = 10;

/// The longest delimiter a C++ raw string allows between `"` and `(` (C++11 [lex.string]).
const MAX_CPP_RAW_DELIMITER_CHARS: usize = 16;

/// The encoding prefixes a C++ raw string may carry before its `R`, longest first.
const CPP_RAW_PREFIXES: [&str; 5] = ["u8R", "uR", "UR", "LR", "R"];

/// How many nested preprocessor conditionals (`#if` ... `#endif`) the scanner tracks.
///
/// A conditional nested deeper is not tracked: its branches' braces all count, as before the
/// scanner tracked any. It bounds the scanner's memory by a constant, whatever the input; real
/// code nests a handful of levels.
pub const MAX_CONDITIONAL_NESTING: usize = 64;

/// A preprocessor directive that opens, branches or closes a conditional.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConditionalDirective {
    /// `#if`, `#ifdef`, `#ifndef`.
    Open,
    /// `#elif`, `#elifdef`, `#elifndef` (C23), `#else`.
    Branch,
    /// `#endif`.
    Close,
}

impl ConditionalDirective {
    /// The conditional directive `line` holds, or `None` for any other line.
    ///
    /// # Arguments
    ///
    /// * `line` - A line whose first non-blank character is `#`; blanks and block comments,
    ///   which C reads as blanks, may follow the `#` (`#  ifdef X`, `# /* c */ else`).
    ///
    /// # Returns
    ///
    /// The directive named by the word after `#`, or `None` when it is no conditional
    /// (`#define`, `#include`), a block comment after the `#` does not close on the line, or the
    /// line holds no `#`. O(n) in the line's length.
    fn parse(line: &str) -> Option<Self> {
        let mut rest = line.trim_start().strip_prefix('#')?.trim_start();
        // MAGI FIX (B2a 3): `# /* c */ ifdef X` hid the word behind the comment.
        while let Some(comment) = rest.strip_prefix("/*") {
            rest = comment.split_once("*/")?.1.trim_start();
        }
        let word = rest
            .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .next()
            .unwrap_or("");
        match word {
            "if" | "ifdef" | "ifndef" => Some(Self::Open),
            "elif" | "elifdef" | "elifndef" | "else" => Some(Self::Branch),
            "endif" => Some(Self::Close),
            _ => None,
        }
    }
}

/// A preprocessor conditional open at the scanner's position.
#[derive(Debug, Clone, Copy)]
struct ConditionalFrame {
    /// The brace depth at its `#if`, where each later branch restarts.
    base: u32,
    /// The brace depth its first branch reached, recorded at its first `#elif` or `#else`;
    /// `None` while it has only one branch.
    first_branch_end: Option<u32>,
}

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
    /// Inside a Java text block (`"""`).
    TripleQuote { quote: char },
    /// Inside a Python string, f-strings and their replacement fields included; the scanner's
    /// [`PythonStrings`] holds what is open.
    PythonString,
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
    /// Java's `"""` text blocks; a backslash escapes the next character.
    TripleQuote,
    /// Python's strings: an optional prefix, one or three quotes, and the replacement fields
    /// of f-strings and t-strings (PEP 701, PEP 750), read by `PythonStrings`.
    PythonStrings,
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
    pub(crate) line_comment: &'static str,
    /// Whether `/* */` block comments exist.
    pub(crate) block_comments: bool,
    /// Whether block comments can nest (`/* /* */ */` — Rust only).
    pub(crate) nested_comments: bool,
    /// Whether the language has character literals with `'` (C, Rust, Java).
    /// `false` for Python (single-quote strings) and JS (single-quote strings).
    pub(crate) char_literals: bool,
    /// Raw string syntax variant.
    pub(crate) raw_strings: RawStringSyntax,
    /// Whether backtick template literals exist (JS/TS only).
    pub(crate) template_literals: bool,
    /// Whether `/.../` regular expression literals exist (JS/TS only).
    pub(crate) regex_literals: bool,
    /// Whether a line whose first non-blank character is `#` is a preprocessor directive
    /// (C, C++), continued while a line ends in `\`: its strings and comments are read as in
    /// code, but its braces, parentheses and brackets are no structure (`#define BEGIN {`).
    ///
    /// Conditionals (`#if`, `#ifdef`, `#ifndef`, `#elif`, `#else`, `#endif`) shape the depth:
    /// only the first branch of a conditional counts after its `#endif`. Each later branch is
    /// scanned from the depth at the `#if`, so a header it opens (`#else` / `int f(void) {`)
    /// still opens a block within it, but the depth it leaves is dropped at `#endif`. The limit:
    /// a block opened or closed only in a later branch is invisible after the conditional, and
    /// the first branch counts even when it is dead (`#if 0` ... `#else`). An unbalanced
    /// directive (a stray `#endif`, a missing one) degrades to plain counting, never a panic,
    /// as does a conditional nested past [`MAX_CONDITIONAL_NESTING`]. A directive inside a
    /// comment, a string or a `\`-continued directive is no conditional.
    ///
    /// The same languages splice a line ending in `\` onto the next one, so a `//` comment
    /// ending in `\` continues on the next line, which is then comment text too.
    // MAGI FIX (B2a): `#ifdef` / `#else` around two headers of one function counted both `{`,
    // so the function ran to the end of the file and a large file failed its batch.
    pub(crate) preprocessor_directives: bool,
    /// Whether a `'` may open a character constant of several characters (`'ab'`, C and C++),
    /// besides the one-character literal every `char_literals` language has; a C++14 digit
    /// separator (`1'000`) opens none.
    pub(crate) multichar_literals: bool,
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
            preprocessor_directives: false,
            multichar_literals: false,
        }
    }

    /// C: `//` line comments, `/* */` non-nested block comments,
    /// `'` char literals of one or several characters, `#` preprocessor directives.
    pub fn c() -> Self {
        Self {
            line_comment: "//",
            block_comments: true,
            nested_comments: false,
            char_literals: true,
            raw_strings: RawStringSyntax::None,
            template_literals: false,
            regex_literals: false,
            // MAGI FIX (S6 loop 19, B2a-2): a directive's braces moved the depth, so
            // `#define BEGIN {` shifted every later block by one level.
            preprocessor_directives: true,
            // MAGI FIX (S6 loop 19, B2a-8): `'{}'` read as code, so its braces moved the depth.
            multichar_literals: true,
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
            preprocessor_directives: false,
            multichar_literals: false,
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
            preprocessor_directives: false,
            multichar_literals: false,
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
            preprocessor_directives: false,
            multichar_literals: false,
        }
    }

    /// TypeScript: same lexical rules as JavaScript.
    pub fn typescript() -> Self {
        Self::javascript()
    }

    /// Python: `#` line comments, no block comments, no char literals, and Python's strings
    /// ([`RawStringSyntax::PythonStrings`]): prefixes, triple quotes and f-string fields.
    ///
    /// Python's string syntax is read in a second place, `LogicalLines` in
    /// `review::languages::python`, which tracks `(` and `[` as well to find where a logical
    /// line ends. Both hand every string to `PythonStrings` in
    /// `review::languages::python_strings`, so the two readers share one string grammar.
    // MAGI FIX (S6 loop 8): cross-referenced so a fix to one Python string reader reaches both.
    pub fn python() -> Self {
        Self {
            line_comment: "#",
            block_comments: false,
            nested_comments: false,
            char_literals: false,
            // MAGI FIX (PEP 701): one open string at a time closed `f"{"("}"` at its inner
            // quote, so the field's brackets and quotes were read as code.
            raw_strings: RawStringSyntax::PythonStrings,
            template_literals: false,
            regex_literals: false,
            preprocessor_directives: false,
            multichar_literals: false,
        }
    }
}

/// Result of scanning a single source line.
///
/// [`CodeScanner::scan_all`] keeps one per line, so its size bounds the scan's memory: 24 bytes,
/// about 200 MiB for a capped file of 8 MiB of newlines, as much as the line slices every
/// extractor already holds. That bound is transient and linear: an extraction keeps the records
/// of the one text it scans only while it runs and drops them on return, so they never add up
/// across files. Pinned by `a_scanned_line_takes_at_most_twenty_four_bytes`; packing
/// it further would narrow the per-line counts, which a line of the capped size can overflow.
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
    /// Net change of the parentheses and square brackets open on this line, code context only
    /// (`(` and `[` count +1, `)` and `]` count -1), so a delimiter inside a string, a character
    /// literal, a regex literal or a comment does not count: `var (` gives 1, `b = ")"` gives 0.
    pub group_delta: i32,
    /// Whether the last non-blank character the line has in code context is `;`: a comment or
    /// string after it does not count, so `int f(void) // note;` does not end a statement.
    pub ends_statement: bool,
    /// Whether the last non-blank character the line has in code context is `}`: a comment or
    /// string after it does not count, so `void f() {} // note` ends a block and
    /// `void g(const char* s = "}",` does not. A `}` that closes a template expression counts
    /// as the template text it returns to, not as a brace.
    pub ends_block: bool,
    /// `true` when the line starts inside a multi-line comment or string and never returns to
    /// structural code. A line that starts in code is `false` even when it is all comment
    /// (`// note`): keyword detection does not need to skip it, since no keyword opens with a
    /// comment. A line that starts inside a `${...}` expression of a template literal counts as
    /// starting inside the template, so it is `true` while it stays in the expression: the
    /// expression is code, but its braces are not structure and no unit starts inside it, so
    /// keyword detection skips it on purpose.
    // MAGI FIX (S6 loop 18, B2a-2): the template expression case is stated.
    pub is_non_code: bool,
    /// `true` when the line starts inside a multi-line comment or string, whether or not it
    /// returns to code: its leading whitespace is literal text, not indentation. A line that
    /// starts inside a `${...}` template expression is `true` too, as for [`Self::is_non_code`].
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
    /// The Python strings open ([`RawStringSyntax::PythonStrings`]), read while the state is
    /// `LexicalState::PythonString`.
    python: PythonStrings,
    /// For each open `${...}` template expression, innermost last, how many `{` it holds still
    /// open: its `}` with none open returns to the template text.
    templates: Vec<u32>,
    /// Whether the previous line was a preprocessor directive ending in `\`, so this line
    /// continues it ([`LexicalRules::preprocessor_directives`]).
    in_directive: bool,
    /// The preprocessor conditionals open at this point, innermost last, at most
    /// [`MAX_CONDITIONAL_NESTING`] ([`CodeScanner::apply_conditional`]).
    conditionals: Vec<ConditionalFrame>,
    /// How many conditionals are open past [`MAX_CONDITIONAL_NESTING`]: not tracked, only
    /// counted, so their `#elif`, `#else` and `#endif` leave the tracked ones alone.
    untracked_conditionals: u32,
    /// The rules' line comment token as characters, collected once ([`LexicalRules::line_comment`]).
    line_comment: Vec<char>,
    /// The characters of the line being scanned, a buffer kept between lines so that a scan of
    /// n lines allocates as for its longest line, not once per line.
    chars: Vec<char>,
    /// Whether a `/` before any code on the next line opens a regex literal: what the last code
    /// line ended with decides ([`regex_may_start`], read on that line's own characters, so no
    /// index crosses lines), as JavaScript reads a line that opens with `/` as the continuation
    /// of the line above (`a\n  / b` divides). `true` before the first code line.
    regex_at_line_start: bool,
}

impl CodeScanner {
    /// Create a new scanner with language-specific lexical rules.
    pub fn new(rules: LexicalRules) -> Self {
        // MAGI FIX (S6 loop 19, B2a-5): the comment token and the line's characters were
        // collected into a new `Vec` on every line, millions of allocations for a capped file of
        // short lines; the token is collected once and the line buffer is reused.
        Self {
            line_comment: rules.line_comment.chars().collect(),
            chars: Vec::new(),
            rules,
            state: LexicalState::Code,
            depth: 0,
            parens: Vec::new(),
            python: PythonStrings::default(),
            templates: Vec::new(),
            in_directive: false,
            conditionals: Vec::new(),
            untracked_conditionals: 0,
            regex_at_line_start: true,
        }
    }

    /// Apply a preprocessor conditional directive to the brace depth.
    ///
    /// `#if` records the depth; `#elif` and `#else` keep the depth the first branch reached and
    /// restart from the recorded one; `#endif` returns to the first branch's depth when the
    /// conditional had a later branch. A directive with no open conditional (a stray `#endif` or
    /// `#else`) changes nothing, and an `#if` past [`MAX_CONDITIONAL_NESTING`] is counted but not
    /// tracked, so its own `#endif` cannot pop a tracked one. O(1).
    fn apply_conditional(&mut self, kind: ConditionalDirective) {
        match kind {
            ConditionalDirective::Open if self.conditionals.len() < MAX_CONDITIONAL_NESTING => {
                self.conditionals.push(ConditionalFrame {
                    base: self.depth,
                    first_branch_end: None,
                });
            }
            ConditionalDirective::Open => {
                self.untracked_conditionals = self.untracked_conditionals.saturating_add(1);
            }
            ConditionalDirective::Branch if self.untracked_conditionals == 0 => {
                if let Some(frame) = self.conditionals.last_mut() {
                    frame.first_branch_end.get_or_insert(self.depth);
                    self.depth = frame.base;
                }
            }
            ConditionalDirective::Close if self.untracked_conditionals == 0 => {
                if let Some(end) = self.conditionals.pop().and_then(|f| f.first_branch_end) {
                    self.depth = end;
                }
            }
            ConditionalDirective::Branch => {}
            ConditionalDirective::Close => {
                self.untracked_conditionals = self.untracked_conditionals.saturating_sub(1);
            }
        }
    }

    /// Scan a single line, updating internal state.
    ///
    /// Must be called in order (line 1, line 2, ...) because the
    /// scanner carries context across lines (e.g., multi-line strings,
    /// block comments, open preprocessor conditionals).
    ///
    /// A conditional directive line moves no brace itself (its `brace_delta` and `min_delta`
    /// are 0) but may move `depth_after`: `#elif` and `#else` return it to the depth at `#if`,
    /// `#endif` to the depth the first branch reached
    /// (the `preprocessor_directives` rule of [`LexicalRules`]). O(n) in the line's length.
    ///
    /// With preprocessor rules, a line that ends in `\` continues its directive or its `//`
    /// comment onto the next line even when blanks follow the backslash: GCC and Clang splice
    /// such a line (with a warning), and C++23 makes it the standard reading (P2223R2, trailing
    /// whitespace after a line-splicing backslash). A compiler that does not splice it reads one
    /// line differently, never a lost one.
    pub fn scan_line(&mut self, line_number: u32, line: &str) -> ScannedLine {
        let mut chars = std::mem::take(&mut self.chars);
        chars.clear();
        chars.extend(line.chars());
        let len = chars.len();
        let mut i = 0;
        let mut delta: i32 = 0;
        let mut min_delta: i32 = 0;
        let mut group_delta: i32 = 0;
        // The last non-blank character read in code context (a string counts as its quote).
        let mut last_code: Option<char> = None;
        // Where the code character `last_code` records ends (0 when none): the word a regex or a
        // condition follows is read there, so a comment after it (`return /* c */ /x/`) does
        // not hide it.
        let mut last_code_end: usize = 0;
        // A line that starts inside a template expression starts inside the template literal.
        let was_code_at_start = self.state == LexicalState::Code && self.templates.is_empty();
        let mut entered_code = false;
        // A directive's strings and comments are lexed as in code, so a `/*` it opens still
        // spans lines, but nothing else in it is structure.
        // MAGI FIX (B2a 6): a block comment before the `#` (`/* c */ #ifdef X`) hid the
        // directive; C reads a comment as a blank before preprocessing.
        let directive_text = without_leading_block_comments(line);
        let starts_directive = self.rules.preprocessor_directives
            && !self.in_directive
            && was_code_at_start
            && directive_text.starts_with('#');
        let directive =
            starts_directive || (self.rules.preprocessor_directives && self.in_directive);
        self.in_directive = directive && line.trim_end().ends_with('\\');
        if starts_directive {
            if let Some(kind) = ConditionalDirective::parse(directive_text) {
                self.apply_conditional(kind);
            }
        }

        while i < len {
            match &self.state {
                LexicalState::Code => {
                    entered_code |= self.templates.is_empty();

                    // MAGI FIX (S6 loop 6): only `//` and `#` were matched, so any other
                    // configured token was ignored and the braces after it counted.
                    if !self.line_comment.is_empty()
                        && chars
                            .get(i..)
                            .is_some_and(|rest| rest.starts_with(&self.line_comment))
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
                        && regex_may_start(
                            &chars,
                            last_code_end,
                            last_code,
                            self.regex_at_line_start,
                        )
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

                    if self.rules.raw_strings == RawStringSyntax::PythonStrings {
                        if let Some(end) = self.python.open(&chars, i) {
                            last_code = Some(STRING_MARK);
                            if self.python.is_open() {
                                self.state = LexicalState::PythonString;
                            }
                            i = end;
                            continue;
                        }
                    }

                    // Check for triple-quote strings (Java)
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
                        // C and C++ have no lifetime: a `'` opens a constant of any width that
                        // closes on the line, unless it separates the digits of a number.
                        let literal = if self.rules.multichar_literals {
                            !is_digit_separator(&chars, i) && multichar_literal_closes(&chars, i)
                        } else {
                            self.is_char_literal(&chars, i)
                        };
                        if literal {
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

                    // MAGI FIX (S6 loop 19, B2a-3): a `b"` branch here opened a plain string at the
                    // `b`, before the `"` reached the triple-quote check, so Python's `b"""` read
                    // as an empty string and a reopened one. A byte string needs no branch: its
                    // `b` is code like any prefix letter, and its `"` (or `"""`) opens the string
                    // on the next character.

                    if directive {
                        i += 1;
                        continue;
                    }

                    let c = char_at(&chars, i);
                    // MAGI FIX (S6 loop 19, B2a-4): the condition's keyword was read right before
                    // the `(`, so a comment between them (`if /* c */ (ok)`) hid it.
                    let word_end = if last_code.is_some_and(is_js_word_char) {
                        last_code_end
                    } else {
                        0
                    };
                    if !c.is_whitespace() {
                        last_code = Some(c);
                        last_code_end = i + 1;
                    }
                    // MAGI FIX (S6 loop 6): `if (ok) /}/.test(s)` read the `/` as a division,
                    // so the regex's brace closed the enclosing block.
                    if self.rules.regex_literals {
                        if c == '(' {
                            let word = word_before(&chars, word_end);
                            self.parens
                                .push(CONDITION_KEYWORDS.iter().any(|k| word_is(word, k)));
                        } else if c == ')' && self.parens.pop() == Some(true) {
                            last_code = Some(CONDITION_END_MARK);
                        }
                    }

                    // MAGI FIX (S6 loop 17, B2-1): the extractors counted parentheses on the
                    // line text, strings and comments included; the code ones are counted here.
                    match c {
                        '(' | '[' => group_delta = group_delta.saturating_add(1),
                        ')' | ']' => group_delta = group_delta.saturating_sub(1),
                        _ => {}
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

                LexicalState::PythonString => {
                    i = self.python.read(&chars, i);
                    if !self.python.is_open() {
                        self.state = LexicalState::Code;
                    }
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
        // MAGI FIX (B2a 4): in C and C++ a backslash before the newline splices the next line
        // into the comment (translation phase 2), so `// note \` hides the next line's braces.
        // Trailing blanks after the backslash are tolerated, as for a continued directive.
        if self.state == LexicalState::PythonString {
            self.python.end_line();
            if !self.python.is_open() {
                self.state = LexicalState::Code;
            }
        }
        let spliced_comment = self.rules.preprocessor_directives
            && self.state == LexicalState::LineComment
            && line.trim_end().ends_with('\\');
        if !spliced_comment
            && matches!(
                self.state,
                LexicalState::LineComment | LexicalState::RegexLiteral { .. }
            )
        {
            self.state = LexicalState::Code;
        }

        // MAGI FIX (B2a 2, 4): each line started as if nothing preceded it, so a `/` opening a
        // line after an operand (`a\n  / b`) opened a regex and hid the rest of the line. A line
        // with no code (blank, a comment) leaves the answer of the code line above it.
        if self.rules.regex_literals && last_code.is_some() {
            self.regex_at_line_start =
                regex_may_start(&chars, last_code_end, last_code, self.regex_at_line_start);
        }
        let is_non_code = !was_code_at_start && !entered_code;
        self.chars = chars;

        ScannedLine {
            line_number,
            brace_delta: delta,
            depth_after: self.depth,
            min_delta,
            group_delta,
            ends_statement: last_code == Some(';'),
            ends_block: last_code == Some('}'),
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
    ///
    /// It continues from the scanner's current state, as [`scan_line`](Self::scan_line) does:
    /// it resets neither the depth nor the lexical context, so a scanner scans one text. Each
    /// text gets a new scanner ([`CodeScanner::new`]), as every extractor does.
    // MAGI FIX (S6 loop 18, B2a-3): the one-text contract is stated.
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

    /// Check if `char_at(chars, i)` starts a Rust raw string (`r"`, `r#"`, `br"`, `br#"`, and
    /// the `r` of `cr"`). Returns the hash count if it does.
    ///
    /// The prefix counts only at the start of a token, as [`cpp_raw_string_start`]'s does: an
    /// `r` or `b` preceded by an identifier character is the tail of a longer identifier
    /// (`iter"a\"{"`, an identifier then a plain string), except the `r` of a `cr` that itself
    /// starts a token. O(h) for the h hashes.
    fn try_rust_raw_string(&self, chars: &[char], i: usize) -> Option<u32> {
        // MAGI FIX (B2a A1): the prefix was matched inside a longer identifier, so the plain
        // string after `iter` read as raw, closed on its escaped quote, and its brace counted.
        let word_char = |k: usize| is_js_word_char(char_at(chars, k));
        if i > 0
            && word_char(i - 1)
            && !(char_at(chars, i) == 'r'
                && char_at(chars, i - 1) == 'c'
                && (i < 2 || !word_char(i - 2)))
        {
            return None;
        }
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
        // MAGI FIX (loop 26, 2): `_` was no escape-body char, though Rust allows it between the
        // digits of a unicode escape (`'\u{1_0}'`), so the quote opened no char literal.
        if next == '\\' {
            return ((i + 3)..len.min(i + 3 + MAX_CHAR_ESCAPE_CHARS))
                .map(|k| char_at(chars, k))
                .take_while(|&c| {
                    c == '\'' || c.is_ascii_alphanumeric() || matches!(c, '{' | '}' | '_')
                })
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
// MAGI FIX (S6 loop 19, B2a-4): `await /x/.test(s)` read as a division.
const REGEX_PRECEDING_KEYWORDS: [&str; 14] = [
    "return",
    "await",
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

/// Whether a `/` (not `//` nor `/*`) starts a JS regular expression literal, by what precedes
/// it on the line in code (`last_code`, the last non-blank code character, which ends at
/// `code_end`; comments between it and the `/` are passed over): nothing on the line, when
/// `at_line_start` says the last code line above allows one (it was asked the same question at
/// its end, [`CodeScanner::scan_line`]), an
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
/// O(k) for the k characters before `code_end`.
fn regex_may_start(
    chars: &[char],
    code_end: usize,
    last_code: Option<char>,
    at_line_start: bool,
) -> bool {
    let before = chars.get(..code_end).unwrap_or(&[]);
    match last_code {
        None => at_line_start,
        Some(c) if c == STRING_MARK || matches!(c, ')' | ']') => false,
        Some(c) if is_js_word_char(c) => {
            let word = word_before(chars, code_end);
            REGEX_PRECEDING_KEYWORDS.iter().any(|k| word_is(word, k))
        }
        // `a++ / 2` divides; a single `+` or `-` is an operator, so a regex may follow it.
        // MAGI FIX (B2a 3, 5): a lone sign (`+/x/` opening a line) and a binary one after a
        // postfix (`x++ + /x/`) read as `++`. The run of adjacent signs ending the code decides:
        // tokens pair from its left, so an even run ends in `++` (an operand's end) and an odd
        // one in a lone operator.
        Some(c @ ('+' | '-')) => before.iter().rev().take_while(|&&p| p == c).count() % 2 == 1,
        Some(_) => true,
    }
}

/// `line` without its leading blanks and the block comments that open it and close on it
/// (`/* c */ #ifdef X` gives `#ifdef X`), as C reads a comment before preprocessing: a blank.
/// A comment that does not close on the line leaves `line` trimmed, so it still opens with
/// `/*` and no directive is read on it. O(n) in the line's length.
fn without_leading_block_comments(line: &str) -> &str {
    let mut rest = line.trim_start();
    while let Some(comment) = rest.strip_prefix("/*") {
        match comment.split_once("*/") {
            Some((_, after)) => rest = after.trim_start(),
            None => return line.trim_start(),
        }
    }
    rest
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
    // MAGI FIX (S6 loop 18, B2a-8): each prefix was collected into a `Vec` at every position of
    // a C++ line; it is compared in place, so a position that opens no raw string allocates
    // nothing.
    let prefix_chars = CPP_RAW_PREFIXES.iter().find_map(|p| {
        let n = p.chars().count();
        (rest.iter().take(n).copied().eq(p.chars()) && rest.get(n) == Some(&'"')).then_some(n)
    })?;
    let open = prefix_chars + 1;
    let delimiter: Vec<char> = rest
        .get(open..)?
        .iter()
        .take(MAX_CPP_RAW_DELIMITER_CHARS + 1)
        .take_while(|&&c| c != '(')
        .copied()
        .collect();
    // MAGI FIX (S6 loop 19, B2a-7): a `"` was rejected too, though C++ allows it in a delimiter
    // ([lex.string] excludes only blanks, `(`, `)`, `\` and control characters), so such a raw
    // string read as plain strings and its braces moved the depth.
    let valid = delimiter.len() <= MAX_CPP_RAW_DELIMITER_CHARS
        && rest.get(open + delimiter.len()) == Some(&'(')
        && !delimiter
            .iter()
            .any(|&c| c.is_whitespace() || c.is_control() || matches!(c, ')' | '\\'));
    valid.then(|| {
        let skip = open + delimiter.len() + 1;
        (delimiter, skip)
    })
}

/// The most characters between the quotes of a C or C++ multicharacter constant that
/// [`multichar_literal_closes`] looks at: four characters each written as a four-character
/// escape (`'\x41\x42\x43\x44'`), the widest an `int` holds. A starting point, not a
/// measurement; a longer one reads as code, as before this rule.
const MAX_MULTICHAR_LITERAL_CHARS: usize = 16;

/// Whether the `'` at `i` opens a C or C++ character constant closed on the line within
/// [`MAX_MULTICHAR_LITERAL_CHARS`] characters (`'{}'`, `'ab'`), a backslash escaping the next
/// character. O(1): the look-ahead is bounded.
fn multichar_literal_closes(chars: &[char], i: usize) -> bool {
    let mut k = i + 1;
    let end = i
        .saturating_add(2)
        .saturating_add(MAX_MULTICHAR_LITERAL_CHARS);
    while k < chars.len().min(end) {
        match char_at(chars, k) {
            '\\' => k += 2,
            '\'' => return k > i + 1,
            _ => k += 1,
        }
    }
    false
}

/// Whether the `'` at `i` is a C++14 digit separator (`1'000`, `0x1'F`): the word right before
/// it is a number, whose first character is a digit. A prefix (`u8'a'`, `L'a'`) starts with a
/// letter, so its quote opens a literal. O(k) for the k characters before `i`.
fn is_digit_separator(chars: &[char], i: usize) -> bool {
    chars
        .get(..i)
        .unwrap_or(&[])
        .iter()
        .rev()
        .take_while(|&&c| c.is_alphanumeric() || c == '_' || c == '.')
        .last()
        .is_some_and(char::is_ascii_digit)
}

/// Whether `c` can be part of a JS identifier or keyword (`$` included).
fn is_js_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '$'
}

/// The identifier or keyword that ends right before `i`, blanks skipped (`if (` gives `if`);
/// empty when a non-word character precedes. A sub-slice of `chars`, so nothing is allocated.
/// O(k) for the k characters before `i`.
fn word_before(chars: &[char], i: usize) -> &[char] {
    let before = chars.get(..i).unwrap_or(&[]);
    let blanks = before
        .iter()
        .rev()
        .take_while(|c| c.is_whitespace())
        .count();
    let end = before.len() - blanks;
    let word = before
        .get(..end)
        .unwrap_or(&[])
        .iter()
        .rev()
        .take_while(|&&c| is_js_word_char(c))
        .count();
    before.get(end - word..end).unwrap_or(&[])
}

/// Whether the characters `word` spell `keyword`. O(k) for a keyword of k characters.
fn word_is(word: &[char], keyword: &str) -> bool {
    word.iter().copied().eq(keyword.chars())
}

/// Find the closing brace for a construct that opens at `open_idx`.
///
/// Uses pre-scanned brace depths instead of naive counting.
/// The closing brace is the first line after `open_idx` where the
/// depth, at any point within the line, returns to the level it was
/// at before the opening brace: `} else {` closes the block it ends
/// although it opens another one. On the opening line, the braces opened after its lowest point
/// are the construct's, so a line that closes first and then opens (`}) {`) opens one.
///
/// # Arguments
///
/// * `scanned` - Pre-scanned line results from [`CodeScanner::scan_all`].
/// * `open_idx` - Index into `scanned` of the line containing the opening brace.
///
/// # Returns
///
/// Index into `scanned` of the line containing the matching closing brace.
/// If no closing brace is found, returns the last line index (graceful degradation). When
/// `open_idx` is outside `scanned`, it returns the last line index too, which for an empty
/// `scanned` is 0, an index that does not exist: callers pass the index of a line they read
/// from `scanned`, and read the result with `get`, never by indexing.
// MAGI FIX (S6 loop 18, B2a-6): the result for an empty scan is stated.
///
/// # Precondition
///
/// `scanned[open_idx]` opens a block (`brace_delta > 0`), as every caller guarantees: the
/// extractors pass the line [`ScannedLine::brace_delta`] says opens one. For a line that opens
/// none, the target depth is the depth after it, so the result is the first later line that
/// closes below that depth, or the last line; never a panic. It is not asserted: a
/// `debug_assert!` would hold the precondition only in debug builds.
pub fn find_closing_brace(scanned: &[ScannedLine], open_idx: usize) -> usize {
    find_closing_brace_within(scanned, open_idx, scanned.len(), || true)
        .unwrap_or_else(|| scanned.len().saturating_sub(1))
}

/// The line closing the construct that opens at `open_idx`, as [`find_closing_brace`] finds it,
/// searched only before index `end` and while `visit` allows each line looked at.
///
/// # Arguments
///
/// * `scanned` - Pre-scanned line results from [`CodeScanner::scan_all`].
/// * `open_idx` - Index into `scanned` of the line containing the opening brace.
/// * `end` - Index of the first line not searched (exclusive); past `scanned`, its end.
/// * `visit` - Called once before each line looked at; `false` stops the search there.
///
/// # Returns
///
/// Index into `scanned` of the closing line, or `None` when `open_idx` is outside `scanned`, no
/// line before `end` closes the construct, or `visit` stopped the search first. O(k) for the k
/// lines searched.
// MAGI FIX (S6 loop 19, B2b-2): a Go type group's member searched to the end of the file,
// outside the pass budget; the bound and the budget are now the caller's.
pub(crate) fn find_closing_brace_within(
    scanned: &[ScannedLine],
    open_idx: usize,
    end: usize,
    mut visit: impl FnMut() -> bool,
) -> Option<usize> {
    let open_line = scanned.get(open_idx)?;

    // The depth before the line: a line that opens several blocks (`fn f() { if x {`) closes
    // when the depth falls back to it, not after its innermost block.
    // MAGI FIX (B2b1 7): a line that closes before it opens (`}) {`, the end of a default
    // argument's lambda then the function's brace) opens what it holds after its lowest point.
    let opened = u32::try_from(
        open_line
            .brace_delta
            .saturating_sub(open_line.min_delta)
            .max(0),
    )
    .unwrap_or(0);
    let target_depth = open_line.depth_after.saturating_sub(opened);
    // MAGI FIX (S6 loop 4): a line that closes and reopens (`} else {`) nets zero; the lowest
    // depth within the line, not the depth after it, says whether it closed the block.
    let mut start_depth = open_line.depth_after;
    for (i, line) in scanned.iter().enumerate().take(end).skip(open_idx + 1) {
        if !visit() {
            return None;
        }
        let lowest = i64::from(start_depth) + i64::from(line.min_delta);
        if line.min_delta < 0 && lowest <= i64::from(target_depth) {
            return Some(i);
        }
        start_depth = line.depth_after;
    }
    // Unclosed brace: the caller degrades (`find_closing_brace` returns the last line).
    None
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

    #[test]
    fn an_identifier_ending_in_r_before_a_string_opens_no_raw_string() {
        // [MAGI B2a A1] the raw-string prefix was matched inside a longer identifier, so in a
        // macro's tokens (`iter"a\"{"`, an identifier then a string before edition 2021) the
        // string read as raw, closed on its escaped quote, and its brace counted.
        let mut scanner = CodeScanner::new(LexicalRules::rust());
        let results = scanner.scan_all("m!(iter\"a\\\"{\");\nfn g() {\n}\n");
        assert_eq!(results[0].brace_delta, 0, "the brace is string text");
        assert!(!results[1].starts_in_literal, "the string closed on line 1");
        assert_eq!(results[1].brace_delta, 1);
        assert_eq!(find_closing_brace(&results, 1), 2, "g keeps its bounds");
        // `abr"` is the identifier `abr` too; the real prefixes still open raw strings.
        let mut s = CodeScanner::new(LexicalRules::rust());
        assert_eq!(s.scan_line(1, r#"m!(abr"a\"{");"#).brace_delta, 0);
        for line in [
            r#"let s = r"{\";"#,
            r#"let s = br"{\";"#,
            r#"let s = cr"{\";"#,
            r##"let s = (r#"{"#);"##,
        ] {
            let mut s = CodeScanner::new(LexicalRules::rust());
            assert_eq!(s.scan_line(1, line).brace_delta, 0, "{line}");
            assert_eq!(s.current_depth(), 0, "{line}");
        }
        // A raw string ends on its first quote, so the brace after it is code: read as a plain
        // string, the escaped quote would keep the string open and hide that brace.
        for prefix in ["r", "br", "cr"] {
            let line = format!(r#"let s = {prefix}"\"; {{"#);
            let mut s = CodeScanner::new(LexicalRules::rust());
            assert_eq!(s.scan_line(1, &line).brace_delta, 1, "{line}");
        }
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
    fn a_slash_opening_a_line_divides_when_the_line_above_ends_in_an_operand() {
        // [MAGI B2a 2, 4] every line started with nothing before it, so a `/` opening a line
        // always opened a regex, even when it continued a division from the line above
        // (`a\n  / b`), as JavaScript reads it: the rest of the line, its braces included, was
        // hidden. What the line above ends with decides, as within a line.
        let delta_of_last = |code: &str| {
            let results = CodeScanner::new(LexicalRules::javascript()).scan_all(code);
            results.last().map(|s| s.brace_delta)
        };
        let cases = [
            ("x = total\n  / count; }", Some(-1)),
            ("x = f(a)\n  / 2; }", Some(-1)),
            ("x = a\n// note\n\n  / b; }", Some(-1)),
            ("x = a /* c */\n  / b; }", Some(-1)),
            // A regex still opens after a statement, an operator, an opening bracket or a block.
            ("g();\n/}/.test(s);", Some(0)),
            ("x =\n  /}/.test(s);", Some(0)),
            ("f(\n  /}/);", Some(0)),
            ("function f() {}\n/{/.test(s);", Some(0)),
            ("if (ok)\n  /}/.test(s);", Some(0)),
            ("return\n/}/.test(s);", Some(0)),
        ];
        for (code, delta) in cases {
            assert_eq!(delta_of_last(code), delta, "{code:?}");
        }
        let code = "function f() {
  if (a
      / b > 1) {
    g();
  }
}
function h() {
}
";
        let results = CodeScanner::new(LexicalRules::javascript()).scan_all(code);
        assert_eq!(
            results[2].brace_delta, 1,
            "the division's line opens the if block"
        );
        assert_eq!(
            find_closing_brace(&results, 0),
            5,
            "f ends on its own brace"
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

    #[test]
    fn a_line_closing_more_groups_than_it_opens_reports_a_negative_net() {
        // [MAGI S6 loop 18, B2a-1] `group_delta` is a signed net per line, as `brace_delta` is:
        // the closing line of a wrapped parameter list (`) {`) reports -1, which the extractor's
        // brace search adds up to see the list close (`open_brace_line`).
        let code = "fn f(
    a: u8,
) {
    g(a)(b
)(";
        let results = CodeScanner::new(LexicalRules::rust()).scan_all(code);
        let nets: Vec<i32> = results.iter().map(|r| r.group_delta).collect();
        assert_eq!(nets, vec![1, 0, -1, 1, 0]);
    }

    #[test]
    fn a_closing_brace_search_outside_the_scan_returns_its_last_index() {
        // [MAGI S6 loop 18, B2a-6] the documented result for an index outside the scan: the last
        // line, which an empty scan does not have.
        let results = CodeScanner::new(LexicalRules::rust()).scan_all("fn f() {\n}\n");
        assert_eq!(find_closing_brace(&results, 5), 1);
        assert_eq!(find_closing_brace(&[], 0), 0);
    }

    #[test]
    fn a_line_whose_code_ends_in_a_closing_brace_ends_a_block() {
        // [MAGI S6 loop 18, B2b-2/B2b-8] the extractors told a one-line construct by its text
        // ending in `}` or holding a `{`; the scanner records whether the line's code ends in `}`,
        // which no comment or string after it, or brace inside it, moves.
        let code = "void f() {} // note\nvoid g(const char* s = \"}\",\nint h() { return 0; }\nint k() {\n}";
        let results = CodeScanner::new(LexicalRules::cpp()).scan_all(code);
        let ends: Vec<bool> = results.iter().map(|r| r.ends_block).collect();
        assert_eq!(ends, vec![true, false, true, false, true]);
    }

    #[test]
    fn a_c_preprocessor_directive_moves_no_brace_depth() {
        // [MAGI S6 loop 19, B2a-2] a directive's braces counted as code, so `#define BEGIN {`
        // left the depth one level up for the rest of the file and a later `}` closed the
        // wrong block.
        let code = "#define BEGIN {\n#define END }\n#define BLOCK(x) \\\n    { x; } \\\n    {\nint f(void) {\n}\n";
        for rules in [LexicalRules::c(), LexicalRules::cpp()] {
            let results = CodeScanner::new(rules).scan_all(code);
            for (idx, line) in results.iter().take(5).enumerate() {
                assert_eq!(
                    (line.brace_delta, line.depth_after, line.ends_block),
                    (0, 0, false),
                    "directive line {}",
                    idx + 1
                );
            }
            assert_eq!(results[5].brace_delta, 1, "f opens its block");
            assert_eq!(
                find_closing_brace(&results, 5),
                6,
                "f ends on its own brace"
            );
        }
        // Not a directive: a `#` past the start of a line, and Rust's attributes.
        let mut c = CodeScanner::new(LexicalRules::c());
        assert_eq!(c.scan_line(1, "int a = b # c; {").brace_delta, 1);
        let mut rust = CodeScanner::new(LexicalRules::rust());
        assert_eq!(rust.scan_line(1, "#[cfg(x)] fn f() {").brace_delta, 1);
    }

    #[test]
    fn a_python_byte_triple_quoted_string_is_one_string() {
        // [MAGI S6 loop 19, B2a-3] the `b"` byte-string branch fired before the triple quote, so
        // `b"""` read as an empty string and a reopened one, and its text was scanned as code.
        for prefix in ["b", "rb", "br", "B", "Rb"] {
            let code = format!("x = {prefix}\"\"\"a \" {{\n\"\"\"\ny = 1\n");
            let results = CodeScanner::new(LexicalRules::python()).scan_all(&code);
            assert_eq!(results[0].brace_delta, 0, "{prefix}: the brace is text");
            assert!(
                results[1].starts_in_literal,
                "{prefix}: line 2 is string text"
            );
            assert!(!results[2].starts_in_literal, "{prefix}: the string closed");
        }
        let mut rust = CodeScanner::new(LexicalRules::rust());
        assert_eq!(rust.scan_line(1, r#"let s = b"{"; {"#).brace_delta, 1);
    }

    #[test]
    fn a_regex_after_await_or_a_comment_after_a_keyword_is_not_counted() {
        // [MAGI S6 loop 19, B2a-4] `await` was no keyword a regex may follow, and a comment
        // between a keyword and the `/` (or the `(` of a condition) hid the keyword, so the
        // regex read as a division and its braces moved the depth.
        let code = "async function f(s) {\n  await /{/.test(s);\n  return /* c */ /}/.test(s);\n}\nif /* c */ (ok) /}/.test(s);\nfunction g() {\n}\n";
        let results = CodeScanner::new(LexicalRules::javascript()).scan_all(code);
        for idx in [1, 2, 4] {
            assert_eq!(results[idx].brace_delta, 0, "line {}", idx + 1);
        }
        assert_eq!(
            find_closing_brace(&results, 0),
            3,
            "f ends on its own brace"
        );
        assert_eq!(find_closing_brace(&results, 5), 6, "g is still found");
        let mut js = CodeScanner::new(LexicalRules::javascript());
        assert_eq!(
            js.scan_line(1, "x = a /* c */ / b; {").brace_delta,
            1,
            "an operand before the comment still divides"
        );
    }

    #[test]
    fn a_scanned_line_takes_at_most_twenty_four_bytes() {
        // [MAGI S6 loop 19, B2a-5] `scan_all` keeps one record per line, so a capped file of
        // 8 MiB of newlines holds about 8.4 M of them: 24 bytes each is about 200 MiB, and a
        // field added carelessly grows that by a third. Pinned at the measured size.
        assert!(
            std::mem::size_of::<ScannedLine>() <= 24,
            "{} bytes",
            std::mem::size_of::<ScannedLine>()
        );
    }

    #[test]
    fn a_cpp_raw_string_delimiter_may_hold_a_quote() {
        // [MAGI S6 loop 19, B2a-7] a delimiter holding `"` is legal C++ (only blanks, `(`, `)`,
        // `\` and control characters are excluded), but it was rejected, so the raw string read
        // as plain strings and code and its braces moved the depth.
        let code = "auto s = R\"d\"(a { b)d\"\";\nint x;\n";
        let results = CodeScanner::new(LexicalRules::cpp()).scan_all(code);
        assert_eq!(results[0].brace_delta, 0);
        assert!(!results[1].starts_in_literal, "the raw string closed");
    }

    #[test]
    fn a_c_multicharacter_constant_is_a_literal() {
        // [MAGI S6 loop 19, B2a-8] C and C++ allow multicharacter constants (`'{}'`); only a
        // one-character literal was recognized, so their braces moved the depth.
        for rules in [LexicalRules::c(), LexicalRules::cpp()] {
            let mut s = CodeScanner::new(rules);
            assert_eq!(s.scan_line(1, "int c = '{{'; int d = 'a}';").brace_delta, 0);
            assert_eq!(s.current_depth(), 0);
        }
        // A C++14 digit separator opens no literal.
        let mut cpp = CodeScanner::new(LexicalRules::cpp());
        assert_eq!(cpp.scan_line(1, "int n = 1'000'000; {").brace_delta, 1);
        assert_eq!(cpp.scan_line(2, "int m = 0x1'F; }").brace_delta, -1);
        // Rust and Java keep their one-character rule: a lifetime opens no literal.
        let mut rust = CodeScanner::new(LexicalRules::rust());
        assert_eq!(rust.scan_line(1, "fn f<'a>(x: &'a str) {").brace_delta, 1);
    }

    #[test]
    fn an_ifdef_else_pair_of_function_headers_closes_at_its_own_brace() {
        // [B2a] each branch of `#ifdef`/`#else` opened the block of `f`, so the depth stayed one
        // level up after its `}`: `f` ran to the end of the file and absorbed `g`.
        let code = "#ifdef X\nint f(int a) {\n#else\nint f(void) {\n#endif\n    return 0;\n}\nint g(void) { return 1; }\n";
        for rules in [LexicalRules::c(), LexicalRules::cpp()] {
            let results = CodeScanner::new(rules).scan_all(code);
            assert_eq!(find_closing_brace(&results, 1), 6, "first header");
            assert_eq!(find_closing_brace(&results, 3), 6, "second header");
            // The second header opens its block from the top level, as the first does.
            let depths: Vec<u32> = results.iter().take(5).map(|l| l.depth_after).collect();
            assert_eq!(depths, vec![0, 1, 0, 1, 1], "depth through the conditional");
            assert_eq!(results[6].depth_after, 0, "back at the top level");
            assert_eq!(results[7].depth_after, 0, "g opens and closes on its line");
        }
    }

    #[test]
    fn nested_conditionals_count_only_each_first_branch() {
        // [B2a] an inner conditional inside the first branch of an outer one: one `{` survives.
        let code = "#ifdef A\n#ifdef B\nvoid f(int a, int b) {\n#else\nvoid f(int a) {\n#endif\n#else\nvoid f(void) {\n#endif\n    return;\n}\nvoid g(void) {}\n";
        let results = CodeScanner::new(LexicalRules::c()).scan_all(code);
        for open in [2, 4, 7] {
            assert_eq!(find_closing_brace(&results, open), 10, "header {open}");
        }
        assert_eq!(results[10].depth_after, 0);
        assert_eq!(results[11].depth_after, 0);
    }

    #[test]
    fn an_elif_chain_counts_only_its_first_branch() {
        // [B2a] `#if`/`#elif`/`#elif`/`#else`: four headers, one block.
        let code = "#if A\nint f(int a) {\n#elif B\nint f(long b) {\n#elif defined(C)\nint f(char c) {\n#else\nint f(void) {\n#endif\n    return 0;\n}\nint g(void) {\n}\n";
        let results = CodeScanner::new(LexicalRules::cpp()).scan_all(code);
        for open in [1, 3, 5, 7] {
            assert_eq!(find_closing_brace(&results, open), 10, "header {open}");
        }
        assert_eq!(results[10].depth_after, 0);
        assert_eq!(find_closing_brace(&results, 11), 12, "g is its own block");
    }

    #[test]
    fn a_conditional_with_blanks_after_the_hash_is_still_a_conditional() {
        // [B2a] `#  ifdef` and `# else` are the same directives as `#ifdef` and `#else`.
        let code = "#  ifdef X\nint f(int a) {\n  # else\nint f(void) {\n#\tendif\n}\n";
        let results = CodeScanner::new(LexicalRules::c()).scan_all(code);
        assert_eq!(find_closing_brace(&results, 1), 5);
        assert_eq!(results[5].depth_after, 0);
    }

    #[test]
    fn a_conditional_with_a_comment_after_the_hash_is_still_a_conditional() {
        // [MAGI B2a 3] a block comment between `#` and the directive's word (`# /* c */ ifdef X`,
        // valid C: a comment is a blank) hid the word, so both branches' braces counted.
        let code = "# /* c */ ifdef X\nint f(int a) {\n#/**/else\nint f(void) {\n# /* a */ /* b */ endif\n}\nint g(void) { return 1; }\n";
        let results = CodeScanner::new(LexicalRules::c()).scan_all(code);
        assert_eq!(find_closing_brace(&results, 1), 5);
        assert_eq!(results[5].depth_after, 0);
        assert_eq!(results[6].depth_after, 0);
        assert_eq!(
            ConditionalDirective::parse("# /* never closed ifdef X"),
            None,
            "an unclosed comment hides the rest of the line"
        );
    }

    #[test]
    fn a_stray_endif_or_else_leaves_the_depth_as_counted() {
        // [B2a] an unbalanced directive degrades to plain brace counting, never a panic.
        // Passes before the fix by design: it pins the degraded behaviour.
        let code = "#endif\n#else\nint f(void) {\n#endif\n    return 0;\n}\n";
        let results = CodeScanner::new(LexicalRules::c()).scan_all(code);
        assert_eq!(find_closing_brace(&results, 2), 5);
        assert_eq!(results[5].depth_after, 0);
    }

    #[test]
    fn the_first_branch_decides_the_depth_after_an_unequal_conditional() {
        // [B2a] branches that leave different depths: the first one's survives `#endif`, and a
        // brace opened only in a later branch is dropped there (the declared limit).
        let first_opens = "#if A\nint f(void) {\n#else\n#endif\n    return 0;\n}\n";
        let results = CodeScanner::new(LexicalRules::c()).scan_all(first_opens);
        assert_eq!(results[3].depth_after, 1, "after #endif");
        assert_eq!(find_closing_brace(&results, 1), 5);
        assert_eq!(results[5].depth_after, 0);
        let later_opens = "#if A\n#else\nint f(void) {\n#endif\nint g(void) {\n}\n";
        let results = CodeScanner::new(LexicalRules::c()).scan_all(later_opens);
        assert_eq!(
            results[3].depth_after, 0,
            "the later branch's brace is dropped"
        );
        assert_eq!(find_closing_brace(&results, 4), 5, "g is its own block");
    }

    #[test]
    fn a_missing_endif_still_restores_the_depth_at_its_else() {
        // [B2a] a conditional the file never closes still restores at `#else`; its frame only
        // stays on the stack.
        let code = "#ifdef X\nint f(int a) {\n#else\nint f(void) {\n    return 0;\n}\nint g(void) { return 1; }\n";
        let results = CodeScanner::new(LexicalRules::c()).scan_all(code);
        assert_eq!(find_closing_brace(&results, 1), 5);
        assert_eq!(results[5].depth_after, 0);
        assert_eq!(results[6].depth_after, 0);
    }

    #[test]
    fn a_brace_inside_an_if_zero_block_without_else_still_counts() {
        // [B2a] a conditional without `#else` has one branch, which counts as code: a balanced
        // block inside `#if 0` is still a block. Passes before the fix by design.
        let code = "#if 0\nint old(void) {\n}\n#endif\nint g(void) {\n}\n";
        let results = CodeScanner::new(LexicalRules::c()).scan_all(code);
        assert_eq!(find_closing_brace(&results, 1), 2);
        assert_eq!(find_closing_brace(&results, 4), 5);
        assert_eq!(results[5].depth_after, 0);
    }

    #[test]
    fn a_conditional_inside_a_comment_or_a_continued_directive_is_no_conditional() {
        // [B2a] `#else` inside a block comment or a `\`-continued `#define` is text: it restores
        // nothing, so the `{` of `h` still counts on top of the one of `f`. Passes before the
        // fix by design.
        let code = "#ifdef X\nint f(int a) {\n/*\n#else\n*/\n#define M \\\n#else\nint h(void) {\n}\n#endif\n}\n";
        let results = CodeScanner::new(LexicalRules::c()).scan_all(code);
        assert_eq!(find_closing_brace(&results, 7), 8, "h is nested in f");
        assert_eq!(
            find_closing_brace(&results, 1),
            10,
            "f closes at the last brace"
        );
        assert_eq!(results[10].depth_after, 0);
    }

    #[test]
    fn conditionals_nested_past_the_cap_degrade_to_plain_counting() {
        // [B2a] past `MAX_CONDITIONAL_NESTING` a conditional is not tracked, and its `#endif`
        // must not pop a tracked one: the outer `#ifdef`/`#else` still restores.
        let deep = MAX_CONDITIONAL_NESTING + 3;
        let mut code = String::from("#ifdef X\nint f(int a) {\n");
        for _ in 0..deep {
            code.push_str("#if 1\n");
        }
        for _ in 0..deep {
            code.push_str("#endif\n");
        }
        code.push_str("#else\nint f(void) {\n#endif\n}\nint g(void) {\n}\n");
        let results = CodeScanner::new(LexicalRules::c()).scan_all(&code);
        let close = results.len() - 3;
        assert_eq!(find_closing_brace(&results, 1), close);
        assert_eq!(results[close].depth_after, 0);
        assert_eq!(find_closing_brace(&results, close + 1), close + 2);
    }

    #[test]
    fn a_hash_else_line_is_no_conditional_without_preprocessor_rules() {
        // [B2a] only languages with preprocessor directives track conditionals. Passes before
        // the fix by design.
        let results = CodeScanner::new(LexicalRules::rust()).scan_all("fn f() {\n#else\n}\n");
        assert_eq!(results[1].depth_after, 1);
        assert_eq!(results[2].depth_after, 0);
    }

    #[test]
    fn a_c_line_comment_ending_in_a_backslash_continues_onto_the_next_line() {
        // [MAGI B2a 4] a `//` comment ended with its line, but in C and C++ a backslash before
        // the newline splices the next line into it (translation phase 2), so the brace there is
        // comment text, not a block.
        for rules in [LexicalRules::c(), LexicalRules::cpp()] {
            let results = CodeScanner::new(rules).scan_all("// note \\\nint f() {\nint g() {\n}\n");
            assert_eq!(results[1].brace_delta, 0, "the spliced line is comment");
            assert!(results[1].is_non_code, "the spliced line holds no code");
            assert_eq!(results[2].brace_delta, 1, "the splice ends with its line");
            assert_eq!(
                results[3].depth_after, 0,
                "the depth closes on the last line"
            );
        }
        // Rust has no line splicing: the next line is code.
        let results = CodeScanner::new(LexicalRules::rust()).scan_all("// note \\\nfn f() {\n}\n");
        assert_eq!(results[1].brace_delta, 1);
    }

    #[test]
    fn a_regex_after_a_unary_or_binary_plus_or_minus_is_not_counted() {
        // [MAGI B2a 3, 5] a `+` or `-` was read as the tail of `++`/`--` whenever it was the
        // only operator character before the `/` on the line, or when the one before it,
        // across a blank, was the same sign (`x++ + /}/`): the regex read as a division and
        // its brace moved the depth.
        let cases = [
            ("+/}/.test(s);", 0),
            ("-/{/.exec(s);", 0),
            ("y = x++ + /}/.test(s);", 0),
            ("y = x-- - /{/.test(s);", 0),
            ("y = a + +/}/.test(s);", 0),
            // A postfix `++` or `--` still ends an operand: the `/` after it divides.
            ("y = a++ / b; {", 1),
            ("y = a-- / b; }", -1),
            ("y = a+++ /}/.test(s);", 0),
        ];
        for (line, delta) in cases {
            let mut scanner = CodeScanner::new(LexicalRules::javascript());
            assert_eq!(scanner.scan_line(1, line).brace_delta, delta, "{line}");
        }
    }

    #[test]
    fn a_conditional_after_a_closed_block_comment_is_still_a_conditional() {
        // [MAGI B2a 6] a block comment before the `#` (`/* c */ #ifdef X`, valid C: a comment
        // is a blank before preprocessing) hid the directive, so both branches' braces counted
        // and the function after them was absorbed.
        let code = "/* c */ #ifdef X\nint f(int a) {\n/* a */ /* b */ #else\nint f(void) {\n#endif\n}\nint g(void) {\n}\n";
        let results = CodeScanner::new(LexicalRules::c()).scan_all(code);
        assert_eq!(find_closing_brace(&results, 1), 5);
        assert_eq!(results[5].depth_after, 0);
        assert_eq!(find_closing_brace(&results, 6), 7);
        // A comment that does not close on the line hides the `#`: the line is no directive.
        let open = "/* c #else */\nint f(void) {\n}\n";
        let results = CodeScanner::new(LexicalRules::c()).scan_all(open);
        assert_eq!(results[1].brace_delta, 1);
    }

    /// Asserts that `head`, one Python line, moves no brace and leaves no literal open: the
    /// `y = {` after it starts in code and opens one block.
    fn assert_python_line_closes_its_strings(head: &str) {
        let code = format!("{head}\ny = {{\n");
        let results = CodeScanner::new(LexicalRules::python()).scan_all(&code);
        assert_eq!(
            results[0].brace_delta, 0,
            "{head}: its braces are string text"
        );
        assert!(!results[1].starts_in_literal, "{head}: every string closed");
        assert_eq!(results[1].brace_delta, 1, "{head}");
    }

    #[test]
    fn a_string_reusing_its_quote_inside_an_fstring_field_keeps_the_fstring_open() {
        // [MAGI residual, PEP 701] since Python 3.12 a string inside a replacement field may
        // reuse the outer quote; it closed the f-string, so the field's `{` counted as code and
        // the `'''` of `d["'''"]` opened a triple-quoted string that ran to the end of the file.
        for head in [
            r#"x = f"{"{"}""#,
            r#"x = f"{d["'''"]}""#,
            r#"x = f"{f"{"{"}"}""#,
            r#"x = t"{"{"}""#,
        ] {
            assert_python_line_closes_its_strings(head);
        }
    }

    #[test]
    fn python_fstring_specs_escapes_and_prefixes_leave_no_literal_open() {
        // Declared green before the PEP 701 fix, pinned so the f-string reader keeps them: a
        // quote in a format spec is text, `{{` is no field, `\N{...}` is an escape, a raw
        // string's backslash keeps its quote from closing it, and a plain or byte string has
        // no field.
        for head in [
            r#"x = f"{d["key"]}""#,
            r#"x = f"{v:'^10}""#,
            r#"x = f"{v:{w}>{n}}""#,
            r#"x = f"{{ }}""#,
            r#"x = f"\N{EM DASH}""#,
            r#"x = rf"\N{v}""#,
            r#"x = fr"{v}\"""#,
            r#"x = "{" + "}""#,
            r#"x = b"{" + rb"}""#,
        ] {
            assert_python_line_closes_its_strings(head);
        }
        let code = "x = f\"\"\"{v} {{\n\"\"\"\ny = {\n";
        let results = CodeScanner::new(LexicalRules::python()).scan_all(code);
        assert!(results[1].starts_in_literal, "line 2 is string text");
        assert!(!results[2].starts_in_literal);
        assert_eq!(results[2].brace_delta, 1);
    }

    #[test]
    fn a_unicode_escape_with_an_underscore_is_a_char_literal() {
        // [MAGI loop 26, 2] the escape body rejected `_`, which Rust allows (`'\u{1_0}'`), so the
        // quote opened no char literal and a later `'{'` counted as a brace: the function ran on.
        let mut scanner = CodeScanner::new(LexicalRules::rust());
        let results = scanner.scan_all("    f('\\u{1_0}','{');");
        assert_eq!(results[0].brace_delta, 0);
    }

    #[test]
    fn a_brace_closed_inside_a_template_expression_ends_no_block() {
        // [MAGI loop 27, B2a] the `}` closing a brace nested in a `${...}` expression left
        // `ends_block` true when it ended the line, although no brace of the expression is
        // structure; the `}` that ends the expression already counted as template text.
        let mut scanner = CodeScanner::new(LexicalRules::javascript());
        let r = scanner.scan_all("const s = `${ f({a: 1}\n) }`;\nfunction g() {\n}\n");
        assert_eq!(
            r.iter().map(|l| l.ends_block).collect::<Vec<_>>(),
            vec![false, false, false, true]
        );
        let mut scanner = CodeScanner::new(LexicalRules::javascript());
        let r = scanner.scan_all("const s = `${ {a: 1}\n}`;\n");
        assert!(!r[0].ends_block);
    }
}
