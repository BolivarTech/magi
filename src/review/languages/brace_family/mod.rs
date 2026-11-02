// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! Brace-delimited language extractor (Rust, C, C++, Go, Java, JS, TS).
//!
//! Uses [`CodeScanner`] for lexical-aware
//! brace counting and keyword detection to extract complete semantic
//! units (functions, structs, classes, declaration groups).
//!
//! # Declared limits
//!
//! Heuristics kept on purpose. Each keeps every line in exactly one unit (RV03): a miss splits
//! a construct from part of its lines, which are still reviewed, never dropped. Each is pinned
//! by the test named:
//!
//! - The forward searches of a pass share one budget linear in the lines; once it is spent a
//!   search finds no brace and the construct ends on its own line
//!   (`the_brace_searches_of_a_whole_file_spend_a_budget_linear_in_its_lines`,
//!   `the_walk_up_from_a_wrapped_attribute_tail_spends_the_pass_budget`).
//! - A `.h` file is read with the C++ rules, so a C identifier spelled like a C++ keyword opens
//!   a one-line construct (`a_c_header_reads_the_same_with_the_cpp_rules`,
//!   `a_c_identifier_spelled_like_a_cpp_keyword_loses_no_line_in_a_header`).
//! - A unit's content joins its lines with LF, whatever the file's line ends
//!   (`a_crlf_file_gives_the_units_of_its_lf_twin_with_lf_line_ends`).
//! - A C `typedef struct` or `typedef enum` is a declaration group, one unit with its
//!   `} Name;` line, not a struct (`loop_28_shapes_read_as_one_unit_each`).
//! - A comment after a return type's brace (`function f(): { // shape`) hides the type, and the
//!   brace reads as the body's (`a_comment_after_a_return_type_brace_hides_the_type_declared`).
//! - A C++ braced member initializer wrapped over lines (`Foo::Foo() : arr{`) splits the
//!   constructor from its body
//!   (`a_braced_member_initializer_wrapped_over_lines_splits_the_constructor_declared`).

use std::borrow::Cow;
use std::cell::Cell;

use super::scanner::{
    find_closing_brace, find_closing_brace_within, CodeScanner, LexicalRules, ScannedLine,
};
use super::{
    gap_units, split_into_members, ChangedLines, LanguageExtractor, Member, SemanticUnit, UnitKind,
};
use crate::review::context::FileContext;

mod detect;
mod lexical;
mod search;

use lexical::*;
use search::*;

/// Configuration for a brace-delimited language extractor.
///
/// Pre-configured constructors exist for 7 languages: Rust, C, C++,
/// Go, Java, JavaScript, TypeScript.
// MAGI FIX (S6 loop 11): `Debug`, so a failing review can print the extractor it chose.
#[derive(Clone, Debug)]
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
    /// Whether a signature line (`int main(void) {`, `void run() {`) starts a function named by
    /// the identifier before its `(`: C, C++ and Java have no function keyword, and neither has
    /// a JavaScript or TypeScript class method.
    signature_functions: bool,
    /// Whether any code line ending in `;` outside every construct is a declaration: C and C++
    /// globals (`int counter = 0;`, `static int state;`) and prototypes have no declaration
    /// keyword.
    semicolon_declarations: bool,
    /// Whether a C++ `template <...>` header may open a construct's line
    /// ([`strip_template_header`]) or stand on its own line above it (then it is one of the
    /// `attribute_prefixes`): C++ only.
    template_headers: bool,
    /// The words that may stand right before a `(` without naming a function in this language
    /// ([`Self::signature_name`]): [`NOT_FUNCTION_NAMES`], or for Java, C and C++ the words that
    /// language reserves ([`JAVA_NOT_FUNCTION_NAMES`], [`C_NOT_FUNCTION_NAMES`],
    /// [`CPP_NOT_FUNCTION_NAMES`]), since a word reserved only in another language (`delete` in
    /// Java or C, `function` in C++) is a legal name there.
    // MAGI FIX (S6 loop 18, B2b-5): one list served every language.
    not_function_names: &'static [&'static str],
    /// Whether a method directly in a class body may be named by a word of
    /// `not_function_names` (`delete() {`, `catch(e) {`): JavaScript and TypeScript allow any
    /// reserved word as a method name, and a class body holds no statement
    /// ([`BraceFamilyExtractor::try_detect_constructs`]).
    reserved_word_methods: bool,
    /// Whether a member may be named by a computed key (`[Symbol.iterator]() {`) or a private
    /// name (`#secret() {`), named as written ([`private_or_computed_name`]): JavaScript and
    /// TypeScript, where a line opening with `#` is no preprocessor directive.
    computed_member_names: bool,
    /// Whether a function keyword glued to its parameter list declares a function with no name
    /// (`export default function() {`): JavaScript and TypeScript, where an anonymous function
    /// declaration is the default export. In Rust or Go a line opening with `fn(` or `func(` is
    /// a type or a function literal, never a declaration.
    // MAGI FIX (S6 loop 18, B2b-1): `export default function() {` was a declaration group.
    anonymous_functions: bool,
    /// Whether the language overloads operators, so `operator` is a keyword that names one
    /// (`bool operator <(`, `operator()`) on a signature line ([`operator_overload_name`]): C++.
    operator_overloads: bool,
    /// Whether a member may be optional, a `?` glued between its name and its parameter list
    /// (`foo?(x: number): void;`), named without the `?` ([`optional_member_name`]): TypeScript.
    optional_members: bool,
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
        let constructs = self.detect_constructs(&lines, &scanned, false);
        let ranges: Vec<(u32, u32)> = constructs
            .iter()
            .map(|c| (c.start_line, c.end_line))
            .collect();
        // MAGI FIX (S6 loop 3): a whole file hands out its lines by range, without a set of all.
        // The `u32::MAX` fallback is out of reach: a reviewed file is at most
        // `MAX_REVIEW_FILE_BYTES` (8 MiB), so at most 8 388 608 lines, far below `u32::MAX`; it
        // only keeps the conversion total, without a panic. Detection itself does not rely on
        // it: it aborts on a line number that does not fit (`try_detect_constructs`).
        // MAGI FIX (S6 loop 14): why the fallback is safe, written where it is used.
        let total = u32::try_from(lines.len()).unwrap_or(u32::MAX);
        let (shares, orphans) = changed_lines.assign(&ranges, total);
        // MAGI FIX (S6 loop 15): a `zip` with fewer shares than constructs dropped the last
        // constructs' changed lines from every unit; a mismatch, out of reach while `assign`
        // keeps its contract, now reviews the changed lines as gaps, none lost.
        let (pairs, ranges, orphans) = match construct_shares(&constructs, shares) {
            Some(pairs) => (pairs, ranges, orphans),
            None => (Vec::new(), Vec::new(), changed_lines.assign(&[], total).1),
        };

        // Filter to constructs that overlap with changed lines
        let mut units: Vec<SemanticUnit> = Vec::new();
        for (construct, changed_in_unit) in pairs {
            if !changed_in_unit.is_empty() {
                let start_idx = construct.start_line.saturating_sub(1) as usize;
                let end_idx = construct.end_line as usize;
                // MAGI FIX (S6 loop 16): declared, not changed. The unit's lines are joined by
                // '\n', so a CRLF file's unit carries LF line ends: `split_into_members`,
                // `gap_units` and the Python extractor build and expect content that way, and
                // preserving '\r' here alone would break that shared contract. Line numbers and
                // every line's text are unchanged; only the terminators differ from the file's
                // bytes (a file sent whole, unsplit, keeps its own). Pinned by
                // `a_crlf_file_gives_the_units_of_its_lf_twin_with_lf_line_ends`.
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

    /// Splits an `impl`, class, trait, interface, namespace or Rust module at its members: the
    /// constructs between the block's opening brace and its matching closing brace, found as at
    /// the top level (indented functions included). O(n) in the block's lines.
    ///
    /// A heuristic, declared: the opening brace is the first code line of the unit, from the
    /// block's header on (the first code line a block keyword opens, so a decorator or annotation
    /// wrapped over lines above it is passed over), whose braces open more than they close. A
    /// header line before it that leaves a brace open (a const
    /// generic argument wrapped over lines, `impl Foo<{` then `N` then `}> {`) is taken for the
    /// block, so the members are searched in the wrong span and the block may not split, or split
    /// at fewer members. Nothing is lost either way: `split_into_members` covers every line of
    /// the unit with exactly one piece, and an unsplit unit is batched whole.
    ///
    /// A declaration group splits at its declarations instead
    /// (`split_declaration_group`). Only those kinds split, declared: an enum or a
    /// struct is reviewed whole, as is a Rust `macro_rules!` (a function-like unit whose token
    /// tree is opaque: no item is read inside it, `MACRO_RULES_KEYWORD`), and a construct no
    /// keyword table names (a JavaScript IIFE, a TypeScript `declare module "x" {`) is no
    /// construct at all, so its lines fall into gap units. A Go interface
    /// (`type Reader interface {`, kind trait) never splits at its methods either: a Go method
    /// signature carries no keyword (`Read(p []byte) (n int, err error)`), so no member is found
    /// in its body and the interface is reviewed whole, declared. A single such unit, or one
    /// declaration, too large for one
    /// batch therefore ends the review with the typed `ReviewError::TooLargeForBatch` (RV03),
    /// never a silent cut: no line is lost.
    fn split_block(&self, unit: &SemanticUnit) -> Vec<SemanticUnit> {
        if unit.kind == UnitKind::DeclarationGroup {
            return self.split_declaration_group(unit);
        }
        // MAGI FIX (S6 loop 3): a block too large for one batch is reviewed method by method.
        if !matches!(
            unit.kind,
            UnitKind::Impl | UnitKind::Class | UnitKind::Trait | UnitKind::Module
        ) {
            return Vec::new();
        }
        let lines: Vec<&str> = unit.content.lines().collect();
        let scanned = CodeScanner::new(self.lexical_rules.clone()).scan_all(&unit.content);
        // MAGI FIX (S6 loop 19, B2b-3): a multi-line decorator above a class
        // (`@Component({` ... `})`) held the unit's first opening brace, so the members were
        // sought inside the decorator's object and the class did not split. The search starts
        // at the block's own header, the first code line its keyword opens.
        let is_code = |j: usize| scanned.get(j).is_some_and(|s| !s.is_non_code);
        let header = lines
            .iter()
            .enumerate()
            .position(|(j, l)| is_code(j) && self.construct_keyword(l.trim()).is_some())
            .unwrap_or(0);
        let Some(open) = scanned
            .iter()
            .enumerate()
            .skip(header)
            .position(|(_, s)| !s.is_non_code && s.brace_delta > 0)
            .map(|p| p + header)
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
        let members = shift_members(
            self.detect_constructs(inner_lines, inner_scanned, true),
            open,
        )
        .unwrap_or_default();
        split_into_members(unit, members)
    }
}

/// The members found in a block's body, `open` being the 0-based line of the block's opening
/// brace, shifted from the body's line numbers to the block's.
///
/// # Arguments
///
/// * `constructs` - The constructs of the body, lines 1-based from the line after `open`.
/// * `open` - The 0-based line of the opening brace in the block.
///
/// # Returns
///
/// The members, or `None` when a shifted line number does not fit `u32`: such a block is
/// reviewed whole, never split at fabricated coordinates. O(m) in the members.
fn shift_members(constructs: Vec<Construct>, open: usize) -> Option<Vec<Member>> {
    // MAGI FIX (S6 loop 9): a `u32::MAX` fallback offset and saturating additions split a block
    // at fabricated ranges; an unrepresentable line now leaves the block whole.
    let offset = u32::try_from(open.checked_add(1)?).ok()?;
    constructs
        .into_iter()
        .map(|c| {
            Some((
                c.kind,
                c.name,
                c.start_line.checked_add(offset)?,
                c.end_line.checked_add(offset)?,
            ))
        })
        .collect()
}

/// Each construct paired with its share of the changed lines, or `None` when `shares` does not
/// hold exactly one share per construct, the contract of `ChangedLines::assign`. A `zip` stops
/// at the shorter side, so a share list one short would drop the last constructs and their
/// changed lines from every unit; on a mismatch the caller reviews the changed lines as gaps
/// instead, so none is lost. O(n) in the constructs.
///
/// # Arguments
///
/// * `constructs` - The file's constructs, sorted and disjoint.
/// * `shares` - The changed lines of each construct, in the same order.
///
/// # Returns
///
/// The pairs, or `None` when the two lengths differ.
fn construct_shares(
    constructs: &[Construct],
    shares: Vec<Vec<u32>>,
) -> Option<Vec<(&Construct, Vec<u32>)>> {
    (shares.len() == constructs.len()).then(|| constructs.iter().zip(shares).collect())
}

/// The modifier words of a language, stripped before its keyword tables are matched.
#[derive(Debug)]
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

/// Java's access modifiers and the other modifiers a class, interface or member may carry.
const JAVA_MODIFIERS: Modifiers = Modifiers {
    visibility: &["public", "private", "protected"],
    qualifiers: &[
        "static",
        "final",
        "abstract",
        "sealed",
        "non-sealed",
        "strictfp",
        "default",
        "synchronized",
        "native",
        "transient",
        "volatile",
    ],
};

/// Keywords that may stand right before a `(` without naming a function: control flow, operators
/// and statements of the C family, Java and JavaScript (`catch (e) {`, `synchronized (lock) {`,
/// `function (x) {`).
const NOT_FUNCTION_NAMES: [&str; 23] = [
    "if",
    "for",
    "while",
    "switch",
    "return",
    "sizeof",
    "else",
    "do",
    "catch",
    "try",
    "finally",
    "new",
    "delete",
    "throw",
    "synchronized",
    "function",
    "typeof",
    "await",
    "yield",
    "case",
    // MAGI FIX (S6 loop 9): a C++ trailing clause (`requires (C)`, `noexcept(true)`,
    // `-> decltype(a + b)`) on a wrapped signature's line read as a function name.
    "requires",
    "noexcept",
    "decltype",
];

/// The words of [`NOT_FUNCTION_NAMES`] that Java reserves: its control flow and statements
/// (`catch (e) {`, `synchronized (lock) {`, `try (var r = open()) {`) and `yield`, which a
/// switch expression's statement opens with. The others (`delete`, `typeof`, `await`,
/// `function`, `sizeof`, the C++ trailing clauses) are legal Java method names.
const JAVA_NOT_FUNCTION_NAMES: [&str; 15] = [
    "if",
    "for",
    "while",
    "switch",
    "return",
    "else",
    "do",
    "catch",
    "try",
    "finally",
    "new",
    "throw",
    "synchronized",
    "yield",
    "case",
];

/// The words of [`NOT_FUNCTION_NAMES`] that C reserves: its control flow, `sizeof` and C23's
/// `typeof`. The others (`delete`, `new`, `try`, `function`, `await`, ...) are legal C function
/// names (`int delete(int x) {`).
// MAGI FIX (B2b1 9): C shared the list of every language, so such a function was no unit.
const C_NOT_FUNCTION_NAMES: [&str; 10] = [
    "if", "for", "while", "switch", "return", "sizeof", "else", "do", "case", "typeof",
];

/// The words of [`NOT_FUNCTION_NAMES`] that C++ reserves: C's ([`C_NOT_FUNCTION_NAMES`]),
/// exception handling, `new` and `delete`, and the trailing clauses of a wrapped signature
/// (`requires (C)`, `noexcept(true)`, `-> decltype(a + b)`). `function`, `finally`, `await` and
/// `yield` are legal C++ names.
const CPP_NOT_FUNCTION_NAMES: [&str; 18] = [
    "if", "for", "while", "switch", "return", "sizeof", "else", "do", "case", "typeof", "catch",
    "try", "throw", "new", "delete", "requires", "noexcept", "decltype",
];

/// The C++ words a namespace or a class may open with: `inline namespace v1 {` (C++11) and the
/// C++20 module export `export namespace m {`. Qualifiers only, so they are stripped before the
/// block and function tables and never before the declaration table.
// MAGI FIX (B2b1 10): both namespaces matched no keyword and their functions became top-level.
// MAGI FIX (loop 28, B2b1a): C's storage classes and cv-qualifiers ([`C_QUALIFIERS`]) too.
const CPP_MODIFIERS: Modifiers = Modifiers {
    visibility: &[],
    qualifiers: &[
        "inline",
        "export",
        "static",
        "extern",
        "const",
        "volatile",
        "register",
        "thread_local",
        "_Thread_local",
    ],
};

/// The C storage classes and cv-qualifiers an aggregate defined with a variable may open with
/// (`static struct S {` ... `} s;`, `const struct S {` ... `} s = {1};`). Qualifiers only, so
/// they are stripped before the block and function tables and never before the declaration
/// table: `static int n;` stays a declaration. A variable of the type (`static struct S s;`) or
/// a function returning it (`static struct S *make(void) {`) still reads as one
/// ([`declares_variable`], [`returns_from_function`]).
const C_QUALIFIERS: [&str; 7] = [
    "static",
    "extern",
    "const",
    "volatile",
    "register",
    "thread_local",
    "_Thread_local",
];

/// C's modifiers: no visibility, the qualifiers of [`C_QUALIFIERS`].
// MAGI FIX (loop 28, B2b1a): `static struct S {` matched no keyword and fell into gap units.
const C_MODIFIERS: Modifiers = Modifiers {
    visibility: &[],
    qualifiers: &C_QUALIFIERS,
};

/// The words a JavaScript or TypeScript declaration may open with (`export default`,
/// `export abstract class`, `async function`).
const JS_MODIFIERS: Modifiers = Modifiers {
    visibility: &["export", "default", "declare"],
    qualifiers: &["async", "abstract"],
};

/// Intermediate representation of a detected construct.
struct Construct {
    kind: UnitKind,
    name: String,
    start_line: u32,
    end_line: u32,
}

/// The line visits per line of the text that the forward searches of one detection pass may
/// spend in all ([`SearchBudget`]). A starting point, not a measurement: a construct's searches
/// walk its header, a few lines even when wrapped, two or three times, so ordinary code spends a
/// small fraction of it; only a text built to keep one parameter list open per line reaches it.
///
/// Why this value: about ten times what ordinary code spends (two or three walks over a header
/// of a few lines), so no real file meets the cap, while an 8 MiB file still costs at most
/// about 2.7 × 10⁸ line visits in the worst case, bounded and linear. It is fixed, not a
/// setting, for the reason given at [`MAX_BRACE_SEARCH_LINES`]; once it is spent the searches
/// degrade as [`SearchBudget`] says, with no line lost.
// MAGI FIX (S6 loop 14): why the value, and why it is no setting.
const BRACE_SEARCH_VISITS_PER_LINE: usize = 32;

/// The line visits the forward searches of one detection pass may spend:
/// [`BRACE_SEARCH_VISITS_PER_LINE`] per line of the text, plus [`MAX_BRACE_SEARCH_LINES`] so
/// that a short text still allows one full search. Each search is bounded on its own
/// ([`open_brace_line`]), but a file of `f(` lines starts one per line, O(n ×
/// [`MAX_BRACE_SEARCH_LINES`]) visits (minutes for an 8 MiB file, before the review clock and
/// with the TUI not cancellable); the shared budget keeps a pass O(n).
///
/// Once it is spent, every later search of the pass degrades as one past its own bound does:
/// a construct whose brace is on its own line is still found, one whose brace is on a later line
/// ends on its own line, and the lines after it fall into later units. No line is lost: every
/// line stays in exactly one unit. Pinned by
/// `the_brace_searches_of_a_whole_file_spend_a_budget_linear_in_its_lines`.
///
/// The worst case is accepted, declared: a file crafted to keep a parameter list open on every
/// line spends the whole budget, about 2.7 × 10⁸ line visits for an 8 MiB file. The budget counts
/// visits, not the work inside one: a visit reads the line's scanner record and also walks its
/// text a few times (the trims, the stop rule's keyword and signature checks,
/// the C attribute scan), so it costs O(m) for a line of m characters, and a pass costs at most
/// its visits times the longest line's length. A crafted file whose few lines are megabytes long
/// spends few visits, since they are counted in lines, but each is that long: about a thousand
/// visits of a multi-megabyte line, seconds of work. Either way the pass stays linear in the
/// file's size, with a constant of up to about a thousand character reads per byte; it runs
/// while the review is prepared, before the review clock starts (RV06), and in the TUI it cannot
/// be cancelled (REQ-TUI-1); ordinary code spends a small fraction of it. The value is not
/// changed without a measurement.
// MAGI FIX (S6 loop 15): the accepted worst case, written where the budget is defined.
// MAGI FIX (S6 loop 18, B2b-10): the cost of one visit, O(line length), is declared.
struct SearchBudget {
    /// The visits the pass may spend in all.
    cap: usize,
    /// The visits spent so far.
    spent: Cell<usize>,
}

impl SearchBudget {
    /// The budget of a detection pass over a text of `lines` lines. O(1).
    fn for_lines(lines: usize) -> Self {
        // MAGI FIX (S6 loop 13): each search was bounded but a pass was not, so a file of `f(`
        // lines took O(n × MAX_BRACE_SEARCH_LINES) before the review clock.
        Self {
            cap: lines
                .saturating_mul(BRACE_SEARCH_VISITS_PER_LINE)
                .saturating_add(MAX_BRACE_SEARCH_LINES),
            spent: Cell::new(0),
        }
    }

    /// Spends one line visit; `false` when the pass has none left, so the search stops there.
    /// O(1).
    fn spend(&self) -> bool {
        let spent = self.spent.get();
        if spent >= self.cap {
            return false;
        }
        self.spent.set(spent + 1);
        true
    }

    /// The line visits spent so far.
    #[cfg(test)]
    fn spent(&self) -> usize {
        self.spent.get()
    }
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
            // MAGI FIX (S6 loop 17, B2-2): an item inside a `macro_rules!` token tree was read
            // as a function of its own; the macro is one function-like unit.
            function_keywords: vec!["fn ".to_string(), MACRO_RULES_KEYWORD.to_string()],
            block_keywords: vec![
                "struct ".to_string(),
                // MAGI FIX (S6 loop 9): a `union` fell into a gap unit; it is a struct-like type.
                "union ".to_string(),
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
            // MAGI FIX (S6 loop 6): a `/** ... */` rustdoc above an item fell into a gap.
            doc_comment_prefixes: vec![
                "///".to_string(),
                "//!".to_string(),
                "/**".to_string(),
                "/*!".to_string(),
            ],
            attribute_prefixes: vec!["#[".to_string()],
            lexical_rules: LexicalRules::rust(),
            signature_functions: false,
            semicolon_declarations: false,
            template_headers: false,
            not_function_names: &NOT_FUNCTION_NAMES,
            reserved_word_methods: false,
            computed_member_names: false,
            anonymous_functions: false,
            operator_overloads: false,
            optional_members: false,
        }
    }

    /// C extractor. It claims `.c` only: a `.h` header goes to the C++ extractor
    /// ([`Self::cpp`]), which reads every C construct this one does.
    pub fn c() -> Self {
        Self {
            language: "c".to_string(),
            // MAGI FIX (S6 loop 14): `.h` is read with the C++ rules ([`Self::cpp`]).
            exts: &["c"],
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
            attribute_prefixes: c_attribute_prefixes(),
            modifiers: &C_MODIFIERS,
            lexical_rules: LexicalRules::c(),
            signature_functions: true,
            semicolon_declarations: true,
            template_headers: false,
            not_function_names: &C_NOT_FUNCTION_NAMES,
            reserved_word_methods: false,
            computed_member_names: false,
            anonymous_functions: false,
            operator_overloads: false,
            optional_members: false,
        }
    }

    /// C++ extractor.
    ///
    /// It also claims `.h`: the extension does not say which language a header is written in,
    /// and this extractor is the C one ([`Self::c`]) plus classes, namespaces, template headers
    /// and raw strings, so it is the safer reading of both (RV03). A C++ header named `.h` gets
    /// its classes and namespaces as units, and a class too large for a batch splits at its
    /// methods; read with the C rules, they fell into gap units. A C header reads the same
    /// (pinned by `a_c_header_reads_the_same_with_the_cpp_rules`); the residual cost, declared, is a
    /// C identifier that opens a line as a C++ keyword would (`class = 1;`, `namespace = 2;`),
    /// whose line becomes a one-line unit of kind class instead of a declaration, and a C string
    /// glued to an `R` macro (`R"(`), read as a raw string that may run to the end of the file.
    /// Either way every line stays in exactly one unit, so no line is lost. The prompt's code
    /// fence says `cpp` for a `.h` file too (`review::context::detect_language`).
    ///
    /// A `template <...>` header belongs to the class or function it declares, on the line
    /// right above it or on the construct's own line (`template<typename T> class Box {`).
    pub fn cpp() -> Self {
        let mut ext = Self::c();
        ext.language = "cpp".to_string();
        ext.exts = &["cpp", "cxx", "cc", "hpp", "hxx", "h"];
        ext.block_keywords
            .extend(["class ".to_string(), "namespace ".to_string()]);
        ext.lexical_rules = LexicalRules::cpp();
        // MAGI FIX (S6 loop 8): a `template <...>` header above a class or a function was left
        // out of the unit it declares.
        ext.template_headers = true;
        ext.operator_overloads = true;
        ext.not_function_names = &CPP_NOT_FUNCTION_NAMES;
        ext.modifiers = &CPP_MODIFIERS;
        ext.attribute_prefixes
            .extend(["template<".to_string(), "template <".to_string()]);
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
            semicolon_declarations: false,
            template_headers: false,
            not_function_names: &NOT_FUNCTION_NAMES,
            reserved_word_methods: false,
            computed_member_names: false,
            anonymous_functions: false,
            operator_overloads: false,
            optional_members: false,
        }
    }

    /// Java extractor.
    pub fn java() -> Self {
        Self {
            language: "java".to_string(),
            exts: &["java"],
            // MAGI FIX (S6 loop 4): the modifiers were the function keywords, so a field
            // (`private String name;`) read as a method named after its type and a method with no
            // modifier was no unit; a method is now a signature, named by the identifier before
            // its `(`, and a field is not one. MAGI FIX (S6 loop 3), kept through the signature
            // path: a Java method is named by the identifier before `(`.
            function_keywords: vec![],
            // MAGI FIX (S6 loop 4): modifiers combine freely (`public final class`), so they are
            // stripped and the tables hold the bare keywords.
            block_keywords: vec![
                "class ".to_string(),
                "interface ".to_string(),
                "@interface ".to_string(),
                "enum ".to_string(),
                "record ".to_string(),
            ],
            declaration_keywords: vec!["import ".to_string(), "package ".to_string()],
            doc_comment_prefixes: vec!["/**".to_string(), "///".to_string()],
            attribute_prefixes: vec!["@".to_string()],
            modifiers: &JAVA_MODIFIERS,
            lexical_rules: LexicalRules::java(),
            signature_functions: true,
            semicolon_declarations: false,
            template_headers: false,
            not_function_names: &JAVA_NOT_FUNCTION_NAMES,
            reserved_word_methods: false,
            computed_member_names: false,
            anonymous_functions: false,
            operator_overloads: false,
            optional_members: false,
        }
    }

    /// JavaScript extractor.
    ///
    /// It does not claim `.jsx`, and the TypeScript one does not claim `.tsx`: the lexical rules
    /// have no JSX text mode, so an apostrophe in JSX text (`<p>it's</p>`) opens a string that
    /// runs to the next `'`, which may be the end of the file, and the unit around it runs there
    /// too. Such files are reviewed as text with no extractor. A `.js` or `.ts` file holding JSX
    /// keeps the limit, declared: a quote in its JSX text may make the unit around it run to the
    /// end of the file.
    pub fn javascript() -> Self {
        Self {
            language: "javascript".to_string(),
            // MAGI FIX (S6 loop 7): `.jsx` is not read with rules that have no JSX text mode.
            // MAGI FIX (S6 loop 9): a CommonJS `.cjs` file had no extractor.
            exts: &["js", "mjs", "cjs"],
            // `export`, `default`, `async` and `abstract` are stripped before these tables are
            // matched ([`JS_MODIFIERS`]), so they hold the bare keywords.
            function_keywords: vec!["function ".to_string()],
            block_keywords: vec!["class ".to_string()],
            declaration_keywords: vec![
                "import ".to_string(),
                "export ".to_string(),
                "const ".to_string(),
                "let ".to_string(),
                "var ".to_string(),
            ],
            doc_comment_prefixes: vec!["/**".to_string(), "///".to_string()],
            attribute_prefixes: vec!["@".to_string()],
            modifiers: &JS_MODIFIERS,
            lexical_rules: LexicalRules::javascript(),
            // MAGI FIX (S6 loop 4): a class method (`run() {`, `async run() {`, `get x() {`)
            // has no keyword, so a class too large for one batch could not split; a method is a
            // signature, as in C and Java.
            signature_functions: true,
            semicolon_declarations: false,
            template_headers: false,
            // MAGI FIX (S6 loop 14): a method named `delete`, `new` or `catch` was no member.
            not_function_names: &NOT_FUNCTION_NAMES,
            reserved_word_methods: true,
            // MAGI FIX (S6 loop 17, B2-3): `[key]() {` and `#m() {` were no members.
            computed_member_names: true,
            anonymous_functions: true,
            operator_overloads: false,
            optional_members: false,
        }
    }

    /// TypeScript extractor.
    pub fn typescript() -> Self {
        let mut ext = Self::javascript();
        ext.language = "typescript".to_string();
        // MAGI FIX (S6 loop 7): `.tsx` is not claimed, as `.jsx` is not ([`Self::javascript`]).
        // MAGI FIX (S6 loop 9): a CommonJS `.cts` file had no extractor.
        ext.exts = &["ts", "mts", "cts"];
        // MAGI FIX (B2b1 9): an enum or a namespace fell into gap units.
        ext.block_keywords
            .extend(["interface ", "enum ", "const enum ", "namespace "].map(str::to_string));
        ext.declaration_keywords.push("type ".to_string());
        ext.declaration_keywords.push("export type ".to_string());
        ext.lexical_rules = LexicalRules::typescript();
        ext.optional_members = true;
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
    fn an_ifdef_else_pair_of_headers_leaves_the_next_function_its_own_unit() {
        // [B2a] both branches of `#ifdef`/`#else` opened the block of `f`, so `f` ran to the end
        // of the file, absorbed `g`, and a large file failed as too large for a batch.
        let code = "#ifdef X\nint f(int a) {\n#else\nint f(void) {\n#endif\n    return 0;\n}\nint g(void) { return 1; }\n";
        for ext in [c_extractor(), BraceFamilyExtractor::cpp()] {
            let units = extract(&ext, code, &[6, 8]);
            let found = spans(&units);
            assert!(
                found
                    .iter()
                    .any(|&(n, k, _, end)| n == "f" && k == UnitKind::Function && end == 7),
                "{found:?}"
            );
            assert!(
                found.contains(&("g", UnitKind::Function, 8, 8)),
                "{found:?}"
            );
        }
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
    fn a_comment_ending_in_a_comma_does_not_continue_a_signature_into_the_next_one() {
        // [MAGI S6 loop 11, B2b-c1] the line before a construct counted as a wrapped
        // continuation when its raw text ended in `,`, comment or not: `foo`'s brace search
        // crossed `// note,` into `bar` and took `bar`'s brace, merging both into one unit.
        let code = "int foo(void)
// note,
int bar(void) {
    return 0;
}
";
        let units = extract(&c_extractor(), code, &[4]);
        assert_eq!(spans(&units), vec![("bar", UnitKind::Function, 3, 5)]);
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
    fn a_rust_module_splits_into_its_items() {
        // [MAGI S6 loop 6, B2b-m1] a `mod` block was a function-like unit that never split, so
        // a whole-file review of a file whose `mod tests` exceeded the batch failed outright.
        let code = "#[cfg(test)]\nmod tests {\n    use super::*;\n\n    fn a() {\n        x();\n    }\n\n    struct S {\n        n: u8,\n    }\n}\nfn after() {\n}\n";
        let ext = rust_extractor();
        let units = extract(&ext, code, &(1..=14).collect::<Vec<u32>>());
        let module = units
            .iter()
            .find(|u| u.name == "tests")
            .expect("the module is one unit");
        assert_eq!((module.start_line, module.end_line), (1, 12));
        assert!(
            units.iter().any(|u| u.name == "after"),
            "the next item is found"
        );
        let parts = ext.split_block(module);
        let names: Vec<&str> = parts.iter().map(|p| p.name.as_str()).collect();
        assert!(names.contains(&"a") && names.contains(&"S"), "{names:?}");
        assert_eq!(parts[0].content, "#[cfg(test)]\nmod tests {");
        assert_each_line_in_one_unit(&parts, 12);
    }

    #[test]
    fn a_call_or_a_one_line_function_does_not_take_a_following_control_block() {
        // [MAGI S6 loop 6, B2b-c1] a top-level `foo()` or a one-line function read as a
        // signature looking for its brace, and took the `{` of the `if` or `while` after it.
        let code = "foo()\nif (x) {\n  a();\n}\nfunction a() { x(); }\nwhile (y) {\n  b();\n}\nfunction g() {\n}\n";
        let units = extract(
            &BraceFamilyExtractor::javascript(),
            code,
            &(1..=10).collect::<Vec<u32>>(),
        );
        let named: Vec<(&str, u32, u32)> = units
            .iter()
            .filter(|u| u.kind == UnitKind::Function)
            .map(|u| (u.name.as_str(), u.start_line, u.end_line))
            .collect();
        assert_eq!(named, vec![("a", 5, 5), ("g", 9, 10)]);
        assert_each_line_in_one_unit(&units, 10);
        let c = "int f(void)\n// note\n{\n    return 0;\n}\n";
        let units = extract(&c_extractor(), c, &(1..=5).collect::<Vec<u32>>());
        assert!(
            units.iter().any(|u| u.name == "f" && u.end_line == 5),
            "a brace on its own line still opens the function"
        );
    }

    #[test]
    fn a_js_identifier_with_a_dollar_names_its_function_and_its_member() {
        // [MAGI S6 loop 6, B2b-c2] `$` was not an identifier character, so `$get() {` was no
        // member to split at and `function $bar()` was unnamed.
        let code =
            "class K {\n  $get() {\n    return 1;\n  }\n  a$b() {\n  }\n}\nfunction $bar() {\n}\n";
        let ext = BraceFamilyExtractor::javascript();
        let units = extract(&ext, code, &(1..=9).collect::<Vec<u32>>());
        assert!(
            units.iter().any(|u| u.name == "$bar"),
            "{:?}",
            spans(&units)
        );
        let class = units
            .iter()
            .find(|u| u.kind == UnitKind::Class)
            .expect("the class");
        let parts = ext.split_block(class);
        let names: Vec<&str> = parts.iter().map(|p| p.name.as_str()).collect();
        assert!(
            names.contains(&"$get") && names.contains(&"a$b"),
            "{names:?}"
        );
    }

    #[test]
    fn an_anonymous_default_class_is_a_class() {
        // [MAGI S6 loop 6, B2b-c5] `class {` already matched `class `; `class{`, with no blank
        // after the keyword, matched none, and its body fell into gap units.
        for head in ["export default class {", "export default class{"] {
            let code = format!("{head}\n  run() {{\n    go();\n  }}\n}}\nconst x = 1;\n");
            assert_an_anonymous_class(&code);
        }
    }

    fn assert_an_anonymous_class(code: &str) {
        let ext = BraceFamilyExtractor::typescript();
        let units = extract(&ext, code, &(1..=6).collect::<Vec<u32>>());
        let class = units
            .iter()
            .find(|u| u.kind == UnitKind::Class)
            .unwrap_or_else(|| panic!("no class: {:?}", spans(&units)));
        assert_eq!((class.start_line, class.end_line), (1, 5));
        assert!(ext.split_block(class).iter().any(|p| p.name == "run"));
        assert_each_line_in_one_unit(&units, 6);
    }

    #[test]
    fn a_c_global_with_a_multi_line_initializer_is_one_declaration() {
        // [MAGI S6 loop 6, B2b-c4] only a line ending in `;` was a C declaration, so an
        // initializer opened on its first line fell into gap units.
        let code = "int arr[] = {\n    1, 2,\n    3,\n};\nstatic int n =\n    4;\nint f(void) {\n    return 0;\n}\n";
        let units = extract(&c_extractor(), code, &(1..=9).collect::<Vec<u32>>());
        let kinds: Vec<(UnitKind, u32, u32)> = spans(&units)
            .into_iter()
            .map(|(_, kind, start, end)| (kind, start, end))
            .collect();
        assert_eq!(
            kinds,
            vec![
                (UnitKind::DeclarationGroup, 1, 6),
                (UnitKind::Function, 7, 9),
            ]
        );
    }

    #[test]
    fn a_receiver_an_arrow_bound_or_a_struct_variable_names_the_right_unit() {
        // [MAGI S6 loop 6, B2b-m4] a Go method with a receiver was unnamed, the `->` of an
        // impl bound closed its generics, and a `struct Foo x = {1};` global was a struct.
        let go = "package p\n\nfunc (r *T) Name() string {\n\treturn r.n\n}\n";
        let units = extract(&BraceFamilyExtractor::go(), go, &[4]);
        assert!(
            units.iter().any(|u| u.name == "Name"),
            "{:?}",
            spans(&units)
        );
        let rust = "impl<T: Fn(u32) -> u32> Foo<T> {\n    fn a() {}\n}\n";
        let units = extract(&rust_extractor(), rust, &[2]);
        assert!(units.iter().any(|u| u.name == "Foo"), "{:?}", spans(&units));
        let c = "struct Foo x = {1};\nstruct Bar {\n    int n;\n};\n";
        let units = extract(&c_extractor(), c, &[1, 3]);
        let kinds: Vec<(UnitKind, u32, u32)> = spans(&units)
            .into_iter()
            .map(|(_, kind, start, end)| (kind, start, end))
            .collect();
        assert_eq!(
            kinds,
            vec![(UnitKind::DeclarationGroup, 1, 1), (UnitKind::Struct, 2, 4)]
        );
    }

    #[test]
    fn a_rust_block_doc_comment_belongs_to_its_item() {
        // [MAGI S6 loop 6, B2b-m5] Rust's doc prefixes were `///` and `//!` only, so a
        // `/** ... */` rustdoc above an item fell into a gap unit.
        let code = "use a::b;\n\n/**\n * Doc.\n */\nfn a() {\n    x();\n}\n";
        let units = extract(&rust_extractor(), code, &[7]);
        let a = units.iter().find(|u| u.name == "a").expect("fn a");
        assert_eq!((a.start_line, a.end_line), (3, 8));
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
        let parts = split(&BraceFamilyExtractor::cpp(), cpp, 3, UnitKind::Module);
        assert_eq!(
            spans(&parts),
            vec![
                ("ns", UnitKind::Module, 1, 1),
                ("f", UnitKind::Function, 2, 4),
                ("ns", UnitKind::Module, 5, 5)
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
    fn control_keywords_before_a_parenthesis_are_not_function_names() {
        // [MAGI S6 loop 4, B2] `catch (e) {` read as a function named `catch`.
        for (ext, line) in [
            (BraceFamilyExtractor::cpp(), "    catch (const E& e) {"),
            (BraceFamilyExtractor::cpp(), "    throw (e);"),
            (BraceFamilyExtractor::cpp(), "    delete (p);"),
            (BraceFamilyExtractor::java(), "    catch (Exception e) {"),
            (BraceFamilyExtractor::java(), "    synchronized (lock) {"),
            (BraceFamilyExtractor::java(), "    @SuppressWarnings(\"x\")"),
            (BraceFamilyExtractor::javascript(), "  catch (e) {"),
            (BraceFamilyExtractor::javascript(), "  function (e) {"),
            (BraceFamilyExtractor::javascript(), "  switch (k) {"),
        ] {
            assert_eq!(ext.signature_name(line, None, false), None, "{line}");
        }
    }

    #[test]
    fn java_fields_are_not_functions_and_methods_without_modifiers_are() {
        // [MAGI S6 loop 4, B2] a field with a modifier read as a function named after its type,
        // and a method with no modifier was no unit at all.
        let java = "public class Foo {\n    public static final int X = 1;\n    private String name;\n\n    void run() {\n        a();\n    }\n\n    @Override\n    public String toString() {\n        return name;\n    }\n}\n";
        let parts = split(&BraceFamilyExtractor::java(), java, 6, UnitKind::Class);
        let found = spans(&parts);
        assert!(
            found.contains(&("run", UnitKind::Method, 5, 7)),
            "{found:?}"
        );
        assert!(
            found.contains(&("toString", UnitKind::Method, 9, 12)),
            "{found:?}"
        );
        assert!(
            !found
                .iter()
                .any(|(n, k, ..)| *k == UnitKind::Method && ["int", "String", "X"].contains(n)),
            "{found:?}"
        );
        assert_each_line_in_one_unit(&parts, 13);
    }

    #[test]
    fn a_javascript_class_splits_into_its_methods() {
        // [MAGI S6 loop 4, B2] class methods matched no keyword, so a class too large for one
        // batch could not split and failed the review as too large.
        let js = "class C {\n  constructor(a) {\n    this.a = a;\n  }\n\n  async run() {\n    work();\n  }\n\n  get x() {\n    return 1;\n  }\n}\n";
        for ext in [
            BraceFamilyExtractor::javascript(),
            BraceFamilyExtractor::typescript(),
        ] {
            let parts = split(&ext, js, 3, UnitKind::Class);
            let found = spans(&parts);
            for member in [
                ("constructor", UnitKind::Method, 2, 4),
                ("run", UnitKind::Method, 6, 8),
                ("x", UnitKind::Method, 10, 12),
            ] {
                assert!(found.contains(&member), "{found:?}");
            }
            assert_each_line_in_one_unit(&parts, 13);
        }
    }

    #[test]
    fn c_globals_form_a_declaration_group() {
        // [MAGI S6 loop 4, B2] a global with no declaration keyword fell into a gap.
        let code = "int counter = 0;\nstatic int state;\nconst char *name = \"x\";\n\nint main(void) {\n    return 0;\n}\n";
        for ext in [c_extractor(), BraceFamilyExtractor::cpp()] {
            let units = extract(&ext, code, &[1, 2, 3]);
            assert_eq!(
                spans(&units),
                vec![("declarations", UnitKind::DeclarationGroup, 1, 3)]
            );
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

    #[test]
    fn a_generic_rust_impl_is_an_impl_block_that_splits_into_its_methods() {
        // [MAGI S6 loop 5, B2-m2] the keyword was the literal `impl `, so `impl<T> Foo<T>` was no
        // block: its methods became top-level functions without the impl header and bounds.
        for (code, name) in [
            ("impl<T> Foo<T> {\n    fn a(&self) {\n        x();\n    }\n}\n", "Foo"),
            (
                "unsafe impl<T: Send> Send for Foo<T> {\n    fn a(&self) {\n        x();\n    }\n}\n",
                "Send",
            ),
            (
                "pub(crate) impl<'a, T: Iterator<Item = u8>> Trait for Wrap<'a, T> {\n    fn a(&self) {\n        x();\n    }\n}\n",
                "Trait",
            ),
        ] {
            let units = extract(&rust_extractor(), code, &[3]);
            assert_eq!(spans(&units), vec![(name, UnitKind::Impl, 1, 5)], "{code}");
            let parts = split(&rust_extractor(), code, 3, UnitKind::Impl);
            assert!(
                spans(&parts).contains(&("a", UnitKind::Method, 2, 4)),
                "{code}: {:?}",
                spans(&parts)
            );
        }
    }

    #[test]
    fn an_arrow_function_or_function_expression_keeps_its_body() {
        // [MAGI S6 loop 5, B2-m1] `const f = () => {` was a one-line declaration, so its body fell
        // into a gap reviewed without the signature.
        let js = "export const f = () => {\n  return 1;\n};\n\nfunction g() {\n  x();\n}\n";
        let units = extract(&BraceFamilyExtractor::javascript(), js, &[2]);
        assert_eq!(spans(&units), vec![("f", UnitKind::Function, 1, 3)]);
        for (ext, code, name, end) in [
            (
                BraceFamilyExtractor::typescript(),
                "const h = async (a: number): Promise<void> => {\n  await a;\n};\n",
                "h",
                3,
            ),
            (
                BraceFamilyExtractor::javascript(),
                "const k = function (x) {\n  return x;\n};\n",
                "k",
                3,
            ),
            (
                BraceFamilyExtractor::javascript(),
                "let w = (\n  a,\n  b,\n) => {\n  return a;\n};\n",
                "w",
                6,
            ),
        ] {
            let units = extract(&ext, code, &[2]);
            assert_eq!(
                spans(&units),
                vec![(name, UnitKind::Function, 1, end)],
                "{code}"
            );
        }
    }

    #[test]
    fn a_declaration_whose_value_opens_a_block_keeps_the_block() {
        // [MAGI S6 loop 5, B2-m1] an object literal or a wrapped `use` ended on its first line and
        // the rest of the declaration fell into a gap.
        let js = "const o = {\n  a: 1,\n};\nconst p = 2;\n";
        let units = extract(&BraceFamilyExtractor::javascript(), js, &[2]);
        assert_eq!(
            spans(&units),
            vec![("constants", UnitKind::DeclarationGroup, 1, 4)]
        );
        let rust = "use a::{\n    b,\n    c,\n};\n\nfn f() {}\n";
        let units = extract(&rust_extractor(), rust, &[2]);
        assert_eq!(
            spans(&units),
            vec![("imports", UnitKind::DeclarationGroup, 1, 4)]
        );
        // A complete declaration takes no later block: the `if` stays out of it.
        let js = "const a = 1\nif (a) {\n  x()\n}\n";
        let units = extract(&BraceFamilyExtractor::javascript(), js, &[1]);
        assert_eq!(
            spans(&units),
            vec![("constants", UnitKind::DeclarationGroup, 1, 1)]
        );
    }

    #[test]
    fn an_extern_c_block_keeps_every_line_of_its_contents_in_a_unit() {
        // [MAGI S6 loop 5, B2-c3, verified as a false positive] the report said the contents of
        // `extern "C" {` fell into the gap fallback. They do not: the block only sets linkage, so
        // its prototypes, globals and functions are top-level constructs (the header joins the
        // declarations right after it), and its closing brace is a unit of its own; no line is
        // lost or reviewed twice.
        let code = "extern \"C\" {\n    int add(int a, int b);\n    extern int count;\n\n    int sub(int a, int b) {\n        return a - b;\n    }\n}\n";
        for ext in [c_extractor(), BraceFamilyExtractor::cpp()] {
            let every: HashSet<u32> = (1..=8).collect();
            let units = ext.extract_units(code, "a.h", &every);
            let found = spans(&units);
            assert!(
                found.contains(&("sub", UnitKind::Function, 5, 7)),
                "{found:?}"
            );
            assert!(
                found
                    .iter()
                    .any(|(_, k, s, e)| *k == UnitKind::DeclarationGroup && (*s, *e) == (1, 3)),
                "{found:?}"
            );
            assert_each_line_in_one_unit(&units, 8);
        }
    }

    #[test]
    fn an_interface_splits_into_its_abstract_method_signatures() {
        // [MAGI S6 loop 5, B2-m3] a member signature ending in `;` was read as a C prototype and
        // skipped, so a Java or TypeScript interface had no members and could not split.
        let java = "public interface Repo {\n    /** Finds. */\n    User find(long id);\n\n    void save(\n        User u);\n}\n";
        let parts = split(&BraceFamilyExtractor::java(), java, 3, UnitKind::Trait);
        let found = spans(&parts);
        for member in [
            ("find", UnitKind::Method, 2, 3),
            ("save", UnitKind::Method, 5, 6),
        ] {
            assert!(found.contains(&member), "{found:?}");
        }
        assert_each_line_in_one_unit(&parts, 7);
        let ts =
            "export interface Api {\n  get(id: string): Promise<User>;\n  put(u: User): void;\n}\n";
        let parts = split(&BraceFamilyExtractor::typescript(), ts, 2, UnitKind::Trait);
        let found = spans(&parts);
        for member in [
            ("get", UnitKind::Method, 2, 2),
            ("put", UnitKind::Method, 3, 3),
        ] {
            assert!(found.contains(&member), "{found:?}");
        }
        assert_each_line_in_one_unit(&parts, 4);
    }

    #[test]
    fn a_multi_line_doc_block_comment_belongs_to_the_declaration_below_it() {
        // [MAGI S6 loop 5, B2-m4] only the line right above a declaration was looked at, and for a
        // `/** ... */` block that line is `*/`, so JavaDoc, JSDoc and C doc blocks were left out.
        for (ext, code, name, end) in [
            (
                BraceFamilyExtractor::java(),
                "/**\n * Runs.\n * @param x the x\n */\n@Override\npublic void run(int x) {\n    a();\n}\n",
                "run",
                8,
            ),
            (
                BraceFamilyExtractor::javascript(),
                "/**\n * Adds.\n */\nexport function add(a, b) {\n  return a + b;\n}\n",
                "add",
                6,
            ),
            (
                c_extractor(),
                "/** Doc.\n *  More.\n */\nint f(void) {\n    return 0;\n}\n",
                "f",
                6,
            ),
        ] {
            let units = extract(&ext, code, &[end - 1]);
            assert_eq!(spans(&units), vec![(name, UnitKind::Function, 1, end)], "{code}");
        }
        // A plain block comment is no doc comment and stays out, as a plain line comment does.
        let code = "/*\n * plain\n */\nint f(void) {\n    return 0;\n}\n";
        let units = extract(&c_extractor(), code, &[5]);
        assert_eq!(spans(&units), vec![("f", UnitKind::Function, 4, 6)]);
    }

    /// Every line of `code`, 1-based.
    fn all_lines(code: &str) -> Vec<u32> {
        (1..=u32::try_from(code.lines().count()).expect("small")).collect()
    }

    #[test]
    fn a_wrapped_signature_with_a_function_pointer_parameter_is_one_function() {
        // [MAGI S6 loop 7, B2b-c1] a parameter line holding `(` before its own name
        // (`int (*callback)(int),`) read as the next construct's signature, so the brace search
        // stopped there and the function was cut at that line.
        for (ext, code) in [
            (
                c_extractor(),
                "void foo(\n    int (*callback)(int),\n    int y\n) {\n    callback(y);\n}\nint bar(void) {\n    return 0;\n}\n",
            ),
            (
                BraceFamilyExtractor::cpp(),
                "void foo(\n    const std::function<void(int)>& callback,\n    int y\n) {\n    callback(y);\n}\nint bar(void) {\n    return 0;\n}\n",
            ),
        ] {
            let units = extract(&ext, code, &all_lines(code));
            assert_eq!(
                spans(&units),
                vec![
                    ("foo", UnitKind::Function, 1, 6),
                    ("bar", UnitKind::Function, 7, 9)
                ],
                "{code}"
            );
        }
    }

    #[test]
    fn a_multi_line_define_keeps_its_continuation_lines() {
        // [MAGI S6 loop 7, B2b-c2] a `\` did not continue a declaration, so a macro body was
        // reviewed in a gap without its `#define` line.
        for (code, changed, end) in [
            (
                "#define MAX(a, b) \\\n    ((a) > (b) ? (a) : (b))\n#define ONE 1\n\nint f(void) {\n    return MAX(1, 2);\n}\n",
                2,
                3,
            ),
            (
                "#define SWAP(a, b) do { \\\n    int t = a; \\\n    a = b; b = t; \\\n} while (0)\n\nint f(void) {\n    return 0;\n}\n",
                3,
                4,
            ),
        ] {
            let units = extract(&c_extractor(), code, &[changed]);
            assert_eq!(
                spans(&units),
                vec![("defines", UnitKind::DeclarationGroup, 1, end)],
                "{code}"
            );
        }
    }

    #[test]
    fn a_crlf_file_gives_the_units_of_its_lf_twin_with_lf_line_ends() {
        // [MAGI S6 loop 16, B2-2] pins the declared normalization: unit content is the unit's
        // lines joined by '\n' (`str::lines` drops the '\r' of "\r\n"), as every extractor,
        // `gap_units` and `split_into_members` build and expect it. So a CRLF file's units hold
        // the same lines, numbers and text as its LF twin, with no '\r' and no line lost.
        let lf = "use a::b;\n\nfn f() {\n    let x = 1;\n}\n\nconst N: u8 = 2;\n";
        let crlf = lf.replace('\n', "\r\n");
        let ext = rust_extractor();
        let changed = all_lines(lf);
        let from_lf = extract(&ext, lf, &changed);
        let from_crlf = extract(&ext, &crlf, &changed);
        assert_eq!(spans(&from_crlf), spans(&from_lf));
        for (c, l) in from_crlf.iter().zip(&from_lf) {
            assert_eq!(c.content, l.content, "{}", c.name);
            assert!(!c.content.contains('\r'), "{}", c.name);
        }
        assert_each_line_in_one_unit(&from_crlf, 7);
    }

    #[test]
    fn a_c_declaration_with_a_trailing_comment_joins_the_declaration_group() {
        // [MAGI S6 loop 16, B2-1] `is_declaration` asked whether the trimmed text ended in `;`,
        // so `static int counter = 0; // note` fell into a gap unit; the scanner's end of
        // statement, as in `signature_candidate`, ignores a comment after the code.
        let c = c_extractor();
        let code = "static int counter = 0; // note\nstatic int limit = 9; /* cap */\nint g(void) {\n    return 0;\n}\n";
        let units = extract(&c, code, &[1]);
        assert_eq!(
            spans(&units),
            vec![("statics", UnitKind::DeclarationGroup, 1, 2)]
        );
        // A comment that merely ends in `;` after code that does not end a statement is no
        // declaration: `int f(void) // note;` is a signature line.
        let signature = "int f(void) // note;\n{\n    return 1;\n}\n";
        let units = extract(&c, signature, &[1]);
        assert_eq!(spans(&units), vec![("f", UnitKind::Function, 1, 4)]);
    }

    #[test]
    fn a_braced_initializer_of_a_struct_variable_is_a_declaration() {
        // [MAGI S6 loop 7, B2b-c3] `struct Foo x{1};` read its initializer's `{` as the struct
        // body, so a variable was named as the struct `Foo`.
        let cpp = BraceFamilyExtractor::cpp();
        let code = "struct Foo x{1};\nint g(void) {\n    return 0;\n}\n";
        let units = extract(&cpp, code, &[1]);
        assert_eq!(
            spans(&units),
            vec![("declarations", UnitKind::DeclarationGroup, 1, 1)]
        );
        // A struct definition stays a struct, `final` and a one-line body included.
        for (code, name, end) in [
            (
                "struct Foo final {
    int x;
};
",
                "Foo",
                3,
            ),
            (
                "struct P { int x; };
",
                "P",
                1,
            ),
        ] {
            let units = extract(&cpp, code, &all_lines(code));
            assert_eq!(
                spans(&units),
                vec![(name, UnitKind::Struct, 1, end)],
                "{code}"
            );
        }
    }

    #[test]
    fn a_c_function_returning_a_struct_or_an_enum_is_a_function() {
        // [MAGI S6 loop 7, B2b-m2] the block keyword was matched before the signature, so
        // `struct Point make_point(void) {` became a struct named `Point`.
        let code = "struct Point make_point(void) {\n    struct Point p = {0, 0};\n    return p;\n}\nenum Color pick(void) {\n    return RED;\n}\nstruct Point *origin(void) {\n    return 0;\n}\n";
        for ext in [c_extractor(), BraceFamilyExtractor::cpp()] {
            let units = extract(&ext, code, &all_lines(code));
            assert_eq!(
                spans(&units),
                vec![
                    ("make_point", UnitKind::Function, 1, 4),
                    ("pick", UnitKind::Function, 5, 7),
                    ("origin", UnitKind::Function, 8, 10)
                ]
            );
        }
        // An attribute's parentheses before the type's name keep it a struct, and a prototype
        // returning a struct stays a declaration.
        let code = "struct __attribute__((packed)) Hdr {\n    int a;\n};\nstruct Point make_point(void);\n";
        let units = extract(&c_extractor(), code, &all_lines(code));
        let kinds: Vec<UnitKind> = units.iter().map(|u| u.kind).collect();
        assert_eq!(kinds, vec![UnitKind::Struct, UnitKind::DeclarationGroup]);
    }

    #[test]
    fn a_cpp_scoped_enum_is_named_by_its_own_name() {
        // [MAGI S6 loop 7, B2b-m2, B2b-b1] `enum class Color {` was named `class`.
        let code = "enum class Color : int {\n    Red,\n};\nenum struct Mode {\n    A,\n};\n";
        let units = extract(&BraceFamilyExtractor::cpp(), code, &all_lines(code));
        assert_eq!(
            spans(&units),
            vec![
                ("Color", UnitKind::Enum, 1, 3),
                ("Mode", UnitKind::Enum, 4, 6)
            ]
        );
    }

    #[test]
    fn a_cpp_template_header_belongs_to_the_construct_it_declares() {
        // [MAGI S6 loop 8, B2b-m1, B2b-c1] a `template <...>` header on its own line fell into a
        // gap unit, and on the construct's line (`template<typename T> class Box {`) the line
        // was no construct at all, so its body was cut into gap units.
        let code = "template <typename T>\nclass Vec {\n    T x;\n};\ntemplate<typename T> class Box {\n    T y;\n};\ntemplate <typename T>\nT twice(T x) {\n    return x;\n}\ntemplate <void (*F)(int)> void call() {\n    F(1);\n}\n";
        let units = extract(&BraceFamilyExtractor::cpp(), code, &all_lines(code));
        assert_eq!(
            spans(&units),
            vec![
                ("Vec", UnitKind::Class, 1, 4),
                ("Box", UnitKind::Class, 5, 7),
                ("twice", UnitKind::Function, 8, 11),
                ("call", UnitKind::Function, 12, 14),
            ]
        );
    }

    #[test]
    fn a_c_declaration_list_wrapped_after_a_comma_is_one_declaration() {
        // [MAGI S6 loop 8, B2b-b1] `int x,` opened no declaration, so the first line of a
        // declaration list fell into a gap unit apart from the rest of it.
        let code = "int x,\n    y;\nstatic const char *a = \"x\",\n                  *b = \"y\";\nint n = 1,\n    m[] = {\n        2,\n    };\nvoid f(void) {\n}\n";
        let units = extract(&c_extractor(), code, &all_lines(code));
        assert_eq!(
            spans(&units),
            vec![
                ("declarations", UnitKind::DeclarationGroup, 1, 8),
                ("f", UnitKind::Function, 9, 10),
            ]
        );
        // A constructor's initializer lines end in `,` too; they stay part of its function.
        let cpp = "Foo::Foo(int a)\n    : a_{a},\n      b_{0},\n      c_{1} {\n    run();\n}\n";
        let units = extract(&BraceFamilyExtractor::cpp(), cpp, &all_lines(cpp));
        assert_eq!(spans(&units), vec![("Foo", UnitKind::Function, 1, 6)]);
    }

    #[test]
    fn a_split_declaration_group_keeps_a_wrapped_prototype_in_one_piece() {
        // [MAGI S6 loop 25, B2b1a] the group joined a wrapped prototype (`int foo(` ... `);`)
        // but its split did not see the opening line as a declaration, so the prototype was cut
        // between a gap piece and a piece starting at its first parameter.
        let code = "int a;\nint foo(\n    int x,\n    int y);\nint b;\nvoid g(void) {\n}\n";
        let parts = split(&c_extractor(), code, 1, UnitKind::DeclarationGroup);
        let decl = |a, b| ("declarations", UnitKind::DeclarationGroup, a, b);
        assert_eq!(spans(&parts), vec![decl(1, 1), decl(2, 4), decl(5, 5)]);
    }

    #[test]
    fn a_c_declaration_list_with_a_function_declarator_splits_as_one_declaration() {
        // [MAGI B2b1 8] a list line holding a function declarator (`g(void),`) read as a
        // signature and ended the list before it, so a split group cut `int a, g(void), b;`
        // in two pieces.
        for code in [
            "int a,\n    g(void),\n    b;\nint c;\nvoid h(void) {\n}\n",
            "int a,\n    g(int x,\n      int y),\n    b;\nint c;\nvoid h(void) {\n}\n",
        ] {
            let parts = split(&c_extractor(), code, 1, UnitKind::DeclarationGroup);
            let last = u32::try_from(code.lines().count()).expect("small") - 3;
            assert_eq!(
                spans(&parts),
                vec![
                    ("declarations", UnitKind::DeclarationGroup, 1, last),
                    (
                        "declarations",
                        UnitKind::DeclarationGroup,
                        last + 1,
                        last + 1
                    ),
                ],
                "{code}"
            );
        }
    }

    #[test]
    fn a_cpp_constructor_initializer_or_trailing_clause_continues_its_signature() {
        // [MAGI S6 loop 9, B2b-b1] `: value_(value) {` read as a signature named `value_`, so the
        // brace search stopped there and the constructor fell into gap units.
        let cases = [
            (
                "Widget::Widget(int value)\n    : value_(value),\n      count_(0) {\n    run();\n}\nvoid g() {\n}\n",
                "Widget",
                5,
            ),
            (
                "Widget::Widget(int v)\n    : value_(v)\n    , count_(0)\n{\n    run();\n}\nvoid g() {\n}\n",
                "Widget",
                6,
            ),
            (
                "Widget::Widget(int v) : value_(v) {\n    run();\n}\nvoid g() {\n}\n",
                "Widget",
                3,
            ),
            (
                "template <typename T>\nvoid f(T t)\n    requires (sizeof(T) > 4)\n{\n    run();\n}\nvoid g() {\n}\n",
                "f",
                6,
            ),
            (
                "auto add(int a, int b)\n    -> decltype(a + b)\n{\n    return a + b;\n}\nvoid g() {\n}\n",
                "add",
                5,
            ),
            (
                "void h()\n    noexcept(true)\n{\n    run();\n}\nvoid g() {\n}\n",
                "h",
                5,
            ),
        ];
        for (code, name, end) in cases {
            let units = extract(&BraceFamilyExtractor::cpp(), code, &all_lines(code));
            assert_eq!(
                spans(&units),
                vec![
                    (name, UnitKind::Function, 1, end),
                    ("g", UnitKind::Function, end + 1, end + 2),
                ],
                "{code}"
            );
        }
    }

    #[test]
    fn a_cpp_assignment_operator_overload_is_a_function() {
        // [MAGI S6 loop 9, B2b-m1] the `=` of `operator=` read as an initializer before the `(`,
        // so an overload was no function and a class made of them could not split.
        let cases = [
            (
                "Foo& Foo::operator=(const Foo& o) {
    return *this;
}
void g() {
}
",
                "operator=",
            ),
            (
                "bool operator==(const Foo& a, const Foo& b) {
    return true;
}
void g() {
}
",
                "operator==",
            ),
            (
                "Foo& operator<<=(int n) {
    return *this;
}
void g() {
}
",
                "operator<<=",
            ),
        ];
        for (code, name) in cases {
            let units = extract(&BraceFamilyExtractor::cpp(), code, &all_lines(code));
            assert_eq!(
                spans(&units),
                vec![
                    (name, UnitKind::Function, 1, 3),
                    ("g", UnitKind::Function, 4, 5)
                ],
                "{code}"
            );
        }
    }

    #[test]
    fn a_rust_union_is_a_struct_like_unit() {
        // [MAGI S6 loop 9, B2b-b3] `union` was missing from Rust's block keywords, so a union's
        // lines fell into a gap unit instead of a type unit.
        let code = "pub union U {
    a: u32,
    b: f32,
}
fn g() {
}
";
        let units = extract(&rust_extractor(), code, &all_lines(code));
        assert_eq!(
            spans(&units),
            vec![
                ("U", UnitKind::Struct, 1, 4),
                ("g", UnitKind::Function, 5, 6)
            ]
        );
    }

    #[test]
    fn a_brace_in_an_open_parameter_list_loses_no_line() {
        // [MAGI S6 loop 8, B2b-m2] a lambda default argument's brace was taken for the
        // function's; every line stays in exactly one unit either way (since B2b1 7 the function
        // is whole, `a_lambda_default_argument_over_lines_keeps_the_function_whole`).
        let code = "void f(int cb = []() {
           return 0;
       }) {
    run();
}
void g(void) {
}
";
        let units = extract(&BraceFamilyExtractor::cpp(), code, &all_lines(code));
        assert_each_line_in_one_unit(&units, 7);
    }

    #[test]
    fn a_lambda_default_argument_over_lines_keeps_the_function_whole() {
        // [MAGI B2b1 7] a brace opened inside the parameter list (a lambda default argument)
        // was taken for the function's, so the function ended at the lambda's `}` and its body
        // fell into a gap unit.
        let cpp = "void f(int cb = []() {\n           return 0;\n       }) {\n    if (x) {\n        run();\n    }\n}\nvoid g(void) {\n}\n";
        assert_eq!(
            spans(&extract(&BraceFamilyExtractor::cpp(), cpp, &all_lines(cpp))),
            vec![
                ("f", UnitKind::Function, 1, 7),
                ("g", UnitKind::Function, 8, 9)
            ]
        );
        let js = "function f(cb = () => {\n  x();\n}) {\n  y();\n}\nfunction g() {\n}\n";
        for ext in [
            BraceFamilyExtractor::javascript(),
            BraceFamilyExtractor::typescript(),
        ] {
            assert_eq!(
                spans(&extract(&ext, js, &all_lines(js))),
                vec![
                    ("f", UnitKind::Function, 1, 5),
                    ("g", UnitKind::Function, 6, 7)
                ]
            );
        }
        // A default argument whose braces close on the signature's line was already whole.
        let one_line = "function f(cb = () => { x(); }) {\n  y();\n}\nfunction g() {\n}\n";
        assert_eq!(
            spans(&extract(
                &BraceFamilyExtractor::javascript(),
                one_line,
                &all_lines(one_line)
            )),
            vec![
                ("f", UnitKind::Function, 1, 3),
                ("g", UnitKind::Function, 4, 5)
            ]
        );
    }

    #[test]
    fn a_grouped_go_type_declaration_yields_each_member() {
        // [MAGI S6 loop 7, B2b-m1] `type (` opened one nameless struct that ended at the first
        // member's brace, so the later members fell into gap units.
        let code = "type (\n\tPoint struct {\n\t\tX int\n\t}\n\n\t// Reader reads.\n\tReader interface {\n\t\tRead() error\n\t}\n\tID int\n)\n\nfunc f() {\n}\n";
        let total = u32::try_from(code.lines().count()).expect("small");
        let units = extract(&BraceFamilyExtractor::go(), code, &all_lines(code));
        assert_each_line_in_one_unit(&units, total);
        let named: Vec<_> = spans(&units)
            .into_iter()
            .filter(|s| s.1 != UnitKind::TopLevel)
            .collect();
        assert_eq!(
            named,
            vec![
                ("Point", UnitKind::Struct, 2, 4),
                ("Reader", UnitKind::Trait, 6, 9),
                ("ID", UnitKind::Struct, 10, 10),
                ("f", UnitKind::Function, 13, 14)
            ]
        );
        // An unclosed group still finds its members and covers every line once.
        let code = "type (\n\tA struct {\n\t\tX int\n\t}\n\tB int\n";
        let units = extract(&BraceFamilyExtractor::go(), code, &all_lines(code));
        assert_each_line_in_one_unit(&units, 5);
        assert!(units.iter().any(|u| u.name == "B"), "{:?}", spans(&units));
    }

    #[test]
    fn a_javascript_generator_method_is_a_member() {
        // [MAGI S6 loop 7, B2b-b2] a line opening with `*` was taken for a comment line, so
        // `*items() {` was no member and a large class could not split there.
        let code = "class Gen {\n  /**\n   * Calls helper(x).\n   */\n  *items() {\n    yield 1;\n  }\n  async *more() {\n    yield 2;\n  }\n}\n";
        let parts = split(
            &BraceFamilyExtractor::javascript(),
            code,
            6,
            UnitKind::Class,
        );
        let methods: Vec<&str> = parts
            .iter()
            .filter(|p| p.kind == UnitKind::Method)
            .map(|p| p.name.as_str())
            .collect();
        assert_eq!(methods, vec!["items", "more"]);
    }

    #[test]
    fn no_extractor_claims_jsx_or_tsx() {
        // [MAGI S6 loop 7, B2a] the JavaScript and TypeScript rules have no JSX text mode: an
        // apostrophe in JSX text (`<p>it's</p>`) opened a string to the end of the file, so the
        // unit around it ran to the end of the file. Such files are reviewed without an extractor.
        for ext in all_extractors() {
            for claimed in ["jsx", "tsx"] {
                assert!(
                    !ext.extensions().contains(&claimed),
                    "{} claims .{claimed}",
                    ext.language_id()
                );
            }
        }
    }

    #[test]
    fn commonjs_extensions_are_claimed_by_their_language() {
        // [MAGI S6 loop 9, B2b-b2] `.cjs` and `.cts` had no extractor, so a CommonJS file got no
        // lexical language context.
        let registry = crate::review::languages::LanguageRegistry::new();
        for (path, language) in [("a.cjs", "javascript"), ("a.cts", "typescript")] {
            let extractor = registry.get(path).map(|e| e.language_id().to_string());
            assert_eq!(extractor.as_deref(), Some(language), "{path}");
        }
    }

    #[test]
    fn a_member_line_that_does_not_fit_u32_leaves_the_block_unsplit() {
        // [MAGI S6 loop 9, B2b-c1] a `u32::MAX` fallback offset and saturating additions gave
        // members fabricated ranges that no longer partitioned the block.
        let member = |start, end| Construct {
            kind: UnitKind::Function,
            name: "a".to_string(),
            start_line: start,
            end_line: end,
        };
        assert_eq!(
            shift_members(vec![member(1, 2)], 2),
            Some(vec![(UnitKind::Function, "a".to_string(), 4, 5)])
        );
        let max = usize::try_from(u32::MAX).expect("usize holds u32");
        assert_eq!(shift_members(vec![member(1, 2)], max - 1), None);
        assert_eq!(shift_members(vec![member(1, 2)], max), None);
        assert_eq!(shift_members(vec![member(1, 2)], usize::MAX), None);
    }

    #[test]
    fn construct_ranges_are_sorted_and_disjoint() {
        // [MAGI S6 loop 7, B2a] `ChangedLines::assign` gives the same units for a diff and a
        // whole file only when the ranges are sorted and disjoint; this pins it per extractor.
        let corpus = [
            (rust_extractor(), "use a::b;\n\n/// Doc.\nimpl Foo {\n    fn a() {}\n}\nconst N: u8 = 1;\nfn f(\n    x: u8,\n) {\n}\nmod m {\n    fn g() {}\n}\n"),
            (c_extractor(), "#define MAX(a, b) \\n    ((a) > (b))\nstruct Point make_point(void) {\n    return p;\n}\nstruct S {\n    int a;\n};\nint g;\nvoid foo(\n    int (*cb)(int)\n) {\n}\n"),
            (BraceFamilyExtractor::cpp(), "namespace n {\nclass A {\n    void f();\n};\n}\nenum class E {\n    X,\n};\nstruct Foo x{1};\n"),
            (BraceFamilyExtractor::go(), "import \"fmt\"\n\ntype (\n\tA struct {\n\t\tX int\n\t}\n\tB int\n)\nfunc f() {\n}\ntype C struct {\n}\n"),
            (BraceFamilyExtractor::java(), "package p;\n\npublic class A {\n    /** Doc. */\n    void run() {\n    }\n}\ninterface I {\n    void f();\n}\n"),
            (BraceFamilyExtractor::javascript(), "import x from 'y';\nconst f = () => {\n};\nclass G {\n  *items() {}\n}\nexport function h() {\n}\n"),
            (BraceFamilyExtractor::typescript(), "type T = number;\ninterface I {\n  f(): void;\n}\nexport class C {\n  run() {}\n}\n"),
            // [MAGI S6 loop 12, B2b-c1] a one-line construct that opens with an attribute prefix
            // was taken by the scan back of the construct after it.
            (BraceFamilyExtractor::java(), "@Override public String toString() { return \"x\"; }\npublic void g() {\n    int y = 1;\n}\n"),
            (BraceFamilyExtractor::cpp(), "template<typename T> void f() { return; }\nvoid g() {\n    x();\n}\n"),
        ];
        for (ext, code) in corpus {
            let lines: Vec<&str> = code.lines().collect();
            let scanned = CodeScanner::new(ext.lexical_rules.clone()).scan_all(code);
            let constructs = ext.detect_constructs(&lines, &scanned, false);
            assert!(!constructs.is_empty(), "{code}");
            for c in &constructs {
                assert!(c.start_line <= c.end_line, "{}: {code}", c.name);
            }
            for pair in constructs.windows(2) {
                assert!(
                    pair[0].end_line < pair[1].start_line,
                    "{} overlaps {}: {code}",
                    pair[0].name,
                    pair[1].name
                );
            }
        }
    }

    #[test]
    fn a_block_without_members_gives_no_pieces_so_the_caller_keeps_it_whole() {
        // [MAGI S6 loop 7, B2a] an empty split means "do not split": the one caller,
        // `review::batching`, keeps the unit whole when `split_block` returns nothing, so no line
        // of a memberless block is dropped.
        for (ext, code, kind) in [
            (rust_extractor(), "impl Foo {}\n", UnitKind::Impl),
            (
                BraceFamilyExtractor::java(),
                "class A {\n    int x;\n}\n",
                UnitKind::Class,
            ),
        ] {
            assert!(split(&ext, code, 1, kind).is_empty(), "{code}");
        }
    }

    #[test]
    fn a_multi_line_rust_attribute_belongs_to_its_item() {
        // [MAGI S6 loop 10, B2b-m1] the line above the item was `)]`, so the scan back stopped
        // there and a change to a `serde` or `cfg` attribute was reviewed without its item.
        let code = "fn before() {\n}\n#[derive(Debug)]\n#[serde(\n    rename = \"x\",\n    default\n)]\npub struct Config {\n    pub x: u32,\n}\n#[cfg(all(\n    test,\n    unix\n))]\nfn only_unix() {\n    a();\n}\nlist! [\n    1\n]\nfn after() {\n    b();\n}\n";
        let units = extract(&rust_extractor(), code, &[5, 9, 12, 16, 19, 22]);
        let found = spans(&units);
        for expected in [
            ("Config", UnitKind::Struct, 3, 10),
            ("only_unix", UnitKind::Function, 11, 17),
            ("after", UnitKind::Function, 21, 23),
        ] {
            assert!(found.contains(&expected), "{expected:?} in {found:?}");
        }
    }

    #[test]
    fn a_java_annotation_on_the_declaration_line_is_part_of_the_declaration() {
        // [MAGI S6 loop 10, B2b-m2] a line opening with `@` was never a signature, so
        // `@Override public void run() {` was no method and an oversized class did not split there.
        let java = BraceFamilyExtractor::java();
        let code = "@Deprecated public class A {\n    @Override public void run() {\n        a();\n    }\n    @SuppressWarnings(\"x\") private int f() {\n        return 0;\n    }\n    @Override\n    public void g() {\n        b();\n    }\n}\n";
        let parts = split(&java, code, 3, UnitKind::Class);
        let found = spans(&parts);
        for expected in [
            ("run", UnitKind::Method, 2, 4),
            ("f", UnitKind::Method, 5, 7),
            ("g", UnitKind::Method, 8, 11),
        ] {
            assert!(found.contains(&expected), "{expected:?} in {found:?}");
        }
        assert_each_line_in_one_unit(&parts, 12);
        let top = "@Override public void run() {\n    a();\n}\n@interface Marker {\n}\n";
        assert_eq!(
            spans(&extract(&java, top, &[2, 4])),
            vec![
                ("run", UnitKind::Function, 1, 3),
                ("Marker", UnitKind::Class, 4, 5)
            ]
        );
    }

    #[test]
    fn a_declaration_group_splits_at_its_declarations() {
        // [MAGI S6 loop 12, B2b-b1] a group was never split, so a generated header of thousands
        // of `#define`s was one unit over the batch limit and the whole review was refused.
        let c = c_extractor();
        let code = "#define A 1\n#define B 2\n\nint c;\nint d,\n    e;\n";
        assert_eq!(
            spans(&extract(&c, code, &[1])),
            vec![("defines", UnitKind::DeclarationGroup, 1, 6)]
        );
        let parts = split(&c, code, 1, UnitKind::DeclarationGroup);
        let ranges: Vec<(UnitKind, u32, u32)> = parts
            .iter()
            .map(|u| (u.kind, u.start_line, u.end_line))
            .collect();
        assert_eq!(
            ranges,
            vec![
                (UnitKind::DeclarationGroup, 1, 1),
                (UnitKind::DeclarationGroup, 2, 2),
                (UnitKind::DeclarationGroup, 3, 3),
                (UnitKind::DeclarationGroup, 4, 4),
                (UnitKind::DeclarationGroup, 5, 6),
            ]
        );
        assert_each_line_in_one_unit(&parts, 6);
        let rust = "/// A.\nconst A: u8 = 1;\nconst B: S = S {\n    a: 1,\n};\n";
        let parts = split(&rust_extractor(), rust, 2, UnitKind::DeclarationGroup);
        assert_eq!(
            spans(&parts).iter().map(|s| (s.2, s.3)).collect::<Vec<_>>(),
            vec![(1, 2), (3, 5)],
            "a declaration keeps its doc comment"
        );
    }

    #[test]
    fn the_brace_search_of_one_signature_stops_after_a_bounded_number_of_lines() {
        // [MAGI S6 loop 12, B2b-c2] inside an open parameter list the search ran to the end of
        // the file, so a file of `f(a` lines took quadratic time before the review clock started.
        let ext = BraceFamilyExtractor::c();
        let search = |params: usize| {
            let code = format!("void f(\n{}) {{\n}}\n", "    a,\n".repeat(params));
            let lines: Vec<&str> = code.lines().collect();
            let scanned = CodeScanner::new(ext.lexical_rules.clone()).scan_all(&code);
            open_brace_line(
                &lines,
                &scanned,
                0,
                &BraceSearch {
                    stops: &|_| false,
                    budget: &SearchBudget::for_lines(lines.len()),
                },
            )
        };
        assert_eq!(
            search(64),
            Some(65),
            "a long parameter list still finds its brace"
        );
        assert_eq!(search(100_000), None, "the search degrades to no brace");
    }

    #[test]
    fn a_multi_line_annotation_belongs_to_its_member() {
        // [MAGI S6 loop 12, B2b-m1] the line above the member was `)` or `})`, so the scan back
        // stopped there and the annotation that governs the member landed in a gap unit.
        let java = BraceFamilyExtractor::java();
        let code = "public class A {\n    void before() {\n    }\n    @SuppressWarnings(\n        \"unchecked\"\n    )\n    public void f() {\n        a();\n    }\n}\n";
        let parts = split(&java, code, 8, UnitKind::Class);
        let found = spans(&parts);
        assert!(
            found.contains(&("f", UnitKind::Method, 4, 9)),
            "f with its annotation in {found:?}"
        );
        assert_each_line_in_one_unit(&parts, 10);
        let ts = BraceFamilyExtractor::typescript();
        let code = "import x from 'y';\n\n@Component({\n  selector: 'app',\n})\nexport class C {\n  run() {}\n}\n";
        let units = extract(&ts, code, &[4, 7]);
        let found = spans(&units);
        assert!(
            found.contains(&("C", UnitKind::Class, 3, 8)),
            "C with its decorator in {found:?}"
        );
    }

    #[test]
    fn a_line_number_that_does_not_fit_u32_is_none_not_a_fallback() {
        // [MAGI S6 loop 10, B2b-b2] construct detection fell back to `u32::MAX`, `1` or the
        // current line where `shift_members` aborted; both now abort on the same check.
        assert_eq!(line_number(0), Some(1));
        let max = usize::try_from(u32::MAX).expect("usize holds u32");
        assert_eq!(line_number(max - 1), Some(u32::MAX));
        assert_eq!(line_number(max), None);
        assert_eq!(line_number(usize::MAX), None);
    }

    #[test]
    fn any_blank_separates_a_keyword_from_the_name_it_declares() {
        // [MAGI S6 loop 13, B2b1-c4] the keyword tables matched with one literal space, so a tab
        // after the keyword (`fn\tfoo`, `const\tX`) left the construct in a gap unit.
        let rust = "fn\tfoo() {\n}\npub\tstruct\tS {\n    a: u8,\n}\nconst\tX: u8 = 1;\n";
        assert_eq!(
            spans(&extract(&rust_extractor(), rust, &all_lines(rust))),
            vec![
                ("foo", UnitKind::Function, 1, 2),
                ("S", UnitKind::Struct, 3, 5),
                ("constants", UnitKind::DeclarationGroup, 6, 6)
            ]
        );
        let js = "const\tf = () => {\n  a();\n};\n";
        let ext = BraceFamilyExtractor::javascript();
        assert_eq!(
            spans(&extract(&ext, js, &all_lines(js))),
            vec![("f", UnitKind::Function, 1, 3)]
        );
        let go = "func\tmain() {\n}\n";
        let ext = BraceFamilyExtractor::go();
        assert_eq!(
            spans(&extract(&ext, go, &all_lines(go))),
            vec![("main", UnitKind::Function, 1, 2)]
        );
    }

    #[test]
    fn a_c_attribute_with_parentheses_is_no_function_and_no_name() {
        // [MAGI S6 loop 13, B2b1-c1] an attribute's parentheses (`__attribute__((packed))`,
        // `alignas(16)`) were read as a call: the struct was named `__attribute__`, and a
        // function or struct opening with one was a function named after the attribute.
        let c = "struct __attribute__((packed)) Hdr {\n    int a;\n};\n__attribute__((noreturn)) void die(void) {\n}\n";
        assert_eq!(
            spans(&extract(&c_extractor(), c, &all_lines(c))),
            vec![
                ("Hdr", UnitKind::Struct, 1, 3),
                ("die", UnitKind::Function, 4, 5)
            ]
        );
        let cpp = "class [[deprecated]] W {\n};\n[[nodiscard]] int f(int a) {\n    return a;\n}\nalignas(16) struct V {\n    float x;\n};\nstruct alignas(8) B {\n};\n__declspec(dllexport) int g(void) {\n}\n";
        let ext = BraceFamilyExtractor::cpp();
        assert_eq!(
            spans(&extract(&ext, cpp, &all_lines(cpp))),
            vec![
                ("W", UnitKind::Class, 1, 2),
                ("f", UnitKind::Function, 3, 5),
                ("V", UnitKind::Struct, 6, 8),
                ("B", UnitKind::Struct, 9, 10),
                ("g", UnitKind::Function, 11, 12)
            ]
        );
    }

    #[test]
    fn a_nested_template_argument_closing_in_a_double_angle_ends_its_header() {
        // [MAGI S6 loop 13, B2b1-c2] declared green, a false positive: each `>` of a `>>` closes
        // one level of the list in `skip_generics`, so `std::vector<int>>` closes both the inner
        // and the outer list and the construct after the header is found.
        assert_eq!(
            skip_generics("<typename T = std::vector<int>> class Box {"),
            "class Box {"
        );
        let cpp = "template<typename T = std::vector<std::pair<int, int>>> class Box {\n    T t;\n};\ntemplate <typename T = std::vector<int>>\nstruct S {\n};\n";
        let ext = BraceFamilyExtractor::cpp();
        assert_eq!(
            spans(&extract(&ext, cpp, &all_lines(cpp))),
            vec![
                ("Box", UnitKind::Class, 1, 3),
                ("S", UnitKind::Struct, 4, 6)
            ]
        );
    }

    #[test]
    fn a_function_expression_whose_body_opens_on_the_next_line_keeps_it() {
        // [MAGI S6 loop 13, B2b1-c3] a declaration continued only when its line ended in `=`,
        // `(`, `[` or `,`, so `const f = () =>` or `const g = function (x)` with the brace on
        // the next line ended on its own line and the body fell into a gap unit.
        let js = "const f = () =>\n{\n  a();\n};\nconst g = async function (x)\n{\n  return x;\n};\nconst h = g(1)\nconst k = 2;\n";
        let ext = BraceFamilyExtractor::javascript();
        assert_eq!(
            spans(&extract(&ext, js, &all_lines(js))),
            vec![
                ("f", UnitKind::Function, 1, 4),
                ("g", UnitKind::Function, 5, 8),
                ("constants", UnitKind::DeclarationGroup, 9, 10)
            ]
        );
        let ts = "const k = (): void =>\n{\n};\n";
        let ext = BraceFamilyExtractor::typescript();
        assert_eq!(
            spans(&extract(&ext, ts, &all_lines(ts))),
            vec![("k", UnitKind::Function, 1, 3)]
        );
    }

    #[test]
    fn a_standalone_go_interface_is_a_trait() {
        // [MAGI S6 loop 13, B2b1-c6] the `type` keyword named a struct unconditionally, so
        // `type Reader interface {` was a struct, unlike the same interface inside `type (`.
        let go = "type Reader interface {\n\tRead() error\n}\ntype Box[T any] interface {\n\tGet() T\n}\ntype P struct {\n\tX int\n}\n";
        let ext = BraceFamilyExtractor::go();
        assert_eq!(
            spans(&extract(&ext, go, &all_lines(go))),
            vec![
                ("Reader", UnitKind::Trait, 1, 3),
                ("Box", UnitKind::Trait, 4, 6),
                ("P", UnitKind::Struct, 7, 9)
            ]
        );
    }

    #[test]
    fn a_block_comment_line_opening_with_a_statement_keyword_keeps_the_body_with_its_signature() {
        // [MAGI S6 loop 25, B2b1b] the brace search read a block comment's line as code, so a
        // comment line opening with `return` or `if` between a signature and its brace ended the
        // search and the body was cut from its signature.
        let c = "int f(int a)\n/* note\n   return early */\n{\n    return a;\n}\n";
        assert_eq!(
            sorted_spans(&c_extractor(), c),
            vec![("f".to_string(), UnitKind::Function, 1, 6)]
        );
        let rust = "fn g()\n/* see\n   if needed */\n{\n    x();\n}\n";
        assert_eq!(
            sorted_spans(&rust_extractor(), rust),
            vec![("g".to_string(), UnitKind::Function, 1, 6)]
        );
    }

    #[test]
    fn a_comment_line_ending_in_a_semicolon_joins_no_declaration_group() {
        // [MAGI S6 loop 13, B2b1-m1] the group's forward scan asked the line text alone, so a
        // block comment's body line ending in `;` (`note;`) was taken for a declaration and
        // joined the group, unlike the main scan, which passes comment lines over.
        let c = "int x,\n    y; /* c\nnote;\n*/\nint f(void) {\n}\n";
        let units = extract(&c_extractor(), c, &all_lines(c));
        let found = spans(&units);
        assert!(
            found.contains(&("declarations", UnitKind::DeclarationGroup, 1, 2)),
            "{found:?}"
        );
        assert!(
            found.contains(&("f", UnitKind::Function, 5, 6)),
            "{found:?}"
        );
        assert_each_line_in_one_unit(&units, 6);
    }

    #[test]
    fn a_c_initializer_wrapped_after_a_calls_parenthesis_is_one_declaration() {
        // [MAGI S6 loop 13, B2b1-m2] only a trailing `=` or `= {` opened an initializer, so
        // `static const struct config c = make_config(` was no declaration: its header and
        // argument lines fell into a gap unit and its `);` became a declaration of its own.
        let c =
            "static const struct config c = make_config(\n    DEFAULT_A\n);\nint f(void) {\n}\n";
        assert_eq!(
            spans(&extract(&c_extractor(), c, &all_lines(c))),
            vec![
                ("statics", UnitKind::DeclarationGroup, 1, 3),
                ("f", UnitKind::Function, 4, 5)
            ]
        );
    }

    #[test]
    fn the_brace_searches_of_a_whole_file_spend_a_budget_linear_in_its_lines() {
        // [MAGI S6 loop 13, B2b1-c5] each search was bounded, but a file of `f(` lines started
        // one per line, O(n × MAX_BRACE_SEARCH_LINES) before the review clock; the searches of a
        // pass now share one budget, and a construct whose brace is on its own line is still
        // found once it is spent.
        let lines_of_open_lists = 5_000;
        let code = format!(
            "{}\nint g(void) {{\n}}\n",
            "f(\n".repeat(lines_of_open_lists)
        );
        let ext = c_extractor();
        let lines: Vec<&str> = code.lines().collect();
        let scanned = CodeScanner::new(ext.lexical_rules.clone()).scan_all(&code);
        let budget = SearchBudget::for_lines(lines.len());
        let constructs = ext
            .try_detect_constructs(&lines, &scanned, false, &budget)
            .expect("small line numbers");
        // Linear in the lines, with room: an oracle of this test, not the production budget.
        let linear_bound_per_line = 64;
        let cap = lines.len() * linear_bound_per_line + MAX_BRACE_SEARCH_LINES;
        assert!(
            budget.spent() <= cap,
            "{} visits for {} lines",
            budget.spent(),
            lines.len()
        );
        let g = constructs.iter().find(|c| c.name == "g").expect("g");
        assert_eq!(
            (g.kind, g.start_line, g.end_line),
            (UnitKind::Function, 5_002, 5_003)
        );
        let total = u32::try_from(lines.len()).expect("small");
        let units = extract(&ext, &code, &(1..=total).collect::<Vec<_>>());
        assert_each_line_in_one_unit(&units, total);
    }

    #[test]
    fn the_walk_up_from_a_wrapped_attribute_tail_spends_the_pass_budget() {
        // [MAGI loop 28] a line opening with a closer and holding a construct after it
        // (`) struct S {`) walks up to the attribute that opens it; a file of such lines with no
        // attribute would walk to the floor from each, O(n²), unless every line walked is
        // charged to the pass's budget, which is linear in the lines ([`SearchBudget`]).
        let code = "x(\n    a,\n    b) struct S {\n";
        let ext = c_extractor();
        let lines: Vec<&str> = code.lines().collect();
        let scanned = CodeScanner::new(ext.lexical_rules.clone()).scan_all(code);
        let budget = SearchBudget::for_lines(lines.len());
        let stops = |j: usize| ext.starts_construct(&lines, &scanned, j, false);
        let search = BraceSearch {
            stops: &stops,
            budget: &budget,
        };
        let found = ext.after_wrapped_attribute_tail(&lines, &scanned, 2, 0, &search, |rest| {
            ext.construct_keyword(rest)
        });
        assert!(found.is_none(), "`x(` opens no attribute");
        assert_eq!(budget.spent(), 2, "each line walked is charged");
    }

    #[test]
    fn a_wrapped_signature_spends_its_brace_search_once() {
        // [MAGI S6 loop 15, B2b1] The signature path searched for the brace to know whether one
        // opened, then searched again to find the block's end, so every wrapped signature spent
        // its header's visits twice and a file reached the budget's degradation sooner.
        let code = "int f(\n    int a,\n    int b)\n{\n    x();\n}\n";
        let ext = c_extractor();
        let lines: Vec<&str> = code.lines().collect();
        let scanned = CodeScanner::new(ext.lexical_rules.clone()).scan_all(code);
        let one_search = SearchBudget::for_lines(lines.len());
        let stops = |j: usize| ext.starts_construct(&lines, &scanned, j, false);
        let search = BraceSearch {
            stops: &stops,
            budget: &one_search,
        };
        assert_eq!(open_brace_line(&lines, &scanned, 0, &search), Some(3));
        let pass = SearchBudget::for_lines(lines.len());
        let constructs = ext
            .try_detect_constructs(&lines, &scanned, false, &pass)
            .expect("small line numbers");
        assert_eq!(
            constructs
                .iter()
                .map(|c| (c.name.as_str(), c.start_line, c.end_line))
                .collect::<Vec<_>>(),
            vec![("f", 1, 6)]
        );
        assert_eq!(pass.spent(), one_search.spent());
    }

    #[test]
    fn constructs_pair_with_their_shares_only_one_to_one() {
        // [MAGI S6 loop 15, B2b1] `extract_units` paired constructs and shares with `zip`, which
        // stops at the shorter side: a share list one short would have dropped the last
        // construct's changed lines from every unit, silently.
        let construct = |name: &str, start, end| Construct {
            kind: UnitKind::Function,
            name: name.to_string(),
            start_line: start,
            end_line: end,
        };
        let constructs = vec![construct("a", 1, 3), construct("b", 5, 7)];
        let pairs = construct_shares(&constructs, vec![vec![2], vec![]]).expect("one to one");
        assert_eq!(
            pairs
                .iter()
                .map(|(c, share)| (c.name.as_str(), share.clone()))
                .collect::<Vec<_>>(),
            vec![("a", vec![2]), ("b", vec![])]
        );
        assert!(construct_shares(&constructs, vec![vec![2]]).is_none());
        assert!(construct_shares(&constructs, vec![vec![], vec![], vec![6]]).is_none());
        assert!(construct_shares(&[], Vec::new()).is_some_and(|p| p.is_empty()));
        // `assign` keeps the contract for both of its forms, an empty set included.
        let ranges = [(1, 3), (5, 7)];
        let only: HashSet<u32> = [2, 4].into_iter().collect();
        for changed in [ChangedLines::All, ChangedLines::from(&only)] {
            assert_eq!(changed.assign(&ranges, 8).0.len(), ranges.len());
        }
        let none: HashSet<u32> = HashSet::new();
        assert_eq!(ChangedLines::from(&none).assign(&ranges, 8).0.len(), 2);
    }

    /// Every line number of `code`, 1-based.
    fn every_line(code: &str) -> Vec<u32> {
        (1..=u32::try_from(code.lines().count()).expect("small")).collect()
    }

    #[test]
    fn a_raw_string_in_a_rust_attribute_hides_its_quotes_and_brackets() {
        // [MAGI S6 loop 25, B2b1a] the attribute's brackets were balanced with `"` as the only
        // quote, so a raw string holding a `"` (`r#"a"b"#`) left the attribute unclosed and the
        // item after it on the line was no construct.
        let code = "#[doc = r#\"a\"b]\"#] fn f() {\n    x();\n}\n#[doc = r#\"a\"b\"#] const X: u8 = 1;\nfn g() {\n    y();\n}\n";
        assert_eq!(
            sorted_spans(&rust_extractor(), code),
            vec![
                ("f".to_string(), UnitKind::Function, 1, 3),
                // A group is named by its first line as written, attribute included.
                ("declarations".to_string(), UnitKind::DeclarationGroup, 4, 4),
                ("g".to_string(), UnitKind::Function, 5, 7),
            ]
        );
    }

    #[test]
    fn a_rust_attribute_ends_past_its_strings_and_raw_strings_only() {
        // The bounds of `rust_attribute_end`: strings and raw strings hide their delimiters, an
        // `r` inside a word opens no raw string, and an unclosed string closes nothing.
        fn end(t: &str) -> Option<&str> {
            rust_attribute_end(t).and_then(|e| t.get(..e))
        }
        assert_eq!(end("[doc = \"]\"] fn f"), Some("[doc = \"]\"]"));
        assert_eq!(end("[doc = r\"]\"] x"), Some("[doc = r\"]\"]"));
        assert_eq!(
            end("[doc = r##\"a\"#]\"##] x"),
            Some("[doc = r##\"a\"#]\"##]")
        );
        assert_eq!(end("[doc = br#\"]\"#] x"), Some("[doc = br#\"]\"#]"));
        // A raw string would end at the escaped quote; a plain one runs past it.
        assert_eq!(end("[cfg(for\"\\\"]\")] x"), Some("[cfg(for\"\\\"]\")]"));
        assert_eq!(end("[doc = \"a\\\"]\"] x"), Some("[doc = \"a\\\"]\"]"));
        assert_eq!(
            end("[cfg(feature = \"x\")][y]"),
            Some("[cfg(feature = \"x\")]")
        );
        assert_eq!(end("[doc = r#\"a]"), None);
        assert_eq!(end("[doc = \"a]"), None);
        assert_eq!(end("[inline"), None);
    }

    #[test]
    fn a_rust_item_after_an_attribute_on_its_own_line_is_a_unit_with_its_body() {
        // [MAGI S6 loop 14, B2b1] `construct_keyword` never read past a leading `#[...]`, so
        // `#[inline] fn f() {` and `#[derive(Debug)] struct S {` were no constructs and their
        // bodies fell into gap units (RV03).
        let code = "#[inline] fn f() {
    x();
}
#[derive(Debug)] pub struct S {
    n: u8,
}
#[cfg(test)] #[allow(dead_code)] mod tests {
    fn g() {
    }
}
#[doc = \"]\"] fn h() {
}
";
        let units = extract(&rust_extractor(), code, &every_line(code));
        assert_each_line_in_one_unit(&units, 12);
        assert_eq!(
            spans(&units),
            vec![
                ("f", UnitKind::Function, 1, 3),
                ("S", UnitKind::Struct, 4, 6),
                ("tests", UnitKind::Module, 7, 10),
                ("h", UnitKind::Function, 11, 12),
            ]
        );
    }

    #[test]
    fn a_parenthesis_inside_an_annotation_string_does_not_hide_the_declaration() {
        // [MAGI S6 loop 14, B2b1] `without_annotations` counted the parentheses inside the
        // annotation's strings, so `@Component(")")` closed early and the class was no unit.
        let java = "@Component(\")\") public class C {\n    void run() {\n    }\n}\n";
        let units = extract(&BraceFamilyExtractor::java(), java, &every_line(java));
        assert_eq!(spans(&units), vec![("C", UnitKind::Class, 1, 4)]);
        let ts = "@Component(')', `(`) export class D {\n    run() {\n    }\n}\n";
        let units = extract(&BraceFamilyExtractor::typescript(), ts, &every_line(ts));
        assert_eq!(spans(&units), vec![("D", UnitKind::Class, 1, 4)]);
        // An escaped quote stays inside its string.
        let escaped = "@A(\"\\\")\") class E {\n}\n";
        let units = extract(&BraceFamilyExtractor::java(), escaped, &every_line(escaped));
        assert_eq!(spans(&units), vec![("E", UnitKind::Class, 1, 2)]);
    }

    #[test]
    fn extra_blanks_after_a_keyword_do_not_change_the_construct_or_its_name() {
        // [MAGI S6 loop 14, B2b1] `blank_keyword_rest` returned the rest untrimmed when the
        // keyword matched as written, so `const  f = () => {` named an empty function and was
        // no function expression, and `enum  class Color` was named `class`.
        let js = "const  f = () => {\n  x();\n};\n";
        let units = extract(&BraceFamilyExtractor::javascript(), js, &every_line(js));
        assert_eq!(spans(&units), vec![("f", UnitKind::Function, 1, 3)]);
        let cpp = "enum  class Color {\n    Red,\n};\n";
        let units = extract(&BraceFamilyExtractor::cpp(), cpp, &every_line(cpp));
        assert_eq!(spans(&units), vec![("Color", UnitKind::Enum, 1, 3)]);
        assert_eq!(blank_keyword_rest("fn  foo()", "fn "), Some("foo()"));
        assert_eq!(blank_keyword_rest("fn\t foo()", "fn "), Some("foo()"));
        assert_eq!(blank_keyword_rest("fnord()", "fn "), None);
    }

    #[test]
    fn a_cpp_allocation_operator_overload_is_a_function_with_its_body() {
        // [MAGI S6 loop 14, B2b1] `operator_name` knew only symbol operators, so the word `new`
        // or `delete` after `operator` reached `NOT_FUNCTION_NAMES` and the allocator's body fell
        // into a gap unit.
        let code = "void* operator new(size_t size) {\n    return malloc(size);\n}\nvoid operator delete[](void* p) noexcept {\n    free(p);\n}\nvoid operator delete (void* p) {\n    free(p);\n}\nvoid* operator new [] (size_t n) {\n    return malloc(n);\n}\n";
        let units = extract(&BraceFamilyExtractor::cpp(), code, &every_line(code));
        assert_eq!(
            spans(&units),
            vec![
                ("operator new", UnitKind::Function, 1, 3),
                ("operator delete[]", UnitKind::Function, 4, 6),
                ("operator delete", UnitKind::Function, 7, 9),
                ("operator new[]", UnitKind::Function, 10, 12),
            ]
        );
        // A word glued to `operator` is an identifier, not an overload.
        assert_eq!(operator_name("void operatornew"), None);
        assert_eq!(operator_name("int my_operator new"), None);
    }

    #[test]
    fn a_cpp_function_returning_decltype_auto_is_a_function_with_its_body() {
        // [MAGI S6 loop 15, B2b1] The first `(` was `decltype`'s, so the name read was
        // `decltype`, a word of `NOT_FUNCTION_NAMES`, and the body fell into a gap unit.
        let code = "decltype(auto) f() {\n    return x;\n}\nstatic decltype(a + b) g(int a, int b) {\n    return a + b;\n}\nauto add(int a, int b)\n    -> decltype(a + b)\n{\n    return a + b;\n}\n";
        let units = extract(&BraceFamilyExtractor::cpp(), code, &every_line(code));
        assert_eq!(
            spans(&units),
            vec![
                ("f", UnitKind::Function, 1, 3),
                ("g", UnitKind::Function, 4, 6),
                ("add", UnitKind::Function, 7, 11),
            ]
        );
        // A `decltype` alone, with no function after it, names nothing.
        assert_eq!(identifier_before_paren("decltype(x)"), None);
    }

    #[test]
    fn a_return_type_holding_parentheses_does_not_name_the_function() {
        // [MAGI S6 loop 15, B2b1] The name was read before the line's first `(`, so a template
        // argument (`std::function<void(int)>`) or a declarator in parentheses (a function
        // returning a function pointer) named the function `void`; the symbol check (RV09)
        // then missed the real name.
        let code = "std::function<void(int)> make(int n) {\n    return f;\n}\nvoid (*get_handler(int sig))(int) {\n    return h;\n}\nvoid (*f)() {\n    x();\n}\nstd::map<int, std::pair<int, int>> table() {\n    return t;\n}\n";
        let units = extract(&BraceFamilyExtractor::cpp(), code, &every_line(code));
        assert_eq!(
            spans(&units),
            vec![
                ("make", UnitKind::Function, 1, 3),
                ("get_handler", UnitKind::Function, 4, 6),
                ("f", UnitKind::Function, 7, 9),
                ("table", UnitKind::Function, 10, 12),
            ]
        );
        // A statement's parenthesized operand is no declarator: it names the keyword, which
        // `NOT_FUNCTION_NAMES` rules out, and a call is named by its callee.
        assert_eq!(identifier_before_paren("if (*p) {"), Some("if".to_string()));
        assert_eq!(
            identifier_before_paren("while (*s)(x) {"),
            Some("while".to_string())
        );
        assert_eq!(
            identifier_before_paren("run(*p) {"),
            Some("run".to_string())
        );
        // A `<` with no matching `>` before the `(` (a comparison) keeps the first `(`.
        assert_eq!(identifier_before_paren("a < f(x) {"), Some("f".to_string()));
    }

    #[test]
    fn a_declaration_keyword_glued_to_its_list_opens_a_declaration_group() {
        // [MAGI S6 loop 14, B2b1] `match_declaration_keyword` required a blank after the
        // keyword, so `use::{a, b};`, `import{foo} from "x";` and `#include<stdio.h>` were no
        // declarations and fell into gap units.
        let rust = "use::{a, b};\nfn f() {\n}\n";
        let units = extract(&rust_extractor(), rust, &[1]);
        assert_eq!(
            spans(&units),
            vec![("imports", UnitKind::DeclarationGroup, 1, 1)]
        );
        let js = "import{foo} from \"x\";\nexport{foo};\nimport*as m from 'm';\n";
        let units = extract(&BraceFamilyExtractor::javascript(), js, &every_line(js));
        assert_eq!(
            spans(&units),
            vec![("imports", UnitKind::DeclarationGroup, 1, 3)]
        );
        let c = "#include<stdio.h>\n#include\"a.h\"\n";
        let units = extract(&c_extractor(), c, &every_line(c));
        assert_eq!(
            spans(&units),
            vec![("declarations", UnitKind::DeclarationGroup, 1, 2)]
        );
        // A longer word that starts like a keyword is no keyword.
        let js = "usefoo();\nimporter();\n";
        let units = extract(&BraceFamilyExtractor::javascript(), js, &every_line(js));
        assert_eq!(spans(&units), vec![("lines 1-2", UnitKind::TopLevel, 1, 2)]);
    }

    #[test]
    fn a_variable_of_an_anonymous_type_is_one_unit_with_its_declarator() {
        // [MAGI S6 loop 14, B2b1] reported: `struct { int a; } x;` reads as a type, so its
        // trailing `x;` falls into a gap unit. The closing line holds the declarator, and the
        // block ends on that line, so the declaration stays whole.
        let c = "struct { int a; } x;\nstruct {\n    int b;\n} y, *z;\nunion\n{\n    int c;\n} w = {0};\n";
        let units = extract(&c_extractor(), c, &every_line(c));
        assert_each_line_in_one_unit(&units, 8);
        let ranges: Vec<(u32, u32)> = units.iter().map(|u| (u.start_line, u.end_line)).collect();
        assert_eq!(ranges, vec![(1, 1), (2, 4), (5, 8)], "{:?}", spans(&units));
        let cpp = "class {\n    int d;\n} v;\n";
        let units = extract(&BraceFamilyExtractor::cpp(), cpp, &every_line(cpp));
        let ranges: Vec<(u32, u32)> = units.iter().map(|u| (u.start_line, u.end_line)).collect();
        assert_eq!(ranges, vec![(1, 3)], "{:?}", spans(&units));
    }

    #[test]
    fn a_keyword_alone_on_its_line_with_no_brace_after_it_takes_no_later_construct() {
        // [MAGI S6 loop 14, B2b1] the bound of the bare-keyword rule: with no brace before a
        // blank line or the next construct, the keyword's line is a unit of its own.
        let c = "struct\n\nint f(void) {\n}\nunion\nint g(void) {\n}\n";
        let units = extract(&c_extractor(), c, &every_line(c));
        assert_each_line_in_one_unit(&units, 7);
        let ranges: Vec<(u32, u32)> = units.iter().map(|u| (u.start_line, u.end_line)).collect();
        assert_eq!(
            ranges,
            vec![(1, 1), (3, 4), (5, 5), (6, 7), (2, 2)],
            "{:?}",
            spans(&units)
        );
        let rust = "impl";
        let units = extract(&rust_extractor(), rust, &[1]);
        assert_eq!(spans(&units), vec![("unnamed", UnitKind::Impl, 1, 1)]);
    }

    #[test]
    fn a_signature_without_a_body_wrapped_over_lines_ends_at_its_semicolon() {
        // [MAGI S6 loop 14, B2b1] with no brace found, a construct ended on its keyword's line,
        // so a wrapped trait method or foreign function lost the rest of its signature to the
        // next unit (RV03).
        let top = "fn f(\n    a: u8,\n) -> u8;\nfn g() {\n}\n";
        let units = extract(&rust_extractor(), top, &every_line(top));
        assert_eq!(
            spans(&units),
            vec![
                ("f", UnitKind::Function, 1, 3),
                ("g", UnitKind::Function, 4, 5)
            ]
        );
        let tr = "trait T {\n    fn f<U>(\n        a: U,\n    ) -> U\n    where\n        U: Clone;\n    fn g(&self) {\n    }\n}\n";
        let parts = split(&rust_extractor(), tr, 2, UnitKind::Trait);
        let ranges: Vec<(&str, u32, u32)> = parts
            .iter()
            .map(|p| (p.name.as_str(), p.start_line, p.end_line))
            .collect();
        assert!(ranges.contains(&("f", 2, 6)), "{ranges:?}");
        assert!(ranges.contains(&("g", 7, 8)), "{ranges:?}");
    }

    #[test]
    fn a_blank_line_inside_a_wrapped_parameter_list_keeps_the_body_with_its_signature() {
        // [MAGI S6 loop 14, B2b1] the brace search stopped at any blank line, so a blank line
        // between two parameters detached the body from its signature.
        let rust = "fn h(\n    a: u8,\n\n    b: u8,\n) -> u8 {\n    a\n}\n";
        let units = extract(&rust_extractor(), rust, &every_line(rust));
        assert_eq!(spans(&units), vec![("h", UnitKind::Function, 1, 7)]);
        let c = "int k(int a,\n\n      int b)\n{\n    return a;\n}\n";
        let units = extract(&c_extractor(), c, &every_line(c));
        assert_eq!(spans(&units), vec![("k", UnitKind::Function, 1, 6)]);
    }

    #[test]
    fn a_blank_line_after_a_line_that_closes_its_code_still_ends_the_brace_search() {
        // [MAGI S6 loop 14, B2b1] the bound of the rule above: a `(` inside a character literal
        // leaves the parentheses counted open, but the line does not end in `,` or `(`, so the
        // blank line after it still ends the search and the next function keeps its brace.
        let code = "fn p() -> char { '(' }\n\nfn q() {\n    x();\n}\n";
        let units = extract(&rust_extractor(), code, &every_line(code));
        assert_each_line_in_one_unit(&units, 5);
        let found = spans(&units);
        assert!(
            found.contains(&("p", UnitKind::Function, 1, 1)),
            "{found:?}"
        );
        assert!(
            found.contains(&("q", UnitKind::Function, 3, 5)),
            "{found:?}"
        );
    }

    #[test]
    fn a_one_line_member_with_balanced_braces_does_not_take_the_statement_after_it() {
        // [MAGI S6 loop 14, B2b1] the bound of the bodiless-signature rule: a line whose braces
        // balance (`run() {}`) ends on its own line, so a class field after it stays apart.
        let js = "class C {\n  run() {}\n  x = 1;\n  go() {\n  }\n}\n";
        let parts = split(&BraceFamilyExtractor::javascript(), js, 2, UnitKind::Class);
        let ranges: Vec<(u32, u32)> = parts.iter().map(|p| (p.start_line, p.end_line)).collect();
        assert!(ranges.contains(&(2, 2)), "{ranges:?}");
        assert!(ranges.contains(&(3, 3)), "{ranges:?}");
        assert!(ranges.contains(&(4, 5)), "{ranges:?}");
    }

    /// `(name, start, end)` of each piece.
    fn piece_spans(parts: &[SemanticUnit]) -> Vec<(&str, u32, u32)> {
        parts
            .iter()
            .map(|p| (p.name.as_str(), p.start_line, p.end_line))
            .collect()
    }

    #[test]
    fn a_javascript_method_named_by_a_reserved_word_is_a_member() {
        // [MAGI S6 loop 14, B2b1] `NOT_FUNCTION_NAMES` applied to every line, so a class
        // method JavaScript allows to be named `delete`, `new` or `catch` was no member and a
        // large class could not split there.
        let js =
            "class C {\n  delete(key) {\n    x();\n  }\n  new() {\n  }\n  catch(e) {\n  }\n}\n";
        let parts = split(&BraceFamilyExtractor::javascript(), js, 2, UnitKind::Class);
        let found = piece_spans(&parts);
        for member in [("delete", 2, 4), ("new", 5, 6), ("catch", 7, 8)] {
            assert!(found.contains(&member), "{member:?} in {found:?}");
        }
        let ts = "export class D {\n  delete(): void {\n  }\n}\n";
        let parts = split(&BraceFamilyExtractor::typescript(), ts, 2, UnitKind::Class);
        let found = piece_spans(&parts);
        assert!(found.contains(&("delete", 2, 3)), "{found:?}");
    }

    #[test]
    fn a_reserved_word_is_a_method_name_only_directly_in_a_class_body() {
        // [MAGI S6 loop 14, B2b1] the bound of the rule above: a statement inside a static
        // block is deeper than the class body, and a top-level `if (x) {` is no method either.
        let js =
            "class C {\n  static {\n    if (ready) {\n      go();\n    }\n  }\n  run() {\n  }\n}\n";
        let parts = split(&BraceFamilyExtractor::javascript(), js, 3, UnitKind::Class);
        let found = piece_spans(&parts);
        assert!(!found.iter().any(|(name, ..)| *name == "if"), "{found:?}");
        assert!(found.contains(&("run", 7, 8)), "{found:?}");
        let top = "if (ready) {\n  go();\n}\n";
        let units = extract(&BraceFamilyExtractor::javascript(), top, &every_line(top));
        assert_eq!(spans(&units), vec![("lines 1-3", UnitKind::TopLevel, 1, 3)]);
        // Java has no such methods: `catch (e) {` stays a clause in a class body.
        let java = "class K {\n  void a() {\n  }\n  catch (e) {\n  }\n}\n";
        let parts = split(&BraceFamilyExtractor::java(), java, 2, UnitKind::Class);
        let found = piece_spans(&parts);
        assert!(
            !found.iter().any(|(name, ..)| *name == "catch"),
            "{found:?}"
        );
    }

    #[test]
    fn a_dot_h_header_is_read_with_the_cpp_rules_so_its_classes_are_units() {
        // [MAGI S6 loop 14, B2b1] the C extractor owned `.h`, so a C++ header named `.h` had no
        // class or namespace units and a class too large for a batch could not split.
        let registry = crate::review::languages::LanguageRegistry::new();
        let header = registry
            .get("include/widget.h")
            .expect("an extractor for .h");
        assert_eq!(header.language_id(), "cpp");
        let code = "namespace ui {\nclass Widget {\n  public:\n    void draw() {\n    }\n};\n}\n";
        let units = header.extract_units(code, "widget.h", ChangedLines::All);
        let found: Vec<(&str, UnitKind)> =
            units.iter().map(|u| (u.name.as_str(), u.kind)).collect();
        assert_eq!(found, vec![("ui", UnitKind::Module)], "{:?}", spans(&units));
    }

    #[test]
    fn a_c_header_reads_the_same_with_the_cpp_rules() {
        // [MAGI S6 loop 14, B2b1] the cost side of giving `.h` to C++: the C++ extractor is the
        // C one plus classes, namespaces, templates and raw strings, so a C header's units do not
        // change.
        let code = "#ifndef A_H\n#define A_H\n#include <stddef.h>\n\n#ifdef __cplusplus\nextern \"C\" {\n#endif\n\ntypedef struct node {\n    int value;\n    struct node *next;\n} node_t;\n\nextern int count;\nint parse(const char *s,\n          size_t n);\nstatic inline int twice(int x) {\n    return 2 * x;\n}\nenum color { RED, GREEN };\n\n#ifdef __cplusplus\n}\n#endif\n#endif\n";
        let every = every_line(code);
        let c = extract(&c_extractor(), code, &every);
        let cpp = extract(&BraceFamilyExtractor::cpp(), code, &every);
        assert_eq!(spans(&cpp), spans(&c));
        assert_each_line_in_one_unit(&cpp, 25);
    }

    #[test]
    fn a_c_identifier_spelled_like_a_cpp_keyword_loses_no_line_in_a_header() {
        // [MAGI S6 loop 14, B2b1] the declared cost of reading `.h` with the C++ rules: a C
        // identifier that opens a line as a C++ keyword does reads as a one-line construct, and
        // every line still lands in exactly one unit.
        let code = "int f(void) {\n    return 0;\n}\nclass = 1;\nnamespace = 2;\nint g(void) {\n    return 1;\n}\n";
        let units = extract(&BraceFamilyExtractor::cpp(), code, &every_line(code));
        assert_each_line_in_one_unit(&units, 8);
        let ranges: Vec<(u32, u32)> = units.iter().map(|u| (u.start_line, u.end_line)).collect();
        assert_eq!(
            ranges,
            vec![(1, 3), (4, 4), (5, 5), (6, 8)],
            "{:?}",
            spans(&units)
        );
        assert_eq!(units[1].kind, UnitKind::Class, "{:?}", spans(&units));
    }

    #[test]
    fn the_quote_aware_delimiter_match_skips_quoted_runs_and_keeps_an_unclosed_text() {
        // [MAGI S6 loop 14, B2b1] the bounds of `balanced_end_outside_quotes`, which the Rust
        // attribute and annotation rules share.
        let quotes = ['"', '\'', '`'];
        assert_eq!(
            balanced_end_outside_quotes("(\")\")x", '(', ')', &quotes),
            Some(5)
        );
        assert_eq!(
            balanced_end_outside_quotes("('(', `)`)", '(', ')', &quotes),
            Some(10)
        );
        assert_eq!(
            balanced_end_outside_quotes("(\"\\\")\")", '(', ')', &quotes),
            Some(7)
        );
        assert_eq!(balanced_end_outside_quotes("(\")", '(', ')', &quotes), None);
        assert_eq!(balanced_end_outside_quotes("((a)", '(', ')', &quotes), None);
        assert_eq!(
            balanced_end_outside_quotes("(\")\")", '(', ')', &[]),
            Some(3)
        );
        // An attribute or annotation that never closes keeps its line as it was, so the item
        // after it is no construct, as before the rule: its lines still land in one unit each.
        let rust = "#[doc = \"] fn h() {\n}\n";
        let units = extract(&rust_extractor(), rust, &every_line(rust));
        assert_each_line_in_one_unit(&units, 2);
        assert!(!units.iter().any(|u| u.name == "h"), "{:?}", spans(&units));
        let java = "@A(\") class F {\n}\n";
        let units = extract(&BraceFamilyExtractor::java(), java, &every_line(java));
        assert_each_line_in_one_unit(&units, 2);
        assert!(!units.iter().any(|u| u.name == "F"), "{:?}", spans(&units));
    }

    #[test]
    fn a_grouped_or_bracketed_declaration_runs_to_its_matching_closing_delimiter() {
        // [MAGI S6 loop 17, B2-1] a declaration ending in `(` or `[` sought only a `{`, so a Go
        // `var (` group, an array literal over lines or a wrapped call kept only its first line
        // and its members fell into gap units, a changed member sent without its header. The
        // delimiter in a string or a comment does not close the group.
        let go = "var (\n\ta = 1\n\n\tb = \")\"\n)\nfunc f() {\n}\n";
        let units = extract(&BraceFamilyExtractor::go(), go, &[4]);
        assert_eq!(
            spans(&units),
            vec![("variables", UnitKind::DeclarationGroup, 1, 5)]
        );
        let go = "var (\n\tcfg = Config{\n\t\tName: \"x\",\n\t}\n\tn = 2\n)\nfunc f() {\n}\n";
        let units = extract(&BraceFamilyExtractor::go(), go, &[5]);
        assert_eq!(
            spans(&units),
            vec![("variables", UnitKind::DeclarationGroup, 1, 6)]
        );
        let js = "const items = [\n  { a: 1 },\n  { b: \"]\" },\n];\nfunction g() {\n}\n";
        for ext in [
            BraceFamilyExtractor::javascript(),
            BraceFamilyExtractor::typescript(),
        ] {
            let units = extract(&ext, js, &[3]);
            assert_eq!(
                spans(&units),
                vec![("constants", UnitKind::DeclarationGroup, 1, 4)]
            );
        }
        let rust = "const NAMES: &[&str] = &[\n    \"a\",\n    // ] not the end\n    \"b\",\n];\nfn h() {}\n";
        let units = extract(&rust_extractor(), rust, &[4]);
        assert_eq!(
            spans(&units),
            vec![("constants", UnitKind::DeclarationGroup, 1, 5)]
        );
        let call = "const x = make(a,\n  b,\n  c);\nfunction g() {\n}\n";
        let units = extract(&BraceFamilyExtractor::javascript(), call, &[2]);
        assert_eq!(
            spans(&units),
            vec![("constants", UnitKind::DeclarationGroup, 1, 3)]
        );
    }

    #[test]
    fn a_bracketed_declaration_that_never_closes_loses_no_line_and_spends_a_linear_budget() {
        // [MAGI S6 loop 17, B2-1] the walk to the closing delimiter shares the pass's budget, so
        // a file of unclosed `[` lines stays linear; every line still lands in exactly one unit.
        let open = "const t = [\n  1,\n\nfunction g() {\n}\n";
        let ext = BraceFamilyExtractor::javascript();
        let units = extract(&ext, open, &every_line(open));
        assert_each_line_in_one_unit(&units, 5);
        let code = format!("{}function g() {{\n}}\n", "const a = [\n".repeat(5_000));
        let lines: Vec<&str> = code.lines().collect();
        let scanned = CodeScanner::new(ext.lexical_rules.clone()).scan_all(&code);
        let budget = SearchBudget::for_lines(lines.len());
        ext.try_detect_constructs(&lines, &scanned, false, &budget)
            .expect("small line numbers");
        let cap = lines.len() * 64 + MAX_BRACE_SEARCH_LINES;
        assert!(budget.spent() <= cap, "{} visits", budget.spent());
        let total = u32::try_from(lines.len()).expect("small");
        assert_each_line_in_one_unit(&extract(&ext, &code, &every_line(&code)), total);
    }

    #[test]
    fn a_parenthesis_in_a_string_or_comment_does_not_move_a_go_type_group_end() {
        // [MAGI B2b1 B1] the group's end was found by counting parentheses on the line text, so
        // a `)` in a raw string cut the group short and a `(` in a comment ran it to the end of
        // the file, absorbing the function after it as a member.
        let go = BraceFamilyExtractor::go();
        let named = |code: &str| -> Vec<(String, UnitKind, u32, u32)> {
            let total = u32::try_from(code.lines().count()).expect("small");
            let units = extract(&go, code, &all_lines(code));
            assert_each_line_in_one_unit(&units, total);
            spans(&units)
                .into_iter()
                .filter(|s| s.1 != UnitKind::TopLevel)
                .map(|(n, k, a, b)| (n.to_string(), k, a, b))
                .collect()
        };
        let owned = |v: &[(&str, UnitKind, u32, u32)]| -> Vec<(String, UnitKind, u32, u32)> {
            v.iter()
                .map(|&(n, k, a, b)| (n.to_string(), k, a, b))
                .collect()
        };
        for tag in ["`json:\")\"`", "`json:\"(\"`"] {
            let code = format!(
                "type (\n\tA struct {{\n\t\tTag string {tag}\n\t}}\n\tB int\n)\n\nfunc f() {{\n}}\n"
            );
            assert_eq!(
                named(&code),
                owned(&[
                    ("A", UnitKind::Struct, 2, 4),
                    ("B", UnitKind::Struct, 5, 5),
                    ("f", UnitKind::Function, 8, 9)
                ]),
                "{tag}"
            );
        }
        for comment in ["// A holds :(", "// A holds :)"] {
            let code = format!("type (\n\t{comment}\n\tA int\n)\n\nfunc f() {{\n}}\n");
            assert_eq!(
                named(&code),
                owned(&[
                    ("A", UnitKind::Struct, 2, 3),
                    ("f", UnitKind::Function, 6, 7)
                ]),
                "{comment}"
            );
        }
    }

    #[test]
    fn a_go_type_group_member_spanning_lines_ends_where_its_declaration_ends() {
        // [MAGI B2b1 7] a member ended on its first line unless that line opened a brace, so the
        // parameters of a wrapped `func` type and the fields of a struct whose type parameters
        // wrap became members of their own (`w`, `r`, `X`).
        let code = "type (\n\tF func(\n\t\tw io.Writer,\n\t\tr io.Reader,\n\t) error\n\tList[\n\t\tT any,\n\t] struct {\n\t\tX T\n\t}\n\tB int\n)\n\nfunc f() {\n}\n";
        let total = u32::try_from(code.lines().count()).expect("small");
        let units = extract(&BraceFamilyExtractor::go(), code, &all_lines(code));
        assert_each_line_in_one_unit(&units, total);
        let named: Vec<_> = spans(&units)
            .into_iter()
            .filter(|s| s.1 != UnitKind::TopLevel)
            .collect();
        assert_eq!(
            named,
            vec![
                ("F", UnitKind::Struct, 2, 5),
                ("List", UnitKind::Struct, 6, 10),
                ("B", UnitKind::Struct, 11, 11),
                ("f", UnitKind::Function, 14, 15)
            ]
        );
    }

    #[test]
    fn a_parenthesis_in_a_comment_does_not_move_a_call_initializer_end() {
        // [MAGI B2b1 B1] the call's end was found by counting parentheses on the line text, so a
        // `(` and a `)` in two comments ran the declaration over the next one.
        let ext = BraceFamilyExtractor::java();
        let code = "int x = make(\n    2); // :(\nint y = 3; // :)\nint z;\n";
        let lines: Vec<&str> = code.lines().collect();
        let scanned = CodeScanner::new(ext.lexical_rules.clone()).scan_all(code);
        let stops = |_: usize| false;
        let budget = SearchBudget::for_lines(lines.len());
        let search = BraceSearch {
            stops: &stops,
            budget: &budget,
        };
        assert_eq!(call_initializer_end(&lines, &scanned, 0, &search), 1);
        let string = "int x = make(\n    \"(\",\n    2);\nint y = g(\")\");\n";
        let lines: Vec<&str> = string.lines().collect();
        let scanned = CodeScanner::new(ext.lexical_rules.clone()).scan_all(string);
        let budget = SearchBudget::for_lines(lines.len());
        let search = BraceSearch {
            stops: &stops,
            budget: &budget,
        };
        assert_eq!(call_initializer_end(&lines, &scanned, 0, &search), 2);
    }

    #[test]
    fn the_walk_to_a_closing_delimiter_stops_when_the_pass_budget_is_spent() {
        // [MAGI S6 loop 17, B2-1] the walk spends the pass's budget like every forward search;
        // with none left it ends on its own line, as a search past its bound does.
        let code = "var (\n\ta = 1\n)\n";
        let ext = BraceFamilyExtractor::go();
        let lines: Vec<&str> = code.lines().collect();
        let scanned = CodeScanner::new(ext.lexical_rules.clone()).scan_all(code);
        let stops = |_: usize| false;
        let fresh = SearchBudget::for_lines(lines.len());
        let search = BraceSearch {
            stops: &stops,
            budget: &fresh,
        };
        assert_eq!(group_end(&lines, &scanned, 0, &search), 2);
        let spent = SearchBudget::for_lines(lines.len());
        while spent.spend() {}
        let search = BraceSearch {
            stops: &stops,
            budget: &spent,
        };
        assert_eq!(group_end(&lines, &scanned, 0, &search), 0);
    }

    #[test]
    fn a_rust_macro_rules_is_one_opaque_unit_and_no_item_is_read_inside_it() {
        // [MAGI S6 loop 17, B2-2] `macro_rules!` was no construct, so the `fn generated() {}` of
        // its token tree was read as a function of its own and the macro's header and arms fell
        // into gap units around it.
        let code = "#[macro_export]\nmacro_rules! make {\n    ($n:ident) => {\n        fn generated() {}\n    };\n}\nfn after() {}\n";
        let units = extract(&rust_extractor(), code, &[4]);
        assert_eq!(spans(&units), vec![("make", UnitKind::Function, 1, 6)]);
        assert!(rust_extractor().split_block(&units[0]).is_empty());
        let every = every_line(code);
        let units = extract(&rust_extractor(), code, &every);
        assert!(
            !units.iter().any(|u| u.name == "generated"),
            "{:?}",
            spans(&units)
        );
        assert_each_line_in_one_unit(&units, 7);
        let paren = "macro_rules! p (\n    () => { fn q() {} }\n);\nfn after() {}\n";
        let units = extract(&rust_extractor(), paren, &[2]);
        assert_eq!(spans(&units), vec![("p", UnitKind::Function, 1, 3)]);
        let in_module = "mod m {\n    macro_rules! inner {\n        () => { fn hidden() {} };\n    }\n    fn real() {}\n}\n";
        let parts = split(&rust_extractor(), in_module, 3, UnitKind::Module);
        let names: Vec<&str> = parts.iter().map(|u| u.name.as_str()).collect();
        assert!(names.contains(&"inner"), "{names:?}");
        assert!(!names.contains(&"hidden"), "{names:?}");
    }

    #[test]
    fn a_parenthesized_macro_rules_with_a_braced_arm_ends_at_its_own_closing_delimiter() {
        // [MAGI S6 loop 17, B2-2] the brace search of `macro_rules! p (` found its first arm's
        // `{`, so the macro ended at that arm and its `);` fell into a gap unit.
        let code = "macro_rules! p (\n    () => {\n        fn q() {}\n    };\n);\nfn after() {}\n";
        let units = extract(&rust_extractor(), code, &[3]);
        assert_eq!(spans(&units), vec![("p", UnitKind::Function, 1, 5)]);
    }

    #[test]
    fn a_js_computed_or_private_method_is_a_member_named_as_written() {
        // [MAGI S6 loop 17, B2-3] `[key]() {` ended in `]`, so no identifier named it, and
        // `#m() {` opened with `#`, read as a preprocessor line: neither was a member, and a class
        // holding one could not split there.
        let code = "class A {\n  #secret() {\n    return 1;\n  }\n  *[Symbol.iterator]() {\n    yield 1;\n  }\n  static #s() {}\n  async [key]() {\n  }\n  get #x() {\n    return 2;\n  }\n}\n";
        for ext in [
            BraceFamilyExtractor::javascript(),
            BraceFamilyExtractor::typescript(),
        ] {
            let parts = split(&ext, code, 3, UnitKind::Class);
            let members: Vec<(&str, u32, u32)> = parts
                .iter()
                .filter(|u| u.kind == UnitKind::Method)
                .map(|u| (u.name.as_str(), u.start_line, u.end_line))
                .collect();
            assert_eq!(
                members,
                vec![
                    ("#secret", 2, 4),
                    ("[Symbol.iterator]", 5, 7),
                    ("#s", 8, 8),
                    ("[key]", 9, 10),
                    ("#x", 11, 13)
                ]
            );
            assert_each_line_in_one_unit(&parts, 14);
        }
    }

    #[test]
    fn a_comment_before_a_blank_line_inside_an_open_parameter_list_keeps_the_body() {
        // [MAGI S6 loop 17, B2-4] the blank line was passed over only after a code line ending in
        // `,` or `(`, so a parameter with a trailing comment, or a comment line, before it ended
        // the brace search: the function ended on its first line and its body became gap units.
        let trailing = "fn f(\n    a: u8, // first\n\n    b: u8,\n) -> u8 {\n    a\n}\n";
        let units = extract(&rust_extractor(), trailing, &[6]);
        assert_eq!(spans(&units), vec![("f", UnitKind::Function, 1, 7)]);
        let own_line = "fn g(\n    a: u8,\n    // the second one\n\n    b: u8,\n) {\n}\n";
        let units = extract(&rust_extractor(), own_line, &[6]);
        assert_eq!(spans(&units), vec![("g", UnitKind::Function, 1, 7)]);
        let c = "int h(int a, /* (first */\n\n      int b)\n{\n    return a;\n}\n";
        let units = extract(&c_extractor(), c, &[5]);
        assert_eq!(spans(&units), vec![("h", UnitKind::Function, 1, 6)]);
    }

    #[test]
    fn a_cpp_comparison_call_or_subscript_operator_is_named_in_full() {
        // [MAGI S6 loop 17, B2-5] `operator<` left a `<` open and `operator()` put its symbol
        // after the first `(`, so both were named `operator`, as was any overload whose symbol
        // the head rule misread.
        let cases = [
            ("bool operator<(const Foo& other) const {", "operator<"),
            ("bool operator<(const Foo<T>& other) const {", "operator<"),
            ("bool operator()(int x) const {", "operator()"),
            ("auto operator<=>(const Foo& o) const {", "operator<=>"),
            ("T& operator[](size_t i) {", "operator[]"),
            ("bool operator>(const Foo& o) const {", "operator>"),
            ("Foo& operator<<(int n) {", "operator<<"),
            ("Foo& operator>>=(int n) {", "operator>>="),
            ("bool operator==(const Foo& o) const {", "operator=="),
        ];
        for (signature, name) in cases {
            let code = format!("{signature}\n    return x;\n}}\n");
            let units = extract(&BraceFamilyExtractor::cpp(), &code, &[2]);
            assert_eq!(
                spans(&units),
                vec![(name, UnitKind::Function, 1, 3)],
                "{signature}"
            );
        }
    }

    #[test]
    fn a_rust_inner_attribute_stays_with_its_container_and_no_line_is_lost() {
        // [MAGI S6 loop 17, B2-6] an inner attribute (`#![allow(x)]`) applies to the module or
        // crate that holds it, not to the item below it, so it is never joined to that item: at
        // the top of a file it is a unit of its own, inside a module it is in the module's header.
        let top = "#![allow(dead_code)]\nfn f() {\n    x();\n}\n";
        let units = extract(&rust_extractor(), top, &every_line(top));
        assert_each_line_in_one_unit(&units, 4);
        assert!(
            spans(&units).contains(&("f", UnitKind::Function, 2, 4)),
            "{:?}",
            spans(&units)
        );
        let module = "mod m {\n    #![allow(unused)]\n    fn a() {}\n}\n";
        let parts = split(&rust_extractor(), module, 3, UnitKind::Module);
        assert_eq!(
            spans(&parts),
            vec![
                ("m", UnitKind::Module, 1, 2),
                ("a", UnitKind::Function, 3, 3),
                ("m", UnitKind::Module, 4, 4)
            ]
        );
    }

    #[test]
    fn a_java_method_named_by_a_word_reserved_only_elsewhere_is_a_method() {
        // [MAGI S6 loop 18, B2b-5] one list of non-function words served every language, so a
        // Java method named `delete` was no unit and a call in its body was read as a member.
        let java = "class A {\n    public void delete(Long id) {\n        repo.remove(id);\n    }\n    void await(int n) {\n        x(n);\n    }\n}\n";
        let parts = split(&BraceFamilyExtractor::java(), java, 1, UnitKind::Class);
        assert_eq!(
            spans(&parts),
            vec![
                ("A", UnitKind::Class, 1, 1),
                ("delete", UnitKind::Method, 2, 4),
                ("await", UnitKind::Method, 5, 7),
                ("A", UnitKind::Class, 8, 8),
            ]
        );
        // A Java keyword is still no method name.
        let ext = BraceFamilyExtractor::java();
        for line in ["    catch (Exception e) {", "    synchronized (lock) {"] {
            assert_eq!(ext.signature_name(line, None, false), None, "{line}");
        }
    }

    #[test]
    fn an_anonymous_default_exported_function_is_a_function() {
        // [MAGI S6 loop 18, B2b-1] `function(` matched no `function ` keyword, so the export
        // was a declaration group named `declarations`.
        for head in [
            "export default function() {",
            "export default async function() {",
        ] {
            let code = format!("{head}\n    x();\n}}\n");
            for ext in [
                BraceFamilyExtractor::javascript(),
                BraceFamilyExtractor::typescript(),
            ] {
                let units = extract(&ext, &code, &every_line(&code));
                assert_eq!(
                    spans(&units),
                    vec![("unnamed", UnitKind::Function, 1, 3)],
                    "{head}"
                );
            }
        }
        // A Rust function pointer type opening a line is no function.
        let rust = "type F =\n    fn(u8) -> u8;\n";
        let units = extract(&rust_extractor(), rust, &every_line(rust));
        assert!(
            units.iter().all(|u| u.kind != UnitKind::Function),
            "{:?}",
            spans(&units)
        );
    }

    #[test]
    fn a_generator_or_a_raw_identifier_names_its_function() {
        // [MAGI S6 loop 18, B2b-6] `function *generate()` was named `` and `fn r#match()` was
        // named `r`.
        let js = "function *generate() {\n    yield 1;\n}\nfunction* other() {\n    yield 2;\n}\n";
        let units = extract(&BraceFamilyExtractor::javascript(), js, &every_line(js));
        assert_eq!(
            spans(&units),
            vec![
                ("generate", UnitKind::Function, 1, 3),
                ("other", UnitKind::Function, 4, 6),
            ]
        );
        let rust = "fn r#match() {\n    x();\n}\nimpl S {\n    pub fn r#type(&self) {}\n}\n";
        let units = extract(&rust_extractor(), rust, &every_line(rust));
        assert_eq!(
            spans(&units),
            vec![
                ("r#match", UnitKind::Function, 1, 3),
                ("S", UnitKind::Impl, 4, 6),
            ]
        );
        let parts = split(&rust_extractor(), rust, 4, UnitKind::Impl);
        assert!(
            parts.iter().any(|p| p.name == "r#type"),
            "{:?}",
            spans(&parts)
        );
    }

    #[test]
    fn an_arrow_in_a_type_parameter_bound_closes_no_angle_bracket() {
        // [MAGI S6 loop 25, B2b1b] the `>` of `=>` or `->` in a bound counted as the list's
        // close, so `foo<T extends () => void>(cb: T) {` named no member and its class did not
        // split there.
        let ts = BraceFamilyExtractor::typescript();
        let class = "class A {\n    foo<T extends () => void>(cb: T) {\n        cb();\n    }\n    bar<U extends (x: number) => void>(u: U): void {\n        u(1);\n    }\n}\n";
        assert_eq!(
            spans(&split(&ts, class, 3, UnitKind::Class)),
            vec![
                ("A", UnitKind::Class, 1, 1),
                ("foo", UnitKind::Method, 2, 4),
                ("bar", UnitKind::Method, 5, 7),
                ("A", UnitKind::Class, 8, 8),
            ]
        );
        for line in [
            "foo<T extends () => void>(cb: T) {",
            "foo<T: Fn() -> U>(x: T) {",
        ] {
            assert_eq!(
                identifier_before_paren(line),
                Some("foo".to_string()),
                "{line}"
            );
        }
        assert_eq!(skip_generics("<T extends () => void>(cb: T)"), "(cb: T)");
    }

    #[test]
    fn a_generic_typescript_method_is_a_method() {
        // [MAGI S6 loop 18, B2b-4] the text before `(` ended in `>`, so no identifier named the
        // method and a generic method of a class or an interface was no member.
        let ts = BraceFamilyExtractor::typescript();
        let class = "class A {\n    foo<T>(x: T) {\n        return x;\n    }\n    static #bar<K>(k: K) {\n        return k;\n    }\n}\n";
        assert_eq!(
            spans(&split(&ts, class, 1, UnitKind::Class)),
            vec![
                ("A", UnitKind::Class, 1, 1),
                ("foo", UnitKind::Method, 2, 4),
                ("#bar", UnitKind::Method, 5, 7),
                ("A", UnitKind::Class, 8, 8),
            ]
        );
        let interface = "interface I {\n    foo<T>(x: T): T;\n    bar(): void;\n}\n";
        assert_eq!(
            spans(&split(&ts, interface, 1, UnitKind::Trait)),
            vec![
                ("I", UnitKind::Trait, 1, 1),
                ("foo", UnitKind::Method, 2, 2),
                ("bar", UnitKind::Method, 3, 3),
                ("I", UnitKind::Trait, 4, 4),
            ]
        );
        // A C++ explicit specialization is named the same way.
        assert_eq!(
            identifier_before_paren("void f<int>(int x) {"),
            Some("f".to_string())
        );
    }

    #[test]
    fn a_cpp_operator_spaced_from_its_symbol_is_a_function() {
        // [MAGI S6 loop 18, B2b-7] the symbol had to touch the word `operator`, so a spaced
        // overload was no function and its body fell into gap units.
        let cpp = "bool operator <(int a, int b) {\n    return a;\n}\nT& operator = (const T& o) {\n    return *this;\n}\nbool operator [ ] (int i) const {\n    return v[i];\n}\n";
        let units = extract(&BraceFamilyExtractor::cpp(), cpp, &every_line(cpp));
        assert_eq!(
            spans(&units),
            vec![
                ("operator<", UnitKind::Function, 1, 3),
                ("operator=", UnitKind::Function, 4, 6),
                ("operator[]", UnitKind::Function, 7, 9),
            ]
        );
        // Outside C++ the word `operator` is an identifier like any other.
        let js = "const operator = (x) => {\n    return x;\n};\n";
        let units = extract(&BraceFamilyExtractor::javascript(), js, &every_line(js));
        assert_eq!(spans(&units), vec![("operator", UnitKind::Function, 1, 3)]);
    }

    #[test]
    fn a_brace_in_a_comment_does_not_end_a_wrapped_signature_on_its_first_line() {
        // [MAGI S6 loop 18, B2b-2] a body-less signature was told from a one-line construct by a
        // `{` anywhere in its first line, so one in a comment ended it there and the rest of the
        // signature fell into a gap unit.
        let rust = "trait T {\n    fn f(a: u8, // {\n        b: u8);\n    fn g(&self);\n}\n";
        assert_eq!(
            spans(&split(&rust_extractor(), rust, 1, UnitKind::Trait)),
            vec![
                ("T", UnitKind::Trait, 1, 1),
                ("f", UnitKind::Method, 2, 3),
                ("g", UnitKind::Method, 4, 4),
                ("T", UnitKind::Trait, 5, 5),
            ]
        );
        // A one-line construct whose code ends in `}` still ends on its line.
        let one_line = "trait T {\n    fn f(&self) { x(); } // c\n    fn g(&self);\n}\n";
        assert_eq!(
            spans(&split(&rust_extractor(), one_line, 1, UnitKind::Trait)),
            vec![
                ("T", UnitKind::Trait, 1, 1),
                ("f", UnitKind::Method, 2, 2),
                ("g", UnitKind::Method, 3, 3),
                ("T", UnitKind::Trait, 4, 4),
            ]
        );
    }

    #[test]
    fn a_one_line_function_with_a_trailing_comment_is_a_unit() {
        // [MAGI S6 loop 18, B2b-8] a one-line function was told by its text ending in `}`, so a
        // comment after the brace left it in a gap unit.
        let c = "void f() {} // note\nint g(void) {\n    return 0;\n}\n";
        let units = extract(&c_extractor(), c, &every_line(c));
        assert_eq!(
            spans(&units),
            vec![
                ("f", UnitKind::Function, 1, 1),
                ("g", UnitKind::Function, 2, 4),
            ]
        );
        let cpp =
            "class A {\n    void f() {} // note\n    int g() {\n        return 0;\n    }\n};\n";
        assert_eq!(
            spans(&split(
                &BraceFamilyExtractor::cpp(),
                cpp,
                1,
                UnitKind::Class
            )),
            vec![
                ("A", UnitKind::Class, 1, 1),
                ("f", UnitKind::Method, 2, 2),
                ("g", UnitKind::Method, 3, 5),
                ("A", UnitKind::Class, 6, 6),
            ]
        );
    }

    #[test]
    fn a_method_after_an_unterminated_member_keeps_its_own_body() {
        // [MAGI S6 loop 18, B2b-9] the forward search's stop rule read a method named by a
        // reserved word (`delete() {`) as no construct, unlike the main scan, so a member with
        // no `;` before it took that method's body.
        let ts = "abstract class A {\n    abstract foo(): void\n    delete() {\n        x();\n    }\n}\n";
        assert_eq!(
            spans(&split(
                &BraceFamilyExtractor::typescript(),
                ts,
                1,
                UnitKind::Class
            )),
            vec![
                ("A", UnitKind::Class, 1, 1),
                ("foo", UnitKind::Method, 2, 2),
                ("delete", UnitKind::Method, 3, 5),
                ("A", UnitKind::Class, 6, 6),
            ]
        );
    }

    #[test]
    fn a_class_under_a_multi_line_decorator_splits_at_its_own_brace() {
        // [MAGI S6 loop 19, B2b-3] the block's opening brace was the first code line opening a
        // brace, the decorator's `@Component({`, so the members were sought inside the
        // decorator's object and the class did not split.
        let ts = "@Component({\n  selector: 'app',\n})\nexport class C {\n  run() {\n    x();\n  }\n  stop() {}\n}\n";
        assert_eq!(
            spans(&split(
                &BraceFamilyExtractor::typescript(),
                ts,
                4,
                UnitKind::Class
            )),
            vec![
                ("C", UnitKind::Class, 1, 4),
                ("run", UnitKind::Method, 5, 7),
                ("stop", UnitKind::Method, 8, 8),
                ("C", UnitKind::Class, 9, 9),
            ]
        );
    }

    #[test]
    fn an_annotation_whose_argument_holds_a_multi_line_literal_stays_with_its_member() {
        // [MAGI S6 loop 19, B2b-4] the walk up an annotation's lines gave up at a line that lies
        // wholly inside a string (a text block, a template literal), so the annotation was cut
        // from its member and reviewed as a gap unit.
        let java = "public class A {\n    @Query(\"\"\"\n        SELECT *\n        \"\"\")\n    public void f() {\n        a();\n    }\n}\n";
        let parts = split(&BraceFamilyExtractor::java(), java, 5, UnitKind::Class);
        let found_java = spans(&parts);
        assert!(
            found_java.contains(&("f", UnitKind::Method, 2, 7)),
            "f with its annotation in {found_java:?}"
        );
        let ts = "@Component({\n  template: `\n    <div></div>\n  `,\n})\nexport class C {\n  run() {}\n}\n";
        let units = extract(&BraceFamilyExtractor::typescript(), ts, &all_lines(ts));
        assert!(
            spans(&units).contains(&("C", UnitKind::Class, 1, 8)),
            "C with its decorator in {:?}",
            spans(&units)
        );
    }

    #[test]
    fn a_declaration_wrapped_after_a_bare_equal_sign_keeps_its_value() {
        // [MAGI S6 loop 19, B2b-5] a declaration whose line ends in `=` ended on that line, so
        // its value fell into a gap unit, cut from the name it initializes.
        let rust = "pub static TABLE: &[u8] =\n    &[1, 2, 3];\nfn f() {\n}\n";
        assert_eq!(
            spans(&extract(&rust_extractor(), rust, &all_lines(rust))),
            vec![
                ("statics", UnitKind::DeclarationGroup, 1, 2),
                ("f", UnitKind::Function, 3, 4)
            ]
        );
        let c = "static int n =\n    5;\nint g(void) {\n}\n";
        assert_eq!(
            spans(&extract(&c_extractor(), c, &all_lines(c))),
            vec![
                ("statics", UnitKind::DeclarationGroup, 1, 2),
                ("g", UnitKind::Function, 3, 4)
            ]
        );
        // With no `;`, the value is the next line, never the declaration after it.
        let js = "const x =\n  5\nconst y = 1\n";
        assert_eq!(
            spans(&extract(
                &BraceFamilyExtractor::javascript(),
                js,
                &all_lines(js)
            )),
            vec![("constants", UnitKind::DeclarationGroup, 1, 3)]
        );
    }

    #[test]
    fn a_value_wrapped_after_a_bare_equal_sign_runs_to_its_closing_delimiter_without_a_semicolon() {
        // [MAGI B2b1 6] in Go or JavaScript without semicolons, the value after a bare `=` was
        // only the next line, so a call or a literal wrapped over lines (`g(` ... `)`, `[` ...
        // `]`) lost its other lines to a gap unit.
        let go = "var x =\n\tg(\n\t\t1,\n\t)\nfunc f() {\n}\n";
        assert_eq!(
            spans(&extract(&BraceFamilyExtractor::go(), go, &all_lines(go))),
            vec![
                ("variables", UnitKind::DeclarationGroup, 1, 4),
                ("f", UnitKind::Function, 5, 6)
            ]
        );
        let js = "const x =\n  [\n    1,\n    2\n  ]\nfunction f() {\n}\n";
        assert_eq!(
            spans(&extract(
                &BraceFamilyExtractor::javascript(),
                js,
                &all_lines(js)
            )),
            vec![
                ("constants", UnitKind::DeclarationGroup, 1, 5),
                ("f", UnitKind::Function, 6, 7)
            ]
        );
        let braced = "var m =\n\tmap[string]int{\n\t\t\"a\": 1,\n\t}\nfunc f() {\n}\n";
        assert_eq!(
            spans(&extract(
                &BraceFamilyExtractor::go(),
                braced,
                &all_lines(braced)
            )),
            vec![
                ("variables", UnitKind::DeclarationGroup, 1, 4),
                ("f", UnitKind::Function, 5, 6)
            ]
        );
    }

    #[test]
    fn the_member_searches_of_a_go_type_group_spend_the_pass_budget() {
        // [MAGI S6 loop 19, B2b-2] a member's closing-brace search ran past the group's `)` to
        // the end of the file, outside the pass budget, so a file of groups whose members never
        // close inside them took O(n²) line visits before the review clock.
        let ext = BraceFamilyExtractor::go();
        let group = "type (\n\tA struct {\n\t\tX int\n\t}\n)\n";
        let lines: Vec<&str> = group.lines().collect();
        let scanned = CodeScanner::new(ext.lexical_rules.clone()).scan_all(group);
        let budget = SearchBudget::for_lines(lines.len());
        ext.try_detect_constructs(&lines, &scanned, false, &budget)
            .expect("small line numbers");
        assert!(budget.spent() > 0, "the member search spent no visit");

        let groups = 2_000;
        let code = "type (\n\tA struct {\n)\n".repeat(groups);
        let lines: Vec<&str> = code.lines().collect();
        let scanned = CodeScanner::new(ext.lexical_rules.clone()).scan_all(&code);
        let budget = SearchBudget::for_lines(lines.len());
        let constructs = ext
            .try_detect_constructs(&lines, &scanned, false, &budget)
            .expect("small line numbers");
        // Each group's member search stays inside its group: a few visits per line at most.
        let linear_bound_per_line = 4;
        assert!(
            budget.spent() <= lines.len() * linear_bound_per_line,
            "{} visits for {} lines",
            budget.spent(),
            lines.len()
        );
        assert_eq!(constructs.len(), groups);
        let total = u32::try_from(lines.len()).expect("small");
        let units = extract(&ext, &code, &(1..=total).collect::<Vec<_>>());
        assert_each_line_in_one_unit(&units, total);
    }
    /// The `(start, end)` of the unit named `name` in `units`.
    fn span_of(units: &[SemanticUnit], name: &str) -> (u32, u32) {
        let unit = units
            .iter()
            .find(|u| u.name == name)
            .unwrap_or_else(|| panic!("no unit {name} in {:?}", spans(units)));
        (unit.start_line, unit.end_line)
    }

    #[test]
    fn a_one_line_construct_never_takes_the_brace_of_the_block_after_it() {
        // A one-line construct whose code does not end in `;` searched on for its brace and took
        // the `{` of the block after it, so the two became one unit.
        let cases: [(BraceFamilyExtractor, &str, &str); 4] = [
            (
                rust_extractor(),
                "fn f() -> u8 { 1 }
extern \"C\" {
    fn g();
}
",
                "f",
            ),
            (
                BraceFamilyExtractor::java(),
                "void f() { x(); }
static {
    y();
}
",
                "f",
            ),
            (
                BraceFamilyExtractor::javascript(),
                "function a() { x(); }
{
  b();
}
",
                "a",
            ),
            (
                BraceFamilyExtractor::java(),
                "void f() { x(); }
synchronized (lock) {
    y();
}
",
                "f",
            ),
        ];
        let got: Vec<(u32, u32)> = cases
            .iter()
            .map(|(ext, code, name)| span_of(&extract(ext, code, &[1, 2, 3, 4]), name))
            .collect();
        assert_eq!(got, vec![(1, 1); cases.len()]);
        for (ext, code, _) in &cases {
            assert_each_line_in_one_unit(&extract(ext, code, &[1, 2, 3, 4]), 4);
        }
    }

    #[test]
    fn a_parenthesis_in_a_comment_or_a_string_does_not_hold_a_signature_open() {
        // The stop rule counted the parentheses on the line text, so a `(` inside a comment or a
        // string kept the parameter list "open" and the function without a body took the next
        // function's brace.
        let cases: [(BraceFamilyExtractor, &str); 4] = [
            (
                BraceFamilyExtractor::go(),
                "func f(x int) // see (
func g() {
	x()
}
",
            ),
            (
                BraceFamilyExtractor::go(),
                "func f(x int) /* ( */
func g() {
	x()
}
",
            ),
            (
                BraceFamilyExtractor::typescript(),
                "function f(a: \"(\"): void
function g() {
  x()
}
",
            ),
            (
                c_extractor(),
                "int f(char c) // see (
int g(void) {
    return 0;
}
",
            ),
        ];
        let got: Vec<Option<(u32, u32)>> = cases
            .iter()
            .map(|(ext, code)| {
                extract(ext, code, &[1, 2, 3, 4])
                    .iter()
                    .find(|u| u.name == "g")
                    .map(|u| (u.start_line, u.end_line))
            })
            .collect();
        assert_eq!(got, vec![Some((2, 4)); cases.len()]);
        for (ext, code) in &cases {
            assert_each_line_in_one_unit(&extract(ext, code, &[1, 2, 3, 4]), 4);
        }
    }

    /// The spans of the units of `code` under `ext`, every line changed, sorted by first line.
    fn sorted_spans(ext: &BraceFamilyExtractor, code: &str) -> Vec<(String, UnitKind, u32, u32)> {
        let total = u32::try_from(code.lines().count()).expect("small");
        let units = extract(ext, code, &(1..=total).collect::<Vec<_>>());
        assert_each_line_in_one_unit(&units, total);
        let mut all: Vec<(String, UnitKind, u32, u32)> = spans(&units)
            .into_iter()
            .map(|(n, k, a, b)| (n.to_string(), k, a, b))
            .collect();
        all.sort_by_key(|u| u.2);
        all
    }

    #[test]
    fn a_wrapped_c_prototype_is_one_declaration_with_its_return_type_line() {
        // A prototype wrapped over lines (`int foo(` ... `int b);`) or written GNU style, with
        // its return type on a line of its own, split into a gap and a declaration.
        let decl = |a, b| ("declarations".to_string(), UnitKind::DeclarationGroup, a, b);
        let func = |n: &str, a, b| (n.to_string(), UnitKind::Function, a, b);
        let cases = [
            (
                "int foo(\n    int a,\n    int b);\nint g(void) {\n}\n",
                vec![decl(1, 3), func("g", 4, 5)],
            ),
            (
                "int foo(\n    int a,\n    int b\n);\nint g(void) {\n}\n",
                vec![decl(1, 4), func("g", 5, 6)],
            ),
            (
                "int\nfoo (int a);\nint g(void) {\n}\n",
                vec![decl(1, 2), func("g", 3, 4)],
            ),
            (
                "int\nfoo (int a,\n     int b);\nint g(void) {\n}\n",
                vec![decl(1, 3), func("g", 4, 5)],
            ),
            (
                "static int\nfoo (int a,\n     int b)\n{\n  return a;\n}\n",
                vec![func("foo", 1, 6)],
            ),
        ];
        let got: Vec<_> = cases
            .iter()
            .map(|(code, _)| sorted_spans(&c_extractor(), code))
            .collect();
        let want: Vec<_> = cases.iter().map(|(_, w)| w.clone()).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn a_closing_parenthesis_line_never_opens_a_declaration_of_its_own() {
        // The `);` that closes a wrapped prototype ends in `;`, so it read as a declaration of
        // its own and split the prototype; it now stays with the lines it closes, whether the
        // opener is recognised (`int foo(`) or not (a comment before it).
        let cases = [
            "int foo(\n    int a\n);\nint g(void) {\n}\n",
            "/* note */ int foo(\n    int a\n);\nint g(void) {\n}\n",
        ];
        let got: Vec<Vec<u32>> = cases
            .iter()
            .map(|code| {
                sorted_spans(&c_extractor(), code)
                    .iter()
                    .map(|u| u.2)
                    .collect()
            })
            .collect();
        assert_eq!(got, vec![vec![1, 4]; cases.len()]);
    }

    #[test]
    fn a_method_inside_a_field_initializer_is_no_member_of_the_class() {
        // The member pass promoted a signature at any depth, so the method of an anonymous class
        // in a field's initializer became a member and cut the field's header from its body.
        let code = "class C {\n    Runnable r = new Runnable() {\n        public void run() {\n            go();\n        }\n    };\n\n    void b() {\n        x();\n    }\n}\n";
        let parts = split(&BraceFamilyExtractor::java(), code, 9, UnitKind::Class);
        assert!(
            parts.iter().any(|p| p.start_line <= 2 && 6 <= p.end_line),
            "the field is split: {:?}",
            spans(&parts)
        );
        assert!(parts.iter().all(|p| p.name != "run"), "{:?}", spans(&parts));
        assert!(
            spans(&parts).contains(&("b", UnitKind::Method, 8, 10)),
            "{:?}",
            spans(&parts)
        );
    }

    #[test]
    fn a_typescript_enum_or_namespace_is_a_unit_and_a_namespace_splits_into_its_items() {
        // [MAGI B2b1 9] TypeScript's `enum` and `namespace` were in no keyword table, so their
        // lines fell into gap units (or a declaration group, after `export` or `const`).
        let code = "enum E {\n  A,\n  B,\n}\nexport enum F {\n  X,\n}\nconst enum G {\n  Y,\n}\nnamespace N {\n  function f() {\n  }\n}\ndeclare namespace D {\n  function g(): void;\n}\n";
        let ext = BraceFamilyExtractor::typescript();
        let units = extract(&ext, code, &all_lines(code));
        assert_eq!(
            spans(&units),
            vec![
                ("E", UnitKind::Enum, 1, 4),
                ("F", UnitKind::Enum, 5, 7),
                ("G", UnitKind::Enum, 8, 10),
                ("N", UnitKind::Module, 11, 14),
                ("D", UnitKind::Module, 15, 17)
            ]
        );
        let parts = split(&ext, code, 12, UnitKind::Module);
        assert_eq!(
            spans(&parts),
            vec![
                ("N", UnitKind::Module, 11, 11),
                ("f", UnitKind::Function, 12, 13),
                ("N", UnitKind::Module, 14, 14)
            ]
        );
        // `namespace` is no reserved word: a variable or an assignment by that name stays one.
        let named = "const namespace = 1;\nnamespace = 2;\n";
        assert_eq!(
            spans(&extract(&ext, named, &all_lines(named))),
            vec![
                ("constants", UnitKind::DeclarationGroup, 1, 1),
                ("lines 2-2", UnitKind::TopLevel, 2, 2)
            ]
        );
    }

    #[test]
    fn a_typescript_optional_method_is_a_member() {
        // [MAGI B2b1 10] `foo?(x: number): void;` ended in `?` before its `(`, so no identifier
        // named it and an interface or class could not split there.
        let ext = BraceFamilyExtractor::typescript();
        let interface =
            "interface I {\n  foo?(x: number): void;\n  qux?<T>(x: T): T;\n  bar(): void;\n}\n";
        let parts = split(&ext, interface, 2, UnitKind::Trait);
        assert_eq!(
            spans(&parts),
            vec![
                ("I", UnitKind::Trait, 1, 1),
                ("foo", UnitKind::Method, 2, 2),
                ("qux", UnitKind::Method, 3, 3),
                ("bar", UnitKind::Method, 4, 4),
                ("I", UnitKind::Trait, 5, 5)
            ]
        );
        let class = "class C {\n  baz?(): void {\n    x();\n  }\n  q() {\n  }\n}\n";
        let parts = split(&ext, class, 3, UnitKind::Class);
        assert_eq!(
            spans(&parts),
            vec![
                ("C", UnitKind::Class, 1, 1),
                ("baz", UnitKind::Method, 2, 4),
                ("q", UnitKind::Method, 5, 6),
                ("C", UnitKind::Class, 7, 7)
            ]
        );
    }

    #[test]
    fn a_free_function_in_a_module_or_a_namespace_stays_a_function_when_split() {
        // [MAGI B2a 1] every block promoted its functions to methods, so a free function in a
        // Rust `mod` or a C++ `namespace` was labelled a method; only a type's block has methods.
        let rust = "mod m {\n    fn f() {\n        x();\n    }\n}\n";
        let parts = split(&rust_extractor(), rust, 3, UnitKind::Module);
        assert_eq!(
            spans(&parts),
            vec![
                ("m", UnitKind::Module, 1, 1),
                ("f", UnitKind::Function, 2, 4),
                ("m", UnitKind::Module, 5, 5)
            ]
        );
        let cpp = "namespace n {\n    int f(int a) {\n        return a;\n    }\n}\n";
        let parts = split(&BraceFamilyExtractor::cpp(), cpp, 3, UnitKind::Module);
        assert_eq!(
            spans(&parts),
            vec![
                ("n", UnitKind::Module, 1, 1),
                ("f", UnitKind::Function, 2, 4),
                ("n", UnitKind::Module, 5, 5)
            ]
        );
        let impl_block = "impl S {\n    fn f() {\n        x();\n    }\n}\n";
        let parts = split(&rust_extractor(), impl_block, 3, UnitKind::Impl);
        assert!(
            spans(&parts).contains(&("f", UnitKind::Method, 2, 4)),
            "{:?}",
            spans(&parts)
        );
    }

    #[test]
    fn a_c_function_named_by_a_word_reserved_only_elsewhere_is_a_function() {
        // [MAGI B2b1 9] C shared the word list of JavaScript, Java and C++, so a C function named
        // `delete`, `new`, `try` or `function` (no C keyword) was no function and its body fell
        // into gap units.
        let code = "int delete(int x) {\n    return x;\n}\nint new(int y) {\n    return y;\n}\nint try(void) {\n    return 0;\n}\nint function(void) {\n    return 1;\n}\n";
        assert_eq!(
            spans(&extract(&c_extractor(), code, &all_lines(code))),
            vec![
                ("delete", UnitKind::Function, 1, 3),
                ("new", UnitKind::Function, 4, 6),
                ("try", UnitKind::Function, 7, 9),
                ("function", UnitKind::Function, 10, 12)
            ]
        );
        // C++ reserves `delete`, `new` and `try`, not `function`; C still reserves its own words.
        let cpp = BraceFamilyExtractor::cpp();
        for line in ["    delete (p);", "    try (x) {", "    new (p) T {"] {
            assert_eq!(cpp.signature_name(line, None, false), None, "{line}");
        }
        assert_eq!(
            cpp.signature_name("int function(void) {", None, false),
            Some("function".to_string())
        );
        for line in ["    if (x) {", "    while (x) {", "    sizeof (x) {"] {
            assert_eq!(
                c_extractor().signature_name(line, None, false),
                None,
                "{line}"
            );
        }
    }

    #[test]
    fn a_cpp_inline_exported_nested_or_anonymous_namespace_is_a_module() {
        // [MAGI B2b1 10] `inline namespace v1 {` and `export namespace m {` matched no keyword,
        // so their functions were top-level units and their header and closing lines gaps; a
        // nested `namespace a::b {` was named `a`.
        let code = "inline namespace v1 {\nint f(int a) {\n    return a;\n}\n}\nnamespace a::b {\nint g() {\n    return 1;\n}\n}\nnamespace {\nint h() {\n    return 2;\n}\n}\nexport namespace m {\nint k() {\n    return 3;\n}\n}\n";
        assert_eq!(
            spans(&extract(
                &BraceFamilyExtractor::cpp(),
                code,
                &all_lines(code)
            )),
            vec![
                ("v1", UnitKind::Module, 1, 5),
                ("a::b", UnitKind::Module, 6, 10),
                ("unnamed", UnitKind::Module, 11, 15),
                ("m", UnitKind::Module, 16, 20)
            ]
        );
    }

    #[test]
    fn a_blank_line_between_an_attribute_and_its_construct_keeps_them_together() {
        // [MAGI B2b1 12] the walk up over attributes stopped at the first blank line, so an
        // attribute separated from its item by one (`#[derive(Debug)]`, blank, `struct S {`) was
        // a gap unit of its own, cut from the item it governs.
        let rust = "fn a() {\n}\n#[derive(Debug)]\n\nstruct S {\n    x: u8,\n}\n";
        assert_eq!(
            spans(&extract(&rust_extractor(), rust, &all_lines(rust))),
            vec![
                ("a", UnitKind::Function, 1, 2),
                ("S", UnitKind::Struct, 3, 7)
            ]
        );
        let java = "@Deprecated\n\npublic class A {\n    int x;\n}\n";
        assert_eq!(
            spans(&extract(
                &BraceFamilyExtractor::java(),
                java,
                &all_lines(java)
            )),
            vec![("A", UnitKind::Class, 1, 5)]
        );
        let member = "class A {\n    @Override\n\n    public void f() {\n        x();\n    }\n}\n";
        let parts = split(&BraceFamilyExtractor::java(), member, 5, UnitKind::Class);
        assert!(
            spans(&parts).contains(&("f", UnitKind::Method, 2, 6)),
            "{:?}",
            spans(&parts)
        );
        // A plain comment above a blank line is no attribute: it stays apart.
        let comment = "// note\n\nfn f() {\n}\n";
        assert_eq!(
            spans(&extract(&rust_extractor(), comment, &all_lines(comment))),
            vec![
                ("f", UnitKind::Function, 3, 4),
                ("lines 1-2", UnitKind::TopLevel, 1, 2)
            ]
        );
    }

    #[test]
    fn a_brace_opened_inside_a_parameter_list_is_not_the_function_body() {
        // [MAGI B2a 1] a parameter whose type is a literal block (`opts: {`, `struct {`) opened
        // the first brace of the signature, so the function ended where the parameter's block
        // closed (`}) {`) and its body fell into a gap unit.
        let ts = "function f(opts: {\n  a: number;\n}) {\n  body();\n}\nfunction g() {\n}\n";
        assert_eq!(
            spans(&extract(
                &BraceFamilyExtractor::typescript(),
                ts,
                &all_lines(ts)
            )),
            vec![
                ("f", UnitKind::Function, 1, 5),
                ("g", UnitKind::Function, 6, 7)
            ]
        );
        let c = "void f(struct {\n    int a;\n} s) {\n    use(s);\n}\nint g(void) {\n    return 0;\n}\n";
        assert_eq!(
            spans(&extract(&c_extractor(), c, &all_lines(c))),
            vec![
                ("f", UnitKind::Function, 1, 5),
                ("g", UnitKind::Function, 6, 8)
            ]
        );
        // A call passing a callback closes its list with no brace after it (`});`): the call
        // keeps the block it opened, as before.
        let call = "describe('x', () => {\n  it();\n});\nfunction g() {\n}\n";
        assert_eq!(
            spans(&extract(
                &BraceFamilyExtractor::javascript(),
                call,
                &all_lines(call)
            )),
            vec![
                ("describe", UnitKind::Function, 1, 3),
                ("g", UnitKind::Function, 4, 5)
            ]
        );
    }

    #[test]
    fn a_wrapped_attribute_search_never_climbs_above_the_previous_construct() {
        // [MAGI loop 26, 1] the backward search for a wrapped attribute climbed to the top of
        // the file while its brackets stayed unbalanced, so a file of `)` lines between
        // functions took O(n²) line visits; it stops at the line after the previous construct.
        // The `(` that balances line 3 lies inside `a`, so no attribute is recovered for `b`.
        let java = "@Ann void a() { f(\n}\n)\nvoid b() {}\n";
        assert_eq!(
            spans(&extract(
                &BraceFamilyExtractor::java(),
                java,
                &all_lines(java)
            )),
            vec![
                ("a", UnitKind::Function, 1, 2),
                ("b", UnitKind::Function, 4, 4),
                ("lines 3-3", UnitKind::TopLevel, 3, 3)
            ]
        );
    }

    #[test]
    fn a_template_header_wrapped_over_lines_belongs_to_its_construct() {
        // [MAGI loop 26, 3] a `template <` line that left its `<` open was read as a declaration
        // list, so the header became a declaration group apart from the function it declares.
        let cpp = "template <typename T,\n          typename U>\nvoid f(T t, U u) {\n    g();\n}\n";
        assert_eq!(
            spans(&extract(&BraceFamilyExtractor::cpp(), cpp, &all_lines(cpp))),
            vec![("f", UnitKind::Function, 1, 5)]
        );
        let three = "template <typename T,\n          typename U,\n          typename V>\nvoid f(T t) {\n    g();\n}\n";
        assert_eq!(
            spans(&extract(
                &BraceFamilyExtractor::cpp(),
                three,
                &all_lines(three)
            )),
            vec![("f", UnitKind::Function, 1, 6)]
        );
        let class = "template <typename T,\n          typename U>\nclass Box {\n    T t;\n};\n";
        assert_eq!(
            spans(&extract(
                &BraceFamilyExtractor::cpp(),
                class,
                &all_lines(class)
            )),
            vec![("Box", UnitKind::Class, 1, 5)]
        );
    }

    #[test]
    fn an_allman_brace_after_a_parameter_that_opens_a_block_is_the_body() {
        // [MAGI loop 26, 4] with a parameter typed by a literal block (`opts: {`), the body's
        // brace was sought only on the line closing the list (`})`), so an Allman `{` on the next
        // line left the body in a gap unit.
        let ts = "function f(opts: {\n    a: number;\n})\n{\n    body();\n}\nfunction g() {\n}\n";
        assert_eq!(
            spans(&extract(
                &BraceFamilyExtractor::typescript(),
                ts,
                &all_lines(ts)
            )),
            vec![
                ("f", UnitKind::Function, 1, 6),
                ("g", UnitKind::Function, 7, 8)
            ]
        );
        let c = "void f(struct {\n    int a;\n} s)\n{\n    use(s);\n}\nint g(void) {\n    return 0;\n}\n";
        assert_eq!(
            spans(&extract(&c_extractor(), c, &all_lines(c))),
            vec![
                ("f", UnitKind::Function, 1, 6),
                ("g", UnitKind::Function, 7, 9)
            ]
        );
        // A call that closes its list with `});` keeps the block it opened, as before.
        let call = "run({\n  a: 1,\n});\n{\n  x();\n}\n";
        assert_eq!(
            spans(&extract(
                &BraceFamilyExtractor::javascript(),
                call,
                &all_lines(call)
            ))
            .first()
            .map(|s| (s.2, s.3)),
            Some((1, 3))
        );
    }

    #[test]
    fn a_generic_default_in_a_method_signature_is_no_assignment() {
        // [MAGI loop 26, 5] the `=` of a type parameter's default (`get<T = unknown>(`) read as
        // an assignment, so the method was no member and stayed in the class's header piece.
        let ts = "class K {\n    get<T = unknown>(key: string): T {\n        return x;\n    }\n    set(k: string) {\n        y();\n    }\n}\n";
        let parts = split(&BraceFamilyExtractor::typescript(), ts, 2, UnitKind::Class);
        assert_eq!(
            piece_spans(&parts),
            vec![("K", 1, 1), ("get", 2, 4), ("set", 5, 7), ("K", 8, 8)]
        );
    }

    #[test]
    fn a_rust_visibility_spaced_before_its_path_still_opens_the_item() {
        // [MAGI loop 26, 6] `pub (crate)` was no modifier, since the `(` had to touch `pub`, so
        // the function was no construct and fell into a gap unit.
        let rust =
            "pub (crate) fn f() {\n    x();\n}\npub (in crate::a) struct S {\n    n: u8,\n}\n";
        assert_eq!(
            spans(&extract(&rust_extractor(), rust, &all_lines(rust))),
            vec![
                ("f", UnitKind::Function, 1, 3),
                ("S", UnitKind::Struct, 4, 6)
            ]
        );
    }

    #[test]
    fn a_c_attribute_on_its_own_line_belongs_to_the_construct_below_it() {
        // [MAGI loop 27, B2b1a] a standard attribute (`[[nodiscard]]`, C23 and C++11) or a GCC or
        // MSVC one (`__attribute__((noreturn))`) on the line above a function was no attribute
        // line, so it was reviewed as a gap unit apart from the function it qualifies; on the
        // construct's own line it was already read past.
        let cases: [(BraceFamilyExtractor, &str, &str, (u32, u32)); 5] = [
            (
                BraceFamilyExtractor::c(),
                "int a;\n\n__attribute__((noreturn))\nvoid die(void) {\n    exit(1);\n}\n",
                "die",
                (3, 6),
            ),
            (
                BraceFamilyExtractor::c(),
                "int a;\n\n[[nodiscard]]\nint f(void) {\n    return 1;\n}\n",
                "f",
                (3, 6),
            ),
            (
                BraceFamilyExtractor::cpp(),
                "int a;\n\n[[nodiscard]]\n[[deprecated(\"use g\")]]\nint f(int x) {\n    return x;\n}\n",
                "f",
                (3, 7),
            ),
            (
                BraceFamilyExtractor::cpp(),
                "int a;\n\n[[deprecated(\n    \"use g\")]]\nint f(int x) {\n    return x;\n}\n",
                "f",
                (3, 7),
            ),
            (
                BraceFamilyExtractor::cpp(),
                "int a;\n\n__declspec(dllexport)\nint f(int x) {\n    return x;\n}\n",
                "f",
                (3, 6),
            ),
        ];
        for (ext, code, name, span) in &cases {
            let total = u32::try_from(code.lines().count()).expect("small");
            let units = extract(ext, code, &(1..=total).collect::<Vec<_>>());
            assert_each_line_in_one_unit(&units, total);
            assert_eq!(span_of(&units, name), *span, "{code}: {:?}", spans(&units));
        }
        let code = "class C {\n    [[nodiscard]]\n    int m() {\n        return 1;\n    }\n};\n";
        let ext = BraceFamilyExtractor::cpp();
        let units = extract(&ext, code, &[1, 2, 3, 4, 5, 6]);
        let parts = ext.split_block(&units[0]);
        assert_eq!(span_of(&parts, "m"), (2, 5), "{:?}", spans(&parts));
    }

    #[test]
    fn an_annotation_after_a_modifier_is_read_past() {
        // [MAGI loop 27, B2b1a] annotations were read past only before the modifiers, so
        // `public @Deprecated class C {` (legal Java, annotations and modifiers mix in any order)
        // was no class, `public @A final @B("x") class C {` a function named `B`, and
        // TypeScript's `export @dec class C {` (decorators after `export`, TS 5.0) no class.
        // After an import group it also joined that group, a defect of its own pinned by
        // `a_construct_after_a_declaration_group_is_never_part_of_it`.
        let class = |n: &str, a, b| (n.to_string(), UnitKind::Class, a, b);
        let cases: [(BraceFamilyExtractor, &str); 3] = [
            (
                BraceFamilyExtractor::java(),
                "import a.B;\n\npublic @Deprecated class C {\n    void m() {\n    }\n}\n",
            ),
            (
                BraceFamilyExtractor::java(),
                "import a.B;\n\npublic @A final @B(\"x\") class C {\n    void m() {\n    }\n}\n",
            ),
            (
                BraceFamilyExtractor::typescript(),
                "foo();\n\nexport @dec class C {\n  m() {\n  }\n}\n",
            ),
        ];
        for (ext, code) in &cases {
            let found = sorted_spans(ext, code);
            assert!(found.contains(&class("C", 3, 6)), "{code}: {found:?}");
        }
        let ext = BraceFamilyExtractor::java();
        let code =
            "class C {\n    public @Ann(\"x\") void m() {\n    }\n    void n() {\n    }\n}\n";
        let units = extract(&ext, code, &[1, 2, 3, 4, 5, 6]);
        let parts = ext.split_block(&units[0]);
        assert_eq!(span_of(&parts, "m"), (2, 3), "{:?}", spans(&parts));
    }

    #[test]
    fn a_construct_after_a_declaration_group_is_never_part_of_it() {
        // [MAGI loop 27, found verifying B2b1a] `export` is a declaration keyword of JavaScript
        // and TypeScript, so a line it opens joined the declaration group above it even when it
        // opens a class or a function (`export class C {`), and the group ran through the whole
        // block: the main scan tries constructs before declarations, the group's join did not.
        let cases: [(BraceFamilyExtractor, &str, &str, UnitKind); 4] = [
            (
                BraceFamilyExtractor::typescript(),
                "import a from 'a';\n\nexport class C {\n  m() {\n  }\n}\n",
                "C",
                UnitKind::Class,
            ),
            (
                BraceFamilyExtractor::javascript(),
                "import a from 'a';\nexport function f() {\n  return 1;\n}\n",
                "f",
                UnitKind::Function,
            ),
            (
                BraceFamilyExtractor::typescript(),
                "import a from 'a';\n\nexport @dec class C {\n  m() {\n  }\n}\n",
                "C",
                UnitKind::Class,
            ),
            (
                BraceFamilyExtractor::typescript(),
                "export const x = 1;\nexport default class C {\n  m() {\n  }\n}\n",
                "C",
                UnitKind::Class,
            ),
        ];
        for (ext, code, name, kind) in &cases {
            let found = sorted_spans(ext, code);
            let construct = found
                .iter()
                .find(|u| u.0 == *name)
                .unwrap_or_else(|| panic!("no {name}: {found:?}"));
            assert_eq!(construct.1, *kind, "{code}: {found:?}");
            assert!(
                found
                    .iter()
                    .filter(|u| u.1 == UnitKind::DeclarationGroup)
                    .all(|u| u.3 < construct.2),
                "{code}: {found:?}"
            );
        }
        // A declaration still joins: `export const` after an import is one group.
        let found = sorted_spans(
            &BraceFamilyExtractor::typescript(),
            "import a from 'a';\nexport const x = 1;\n",
        );
        assert_eq!(
            found,
            vec![("imports".to_string(), UnitKind::DeclarationGroup, 1, 2)]
        );
    }

    #[test]
    fn a_comparison_in_parentheses_inside_a_template_header_closes_no_bracket() {
        // [MAGI loop 27, B2b1b] C++ requires a `>` or `>=` inside a template argument to sit in
        // parentheses (`template <int N = (M >= 0)>`), but the angle brackets were counted
        // there too, so the `>` of `>=` closed the header early: the function after it was read
        // as a declaration and the class was no unit.
        assert_eq!(skip_generics("<int N = (M >= 0)> void f() {"), "void f() {");
        assert_eq!(skip_generics("<T: Fn(Vec<u8>) -> u8> Foo<T> {"), "Foo<T> {");
        assert_eq!(
            skip_generics("<T extends (a: X<Y>) => void>(x: T) {"),
            "(x: T) {"
        );
        let ext = BraceFamilyExtractor::cpp();
        let cases: [(&str, &str, UnitKind, (u32, u32)); 3] = [
            (
                "int a;\n\ntemplate <int N = (M >= 0)> void f() {\n    g();\n}\n",
                "f",
                UnitKind::Function,
                (3, 5),
            ),
            (
                "int a;\n\ntemplate <bool B = (N >= 0)> class C {\n    void m() {\n    }\n};\n",
                "C",
                UnitKind::Class,
                (3, 6),
            ),
            (
                // After a declaration group a wrapped header has a defect of its own, pinned by
                // `a_wrapped_template_header_after_a_declaration_group_is_never_part_of_it`.
                "// a\n\ntemplate <int N,\n          bool B = (N >= 0)>\nvoid f() {\n    g();\n}\n",
                "f",
                UnitKind::Function,
                (3, 7),
            ),
        ];
        for (code, name, kind, span) in &cases {
            let found = sorted_spans(&ext, code);
            assert!(
                found.contains(&(name.to_string(), *kind, span.0, span.1)),
                "{code}: {found:?}"
            );
        }
    }

    #[test]
    fn a_wrapped_template_header_after_a_declaration_group_is_never_part_of_it() {
        // [MAGI loop 27, found verifying B2b1b] the first line of a template header wrapped over
        // lines ends in `,`, so it joined the declaration group above it as a declaration list
        // and its function lost its header; the main scan passes over such a header before it
        // looks for declarations, and the group's join now does too.
        let ext = BraceFamilyExtractor::cpp();
        for code in [
            "int a;\n\ntemplate <typename T,\n          typename U>\nvoid f() {\n    g();\n}\n",
            "int a;\n\ntemplate <int N,\n          bool B = (N >= 0)>\nvoid f() {\n    g();\n}\n",
        ] {
            let found = sorted_spans(&ext, code);
            assert_eq!(
                found,
                vec![
                    ("declarations".to_string(), UnitKind::DeclarationGroup, 1, 1),
                    ("lines 2-2".to_string(), UnitKind::TopLevel, 2, 2),
                    ("f".to_string(), UnitKind::Function, 3, 7),
                ],
                "{code}"
            );
        }
    }

    #[test]
    fn a_parenthesis_ending_a_comment_opens_no_call_initializer() {
        // [MAGI loop 27, B2b1b] a call initializer was told by the line's text ending in `(`, so a
        // complete declaration whose comment ends in one (`int x = 1; // note (`) opened a call
        // and ran over the declaration below it; only a `(` the line leaves open in code counts.
        let ext = BraceFamilyExtractor::c();
        let code = "int x = 1; // note (\nint y = 2;\nint z;\n";
        let lines: Vec<&str> = code.lines().collect();
        let scanned = CodeScanner::new(ext.lexical_rules.clone()).scan_all(code);
        let stops = |_: usize| false;
        let budget = SearchBudget::for_lines(lines.len());
        let search = BraceSearch {
            stops: &stops,
            budget: &budget,
        };
        assert_eq!(call_initializer_end(&lines, &scanned, 0, &search), 0);
        let units = extract(&ext, code, &[1, 2, 3]);
        let parts = ext.split_block(&units[0]);
        assert_eq!(
            parts
                .iter()
                .map(|u| (u.start_line, u.end_line))
                .collect::<Vec<_>>(),
            vec![(1, 1), (2, 2), (3, 3)]
        );
    }

    #[test]
    fn a_compound_assignment_opens_no_initializer() {
        // [MAGI loop 27, B2b1b] the initializer checks excluded only the comparisons (`==`, `!=`,
        // `<=`, `>=`) from a line ending in `=`, so a compound assignment (`x +=`) read as a
        // declaration's bare `=`, unlike `holds_bare_assignment`; both now agree.
        for line in [
            "static int n =",
            "int a[] = {",
            "T c = make(",
            "x =",
            "let x: u8=",
        ] {
            assert!(opens_initializer(line), "{line}");
        }
        for op in [
            "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=", "==", "!=", "<=", ">=",
        ] {
            let line = format!("x {op}");
            assert!(!opens_initializer(&line), "{line}");
        }
        let ext = BraceFamilyExtractor::c();
        let code = "x +=\n    1;\nint y;\n";
        let lines: Vec<&str> = code.lines().collect();
        let scanned = CodeScanner::new(ext.lexical_rules.clone()).scan_all(code);
        let stops = |_: usize| false;
        let budget = SearchBudget::for_lines(lines.len());
        let search = BraceSearch {
            stops: &stops,
            budget: &budget,
        };
        assert_eq!(bare_initializer_end(&lines, &scanned, 0, &search), 0);
    }

    #[test]
    fn unicode_names_arrows_in_strings_and_calls_in_default_arguments_keep_their_units() {
        // [MAGI loop 27, B2b1a] declared green, three false positives measured: an identifier
        // character is any Unicode letter or digit (`is_identifier_char`), so `fn café()` is a
        // function; an `=>` inside a string makes no function expression of an object
        // initializer; and a wrapped parameter whose default value is a call
        // (`int x = make(1),`) names no function, since an `=` before the `(` rules a signature
        // out (`identifier_before_paren`), so the outer function keeps its body.
        let func = |n: &str, a, b| (n.to_string(), UnitKind::Function, a, b);
        for (ext, code) in [
            (
                BraceFamilyExtractor::rust(),
                "fn café() {\n    x();\n}\nfn z() {\n}\n",
            ),
            (
                BraceFamilyExtractor::go(),
                "func café() {\n    x()\n}\nfunc z() {\n}\n",
            ),
            (
                BraceFamilyExtractor::typescript(),
                "function café() {\n    x();\n}\nfunction z() {\n}\n",
            ),
            (
                BraceFamilyExtractor::c(),
                "int café(void) {\n    return 1;\n}\nint z(void) {\n}\n",
            ),
        ] {
            assert_eq!(
                sorted_spans(&ext, code),
                vec![func("café", 1, 3), func("z", 4, 5)],
                "{code}"
            );
        }
        assert_eq!(
            sorted_spans(
                &BraceFamilyExtractor::javascript(),
                "const options = { note: \"=>\" };\nfunction f() {\n}\n"
            ),
            vec![
                ("constants".to_string(), UnitKind::DeclarationGroup, 1, 1),
                func("f", 2, 3),
            ]
        );
        assert_eq!(
            sorted_spans(
                &BraceFamilyExtractor::cpp(),
                "int f(int a,\n      int x = make(1),\n      int y = 2) {\n    return a;\n}\nint g() {\n}\n"
            ),
            vec![func("f", 1, 5), func("g", 6, 7)]
        );
    }

    /// `(name, kind, start, end)` with an owned name, as [`sorted_spans`] gives them.
    fn unit(name: &str, kind: UnitKind, start: u32, end: u32) -> (String, UnitKind, u32, u32) {
        (name.to_string(), kind, start, end)
    }

    #[test]
    fn an_arrow_in_a_wrapped_template_header_closes_no_angle_bracket() {
        // [MAGI loop 28, B2b1b] `template_angle_delta` read the `>` of a trailing return type
        // (`-> void`) in a default template argument as a close, unlike `skip_generics`: the
        // header balanced on the wrong line and was cut from the function it declares.
        for code in [
            "template <typename F = auto (*)(int) -> void,\n          typename G = int>\nvoid f() {\n    x();\n}\nint g() {\n}\n",
            "template <typename T,\n          typename F = auto (*)(T) -> void>\nvoid f() {\n    x();\n}\nint g() {\n}\n",
        ] {
            assert_eq!(
                sorted_spans(&BraceFamilyExtractor::cpp(), code),
                vec![
                    unit("f", UnitKind::Function, 1, 5),
                    unit("g", UnitKind::Function, 6, 7),
                ],
                "{code}"
            );
        }
    }

    #[test]
    fn an_equals_sign_inside_a_parenthesized_return_type_is_no_assignment() {
        // [MAGI loop 28, B2b1b] `ListHead::assigns` read the `=` of a `decltype` operand, which
        // `parameter_list_head` reads past, as an assignment, so the function was no unit. An
        // `=` before the list at parenthesis depth zero still makes an initializer.
        assert_eq!(
            sorted_spans(
                &BraceFamilyExtractor::cpp(),
                "decltype(x = 1) f() {\n    x();\n}\nint g() {\n}\n"
            ),
            vec![
                unit("f", UnitKind::Function, 1, 3),
                unit("g", UnitKind::Function, 4, 5),
            ]
        );
        assert_eq!(
            sorted_spans(
                &BraceFamilyExtractor::cpp(),
                "int (*p)(int) = make(1);\nint g() {\n}\n"
            ),
            vec![
                unit("declarations", UnitKind::DeclarationGroup, 1, 1),
                unit("g", UnitKind::Function, 2, 3),
            ]
        );
    }

    #[test]
    fn a_nested_braced_initializer_opening_its_line_is_one_declaration() {
        // [MAGI loop 28, B2b1b] `opens_initializer` stripped one trailing `{`, so `= {{` (a
        // nested initializer opening the next lines) opened none and its first lines fell into
        // a gap unit.
        for code in [
            "int grid[2][2] = {{\n    1, 2}, {\n    3, 4}};\nint g(void) {\n}\n",
            "int grid[2][2] = { {\n    1, 2}, {\n    3, 4} };\nint g(void) {\n}\n",
        ] {
            assert_eq!(
                sorted_spans(&BraceFamilyExtractor::c(), code),
                vec![
                    unit("declarations", UnitKind::DeclarationGroup, 1, 3),
                    unit("g", UnitKind::Function, 4, 5),
                ],
                "{code}"
            );
        }
    }

    #[test]
    fn a_line_closing_a_wrapped_list_opens_no_declaration_list() {
        // [MAGI loop 28, B2b1a] `opens_declaration_list` lacked `is_declaration`'s `)` rule, so a
        // stray `),` outside every construct opened a declaration list that ran through the
        // function after it.
        assert_eq!(
            sorted_spans(
                &BraceFamilyExtractor::c(),
                "),\nint g(void) {\n    return 1;\n}\n"
            ),
            vec![
                unit("lines 1-1", UnitKind::TopLevel, 1, 1),
                unit("g", UnitKind::Function, 2, 4),
            ]
        );
    }

    #[test]
    fn a_storage_class_before_an_aggregate_keyword_keeps_the_aggregate_one_unit() {
        // [MAGI loop 28, B2b1a] the C and C++ extractors stripped no storage class, so
        // `static struct S {` matched no keyword: its header and its `} s;` line fell into gap
        // units around its fields. A variable of the type, a function returning it and a
        // plain static function read as before.
        for ext in [BraceFamilyExtractor::c(), BraceFamilyExtractor::cpp()] {
            for (code, kind) in [
                (
                    "static struct S {\n    int a;\n} s;\nint g(void) {\n}\n",
                    UnitKind::Struct,
                ),
                (
                    "const struct S {\n    int a;\n} s = {1};\nint g(void) {\n}\n",
                    UnitKind::Struct,
                ),
                (
                    "static union S {\n    int a;\n    float b;\n} s;\nint g(void) {\n}\n",
                    UnitKind::Struct,
                ),
            ] {
                let spans = sorted_spans(&ext, code);
                let last = u32::try_from(code.lines().count()).expect("small");
                let close = last - 2;
                assert_eq!(
                    spans,
                    vec![
                        unit("S", kind, 1, close),
                        unit("g", UnitKind::Function, close + 1, last),
                    ],
                    "{code}"
                );
            }
            assert_eq!(
                sorted_spans(
                    &ext,
                    "static struct S *make(void) {\n    return 0;\n}\nstatic int h(void) {\n}\nstatic struct S s = {\n    1,\n};\n"
                ),
                vec![
                    unit("make", UnitKind::Function, 1, 3),
                    unit("h", UnitKind::Function, 4, 5),
                    unit("statics", UnitKind::DeclarationGroup, 6, 8),
                ]
            );
        }
    }

    #[test]
    fn a_return_type_literal_block_keeps_the_function_body() {
        // [MAGI loop 28 sweep, B2a/B2b1b] the brace a return type opens (`function f(): {`, a
        // TypeScript object type, or `func f() struct {`, a Go anonymous struct) was taken for
        // the body's, so the function ended where the type closed and its body fell into a gap
        // unit. The body's brace is the one the type's closing line opens after its lowest point
        // (`} {`, `}> {`, `} => {`) or an Allman `{` on the next line.
        let ts = BraceFamilyExtractor::typescript();
        let body = "\n    return { a: 1 };\n}\nfunction g() {\n}\n";
        for head in [
            "function f(): {\n    a: number;\n} {",
            "async function f(): Promise<{\n    a: number;\n}> {",
        ] {
            let code = format!("{head}{body}");
            assert_eq!(
                sorted_spans(&ts, &code),
                vec![
                    unit("f", UnitKind::Function, 1, 5),
                    unit("g", UnitKind::Function, 6, 7),
                ],
                "{code}"
            );
        }
        assert_eq!(
            sorted_spans(
                &ts,
                &format!("function f(): {{\n    a: number;\n}}\n{{{body}")
            ),
            vec![
                unit("f", UnitKind::Function, 1, 6),
                unit("g", UnitKind::Function, 7, 8),
            ]
        );
        let class = "class C {\n    m(): {\n        a: number;\n    } {\n        return { a: 1 };\n    }\n}\n";
        assert!(
            spans(&split(&ts, class, 5, UnitKind::Class)).contains(&("m", UnitKind::Method, 2, 6)),
            "{:?}",
            spans(&split(&ts, class, 5, UnitKind::Class))
        );
        assert_eq!(
            sorted_spans(
                &BraceFamilyExtractor::go(),
                "func f() struct {\n\tA int\n} {\n\treturn struct{ A int }{1}\n}\nfunc g() {\n}\n"
            ),
            vec![
                unit("f", UnitKind::Function, 1, 5),
                unit("g", UnitKind::Function, 6, 7),
            ]
        );
        // Unchanged: a method signature with no body ends where its type closes, and a C++
        // ref-qualifier's `&` before the body's brace is no type.
        assert_eq!(
            sorted_spans(
                &ts,
                "interface I {\n    m(): {\n        a: number;\n    };\n}\nfunction g() {\n}\n"
            ),
            vec![
                unit("I", UnitKind::Trait, 1, 5),
                unit("g", UnitKind::Function, 6, 7),
            ]
        );
        assert_eq!(
            sorted_spans(
                &BraceFamilyExtractor::cpp(),
                "void S::f() & {\n    x();\n}\nint g() {\n}\n"
            ),
            vec![
                unit("f", UnitKind::Function, 1, 3),
                unit("g", UnitKind::Function, 4, 5),
            ]
        );
    }

    #[test]
    fn a_construct_on_the_line_closing_a_wrapped_attribute_keeps_the_attribute() {
        // [MAGI loop 28, B2b1a, and its sweep] a construct on the line that closes a template
        // header or an attribute wrapped from the lines above (`typename U> class Box {`,
        // `}) export class C {`, `Debug)] struct S {`) matched no keyword, since the line opens
        // with the attribute's tail: the construct fell into gap units and a large class did
        // not split at its methods.
        let class_box = "template <typename T,\n          typename U> class Box {\n    void m() {\n        x();\n    }\n};\nint g() {\n}\n";
        assert_eq!(
            sorted_spans(&BraceFamilyExtractor::cpp(), class_box),
            vec![
                unit("Box", UnitKind::Class, 1, 6),
                unit("g", UnitKind::Function, 7, 8),
            ]
        );
        assert!(spans(&split(
            &BraceFamilyExtractor::cpp(),
            class_box,
            4,
            UnitKind::Class
        ))
        .contains(&("m", UnitKind::Method, 3, 5)));
        for (ext, code, expected) in [
            (
                BraceFamilyExtractor::cpp(),
                "template <typename T,\n          typename U> struct Box : Base<T> {\n    T t;\n};\nint g() {\n}\n",
                vec![
                    unit("Box", UnitKind::Struct, 1, 4),
                    unit("g", UnitKind::Function, 5, 6),
                ],
            ),
            (
                BraceFamilyExtractor::typescript(),
                "@Component({\n    selector: 'x',\n}) export class C {\n    m() {\n        x();\n    }\n}\nfunction g() {\n}\n",
                vec![
                    unit("C", UnitKind::Class, 1, 7),
                    unit("g", UnitKind::Function, 8, 9),
                ],
            ),
            (
                BraceFamilyExtractor::java(),
                "@SuppressWarnings({\n    \"unchecked\"}) public class C {\n    void m() {\n        x();\n    }\n}\n",
                vec![unit("C", UnitKind::Class, 1, 6)],
            ),
            (
                BraceFamilyExtractor::rust(),
                "#[derive(\n    Debug)] struct S {\n    a: u8,\n}\nfn g() {\n}\n",
                vec![
                    unit("S", UnitKind::Struct, 1, 4),
                    unit("g", UnitKind::Function, 5, 6),
                ],
            ),
        ] {
            assert_eq!(sorted_spans(&ext, code), expected, "{code}");
        }
        // A method under an annotation whose arguments close on the method's line.
        let java = "class C {\n    @Ann(\n        x = 1) void m() {\n        y();\n    }\n}\n";
        assert!(
            spans(&split(
                &BraceFamilyExtractor::java(),
                java,
                4,
                UnitKind::Class
            ))
            .contains(&("m", UnitKind::Method, 2, 5)),
            "{:?}",
            spans(&split(
                &BraceFamilyExtractor::java(),
                java,
                4,
                UnitKind::Class
            ))
        );
    }

    #[test]
    fn loop_28_shapes_read_as_one_unit_each() {
        // [MAGI loop 28] declared green, false positives each: a TypeScript type literal over
        // lines is one declaration (B2a); a `#define` whose text holds a keyword or a function,
        // continued by `\`, stays in the defines group, its braces no structure (B2a); a C
        // `typedef struct` or `typedef enum` is one declaration with its `} Name;` line (B2b1a);
        // a parameter's literal block closing before the line that closes the list, which opens
        // the body (`other: number) {`), keeps the body (B2b1b).
        let decl = |name: &str, end| unit(name, UnitKind::DeclarationGroup, 1, end);
        let func = |name: &str, start, end| unit(name, UnitKind::Function, start, end);
        let ts = BraceFamilyExtractor::typescript();
        let c = BraceFamilyExtractor::c();
        for (ext, code, expected) in [
            (
                &ts,
                "type T = {\n    a: number;\n};\nfunction f() {\n}\n",
                vec![decl("type aliases", 3), func("f", 4, 5)],
            ),
            (
                &c,
                "#define DECL struct S {\nint g(void) {\n}\n",
                vec![decl("defines", 1), func("g", 2, 3)],
            ),
            (
                &c,
                "#define FOO(x) \\\n    static int f(void) { return x; }\nint g(void) {\n}\n",
                vec![decl("defines", 2), func("g", 3, 4)],
            ),
            (
                &c,
                "typedef struct Point {\n    int x;\n} Point;\nint g(void) {\n}\n",
                vec![decl("typedefs", 3), func("g", 4, 5)],
            ),
            (
                &c,
                "typedef enum {\n    A,\n    B\n} E;\nint g(void) {\n}\n",
                vec![decl("typedefs", 4), func("g", 5, 6)],
            ),
            (
                &ts,
                "function f(opts: {\n    a: number;\n},\nother: number) {\n    body();\n}\nfunction g() {\n}\n",
                vec![func("f", 1, 6), func("g", 7, 8)],
            ),
        ] {
            assert_eq!(sorted_spans(ext, code), expected, "{code}");
        }
    }

    #[test]
    fn a_comment_after_a_return_type_brace_hides_the_type_declared() {
        // [MAGI loop 28 sweep] declared: `opens_return_type_literal` reads the line's text, so a
        // comment after the type's brace (`function f(): { // shape`) hides the rule and the line
        // reads as opening the body, as before the rule: the function ends where the type closes
        // and its body falls into a later unit, every line still in exactly one unit.
        assert_eq!(
            sorted_spans(
                &BraceFamilyExtractor::typescript(),
                "function f(): { // shape\n    a: number;\n} {\n    return x;\n}\nfunction g() {\n}\n"
            ),
            vec![
                unit("f", UnitKind::Function, 1, 3),
                unit("lines 4-5", UnitKind::TopLevel, 4, 5),
                unit("g", UnitKind::Function, 6, 7),
            ]
        );
    }

    #[test]
    fn a_braced_member_initializer_wrapped_over_lines_splits_the_constructor_declared() {
        // [MAGI loop 28 sweep] declared: the brace a C++ constructor's member initializer opens
        // (`Foo::Foo() : arr{`) is taken for the body's when the initializer wraps over lines,
        // so the constructor ends where it closes and its body falls into later units. Telling it
        // from a TypeScript return type (`): Foo {`, whose brace is the body) needs the language
        // in the brace search; every line stays in exactly one unit.
        assert_eq!(
            sorted_spans(
                &BraceFamilyExtractor::cpp(),
                "Foo::Foo() : arr{\n    1, 2} {\n    x();\n}\nint g() {\n}\n"
            ),
            vec![
                unit("Foo", UnitKind::Function, 1, 2),
                unit("declarations", UnitKind::DeclarationGroup, 3, 3),
                unit("lines 4-4", UnitKind::TopLevel, 4, 4),
                unit("g", UnitKind::Function, 5, 6),
            ]
        );
    }

    #[test]
    fn a_char_literal_in_a_rust_attribute_hides_its_bracket() {
        // [MAGI loop 29, B2b1b] `rust_attribute_end` read no `'`, since one may open a lifetime,
        // so the `]` of a char literal (`#[doc = ']']`) closed the attribute and the item after
        // it on the line was no construct: its lines fell into a gap unit.
        let code = "#[doc = ']'] fn f() {\n    x();\n}\n#[a(']', '\\'')] struct S {\n    x: u8,\n}\nfn g<'a>(x: &'a u8) {\n}\n";
        assert_eq!(
            sorted_spans(&rust_extractor(), code),
            vec![
                unit("f", UnitKind::Function, 1, 3),
                unit("S", UnitKind::Struct, 4, 6),
                unit("g", UnitKind::Function, 7, 8),
            ]
        );
        fn end(t: &str) -> Option<&str> {
            rust_attribute_end(t).and_then(|e| t.get(..e))
        }
        assert_eq!(end("[doc = ']'] x"), Some("[doc = ']']"));
        assert_eq!(end("[a('\\u{5D}', ']')] x"), Some("[a('\\u{5D}', ']')]"));
        // A lifetime has no closing quote and stays plain text.
        assert_eq!(end("[b(T: 'a)] x"), Some("[b(T: 'a)]"));
        assert_eq!(end("[b(T: 'a, U: 'b)] x"), Some("[b(T: 'a, U: 'b)]"));
    }
}
