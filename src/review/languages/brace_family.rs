// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! Brace-delimited language extractor (Rust, C, C++, Go, Java, JS, TS).
//!
//! Uses [`CodeScanner`] for lexical-aware
//! brace counting and keyword detection to extract complete semantic
//! units (functions, structs, classes, declaration groups).

use std::borrow::Cow;
use std::cell::Cell;

use super::scanner::{
    find_closing_brace, find_closing_brace_within, CodeScanner, LexicalRules, ScannedLine,
};
use super::{
    gap_units, split_into_members, ChangedLines, LanguageExtractor, Member, SemanticUnit, UnitKind,
};
use crate::review::context::FileContext;

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
    /// ([`Self::signature_name`]): [`NOT_FUNCTION_NAMES`], or for Java
    /// [`JAVA_NOT_FUNCTION_NAMES`], since a word reserved only in another language (`delete`,
    /// `typeof`) is a legal Java method name.
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
    /// keyword table names (TypeScript's `enum` or `namespace`, a JavaScript IIFE) is no
    /// construct at all, so its lines fall into gap units. A single such unit, or one
    /// declaration, too large for one
    /// batch therefore ends the review with the typed `ReviewError::TooLargeForBatch` (RV03),
    /// never a silent cut.
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

impl BraceFamilyExtractor {
    /// Detect all constructs in the file or, with `members`, in a block's body, where a
    /// signature that ends in `;` is a member too ([`Self::member_prototypes`]).
    ///
    /// No constructs when a line number does not fit `u32` ([`Self::try_detect_constructs`]):
    /// the file's changed lines then fall into gap units and a block is reviewed whole, as
    /// [`shift_members`] leaves it, never split at fabricated coordinates.
    ///
    /// Invariant: the constructs come sorted by their first line and their ranges are disjoint,
    /// which `ChangedLines::assign` and `split_into_members` rely on; lines no construct holds
    /// are the gaps. O(n) in the lines ([`Self::try_detect_constructs`]).
    ///
    /// # Arguments
    ///
    /// * `lines` - The text's lines: a whole file, or the lines of a block's body.
    /// * `scanned` - The scanner's record of each of those lines, in the same order.
    /// * `members` - Whether the lines are a block's body (a class, an `impl`, a module).
    ///
    /// # Returns
    ///
    /// The constructs, 1-based line numbers counted from the first of `lines`.
    // MAGI FIX (S6 loop 14): the arguments, the result and the invariant are documented.
    fn detect_constructs(
        &self,
        lines: &[&str],
        scanned: &[ScannedLine],
        members: bool,
    ) -> Vec<Construct> {
        let budget = SearchBudget::for_lines(lines.len());
        self.try_detect_constructs(lines, scanned, members, &budget)
            .unwrap_or_default()
    }

    /// The constructs of [`Self::detect_constructs`], or `None` when a line number does not
    /// fit `u32`. The 8 MiB file cap keeps that out of reach (8 388 608 lines at most); the
    /// check makes the failure mode explicit instead of a `u32::MAX` fallback.
    ///
    /// A heuristic of several rules, tried in this order on each line: a Go type group, a block
    /// or function keyword, a C-family signature, then a declaration or a declaration group.
    /// Each rule lives in its own helper and the order is the contract, since a later rule would
    /// misread what an earlier one claims (Java's `public class` as a function): a new rule or
    /// language goes into a helper placed in that order, and the test
    /// `construct_ranges_are_sorted_and_disjoint` with the per-language tests pins the result.
    /// A known maintenance hotspot, declared. Splitting it into a table of rule strategies is
    /// deferred until the heuristic set settles after MS3: the rules' order is the contract,
    /// and a large refactor at the end of the review gate risks moving it.
    /// O(n) in the lines, plus the helpers' forward searches, which share `budget`
    /// ([`SearchBudget`]) and so add O(n) line visits in all.
    ///
    /// Each construct found moves the scan past its last line, so no two ranges overlap; the
    /// scan back for its doc comments never goes above the line after the previous construct
    /// ([`Self::scan_back_for_docs`]).
    ///
    /// # Arguments
    ///
    /// * `lines` - The text's lines.
    /// * `scanned` - The scanner's record of each line.
    /// * `members` - Whether the lines are a block's body: a signature ending in `;` is then a
    ///   member ([`Self::member_prototypes`]), and a JavaScript or TypeScript method directly in
    ///   it may be named by a reserved word (`reserved_word_methods`). A signature deeper than
    ///   the body's first line is then no member: it sits inside a member no rule read (a
    ///   method of an anonymous class in a field's initializer), which stays one piece.
    /// * `budget` - The line visits the forward searches of this pass may spend.
    ///
    /// # Returns
    ///
    /// The constructs, sorted and disjoint, or `None` when a line number does not fit `u32`.
    fn try_detect_constructs(
        &self,
        lines: &[&str],
        scanned: &[ScannedLine],
        members: bool,
        budget: &SearchBudget,
    ) -> Option<Vec<Construct>> {
        // MAGI FIX (S6 loop 10): the conversions fell back to `u32::MAX`, `1` or the line
        // number, unlike `shift_members`; an unrepresentable line now aborts detection.
        let mut constructs = Vec::new();
        let mut i = 0;
        let depth_before = |k: usize| {
            scanned
                .get(k)
                .map(|s| i64::from(s.depth_after) - i64::from(s.brace_delta))
        };
        let stops = |j: usize| self.starts_construct(lines, scanned, j, members);
        let search = BraceSearch {
            stops: &stops,
            budget,
        };

        while let Some(line) = lines.get(i) {
            if scanned.get(i).is_some_and(|s| s.is_non_code) {
                i += 1;
                continue;
            }

            let trimmed = line.trim();
            let line_num = line_number(i)?;
            // MAGI FIX (S6 loop 12): the scan back for docs crossed into a one-line construct
            // that opens with an attribute prefix (`@Override public String toString() { .. }`),
            // so two ranges overlapped; it now stops at the line after the previous construct.
            let floor = constructs
                .last()
                .map_or(0, |c: &Construct| c.end_line as usize);

            // MAGI FIX (S6 loop 7): Go's `type (` opened one nameless struct and its later
            // members fell into gaps; each member is a construct of its own.
            if let Some((last, group)) = self.grouped_types(lines, scanned, i, &search) {
                constructs.extend(group?);
                i = last + 1;
                continue;
            }
            if let Some((kind, name)) = self.construct_keyword(trimmed) {
                let start = self.scan_back_for_docs(lines, scanned, i, floor);
                let mut end = brace_block_end(lines, scanned, i, line_num, &search)?;
                // MAGI FIX (S6 loop 17, B2-2): a `macro_rules! m (` ... `);` ended at the first
                // brace of its arms; it runs to its own closing delimiter.
                if self.opens_macro_rules(trimmed) {
                    end = end.max(line_number(group_end(lines, scanned, i, &search))?);
                }
                constructs.push(Construct {
                    kind,
                    name,
                    start_line: line_number(start)?,
                    end_line: end,
                });
                // Skip past this construct
                i = (end as usize).max(i + 1);
                continue;
            }
            let prototypes = members && self.member_prototypes();
            // A signature deeper than the body's own lines sits inside a member no rule read
            // (an anonymous class in a field's initializer, `r = new Runnable() {`): taking it
            // for a member cut the field's header from its body.
            let nested = members && depth_before(i) != depth_before(0);
            if let Some(name) = (!nested)
                .then(|| self.line_signature(line, scanned, i, members))
                .flatten()
            {
                // MAGI FIX (S6 loop 15): the brace search ran twice, once here and once in
                // `brace_block_end`, spending a wrapped header's visits of the budget twice.
                let open = open_brace_line(lines, scanned, i, &search);
                let opens = open.is_some();
                // MAGI FIX (S6 loop 18, B2b-8): a one-line function was told by its text ending
                // in `}`, so a comment after the brace (`void f() {} // note`) left it in a gap;
                // the scanner says whether its code ends in `}`.
                let one_line = scanned.get(i).is_some_and(|s| s.ends_block);
                if opens || one_line || prototypes {
                    let head = self.return_type_start(lines, scanned, i, floor, members);
                    let start = self.scan_back_for_docs(lines, scanned, head, floor);
                    // MAGI FIX (S6 loop 5): an abstract member (`User find(long id);`) ends at its
                    // `;`, which may sit on a later line of a wrapped signature.
                    let end = if opens || one_line {
                        block_end_after(lines, scanned, i, line_num, open, &search)?
                    } else {
                        line_number(statement_end(lines, scanned, i, &search))?
                    };
                    constructs.push(Construct {
                        kind: UnitKind::Function,
                        name,
                        start_line: line_number(start)?,
                        end_line: end,
                    });
                    i = (end as usize).max(i + 1);
                    continue;
                }
            }
            // Check declaration keywords
            // MAGI FIX (S6 loop 8): `int x,` opened no declaration, so a declaration list
            // wrapped after a comma lost its first line to a gap unit.
            if self.is_declaration(trimmed, scanned.get(i))
                || self.opens_declaration_list(trimmed)
                || self.opens_wrapped_prototype(lines, scanned, i, &search)
            {
                // MAGI FIX (S6 loop 5): `const f = () => {` ended on its first line and its body
                // fell into a gap; a function expression is a function, with its whole body.
                if let Some((name, end)) = self.function_expression(lines, scanned, i, &search) {
                    let start = self.scan_back_for_docs(lines, scanned, i, floor);
                    constructs.push(Construct {
                        kind: UnitKind::Function,
                        name,
                        start_line: line_number(start)?,
                        end_line: line_number(end)?,
                    });
                    i = end + 1;
                    continue;
                }
                let group_start = i;
                // MAGI FIX (S6 loop 7): a `#define` continued by `\` ends on its last line.
                let end_of = |j: usize| self.declaration_end(lines, scanned, j, &search);
                // A declaration joins the group unless it is a function expression, its own unit.
                // MAGI FIX (S6 loop 13): a comment line ending in `;` joined the group; only code
                // does, as in the main scan.
                let joins = |j: usize| {
                    scanned.get(j).is_some_and(|s| !s.is_non_code)
                        && (lines.get(j).is_some_and(|l| {
                            self.is_declaration(l.trim(), scanned.get(j))
                                || self.opens_declaration_list(l.trim())
                        }) || self.opens_wrapped_prototype(lines, scanned, j, &search))
                        && self
                            .function_expression(lines, scanned, j, &search)
                            .is_none()
                };
                let mut group_end = end_of(i);
                // Scan forward for contiguous declarations, one blank line allowed between two.
                loop {
                    let next = group_end + 1;
                    if joins(next) {
                        group_end = end_of(next);
                    } else if lines.get(next).is_some_and(|l| l.trim().is_empty())
                        && joins(next + 1)
                    {
                        group_end = end_of(next + 1);
                    } else {
                        break;
                    }
                }
                let head = self.return_type_start(lines, scanned, group_start, floor, members);
                let doc_start = self.scan_back_for_docs(lines, scanned, head, floor);
                let name = self.declaration_group_name(self.strip_modifiers(trimmed, false));
                constructs.push(Construct {
                    kind: UnitKind::DeclarationGroup,
                    name,
                    start_line: line_number(doc_start)?,
                    end_line: line_number(group_end)?,
                });
                i = group_end + 1;
                continue;
            }

            i += 1;
        }

        Some(constructs)
    }

    /// The 0-based last line of the declaration at line `j`: the end of the block its value
    /// opens ([`declaration_block`]), of a `#define` continued by `\` ([`preprocessor_end`]), of
    /// a declaration list wrapped after a comma ([`Self::declaration_list_end`]), of a call
    /// its initializer opens ([`call_initializer_end`]), of the parentheses or brackets its line
    /// leaves open ([`group_end`]) or of the value after a bare `=` ([`bare_initializer_end`]),
    /// whichever is last; `j` itself otherwise. O(k) for the k lines
    /// of the declaration.
    ///
    /// A declaration that runs to its closing delimiter is one declaration however many members
    /// it holds (a Go `var (` group, a long array literal), so a declaration group too large for
    /// one batch does not split inside it ([`Self::split_declaration_group`]) and ends the review
    /// with the typed `ReviewError::TooLargeForBatch` (RV03), declared: never a silent cut.
    ///
    /// # Arguments
    ///
    /// * `lines` - The text's lines.
    /// * `scanned` - The scanner's record of each line.
    /// * `j` - The 0-based line the declaration starts on.
    /// * `search` - The bounds of the forward searches ([`BraceSearch`]).
    ///
    /// # Returns
    ///
    /// The declaration's last line, 0-based, never before `j`.
    fn declaration_end(
        &self,
        lines: &[&str],
        scanned: &[ScannedLine],
        j: usize,
        search: &BraceSearch<'_>,
    ) -> usize {
        // MAGI FIX (S6 loop 17, B2-1): a declaration that leaves a `(` or `[` open (Go's
        // `var (`, an array literal over lines, a wrapped call) ended at its first line or at a
        // brace inside it; it now runs to its matching closing delimiter.
        declaration_block(lines, scanned, j, search)
            .map_or(j, |b| b.1)
            .max(preprocessor_end(lines, j))
            .max(self.declaration_list_end(lines, scanned, j))
            .max(call_initializer_end(lines, scanned, j, search))
            .max(group_end(lines, scanned, j, search))
            // MAGI FIX (S6 loop 19, B2b-5): a declaration wrapped after a bare `=` ended on its
            // first line, and its value fell into a gap unit.
            .max(bare_initializer_end(lines, scanned, j, search))
    }

    /// The pieces of a declaration group too large for one batch: each declaration with its doc
    /// comments and attributes, and each line between two (a blank line) as a piece with the
    /// group's kind and name ([`split_into_members`]). Splitting at declarations is no silent
    /// cut (RV03): every line stays in exactly one piece. Empty when the group holds no
    /// declaration or a line number does not fit `u32`, so the caller batches it whole.
    /// O(n) in the group's lines, plus each declaration's forward search.
    fn split_declaration_group(&self, unit: &SemanticUnit) -> Vec<SemanticUnit> {
        // MAGI FIX (S6 loop 12): a group was never split, so a generated header or a constant
        // table over the batch limit refused the whole review although each declaration fits.
        let lines: Vec<&str> = unit.content.lines().collect();
        let scanned = CodeScanner::new(self.lexical_rules.clone()).scan_all(&unit.content);
        let stops = |j: usize| self.starts_construct(&lines, &scanned, j, false);
        let budget = SearchBudget::for_lines(lines.len());
        let search = BraceSearch {
            stops: &stops,
            budget: &budget,
        };
        let mut members: Vec<Member> = Vec::new();
        let mut floor = 0;
        let mut j = 0;
        while let Some(line) = lines.get(j) {
            let trimmed = line.trim();
            let declares = scanned.get(j).is_some_and(|s| !s.is_non_code)
                && (self.is_declaration(trimmed, scanned.get(j))
                    || self.opens_declaration_list(trimmed));
            if !declares {
                j += 1;
                continue;
            }
            let end = self.declaration_end(&lines, &scanned, j, &search).max(j);
            let start = self.scan_back_for_docs(&lines, &scanned, j, floor);
            let Some((first, last)) = line_number(start).zip(line_number(end)) else {
                return Vec::new();
            };
            members.push((UnitKind::DeclarationGroup, unit.name.clone(), first, last));
            floor = end + 1;
            j = end + 1;
        }
        split_into_members(unit, members)
    }

    /// The members of a grouped type declaration at line `i` (Go's `type (` ... `)`) and the
    /// group's last line (0-based, its `)`), or `None` when line `i` opens none. Each member is a
    /// construct with its doc comments: a struct, an interface (as a trait) or a defined type or
    /// alias (as a struct, as a lone `type ID int` is), named by its first word and ending on the
    /// line that closes the parentheses and brackets it opens (a wrapped `func(` type), then at
    /// the closing brace of a block that line opens, or on its own line when it opens neither,
    /// never past the group's last member line. The `type (` and `)` lines
    /// belong to no member, so they fall into gap units. O(k) for the k lines of the group.
    ///
    /// The group's end is found by counting parentheses in code context ([`code_group_delta`]),
    /// so one inside a raw string, a string or a comment moves nothing; a
    /// group that never closes runs to the end of the file, its members still found one by one,
    /// so every line stays in exactly one unit either way. A member's closing-brace search stays
    /// inside the group and spends the pass's `search` budget ([`SearchBudget`]); a member it
    /// does not close runs to the group's end. The groups of a file are disjoint, so their walks
    /// add O(n) in all. The members are `None` when a line
    /// number does not fit `u32`, which aborts detection ([`Self::try_detect_constructs`]).
    fn grouped_types(
        &self,
        lines: &[&str],
        scanned: &[ScannedLine],
        i: usize,
        search: &BraceSearch<'_>,
    ) -> Option<(usize, Option<Vec<Construct>>)> {
        if !self.block_keywords.iter().any(|k| k == "type ") {
            return None;
        }
        let rest = keyword_rest(lines.get(i)?.trim(), "type ")?;
        if !rest.trim_start().starts_with('(') {
            return None;
        }
        let mut depth = 0i64;
        let close = (i..lines.len()).find(|&j| {
            // MAGI FIX (B2b1 B1): the parentheses were counted on the line text, so a `)` in a
            // raw string cut the group short and a `(` in a comment absorbed the next function.
            depth += code_group_delta(scanned, j);
            depth <= 0
        });
        let last = close.unwrap_or(lines.len().saturating_sub(1));
        let body_end = close.unwrap_or(lines.len());
        let mut members = Some(Vec::new());
        let mut floor = i + 1;
        let mut j = i + 1;
        while j < body_end {
            let trimmed = lines.get(j).map_or("", |l| l.trim());
            let name: String = trimmed
                .chars()
                .take_while(|&c| is_identifier_char(c))
                .collect();
            if scanned.get(j).is_none_or(|s| s.is_non_code) || name.is_empty() {
                j += 1;
                continue;
            }
            let kind = go_type_kind(trimmed);
            // MAGI FIX (S6 loop 19, B2b-2): the search ran past the group's `)` to the end of
            // the file, outside the pass budget, so a file of groups whose members never close
            // inside them took O(n²) visits; it stays inside the group and spends the budget. A
            // member not closed there, or whose search the budget cut, runs to the group's end,
            // so no line of the group is lost.
            // MAGI FIX (B2b1 7): a member ended on its first line unless that line opened a
            // brace, so a wrapped `func(` type or type parameter list (`List[`) gave each later
            // line a member of its own. The member first runs to the line closing the
            // parentheses and brackets it opens ([`group_end`]), kept inside the group, then to
            // the closing brace of a block that line opens.
            let open = group_end(lines, scanned, j, search)
                .min(body_end.saturating_sub(1))
                .max(j);
            let end = if scanned.get(open).is_some_and(|s| s.brace_delta > 0) {
                find_closing_brace_within(scanned, open, body_end, || search.spend())
                    .unwrap_or(body_end.saturating_sub(1))
                    .max(open)
            } else {
                open
            };
            let start = self.scan_back_for_docs(lines, scanned, j, floor);
            let member = line_number(start)
                .zip(line_number(end))
                .map(|(start_line, end_line)| Construct {
                    kind,
                    name,
                    start_line,
                    end_line,
                });
            members = members.zip(member).map(|(mut all, m)| {
                all.push(m);
                all
            });
            floor = end + 1;
            j = end + 1;
        }
        Some((last, members))
    }

    /// The kind and name of the block or function construct a trimmed line opens, matched on
    /// the line with its visibility and qualifiers stripped ([`Self::strip_modifiers`]) or,
    /// failing that, on the line as written. Block keywords are tried before function keywords:
    /// Java's `public class` would otherwise read as a function named `class`. O(k + n) for k
    /// keywords and a line of n characters.
    fn construct_keyword(&self, trimmed: &str) -> Option<(UnitKind, String)> {
        // MAGI FIX (S6 loop 4): qualifiers combine freely (`pub(super) async unsafe fn`); the
        // tables of fixed prefixes missed most combinations.
        // MAGI FIX (S6 loop 8): `template<typename T> class Box {` was no construct at all.
        // MAGI FIX (S6 loop 10): `@Deprecated public class A {` was no construct at all.
        // MAGI FIX (S6 loop 13): `struct __attribute__((packed)) Hdr {` was named after the
        // attribute; the C attributes are read past first.
        // MAGI FIX (S6 loop 14): `#[inline] fn f() {` was no construct and its body fell into a
        // gap unit; the Rust attributes on the line are read past first.
        let trimmed = self.without_rust_attributes(trimmed);
        let attributes = self.without_c_attributes(trimmed);
        let trimmed = self.without_template_header(self.without_annotations(&attributes));
        let candidates = [self.strip_modifiers(trimmed, true), trimmed];
        candidates
            .iter()
            .find_map(|&text| {
                // MAGI FIX (S6 loop 6): `struct Foo x = {1};` is a variable of a struct type,
                // a declaration, not a struct.
                // MAGI FIX (S6 loop 7): `struct Point make_point(void) {` is a function returning
                // a struct; the signature path names it.
                self.match_block_keyword(text)
                    .filter(|(_, rest)| {
                        !(self.semicolon_declarations
                            && (declares_variable(rest) || returns_from_function(rest)))
                    })
                    .map(|(kind, rest)| (kind, construct_name(rest, text)))
            })
            .or_else(|| {
                candidates.iter().find_map(|&text| {
                    self.match_function_keyword(text)
                        .map(|rest| (UnitKind::Function, construct_name(rest, text)))
                })
            })
    }

    /// Whether a trimmed line opens a declaration: a declaration keyword starts the line as
    /// written or with its visibility stripped (`pub(crate) use`), or, with
    /// `semicolon_declarations`, it is code ending in `;` that does not open with `}`, `)`, `#` or
    /// a comment: a line opening with `)` closes a list wrapped from a line above (`);`,
    /// `) const;`), so it is never a declaration of its own. With `scanned`, "ending in `;`" is
    /// the scanner's end of statement, so a comment after the code (`static int n = 0; // note`)
    /// does not hide it, as in [`Self::signature_candidate`]; without it, the trimmed text must
    /// end in `;`. Only lines outside every construct are asked, so a statement inside a function
    /// body is never taken for one. O(k + n).
    ///
    /// # Arguments
    ///
    /// * `trimmed` - The line, trimmed.
    /// * `scanned` - The scanner's record of the line; `None` falls back to the trimmed text.
    ///
    /// # Returns
    ///
    /// Whether the line opens a declaration.
    fn is_declaration(&self, trimmed: &str, scanned: Option<&ScannedLine>) -> bool {
        // MAGI FIX (S6 loop 14): `#[allow(x)] const X: u8 = 1;` fell into a gap unit.
        let trimmed = self.without_rust_attributes(trimmed);
        // MAGI FIX (S6 loop 4): a C global with no declaration keyword fell into a gap.
        // MAGI FIX (S6 loop 6): so did one whose initializer opens on a later line
        // (`int a[] = {`, `int n =`); its value is the declaration's block or next statement.
        let semicolon = self.semicolon_declarations
            // MAGI FIX (S6 loop 16): `static int n = 0; // note` fell into a gap unit.
            && (scanned.map_or(trimmed.ends_with(';'), |s| s.ends_statement)
                || opens_initializer(trimmed))
            && !["}", ")", "#", "//", "/*", "*"]
                .iter()
                .any(|p| trimmed.starts_with(p));
        semicolon
            || self.match_declaration_keyword(trimmed)
            || self.match_declaration_keyword(self.strip_modifiers(trimmed, false))
    }

    /// Whether a trimmed line opens a C or C++ declaration list wrapped after a comma (`int x,`,
    /// `static const char *a = "x",`), for extractors with `semicolon_declarations`: code ending
    /// in `,` that does not open with `}`, `#`, a comment, `:` or `,`. Only the main scan and its
    /// declaration groups ask it, never [`Self::starts_construct`]: a constructor's initializer
    /// line (`b_{0},`) ends in `,` too, and stopping a signature's brace search there cut the
    /// function. A comment after the comma (`int x, // note`) hides it, declared: that line falls
    /// into a gap unit as before. O(n) in the line.
    fn opens_declaration_list(&self, trimmed: &str) -> bool {
        self.semicolon_declarations
            && trimmed.ends_with(',')
            && !["}", "#", "//", "/*", "*", ":", ","]
                .iter()
                .any(|p| trimmed.starts_with(p))
    }

    /// The 0-based last line of the declaration list that line `i` opens
    /// ([`Self::opens_declaration_list`]): the first later code line that ends in `;`. The walk
    /// ends earlier, on the line before, at a blank line, a line whose braces do not balance (an
    /// initializer block, which joins the group as a declaration of its own) or a line that opens
    /// a block or function construct, so the list never takes a later construct; a line ending
    /// in `,` is a declarator of the list (`g(void),`, `(*fp)(int),`), never a function header.
    /// `i` when line `i`
    /// opens no list. O(k) for the k lines walked.
    fn declaration_list_end(&self, lines: &[&str], scanned: &[ScannedLine], i: usize) -> usize {
        if !lines
            .get(i)
            .is_some_and(|l| self.opens_declaration_list(l.trim()))
        {
            return i;
        }
        let mut last = i;
        for (k, line) in lines.iter().enumerate().skip(i + 1) {
            let trimmed = line.trim();
            let Some(record) = scanned.get(k) else {
                break;
            };
            if trimmed.is_empty() {
                break;
            }
            if record.is_non_code {
                continue;
            }
            if record.ends_statement {
                return k;
            }
            // MAGI FIX (B2b1 8): a list line holding a function declarator (`g(void),`) read as a
            // signature and ended the list before it; a line that ends in `,` continues the list,
            // as no function header does (its parameters would close first).
            if record.brace_delta != 0
                || self.construct_keyword(trimmed).is_some()
                || (!trimmed.ends_with(',')
                    && self.signature_name(line, Some(record), false).is_some())
            {
                break;
            }
            last = k;
        }
        last
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

    /// `trimmed` without the C++ template header it opens with ([`strip_template_header`]), for
    /// extractors with `template_headers`; `trimmed` otherwise. O(n) in the line.
    fn without_template_header<'t>(&self, trimmed: &'t str) -> &'t str {
        if self.template_headers {
            strip_template_header(trimmed)
        } else {
            trimmed
        }
    }

    /// `trimmed` without the complete annotations it opens with (`@Override`,
    /// `@SuppressWarnings("x")`, `@Component({ a: 1 })`), for extractors whose
    /// `attribute_prefixes` hold `@` (Java, JavaScript, TypeScript); `trimmed` otherwise.
    ///
    /// An annotation is `@`, a name of identifier characters and dots, and optionally its
    /// arguments in balanced parentheses on the same line. `@interface` declares an annotation
    /// type and is kept. An annotation whose arguments do not close on the line is kept with
    /// everything after it, so the line still opens with `@` and is no signature, as before.
    /// Parentheses inside the arguments' strings do not count ([`balanced_end_outside_quotes`]
    /// with [`ANNOTATION_QUOTES`]), so `@Component(")")` closes at its last `)`; a string that
    /// never closes on the line keeps the annotation, as above. O(n) in the line.
    ///
    /// # Arguments
    ///
    /// * `trimmed` - A trimmed line.
    ///
    /// # Returns
    ///
    /// The rest of the line after its leading annotations, trimmed at its start.
    fn without_annotations<'t>(&self, trimmed: &'t str) -> &'t str {
        if !self.attribute_prefixes.iter().any(|p| p == "@") {
            return trimmed;
        }
        let mut rest = trimmed;
        while let Some(after_at) = rest.strip_prefix('@') {
            let name_len: usize = after_at
                .chars()
                .take_while(|&c| is_identifier_char(c) || c == '.')
                .map(char::len_utf8)
                .sum();
            let (name, mut tail) = after_at.split_at(name_len);
            if name.is_empty() || name == "interface" {
                return rest;
            }
            if tail.starts_with('(') {
                // MAGI FIX (S6 loop 14): `@Component(")")` closed at the quoted `)` and the
                // class after it was no unit; the quoted parentheses no longer count.
                let Some(close) = balanced_end_outside_quotes(tail, '(', ')', &ANNOTATION_QUOTES)
                else {
                    return rest;
                };
                tail = tail.get(close..).unwrap_or("");
            }
            rest = tail.trim_start();
        }
        rest
    }

    /// `trimmed` without the complete Rust outer attributes it opens with (`#[inline]`,
    /// `#[derive(Debug)] #[cfg(test)]`), for extractors whose `attribute_prefixes` hold
    /// [`RUST_ATTRIBUTE_PREFIX`] (Rust); `trimmed` otherwise.
    ///
    /// An attribute is `#[` up to its matching `]`, read past its double-quoted strings
    /// ([`balanced_end_outside_quotes`] with [`RUST_ATTRIBUTE_QUOTES`]), so `#[doc = "]"]` closes
    /// at its last `]`. An attribute that does not close on the line is kept with everything
    /// after it, so the line still opens with `#[` and reads as before: an attribute line,
    /// joined to the item below it ([`Self::scan_back_for_docs`]). A raw string holding a `"`
    /// (`#[doc = r#"a"b"#]`) may miscount; the line then opens with no keyword and its item falls
    /// into a gap unit, as before this rule, so no line is lost. O(n) in the line.
    ///
    /// # Arguments
    ///
    /// * `trimmed` - A trimmed line.
    ///
    /// # Returns
    ///
    /// The rest of the line after its leading attributes, trimmed at its start.
    fn without_rust_attributes<'t>(&self, trimmed: &'t str) -> &'t str {
        if !self
            .attribute_prefixes
            .iter()
            .any(|p| p == RUST_ATTRIBUTE_PREFIX)
        {
            return trimmed;
        }
        let mut rest = trimmed;
        while rest.starts_with(RUST_ATTRIBUTE_PREFIX) {
            let attribute = rest.get(1..).unwrap_or("");
            let Some(end) =
                balanced_end_outside_quotes(attribute, '[', ']', &RUST_ATTRIBUTE_QUOTES)
            else {
                return rest;
            };
            rest = attribute.get(end..).unwrap_or("").trim_start();
        }
        rest
    }

    /// `trimmed` without its C and C++ attributes ([`strip_c_attributes`]), for extractors with
    /// `semicolon_declarations` (C and C++); `trimmed` otherwise. O(n) in the line.
    fn without_c_attributes<'t>(&self, trimmed: &'t str) -> Cow<'t, str> {
        if self.semicolon_declarations {
            strip_c_attributes(trimmed)
        } else {
            Cow::Borrowed(trimmed)
        }
    }

    /// Whether a trimmed line opens a Rust `macro_rules!` definition ([`MACRO_RULES_KEYWORD`]),
    /// its outer attributes read past (`#[macro_export] macro_rules! m {`). Only the Rust
    /// extractor holds the keyword in its function table, so no other language matches. O(n).
    fn opens_macro_rules(&self, trimmed: &str) -> bool {
        self.function_keywords
            .iter()
            .any(|k| k == MACRO_RULES_KEYWORD)
            && keyword_rest(self.without_rust_attributes(trimmed), MACRO_RULES_KEYWORD).is_some()
    }

    /// The text after the function keyword `trimmed` opens with ([`keyword_rest`]), if any, or,
    /// with `anonymous_functions`, after the keyword's bare word glued to a `(`
    /// (`function() {` gives `() {`).
    fn match_function_keyword<'t>(&self, trimmed: &'t str) -> Option<&'t str> {
        self.function_keywords.iter().find_map(|kw| {
            keyword_rest(trimmed, kw).or_else(|| {
                let word = kw.strip_suffix(' ')?;
                trimmed
                    .strip_prefix(word)
                    .filter(|rest| self.anonymous_functions && rest.starts_with('('))
            })
        })
    }

    /// Whether line `j` (0-based) is code that opens a construct: a function, block or
    /// declaration keyword, or a signature as the main scan reads it
    /// ([`Self::line_signature`]). O(k) in the keywords.
    fn starts_construct(
        &self,
        lines: &[&str],
        scanned: &[ScannedLine],
        j: usize,
        members: bool,
    ) -> bool {
        if scanned.get(j).is_none_or(|s| s.is_non_code) {
            return false;
        }
        let Some(line) = lines.get(j) else {
            return false;
        };
        let trimmed = line.trim();
        // MAGI FIX (S6 loop 18, B2b-9): the stop rule ruled out a reserved word unlike the main
        // scan, so a member with no `;` (`abstract foo(): void`) took the body of the
        // `delete() {` after it; both read a signature through `line_signature`.
        self.construct_keyword(trimmed).is_some()
            || self.is_declaration(trimmed, scanned.get(j))
            || self.line_signature(line, scanned, j, members).is_some()
    }

    /// The function name of the signature on line `j` (0-based), as the main scan reads it:
    /// [`Self::signature_name`], or, for a line directly in a class body of a language with
    /// `reserved_word_methods`, [`Self::signature_candidate`], where a reserved word names a
    /// method (`delete() {`). Shared by the main scan and the forward searches' stop rule
    /// ([`Self::starts_construct`]), so the two never disagree on where a construct starts. O(n)
    /// in the line.
    ///
    /// The depth of a block body's own lines is the depth before its first line: a line at that
    /// depth is a direct member, one deeper sits inside a member (a static block).
    ///
    /// # Arguments
    ///
    /// * `line` - Line `j`, untrimmed.
    /// * `scanned` - The scanner's record of each line of the text.
    /// * `j` - The 0-based line.
    /// * `members` - Whether the lines are a block's body.
    ///
    /// # Returns
    ///
    /// The function name, or `None` when the line is no signature.
    // MAGI FIX (S6 loop 14): `delete() {` in a JavaScript class was no member; directly in a
    // class body a reserved word names a method.
    fn line_signature(
        &self,
        line: &str,
        scanned: &[ScannedLine],
        j: usize,
        members: bool,
    ) -> Option<String> {
        let depth_before = |k: usize| {
            scanned
                .get(k)
                .map(|s| i64::from(s.depth_after) - i64::from(s.brace_delta))
        };
        let prototypes = members && self.member_prototypes();
        if members && self.reserved_word_methods && depth_before(j) == depth_before(0) {
            self.signature_candidate(line, scanned.get(j), prototypes)
        } else {
            self.signature_name(line, scanned.get(j), prototypes)
        }
    }

    /// The function name of a C-family signature line, for extractors with
    /// `signature_functions`: a line that holds `(`, whose code does not end in `;` (a prototype;
    /// with `scanned`, a comment after the code does not count), does not
    /// open with `#` (but for `computed_member_names`, where it opens a private name), a comment, `}`, `@`, `,` or a single `:` (a C++ constructor's initializer line,
    /// `: value_(v) {`, continues the signature above it), has no `=` before its `(`, and names an identifier,
    /// not a word of its `not_function_names`, just before the `(` (`int *parse(` gives `parse`, `void Foo::bar(` gives
    /// `bar`). The line may be indented: a function inside a namespace, a class or an
    /// `extern "C"` block is a function too. Only lines outside every construct are asked, so a
    /// call inside a recognised function body is never taken for one. O(n) in the line.
    ///
    /// A C++ template header is read past first ([`Self::without_template_header`]): alone on its
    /// line (`template <typename T>`) it is no signature, since nothing follows it, so it never
    /// names a function after its last parameter; it joins the construct below as one of the
    /// extractor's `attribute_prefixes`.
    ///
    /// # Arguments
    ///
    /// * `line` - The line, untrimmed.
    /// * `scanned` - The scanner's record of the line; `None` falls back to the trimmed text.
    /// * `prototypes` - Whether a line whose code ends in `;` is a signature too: an abstract
    ///   member of an interface or class body ([`Self::member_prototypes`]).
    ///
    /// # Returns
    ///
    /// The function name, or `None` when the line is no signature.
    fn signature_name(
        &self,
        line: &str,
        scanned: Option<&ScannedLine>,
        prototypes: bool,
    ) -> Option<String> {
        // MAGI FIX (S6 loop 4): `catch (e) {` read as a function named `catch`.
        self.signature_candidate(line, scanned, prototypes)
            .filter(|name| !self.not_function_names.contains(&name.as_str()))
    }

    /// The name [`Self::signature_name`] reads on a signature line before it rules out the words
    /// of `not_function_names`: the same rules and arguments, with a reserved word kept as a
    /// name. Only a direct member of a JavaScript or TypeScript class body is read with it
    /// (`reserved_word_methods`), where `delete() {` declares a method. O(n) in the line.
    ///
    /// # Arguments
    ///
    /// * `line` - The line, untrimmed.
    /// * `scanned` - The scanner's record of the line; `None` falls back to the trimmed text.
    /// * `prototypes` - Whether a line whose code ends in `;` is a signature too.
    ///
    /// # Returns
    ///
    /// The identifier before the line's `(`, or `None` when the line is no signature.
    fn signature_candidate(
        &self,
        line: &str,
        scanned: Option<&ScannedLine>,
        prototypes: bool,
    ) -> Option<String> {
        // MAGI FIX (S6 loop 3): indentation no longer disqualifies a signature.
        if !self.signature_functions {
            return None;
        }
        // MAGI FIX (S6 loop 10): `@Override public void run() {` opened with `@` and was no
        // signature; the complete annotations on the line are read past first.
        let trimmed = self.without_annotations(line.trim());
        // MAGI FIX (S6 loop 4): a trailing comment ending in `;` is not a prototype's `;`.
        let prototype = scanned.map_or(trimmed.ends_with(';'), |s| s.ends_statement);
        // An annotation or decorator (`@SuppressWarnings("x")`) is no signature either.
        // MAGI FIX (S6 loop 7): a `*` opens a comment's continuation line only when the line
        // starts inside the comment; in code it is a generator method (`*items() {`).
        let comment_star = trimmed.starts_with('*') && scanned.is_none_or(|s| s.starts_in_literal);
        // MAGI FIX (S6 loop 9): a C++ constructor's initializer line (`: value_(v) {`,
        // `, count_(0)`) read as a signature named after the member, so the constructor's brace
        // search stopped there; it continues the signature above it.
        let initializer =
            trimmed.starts_with(',') || (trimmed.starts_with(':') && !trimmed.starts_with("::"));
        // MAGI FIX (S6 loop 17, B2-3): a private method (`#m() {`) opened with `#` and was read
        // as a preprocessor line; in JavaScript and TypeScript `#` opens a private name.
        let directive = trimmed.starts_with('#') && !self.computed_member_names;
        if (prototype && !prototypes)
            || comment_star
            || initializer
            || directive
            || ["//", "/*", "}", "@"]
                .iter()
                .any(|p| trimmed.starts_with(p))
        {
            return None;
        }
        // MAGI FIX (S6 loop 8): `template <void (*F)(int)> void call() {` was named `void`; a
        // header alone on its line (`template <typename T>`) is no signature: nothing follows it.
        // MAGI FIX (S6 loop 13): `__attribute__((noreturn)) void die(void) {` was a function
        // named `__attribute__`; the C attributes are read past first.
        let attributes = self.without_c_attributes(trimmed);
        let text = self.without_template_header(&attributes);
        self.operator_overloads
            .then(|| operator_overload_name(text))
            .flatten()
            .or_else(|| {
                self.computed_member_names
                    .then(|| private_or_computed_name(text))
                    .flatten()
            })
            .or_else(|| identifier_before_paren(text))
    }

    /// The kind of the block `trimmed` opens and the text after its keyword
    /// ([`keyword_rest`]), if it opens one.
    fn match_block_keyword<'t>(&self, trimmed: &'t str) -> Option<(UnitKind, &'t str)> {
        for kw in &self.block_keywords {
            if let Some(rest) = keyword_rest(trimmed, kw) {
                // MAGI FIX (S6 loop 5): the tables hold bare keywords and the modifiers are
                // stripped before they are matched, so the `pub struct` and `export class` arms
                // could never match; a bare keyword names the kind.
                let kind = match kw.trim() {
                    // MAGI FIX (S6 loop 13): `type Reader interface {` was a struct.
                    "type" => go_type_kind(rest),
                    "struct" | "union" => UnitKind::Struct,
                    "enum" => UnitKind::Enum,
                    "trait" | "interface" => UnitKind::Trait,
                    "impl" => UnitKind::Impl,
                    // MAGI FIX (S6 loop 6): a module was function-like, so it never split.
                    // MAGI FIX (B2a 1): a C++ namespace holds free functions, as a module does.
                    "mod" | "namespace" => UnitKind::Module,
                    _ => UnitKind::Class, // class, record, @interface
                };
                // MAGI FIX (S6 loop 7): C++'s `enum class Color` was named `class`.
                let rest = if kind == UnitKind::Enum {
                    ["class ", "struct "]
                        .iter()
                        .find_map(|w| rest.strip_prefix(w))
                        .unwrap_or(rest)
                } else {
                    rest
                };
                return Some((kind, rest));
            }
        }
        None
    }

    /// Whether the line opens with a declaration keyword followed by any blank or by the list,
    /// path or string it declares ([`declaration_keyword_rest`]).
    fn match_declaration_keyword(&self, trimmed: &str) -> bool {
        // MAGI FIX (S6 loop 14): `use::{a, b};`, `import{foo} from "x";` and
        // `#include<stdio.h>` fell into gap units.
        self.declaration_keywords
            .iter()
            .any(|kw| declaration_keyword_rest(trimmed, kw).is_some())
    }

    /// The 0-based first line of the construct whose declaration is at `idx`: `idx` moved up
    /// over the doc comments and attributes right above it. A doc line opens with a doc prefix
    /// (`///`, `/**`); a block comment that spans lines (`/**` ... `*/`) counts whole when its
    /// opening line does ([`Self::block_doc_start`]). O(k) for the k lines passed over.
    ///
    /// The walk never goes above `floor`, the 0-based line after the previous construct: a
    /// one-line construct may itself open with an attribute prefix (Java's
    /// `@Override public String toString() { .. }`, C++'s `template<typename T> void f() { .. }`),
    /// and taking it would make two construct ranges overlap, which `ChangedLines::assign` and
    /// `split_into_members` rule out. `floor` is also returned when a multi-line comment or
    /// attribute recovered below it. O(k) for the k lines passed over, at most `idx - floor`
    /// plus the lines of one such comment or attribute.
    fn scan_back_for_docs(
        &self,
        lines: &[&str],
        scanned: &[ScannedLine],
        idx: usize,
        floor: usize,
    ) -> usize {
        let mut start = idx;
        while start > floor {
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
            } else if let Some(open) = self
                .block_doc_start(lines, scanned, start - 1)
                .or_else(|| self.multi_line_attribute_start(lines, scanned, start - 1))
            {
                start = open;
            } else {
                break;
            }
        }
        start.max(floor)
    }

    /// The 0-based opening line of the doc block comment that ends on line `last` (its `*/`):
    /// the lines above it that start inside the comment are passed over up to the line that
    /// opens it, which must open with a doc prefix (`/**`). `None` when `last` ends no block
    /// comment or the comment is no doc comment (`/* plain */`). O(k) for the k lines of the
    /// comment.
    fn block_doc_start(
        &self,
        lines: &[&str],
        scanned: &[ScannedLine],
        last: usize,
    ) -> Option<usize> {
        // MAGI FIX (S6 loop 5): the line above a declaration is `*/` for a multi-line JavaDoc,
        // JSDoc or C doc block, so the whole comment was left out of the unit.
        if !lines.get(last)?.trim_end().ends_with("*/") {
            return None;
        }
        let mut open = last;
        while scanned.get(open)?.starts_in_literal {
            open = open.checked_sub(1)?;
        }
        let opener = lines.get(open)?.trim();
        (open < last
            && self
                .doc_comment_prefixes
                .iter()
                .any(|p| p.starts_with("/*") && opener.starts_with(p.as_str())))
        .then_some(open)
    }

    /// The 0-based opening line of the attribute that spans lines and ends on line `last`
    /// (Rust's `#[serde(` ... `)]`): the lines above `last` are walked up while its square
    /// brackets stay unbalanced, up to the line that balances them, which must open with an
    /// attribute prefix ending in `[` (`#[`). `None` when `last` does not end in `]`, the
    /// extractor has no such prefix, `last` is not code, or the line that balances
    /// the brackets opens no attribute (a macro call `list! [` ... `]`).
    ///
    /// A line on the way that lies wholly inside a string or a comment (a text block, a template
    /// literal in the arguments) is passed over: it holds no bracket of the attribute.
    ///
    /// Brackets are counted on the line text: a bracket
    /// inside a string may miscount, and the attribute then falls into a gap unit as before, so
    /// no line is lost. O(k) for the k lines walked.
    ///
    /// An annotation of Java, JavaScript or TypeScript whose arguments wrap over lines
    /// (`@SuppressWarnings(` ... `)`, `@Component({` ... `})`) is recovered the same way on its
    /// parentheses: `last` ends in `)` and the line that balances them opens with an attribute
    /// prefix that does not end in `[` (`@`).
    fn multi_line_attribute_start(
        &self,
        lines: &[&str],
        scanned: &[ScannedLine],
        last: usize,
    ) -> Option<usize> {
        // MAGI FIX (S6 loop 10): the line above an item under a multi-line attribute is `)]`,
        // so the attribute was cut from the item it controls and reviewed as a gap unit.
        // MAGI FIX (S6 loop 12): a wrapped `@` annotation ends in `)` or `})`, which the `[`
        // rule never matched, so it was cut from its member too.
        let (open, close, bracket_prefix) = match lines.get(last)?.trim_end().chars().last()? {
            ']' => ('[', ']', true),
            ')' => ('(', ')', false),
            _ => return None,
        };
        let opens_attribute = |text: &str| {
            self.attribute_prefixes
                .iter()
                .any(|p| p.ends_with('[') == bracket_prefix && text.starts_with(p.as_str()))
        };
        let mut unclosed = 0i64;
        let mut j = last;
        loop {
            let record = scanned.get(j)?;
            // MAGI FIX (S6 loop 19, B2b-4): a line wholly inside a string of the arguments (a
            // Java text block, a template literal) ended the walk, so the annotation was cut from
            // its member; such a line (or one wholly inside a comment) holds no bracket and is
            // passed over. The line the walk starts on must still be code.
            if record.is_non_code {
                if j == last {
                    return None;
                }
                j = j.checked_sub(1)?;
                continue;
            }
            let line = lines.get(j)?;
            unclosed += line.chars().fold(0, |n, c| {
                if c == close {
                    n + 1
                } else if c == open {
                    n - 1
                } else {
                    n
                }
            });
            if unclosed <= 0 {
                return (unclosed == 0 && j < last && opens_attribute(line.trim())).then_some(j);
            }
            j = j.checked_sub(1)?;
        }
    }

    /// The name and last line (0-based) of the function expression the declaration at line `i`
    /// assigns (`const f = () => {`, `let g = async function (x) {`, an arrow whose parameters
    /// wrap over several lines): a `const`, `let` or `var` whose value is a `function` or whose
    /// lines up to the block's opening brace hold `=>`, and whose value opens a block
    /// ([`declaration_block`]). O(k) for the k lines of its header and body.
    fn function_expression(
        &self,
        lines: &[&str],
        scanned: &[ScannedLine],
        i: usize,
        search: &BraceSearch<'_>,
    ) -> Option<(String, usize)> {
        let text = self.strip_modifiers(lines.get(i)?.trim(), false);
        let rest = FUNCTION_EXPRESSION_KEYWORDS
            .iter()
            .find_map(|kw| blank_keyword_rest(text, kw))?;
        let name: String = rest
            .chars()
            .take_while(|&c| is_identifier_char(c))
            .collect();
        let value = rest.split_once('=')?.1.trim_start();
        let (open, close) = declaration_block(lines, scanned, i, search)?;
        let arrow = lines.get(i..=open)?.iter().any(|l| l.contains("=>"));
        let function = opens_function_value(value);
        (!name.is_empty() && (arrow || function)).then_some((name, close))
    }

    /// Whether a signature ending in `;` is a member of a block body: in Java, JavaScript and
    /// TypeScript an interface or abstract class declares its methods so (`User find(long id);`).
    /// C and C++ read such a line as a declaration instead (`semicolon_declarations`), which
    /// groups the prototypes of a class body. O(1).
    fn member_prototypes(&self) -> bool {
        self.signature_functions && !self.semicolon_declarations
    }

    /// Whether line `j` opens a C or C++ prototype wrapped over lines (`int foo(` ...
    /// `int b);`), for extractors with `semicolon_declarations`: a code line that names a
    /// function ([`Self::signature_name`]), leaves a parenthesis open in code context
    /// ([`ScannedLine::group_delta`]) and whose list closes on a line whose code ends in `;`
    /// ([`group_end`]). Such a line opens no declaration by [`Self::is_declaration`] (its code
    /// ends in `(` or in a parameter), so without this rule its first lines fell into a gap and
    /// the closing `);` read as a declaration of its own. O(k) for the k lines of the list, from
    /// the pass's [`SearchBudget`].
    ///
    /// # Arguments
    ///
    /// * `lines` - The text's lines.
    /// * `scanned` - The scanner's record of each line.
    /// * `j` - The 0-based line asked about.
    /// * `search` - The bounds of the forward searches ([`BraceSearch`]).
    ///
    /// # Returns
    ///
    /// `true` when line `j` opens a wrapped prototype.
    fn opens_wrapped_prototype(
        &self,
        lines: &[&str],
        scanned: &[ScannedLine],
        j: usize,
        search: &BraceSearch<'_>,
    ) -> bool {
        let Some(record) = scanned.get(j).filter(|s| !s.is_non_code) else {
            return false;
        };
        if !self.semicolon_declarations
            || record.group_delta <= 0
            || !lines
                .get(j)
                .is_some_and(|l| self.signature_name(l, Some(record), false).is_some())
        {
            return false;
        }
        let close = group_end(lines, scanned, j, search);
        close > j && scanned.get(close).is_some_and(|s| s.ends_statement)
    }

    /// The 0-based first line of the construct whose signature is on line `i`: the line above
    /// it when that line holds only its return type, written GNU style on a line of its own
    /// (`static int` over `foo (int a)`), else `i`. Only for extractors with
    /// `semicolon_declarations` and outside a block body (`members`), where a line of type words
    /// alone (identifier characters, blanks, `*`, `&`, `<`, `>` and `::`, not ending in `:`) is
    /// no construct of its own: it fell into a gap unit, split from its function or prototype.
    /// Never above `floor`, the line after the previous construct. O(n) in the two lines.
    ///
    /// # Arguments
    ///
    /// * `lines` - The text's lines.
    /// * `scanned` - The scanner's record of each line.
    /// * `i` - The 0-based line of the signature.
    /// * `floor` - The lowest 0-based line the construct may start on.
    /// * `members` - Whether the lines are a block's body.
    ///
    /// # Returns
    ///
    /// `i - 1` when that line is the signature's return type, `i` otherwise.
    fn return_type_start(
        &self,
        lines: &[&str],
        scanned: &[ScannedLine],
        i: usize,
        floor: usize,
        members: bool,
    ) -> usize {
        let Some(above) = i.checked_sub(1).filter(|&a| a >= floor) else {
            return i;
        };
        let code = |k: usize| {
            scanned
                .get(k)
                .is_some_and(|s| !s.is_non_code && !s.starts_in_literal)
        };
        let type_words = lines.get(above).map(|l| l.trim()).is_some_and(|t| {
            !t.is_empty()
                && !t.ends_with(':')
                && t.chars()
                    .all(|c| is_identifier_char(c) || RETURN_TYPE_CHARS.contains(c))
        });
        let signature = lines
            .get(i)
            .is_some_and(|l| self.signature_name(l, scanned.get(i), true).is_some());
        if self.semicolon_declarations && !members && code(above) && type_words && signature {
            above
        } else {
            i
        }
    }

    /// Generate a descriptive name for a declaration group from its first line, with the
    /// visibility words already stripped ([`Self::strip_modifiers`]): only bare keywords can
    /// match here, so `pub use` reads as `use` and a JavaScript `export { a };` has no name of
    /// its own (`declarations`).
    fn declaration_group_name(&self, first_line: &str) -> String {
        // MAGI FIX (S6 loop 12): the `pub ...` and `export ` arms could never match, since the
        // caller strips the visibility words first.
        let trimmed = first_line.trim();
        let opens = |kw: &str| declaration_keyword_rest(trimmed, kw).is_some();
        if opens("use ") || opens("import ") || opens("from ") {
            "imports".to_string()
        } else if opens("const ") {
            "constants".to_string()
        } else if opens("static ") {
            "statics".to_string()
        } else if opens("#define ") {
            "defines".to_string()
        } else if opens("typedef ") {
            "typedefs".to_string()
        } else if opens("extern ") {
            "externs".to_string()
        } else if opens("type ") {
            "type aliases".to_string()
        } else if opens("let ") || opens("var ") {
            "variables".to_string()
        } else if opens("package ") {
            "package".to_string()
        } else {
            "declarations".to_string()
        }
    }
}

/// The openers of a comment that make a line before a construct no wrapped continuation of
/// the signature above it, though it ends in `,` ([`open_brace_line`]).
const LINE_OPENING_COMMENTS: [&str; 2] = ["//", "/*"];

/// The characters besides identifier ones that a GNU-style return type line may hold
/// (`static const char *`, `std::vector<int> &`), read by
/// [`BraceFamilyExtractor::return_type_start`].
const RETURN_TYPE_CHARS: &str = " 	*&<>:";

/// The most lines past a construct's first line that [`open_brace_line`] looks at for its
/// opening brace. A starting point, not a measurement: a signature wrapped over more lines
/// than this (a generated parameter list) is read as having no brace, so its lines fall into
/// later units, reviewed but split from their signature. It bounds one search; the searches of a
/// whole detection pass share [`SearchBudget`], O(n) line visits for a text of n lines.
///
/// Why this value: rustfmt and clang-format put one parameter per line, so a hand-written
/// signature spans tens of lines at most; 1 024 is two orders of magnitude above that, so only
/// generated code reaches it, and it keeps one search over a pathological text short (a few
/// thousand line visits). It is a fixed constant, not a setting: no operator knob changes how
/// a file is cut into units, so the same file always gives the same units. Raising it only
/// moves where a generated list is split from its body; lowering it splits real signatures.
// MAGI FIX (S6 loop 14): why the value, and why it is no setting.
const MAX_BRACE_SEARCH_LINES: usize = 1_024;

/// What a forward brace search of one detection pass is bounded by: where the next construct
/// starts (`stops`, [`BraceFamilyExtractor::starts_construct`]) and the line visits the pass may
/// still spend ([`SearchBudget`]).
struct BraceSearch<'s> {
    /// Whether a line (0-based) starts the next construct.
    stops: &'s dyn Fn(usize) -> bool,
    /// The line visits left to the whole detection pass.
    budget: &'s SearchBudget,
}

impl BraceSearch<'_> {
    /// Whether line `j` (0-based) starts the next construct. O(k) in the keywords.
    fn stops(&self, j: usize) -> bool {
        (self.stops)(j)
    }

    /// Spends one line visit of the pass ([`SearchBudget::spend`]).
    fn spend(&self) -> bool {
        self.budget.spend()
    }
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

/// The 0-based line holding the opening brace of the construct that starts at line `i`: line
/// `i` when it opens one, else the first later line that does, so a signature wrapped over any
/// number of lines (rustfmt's `fn f(\n    a: u8,\n) -> u8 {`) finds its brace.
///
/// The search ends without a brace when line `i` already ends the declaration (its code ends in
/// `;`) or is a one-line construct (its code ends in `}` with nothing left open: `fn a() { x(); }`,
/// `fn f() -> u8 { 1 }`), so a one-line construct never takes the brace of the block after it.
/// Otherwise it ends before the first later line that is blank, is code ending in `;` (a
/// statement: the declaration had no block; a comment ending in `;` does not count), opens a
/// statement with a keyword of [`STATEMENT_KEYWORDS`] (`if (x) {`: a signature's brace is never
/// there), or where `stops` holds (the next construct's start) while no parenthesis opened since
/// line `i` is still open and the line before it is no code ending in `,` (a C++ constructor's
/// initializer list wrapped after a comma, `count_(0) {`, continues the declaration; a comment
/// line ending in `,` does not, while a trailing comment after the code, `int f() // a,`, still
/// reads as one, declared); comment lines are passed over.
/// O(k) for the k lines scanned, which never pass the next construct.
///
/// Inside an open parameter list `stops` is not asked: a parameter line holding a `(` after a
/// word (`int (*callback)(int),`, `std::function<void(int)> cb,`) reads like a signature of its
/// own, and stopping there cut the function at that line. The list counts as open while a
/// parenthesis or bracket opened since line `i` is still open, counted by the scanner in code
/// context only ([`ScannedLine::group_delta`]), so a `(` inside a string or a comment
/// (`func f(x int) // see (`) holds nothing open and the next construct keeps its own brace.
///
/// A blank line does not end the search inside an open parameter list: while a parenthesis or
/// bracket opened since line `i` is still open, counted by the scanner in code context only
/// ([`ScannedLine::group_delta`]), it is passed over, so a blank line between two parameters
/// keeps the body with its signature, whatever comment comes before it (`a: u8, // first`, a
/// comment line). A delimiter inside a string, a character literal or a comment does not count
/// (`fn p() -> char { '(' }` leaves nothing open), so the blank line after such a line still
/// ends the search and the next construct keeps its own brace.
///
/// A brace on a parameter line is taken for the construct's brace, declared: a default argument
/// that opens a lambda body on that line (`void f(int cb = []() {`) or an anonymous struct
/// parameter makes the function end at that parameter's closing brace, and the rest of its body
/// falls into later units. No line is lost: every line stays in exactly one unit, so the body is
/// still reviewed, only split from its signature.
///
/// The search looks at most [`MAX_BRACE_SEARCH_LINES`] lines past line `i`, and no more than the
/// detection pass has left ([`SearchBudget`]), and then degrades to "no brace found", which
/// already means the construct ends on its own line: the lines after it fall into later units,
/// so none is lost. Preparation runs before the review clock (RV06), so these two bounds are
/// the only ones on a file built to keep a parameter list open.
fn open_brace_line(
    lines: &[&str],
    scanned: &[ScannedLine],
    i: usize,
    search: &BraceSearch<'_>,
) -> Option<usize> {
    let opens = |idx: usize| scanned.get(idx).is_some_and(|s| s.brace_delta > 0);
    let is_code = |idx: usize| scanned.get(idx).is_some_and(|s| !s.is_non_code);
    // MAGI FIX (S6 loop 4): a comment ending in `;` (`// keeps state;`) is not a statement.
    let ends_statement =
        |idx: usize| is_code(idx) && scanned.get(idx).is_some_and(|s| s.ends_statement);
    if opens(i) {
        return Some(i);
    }
    // A line whose code ends in `}` after opening nothing net (`fn f() -> u8 { 1 }`) is a
    // one-line construct: searching on took the brace of the block after it.
    let ends_block = |idx: usize| is_code(idx) && scanned.get(idx).is_some_and(|s| s.ends_block);
    if ends_statement(i) || ends_block(i) {
        return None;
    }
    // MAGI FIX (S6 loop 3): no fixed two-line cap; the declaration's own end bounds the search.
    // MAGI FIX (S6 loop 6): a top-level `foo()` took the brace of the `if (x) {` after it.
    // MAGI FIX (S6 loop 7): a parameter line inside the open list no longer stops the search.
    // MAGI FIX (S6 loop 12): inside an open parameter list the search ran to the end of the
    // file, quadratic over a file of `f(a` lines and outside any clock; it is now bounded.
    let budget_end = i
        .saturating_add(1)
        .saturating_add(MAX_BRACE_SEARCH_LINES)
        .min(lines.len());
    // MAGI FIX (S6 loop 14): a blank line between two parameters ended the search, so the body
    // was cut from its signature; inside an open list it is passed over.
    // MAGI FIX (S6 loop 17, B2-4): the list counted as open only after a code line ending in `,`
    // or `(`, so a comment before the blank line ended the search; the scanner's count of the
    // delimiters in code, which no comment or string moves, now decides.
    let group_open = |idx: usize| scanned.get(idx).map_or(0, |s| i64::from(s.group_delta));
    let mut list_open = group_open(i);
    for j in (i + 1)..budget_end {
        if !search.spend() {
            return None;
        }
        if lines.get(j).is_some_and(|l| l.trim().is_empty()) {
            if list_open > 0 {
                continue;
            }
            return None;
        }
        // The list still open before line `j`, in code context: a `(` inside a comment or a
        // string (`func f(x int) // see (`) held it open and the next function's brace was taken.
        let open_before = list_open;
        list_open += group_open(j);
        // MAGI FIX (S6 loop 9): a constructor initializer after a `,` (`count_(0) {`) read as
        // the next construct; a line after one ending in `,` continues the declaration.
        // MAGI FIX (S6 loop 11): a comment line ending in `,` (`// note,`) between two signatures
        // continued the first one into the second, which took the second's brace; the line before
        // must be code (scanner record) and open with no comment.
        let continues = j.checked_sub(1).is_some_and(|p| {
            is_code(p)
                && lines.get(p).is_some_and(|l| {
                    let t = l.trim();
                    t.ends_with(',') && !LINE_OPENING_COMMENTS.iter().any(|c| t.starts_with(c))
                })
        });
        let ends_search = (open_before <= 0 && !continues && search.stops(j))
            || lines.get(j).is_some_and(|l| opens_statement(l.trim()));
        if ends_search {
            return None;
        }
        if opens(j) {
            return Some(j);
        }
        if ends_statement(j) {
            return None;
        }
    }
    None
}

/// The parentheses and square brackets line `idx` opens minus those it closes, counted by the
/// scanner in code context only ([`ScannedLine::group_delta`]): one inside a string, a raw
/// string, a character literal or a comment counts nothing. 0 for a line that does not exist.
/// O(1).
fn code_group_delta(scanned: &[ScannedLine], idx: usize) -> i64 {
    scanned.get(idx).map_or(0, |s| i64::from(s.group_delta))
}

/// Keywords that open a statement, never the rest of a declaration: the brace of a line that
/// opens with one belongs to that statement.
const STATEMENT_KEYWORDS: [&str; 10] = [
    "if", "else", "for", "while", "do", "switch", "try", "catch", "finally", "return",
];

/// Whether a trimmed line opens with a keyword of [`STATEMENT_KEYWORDS`] as a whole word
/// (`if (x) {`, `else {`; not `iffy()`). O(1): the keywords are short.
fn opens_statement(trimmed: &str) -> bool {
    STATEMENT_KEYWORDS.iter().any(|kw| {
        trimmed.strip_prefix(kw).is_some_and(|rest| {
            !rest.starts_with(|c: char| c.is_alphanumeric() || c == '_' || c == '$')
        })
    })
}

/// The words that open a JavaScript or TypeScript variable declaration, whose value may be a
/// function expression.
const FUNCTION_EXPRESSION_KEYWORDS: [&str; 3] = ["const ", "let ", "var "];

/// What a declaration line may end with when its value goes on to the next line (`const f = (`,
/// `const t = [`, `int a,`).
const DECLARATION_CONTINUATIONS: [char; 4] = ['=', '(', '[', ','];

/// The arrow a JavaScript or TypeScript arrow function ends its header with (`() =>`).
const ARROW: &str = "=>";

/// The words a JavaScript function expression's value opens with (`function (x) {`).
const FUNCTION_VALUE_WORDS: [&str; 2] = ["function", "async function"];

/// Whether a declaration's value (the text after its `=`, trimmed at its start) is a function
/// expression ([`FUNCTION_VALUE_WORDS`], as whole words: `functional(x)` is a call). O(1).
fn opens_function_value(value: &str) -> bool {
    FUNCTION_VALUE_WORDS.iter().any(|w| {
        value
            .strip_prefix(w)
            .is_some_and(|after| !after.starts_with(is_identifier_char))
    })
}

/// Whether a trimmed declaration line goes on to the next line: it ends in a
/// [`DECLARATION_CONTINUATIONS`] character, in an arrow ([`ARROW`], `const f = () =>`), or in
/// the `)` of a function expression's parameters (`const g = function (x)`), whose body may open
/// on the next line. A call's `)` (`const h = g(1)`) ends the declaration. O(n) in the line.
fn continues_on_next_line(trimmed: &str) -> bool {
    // MAGI FIX (S6 loop 13): an arrow or function expression whose brace opened on the next
    // line ended on its own line, and its body fell into a gap unit.
    trimmed.ends_with(DECLARATION_CONTINUATIONS)
        || trimmed.ends_with(ARROW)
        || (trimmed.ends_with(')')
            && trimmed
                .split_once('=')
                .is_some_and(|(_, value)| opens_function_value(value.trim_start())))
}

/// The opening and closing lines (0-based) of the block a declaration at line `i` holds: an
/// object or struct literal, a wrapped `use a::{`, a function expression's body. There is one
/// when the declaration's line opens a brace, or goes on to the next line
/// ([`continues_on_next_line`]) and a later line opens one before the declaration ends
/// ([`open_brace_line`]). A
/// complete one-line declaration (`const a = 1`) takes no later block. A linkage block
/// (`extern "C" {`, [`opens_linkage_block`]) holds none: its contents stay top-level constructs.
/// O(k) for the k lines of the block.
fn declaration_block(
    lines: &[&str],
    scanned: &[ScannedLine],
    i: usize,
    search: &BraceSearch<'_>,
) -> Option<(usize, usize)> {
    let opens = scanned.get(i).is_some_and(|s| s.brace_delta > 0);
    let continues = lines
        .get(i)
        .is_some_and(|l| continues_on_next_line(l.trim()));
    if !(opens || continues) || lines.get(i).is_some_and(|l| opens_linkage_block(l.trim())) {
        return None;
    }
    let open = open_brace_line(lines, scanned, i, search)?;
    Some((open, find_closing_brace(scanned, open)))
}

/// Whether the text after a C block keyword declares a variable of that type rather than the
/// type: before its first `{`, `=` or `;` stand two words or more (`Foo x = {1};`,
/// `Foo *p;`), with no `:` or `<` (a base clause or a template, which only a type has). A type
/// names itself once (`Foo {`, `Foo;`). When that first character is `{`, it is a C++ braced
/// initializer (`Foo x{1};`) only if the line also ends in `;` and the word before the `{` is
/// not `final` (`Foo final {`); a braced initializer wrapped over lines is read as the type's
/// body, declared. O(n) in the text.
fn declares_variable(rest: &str) -> bool {
    let Some(end) = rest.find(['{', '=', ';']) else {
        return false;
    };
    let head = rest.get(..end).unwrap_or("");
    let names_variable = !head.contains([':', '<']) && head.split_whitespace().count() >= 2;
    if !rest.get(end..).is_some_and(|t| t.starts_with('{')) {
        return names_variable;
    }
    // MAGI FIX (S6 loop 7): `struct Foo x{1};` is a variable with a braced initializer.
    names_variable
        && rest.trim_end().ends_with(';')
        && head.split_whitespace().last() != Some("final")
}

/// Whether the text after a C block keyword is the return type of a function rather than the
/// type: a `(` comes before any `{` and two words or more stand before it (`Point make_point(`,
/// `Point *origin(`), so a single word before it is an attribute (`__attribute__((packed)) Hdr`,
/// `alignas(8) Foo`). O(n) in the text.
fn returns_from_function(rest: &str) -> bool {
    let Some(paren) = rest.find('(') else {
        return false;
    };
    if rest.find('{').is_some_and(|brace| brace < paren) {
        return false;
    }
    rest.get(..paren)
        .unwrap_or("")
        .split(|c: char| c.is_whitespace() || c == '*' || c == '&')
        .filter(|w| !w.is_empty())
        .count()
        >= 2
}

/// Whether a trimmed line opens an initializer it continues on the next lines: it ends in `=`
/// (`static int n =`), in `= {` (`int a[] = {`) or in a call's `(` after the `=`
/// ([`opens_call_initializer`]). O(n) in the line.
fn opens_initializer(trimmed: &str) -> bool {
    trimmed
        .strip_suffix('{')
        .unwrap_or(trimmed)
        .trim_end()
        .ends_with('=')
        || opens_call_initializer(trimmed)
}

/// Whether a trimmed line opens an initializer whose call wraps onto the next lines
/// (`static const struct config c = make_config(`): it ends in `(` and holds a `=` before its
/// first `(`, so a signature with a default argument (`void f(int a = g(`) is none. O(n).
fn opens_call_initializer(trimmed: &str) -> bool {
    // MAGI FIX (S6 loop 13): `T c = make(` was no declaration, so its header and arguments
    // fell into a gap unit and its `);` became a declaration of its own.
    trimmed.ends_with('(')
        && trimmed
            .split_once('(')
            .is_some_and(|(before, _)| before.contains('='))
}

/// The 0-based line that closes the call an initializer opens at line `i`
/// ([`opens_call_initializer`]): the first code line after it where the parentheses opened
/// since line `i` all close, counted in code context ([`code_group_delta`]). The walk ends without
/// one at a blank line or after [`MAX_BRACE_SEARCH_LINES`] lines; `i` then, and when line `i`
/// opens no such call, so the declaration ends on its own line and the lines after it fall into
/// later units, none lost. O(k) for the k lines walked.
fn call_initializer_end(
    lines: &[&str],
    scanned: &[ScannedLine],
    i: usize,
    search: &BraceSearch<'_>,
) -> usize {
    if !lines
        .get(i)
        .is_some_and(|l| opens_call_initializer(l.trim()))
    {
        return i;
    }
    // MAGI FIX (B2b1 B1): counted on the line text, a `(` and a `)` in two comments ran the
    // declaration over the next one.
    let mut depth = code_group_delta(scanned, i);
    let budget_end = i
        .saturating_add(1)
        .saturating_add(MAX_BRACE_SEARCH_LINES)
        .min(lines.len());
    for j in (i + 1)..budget_end {
        if !search.spend() || lines.get(j).is_some_and(|l| l.trim().is_empty()) {
            return i;
        }
        depth += code_group_delta(scanned, j);
        if depth <= 0 {
            return j;
        }
    }
    i
}

/// The 0-based line that closes the parentheses and square brackets line `i` leaves open
/// (Go's `var (` ... `)`, `const t = [` ... `];`, `const x = make(a,` ... `c);`): the first later
/// line where the delimiters opened since line `i` all close, counted by the scanner in code
/// context only ([`ScannedLine::group_delta`]), so a `)` in a string or a comment closes
/// nothing. Blank lines and comment lines inside are passed over.
///
/// `i` when line `i` leaves none open or is a preprocessor directive (`#define LP (` opens a
/// parenthesis its expansion closes elsewhere), and when the walk ends without a match: at a line
/// that closes a brace opened before line `i` (the delimiter was left open in its own scope),
/// after [`MAX_BRACE_SEARCH_LINES`] lines, or when the pass's budget is spent ([`SearchBudget`]).
/// The declaration then ends as it did before this rule and the lines after it fall into later
/// units, none lost. O(k) for the k lines walked.
///
/// # Arguments
///
/// * `lines` - The text's lines.
/// * `scanned` - The scanner's record of each line.
/// * `i` - The 0-based line the declaration starts on.
/// * `search` - The bounds of the forward searches ([`BraceSearch`]).
///
/// # Returns
///
/// The 0-based closing line, never before `i`.
fn group_end(lines: &[&str], scanned: &[ScannedLine], i: usize, search: &BraceSearch<'_>) -> usize {
    let Some(first) = scanned.get(i).filter(|s| !s.is_non_code) else {
        return i;
    };
    let directive = lines
        .get(i)
        .is_some_and(|l| l.trim_start().starts_with('#'));
    if first.group_delta <= 0 || directive {
        return i;
    }
    let mut open = i64::from(first.group_delta);
    // The braces opened since the start of line `i`: a line that falls below zero closes a
    // brace of the enclosing scope.
    let mut braces = i64::from(first.brace_delta);
    let walk_end = i
        .saturating_add(1)
        .saturating_add(MAX_BRACE_SEARCH_LINES)
        .min(scanned.len());
    for j in (i + 1)..walk_end {
        let Some(record) = scanned.get(j) else {
            return i;
        };
        if !search.spend() || braces + i64::from(record.min_delta) < 0 {
            return i;
        }
        braces += i64::from(record.brace_delta);
        open += i64::from(record.group_delta);
        if open <= 0 {
            return j;
        }
    }
    i
}

/// Whether a trimmed line ends in an assignment's bare `=` (`static int n =`), not in a
/// comparison (`==`, `!=`, `<=`, `>=`). O(1).
fn ends_in_bare_equal(trimmed: &str) -> bool {
    trimmed
        .strip_suffix('=')
        .is_some_and(|rest| !rest.ends_with(['=', '!', '<', '>']))
}

/// The 0-based last line of the value a declaration at line `i` puts after a bare `=` on the
/// next lines (`pub static T: &[u8] =` then `&[1, 2, 3];`): the end of that statement, the first
/// later line whose code ends in `;` ([`statement_end`], which stops at a blank line, at the next
/// construct and when the pass's budget is spent); with none, as in JavaScript or Go without
/// semicolons, the next line when it is code that starts no construct, so the value is kept and
/// the declaration after it is never taken. `i` when line `i` is no code ending in a bare `=`
/// ([`ends_in_bare_equal`]), or no line qualifies. O(k) for the k lines searched.
fn bare_initializer_end(
    lines: &[&str],
    scanned: &[ScannedLine],
    i: usize,
    search: &BraceSearch<'_>,
) -> usize {
    let is_code = |j: usize| scanned.get(j).is_some_and(|s| !s.is_non_code);
    if !is_code(i) || !lines.get(i).is_some_and(|l| ends_in_bare_equal(l.trim())) {
        return i;
    }
    let end = statement_end(lines, scanned, i, search);
    if end > i {
        return end;
    }
    let next = i + 1;
    let value = is_code(next)
        && lines.get(next).is_some_and(|l| !l.trim().is_empty())
        && !search.stops(next);
    if value {
        next
    } else {
        i
    }
}

/// Whether a trimmed line opens a C or C++ linkage block (`extern "C" {`, `extern "C++" {`).
/// Such a block only sets the linkage of what it holds, so its functions, prototypes and globals
/// are found as top-level constructs (indented signatures count, [`BraceFamilyExtractor::signature_name`])
/// and its header and closing brace are units of their own: no line of it is lost. O(n).
fn opens_linkage_block(trimmed: &str) -> bool {
    trimmed
        .strip_prefix("extern")
        .map(str::trim_start)
        .and_then(|r| r.strip_prefix('"'))
        .and_then(|r| r.split_once('"'))
        .is_some_and(|(_, after)| after.trim_start().starts_with('{'))
}

/// The 0-based last line of the preprocessor directive at line `i` (`#define MAX(a, b) \`):
/// the directive goes on while a line ends in `\`. `i` when line `i` is no directive (its text
/// does not open with `#`) or does not end in `\`. O(k) for the k lines of the directive.
fn preprocessor_end(lines: &[&str], i: usize) -> usize {
    if !lines
        .get(i)
        .is_some_and(|l| l.trim_start().starts_with('#'))
    {
        return i;
    }
    let mut end = i;
    while end + 1 < lines.len() && lines.get(end).is_some_and(|l| l.trim_end().ends_with('\\')) {
        end += 1;
    }
    end
}

/// The 0-based line where the statement that starts at line `i` ends: the first line from `i`
/// whose code ends in `;`, searched up to a blank line or the next construct (`stops`); `i`
/// when none does. Each line searched spends one visit of the pass ([`SearchBudget`]); with
/// none left the search ends there, and the statement then ends on line `i`, its later lines
/// falling into later units, none lost. O(k) for the k lines searched.
fn statement_end(
    lines: &[&str],
    scanned: &[ScannedLine],
    i: usize,
    search: &BraceSearch<'_>,
) -> usize {
    let ends = |j: usize| {
        scanned
            .get(j)
            .is_some_and(|s| !s.is_non_code && s.ends_statement)
    };
    if ends(i) {
        return i;
    }
    ((i + 1)..lines.len())
        .take_while(|&j| {
            search.spend() && !search.stops(j) && !lines.get(j).is_some_and(|l| l.trim().is_empty())
        })
        .find(|&j| ends(j))
        .unwrap_or(i)
}

/// Find the 1-indexed line where the construct starting at line index `i` ends.
///
/// The opening brace is the one [`open_brace_line`] finds. A construct with no brace ends at
/// its statement's `;` ([`statement_end`]): a signature without a body wrapped over lines
/// (`fn f(` ... `) -> u8;`, a trait method with a `where` clause) is one unit. That holds only
/// when the code of line `i` does not end in `}` ([`ScannedLine::ends_block`]): a one-line
/// construct whose braces balance (`run() {}`) ends on `line_num`, so it never takes a later
/// statement (a class field `x = 1;`), while a `{` in a comment or a string of line `i` cuts
/// nothing. A signature with
/// no body and no `;` (a Go function implemented in assembly) ends on `line_num` too, as does
/// any construct whose search runs out of budget; the lines after it fall into later units,
/// none lost.
///
/// # Arguments
///
/// * `lines` - The text's lines.
/// * `scanned` - The scanner's record of each line.
/// * `i` - The 0-based line the construct starts on.
/// * `line_num` - The 1-based number of line `i`.
/// * `search` - The bounds of the forward searches ([`BraceSearch`]).
///
/// # Returns
///
/// The construct's last line, 1-based, or `None` when it does not fit `u32`.
fn brace_block_end(
    lines: &[&str],
    scanned: &[ScannedLine],
    i: usize,
    line_num: u32,
    search: &BraceSearch<'_>,
) -> Option<u32> {
    let open = open_brace_line(lines, scanned, i, search);
    block_end_after(lines, scanned, i, line_num, open, search)
}

/// The 1-indexed last line of the construct starting at line index `i`, as [`brace_block_end`]
/// finds it, given the result of its brace search (`open`, [`open_brace_line`]) already run, so
/// a caller that needed the search's answer first does not spend its visits a second time.
///
/// With no opening brace, a line whose code ends in `}` ([`ScannedLine::ends_block`]) is a
/// one-line construct (`fn a() { x(); } // note`) and ends there; any other line is a
/// signature without a body, which runs to its statement's end ([`statement_end`]), however
/// wrapped, so a brace in a comment or a string of its first line does not cut it.
///
/// # Arguments
///
/// * `lines` - The text's lines.
/// * `scanned` - The scanner's record of each line.
/// * `i` - The 0-based line the construct starts on.
/// * `line_num` - The 1-based number of line `i`.
/// * `open` - The 0-based line of the construct's opening brace, or `None` when it has none.
/// * `search` - The bounds of the forward searches ([`BraceSearch`]).
///
/// # Returns
///
/// The construct's last line, 1-based, or `None` when it does not fit `u32`.
fn block_end_after(
    lines: &[&str],
    scanned: &[ScannedLine],
    i: usize,
    line_num: u32,
    open: Option<usize>,
    search: &BraceSearch<'_>,
) -> Option<u32> {
    match open {
        Some(open_idx) => line_number(find_closing_brace(scanned, open_idx)),
        // MAGI FIX (S6 loop 14): a wrapped signature without a body ended on its first line
        // and the rest of it fell into the next unit.
        // MAGI FIX (S6 loop 18, B2b-2): a `{` anywhere in the line's text, a comment's or a
        // string's included, made it a one-line construct; the scanner says whether its code
        // ends in `}`.
        None if scanned.get(i).is_none_or(|s| !s.ends_block) => {
            line_number(statement_end(lines, scanned, i, search))
        }
        None => Some(line_num),
    }
}

/// The 1-based line number of the 0-based line `idx`.
///
/// # Arguments
///
/// * `idx` - A 0-based line index.
///
/// # Returns
///
/// `idx + 1`, or `None` when it does not fit `u32`. O(1).
fn line_number(idx: usize) -> Option<u32> {
    u32::try_from(idx.checked_add(1)?).ok()
}

/// Whether `c` can be part of an identifier in a brace-family language: a letter or digit
/// (Unicode included), `_`, or `$` (JavaScript and TypeScript; GCC accepts it in C too).
fn is_identifier_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '$'
}

/// The identifier that names the function `line` declares, read just before the `(` that opens
/// its parameter list ([`parameter_list_head`]; `int *parse(` gives `parse`,
/// `public static <T> List<T> wrap(` gives `wrap`, `decltype(auto) f(` gives `f`,
/// `std::function<void(int)> make(` gives `make`), or inside the declarator in parentheses of a
/// function returning a function pointer (`void (*get_handler(int sig))(int)` gives
/// `get_handler`).
///
/// A C++ operator overload is named before this rule applies ([`operator_overload_name`]).
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
    // MAGI FIX (S6 loop 15): `decltype(auto) f() {` was named `decltype`, its operand's `(`
    // being taken for the parameter list; the list's own `(` is looked for instead.
    let head = parameter_list_head(line)?;
    if head.before.contains('=') {
        return None;
    }
    // MAGI FIX (S6 loop 6): `$get() {` named no member, so a class could not split there.
    // MAGI FIX (S6 loop 18, B2b-4): `foo<T>(x: T) {` ended in `>`, so no identifier named it
    // and a generic TypeScript method was no member; its type parameters are read past.
    Some(trailing_identifier(without_type_parameters(head.name_text)))
        .filter(|n| !n.is_empty())
        .map(str::to_string)
}

/// The name of a JavaScript or TypeScript member whose key is computed or private, as written
/// before the `(` of its parameter list ([`parameter_list_head`]): a computed key in balanced
/// square brackets (`*[Symbol.iterator]() {` gives `[Symbol.iterator]`) or an identifier right
/// after a `#` (`static #s() {` gives `#s`). `None` for any other line, which
/// [`identifier_before_paren`] then names, and when an `=` comes before the `(` (an assignment).
/// O(n) in the line.
///
/// # Arguments
///
/// * `line` - A signature line, trimmed.
///
/// # Returns
///
/// The member's name, or `None`.
fn private_or_computed_name(line: &str) -> Option<String> {
    let head = parameter_list_head(line)?;
    if head.before.contains('=') {
        return None;
    }
    let text = without_type_parameters(head.name_text);
    if text.ends_with(']') {
        let open = matching_open(text, '[', ']')?;
        return text.get(open..).map(str::to_string);
    }
    let ident = trailing_identifier(text);
    let before = text.get(..text.len().checked_sub(ident.len())?)?;
    (!ident.is_empty() && before.ends_with('#')).then(|| format!("#{ident}"))
}

/// `text` without the type parameter list it ends with (`foo<T>` gives `foo`,
/// `[key]<K, V>` gives `[key]`), blanks at its end passed over; `text` trimmed at its end when it
/// ends in no `>` or the list never opens. A generic method (`foo<T>(x: T) {`) and a C++
/// explicit specialization (`void f<int>(int x) {`) are named so before their `(`. O(n) in the
/// text.
///
/// # Arguments
///
/// * `text` - The text of a signature line before its parameter list.
///
/// # Returns
///
/// The text before the type parameter list.
fn without_type_parameters(text: &str) -> &str {
    let text = text.trim_end();
    if !text.ends_with('>') {
        return text;
    }
    matching_open(text, '<', '>')
        .and_then(|at| text.get(..at))
        .map_or(text, str::trim_end)
}

/// The byte offset of the `open` that matches the `close` `text` ends with (`[` for `]`, `<` for
/// `>`), counting the pairs between, or `None` when it never opens. O(n) in the text.
fn matching_open(text: &str, open: char, close: char) -> Option<usize> {
    let mut depth = 0u32;
    for (at, c) in text.char_indices().rev() {
        match c {
            c if c == close => depth = depth.saturating_add(1),
            c if c == open => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(at);
                }
            }
            _ => {}
        }
    }
    None
}

/// The words of a C++ return type whose operand is parenthesized (`decltype(auto)`,
/// `decltype(a + b)`): the `(` right after one opens no parameter list.
const TYPE_OPERATORS: [&str; 1] = ["decltype"];

/// The characters a declarator in parentheses opens with (`(*f)`, `(&r)`, a block's `(^b)`).
const DECLARATOR_OPENERS: [char; 3] = ['*', '&', '^'];

/// Where a signature line's parameter list starts ([`parameter_list_head`]).
struct ListHead<'t> {
    /// The line's text before the `(` the head was found at, for the `=` rule.
    before: &'t str,
    /// The text whose last identifier names the function: `before` itself, or the inside of a
    /// declarator in parentheses up to its own `(`.
    name_text: &'t str,
}

impl<'t> ListHead<'t> {
    /// A head named by the identifier just before its `(`. O(1).
    fn at(before: &'t str) -> Self {
        Self {
            before,
            name_text: before,
        }
    }
}

/// The head of the parameter list `line` opens, read past what a return type may put in
/// parentheses before it:
///
/// * a `(` right after a word of [`TYPE_OPERATORS`] is that word's operand, read past up to its
///   matching `)` (`decltype(auto) f(`);
/// * a `(` inside angle brackets is a template argument, read past the same way
///   (`std::function<void(int)> make(`); a `>` closes a bracket however it is preceded, so `->`
///   counts as one too;
/// * a `(` after a return type word, opening with a [`DECLARATOR_OPENERS`] character and followed
///   by another `(` once it closes, is a declarator in parentheses: the function is named
///   inside it (`void (*f)()`, `void (*get_handler(int sig))(int)`). A keyword of
///   [`NOT_FUNCTION_NAMES`] before it (`while (*s)(x)`) is no return type.
///
/// When a `<` is left open (a comparison, `a < f(x)`, or an operator, `operator<(`), the line's
/// first `(` is taken, as before this rule, so nothing it named before is lost; an operand that
/// never closes on the line leaves its `(` as the list's, so the line is named by the type word,
/// which [`NOT_FUNCTION_NAMES`] rules out. A heuristic on one line's text: a parenthesis inside a
/// string of the return type miscounts. O(n) in the line.
///
/// # Arguments
///
/// * `line` - A signature line.
///
/// # Returns
///
/// The head, or `None` when the line holds no `(` that opens a parameter list.
fn parameter_list_head(line: &str) -> Option<ListHead<'_>> {
    // MAGI FIX (S6 loop 15): `std::function<void(int)> make() {` and `void (*f)() {` were
    // named `void`, the first `(` being taken for the parameter list.
    let first = || line.split_once('(').map(|(before, _)| ListHead::at(before));
    let mut from = 0;
    let mut angles = 0usize;
    loop {
        let Some((offset, c)) = line
            .get(from..)?
            .char_indices()
            .find(|&(_, c)| matches!(c, '(' | '<' | '>'))
        else {
            return if angles > 0 { first() } else { None };
        };
        let at = from + offset;
        if c != '(' {
            angles = if c == '<' {
                angles + 1
            } else {
                angles.saturating_sub(1)
            };
            from = at + c.len_utf8();
            continue;
        }
        let before = line.get(..at)?;
        let group_end = line
            .get(at..)
            .and_then(|t| balanced_end(t, '(', ')'))
            .map(|end| at + end);
        let word = trailing_identifier(before);
        let read_past = angles > 0 || TYPE_OPERATORS.contains(&word);
        match group_end {
            Some(end) if read_past => from = end,
            None if angles > 0 => return first(),
            None => return Some(ListHead::at(before)),
            Some(end) => {
                let name_text = declarator_name_text(line, at, end, word).unwrap_or(before);
                return Some(ListHead { before, name_text });
            }
        }
    }
}

/// The inside of the declarator in parentheses that spans `line[open..end]`, up to its own `(`
/// (`*get_handler` in `void (*get_handler(int sig))(int)`, `*f` in `void (*f)()`), when it is
/// one: `word`, the identifier before it, is a return type (not empty, not a keyword of
/// [`NOT_FUNCTION_NAMES`]), its inside opens with a [`DECLARATOR_OPENERS`] character, and a `(`
/// follows its `)`. O(n) in the line.
///
/// # Arguments
///
/// * `line` - A signature line.
/// * `open` - The byte offset of the group's `(`.
/// * `end` - The byte offset just past its matching `)`.
/// * `word` - The identifier just before the `(`.
///
/// # Returns
///
/// The declarator's text, or `None` when the group is no declarator.
fn declarator_name_text<'t>(line: &'t str, open: usize, end: usize, word: &str) -> Option<&'t str> {
    if word.is_empty() || NOT_FUNCTION_NAMES.contains(&word) {
        return None;
    }
    let inner = line.get(open + 1..end.checked_sub(1)?)?.trim_start();
    let followed = line.get(end..)?.trim_start().starts_with('(');
    if !(followed && inner.starts_with(DECLARATOR_OPENERS)) {
        return None;
    }
    Some(inner.split_once('(').map_or(inner, |(name, _)| name))
}

/// The identifier `text` ends with, blanks at its end passed over (`int *parse ` gives
/// `parse`); empty when it ends in no identifier character ([`is_identifier_char`]). O(n) in the
/// text.
fn trailing_identifier(text: &str) -> &str {
    let text = text.trim_end();
    let start = text
        .char_indices()
        .rev()
        .take_while(|&(_, c)| is_identifier_char(c))
        .last()
        .map_or(text.len(), |(at, _)| at);
    text.get(start..).unwrap_or("")
}

/// The characters a C++ overloaded operator's symbol is made of (`=`, `==`, `<<=`, `[]`).
const OPERATOR_SYMBOL_CHARS: &str = "=!<>+-*/%&|^~[],";

/// The name of the C++ operator overload `before` declares, when it ends in the word `operator`
/// and a symbol, blanks between them or inside it passed over (`Foo& operator=` gives
/// `operator=`, `bool operator ==` gives `operator==`, `int operator [ ]` gives `operator[]`), or
/// in the word `operator`, a blank and an allocation operator of [`ALLOCATION_OPERATORS`]
/// (`void* operator new` gives `operator new`, `void operator delete [ ]` gives
/// `operator delete[]`).
///
/// # Arguments
///
/// * `before` - The text of a line before its first `(`.
///
/// # Returns
///
/// The operator's name, or `None` when `before` names no such operator (a call operator,
/// `operator()`, has its symbol after the first `(` and is named before this rule
/// ([`call_operator`]); a conversion operator, `operator bool`, is named by
/// [`identifier_before_paren`]'s general rule, after its type). O(n) in the text.
fn operator_name(before: &str) -> Option<String> {
    let (head, symbol) = before.trim_end().rsplit_once("operator")?;
    let word_starts = head
        .chars()
        .next_back()
        .is_none_or(|c| !is_identifier_char(c));
    if !word_starts {
        return None;
    }
    // MAGI FIX (S6 loop 18, B2b-7): the symbol had to touch the word, so `bool operator <(` was
    // no function; only C++ asks ([`operator_overload_name`]), where the blanks mean nothing.
    let word: String = symbol.chars().filter(|c| !c.is_whitespace()).collect();
    if !word.is_empty() && word.chars().all(|c| OPERATOR_SYMBOL_CHARS.contains(c)) {
        return Some(format!("operator{word}"));
    }
    // MAGI FIX (S6 loop 14): `void* operator new(size_t n) {` reached `NOT_FUNCTION_NAMES` with
    // the word `new`, so the allocator was no function and its body fell into a gap unit.
    let allocation =
        symbol.starts_with(char::is_whitespace) && ALLOCATION_OPERATORS.contains(&word.as_str());
    allocation.then(|| format!("operator {word}"))
}

/// The name of the C++ operator overload `line` declares: a call operator ([`call_operator`],
/// `operator()`) or an operator named before the `(` of its parameter list ([`operator_name`],
/// `operator=`, `operator <`, `operator new`). Asked only for an extractor with
/// `operator_overloads` (C++), where `operator` is a keyword: elsewhere it is an identifier
/// like any other, so a JavaScript `const operator = (x) => {` names no overload. O(n × k) in
/// the line, for k occurrences of the word.
///
/// # Arguments
///
/// * `line` - A signature line.
///
/// # Returns
///
/// The operator's name, or `None` when the line declares no operator overload.
// MAGI FIX (S6 loop 18, B2b-7): moved out of `identifier_before_paren`, so the spaced forms
// are read in C++ only.
fn operator_overload_name(line: &str) -> Option<String> {
    // MAGI FIX (S6 loop 17, B2-5): `bool operator()(int x) {` was named `operator`: its symbol is
    // the line's first `()`, taken for the parameter list.
    if call_operator(line) {
        return Some(CALL_OPERATOR_NAME.to_string());
    }
    // MAGI FIX (S6 loop 9): `Foo& operator=(` held an `=` before its `(` and named nothing.
    operator_name(parameter_list_head(line)?.name_text)
}

/// The name of a C++ call operator overload ([`call_operator`]).
const CALL_OPERATOR_NAME: &str = "operator()";

/// Whether `line` declares a C++ call operator overload (`bool operator()(int x) const {`,
/// `auto operator ( ) (int x) {`): the word `operator` (no identifier character before it), then
/// after optional blanks an empty `()`, then after optional blanks the `(` of the parameter list.
/// Its symbol is the line's first `()`, so the general rule ([`parameter_list_head`]) takes it
/// for the parameter list and names the function `operator`. O(n × k) in the line, for k
/// occurrences of the word.
fn call_operator(line: &str) -> bool {
    line.match_indices("operator").any(|(at, word)| {
        let word_starts = line
            .get(..at)
            .and_then(|before| before.chars().next_back())
            .is_none_or(|c| !is_identifier_char(c));
        let after = line.get(at + word.len()..).unwrap_or("").trim_start();
        let parameters = after
            .strip_prefix('(')
            .map(str::trim_start)
            .and_then(|rest| rest.strip_prefix(')'))
            .map(str::trim_start);
        word_starts && parameters.is_some_and(|rest| rest.starts_with('('))
    })
}

/// The C++ allocation operators, whose symbol is a word with an optional `[]`
/// (`operator new`, `operator delete[]`), named with one space after `operator` and the
/// brackets glued to the word, however they were spaced on the line.
const ALLOCATION_OPERATORS: [&str; 4] = ["new", "new[]", "delete", "delete[]"];

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

/// The text after keyword `kw` at the start of `text`. A table keyword ends in a blank
/// (`impl `), and the word may also be followed directly by a generic parameter list
/// (`impl<T: Clone> Foo<T>`), which is passed over, or by the brace of an anonymous construct
/// (`class{`), which is kept. The keyword may also be the whole line (`union`, its brace on the
/// next line, Allman style); the rest is then empty and the construct is named `unnamed`
/// ([`construct_name`]). `None` when `text` does not open with the keyword. O(n) in the text.
fn keyword_rest<'t>(text: &'t str, kw: &str) -> Option<&'t str> {
    if let Some(rest) = blank_keyword_rest(text, kw) {
        return Some(rest);
    }
    // MAGI FIX (S6 loop 5): `impl<T> Foo<T>` matched no `impl ` and its methods lost the header.
    let word = kw.strip_suffix(' ')?;
    let rest = text.strip_prefix(word)?;
    // MAGI FIX (S6 loop 6): `export default class{` matched no `class ` and was no class.
    // MAGI FIX (S6 loop 14): a keyword alone on its line (`union`, then `{` on the next) was no
    // construct, so an anonymous type's variable fell into gap units around its declarator.
    if rest.is_empty() || rest.starts_with('{') {
        return Some(rest);
    }
    rest.starts_with('<').then(|| skip_generics(rest))
}

/// The text after keyword `kw` at the start of `text`: `text` past `kw` as written or, for a
/// table keyword that ends in a blank (`fn `, `const `), past the bare word and any run of
/// blanks after it (`fn\tfoo`, `const  X`), trimmed at its start. `None` when `text` does not
/// open with the keyword followed by a blank (`constant`). O(n) in the text.
///
/// # Arguments
///
/// * `text` - A trimmed line, or its rest after the modifiers.
/// * `kw` - A keyword of a table.
///
/// # Returns
///
/// The text after the keyword, or `None`.
fn blank_keyword_rest<'t>(text: &'t str, kw: &str) -> Option<&'t str> {
    // MAGI FIX (S6 loop 13): a tab after the keyword (`fn\tfoo`) matched no `fn `.
    // MAGI FIX (S6 loop 14): a keyword matched as written kept the blanks after it, so
    // `const  f = () => {` named an empty function and `enum  class Color` read as `class`.
    if let Some(rest) = text.strip_prefix(kw) {
        return Some(rest.trim_start());
    }
    let rest = text.strip_prefix(kw.strip_suffix(' ')?)?;
    rest.starts_with(char::is_whitespace)
        .then(|| rest.trim_start())
}

/// What may follow a declaration keyword's bare word with no blank between and still open the
/// declaration: a list (`use::{a}`, `import{foo}`, `export{foo}`), a path (`use::a`), an included
/// file (`#include<stdio.h>`, `#include"a.h"`), a quoted module or linkage (`import'./m'`,
/// `extern"C"`) or a namespace import (`import*as m`). An identifier character never does, so a
/// longer word (`usefoo`, `importer`) is no keyword.
const GLUED_DECLARATION_OPENERS: [&str; 6] = ["{", "::", "<", "\"", "'", "*"];

/// The text after declaration keyword `kw` at the start of `text`: as [`blank_keyword_rest`],
/// or, when the keyword's bare word is followed directly by one of
/// [`GLUED_DECLARATION_OPENERS`], the text from that opener on. O(n) in the text.
///
/// # Arguments
///
/// * `text` - A trimmed line, or its rest after the modifiers.
/// * `kw` - A declaration keyword of a table, ending in a blank.
///
/// # Returns
///
/// The text after the keyword, or `None` when `text` does not open with it.
fn declaration_keyword_rest<'t>(text: &'t str, kw: &str) -> Option<&'t str> {
    if let Some(rest) = blank_keyword_rest(text, kw) {
        return Some(rest);
    }
    let rest = text.strip_prefix(kw.strip_suffix(' ')?)?;
    GLUED_DECLARATION_OPENERS
        .iter()
        .any(|opener| rest.starts_with(opener))
        .then_some(rest)
}

/// `text` after the balanced `<...>` it opens with, trimmed at its start; empty when the list
/// never closes. The `>` of a `->` (`impl<T: Fn(u8) -> u8>`) closes nothing. O(n) in the text.
fn skip_generics(text: &str) -> &str {
    let mut depth = 0u32;
    let mut previous = '\0';
    for (i, c) in text.char_indices() {
        let after_dash = std::mem::replace(&mut previous, c) == '-';
        match c {
            '<' => depth = depth.saturating_add(1),
            // MAGI FIX (S6 loop 6): the `>` of a `->` in a bound closed the list early.
            '>' if after_dash => {}
            '>' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return text.get(i + c.len_utf8()..).unwrap_or("").trim_start();
                }
            }
            _ => {}
        }
    }
    ""
}

/// `text` after the C++ template header it opens with: the word `template` followed, after any
/// blanks, by a balanced `<...>` ([`skip_generics`]), trimmed at its start; empty when the
/// header is the whole line or never closes on it. `text` when it opens with no such header
/// (`templated()`, `template class Foo<int>;`). O(n) in the text.
///
/// A header wrapped over several lines (`template <typename T,` then `typename U>`) is not
/// recognised past its first line: the first line attaches to the construct only when it is the
/// line right above it, so such a header's lines fall into a gap unit, each line in exactly one.
fn strip_template_header(text: &str) -> &str {
    match text.strip_prefix(TEMPLATE_KEYWORD).map(str::trim_start) {
        Some(rest) if rest.starts_with('<') => skip_generics(rest),
        _ => text,
    }
}

/// The Go keyword that declares an interface type (`type Reader interface {`).
const GO_INTERFACE_KEYWORD: &str = "interface";

/// The kind of a Go type declaration from the text after `type` (or a member line of a
/// `type (` group): its name, an optional type parameter list in brackets (`Box[T any]`), then
/// the keyword [`GO_INTERFACE_KEYWORD`] makes a trait, anything else (a struct, a defined type,
/// an alias) a struct. A Go interface's methods carry no keyword, so it is reviewed whole, never
/// split at them. O(n) in the text.
fn go_type_kind(rest: &str) -> UnitKind {
    let rest = rest.trim_start();
    let after_name = rest.trim_start_matches(is_identifier_char).trim_start();
    let after_params = if after_name.starts_with('[') {
        balanced_end(after_name, '[', ']')
            .and_then(|end| after_name.get(end..))
            .unwrap_or("")
            .trim_start()
    } else {
        after_name
    };
    let interface = after_params
        .strip_prefix(GO_INTERFACE_KEYWORD)
        .is_some_and(|after| !after.starts_with(is_identifier_char));
    if interface {
        UnitKind::Trait
    } else {
        UnitKind::Struct
    }
}

/// The C++ keyword that opens a template header (`template <typename T>`).
const TEMPLATE_KEYWORD: &str = "template";

/// The C and C++ attribute words whose argument list in parentheses is no call: GCC's
/// `__attribute__((packed))`, MSVC's `__declspec(dllexport)`, and the alignment specifiers
/// `alignas(16)` (C++11, C23) and `_Alignas(16)` (C11).
const C_ATTRIBUTE_WORDS: [&str; 4] = ["__attribute__", "__declspec", "alignas", "_Alignas"];

/// `text` without its C and C++ attributes, wherever they stand on the line: a standard
/// attribute (`[[nodiscard]]`, `[[gnu::packed]]`) and a word of [`C_ATTRIBUTE_WORDS`] with its
/// argument list in balanced parentheses. The blanks are collapsed to one space, so the
/// keyword and name read as without them (`struct __attribute__((packed)) Hdr {` reads
/// `struct Hdr {`). An attribute that does not close on the line is kept with the rest of the
/// line, so the line reads as before. The text is only matched, never reviewed: the unit keeps
/// the line as written. O(n × k) in the line, for k attribute words.
///
/// # Arguments
///
/// * `text` - A trimmed line.
///
/// # Returns
///
/// The line without its attributes, borrowed when it holds none.
fn strip_c_attributes(text: &str) -> Cow<'_, str> {
    if !text.contains("[[") && !C_ATTRIBUTE_WORDS.iter().any(|w| text.contains(w)) {
        return Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some((at, end)) = next_c_attribute(rest) {
        out.push_str(rest.get(..at).unwrap_or(""));
        out.push(' ');
        rest = rest.get(end..).unwrap_or("");
    }
    out.push_str(rest);
    Cow::Owned(out.split_whitespace().collect::<Vec<_>>().join(" "))
}

/// The byte span (`start`, `end`) of the first complete C or C++ attribute in `text`
/// ([`strip_c_attributes`]), or `None` when there is none: an attribute word starts only where
/// no identifier character comes before it, and is followed, after any blanks, by a `(` whose
/// match closes on the line; a `[[` is matched by its closing `]]`. O(n × k) in the text.
fn next_c_attribute(text: &str) -> Option<(usize, usize)> {
    let mut previous: Option<char> = None;
    for (at, c) in text.char_indices() {
        let word_starts = previous.is_none_or(|p| !is_identifier_char(p));
        previous = Some(c);
        let rest = text.get(at..).unwrap_or("");
        if rest.starts_with("[[") {
            return balanced_end(rest, '[', ']').map(|end| (at, at + end));
        }
        if !word_starts {
            continue;
        }
        let Some(word) = C_ATTRIBUTE_WORDS.iter().find(|w| {
            rest.strip_prefix(**w)
                .is_some_and(|after| !after.starts_with(is_identifier_char))
        }) else {
            continue;
        };
        let after = rest.get(word.len()..).unwrap_or("");
        let args = after.trim_start();
        if !args.starts_with('(') {
            continue;
        }
        let offset = at + word.len() + (after.len() - args.len());
        return balanced_end(args, '(', ')').map(|end| (at, offset + end));
    }
    None
}

/// The byte offset just past the `close` that matches the `open` `text` starts with, or `None`
/// when it never closes. Every character counts, strings included. O(n) in the text.
fn balanced_end(text: &str, open: char, close: char) -> Option<usize> {
    balanced_end_outside_quotes(text, open, close, &[])
}

/// The quotes an annotation's arguments are read past ([`BraceFamilyExtractor::without_annotations`]):
/// Java's strings and character literals, and JavaScript's and TypeScript's three string forms.
/// A backtick is no Java token, so reading it as a quote there changes no valid line.
const ANNOTATION_QUOTES: [char; 3] = ['"', '\'', '`'];

/// The keyword that opens a Rust macro-by-example definition (`macro_rules! name {`). It is in
/// the Rust function table: a macro is reviewed as one function-like unit, its token tree opaque,
/// so an item written inside it (`fn generated() {}`) is never a construct of its own, and it is
/// never split (`split_block` splits only blocks of items). Its end is the later of its brace
/// block and its own closing delimiter ([`group_end`]), so the `(` ... `);` and `[` ... `];`
/// forms end on their last line too. A macro invoked at the top level (`thread_local! {`) is no
/// definition and is read as before.
const MACRO_RULES_KEYWORD: &str = "macro_rules! ";

/// The opening of a Rust outer attribute (`#[inline]`).
///
/// An inner attribute (`#![allow(dead_code)]`) does not open with it, on purpose: the Rust
/// reference applies an inner attribute to the item that contains it (the crate, a module, a
/// function), never to the item written after it, so joining it to that item
/// ([`BraceFamilyExtractor::scan_back_for_docs`]) would review it as part of code it does not
/// govern. It stays with its container instead: inside a module or `impl` body it falls into the
/// block's header piece when the block splits ([`BraceFamilyExtractor::split_block`]), and at the
/// top of a file it is a gap unit of its own. Either way its line lands in exactly one unit, so
/// none is lost. Pinned by `a_rust_inner_attribute_stays_with_its_container_and_no_line_is_lost`.
/// An inner doc comment (`//!`, `/*!`) is still joined to the item below it (it is one of the
/// Rust `doc_comment_prefixes`), a known difference, declared.
// MAGI FIX (S6 loop 17, B2-6): the inner-attribute policy, written where the prefix is defined.
const RUST_ATTRIBUTE_PREFIX: &str = "#[";

/// The quotes a Rust attribute's arguments are read past: only `"`, since a `'` may open a
/// lifetime (`T: 'a`), which has no closing quote.
const RUST_ATTRIBUTE_QUOTES: [char; 1] = ['"'];

/// The byte offset just past the `close` that matches the `open` `text` starts with, counting
/// only the delimiters outside quoted text, or `None` when it never closes.
///
/// A character of `quotes` opens a quoted run that the same character ends; inside it a `\`
/// escapes the next character (`"a\"b"`). The scanner ([`CodeScanner`]) reports braces per
/// line, not where a string ends on it, so this reads one line's text on its own, with the
/// quotes of the language it reads. A quote that never closes leaves the delimiters after it
/// uncounted, so the result is `None` and the caller keeps its text as it was. O(n) in the
/// text.
///
/// # Arguments
///
/// * `text` - The text, starting with `open`.
/// * `open` - The opening delimiter.
/// * `close` - The closing delimiter.
/// * `quotes` - The characters that open and close quoted text; empty to count every character.
///
/// # Returns
///
/// The offset just past the matching `close`, or `None`.
fn balanced_end_outside_quotes(
    text: &str,
    open: char,
    close: char,
    quotes: &[char],
) -> Option<usize> {
    let mut depth = 0u32;
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for (at, c) in text.char_indices() {
        if let Some(q) = quote {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == q {
                quote = None;
            }
            continue;
        }
        if quotes.contains(&c) {
            quote = Some(c);
        } else if c == open {
            depth = depth.saturating_add(1);
        } else if c == close {
            depth = depth.saturating_sub(1);
            if depth == 0 {
                return Some(at + c.len_utf8());
            }
        }
    }
    None
}

/// The name of a construct that has none (`export default function() {`, `class {`).
const UNNAMED_CONSTRUCT: &str = "unnamed";

/// The prefix of a Rust raw identifier (`r#match`), which [`construct_name`] keeps in the name.
const RAW_IDENTIFIER_PREFIX: &str = "r#";

/// The marker of a JavaScript generator declaration (`function *gen()`), which
/// [`construct_name`] passes over before the name.
const GENERATOR_MARKER: char = '*';

/// The name of a construct: the identifier that opens `rest`, the text after its keyword (after
/// a parenthesized receiver it opens with, as in Go's `func (r *T) Name(`), or, when none
/// does, the identifier of the second word of `text`, the matched line, or [`UNNAMED_CONSTRUCT`]
/// when that is no identifier either (`function() {`). A generator's `*` before
/// the name is passed over (`function *gen()` gives `gen`), and a Rust raw identifier keeps its
/// prefix (`fn r#match()` gives `r#match`). O(n) in the line.
fn construct_name(rest: &str, text: &str) -> String {
    // MAGI FIX (S6 loop 18, B2b-6): `fn r#match()` was named `r` and `function *gen()` was
    // unnamed; a raw identifier keeps its prefix, as written, and a generator's `*` is passed
    // over.
    let ident = |t: &str| -> String {
        let (prefix, body) = t
            .strip_prefix(RAW_IDENTIFIER_PREFIX)
            .filter(|body| body.starts_with(is_identifier_char))
            .map_or(("", t), |body| (RAW_IDENTIFIER_PREFIX, body));
        let name: String = body
            .chars()
            .take_while(|&c| is_identifier_char(c))
            .collect();
        if name.is_empty() {
            name
        } else {
            format!("{prefix}{name}")
        }
    };
    let rest = rest
        .trim_start()
        .strip_prefix(GENERATOR_MARKER)
        .map_or(rest, str::trim_start);
    // MAGI FIX (S6 loop 6): a Go method's receiver left it unnamed.
    let rest = match rest.trim_start().strip_prefix('(') {
        Some(inner) => inner
            .split_once(')')
            .map_or(rest, |(_, after)| after.trim_start()),
        None => rest,
    };
    let name = ident(rest);
    if !name.is_empty() {
        return name;
    }
    let second = ident(text.split_whitespace().nth(1).unwrap_or(""));
    if second.is_empty() {
        UNNAMED_CONSTRUCT.to_string()
    } else {
        second
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
            attribute_prefixes: vec![],
            modifiers: &NO_MODIFIERS,
            lexical_rules: LexicalRules::c(),
            signature_functions: true,
            semicolon_declarations: true,
            template_headers: false,
            not_function_names: &NOT_FUNCTION_NAMES,
            reserved_word_methods: false,
            computed_member_names: false,
            anonymous_functions: false,
            operator_overloads: false,
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
        }
    }

    /// TypeScript extractor.
    pub fn typescript() -> Self {
        let mut ext = Self::javascript();
        ext.language = "typescript".to_string();
        // MAGI FIX (S6 loop 7): `.tsx` is not claimed, as `.jsx` is not ([`Self::javascript`]).
        // MAGI FIX (S6 loop 9): a CommonJS `.cts` file had no extractor.
        ext.exts = &["ts", "mts", "cts"];
        ext.block_keywords.push("interface ".to_string());
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
        // [MAGI S6 loop 8, B2b-m2] declared on `open_brace_line`: a lambda default argument's
        // brace is taken for the function's, so the body is split from its signature, but every
        // line stays in exactly one unit.
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
}
