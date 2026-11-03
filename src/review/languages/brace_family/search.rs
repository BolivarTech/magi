// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! Forward brace and statement search: where a construct's opening brace is, where a
//! statement, declaration group or initializer ends, all bounded by one shared [`SearchBudget`].

use super::*;

/// The openers of a comment that make a line before a construct no wrapped continuation of
/// the signature above it, though it ends in `,` ([`open_brace_line`]).
const LINE_OPENING_COMMENTS: [&str; 2] = ["//", "/*"];

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
pub(super) const MAX_BRACE_SEARCH_LINES: usize = 1_024;

/// What a forward brace search of one detection pass is bounded by: where the next construct
/// starts (`stops`, [`BraceFamilyExtractor::starts_construct`]) and the line visits the pass may
/// still spend ([`SearchBudget`]).
pub(super) struct BraceSearch<'s> {
    /// Whether a line (0-based) starts the next construct.
    pub(super) stops: &'s dyn Fn(usize) -> bool,
    /// The line visits left to the whole detection pass.
    pub(super) budget: &'s SearchBudget,
}

impl BraceSearch<'_> {
    /// Whether line `j` (0-based) starts the next construct. O(k) in the keywords.
    fn stops(&self, j: usize) -> bool {
        (self.stops)(j)
    }

    /// Spends one line visit of the pass ([`SearchBudget::spend`]).
    pub(super) fn spend(&self) -> bool {
        self.budget.spend()
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
/// A brace that a default argument opens while the list is still open (`void f(int cb = []() {`,
/// `function f(cb = () => {`: a line holding a bare `=`, [`holds_bare_assignment`]) is the
/// argument's: the search passes over the lines to the one that closes the list, whose brace
/// after its lowest point (`}) {`) is the function's. A brace that line `i` opens with its list
/// still open and no `=` (a parameter typed by a literal block, `function f(opts: {`,
/// `void f(struct {`) is a parameter's too when the line closing the list opens one after its
/// lowest point (`}) {`, [`brace_after_parameter_list`]); when that line opens none (a call
/// passing a callback, `describe('x', () => {` ... `});`, which reads as a signature) line `i`'s
/// brace is kept, so the call keeps its block. A brace line `i` ends with in its return type
/// (`function f(): {`, `): Promise<{`, `func f() struct {`, [`opens_return_type_literal`]) is the
/// type's too, and the body's is found the same way after the type closes
/// ([`brace_after_return_type_literal`]). A brace a later parameter line opens while the outer
/// list stays open after it (`options: {` under `function f(`) is a parameter's too, read the
/// same way: the body's brace is the one the line closing the list opens (`) {`, `}) {`) or an
/// Allman `{` after it ([`brace_after_open_list`]), and when that line opens none the later
/// line's brace is kept, as for line `i`'s.
///
/// The search looks at most [`MAX_BRACE_SEARCH_LINES`] lines past line `i`, and no more than the
/// detection pass has left ([`SearchBudget`]), and then degrades to "no brace found", which
/// already means the construct ends on its own line: the lines after it fall into later units,
/// so none is lost. Preparation runs before the review clock (RV06), so these two bounds are
/// the only ones on a file built to keep a parameter list open.
///
/// `lines` and `scanned` describe the same text, both split with [`str::lines`]
/// ([`CodeScanner::scan_all`]), so they have one length; a line with no record would read as
/// opening no brace and being no code, which ends the search, never a panic.
pub(super) fn open_brace_line(
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
    let group_open = |idx: usize| scanned.get(idx).map_or(0, |s| i64::from(s.group_delta));
    // MAGI FIX (B2b1 7): a brace a default argument opens while the parameter list is still open
    // (a lambda, `f(cb = () => {`) is the argument's, not the function's: the search passes over
    // the argument to the line that closes the list.
    let mut in_default_argument = opens(i)
        && group_open(i) > 0
        && lines.get(i).is_some_and(|l| {
            l.split_once('(')
                .is_some_and(|(_, list)| holds_bare_assignment(list))
        });
    if opens(i) && !in_default_argument {
        // MAGI FIX (B2a 1): a parameter typed by a literal block (`opts: {`, `struct {`) opened
        // the signature's first brace, so the function ended where that block closed (`}) {`).
        if group_open(i) > 0 {
            return Some(brace_after_parameter_list(lines, scanned, i, search).unwrap_or(i));
        }
        // MAGI FIX (loop 28 sweep): the brace of a return type's literal block
        // (`function f(): {`, `func f() struct {`) was the body's, and the body fell into a gap.
        if lines
            .get(i)
            .is_some_and(|l| opens_return_type_literal(l.trim()))
        {
            return Some(brace_after_return_type_literal(lines, scanned, i, search).unwrap_or(i));
        }
        return Some(i);
    }
    // A line whose code ends in `}` after opening nothing net (`fn f() -> u8 { 1 }`) is a
    // one-line construct: searching on took the brace of the block after it.
    let ends_block = |idx: usize| is_code(idx) && scanned.get(idx).is_some_and(|s| s.ends_block);
    if !in_default_argument && (ends_statement(i) || ends_block(i)) {
        return None;
    }
    // The line that closes the list after a default argument's brace (`}) {`) nets no brace,
    // yet opens the function's after its lowest point.
    let reopens = |idx: usize| {
        scanned
            .get(idx)
            .is_some_and(|s| s.brace_delta > s.min_delta)
    };
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
        if in_default_argument {
            if list_open > 0 {
                continue;
            }
            in_default_argument = false;
            if reopens(j) {
                return Some(j);
            }
        } else if open_before > 0
            && list_open > 0
            && opens(j)
            && lines.get(j).is_some_and(|l| holds_bare_assignment(l))
        {
            in_default_argument = true;
            continue;
        }
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
        // MAGI FIX (S6 loop 25, B2b1b): a block comment's line opening with a keyword
        // (`   return early */`) read as a statement and cut the body from its signature; a line
        // that starts inside a comment or a string opens no statement.
        let opens_code_statement = scanned.get(j).is_some_and(|s| !s.starts_in_literal)
            && lines.get(j).is_some_and(|l| opens_statement(l.trim()));
        let ends_search =
            (open_before <= 0 && !continues && search.stops(j)) || opens_code_statement;
        if ends_search {
            return None;
        }
        if opens(j) {
            // MAGI FIX (S6 loop 30, B2b1b): a brace on a later parameter line with the list still
            // open (`options: {` under `function f(`) was taken for the body's; it is the
            // parameter's type, and the body's brace is found after the list closes.
            if open_before > 0 && list_open > 0 {
                return Some(
                    brace_after_open_list(lines, scanned, j, list_open, search).unwrap_or(j),
                );
            }
            return Some(j);
        }
        if ends_statement(j) {
            return None;
        }
    }
    None
}

/// The 0-based line that closes the parameter list line `i` leaves open and, after its lowest
/// point, opens a brace (`}) {`): the body of a function one of whose parameters opened a block
/// on line `i` (`function f(opts: {`, `void f(struct {`). `None` when the line that closes the
/// list opens nothing after it (a call passing a callback, `});`), when no line closes the list
/// within [`MAX_BRACE_SEARCH_LINES`] lines, or when the pass's budget is spent
/// ([`SearchBudget`]); the caller then keeps line `i`'s brace, as [`open_brace_line`] did before
/// this rule. The list is counted by the scanner in code context only
/// ([`ScannedLine::group_delta`]). O(k) for the k lines walked.
///
/// When the line that closes the list opens nothing and ends no statement (`})`), an Allman
/// brace opening the next code line (`{`) is the body's; `});` still opens none.
///
/// # Arguments
///
/// * `lines` - The text's lines.
/// * `scanned` - The scanner's record of each line.
/// * `i` - The 0-based line whose parameter list stays open after a brace it opens.
/// * `search` - The pass's search bounds; only its budget is spent here.
///
/// # Returns
///
/// The 0-based line holding the function's opening brace, or `None`.
fn brace_after_parameter_list(
    lines: &[&str],
    scanned: &[ScannedLine],
    i: usize,
    search: &BraceSearch<'_>,
) -> Option<usize> {
    brace_after_open_list(lines, scanned, i, code_group_delta(scanned, i), search)
}

/// The 0-based line that closes a parameter list still open after line `i`, `open` of its
/// parentheses and brackets open there, and opens a brace after its lowest point, read as
/// [`brace_after_parameter_list`] reads it: line `i` holds a parameter's brace, either on the
/// line the list opens (`function f(opts: {`) or on a later parameter line (`options: {` under
/// `function f(`). `None` as there. O(k) for the k lines walked, at most
/// [`MAX_BRACE_SEARCH_LINES`] past line `i`, each charged to the pass's [`SearchBudget`].
///
/// # Arguments
///
/// * `lines` - The text's lines.
/// * `scanned` - The scanner's record of each line.
/// * `i` - The 0-based line holding the parameter's brace.
/// * `open` - The parentheses and brackets of the list still open after line `i`.
/// * `search` - The pass's search bounds; only its budget is spent here.
///
/// # Returns
///
/// The 0-based line holding the function's opening brace, or `None`.
fn brace_after_open_list(
    lines: &[&str],
    scanned: &[ScannedLine],
    i: usize,
    mut open: i64,
    search: &BraceSearch<'_>,
) -> Option<usize> {
    let end = i
        .saturating_add(1)
        .saturating_add(MAX_BRACE_SEARCH_LINES)
        .min(scanned.len());
    for j in (i + 1)..end {
        if !search.spend() {
            return None;
        }
        open += code_group_delta(scanned, j);
        if open <= 0 {
            return body_brace_from(lines, scanned, j, search);
        }
    }
    None
}

/// The 0-based line holding a function's opening brace, given line `j`, where what kept the brace
/// of line `i` from being the body's (a parameter list, a return type's literal block) closes:
/// line `j` itself when it opens a brace after its lowest point (`}) {`, `} {`, `}> {`,
/// `} => {`), else the next line when line `j` ends no statement and the next line is code
/// opening with an Allman `{`. `None` otherwise (`});`, `};`), so the caller keeps line `i`'s
/// brace. Looking at the next line spends one visit of the pass ([`SearchBudget`]). O(1).
fn body_brace_from(
    lines: &[&str],
    scanned: &[ScannedLine],
    j: usize,
    search: &BraceSearch<'_>,
) -> Option<usize> {
    let record = scanned.get(j)?;
    if record.brace_delta > record.min_delta {
        return Some(j);
    }
    // MAGI FIX (loop 26, 4): an Allman brace on the line after `})` was never looked at,
    // so the body of `function f(opts: {` ... `})` then `{` fell into a gap unit.
    // MAGI FIX (loop 28, B2b1b): the look at the next line was charged only when it found a
    // brace; every look is charged now.
    let next = j + 1;
    let allman = !record.ends_statement
        && next < lines.len()
        && search.spend()
        && scanned.get(next).is_some_and(|s| !s.is_non_code)
        && lines
            .get(next)
            .is_some_and(|l| l.trim_start().starts_with('{'));
    allman.then_some(next)
}

/// What the text before the `{` a TypeScript signature line ends with may end in when that brace
/// opens an object type in the return type, not the body: the annotation's `:`
/// (`function f(): {`), a type argument (`): Promise<{`, `): Map<K, {`), a union or an
/// intersection (`): A | {`, `): A & {`).
const RETURN_TYPE_POSITION_ENDS: [char; 5] = [':', '<', ',', '|', '&'];

/// The Go keywords whose `{` after a function's parameter list opens an anonymous type in its
/// result (`func f() struct {`), not the body.
const GO_RESULT_TYPE_LITERALS: [&str; 2] = ["struct", "interface"];

/// Whether the `{` a trimmed signature line ends with opens a literal type in the function's
/// return type rather than its body: after the line's last `)` comes a TypeScript annotation
/// (`:`) and the text before the brace ends in a type position ([`RETURN_TYPE_POSITION_ENDS`]),
/// or a Go anonymous type keyword alone ([`GO_RESULT_TYPE_LITERALS`]). The `:` rules out a C++
/// ref-qualifier (`void f() & {`) and a constructor's initializer list (`S() : a_{`), whose brace
/// is a body or an initializer. A comment after the brace hides it, declared: the line then
/// reads as opening the body, as before this rule. O(n) in the line.
fn opens_return_type_literal(trimmed: &str) -> bool {
    let Some(before) = trimmed.strip_suffix('{').map(str::trim_end) else {
        return false;
    };
    let Some((_, after_list)) = before.rsplit_once(')') else {
        return false;
    };
    let after_list = after_list.trim();
    (after_list.starts_with(':') && before.ends_with(RETURN_TYPE_POSITION_ENDS))
        || GO_RESULT_TYPE_LITERALS.contains(&after_list)
}

/// The 0-based line holding the body's brace of a function whose line `i` ends in a return
/// type's literal block ([`opens_return_type_literal`]): the line where the braces line `i`
/// leaves open all close, read as [`body_brace_from`] reads it. `None` when that line opens no
/// body (a method signature with no body, `};`), when no line closes them within
/// [`MAX_BRACE_SEARCH_LINES`] lines, or when the pass's budget is spent ([`SearchBudget`]); the
/// caller then keeps line `i`'s brace, as before this rule. O(k) for the k lines walked.
fn brace_after_return_type_literal(
    lines: &[&str],
    scanned: &[ScannedLine],
    i: usize,
    search: &BraceSearch<'_>,
) -> Option<usize> {
    let end = i
        .saturating_add(1)
        .saturating_add(MAX_BRACE_SEARCH_LINES)
        .min(scanned.len());
    let mut open = i64::from(scanned.get(i)?.brace_delta);
    for j in (i + 1)..end {
        if !search.spend() {
            return None;
        }
        let record = scanned.get(j)?;
        if open + i64::from(record.min_delta) <= 0 {
            return body_brace_from(lines, scanned, j, search);
        }
        open += i64::from(record.brace_delta);
    }
    None
}

/// The parentheses and square brackets line `idx` opens minus those it closes, counted by the
/// scanner in code context only ([`ScannedLine::group_delta`]): one inside a string, a raw
/// string, a character literal or a comment counts nothing. 0 for a line that does not exist.
/// O(1).
pub(super) fn code_group_delta(scanned: &[ScannedLine], idx: usize) -> i64 {
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
pub(super) const FUNCTION_EXPRESSION_KEYWORDS: [&str; 3] = ["const ", "let ", "var "];

/// What a declaration line may end with when its value goes on to the next line (`const f = (`,
/// `const t = [`, `int a,`).
const DECLARATION_CONTINUATIONS: [char; 4] = ['=', '(', '[', ','];

/// The arrow a JavaScript or TypeScript arrow function ends its header with (`() =>`).
const ARROW: &str = "=>";

/// The words a JavaScript function expression's value opens with (`function (x) {`).
const FUNCTION_VALUE_WORDS: [&str; 2] = ["function", "async function"];

/// Whether a declaration's value (the text after its `=`, trimmed at its start) is a function
/// expression ([`FUNCTION_VALUE_WORDS`], as whole words: `functional(x)` is a call). O(1).
pub(super) fn opens_function_value(value: &str) -> bool {
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
pub(super) fn declaration_block(
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
pub(super) fn declares_variable(rest: &str) -> bool {
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
pub(super) fn returns_from_function(rest: &str) -> bool {
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

/// Whether a trimmed line opens an initializer it continues on the next lines: it ends in a bare
/// `=` ([`ends_in_bare_equal`]: `static int n =`), in `=` followed by any run of `{` and blanks
/// (`int a[] = {`, a nested `int m[2][2] = {{`) or in a call's `(` after the `=`
/// ([`opens_call_initializer`]). O(n) in the line.
pub(super) fn opens_initializer(trimmed: &str) -> bool {
    // MAGI FIX (loop 27, B2b1b): any line ending in `=` opened one, a compound assignment
    // (`x +=`) and a comparison (`x ==`) included.
    // MAGI FIX (loop 28, B2b1b): only one trailing `{` was stripped, so `= {{` opened none.
    ends_in_bare_equal(trimmed.trim_end_matches(|c: char| c == '{' || c.is_whitespace()))
        || opens_call_initializer(trimmed)
}

/// The characters that, right before a `=`, make it part of another operator: a comparison
/// (`==`, `!=`, `<=`, `>=`) or a compound assignment (`+=`, `<<=`, `^=`, ...). Shared by
/// [`ends_in_bare_equal`] and [`holds_bare_assignment`], so the two never disagree on which `=`
/// is bare.
const OPERATOR_BEFORE_EQUAL: &str = "=!<>+-*/%&|^";

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
/// later units, none lost. A line whose text ends in `(` but leaves no parenthesis open in code
/// (`int x = 1; // note (`) opens none. O(k) for the k lines walked.
pub(super) fn call_initializer_end(
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
    // MAGI FIX (loop 27, B2b1b): a `(` ending the line's comment opened a call and ran the
    // declaration over the next one.
    if depth <= 0 {
        return i;
    }
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
pub(super) fn group_end(
    lines: &[&str],
    scanned: &[ScannedLine],
    i: usize,
    search: &BraceSearch<'_>,
) -> usize {
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
/// comparison (`==`, `!=`, `<=`, `>=`) nor a compound assignment (`+=`, `<<=`), as
/// [`holds_bare_assignment`] reads them ([`OPERATOR_BEFORE_EQUAL`]). O(1).
fn ends_in_bare_equal(trimmed: &str) -> bool {
    // MAGI FIX (loop 27, B2b1b): `x +=` read as a bare `=`, unlike `holds_bare_assignment`.
    trimmed.strip_suffix('=').is_some_and(|rest| {
        !rest
            .chars()
            .next_back()
            .is_some_and(|c| OPERATOR_BEFORE_EQUAL.contains(c))
    })
}

/// Whether `text` holds a bare `=`, an assignment or a default argument (`cb = () => {`), not
/// part of `==`, `!=`, `<=`, `>=`, `=>` or a compound assignment (`+=`). A heuristic on the
/// text: a `=` inside a string counts too. O(n) in the text.
///
/// # Arguments
///
/// * `text` - A line's text, or part of it.
///
/// # Returns
///
/// Whether a bare `=` is in it.
fn holds_bare_assignment(text: &str) -> bool {
    let mut previous = ' ';
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        let next = chars.peek().copied().unwrap_or(' ');
        if c == '=' && !OPERATOR_BEFORE_EQUAL.contains(previous) && !matches!(next, '=' | '>') {
            return true;
        }
        previous = c;
    }
    false
}

/// The 0-based last line of the value a declaration at line `i` puts after a bare `=` on the
/// next lines (`pub static T: &[u8] =` then `&[1, 2, 3];`): the end of that statement, the first
/// later line whose code ends in `;` ([`statement_end`], which stops at a blank line, at the next
/// construct and when the pass's budget is spent); with none, as in JavaScript or Go without
/// semicolons, the next line when it is code that starts no construct, through the parentheses
/// or brackets it leaves open ([`group_end`], `g(` ... `)`), so the value is kept and the
/// declaration after it is never taken. `i` when line `i` is no code ending in a bare `=`
/// ([`ends_in_bare_equal`]), or no line qualifies. O(k) for the k lines searched.
pub(super) fn bare_initializer_end(
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
    // MAGI FIX (B2b1 6): a value wrapped over lines without a `;` (`g(` ... `)`, `[` ... `]`)
    // ended on its first line; it runs to the delimiter that line leaves open.
    if value {
        group_end(lines, scanned, next, search)
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
///
/// It spends no visit of the pass's [`SearchBudget`], unlike the forward searches: its callers
/// (a declaration group and its split, `BraceFamilyExtractor::declaration_end`) resume after the
/// line it returns, so each directive's lines are walked once and a pass's walks add O(n) in all.
pub(super) fn preprocessor_end(lines: &[&str], i: usize) -> usize {
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
pub(super) fn statement_end(
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
pub(super) fn brace_block_end(
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
pub(super) fn block_end_after(
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
