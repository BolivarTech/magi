// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! Python-specific semantic extractor using indentation-based block detection.
//!
//! Recognizes `def`, `async def`, `class`, `import`/`from`, and module-level
//! assignments. Block boundaries are determined by indentation level changes.

use super::python_strings::PythonStrings;
use super::scanner::{CodeScanner, LexicalRules};
use super::{
    gap_units, split_into_members, ChangedLines, LanguageExtractor, Member, SemanticUnit, UnitKind,
};
use crate::review::context::FileContext;

/// Indentation-based Python code extractor.
///
/// Detects:
/// - `def` / `async def` → [`UnitKind::Function`]
/// - `class` → [`UnitKind::Class`]
/// - `import` / `from ... import` → [`UnitKind::DeclarationGroup`]
/// - Module-level assignments (indent 0) → [`UnitKind::DeclarationGroup`]
#[derive(Debug, Clone, Copy)]
pub struct PythonExtractor;

impl LanguageExtractor for PythonExtractor {
    fn language_id(&self) -> &str {
        "python"
    }

    fn extensions(&self) -> &[&str] {
        &["py", "pyi"]
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
        let constructs = detect_python_constructs(&lines);
        let ranges: Vec<(u32, u32)> = constructs
            .iter()
            .map(|c| (c.start_line, c.end_line))
            .collect();
        // MAGI FIX (S6 loop 3): a whole file hands out its lines by range, without a set of all.
        let total = u32::try_from(lines.len()).unwrap_or(u32::MAX);
        let (shares, orphans) = changed_lines.assign(&ranges, total);

        let mut units = Vec::new();
        for (construct, changed_in_unit) in constructs.iter().zip(shares) {
            if !changed_in_unit.is_empty() {
                // MAGI FIX (S6 loop 9): the start is clamped to the end before slicing, so a
                // range past the file never indexes out of it. A construct's lines always fit
                // `u32`: a reviewed file is at most `MAX_REVIEW_FILE_BYTES` (8 MiB), so at most
                // 8 388 608 lines, far below `u32::MAX`.
                let end_idx = (construct.end_line as usize).min(lines.len());
                let start_idx = (construct.start_line.saturating_sub(1) as usize).min(end_idx);
                let unit_content = lines.get(start_idx..end_idx).unwrap_or(&[]).join("\n");

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

    /// Splits a class at its methods: each `def` / `async def` at the indentation of the
    /// class body, with its decorators, on a line that starts a logical line (outside
    /// triple-quoted strings and brackets; see `continued_lines`). O(n) in the class's lines.
    /// A `class` nested at that indentation is a member too, of kind class, as the trait asks of
    /// other nested constructs; a function nested in a method stays in its method (RV03).
    ///
    /// A class with no such member (one holding only data attributes, as generated code may) or
    /// a unit of another kind gives no pieces, and is batched whole: if it is larger than one
    /// batch, the review ends with the typed `ReviewError::TooLargeForBatch` (RV03) before the
    /// clock starts, never a silent cut. Pinned by
    /// `construct_ranges_are_sorted_and_disjoint_and_a_memberless_class_does_not_split` here and
    /// `review::batching`'s `a_single_unit_larger_than_the_limit_is_a_typed_error`.
    ///
    /// A declaration group (imports or module-level assignments) splits at its statements
    /// instead (`split_declaration_group`).
    fn split_block(&self, unit: &SemanticUnit) -> Vec<SemanticUnit> {
        if unit.kind == UnitKind::DeclarationGroup {
            return split_declaration_group(unit);
        }
        // MAGI FIX (S6 loop 3): a class too large for one batch is reviewed method by method.
        if unit.kind != UnitKind::Class {
            return Vec::new();
        }
        let lines: Vec<&str> = unit.content.split('\n').collect();
        let continued = continued_lines(&lines);
        let code = |j: usize| !continued.get(j).copied().unwrap_or(false);
        let Some(class_idx) = lines
            .iter()
            .enumerate()
            // MAGI FIX (S6 loop 13): a `class` line inside a decorator's string was the header.
            .position(|(j, l)| code(j) && after_keyword(l.trim(), "class").is_some())
        else {
            return Vec::new();
        };
        let Some(body_indent) = lines
            .iter()
            .enumerate()
            .skip(class_idx + 1)
            .find(|&(j, l)| code(j) && !l.trim().is_empty() && !l.trim().starts_with('#'))
            .map(|(_, l)| indentation(l))
        else {
            return Vec::new();
        };
        let mut members: Vec<Member> = Vec::new();
        let mut j = class_idx + 1;
        while let Some(&line) = lines.get(j) {
            let trimmed = line.trim();
            // MAGI FIX (loop 26, 8): only `def` lines were members, so a nested `class Inner:`
            // stayed in the outer class's header piece; it is a member of kind class.
            let member = def_rest(trimmed)
                .map(|rest| (UnitKind::Function, rest))
                .or_else(|| after_keyword(trimmed, "class").map(|rest| (UnitKind::Class, rest)));
            if let Some((kind, rest)) =
                member.filter(|_| code(j) && indentation(line) == body_indent)
            {
                let start = scan_back_decorators(&lines, &continued, j);
                let end = find_block_end(&lines, &continued, j);
                // MAGI FIX (S6 loop 9): a `u32::MAX` fallback split the class at a fabricated
                // range; a line that does not fit `u32` leaves the class whole.
                let Some((first, last)) = member_span(start, end) else {
                    return Vec::new();
                };
                members.push((kind, python_name(Some(rest)), first, last));
                j = end + 1;
            } else {
                j += 1;
            }
        }
        split_into_members(unit, members)
    }
}

/// The pieces of a declaration group too large for one batch: each statement (a logical line,
/// with the lines that continue it, see [`continued_lines`]) as a piece of the group's kind and
/// name, and each line between two (a blank line) as a stretch piece ([`split_into_members`]).
/// Splitting at statements is no silent cut (RV03): every line stays in exactly one piece.
///
/// # Arguments
///
/// * `unit` - A [`UnitKind::DeclarationGroup`] unit of this extractor.
///
/// # Returns
///
/// The pieces in line order; empty when the group holds no statement or a line number does not
/// fit `u32`, so the caller batches it whole. O(n) in the group's lines.
fn split_declaration_group(unit: &SemanticUnit) -> Vec<SemanticUnit> {
    // MAGI FIX (S6 loop 12): a group was never split, so a generated constant table over the
    // batch limit refused the whole review although each statement fits.
    let lines: Vec<&str> = unit.content.split('\n').collect();
    let continued = continued_lines(&lines);
    let continues = |j: usize| continued.get(j).copied().unwrap_or(false);
    let mut members: Vec<Member> = Vec::new();
    let mut j = 0;
    while let Some(line) = lines.get(j) {
        if continues(j) || line.trim().is_empty() {
            j += 1;
            continue;
        }
        let mut end = j;
        while continues(end + 1) {
            end += 1;
        }
        let Some((first, last)) = member_span(j, end) else {
            return Vec::new();
        };
        members.push((UnitKind::DeclarationGroup, unit.name.clone(), first, last));
        j = end + 1;
    }
    split_into_members(unit, members)
}

/// The 1-based, inclusive line span of a member found at the 0-based lines `start..=end`.
///
/// # Arguments
///
/// * `start` - The member's first line, 0-based.
/// * `end` - The member's last line, 0-based.
///
/// # Returns
///
/// The span, or `None` when a line number does not fit `u32`. O(1).
fn member_span(start: usize, end: usize) -> Option<(u32, u32)> {
    Some((line_number(start)?, line_number(end)?))
}

/// The 1-based line number of the 0-based line `idx`, or `None` when it does not fit `u32`.
/// O(1).
fn line_number(idx: usize) -> Option<u32> {
    u32::try_from(idx.checked_add(1)?).ok()
}

/// Whether each of `lines` continues a logical line begun on an earlier one: it starts inside a
/// triple-quoted string (as the Python scanner reads it), inside an open bracket (`(`, `[`,
/// `{`), or after a line ending in a `\` continuation. Python ignores the indentation of such a
/// line, so it never ends a block nor starts a construct. O(n) in the text.
///
/// Declared: an unclosed bracket (invalid code) carries the continuation to the end of the
/// text, so the construct open there takes the rest of the file; no line is lost.
fn continued_lines(lines: &[&str]) -> Vec<bool> {
    // MAGI FIX (S6 loop 4): a wrapped header (`def f(\n    x,\n) -> None:`) or a dedented
    // continuation (`)` at column 0) ended the block by its indentation.
    let literal = CodeScanner::new(LexicalRules::python()).scan_all(&lines.join("\n"));
    let mut brackets = LogicalLines::default();
    lines
        .iter()
        .enumerate()
        .map(|(j, line)| {
            let continued = brackets.continues();
            brackets.read(line);
            continued || literal.get(j).is_some_and(|s| s.starts_in_literal)
        })
        .collect()
}

/// What carries a Python logical line over to the next physical line: open brackets, an open
/// string (a triple-quoted one, a replacement field of an f-string, or a one-quote string
/// continued by an escaped newline), or a trailing `\`. Strings and comments are read so that
/// their brackets do not count.
///
/// It overlaps [`CodeScanner`] on purpose: the scanner counts braces only and knows nothing of
/// `(` and `[`, which carry a Python line as much as `{` does. Both hand every string to
/// [`PythonStrings`], so they share one string grammar, f-strings included (PEP 701: a string
/// inside a replacement field may reuse the outer quote, as in `f"{"("}"`, and a field may
/// span lines); [`continued_lines`] takes the union of both.
///
/// In Python's tokenizer a backslash escapes the next character in a raw string too as far as
/// where the string ends (`r"\""` is the two characters `\"`, and `r"""a\"""` never closes),
/// and [`PythonStrings`] reads it so. Only a `\` outside every string and comment continues a
/// line, and Python accepts one there only right before the newline, so no parity of
/// backslashes is counted. Both are pinned by
/// `a_backslash_before_a_quote_in_a_raw_string_never_closes_it` and
/// `an_escaped_backslash_or_a_comment_ending_in_one_continues_no_line`.
// MAGI FIX (S6 loop 13): the raw-string and backslash behavior is declared.
// MAGI FIX (PEP 701): one open string at a time read `f"{"("}"` as closing early and kept its
// `(` counted open, so the construct there took the rest of the file.
#[derive(Default)]
struct LogicalLines {
    depth: u32,
    strings: PythonStrings,
    backslash: bool,
}

impl LogicalLines {
    /// Whether the next line to read continues the current logical line.
    fn continues(&self) -> bool {
        self.depth > 0 || self.backslash || self.strings.is_open()
    }

    /// Reads one physical line. O(n) in the line.
    fn read(&mut self, line: &str) {
        let chars: Vec<char> = line.chars().collect();
        let mut last_code: Option<char> = None;
        let mut k = self.strings.read(&chars, 0);
        while let Some(&c) = chars.get(k) {
            match c {
                '#' => break,
                '"' | '\'' => {
                    last_code = Some(c);
                    k = self.strings.open(&chars, k).unwrap_or(k + 1);
                    continue;
                }
                '(' | '[' | '{' => self.depth = self.depth.saturating_add(1),
                ')' | ']' | '}' => self.depth = self.depth.saturating_sub(1),
                _ => {}
            }
            if !c.is_whitespace() {
                last_code = Some(c);
            }
            k += 1;
        }
        self.strings.end_line();
        self.backslash = !self.strings.is_open() && last_code == Some('\\');
    }
}

/// Intermediate detected construct.
struct PythonConstruct {
    kind: UnitKind,
    name: String,
    start_line: u32,
    end_line: u32,
}

/// Detect Python constructs using indentation.
///
/// No constructs when a line number does not fit `u32` ([`try_detect_python_constructs`]): the
/// file's changed lines then fall into gap units, never into fabricated ranges.
fn detect_python_constructs(lines: &[&str]) -> Vec<PythonConstruct> {
    try_detect_python_constructs(lines).unwrap_or_default()
}

/// The constructs of [`detect_python_constructs`], or `None` when a line number does not fit
/// `u32`. The 8 MiB file cap keeps that out of reach (8 388 608 lines at most); the check makes
/// the failure mode explicit, as the brace family's detection and [`member_span`] do. O(n) in
/// the lines.
///
/// Two limitations, declared, both in the safe direction (every line stays in exactly one unit):
///
/// - Definitions, classes, imports and assignments are found at indent 0 only. One inside a
///   module-level compound statement (`if TYPE_CHECKING:`, a `try:` around a compatibility
///   import, a `with` block) is no unit of its own: the whole block is one gap unit, never
///   split, so a block larger than a batch ends the review with the typed
///   `ReviewError::TooLargeForBatch` (RV03) instead of being cut. Pinned by
///   `a_definition_inside_a_module_level_block_falls_into_a_whole_gap_unit`.
/// - A UTF-8 byte order mark before the first line's keyword (`\u{FEFF}def f():`) is kept by
///   `str::trim`, since U+FEFF is no whitespace, so that first construct is not recognised
///   (its header may read as a one-line assignment and its body falls into a gap unit). Only
///   the unit structure of that construct degrades; every line is still reviewed. Pinned by
///   `a_byte_order_mark_on_the_first_line_loses_no_line`.
// MAGI FIX (S6 loop 14): the indent-0 and byte-order-mark limitations are declared and pinned.
fn try_detect_python_constructs(lines: &[&str]) -> Option<Vec<PythonConstruct>> {
    // MAGI FIX (S6 loop 12): the conversions fell back to `u32::MAX`, `1` or the line number,
    // unlike the brace family since loop 10; an unrepresentable line now aborts detection.
    let continued = continued_lines(lines);
    let mut constructs = Vec::new();
    let mut i = 0;

    while let Some(&line) = lines.get(i) {
        // MAGI FIX (S6 loop 3): a line that starts inside a triple-quoted string is string text,
        // never a definition, import or assignment; (S6 loop 4) nor is a line that continues a
        // bracket or a `\` continuation.
        if continued.get(i).copied().unwrap_or(false) {
            i += 1;
            continue;
        }
        let trimmed = line.trim();
        let indent = indentation(line);
        let line_num = line_number(i)?;

        // MAGI FIX (S6 loop 13): a keyword matched with one literal space, so `def\tf():` or
        // `class  C:` (legal Python) was no construct; any blank separates it from its name.
        // def / async def at indent 0 → Function
        if let Some(rest) = def_rest(trimmed).filter(|_| indent == 0) {
            let name = python_name(Some(rest));
            let start = scan_back_decorators(lines, &continued, i);
            let end = find_block_end(lines, &continued, i);
            constructs.push(PythonConstruct {
                kind: UnitKind::Function,
                name,
                start_line: line_number(start)?,
                end_line: line_number(end)?,
            });
            i = end + 1;
            continue;
        }

        // class at indent 0 → Class
        if let Some(rest) = after_keyword(trimmed, "class").filter(|_| indent == 0) {
            let name = python_name(Some(rest));
            let start = scan_back_decorators(lines, &continued, i);
            let end = find_block_end(lines, &continued, i);
            constructs.push(PythonConstruct {
                kind: UnitKind::Class,
                name,
                start_line: line_number(start)?,
                end_line: line_number(end)?,
            });
            i = end + 1;
            continue;
        }

        // import / from ... import at indent 0 → DeclarationGroup
        if opens_import(trimmed) && indent == 0 {
            let group_end = find_import_group_end(lines, &continued, i);
            constructs.push(PythonConstruct {
                kind: UnitKind::DeclarationGroup,
                name: "imports".to_string(),
                start_line: line_num,
                end_line: line_number(group_end)?,
            });
            i = group_end + 1;
            continue;
        }

        // Module-level assignment at indent 0 → DeclarationGroup
        if indent == 0 && is_module_assignment(trimmed) {
            let group_end = find_assignment_group_end(lines, &continued, i);
            let name = trimmed
                .split(['=', ':', ' '])
                .next()
                .unwrap_or("variable")
                .to_string();
            constructs.push(PythonConstruct {
                kind: UnitKind::DeclarationGroup,
                name,
                start_line: line_num,
                end_line: line_number(group_end)?,
            });
            i = group_end + 1;
            continue;
        }

        i += 1;
    }

    Some(constructs)
}

/// Words that open a statement which is never a module-level assignment: definitions, imports,
/// the compound statements and the simple-statement keywords, whose `:`, `=` and `==` would
/// otherwise read as an assignment.
const NOT_ASSIGNMENT_WORDS: &[&str] = &[
    "def", "async", "class", "import", "from", "if", "elif", "else", "for", "while", "with", "try",
    "except", "finally", "assert", "raise", "del", "global", "nonlocal", "return", "yield",
    "await", "pass", "lambda", "not",
];

/// The soft keywords that open a compound statement (Python 3.10, PEP 634): `match` and `case`
/// open one only in the statement's own form, whose header ends in `:`; anywhere else they are
/// plain names (`match = 1`, `case: int = 2`), which CPython accepts.
const SOFT_STATEMENT_WORDS: &[&str] = &["match", "case"];

/// Whether an unindented, trimmed line is a module-level assignment (`X = 1`, `x: int = 1`,
/// `y += 1`): it holds an assignment `=` or an annotation `:` outside every bracket and string
/// ([`has_top_level_assignment`]), is no comment or decorator, does not open with a word of
/// [`NOT_ASSIGNMENT_WORDS`], and is no statement a word of [`SOFT_STATEMENT_WORDS`] opens (one
/// whose code ends in `:`, [`ends_in_colon`]). O(n) in the line.
fn is_module_assignment(trimmed: &str) -> bool {
    // MAGI FIX (loop 27, B2b2): `match = 1` and `case = 2` (legal names since 3.10) fell into
    // gap units, the soft keywords being read as hard ones.
    let first_word = trimmed
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .next()
        .unwrap_or("");
    let soft_statement = SOFT_STATEMENT_WORDS.contains(&first_word) && ends_in_colon(trimmed);
    !trimmed.is_empty()
        && !trimmed.starts_with('#')
        && !trimmed.starts_with('@')
        && !NOT_ASSIGNMENT_WORDS.contains(&first_word)
        && !soft_statement
        && has_top_level_assignment(trimmed)
}

/// Whether the last character `line` has outside every string and comment is `:`, as the
/// header of a compound statement ends (`match x:  # note`). Strings are read with
/// [`PythonStrings`]; a string still open at the line's end ends no header. O(n) in the line.
fn ends_in_colon(line: &str) -> bool {
    let chars: Vec<char> = line.chars().collect();
    let mut strings = PythonStrings::default();
    let mut last: Option<char> = None;
    let mut k = 0;
    while let Some(&c) = chars.get(k) {
        match c {
            '#' => break,
            '"' | '\'' => {
                last = Some(c);
                k = strings.open(&chars, k).unwrap_or(k + 1);
                if strings.is_open() {
                    return false;
                }
                continue;
            }
            _ if !c.is_whitespace() => last = Some(c),
            _ => {}
        }
        k += 1;
    }
    last == Some(':')
}

/// Whether `line` holds, outside every bracket, string and comment, an assignment `=` (plain or
/// augmented: `+=`, `<<=`; not the comparisons `==`, `!=`, `<=`, `>=`) or an annotation `:`
/// (not `:=`). A call's keyword argument (`f(level=1)`) and a string's `:` (`print("a:b")`) are
/// inside a bracket or a string and do not count. O(n) in the line.
///
/// Strings are read with [`PythonStrings`], the reader `LogicalLines` uses, so triple quotes,
/// prefixes and f-string fields follow one grammar: an `=` inside a triple-quoted string holding
/// an apostrophe (`'''It's a = b'''`) stays string text. A string still open at the line's end
/// (a triple-quoted string that continues) holds the rest of the line, so no operator follows.
fn has_top_level_assignment(line: &str) -> bool {
    // MAGI FIX (S6 loop 5): any `=` or `:` made a call (`setup(level=1)`) an assignment.
    // MAGI FIX (loop 26, 9): a triple-quoted string was read as three one-quote strings, so an
    // apostrophe inside it (`'''It's a = b'''`) left `a = b` outside every quote and the
    // docstring read as an assignment; the shared string reader tracks triple quotes.
    let chars: Vec<char> = line.chars().collect();
    let at = |k: usize| chars.get(k).copied();
    let mut depth = 0u32;
    let mut strings = PythonStrings::default();
    let mut k = 0;
    while let Some(c) = at(k) {
        match c {
            '#' => return false,
            '"' | '\'' => {
                k = strings.open(&chars, k).unwrap_or(k + 1);
                if strings.is_open() {
                    return false;
                }
                continue;
            }
            '(' | '[' | '{' => depth = depth.saturating_add(1),
            ')' | ']' | '}' => depth = depth.saturating_sub(1),
            ':' if depth == 0 => return at(k + 1) != Some('='),
            '=' if depth == 0 => {
                if at(k + 1) == Some('=') {
                    k += 2;
                    continue;
                }
                let prev = k.checked_sub(1).and_then(at);
                let shift = k.checked_sub(2).and_then(at) == prev;
                return match prev {
                    Some('!') => false,
                    Some('<' | '>') => shift,
                    _ => true,
                };
            }
            _ => {}
        }
        k += 1;
    }
    false
}

/// Get indentation level (number of leading spaces/tabs).
fn indentation(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

/// Find the end of a Python indented block: the last line before the first code line indented
/// no deeper than the definition.
///
/// As in Python's own tokenizer, a comment-only line and a line that continues a logical line
/// (`continued[j]`, from [`continued_lines`]: inside a triple-quoted string or a bracket, or
/// after a `\`) never end the block. A continued line extends the block; an
/// unindented comment does not, so a comment between two definitions stays outside both.
/// O(n) in the lines after `def_line`.
fn find_block_end(lines: &[&str], continued: &[bool], def_line: usize) -> usize {
    let def_indent = lines.get(def_line).map_or(0, |line| indentation(line));
    let mut end = def_line;
    for (j, line) in lines.iter().enumerate().skip(def_line + 1) {
        if continued.get(j).copied().unwrap_or(false) {
            end = j;
            continue;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue; // blank lines inside block
        }
        let ind = indentation(line);
        if trimmed.starts_with('#') && ind <= def_indent {
            continue;
        }
        if ind <= def_indent {
            break;
        }
        end = j;
    }
    end
}

/// Find end of a contiguous import group.
///
/// A line that continues a logical line (`continued[j]`: inside the brackets of
/// `from a import (`, or after a `\`) belongs to the import that opened it, so it extends the
/// group. O(n) in the lines after `start`.
fn find_import_group_end(lines: &[&str], continued: &[bool], start: usize) -> usize {
    let mut end = start;
    for (j, line) in lines.iter().enumerate().skip(start + 1) {
        // MAGI FIX (S6 loop 5): a wrapped import ended on its header line, and its names became
        // orphan units reviewed without it.
        if continued.get(j).copied().unwrap_or(false) {
            end = j;
            continue;
        }
        let trimmed = line.trim();
        if opens_import(trimmed) {
            end = j;
        } else if trimmed.is_empty() {
            // Allow one blank line
            if let Some(next) = lines.get(j + 1).map(|l| l.trim()) {
                if opens_import(next) {
                    continue;
                }
            }
            break;
        } else {
            break;
        }
    }
    end
}

/// Find end of a contiguous assignment group at indent 0: the assignments that follow one
/// another, one blank line allowed between two, as the import group and the brace-family
/// declaration groups allow.
///
/// A line that continues a logical line (`continued[j]`: inside a triple-quoted string or a
/// bracket, or after a `\`) belongs to the assignment that opened it, so it extends the group.
/// O(n) in the lines after `start`.
///
/// A run of adjacent assignments too large for one batch is split at its statements by
/// [`split_declaration_group`], never cut; one statement too large alone is reported as too
/// large.
fn find_assignment_group_end(lines: &[&str], continued: &[bool], start: usize) -> usize {
    let mut end = start;
    for (j, line) in lines.iter().enumerate().skip(start + 1) {
        if continued.get(j).copied().unwrap_or(false) {
            end = j;
            continue;
        }
        let trimmed = line.trim();
        let indent = indentation(line);
        if indent == 0 && !trimmed.is_empty() && !trimmed.starts_with('#') {
            if is_module_assignment(trimmed) {
                end = j;
                continue;
            }
            break;
        } else if trimmed.is_empty() {
            // MAGI FIX (S6 loop 7): any number of blank lines kept the group open, so far-apart
            // constants became one unit; one blank line is allowed, before an assignment.
            let next_assigns = !continued.get(j + 1).copied().unwrap_or(false)
                && lines
                    .get(j + 1)
                    .is_some_and(|l| indentation(l) == 0 && is_module_assignment(l.trim()));
            if !next_assigns {
                break;
            }
        } else {
            break;
        }
    }
    end
}

/// Scan backward for Python decorators (`@...`): the 0-based first line of the decorators
/// right above line `idx`, or `idx` when there are none. A decorator is a whole logical line, so
/// one wrapped over several lines (`@route(\n    "/a",\n)`) is taken with its continuation
/// lines, as `continued` marks them ([`continued_lines`]). Blank and comment-only lines between
/// two decorators or between a decorator and its definition are passed over, as Python's
/// tokenizer does, and taken only when a decorator stands above them: a comment right above a
/// definition with no decorator stays outside it. O(k) for the k lines looked at.
fn scan_back_decorators(lines: &[&str], continued: &[bool], idx: usize) -> usize {
    // MAGI FIX (S6 loop 6): a wrapped decorator's continuation lines were left out.
    // MAGI FIX (S6 loop 13): a blank or comment line between a decorator and its definition
    // stopped the scan, so the decorator was reviewed apart from the function.
    let is_code = |j: usize| !continued.get(j).copied().unwrap_or(false);
    let is_filler = |j: usize| {
        is_code(j)
            && lines
                .get(j)
                .is_some_and(|l| l.trim().is_empty() || l.trim().starts_with('#'))
    };
    let mut start = idx;
    while let Some(mut first) = start.checked_sub(1) {
        while is_filler(first) {
            let Some(above) = first.checked_sub(1) else {
                return start;
            };
            first = above;
        }
        while first > 0 && continued.get(first).copied().unwrap_or(false) {
            first -= 1;
        }
        if lines.get(first).is_some_and(|l| l.trim().starts_with('@')) {
            start = first;
        } else {
            break;
        }
    }
    start
}

/// The text after `keyword` at the start of a trimmed line, past the blanks that end the
/// keyword: Python separates a keyword from what follows it with any run of spaces and tabs
/// (`def\tf():`, `class  C:`). `None` when the line does not open with the keyword followed by
/// a blank (`define = 1`, `classes = []`). O(n) in the line.
///
/// # Arguments
///
/// * `trimmed` - A trimmed line.
/// * `keyword` - The bare keyword (`def`, `class`, `import`, `from`).
///
/// # Returns
///
/// The rest of the line, trimmed at its start, or `None`.
fn after_keyword<'t>(trimmed: &'t str, keyword: &str) -> Option<&'t str> {
    let rest = trimmed.strip_prefix(keyword)?;
    rest.starts_with(char::is_whitespace)
        .then(|| rest.trim_start())
}

/// The text after `def` or `async def` at the start of a trimmed line ([`after_keyword`]), or
/// `None` when the line opens no function definition. O(n) in the line.
fn def_rest(trimmed: &str) -> Option<&str> {
    after_keyword(trimmed, "def").or_else(|| after_keyword(after_keyword(trimmed, "async")?, "def"))
}

/// Whether a trimmed line opens an `import` or `from` statement ([`after_keyword`]). O(n).
fn opens_import(trimmed: &str) -> bool {
    after_keyword(trimmed, "import").is_some() || after_keyword(trimmed, "from").is_some()
}

/// The name of a definition: the identifier that opens `rest`, the text after its keyword
/// ([`after_keyword`]); empty when there is no such text or identifier. O(n) in the name.
fn python_name(rest: Option<&str>) -> String {
    rest.unwrap_or("")
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn extract(code: &str, changed: &[u32]) -> Vec<SemanticUnit> {
        let ext = PythonExtractor;
        let changed_set: HashSet<u32> = changed.iter().copied().collect();
        ext.extract_units(code, "test.py", &changed_set)
    }

    // --- Test 29: python_extracts_function ---
    #[test]
    fn python_extracts_function() {
        let code = "def foo():\n    x = 1\n    return x\n";
        let units = extract(code, &[2]);
        assert_eq!(units.len(), 1, "Should extract one function");
        assert_eq!(units[0].kind, UnitKind::Function);
        assert_eq!(units[0].name, "foo");
    }

    // --- Test 30: python_extracts_class ---
    #[test]
    fn python_extracts_class() {
        let code = "class Foo:\n    def __init__(self):\n        self.x = 1\n";
        let units = extract(code, &[3]);
        assert_eq!(units.len(), 1, "Should extract one class");
        assert_eq!(units[0].kind, UnitKind::Class);
        assert_eq!(units[0].name, "Foo");
    }

    // --- Test 31: python_extracts_module_level_assignment ---
    #[test]
    fn python_extracts_module_level_assignment() {
        let code = "MAX_SIZE = 100\nMIN_SIZE = 10\n\ndef foo():\n    pass\n";
        let units = extract(code, &[1]);
        assert_eq!(units.len(), 1, "Should extract one declaration group");
        assert_eq!(units[0].kind, UnitKind::DeclarationGroup);
    }

    // --- Test 32: python_extracts_import_group ---
    #[test]
    fn python_extracts_import_group() {
        let code = "import os\nimport sys\nfrom pathlib import Path\n\ndef main():\n    pass\n";
        let units = extract(code, &[2]);
        assert_eq!(units.len(), 1, "Should extract one import group");
        assert_eq!(units[0].kind, UnitKind::DeclarationGroup);
        assert_eq!(units[0].name, "imports");
    }

    #[test]
    fn every_gap_between_constructs_is_its_own_fallback_unit() {
        // Orphans around several constructs gave one window from the first to the last of
        // them, which held every construct in between.
        let code = "import os\n\n\ndef f():\n    pass\n\n# note\n\ndef g():\n    pass\n\n\ndef h():\n    pass\n";
        let every: Vec<u32> = (1..=14).collect();
        let units = extract(code, &every);
        for line in 1..=14u32 {
            let holders: Vec<&str> = units
                .iter()
                .filter(|u| u.start_line <= line && line <= u.end_line)
                .map(|u| u.name.as_str())
                .collect();
            assert_eq!(holders.len(), 1, "line {line} is in {holders:?}");
        }
    }

    #[test]
    fn an_unindented_comment_inside_a_function_does_not_end_it() {
        // Python ignores comment-only lines for indentation.
        let code = "def f():
    x = 1
# note
    return x

def g():
    pass
";
        let units = extract(code, &[4]);
        assert_eq!(units.len(), 1, "{units:?}");
        assert_eq!(
            (
                units[0].name.as_str(),
                units[0].start_line,
                units[0].end_line
            ),
            ("f", 1, 4)
        );
    }

    #[test]
    fn a_dedented_line_inside_a_triple_quoted_string_does_not_end_the_function() {
        for quote in ["\"\"\"", "'''"] {
            let code = format!(
                "def f():
    s = {quote}
not code
{quote}
    return s

def g():
    pass
"
            );
            let units = extract(&code, &[5]);
            assert_eq!(units.len(), 1, "{quote}: {units:?}");
            assert_eq!(
                (
                    units[0].name.as_str(),
                    units[0].start_line,
                    units[0].end_line
                ),
                ("f", 1, 5),
                "{quote}"
            );
        }
    }

    #[test]
    fn a_definition_inside_a_triple_quoted_string_is_not_a_construct() {
        // [MAGI S6 loop 3, B1] `def fake():` at column 0 inside a module string read as a
        // function, and the string's lines became its body.
        for quote in ["\"\"\"", "'''"] {
            let code = format!(
                "DOC = {quote}\ndef fake():\nclass Ghost:\nimport os\nX = 1\n{quote}\n\ndef real():\n    return 1\n"
            );
            let every: Vec<u32> = (1..=9).collect();
            let units = extract(&code, &every);
            let found: Vec<(&str, UnitKind, u32, u32)> = units
                .iter()
                .map(|u| (u.name.as_str(), u.kind, u.start_line, u.end_line))
                .collect();
            assert!(
                !found
                    .iter()
                    .any(|(n, ..)| ["fake", "Ghost", "imports", "X"].contains(n)),
                "{quote}: {found:?}"
            );
            assert!(
                found.contains(&("DOC", UnitKind::DeclarationGroup, 1, 6)),
                "{quote}: the string belongs to its assignment: {found:?}"
            );
            assert!(
                found.contains(&("real", UnitKind::Function, 8, 9)),
                "{quote}: {found:?}"
            );
        }
    }

    #[test]
    fn a_class_splits_into_its_methods_and_keeps_its_header() {
        // [MAGI S6 loop 3, B4] a class was one unit: one larger than the batch failed the review
        // although each of its methods fits. A `def` inside a string is not a method.
        let code = "@dataclass\nclass Foo:\n    x = 1\n\n    @property\n    def a(self):\n        return 1\n\n    async def b(self):\n        s = '''\n    def fake():\n'''\n        return s\n";
        let every: HashSet<u32> = (1..=13).collect();
        let block = PythonExtractor
            .extract_units(code, "test.py", &every)
            .into_iter()
            .find(|u| u.kind == UnitKind::Class)
            .expect("the class");
        let parts = PythonExtractor.split_block(&block);
        let found: Vec<(&str, UnitKind, u32, u32)> = parts
            .iter()
            .map(|u| (u.name.as_str(), u.kind, u.start_line, u.end_line))
            .collect();
        assert_eq!(
            found,
            vec![
                ("Foo", UnitKind::Class, 1, 4),
                ("a", UnitKind::Method, 5, 7),
                ("Foo", UnitKind::Class, 8, 8),
                ("b", UnitKind::Method, 9, 13),
            ]
        );
        assert_eq!(parts[0].content, "@dataclass\nclass Foo:\n    x = 1\n");
        let def = extract("def f():\n    pass\n", &[2]);
        assert!(PythonExtractor.split_block(&def[0]).is_empty());
    }

    #[test]
    fn a_declaration_group_splits_at_its_statements() {
        // [MAGI S6 loop 12, B2b-b1] a group was never split, so a generated constant table was
        // one unit over the batch limit and the whole review was refused.
        for (code, expected) in [
            (
                "A = 1\nB = (\n    2,\n)\n\nC = 3\n",
                vec![(1, 1), (2, 4), (5, 5), (6, 6)],
            ),
            (
                "import a\nfrom b import (\n    c,\n)\n",
                vec![(1, 1), (2, 4)],
            ),
        ] {
            let units = extract(code, &[1]);
            assert_eq!(units.len(), 1, "one group: {code}");
            assert_eq!(units[0].kind, UnitKind::DeclarationGroup);
            let parts = PythonExtractor.split_block(&units[0]);
            let found: Vec<(u32, u32)> = parts.iter().map(|u| (u.start_line, u.end_line)).collect();
            assert_eq!(found, expected, "{code}");
            assert!(parts.iter().all(|u| u.kind == UnitKind::DeclarationGroup));
        }
    }

    /// `(name, kind, start, end)` of each unit.
    fn spans(units: &[SemanticUnit]) -> Vec<(&str, UnitKind, u32, u32)> {
        units
            .iter()
            .map(|u| (u.name.as_str(), u.kind, u.start_line, u.end_line))
            .collect()
    }

    #[test]
    fn a_header_wrapped_over_several_lines_keeps_its_body() {
        // [MAGI S6 loop 4, B2] the `) -> None:` line, unindented, ended the function before its
        // header closed, so the body became a gap.
        let code = "def f(\n    x,\n) -> None:\n    return x\n\ndef g():\n    pass\n";
        let every: Vec<u32> = (1..=7).collect();
        assert_eq!(
            spans(&extract(code, &every)),
            vec![
                ("f", UnitKind::Function, 1, 4),
                ("g", UnitKind::Function, 6, 7),
                ("lines 5-5", UnitKind::TopLevel, 5, 5),
            ]
        );
    }

    #[test]
    fn a_dedented_continuation_line_does_not_end_the_block() {
        // [MAGI S6 loop 4, B2] inside brackets indentation means nothing, so `)` at column 0
        // continues the statement; it neither ends the function nor starts an assignment.
        let code = "def h():\n    x = call(\n1,\n)\n    return x\n";
        assert_eq!(
            spans(&extract(code, &[5])),
            vec![("h", UnitKind::Function, 1, 5)]
        );
        let module = "setup(\nname='x',\n)\n";
        assert!(
            extract(module, &[1, 2, 3])
                .iter()
                .all(|u| u.kind != UnitKind::DeclarationGroup),
            "{:?}",
            spans(&extract(module, &[1, 2, 3]))
        );
    }

    #[test]
    fn a_class_with_a_wrapped_header_splits_at_its_methods() {
        // [MAGI S6 loop 4, B2] the body indentation was read off the first base-class line.
        let code = "class C(\n        Base,\n):\n    def a(self):\n        return 1\n\n    def b(self):\n        return 2\n";
        let every: HashSet<u32> = (1..=8).collect();
        let block = PythonExtractor
            .extract_units(code, "test.py", &every)
            .into_iter()
            .find(|u| u.kind == UnitKind::Class)
            .expect("the class");
        assert_eq!((block.start_line, block.end_line), (1, 8));
        let parts = PythonExtractor.split_block(&block);
        assert!(
            spans(&parts).contains(&("a", UnitKind::Method, 4, 5))
                && spans(&parts).contains(&("b", UnitKind::Method, 7, 8)),
            "{:?}",
            spans(&parts)
        );
    }

    #[test]
    fn a_module_level_statement_with_an_equals_sign_is_not_a_declaration_group() {
        for statement in [
            "assert x == 1",
            "raise E(code=1)",
            "del d[k == 1]",
            "await f(x=1)",
        ] {
            let code = format!(
                "{statement}
"
            );
            let units = extract(&code, &[1]);
            assert!(
                units.iter().all(|u| u.kind != UnitKind::DeclarationGroup),
                "{statement}: {units:?}"
            );
        }
    }

    #[test]
    fn a_control_flow_block_is_not_a_declaration_group() {
        for opener in [
            "if __name__ == \"__main__\":",
            "for i in range(3):",
            "while x == 1:",
            "with open(p) as h:",
            "try:",
        ] {
            let code = format!("{opener}\n    x = 1\n");
            let units = extract(&code, &[1]);
            assert!(
                units.iter().all(|u| u.kind != UnitKind::DeclarationGroup),
                "{opener}: {units:?}"
            );
            assert!(
                units.iter().any(|u| u.start_line <= 1 && u.end_line >= 2),
                "{opener}: the block's body belongs with its line: {units:?}"
            );
        }
    }

    #[test]
    fn a_module_level_call_is_not_an_assignment() {
        // [MAGI S6 loop 5, B2-c1] any `=` or `:` made a line an assignment, so a call with a
        // keyword argument or a string holding `:` became a declaration group with the
        // assignments around it.
        let code = "print(\"a:b\")\nsetup_logging(level=INFO)\nx = 1\n";
        let units = extract(code, &[1, 2, 3]);
        assert_eq!(
            spans(&units),
            vec![
                ("x", UnitKind::DeclarationGroup, 3, 3),
                ("lines 1-2", UnitKind::TopLevel, 1, 2)
            ]
        );
        for line in [
            "x = 1",
            "x: int = 1",
            "x: int",
            "a[i] = 2",
            "obj.attr = f(a=1)",
            "a, b = 1, 2",
            "y += 1",
            "MAP = {\"k\": 1}",
            "handler = lambda e: e",
        ] {
            assert!(is_module_assignment(line), "{line}");
        }
        for line in [
            "print(\"a:b\")",
            "setup_logging(level=INFO)",
            "x == 1",
            "f(x) != g(y)",
            "d[\"k:v\"]",
            "run(a=1)  # x = 2",
        ] {
            assert!(!is_module_assignment(line), "{line}");
        }
    }

    #[test]
    fn a_decorator_wrapped_over_several_lines_belongs_to_its_function() {
        // [MAGI S6 loop 6, B2b-c3] only lines opening with `@` were taken back, so the
        // continuation lines of a wrapped decorator were reviewed apart from the function.
        let code = "x = 1\n\n@route(\n    \"/a\",\n)\n@cache\ndef f():\n    return 1\n";
        let units = extract(code, &[4]);
        assert_eq!(spans(&units), vec![("f", UnitKind::Function, 3, 8)]);
    }

    #[test]
    fn a_backslash_continued_single_quoted_string_stays_with_its_statement() {
        // [MAGI S6 loop 6, B2b-m2] declared green: the bracket reader drops a one-quote string
        // at its line's end, but the scanner keeps it open past an escaped newline, so the
        // continuation lines are marked continued and never read as a definition.
        let code = "x = 'abc\\\ndef g():\\\nend'\ny = 1\n\ndef h():\n    return 1\n";
        let units = extract(code, &[2, 7]);
        assert_eq!(
            spans(&units),
            vec![
                ("x", UnitKind::DeclarationGroup, 1, 4),
                ("h", UnitKind::Function, 6, 7),
            ]
        );
    }

    #[test]
    fn a_wrapped_import_is_one_unit_with_its_header() {
        // [MAGI S6 loop 5, B2-c2] the import group ended on `from foo import (`, so its
        // continuation lines became orphan units reviewed without the header.
        let code = "from foo import (\n    a,\n    b,\n)\nimport os, \\\n    sys\n\nx = 1\n";
        let units = extract(code, &[3, 6]);
        assert_eq!(
            spans(&units),
            vec![("imports", UnitKind::DeclarationGroup, 1, 6)]
        );
    }

    #[test]
    fn assignments_apart_by_more_than_one_blank_line_are_separate_groups() {
        // [MAGI S6 loop 7, B2b-m3] the assignment group went on through any number of blank
        // lines, so a module of constants far apart became one unit that never splits.
        let code = "X = 1\n\n\nY = 2\n\nZ = 3\n";
        let units = extract(code, &[1, 4, 6]);
        assert_eq!(
            spans(&units),
            vec![
                ("X", UnitKind::DeclarationGroup, 1, 1),
                ("Y", UnitKind::DeclarationGroup, 4, 6),
            ]
        );
    }

    #[test]
    fn construct_ranges_are_sorted_and_disjoint_and_a_memberless_class_does_not_split() {
        // [MAGI S6 loop 7, B2a] `ChangedLines::assign` needs sorted, disjoint ranges; and an
        // empty split means "keep the class whole" to its one caller, `review::batching`.
        let code = "import os\n\n\nX = 1\n\n@dec\ndef f(\n    a,\n):\n    return a\n\nclass A:\n    y = 2\n\nclass B:\n    def m(self):\n        pass\n";
        let lines: Vec<&str> = code.lines().collect();
        let constructs = detect_python_constructs(&lines);
        assert_eq!(constructs.len(), 5);
        for pair in constructs.windows(2) {
            assert!(pair[0].start_line <= pair[0].end_line);
            assert!(
                pair[0].end_line < pair[1].start_line,
                "{} overlaps {}",
                pair[0].name,
                pair[1].name
            );
        }
        let all: Vec<u32> = (1..=u32::try_from(lines.len()).expect("small")).collect();
        let class_a = extract(code, &all)
            .into_iter()
            .find(|u| u.name == "A")
            .expect("class A");
        assert!(PythonExtractor.split_block(&class_a).is_empty());
    }

    #[test]
    fn a_method_line_that_does_not_fit_u32_has_no_span() {
        // [MAGI S6 loop 9, B2b-c2] a `u32::MAX` fallback split a class at a fabricated range.
        assert_eq!(member_span(0, 4), Some((1, 5)));
        let max = usize::try_from(u32::MAX).expect("usize holds u32");
        assert_eq!(member_span(0, max), None);
        assert_eq!(member_span(max, max), None);
        assert_eq!(member_span(0, usize::MAX), None);
    }

    #[test]
    fn a_shift_assignment_is_an_assignment_and_a_comparison_is_not() {
        // [MAGI S6 loop 9, B2b-c4] `<<=` and `>>=` are told from `<=` and `>=` by the character
        // two places before the `=`; pin both sides so a refactor of the scan cannot flip them.
        for line in ["x <<= 1", "x >>= 2", "x = 1", "x += 1", "x: int = 1"] {
            assert!(has_top_level_assignment(line), "{line}");
        }
        for line in ["x <= y", "x >= y", "x == y", "x != y", "f(a=1)", "y := 1"] {
            assert!(!has_top_level_assignment(line), "{line}");
        }
    }

    #[test]
    fn any_blank_separates_a_keyword_from_what_follows_it() {
        // [MAGI S6 loop 13, B2b2-b2, B2b2-m2, B2b2-b1] the keywords matched with one literal
        // space, so `def\tf():`, `class\t C:` or `import\tos` (legal Python) were no construct;
        // a class is named by its own identifier, never by its keyword.
        let code = "import\tos\nfrom\tx import y\n\ndef\tf():\n    pass\n\nasync  def g():\n    pass\n\nclass\t C:\n    def\tm(self):\n        pass\n";
        let all: Vec<u32> = (1..=12).collect();
        let units = extract(code, &all);
        let found = spans(&units);
        for expected in [
            ("imports", UnitKind::DeclarationGroup, 1, 2),
            ("f", UnitKind::Function, 4, 5),
            ("g", UnitKind::Function, 7, 8),
            ("C", UnitKind::Class, 10, 12),
        ] {
            assert!(found.contains(&expected), "{expected:?} in {found:?}");
        }
        let class = units
            .iter()
            .find(|u| u.kind == UnitKind::Class)
            .expect("the class");
        let parts = PythonExtractor.split_block(class);
        assert_eq!(
            spans(&parts),
            vec![
                ("C", UnitKind::Class, 10, 10),
                ("m", UnitKind::Method, 11, 12)
            ]
        );
    }

    #[test]
    fn a_class_line_inside_a_decorator_string_is_not_the_class_header() {
        // [MAGI S6 loop 13, B2b2-m1] the header was the first line opening with `class`, even
        // one inside a decorator's triple-quoted argument, so the body's indentation was read
        // from the wrong line and the class never split.
        let code = "@doc(\"\"\"\nclass Fake:\n\"\"\")\nclass Real:\n    def a(self):\n        pass\n    def b(self):\n        pass\n";
        let units = extract(code, &[6]);
        assert_eq!(spans(&units), vec![("Real", UnitKind::Class, 1, 8)]);
        assert_eq!(
            spans(&PythonExtractor.split_block(&units[0])),
            vec![
                ("Real", UnitKind::Class, 1, 4),
                ("a", UnitKind::Method, 5, 6),
                ("b", UnitKind::Method, 7, 8)
            ]
        );
    }

    #[test]
    fn a_blank_or_comment_line_between_a_decorator_and_its_definition_keeps_them_together() {
        // [MAGI S6 loop 13, B2b2-c2] the scan back stopped at the first line that was no
        // decorator, so a decorator above a blank or comment line (legal Python) was reviewed
        // apart from its function. A comment with no decorator above it is still left out.
        let code = "x = 1\n\n@cache\n\n# keeps state\ndef f():\n    return 1\n";
        assert_eq!(
            spans(&extract(code, &[3])),
            vec![("f", UnitKind::Function, 3, 7)]
        );
        let code = "x = 1\n\n# note\ndef g():\n    pass\n";
        assert_eq!(
            spans(&extract(code, &[4])),
            vec![("g", UnitKind::Function, 4, 5)]
        );
        let code = "class A:\n    def a(self):\n        pass\n\n    @property\n    # cached\n    def b(self):\n        return 1\n";
        let units = extract(code, &[5]);
        assert_eq!(
            spans(&PythonExtractor.split_block(&units[0])),
            vec![
                ("A", UnitKind::Class, 1, 1),
                ("a", UnitKind::Method, 2, 3),
                ("A", UnitKind::Class, 4, 4),
                ("b", UnitKind::Method, 5, 8)
            ]
        );
    }

    #[test]
    fn a_backslash_before_a_quote_in_a_raw_string_never_closes_it() {
        // [MAGI S6 loop 13, B2b2-c1] declared green, a false positive: in Python's tokenizer a
        // backslash escapes the next character in a raw string too, as far as where the string
        // ends (`r"\""` is the two characters `\"`; `r"""a\"""` never closes and CPython
        // rejects it), so reading every string alike is Python's own rule. Each line below
        // parses with CPython 3.15 as one assignment, and the definition after them is found.
        let code = r#"x = r"\""
y = r'a\'b'
z = rb"""q\""""
w = r"\\"
v = "a\\"
def f():
    return 1
"#;
        for line in code.lines().take(5) {
            assert!(has_top_level_assignment(line), "{line}");
        }
        let units = extract(code, &[1, 7]);
        assert_eq!(
            spans(&units),
            vec![
                ("x", UnitKind::DeclarationGroup, 1, 5),
                ("f", UnitKind::Function, 6, 7)
            ]
        );
    }

    #[test]
    fn an_escaped_backslash_or_a_comment_ending_in_one_continues_no_line() {
        // [MAGI S6 loop 13, B2b2-b3] declared green, a false positive: only a `\` outside every
        // string and comment continues a line, and Python accepts a `\` there only right before
        // the newline. A string ending in an escaped backslash (`"a\\"`) closes on its quote,
        // and a comment's backslash is comment text, so neither carries the line on.
        let code = r#"v = "a\\"
# dir C:\
def f():
    return 1
w = 'b\\'
def g():
    return 2
"#;
        let units = extract(code, &[1, 4, 5, 7]);
        assert_eq!(
            spans(&units),
            vec![
                ("v", UnitKind::DeclarationGroup, 1, 1),
                ("f", UnitKind::Function, 3, 4),
                ("w", UnitKind::DeclarationGroup, 5, 5),
                ("g", UnitKind::Function, 6, 7)
            ]
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
    fn a_byte_order_mark_on_the_first_line_loses_no_line() {
        // [MAGI S6 loop 14, B2b2] a UTF-8 BOM before `def` hides the keyword (U+FEFF is no
        // whitespace), a declared limitation: the first construct's unit structure degrades,
        // but every line still lands in exactly one unit and the next construct is found.
        let code = "\u{FEFF}def f():\n    return 1\n\ndef g():\n    pass\n";
        let units = extract(code, &[1, 2, 3, 4, 5]);
        assert_each_line_in_one_unit(&units, 5);
        assert!(
            spans(&units).contains(&("g", UnitKind::Function, 4, 5)),
            "{:?}",
            spans(&units)
        );
    }

    #[test]
    fn a_definition_inside_a_module_level_block_falls_into_a_whole_gap_unit() {
        // [MAGI S6 loop 14, B2b2] definitions are found at indent 0 only, a declared
        // limitation: one inside `if TYPE_CHECKING:` is no unit of its own, its block is one gap
        // unit, and that unit is never split, so a block larger than a batch is refused with the
        // typed `ReviewError::TooLargeForBatch` (RV03) instead of being cut; no line is lost.
        let code = "import os\n\nif TYPE_CHECKING:\n    def helper():\n        return 1\n\n    class Shim:\n        pass\n\ndef after():\n    return 2\n";
        let every: Vec<u32> = (1..=11).collect();
        let units = extract(code, &every);
        assert_each_line_in_one_unit(&units, 11);
        let found = spans(&units);
        assert!(
            found.contains(&("after", UnitKind::Function, 10, 11)),
            "{found:?}"
        );
        assert!(
            !units.iter().any(|u| u.name == "helper" || u.name == "Shim"),
            "{found:?}"
        );
        let block = units
            .iter()
            .find(|u| u.start_line <= 4 && 4 <= u.end_line)
            .expect("a unit holds the nested def");
        assert_eq!(block.kind, UnitKind::TopLevel, "{found:?}");
        assert!(PythonExtractor.split_block(block).is_empty());
    }

    /// What follows each f-string line in the PEP 701 tests: two definitions, so a reader that
    /// keeps a string or a bracket open past the line swallows them into the line's unit.
    const FSTRING_TAIL: &str = "\ndef f():\n    return 1\n\ndef g():\n    return 2\n";

    /// Asserts that `head`, one line assigning `x`, followed by [`FSTRING_TAIL`] gives three
    /// units: the assignment on its own line and both definitions.
    fn assert_line_ends_before_the_definitions(head: &str) {
        let code = format!("{head}{FSTRING_TAIL}");
        let units = extract(&code, &[1, 3, 6]);
        assert_eq!(
            spans(&units),
            vec![
                ("x", UnitKind::DeclarationGroup, 1, 1),
                ("f", UnitKind::Function, 2, 3),
                ("g", UnitKind::Function, 5, 6),
            ],
            "{head}"
        );
    }

    #[test]
    fn a_plain_assignment_ends_before_the_definitions() {
        // The control of the PEP 701 tests: the spans every f-string line below must give.
        assert_line_ends_before_the_definitions("x = 1");
    }

    #[test]
    fn an_fstring_field_reusing_its_quote_around_a_bracket_ends_with_its_line() {
        // [MAGI residual, PEP 701] a string inside a replacement field that reuses the outer
        // quote closed the f-string early, so its `(` stayed counted open and the assignment
        // took both definitions. Each line parses with CPython 3.12 (`t"` with 3.14).
        for head in [
            r#"x = f"{"(" if a else ""}""#,
            r#"x = f"{f"{"(" if a else ""}"}""#,
            r#"x = t"{"(" if a else ""}""#,
        ] {
            assert_line_ends_before_the_definitions(head);
        }
    }

    #[test]
    fn a_triple_quote_inside_an_fstring_field_opens_no_string() {
        // [MAGI residual, PEP 701] the `'''` of a string inside a field read as the opening of a
        // triple-quoted string, which ran to the end of the file.
        assert_line_ends_before_the_definitions(r#"x = f"{d["'''"]}""#);
        assert_line_ends_before_the_definitions(r#"x = f"""{d['"""']}""""#);
    }

    #[test]
    fn an_fstring_field_spanning_lines_continues_its_statement() {
        // [MAGI residual, PEP 701] a replacement field may span lines, a comment included; its
        // lines continue the assignment and its `(` is string text.
        for code in [
            "x = f\"{\n    a\n} (\"\ndef f():\n    return 1\n",
            "x = f\"{a # c (\n    + b\n} (\"\ndef f():\n    return 1\n",
        ] {
            let units = extract(code, &[1, 2, 5]);
            assert_each_line_in_one_unit(&units, 5);
            assert_eq!(
                spans(&units),
                vec![
                    ("x", UnitKind::DeclarationGroup, 1, 3),
                    ("f", UnitKind::Function, 4, 5),
                ],
                "{code}"
            );
        }
    }

    #[test]
    fn fstring_formats_escapes_and_prefixes_keep_their_unit_boundaries() {
        // Declared green before the PEP 701 fix: the lines the old reader already got right,
        // pinned so the f-string reader keeps them. A quote in a format spec is text, `{{` is
        // no field, `\N{...}` is an escape except in a raw string, and a plain, byte or `u`
        // string has no field at all.
        for head in [
            r#"x = f"{d["key"]}""#,
            r#"x = f"{", ".join(y)}""#,
            r#"x = f"{v:#x}""#,
            r#"x = f"{v:'^10} (""#,
            r#"x = f"{v:{w}>{n}} (""#,
            r#"x = f"{v!r:>10} (""#,
            r#"x = f"{{ ( }}""#,
            r#"x = f"\N{EM DASH} (""#,
            r#"x = rf"\{v} (""#,
            r#"x = rf"\N{v} (""#,
            r#"x = fr"{v}\" (""#,
            r#"x = Rf"{v} (""#,
            r#"x = fR"{v} (""#,
            r#"x = F"{v} (""#,
            r#"x = T"{v} (""#,
            r#"x = f"{f"{v}"} (""#,
            r#"x = "{" + ")""#,
            r#"x = b"{" + b")""#,
            r#"x = rb"{" + br")""#,
            r#"x = u"{" + ")""#,
        ] {
            assert_line_ends_before_the_definitions(head);
        }
        let code = "x = f\"\"\"a {v} (\n{w:>{n}} \"\"\"\ndef f():\n    return 1\n";
        let units = extract(code, &[1, 4]);
        assert_eq!(
            spans(&units),
            vec![
                ("x", UnitKind::DeclarationGroup, 1, 2),
                ("f", UnitKind::Function, 3, 4),
            ]
        );
    }

    #[test]
    fn any_mix_of_string_prefixes_quotes_and_braces_loses_no_line() {
        // The RV03 invariant under the f-string reader, declared green: whatever the quotes,
        // prefixes, fields, specs, escapes and comments, valid or not, every line lands in
        // exactly one unit and nothing panics. A fixed linear congruential generator keeps the
        // texts deterministic.
        const PIECES: [&str; 16] = [
            "f\"",
            "rf'",
            "t\"\"\"",
            "\"",
            "'''",
            "{",
            "}",
            "{{",
            ":",
            "(",
            "]",
            "\\",
            "#",
            "def f():",
            "\n",
            "\n    x = 1\n",
        ];
        const TEXTS: usize = 400;
        const PIECES_PER_TEXT: usize = 40;
        let mut seed: u64 = 0x5EED;
        for _ in 0..TEXTS {
            let mut code = String::new();
            for _ in 0..PIECES_PER_TEXT {
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let pick = usize::try_from(seed >> 60).unwrap_or(0);
                code.push_str(PIECES.get(pick).copied().unwrap_or(""));
            }
            let total = u32::try_from(code.lines().count()).unwrap_or(u32::MAX);
            let every: Vec<u32> = (1..=total).collect();
            let units = extract(&code, &every);
            assert_each_line_in_one_unit(&units, total);
        }
    }

    #[test]
    fn a_class_nested_in_a_class_body_is_a_member_of_its_own_kind() {
        // [MAGI loop 26, 8] the members split took only `def` lines, so a nested `class Inner:`
        // stayed in the outer class's header piece, against the `split_block` contract (other
        // nested constructs keep their kind), which the brace family honors.
        let code = "class C:\n    def a(self):\n        return 1\n\n    @decorate\n    class Inner:\n        def b(self):\n            return 2\n";
        let every: HashSet<u32> = (1..=8).collect();
        let block = PythonExtractor
            .extract_units(code, "test.py", &every)
            .into_iter()
            .find(|u| u.kind == UnitKind::Class)
            .expect("the class");
        assert_eq!(
            spans(&PythonExtractor.split_block(&block)),
            vec![
                ("C", UnitKind::Class, 1, 1),
                ("a", UnitKind::Method, 2, 3),
                ("C", UnitKind::Class, 4, 4),
                ("Inner", UnitKind::Class, 5, 8),
            ]
        );
        // A function nested in a function stays in its parent (RV03).
        let def = extract("def f():\n    def g():\n        pass\n", &[2]);
        assert!(PythonExtractor.split_block(&def[0]).is_empty());
    }

    #[test]
    fn an_equals_sign_inside_a_triple_quoted_docstring_is_no_assignment() {
        // [MAGI loop 26, 9] the assignment check read a triple-quoted string as one-quote
        // strings, so an apostrophe inside it (`It's`) left `a = b` outside every quote and the
        // docstring merged with the assignment below it under a junk name.
        assert!(!has_top_level_assignment("'''It's a = b, see below.'''"));
        assert!(!has_top_level_assignment("\"\"\"Say \"x\" = 1.\"\"\""));
        assert!(has_top_level_assignment("X = '''It's a = b'''"));
        let code = "'''It's a = b, see below.'''\nX = 1\n";
        assert_eq!(
            spans(&extract(code, &[1, 2])),
            vec![
                ("X", UnitKind::DeclarationGroup, 2, 2),
                ("lines 1-1", UnitKind::TopLevel, 1, 1),
            ]
        );
    }

    #[test]
    fn a_soft_keyword_used_as_a_name_is_an_assignment_and_its_statement_is_not() {
        // [MAGI loop 27, B2b2] `match` and `case` are soft keywords (Python 3.10): they name a
        // variable as legally as any word, so `match = 1` (which CPython parses) was an orphan
        // gap unit instead of a declaration group. A `match` statement is still none.
        for line in [
            "match = 1",
            "case = 2",
            "match: int = 1",
            "match[0] = 1",
            "case += 1",
        ] {
            assert!(is_module_assignment(line), "{line}");
        }
        for line in [
            "match x:",
            "match (a, b):",
            "case _:",
            "match x:  # note = 1",
        ] {
            assert!(!is_module_assignment(line), "{line}");
        }
        let code = "match = 1\ncase = 2\n\ndef f():\n    return match\n";
        assert_eq!(
            spans(&extract(code, &[1, 2, 5])),
            vec![
                ("match", UnitKind::DeclarationGroup, 1, 2),
                ("f", UnitKind::Function, 4, 5),
            ]
        );
    }
}
