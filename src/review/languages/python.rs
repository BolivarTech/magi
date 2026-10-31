// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! Python-specific semantic extractor using indentation-based block detection.
//!
//! Recognizes `def`, `async def`, `class`, `import`/`from`, and module-level
//! assignments. Block boundaries are determined by indentation level changes.

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
    fn split_block(&self, unit: &SemanticUnit) -> Vec<SemanticUnit> {
        // MAGI FIX (S6 loop 3): a class too large for one batch is reviewed method by method.
        if unit.kind != UnitKind::Class {
            return Vec::new();
        }
        let lines: Vec<&str> = unit.content.split('\n').collect();
        let continued = continued_lines(&lines);
        let code = |j: usize| !continued.get(j).copied().unwrap_or(false);
        let Some(class_idx) = lines
            .iter()
            .position(|l| l.trim_start().starts_with("class "))
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
            let is_def = trimmed.starts_with("def ") || trimmed.starts_with("async def ");
            if code(j) && is_def && indentation(line) == body_indent {
                let start = scan_back_decorators(&lines, &continued, j);
                let end = find_block_end(&lines, &continued, j);
                // MAGI FIX (S6 loop 9): a `u32::MAX` fallback split the class at a fabricated
                // range; a line that does not fit `u32` leaves the class whole.
                let Some((first, last)) = member_span(start, end) else {
                    return Vec::new();
                };
                members.push((
                    UnitKind::Function,
                    extract_python_name(trimmed, "def "),
                    first,
                    last,
                ));
                j = end + 1;
            } else {
                j += 1;
            }
        }
        split_into_members(unit, members)
    }
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
    let line = |idx: usize| u32::try_from(idx.checked_add(1)?).ok();
    Some((line(start)?, line(end)?))
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

/// A Python string being read: its quote and whether it is triple-quoted.
#[derive(Clone, Copy)]
struct OpenString {
    quote: char,
    triple: bool,
}

/// What carries a Python logical line over to the next physical line: open brackets, an open
/// triple-quoted string, or a trailing `\`. Strings and comments are read so that their
/// brackets do not count.
///
/// It overlaps [`CodeScanner`] on purpose: the scanner counts braces only and knows nothing of
/// `(` and `[`, which carry a Python line as much as `{` does. [`continued_lines`] takes the
/// union of both, so a string state this reader drops (a one-quote string continued by an
/// escaped newline) is still marked by the scanner's `starts_in_literal`. A change to Python's
/// string syntax must reach both.
#[derive(Default)]
struct LogicalLines {
    depth: u32,
    string: Option<OpenString>,
    backslash: bool,
}

impl LogicalLines {
    /// Whether the next line to read continues the current logical line.
    fn continues(&self) -> bool {
        self.depth > 0 || self.backslash || self.string.is_some_and(|s| s.triple)
    }

    /// Reads one physical line. O(n) in the line.
    fn read(&mut self, line: &str) {
        let chars: Vec<char> = line.chars().collect();
        let at = |k: usize| chars.get(k).copied();
        let mut last_code: Option<char> = None;
        let mut k = 0;
        while let Some(c) = at(k) {
            match self.string {
                Some(open) => {
                    if c == '\\' {
                        k += 2;
                        continue;
                    }
                    let closes = c == open.quote
                        && (!open.triple
                            || (at(k + 1) == Some(open.quote) && at(k + 2) == Some(open.quote)));
                    if closes {
                        self.string = None;
                        k += if open.triple { 3 } else { 1 };
                        continue;
                    }
                }
                None => match c {
                    '#' => break,
                    '"' | '\'' => {
                        let triple = at(k + 1) == Some(c) && at(k + 2) == Some(c);
                        self.string = Some(OpenString { quote: c, triple });
                        last_code = Some(c);
                        k += if triple { 3 } else { 1 };
                        continue;
                    }
                    '(' | '[' | '{' => self.depth = self.depth.saturating_add(1),
                    ')' | ']' | '}' => self.depth = self.depth.saturating_sub(1),
                    _ => {}
                },
            }
            if self.string.is_none() && !c.is_whitespace() {
                last_code = Some(c);
            }
            k += 1;
        }
        // A one-quote string cannot span lines without a `\`; an unclosed one ends here.
        if self.string.is_some_and(|s| !s.triple) {
            self.string = None;
        }
        self.backslash = self.string.is_none() && last_code == Some('\\');
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
fn detect_python_constructs(lines: &[&str]) -> Vec<PythonConstruct> {
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
        let line_num = u32::try_from(i + 1).unwrap_or(u32::MAX);

        // def / async def at indent 0 → Function
        if (trimmed.starts_with("def ") || trimmed.starts_with("async def ")) && indent == 0 {
            let name = extract_python_name(trimmed, "def ");
            let start = scan_back_decorators(lines, &continued, i);
            let end = find_block_end(lines, &continued, i);
            constructs.push(PythonConstruct {
                kind: UnitKind::Function,
                name,
                start_line: u32::try_from(start + 1).unwrap_or(1),
                end_line: u32::try_from(end + 1).unwrap_or(line_num),
            });
            i = end + 1;
            continue;
        }

        // class at indent 0 → Class
        if trimmed.starts_with("class ") && indent == 0 {
            let name = extract_python_name(trimmed, "class ");
            let start = scan_back_decorators(lines, &continued, i);
            let end = find_block_end(lines, &continued, i);
            constructs.push(PythonConstruct {
                kind: UnitKind::Class,
                name,
                start_line: u32::try_from(start + 1).unwrap_or(1),
                end_line: u32::try_from(end + 1).unwrap_or(line_num),
            });
            i = end + 1;
            continue;
        }

        // import / from ... import at indent 0 → DeclarationGroup
        if (trimmed.starts_with("import ") || trimmed.starts_with("from ")) && indent == 0 {
            let group_end = find_import_group_end(lines, &continued, i);
            constructs.push(PythonConstruct {
                kind: UnitKind::DeclarationGroup,
                name: "imports".to_string(),
                start_line: line_num,
                end_line: u32::try_from(group_end + 1).unwrap_or(line_num),
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
                end_line: u32::try_from(group_end + 1).unwrap_or(line_num),
            });
            i = group_end + 1;
            continue;
        }

        i += 1;
    }

    constructs
}

/// Words that open a statement which is never a module-level assignment: definitions, imports,
/// the compound statements and the simple-statement keywords, whose `:`, `=` and `==` would
/// otherwise read as an assignment.
const NOT_ASSIGNMENT_WORDS: &[&str] = &[
    "def", "async", "class", "import", "from", "if", "elif", "else", "for", "while", "with", "try",
    "except", "finally", "match", "case", "assert", "raise", "del", "global", "nonlocal", "return",
    "yield", "await", "pass", "lambda", "not",
];

/// Whether an unindented, trimmed line is a module-level assignment (`X = 1`, `x: int = 1`,
/// `y += 1`): it holds an assignment `=` or an annotation `:` outside every bracket and string
/// ([`has_top_level_assignment`]), is no comment or decorator, and does not open with a word of
/// [`NOT_ASSIGNMENT_WORDS`]. O(n) in the line.
fn is_module_assignment(trimmed: &str) -> bool {
    let first_word = trimmed
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .next()
        .unwrap_or("");
    !trimmed.is_empty()
        && !trimmed.starts_with('#')
        && !trimmed.starts_with('@')
        && !NOT_ASSIGNMENT_WORDS.contains(&first_word)
        && has_top_level_assignment(trimmed)
}

/// Whether `line` holds, outside every bracket, string and comment, an assignment `=` (plain or
/// augmented: `+=`, `<<=`; not the comparisons `==`, `!=`, `<=`, `>=`) or an annotation `:`
/// (not `:=`). A call's keyword argument (`f(level=1)`) and a string's `:` (`print("a:b")`) are
/// inside a bracket or a string and do not count. O(n) in the line.
fn has_top_level_assignment(line: &str) -> bool {
    // MAGI FIX (S6 loop 5): any `=` or `:` made a call (`setup(level=1)`) an assignment.
    let chars: Vec<char> = line.chars().collect();
    let at = |k: usize| chars.get(k).copied();
    let mut depth = 0u32;
    let mut quote: Option<char> = None;
    let mut k = 0;
    while let Some(c) = at(k) {
        if let Some(q) = quote {
            if c == '\\' {
                k += 2;
                continue;
            }
            if c == q {
                quote = None;
            }
            k += 1;
            continue;
        }
        match c {
            '#' => return false,
            '"' | '\'' => quote = Some(c),
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
        if trimmed.starts_with("import ") || trimmed.starts_with("from ") {
            end = j;
        } else if trimmed.is_empty() {
            // Allow one blank line
            if let Some(next) = lines.get(j + 1).map(|l| l.trim()) {
                if next.starts_with("import ") || next.starts_with("from ") {
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
/// Declared: a group is never split, so a run of adjacent assignments too large for one batch
/// is reported as too large rather than cut.
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
/// lines, as `continued` marks them ([`continued_lines`]). O(k) for the k lines taken.
fn scan_back_decorators(lines: &[&str], continued: &[bool], idx: usize) -> usize {
    // MAGI FIX (S6 loop 6): a wrapped decorator's continuation lines were left out.
    let mut start = idx;
    while let Some(mut first) = start.checked_sub(1) {
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

/// Extract name from a Python def/class line.
fn extract_python_name(trimmed: &str, prefix: &str) -> String {
    let rest = if let Some(r) = trimmed.strip_prefix("async def ") {
        r
    } else if let Some(r) = trimmed.strip_prefix(prefix) {
        r
    } else {
        trimmed
    };

    rest.chars()
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
}
