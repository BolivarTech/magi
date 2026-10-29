// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! Python-specific semantic extractor using indentation-based block detection.
//!
//! Recognizes `def`, `async def`, `class`, `import`/`from`, and module-level
//! assignments. Block boundaries are determined by indentation level changes.

use super::scanner::{CodeScanner, LexicalRules};
use super::{gap_units, ChangedLines, LanguageExtractor, SemanticUnit, UnitKind};
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
    let in_literal: Vec<bool> = CodeScanner::new(LexicalRules::python())
        .scan_all(&lines.join(
            "
",
        ))
        .iter()
        .map(|s| s.starts_in_literal)
        .collect();
    let mut constructs = Vec::new();
    let mut i = 0;

    while let Some(&line) = lines.get(i) {
        // MAGI FIX (S6 loop 3): a line that starts inside a triple-quoted string is string text,
        // never a definition, import or assignment.
        if in_literal.get(i).copied().unwrap_or(false) {
            i += 1;
            continue;
        }
        let trimmed = line.trim();
        let indent = indentation(line);
        let line_num = u32::try_from(i + 1).unwrap_or(u32::MAX);

        // def / async def at indent 0 → Function
        if (trimmed.starts_with("def ") || trimmed.starts_with("async def ")) && indent == 0 {
            let name = extract_python_name(trimmed, "def ");
            let start = scan_back_decorators(lines, i);
            let end = find_block_end(lines, &in_literal, i);
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
            let start = scan_back_decorators(lines, i);
            let end = find_block_end(lines, &in_literal, i);
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
            let group_end = find_import_group_end(lines, i);
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
            let group_end = find_assignment_group_end(lines, &in_literal, i);
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

/// Whether an unindented, trimmed line is a module-level assignment (`X = 1`, `x: int = 1`):
/// it holds `=` or `:`, is no comment or decorator, and does not open with a word of
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
        && (trimmed.contains('=') || trimmed.contains(':'))
}

/// Get indentation level (number of leading spaces/tabs).
fn indentation(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

/// Find the end of a Python indented block: the last line before the first code line indented
/// no deeper than the definition.
///
/// As in Python's own tokenizer, a comment-only line and a line that starts inside a
/// triple-quoted string never end the block (`in_literal[j]` is the scanner's
/// [`super::scanner::ScannedLine::starts_in_literal`]). A string line extends the block; an
/// unindented comment does not, so a comment between two definitions stays outside both.
/// O(n) in the lines after `def_line`.
fn find_block_end(lines: &[&str], in_literal: &[bool], def_line: usize) -> usize {
    let def_indent = lines.get(def_line).map_or(0, |line| indentation(line));
    let mut end = def_line;
    for (j, line) in lines.iter().enumerate().skip(def_line + 1) {
        if in_literal.get(j).copied().unwrap_or(false) {
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
fn find_import_group_end(lines: &[&str], start: usize) -> usize {
    let mut end = start;
    for (j, line) in lines.iter().enumerate().skip(start + 1) {
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

/// Find end of a contiguous assignment group at indent 0.
///
/// A line that starts inside a triple-quoted string (`in_literal[j]`) belongs to the assignment
/// that opened the string, so it extends the group. O(n) in the lines after `start`.
fn find_assignment_group_end(lines: &[&str], in_literal: &[bool], start: usize) -> usize {
    let mut end = start;
    for (j, line) in lines.iter().enumerate().skip(start + 1) {
        if in_literal.get(j).copied().unwrap_or(false) {
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
            continue;
        } else {
            break;
        }
    }
    end
}

/// Scan backward for Python decorators (`@...`).
fn scan_back_decorators(lines: &[&str], idx: usize) -> usize {
    let mut start = idx;
    while start > 0 {
        let is_decorator = lines
            .get(start - 1)
            .is_some_and(|prev| prev.trim().starts_with('@'));
        if is_decorator {
            start -= 1;
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
}
