// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! Language-aware semantic extraction for source code.
//!
//! This module provides extractors that recognize syntactic boundaries
//! (functions, classes, structs, declaration groups) in source code
//! and return the units that contain changed lines. Used by the
//! review pipeline to send complete code units to the trio.
//!
//! # Architecture
//!
//! - [`LanguageExtractor`] trait defines the extraction interface.
//! - [`ExtractorKind`] enum provides zero-cost dispatch to concrete extractors.
//! - [`LanguageRegistry`] maps file extensions to extractors.

pub mod brace_family;
pub mod python;
pub mod scanner;

use std::collections::{HashMap, HashSet};

use crate::review::context::FileContext;

/// Kind of semantic code unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitKind {
    /// A standalone function (Rust `fn`, C function, Python `def`, JS function).
    Function,
    /// A method inside a class or impl block.
    Method,
    /// A class definition (Python `class`, Java/C#/JS class).
    Class,
    /// A struct definition (Rust, C, Go).
    Struct,
    /// An enum definition.
    Enum,
    /// A trait or interface definition.
    Trait,
    /// An impl block (Rust-specific).
    Impl,
    /// A group of related top-level declarations: constants, statics,
    /// type aliases, imports, macros, global variables.
    DeclarationGroup,
    /// Fallback: changed lines not inside any recognized construct.
    /// Context is lines around the change.
    TopLevel,
}

/// A semantic unit of code extracted from a source file.
#[derive(Debug, Clone)]
pub struct SemanticUnit {
    /// Kind of code block.
    pub kind: UnitKind,
    /// Name of the construct (function name, struct name, or
    /// descriptive label like "imports" or "constants").
    pub name: String,
    /// Source file path.
    pub file: String,
    /// Start line (1-indexed, inclusive).
    pub start_line: u32,
    /// End line (1-indexed, inclusive).
    pub end_line: u32,
    /// Complete source code of the unit (including doc comments
    /// and attributes above the declaration).
    pub content: String,
    /// Lines within this unit that were modified in the PR.
    pub changed_lines: Vec<u32>,
    /// File context metadata for this unit's file.
    ///
    /// The extractors leave it at [`FileContext::default`] in v0.25.0 and no stage reads it: a
    /// batch names its file's language in its [`file_header`](crate::review::batching::file_header),
    /// computed from the path. It is kept for the retrieval stage that will fill it.
    pub context: FileContext,
}

/// Language-specific code extractor.
///
/// Implementors recognize syntactic boundaries (functions, classes,
/// declaration groups) in source code and return the units that
/// contain changed lines.
///
/// Concrete types: [`brace_family::BraceFamilyExtractor`],
/// [`python::PythonExtractor`].
/// Dispatched via [`ExtractorKind`] enum.
pub trait LanguageExtractor {
    /// Language identifier (e.g., "rust", "c", "python").
    fn language_id(&self) -> &str;
    /// File extensions this extractor handles (without dot).
    fn extensions(&self) -> &[&str];
    /// Extract semantic units from full file content, filtering
    /// to only those that overlap with `changed_lines`.
    ///
    /// Each unit's `changed_lines` is sorted. With [`ChangedLines::Only`] the changed lines are
    /// sorted once and each construct takes its share by binary search, O(c log c + n) for c
    /// changed lines; with [`ChangedLines::All`] (a file reviewed whole) no set of every line is
    /// built and each construct's share is its own range, O(n). Both give the same units for
    /// the same lines.
    ///
    /// # Arguments
    ///
    /// * `content` - Full file content.
    /// * `file_path` - Path to the file, copied into each unit.
    /// * `changed_lines` - The changed lines: a `&HashSet<u32>` converts to
    ///   [`ChangedLines::Only`].
    ///
    /// # Returns
    ///
    /// The units of the constructs that hold a changed line, then one fallback unit per gap
    /// between constructs that holds one; empty when no line changed.
    fn extract_units<'a>(
        &self,
        content: &str,
        file_path: &str,
        changed_lines: impl Into<ChangedLines<'a>>,
    ) -> Vec<SemanticUnit>;

    /// Splits a block unit (an `impl`, a class, a trait or a namespace) one level down: its
    /// methods become [`UnitKind::Method`] units, other nested constructs keep their kind, and
    /// the lines between them (the block's header, the blank lines and its closing line) become
    /// pieces with the block's kind and name, so the block's header is never lost.
    ///
    /// # Arguments
    ///
    /// * `unit` - A unit this extractor returned from [`Self::extract_units`].
    ///
    /// # Returns
    ///
    /// The pieces in line order, covering every line of `unit` exactly once, each with the
    /// changed lines of `unit` it holds; empty when `unit` is not a block or holds no member.
    fn split_block(&self, unit: &SemanticUnit) -> Vec<SemanticUnit>;
}

/// A construct found inside a block: `(kind, name, first, last)`, lines 1-based and inclusive,
/// counted from the block's first line.
pub(crate) type Member = (UnitKind, String, u32, u32);

/// The pieces of `block` given its `members`: each member as a unit ([`UnitKind::Function`]
/// becomes [`UnitKind::Method`]), and each stretch between members (the header, blank lines,
/// the closing line) as a piece with the block's kind and name. Shared by the extractors'
/// [`LanguageExtractor::split_block`].
///
/// # Arguments
///
/// * `block` - The block unit.
/// * `members` - Its members in block lines, sorted by start and disjoint.
///
/// # Returns
///
/// The pieces sorted by start line, covering every line of `block` once, each with the changed
/// lines of `block` it holds; empty when `members` is. O(n log n) for the block's n lines.
///
/// A piece may hold no changed line (a stretch between two members of a block whose changes
/// are elsewhere): coverage of every line is the contract, so the block's header is never lost.
/// Its one production caller splits a file reviewed whole, where every line is changed and no
/// piece is left without one; batching packs a piece by its content, never by its changed
/// lines.
pub(crate) fn split_into_members(block: &SemanticUnit, members: Vec<Member>) -> Vec<SemanticUnit> {
    if members.is_empty() {
        return Vec::new();
    }
    // `content` is the block's lines joined by '\n': splitting on it gives them back exactly.
    let lines: Vec<&str> = block.content.split('\n').collect();
    let total = u32::try_from(lines.len()).unwrap_or(u32::MAX);
    let base = block.start_line.saturating_sub(1);
    let piece = |kind: UnitKind, name: String, first: u32, last: u32| SemanticUnit {
        kind,
        name,
        file: block.file.clone(),
        start_line: first.saturating_add(base),
        end_line: last.saturating_add(base),
        content: lines
            .get(first.saturating_sub(1) as usize..(last as usize).min(lines.len()))
            .unwrap_or(&[])
            .join("\n"),
        changed_lines: lines_in_range(
            &block.changed_lines,
            first.saturating_add(base),
            last.saturating_add(base),
        )
        .to_vec(),
        context: block.context.clone(),
    };
    let ranges: Vec<(u32, u32)> = members.iter().map(|m| (m.2, m.3)).collect();
    let (_, orphans) = ChangedLines::All.assign(&ranges, total);
    let gaps = gap_units(&lines, &block.file, &ranges, orphans);
    let mut parts: Vec<SemanticUnit> = members
        .into_iter()
        .map(|(kind, name, first, last)| {
            let kind = if kind == UnitKind::Function {
                UnitKind::Method
            } else {
                kind
            };
            piece(kind, name, first, last)
        })
        .chain(
            gaps.into_iter()
                .map(|g| piece(block.kind, block.name.clone(), g.start_line, g.end_line)),
        )
        .collect();
    parts.sort_by_key(|u| u.start_line);
    parts
}

/// The lines [`LanguageExtractor::extract_units`] treats as changed.
///
/// # Examples
///
/// ```
/// use magi_rs::review::languages::{ChangedLines, LanguageExtractor, LanguageRegistry};
///
/// let registry = LanguageRegistry::new();
/// let rust = registry.get("a.rs").expect("registered");
/// let units = rust.extract_units("fn f() {\n    x();\n}\n", "a.rs", ChangedLines::All);
/// assert_eq!(units.len(), 1);
/// ```
#[derive(Debug, Clone, Copy)]
pub enum ChangedLines<'a> {
    /// Every line of the file (a file reviewed whole), without materializing them.
    All,
    /// Only these lines (1-based).
    Only(&'a HashSet<u32>),
}

impl<'a> From<&'a HashSet<u32>> for ChangedLines<'a> {
    fn from(lines: &'a HashSet<u32>) -> Self {
        Self::Only(lines)
    }
}

impl ChangedLines<'_> {
    /// Whether no line is changed. [`ChangedLines::All`] is never empty, even for an empty file
    /// (which has no line to give out).
    pub fn is_empty(&self) -> bool {
        match self {
            Self::All => false,
            Self::Only(lines) => lines.is_empty(),
        }
    }

    /// Hands the changed lines out to the constructs and the gaps between them.
    ///
    /// # Arguments
    ///
    /// * `ranges` - `(start, end)` line ranges (1-based, inclusive) of the constructs, sorted by
    ///   start and disjoint, as the extractors detect them.
    /// * `total` - Line count of the file: [`ChangedLines::All`] means `1..=total`.
    ///
    /// # Returns
    ///
    /// The sorted changed lines of each range, in the order of `ranges`, and the changed lines
    /// outside every range (the orphans, sorted). O(c log c + c log r) for
    /// [`ChangedLines::Only`] with c lines and r ranges, O(total) for [`ChangedLines::All`].
    pub(crate) fn assign(&self, ranges: &[(u32, u32)], total: u32) -> (Vec<Vec<u32>>, Vec<u32>) {
        match self {
            Self::Only(set) => {
                let mut sorted: Vec<u32> = set.iter().copied().collect();
                sorted.sort_unstable();
                let shares = ranges
                    .iter()
                    .map(|&(start, end)| lines_in_range(&sorted, start, end).to_vec())
                    .collect();
                let orphans = sorted
                    .into_iter()
                    .filter(|&line| {
                        let next = ranges.partition_point(|&(start, _)| start <= line);
                        next.checked_sub(1)
                            .and_then(|i| ranges.get(i))
                            .is_none_or(|&(_, end)| end < line)
                    })
                    .collect();
                (shares, orphans)
            }
            Self::All => {
                let shares = ranges
                    .iter()
                    .map(|&(start, end)| (start..=end.min(total)).collect())
                    .collect();
                let mut orphans = Vec::new();
                let mut next = 1u32;
                for &(start, end) in ranges {
                    orphans.extend(next..start.min(total.saturating_add(1)));
                    next = next.max(end.saturating_add(1));
                }
                orphans.extend(next..=total);
                (shares, orphans)
            }
        }
    }
}

/// Enum dispatch for language extractors.
///
/// Each variant wraps a concrete extractor. The enum delegates
/// [`LanguageExtractor`] methods to the inner type via match arms.
/// This avoids `Box<dyn>` and is trivially [`Clone`]-able.
#[derive(Clone)]
pub enum ExtractorKind {
    /// Brace-delimited languages (Rust, C, C++, Go, Java, JS, TS).
    BraceFamily(brace_family::BraceFamilyExtractor),
    /// Indentation-delimited (Python).
    Python(python::PythonExtractor),
}

impl LanguageExtractor for ExtractorKind {
    fn language_id(&self) -> &str {
        match self {
            Self::BraceFamily(e) => e.language_id(),
            Self::Python(e) => e.language_id(),
        }
    }

    fn extensions(&self) -> &[&str] {
        match self {
            Self::BraceFamily(e) => e.extensions(),
            Self::Python(e) => e.extensions(),
        }
    }

    fn extract_units<'a>(
        &self,
        content: &str,
        file_path: &str,
        changed_lines: impl Into<ChangedLines<'a>>,
    ) -> Vec<SemanticUnit> {
        let changed_lines = changed_lines.into();
        match self {
            Self::BraceFamily(e) => e.extract_units(content, file_path, changed_lines),
            Self::Python(e) => e.extract_units(content, file_path, changed_lines),
        }
    }

    fn split_block(&self, unit: &SemanticUnit) -> Vec<SemanticUnit> {
        match self {
            Self::BraceFamily(e) => e.split_block(unit),
            Self::Python(e) => e.split_block(unit),
        }
    }
}

/// Registry of language extractors, keyed by file extension.
///
/// # Examples
///
/// ```
/// use magi_rs::review::languages::LanguageRegistry;
///
/// let registry = LanguageRegistry::new();
/// assert!(registry.get("src/main.rs").is_some());
/// assert!(registry.get("unknown.xyz").is_none());
/// ```
pub struct LanguageRegistry {
    extractors: HashMap<String, ExtractorKind>,
}

impl LanguageRegistry {
    /// Create a registry with all built-in extractors.
    pub fn new() -> Self {
        let mut registry = Self {
            extractors: HashMap::new(),
        };

        // Register brace-family extractors
        for extractor in brace_family::all_extractors() {
            for ext in extractor.extensions() {
                registry.extractors.insert(
                    ext.to_string(),
                    ExtractorKind::BraceFamily(extractor.clone()),
                );
            }
        }

        // Register Python
        let py = python::PythonExtractor;
        for ext in py.extensions() {
            registry
                .extractors
                .insert(ext.to_string(), ExtractorKind::Python(py));
        }

        registry
    }

    /// File extensions (without the dot) that have a registered extractor.
    pub fn extensions(&self) -> impl Iterator<Item = &str> {
        self.extractors.keys().map(String::as_str)
    }

    /// Get the extractor for a file path, or `None` for unknown extensions. The extension is
    /// matched ignoring its ASCII case (`A.RS` is Rust), as [`detect_language`] does.
    ///
    /// [`detect_language`]: crate::review::context::detect_language
    pub fn get(&self, file_path: &str) -> Option<&ExtractorKind> {
        // MAGI FIX (S6 loop 4): an upper-case extension lost its extractor.
        let ext = file_path.rsplit('.').next()?;
        self.extractors.get(&ext.to_ascii_lowercase())
    }
}

impl Default for LanguageRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// The changed lines of `sorted` that fall in `start..=end`.
///
/// Extractors call it once per construct with the changed lines sorted once, so a whole file
/// (every line changed) costs O(c log n + n) for c constructs and n lines instead of O(c × n).
///
/// # Arguments
///
/// * `sorted` - Changed lines, sorted ascending and without repeats.
/// * `start` - First line of the range (1-based, inclusive).
/// * `end` - Last line of the range (1-based, inclusive).
///
/// # Returns
///
/// The sub-slice of `sorted` inside the range, empty when none is.
pub(crate) fn lines_in_range(sorted: &[u32], start: u32, end: u32) -> &[u32] {
    let lo = sorted.partition_point(|&l| l < start);
    let hi = sorted.partition_point(|&l| l <= end).max(lo);
    sorted.get(lo..hi).unwrap_or(&[])
}

/// The fallback units of the changed lines no construct holds: one [`UnitKind::TopLevel`] unit
/// per gap between constructs that holds orphans, spanning that whole gap and nothing else.
///
/// A gap runs from the line after the previous construct (or line 1) to the line before the
/// next one (or the last line), so a fallback unit never holds a construct and a file split by
/// units keeps each unit as small as its source allows.
///
/// # Arguments
///
/// * `lines` - The file's lines.
/// * `file_path` - Path to the file.
/// * `constructs` - `(start, end)` line ranges (1-based, inclusive) of the file's constructs,
///   sorted by start and disjoint.
/// * `orphans` - Changed lines outside every construct.
///
/// # Returns
///
/// The units, in line order. O(o log o + o log c) for o orphans and c constructs.
pub(crate) fn gap_units(
    lines: &[&str],
    file_path: &str,
    constructs: &[(u32, u32)],
    mut orphans: Vec<u32>,
) -> Vec<SemanticUnit> {
    orphans.sort_unstable();
    orphans.dedup();
    let total = u32::try_from(lines.len()).unwrap_or(u32::MAX);
    let mut units: Vec<SemanticUnit> = Vec::new();
    for orphan in orphans {
        let next = constructs.partition_point(|&(start, _)| start <= orphan);
        let first = next
            .checked_sub(1)
            .and_then(|i| constructs.get(i))
            .map_or(1, |&(_, end)| end.saturating_add(1));
        let last = constructs
            .get(next)
            .map_or(total, |&(start, _)| start.saturating_sub(1));
        if let Some(unit) = units
            .last_mut()
            .filter(|u| u.start_line == first && u.end_line == last)
        {
            unit.changed_lines.push(orphan);
            continue;
        }
        let content = lines
            .get(first.saturating_sub(1) as usize..(last as usize).min(lines.len()))
            .unwrap_or(&[])
            .join("\n");
        units.push(SemanticUnit {
            kind: UnitKind::TopLevel,
            name: format!("lines {first}-{last}"),
            file: file_path.to_string(),
            start_line: first,
            end_line: last,
            content,
            changed_lines: vec![orphan],
            context: FileContext::default(),
        });
    }
    units
}

/// Lines of context [`fallback_extract`] keeps on each side of a changed line.
const FALLBACK_CONTEXT_LINES: u32 = 20;

/// Create fallback semantic units for changed lines in unknown file types.
///
/// Each changed line gets a window of ±`FALLBACK_CONTEXT_LINES` (20) lines, clamped to the file;
/// windows that overlap or touch merge, so each `TopLevel` unit holds its changes and their
/// context, never the unchanged stretch between two distant changes.
///
/// # Arguments
///
/// * `content` - Full file content.
/// * `file_path` - Path to the file.
/// * `changed_lines` - Set of changed line numbers.
/// * `context` - File context metadata.
///
/// # Returns
///
/// The units in line order, each with its changed lines sorted; empty when nothing changed.
/// O(c log c + n) for c changed lines and n lines of the file.
pub fn fallback_extract(
    content: &str,
    file_path: &str,
    changed_lines: &HashSet<u32>,
    context: &FileContext,
) -> Vec<SemanticUnit> {
    let lines: Vec<&str> = content.lines().collect();
    let total = u32::try_from(lines.len()).unwrap_or(u32::MAX);
    let mut sorted: Vec<u32> = changed_lines.iter().copied().collect();
    sorted.sort_unstable();

    // MAGI FIX (S6 loop 3): one window per change, merged when they overlap or touch, instead of
    // one span from the first change to the last.
    let mut windows: Vec<(u32, u32, Vec<u32>)> = Vec::new();
    for line in sorted {
        let start = line.saturating_sub(FALLBACK_CONTEXT_LINES).max(1);
        let end = line.saturating_add(FALLBACK_CONTEXT_LINES).min(total);
        match windows.last_mut() {
            Some((_, last_end, changed)) if last_end.saturating_add(1) >= start => {
                *last_end = (*last_end).max(end);
                changed.push(line);
            }
            _ => windows.push((start, end, vec![line])),
        }
    }

    windows
        .into_iter()
        .map(|(start, end, changed)| SemanticUnit {
            kind: UnitKind::TopLevel,
            name: format!("lines {start}-{end}"),
            file: file_path.to_string(),
            start_line: start,
            end_line: end,
            content: lines
                .get(start.saturating_sub(1) as usize..(end as usize).min(lines.len()))
                .unwrap_or(&[])
                .join("\n"),
            changed_lines: changed,
            context: context.clone(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_in_range_returns_exactly_the_changed_lines_inside_the_range() {
        let sorted = [2, 4, 5, 9, 12];
        assert_eq!(lines_in_range(&sorted, 4, 9), &[4, 5, 9]);
        assert_eq!(lines_in_range(&sorted, 1, 2), &[2]);
        assert_eq!(lines_in_range(&sorted, 6, 8), &[] as &[u32]);
        assert_eq!(lines_in_range(&sorted, 13, 20), &[] as &[u32]);
        assert_eq!(lines_in_range(&[], 1, 5), &[] as &[u32]);
        assert_eq!(
            lines_in_range(&sorted, 9, 4),
            &[] as &[u32],
            "an empty range holds nothing"
        );
    }

    #[test]
    fn fallback_extract_gives_one_window_per_change_and_merges_overlapping_ones() {
        // [MAGI S6 loop 3, B6] one span from the first change to the last held everything
        // between two distant changes.
        let content: String = (1..=200).map(|n| format!("line {n}\n")).collect();
        let ctx = FileContext::default();
        let spans = |changed: &[u32]| -> Vec<(u32, u32, Vec<u32>)> {
            let set: HashSet<u32> = changed.iter().copied().collect();
            fallback_extract(&content, "a.txt", &set, &ctx)
                .into_iter()
                .map(|u| (u.start_line, u.end_line, u.changed_lines))
                .collect()
        };
        assert_eq!(
            spans(&[5, 100, 199]),
            vec![
                (1, 25, vec![5]),
                (80, 120, vec![100]),
                (179, 200, vec![199])
            ]
        );
        assert_eq!(spans(&[30, 50, 71]), vec![(10, 91, vec![30, 50, 71])]);
        assert!(spans(&[]).is_empty());
    }

    #[test]
    fn every_line_of_a_whole_file_lands_in_one_unit_and_so_does_every_split_part() {
        // [MAGI S6 loop 4, B2] the extractors are heuristics; whatever they misread, a file
        // reviewed by units must send each of its lines exactly once, unit split or not. The
        // samples hold the constructs the loop-4 review named.
        let samples = [
            ("a.rs", "pub(super) async unsafe fn f() {\n    x();\n}\nunsafe impl Send for S {\n    fn a() {\n        if x {\n        } else {\n        }\n    }\n}\n"),
            ("a.c", "int counter = 0;\nint f(void)\n// keeps state;\n{\n    return 0;\n}\nstatic int g;\n"),
            ("a.cpp", "namespace n {\n    void f() {\n        try {\n        } catch (const E& e) {\n        }\n    }\n}\n"),
            ("a.java", "public final class C {\n    private static final int X = 1;\n    @Override\n    public String toString() {\n        return \"}\";\n    }\n    void run() {\n    }\n}\n"),
            ("a.js", "const r = /}/;\nclass K {\n  constructor(a) {\n    this.a = a.replace(/\\{/g, '');\n  }\n  get x() {\n    return `${ `n${1}` }`;\n  }\n}\ndescribe('x', () => {\n  it('y', () => {});\n});\n"),
            ("a.ts", "export abstract class A {\n  abstract run(): void;\n  async go() {\n    await x();\n  }\n}\n"),
            ("a.py", "def f(\n    x,\n) -> None:\n    y = call(\n1,\n)\n    return y\n\nclass C(\n        Base,\n):\n    def a(self):\n        return 1\n"),
            ("a.go", "package p\n\nfunc f() {\n\tif x {\n\t} else {\n\t}\n}\n"),
        ];
        let registry = LanguageRegistry::new();
        for (path, code) in samples {
            let ext = registry.get(path).expect("a registered extension");
            let total = u32::try_from(code.lines().count()).expect("small");
            let units = ext.extract_units(code, path, ChangedLines::All);
            let parts: Vec<SemanticUnit> = units
                .iter()
                .flat_map(|u| {
                    let split = ext.split_block(u);
                    if split.is_empty() {
                        vec![u.clone()]
                    } else {
                        split
                    }
                })
                .collect();
            for set in [&units, &parts] {
                for line in 1..=total {
                    let holders = set
                        .iter()
                        .filter(|u| u.start_line <= line && line <= u.end_line)
                        .count();
                    assert_eq!(holders, 1, "{path}: line {line} in {holders} units");
                }
            }
        }
    }

    #[test]
    fn all_lines_gives_the_same_units_as_the_set_of_every_line() {
        // [MAGI S6 loop 3, B9] `ChangedLines::All` replaces a set of every line for a file
        // reviewed whole; the units must not change.
        let samples = [
            ("a.rs", "use a::b;\n\n/// doc\npub fn plan(\n    a: u8,\n) -> u8 {\n    a\n}\n// gap\nimpl X {\n    fn y() {}\n}\nconst C: u8 = 1;\n"),
            ("a.c", "#include <x.h>\n\nint f(int a)\n{\n    return a;\n}\nstatic int g;\nextern \"C\" {\n    int h(void) {\n        return 0;\n    }\n}\n"),
            ("a.py", "import os\n\nX = 1\n\n@d\ndef f():\n    s = '''\ndef g():\n'''\n    return s\n\nclass C:\n    pass\n# tail\n"),
            ("a.java", "package p;\n\npublic class Foo {\n    public void run() {\n    }\n}\n"),
            ("a.js", "const a = 1;\nfunction f() {\n  return `${a}`;\n}\n\nclass K {}\n"),
            ("a.rs", ""),
            ("a.rs", "x"),
        ];
        let registry = LanguageRegistry::new();
        let view =
            |units: Vec<SemanticUnit>| -> Vec<(UnitKind, String, u32, u32, String, Vec<u32>)> {
                units
                    .into_iter()
                    .map(|u| {
                        (
                            u.kind,
                            u.name,
                            u.start_line,
                            u.end_line,
                            u.content,
                            u.changed_lines,
                        )
                    })
                    .collect()
            };
        for (path, text) in samples {
            let ex = registry.get(path).expect("registered");
            let total = u32::try_from(text.lines().count()).expect("small");
            let every: HashSet<u32> = (1..=total).collect();
            let from_set = view(ex.extract_units(text, path, &every));
            let from_all = view(ex.extract_units(text, path, ChangedLines::All));
            assert_eq!(from_all, from_set, "{path}: {text:?}");
        }
    }

    // --- Test 37: registry_dispatches_by_extension ---
    #[test]
    fn registry_dispatches_by_extension() {
        let registry = LanguageRegistry::new();
        let ext = registry.get("src/main.rs");
        assert!(ext.is_some(), ".rs should be registered");
        assert!(
            matches!(ext.unwrap(), ExtractorKind::BraceFamily(_)),
            ".rs should use BraceFamily extractor"
        );
    }

    // --- Test 38: registry_returns_none_for_unknown ---
    #[test]
    fn registry_returns_none_for_unknown() {
        let registry = LanguageRegistry::new();
        assert!(
            registry.get("data.xyz").is_none(),
            "Unknown extension should return None"
        );
    }

    #[test]
    fn no_extractor_panics_on_any_prefix_of_hostile_source() {
        // Release is panic=abort: hostile input must never abort an extractor.
        let samples = [
            "fn a() { let s = \"}\"; '\\'' /* { */ }\n#[x] fn b(",
            "def f(x):\n    '''\n  @d\nclass C:\n\tpass\"\"\"",
            "/* open\n} } { 'a' \"\\\" `t ${x}` r#\"x\"# \u{e9}\u{1F600}",
        ];
        let registry = LanguageRegistry::new();
        let every: HashSet<u32> = (1..=20).collect();
        for path in [
            "a.rs", "a.py", "a.c", "a.cpp", "a.js", "a.ts", "a.go", "a.java",
        ] {
            let ex = registry.get(path).expect("registered");
            for s in samples {
                for end in s.char_indices().map(|(i, _)| i).chain([s.len()]) {
                    for lines in [&every, &HashSet::new()] {
                        let _ = ex.extract_units(&s[..end], path, lines);
                    }
                }
            }
        }
    }

    #[test]
    fn the_default_registry_has_the_ported_extractors_and_none_for_config_files() {
        // `.rs` has an extractor and `rust` as its language; config and unknown files have
        // none (config_files is not ported).
        let registry = LanguageRegistry::new();
        assert!(registry.get("src/a.rs").is_some());
        for path in [
            "Cargo.toml",
            "notes.txt",
            "x.unknownlang",
            "a.yaml",
            "README.md",
        ] {
            assert!(registry.get(path).is_none(), "{path}");
        }
        assert_eq!(crate::review::context::detect_language("src/a.rs"), "rust");
    }

    #[test]
    fn a_file_name_without_an_extension_has_no_extractor() {
        // [MAGI S6 loop 6, B2a-m2] the text after the last dot was taken even when the name had
        // no dot, so a file named `rs` or `.rs` was scanned as Rust.
        let registry = LanguageRegistry::new();
        for path in [
            "rs",
            "src/py",
            ".rs",
            "src/.js",
            "dir.rs/file",
            "dir.rs\\file",
        ] {
            assert!(registry.get(path).is_none(), "{path}");
        }
        for path in ["a.rs", "src/b.test.ts", "C:\\x\\y.go"] {
            assert!(registry.get(path).is_some(), "{path}");
        }
    }

    #[test]
    fn an_upper_case_extension_finds_its_extractor() {
        // [MAGI S6 loop 4, B1] a whole file named `A.RS` too large for one batch had no
        // extractor to split it, so the review failed as too large.
        let registry = LanguageRegistry::new();
        for path in ["src/A.RS", "m.Py", "x.JAVA", "y.Ts"] {
            assert!(registry.get(path).is_some(), "{path}");
        }
        assert!(registry.get("Makefile").is_none());
    }

    #[test]
    fn every_extractor_reports_its_extensions_through_the_trait() {
        let registry = LanguageRegistry::new();
        for ext in registry.extensions() {
            let ex = registry.get(&format!("a.{ext}")).expect("registered");
            assert!(
                ex.extensions().contains(&ext),
                "{ext}: {:?}",
                ex.extensions()
            );
        }
    }
}
