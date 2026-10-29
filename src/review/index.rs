// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! In-memory lexical index and the record of what a review showed (REQ-RV09, REQ-RV19).

use std::collections::{BTreeMap, HashMap, HashSet};

use super::languages::LanguageRegistry;

/// Distinct identifiers the index holds before it stops growing (memory bound; a starting point,
/// not a measurement). At the cap, roughly 100-150 MB of heap (an owned String plus a hash slot
/// each; an estimate), on top of the content cache.
pub const MAX_INDEX_IDENTIFIERS: usize = 2_000_000;

/// Extensions that name a file even when nothing reviewed has them: a citation such as `conf.toml`
/// is a file name, not the symbol `toml`. A citation whose extension is in none of this list, the
/// registry or the reviewed files (`notes.zzz`) is looked up as the symbol `zzz` and counted by the
/// symbol-discard tally.
const KNOWN_FILE_EXTENSIONS: [&str; 14] = [
    "toml", "yaml", "yml", "json", "md", "txt", "lock", "cfg", "ini", "xml", "html", "css", "sh",
    "env",
];

/// Separator of a qualified citation (`Baz::qux`).
pub(crate) const PATH_SEPARATOR: &str = "::";

/// Last segment of a citation joined by `::` or `.`.
fn last_segment(citation: &str) -> &str {
    let after_path = citation.rsplit(PATH_SEPARATOR).next().unwrap_or(citation);
    after_path.rsplit('.').next().unwrap_or(after_path)
}

/// Extension of a path: the text after its last `.`, if it has one.
fn extension_of(path: &str) -> Option<&str> {
    path.rsplit_once('.').map(|(_, ext)| ext)
}

/// Maximal ASCII identifier runs of `text`; runs that start with a digit are not identifiers.
fn identifier_runs(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .filter(|run| run.chars().next().is_some_and(|c| !c.is_ascii_digit()))
}

/// Every identifier of the reviewed code, plus the files that have a language extractor.
///
/// The corpus is every ASCII identifier of the text given to [`LexicalIndex::build`], comments and
/// strings included (a name invented in a comment passes: declared). It holds at most
/// [`MAX_INDEX_IDENTIFIERS`] distinct identifiers; an index that hit the cap is incomplete and
/// cannot say that a symbol is absent.
pub struct LexicalIndex {
    identifiers: HashSet<String>,
    extractor_files: HashSet<String>,
    file_extensions: HashSet<String>,
    complete: bool,
}

impl LexicalIndex {
    /// Builds the index from `(path, text)` pairs (both sides of a diff may appear under one path).
    ///
    /// Complexity: O(total bytes of `files`) time; O(distinct identifiers), bounded by the cap,
    /// space.
    pub fn build(files: &[(String, String)], registry: &LanguageRegistry) -> Self {
        Self::build_with_limit(files, registry, MAX_INDEX_IDENTIFIERS)
    }

    #[cfg(test)]
    pub(crate) fn build_with_cap(
        files: &[(String, String)],
        registry: &LanguageRegistry,
        cap: usize,
    ) -> Self {
        Self::build_with_limit(files, registry, cap)
    }

    fn build_with_limit(
        files: &[(String, String)],
        registry: &LanguageRegistry,
        cap: usize,
    ) -> Self {
        let mut index = Self {
            identifiers: HashSet::new(),
            extractor_files: HashSet::new(),
            file_extensions: registry
                .extensions()
                .chain(KNOWN_FILE_EXTENSIONS)
                .map(str::to_string)
                .collect(),
            complete: true,
        };
        for (path, _) in files {
            if let Some(ext) = extension_of(path) {
                index.file_extensions.insert(ext.to_string());
            }
            if registry.get(path).is_some() {
                index.extractor_files.insert(path.clone());
            }
        }
        'files: for (_, text) in files {
            for run in identifier_runs(text) {
                if index.identifiers.contains(run) {
                    continue;
                }
                if index.identifiers.len() >= cap {
                    index.complete = false;
                    break 'files;
                }
                index.identifiers.insert(run.to_string());
            }
        }
        index
    }

    /// Whether `name` occurs in the reviewed code; a qualified citation (`Baz::qux`,
    /// `self.helper`) is looked up by its last segment. O(1) expected.
    pub fn contains(&self, name: &str) -> bool {
        self.identifiers.contains(last_segment(name))
    }

    /// Whether `file` was given to `build` and has a language extractor. O(1) expected.
    pub fn has_extractor(&self, file: &str) -> bool {
        self.extractor_files.contains(file)
    }

    /// Whether a `.`-joined citation names a file (`consts.rs`, `conf.toml`) rather than a symbol:
    /// its last segment is the extension of a file given to `build`, of the registry or of
    /// `KNOWN_FILE_EXTENSIONS`. A citation joined by `::` is a symbol path, never a file name.
    /// Cost declared: a field access ending in an extension (`self.c`) is not checked either.
    pub fn is_file_name(&self, citation: &str) -> bool {
        !citation.contains(PATH_SEPARATOR)
            && extension_of(citation).is_some_and(|ext| self.file_extensions.contains(ext))
    }

    /// False when `build` stopped at the identifier cap: the symbol check is off for the review.
    pub fn is_complete(&self) -> bool {
        self.complete
    }
}

/// Lines of each file that some completed batch showed to the trio.
///
/// Per file the ranges are kept merged in a `BTreeMap<start, end>`: inserting is O(log r)
/// amortized and a line lookup is O(log r), with r the ranges of that file.
#[derive(Default)]
pub struct Reviewed {
    ranges: HashMap<String, BTreeMap<u32, u32>>,
}

impl Reviewed {
    /// Records lines `first..=last` of `file`; an empty range (`first > last`, the empty side of a
    /// hunk) records nothing and does not register the file.
    pub fn add_lines(&mut self, file: &str, first: u32, last: u32) {
        if first > last {
            return;
        }
        let map = self.ranges.entry(file.to_string()).or_default();
        let (mut start, mut end) = (first, last);
        if let Some((&s, &e)) = map.range(..=start).next_back() {
            if e.saturating_add(1) >= start {
                start = s;
                end = end.max(e);
                map.remove(&s);
            }
        }
        while let Some((&s, &e)) = map.range(start..).next() {
            if s > end.saturating_add(1) {
                break;
            }
            end = end.max(e);
            map.remove(&s);
        }
        map.insert(start, end);
    }

    /// Records lines `1..=lines` of `file`; zero lines record nothing.
    pub fn add_whole_file(&mut self, file: &str, lines: u32) {
        self.add_lines(file, 1, lines);
    }

    /// Whether `file` has at least one recorded line. O(1) expected.
    pub(crate) fn has_file(&self, file: &str) -> bool {
        self.ranges.contains_key(file)
    }

    /// Whether `line` of `file` was recorded. O(log r).
    pub(crate) fn contains_line(&self, file: &str, line: u32) -> bool {
        self.ranges
            .get(file)
            .and_then(|map| map.range(..=line).next_back())
            .is_some_and(|(_, &end)| end >= line)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idx(text: &str) -> LexicalIndex {
        LexicalIndex::build(
            &[("a.rs".to_string(), text.to_string())],
            &LanguageRegistry::new(),
        )
    }

    #[test]
    fn identifiers_are_maximal_ascii_runs_and_digit_runs_are_not_identifiers() {
        let i = idx("let x_1 = 0x1F + foo2(bar);\n");
        assert!(i.contains("x_1") && i.contains("foo2") && i.contains("bar") && i.contains("let"));
        assert!(!i.contains("x1F") && !i.contains("1F") && !i.contains("foo"));
    }

    #[test]
    fn a_qualified_name_is_looked_up_by_its_last_segment() {
        let i = idx("fn helper() {}\n");
        assert!(i.contains("self.helper") && i.contains("Ghost::helper"));
        assert!(!i.contains("self.ghost"));
    }

    #[test]
    fn only_files_with_an_extractor_are_marked_and_extensions_of_the_registry_name_files() {
        let files = vec![
            ("a.rs".to_string(), "fn a() {}".to_string()),
            ("n.unknownlang".to_string(), "x".to_string()),
        ];
        let i = LexicalIndex::build(&files, &LanguageRegistry::new());
        assert!(i.has_extractor("a.rs") && !i.has_extractor("n.unknownlang"));
        assert!(i.is_file_name("mod.rs") && i.is_file_name("x.unknownlang"));
        assert!(!i.is_file_name("plain") && !i.is_file_name("a.b.zzz"));
    }

    #[test]
    fn a_field_access_on_a_receiver_is_a_symbol_and_a_file_or_path_is_a_file() {
        // [MAGI S6 loop 3, B2] `self.c` ends in the C extension but is the field `c`, looked up
        // by its last segment; `main.c` and `src/a.rs` name files.
        let i = idx("fn a() { let b = 1; }\n");
        for field in [
            "self.c",
            "this.h",
            "Self.rs",
            "super.go",
            "cls.py",
            "self.inner.c",
        ] {
            assert!(!i.is_file_name(field), "{field}");
        }
        assert!(
            !i.contains("self.c"),
            "`c` is absent from the reviewed code"
        );
        assert!(i.contains("self.b"), "`b` is present");
        for file in ["main.c", "src/a.rs", "src\\self.c", "mod.rs", "app.test.ts"] {
            assert!(i.is_file_name(file), "{file}");
        }
        let reviewed = LexicalIndex::build(
            &[("lib/this.ts".to_string(), "x".to_string())],
            &LanguageRegistry::new(),
        );
        assert!(
            reviewed.is_file_name("this.ts"),
            "a reviewed file's name stays a file name"
        );
    }

    #[test]
    fn overlapping_and_adjacent_ranges_merge_and_gaps_stay() {
        let mut r = Reviewed::default();
        r.add_lines("a.rs", 10, 20);
        r.add_lines("a.rs", 21, 30);
        r.add_lines("a.rs", 5, 12);
        r.add_lines("a.rs", 40, 50);
        r.add_lines("a.rs", 25, 45);
        assert!(r.contains_line("a.rs", 5) && r.contains_line("a.rs", 50));
        assert!(!r.contains_line("a.rs", 4) && !r.contains_line("a.rs", 51));
        r.add_lines("b.rs", 1, 3);
        r.add_lines("b.rs", 7, 9);
        assert!(!r.contains_line("b.rs", 5) && r.has_file("b.rs") && !r.has_file("c.rs"));
    }

    #[test]
    fn a_whole_file_covers_line_one_to_its_last_and_zero_lines_registers_nothing() {
        let mut r = Reviewed::default();
        r.add_whole_file("a.rs", 3);
        r.add_whole_file("empty.rs", 0);
        assert!(r.contains_line("a.rs", 1) && r.contains_line("a.rs", 3));
        assert!(!r.contains_line("a.rs", 4) && !r.has_file("empty.rs"));
    }

    #[test]
    fn a_line_near_u32_max_does_not_overflow_the_merge() {
        let mut r = Reviewed::default();
        r.add_lines("a.rs", u32::MAX - 1, u32::MAX);
        r.add_lines("a.rs", 1, 2);
        assert!(r.contains_line("a.rs", u32::MAX) && r.contains_line("a.rs", 2));
    }

    #[test]
    fn a_range_that_ends_at_u32_max_merges_with_a_later_one_without_overflow() {
        let mut r = Reviewed::default();
        r.add_lines("a.rs", 5, u32::MAX);
        r.add_lines("a.rs", 10, 20);
        r.add_lines("a.rs", u32::MAX, u32::MAX);
        assert!(r.contains_line("a.rs", 5) && r.contains_line("a.rs", u32::MAX));
        assert!(!r.contains_line("a.rs", 4));
    }
}
