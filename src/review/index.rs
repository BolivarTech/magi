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

/// Every identifier of the reviewed code, plus the files that have a language extractor.
#[allow(dead_code)] // Red stub
pub struct LexicalIndex {
    identifiers: HashSet<String>,
    extractor_files: HashSet<String>,
    complete: bool,
}

impl LexicalIndex {
    /// Builds the index from `(path, text)` pairs. O(total bytes).
    pub fn build(files: &[(String, String)], registry: &LanguageRegistry) -> Self {
        todo!("{} {}", files.len(), registry.get("x.rs").is_some())
    }

    /// Whether `name` (or the last segment of a qualified citation) occurs in the code.
    pub fn contains(&self, name: &str) -> bool {
        todo!("{name}")
    }

    /// Whether `file` has a language extractor.
    pub fn has_extractor(&self, file: &str) -> bool {
        todo!("{file}")
    }

    /// Whether a `.`-joined citation names a file rather than a symbol.
    pub fn is_file_name(&self, citation: &str) -> bool {
        todo!("{citation}")
    }

    /// False when `build` stopped at the identifier cap.
    pub fn is_complete(&self) -> bool {
        todo!()
    }

    #[cfg(test)]
    pub(crate) fn build_with_cap(
        files: &[(String, String)],
        registry: &LanguageRegistry,
        cap: usize,
    ) -> Self {
        todo!("{} {} {cap}", files.len(), registry.get("x.rs").is_some())
    }
}

/// Lines of each file that some completed batch showed to the trio.
#[derive(Default)]
#[allow(dead_code)] // Red stub
pub struct Reviewed {
    ranges: HashMap<String, BTreeMap<u32, u32>>,
}

impl Reviewed {
    /// Records lines `first..=last` of `file`; an empty range records nothing.
    pub fn add_lines(&mut self, file: &str, first: u32, last: u32) {
        todo!("{file} {first} {last}")
    }

    /// Records lines `1..=lines` of `file`.
    pub fn add_whole_file(&mut self, file: &str, lines: u32) {
        todo!("{file} {lines}")
    }

    /// Whether `file` has at least one recorded line.
    pub fn has_file(&self, file: &str) -> bool {
        todo!("{file}")
    }

    /// Whether `line` of `file` was recorded.
    pub fn contains_line(&self, file: &str, line: u32) -> bool {
        todo!("{file} {line}")
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
}
