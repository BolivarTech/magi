// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! In-memory lexical index and the record of what a review showed (REQ-RV09, REQ-RV19).

use std::collections::{BTreeMap, HashMap, HashSet};

use super::languages::{
    ChangedLines, ExtractorKind, LanguageExtractor, LanguageRegistry, UnitKind,
};

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

/// Receivers of a field or method access (`self.c`, `this.h`): a citation whose first `.`-segment
/// is one of them is a member, never a file name, even when its last segment is an extension.
const FIELD_ACCESS_RECEIVERS: [&str; 5] = ["self", "Self", "this", "super", "cls"];

/// Separator of a qualified citation (`Baz::qux`).
pub(crate) const PATH_SEPARATOR: &str = "::";

/// Last segment of a citation joined by `::` or `.`.
fn last_segment(citation: &str) -> &str {
    let after_path = citation.rsplit(PATH_SEPARATOR).next().unwrap_or(citation);
    after_path.rsplit('.').next().unwrap_or(after_path)
}

/// Last component of a path, after its last `/` or `\`.
fn base_name(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

/// Extension of a path: the text after the last `.` of its file name, if it has one, in ASCII
/// lowercase. A dot in a directory (`src.d/foo`) is not an extension.
fn extension_of(path: &str) -> Option<String> {
    // MAGI FIX (S6 loop 5): `FOO.ZZZ` and `notes.zzz` share an extension; both sides compare
    // lowercased, as `detect_language` does.
    // MAGI FIX (S6 loop 18, B1-5): the file name is isolated first, as in
    // `context::file_extension`. Unlike that function, a dot-led name keeps its extension
    // (`.toml` gives `toml`) on purpose: a citation that starts with a dot is never a symbol,
    // so reading it as a file name skips a check that could only discard the finding.
    base_name(path)
        .rsplit_once('.')
        .map(|(_, ext)| ext.to_ascii_lowercase())
}

/// Maximal ASCII identifier runs of `text`; runs that start with a digit are not identifiers.
///
/// ASCII on purpose, paired with the citation rule of REQ-RV09
/// (`crate::review::validate::cited_identifiers`), which only takes ASCII identifiers: a
/// citation of a non-ASCII name (`größe`) is never checked, so it can never be discarded as an
/// unknown symbol, and an ASCII citation is looked up among ASCII runs. A Unicode index would
/// only matter with Unicode citations, which would add a discard path that a different
/// normalization of the same name (NFC in the finding, NFD in the code) turns into the loss of
/// a real finding. The `citations_the_validator_checks_are_whole_index_identifiers` test pins
/// the pairing.
pub(crate) fn identifier_runs(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .filter(|run| run.chars().next().is_some_and(|c| !c.is_ascii_digit()))
}

/// Every identifier of the reviewed code, plus the files that have a language extractor.
///
/// The corpus is every ASCII identifier of the text given to [`LexicalIndex::build`], comments and
/// strings included (a name invented in a comment passes: declared). Indexing code only would
/// need a lexer per language and would turn a name the scanner misreads into a discarded real
/// finding; a wider corpus can only keep a finding (the fail-open direction of REQ-RV09), at the
/// cost of a laxer `unknown symbol` count (PM-S-2). It holds at most
/// [`MAX_INDEX_IDENTIFIERS`] distinct identifiers; an index that hit the cap is incomplete and
/// cannot say that a symbol is absent.
pub struct LexicalIndex {
    identifiers: HashSet<String>,
    extractor_files: HashSet<String>,
    file_extensions: HashSet<String>,
    /// Last path component of every file given to `build` (`lib/this.ts` gives `this.ts`).
    file_names: HashSet<String>,
    /// Name of each named top-level unit, with `(file, first_line, last_line)`.
    definitions: HashMap<String, Vec<(String, u32, u32)>>,
    /// `(file, line)` of each line an identifier appears on. Holds the identifiers of
    /// `identifiers` only, so it grows with the occurrences of at most
    /// [`MAX_INDEX_IDENTIFIERS`] names.
    occurrences: HashMap<String, Vec<(String, u32)>>,
    complete: bool,
}

impl LexicalIndex {
    /// Builds the index from `(path, text)` pairs (both sides of a diff may appear under one path).
    ///
    /// Also records where each identifier occurs and where each named top-level unit is defined
    /// (the extractors run once over every file that has one).
    ///
    /// Complexity: O(total bytes of `files`) time; O(distinct identifiers, bounded by the cap,
    /// plus occurrences) space.
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
                .map(str::to_ascii_lowercase)
                .collect(),
            file_names: HashSet::new(),
            definitions: HashMap::new(),
            occurrences: HashMap::new(),
            complete: true,
        };
        for (path, _) in files {
            if let Some(ext) = extension_of(path) {
                index.file_extensions.insert(ext);
            }
            index.file_names.insert(base_name(path).to_string());
            if registry.get(path).is_some() {
                index.extractor_files.insert(path.clone());
            }
        }
        'files: for (path, text) in files {
            for (idx, line_text) in text.lines().enumerate() {
                let line = u32::try_from(idx.saturating_add(1)).unwrap_or(u32::MAX);
                for run in identifier_runs(line_text) {
                    if !index.identifiers.contains(run) {
                        if index.identifiers.len() >= cap {
                            index.complete = false;
                            break 'files;
                        }
                        index.identifiers.insert(run.to_string());
                    }
                    index.record_occurrence(run, path, line);
                }
            }
        }
        for (path, text) in files {
            if let Some(extractor) = registry.get(path) {
                index.record_definitions(extractor, path, text);
            }
        }
        index
    }

    /// Records that `name` occurs at `line` of `file`; a second occurrence on the same line is
    /// not recorded again. O(1) expected.
    fn record_occurrence(&mut self, name: &str, file: &str, line: u32) {
        let entry = (file.to_string(), line);
        match self.occurrences.get_mut(name) {
            Some(places) => {
                if places.last() != Some(&entry) {
                    places.push(entry);
                }
            }
            None => {
                self.occurrences.insert(name.to_string(), vec![entry]);
            }
        }
    }

    /// Records the named top-level units of `text` as definitions: a function, a method, a type
    /// (class, struct, enum, trait) or a module. An impl block is not a definition of its type, and
    /// the gap and declaration-group units an extractor emits carry a label, not a name.
    /// O(bytes of `text`) for the extraction.
    fn record_definitions(&mut self, extractor: &ExtractorKind, file: &str, text: &str) {
        for unit in extractor.extract_units(text, file, ChangedLines::All) {
            if !matches!(
                unit.kind,
                UnitKind::Function
                    | UnitKind::Method
                    | UnitKind::Class
                    | UnitKind::Struct
                    | UnitKind::Enum
                    | UnitKind::Trait
                    | UnitKind::Module
            ) {
                continue;
            }
            let place = (file.to_string(), unit.start_line, unit.end_line);
            let places = self.definitions.entry(unit.name).or_default();
            // Both sides of a diff may be given under one path: the same span is one definition.
            if !places.contains(&place) {
                places.push(place);
            }
        }
    }

    /// Whether `name` occurs in the reviewed code; a qualified citation (`Baz::qux`,
    /// `self.helper`) is looked up by its last segment. O(1) expected.
    pub fn contains(&self, name: &str) -> bool {
        self.identifiers.contains(last_segment(name))
    }

    /// Where `name` is defined, as `(file, first_line, last_line)`, in the order the files were
    /// given. Only named top-level units of files with an extractor are definitions; a name that
    /// is defined more than once (homonyms) lists every place. O(1) expected.
    ///
    /// # Arguments
    ///
    /// * `name` - The exact name of the unit.
    ///
    /// # Returns
    ///
    /// The places, empty when `name` has no definition.
    pub fn definitions(&self, name: &str) -> &[(String, u32, u32)] {
        self.definitions.get(name).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Where the identifier `name` occurs, as `(file, line)`: one entry per line, in the order the
    /// files were given and then line order. Comments and strings count, as in
    /// [`Self::contains`]. Occurrences of an identifier that arrived after the index stopped at
    /// the cap ([`Self::is_complete`]) are missing. O(1) expected.
    ///
    /// # Arguments
    ///
    /// * `name` - The identifier.
    ///
    /// # Returns
    ///
    /// The places, empty when `name` does not occur.
    pub fn occurrences(&self, name: &str) -> &[(String, u32)] {
        self.occurrences.get(name).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Whether `file` was given to `build` and has a language extractor. O(1) expected.
    pub fn has_extractor(&self, file: &str) -> bool {
        self.extractor_files.contains(file)
    }

    /// Whether a citation names a file (`consts.rs`, `conf.toml`, `src/a.rs`) rather than a symbol.
    ///
    /// A citation joined by `::` is a symbol path, never a file name. A citation whose last path
    /// component is the name of a file given to `build` is a file name, with or without an
    /// extension (`Makefile`, `docker/Dockerfile` when that file was reviewed). Otherwise it names
    /// a file when its extension (the text after the last `.`) is the extension of a file given to
    /// `build`, of the registry or of `KNOWN_FILE_EXTENSIONS`, and it is plausibly a file:
    ///
    /// * it has the form of a path (holds `/` or `\`), or
    /// * its first `.`-segment is not a receiver of `FIELD_ACCESS_RECEIVERS` (`self`, `Self`,
    ///   `this`, `super`, `cls`).
    ///
    /// So `self.c` and `this.inner.h` are field accesses, looked up as the symbols `c` and `h` by
    /// [`Self::contains`], while `main.c` stays a file name. Cost declared: any other
    /// `<name>.<extension>` (`node.ts`) is ambiguous and reads as a file, which skips its symbol
    /// check and keeps the finding (the fail-open direction of REQ-RV09).
    ///
    /// The name of a reviewed file wins over the receiver rule: with `self.rs` reviewed, the
    /// citation `self.rs` reads as that file, which skips one symbol check and keeps the finding
    /// (fail-open, as above). An extensionless path that names no reviewed file (`src/foo`) is
    /// not a file name; `cited_identifiers` never yields a `/` or `\`, so the path clauses only
    /// serve direct callers.
    ///
    /// # Arguments
    ///
    /// * `citation` - The cited text, as `cited_identifiers` returns it, or a path.
    ///
    /// # Returns
    ///
    /// `true` when the citation is read as a file name. O(len of `citation`).
    pub fn is_file_name(&self, citation: &str) -> bool {
        if citation.contains(PATH_SEPARATOR) {
            return false;
        }
        // MAGI FIX (S6 loop 16): a reviewed name without an extension (`Makefile`,
        // `docker/Dockerfile`) is a file; the extension guard below rejected it first.
        if self.file_names.contains(base_name(citation)) {
            return true;
        }
        // MAGI FIX (S6 loop 3): a field access on a receiver (`self.c`) is a symbol, not a file.
        if !extension_of(citation).is_some_and(|ext| self.file_extensions.contains(&ext)) {
            return false;
        }
        let is_path = citation.contains(['/', '\\']);
        let receiver = citation.split('.').next().unwrap_or(citation);
        // MAGI FIX (S6 loop 17, B1-2): the name of a file given to `build` already returned
        // above, so a second lookup of it here could never be the deciding clause.
        is_path || !FIELD_ACCESS_RECEIVERS.contains(&receiver)
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
    fn definitions_are_recorded_by_name_with_their_lines() {
        let files = vec![(
            "a.rs".to_string(),
            "fn parse() {}\n\nfn only() {}\n".to_string(),
        )];
        let index = LexicalIndex::build(&files, &LanguageRegistry::new());
        assert_eq!(index.definitions("parse"), &[("a.rs".to_string(), 1, 1)]);
        assert!(index.definitions("missing").is_empty());
        // [RED-DRYRUN 2026-10-18] measured: the extractor also yields a gap unit "lines 2-3"; it is
        // not a definition.
        assert!(index.definitions("lines 2-3").is_empty());
    }

    #[test]
    fn occurrences_are_recorded_with_their_line() {
        let files = vec![(
            "b.rs".to_string(),
            "fn use_it() {\n    parse();\n}\n".to_string(),
        )];
        let index = LexicalIndex::build(&files, &LanguageRegistry::new());
        assert_eq!(index.occurrences("parse"), &[("b.rs".to_string(), 2)]);
        assert!(index.occurrences("missing").is_empty());
    }

    #[test]
    fn identifiers_are_maximal_ascii_runs_and_digit_runs_are_not_identifiers() {
        let i = idx("let x_1 = 0x1F + foo2(bar);\n");
        assert!(i.contains("x_1") && i.contains("foo2") && i.contains("bar") && i.contains("let"));
        assert!(!i.contains("x1F") && !i.contains("1F") && !i.contains("foo"));
    }

    #[test]
    fn citations_the_validator_checks_are_whole_index_identifiers() {
        // [MAGI S6 loop 4, B1] a finding naming a real non-ASCII identifier must never be
        // discarded as an unknown symbol: whatever the validator cites from such a finding is
        // in the index built from the code that holds it.
        use crate::review::validate::cited_identifiers;
        let code = "fn größe() {}\nlet café_total = 1;\nstruct 変数;\nfn x\u{301}y() {}\n";
        let index = idx(code);
        let finding =
            "`größe` overflows; `café_total` and `変数` too; see `x\u{301}y` and `Self::größe`";
        for cited in cited_identifiers(finding) {
            assert!(index.contains(&cited), "{cited} would be discarded");
        }
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
    fn an_extensionless_reviewed_file_name_is_a_file_name_bare_or_with_a_directory() {
        // [MAGI S6 loop 16, B1-1] the extension guard ran before the reviewed names, so a
        // reviewed `Makefile` or `docker/Dockerfile`, cited in a finding, was checked as a
        // symbol and RV09 could discard the finding as naming an unknown one.
        let i = LexicalIndex::build(
            &[
                ("Makefile".to_string(), "all:\n\tcc a.c\n".to_string()),
                ("docker/Dockerfile".to_string(), "FROM x\n".to_string()),
            ],
            &LanguageRegistry::new(),
        );
        for file in [
            "Makefile",
            "Dockerfile",
            "dir/Makefile",
            "docker/Dockerfile",
            "docker\\Dockerfile",
        ] {
            assert!(i.is_file_name(file), "{file}");
        }
        assert!(
            !i.is_file_name("Rakefile") && !i.is_file_name("Makefile::build"),
            "an unreviewed extensionless name and a symbol path stay symbols"
        );
    }

    #[test]
    fn file_extensions_are_matched_without_regard_to_ascii_case() {
        // [MAGI S6 loop 5, B1-c4] a reviewed file's extension was kept as written and a
        // citation's compared as written, so `notes.zzz` was no file name after `FOO.ZZZ` was
        // reviewed, nor `MAIN.RS` beside the registry's `rs`, and the citation was checked as a
        // symbol.
        let i = LexicalIndex::build(
            &[("FOO.ZZZ".to_string(), "x".to_string())],
            &LanguageRegistry::new(),
        );
        for file in ["notes.zzz", "notes.ZZZ", "MAIN.RS", "src/Lib.Rs"] {
            assert!(i.is_file_name(file), "{file}");
        }
    }

    #[test]
    fn a_dot_in_a_directory_is_not_read_as_the_file_extension() {
        // [MAGI S6 loop 18, B1-5] the extension was taken from the whole path, so a dotted
        // directory over an extensionless file (`src.d/foo`) gave the extension `d/foo`, unlike
        // `context::file_extension`, which isolates the file name first.
        for path in ["src.d/foo", "src/v1.2/main", "conf.d\\settings"] {
            assert_eq!(extension_of(path), None, "{path}");
        }
        assert_eq!(extension_of("pkg.d/Lib.RS").as_deref(), Some("rs"));
        assert_eq!(extension_of("notes.zzz").as_deref(), Some("zzz"));
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
