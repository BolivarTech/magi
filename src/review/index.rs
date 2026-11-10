// Author: Julian Bolivar
// Version: 0.25.0
// Date: 2026-11-08

//! In-memory lexical index and the record of what a review showed (REQ-RV09, REQ-RV19).

use std::collections::{BTreeMap, HashMap, HashSet};

use super::languages::{
    ChangedLines, ExtractorKind, LanguageExtractor, LanguageRegistry, UnitKind,
};

/// Distinct identifiers the index holds before it stops growing (memory bound; a starting point,
/// not a measurement). Each name is stored once (interned): its `String` (24 bytes plus its
/// bytes, rounded up by the allocator), a `u32` id and a hash slot, plus the 24-byte header of
/// its place list, about 70 bytes plus the name. At the cap, with 16-byte names, about 170 MB
/// of heap (computed from that layout, not measured), plus at most 64 MB of places
/// ([`MAX_INDEX_OCCURRENCES`]), on top of the content cache. The definitions map is not under
/// this cap: one entry per named top-level unit (the name, a copy of its path and two lines,
/// about 60 bytes plus both strings), so it grows with the reviewed units and is bounded by the
/// reviewed bytes the content cache admits (`MAX_REVIEW_TOTAL_BYTES`), not by a count of its own.
// MAGI FIX (MS3 §6 loop5): each occurring name was stored twice (set and place-map key), and the
// estimate left the second copy, the list headers and the map slots out.
pub const MAX_INDEX_IDENTIFIERS: usize = 2_000_000;

/// Places (`(file, line)` of an identifier) the index records before it stops recording more:
/// 8 bytes each, so about 64 MB at the cap (an estimate, a starting point). Without it the places
/// grew with the reviewed lines, up to the review's 256 MiB total, since one name repeated on
/// every line of a generated file adds one place per line. Past it some callers are missing,
/// declared ([`LexicalIndex::callers_complete`]).
pub const MAX_INDEX_OCCURRENCES: usize = 8_000_000;

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
    /// Each distinct identifier, once, with its id: the index of its places in
    /// [`Self::occurrences`]. Interned so a name is stored once, not again as a key of its places.
    identifiers: HashMap<String, u32>,
    extractor_files: HashSet<String>,
    file_extensions: HashSet<String>,
    /// Last path component of every file given to `build` (`lib/this.ts` gives `this.ts`).
    file_names: HashSet<String>,
    /// Name of each named top-level unit, with `(file, first_line, last_line)`.
    definitions: HashMap<String, Vec<(String, u32, u32)>>,
    /// Paths of the new-side files given to `build`; an occurrence names its file by position.
    paths: Vec<String>,
    /// By identifier id, `(file id, line)` of each new-side line it appears on, 8 bytes each: at
    /// most one entry per name per line, and at most [`MAX_INDEX_OCCURRENCES`] in all.
    occurrences: Vec<Vec<(u32, u32)>>,
    complete: bool,
    /// Places recorded so far, against `places_cap`.
    places: usize,
    /// [`MAX_INDEX_OCCURRENCES`], or a smaller cap in tests.
    places_cap: usize,
    /// False once a place was dropped at `places_cap`.
    callers_complete: bool,
}

impl LexicalIndex {
    /// Builds the index from `(path, text)` pairs, the new side of every file; the old side of a
    /// diff goes to [`Self::build_with_old`], never under the same path here.
    ///
    /// Also records where each identifier occurs and where each named top-level unit is defined
    /// (the extractors run once over every file that has one).
    ///
    /// Complexity: O(total bytes of `files`) time; O(distinct identifiers, bounded by the cap,
    /// plus occurrences) space.
    pub fn build(files: &[(String, String)], registry: &LanguageRegistry) -> Self {
        Self::build_with_limit(
            files,
            &[],
            registry,
            MAX_INDEX_IDENTIFIERS,
            MAX_INDEX_OCCURRENCES,
        )
    }

    /// Like [`Self::build`], for a diff: `files` is the new side of every reviewed file and
    /// `old_files` the old side of the diffed ones.
    ///
    /// Definitions and occurrences are line-keyed, so they come from `files` only; the old side
    /// adds its identifiers (and nothing else), so [`Self::contains`] answers for both sides, as
    /// the validator needs. Same complexity as [`Self::build`].
    pub fn build_with_old(
        files: &[(String, String)],
        old_files: &[(String, String)],
        registry: &LanguageRegistry,
    ) -> Self {
        Self::build_with_limit(
            files,
            old_files,
            registry,
            MAX_INDEX_IDENTIFIERS,
            MAX_INDEX_OCCURRENCES,
        )
    }

    #[cfg(test)]
    pub(crate) fn build_with_cap(
        files: &[(String, String)],
        registry: &LanguageRegistry,
        cap: usize,
    ) -> Self {
        Self::build_with_limit(files, &[], registry, cap, MAX_INDEX_OCCURRENCES)
    }

    fn build_with_limit(
        files: &[(String, String)],
        old_files: &[(String, String)],
        registry: &LanguageRegistry,
        cap: usize,
        places_cap: usize,
    ) -> Self {
        let mut index = Self {
            identifiers: HashMap::new(),
            extractor_files: HashSet::new(),
            file_extensions: registry
                .extensions()
                .chain(KNOWN_FILE_EXTENSIONS)
                .map(str::to_ascii_lowercase)
                .collect(),
            file_names: HashSet::new(),
            definitions: HashMap::new(),
            paths: Vec::new(),
            occurrences: Vec::new(),
            complete: true,
            places: 0,
            places_cap,
            callers_complete: true,
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
        index.paths = files.iter().map(|(path, _)| path.clone()).collect();
        // New side first, with occurrences; then the old side, identifiers only (the validator
        // accepts a name from either side, but line numbers of the old side mean nothing here).
        // MAGI FIX (MS3 §6 loop7): a path given twice is one file: only its first copy records
        // places, so a line is never listed (or counted against the cap) twice; a later copy adds
        // its identifiers only (`a_symbol_only_on_the_old_side_of_a_diff_is_known`).
        let mut first_copy: HashSet<&str> = HashSet::new();
        let scanned = files
            .iter()
            .enumerate()
            .map(|(id, (path, text))| (first_copy.insert(path.as_str()).then_some(id), text))
            .chain(old_files.iter().map(|(_, text)| (None, text)));
        for (id, text) in scanned {
            if !index.scan(text, id, cap) {
                break;
            }
        }
        // A path given twice is one file: its definitions are recorded once.
        let mut defined: HashSet<&str> = HashSet::new();
        for (path, text) in files {
            if let Some(extractor) = registry.get(path).filter(|_| defined.insert(path)) {
                index.record_definitions(extractor, path, text);
            }
        }
        index
    }

    /// Adds the identifiers of `text` and, for a new-side file (`file` is its id), where each
    /// occurs. Returns false when the identifier cap stopped the scan. O(bytes of `text`).
    fn scan(&mut self, text: &str, file: Option<usize>, cap: usize) -> bool {
        let id = file.map(|f| u32::try_from(f).unwrap_or(u32::MAX));
        for (idx, line_text) in text.lines().enumerate() {
            let line = u32::try_from(idx.saturating_add(1)).unwrap_or(u32::MAX);
            for run in identifier_runs(line_text) {
                let name = match self.identifiers.get(run) {
                    Some(&name) => name,
                    None => {
                        if self.identifiers.len() >= cap {
                            self.complete = false;
                            // MAGI FIX (MS3 §6 loop5): the files after this point record no
                            // occurrence either, so callers may be missing too.
                            // MAGI FIX (MS3 §6 loop6): only on the new side, which records
                            // occurrences; the old side runs last and records none.
                            if id.is_some() {
                                self.callers_complete = false;
                            }
                            return false;
                        }
                        let name = u32::try_from(self.occurrences.len()).unwrap_or(u32::MAX);
                        self.identifiers.insert(run.to_string(), name);
                        self.occurrences.push(Vec::new());
                        name
                    }
                };
                if let Some(id) = id {
                    self.record_occurrence(name, id, line);
                }
            }
        }
        true
    }

    /// Records that the identifier with id `name` occurs at `line` of the file with id `file`; a
    /// second occurrence on the same line is not recorded again. Allocates only when a place is
    /// really added. O(1) amortized.
    // MAGI FIX (MS3 §6 loop4): the places grew with the reviewed lines, without a bound.
    // MAGI FIX (MS3 §6 loop5): keyed by the interned id; the name was stored a second time.
    fn record_occurrence(&mut self, name: u32, file: u32, line: u32) {
        let entry = (file, line);
        let Some(places) = self.occurrences.get_mut(name as usize) else {
            return;
        };
        if places.last() == Some(&entry) {
            return;
        }
        if self.places >= self.places_cap {
            self.callers_complete = false;
            return;
        }
        self.places = self.places.saturating_add(1);
        places.push(entry);
    }

    /// Records the named units of `text` as definitions: a function, a method, a type (class,
    /// struct, enum, trait) or a module, at the top level and one level down a block (an `impl`,
    /// class, trait or module, cut by [`LanguageExtractor::split_block`] as the containing-unit
    /// lookup does), so a method is a definition (`methods_are_retrieved_as_defs`). An impl
    /// block is not a definition of its type; a block's own header pieces, and the gap and
    /// declaration-group units, carry no new name. O(bytes of `text`) for the extraction.
    // MAGI FIX (Loop1 r1, M5): the limitation is declared where it lives.
    // MAGI FIX (MS3 §6 loop9): methods were never recorded; blocks are split one level now.
    fn record_definitions(&mut self, extractor: &ExtractorKind, file: &str, text: &str) {
        // MAGI FIX (MS3 §6 loop5): the duplicate check scanned each name's list, O(n^2) for a
        // name defined many times in one file (generated code); spans seen are now a set.
        let mut seen: HashSet<(String, u32, u32)> = HashSet::new();
        let units = extractor.extract_units(text, file, ChangedLines::All);
        let members = units.iter().flat_map(|block| {
            extractor
                .split_block(block)
                .into_iter()
                .filter(move |p| p.kind != block.kind || p.name != block.name)
        });
        for unit in units.iter().cloned().chain(members) {
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
            // The same span yielded twice is one definition (a path given twice is skipped by
            // the caller).
            if !seen.insert((unit.name.clone(), unit.start_line, unit.end_line)) {
                continue;
            }
            let place = (file.to_string(), unit.start_line, unit.end_line);
            self.definitions.entry(unit.name).or_default().push(place);
        }
    }

    /// Whether `name` occurs in the reviewed code; a qualified citation (`Baz::qux`,
    /// `self.helper`) is looked up by its last segment. O(1) expected.
    pub fn contains(&self, name: &str) -> bool {
        self.identifiers.contains_key(last_segment(name))
    }

    /// Where `name` is defined, as `(file, first_line, last_line)`, in the order the files were
    /// given. Only named top-level units of files with an extractor are definitions (a method
    /// inside an `impl` or `class` block is not one; see `record_definitions`); a name that is
    /// defined more than once (homonyms) lists every place. O(1) expected.
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
    pub fn occurrences(&self, name: &str) -> impl Iterator<Item = (&str, u32)> + '_ {
        self.identifiers
            .get(name)
            .and_then(|&id| self.occurrences.get(id as usize))
            .into_iter()
            .flatten()
            .filter_map(|&(file, line)| {
                self.paths
                    .get(file as usize)
                    .map(|path| (path.as_str(), line))
            })
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

    /// False when `build` stopped recording occurrences, at their cap or because the identifier
    /// cap stopped the scan: some callers may be missing from the context.
    pub fn callers_complete(&self) -> bool {
        self.callers_complete
    }

    #[cfg(test)]
    pub(crate) fn build_with_occurrence_cap(
        files: &[(String, String)],
        registry: &LanguageRegistry,
        cap: usize,
    ) -> Self {
        Self::build_with_limit(files, &[], registry, MAX_INDEX_IDENTIFIERS, cap)
    }
}

/// Lines of each file that some completed batch showed to the trio.
///
/// Per file the ranges are kept merged in a `BTreeMap<start, end>`: inserting is O(log r)
/// amortized and a line lookup is O(log r), with r the ranges of that file.
#[derive(Default)]
pub struct Reviewed {
    ranges: HashMap<String, BTreeMap<u32, u32>>,
    /// Files a completed batch showed only as retrieved context (RV09's `context only`).
    context: HashSet<String>,
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

    /// Records that a completed batch showed `file` as retrieved context (a definition, a
    /// caller, documentation or history), not as code under review. A file that also has
    /// reviewed lines stays reviewed: the context mark only names why an unreviewed file is not
    /// reviewed. O(1) expected.
    ///
    /// # Arguments
    ///
    /// * `file` - The workspace-relative path of the context item.
    pub fn add_context_file(&mut self, file: &str) {
        self.context.insert(file.to_string());
    }

    /// Whether `file` was shown only as retrieved context. O(1) expected.
    pub(crate) fn is_context_file(&self, file: &str) -> bool {
        self.context.contains(file)
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
        // [RED-DRYRUN 2026-10-18] measured: the extractor also yields a gap unit, labelled
        // "lines 2-2" for the one blank line here; it is not a definition.
        assert!(index.definitions("lines 2-2").is_empty());
    }

    #[test]
    fn a_gap_between_definitions_is_not_a_definition() {
        // Same measured label as above, checked on its own with the line of the next definition
        // too: the gap between two functions is named after the lines it spans, and that label is
        // not a definition.
        let files = vec![(
            "a.rs".to_string(),
            "fn parse() {}\n\nfn only() {}\n".to_string(),
        )];
        let index = LexicalIndex::build(&files, &LanguageRegistry::new());
        assert!(index.definitions("lines 2-2").is_empty());
        assert_eq!(index.definitions("only"), &[("a.rs".to_string(), 3, 3)]);
    }

    #[test]
    fn occurrences_are_recorded_with_their_line() {
        let files = vec![(
            "b.rs".to_string(),
            "fn use_it() {\n    parse();\n}\n".to_string(),
        )];
        let index = LexicalIndex::build(&files, &LanguageRegistry::new());
        assert_eq!(
            index.occurrences("parse").collect::<Vec<_>>(),
            vec![("b.rs", 2)]
        );
        assert_eq!(index.occurrences("missing").count(), 0);
    }

    #[test]
    fn a_repeated_occurrence_on_one_line_is_recorded_once() {
        let files = vec![(
            "b.rs".to_string(),
            "fn use_it() { parse(); parse(); }\nfn again() {\n    parse();\n}\n".to_string(),
        )];
        let index = LexicalIndex::build(&files, &LanguageRegistry::new());
        assert_eq!(
            index.occurrences("parse").collect::<Vec<_>>(),
            vec![("b.rs", 1), ("b.rs", 3)]
        );
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

    #[test]
    fn occurrences_stop_at_their_cap_and_the_index_says_callers_may_be_missing() {
        // [MAGI S6 MS3 loop4, R2 Caspar] One name repeated on millions of lines (generated code)
        // grew the place lists without bound; past the cap no place is added and the index
        // says so, so the context declares that callers may be missing.
        let text = "x\n".repeat(10);
        let files = [("a.rs".to_string(), text)];
        let capped = LexicalIndex::build_with_occurrence_cap(&files, &LanguageRegistry::new(), 3);
        assert_eq!(capped.occurrences("x").count(), 3);
        assert!(!capped.callers_complete());
        assert!(capped.contains("x"), "identifiers are not cut by this cap");
        let whole = LexicalIndex::build(&files, &LanguageRegistry::new());
        assert_eq!(whole.occurrences("x").count(), 10);
        assert!(whole.callers_complete());
    }

    #[test]
    fn the_identifier_cap_also_says_callers_may_be_missing() {
        // [MAGI S6 MS3 loop5, R2 Melchior] The identifier cap stopped the scan, so no occurrence
        // of any later file was recorded, but only the places cap cleared `callers_complete`:
        // the context lacked those callers and declared nothing.
        let files = [
            ("a.rs".to_string(), "alpha beta\n".to_string()),
            ("b.rs".to_string(), "gamma alpha\n".to_string()),
        ];
        let capped = LexicalIndex::build_with_cap(&files, &LanguageRegistry::new(), 2);
        assert!(!capped.is_complete());
        assert!(!capped.callers_complete());
        assert!(LexicalIndex::build(&files, &LanguageRegistry::new()).callers_complete());
    }

    #[test]
    fn an_identifier_cap_reached_on_the_old_side_keeps_the_callers_complete() {
        // [MAGI S6 MS3 loop6, R2 Caspar] The old side records identifiers only, so a cap reached
        // there loses no occurrence: every new-side place was recorded, and the context must not
        // declare that callers may be missing.
        let files = [("a.rs".to_string(), "alpha beta\n".to_string())];
        let old = [("a.rs".to_string(), "gamma delta epsilon\n".to_string())];
        let registry = LanguageRegistry::new();
        let capped =
            LexicalIndex::build_with_limit(&files, &old, &registry, 3, MAX_INDEX_OCCURRENCES);
        assert!(
            !capped.is_complete(),
            "precondition: the old side hit the cap"
        );
        assert!(capped.callers_complete());
        assert_eq!(capped.occurrences("alpha").count(), 1);
    }

    #[test]
    fn a_path_given_twice_is_one_file_for_its_occurrences() {
        // [MAGI S6 MS3 loop7, R2 Melchior] Each copy got its own file id, so one line was listed
        // twice and counted twice against MAX_INDEX_OCCURRENCES.
        let file = ("a.rs".to_string(), "fn alpha() {}\n".to_string());
        let index = LexicalIndex::build(&[file.clone(), file], &LanguageRegistry::new());
        assert_eq!(
            index.occurrences("alpha").collect::<Vec<_>>(),
            vec![("a.rs", 1)]
        );
    }
}
