// Author: Julian Bolivar
// Version: 0.25.0
// Date: 2026-11-08

//! Review context assembled by a fixed priority (REQ-RV07, RV03).
//!
//! Each batch carries, besides its own code, what the trio needs to judge it: the unit that
//! contains each change, the definitions of what the code uses, the units that call a changed
//! one, project documentation and validated history. The priority is fixed (the order of
//! [`ContextKind`]): the batch's code, its containing units, definitions and callers,
//! documentation, history. What does not fit the batch budget is cut from the end and named; the
//! containing unit is never cut, so a unit that cannot fit is a typed error.
//!
//! # Secrets (RV22)
//!
//! The units, definitions and callers are read from the `sources` they are given, and the
//! pipeline gives the review's `CachedSource`: it holds every text side with each one-line secret
//! masked before any stage reads it, and refuses a side the filter left out for a private-key
//! block. So these items are masked when they are built, and the text charged to the budget is the
//! masked text (pinned by
//! `a_secret_outside_the_changed_lines_reaches_every_code_context_item_masked` in
//! `memory::review_rag`). Documentation is masked when it is chunked, history when it is
//! retrieved, and the doc query in [`doc_search_query`]; the pipeline's egress pass over every
//! item is a backstop on top of that.

use std::collections::{HashMap, HashSet};

use super::batching::{number_lines, BatchItem};
use super::docs::MAX_DOC_CHUNKS;
use super::error::ReviewError;
use super::index::{identifier_runs, LexicalIndex, MAX_INDEX_OCCURRENCES};
use super::input::renamed_from;
use super::languages::{
    enclosing_unit, ChangedLines, ExtractorKind, LanguageExtractor, LanguageRegistry, SemanticUnit,
    UnitKind,
};
use super::secrets::{display_path, mask_secrets};
use super::source::ContentSource;
use super::window::estimate_tokens;

/// What a context item is. The declaration order is the priority order.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextKind {
    /// The unit that contains a change.
    Unit,
    /// The definition of a name the batch uses.
    Definition,
    /// A unit that names a changed unit.
    Caller,
    /// A chunk of project documentation.
    Documentation,
    /// A validated historical finding.
    History,
}

/// One piece of retrieved context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextItem {
    /// What the item is.
    pub kind: ContextKind,
    /// The file it comes from.
    pub file: String,
    /// A short name for the item.
    pub label: String,
    /// The text sent to the trio.
    pub text: String,
    /// Whether the item is one of several homonyms.
    pub candidate: bool,
    /// The lines of `file` the text covers, when it is a span of a file.
    pub lines: Option<(u32, u32)>,
    /// Whether the text was read from the old side of `file`: a containing unit of lines a
    /// hunk removed (RV21, [`BatchItem::hunk_change`]). `lines` are then old-side coordinates;
    /// `false` for every other item.
    // MAGI FIX (Loop1 r2, Minor 4): wrapped to the line limit.
    pub old_side: bool,
}

/// The context that fits a batch, and what was left out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assembled {
    /// The items that entered, in priority order.
    pub items: Vec<ContextItem>,
    /// The items left out, in priority order.
    pub trimmed: Vec<(ContextKind, String)>,
    /// Whether documentation and history were skipped because no embedder answered: none is
    /// configured, or the configured one failed (then `trimmed` also carries "the embedder did
    /// not answer", once per review, so the report tells the two apart).
    pub without_embedder: bool,
    /// Whether `--no-memory` left out the stores.
    pub without_memory: bool,
    /// Items the secret detector left out: one per history entry and one per documentation
    /// source (a source is withheld whole, however many of its chunks held the secret; U29).
    pub withheld: usize,
}

/// Callers kept per name.
pub const MAX_CALLERS_PER_NAME: usize = 10;

/// What indexing left out.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IndexDeclarations {
    /// Documentation sources the secret detector left out, one per source however many of its
    /// chunks held the secret (a source is indexed whole or not at all, U29).
    pub withheld: usize,
    /// Whether indexing stopped at its cap.
    pub truncated: bool,
    /// Whether a source was left out because reading it passed the review's size cap.
    pub size_capped: bool,
    /// Whether a source was left out because its bytes are not UTF-8 text.
    pub not_utf8: bool,
    /// Whether a working-tree source was left out because it could not be read.
    pub unreadable: bool,
}

/// What the report says when a documentation source was left out for the review's size cap.
const SIZE_CAPPED_SOURCE: &str = "a documentation source over the review's size cap was left out";

/// What the report says when a documentation source was left out because it is not UTF-8.
const NOT_UTF8_SOURCE: &str = "a documentation source that is not UTF-8 text was left out";

/// What the report says when a working-tree documentation source could not be read.
const UNREADABLE_SOURCE: &str = "a documentation source that could not be read was left out";

/// Fixed tokens of one item's frame in the prompt.
pub const CONTEXT_ITEM_FRAME_TOKENS: usize = 18;

/// Longest doc-search query, in bytes.
pub const DOC_QUERY_MAX_BYTES: usize = 8_192;

/// Deepest a block is cut to find the narrowest unit around a line (a method of an impl in a
/// module in a module is three levels); each level must be strictly smaller, so the bound only
/// stops a pathological nesting.
const MAX_BLOCK_SPLIT_DEPTH: usize = 8;

impl ContextItem {
    /// The text the prompt carries for the item (before fence escaping): a containing unit
    /// with known `lines` has each line prefixed with its absolute number, as the reviewed code
    /// is (`number_lines`); every other item is reference text and stays as it is.
    ///
    /// RV09 counts the containing unit as reviewed, so a finding on one of its lines must be
    /// locatable by the number shown. `lines` are old-side coordinates when `old_side`, and the
    /// prefix then shows old-side numbers, the ones the validator accepts for that unit.
    ///
    /// # Returns
    ///
    /// The text to write into the prompt and to charge to the budget. O(bytes).
    // MAGI FIX (T3.7): an unnumbered reviewed line broke "the number shown is the one RV09 accepts".
    pub fn rendered(&self) -> String {
        match (&self.kind, self.lines) {
            (ContextKind::Unit, Some((first, _))) => number_lines(&self.text, first),
            _ => self.text.clone(),
        }
    }
}

/// What `assemble` charges for one item: the tokens of what the prompt carries for it, its text,
/// its path ([`display_path`]) and its fixed frame ([`CONTEXT_ITEM_FRAME_TOKENS`]). The label is
/// not charged because it is never written: the frame carries the kind, the path and the
/// candidate mark only (`an_items_label_is_never_written_so_it_is_never_charged`).
///
/// # Arguments
///
/// * `item` - The context item.
///
/// # Returns
///
/// The item's cost in tokens. O(len of the item's text).
pub fn item_cost(item: &ContextItem) -> usize {
    estimate_tokens(&item.rendered())
        .saturating_add(estimate_tokens(&display_path(&item.file)))
        .saturating_add(CONTEXT_ITEM_FRAME_TOKENS)
}

/// The key [`assemble`] deduplicates a unit, definition or caller by: `(file, lines, side)`; an
/// item without `lines` has none and is never deduplicated.
fn dedup_key(item: &ContextItem) -> Option<(String, (u32, u32), bool)> {
    item.lines
        .map(|lines| (item.file.clone(), lines, item.old_side))
}

/// What [`assemble`] would charge for the batch with `units` and every item of `code`, with the
/// same deduplication: a repeated unit, and a definition or caller that repeats a unit or another
/// item, are charged once. A caller asks it whether anything could still follow `code` before it
/// pays to fetch documentation and history. O(n) in items.
///
/// # Arguments
///
/// * `batch_tokens` - Tokens of the batch's own code.
/// * `units` - The containing units of the batch's changes.
/// * `code` - Definitions followed by callers.
///
/// # Returns
///
/// The tokens charged, saturating.
// MAGI FIX (MS3 §6 loop2): the room check charged a unit once per hunk inside it.
pub fn charged_tokens(batch_tokens: usize, units: &[ContextItem], code: &[ContextItem]) -> usize {
    let mut seen = HashSet::new();
    // `Option::is_none_or` is stable since Rust 1.82; the crate declares `rust-version = 1.98`.
    units
        .iter()
        .chain(code)
        .filter(|item| dedup_key(item).is_none_or(|key| seen.insert(key)))
        .fold(batch_tokens, |acc, item| {
            acc.saturating_add(item_cost(item))
        })
}

/// Adds to the batch the context that fits, by the fixed priority: containing units (mandatory),
/// then definitions and callers, documentation and history.
///
/// `batch_tokens` is `ReviewBatch::tokens`, which does not include the companions: the units are
/// charged here, once, and nothing else charges them. What enters is a prefix of the optional
/// list in priority order: the first item that does not fit and everything after it go to
/// `trimmed`, in that order, even if a later one is smaller. A unit repeated by `(file, lines,
/// side)` (two hunks of one function) is kept once and charged once; a definition or caller with
/// the `(file, lines)` of a new-side unit is not added, because the unit already carries it. An
/// old-side unit and a new-side one with the same numbers are different code and both enter. An
/// item without `lines` is never deduplicated.
///
/// `docs` and `history` are `None` when their source is not there; the result then reports
/// `without_embedder`, a default the caller corrects: with a live embedder the caller always
/// passes `Some` (empty when there is nothing or the source degraded), and `without_memory` is
/// the caller's to set.
///
/// O(n) in items, with one key set.
///
/// # Arguments
///
/// * `batch_tokens` - Tokens of the batch's own code.
/// * `limit_tokens` - The budget for the batch and its context.
/// * `units` - The containing units of the batch's changes.
/// * `definitions` - Definitions followed by callers, in that order.
/// * `docs` - Documentation chunks, best first, or `None`.
/// * `history` - Validated historical findings, best first, or `None`.
///
/// # Returns
///
/// The items that entered and the ones that did not.
///
/// # Errors
///
/// [`ReviewError::TooLargeForBatch`] when `batch_tokens` is over the limit, or a containing unit
/// does not fit with the batch and the units before it; it names the unit, its tokens and the
/// limit.
pub fn assemble(
    batch_tokens: usize,
    limit_tokens: usize,
    units: Vec<ContextItem>,
    definitions: Vec<ContextItem>,
    docs: Option<Vec<ContextItem>>,
    history: Option<Vec<ContextItem>>,
) -> Result<Assembled, ReviewError> {
    if batch_tokens > limit_tokens {
        return Err(ReviewError::TooLargeForBatch {
            what: "the batch".to_string(),
            tokens: batch_tokens,
            limit: limit_tokens,
        });
    }
    let mut used = batch_tokens;
    let mut seen: HashSet<(String, (u32, u32), bool)> = HashSet::new();
    let mut items = Vec::new();
    for unit in units {
        if dedup_key(&unit).is_some_and(|key| !seen.insert(key)) {
            continue;
        }
        let cost = item_cost(&unit);
        used = used.saturating_add(cost);
        if used > limit_tokens {
            return Err(ReviewError::TooLargeForBatch {
                what: format!(
                    "containing unit {} of {}",
                    unit.label,
                    display_path(&unit.file)
                ),
                tokens: cost,
                limit: limit_tokens,
            });
        }
        items.push(unit);
    }
    let without_embedder = docs.is_none() || history.is_none();
    let optional = definitions
        .into_iter()
        .chain(docs.into_iter().flatten())
        .chain(history.into_iter().flatten());
    let mut trimmed = Vec::new();
    let mut full = false;
    for item in optional {
        // MAGI FIX (MS3 §6 loop5): a span counts as seen only once its item entered; a trimmed
        // one marked it too, and a later item of that span went undeclared.
        let key = matches!(item.kind, ContextKind::Definition | ContextKind::Caller)
            .then(|| dedup_key(&item))
            .flatten();
        if key.as_ref().is_some_and(|key| seen.contains(key)) {
            continue;
        }
        if !full {
            let cost = item_cost(&item);
            if used.saturating_add(cost) <= limit_tokens {
                used = used.saturating_add(cost);
                seen.extend(key);
                items.push(item);
                continue;
            }
            full = true;
        }
        trimmed.push((item.kind, item.label));
    }
    Ok(Assembled {
        items,
        trimmed,
        without_embedder,
        without_memory: false,
        withheld: 0,
    })
}

/// Adds what indexing left out to `a`: `withheld` is summed, a truncation becomes a
/// `Documentation` entry in `trimmed` that names [`MAX_DOC_CHUNKS`], and a source left out for
/// the size cap an entry of its own (the report names the cap that cut), as does one left out
/// for not being UTF-8 text.
///
/// # Arguments
///
/// * `a` - The assembled context to declare on.
/// * `d` - What indexing left out.
pub fn declare_index(a: &mut Assembled, d: &IndexDeclarations) {
    a.withheld = a.withheld.saturating_add(d.withheld);
    if d.truncated {
        a.trimmed.push((
            ContextKind::Documentation,
            format!("document index truncated at MAX_DOC_CHUNKS ({MAX_DOC_CHUNKS})"),
        ));
    }
    // MAGI FIX (MS3 §6 loop4): a source over the size cap was declared as the chunk cap.
    if d.size_capped {
        a.trimmed
            .push((ContextKind::Documentation, SIZE_CAPPED_SOURCE.to_string()));
    }
    // MAGI FIX (MS3 §6 loop5): a source that is not UTF-8 left with an INFO entry only.
    if d.not_utf8 {
        a.trimmed
            .push((ContextKind::Documentation, NOT_UTF8_SOURCE.to_string()));
    }
    // MAGI FIX (MS3 §6 loop7): a working-tree source that could not be read was only logged.
    if d.unreadable {
        a.trimmed
            .push((ContextKind::Documentation, UNREADABLE_SOURCE.to_string()));
    }
}

/// Adds to `a` what the lexical index left out: when it stopped recording occurrences at
/// [`MAX_INDEX_OCCURRENCES`], a `Caller` entry in `trimmed` that says callers may be missing.
///
/// # Arguments
///
/// * `a` - The assembled context to declare on (the first batch's, once per review).
/// * `index` - The review's lexical index.
pub fn declare_lexical(a: &mut Assembled, index: &LexicalIndex) {
    if !index.callers_complete() {
        a.trimmed.push((
            ContextKind::Caller,
            format!(
                "occurrence index cut at MAX_INDEX_OCCURRENCES ({MAX_INDEX_OCCURRENCES}): some \
                 callers may be missing"
            ),
        ));
    }
}

/// Reads one side of `file` as UTF-8 text; `None` when that side is gone.
///
/// # Errors
///
/// Any error of `sources`, unchanged, and [`ReviewError::NotReviewable`] for non-UTF-8 bytes.
fn read_text(
    sources: &dyn ContentSource,
    file: &str,
    old_side: bool,
) -> Result<Option<String>, ReviewError> {
    let bytes = if old_side {
        sources.read_old(file)?
    } else {
        sources.read(file)?
    };
    bytes
        .map(|b| {
            String::from_utf8(b).map_err(|_| ReviewError::NotReviewable {
                path: file.to_string(),
                cause: "not utf-8",
            })
        })
        .transpose()
}

/// Lines `first..=last` (1-based) of `text`, joined by `\n`. O(`last`).
fn lines_text(text: &str, first: u32, last: u32) -> String {
    let skip = first.saturating_sub(1) as usize;
    let take = (last.saturating_add(1).saturating_sub(first)) as usize;
    text.lines()
        .skip(skip)
        .take(take)
        .collect::<Vec<_>>()
        .join("\n")
}

/// A unit as a context item of `kind`.
fn unit_item(kind: ContextKind, unit: SemanticUnit) -> ContextItem {
    ContextItem {
        kind,
        file: unit.file,
        label: unit.name,
        text: unit.content,
        candidate: false,
        lines: Some((unit.start_line, unit.end_line)),
        old_side: false,
    }
}

/// The units of one file, to find the narrowest one around a line.
struct FileUnits {
    extractor: ExtractorKind,
    units: Vec<SemanticUnit>,
}

impl FileUnits {
    /// Extracts the units of `text`, leaving out the gap units an extractor emits between
    /// constructs: a line outside every construct has no containing unit. O(bytes of `text`).
    fn load(extractor: &ExtractorKind, file: &str, text: &str) -> Self {
        let units = extractor
            .extract_units(text, file, ChangedLines::All)
            .into_iter()
            .filter(|u| u.kind != UnitKind::TopLevel)
            .collect();
        Self {
            extractor: extractor.clone(),
            units,
        }
    }

    /// The narrowest unit around `line`. A block (impl, class, trait, module) is cut by the
    /// extractor and the piece that holds the line wins, again while that piece is a strictly
    /// smaller block, up to [`MAX_BLOCK_SPLIT_DEPTH`] levels: a hunk in one method of a large
    /// impl, even one nested in a module, is not given the whole impl. O(depth · units).
    // MAGI FIX (MS3 §6 loop1): a block was cut one level only, so a method of an impl inside a
    // module took the whole impl as its mandatory unit.
    fn at(&self, line: u32) -> Option<SemanticUnit> {
        let mut unit = enclosing_unit(&self.units, line)?.clone();
        for _ in 0..MAX_BLOCK_SPLIT_DEPTH {
            if !matches!(
                unit.kind,
                UnitKind::Impl | UnitKind::Class | UnitKind::Trait | UnitKind::Module
            ) {
                break;
            }
            let pieces = self.extractor.split_block(&unit);
            match enclosing_unit(&pieces, line) {
                Some(piece)
                    if piece.end_line.saturating_sub(piece.start_line)
                        < unit.end_line.saturating_sub(unit.start_line) =>
                {
                    unit = piece.clone();
                }
                _ => break,
            }
        }
        Some(unit)
    }
}

/// The definitions of the names the batch code uses, one item per definition.
///
/// The input is the batch's raw code: every identifier counts, quoted or not, in the order it
/// first appears. A name with more than one definition (homonyms) marks each as a `candidate`;
/// a name with none (a keyword, an outside symbol) adds nothing. Each file is read once. Needs a
/// complete index to be exhaustive: when `index.is_complete()` is false some definitions may be
/// missing.
///
/// O(bytes of `text` + bytes of the files read).
///
/// # Arguments
///
/// * `text` - The raw code of the batch.
/// * `index` - The lexical index of the review.
/// * `sources` - The content source of the review (the cached one).
///
/// # Returns
///
/// The `Definition` items.
///
/// # Errors
///
/// An error of `sources`, unchanged; a file that is gone (`Ok(None)`) is skipped.
pub fn definitions_for(
    text: &str,
    index: &LexicalIndex,
    sources: &dyn ContentSource,
) -> Result<Vec<ContextItem>, ReviewError> {
    let mut names = HashSet::new();
    let mut files: HashMap<&str, Option<String>> = HashMap::new();
    let mut items = Vec::new();
    for name in identifier_runs(text) {
        if !names.insert(name) {
            continue;
        }
        let places = index.definitions(name);
        for (file, first, last) in places {
            if !files.contains_key(file.as_str()) {
                files.insert(file, read_text(sources, file, false)?);
            }
            let Some(Some(body)) = files.get(file.as_str()) else {
                continue;
            };
            items.push(ContextItem {
                kind: ContextKind::Definition,
                file: file.clone(),
                label: name.to_string(),
                text: lines_text(body, *first, *last),
                candidate: places.len() > 1,
                lines: Some((*first, *last)),
                old_side: false,
            });
        }
    }
    Ok(items)
}

/// The units that name any of `names` (the changed units), one item per `(file, unit)` and never
/// the definition itself nor a unit with the same name (a nested method the index does not
/// record, a homonym, a recursive call); at most [`MAX_CALLERS_PER_NAME`] per name, the first by
/// `(file, line)`.
///
/// Each file is read and split once. A file without an extractor, or that is gone, gives no
/// caller. A unit already given for an earlier name or occurrence is not repeated.
///
/// O(occurrences of `names` log occurrences + bytes of the files read).
///
/// # Arguments
///
/// * `names` - The names of the changed units.
/// * `index` - The lexical index of the review.
/// * `sources` - The content source of the review (the cached one).
/// * `registry` - The language extractors.
///
/// # Returns
///
/// The `Caller` items (labelled with the caller unit's name), and the names whose callers the
/// cap cut, in `names` order, for the caller to declare.
///
/// # Errors
///
/// An error of `sources`, unchanged.
pub fn callers_for(
    names: &[&str],
    index: &LexicalIndex,
    sources: &dyn ContentSource,
    registry: &LanguageRegistry,
) -> Result<(Vec<ContextItem>, Vec<String>), ReviewError> {
    let mut items = Vec::new();
    let mut capped = Vec::new();
    let mut seen: HashSet<(String, u32, u32)> = HashSet::new();
    let mut files: HashMap<String, Option<FileUnits>> = HashMap::new();
    for name in names {
        let mut places: Vec<(&str, u32)> = index.occurrences(name).collect();
        // Not redundant: `occurrences` yields the files in the order they were given to the
        // index (the listing's order), not by path, and the cap keeps "the first by
        // `(file, line)`"; the sort makes which callers survive the cap independent of that order.
        places.sort_unstable();
        let defined = index.definitions(name);
        let mut kept = 0usize;
        for (file, line) in places {
            if defined
                .iter()
                .any(|(f, first, last)| f == file && (*first..=*last).contains(&line))
            {
                continue;
            }
            if !files.contains_key(file) {
                let loaded = match registry.get(file) {
                    Some(extractor) => read_text(sources, file, false)?
                        .map(|text| FileUnits::load(extractor, file, &text)),
                    None => None,
                };
                files.insert(file.to_string(), loaded);
            }
            let Some(Some(units)) = files.get(file) else {
                continue;
            };
            let Some(unit) = units.at(line) else {
                continue;
            };
            // MAGI FIX (MS3 §6 loop1): a unit with the name's own name is its definition (a nested
            // method the index does not record) or a homonym, never a caller; it took a slot.
            if unit.name == *name {
                continue;
            }
            let key = (file.to_string(), unit.start_line, unit.end_line);
            if seen.contains(&key) {
                continue;
            }
            if kept >= MAX_CALLERS_PER_NAME {
                capped.push((*name).to_string());
                break;
            }
            seen.insert(key);
            kept += 1;
            items.push(unit_item(ContextKind::Caller, unit));
        }
    }
    Ok((items, capped))
}

/// The units that contain a batch item's changed lines, as `Unit` items with their `lines`
/// (RV07(1): "the containing unit of each change"; RV21).
///
/// The anchors are [`BatchItem::hunk_change`]'s: the added lines, looked up on the new side, and
/// the removed lines, looked up on the old side, because the new file has no line for them. The
/// old side of a renamed (or copied) file is read under the `rename from` path its diff header
/// names, while the unit keeps the reviewed path; each side is parsed with the extractor of the
/// path it is read under, so a rename that changes the language reads its old side as that
/// language (`the_old_side_of_a_rename_that_changes_language_is_read_with_its_own_extractor`),
/// and a side whose path has no extractor gives no unit. git's context lines are never anchors, so a
/// change at the top of a function gets that function, not the one above it. Each distinct unit
/// is returned once, new side first, each side in line order: a hunk that spans two units brings
/// both, and a hunk that edits `f` and deletes `g` brings the new `f` and the old `g`. An
/// old-side unit that is the edited unit itself, the same name starting on the line its start
/// moves to across the hunk ([`BatchItem::new_line_of_old`]), is not brought again: its removed
/// lines are in the hunk, and the new version is its unit. A deleted homonym or overload starts
/// on a removed line, so it keeps its old-side unit; so does a unit whose first line was edited
/// (more context, never less). The side
/// is recorded in [`ContextItem::old_side`], so no later stage infers it from the hunks.
/// Planning's companions and the context both use this, so what was charged is what `assemble`
/// adds.
///
/// # Arguments
///
/// * `item` - The batch item.
/// * `sources` - The content source of the review (the cached one).
/// * `registry` - The language extractors.
///
/// # Returns
///
/// The units, possibly none: a unit or a whole file (already a unit), changed lines outside
/// every unit, a file without an extractor, or a side that is gone give none. O(changed lines ·
/// units) plus one read and one extraction of each side with changed lines.
///
/// # Errors
///
/// An error of `sources`, unchanged; [`ReviewError::NotReviewable`] for non-UTF-8 content.
// MAGI FIX (Loop1 r1, I1): one unit at the hunk's first line (git's leading context) became
// every unit of the changed lines.
// MAGI FIX (Loop1 r2, Minor 3): the removed lines of a hunk that also adds bring their units.
pub fn containing_units(
    item: &BatchItem,
    sources: &dyn ContentSource,
    registry: &LanguageRegistry,
) -> Result<Vec<ContextItem>, ReviewError> {
    containing_units_cached(item, sources, registry, &UnitCache::default())
}

/// A file's read path and its units, as [`UnitCache`] holds them for one side.
type HeldUnits = Option<(String, std::sync::Arc<FileUnits>)>;

/// The units of the last file read on each side, so consecutive hunks of one file (the order
/// the planner and a batch walk them in) extract it once instead of once per hunk. Holds at most
/// one file per side, so its memory is one file's units.
#[derive(Default)]
pub struct UnitCache {
    /// The last file extracted on each side (`[new, old]`): its read path and its units.
    last: std::sync::Mutex<[HeldUnits; 2]>,
    /// Extractions done, for tests that count them.
    loads: std::sync::atomic::AtomicUsize,
}

impl UnitCache {
    /// How many files this cache extracted.
    pub fn loads(&self) -> usize {
        self.loads.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// The units of `text` (side `old_side`, read under `read_path`), extracted unless they
    /// are the ones held for that side. The lock is held through the extraction, which never
    /// awaits; a review reads its batches in series (RV05), so it is never contended. Keyed by
    /// path because a review's content source gives the same bytes for a path on every read.
    fn units(
        &self,
        old_side: bool,
        read_path: &str,
        file: &str,
        text: &str,
        extractor: &ExtractorKind,
    ) -> std::sync::Arc<FileUnits> {
        let slot = usize::from(old_side);
        let mut last = self
            .last
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(Some((path, units))) = last.get(slot) {
            if path == read_path {
                return std::sync::Arc::clone(units);
            }
        }
        let units = std::sync::Arc::new(FileUnits::load(extractor, file, text));
        self.loads.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if let Some(held) = last.get_mut(slot) {
            *held = Some((read_path.to_string(), std::sync::Arc::clone(&units)));
        }
        units
    }
}

/// [`containing_units`] through `cache`: a file the cache already holds on that side is not
/// extracted again.
///
/// # Errors
///
/// As [`containing_units`].
pub fn containing_units_cached(
    item: &BatchItem,
    sources: &dyn ContentSource,
    registry: &LanguageRegistry,
    cache: &UnitCache,
) -> Result<Vec<ContextItem>, ReviewError> {
    let (BatchItem::Hunk { file, text, .. }, Some(change)) = (item, item.hunk_change()) else {
        return Ok(Vec::new());
    };
    // MAGI FIX (Loop1 r1, M1): the old side of a rename lives under its `rename from` path,
    // as the filter and the lexical index read it; the hunk's diff header names it.
    let old_path = renamed_from(text).unwrap_or_else(|| file.clone());
    // MAGI FIX (MS3 §6 loop6): each side is parsed with its own path's extractor, so a rename
    // that changes the language (`a.py` to `b.rs`) reads its old side as what it was.
    let side = |lines: &[u32], old_side: bool, read_path: &str| match registry.get(read_path) {
        Some(extractor) => side_units(lines, old_side, file, read_path, sources, extractor, cache),
        None => Ok(Vec::new()),
    };
    let mut found = side(&change.added, false, file)?;
    if !change.removed.is_empty() {
        let old = side(&change.removed, true, &old_path)?;
        // MAGI FIX (Loop1 r3, Minor 2): the same unit by position, never by name alone, so a
        // deleted homonym or overload keeps its old-side unit.
        let new_units: Vec<(String, Option<u32>)> = found
            .iter()
            .map(|u| (u.label.clone(), u.lines.map(|(a, _)| a)))
            .collect();
        found.extend(old.into_iter().filter(|u| {
            let moved = u.lines.and_then(|(a, _)| item.new_line_of_old(a));
            !new_units
                .iter()
                .any(|(name, start)| *name == u.label && moved.is_some() && *start == moved)
        }));
    }
    Ok(found)
}

/// The distinct units around `lines` on one side of `file`, read under `read_path`, as `Unit`
/// items marked with that side; none when `lines` is empty or the side is gone. O(lines ·
/// units) plus one read and one extraction.
///
/// A line inside a unit already found is skipped: that unit carries every line of any narrower
/// unit around it (an impl taken for a change on its header holds the method of a later change),
/// so every changed line still lies in a unit that is sent and nothing is repeated
/// (`a_changed_line_inside_a_unit_already_found_is_covered_by_that_unit`).
///
/// # Errors
///
/// An error of `sources`, unchanged; [`ReviewError::NotReviewable`] for non-UTF-8 content.
fn side_units(
    lines: &[u32],
    old_side: bool,
    file: &str,
    read_path: &str,
    sources: &dyn ContentSource,
    extractor: &ExtractorKind,
    cache: &UnitCache,
) -> Result<Vec<ContextItem>, ReviewError> {
    if lines.is_empty() {
        return Ok(Vec::new());
    }
    let Some(text) = read_text(sources, read_path, old_side)? else {
        return Ok(Vec::new());
    };
    // MAGI FIX (MS3 §6 loop4): each hunk of one file extracted the whole file again.
    let units = cache.units(old_side, read_path, file, &text, extractor);
    let mut found: Vec<ContextItem> = Vec::new();
    for &line in lines {
        let covered = found
            .iter()
            .any(|u| u.lines.is_some_and(|(a, b)| a <= line && line <= b));
        if covered {
            continue;
        }
        if let Some(unit) = units.at(line) {
            found.push(ContextItem {
                old_side,
                ..unit_item(ContextKind::Unit, unit)
            });
        }
    }
    Ok(found)
}
/// The doc-search query: `text` cut to at most [`DOC_QUERY_MAX_BYTES`] bytes, on a char boundary.
///
/// # Arguments
///
/// * `text` - The text to search with.
///
/// # Returns
///
/// A prefix of `text`. O(1) in the cut, never slices inside a character.
pub fn doc_query(text: &str) -> &str {
    cut_on_char(text, DOC_QUERY_MAX_BYTES)
}

/// `text` cut to at most `max` bytes, on a char boundary. O(1) in the cut.
fn cut_on_char(text: &str, max: usize) -> &str {
    let mut end = max.min(text.len());
    while !text.is_char_boundary(end) {
        end = end.saturating_sub(1);
    }
    text.get(..end).unwrap_or_default()
}

/// The text a batch's documentation query sends to the embedder: `prefix` (the embedder's query
/// prefix) followed by `text` with every one-line secret masked ([`mask_secrets`], RV22
/// amended), the whole query at most [`DOC_QUERY_MAX_BYTES`] bytes.
///
/// The text is masked before it is cut: a cut through a raw secret could leave a fragment that
/// no longer matches its pattern. The prefix is masked the same way (it is `[embedding]`
/// operator text, and it leaves the process too). A `prefix` longer than the bound is cut to it
/// (on a char boundary) and leaves no room for the text.
///
/// # Arguments
///
/// * `prefix` - The embedder's query prefix.
/// * `text` - The batch's code.
///
/// # Returns
///
/// The query to embed. O(bytes of `text`).
// MAGI FIX (MS3 §6 loop3): the query left raw, and the bound did not count the prefix.
pub fn doc_search_query(prefix: &str, text: &str) -> String {
    // MAGI FIX (MS3 §6 loop4): an oversized prefix passed whole, over the bound.
    // MAGI FIX (MS3 §6 loop5): the prefix is operator text that leaves too: masked, then cut.
    let masked_prefix = mask_secrets(prefix);
    let prefix = cut_on_char(&masked_prefix, DOC_QUERY_MAX_BYTES);
    let masked = mask_secrets(text);
    let room = DOC_QUERY_MAX_BYTES.saturating_sub(prefix.len());
    format!("{prefix}{}", cut_on_char(&masked, room))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::review::secrets::display_path;
    use crate::review::window::estimate_tokens;

    fn item(kind: ContextKind, label: &str, tokens: usize) -> ContextItem {
        // [CP2 MS3 loop 11] `tokens` is the item's whole cost: its frame, its path ("a.rs" = 1 token)
        // and its text.
        let text_tokens = tokens
            .checked_sub(CONTEXT_ITEM_FRAME_TOKENS + 1)
            .expect("tokens covers the frame");
        ContextItem {
            kind,
            file: "a.rs".into(),
            label: label.into(),
            text: "x".repeat(text_tokens * 4),
            candidate: false,
            lines: None,
            old_side: false,
        }
    }

    fn spanning(kind: ContextKind, lines: Option<(u32, u32)>) -> ContextItem {
        ContextItem {
            kind,
            file: "a.rs".into(),
            label: "f".into(),
            text: "alpha\nbeta\ngamma".into(),
            candidate: false,
            lines,
            old_side: false,
        }
    }

    #[test]
    fn the_embedder_query_prefix_is_masked_too() {
        // [MAGI S6 MS3 loop5, R2 Caspar] Only the batch text was masked; the prefix comes from
        // `[embedding]` operator config, and RV22 covers every byte that leaves the process.
        let token = concat!("ghp_", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"); // allow-secret-scan
        let q = doc_search_query(&format!("search {token}: "), "fn a() {}");
        assert!(!q.contains(token), "{q}");
        assert!(q.ends_with("fn a() {}"), "{q}");
    }

    #[test]
    fn a_trimmed_definition_does_not_hide_a_later_item_of_its_span_from_the_declaration() {
        // [MAGI S6 MS3 loop5, R2 Melchior] The dedup key was taken before the budget check, so a
        // definition that did not fit marked its span as seen, and a caller with the same span
        // after it was dropped silently instead of being declared as trimmed.
        let mut def = spanning(ContextKind::Definition, Some((1, 3)));
        def.label = "def f".into();
        let mut caller = spanning(ContextKind::Caller, Some((1, 3)));
        caller.label = "caller f".into();
        let a = assemble(
            10,
            10,
            vec![],
            vec![def, caller],
            Some(vec![]),
            Some(vec![]),
        )
        .expect("assembles");
        let labels: Vec<&str> = a.trimmed.iter().map(|(_, l)| l.as_str()).collect();
        assert_eq!(labels, ["def f", "caller f"]);
    }

    #[test]
    fn a_containing_unit_is_rendered_with_its_absolute_line_numbers() {
        // [T3.7-fix3] RV09 counts the unit as reviewed, so its lines carry the numbers it accepts.
        let u = spanning(ContextKind::Unit, Some((5, 7)));
        assert_eq!(u.rendered(), "     5|alpha\n     6|beta\n     7|gamma");
    }

    #[test]
    fn the_unit_of_a_deletion_only_hunk_is_numbered_on_the_old_side() {
        // [T3.7-fix3 round 2] its `lines` are old-side coordinates, and so are its numbers.
        use crate::review::test_support::MapSource;
        let old = "fn a() {}\n\nfn gone() {\n    y();\n}\n";
        let src = MapSource::new(&[("a.rs", "fn a() {}\n")], &[("a.rs", old)]);
        let deleted = BatchItem::Hunk {
            file: "a.rs".into(),
            text: "-    y();".into(),
            first_line: 4,
            last_line: 3,
            old_first_line: 4,
            old_last_line: 4,
        };
        let u = containing_units(&deleted, &src, &LanguageRegistry::new())
            .expect("reads")
            .into_iter()
            .next()
            .expect("old side");
        assert!(u.old_side);
        assert_eq!(u.lines, Some((3, 5)));
        assert_eq!(
            u.rendered(),
            "     3|fn gone() {\n     4|    y();\n     5|}"
        );
    }

    #[test]
    fn other_context_stays_unnumbered_and_a_unit_without_lines_too() {
        // [T3.7-fix3] definitions, callers, docs and history are reference only (`context only`).
        for kind in [
            ContextKind::Definition,
            ContextKind::Caller,
            ContextKind::Documentation,
            ContextKind::History,
        ] {
            let item = spanning(kind, Some((5, 7)));
            assert_eq!(item.rendered(), item.text);
        }
        let unit = spanning(ContextKind::Unit, None);
        assert_eq!(unit.rendered(), unit.text);
    }

    #[test]
    fn a_unit_costs_the_tokens_of_its_numbered_text() {
        // [T3.7-fix3] the budget counts what the prompt carries.
        let u = spanning(ContextKind::Unit, Some((5, 7)));
        let numbered = "     5|alpha\n     6|beta\n     7|gamma";
        assert_eq!(
            item_cost(&u),
            estimate_tokens(numbered) + estimate_tokens("a.rs") + CONTEXT_ITEM_FRAME_TOKENS
        );
    }

    #[test]
    fn the_frame_of_each_item_counts_in_its_cost() {
        // [CP2 MS3 loop 11] The prompt carries the delimiters and the path too: the budget counts
        // what is sent.
        let def = || ContextItem {
            kind: ContextKind::Definition,
            file: "a.rs".into(),
            label: "d".into(),
            text: "x".repeat(40),
            candidate: false,
            lines: None,
            old_side: false,
        };
        let cost = 10 + 1 + CONTEXT_ITEM_FRAME_TOKENS; // text, path, frame [CP2 MS3 loop 12: symbolic]
        assert_eq!(item_cost(&def()), cost);
        let fits = assemble(
            100 - cost,
            100,
            vec![],
            vec![def()],
            Some(vec![]),
            Some(vec![]),
        )
        .expect("fits");
        assert_eq!(fits.items.len(), 1, "exactly the limit");
        let over = assemble(
            100 - cost + 1,
            100,
            vec![],
            vec![def()],
            Some(vec![]),
            Some(vec![]),
        )
        .expect("ok");
        assert!(
            over.items.is_empty() && over.trimmed.len() == 1,
            "one token over: trimmed"
        );
    }

    #[test]
    fn an_old_side_unit_and_a_new_side_unit_on_the_same_lines_both_enter() {
        // [MS3 final review m2] Same numbers, different sides: two units, and a definition with
        // the old unit's range is still added (the old unit does not carry it).
        let unit = |old_side: bool, label: &str| {
            let mut u = item(ContextKind::Unit, label, 30);
            u.lines = Some((1, 3));
            u.old_side = old_side;
            u
        };
        let mut def = item(ContextKind::Definition, "d", 30);
        def.lines = Some((1, 3));
        let a = assemble(
            0,
            1_000,
            vec![unit(true, "gone"), unit(false, "kept")],
            vec![def],
            Some(vec![]),
            Some(vec![]),
        )
        .expect("fits");
        let labels: Vec<&str> = a.items.iter().map(|i| i.label.as_str()).collect();
        assert_eq!(
            labels,
            vec!["gone", "kept"],
            "the definition is the new unit's"
        );
        let a = assemble(
            0,
            1_000,
            vec![unit(true, "gone")],
            vec![{
                let mut d = item(ContextKind::Definition, "d", 30);
                d.lines = Some((1, 3));
                d
            }],
            Some(vec![]),
            Some(vec![]),
        )
        .expect("fits");
        let labels: Vec<&str> = a.items.iter().map(|i| i.label.as_str()).collect();
        assert_eq!(labels, vec!["gone", "d"]);
    }

    #[test]
    fn a_containing_unit_records_the_side_it_was_read_from() {
        // [MS3 final review m2] Recorded where it is derived, never inferred from the hunks later.
        use crate::review::test_support::MapSource;
        let sources = MapSource::new(
            &[("a.rs", "fn kept() {\n    y();\n}\n")],
            &[("a.rs", "fn gone() {\n    x();\n}\n")],
        );
        let registry = LanguageRegistry::new();
        let hunk = |first_line, last_line| BatchItem::Hunk {
            file: "a.rs".into(),
            text: "x".into(),
            first_line,
            last_line,
            old_first_line: 2,
            old_last_line: 2,
        };
        let deletion = containing_units(&hunk(2, 1), &sources, &registry)
            .expect("ok")
            .into_iter()
            .next()
            .expect("unit");
        assert!(deletion.old_side && deletion.text.contains("fn gone()"));
        let change = containing_units(&hunk(2, 2), &sources, &registry)
            .expect("ok")
            .into_iter()
            .next()
            .expect("unit");
        assert!(!change.old_side && change.text.contains("fn kept()"));
    }

    #[test]
    fn a_unit_shared_by_two_hunks_enters_once() {
        // [CP2 MS3 loop 11] Two hunks of one function share their unit: charged once, never a false
        // TooLargeForBatch.
        let unit = || {
            let mut u = item(ContextKind::Unit, "f", 30);
            u.lines = Some((3, 9));
            u
        };
        let a = assemble(60, 100, vec![unit(), unit()], vec![], None, None).expect("one copy fits");
        assert_eq!(a.items.len(), 1);
    }

    #[test]
    fn a_definition_with_the_lines_of_a_unit_is_not_repeated() {
        // [CP2 MS3 loop 12] Dedup by (file, lines) across kinds, units first: never charged twice,
        // never trimmed.
        let unit = {
            let mut u = item(ContextKind::Unit, "f", 30);
            u.lines = Some((3, 9));
            u
        };
        let same = {
            let mut d = item(ContextKind::Definition, "f", 30);
            d.lines = Some((3, 9));
            d
        };
        let a =
            assemble(60, 100, vec![unit], vec![same], Some(vec![]), Some(vec![])).expect("fits");
        assert_eq!(
            a.items.iter().map(|i| i.kind.clone()).collect::<Vec<_>>(),
            vec![ContextKind::Unit]
        );
        assert!(a.trimmed.is_empty(), "{:?}", a.trimmed);
    }

    #[test]
    fn items_enter_by_priority_and_the_tail_is_trimmed() {
        let a = assemble(
            10,
            100,
            vec![item(ContextKind::Unit, "u", 20)],
            vec![item(ContextKind::Definition, "d", 30)],
            Some(vec![item(ContextKind::Documentation, "doc", 30)]),
            Some(vec![item(ContextKind::History, "h", 30)]),
        )
        .expect("fits");
        assert_eq!(
            a.items.iter().map(|i| i.label.as_str()).collect::<Vec<_>>(),
            vec!["u", "d", "doc"]
        );
        assert_eq!(a.trimmed, vec![(ContextKind::History, "h".to_string())]);
        assert!(
            !a.without_embedder,
            "[CP2 MS3 loop 12] Some docs and history: the embedder is there"
        );
    }

    #[test]
    fn with_no_room_left_only_the_containing_unit_is_added() {
        let a = assemble(
            80,
            100,
            vec![item(ContextKind::Unit, "u", 20)],
            vec![item(
                ContextKind::Definition,
                "d",
                CONTEXT_ITEM_FRAME_TOKENS + 1,
            )],
            Some(vec![]),
            Some(vec![]),
        )
        .expect("unit fits exactly");
        assert_eq!(a.items.len(), 1);
        assert_eq!(a.trimmed.len(), 1);
    }

    #[test]
    fn a_containing_unit_that_does_not_fit_is_an_error_never_a_trim() {
        assert!(matches!(
            assemble(
                90,
                100,
                vec![item(ContextKind::Unit, "u", 20)],
                vec![],
                None,
                None
            ),
            Err(ReviewError::TooLargeForBatch { .. })
        ));
    }

    #[test]
    fn a_second_containing_unit_that_does_not_fit_is_an_error_too() {
        // [CP2 MS3 loop 6] Units are mandatory one by one: never the first kept and the second
        // dropped.
        let units = vec![
            item(ContextKind::Unit, "u1", 20),
            item(ContextKind::Unit, "u2", 40),
        ];
        assert!(matches!(
            assemble(50, 100, units, vec![], None, None),
            Err(ReviewError::TooLargeForBatch { .. })
        ));
    }

    #[test]
    fn a_batch_over_the_limit_is_an_error_never_an_empty_context() {
        // [CP2 MS3 loop 6, 12] Planning never hands one over (plan_batches_with); a caller that did
        // is a visible defect.
        assert!(matches!(
            assemble(101, 100, vec![], vec![], None, None),
            Err(ReviewError::TooLargeForBatch { .. })
        ));
    }

    #[test]
    fn a_source_read_error_while_gathering_context_is_returned_never_skipped() {
        // [CP2 MS3 loop 6] No silent failure: a definition or caller that cannot be read fails the
        // preparation, typed.
        struct Broken;
        impl ContentSource for Broken {
            fn read(&self, _p: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Err(ReviewError::Io("disk".into()))
            }
            fn read_old(&self, _p: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Ok(None)
            }
        }
        let files = vec![
            ("a.rs".to_string(), "fn parse() {}\n".to_string()),
            (
                "b.rs".to_string(),
                "fn use_it() {\n    parse();\n}\n".to_string(),
            ),
        ];
        let index = LexicalIndex::build(&files, &LanguageRegistry::new());
        assert!(matches!(
            definitions_for("parse();", &index, &Broken),
            Err(ReviewError::Io(_))
        ));
        assert!(matches!(
            callers_for(&["parse"], &index, &Broken, &LanguageRegistry::new()),
            Err(ReviewError::Io(_))
        ));
    }

    #[test]
    fn the_containing_unit_of_a_hunk_is_found_on_its_side_and_priced_as_assemble_charges_it() {
        // [CP2 MS3 loop 12, option 1] One definition for planning's companion and for the context:
        // what plan_batches_with charged is what assemble adds.
        use crate::review::test_support::MapSource;
        let new = "fn f() {\n    let x = 1;\n}\n";
        let old = "fn gone() {\n    y();\n}\n";
        let src = MapSource::new(&[("a.rs", new)], &[("a.rs", old)]);
        let reg = LanguageRegistry::new();
        let hunk = BatchItem::Hunk {
            file: "a.rs".into(),
            text: "+    let x = 1;".into(),
            first_line: 2,
            last_line: 2,
            old_first_line: 2,
            old_last_line: 2,
        };
        let u = containing_units(&hunk, &src, &reg)
            .expect("reads")
            .into_iter()
            .next()
            .expect("inside f");
        assert_eq!(
            (u.kind.clone(), u.label.as_str(), u.lines),
            (ContextKind::Unit, "f", Some((1, 3)))
        );
        assert_eq!(
            item_cost(&u),
            // T3.7: the unit is charged as numbered, what the prompt carries.
            estimate_tokens(&u.rendered())
                + estimate_tokens(&display_path(&u.file))
                + CONTEXT_ITEM_FRAME_TOKENS
        );
        let deleted = BatchItem::Hunk {
            file: "a.rs".into(),
            text: "-    y();".into(),
            first_line: 2,
            last_line: 1,
            old_first_line: 2,
            old_last_line: 2,
        };
        assert_eq!(
            containing_units(&deleted, &src, &reg)
                .expect("reads")
                .into_iter()
                .next()
                .expect("old side")
                .label,
            "gone"
        );
        let whole = BatchItem::WholeFile {
            file: "a.rs".into(),
            text: new.into(),
            lines: 3,
        };
        assert!(
            containing_units(&whole, &src, &reg)
                .expect("reads")
                .is_empty(),
            "a whole file is its own unit"
        );
    }

    /// `a.rs` before the change: `e` (lines 1-3), `f` (5-8), `g` (10-12).
    const THREE_FNS: &str =
        "fn e() {\n    a();\n}\n\nfn f() {\n    b();\n    c();\n}\n\nfn g() {\n    d();\n}\n";

    /// A hunk of `a.rs` with its real `git diff` text (default `-U3` context).
    fn real_hunk(text: &str, new: (u32, u32), old: (u32, u32)) -> BatchItem {
        BatchItem::Hunk {
            file: "a.rs".into(),
            text: text.into(),
            first_line: new.0,
            last_line: new.1,
            old_first_line: old.0,
            old_last_line: old.1,
        }
    }

    /// A containing unit as the tests compare it: `(label, lines, old_side)`.
    type UnitView = (String, Option<(u32, u32)>, bool);

    /// The [`UnitView`] of each containing unit of `item` over `new` and `old`.
    fn units_of(item: &BatchItem, new: &str, old: &str) -> Vec<UnitView> {
        use crate::review::test_support::MapSource;
        let src = MapSource::new(&[("a.rs", new)], &[("a.rs", old)]);
        containing_units(item, &src, &LanguageRegistry::new())
            .expect("reads")
            .into_iter()
            .map(|u| (u.label, u.lines, u.old_side))
            .collect()
    }

    #[test]
    fn consecutive_hunks_of_one_file_extract_it_once() {
        // [MAGI S6 MS3 loop4, R2 Caspar] A batch with many hunks of one large file extracted it
        // once per hunk: O(hunks x file size) before any embedder call.
        use crate::review::test_support::MapSource;
        let new =
            "fn e() {\n    aa();\n}\n\nfn f() {\n    bb();\n    c();\n}\n\nfn g() {\n    d();\n}\n";
        let src = MapSource::new(&[("a.rs", new)], &[("a.rs", THREE_FNS)]);
        let reg = LanguageRegistry::new();
        let first = real_hunk(
            "diff --git a/a.rs b/a.rs\n--- a/a.rs\n+++ b/a.rs\n@@ -2 +2 @@\n-    a();\n+    aa();\n",
            (2, 2),
            (2, 2),
        );
        let second = real_hunk(
            "diff --git a/a.rs b/a.rs\n--- a/a.rs\n+++ b/a.rs\n@@ -6 +6 @@\n-    b();\n+    bb();\n",
            (6, 6),
            (6, 6),
        );
        let cache = UnitCache::default();
        for hunk in [&first, &second] {
            assert_eq!(
                containing_units_cached(hunk, &src, &reg, &cache).expect("reads"),
                containing_units(hunk, &src, &reg).expect("reads")
            );
        }
        assert_eq!(cache.loads(), 2, "one extraction per side, not per hunk");
    }

    #[test]
    fn a_changed_line_inside_a_unit_already_found_is_covered_by_that_unit() {
        // [MAGI S6 MS3 loop4, R2 Balthasar] A change on an impl's header line takes the whole
        // impl (no method holds that line); a later change inside one of its methods is then
        // already covered: the impl carries every line of the method, so the narrower unit
        // would only repeat text. Every changed line still lies inside a unit that is sent.
        let old = "impl T {\n    fn m() {\n        y();\n    }\n}\n";
        let new = "impl S {\n    fn m() {\n        x();\n    }\n}\n";
        let text = "diff --git a/a.rs b/a.rs\nindex 1111111..2222222 100644\n--- a/a.rs\n+++ b/a.rs\n\
                    @@ -1,5 +1,5 @@\n-impl T {\n+impl S {\n     fn m() {\n-        y();\n+        x();\n     }\n }\n";
        let hunk = real_hunk(text, (1, 5), (1, 5));
        let units = units_of(&hunk, new, old);
        for line in [1, 3] {
            assert!(
                units
                    .iter()
                    .any(|(_, l, old_side)| !old_side
                        && l.is_some_and(|(a, b)| a <= line && line <= b)),
                "changed line {line} has no new-side unit: {units:?}"
            );
        }
    }

    #[test]
    fn a_change_at_the_top_of_a_function_gets_that_function_not_the_one_above() {
        // [Loop1 r1, I1] Taken from a real `git diff`: the hunk opens three context lines above
        // the change, on the closing brace of `e`; the change is the first body line of `f`.
        let new =
            "fn e() {\n    a();\n}\n\nfn f() {\n    bb();\n    c();\n}\n\nfn g() {\n    d();\n}\n";
        let text = "diff --git a/a.rs b/a.rs\nindex 31d2d81..9c46387 100644\n--- a/a.rs\n+++ b/a.rs\n\
                    @@ -3,7 +3,7 @@ fn e() {\n }\n \n fn f() {\n-    b();\n+    bb();\n     c();\n }\n \n";
        let hunk = real_hunk(text, (3, 9), (3, 9));
        assert_eq!(
            units_of(&hunk, new, THREE_FNS),
            vec![("f".to_string(), Some((5, 8)), false)]
        );
    }

    #[test]
    fn a_deletion_inside_git_context_takes_the_old_side_unit_of_the_removed_line() {
        // [Loop1 r1, I1] With context the new side is not empty, yet the change only deletes:
        // its unit is the one that held the removed line, on the old side.
        let new = "fn e() {\n    a();\n}\n\nfn f() {\n    c();\n}\n\nfn g() {\n    d();\n}\n";
        let text =
            "diff --git a/a.rs b/a.rs\nindex 31d2d81..7b5f4c7 100644\n--- a/a.rs\n+++ b/a.rs\n\
                    @@ -3,7 +3,6 @@ fn e() {\n }\n \n fn f() {\n-    b();\n     c();\n }\n \n";
        let hunk = real_hunk(text, (3, 8), (3, 9));
        assert_eq!(
            units_of(&hunk, new, THREE_FNS),
            vec![("f".to_string(), Some((5, 8)), true)]
        );
    }

    #[test]
    fn a_hunk_that_changes_two_functions_brings_both_units() {
        // [Loop1 r1, I1] git merged two nearby changes into one hunk that opens on a blank line
        // between units: each changed function is a containing unit, once.
        let new =
            "fn e() {\n    a();\n}\n\nfn f() {\n    b();\n    cc();\n}\n\nfn g() {\n    dd();\n}\n";
        let text = "diff --git a/a.rs b/a.rs\nindex 31d2d81..eec0b67 100644\n--- a/a.rs\n+++ b/a.rs\n\
                    @@ -4,9 +4,9 @@ fn e() {\n \n fn f() {\n     b();\n-    c();\n+    cc();\n }\n \n fn g() {\n\
                    -    d();\n+    dd();\n }\n";
        let hunk = real_hunk(text, (4, 12), (4, 12));
        assert_eq!(
            units_of(&hunk, new, THREE_FNS),
            vec![
                ("f".to_string(), Some((5, 8)), false),
                ("g".to_string(), Some((10, 12)), false)
            ]
        );
    }

    #[test]
    fn a_hunk_that_edits_one_function_and_deletes_another_brings_both_units() {
        // [Loop1 r2, Minor 3] Verbatim `git diff`: `f` is edited and `g` deleted in one hunk.
        // The edited `f` comes from the new side; the deleted `g` exists only on the old side
        // and comes from there. The old `f` is not brought again: the new one is its unit.
        let new = "fn e() {\n    a();\n}\n\nfn f() {\n    b();\n    cc();\n}\n";
        let text = "diff --git a/a.rs b/a.rs\nindex 31d2d81..409fd04 100644\n--- a/a.rs\n+++ b/a.rs\n\
                    @@ -4,9 +4,5 @@ fn e() {\n \n fn f() {\n     b();\n-    c();\n-}\n-\n-fn g() {\n\
                    -    d();\n+    cc();\n }\n";
        let hunk = real_hunk(text, (4, 8), (4, 12));
        assert_eq!(
            units_of(&hunk, new, THREE_FNS),
            vec![
                ("f".to_string(), Some((5, 8)), false),
                ("g".to_string(), Some((10, 12)), true)
            ]
        );
    }

    #[test]
    fn a_deleted_homonym_keeps_its_old_side_unit_next_to_the_edited_one() {
        // [Loop1 r3, Minor 2] Verbatim `git diff`: the first `f` is edited, the second `f` (same
        // name) is deleted. Only the edited `f` is the same unit on both sides; the deleted one
        // keeps its old-side unit.
        use crate::review::test_support::MapSource;
        let old = "def f(a):\n    return 1\n\n\ndef f(a, b):\n    return 2\n";
        let new = "def f(a):\n    return 11\n";
        let src = MapSource::new(&[("m.py", new)], &[("m.py", old)]);
        let text = "diff --git a/m.py b/m.py\nindex 26c5b48..d0065db 100644\n--- a/m.py\n+++ b/m.py\n\
                    @@ -1,6 +1,2 @@\n def f(a):\n-    return 1\n-\n-\n-def f(a, b):\n-    return 2\n\
                    +    return 11\n";
        let hunk = BatchItem::Hunk {
            file: "m.py".into(),
            text: text.into(),
            first_line: 1,
            last_line: 2,
            old_first_line: 1,
            old_last_line: 6,
        };
        let units: Vec<UnitView> = containing_units(&hunk, &src, &LanguageRegistry::new())
            .expect("reads")
            .into_iter()
            .map(|u| (u.label, u.lines, u.old_side))
            .collect();
        assert_eq!(
            units,
            vec![
                ("f".to_string(), Some((1, 2)), false),
                ("f".to_string(), Some((5, 6)), true)
            ]
        );
    }

    #[test]
    fn a_deletion_in_a_renamed_file_reads_its_unit_under_the_old_path() {
        // [Loop1 r1, M1] From a real `git diff -M`: the old side of a rename lives under its
        // `rename from` path (RV21), as the filter and the lexical index read it. The unit keeps
        // the reviewed (new) path, the one RV09 and the gate key it by.
        use crate::review::test_support::MapSource;
        let new = "fn e() {\n    a();\n}\n\nfn f() {\n    c();\n}\n\nfn g() {\n    d();\n}\n";
        let src = MapSource::new(&[("b.rs", new)], &[("a.rs", THREE_FNS)]);
        let text =
            "diff --git a/a.rs b/b.rs\nsimilarity index 87%\nrename from a.rs\nrename to b.rs\n\
                    index 31d2d81..7b5f4c7 100644\n--- a/a.rs\n+++ b/b.rs\n\
                    @@ -3,7 +3,6 @@ fn e() {\n }\n \n fn f() {\n-    b();\n     c();\n }\n";
        let hunk = BatchItem::Hunk {
            file: "b.rs".into(),
            text: text.into(),
            first_line: 3,
            last_line: 8,
            old_first_line: 3,
            old_last_line: 9,
        };
        let units: Vec<(String, UnitView)> =
            containing_units(&hunk, &src, &LanguageRegistry::new())
                .expect("reads")
                .into_iter()
                .map(|u| (u.file, (u.label, u.lines, u.old_side)))
                .collect();
        assert_eq!(
            units,
            vec![("b.rs".to_string(), ("f".to_string(), Some((5, 8)), true))]
        );
    }

    #[test]
    fn the_old_side_of_a_rename_that_changes_language_is_read_with_its_own_extractor() {
        // [MAGI S6 MS3 loop6, R2 Melchior] The extractor was chosen by the reviewed (new) path, so
        // the old side of `a.py` renamed to `b.rs` was parsed as Rust and its unit was lost.
        use crate::review::test_support::MapSource;
        let old = "def f():\n    b()\n    c()\n";
        let src = MapSource::new(&[("b.rs", "fn f() {\n    c();\n}\n")], &[("a.py", old)]);
        let text = "diff --git a/a.py b/b.rs\nsimilarity index 51%\nrename from a.py\n\
                    rename to b.rs\n--- a/a.py\n+++ b/b.rs\n\
                    @@ -1,3 +1,2 @@\n def f():\n-    b()\n     c()\n";
        let hunk = BatchItem::Hunk {
            file: "b.rs".into(),
            text: text.into(),
            first_line: 1,
            last_line: 2,
            old_first_line: 1,
            old_last_line: 3,
        };
        let units: Vec<(String, UnitView)> =
            containing_units(&hunk, &src, &LanguageRegistry::new())
                .expect("reads")
                .into_iter()
                .map(|u| (u.file, (u.label, u.lines, u.old_side)))
                .collect();
        assert_eq!(
            units,
            vec![("b.rs".to_string(), ("f".to_string(), Some((1, 3)), true))]
        );
    }

    #[test]
    fn a_hunk_in_one_method_of_a_large_impl_gets_that_method_not_the_whole_impl() {
        // The extractor yields an impl as one top-level unit; narrowing by `split_block` keeps a
        // hunk from dragging every method of the impl into the context.
        use crate::review::test_support::MapSource;
        let new = "struct A;\nimpl A {\n    fn one() {\n        a();\n    }\n    fn two() {\n        b();\n    }\n}\n";
        let src = MapSource::new(&[("a.rs", new)], &[]);
        let hunk = BatchItem::Hunk {
            file: "a.rs".into(),
            text: "+        b();".into(),
            first_line: 7,
            last_line: 7,
            old_first_line: 7,
            old_last_line: 7,
        };
        let u = containing_units(&hunk, &src, &LanguageRegistry::new())
            .expect("reads")
            .into_iter()
            .next()
            .expect("inside two");
        assert_eq!((u.label.as_str(), u.lines), ("two", Some((6, 8))));
        assert!(!u.text.contains("one"), "{}", u.text);
    }

    #[test]
    fn a_hunk_in_a_method_of_an_impl_inside_a_module_gets_that_method() {
        // [MAGI S6 MS3 loop1, R2 Melchior] A block was cut one level only: a change in a method of
        // an impl nested in a module took the whole impl as its (mandatory, untrimmable) unit.
        use crate::review::test_support::MapSource;
        let new = "mod m {\n    struct A;\n    impl A {\n        fn one() {\n            a();\n        }\n        fn two() {\n            b();\n        }\n    }\n}\n";
        let src = MapSource::new(&[("a.rs", new)], &[]);
        let hunk = BatchItem::Hunk {
            file: "a.rs".into(),
            text: "+            b();".into(),
            first_line: 8,
            last_line: 8,
            old_first_line: 8,
            old_last_line: 8,
        };
        let u = containing_units(&hunk, &src, &LanguageRegistry::new())
            .expect("reads")
            .into_iter()
            .next()
            .expect("inside two");
        assert_eq!(
            (u.label.as_str(), u.lines),
            ("two", Some((7, 9))),
            "{}",
            u.text
        );
    }

    #[test]
    fn the_doc_query_is_cut_on_a_char_boundary() {
        // [CP2 MS3 loop 4] A byte cut inside a multi-byte char would panic (§0: no slicing).
        assert_eq!(doc_query("abc"), "abc");
        assert_eq!(
            doc_query(&"é".repeat(DOC_QUERY_MAX_BYTES)).len(),
            DOC_QUERY_MAX_BYTES
        );
        assert_eq!(
            doc_query(&format!("a{}", "é".repeat(DOC_QUERY_MAX_BYTES))).len(),
            DOC_QUERY_MAX_BYTES - 1
        );
    }

    #[test]
    fn the_doc_search_query_masks_its_secrets_and_bounds_the_whole_query() {
        // [MAGI S6 MS3 loop3, R2 Melchior, Caspar, Balthasar] The query is an egress point
        // (RV22): every one-line secret is masked before the cut, so a cut never leaves a
        // fragment that no longer matches its pattern, and the bound covers the prefix too.
        let secret = "sk-ant-api03-AAAAAAAAAAAAAAAAAAAA"; // allow-secret-scan
        let q = doc_search_query("query: ", &format!("let k = \"{secret}\";"));
        assert!(q.starts_with("query: "), "{q}");
        assert!(!q.contains("sk-ant-"), "{q}");
        assert!(q.contains("<redacted: anthropic-key>"), "{q}");
        let long = doc_search_query("query: ", &"a".repeat(2 * DOC_QUERY_MAX_BYTES));
        assert_eq!(long.len(), DOC_QUERY_MAX_BYTES);
        let straddling = format!(
            "{}{secret}",
            "a".repeat(DOC_QUERY_MAX_BYTES - "query: ".len() - 10)
        );
        let cut = doc_search_query("query: ", &straddling);
        assert!(
            !cut.contains("sk-ant-"),
            "a cut fragment of the key left: {cut}"
        );
    }

    #[test]
    fn a_source_over_the_size_cap_is_declared_apart_from_the_chunk_cap() {
        // [MAGI S6 MS3 loop4, R2 Melchior] RV07 declares what was actually cut: a source too
        // large to read is not a stop at MAX_DOC_CHUNKS.
        let declared = |d: IndexDeclarations| {
            let mut a = assemble(10, 1_000, vec![], vec![], None, None).expect("assemble");
            declare_index(&mut a, &d);
            a.trimmed
        };
        assert_eq!(
            declared(IndexDeclarations {
                size_capped: true,
                ..Default::default()
            }),
            vec![(
                ContextKind::Documentation,
                "a documentation source over the review's size cap was left out".to_string()
            )]
        );
        assert_eq!(
            declared(IndexDeclarations {
                truncated: true,
                ..Default::default()
            }),
            vec![(
                ContextKind::Documentation,
                format!("document index truncated at MAX_DOC_CHUNKS ({MAX_DOC_CHUNKS})")
            )]
        );
    }

    #[test]
    fn an_items_label_is_never_written_so_it_is_never_charged() {
        // [MAGI S6 MS3 loop7, R2 Balthasar] `item_cost` leaves the label out because the frame
        // the prompt writes carries the kind, the path and the candidate mark, never the label.
        let label = "A-LABEL-THAT-THE-PROMPT-NEVER-CARRIES";
        let item = |label: &str| ContextItem {
            kind: ContextKind::Documentation,
            file: "README.md".into(),
            label: label.into(),
            text: "how it works".into(),
            candidate: true,
            lines: None,
            old_side: false,
        };
        assert_eq!(item_cost(&item(label)), item_cost(&item("")));
        let (open, close) =
            crate::review::pipeline::context_frame(&ContextKind::Documentation, "README.md", true);
        assert!(!format!("{open}{close}").contains(label));
    }

    #[test]
    fn a_documentation_source_that_could_not_be_read_is_declared() {
        // [MAGI S6 MS3 loop7, R2 Caspar] A working-tree file the review could not read left the
        // index with a WARN only.
        let mut a = assemble(10, 1_000, vec![], vec![], None, None).expect("assemble");
        let d = IndexDeclarations {
            unreadable: true,
            ..Default::default()
        };
        declare_index(&mut a, &d);
        let want = "a documentation source that could not be read was left out";
        assert_eq!(a.trimmed, vec![(ContextKind::Documentation, want.into())]);
    }

    #[test]
    fn a_documentation_source_that_is_not_utf8_is_declared() {
        // [MAGI S6 MS3 loop5, R2 Caspar] It left the index with an INFO entry only; RV07
        // declares what was cut, and every other omission had its declaration.
        let mut a = assemble(10, 1_000, vec![], vec![], None, None).expect("assemble");
        declare_index(
            &mut a,
            &IndexDeclarations {
                not_utf8: true,
                ..Default::default()
            },
        );
        assert_eq!(
            a.trimmed,
            vec![(
                ContextKind::Documentation,
                "a documentation source that is not UTF-8 text was left out".to_string()
            )]
        );
    }

    #[test]
    fn an_occurrence_index_cut_at_its_cap_is_declared_for_the_callers() {
        // [MAGI S6 MS3 loop4, R2 Caspar] RV07 declares what was cut: past the occurrence cap
        // some callers cannot be found, and the report says so.
        let files = [("a.rs".to_string(), "x\n".repeat(10))];
        let mut a = assemble(10, 1_000, vec![], vec![], None, None).expect("assemble");
        declare_lexical(
            &mut a,
            &LexicalIndex::build_with_occurrence_cap(&files, &LanguageRegistry::new(), 3),
        );
        assert_eq!(
            a.trimmed,
            vec![(
                ContextKind::Caller,
                format!(
                    "occurrence index cut at MAX_INDEX_OCCURRENCES ({MAX_INDEX_OCCURRENCES}): \
                     some callers may be missing"
                )
            )]
        );
        let mut whole = assemble(10, 1_000, vec![], vec![], None, None).expect("assemble");
        declare_lexical(
            &mut whole,
            &LexicalIndex::build(&files, &LanguageRegistry::new()),
        );
        assert!(whole.trimmed.is_empty());
    }

    #[test]
    fn a_query_prefix_longer_than_the_bound_is_cut_to_it_too() {
        // [MAGI S6 MS3 loop4, R2 Caspar] The bound is on the whole query, the prefix included.
        let q = doc_search_query(&"é".repeat(DOC_QUERY_MAX_BYTES), "fn a() {}");
        assert!(q.len() <= DOC_QUERY_MAX_BYTES, "{}", q.len());
    }

    #[test]
    fn without_an_embedder_docs_and_history_are_skipped_and_declared() {
        let a = assemble(0, 100, vec![], vec![], None, None).expect("ok");
        assert!(a.without_embedder);
    }

    #[test]
    fn homonym_definitions_are_marked_candidates() {
        use crate::review::test_support::MapSource;
        let files = vec![
            ("a.rs".to_string(), "fn parse() {}\n".to_string()),
            ("b.rs".to_string(), "fn parse() {}\n".to_string()),
            ("c.rs".to_string(), "fn only() {}\n".to_string()),
        ];
        let index = LexicalIndex::build(&files, &LanguageRegistry::new());
        let src = MapSource::new(
            &[
                ("a.rs", "fn parse() {}\n"),
                ("b.rs", "fn parse() {}\n"),
                ("c.rs", "fn only() {}\n"),
            ],
            &[],
        );
        let defs = definitions_for("call `parse` and `only` here", &index, &src).expect("reads");
        let parse: Vec<_> = defs.iter().filter(|d| d.label == "parse").collect();
        assert_eq!(parse.len(), 2);
        assert!(parse
            .iter()
            .all(|d| d.candidate && d.kind == ContextKind::Definition));
        let only: Vec<_> = defs.iter().filter(|d| d.label == "only").collect();
        assert_eq!(only.len(), 1);
        assert!(!only[0].candidate);
    }

    #[test]
    fn definitions_are_found_from_raw_batch_code_without_backticks() {
        // [CP2 MS3 loop 2] The input is the batch's raw code: every identifier counts, quoted or
        // not.
        use crate::review::test_support::MapSource;
        let files = vec![("a.rs".to_string(), "fn parse() {}\n".to_string())];
        let index = LexicalIndex::build(&files, &LanguageRegistry::new());
        let src = MapSource::new(&[("a.rs", "fn parse() {}\n")], &[]);
        let defs = definitions_for(
            "fn caller() {\n    parse();\n    parse();\n}\n",
            &index,
            &src,
        )
        .expect("reads");
        assert_eq!(
            defs.iter().map(|d| d.label.as_str()).collect::<Vec<_>>(),
            vec!["parse"],
            "[CP2 MS3 loop 12] one per definition"
        );
    }

    #[test]
    fn a_referenced_type_reaches_the_definitions() {
        // [CP2 MS3 loop 10] RV07(2) names referenced types too: a `struct` is a top-level unit with
        // a name.
        use crate::review::test_support::MapSource;
        let files = vec![(
            "a.rs".to_string(),
            "struct Config {\n    n: u32,\n}\n".to_string(),
        )];
        let index = LexicalIndex::build(&files, &LanguageRegistry::new());
        let src = MapSource::new(&[("a.rs", "struct Config {\n    n: u32,\n}\n")], &[]);
        let defs =
            definitions_for("fn f(c: Config) -> u32 {\n    c.n\n}\n", &index, &src).expect("reads");
        assert_eq!(
            defs.iter().map(|d| d.label.as_str()).collect::<Vec<_>>(),
            vec!["Config"]
        );
    }

    #[test]
    fn index_drops_reach_the_assembled_declarations() {
        // [CP2 MS3 loop 2] Secret-dropped doc chunks and a MAX_DOC_CHUNKS cut are declared, never
        // silent.
        let mut a = assemble(0, 100, vec![], vec![], Some(vec![]), Some(vec![])).expect("ok");
        declare_index(
            &mut a,
            &IndexDeclarations {
                withheld: 2,
                truncated: true,
                ..Default::default()
            },
        );
        assert_eq!(a.withheld, 2);
        assert!(
            a.trimmed
                .iter()
                .any(|(k, l)| *k == ContextKind::Documentation && l.contains("MAX_DOC_CHUNKS")),
            "{:?}",
            a.trimmed
        );
    }

    #[test]
    fn the_units_that_name_a_changed_unit_are_its_callers() {
        // [CP2 MS3 loop 1] RV07(2).
        use crate::review::test_support::MapSource;
        let a = "fn parse() {}\n";
        let b = "fn use_it() {\n    parse();\n}\n\nfn unrelated() {}\n";
        let files = vec![
            ("a.rs".to_string(), a.to_string()),
            ("b.rs".to_string(), b.to_string()),
        ];
        let index = LexicalIndex::build(&files, &LanguageRegistry::new());
        let src = MapSource::new(&[("a.rs", a), ("b.rs", b)], &[]);
        let (callers, capped) =
            callers_for(&["parse"], &index, &src, &LanguageRegistry::new()).expect("reads");
        assert_eq!(
            callers
                .iter()
                .map(|c| (c.kind.clone(), c.file.as_str(), c.label.as_str()))
                .collect::<Vec<_>>(),
            vec![(ContextKind::Caller, "b.rs", "use_it")],
            "the definition in a.rs is not its own caller"
        );
        assert!(capped.is_empty());
    }

    #[test]
    fn callers_beyond_the_cap_are_named_never_dropped_silently() {
        // [CP2 MS3 loop 8] §0.2: a name with more callers than the cap comes back so StoreRag
        // declares it.
        use crate::review::test_support::MapSource;
        let a = "fn parse() {}\n".to_string();
        let b: String = (0..=MAX_CALLERS_PER_NAME)
            .map(|i| format!("fn c{i}() {{\n    parse();\n}}\n\n"))
            .collect();
        let files = vec![
            ("a.rs".to_string(), a.clone()),
            ("b.rs".to_string(), b.clone()),
        ];
        let index = LexicalIndex::build(&files, &LanguageRegistry::new());
        let src = MapSource::new(&[("a.rs", a.as_str()), ("b.rs", b.as_str())], &[]);
        let (callers, capped) =
            callers_for(&["parse"], &index, &src, &LanguageRegistry::new()).expect("reads");
        assert_eq!(callers.len(), MAX_CALLERS_PER_NAME);
        assert_eq!(capped, vec!["parse".to_string()]);
    }

    #[test]
    fn a_changed_method_is_never_its_own_caller() {
        // [MAGI S6 MS3 loop1, R2 Melchior] The index records only top-level definitions, so a
        // method's own line was an occurrence: its unit took a caller slot and pushed a real
        // caller past the cap.
        use crate::review::test_support::MapSource;
        let a = "struct S;\nimpl S {\n    fn m(&self) {}\n}\n".to_string();
        let b: String = (0..MAX_CALLERS_PER_NAME)
            .map(|i| format!("fn c{i}() {{\n    S.m();\n}}\n\n"))
            .collect();
        let files = vec![
            ("a.rs".to_string(), a.clone()),
            ("b.rs".to_string(), b.clone()),
        ];
        let index = LexicalIndex::build(&files, &LanguageRegistry::new());
        let src = MapSource::new(&[("a.rs", a.as_str()), ("b.rs", b.as_str())], &[]);
        let (callers, capped) =
            callers_for(&["m"], &index, &src, &LanguageRegistry::new()).expect("reads");
        assert!(capped.is_empty(), "{capped:?}");
        assert_eq!(callers.len(), MAX_CALLERS_PER_NAME);
        assert!(callers.iter().all(|c| c.file == "b.rs"), "{callers:?}");
    }

    #[test]
    fn the_kept_context_is_a_prefix_and_everything_after_the_first_misfit_is_trimmed() {
        // [CP2 MS3 loop 8] RV07 «se recorta desde el final»: a smaller item after a misfit is never
        // kept.
        let docs = vec![
            item(ContextKind::Documentation, "big", 95),
            item(
                ContextKind::Documentation,
                "small",
                CONTEXT_ITEM_FRAME_TOKENS + 1,
            ),
        ];
        let a = assemble(10, 100, vec![], vec![], Some(docs), Some(vec![])).expect("fits");
        assert!(
            a.items.is_empty(),
            "{:?}",
            a.items.iter().map(|i| i.label.as_str()).collect::<Vec<_>>()
        );
        assert_eq!(
            a.trimmed,
            vec![
                (ContextKind::Documentation, "big".to_string()),
                (ContextKind::Documentation, "small".to_string())
            ]
        );
    }
}
