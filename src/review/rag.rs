// Author: Julian Bolivar
// Version: 0.25.0
// Date: 2026-11-15

//! Review context assembled by a fixed priority (REQ-RV07, RV03).
//!
//! Each batch carries, besides its code, the unit containing each change, the definitions of what
//! it uses, the units calling a changed one, documentation and validated history, in that fixed
//! priority ([`ContextKind`]). What does not fit is cut from the end and named; a containing unit
//! is never cut, so one that cannot fit is a typed error.
//!
//! # Secrets (RV22)
//!
//! Units, definitions and callers are read from the review's `CachedSource`, which holds every
//! side masked and refuses a key-block side, so they are built masked and charged masked.
//! Documentation is masked when chunked, history when retrieved, the query in
//! [`doc_search_query`]; the pipeline's egress pass is a backstop on top.

use std::collections::{HashMap, HashSet};

use super::batching::{number_lines, BatchItem};
use super::docs::{MAX_DOC_CHUNKS, MAX_DOC_INDEX_BYTES};
use super::error::ReviewError;
use super::index::{identifier_runs, LexicalIndex, MAX_INDEX_IDENTIFIERS, MAX_INDEX_OCCURRENCES};
use super::input::renamed_from;
use super::languages::{
    enclosing_unit, ChangedLines, ExtractorKind, LanguageExtractor, LanguageRegistry, SemanticUnit,
    UnitKind,
};
use super::secrets::{display_path, holds_key_block, mask_secrets};
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

/// Definitions kept per name by [`definitions_for`]: the first by `(file, line)`; homonyms such as
/// `new` or `from` have hundreds, all marked `candidate`.
pub const MAX_DEFINITIONS_PER_NAME: usize = 5;

/// Definitions kept per batch by [`definitions_for`], over all names, so the names of a batch
/// cannot fill the room that documentation and history are served from.
pub const MAX_DEFINITIONS_TOTAL: usize = 24;

/// What a retrieval cap cut in one batch, for the caller to declare: each cause apart, so a
/// name cut by the batch's total is never declared as one with too many entries of its own.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RetrievalCuts {
    /// The names cut by the per-name cap, in the order they were met.
    pub per_name: Vec<String>,
    /// Whether the batch's total cap was reached with a later name still to serve: the scan
    /// stopped there, so the names after it are not listed (one bounded declaration).
    pub total: bool,
}

/// The tracing target of this module's WARNs (the pipeline's).
const LOG_TARGET: &str = "magi_rs::review";

/// The most occurrences of one name [`callers_for`] inspects (the first by `(file, line)`); a
/// name with more is declared cut like one past [`MAX_CALLERS_PER_NAME`]. Bounds the per-name
/// work (file loads, unit lookups) on generated code, where one name can occur millions of times.
pub const MAX_CALLER_SCAN_PER_NAME: usize = 1_024;

/// Callers kept per batch by [`callers_for`], over all names (three names at
/// [`MAX_CALLERS_PER_NAME`]): each carries a whole unit, so the changed names of a large
/// whole-file batch cannot fill the room that documentation and history are served from, as
/// [`MAX_DEFINITIONS_TOTAL`] does for definitions.
pub const MAX_CALLERS_TOTAL: usize = 3 * MAX_CALLERS_PER_NAME;

/// What indexing left out.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IndexDeclarations {
    /// Documentation sources the secret detector left out, one per source however many of its
    /// chunks held the secret (a source is indexed whole or not at all, U29).
    pub withheld: usize,
    /// Whether indexing stopped at its cap.
    pub truncated: bool,
    /// Whether the indexing deadline passed while sources were still being indexed: the
    /// review runs without documentation, but the groups already written are kept, so a later
    /// review resumes (a cold index, not a down one).
    pub still_indexing: bool,
    /// Whether the cut was at the chunk bound (`MAX_DOC_CHUNKS`).
    pub chunk_cap: bool,
    /// Whether the cut was at the byte bound (`MAX_DOC_INDEX_BYTES`).
    pub byte_cap: bool,
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
///
/// One owned constant with the definition index's depth (S6 loop 17, R6b1).
const MAX_BLOCK_SPLIT_DEPTH: usize = crate::review::index::MAX_DEFINITION_DEPTH;

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
/// pays to fetch documentation and history. Charging every code item, not only those that fit,
/// is exact: [`assemble`] keeps a prefix, so once one code item is trimmed every later item is
/// too, and documentation can enter only when all of `code` did (`docs_follow_overflowed_code_out`).
/// O(n) in items.
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
/// `batch_tokens` excludes the companions: the units are charged here, once. What enters is a
/// prefix of the optional list in priority order: the first item that does not fit and all after
/// it go to `trimmed`, even a smaller one. A unit repeated by `(file, lines, side)` is kept and
/// charged once; a definition or caller with a new-side unit's `(file, lines)` is not added (the
/// unit carries it); old- and new-side units with the same numbers both enter; an item without
/// `lines` is never deduplicated.
///
/// `docs`/`history` `None` (no source) reports `without_embedder`, a default the caller corrects
/// (with a live embedder it passes `Some`); `without_memory` is the caller's to set.
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

/// The `trimmed` label of a documentation index cut at its bounds: it names the bound that cut
/// ([`crate::review::docs::bounded_capped`]), and both when the caller did not say which.
fn truncated_label(d: &IndexDeclarations) -> String {
    let chunks = format!("MAX_DOC_CHUNKS ({MAX_DOC_CHUNKS})");
    let bytes = format!("MAX_DOC_INDEX_BYTES ({MAX_DOC_INDEX_BYTES})");
    match (d.chunk_cap, d.byte_cap) {
        (true, false) => format!("document index truncated at {chunks}"),
        (false, true) => format!("document index truncated at {bytes}"),
        _ => format!("document index truncated at {chunks} or {bytes}"),
    }
}

/// Adds what indexing left out to `a`: `withheld` is summed, a truncation becomes a
/// `Documentation` entry in `trimmed` that names both index bounds, and a source left out for
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
        a.trimmed
            .push((ContextKind::Documentation, truncated_label(d)));
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
    if !index.definitions_complete() {
        a.trimmed.push((
            ContextKind::Definition,
            format!(
                "definition index cut at MAX_INDEX_DEFINITIONS ({}): some definitions may be \
                 missing",
                crate::review::index::MAX_INDEX_DEFINITIONS
            ),
        ));
    }
    // MAGI FIX (MS3 §6 loop13): the identifier cap leaves the definitions incomplete as well.
    if !index.is_complete() {
        a.trimmed.push((
            ContextKind::Definition,
            format!(
                "identifier index cut at MAX_INDEX_IDENTIFIERS ({MAX_INDEX_IDENTIFIERS}): some \
                 definitions may be missing"
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
            // MAGI FIX (MS3 §6 loop17, R2 Caspar): the raw path reached the error's text.
            String::from_utf8(b).map_err(|_| ReviewError::NotReviewable {
                path: display_path(file),
                cause: "not utf-8",
            })
        })
        .transpose()
}

/// What a definition or caller file that could not be read, or was gone when read, declares,
/// after its path.
pub const UNREADABLE_CONTEXT: &str = "could not be read and was left out";

/// The new side of a context file, or `None` when it is gone or cannot be read; either pushes
/// the file onto `unread`, so it costs its own items only, declared, never the batch (the index
/// named it, so a side gone since is a loss to declare like a read error).
fn read_or_note(
    sources: &dyn ContentSource,
    file: &str,
    unread: &mut Vec<String>,
) -> Option<String> {
    // MAGI FIX (MS3 §6 loop9): one unreadable file failed the batch's whole context.
    // MAGI FIX (MS3 §6 loop17, R2 Caspar): a file read back gone (`Ok(None)`) was dropped
    // with no declaration.
    let text = read_text(sources, file, false).ok().flatten();
    if text.is_none() {
        unread.push(display_path(file));
    }
    text
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

/// Openers that mark a comment line once its diff prefix and indentation are off: `//`, `/*`
/// and `*` (C family, Rust doc comments, block-comment bodies) and `#` (Python, shell, TOML).
const COMMENT_OPENERS: [&str; 4] = ["//", "/*", "*", "#"];

/// Whether `line` reads as a comment line: after one diff prefix (`+`, `-` or a space) and its
/// indentation, it opens with one of [`COMMENT_OPENERS`]. A language-agnostic heuristic, used only
/// to rank names (never to drop one): a C line that opens with a dereference (`*p = 1;`), a Rust
/// attribute or a C preprocessor line (`#`) rank with the comments, and a comment that follows
/// code on its line ranks with the code. O(length of `line`).
fn is_comment_line(line: &str) -> bool {
    let body = line
        .strip_prefix(['+', '-', ' '])
        .unwrap_or(line)
        .trim_start();
    COMMENT_OPENERS.iter().any(|o| body.starts_with(o))
}

/// The identifiers of `text`: those of its code lines first, then those of its comment lines
/// ([`is_comment_line`]), each group in order of first appearance. O(bytes of `text`).
fn ranked_identifiers(text: &str) -> impl Iterator<Item = &str> {
    let lines = |comments: bool| {
        text.lines()
            .filter(move |l| is_comment_line(l) == comments)
            .flat_map(identifier_runs)
    };
    lines(false).chain(lines(true))
}

/// The definitions of the names the batch code uses, one item per definition, at most
/// [`MAX_DEFINITIONS_PER_NAME`] per name (the first by `(file, line)`) and
/// [`MAX_DEFINITIONS_TOTAL`] for the batch. The two cuts are returned apart, for the caller to
/// declare: each name cut by the per-name cap, and once whether the total was reached with a
/// later name still to serve (the scan stops at that name, so the cost of a spent total is one
/// flag, never one entry per later name).
///
/// The input is the batch's raw code: every identifier counts, quoted or not. The names of code
/// lines are served first and those of comment lines (`is_comment_line`) after them, each in the
/// order it first appears, so a comment block cannot spend [`MAX_DEFINITIONS_TOTAL`] before the
/// callees of the code below it. Accepted over-retrieval (declared): a name in a comment, in a
/// trailing comment or in a string still fetches its definition when slots are left, and a name
/// in a trailing comment or a string ranks as code. It costs slots and budget only: `assemble`
/// holds the batch's limit and can let it displace lower layers (callers, documentation,
/// history), never a containing unit. A name with more than one definition (homonyms) marks each as a `candidate`;
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
/// * `unread` - Receives each file that could not be read, as [`display_path`] shows it.
///
/// # Returns
///
/// The `Definition` items; a file that is gone or cannot be read gives none. And the
/// [`RetrievalCuts`], for the caller to declare.
pub fn definitions_for(
    text: &str,
    index: &LexicalIndex,
    sources: &dyn ContentSource,
    unread: &mut Vec<String>,
) -> (Vec<ContextItem>, RetrievalCuts) {
    let mut names = HashSet::new();
    let mut files: HashMap<&str, Option<String>> = HashMap::new();
    let mut items = Vec::new();
    let mut cuts = RetrievalCuts::default();
    // MAGI FIX (MS3 §6 loop17, R2 Melchior): names of a comment block spent the total before
    // the callee below it was met; code lines rank first now.
    for name in ranked_identifiers(text) {
        if !names.insert(name) {
            continue;
        }
        let places = index.definitions(name);
        if places.is_empty() {
            continue;
        }
        // MAGI FIX (MS3 §6 loop17, R2): a name met with the total spent was declared as over
        // the per-name cap, one entry per later name; the total is its own cause, said once,
        // and the scan stops there.
        let room = MAX_DEFINITIONS_TOTAL.saturating_sub(items.len());
        if room == 0 {
            cuts.total = true;
            break;
        }
        // MAGI FIX (MS3 §6 loop16): the one retrieval layer without a cap (R2, all three
        // mages): the first MAX_DEFINITIONS_PER_NAME by `(file, line)`, whatever the listing
        // order, and at most MAX_DEFINITIONS_TOTAL for the batch; a cut name is declared.
        let mut ordered: Vec<_> = places.iter().collect();
        ordered.sort_unstable();
        let own = ordered.len().min(MAX_DEFINITIONS_PER_NAME);
        let take = own.min(room);
        if own < ordered.len() {
            cuts.per_name.push(name.to_string());
        }
        cuts.total |= take < own;
        for (file, first, last) in ordered.into_iter().take(take) {
            if !files.contains_key(file.as_str()) {
                files.insert(file, read_or_note(sources, file, unread));
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
    (items, cuts)
}

/// The `k` smallest of `items`, in ascending order, and whether `items` held more. A max-heap of
/// at most `k` entries: O(n log k) time and O(k) memory, whatever n.
fn first_k<T: Ord>(items: impl Iterator<Item = T>, k: usize) -> (Vec<T>, bool) {
    let mut heap = std::collections::BinaryHeap::with_capacity(k.saturating_add(1));
    let mut more = false;
    for item in items {
        if heap.len() < k {
            heap.push(item);
            continue;
        }
        more = true;
        if heap.peek().is_some_and(|top| item < *top) {
            heap.pop();
            heap.push(item);
        }
    }
    (heap.into_sorted_vec(), more)
}

/// The units that name any of `names` (the changed units), one item per `(file, unit)`, never
/// the definition nor a unit of the same name (a homonym, a recursive call); at most
/// [`MAX_CALLERS_PER_NAME`] per name, the first by `(file, line)` among the first
/// [`MAX_CALLER_SCAN_PER_NAME`] occurrences, and at most [`MAX_CALLERS_TOTAL`] for the batch
/// (the scan stops at the first caller past it). Each file is read and split once; one without an
/// extractor, or gone, gives none; a unit already given is not repeated.
///
/// O(occurrences of `names` · log S + bytes of the files read) time and O(S) memory per name,
/// S = the scan bound (a bounded selection, `first_k`).
///
/// # Arguments
///
/// * `names` - The names of the changed units.
/// * `index` - The lexical index of the review.
/// * `sources` - The content source of the review (the cached one).
/// * `registry` - The language extractors.
/// * `unread` - Receives each file that could not be read, as [`display_path`] shows it.
///
/// # Returns
///
/// The `Caller` items (labelled with the caller unit's name), and the [`RetrievalCuts`]: the
/// names whose callers the per-name cap or scan bound cut, in `names` order, and whether the
/// batch total was reached; an unreadable file gives none.
pub fn callers_for(
    names: &[&str],
    index: &LexicalIndex,
    sources: &dyn ContentSource,
    registry: &LanguageRegistry,
    unread: &mut Vec<String>,
) -> (Vec<ContextItem>, RetrievalCuts) {
    let mut items = Vec::new();
    let mut capped = Vec::new();
    let mut seen: HashSet<(String, u32, u32)> = HashSet::new();
    let mut files: HashMap<String, Option<FileUnits>> = HashMap::new();
    let mut total = false;
    'names: for name in names {
        // MAGI FIX (MS3 §6 loop11): only the first MAX_CALLER_SCAN_PER_NAME are inspected; a
        // cut scan is declared.
        // MAGI FIX (MS3 §6 loop12): the first k by `(file, line)`, whatever the listing order
        // (`occurrences` yields files in the order they were given to the index);
        // `the_cap_keeps_the_first_by_file` pins it with the listing in the opposite order.
        // MAGI FIX (MS3 §6 loop17, R2 Melchior): every occurrence was collected before the cap
        // (one name can have millions); a bounded selection holds k at a time.
        let (places, cut) = first_k(index.occurrences(name), MAX_CALLER_SCAN_PER_NAME);
        let defined = index.definitions(name);
        let (mut kept, mut over) = (0usize, false);
        for (file, line) in places {
            if defined
                .iter()
                .any(|(f, first, last)| f == file && (*first..=*last).contains(&line))
            {
                continue;
            }
            if !files.contains_key(file) {
                let loaded = match registry.get(file) {
                    Some(extractor) => read_or_note(sources, file, unread)
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
                over = true;
                break;
            }
            // MAGI FIX (MS3 §6 loop17, R2 Caspar): the batch total, met with a caller still to
            // add, ends the scan; said once.
            if items.len() >= MAX_CALLERS_TOTAL {
                total = true;
                break 'names;
            }
            seen.insert(key);
            kept += 1;
            items.push(unit_item(ContextKind::Caller, unit));
        }
        if over || cut {
            capped.push((*name).to_string());
        }
    }
    (
        items,
        RetrievalCuts {
            per_name: capped,
            total,
        },
    )
}

/// The units that contain a batch item's changed lines, as `Unit` items with their `lines`
/// (RV07(1): "the containing unit of each change"; RV21).
///
/// The anchors are [`BatchItem::hunk_change`]'s: added lines looked up on the new side, removed
/// lines on the old side. A renamed (or copied) file's old side is read under its `rename from`
/// path and parsed with that path's extractor (a side with none gives no unit), the unit keeping
/// the reviewed path. git's context lines are never anchors, so a change at a function's top gets
/// that function. Each distinct unit comes once, new side first, in line order (a hunk editing
/// `f` and deleting `g` brings the new `f` and the old `g`). An old-side unit that is the edited
/// unit itself (same name, starting where its start moves to, [`BatchItem::new_line_of_old`]) is
/// not brought again; a deleted homonym, or a unit whose first line was edited, keeps its old-side
/// unit. The side is in [`ContextItem::old_side`]. Planning's companions use this too, so what
/// was charged is what `assemble` adds.
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

/// A file's read path, its reviewed file and its units, as [`UnitCache`] holds them for one side.
type HeldUnits = Option<(String, String, std::sync::Arc<FileUnits>)>;

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
    /// read path (a review's content source gives the same bytes for a path on every read) and
    /// by reviewed file, which labels the units (`unit_cache_keys_by_file_too`).
    fn units(
        &self,
        old_side: bool,
        read_path: &str,
        file: &str,
        text: &str,
        extractor: &ExtractorKind,
    ) -> std::sync::Arc<FileUnits> {
        let slot = usize::from(old_side);
        // MAGI FIX (MS3 §6 loop11): recovered with a WARN and cleared, as review_rag's `relock`;
        // the slot is replaced whole after an extraction, so what it holds is safe to reuse.
        let mut last = match self.last.lock() {
            Ok(g) => g,
            Err(poisoned) => {
                tracing::event!(
                    target: LOG_TARGET,
                    tracing::Level::WARN,
                    "the containing-unit cache lock was poisoned by a panic; its data is used as left"
                );
                self.last.clear_poison();
                poisoned.into_inner()
            }
        };
        // MAGI FIX (MS3 §6 loop8): keyed by read path alone, another file reused these labels.
        if let Some(Some((path, held_file, units))) = last.get(slot) {
            if path == read_path && held_file == file {
                return std::sync::Arc::clone(units);
            }
        }
        let units = std::sync::Arc::new(FileUnits::load(extractor, file, text));
        self.loads.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if let Some(held) = last.get_mut(slot) {
            let units = std::sync::Arc::clone(&units);
            *held = Some((read_path.to_string(), file.to_string(), units));
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
/// (on a char boundary) and leaves no room for the text. A multi-line private-key block in
/// either is withheld whole (RV22's exception): no query at all.
///
/// # Arguments
///
/// * `prefix` - The embedder's query prefix.
/// * `text` - The batch's code.
///
/// # Returns
///
/// The query to embed, `None` when a key block withholds it. O(bytes of `text`).
// MAGI FIX (MS3 §6 loop3): the query left raw, and the bound did not count the prefix.
// MAGI FIX (MS3 §6 loop11): a key block in the prefix was only masked, never withheld.
pub fn doc_search_query(prefix: &str, text: &str) -> Option<String> {
    // MAGI FIX (MS3 §6 loop13): the pieces are masked JOINED, as they leave: a secret opened by
    // the prefix and finished by the text matches in neither alone.
    let joined = format!("{prefix}{text}");
    if holds_key_block(prefix) || holds_key_block(text) || holds_key_block(&joined) {
        return None;
    }
    // MAGI FIX (MS3 §6 loop4): an oversized prefix passed whole, over the bound.
    // MAGI FIX (MS3 §6 loop5): the prefix is operator text that leaves too: masked, then cut.
    // The cut comes last (a cut through a raw secret could leave a fragment that no longer
    // matches), over the whole masked query, the prefix included.
    let masked = mask_secrets(&joined);
    Some(cut_on_char(&masked, DOC_QUERY_MAX_BYTES).to_string())
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

    #[test]
    fn docs_follow_overflowed_code_out() {
        // [MAGI S6 MS3 loop8, R2 Melchior] What enters is a prefix: once a definition does not
        // fit, a documentation chunk that would fit after it is trimmed too. So the room check,
        // which charges every code item, skips exactly what `assemble` would trim.
        let def = item(ContextKind::Definition, "big", 600);
        let doc = item(ContextKind::Documentation, "small", 20);
        let a = assemble(
            500,
            1000,
            Vec::new(),
            vec![def.clone()],
            Some(vec![doc]),
            Some(vec![]),
        )
        .expect("assembled");
        assert!(a.items.is_empty(), "{:?}", a.items);
        let kinds: Vec<ContextKind> = a.trimmed.iter().map(|(k, _)| k.clone()).collect();
        assert_eq!(
            kinds,
            vec![ContextKind::Definition, ContextKind::Documentation]
        );
        let used = charged_tokens(500, &[], &[def]);
        assert!(1000usize.saturating_sub(used) < CONTEXT_ITEM_FRAME_TOKENS);
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
        let q = doc_search_query(&format!("search {token}: "), "fn a() {}").expect("q");
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
    fn unreadable_file_degrades_item() {
        // [CP2 MS3 loop 6] No silent failure. [MAGI S6 MS3 loop9, R2 Melchior] One unreadable
        // definition or caller file failed the whole review; it now leaves out its own items
        // only, each file named for the caller to declare.
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
        let mut unread = Vec::new();
        let (defs, _) = definitions_for("parse();", &index, &Broken, &mut unread);
        assert!(defs.is_empty());
        let reg = LanguageRegistry::new();
        let (callers, _) = callers_for(&["parse"], &index, &Broken, &reg, &mut unread);
        assert!(callers.is_empty());
        assert_eq!(unread, vec!["a.rs".to_string(), "b.rs".to_string()]);
    }

    #[test]
    fn a_context_file_that_reads_back_gone_is_declared_like_an_unreadable_one() {
        // [MAGI S6 MS3 loop17, R2 Caspar] A definition or caller file the index knows but the
        // source answers `Ok(None)` for (gone since the index was built) was dropped silently.
        struct Gone;
        impl ContentSource for Gone {
            fn read(&self, _p: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Ok(None)
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
        let mut unread = Vec::new();
        let (defs, _) = definitions_for("parse();", &index, &Gone, &mut unread);
        assert!(defs.is_empty());
        let reg = LanguageRegistry::new();
        let (callers, _) = callers_for(&["parse"], &index, &Gone, &reg, &mut unread);
        assert!(callers.is_empty());
        assert_eq!(unread, vec!["a.rs".to_string(), "b.rs".to_string()]);
    }

    #[test]
    fn a_non_utf8_side_names_its_file_as_display_path_shows_it() {
        // [MAGI S6 MS3 loop17, R2 Caspar] `read_text` put the raw path into `NotReviewable`, the
        // one error path of this module that bypassed `display_path`.
        struct NotUtf8;
        impl ContentSource for NotUtf8 {
            fn read(&self, _p: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Ok(Some(vec![0xff, 0xfe, b'\n']))
            }
            fn read_old(&self, _p: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Ok(None)
            }
        }
        let file = "src/a\nb\u{202e}.rs";
        let hunk = BatchItem::Hunk {
            file: file.into(),
            text: "+x".into(),
            first_line: 1,
            last_line: 1,
            old_first_line: 1,
            old_last_line: 1,
        };
        match containing_units(&hunk, &NotUtf8, &LanguageRegistry::new()) {
            Err(ReviewError::NotReviewable { path, .. }) => {
                assert_eq!(path, display_path(file));
                assert!(!path.contains('\n'), "{path:?}");
            }
            other => panic!("expected NotReviewable, got {other:?}"),
        }
    }

    #[test]
    fn egress_secret_read_is_noted() {
        // [MAGI S6 MS3 loop10, R6a Melchior] A sealed cache refusing a raw-secret side
        // (`EgressSecret`) costs a definition or caller its own item, declared, never the review.
        struct Sealed;
        impl ContentSource for Sealed {
            fn read(&self, p: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Err(ReviewError::EgressSecret { path: p.into() })
            }
            fn read_old(&self, _p: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Ok(None)
            }
        }
        let files = vec![("a.rs".to_string(), "fn parse() {}\n".to_string())];
        let index = LexicalIndex::build(&files, &LanguageRegistry::new());
        let mut unread = Vec::new();
        assert!(definitions_for("parse();", &index, &Sealed, &mut unread)
            .0
            .is_empty());
        assert_eq!(unread, vec!["a.rs".to_string()]);
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
    fn unit_cache_keys_by_file_too() {
        // [MAGI S6 MS3 loop8, R2 Melchior] Two reviewed files read under one old path (two
        // `copy from` hunks) shared the first file's units, labeled with its name.
        let reg = LanguageRegistry::new();
        let extractor = reg.get("a.rs").expect("rust");
        let cache = UnitCache::default();
        let text = "fn f() {\n    x();\n}\n";
        let a = cache.units(true, "old.rs", "a.rs", text, extractor);
        let b = cache.units(true, "old.rs", "b.rs", text, extractor);
        assert_eq!(cache.loads(), 2, "another reviewed file is extracted again");
        assert!(!std::sync::Arc::ptr_eq(&a, &b));
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
        let doc_query = |t: &str| cut_on_char(t, DOC_QUERY_MAX_BYTES).len();
        assert_eq!(doc_query("abc"), 3);
        assert_eq!(
            doc_query(&"é".repeat(DOC_QUERY_MAX_BYTES)),
            DOC_QUERY_MAX_BYTES
        );
        assert_eq!(
            doc_query(&format!("a{}", "é".repeat(DOC_QUERY_MAX_BYTES))),
            DOC_QUERY_MAX_BYTES - 1
        );
    }

    #[test]
    fn the_doc_search_query_masks_its_secrets_and_bounds_the_whole_query() {
        // [MAGI S6 MS3 loop3, R2 Melchior, Caspar, Balthasar] The query is an egress point
        // (RV22): every one-line secret is masked before the cut, so a cut never leaves a
        // fragment that no longer matches its pattern, and the bound covers the prefix too.
        let secret = "sk-ant-api03-AAAAAAAAAAAAAAAAAAAA"; // allow-secret-scan
        let q = doc_search_query("query: ", &format!("let k = \"{secret}\";")).expect("q");
        assert!(q.starts_with("query: "), "{q}");
        assert!(!q.contains("sk-ant-"), "{q}");
        assert!(q.contains("<redacted: anthropic-key>"), "{q}");
        let long = doc_search_query("query: ", &"a".repeat(2 * DOC_QUERY_MAX_BYTES)).expect("q");
        assert_eq!(long.len(), DOC_QUERY_MAX_BYTES);
        let straddling = format!(
            "{}{secret}",
            "a".repeat(DOC_QUERY_MAX_BYTES - "query: ".len() - 10)
        );
        let cut = doc_search_query("query: ", &straddling).expect("q");
        assert!(
            !cut.contains("sk-ant-"),
            "a cut fragment of the key left: {cut}"
        );
    }

    #[test]
    fn a_truncation_names_the_bound_that_cut() {
        // [MAGI S6 MS3 loop17, R1] The label named both bounds whichever one cut.
        let label = |chunk_cap, byte_cap| {
            let mut a = assemble(10, 1_000, vec![], vec![], None, None).expect("assemble");
            declare_index(
                &mut a,
                &IndexDeclarations {
                    truncated: true,
                    chunk_cap,
                    byte_cap,
                    ..Default::default()
                },
            );
            a.trimmed[0].1.clone()
        };
        let by_bytes = label(false, true);
        assert!(by_bytes.contains("MAX_DOC_INDEX_BYTES"), "{by_bytes}");
        assert!(!by_bytes.contains("MAX_DOC_CHUNKS"), "{by_bytes}");
        let by_chunks = label(true, false);
        assert!(by_chunks.contains("MAX_DOC_CHUNKS"), "{by_chunks}");
        assert!(!by_chunks.contains("MAX_DOC_INDEX_BYTES"), "{by_chunks}");
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
                format!(
                    "document index truncated at MAX_DOC_CHUNKS ({MAX_DOC_CHUNKS}) or \
                     MAX_DOC_INDEX_BYTES ({MAX_DOC_INDEX_BYTES})"
                )
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
    fn an_identifier_index_cut_at_its_cap_is_declared_for_definitions() {
        // MAGI FIX (MS3 §6 loop13, R2 Caspar, Melchior): the identifier cap made the index
        // incomplete (definitions may be missing) and no report said so, only the symbol check.
        let files = [("a.rs".to_string(), "fn one() {}\nfn two() {}\n".to_string())];
        let mut capped = assemble(10, 1_000, vec![], vec![], None, None).expect("assemble");
        declare_lexical(
            &mut capped,
            &LexicalIndex::build_with_cap(&files, &LanguageRegistry::new(), 1),
        );
        // The cap also stops the occurrences, so the callers entry may sit beside it.
        assert!(
            capped.trimmed.contains(&(
                ContextKind::Definition,
                format!(
                    "identifier index cut at MAX_INDEX_IDENTIFIERS ({MAX_INDEX_IDENTIFIERS}): \
                     some definitions may be missing"
                )
            )),
            "{:?}",
            capped.trimmed
        );
        let mut whole = assemble(10, 1_000, vec![], vec![], None, None).expect("assemble");
        declare_lexical(
            &mut whole,
            &LexicalIndex::build(&files, &LanguageRegistry::new()),
        );
        assert!(whole.trimmed.is_empty());
    }

    #[test]
    fn a_secret_across_the_prefix_boundary_is_masked() {
        // MAGI FIX (MS3 §6 loop13, R2 Caspar): prefix and text were masked apart, so a key
        // opened by the prefix and finished by the text matched in neither and went out whole.
        let body = "AAAAAAAAAAAAAAAAAAAA";
        let q = doc_search_query("query: sk-ant-api03-", body).expect("q"); // allow-secret-scan
        assert!(!q.contains(body), "{q}");
        assert!(q.contains("<redacted: anthropic-key>"), "{q}");
    }

    #[test]
    fn a_query_prefix_longer_than_the_bound_is_cut_to_it_too() {
        // [MAGI S6 MS3 loop4, R2 Caspar] The bound is on the whole query, the prefix included.
        let q = doc_search_query(&"é".repeat(DOC_QUERY_MAX_BYTES), "fn a() {}").expect("q");
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
        let (defs, _) = definitions_for(
            "call `parse` and `only` here",
            &index,
            &src,
            &mut Vec::new(),
        );
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
    fn homonym_definitions_are_capped_per_name_and_in_total_and_declared() {
        // [MAGI S6 MS3 loop16, R2 Caspar/Melchior/Balthasar] `definitions_for` was the one layer
        // with no cap: a batch naming `new` read and built every definition of it, charged them
        // all, and left no room for documentation and history.
        use crate::review::test_support::MapSource;
        let body = "fn new() {}\n";
        let files: Vec<(String, String)> = (0..40)
            .map(|i| (format!("f{i:02}.rs"), body.to_string()))
            .collect();
        let borrowed: Vec<(&str, &str)> = files
            .iter()
            .map(|(p, t)| (p.as_str(), t.as_str()))
            .collect();
        let index = LexicalIndex::build(&files, &LanguageRegistry::new());
        let src = MapSource::new(&borrowed, &[]);
        let (defs, capped) = definitions_for("new();", &index, &src, &mut Vec::new());
        assert_eq!(defs.len(), MAX_DEFINITIONS_PER_NAME);
        assert_eq!(capped.per_name, ["new"]);
        assert!(!capped.total, "five definitions never reach the total");
        // The kept ones are the first by (file, line), whatever the listing order.
        let mut kept: Vec<&str> = defs.iter().map(|d| d.file.as_str()).collect();
        let sorted = {
            let mut s = kept.clone();
            s.sort_unstable();
            s
        };
        kept.sort_unstable();
        assert_eq!(kept, sorted);
        assert_eq!(kept.first().copied(), Some("f00.rs"));

        // Many names, a few definitions each: the total bounds the batch.
        let many: Vec<(String, String)> = (0..MAX_DEFINITIONS_TOTAL * 2)
            .map(|i| (format!("g{i:02}.rs"), format!("fn name{i}() {{}}\n")))
            .collect();
        let borrowed: Vec<(&str, &str)> =
            many.iter().map(|(p, t)| (p.as_str(), t.as_str())).collect();
        let index = LexicalIndex::build(&many, &LanguageRegistry::new());
        let src = MapSource::new(&borrowed, &[]);
        let call: String = (0..MAX_DEFINITIONS_TOTAL * 2)
            .map(|i| format!("name{i}();\n"))
            .collect();
        let (defs, capped) = definitions_for(&call, &index, &src, &mut Vec::new());
        assert_eq!(defs.len(), MAX_DEFINITIONS_TOTAL);
        // [MAGI S6 MS3 loop17, R2 all three] A name cut by the total was declared as one with
        // more than MAX_DEFINITIONS_PER_NAME definitions, once per later name.
        assert!(
            capped.per_name.is_empty(),
            "a one-definition name is never declared as over the per-name cap: {:?}",
            capped.per_name
        );
        assert!(capped.total, "the total cut is declared, once");
    }

    #[test]
    fn identifiers_in_comment_lines_never_displace_a_callee_from_the_code() {
        // [MAGI S6 MS3 loop17, R2 Melchior] The total was spent in order of first appearance over
        // the raw text, comments included: names in a doc block above a function took every slot
        // and its real callee was left out, with room to spare.
        use crate::review::test_support::MapSource;
        let many: Vec<(String, String)> = (0..MAX_DEFINITIONS_TOTAL + 6)
            .map(|i| (format!("g{i:02}.rs"), format!("fn name{i}() {{}}\n")))
            .chain([("z.rs".to_string(), "fn callee() {}\n".to_string())])
            .collect();
        let borrowed: Vec<(&str, &str)> =
            many.iter().map(|(p, t)| (p.as_str(), t.as_str())).collect();
        let index = LexicalIndex::build(&many, &LanguageRegistry::new());
        let src = MapSource::new(&borrowed, &[]);
        // A hunk: every line carries its diff prefix, as the batch text does.
        let text: String = (0..MAX_DEFINITIONS_TOTAL + 6)
            .map(|i| format!("+/// See name{i} and its cousins.\n"))
            .chain([
                " # note name0\n".to_string(),
                "+    callee();\n".to_string(),
            ])
            .collect();
        let (defs, cuts) = definitions_for(&text, &index, &src, &mut Vec::new());
        assert_eq!(
            defs.first().map(|d| d.label.as_str()),
            Some("callee"),
            "a name of the code comes before every name of a comment"
        );
        assert_eq!(defs.len(), MAX_DEFINITIONS_TOTAL);
        assert!(
            cuts.total,
            "the comment names past the total are still declared cut"
        );
    }

    #[test]
    fn a_spent_total_with_no_later_definition_declares_no_cut() {
        // [MAGI S6 MS3 loop17, R2] The scan stops at the first later name that would have
        // served; names without a definition (keywords, outside symbols) cut nothing.
        use crate::review::test_support::MapSource;
        let many: Vec<(String, String)> = (0..MAX_DEFINITIONS_TOTAL)
            .map(|i| (format!("g{i:02}.rs"), format!("fn name{i}() {{}}\n")))
            .collect();
        let borrowed: Vec<(&str, &str)> =
            many.iter().map(|(p, t)| (p.as_str(), t.as_str())).collect();
        let index = LexicalIndex::build(&many, &LanguageRegistry::new());
        let src = MapSource::new(&borrowed, &[]);
        let call: String = (0..MAX_DEFINITIONS_TOTAL)
            .map(|i| format!("name{i}();\n"))
            .chain(["let outside = 1;\n".to_string()])
            .collect();
        let (defs, capped) = definitions_for(&call, &index, &src, &mut Vec::new());
        assert_eq!(defs.len(), MAX_DEFINITIONS_TOTAL);
        assert_eq!(capped, RetrievalCuts::default());
    }

    #[test]
    fn definitions_are_found_from_raw_batch_code_without_backticks() {
        // [CP2 MS3 loop 2] The input is the batch's raw code: every identifier counts, quoted or
        // not.
        use crate::review::test_support::MapSource;
        let files = vec![("a.rs".to_string(), "fn parse() {}\n".to_string())];
        let index = LexicalIndex::build(&files, &LanguageRegistry::new());
        let src = MapSource::new(&[("a.rs", "fn parse() {}\n")], &[]);
        let (defs, _) = definitions_for(
            "fn caller() {\n    parse();\n    parse();\n}\n",
            &index,
            &src,
            &mut Vec::new(),
        );
        assert_eq!(
            defs.iter().map(|d| d.label.as_str()).collect::<Vec<_>>(),
            vec!["parse"],
            "[CP2 MS3 loop 12] one per definition"
        );
    }

    #[test]
    fn methods_are_retrieved_as_defs() {
        // [MAGI S6 MS3 loop9, R2 Melchior] RV07(2): a method of a Rust `impl` or a Python class
        // was never recorded, so `obj.method()` brought no definition.
        use crate::review::test_support::MapSource;
        let rs = "struct S;\n\nimpl S {\n    fn run(&self) {}\n}\n";
        let py = "class C:\n    def go(self):\n        pass\n";
        let files = vec![
            ("a.rs".to_string(), rs.to_string()),
            ("c.py".to_string(), py.to_string()),
        ];
        let index = LexicalIndex::build(&files, &LanguageRegistry::new());
        let src = MapSource::new(&[("a.rs", rs), ("c.py", py)], &[]);
        let (defs, _) = definitions_for("s.run(); c.go()", &index, &src, &mut Vec::new());
        let got: Vec<_> = defs.iter().map(|d| (d.label.as_str(), d.lines)).collect();
        assert_eq!(got, vec![("run", Some((4, 4))), ("go", Some((2, 3)))]);
    }

    #[test]
    fn nested_methods_are_defs() {
        // [MAGI S6 MS3 loop10, R2 Melchior] Only one level down a block was recorded, so a method
        // of an impl inside a module (or a nested class) was no definition.
        let files: Vec<(String, String)> = [
            (
                "a.rs",
                "mod m {\n    impl T {\n        fn f() {}\n    }\n}\n",
            ),
            (
                "A.java",
                "class A {\n  class B {\n    void g() {}\n  }\n}\n",
            ),
            (
                "c.py",
                "class A:\n    class B:\n        def h(self):\n            pass\n",
            ),
        ]
        .iter()
        .map(|(p, t)| (p.to_string(), t.to_string()))
        .collect();
        let index = LexicalIndex::build(&files, &LanguageRegistry::new());
        for (name, file) in [("f", "a.rs"), ("g", "A.java"), ("h", "c.py")] {
            let defs = index.definitions(name);
            assert!(defs.iter().any(|d| d.0 == file), "{name}: {defs:?}");
        }
    }

    #[test]
    fn unit_cache_poison_warns() {
        // MAGI FIX (MS3 §6 loop11): UnitCache recovered a poisoned lock in silence.
        use tracing_subscriber::layer::SubscriberExt;
        #[derive(Clone, Default)]
        struct Count(std::sync::Arc<std::sync::atomic::AtomicUsize>);
        impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for Count {
            fn on_event(
                &self,
                e: &tracing::Event<'_>,
                _: tracing_subscriber::layer::Context<'_, S>,
            ) {
                if *e.metadata().level() == tracing::Level::WARN {
                    self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }
            }
        }
        let cache = UnitCache::default();
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _held = cache.last.lock();
            panic!("poison");
        }));
        let count = Count::default();
        let _guard =
            tracing::subscriber::set_default(tracing_subscriber::registry().with(count.clone()));
        let registry = LanguageRegistry::new();
        let x = registry.get("a.rs").expect("rust");
        cache.units(false, "a.rs", "a.rs", "fn a() {}\n", x);
        assert_eq!(count.0.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert!(!cache.last.is_poisoned(), "cleared");
    }

    #[test]
    fn same_name_nested_keeps_defs() {
        // MAGI FIX (MS3 §6 loop11): a block nested in one of its own kind and name was dropped
        // with its definitions.
        let files: Vec<(String, String)> = [
            (
                "a.rs",
                "mod m {\n    mod m {\n        fn f() {}\n    }\n}\n",
            ),
            (
                "c.py",
                "class A:\n    class A:\n        def h(self):\n            pass\n",
            ),
        ]
        .iter()
        .map(|(p, t)| (p.to_string(), t.to_string()))
        .collect();
        let index = LexicalIndex::build(&files, &LanguageRegistry::new());
        for (name, file) in [("f", "a.rs"), ("h", "c.py")] {
            let defs = index.definitions(name);
            assert!(defs.iter().any(|d| d.0 == file), "{name}: {defs:?}");
        }
        let m = index.definitions("m");
        assert_eq!(m.len(), 2, "both modules, no header piece: {m:?}");
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
        let (defs, _) = definitions_for(
            "fn f(c: Config) -> u32 {\n    c.n\n}\n",
            &index,
            &src,
            &mut Vec::new(),
        );
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
        let (callers, capped) = callers_for(
            &["parse"],
            &index,
            &src,
            &LanguageRegistry::new(),
            &mut Vec::new(),
        );
        assert_eq!(
            callers
                .iter()
                .map(|c| (c.kind.clone(), c.file.as_str(), c.label.as_str()))
                .collect::<Vec<_>>(),
            vec![(ContextKind::Caller, "b.rs", "use_it")],
            "the definition in a.rs is not its own caller"
        );
        assert_eq!(capped, RetrievalCuts::default());
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
        let (callers, capped) = callers_for(
            &["parse"],
            &index,
            &src,
            &LanguageRegistry::new(),
            &mut Vec::new(),
        );
        assert_eq!(callers.len(), MAX_CALLERS_PER_NAME);
        assert_eq!(capped.per_name, vec!["parse".to_string()]);
    }

    #[test]
    fn caller_scan_is_bounded() {
        // MAGI FIX (MS3 §6 loop11): every occurrence of a name was inspected before the cap.
        use crate::review::test_support::MapSource;
        let a = "fn parse() {}\n".to_string();
        let mut b = "// parse\n".repeat(MAX_CALLER_SCAN_PER_NAME);
        b.push_str("fn c() {\n    parse();\n}\n");
        let files = vec![
            ("a.rs".to_string(), a.clone()),
            ("b.rs".to_string(), b.clone()),
        ];
        let index = LexicalIndex::build(&files, &LanguageRegistry::new());
        let src = MapSource::new(&[("a.rs", a.as_str()), ("b.rs", b.as_str())], &[]);
        let (callers, capped) = callers_for(
            &["parse"],
            &index,
            &src,
            &LanguageRegistry::new(),
            &mut Vec::new(),
        );
        assert!(
            callers.is_empty(),
            "the scan stops at the bound: {callers:?}"
        );
        assert_eq!(capped.per_name, vec!["parse".to_string()], "and says so");
    }

    #[test]
    fn the_cap_keeps_the_first_by_file() {
        // MAGI FIX (MS3 §6 loop12): the cap keeps the K smallest `(file, line)` whatever the
        // discovery order (`select_nth_unstable` puts the K smallest before index K). The listing
        // gives the late file first; the early file's one caller must still survive the cut.
        use crate::review::test_support::MapSource;
        let d = "fn parse() {}\n".to_string();
        let a = "fn early() {\n    parse();\n}\n".to_string();
        let z = "// parse\n".repeat(MAX_CALLER_SCAN_PER_NAME + 10);
        let files = vec![
            ("z.rs".to_string(), z.clone()),
            ("d.rs".to_string(), d.clone()),
            ("a.rs".to_string(), a.clone()),
        ];
        let index = LexicalIndex::build(&files, &LanguageRegistry::new());
        let src = MapSource::new(
            &[
                ("a.rs", a.as_str()),
                ("d.rs", d.as_str()),
                ("z.rs", z.as_str()),
            ],
            &[],
        );
        let (callers, capped) = callers_for(
            &["parse"],
            &index,
            &src,
            &LanguageRegistry::new(),
            &mut Vec::new(),
        );
        assert_eq!(capped.per_name, vec!["parse".to_string()]);
        let labels: Vec<_> = callers.iter().map(|c| c.label.as_str()).collect();
        assert_eq!(labels, vec!["early"], "the first file's caller survives");
    }

    #[test]
    fn first_k_keeps_the_smallest_in_order_and_says_when_more_were_given() {
        // [MAGI S6 MS3 loop17, R2 Melchior] The caller scan's bounded selection: the k smallest
        // whatever the input order, sorted, and whether the input held more than k.
        let (kept, more) = first_k((0..100u32).rev(), 3);
        assert_eq!((kept, more), (vec![0, 1, 2], true));
        let (kept, more) = first_k([5u32, 1, 9, 1].into_iter(), 3);
        assert_eq!((kept, more), (vec![1, 1, 5], true));
        let (kept, more) = first_k([2u32, 1].into_iter(), 3);
        assert_eq!(
            (kept, more),
            (vec![1, 2], false),
            "fewer than k: all, no cut"
        );
        let (kept, more) = first_k(std::iter::empty::<u32>(), 3);
        assert_eq!((kept, more), (Vec::new(), false));
        let (kept, more) = first_k([7u32].into_iter(), 0);
        assert_eq!(
            (kept, more),
            (Vec::new(), true),
            "k = 0 keeps nothing and says so"
        );
    }

    #[test]
    fn callers_stop_at_their_batch_total_declared_once() {
        // [MAGI S6 MS3 loop17, R2 Caspar] Definitions got a batch total and callers only a
        // per-name cap: a whole-file batch over a hub module brought every name's callers, each
        // a whole unit.
        use crate::review::test_support::MapSource;
        let names: Vec<String> = (0..MAX_CALLERS_TOTAL + 5)
            .map(|i| format!("n{i}"))
            .collect();
        let d: String = names.iter().map(|n| format!("fn {n}() {{}}\n")).collect();
        let c: String = names
            .iter()
            .map(|n| format!("fn call_{n}() {{\n    {n}();\n}}\n\n"))
            .collect();
        let files = vec![
            ("c.rs".to_string(), c.clone()),
            ("d.rs".to_string(), d.clone()),
        ];
        let index = LexicalIndex::build(&files, &LanguageRegistry::new());
        let src = MapSource::new(&[("c.rs", c.as_str()), ("d.rs", d.as_str())], &[]);
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let (callers, cuts) = callers_for(
            &refs,
            &index,
            &src,
            &LanguageRegistry::new(),
            &mut Vec::new(),
        );
        assert_eq!(callers.len(), MAX_CALLERS_TOTAL);
        assert!(
            cuts.per_name.is_empty(),
            "one caller per name is never over the per-name cap: {:?}",
            cuts.per_name
        );
        assert!(cuts.total, "the batch total is declared, once");
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
        let (callers, capped) = callers_for(
            &["m"],
            &index,
            &src,
            &LanguageRegistry::new(),
            &mut Vec::new(),
        );
        assert_eq!(capped, RetrievalCuts::default());
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
