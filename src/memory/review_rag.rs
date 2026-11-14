// Author: Julian Bolivar
// Version: 0.25.0
// Date: 2026-11-08

//! The retrieved context of a `code-review` batch, from the encrypted stores and the embedder
//! (REQ-RV07, RV08, RV19, RV21, RV22).
//!
//! Per review: [`prepare_docs`] lists the reviewed revision's documentation sources, indexes the
//! ones the store lacks and degrades (documentation left out, declared once) when it cannot;
//! [`StoreRag`] then gives each batch its containing units, definitions, callers, documentation
//! and validated history, assembled to the batch budget; after the report, [`record_review`]
//! reclassifies and records the history and [`collect_garbage`] drops the documentation nothing
//! reaches any more.
//!
//! # Lock discipline
//!
//! The query is embedded before any store is touched, and every store call is a short
//! synchronous one: no database lock is ever held across an `.await`.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rusqlite::Connection;
use tokio::time::Instant;

use magi_rs::review::batching::{BatchItem, Companion, ReviewBatch};
use magi_rs::review::docs::{
    bounded_capped, doc_index_id, live_doc_sources, netted_chunks, revision_doc_sources,
    worktree_key, yields_doc_chunks, DocChunk, DocSource,
};
use magi_rs::review::error::ReviewError;
use magi_rs::review::history::{
    on_heads_line, reclassify_then_record, HistoryEntry, MAX_RECLASSIFY_PER_REVIEW,
};

/// What the history is reclassified and served against: everything after the reviewed commit,
/// up to now. One definition for the runner's reclassification and the context read.
pub const HISTORY_HEAD: &str = "HEAD";
use magi_rs::review::index::{definition_units, LexicalIndex};
use magi_rs::review::languages::LanguageRegistry;
use magi_rs::review::pipeline::{escape_fence, RagSource, ReviewMaterial, CONTEXT_HEADING};
use magi_rs::review::rag::{
    assemble, callers_for, charged_tokens, containing_units_cached, declare_index, declare_lexical,
    definitions_for, doc_search_query, item_cost, Assembled, ContextItem, ContextKind,
    IndexDeclarations, UnitCache, CONTEXT_ITEM_FRAME_TOKENS, MAX_CALLERS_PER_NAME,
    MAX_CALLERS_TOTAL, MAX_DEFINITIONS_PER_NAME, MAX_DEFINITIONS_TOTAL, UNREADABLE_CONTEXT,
};
use magi_rs::review::secrets::{
    display_path, find_secret, holds_key_block, mask_secrets, masked_pattern, outward_text,
};
use magi_rs::review::source::{ContentSource, GitCommand};
use magi_rs::review::window::estimate_tokens;
use magi_rs::vault::MaskedDek;

use crate::memory::embedding::EmbeddingProvider;
use crate::memory::error::MemoryError;
use crate::memory::review_docs::{DocLoss, DocVectors, ReviewDocStore, EMBED_BATCH_ITEMS};
use crate::memory::review_history::{ReviewHistoryStore, MAX_HISTORY_PER_BATCH};

/// Wall-clock bound on one reclassification pass of the review history, checked between
/// entries. Each entry's git calls carry MS2's own `GIT_CALL_TIMEOUT_SECS`, so a pass lasts at
/// most this plus the calls of the entry in flight, instead of up to
/// `MAX_RECLASSIFY_PER_REVIEW` hung calls.
pub const RECLASSIFY_BUDGET_SECS: u64 = 60;

/// Bound on embedding one batch's documentation query (one short text): a local embedder
/// answers in well under a second, so 30 s only fires on a hung or unreachable one.
pub const EMBED_QUERY_TIMEOUT_SECS: u64 = 30;

/// Bound on the one embedding [`ReviewMemory::doc_store`] sends to learn the dimension of an
/// embedder that autodetects it: one short text, like a batch's query.
pub const DIMENSION_PROBE_TIMEOUT_SECS: u64 = EMBED_QUERY_TIMEOUT_SECS;

/// Documentation chunks retrieved per batch before `assemble` trims to the budget.
pub const DOC_SEARCH_K: usize = 8;

/// What the report says when history rows that did not decode were skipped.
const CORRUPT_HISTORY: &str = "history rows that could not be decoded were skipped";

/// Why a listed working-tree documentation source gave no bytes: it was gone when read.
const VANISHED: &str = "gone since it was listed";

/// Why no query was sent when [`doc_search_query`] withholds it (a key block, RV22).
const QUERY_WITHHELD: &str = "the query prefix holds a private-key block: no query is sent";

/// Why no history entry is served when the review has no git to check it against `HEAD`.
const HISTORY_NO_GIT: &str = "history not served: there is no git to check it against HEAD";

/// Why a batch's documentation and history were not searched when its own text holds a private-key
/// block (RV22): declared for that batch, apart from a missing or down embedder.
const BATCH_QUERY_WITHHELD: &str =
    "this batch's text holds a private-key block: no query is sent for it";

/// What embedding one batch's query came to.
enum QueryVector {
    /// The query's vector.
    Vector(Vec<f32>),
    /// Withheld for this batch alone (a private-key block in its text).
    Withheld,
    /// The embedder is down or did not answer, for the rest of the review.
    Unavailable,
}

/// Bound on the whole documentation preparation of one review (listing, reads and every
/// reindex), checked between steps and sources; it runs before the review clock (RV06).
pub const REINDEX_TIMEOUT_SECS: u64 = 120;

/// Run-log target of this module's entries.
const LOG_TARGET: &str = "magi_rs::review";

/// The `trimmed` label of a documentation or history source that is unavailable for the review.
const INDEX_UNAVAILABLE: &str = "index unavailable";

/// The `trimmed` label of documentation left out because the indexing deadline
/// ([`REINDEX_TIMEOUT_SECS`] for the whole preparation) passed while sources were still being
/// indexed: each group written is kept (`reindex` commits per group), so a later review resumes.
const INDEX_STILL_BUILDING: &str = "index still being built: the indexing deadline passed; what \
                                    was indexed is kept and a later review resumes it";

/// The `trimmed` label of documentation or history left out of a batch because nothing more
/// could fit in its budget (its query is then never embedded).
const NO_ROOM: &str = "no room left in the batch";

/// The cause declared for documentation and history when the configured embedder failed or did
/// not answer the review's first query.
const EMBEDDER_DOWN: &str = "the embedder did not answer";

/// The fixed run-log line of a reindex that wiped an index of another identity.
const REBUILT_LINE: &str = "documentation index rebuilt: embedding model or chunking changed";

/// What [`ReviewRag::context_for`] answers: the source must be bound to a review first.
const UNBOUND_SOURCE: &str = "the review context was read before it was bound to a review";

/// The text embedded once to learn the dimension of an embedder that autodetects it.
const DIMENSION_PROBE: &str = "dimension";

/// The context source of one review, over the stores and the review's own content.
///
/// MAGI FIX (MS3 §6 loop14), declared limitations. (1) The struct carries many fields (stores,
/// git, per-review memos, once-per-review flags): they are one review's state and all live and die
/// together, so splitting them would only move the same borrows into sub-structs; this is a style
/// limit, not a behaviour one. (2) The stores are synchronous SQLite behind short locks and are
/// never held across an `.await`; the query timeout ([`EMBED_QUERY_TIMEOUT_SECS`]) is a tokio
/// timeout, so on a single-threaded runtime a store call delays the timer by its own (short,
/// `busy_timeout`-bounded) duration and never by more. (3) Poisoned locks: see `relock`.
///
/// # Examples
///
/// ```ignore
/// let rag = StoreRag { docs: Some(&docs), history: Some(&history), /* ... */ }.with_prep(prep);
/// let assembled = rag.context_for(&batch).await?;
/// ```
pub struct StoreRag<'a> {
    /// The documentation index; `None` with `--no-memory`.
    pub docs: Option<&'a ReviewDocStore>,
    /// The review history; `None` with `--no-memory`.
    pub history: Option<&'a ReviewHistoryStore>,
    /// The review's lexical index.
    pub index: &'a LexicalIndex,
    /// The review's content source: the review's `CachedSource`, whose sides are held masked,
    /// so every unit, definition and caller is built masked
    /// (`a_secret_outside_the_changed_lines_reaches_every_code_context_item_masked`).
    pub sources: &'a dyn ContentSource,
    /// The language extractors.
    pub registry: &'a LanguageRegistry,
    /// The `[embedding]` embedder, if any.
    pub embedder: Option<&'a dyn EmbeddingProvider>,
    /// The batch limit in tokens (the context heading is taken off it here).
    pub limit_tokens: usize,
    /// What indexing left out, declared by the first batch.
    pub doc_index: IndexDeclarations,
    /// Whether the index declarations were already reported.
    pub declared: AtomicBool,
    /// The documentation index could not be completed for this revision (it is not searched).
    pub docs_unavailable: bool,
    /// The query embedder failed once: no later batch calls it.
    pub embedder_down: AtomicBool,
    /// A documentation read failed once: no later batch searches.
    pub docs_failed: AtomicBool,
    /// Documentation rows or sources that vanished mid-review were declared once; unlike
    /// `docs_failed` it leaves the rest of the documentation serving.
    pub docs_shrunk: AtomicBool,
    /// A history read failed once: no later batch reads it.
    pub history_failed: AtomicBool,
    /// Skipped history rows (they did not decode) were declared once.
    pub corrupt_declared: AtomicBool,
    /// The store's corrupt-row count before this review's first history read: the store counts
    /// for its lifetime, the review declares only what it skipped itself.
    pub corrupt_baseline: std::sync::OnceLock<usize>,
    /// The absence of a git, which leaves history unserved, was declared once.
    pub history_no_git_declared: AtomicBool,
    /// The reviewed revision's documentation sources: `search` sees only these (RV21).
    pub doc_sources: BTreeSet<DocSource>,
    /// Documentation paths left out for a secret, as `display_path`, returned by `skipped`: a
    /// path that itself matches a secret pattern is written `<path redacted: <pattern>>`, never
    /// raw (pinned by `a_working_tree_review_keys_its_sources_by_their_bytes`, which asserts the
    /// redacted form and that no byte of the key is in the list).
    pub withheld_sources: Vec<String>,
    /// Documentation paths indexed with a secret masked **by this review**, with the first
    /// pattern masked in each, returned by `masked`.
    ///
    /// A source masked when an earlier review indexed it is not in this list: the declaration
    /// RV22 asks for is of what is sent, and it is made when its chunk is served. Every chunk
    /// served carries its `<redacted: …>` mask (stored masked), and the pipeline declares the
    /// file of each documentation or history item that carries one (`masked_store_item`,
    /// `masked_documentation_and_history_are_declared_as_masked` in `review::pipeline`); a
    /// source not served in this review sends nothing, so it needs no declaration
    /// (`a_source_masked_by_an_earlier_review_is_served_with_its_mask`).
    pub masked_sources: Vec<(String, &'static str)>,
    /// Milliseconds the documentation preparation took, returned by `index_ms`.
    pub index_ms: u64,
    /// The validated handle on `doc_sources`, made by the first batch that searches and scanned
    /// by every later one in pages (no vector is held decrypted: the peak is one page plus the
    /// best rows, so a large documentation set never grows the review's memory).
    pub doc_vectors: Mutex<Option<Arc<DocVectors>>>,
    /// The last file's units on each side, so consecutive hunks of one file (planning and each
    /// batch walk them in order) extract it once.
    pub units: UnitCache,
    /// The git each served history entry is checked against (`served_as_validated` on `HEAD`);
    /// `None` serves no history entry, since none can be checked.
    pub history_git: Option<&'a dyn GitCommand>,
    /// `on_heads_line` by reviewed commit, so entries sharing one cost one git call; it reads the
    /// commit alone, and the status is checked per entry (`memo_shares_no_verdict`).
    pub on_line: Mutex<HashMap<String, bool>>,
    /// The history entries already counted as withheld, id to the [`display_path`] of their file,
    /// so an entry two batches meet counts once and its file is declared `secret` (`skipped`).
    pub withheld_history: Mutex<BTreeMap<String, String>>,
}

/// What the documentation preparation of one review produced.
#[derive(Debug, Default, Clone)]
pub struct DocPrep {
    /// The reviewed revision's documentation sources.
    pub doc_sources: BTreeSet<DocSource>,
    /// What indexing left out: sources withheld for a secret, and whether it was cut.
    pub doc_index: IndexDeclarations,
    /// The index could not be completed: the review serves no documentation (B).
    pub docs_unavailable: bool,
    /// Milliseconds the preparation took.
    pub index_ms: u64,
    /// Sources left out for a secret, as `display_path` (a secret-shaped path is redacted, see
    /// [`StoreRag::withheld_sources`]).
    pub withheld_sources: Vec<String>,
    /// Sources indexed with a secret masked, as `display_path`, each with the first pattern
    /// masked in it.
    pub masked_sources: Vec<(String, &'static str)>,
}

impl DocPrep {
    /// The working-tree set [`collect_garbage`] may judge `wt:` sources by: `doc_sources` for a
    /// working-tree review (`rev` is `None`) whose listing completed; `None` otherwise, so a
    /// degraded, partial listing never deletes a source it did not reach (RV19,
    /// `degraded_listing_keeps_worktree`).
    pub fn gc_worktree(&self, rev: Option<&str>) -> Option<BTreeSet<DocSource>> {
        (rev.is_none() && !self.docs_unavailable).then(|| self.doc_sources.clone())
    }
}

/// What one [`record_review`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordStats {
    /// Due entries checked this review.
    pub reclassified: usize,
    /// New entries recorded.
    pub recorded: usize,
    /// Due entries left for a later review (the total minus those checked now; the total is
    /// counted up to `MAX_COUNTED_HISTORY`, so a larger history reads as at least that).
    pub backlog: usize,
}

/// `item` with its text escaped for its frame, so what is charged is what is written.
fn escaped(item: ContextItem) -> ContextItem {
    ContextItem {
        text: escape_fence(&item.text),
        ..item
    }
}

/// Locks `m`; a poisoned lock is recovered explicitly: a WARN, the poison cleared, the data used.
///
/// Every lock of this module guards a cache or a set written in one step (a memo entry, an
/// `Option` replaced whole, a set insert), so a panic elsewhere never leaves it half-written;
/// the next review starts with fresh ones (`poison_is_recovered_and_cleared`).
// MAGI FIX (MS3 §6 loop10): a poisoned lock was recovered in silence.
fn relock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match m.lock() {
        Ok(g) => g,
        Err(poisoned) => {
            tracing::event!(
                target: LOG_TARGET,
                tracing::Level::WARN,
                "a code-review RAG lock was poisoned by a panic; its data is used as left"
            );
            m.clear_poison();
            poisoned.into_inner()
        }
    }
}

/// The file of a batch item.
fn item_file(item: &BatchItem) -> &str {
    match item {
        BatchItem::Hunk { file, .. }
        | BatchItem::Unit { file, .. }
        | BatchItem::WholeFile { file, .. } => file,
    }
}

/// The text of a batch item.
fn item_text(item: &BatchItem) -> &str {
    match item {
        BatchItem::Hunk { text, .. }
        | BatchItem::Unit { text, .. }
        | BatchItem::WholeFile { text, .. } => text,
    }
}

/// The batch's raw code, one item per line block. O(bytes of the batch).
fn batch_text(batch: &ReviewBatch) -> String {
    batch
        .items
        .iter()
        .map(item_text)
        .collect::<Vec<_>>()
        .join("\n")
}

/// The distinct files of a batch, in order.
fn batch_files(batch: &ReviewBatch) -> Vec<String> {
    let mut seen = BTreeSet::new();
    batch
        .items
        .iter()
        .map(item_file)
        .filter(|f| seen.insert(*f))
        .map(str::to_string)
        .collect()
}

/// The names of the changed units: each containing unit's, each unit item's, and each named
/// definition of a whole-file item, its methods included ([`definition_units`]; all of it is under
/// review; `callers_for` caps what this brings). O(bytes of the whole-file items).
// MAGI FIX (MS3 §6 loop10): a whole file named none, so it got no callers (RV07(2)).
fn changed_names(
    batch: &ReviewBatch,
    units: &[ContextItem],
    registry: &LanguageRegistry,
) -> Vec<String> {
    let whole: Vec<String> = batch
        .items
        .iter()
        .filter_map(|i| match i {
            BatchItem::WholeFile { file, text, .. } => registry.get(file).map(|x| (x, file, text)),
            _ => None,
        })
        // MAGI FIX (MS3 §6 loop11): the methods too, not only the top-level units.
        .flat_map(|(x, file, text)| definition_units(x, file, text))
        .map(|u| u.name)
        .filter(|n| !n.is_empty())
        .collect();
    let mut seen = BTreeSet::new();
    units
        .iter()
        .map(|u| u.label.as_str())
        .chain(batch.items.iter().filter_map(|i| match i {
            BatchItem::Unit { name, .. } => Some(name.as_str()),
            _ => None,
        }))
        .chain(whole.iter().map(String::as_str))
        .filter(|n| seen.insert(*n))
        .map(str::to_string)
        .collect()
}

/// A documentation chunk as a context item, labelled by its section (its path without one).
fn doc_item(chunk: DocChunk) -> ContextItem {
    let label = if chunk.section.is_empty() {
        chunk.path.clone()
    } else {
        chunk.section
    };
    ContextItem {
        kind: ContextKind::Documentation,
        file: chunk.path,
        label,
        text: escape_fence(&chunk.text),
        candidate: false,
        lines: None,
        old_side: false,
    }
}

/// A validated history entry as a context item: its title, its status and its detail.
fn history_item(e: HistoryEntry) -> ContextItem {
    let text = format!("{} ({})\n{}", e.title, e.status.as_str(), e.detail);
    ContextItem {
        kind: ContextKind::History,
        // MAGI FIX (MS3 §6 loop8): its text is no span of the file, so it has no `lines`.
        lines: None,
        label: e.title,
        file: e.file,
        text: escape_fence(&text),
        candidate: false,
        old_side: false,
    }
}

/// A duration in whole milliseconds, saturating.
fn millis(d: Duration) -> u64 {
    u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
}

impl StoreRag<'_> {
    /// Moves a preparation into the source: its sources, declarations, degradation, duration and
    /// withheld paths.
    ///
    /// # Arguments
    ///
    /// * `prep` - What [`prepare_docs`] produced for this review.
    ///
    /// # Returns
    ///
    /// The source, ready for the review.
    pub fn with_prep(mut self, prep: DocPrep) -> Self {
        self.doc_sources = prep.doc_sources;
        self.doc_index = prep.doc_index;
        self.docs_unavailable = prep.docs_unavailable;
        self.index_ms = prep.index_ms;
        self.withheld_sources = prep.withheld_sources;
        self.masked_sources = prep.masked_sources;
        self
    }

    /// The query's vector, embedded under [`EMBED_QUERY_TIMEOUT_SECS`]; [`QueryVector::Unavailable`]
    /// when the embedder fails or does not answer, which marks it down for the rest of the review
    /// with one WARN; [`QueryVector::Withheld`] when this batch's own text holds a private-key
    /// block, which sends nothing and leaves the embedder as it was.
    async fn query_vector(&self, embedder: &dyn EmbeddingProvider, text: &str) -> QueryVector {
        if self.embedder_down.load(Ordering::SeqCst) {
            return QueryVector::Unavailable;
        }
        let bound = Duration::from_secs(EMBED_QUERY_TIMEOUT_SECS);
        // MAGI FIX (MS3 §6 loop11): a key block in the query prefix is withheld, never sent.
        // MAGI FIX (MS3 §6 loop12): a key block in this BATCH's text withholds this query only;
        // the embedder is not down and the next batch is served (the filter omits key-block
        // files, so this is a backstop).
        // MAGI FIX (MS3 §6 loop16): that withhold is declared for the batch, with its own cause.
        let prefix_blocked = holds_key_block(embedder.query_prefix());
        if !prefix_blocked && holds_key_block(text) {
            return QueryVector::Withheld;
        }
        let cause = match doc_search_query(embedder.query_prefix(), text) {
            // MAGI FIX (MS3 §6 loop16): a block formed only by prefix and text joined is this
            // batch's alone, not an embedder that "did not answer" for the rest of the review.
            None if !prefix_blocked => return QueryVector::Withheld,
            None => QUERY_WITHHELD.to_string(),
            Some(query) => match tokio::time::timeout(bound, embedder.embed(&[query])).await {
                // MAGI FIX (MS3 §6 loop2): one query, one vector; any other count is not guessed at.
                Ok(Ok(mut vectors)) => match (vectors.pop(), vectors.is_empty()) {
                    (Some(v), true) if !v.is_empty() => return QueryVector::Vector(v),
                    (None, _) => "the embedder answered no vector".to_string(),
                    _ => "the embedder did not answer one vector for the query".to_string(),
                },
                Ok(Err(e)) => outward_text(&e.to_string()),
                Err(_) => format!("no answer within {EMBED_QUERY_TIMEOUT_SECS} s"),
            },
        };
        if !self.embedder_down.swap(true, Ordering::SeqCst) {
            tracing::event!(
                target: LOG_TARGET,
                tracing::Level::WARN,
                error = cause.as_str(),
                "the embedder did not answer: documentation and history are left out of this review"
            );
        }
        QueryVector::Unavailable
    }

    /// The documentation chunks closest to `vector`, among the revision's own sources only.
    /// A read error leaves documentation out for the rest of the review, declared once.
    fn documentation(
        &self,
        vector: &[f32],
        declared: &mut Vec<(ContextKind, String)>,
    ) -> Vec<ContextItem> {
        let Some(docs) = self.docs else {
            return Vec::new();
        };
        if self.docs_unavailable || self.docs_failed.load(Ordering::SeqCst) {
            return Vec::new();
        }
        match self.search_docs(docs, vector) {
            Ok((hits, loss)) => {
                // MAGI FIX (MS3 §6 loop15): rows deleted since the vectors were loaded (another
                // process's clean-up) are declared once; the surviving hits still serve.
                // MAGI FIX (MS3 §6 loop16): the loss touched those rows or sources only; the rest
                // keeps serving (it switched the whole layer off for the review).
                // MAGI FIX (MS3 §6 loop17, R1): the review-level loss (sources and rows gone
                // under it) and the per-batch one (hits missed) are logged apart, never summed.
                if loss.any() && !self.docs_shrunk.swap(true, Ordering::SeqCst) {
                    tracing::event!(
                        target: LOG_TARGET,
                        tracing::Level::WARN,
                        sources_gone = loss.sources_gone,
                        rows_gone = loss.rows_gone,
                        hits_missed = loss.hits_missed,
                        "documentation rows vanished during the review: the index shrank under it"
                    );
                    declared.push((ContextKind::Documentation, INDEX_UNAVAILABLE.to_string()));
                }
                hits.into_iter().map(|(chunk, _)| doc_item(chunk)).collect()
            }
            Err(e) => self.docs_unreadable(&e, declared),
        }
    }

    /// [`Self::documentation`] after checking that the query's model is the index's, not only its
    /// dimension: a vector from another model of the same size would be scored against vectors it
    /// does not belong to. A mismatch leaves documentation out for the rest of the review,
    /// declared once, like a read error.
    // MAGI FIX (MS3 §6 loop17, R1): the identity check guarded only the reindex path.
    fn documentation_of(
        &self,
        embedder: &dyn EmbeddingProvider,
        vector: &[f32],
        declared: &mut Vec<(ContextKind, String)>,
    ) -> Vec<ContextItem> {
        let Some(docs) = self.docs else {
            return Vec::new();
        };
        match docs.require_identity(embedder.model_id(), vector.len()) {
            Ok(()) => self.documentation(vector, declared),
            Err(_) if self.docs_unavailable || self.docs_failed.load(Ordering::SeqCst) => {
                Vec::new()
            }
            Err(e) => self.docs_unreadable(&e, declared),
        }
    }

    /// Leaves documentation out for the rest of the review: one WARN and one declaration.
    fn docs_unreadable(
        &self,
        e: &MemoryError,
        declared: &mut Vec<(ContextKind, String)>,
    ) -> Vec<ContextItem> {
        if !self.docs_failed.swap(true, Ordering::SeqCst) {
            let error = outward_text(&e.to_string());
            tracing::event!(
                target: LOG_TARGET,
                tracing::Level::WARN,
                error = error.as_str(),
                "the documentation index could not be read: documentation is left out"
            );
            declared.push((ContextKind::Documentation, INDEX_UNAVAILABLE.to_string()));
        }
        Vec::new()
    }

    /// The [`DOC_SEARCH_K`] chunks of the revision's sources closest to `query`, scored against
    /// [`Self::doc_vectors`]: the first call validates the sources, every call scans their rows
    /// in pages (the connection lock is held only for each page's raw read). The cache lock is
    /// released before any store call.
    ///
    /// Declared limit (MS3 §6 loop12): the store work is synchronous SQLite behind a mutex, so
    /// the `tokio::time::timeout` around `reindex` cannot preempt it on a single-threaded
    /// runtime; the calls are short and each is bounded by the busy timeout, and production
    /// runs on the multi-thread runtime. `spawn_blocking` would add a thread hop per call for no
    /// bound the store does not already give. A poisoned lock (`relock`) is recovered with one
    /// WARN under the `magi_rs::review` target; the guarded values are replaced whole, so the
    /// recovery is safe and the log is the signal (panics are reported by the process's own hook).
    ///
    /// # Errors
    ///
    /// [`MemoryError`] of the store.
    fn search_docs(
        &self,
        docs: &ReviewDocStore,
        query: &[f32],
    ) -> Result<(Vec<(DocChunk, f32)>, DocLoss), MemoryError> {
        let cached = relock(&self.doc_vectors).clone();
        let vectors = match cached {
            Some(vectors) if vectors.dim() == query.len() => vectors,
            _ => {
                let loaded = Arc::new(docs.load_vectors(query.len(), &self.doc_sources)?);
                *relock(&self.doc_vectors) = Some(Arc::clone(&loaded));
                loaded
            }
        };
        docs.search_loaded_counting(&vectors, query, DOC_SEARCH_K)
    }

    /// Whether `entry` may be served: `served_as_validated` on `HEAD` through
    /// [`Self::history_git`], once per reviewed commit (memoized in [`Self::on_line`]); without a
    /// git, nothing is served. O(1) after the first call for a commit.
    // MAGI FIX (MS3 §6 loop11): the memo held `served_as_validated`'s whole verdict, keyed by
    // commit alone; it holds only `on_heads_line`, which reads nothing but the commit, and the
    // status is checked per entry, so the invariant holds by construction in every build.
    fn served_on_heads_line(&self, entry: &HistoryEntry) -> bool {
        let Some(git) = self.history_git else {
            return false;
        };
        entry.status.is_validated() && self.commit_on_line(git, &entry.reviewed_commit)
    }

    /// [`on_heads_line`] of `commit` on `HEAD`, memoized in [`Self::on_line`]; `false` without a
    /// git. O(1) after the first call for a commit.
    fn commit_on_line(&self, git: &dyn GitCommand, commit: &str) -> bool {
        let mut memo = relock(&self.on_line);
        *memo
            .entry(commit.to_string())
            .or_insert_with(|| on_heads_line(commit, HISTORY_HEAD, git))
    }

    /// Whether the history entry `id`, on `file`, is withheld for the first time in this review.
    /// O(log n).
    fn first_withheld(&self, id: &str, file: &str) -> bool {
        relock(&self.withheld_history)
            .insert(id.to_string(), display_path(file))
            .is_none()
    }

    /// The validated history on `files`, each one-line secret of a title or detail masked
    /// ([`mask_secrets`], RV22 amended); an entry whose title or detail holds a private-key block,
    /// or whose file path quotes a secret (a path is not maskable and still names its file), counts
    /// in `withheld` instead, its file declared `secret` (RV22). A read error leaves history out for the
    /// rest of the review, declared once.
    // MAGI FIX (MS3 §6 loop1): an entry quoting a one-line secret was withheld, not masked.
    fn history_items(
        &self,
        files: &[String],
        declared: &mut Vec<(ContextKind, String)>,
        withheld: &mut usize,
    ) -> Vec<ContextItem> {
        let Some(history) = self.history else {
            return Vec::new();
        };
        if self.history_failed.load(Ordering::SeqCst) {
            return Vec::new();
        }
        // MAGI FIX (MS3 §6 loop17, R2 Melchior/Balthasar): with no git nothing read could be
        // served, so the store is not read (the cause is declared by `declare_no_git`).
        if self.history_git.is_none() {
            return Vec::new();
        }
        let on_line = |commit: &str| {
            self.history_git
                .is_some_and(|g| self.commit_on_line(g, commit))
        };
        let baseline = *self.corrupt_baseline.get_or_init(|| history.corrupt_rows());
        // MAGI FIX (MS3 §6 loop11): the ancestry filter runs before the cap, in the store.
        match history.validated_on_line(files, &on_line) {
            Ok((entries, cut)) => {
                // MAGI FIX (MS3 §6 loop11): a row that did not decode is skipped, declared once.
                // MAGI FIX (MS3 §6 loop16): the store counts for its lifetime (the TUI keeps it
                // across reviews); this review declares the rows it skipped itself.
                if history.corrupt_rows() > baseline
                    && !self.corrupt_declared.swap(true, Ordering::SeqCst)
                {
                    declared.push((ContextKind::History, CORRUPT_HISTORY.to_string()));
                }
                // MAGI FIX (MS3 §6 loop4): the read is bounded now; what it cut is declared.
                if cut {
                    declared.push((
                        ContextKind::History,
                        format!(
                            "more than MAX_HISTORY_PER_BATCH ({MAX_HISTORY_PER_BATCH}) validated \
                             entries on these files: the newest kept"
                        ),
                    ));
                }
                entries
                    .into_iter()
                    // MAGI FIX (MS3 §6 loop7): an entry validated off HEAD's line (another
                    // branch) was served; each one is checked against HEAD first.
                    .filter(|e| self.served_on_heads_line(e))
                    .filter_map(|e| {
                        // MAGI FIX (MS3 §6 loop10): a secret-shaped file path was left raw.
                        if holds_key_block(&e.title)
                            || holds_key_block(&e.detail)
                            || find_secret(&e.file).is_some()
                        {
                            // MAGI FIX (MS3 §6 loop9): counted again by every batch on its file.
                            if self.first_withheld(&e.id, &e.file) {
                                *withheld = withheld.saturating_add(1);
                            }
                            None
                        } else {
                            Some(history_item(HistoryEntry {
                                title: mask_secrets(&e.title),
                                detail: mask_secrets(&e.detail),
                                ..e
                            }))
                        }
                    })
                    .collect()
            }
            Err(e) => {
                if !self.history_failed.swap(true, Ordering::SeqCst) {
                    let error = outward_text(&e.to_string());
                    tracing::event!(
                        target: LOG_TARGET,
                        tracing::Level::WARN,
                        error = error.as_str(),
                        "the review history could not be read: history is left out"
                    );
                    declared.push((ContextKind::History, INDEX_UNAVAILABLE.to_string()));
                }
                Vec::new()
            }
        }
    }
}

impl StoreRag<'_> {
    /// Documentation and history for `batch`, searched with the embedded query of `text`; a down
    /// embedder gives neither and is declared once per review (its error is in the run log).
    async fn retrieve(
        &self,
        embedder: &dyn EmbeddingProvider,
        batch: &ReviewBatch,
        text: &str,
        declared: &mut Vec<(ContextKind, String)>,
        withheld: &mut usize,
    ) -> (Option<Vec<ContextItem>>, Option<Vec<ContextItem>>) {
        let was_down = self.embedder_down.load(Ordering::SeqCst);
        let vector = match self.query_vector(embedder, text).await {
            QueryVector::Vector(v) => v,
            // MAGI FIX (MS3 §6 loop16): this batch only, with its own cause, and neither layer
            // is a missing embedder (`without_embedder` stays false).
            QueryVector::Withheld => {
                for (on, kind) in [
                    (self.docs.is_some(), ContextKind::Documentation),
                    (self.history.is_some(), ContextKind::History),
                ] {
                    if on {
                        declared.push((kind, BATCH_QUERY_WITHHELD.to_string()));
                    }
                }
                return (Some(Vec::new()), Some(Vec::new()));
            }
            QueryVector::Unavailable => {
                // MAGI FIX (MS3 §6 loop2): a down embedder is said apart from a missing one.
                // MAGI FIX (MS3 §6 loop12): only when it is down now (a withheld batch is not).
                if !was_down && self.embedder_down.load(Ordering::SeqCst) {
                    let layers = [
                        (self.docs.is_some(), ContextKind::Documentation),
                        (self.history.is_some(), ContextKind::History),
                    ];
                    // MAGI FIX (MS3 §6 loop14): a prefix that holds a key block never reached the
                    // embedder, so it is not an embedder that "did not answer".
                    let cause = if holds_key_block(embedder.query_prefix()) {
                        QUERY_WITHHELD
                    } else {
                        EMBEDDER_DOWN
                    };
                    for (_, kind) in layers.into_iter().filter(|(on, _)| *on) {
                        declared.push((kind, cause.to_string()));
                    }
                }
                return (None, None);
            }
        };
        let docs = self.documentation_of(embedder, &vector, declared);
        (
            Some(docs),
            Some(self.history_items(&batch_files(batch), declared, withheld)),
        )
    }

    /// Whether neither documentation nor history can enter any later batch: each is absent,
    /// unavailable or failed for the rest of the review. A down embedder is not checked here:
    /// [`Self::query_vector`] answers `Unavailable` for it without a call, so the stores are what this
    /// adds. O(1).
    fn both_layers_down(&self) -> bool {
        let docs =
            self.docs.is_none() || self.docs_unavailable || self.docs_failed.load(Ordering::SeqCst);
        let history = self.history.is_none() || self.history_unservable();
        docs && history
    }

    /// Whether no history entry can be served for the rest of the review, for a cause declared
    /// once on its own: a failed read, or no git to check an entry against `HEAD`. O(1).
    fn history_unservable(&self) -> bool {
        self.history_failed.load(Ordering::SeqCst) || self.history_git.is_none()
    }

    /// Declares, once per review, that history is not served for want of a git; only when the
    /// review has a history and an embedder (without one, history is out for that cause).
    // MAGI FIX (MS3 §6 loop16): no git, no entry can be checked against HEAD, so none is
    // served; said once (R2 Caspar).
    // MAGI FIX (MS3 §6 loop17, R2): said here, whatever the batch's path, now that a no-git
    // review neither embeds a query for the history nor reads it.
    fn declare_no_git(&self, declared: &mut Vec<(ContextKind, String)>) {
        if self.history.is_some()
            && self.embedder.is_some()
            && self.history_git.is_none()
            && !self.history_no_git_declared.swap(true, Ordering::SeqCst)
        {
            declared.push((ContextKind::History, HISTORY_NO_GIT.to_string()));
        }
    }

    /// What a batch with no room left declares for documentation and history: `no room` for each
    /// layer this review has, unless that layer is already out for a cause declared once on its
    /// own (an unavailable or failed index, a down embedder, a failed history read). O(1).
    // MAGI FIX (MS3 §6 loop1): RV07 declares what was cut, this case included.
    // MAGI FIX (MS3 §6 loop2): an unavailable index has its own cause, never room.
    // MAGI FIX (MS3 §6 loop4): nor for what a down embedder or a failed history read already
    // left out, each declared once with its own cause.
    // MAGI FIX (MS3 §6 loop5): one helper, so the next rule changes one place.
    fn no_room_declarations(&self) -> Vec<(ContextKind, String)> {
        let embedder_down = self.embedder_down.load(Ordering::SeqCst);
        let docs_down =
            self.docs_unavailable || self.docs_failed.load(Ordering::SeqCst) || embedder_down;
        let history_down = self.history_unservable() || embedder_down;
        [
            // MAGI FIX (MS3 §6 loop12): a review that indexed no source has nothing to cut. The
            // history layer is not checked the same way: its emptiness is a store read per batch,
            // so its `no room` stays (the declaration says what the batch could not hold).
            (
                self.docs.is_some() && !docs_down && !self.doc_sources.is_empty(),
                ContextKind::Documentation,
            ),
            (
                self.history.is_some() && !history_down,
                ContextKind::History,
            ),
        ]
        .into_iter()
        .filter(|(declare, _)| *declare)
        .map(|(_, kind)| (kind, NO_ROOM.to_string()))
        .collect()
    }
}

#[async_trait::async_trait]
impl RagSource for StoreRag<'_> {
    /// The batch's context: containing units, definitions and callers from the review's own
    /// content, then documentation and history when there is a database and an embedder.
    ///
    /// The query is embedded, masked ([`doc_search_query`]), before any store is touched and only
    /// when an item could still enter (room of at least [`CONTEXT_ITEM_FRAME_TOKENS`], every
    /// item's floor; one that then does not fit is declared trimmed). A missing embedder is
    /// `without_embedder`, no database `without_memory`; a failing embedder or store degrades,
    /// declared once ("the embedder did not answer"), never fails. An unreadable definition or
    /// caller file leaves out its own items, declared. It is embedded even when the index cannot
    /// be searched: the history goes with the embedder (RV07), so it is also the embedder check.
    async fn context_for(&self, batch: &ReviewBatch) -> Result<Assembled, ReviewError> {
        let text = batch_text(batch);
        let units = self.units_for(batch)?;
        // MAGI FIX (MS3 §6 loop9): an unreadable file costs its own items, declared.
        let (mut unread_defs, mut unread_callers) = (Vec::new(), Vec::new());
        let (mut code, capped_defs) =
            definitions_for(&text, self.index, self.sources, &mut unread_defs);
        let names = changed_names(batch, &units, self.registry);
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let (callers, capped) = callers_for(
            &refs,
            self.index,
            self.sources,
            self.registry,
            &mut unread_callers,
        );
        code.extend(callers);
        let code: Vec<ContextItem> = code.into_iter().map(escaped).collect();
        let limit = self
            .limit_tokens
            .saturating_sub(estimate_tokens(CONTEXT_HEADING));
        let used = charged_tokens(batch.tokens, &units, &code);
        let room = limit.saturating_sub(used) >= CONTEXT_ITEM_FRAME_TOKENS;
        let memory = self.docs.is_some() || self.history.is_some();
        let mut declared = Vec::new();
        let mut withheld = 0usize;
        self.declare_no_git(&mut declared);
        let (docs, history) = match (memory, self.embedder) {
            (false, _) | (true, None) => (None, None),
            (true, Some(_)) if !room => {
                declared.extend(self.no_room_declarations());
                (Some(Vec::new()), Some(Vec::new()))
            }
            // MAGI FIX (MS3 §6 loop7): with both layers out (each declared once already), a query
            // could bring nothing in; it is not embedded.
            (true, Some(_)) if self.both_layers_down() => (Some(Vec::new()), Some(Vec::new())),
            (true, Some(embedder)) => {
                self.retrieve(embedder, batch, &text, &mut declared, &mut withheld)
                    .await
            }
        };
        let mut a = assemble(batch.tokens, limit, units, code, docs, history)?;
        if !memory {
            a.without_memory = true;
            a.without_embedder = false;
        }
        a.withheld = a.withheld.saturating_add(withheld);
        a.trimmed.extend(capped.per_name.into_iter().map(|name| {
            (
                ContextKind::Caller,
                format!("{name}: more than {MAX_CALLERS_PER_NAME} callers"),
            )
        }));
        // MAGI FIX (MS3 §6 loop17, R2 Caspar): callers had no batch total; reaching it is one
        // cause, declared once per batch.
        if capped.total {
            a.trimmed.push((
                ContextKind::Caller,
                format!(
                    "callers cut at MAX_CALLERS_TOTAL ({MAX_CALLERS_TOTAL}) for this batch: the \
                     later ones were left out"
                ),
            ));
        }
        a.trimmed
            .extend(capped_defs.per_name.into_iter().map(|name| {
                (
                    ContextKind::Definition,
                    format!("{name}: more than {MAX_DEFINITIONS_PER_NAME} definitions"),
                )
            }));
        // MAGI FIX (MS3 §6 loop17, R2): the total cut was declared as the per-name one, once
        // per later name; it is its own cause, declared once per batch.
        if capped_defs.total {
            a.trimmed.push((
                ContextKind::Definition,
                format!(
                    "definitions cut at MAX_DEFINITIONS_TOTAL ({MAX_DEFINITIONS_TOTAL}) for this \
                     batch: the later names were left out"
                ),
            ));
        }
        a.trimmed.extend(declared);
        let unread = (unread_defs
            .into_iter()
            .map(|f| (ContextKind::Definition, f)))
        .chain(unread_callers.into_iter().map(|f| (ContextKind::Caller, f)));
        a.trimmed
            .extend(unread.map(|(kind, f)| (kind, format!("{f}: {UNREADABLE_CONTEXT}"))));
        if !self.declared.swap(true, Ordering::SeqCst) {
            declare_index(&mut a, &self.doc_index);
            declare_lexical(&mut a, self.index);
            // MAGI FIX (MS3 §6 loop7): a store that was not opened, with an embedder to search
            // it, left documentation out undeclared unless the caller also set the flag.
            let not_opened = self.docs.is_none() && self.embedder.is_some();
            if memory && (self.docs_unavailable || not_opened) {
                // MAGI FIX (MS3 §6 loop17, R2 Caspar): a cold index cut by its deadline read as
                // a down one.
                let label = if self.doc_index.still_indexing && !not_opened {
                    INDEX_STILL_BUILDING
                } else {
                    INDEX_UNAVAILABLE
                };
                a.trimmed
                    .push((ContextKind::Documentation, label.to_string()));
            }
        }
        Ok(a)
    }

    fn index_ms(&self) -> u64 {
        self.index_ms
    }

    /// The item's containing units, each keyed `"<file>:<start>-<end>"`
    /// (`"<file>:old:<start>-<end>"` for an old-side unit, which `assemble` keeps apart from a
    /// new-side one with the same numbers) and charged at its escaped cost, so planning counts
    /// exactly what `context_for` will add.
    fn companion(&self, item: &BatchItem) -> Result<Vec<Companion>, ReviewError> {
        Ok(
            containing_units_cached(item, self.sources, self.registry, &self.units)?
                .into_iter()
                .map(|unit| {
                    let (first, last) = unit.lines.unwrap_or_default();
                    let side = if unit.old_side { "old:" } else { "" };
                    let unit = escaped(unit);
                    (
                        format!("{}:{side}{first}-{last}", unit.file),
                        item_cost(&unit),
                    )
                })
                .collect(),
        )
    }

    /// The containing units of `batch`, escaped, from the review's own content: what
    /// `context_for` assembles first (`store_rag_declares_its_units`). No endpoint call.
    // MAGI FIX (MS3 §6 loop10): declared so the pipeline checks them before any embedder call.
    fn units_for(&self, batch: &ReviewBatch) -> Result<Vec<ContextItem>, ReviewError> {
        let mut units = Vec::new();
        for item in &batch.items {
            units.extend(
                containing_units_cached(item, self.sources, self.registry, &self.units)?
                    .into_iter()
                    .map(escaped),
            );
        }
        Ok(units)
    }

    /// The withheld documentation sources, then the file of each history entry withheld for a
    /// private-key block (`key_block_history_is_declared`), each once.
    // MAGI FIX (MS3 §6 loop10): a withheld history entry's file was never declared.
    fn skipped(&self) -> Vec<String> {
        let history = relock(&self.withheld_history);
        let mut out = self.withheld_sources.clone();
        // MAGI FIX (MS3 §6 loop12): a set, so the merge is O(n log n), not a scan per file.
        let mut seen: BTreeSet<&String> = self.withheld_sources.iter().collect();
        for file in history.values() {
            if seen.insert(file) {
                out.push(file.clone());
            }
        }
        out
    }

    fn masked(&self) -> Vec<(String, &'static str)> {
        self.masked_sources.clone()
    }
}

/// The bound on embedding the documentation queries of `batches` batches: one query per batch,
/// each under [`EMBED_QUERY_TIMEOUT_SECS`]. O(1).
fn query_reserve_secs(batches: usize) -> u64 {
    u64::try_from(batches)
        .unwrap_or(u64::MAX)
        .saturating_mul(EMBED_QUERY_TIMEOUT_SECS)
}

/// The bound on a review's whole outbound preparation under a run deadline (§5.0 stage 4, RV06):
/// the dimension probe [`ReviewMemory::doc_store`] may send ([`DIMENSION_PROBE_TIMEOUT_SECS`]),
/// the documentation indexing ([`REINDEX_TIMEOUT_SECS`]) and one query embedding per batch.
/// A review with no store or no embedder calls no one and reserves nothing.
///
/// Each bound counts whole, not a typical latency, so a many-batch review under a tight
/// `--timeout` can be refused though it would fit: RV06's fail-closed direction, declared.
///
/// The runner's check uses this before the probe and the indexing; the pipeline's later check
/// uses [`ReviewRag`]'s `outbound_reserve_secs` (the queries still to come); both count those
/// with `query_reserve_secs`, so they never drift apart.
///
/// # Arguments
///
/// * `batches` - The batches the review was planned into.
/// * `calls_embedder` - Whether the review has review stores and an embedder.
///
/// # Returns
///
/// The seconds the budget check holds back from a run deadline. O(1).
// MAGI FIX (MS3 §6 loop1): the probe ran after the budget check but outside the reserve, so a
// run deadline could be overshot by up to its bound.
pub fn outbound_prep_reserve_secs(batches: usize, calls_embedder: bool) -> u64 {
    if calls_embedder {
        DIMENSION_PROBE_TIMEOUT_SECS
            .saturating_add(REINDEX_TIMEOUT_SECS)
            .saturating_add(query_reserve_secs(batches))
    } else {
        0
    }
}

/// The context source the review runner hands the pipeline: the stores, the embedder and the
/// documentation preparation, bound to each review's own content by
/// [`RagSource::for_review`] (the pipeline builds the index and the content source).
pub struct ReviewRag<'a> {
    /// The documentation index, when it could be opened for this review. `None` with the
    /// history and an embedder present is declared `index unavailable` by the first batch, flag
    /// or not (`a_documentation_store_that_was_not_opened_is_declared_unavailable`).
    pub docs: Option<&'a ReviewDocStore>,
    /// The review history; `None` without a database.
    pub history: Option<&'a ReviewHistoryStore>,
    /// The language extractors.
    pub registry: &'a LanguageRegistry,
    /// The `[embedding]` embedder, if any.
    pub embedder: Option<&'a dyn EmbeddingProvider>,
    /// The batch limit in tokens.
    pub limit_tokens: usize,
    /// What [`prepare_docs`] produced.
    pub prep: DocPrep,
    /// The workspace's git, for the served history's check (see [`StoreRag::history_git`]).
    pub history_git: Option<&'a dyn GitCommand>,
}

#[async_trait::async_trait]
impl RagSource for ReviewRag<'_> {
    /// Never answers: the pipeline binds the source to a review first.
    ///
    /// # Errors
    ///
    /// Always [`ReviewError::Io`], so a caller that skipped the binding fails loudly.
    async fn context_for(&self, _batch: &ReviewBatch) -> Result<Assembled, ReviewError> {
        Err(ReviewError::Io(UNBOUND_SOURCE.to_string()))
    }

    fn index_ms(&self) -> u64 {
        self.prep.index_ms
    }

    /// One query embedding per batch, each bounded by [`EMBED_QUERY_TIMEOUT_SECS`], when the
    /// bound source will call the embedder (a store and an embedder); 0 otherwise. The pipeline's
    /// second check uses this, after the probe and the indexing already ran under the runner's
    /// [`outbound_prep_reserve_secs`]. The bound `StoreRag` keeps the trait's default because
    /// the pipeline reads the reserve from this unbound source, before binding it.
    // MAGI FIX (Loop1 r1, M6): the run deadline keeps room for these calls.
    fn outbound_reserve_secs(&self, batches: usize) -> u64 {
        let calls = self.embedder.is_some() && (self.docs.is_some() || self.history.is_some());
        if calls {
            query_reserve_secs(batches)
        } else {
            0
        }
    }

    fn skipped(&self) -> Vec<String> {
        self.prep.withheld_sources.clone()
    }

    fn masked(&self) -> Vec<(String, &'static str)> {
        self.prep.masked_sources.clone()
    }

    fn for_review<'r>(&'r self, material: ReviewMaterial<'r>) -> Option<Box<dyn RagSource + 'r>> {
        let rag = StoreRag {
            docs: self.docs,
            history: self.history,
            index: material.index,
            sources: material.sources,
            registry: self.registry,
            embedder: self.embedder,
            limit_tokens: self.limit_tokens,
            doc_index: IndexDeclarations::default(),
            declared: AtomicBool::new(false),
            docs_unavailable: false,
            embedder_down: AtomicBool::new(false),
            docs_failed: AtomicBool::new(false),
            docs_shrunk: AtomicBool::new(false),
            history_failed: AtomicBool::new(false),
            corrupt_declared: AtomicBool::new(false),
            corrupt_baseline: std::sync::OnceLock::new(),
            history_no_git_declared: AtomicBool::new(false),
            doc_sources: BTreeSet::new(),
            withheld_sources: Vec::new(),
            masked_sources: Vec::new(),
            index_ms: 0,
            doc_vectors: Mutex::new(None),
            units: UnitCache::default(),
            history_git: self.history_git,
            on_line: Mutex::new(HashMap::new()),
            withheld_history: Mutex::default(),
        }
        .with_prep(self.prep.clone());
        Some(Box::new(rag))
    }
}

/// The review stores over the open database: the history, opened once, and what a review needs
/// to open its documentation index (whose identity depends on the embedder's dimension).
pub struct ReviewMemory {
    /// The connection shared with `EncryptedSqliteMemory`.
    conn: Arc<Mutex<Connection>>,
    /// A masked duplicate of the data key, duplicated again for each documentation store.
    dek: Mutex<MaskedDek>,
    /// The review history.
    history: ReviewHistoryStore,
}

impl ReviewMemory {
    /// Opens the history over `conn`.
    ///
    /// Idempotent on a shared connection: the schema call is `CREATE ... IF NOT EXISTS` (and
    /// refuses a pre-0.25 database the same way every time), and the stores keep no state but
    /// the connection and a duplicate of the key, so a second open over the same connection
    /// (one per surface in one process) neither fails nor resets anything, and both see the same
    /// rows (`the_review_stores_open_twice_on_one_shared_connection_and_share_their_rows`,
    /// `tools::code_review`).
    ///
    /// # Arguments
    ///
    /// * `conn` - The connection shared with `EncryptedSqliteMemory`.
    /// * `dek` - A masked duplicate of the data key.
    ///
    /// # Errors
    ///
    /// [`MemoryError::Crypto`] when the key cannot be duplicated, [`MemoryError::Storage`] when
    /// the schema cannot be checked.
    pub fn open(conn: Arc<Mutex<Connection>>, mut dek: MaskedDek) -> Result<Self, MemoryError> {
        let history_key = dek
            .duplicate()
            .map_err(|e| MemoryError::Crypto(e.to_string()))?;
        let history = ReviewHistoryStore::new(Arc::clone(&conn), history_key)?;
        Ok(Self {
            conn,
            dek: Mutex::new(dek),
            history,
        })
    }

    /// The shared connection, for tests that read raw rows.
    #[cfg(test)]
    pub(crate) fn connection_for_test(&self) -> &Arc<Mutex<Connection>> {
        &self.conn
    }

    /// The review history.
    pub fn history(&self) -> &ReviewHistoryStore {
        &self.history
    }

    /// The documentation index of `embedder`'s identity. An embedder that autodetects its
    /// dimension is asked once (under [`DIMENSION_PROBE_TIMEOUT_SECS`]) to learn it; the caller
    /// runs this after the budget check, whose reserve counts that bound.
    ///
    /// # Errors
    ///
    /// [`MemoryError::Storage`] when the probe fails or answers other than exactly one non-empty
    /// vector (`a_dimension_probe_answered_with_other_than_one_vector_opens_no_store`), when the
    /// dimension is still unknown or the schema cannot be checked, [`MemoryError::Crypto`] when
    /// the key cannot be duplicated.
    pub async fn doc_store(
        &self,
        embedder: &dyn EmbeddingProvider,
    ) -> Result<ReviewDocStore, MemoryError> {
        if embedder.dim() == 0 {
            let bound = Duration::from_secs(DIMENSION_PROBE_TIMEOUT_SECS);
            let probe = vec![DIMENSION_PROBE.to_string()];
            // Its vectors are not used: a successful call sets the dimension; a failed one leaves
            // it at 0, checked right below.
            // MAGI FIX (MS3 §6 loop5): its error was discarded; the run log now says why.
            // MAGI FIX (MS3 §6 loop7): any successful answer set the identity; like a query, the
            // probe must answer exactly one non-empty vector, or no store opens.
            let failure = match tokio::time::timeout(bound, embedder.embed(&probe)).await {
                Ok(Ok(v)) if v.len() == 1 && v.iter().all(|x| !x.is_empty()) => None,
                Ok(Ok(_)) => Some("the probe was not answered with one vector".to_string()),
                Ok(Err(e)) => Some(outward_text(&e.to_string())),
                Err(_) => Some(format!("no answer within {DIMENSION_PROBE_TIMEOUT_SECS} s")),
            };
            if let Some(error) = failure {
                tracing::event!(
                    target: LOG_TARGET,
                    tracing::Level::WARN,
                    error = error.as_str(),
                    "the embedder's dimension probe failed"
                );
                return Err(MemoryError::Storage(format!(
                    "the embedder's dimension probe failed: {error}"
                )));
            }
        }
        if embedder.dim() == 0 {
            return Err(MemoryError::Storage(
                "the embedder has no known dimension".to_string(),
            ));
        }
        let key = relock(&self.dek)
            .duplicate()
            .map_err(|e| MemoryError::Crypto(e.to_string()))?;
        ReviewDocStore::new(
            Arc::clone(&self.conn),
            key,
            doc_index_id(embedder.model_id(), embedder.dim()),
        )
    }
}

/// Records `path` as withheld for a secret: one per source, by `display_path`.
fn withhold(prep: &mut DocPrep, path: &str) {
    prep.doc_index.withheld = prep.doc_index.withheld.saturating_add(1);
    prep.withheld_sources.push(display_path(path));
}

/// Whether a read error is the size cap (the source is cut, declared) rather than a failure.
fn is_size_cap(e: &ReviewError) -> bool {
    matches!(
        e,
        ReviewError::TooLarge { .. } | ReviewError::InputTooLarge { .. }
    )
}

/// `prep` degraded to no documentation (B), with one WARN naming why, its duration set.
fn degraded(mut prep: DocPrep, cause: &str, started: Instant) -> DocPrep {
    tracing::event!(
        target: LOG_TARGET,
        tracing::Level::WARN,
        error = cause,
        "the documentation index of the reviewed revision is unavailable: the review runs without documentation"
    );
    prep.docs_unavailable = true;
    prep.index_ms = millis(started.elapsed());
    prep
}

/// One listed source and, for the working tree, the bytes already read to key it.
type Listed = (DocSource, Option<Vec<u8>>);

/// The working tree's documentation sources: each path of `paths` that `wanted` accepts, read
/// through `sources` and keyed by [`worktree_key`]; a path that quotes a secret is never read,
/// and a path given twice is one source.
///
/// A file that cannot be read (a hard or escaping link, one another process holds open, one gone
/// since the listing) is left out with one WARN and declared (`unreadable`), never costing the
/// rest; the size cap cuts the
/// source (`size_capped`). Memory: the listed bytes are held until chunked, bounded by the
/// `CachedSource`'s `MAX_REVIEW_TOTAL_BYTES` (twice it at peak with the cache's copy).
///
/// # Errors
///
/// A cause when the deadline passes.
fn list_worktree(
    sources: &dyn ContentSource,
    paths: &[String],
    wanted: &dyn Fn(&str) -> bool,
    deadline: Instant,
    timeout: Duration,
    prep: &mut DocPrep,
) -> Result<Vec<Listed>, String> {
    let mut listed = Vec::new();
    // MAGI FIX (MS3 §6 loop5): a path given twice is one source, read and indexed once.
    let mut seen: HashSet<&str> = HashSet::new();
    for path in paths
        .iter()
        .filter(|p| wanted(p) && seen.insert(p.as_str()))
    {
        if Instant::now() >= deadline {
            return Err(format!("not listed within {} s", timeout.as_secs()));
        }
        if find_secret(path).is_some() {
            listed.push((
                DocSource {
                    path: path.clone(),
                    key: String::new(),
                },
                None,
            ));
            continue;
        }
        match sources.read(path) {
            Ok(Some(bytes)) => listed.push((
                DocSource {
                    path: path.clone(),
                    key: worktree_key(&bytes),
                },
                Some(bytes),
            )),
            Err(e) if is_size_cap(&e) => prep.doc_index.size_capped = true,
            // MAGI FIX (MS3 §6 loop11): a source gone since the listing is unreadable, declared.
            read => {
                // MAGI FIX (MS3 §6 loop7): declared in the report too, not only logged.
                prep.doc_index.unreadable = true;
                let cause = read.err().map_or(VANISHED.to_string(), |e| e.to_string());
                let (shown, error) = (display_path(path), outward_text(&cause));
                tracing::event!(
                    target: LOG_TARGET,
                    tracing::Level::WARN,
                    path = shown.as_str(),
                    error = error.as_str(),
                    "a documentation source could not be read and was left out"
                );
            }
        }
    }
    Ok(listed)
}

/// Groups `sources`, in order, into reindex batches of at most [`EMBED_BATCH_ITEMS`] chunks; a
/// larger source goes alone. A batch is one all-or-nothing [`ReviewDocStore::reindex`], never
/// one embedder request: `reindex` itself sends its chunks in requests of at most
/// [`EMBED_BATCH_ITEMS`] texts (`embedding_requests_are_bounded_and_written_once`), so a source
/// with more chunks still never sends a larger request. O(sources).
fn reindex_batches(
    sources: Vec<(DocSource, Vec<DocChunk>)>,
) -> Vec<Vec<(DocSource, Vec<DocChunk>)>> {
    let mut batches = Vec::new();
    let mut current: Vec<(DocSource, Vec<DocChunk>)> = Vec::new();
    let mut chunks = 0usize;
    for source in sources {
        let n = source.1.len();
        if !current.is_empty() && chunks.saturating_add(n) > EMBED_BATCH_ITEMS {
            batches.push(std::mem::take(&mut current));
            chunks = 0;
        }
        chunks = chunks.saturating_add(n);
        current.push(source);
    }
    if !current.is_empty() {
        batches.push(current);
    }
    batches
}

/// Lists, indexes and, on any failure, degrades the documentation of the reviewed revision
/// (RV07(3), RV19, RV21, U29).
///
/// 1. **List**: `rev` = the revision's new side, listed through `git` without reading a file;
///    `None` = the working tree, each of `worktree_paths` read through `sources`, keyed by
///    [`worktree_key`]. Only paths `keep` accepts that can yield a chunk count; a path that quotes
///    a secret is never read or stored, and counts as withheld.
/// 2. **Index what is missing**: read, `netted_chunks`, `bounded`; a masked source is declared
///    in `masked_sources`, a key-block one (or one the net or the store refuses) withheld; the
///    rest indexed in batches of at most [`EMBED_BATCH_ITEMS`] chunks; what a batch wrote stays.
/// 3. **Degrade (B)**: a failing listing, `missing`, read or reindex, or `timeout`, sets
///    `docs_unavailable` with a WARN and stops. The size cap only cuts (`size_capped`); a source
///    git lists but cannot give back degrades (U29); an unreadable working-tree file is left out.
///
/// Never fails. `timeout` bounds it all, checked between steps and sources. O(sources + bytes).
///
/// # Arguments
/// * `docs` - The documentation index.
/// * `git` - The review's git; `None` without a repository.
/// * `rev` - The reviewed revision's new side, or `None` for the working tree.
/// * `sources` - The revision's single content source (RV21).
/// * `worktree_paths` - The working tree's files, for `rev = None`.
/// * `keep` - The review's path filter (`.gitignore`, never-reviewed paths).
/// * `registry` - The language extractors.
/// * `embedder` - The `[embedding]` embedder.
/// * `timeout` - The bound on the whole preparation.
///
/// # Returns
///
/// What the review's context source needs ([`StoreRag::with_prep`]).
#[allow(clippy::too_many_arguments)]
pub async fn prepare_docs(
    docs: &ReviewDocStore,
    git: Option<&dyn GitCommand>,
    rev: Option<&str>,
    sources: &dyn ContentSource,
    worktree_paths: &[String],
    keep: &(dyn Fn(&str) -> bool + Sync),
    registry: &LanguageRegistry,
    embedder: &dyn EmbeddingProvider,
    timeout: Duration,
) -> DocPrep {
    let started = Instant::now();
    let deadline = started + timeout;
    let mut prep = DocPrep::default();
    // MAGI FIX (Loop1 r2, Minor 1): only a file that can yield chunks is listed and keyed.
    // A path that quotes a secret is listed whatever its kind, on purpose: it is never read, and
    // it is declared as held back for a secret (CP2 MS3 loop 11, pinned by
    // `a_working_tree_review_keys_its_sources_by_their_bytes`). The pipeline merges that entry
    // with the reviewed side's own `secret` skip of the same path, so it is listed once.
    let wanted = |p: &str| keep(p) && (find_secret(p).is_some() || yields_doc_chunks(p, registry));
    let listed = match (rev, git) {
        (Some(rev), Some(git)) => revision_doc_sources(git, rev, &wanted)
            .map(|v| v.into_iter().map(|s| (s, None)).collect())
            .map_err(|e| outward_text(&e.to_string())),
        (Some(_), None) => Err("no repository to list the reviewed revision".to_string()),
        (None, _) => list_worktree(
            sources,
            worktree_paths,
            &wanted,
            deadline,
            timeout,
            &mut prep,
        ),
    };
    let listed: Vec<Listed> = match listed {
        Ok(listed) => listed,
        Err(cause) => return degraded(prep, &cause, started),
    };
    let mut pairs: Vec<Listed> = Vec::with_capacity(listed.len());
    for (source, bytes) in listed {
        if find_secret(&source.path).is_some() {
            withhold(&mut prep, &source.path);
        } else {
            prep.doc_sources.insert(source.clone());
            pairs.push((source, bytes));
        }
    }
    let wanted_sources: Vec<DocSource> = pairs.iter().map(|(s, _)| s.clone()).collect();
    let missing: BTreeSet<DocSource> = match docs.missing(&wanted_sources) {
        Ok(missing) => missing.into_iter().collect(),
        Err(e) => return degraded(prep, &outward_text(&e.to_string()), started),
    };
    // MAGI FIX (MS3 §6 loop9): a slow `missing` read (one synchronous SQLite query, its lock
    // wait bounded by the busy timeout) was not checked against the bound.
    if Instant::now() >= deadline {
        let cause = format!("not indexed within {} s", timeout.as_secs());
        return degraded(prep, &cause, started);
    }
    let mut chunked = Vec::new();
    for (source, bytes) in pairs.into_iter().filter(|(s, _)| missing.contains(s)) {
        if Instant::now() >= deadline {
            // MAGI FIX (MS3 §6 loop8): both causes named the constant, not the bound passed.
            let cause = format!("not indexed within {} s", timeout.as_secs());
            return degraded(prep, &cause, started);
        }
        let bytes = match bytes {
            Some(bytes) => bytes,
            None => match sources.read(&source.path) {
                Ok(Some(bytes)) => bytes,
                // MAGI FIX (MS3 §6 loop4): git listed it, so the revision's index cannot be
                // completed without it (U29); skipping it served a partial set undeclared.
                Ok(None) => {
                    let cause = format!(
                        "the reviewed revision lists `{}` but it could not be read",
                        display_path(&source.path)
                    );
                    return degraded(prep, &cause, started);
                }
                // MAGI FIX (MS3 §6 loop6): cut, so no source of the index: it left `doc_sources`
                // like a not-UTF-8 or key-block source.
                Err(e) if is_size_cap(&e) => {
                    prep.doc_index.size_capped = true;
                    prep.doc_sources.remove(&source);
                    continue;
                }
                Err(e) => return degraded(prep, &outward_text(&e.to_string()), started),
            },
        };
        let Ok(text) = String::from_utf8(bytes) else {
            // Not text, so not documentation: it is no source of the revision, and it is said.
            let path = display_path(&source.path);
            tracing::event!(
                target: LOG_TARGET,
                tracing::Level::INFO,
                path = path.as_str(),
                "a documentation source is not UTF-8 and is left out"
            );
            // MAGI FIX (MS3 §6 loop5): declared in the report too (RV07), not only logged.
            prep.doc_index.not_utf8 = true;
            prep.doc_sources.remove(&source);
            continue;
        };
        // MAGI FIX (MS3 §6 loop1): RV22 amended: a chunk's one-line secrets are masked by the
        // chunker and the file is declared as masked; only a private-key block, anywhere in the
        // file (its code included), still withholds the whole source.
        // `kept` are the chunks safe to send; `dropped` names (once per chunk) the source of
        // each chunk left out for a secret, which after the chunker's masking means a private-key
        // block. A non-empty `dropped` withholds the whole source: U29 indexes a source whole or
        // not at all, so it is never served partly.
        let (kept, dropped) = netted_chunks(&source.path, &text, registry);
        if dropped.is_empty() {
            if let Some(pattern) = kept
                .iter()
                .find_map(|c| masked_pattern(&c.text).or_else(|| masked_pattern(&c.section)))
            {
                prep.masked_sources
                    .push((display_path(&source.path), pattern));
            }
            chunked.push((source, kept));
        } else {
            prep.doc_sources.remove(&source);
            withhold(&mut prep, &source.path);
        }
    }
    let kept = bounded_into(&mut prep, chunked);
    for batch in reindex_batches(kept) {
        let left = deadline.saturating_duration_since(Instant::now());
        prep.doc_index.still_indexing |= left.is_zero();
        if left.is_zero() || !index_batch(docs, &batch, embedder, left, &mut prep).await {
            let cause = "a batch of documentation sources was not indexed";
            return degraded(prep, cause, started);
        }
    }
    prep.index_ms = millis(started.elapsed());
    prep
}

/// [`bounded`] over `chunked`, recording a cut in `prep` (declared truncated); a cut source leaves
/// `doc_sources`, so the load never asks for it (`cut_source_leaves_doc_sources`).
// MAGI FIX (MS3 §6 loop10): a cut source stayed in `doc_sources`.
fn bounded_into(
    prep: &mut DocPrep,
    chunked: Vec<(DocSource, Vec<DocChunk>)>,
) -> Vec<(DocSource, Vec<DocChunk>)> {
    let offered: Vec<DocSource> = chunked.iter().map(|(s, _)| s.clone()).collect();
    let (kept, caps) = bounded_capped(chunked);
    prep.doc_index.truncated |= caps.chunks || caps.bytes;
    prep.doc_index.chunk_cap |= caps.chunks;
    prep.doc_index.byte_cap |= caps.bytes;
    let kept_set: BTreeSet<&DocSource> = kept.iter().map(|(s, _)| s).collect();
    for s in offered.iter().filter(|s| !kept_set.contains(s)) {
        prep.doc_sources.remove(s);
    }
    kept
}

/// One reindex batch for `prep`: `false` when it failed. Each source the store's own check
/// refused (RV22's backstop) is declared withheld and leaves `doc_sources`, never only logged.
// MAGI FIX (MS3 §6 loop9): a refused source was logged only.
async fn index_batch(
    docs: &ReviewDocStore,
    batch: &[(DocSource, Vec<DocChunk>)],
    embedder: &dyn EmbeddingProvider,
    timeout: Duration,
    prep: &mut DocPrep,
) -> bool {
    // MAGI FIX (MS3 §6 loop17, R2 Caspar): a bound that passed is a cold index (kept, resumed
    // by a later review), declared apart from a failure.
    let refused = match reindex_or_degrade(docs, batch, embedder, timeout).await {
        Ok(refused) => refused,
        Err(stop) => {
            prep.doc_index.still_indexing |= stop == IndexStop::TimedOut;
            return false;
        }
    };
    for (source, _) in batch {
        if refused.contains(&display_path(&source.path)) {
            prep.doc_sources.remove(source);
            withhold(prep, &source.path);
        }
    }
    true
}

/// Why a bounded reindex did not complete.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexStop {
    /// The store or the embedder failed.
    Failed,
    /// The bound passed first: the groups written are kept, the rest stays `missing`.
    TimedOut,
}

/// One [`ReviewDocStore::reindex`] under `timeout`: the sources the store refused, as
/// `display_path`, when it wrote; an [`IndexStop`] (with a WARN) when it failed or did not finish. A
/// timeout drops the group in flight, by the store's contract: `reindex` embeds a group, then
/// writes it in **one** transaction with no `.await` inside, so a future dropped here never
/// half-commits a group; the groups already written stay and the sources not yet written stay
/// `missing` for the next review. This is where the documentation indexing deadline is applied
/// (the caller's `timeout`, `prepare_docs`' time left). Leaves an INFO with its duration, the
/// rebuilt line when the index changed identity, and a WARN per source left out for a secret.
///
/// # Arguments
/// * `docs` - The documentation index.
/// * `sources` - Whole sources with their chunks.
/// * `embedder` - The `[embedding]` embedder.
/// * `timeout` - The bound on this reindex.
///
/// # Returns
///
/// The refused sources when the rest is indexed now.
///
/// # Errors
///
/// [`IndexStop`], already logged: a failure apart from a bound that passed first (a cold index
/// that a later review resumes, declared `still_indexing`).
// MAGI FIX (MS3 §6 loop17, R2 Caspar): `None` for both read a cold index as a down one.
pub async fn reindex_or_degrade(
    docs: &ReviewDocStore,
    sources: &[(DocSource, Vec<DocChunk>)],
    embedder: &dyn EmbeddingProvider,
    timeout: Duration,
) -> Result<Vec<String>, IndexStop> {
    let started = Instant::now();
    match tokio::time::timeout(timeout, docs.reindex(sources, embedder)).await {
        Ok(Ok(stats)) => {
            tracing::event!(
                target: LOG_TARGET,
                tracing::Level::INFO,
                ms = millis(started.elapsed()),
                embedded = stats.embedded,
                reused = stats.reused,
                sources = stats.sources,
                "documentation indexed"
            );
            if stats.rebuilt {
                tracing::event!(target: LOG_TARGET, tracing::Level::INFO, "{REBUILT_LINE}");
            }
            for path in &stats.refused {
                tracing::event!(
                    target: LOG_TARGET,
                    tracing::Level::WARN,
                    path = path.as_str(),
                    "a documentation source matches a secret pattern and was left out"
                );
            }
            Ok(stats.refused)
        }
        Ok(Err(e)) => {
            let error = outward_text(&e.to_string());
            tracing::event!(
                target: LOG_TARGET,
                tracing::Level::WARN,
                error = error.as_str(),
                "the documentation index could not be updated"
            );
            Err(IndexStop::Failed)
        }
        Err(_) => {
            tracing::event!(
                target: LOG_TARGET,
                tracing::Level::WARN,
                ms = millis(started.elapsed()),
                "the documentation index was not updated in time"
            );
            Err(IndexStop::TimedOut)
        }
    }
}

/// Reclassifies the due entries, least recently checked first (`due`, then `touch` each one
/// checked), then records `findings` (T3.4 `reclassify_then_record`: reclassify first, so a new
/// entry is never classified against its own commit).
///
/// Stops reclassifying once `budget` is spent, checked between entries; an entry it did not
/// reach is not touched, so it stays first for the next review. O(min(due,
/// [`MAX_RECLASSIFY_PER_REVIEW`])) classifications.
///
/// # Arguments
///
/// * `history` - The review history.
/// * `findings` - This review's entries (`entries_from_review`, anchored to the full SHA).
/// * `head` - The revision to compare against (`HEAD`).
/// * `git` - The workspace's git.
/// * `now` - Unix seconds, stamped on each checked entry.
/// * `budget` - The wall clock of the reclassification.
///
/// # Returns
///
/// What was reclassified, recorded and left over.
///
/// # Errors
///
/// [`MemoryError::Storage`] with the store's cause redacted ([`outward_text`]), never swallowed.
///
/// One transaction (MS3 §6 loop10): git is asked per due entry first, within the `budget`, and
/// the gathered writes (each reclassified status, the `checked_at` of every entry checked, then
/// the findings recorded) are applied together by one `apply_review`, all or nothing
/// (`apply_review_is_atomic`). A failure rolls the whole review's history update back, so the
/// next review reclassifies the same queue; nothing is ever half applied.
pub fn record_review(
    history: &ReviewHistoryStore,
    findings: &[HistoryEntry],
    head: &str,
    git: &dyn GitCommand,
    now: i64,
    budget: Duration,
) -> Result<RecordStats, MemoryError> {
    let started = std::time::Instant::now();
    // MAGI FIX (MS3 §6 loop11): a store error leaves through outward_text, like prepare_docs'.
    let outward = |e: MemoryError| MemoryError::Storage(outward_text(&e.to_string()));
    let (due, total) = history.due(MAX_RECLASSIFY_PER_REVIEW).map_err(outward)?;
    // MAGI FIX (MS3 §6 loop10): the writes are gathered, then applied in one transaction.
    let (mut updates, mut touched) = (Vec::new(), Vec::new());
    for entry in &due {
        if started.elapsed() >= budget {
            break;
        }
        let (changed, _) = reclassify_then_record(std::slice::from_ref(entry), &[], head, git);
        updates.extend(changed);
        touched.push(entry.id.clone());
    }
    let checked = touched.len();
    let recorded = history
        .apply_review(&updates, &touched, now, findings)
        .map_err(outward)?;
    Ok(RecordStats {
        reclassified: checked,
        recorded,
        backlog: total.saturating_sub(checked),
    })
}

/// The wall clock [`record_review`]'s reclassification may spend under `deadline` (the post-report
/// bound, or `None` for none): [`RECLASSIFY_BUDGET_SECS`] or what is left before `deadline`,
/// whichever is smaller; zero once it has passed. The reclassification is derived work, so it
/// is what yields to the deadline; the findings are still recorded (`record_review` applies them
/// with whatever was reclassified, even none). O(1).
///
/// # Arguments
///
/// * `deadline` - When the work must be over, if it must.
///
/// # Returns
///
/// The budget to hand `record_review`.
// MAGI FIX (MS3 §6 loop12): the reclassification was bounded by its own sub-budget alone, so it
// ran past a deadline the reserve did not cover (R2 Caspar).
pub fn reclassify_budget(deadline: Option<std::time::Instant>) -> Duration {
    let own = Duration::from_secs(RECLASSIFY_BUDGET_SECS);
    deadline.map_or(own, |d| {
        own.min(d.saturating_duration_since(std::time::Instant::now()))
    })
}

/// Deletes the documentation sources nothing reaches any more (RV-S39): with `git`, the live
/// set is every local ref's sources plus `worktree` when the review listed the working tree;
/// without git, `worktree` alone; with neither, nothing is deleted. A review that did not list
/// the working tree keeps those sources. O(refs x tree entries) git calls.
///
/// # Arguments
///
/// * `docs` - The documentation index.
/// * `git` - The workspace's git, if there is a repository.
/// * `keep` - The path filter the review listed with.
/// * `worktree` - The working tree's sources, when the review listed them.
///
/// # Returns
///
/// The sources deleted, or `None` when the clean-up was skipped (past `MAX_GC_REFS`, a git that
/// does not answer, or nothing known to be live).
///
/// # Errors
///
/// [`MemoryError::Storage`] with the cause redacted, for a git or store failure.
///
/// # Bounds
///
/// MAGI FIX (MS3 §6 loop14): the work is bounded by structure, not by a clock: at most
/// `MAX_GC_REFS` ref-tree listings (past that the clean-up is skipped, never started) and one
/// `DELETE` per dead source; it runs after the review has produced its report, and a failure only
/// leaves rows that the next review's clean-up deletes (the index is derived data).
///
/// Deadline (MAGI FIX, MS3 §6 loop17, R2 Caspar): unlike [`reclassify_budget`], this function
/// takes none, on purpose. Its clock is its caller's: the runner's post-report step checks the
/// post-report bound and the run's cancellation before starting it (and the history step leaves
/// it a reserve of that bound), and the wait on the whole step is itself bounded, so a clean-up
/// still running when the bound passes keeps running on its blocking thread without holding the
/// review. Once started, each listing is one git call under the runner's own per-call timeout,
/// so its worst case is `MAX_GC_REFS + 1` such calls; checking a clock between listings would cut
/// a clean-up half way for no saving the outer bound does not already give.
pub fn collect_garbage(
    docs: &ReviewDocStore,
    git: Option<&dyn GitCommand>,
    keep: &dyn Fn(&str) -> bool,
    worktree: Option<&BTreeSet<DocSource>>,
) -> Result<Option<usize>, MemoryError> {
    let mut live = match git {
        Some(git) => match live_doc_sources(git, keep) {
            Ok(Some(live)) => live,
            Ok(None) => return Ok(None),
            Err(e) => return Err(MemoryError::Storage(outward_text(&e.to_string()))),
        },
        None if worktree.is_none() => return Ok(None),
        None => BTreeSet::new(),
    };
    if let Some(worktree) = worktree {
        live.extend(worktree.iter().cloned());
    }
    // `gc`'s second argument is `keep_worktree_keys`: a review that did not list the working
    // tree (`worktree` is `None`, a revision review) keeps every `wt:` source, so a commit review
    // never drops the working tree's documentation; one that listed it judges them by `live`.
    docs.gc(&live, worktree.is_none()).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::review_docs::test_embedder::{chunk, src, FailingEmbedder, FakeEmbedder};
    use magi_rs::review::docs::bounded;
    use magi_rs::review::history::MAX_RECLASSIFY_PER_REVIEW;
    use magi_rs::review::history::{HistoryEntry, HistoryStatus};
    use magi_rs::review::rag::ContextKind;
    use magi_rs::review::source::WorktreeSource;

    /// [CP2 MS3 loop 8] The production budget, for the tests that are not about it.
    const BUDGET: Duration = Duration::from_secs(RECLASSIFY_BUDGET_SECS);

    // [RED-DRYRUN 2026-10-20] WorktreeSource carries a lifetime (its optional old-side git): 'static here, no git.
    struct RagFixture {
        docs: ReviewDocStore,
        history: ReviewHistoryStore,
        index: LexicalIndex,
        source: WorktreeSource<'static>,
        registry: LanguageRegistry,
        embedder: FakeEmbedder,
        _db: tempfile::NamedTempFile,
        _ws: tempfile::TempDir,
    }
    impl RagFixture {
        async fn new(files: &[(&str, &str)]) -> Self {
            let db = tempfile::NamedTempFile::new().expect("tmp");
            let mem = crate::system::database::EncryptedSqliteMemory::new(
                db.path().to_path_buf(),
                zeroize::Zeroizing::new(
                    crate::system::database::TEST_MASTER_PASSPHRASE.to_string(),
                ),
            )
            .expect("db");
            let ws = tempfile::tempdir().expect("ws");
            for (p, c) in files {
                std::fs::write(ws.path().join(p), c).expect("write");
            }
            let owned: Vec<(String, String)> = files
                .iter()
                .map(|(p, c)| (p.to_string(), c.to_string()))
                .collect();
            let registry = LanguageRegistry::new();
            Self {
                docs: ReviewDocStore::new(
                    mem.shared_conn(),
                    mem.data_key().expect("dek"),
                    magi_rs::review::docs::doc_index_id("fake", 3),
                )
                .expect("docs"),
                history: ReviewHistoryStore::new(mem.shared_conn(), mem.data_key().expect("dek"))
                    .expect("history"),
                index: LexicalIndex::build(&owned, &registry),
                source: WorktreeSource::new(ws.path().to_path_buf()),
                registry,
                embedder: FakeEmbedder::new(3),
                _db: db,
                _ws: ws,
            }
        }
        fn rag(&self, with_embedder: bool) -> StoreRag<'_> {
            StoreRag {
                docs: Some(&self.docs),
                history: Some(&self.history),
                index: &self.index,
                sources: &self.source,
                registry: &self.registry,
                embedder: with_embedder.then_some(&self.embedder as &dyn EmbeddingProvider),
                limit_tokens: 100_000,
                doc_index: IndexDeclarations::default(),
                declared: AtomicBool::new(false),
                docs_unavailable: false,
                embedder_down: Default::default(),
                docs_failed: Default::default(),
                docs_shrunk: Default::default(),
                history_failed: Default::default(),
                corrupt_declared: Default::default(),
                corrupt_baseline: Default::default(),
                history_no_git_declared: Default::default(),
                doc_sources: BTreeSet::new(),
                withheld_sources: Vec::new(),
                masked_sources: Vec::new(),
                index_ms: 0,
                doc_vectors: Mutex::new(None),
                units: UnitCache::default(),
                history_git: Some(&ON_HEADS_LINE),
                on_line: Mutex::new(HashMap::new()),
                withheld_history: Mutex::default(),
            }
        }
    }

    #[test]
    fn history_items_carry_no_span() {
        // [MAGI S6 MS3 loop8, R2 Melchior] A history item's text is no span of its file, yet it
        // carried `lines`, the field a later stage could read as reviewed lines.
        let e = HistoryEntry {
            id: "i".into(),
            file: "a.rs".into(),
            line: Some(5),
            reviewed_commit: "c1".into(),
            title: "t".into(),
            detail: "d".into(),
            status: HistoryStatus::Attended,
        };
        assert_eq!(history_item(e).lines, None);
    }

    #[tokio::test]
    async fn timeout_cause_names_its_bound() {
        // [MAGI S6 MS3 loop8, R2 Melchior] The cause printed REINDEX_TIMEOUT_SECS whatever bound
        // the caller passed.
        let f = RagFixture::new(&[("README.md", "# A\nx\n")]).await;
        let paths = vec!["README.md".to_string()];
        let mut prep = DocPrep::default();
        let r = list_worktree(
            &f.source,
            &paths,
            &|_: &str| true,
            Instant::now(),
            Duration::from_secs(7),
            &mut prep,
        );
        assert_eq!(r.err().as_deref(), Some("not listed within 7 s"));
    }

    #[tokio::test]
    async fn degraded_listing_keeps_worktree() {
        // [MAGI S6 MS3 loop8, R2 Caspar] A partial working-tree listing (the preparation
        // degraded) is never the live set: the clean-up keeps every `wt:` source then.
        let f = RagFixture::new(&[]).await;
        let wt = src("README.md", "wt:aa");
        f.docs
            .reindex(
                &[(wt.clone(), vec![chunk("README.md", "A", "x")])],
                &f.embedder,
            )
            .await
            .expect("indexed");
        let degraded = DocPrep {
            docs_unavailable: true,
            ..DocPrep::default()
        };
        assert_eq!(degraded.gc_worktree(None), None);
        let gc = collect_garbage(
            &f.docs,
            None,
            &|_: &str| true,
            degraded.gc_worktree(None).as_ref(),
        );
        assert_eq!(
            gc.expect("gc"),
            None,
            "nothing known to be live, nothing deleted"
        );
        assert!(f
            .docs
            .missing(std::slice::from_ref(&wt))
            .expect("m")
            .is_empty());
        let complete = DocPrep::default();
        assert_eq!(complete.gc_worktree(None), Some(BTreeSet::new()));
        assert_eq!(
            complete.gc_worktree(Some("abc")),
            None,
            "a revision lists no worktree"
        );
    }

    #[tokio::test]
    async fn a_query_from_another_model_with_the_same_dimension_is_not_searched() {
        // [MAGI S6 MS3 loop17, R1 Caspar] The identity check guarded only the reindex: a query
        // embedded by another model of the same dimension was scored against the index's vectors.
        struct OtherModel;
        #[async_trait::async_trait]
        impl EmbeddingProvider for OtherModel {
            async fn embed(&self, t: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
                Ok(t.iter().map(|_| vec![0.5; 3]).collect())
            }
            fn model_id(&self) -> &str {
                "other"
            }
            fn dim(&self) -> usize {
                3
            }
            fn query_prefix(&self) -> &str {
                ""
            }
            fn document_prefix(&self) -> &str {
                ""
            }
        }
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        f.docs
            .reindex(
                &[(src("d.md", "k"), vec![chunk("d.md", "A", "x")])],
                &f.embedder,
            )
            .await
            .expect("indexed");
        let mut rag = f.rag(true);
        rag.embedder = Some(&OtherModel);
        rag.doc_sources = BTreeSet::from([src("d.md", "k")]);
        rag.context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("the review still runs");
        assert!(
            rag.docs_failed.load(Ordering::SeqCst),
            "documentation from another model's query must be left out, declared"
        );
    }

    #[tokio::test]
    async fn unindexed_source_fails_closed() {
        // [MAGI S6 MS3 loop8, R2 Caspar] A source without a row in `doc_sources` was skipped.
        // [MAGI S6 MS3 loop10, R1 Melchior] It is the index unavailable now (a source can vanish
        // after the reindex); `prepare_docs` never keeps an unindexed one (`cut_source_leaves`).
        let f = RagFixture::new(&[]).await;
        f.docs
            .reindex(
                &[(src("d.md", "k"), vec![chunk("d.md", "A", "x")])],
                &f.embedder,
            )
            .await
            .expect("indexed");
        let mut rag = f.rag(true);
        rag.doc_sources = BTreeSet::from([src("d.md", "k"), src("cut.md", "k2")]);
        // [loop16, R1 Caspar] ... and it loses that source only: the other serves, the loss is
        // counted for the review to declare.
        let (hits, loss) = rag
            .search_docs(&f.docs, &[0.5, 0.5, 0.5])
            .expect("the indexed source serves");
        assert_eq!((hits.len(), loss.sources_gone), (1, 1));
    }

    /// A git for which every reviewed commit is on `HEAD`'s line (the fixture's history is).
    struct OnHeadsLine;
    impl GitCommand for OnHeadsLine {
        fn run_bytes(&self, a: &[&str]) -> Result<Vec<u8>, ReviewError> {
            match a.first() {
                // The reviewed commit is a strict ancestor of HEAD: the reverse question (is HEAD
                // an ancestor of it?) answers no.
                Some(&"merge-base") if a.get(2) != Some(&"HEAD") => Ok(Vec::new()),
                _ => Err(ReviewError::Git(a.join(" "))),
            }
        }
    }
    static ON_HEADS_LINE: OnHeadsLine = OnHeadsLine;

    #[tokio::test]
    async fn a_stored_reviewed_commit_that_is_not_a_sha_never_reaches_git() {
        // [MAGI S6 MS3 loop17, R2 Caspar] False positive, pinned: a stored `reviewed_commit` is
        // validated by `on_heads_line` (`is_commit_id`) before any git argument is built, so a
        // hand-edited row holding an option (`--output=x`) is never passed to git.
        struct Recording(std::sync::Mutex<Vec<String>>);
        impl GitCommand for Recording {
            fn run_bytes(&self, a: &[&str]) -> Result<Vec<u8>, ReviewError> {
                if let Ok(mut calls) = self.0.lock() {
                    calls.push(a.join(" "));
                }
                Ok(Vec::new())
            }
        }
        let git = Recording(std::sync::Mutex::new(Vec::new()));
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let mut rag = f.rag(false);
        rag.history_git = Some(&git);
        for hostile in ["--output=x", "-c", "HEAD", "c1 --all", ""] {
            let entry = HistoryEntry {
                reviewed_commit: hostile.into(),
                ..attended("h", "d")
            };
            assert!(!rag.served_on_heads_line(&entry), "{hostile:?}");
        }
        assert_eq!(
            git.0.lock().map(|c| c.clone()).unwrap_or_default(),
            Vec::<String>::new(),
            "no git call for a stored commit that is not a SHA"
        );
    }

    #[tokio::test]
    async fn memo_shares_no_verdict() {
        // MAGI FIX (MS3 §6 loop11): the memo keyed by commit cached the whole verdict, so a
        // pending entry sharing a validated one's commit decided for it.
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let rag = f.rag(false);
        let pending = HistoryEntry {
            status: HistoryStatus::Pending,
            ..attended("p", "d")
        };
        assert!(!rag.served_on_heads_line(&pending));
        assert!(rag.served_on_heads_line(&attended("a", "d")));
    }

    #[tokio::test]
    async fn a_validated_entry_off_heads_line_is_never_served() {
        // [MAGI S6 MS3 loop7, R3 Caspar] An entry validated on another branch reached the context
        // of a review on this one; each served entry is now checked against HEAD.
        struct OffLine;
        impl GitCommand for OffLine {
            fn run_bytes(&self, a: &[&str]) -> Result<Vec<u8>, ReviewError> {
                Err(ReviewError::Git(a.join(" ")))
            }
        }
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        f.history
            .record(&[attended("h1", "a was off by one")])
            .expect("record");
        f.history
            .update_status("h1", HistoryStatus::Attended)
            .expect("status");
        let f = &f;
        let served = |git: Option<&'static dyn GitCommand>| async move {
            let mut rag = f.rag(true);
            rag.history_git = git;
            let a = rag
                .context_for(&batch_on("a.rs", "fn a() {}"))
                .await
                .expect("context");
            a.items.iter().any(|i| i.kind == ContextKind::History)
        };
        static OFF: OffLine = OffLine;
        assert!(
            served(Some(&ON_HEADS_LINE)).await,
            "precondition: served on HEAD's line"
        );
        assert!(!served(Some(&OFF)).await, "served off HEAD's line");
        assert!(!served(None).await, "served with no git to check it");
    }

    #[tokio::test]
    async fn a_batch_cut_by_the_definition_total_declares_that_cut_once_and_nothing_else() {
        // [MAGI S6 MS3 loop17, R2 Caspar/Melchior/Balthasar] Every name after the total was
        // declared "more than MAX_DEFINITIONS_PER_NAME definitions", one entry per name: false
        // for a name with one definition, and unbounded.
        let names = MAX_DEFINITIONS_TOTAL * 2;
        let owned: Vec<(String, String)> = (0..names)
            .map(|i| (format!("g{i:02}.rs"), format!("fn name{i}() {{}}\n")))
            .chain([("a.rs".to_string(), "fn a() {}\n".to_string())])
            .collect();
        let borrowed: Vec<(&str, &str)> = owned
            .iter()
            .map(|(p, t)| (p.as_str(), t.as_str()))
            .collect();
        let f = RagFixture::new(&borrowed).await;
        let call: String = (0..names).map(|i| format!("name{i}();\n")).collect();
        let a = f
            .rag(false)
            .context_for(&batch_on("a.rs", &call))
            .await
            .expect("context");
        let defs: Vec<&str> = a
            .trimmed
            .iter()
            .filter(|(k, _)| *k == ContextKind::Definition)
            .map(|(_, label)| label.as_str())
            .collect();
        assert_eq!(
            defs,
            vec![format!(
                "definitions cut at MAX_DEFINITIONS_TOTAL ({MAX_DEFINITIONS_TOTAL}) for this \
                 batch: the later names were left out"
            )],
        );
    }

    fn batch_on(file: &str, text: &str) -> ReviewBatch {
        // `companion_tokens` is a field of MS2's approved `ReviewBatch` (T2.5, option 1).
        ReviewBatch {
            index: 0,
            tokens: 10,
            companion_tokens: 0,
            items: vec![BatchItem::Hunk {
                file: file.into(),
                text: text.into(),
                first_line: 1,
                last_line: 1,
                old_first_line: 1,
                old_last_line: 1,
            }],
        }
    }

    fn attended(id: &str, detail: &str) -> HistoryEntry {
        HistoryEntry {
            id: id.into(),
            file: "a.rs".into(),
            line: Some(1),
            reviewed_commit: "c1".into(),
            title: format!("t {id}"),
            detail: detail.into(),
            status: HistoryStatus::Attended,
        }
    }

    #[tokio::test]
    async fn with_an_embedder_documentation_and_validated_history_reach_the_context() {
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await; // DB, stores, index, source, FakeEmbedder(3)
        f.docs
            .reindex(
                &[(
                    src("README.md", "k"),
                    vec![chunk("README.md", "Usage", "how a works")],
                )],
                &f.embedder,
            )
            .await
            .expect("index");
        f.history
            .record(&[attended("h1", "a was off by one")])
            .expect("record");
        f.history
            .update_status("h1", HistoryStatus::Attended)
            .expect("status");
        let mut rag = f.rag(true);
        rag.doc_sources = BTreeSet::from([src("README.md", "k")]); // [RV21 D+B, 2026-10-20]
        let a = rag
            .context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("context");
        assert!(!a.without_embedder);
        let kinds: Vec<_> = a.items.iter().map(|i| i.kind.clone()).collect();
        assert!(
            kinds.contains(&ContextKind::Documentation) && kinds.contains(&ContextKind::History),
            "{kinds:?}"
        );
    }

    #[tokio::test]
    async fn a_multi_batch_review_scans_the_allowed_vectors_per_batch_and_serves_the_same() {
        // [MS3 final review I3; superseded by the loop16 R1 controller ruling] The decrypted
        // vectors are no longer cached for the whole review (that held every allowed vector at
        // once): each batch scans them in pages, so each decrypts each allowed row once, and
        // every batch is served the same documentation.
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        f.docs
            .reindex(
                &[(
                    src("README.md", "k"),
                    vec![
                        chunk("README.md", "Usage", "how a works"),
                        chunk("README.md", "More", "and more"),
                    ],
                )],
                &f.embedder,
            )
            .await
            .expect("index");
        let mut rag = f.rag(true);
        rag.doc_sources = BTreeSet::from([src("README.md", "k")]);
        let before = f.docs.vector_decrypts();
        let mut served = Vec::new();
        for _ in 0..3 {
            let a = rag
                .context_for(&batch_on("a.rs", "fn a() {}"))
                .await
                .expect("context");
            let docs: Vec<String> = a
                .items
                .iter()
                .filter(|i| i.kind == ContextKind::Documentation)
                .map(|i| i.text.clone())
                .collect();
            served.push(docs);
        }
        assert_eq!(
            f.docs.vector_decrypts() - before,
            2 * 3,
            "each allowed row once per batch"
        );
        assert_eq!(served[0].len(), 2, "{served:?}");
        assert!(served.iter().all(|d| *d == served[0]), "{served:?}");
    }

    #[tokio::test]
    async fn without_an_embedder_documentation_and_history_are_left_out_and_declared() {
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        f.history.record(&[attended("h1", "fine")]).expect("record");
        let a = f
            .rag(false)
            .context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("context");
        assert!(a.without_embedder);
        assert!(a
            .items
            .iter()
            .all(|i| !matches!(i.kind, ContextKind::Documentation | ContextKind::History)));
    }

    #[tokio::test]
    async fn validated_history_cut_at_its_bound_is_declared() {
        // [MAGI S6 MS3 loop4, R3 Balthasar] RV07 declares what was cut: past
        // MAX_HISTORY_PER_BATCH the newest entries are kept and the cut is named.
        use crate::memory::review_history::MAX_HISTORY_PER_BATCH;
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let entries: Vec<HistoryEntry> = (0..=MAX_HISTORY_PER_BATCH)
            .map(|i| attended(&format!("h{i}"), "fine"))
            .collect();
        f.history.record(&entries).expect("record");
        for e in &entries {
            f.history
                .update_status(&e.id, HistoryStatus::Attended)
                .expect("status");
        }
        let a = f
            .rag(true)
            .context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("context");
        assert!(
            a.trimmed.contains(&(
                ContextKind::History,
                format!(
                    "more than MAX_HISTORY_PER_BATCH ({MAX_HISTORY_PER_BATCH}) validated \
                     entries on these files: the newest kept"
                )
            )),
            "{:?}",
            a.trimmed
        );
    }

    #[tokio::test]
    async fn a_history_entry_quoting_a_secret_reaches_the_context_masked() {
        // [MAGI S6 MS3 loop1, R2 all three] RV22 amended lists history among the egress points
        // that mask, not omit: each entry stays, its one-line secret masked in title and detail.
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let leaked = "key sk-ant-api03-AAAAAAAAAAAAAAAAAAAA is hardcoded"; // allow-secret-scan
                                                                           // [CP2 MS3 loop 11] Title and detail are each scanned: one entry per field.
        let mut by_title = attended("h2", "fine");
        by_title.title = leaked.into();
        // RV22's exception: a private-key block is never masked, its entry is withheld.
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        let keyed = attended(
            "h3",
            &format!("{begin}\nMIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8w"),
        );
        f.history
            .record(&[attended("h1", leaked), by_title, keyed])
            .expect("record");
        for id in ["h1", "h2", "h3"] {
            f.history
                .update_status(id, HistoryStatus::Attended)
                .expect("status");
        }
        let a = f
            .rag(true)
            .context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("context");
        assert!(
            a.items.iter().all(|i| !i.text.contains("sk-ant-")
                && !i.label.contains("sk-ant-")
                && !i.text.contains("MIIB")),
            "a secret left the process"
        );
        let history: Vec<&ContextItem> = a
            .items
            .iter()
            .filter(|i| i.kind == ContextKind::History)
            .collect();
        assert_eq!(history.len(), 2, "{history:?}");
        assert!(
            history
                .iter()
                .all(|i| i.text.contains("<redacted: anthropic-key>")),
            "{history:?}"
        );
        assert!(
            history
                .iter()
                .any(|i| i.label.contains("<redacted: anthropic-key>")),
            "the title is the label, masked too: {history:?}"
        );
        assert_eq!(a.withheld, 1, "only the key-block entry is withheld");
    }

    #[tokio::test]
    async fn key_block_entry_counted_once() {
        // [MAGI S6 MS3 loop9, R2 Caspar] The same withheld entry, met by two batches on its
        // file, counted once per batch: the report's total named more entries than exist.
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        f.history
            .record(&[attended("h3", &format!("{begin}\nQQQQ"))])
            .expect("record");
        f.history
            .update_status("h3", HistoryStatus::Attended)
            .expect("status");
        let rag = f.rag(true);
        let mut total = 0;
        for _ in 0..2 {
            let a = rag.context_for(&batch_on("a.rs", "fn a() {}")).await;
            total += a.expect("context").withheld;
        }
        assert_eq!(total, 1);
    }

    #[test]
    fn poison_is_recovered_and_cleared() {
        // [MAGI S6 MS3 loop10, R2 Balthasar] A poisoned lock was swallowed in silence.
        let m = std::sync::Arc::new(Mutex::new(7u8));
        let held = std::sync::Arc::clone(&m);
        let _ = std::thread::spawn(move || {
            let _g = held.lock().expect("lock");
            panic!("poison");
        })
        .join();
        assert!(m.is_poisoned(), "precondition");
        assert_eq!(*relock(&m), 7);
        assert!(
            !m.is_poisoned(),
            "recovery is explicit: the poison is cleared"
        );
    }

    #[tokio::test]
    async fn secret_path_history_withheld() {
        // [MAGI S6 MS3 loop10, R2 Caspar] A history entry's file path left raw, relying on the
        // pipeline's backstop; a secret-shaped path withholds the entry here already.
        let path = concat!("ghp_", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA.rs"); // allow-secret-scan
        let f = RagFixture::new(&[(path, "fn a() {}\n")]).await;
        let h = HistoryEntry {
            file: path.into(),
            ..attended("h4", "off by one")
        };
        f.history.record(&[h]).expect("record");
        f.history
            .update_status("h4", HistoryStatus::Attended)
            .expect("status");
        let rag = f.rag(true);
        let a = rag
            .context_for(&batch_on(path, "fn a() {}"))
            .await
            .expect("context");
        assert!(
            a.items.iter().all(|i| i.kind != ContextKind::History),
            "{a:?}"
        );
        assert_eq!((a.withheld, rag.skipped().len()), (1, 1));
    }

    #[tokio::test]
    async fn key_block_history_is_declared() {
        // [MAGI S6 MS3 loop10, R2 Melchior] A key-block history entry was only counted: its file
        // never reached the report's `secret` omissions (RV22's exception).
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        let h = attended("h3", &format!("{begin}\nQQQQ"));
        f.history.record(&[h]).expect("record");
        f.history
            .update_status("h3", HistoryStatus::Attended)
            .expect("status");
        let rag = f.rag(true);
        for _ in 0..2 {
            rag.context_for(&batch_on("a.rs", "fn a() {}"))
                .await
                .expect("context");
        }
        assert_eq!(rag.skipped(), vec!["a.rs".to_string()]);
    }

    use crate::memory::error::EmbeddingError;

    /// An embedder that never answers. [CP2 MS3 loop 6] FailingEmbedder now lives in review_docs::test_embedder (T3.3).
    struct HangingEmbedder;
    #[async_trait::async_trait]
    impl EmbeddingProvider for HangingEmbedder {
        async fn embed(&self, _t: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            std::future::pending().await
        }
        fn model_id(&self) -> &str {
            "fake"
        } // [CP2 MS3 loop 13] the store's model: the reindex bound is what fires
        fn dim(&self) -> usize {
            3
        }
        fn query_prefix(&self) -> &str {
            ""
        }
        fn document_prefix(&self) -> &str {
            ""
        }
    }

    #[tokio::test]
    async fn a_failed_dimension_probe_is_logged_with_its_cause_masked() {
        // [MAGI S6 MS3 loop5, R2 Caspar] The probe's error was discarded: the run log was silent
        // and the caller saw only "no known dimension". Its cause now reaches the log, through
        // the same scrubbing as every other outward text.
        use tracing_subscriber::layer::SubscriberExt;
        #[derive(Clone, Default)]
        struct Warns(std::sync::Arc<std::sync::Mutex<Vec<String>>>);
        struct Fields(String);
        impl tracing::field::Visit for Fields {
            fn record_str(&mut self, _: &tracing::field::Field, value: &str) {
                self.0.push_str(value);
                self.0.push(' ');
            }
            fn record_debug(&mut self, _: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                self.0.push_str(&format!("{value:?} "));
            }
        }
        impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for Warns {
            fn on_event(
                &self,
                e: &tracing::Event<'_>,
                _: tracing_subscriber::layer::Context<'_, S>,
            ) {
                if *e.metadata().level() == tracing::Level::WARN {
                    let mut f = Fields(String::new());
                    e.record(&mut f);
                    self.0.lock().expect("lock").push(f.0);
                }
            }
        }
        struct Autodetect;
        #[async_trait::async_trait]
        impl EmbeddingProvider for Autodetect {
            async fn embed(&self, _t: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
                Err(EmbeddingError::Http(
                    concat!("refused for ghp_", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA").into(), // allow-secret-scan
                ))
            }
            fn model_id(&self) -> &str {
                "fake"
            }
            fn dim(&self) -> usize {
                0
            }
            fn query_prefix(&self) -> &str {
                ""
            }
            fn document_prefix(&self) -> &str {
                ""
            }
        }
        let warns = Warns::default();
        let _guard =
            tracing::subscriber::set_default(tracing_subscriber::registry().with(warns.clone()));
        let tmp = tempfile::NamedTempFile::new().expect("tmp");
        let mem = crate::system::database::EncryptedSqliteMemory::new(
            tmp.path().to_path_buf(),
            zeroize::Zeroizing::new(crate::system::database::TEST_MASTER_PASSPHRASE.to_string()),
        )
        .expect("db");
        let memory =
            ReviewMemory::open(mem.shared_conn(), mem.data_key().expect("dek")).expect("open");
        assert!(memory.doc_store(&Autodetect).await.is_err());
        let logged = warns.0.lock().expect("lock").join("\n");
        assert!(logged.contains("refused for"), "{logged}");
        assert!(!logged.contains("ghp_AAAA"), "{logged}");
    }

    #[tokio::test]
    async fn a_dimension_probe_answered_with_other_than_one_vector_opens_no_store() {
        // [MAGI S6 MS3 loop7, R2 Caspar] Any successful probe set the index identity, even an
        // answer the query path refuses (no vector, two, or an empty one).
        struct Malformed(Vec<Vec<f32>>, std::sync::atomic::AtomicUsize);
        #[async_trait::async_trait]
        impl EmbeddingProvider for Malformed {
            async fn embed(&self, _t: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
                self.1.store(3, Ordering::SeqCst);
                Ok(self.0.clone())
            }
            fn model_id(&self) -> &str {
                "fake"
            }
            fn dim(&self) -> usize {
                self.1.load(Ordering::SeqCst)
            }
            fn query_prefix(&self) -> &str {
                ""
            }
            fn document_prefix(&self) -> &str {
                ""
            }
        }
        let tmp = tempfile::NamedTempFile::new().expect("tmp");
        let mem = crate::system::database::EncryptedSqliteMemory::new(
            tmp.path().to_path_buf(),
            zeroize::Zeroizing::new(crate::system::database::TEST_MASTER_PASSPHRASE.to_string()),
        )
        .expect("db");
        let memory =
            ReviewMemory::open(mem.shared_conn(), mem.data_key().expect("dek")).expect("open");
        for answer in [vec![], vec![vec![1.0; 3]; 2], vec![vec![]]] {
            let e = Malformed(answer.clone(), Default::default());
            assert!(memory.doc_store(&e).await.is_err(), "{answer:?}");
        }
        let good = Malformed(vec![vec![1.0; 3]], Default::default());
        assert!(memory.doc_store(&good).await.is_ok());
    }

    #[tokio::test]
    async fn a_failing_embedder_degrades_to_no_documentation_or_history() {
        // [CP2 MS3 loop 2] RV07: the review goes on without (3) and (4), declared.
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        f.history.record(&[attended("h1", "fine")]).expect("record");
        let failing = FailingEmbedder;
        let mut rag = f.rag(true);
        rag.embedder = Some(&failing);
        let a = rag
            .context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("never fails the review");
        assert!(a.without_embedder);
        assert!(a
            .items
            .iter()
            .all(|i| !matches!(i.kind, ContextKind::Documentation | ContextKind::History)));
    }

    #[tokio::test]
    async fn a_down_embedder_is_declared_apart_from_a_missing_one_once() {
        // [MAGI S6 MS3 loop2, R2 Melchior] A configured embedder that fails read in the report
        // exactly like no embedder at all; the cause lived only in the run log.
        const DOWN: &str = "the embedder did not answer";
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let failing = FailingEmbedder;
        let mut rag = f.rag(true);
        rag.embedder = Some(&failing);
        let first = rag
            .context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("never fails the review");
        assert_eq!(
            first
                .trimmed
                .iter()
                .filter(|(_, label)| label == DOWN)
                .collect::<Vec<_>>(),
            vec![
                &(ContextKind::Documentation, DOWN.to_string()),
                &(ContextKind::History, DOWN.to_string())
            ]
        );
        let second = rag
            .context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("never fails the review");
        assert!(
            second.trimmed.iter().all(|(_, label)| label != DOWN),
            "declared once per review: {:?}",
            second.trimmed
        );
        let missing = f
            .rag(false)
            .context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("context");
        assert!(missing.trimmed.iter().all(|(_, label)| label != DOWN));
    }

    #[tokio::test(start_paused = true)]
    async fn a_hanging_query_embed_times_out_and_degrades() {
        // [CP2 MS3 loop 4] EMBED_QUERY_TIMEOUT_SECS bounds the query like REINDEX_TIMEOUT_SECS bounds the index.
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let hanging = HangingEmbedder;
        let mut rag = f.rag(true);
        rag.embedder = Some(&hanging);
        let bound = Duration::from_secs(2 * EMBED_QUERY_TIMEOUT_SECS);
        let a = tokio::time::timeout(bound, rag.context_for(&batch_on("a.rs", "fn a() {}")))
            .await
            .expect("the query bound fires first")
            .expect("never fails the review");
        assert!(a.without_embedder);
    }

    #[tokio::test]
    async fn without_memory_the_code_context_stays_and_the_cause_is_its_own() {
        // [CP2 MS3 loop 4] --no-memory opens no DB; it is never reported as a missing embedder.
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let mut rag = f.rag(true);
        (rag.docs, rag.history) = (None, None);
        let a = rag
            .context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("context");
        assert!(a.without_memory && !a.without_embedder);
        assert!(a.items.iter().any(|i| i.kind == ContextKind::Unit));
    }

    #[tokio::test]
    async fn the_index_declarations_are_reported_once_per_review() {
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let mut rag = f.rag(true);
        rag.doc_index = IndexDeclarations {
            withheld: 1,
            truncated: true,
            ..Default::default()
        };
        let first = rag
            .context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("context");
        let second = rag
            .context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("context");
        assert_eq!((first.withheld, second.withheld), (1, 0));
        assert!(first
            .trimmed
            .iter()
            .any(|(k, _)| *k == ContextKind::Documentation));
        assert!(second
            .trimmed
            .iter()
            .all(|(k, _)| *k != ContextKind::Documentation));
    }

    #[tokio::test]
    async fn a_failing_reindex_degrades_and_never_fails_the_review() {
        // [CP2 MS3 loop 3] S51's scenario: the embedder is configured but down while indexing.
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let docs = [(
            src("README.md", "k"),
            vec![chunk("README.md", "Usage", "how a works")],
        )]; // [RV21 D+B, 2026-10-20]
        let bound = Duration::from_secs(REINDEX_TIMEOUT_SECS);
        assert!(
            reindex_or_degrade(&f.docs, &docs, &f.embedder, bound)
                .await
                .is_ok(),
            "a healthy embedder indexes"
        );
        // [CP2 MS3 loop 13] Another source: `docs` is indexed now and would be skipped without calling the embedder, so
        // the failing one must have something new to embed (with the old "failing" model id the index-id gate hid this).
        let other = [(
            src("README.md", "k2"),
            vec![chunk("README.md", "Usage", "how a works, v2")],
        )];
        assert_eq!(
            reindex_or_degrade(&f.docs, &other, &FailingEmbedder, bound).await,
            Err(IndexStop::Failed)
        );
    }

    #[tokio::test]
    async fn an_indexing_cut_by_its_deadline_is_declared_still_building_not_unavailable() {
        // [MAGI S6 MS3 loop17, R2 Caspar] A cold index that did not finish inside the bound read
        // "index unavailable", like a down embedder; progress is kept, and the operator was not
        // told that a later review resumes.
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let docs = [(
            src("README.md", "k"),
            vec![chunk("README.md", "Usage", "how a works")],
        )];
        let mut slow = DocPrep::default();
        assert!(
            !index_batch(
                &f.docs,
                &docs,
                &HangingEmbedder,
                Duration::from_millis(50),
                &mut slow
            )
            .await
        );
        assert!(slow.doc_index.still_indexing, "the deadline cut it");
        let mut down = DocPrep::default();
        assert!(
            !index_batch(
                &f.docs,
                &docs,
                &FailingEmbedder,
                Duration::from_secs(REINDEX_TIMEOUT_SECS),
                &mut down
            )
            .await
        );
        assert!(
            !down.doc_index.still_indexing,
            "a failure is not a cold index"
        );

        let mut rag = f.rag(true);
        rag.docs_unavailable = true;
        rag.doc_index.still_indexing = true;
        let a = rag
            .context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("context");
        let docs: Vec<&str> = a
            .trimmed
            .iter()
            .filter(|(k, _)| *k == ContextKind::Documentation)
            .map(|(_, l)| l.as_str())
            .collect();
        assert_eq!(
            docs,
            vec![
                "index still being built: the indexing deadline passed; what was indexed is kept \
                 and a later review resumes it"
            ]
        );
    }

    #[tokio::test]
    async fn store_refusal_is_declared() {
        // [MAGI S6 MS3 loop9, R2 Caspar] RV22: a source the store's own check refused was only
        // logged; it is declared withheld and leaves the revision's sources.
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let key = format!("sk-ant-api03-{}", "Q1w2E3r4".repeat(5)); // allow-secret-scan
        let (bad, good) = (src("d.md", "k"), src("e.md", "k2"));
        let batch = [
            (bad.clone(), vec![chunk("d.md", "A", &format!("key {key}"))]),
            (good.clone(), vec![chunk("e.md", "B", "fine")]),
        ];
        let mut prep = DocPrep {
            doc_sources: BTreeSet::from([bad.clone(), good.clone()]),
            ..DocPrep::default()
        };
        let bound = Duration::from_secs(REINDEX_TIMEOUT_SECS);
        assert!(index_batch(&f.docs, &batch, &f.embedder, bound, &mut prep).await);
        assert_eq!(prep.withheld_sources, vec!["d.md".to_string()]);
        assert_eq!(prep.doc_index.withheld, 1);
        assert_eq!(prep.doc_sources, BTreeSet::from([good]));
    }

    #[tokio::test]
    async fn a_failed_reindex_is_declared_once_as_its_own_cause() {
        // [CP2 MS3 loop 5] Embedder up, index down: neither without_embedder nor without_memory.
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        f.history.record(&[attended("h1", "fine")]).expect("record");
        // [CP2 MS3 loop 12] Part of the revision was already indexed: B still serves none of it (RV-S38), unlike a trim.
        let indexed = src("README.md", "k");
        f.docs
            .reindex(
                &[(
                    indexed.clone(),
                    vec![chunk("README.md", "Usage", "how a works")],
                )],
                &f.embedder,
            )
            .await
            .expect("indexed");
        let mut rag = f.rag(true);
        rag.doc_sources = BTreeSet::from([indexed]);
        rag.docs_unavailable = true;
        let first = rag
            .context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("context");
        let second = rag
            .context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("context");
        assert!(!first.without_embedder && !first.without_memory);
        // [CP2 MS3 loop 6] The history goes on without the documentation index.
        assert!(
            first.items.iter().any(|i| i.kind == ContextKind::History),
            "{:?}",
            first.items.len()
        );
        assert!(
            first
                .trimmed
                .contains(&(ContextKind::Documentation, "index unavailable".to_string())),
            "{:?}",
            first.trimmed
        );
        assert!(!second
            .trimmed
            .contains(&(ContextKind::Documentation, "index unavailable".to_string())));
        assert!(
            first
                .items
                .iter()
                .all(|i| i.kind != ContextKind::Documentation),
            "zero documentation, not partial"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_hanging_reindex_times_out_and_degrades() {
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let bound = Duration::from_secs(REINDEX_TIMEOUT_SECS);
        let r = tokio::time::timeout(
            2 * bound,
            reindex_or_degrade(
                &f.docs,
                &[(src("README.md", "k"), vec![chunk("README.md", "U", "x")])],
                &HangingEmbedder,
                bound,
            ),
        )
        .await;
        assert!(
            matches!(r, Ok(Err(IndexStop::TimedOut))),
            "the reindex bound fires before the test's own: {r:?}"
        );
    }

    /// Fails every embed and counts the calls. [CP2 MS3 loop 7]
    struct CountingFailingEmbedder(std::sync::atomic::AtomicUsize);
    #[async_trait::async_trait]
    impl EmbeddingProvider for CountingFailingEmbedder {
        async fn embed(&self, _t: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Err(EmbeddingError::Http("down".into()))
        }
        fn model_id(&self) -> &str {
            "counting-failing"
        }
        fn dim(&self) -> usize {
            3
        }
        fn query_prefix(&self) -> &str {
            ""
        }
        fn document_prefix(&self) -> &str {
            ""
        }
    }

    #[tokio::test]
    async fn a_down_embedder_is_tried_once_per_review_not_once_per_batch() {
        // [CP2 MS3 loop 7] Each later batch would otherwise pay up to EMBED_QUERY_TIMEOUT_SECS again.
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let down = CountingFailingEmbedder(Default::default());
        let mut rag = f.rag(true);
        rag.embedder = Some(&down);
        for _ in 0..3 {
            let a = rag
                .context_for(&batch_on("a.rs", "fn a() {}"))
                .await
                .expect("never fails the review");
            assert!(a.without_embedder);
        }
        assert_eq!(down.0.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_history_read_error_mid_review_degrades_and_is_declared_once() {
        // [CP2 MS3 loop 7] Neither silent nor fatal: that source is left out for the review and declared once.
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        f.history
            .connection_for_test()
            .lock()
            .expect("lock")
            .execute_batch("DROP TABLE code_review_history;")
            .expect("drop");
        let rag = f.rag(true);
        let first = rag
            .context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("never fails the review");
        let second = rag
            .context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("never fails the review");
        let unavailable = (ContextKind::History, "index unavailable".to_string());
        assert!(first.trimmed.contains(&unavailable), "{:?}", first.trimmed);
        assert!(!second.trimmed.contains(&unavailable));
        assert!(
            first.items.iter().any(|i| i.kind == ContextKind::Unit),
            "the code context stays"
        );
    }

    #[tokio::test]
    async fn the_unit_definitions_and_callers_of_a_batch_reach_the_context() {
        // [CP2 MS3 loop 3] RV07(1)-(2) from the real StoreRag, not only through FakeRag.
        let f = RagFixture::new(&[
            ("a.rs", "fn a() {\n    b();\n}\n"),
            ("b.rs", "fn b() {}\n"),
            ("c.rs", "fn c() {\n    a();\n}\n"),
        ])
        .await;
        let batch = ReviewBatch {
            index: 0,
            tokens: 10,
            companion_tokens: 0,
            items: vec![BatchItem::Hunk {
                file: "a.rs".into(),
                text: "    b();".into(),
                first_line: 2,
                last_line: 2,
                old_first_line: 2,
                old_last_line: 2,
            }],
        };
        let a = f.rag(false).context_for(&batch).await.expect("context");
        let has =
            |k: ContextKind, file: &str| a.items.iter().any(|i| i.kind == k && i.file == file);
        assert!(
            has(ContextKind::Unit, "a.rs")
                && has(ContextKind::Definition, "b.rs")
                && has(ContextKind::Caller, "c.rs"),
            "{:?}",
            a.items
                .iter()
                .map(|i| (i.kind.clone(), i.file.clone()))
                .collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn store_rag_declares_its_units() {
        // [MAGI S6 MS3 loop10, R4 Caspar] The pipeline checks `units_for` before any embedder
        // call; the production source declares the units its `context_for` assembles.
        let f = RagFixture::new(&[("a.rs", "fn a() {\n    b();\n}\n")]).await;
        let batch = batch_on("a.rs", "    b();");
        let rag = f.rag(false);
        let units = rag.units_for(&batch).expect("units");
        let assembled = rag.context_for(&batch).await.expect("context");
        let from_context: Vec<_> = assembled
            .items
            .into_iter()
            .filter(|i| i.kind == ContextKind::Unit)
            .collect();
        assert!(!units.is_empty());
        assert_eq!(units, from_context);
    }

    #[tokio::test]
    async fn whole_file_gets_callers() {
        // [MAGI S6 MS3 loop10, R2 Melchior] A whole-file batch named no changed unit, so it never
        // retrieved the callers of its functions.
        let a = "fn a() {}\n";
        let f = RagFixture::new(&[("a.rs", a), ("c.rs", "fn c() {\n    a();\n}\n")]).await;
        let batch = ReviewBatch {
            index: 0,
            tokens: 10,
            companion_tokens: 0,
            items: vec![BatchItem::WholeFile {
                file: "a.rs".into(),
                text: a.into(),
                lines: 1,
            }],
        };
        let got = f.rag(false).context_for(&batch).await.expect("context");
        assert!(
            got.items
                .iter()
                .any(|i| i.kind == ContextKind::Caller && i.file == "c.rs"),
            "{:?}",
            got.items
        );
    }

    #[tokio::test]
    async fn whole_file_method_callers() {
        // MAGI FIX (MS3 §6 loop11): a whole file named its top-level units only, so its
        // methods got no callers.
        let a = "struct S;\nimpl S {\n    fn m(&self) {}\n}\n";
        let c = "fn c() {\n    get().m();\n}\n";
        let f = RagFixture::new(&[("a.rs", a), ("c.rs", c)]).await;
        let batch = ReviewBatch {
            index: 0,
            tokens: 10,
            companion_tokens: 0,
            items: vec![BatchItem::WholeFile {
                file: "a.rs".into(),
                text: a.into(),
                lines: 4,
            }],
        };
        let got = f.rag(false).context_for(&batch).await.expect("context");
        assert!(
            got.items
                .iter()
                .any(|i| i.kind == ContextKind::Caller && i.label == "c"),
            "{:?}",
            got.items
        );
    }

    #[tokio::test]
    async fn a_secret_outside_the_changed_lines_reaches_every_code_context_item_masked() {
        // [MAGI S6 MS3 loop4, R2 all three] Units, definitions and callers are read through the
        // review's cache (`CachedSource`, the source the pipeline binds), which holds every text
        // side with its one-line secrets masked before any stage reads it (RV22 amended). So a
        // secret in an unreviewed line of the containing unit, of a definition or of a caller
        // never reaches the context raw, and the text charged to the budget is the masked one.
        let key = "sk-ant-api03-AAAAAAAAAAAAAAAAAAAA"; // allow-secret-scan
        let f = RagFixture::new(&[
            (
                "a.rs",
                &format!("fn a() {{\n    b();\n    let k = \"{key}\";\n}}\n"),
            ),
            ("b.rs", &format!("fn b() {{\n    let k = \"{key}\";\n}}\n")),
            (
                "c.rs",
                &format!("fn c() {{\n    a();\n    let k = \"{key}\";\n}}\n"),
            ),
        ])
        .await;
        let cached = magi_rs::review::source::CachedSource::new(&f.source);
        let rag = StoreRag {
            sources: &cached,
            ..f.rag(false)
        };
        let batch = ReviewBatch {
            index: 0,
            tokens: 10,
            companion_tokens: 0,
            items: vec![BatchItem::Hunk {
                file: "a.rs".into(),
                text: "    b();".into(),
                first_line: 2,
                last_line: 2,
                old_first_line: 2,
                old_last_line: 2,
            }],
        };
        let a = rag.context_for(&batch).await.expect("context");
        let kinds: BTreeSet<String> = a.items.iter().map(|i| format!("{:?}", i.kind)).collect();
        assert_eq!(
            kinds,
            ["Caller", "Definition", "Unit"]
                .into_iter()
                .map(String::from)
                .collect()
        );
        for item in &a.items {
            assert!(
                !item.text.contains("sk-ant-") && item.text.contains("<redacted: anthropic-key>"),
                "{:?} of {} left raw: {}",
                item.kind,
                item.file,
                item.text
            );
        }
    }

    /// Answers every git call with a diff covering line 1 (so `merge-base` succeeds too).
    struct LineOneChanged;
    impl GitCommand for LineOneChanged {
        fn run_bytes(&self, _a: &[&str]) -> Result<Vec<u8>, ReviewError> {
            Ok(b"@@ -1 +1 @@\n".to_vec())
        }
    }

    #[tokio::test]
    async fn record_error_is_outward() {
        // MAGI FIX (MS3 §6 loop11): a store error left record_review raw, unlike prepare_docs.
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let token = concat!("ghp_", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"); // allow-secret-scan
        let conn = rusqlite::Connection::open(f._db.path()).expect("open");
        conn.execute_batch(&format!(
            "CREATE TRIGGER t BEFORE INSERT ON code_review_history \
             BEGIN SELECT RAISE(ABORT, '{token}'); END;"
        ))
        .expect("trigger");
        let entry = HistoryEntry {
            status: HistoryStatus::Pending,
            ..attended("n1", "d")
        };
        let e = record_review(&f.history, &[entry], "HEAD", &LineOneChanged, 100, BUDGET)
            .expect_err("the trigger aborts");
        assert!(!e.to_string().contains(token), "{e}");
    }

    #[tokio::test]
    async fn record_review_reclassifies_first_then_records() {
        // [CP2 MS3 loop 3] The bin composition: reclassify_then_record, then update_status and record.
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let pending = |id: &str| HistoryEntry {
            status: HistoryStatus::Pending,
            ..attended(id, "d")
        };
        f.history.record(&[pending("old")]).expect("seed");
        let stats = record_review(
            &f.history,
            &[pending("n1"), pending("n2")],
            "HEAD",
            &LineOneChanged,
            100,
            BUDGET,
        )
        .expect("ok");
        assert_eq!(
            stats,
            RecordStats {
                reclassified: 1,
                recorded: 2,
                backlog: 0
            }
        );
        let all = f.history.all().expect("all");
        assert_eq!(
            all.iter().find(|e| e.id == "old").map(|e| e.status),
            Some(HistoryStatus::Attended)
        );
        assert_eq!(all.len(), 3);
    }

    #[tokio::test]
    async fn record_review_touches_an_entry_whose_status_did_not_change() {
        // [MAGI S6 MS3 loop4, R3 Melchior] `due()` promises that an entry that never resolves
        // goes to the back of the queue; that holds only because every entry checked is
        // touched, changed or not. A git that cannot answer leaves the entry pending.
        struct NoGit;
        impl GitCommand for NoGit {
            fn run_bytes(&self, a: &[&str]) -> Result<Vec<u8>, ReviewError> {
                Err(ReviewError::Git(a.join(" ")))
            }
        }
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let pending = HistoryEntry {
            status: HistoryStatus::Pending,
            ..attended("stuck", "d")
        };
        f.history.record(&[pending]).expect("seed");
        record_review(&f.history, &[], "HEAD", &NoGit, 4242, BUDGET).expect("ok");
        let all = f.history.all().expect("all");
        assert_eq!(all.first().map(|e| e.status), Some(HistoryStatus::Pending));
        let checked: i64 = f
            .history
            .connection_for_test()
            .lock()
            .expect("lock")
            .query_row(
                "SELECT checked_at FROM code_review_history WHERE id = 'stuck'",
                [],
                |r| r.get(0),
            )
            .expect("row");
        assert_eq!(checked, 4242, "an unchanged entry is touched too");
    }

    #[tokio::test]
    async fn record_review_reports_the_backlog_left_beyond_the_cap() {
        // [CP2 MS3 loop 6] What the INFO carries: due entries this review did not reach.
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let olds: Vec<HistoryEntry> = (0..MAX_RECLASSIFY_PER_REVIEW + 1)
            .map(|i| HistoryEntry {
                id: format!("o{i:04}"),
                status: HistoryStatus::Pending,
                ..attended("x", "d")
            })
            .collect();
        f.history.record(&olds).expect("seed");
        let stats =
            record_review(&f.history, &[], "HEAD", &LineOneChanged, 100, BUDGET).expect("ok");
        assert_eq!(
            (stats.reclassified, stats.backlog),
            (MAX_RECLASSIFY_PER_REVIEW, 1)
        );
    }

    #[tokio::test]
    async fn a_failing_history_store_is_an_error_never_a_silent_success() {
        // [CP2 MS3 loop 6] main.rs turns this into the stderr warning; record_review itself never swallows it.
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        f.history
            .connection_for_test()
            .lock()
            .expect("lock")
            .execute_batch("DROP TABLE code_review_history;")
            .expect("drop");
        let r = record_review(
            &f.history,
            &[attended("n1", "d")],
            "HEAD",
            &LineOneChanged,
            100,
            BUDGET,
        );
        assert!(matches!(r, Err(MemoryError::Storage(_))), "{r:?}");
    }

    #[test]
    fn reclassify_budget_by_deadline() {
        // MAGI FIX (MS3 §6 loop12): a deadline that only covers the old reserve (a few seconds)
        // shortens the reclassification; a passed one gives it nothing; none keeps the sub-budget.
        let now = std::time::Instant::now();
        assert_eq!(reclassify_budget(None), BUDGET);
        let short = reclassify_budget(Some(now + Duration::from_secs(5)));
        assert!(short <= Duration::from_secs(5), "{short:?}");
        assert_eq!(reclassify_budget(Some(now)), Duration::ZERO);
        let far = reclassify_budget(Some(now + 10 * BUDGET));
        assert_eq!(far, BUDGET, "never above its own sub-budget");
    }

    #[tokio::test]
    async fn record_review_stops_reclassifying_when_its_budget_is_spent() {
        // [CP2 MS3 loop 8] Bounded by time, not only by count; an entry it did not reach keeps its place in the queue.
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let pending = |id: &str| HistoryEntry {
            status: HistoryStatus::Pending,
            ..attended(id, "d")
        };
        f.history.record(&[pending("old")]).expect("seed");
        let stats = record_review(
            &f.history,
            &[pending("n1")],
            "HEAD",
            &LineOneChanged,
            100,
            Duration::ZERO,
        )
        .expect("ok");
        assert_eq!(
            stats,
            RecordStats {
                reclassified: 0,
                recorded: 1,
                backlog: 1
            }
        );
        let checked: i64 = f
            .history
            .connection_for_test()
            .lock()
            .expect("lock")
            .query_row(
                "SELECT checked_at FROM code_review_history WHERE id = 'old'",
                [],
                |r| r.get(0),
            )
            .expect("row");
        assert_eq!(checked, 0, "not touched");
    }

    #[tokio::test]
    async fn a_documentation_read_error_mid_review_degrades_and_is_declared_once() {
        // [CP2 MS3 loop 8] The documentation mirror of the history case (docs_failed).
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        f.docs
            .connection_for_test()
            .lock()
            .expect("lock")
            .execute_batch("DROP TABLE code_review_docs;")
            .expect("drop");
        let mut rag = f.rag(true);
        // T3.3 (fix round 1) runs no query for an empty `allowed` set: the revision lists one source, so the search
        // reaches the dropped table.
        rag.doc_sources = BTreeSet::from([src("README.md", "k")]);
        let first = rag
            .context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("never fails the review");
        let second = rag
            .context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("never fails the review");
        let unavailable = (ContextKind::Documentation, "index unavailable".to_string());
        assert!(first.trimmed.contains(&unavailable), "{:?}", first.trimmed);
        assert!(!second.trimmed.contains(&unavailable));
        assert!(
            first.items.iter().any(|i| i.kind == ContextKind::Unit),
            "the code context stays"
        );
    }

    #[tokio::test]
    async fn documentation_of_another_revision_never_reaches_the_context() {
        // [RV21 D+B, 2026-10-20] RV-S36: both versions are indexed and equally similar; only the reviewed one is served.
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        f.docs
            .reindex(
                &[
                    (
                        src("README.md", "theirs"),
                        vec![chunk("README.md", "Usage", "WORKTREE-DOC")],
                    ),
                    (
                        src("README.md", "ours"),
                        vec![chunk("README.md", "Usage", "REVIEWED-DOC")],
                    ),
                ],
                &f.embedder,
            )
            .await
            .expect("index");
        let mut rag = f.rag(true);
        rag.doc_sources = BTreeSet::from([src("README.md", "ours")]);
        let a = rag
            .context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("context");
        let docs: Vec<&str> = a
            .items
            .iter()
            .filter(|i| i.kind == ContextKind::Documentation)
            .map(|i| i.text.as_str())
            .collect();
        assert!(docs.iter().any(|t| t.contains("REVIEWED-DOC")), "{docs:?}");
        assert!(docs.iter().all(|t| !t.contains("WORKTREE-DOC")), "{docs:?}");
    }

    /// The reviewed commit `c1` lists README.md at `k1`; with `false`, listing fails. [CP2 MS3 loop 10]
    struct CommitGit(bool);
    impl GitCommand for CommitGit {
        fn run_bytes(&self, args: &[&str]) -> Result<Vec<u8>, ReviewError> {
            match (self.0, args.join(" ").as_str()) {
                (true, "ls-tree -r -z c1") => {
                    Ok(b"100644 blob k1\tREADME.md\x00100644 blob k2\ta.rs\x00".to_vec())
                }
                (_, other) => Err(ReviewError::Git(other.to_string())),
            }
        }
    }

    #[tokio::test]
    async fn a_revision_that_reads_as_an_option_never_reaches_git() {
        // [MAGI S6 MS3 loop7, R2 Caspar] The listing is refused before git runs; git gets argv,
        // never a shell, so a newline in the revision is one argument, not two.
        struct Recording(Mutex<Vec<String>>);
        impl GitCommand for Recording {
            fn run_bytes(&self, a: &[&str]) -> Result<Vec<u8>, ReviewError> {
                self.0.lock().expect("lock").push(a.join(" "));
                Ok(Vec::new())
            }
        }
        let f = RagFixture::new(&[("README.md", "# A\nx\n")]).await;
        let git = Recording(Mutex::new(Vec::new()));
        for rev in ["--output=x", "-c", "--upload-pack=touch x"] {
            let prep = prepare_docs(
                &f.docs,
                Some(&git),
                Some(rev),
                &f.source,
                &[],
                &|_: &str| true,
                &f.registry,
                &f.embedder,
                BOUND,
            )
            .await;
            assert!(prep.docs_unavailable, "{rev}");
        }
        assert!(git.0.lock().expect("lock").is_empty(), "git ran");
    }

    #[test]
    fn the_chunk_bound_drops_whole_sources_never_part_of_one() {
        // [MAGI S6 MS3 loop7, R2 Caspar] U29 indexes a source whole or not at all: a source
        // whose chunks would cross MAX_DOC_CHUNKS goes entirely, the ones before stay entire.
        let many = |path: &str, n: usize| {
            let chunks = (0..n).map(|i| chunk(path, &format!("s{i}"), "x")).collect();
            (src(path, "k"), chunks)
        };
        let max = magi_rs::review::docs::MAX_DOC_CHUNKS;
        let (kept, truncated) = bounded(vec![many("a.md", max - 1), many("b.md", 2)]);
        assert!(truncated);
        let shape: Vec<(&str, usize)> = kept
            .iter()
            .map(|(s, c)| (s.path.as_str(), c.len()))
            .collect();
        assert_eq!(shape, vec![("a.md", max - 1)]);
    }

    #[test]
    fn cut_source_leaves_doc_sources() {
        // [MAGI S6 MS3 loop10, R2 Caspar] A source `bounded` cut stayed in `doc_sources`.
        let many = |path: &str, n: usize| {
            let chunks = (0..n).map(|i| chunk(path, &format!("s{i}"), "x")).collect();
            (src(path, "k"), chunks)
        };
        let mut prep = DocPrep {
            doc_sources: BTreeSet::from([src("a.md", "k"), src("b.md", "k")]),
            ..Default::default()
        };
        let max = magi_rs::review::docs::MAX_DOC_CHUNKS;
        let kept = bounded_into(&mut prep, vec![many("a.md", 1), many("b.md", max + 1)]);
        assert_eq!(kept.len(), 1);
        assert!(prep.doc_index.truncated);
        assert_eq!(prep.doc_sources, BTreeSet::from([src("a.md", "k")]));
    }

    #[tokio::test]
    async fn allowed_excludes_refused() {
        // MAGI FIX (MS3 §6 loop11, R1 carry): `doc_sources` is `load_vectors`' allowed set; a
        // source the store refused leaves it (the netted and bounded ones:
        // `a_source_whose_code_holds_a_key_block_is_withheld_whole`, `cut_source_leaves_doc_sources`;
        // `bounded_into` runs before every `index_batch` in `prepare_docs`).
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let bad = format!("{}.md", concat!("sk-ant-api03-", "AAAAAAAAAAAAAAAAAAAA")); // allow-secret-scan
        let batch = vec![
            (src(&bad, "k"), vec![chunk(&bad, "s", "x")]),
            (src("ok.md", "k"), vec![chunk("ok.md", "s", "y")]),
        ];
        let mut prep = DocPrep {
            doc_sources: batch.iter().map(|(s, _)| s.clone()).collect(),
            ..Default::default()
        };
        assert!(index_batch(&f.docs, &batch, &f.embedder, BOUND, &mut prep).await);
        assert_eq!(prep.doc_sources, BTreeSet::from([src("ok.md", "k")]));
        assert_eq!(prep.withheld_sources, vec![display_path(&bad)]);
    }

    /// Both sides by path, as `RevisionSource` serves a reviewed diff. [CP2 MS3 loop 10]
    struct TwoSides {
        new: Vec<(&'static str, &'static str)>,
        old: Vec<(&'static str, &'static str)>,
    }
    impl ContentSource for TwoSides {
        fn read(&self, p: &str) -> Result<Option<Vec<u8>>, ReviewError> {
            Ok(self
                .new
                .iter()
                .find(|(k, _)| *k == p)
                .map(|(_, v)| v.as_bytes().to_vec()))
        }
        fn read_old(&self, p: &str) -> Result<Option<Vec<u8>>, ReviewError> {
            Ok(self
                .old
                .iter()
                .find(|(k, _)| *k == p)
                .map(|(_, v)| v.as_bytes().to_vec()))
        }
    }

    const BOUND: Duration = Duration::from_secs(REINDEX_TIMEOUT_SECS);

    /// A working tree of one README whose bytes are not UTF-8.
    struct Latin1Readme;
    impl ContentSource for Latin1Readme {
        fn read(&self, p: &str) -> Result<Option<Vec<u8>>, ReviewError> {
            Ok((p == "README.md").then(|| b"# Caf\xe9\nnot utf-8\n".to_vec()))
        }
        fn read_old(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
            Ok(None)
        }
    }

    #[tokio::test]
    async fn an_unreadable_working_tree_source_is_declared_not_only_logged() {
        // [MAGI S6 MS3 loop7, R2 Caspar] It was left out with a WARN only; the report said nothing.
        struct Unreadable;
        impl ContentSource for Unreadable {
            fn read(&self, p: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Err(ReviewError::Io(format!("{p} is held open")))
            }
            fn read_old(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Ok(None)
            }
        }
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let prep = prepare_docs(
            &f.docs,
            None,
            None,
            &Unreadable,
            &["README.md".to_string()],
            &|_: &str| true,
            &f.registry,
            &f.embedder,
            BOUND,
        )
        .await;
        assert!(prep.doc_index.unreadable, "{prep:?}");
        assert!(!prep.docs_unavailable, "one file, never the whole index");
    }

    #[tokio::test]
    async fn worktree_doc_read_once() {
        // [MAGI S6 MS3 loop11, R5 Caspar] Verified: a working-tree source is read once; the
        // bytes its key hashes are the bytes chunked and embedded, never a second read.
        struct Counted(std::sync::atomic::AtomicUsize);
        impl ContentSource for Counted {
            fn read(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                let n = self.0.fetch_add(1, Ordering::SeqCst);
                Ok(Some(format!("# Setup\nread {n}\n").into_bytes()))
            }
            fn read_old(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Ok(None)
            }
        }
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let (source, embedder) = (Counted(0.into()), Recording(Default::default()));
        let prep = prepare_docs(
            &f.docs,
            None,
            None,
            &source,
            &["README.md".to_string()],
            &|_: &str| true,
            &f.registry,
            &embedder,
            BOUND,
        )
        .await;
        assert_eq!(source.0.load(Ordering::SeqCst), 1, "{prep:?}");
        let sent = embedder.0.lock().expect("lock").clone();
        assert!(sent.iter().all(|t| !t.contains("read 1")), "{sent:?}");
        assert!(sent.iter().any(|t| t.contains("read 0")), "{sent:?}");
    }

    #[tokio::test]
    async fn vanished_doc_is_unreadable() {
        // MAGI FIX (MS3 §6 loop11): a source gone between listing and read left in silence.
        struct Gone;
        impl ContentSource for Gone {
            fn read(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Ok(None)
            }
            fn read_old(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Ok(None)
            }
        }
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let prep = prepare_docs(
            &f.docs,
            None,
            None,
            &Gone,
            &["README.md".to_string()],
            &|_: &str| true,
            &f.registry,
            &f.embedder,
            BOUND,
        )
        .await;
        assert!(prep.doc_index.unreadable, "{prep:?}");
    }

    #[tokio::test]
    async fn unread_def_is_declared() {
        // [MAGI S6 MS3 loop9, R2 Melchior] A definition file that cannot be read failed the
        // batch's whole context; that item is left out and declared, the rest is served.
        struct HalfBroken;
        impl ContentSource for HalfBroken {
            fn read(&self, p: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                match p {
                    "a.rs" => Ok(Some(b"fn a() {\n    b();\n}\n".to_vec())),
                    _ => Err(ReviewError::Io("disk".into())),
                }
            }
            fn read_old(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Ok(None)
            }
        }
        let f =
            RagFixture::new(&[("a.rs", "fn a() {\n    b();\n}\n"), ("b.rs", "fn b() {}\n")]).await;
        let rag = StoreRag {
            sources: &HalfBroken,
            ..f.rag(false)
        };
        let a = rag.context_for(&batch_on("a.rs", "    b();")).await;
        let a = a.expect("the batch keeps its context");
        let cause = (
            ContextKind::Definition,
            format!("b.rs: {UNREADABLE_CONTEXT}"),
        );
        assert!(a.trimmed.contains(&cause), "{:?}", a.trimmed);
    }

    #[tokio::test]
    async fn missing_runs_under_the_deadline() {
        // [MAGI S6 MS3 loop9, R2 Caspar] The `missing` read ran past the bound unchecked; the
        // deadline is read again right after it.
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let prep = prepare_docs(
            &f.docs,
            None,
            None,
            &f.source,
            &[],
            &|_: &str| true,
            &f.registry,
            &f.embedder,
            Duration::ZERO,
        )
        .await;
        assert!(prep.docs_unavailable, "{prep:?}");
    }

    #[tokio::test]
    async fn a_working_tree_path_listed_twice_is_read_and_indexed_once() {
        // [MAGI S6 MS3 loop5, R2 Balthasar] Duplicate paths were each read, chunked and handed
        // to the reindex; the store is content-addressed, so it was waste, not a wrong index.
        struct Counting(std::sync::atomic::AtomicUsize);
        impl ContentSource for Counting {
            fn read(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(Some(b"# Usage\nrun it\n".to_vec()))
            }
            fn read_old(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Ok(None)
            }
        }
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let source = Counting(std::sync::atomic::AtomicUsize::new(0));
        let prep = prepare_docs(
            &f.docs,
            None,
            None,
            &source,
            &["README.md".to_string(), "README.md".to_string()],
            &|_: &str| true,
            &f.registry,
            &f.embedder,
            BOUND,
        )
        .await;
        assert_eq!(prep.doc_sources.len(), 1);
        assert_eq!(source.0.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_documentation_source_that_is_not_utf8_is_left_out_and_declared() {
        // [MAGI S6 MS3 loop5, R2 Caspar] It was removed with an INFO entry only, so the report
        // never said why that documentation was missing (RV07 declares what was cut).
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let prep = prepare_docs(
            &f.docs,
            None,
            None,
            &Latin1Readme,
            &["README.md".to_string()],
            &|_: &str| true,
            &f.registry,
            &f.embedder,
            BOUND,
        )
        .await;
        assert!(prep.doc_sources.is_empty(), "{:?}", prep.doc_sources);
        assert!(prep.doc_index.not_utf8);
        assert!(!prep.docs_unavailable);
    }

    #[tokio::test]
    async fn a_commit_review_indexes_and_serves_the_commits_documentation_not_the_working_trees() {
        // [CP2 MS3 loop 10] RV-S36 through the composition main.rs calls: the working tree's README says something else.
        // [CP2 MS3 loop 11] RV-S36 is about doc comments too: a.rs's comment differs between the commit and the tree.
        let f = RagFixture::new(&[
            ("a.rs", "/// WORKTREE-COMMENT\nfn a() {}\n"),
            ("README.md", "# Usage\nWORKTREE-DOC\n"),
        ])
        .await;
        let rev = TwoSides {
            new: vec![
                ("README.md", "# Usage\nREVIEWED-DOC\n"),
                ("a.rs", "/// REVIEWED-COMMENT\nfn a() {}\n"),
            ],
            old: vec![],
        };
        let prep = prepare_docs(
            &f.docs,
            Some(&CommitGit(true)),
            Some("c1"),
            &rev,
            &[],
            &|_: &str| true,
            &f.registry,
            &f.embedder,
            BOUND,
        )
        .await;
        assert_eq!(
            prep.doc_sources,
            BTreeSet::from([src("README.md", "k1"), src("a.rs", "k2")])
        );
        assert!(!prep.docs_unavailable);
        let mut rag = f.rag(true);
        rag.doc_sources = prep.doc_sources;
        let a = rag
            .context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("context");
        let docs: Vec<&str> = a
            .items
            .iter()
            .filter(|i| i.kind == ContextKind::Documentation)
            .map(|i| i.text.as_str())
            .collect();
        assert!(docs.iter().any(|t| t.contains("REVIEWED-DOC")), "{docs:?}");
        assert!(docs.iter().all(|t| !t.contains("WORKTREE-DOC")), "{docs:?}");
        assert!(
            docs.iter().any(|t| t.contains("REVIEWED-COMMENT")),
            "{docs:?}"
        );
        assert!(
            docs.iter().all(|t| !t.contains("WORKTREE-COMMENT")),
            "{docs:?}"
        );
    }

    #[tokio::test]
    async fn a_revision_source_over_the_size_cap_is_cut_and_left_out_of_the_searched_sources() {
        // [MAGI S6 MS3 loop6, R2 Melchior] The size cap cuts a source (declared `size_capped`):
        // like a not-UTF-8 or key-block source, it is no source of the revision's index, so it
        // leaves `doc_sources` and the search never asks for rows it was never given.
        struct CappedReadme;
        impl ContentSource for CappedReadme {
            fn read(&self, p: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                if p == "README.md" {
                    Err(ReviewError::TooLarge {
                        path: p.to_string(),
                        limit: 1,
                    })
                } else {
                    Ok(Some(b"/// Doc.\nfn a() {}\n".to_vec()))
                }
            }
            fn read_old(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Ok(None)
            }
        }
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let prep = prepare_docs(
            &f.docs,
            Some(&CommitGit(true)),
            Some("c1"),
            &CappedReadme,
            &[],
            &|_: &str| true,
            &f.registry,
            &f.embedder,
            BOUND,
        )
        .await;
        assert!(prep.doc_index.size_capped, "declared");
        assert!(!prep.docs_unavailable, "a cut, never a degradation");
        assert_eq!(prep.doc_sources, BTreeSet::from([src("a.rs", "k2")]));
    }

    #[tokio::test]
    async fn a_revision_source_git_lists_but_cannot_give_back_degrades_the_documentation() {
        // [MAGI S6 MS3 loop4, R2 Caspar] U29: the revision's index cannot be completed, so the
        // review serves no documentation (declared) rather than a partial set.
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let rev = TwoSides {
            new: vec![("a.rs", "/// REVIEWED-COMMENT\nfn a() {}\n")],
            old: vec![],
        };
        let prep = prepare_docs(
            &f.docs,
            Some(&CommitGit(true)),
            Some("c1"),
            &rev,
            &[],
            &|_: &str| true,
            &f.registry,
            &f.embedder,
            BOUND,
        )
        .await;
        assert!(prep.docs_unavailable, "{:?}", prep.doc_sources);
    }

    #[tokio::test]
    async fn a_working_tree_review_reads_only_the_files_that_can_hold_documentation() {
        // [Loop1 r2, Minor 1] Only Markdown doc sources and the languages whose doc comments are
        // chunked (`.rs`, `.py`) can yield a chunk: a Java file has an extractor but no doc
        // chunker, so it is neither read nor keyed for the documentation index.
        struct Counting<'a>(&'a TwoSides, Mutex<Vec<String>>);
        impl ContentSource for Counting<'_> {
            fn read(&self, p: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                self.1.lock().expect("lock").push(p.to_string());
                self.0.read(p)
            }
            fn read_old(&self, p: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                self.0.read_old(p)
            }
        }
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let tree = TwoSides {
            new: vec![
                ("README.md", "# Usage\nDOC\n"),
                ("a.rs", "/// A.\nfn a() {}\n"),
                ("m.py", "\"\"\"M.\"\"\"\n"),
                ("B.java", "/** B. */\nclass B {}\n"),
            ],
            old: vec![],
        };
        let source = Counting(&tree, Mutex::new(Vec::new()));
        let paths: Vec<String> = ["README.md", "a.rs", "m.py", "B.java"]
            .iter()
            .map(|p| p.to_string())
            .collect();
        let prep = prepare_docs(
            &f.docs,
            None,
            None,
            &source,
            &paths,
            &|_: &str| true,
            &f.registry,
            &f.embedder,
            BOUND,
        )
        .await;
        let listed: Vec<String> = prep.doc_sources.iter().map(|s| s.path.clone()).collect();
        assert_eq!(listed, vec!["README.md", "a.rs", "m.py"]);
        let reads = source.1.lock().expect("lock").clone();
        assert!(!reads.iter().any(|p| p == "B.java"), "{reads:?}");
    }

    #[tokio::test]
    async fn a_listing_that_fails_leaves_no_documentation_and_says_so() {
        // [CP2 MS3 loop 10] RV-S38's git trigger: degradation (B), never the working tree's documentation instead.
        let f = RagFixture::new(&[("README.md", "# Usage\nWORKTREE-DOC\n")]).await;
        let prep = prepare_docs(
            &f.docs,
            Some(&CommitGit(false)),
            Some("c1"),
            &f.source,
            &["README.md".to_string()],
            &|_: &str| true,
            &f.registry,
            &f.embedder,
            BOUND,
        )
        .await;
        assert!(prep.docs_unavailable);
        assert!(prep.doc_sources.is_empty());
    }

    #[tokio::test]
    async fn a_git_failure_is_declared_once_as_an_unavailable_index_never_as_no_embedder() {
        // [MAGI S6 MS3 loop3, R6b Melchior] RV21: a revision that cannot be indexed is declared,
        // and accurately. A git failure is no missing embedder: the report says the index was
        // unavailable, once, and `without_embedder` stays false (the embedder answered).
        let f = RagFixture::new(&[("README.md", "# Usage\nWORKTREE-DOC\n")]).await;
        let prep = prepare_docs(
            &f.docs,
            Some(&CommitGit(false)),
            Some("c1"),
            &f.source,
            &["README.md".to_string()],
            &|_: &str| true,
            &f.registry,
            &f.embedder,
            BOUND,
        )
        .await;
        let rag = f.rag(true).with_prep(prep);
        let mut declared = Vec::new();
        for _ in 0..2 {
            let a = rag
                .context_for(&batch_on("README.md", "x"))
                .await
                .expect("context");
            assert!(!a.without_embedder, "{:?}", a.trimmed);
            declared.extend(
                a.trimmed
                    .into_iter()
                    .filter(|(kind, _)| *kind == ContextKind::Documentation),
            );
        }
        assert_eq!(
            declared,
            vec![(ContextKind::Documentation, INDEX_UNAVAILABLE.to_string())]
        );
    }

    #[tokio::test]
    async fn a_working_tree_review_keys_its_sources_by_their_bytes() {
        // [CP2 MS3 loop 10] No revision: the working tree is listed and read, never git.
        // [CP2 MS3 loop 11] A path that quotes a secret is never stored: the source's path sits in a clear column.
        let secret_path = "sk-ant-api03-AAAAAAAAAAAAAAAAAAAA.md"; // allow-secret-scan
        let f = RagFixture::new(&[
            ("README.md", "# Usage\nWORKTREE-DOC\n"),
            (secret_path, "# Notes\nok\n"),
        ])
        .await;
        let prep = prepare_docs(
            &f.docs,
            None,
            None,
            &f.source,
            &["README.md".to_string(), secret_path.to_string()],
            &|_: &str| true,
            &f.registry,
            &f.embedder,
            BOUND,
        )
        .await;
        assert_eq!(
            prep.doc_sources,
            BTreeSet::from([src(
                "README.md",
                &magi_rs::review::docs::worktree_key(b"# Usage\nWORKTREE-DOC\n")
            )])
        );
        assert_eq!(prep.doc_index.withheld, 1);
        // [CP2 MS3 loop 13] Named for RV22's skipped list as display_path: the raw path (it holds the key) never leaves.
        assert_eq!(
            prep.withheld_sources,
            vec![magi_rs::review::secrets::display_path(secret_path)]
        );
        assert!(
            prep.withheld_sources.iter().all(|p| !p.contains("sk-ant-")),
            "{:?}",
            prep.withheld_sources
        );
        assert!(!prep.docs_unavailable);
    }

    #[tokio::test]
    async fn a_source_with_a_one_line_secret_is_indexed_masked_even_read_raw() {
        // [MAGI S6 MS3 loop1, R1 all three] RV22 amended: the documentation chunks are masked,
        // not withheld, whatever the source they were read through (here one that masks nothing),
        // no byte of the key reaches the embedder, and the file is declared as masked.
        let readme = "# Setup\nexport KEY=sk-ant-api03-AAAAAAAAAAAAAAAAAAAA\n# Other\nfine\n"; // allow-secret-scan
        let f = RagFixture::new(&[("README.md", readme)]).await;
        let embedder = Recording(std::sync::Mutex::new(Vec::new()));
        let prep = prepare_docs(
            &f.docs,
            None,
            None,
            &f.source,
            &["README.md".to_string()],
            &|_: &str| true,
            &f.registry,
            &embedder,
            BOUND,
        )
        .await;
        assert!(
            prep.withheld_sources.is_empty(),
            "{:?}",
            prep.withheld_sources
        );
        assert_eq!(prep.doc_sources.len(), 1, "{:?}", prep.doc_sources);
        assert_eq!(
            prep.masked_sources,
            vec![("README.md".to_string(), "anthropic-key")]
        );
        let sent = embedder
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        assert!(sent.iter().all(|t| !t.contains("sk-ant-")), "{sent:?}");
        assert!(
            sent.iter().any(|t| t.contains("<redacted: anthropic-key>")),
            "{sent:?}"
        );
        assert!(sent.iter().any(|t| t.contains("fine")), "{sent:?}");
    }

    #[tokio::test]
    async fn a_source_masked_by_an_earlier_review_is_served_with_its_mask() {
        // [MAGI S6 MS3 loop6, R2 Caspar] A second review finds the source indexed, so its
        // `masked_sources` is empty; what it serves still carries the mask, which is what the
        // pipeline declares the file as masked by (`masked_store_item`).
        let readme = "# Setup\nexport KEY=sk-ant-api03-AAAAAAAAAAAAAAAAAAAA\n"; // allow-secret-scan
        let f = RagFixture::new(&[("README.md", readme), ("a.rs", "fn a() {}\n")]).await;
        let (paths, keep) = (["README.md".to_string()], |_: &str| true);
        let prepare = || {
            prepare_docs(
                &f.docs,
                None,
                None,
                &f.source,
                &paths,
                &keep,
                &f.registry,
                &f.embedder,
                BOUND,
            )
        };
        assert_eq!(prepare().await.masked_sources.len(), 1, "the first review");
        let second = prepare().await;
        assert!(
            second.masked_sources.is_empty(),
            "{:?}",
            second.masked_sources
        );
        let a = f
            .rag(true)
            .with_prep(second)
            .context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("context");
        let docs: Vec<&ContextItem> = a
            .items
            .iter()
            .filter(|i| i.kind == ContextKind::Documentation)
            .collect();
        assert!(!docs.is_empty(), "the source is served");
        assert!(
            docs.iter()
                .all(|i| !i.text.contains("sk-ant-")
                    && masked_pattern(&i.text) == Some("anthropic-key")),
            "{docs:?}"
        );
    }

    #[tokio::test]
    async fn a_source_whose_code_holds_a_key_block_is_withheld_whole() {
        // [MAGI S6 MS3 loop1] RV22's exception covers the file, not the chunk: a key block in a
        // Rust file's code (never a chunk itself) still keeps its doc comments out.
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        let code = format!("/// Loads it.\nfn load() {{}}\nconst K: &str = \"{begin}\\nQQQQ\";\n");
        let f = RagFixture::new(&[("a.rs", code.as_str())]).await;
        let prep = prepare_docs(
            &f.docs,
            None,
            None,
            &f.source,
            &["a.rs".to_string()],
            &|_: &str| true,
            &f.registry,
            &f.embedder,
            BOUND,
        )
        .await;
        assert!(prep.doc_sources.is_empty(), "{:?}", prep.doc_sources);
        assert_eq!(prep.withheld_sources, vec!["a.rs".to_string()]);
        assert_eq!(f.embedder.calls(), 0, "no chunk of it was embedded");
    }

    #[tokio::test]
    async fn a_markdown_source_with_a_key_block_is_omitted_not_masked_and_never_embedded() {
        // [MAGI S6 MS3 loop17, R5 Caspar] The documentation path's contract is OMISSION for a
        // multi-line private-key block (RV22's exception), not masking: the body has no pattern
        // of its own, so masking could not cover it. The embedder's request proves it: neither
        // the key's body nor its clean sibling section is sent.
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        let end = concat!("-----", "END ", "PRIVATE KEY-----");
        let body = "MIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8w";
        let text = format!("# Keys\n{begin}\n{body}\n{end}\n# Other\nfine sibling text\n");
        let f = RagFixture::new(&[("README.md", text.as_str())]).await;
        let embedder = Recording(std::sync::Mutex::new(Vec::new()));
        let prep = prepare_docs(
            &f.docs,
            None,
            None,
            &f.source,
            &["README.md".to_string()],
            &|_: &str| true,
            &f.registry,
            &embedder,
            BOUND,
        )
        .await;
        let sent = embedder.0.lock().expect("lock").clone();
        assert!(sent.is_empty(), "nothing of it was embedded: {sent:?}");
        assert_eq!(prep.withheld_sources, vec!["README.md".to_string()]);
        assert!(prep.doc_sources.is_empty(), "{:?}", prep.doc_sources);
        assert!(prep.masked_sources.is_empty(), "omitted, never masked");
    }

    /// Records every text it is asked to embed. [T3.7]
    struct Recording(std::sync::Mutex<Vec<String>>);
    #[async_trait::async_trait]
    impl EmbeddingProvider for Recording {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            self.0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .extend(texts.iter().cloned());
            Ok(texts.iter().map(|_| vec![0.5; 3]).collect())
        }
        fn model_id(&self) -> &str {
            "fake"
        }
        fn dim(&self) -> usize {
            3
        }
        fn query_prefix(&self) -> &str {
            ""
        }
        fn document_prefix(&self) -> &str {
            ""
        }
    }

    #[tokio::test]
    async fn a_batch_quoting_a_one_line_secret_sends_its_documentation_query_masked() {
        // [MAGI S6 MS3 loop3, R2 Melchior (critical), Caspar] `query_vector` is an egress point
        // (RV22 amended): it masks the query itself, so no byte of a one-line secret in the
        // batch reaches the embedder even when the batch text arrives raw (upstream, the review's
        // cache masks it already: `no_byte_of_a_one_line_secret_reaches_the_embedder_or_the_trio`
        // in the runner).
        let secret = "sk-ant-api03-AAAAAAAAAAAAAAAAAAAA"; // allow-secret-scan
        let code = format!("const K: &str = \"{secret}\";\n");
        let f = RagFixture::new(&[("a.rs", code.as_str())]).await;
        let embedder = Recording(std::sync::Mutex::new(Vec::new()));
        let mut rag = f.rag(true);
        rag.embedder = Some(&embedder);
        rag.context_for(&batch_on("a.rs", code.trim_end()))
            .await
            .expect("context");
        let sent = embedder
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        assert_eq!(sent.len(), 1, "one query: {sent:?}");
        assert!(sent.iter().all(|t| !t.contains("sk-ant-")), "{sent:?}");
        assert!(
            sent.iter().all(|t| t.contains("<redacted: anthropic-key>")),
            "{sent:?}"
        );
    }

    /// An embedder whose query prefix is `0`; records what it is asked to embed in `1`.
    struct KeyPrefix(String, std::sync::Mutex<Vec<String>>);
    #[async_trait::async_trait]
    impl EmbeddingProvider for KeyPrefix {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            self.1
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .extend(texts.iter().cloned());
            Ok(texts.iter().map(|_| vec![0.5; 3]).collect())
        }
        fn model_id(&self) -> &str {
            "fake"
        }
        fn dim(&self) -> usize {
            3
        }
        fn query_prefix(&self) -> &str {
            &self.0
        }
        fn document_prefix(&self) -> &str {
            ""
        }
    }

    #[tokio::test]
    async fn key_prefix_sends_no_query() {
        // MAGI FIX (MS3 §6 loop11): a key block in `[embedding]`'s query prefix left raw.
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        let end = concat!("-----", "END ", "PRIVATE KEY-----");
        let prefix = format!("{begin}\nMIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8w\n{end}\n");
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let embedder = KeyPrefix(prefix, std::sync::Mutex::new(Vec::new()));
        let mut rag = f.rag(true);
        rag.embedder = Some(&embedder);
        rag.context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("context");
        let sent = embedder.1.lock().expect("lock").clone();
        assert!(sent.is_empty(), "nothing leaves: {sent:?}");
        assert!(rag.embedder_down.load(Ordering::SeqCst), "declared down");
    }

    #[tokio::test]
    async fn a_withheld_prefix_is_not_an_unanswered_embedder() {
        // MAGI FIX (MS3 §6 loop14): the report said "the embedder did not answer" for a query that
        // was never sent because its prefix holds a key block.
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        let end = concat!("-----", "END ", "PRIVATE KEY-----");
        let prefix = format!("{begin}\nMIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8w\n{end}\n");
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let embedder = KeyPrefix(prefix, std::sync::Mutex::new(Vec::new()));
        let mut rag = f.rag(true);
        rag.embedder = Some(&embedder);
        let out = rag
            .context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("context");
        assert!(
            out.trimmed
                .iter()
                .all(|(_, label)| label != "the embedder did not answer"),
            "{:?}",
            out.trimmed
        );
        assert!(
            out.trimmed
                .iter()
                .any(|(kind, label)| *kind == ContextKind::Documentation
                    && label.contains("query prefix")),
            "{:?}",
            out.trimmed
        );
    }

    #[tokio::test]
    async fn a_documentation_chunk_read_through_the_review_cache_is_embedded_masked() {
        // [T3.7] RV22 amended: the review's cache masks every secret it holds, so a README with
        // a key is indexed with the key masked, never withheld, and no byte of the key reaches
        // the embedder.
        let password = "S3cr3t";
        let readme =
            format!("# Setup\nDATABASE_URL=postgres://svc:{password}@db/app\n# Other\nfine\n");
        let f = RagFixture::new(&[("README.md", readme.as_str())]).await;
        let cached = magi_rs::review::source::CachedSource::new(&f.source);
        let embedder = Recording(std::sync::Mutex::new(Vec::new()));
        let prep = prepare_docs(
            &f.docs,
            None,
            None,
            &cached,
            &["README.md".to_string()],
            &|_: &str| true,
            &f.registry,
            &embedder,
            BOUND,
        )
        .await;
        assert!(
            prep.withheld_sources.is_empty(),
            "{:?}",
            prep.withheld_sources
        );
        assert_eq!(prep.doc_sources.len(), 1, "{:?}", prep.doc_sources);
        // [MAGI S6 MS3 loop1] Masked by the cache, declared all the same.
        assert_eq!(
            prep.masked_sources,
            vec![("README.md".to_string(), "url-credential")]
        );
        let sent = embedder
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        assert!(!sent.is_empty());
        assert!(sent.iter().all(|t| !t.contains(password)), "{sent:?}");
        assert!(
            sent.iter()
                .any(|t| t.contains("<redacted: url-credential>")),
            "{sent:?}"
        );
    }

    #[tokio::test]
    async fn a_documentation_file_holding_a_key_block_is_withheld_whole_through_the_cache() {
        // [T3.7-fix2 round 4, R3-4] The cache holds a key-block side raw (RV22's exception), so the
        // documentation path must be the one that withholds it: the whole source, the clean
        // sections included, and the embedder never sees a byte of it.
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        let end = concat!("-----", "END ", "PRIVATE KEY-----");
        let readme = format!(
            "# Keys\n{begin}\nMIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8w\nQQQQ\n{end}\n# Other\nCLEAN-SECTION\n"
        );
        let f = RagFixture::new(&[("README.md", readme.as_str())]).await;
        let cached = magi_rs::review::source::CachedSource::new(&f.source);
        let embedder = Recording(std::sync::Mutex::new(Vec::new()));
        let prep = prepare_docs(
            &f.docs,
            None,
            None,
            &cached,
            &["README.md".to_string()],
            &|_: &str| true,
            &f.registry,
            &embedder,
            BOUND,
        )
        .await;
        assert!(prep.doc_sources.is_empty(), "{:?}", prep.doc_sources);
        assert_eq!(prep.withheld_sources, vec!["README.md".to_string()]);
        let sent = embedder
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        assert!(
            sent.iter().all(|t| !t.contains("MIIB")
                && !t.contains("QQQQ")
                && !t.contains("PRIVATE KEY")
                && !t.contains("CLEAN-SECTION")),
            "{sent:?}"
        );
    }

    /// README.md reads; every other path fails the way MS2 skips a listed file. [MS3 final review I2]
    struct SomeUnreadable;
    impl ContentSource for SomeUnreadable {
        fn read(&self, p: &str) -> Result<Option<Vec<u8>>, ReviewError> {
            match p {
                "README.md" => Ok(Some(b"# Usage\nok\n".to_vec())),
                "docs/linked.md" => Err(ReviewError::NotReviewable {
                    path: p.into(),
                    cause: magi_rs::review::source::HARD_LINK_CAUSE,
                }),
                "docs/escape.md" => Err(ReviewError::OutsideWorkspace(p.into())),
                "docs/dir.md" => Err(ReviewError::NotRegularFile(p.into())),
                _ => Err(ReviewError::Io(format!(
                    "{p}: held open by another process"
                ))),
            }
        }
        fn read_old(&self, _p: &str) -> Result<Option<Vec<u8>>, ReviewError> {
            Ok(None)
        }
    }

    #[tokio::test]
    async fn an_unreadable_working_tree_file_is_skipped_and_the_rest_is_indexed() {
        // [MS3 final review I2] One hard link, escaping link, directory link or locked file never
        // costs the whole working tree its documentation.
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let paths: Vec<String> = [
            "README.md",
            "docs/linked.md",
            "docs/escape.md",
            "docs/dir.md",
            "docs/locked.md",
        ]
        .map(String::from)
        .to_vec();
        let prep = prepare_docs(
            &f.docs,
            None,
            None,
            &SomeUnreadable,
            &paths,
            &|_: &str| true,
            &f.registry,
            &f.embedder,
            BOUND,
        )
        .await;
        assert!(!prep.docs_unavailable);
        assert_eq!(
            prep.doc_sources,
            BTreeSet::from([src(
                "README.md",
                &magi_rs::review::docs::worktree_key(b"# Usage\nok\n")
            )])
        );
    }

    /// Answers the first embed request and fails every later one. [CP2 MS3 loop 13]
    struct FailsAfterFirst(std::sync::atomic::AtomicUsize);
    #[async_trait::async_trait]
    impl EmbeddingProvider for FailsAfterFirst {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            if self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                Ok(texts.iter().map(|_| vec![0.5; 3]).collect())
            } else {
                Err(EmbeddingError::Http("down".into()))
            }
        }
        fn model_id(&self) -> &str {
            "fake"
        } // the fixture store's model: it passes the index-id gate
        fn dim(&self) -> usize {
            3
        }
        fn query_prefix(&self) -> &str {
            ""
        }
        fn document_prefix(&self) -> &str {
            ""
        }
    }

    #[tokio::test]
    async fn indexing_progress_survives_a_later_failure() {
        // [CP2 MS3 loop 13] One reindex per batch of sources: a large repository completes its index over several reviews.
        use crate::memory::review_docs::EMBED_BATCH_ITEMS;
        let big: String = (0..EMBED_BATCH_ITEMS)
            .map(|i| format!("# S{i}\nx{i}\n"))
            .collect();
        // CLAUDE.md (1 chunk) sorts first and is the first batch; README.md (EMBED_BATCH_ITEMS chunks) is the second.
        let f = RagFixture::new(&[("CLAUDE.md", "# B\ny\n"), ("README.md", big.as_str())]).await;
        let down = FailsAfterFirst(Default::default());
        let prep = prepare_docs(
            &f.docs,
            None,
            None,
            &f.source,
            &["CLAUDE.md".to_string(), "README.md".to_string()],
            &|_: &str| true,
            &f.registry,
            &down,
            BOUND,
        )
        .await;
        assert!(
            prep.docs_unavailable,
            "a source of the review is still missing: B for this review"
        );
        let claude = src(
            "CLAUDE.md",
            &magi_rs::review::docs::worktree_key(b"# B\ny\n"),
        );
        let readme = src(
            "README.md",
            &magi_rs::review::docs::worktree_key(big.as_bytes()),
        );
        assert_eq!(
            f.docs.missing(&[claude, readme.clone()]).expect("missing"),
            vec![readme],
            "the first batch stays written"
        );
    }

    #[tokio::test]
    async fn with_no_room_left_content_already_out_for_another_cause_is_not_declared_again() {
        // [MAGI S6 MS3 loop4, R2 Melchior] After the embedder went down, or the history read
        // failed (each declared once with its own cause), a batch with no room must not add
        // "no room left in the batch" for content that is out regardless of room.
        let f = RagFixture::new(&[("notes.txt", "plain\n")]).await;
        let b = batch_on("notes.txt", "plain");
        let tight = b.tokens
            + magi_rs::review::window::estimate_tokens(magi_rs::review::pipeline::CONTEXT_HEADING);
        let mut down = f.rag(true);
        down.limit_tokens = tight;
        down.embedder_down = AtomicBool::new(true);
        let a = down.context_for(&b).await.expect("context");
        assert!(
            a.trimmed.iter().all(|(_, label)| label != NO_ROOM),
            "{:?}",
            a.trimmed
        );
        let mut failed = f.rag(true);
        failed.limit_tokens = tight;
        failed.history_failed = AtomicBool::new(true);
        let a = failed.context_for(&b).await.expect("context");
        assert!(
            !a.trimmed
                .iter()
                .any(|(kind, label)| *kind == ContextKind::History && label == NO_ROOM),
            "{:?}",
            a.trimmed
        );
    }

    #[tokio::test]
    async fn a_withheld_batch_is_not_down() {
        // MAGI FIX (MS3 §6 loop12): a key block in the BATCH text withholds that batch's query
        // only; it was declared as a down embedder and left the whole review without RAG.
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        let end = concat!("-----", "END ", "PRIVATE KEY-----");
        let key = format!("{begin}\nMIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8w\n{end}");
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let rag = f.rag(true);
        let a = rag
            .context_for(&batch_on("a.rs", &key))
            .await
            .expect("context");
        assert_eq!(f.embedder.calls(), 0, "nothing was sent");
        assert!(!rag.embedder_down.load(Ordering::SeqCst), "not down");
        assert!(
            a.trimmed.iter().all(|(_, label)| label != EMBEDDER_DOWN),
            "{:?}",
            a.trimmed
        );
        rag.context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("context");
        assert!(f.embedder.calls() > 0, "the next batch is still served");
    }

    #[test]
    fn a_unit_keeps_one_numbered_line_per_source_line_over_escaped_text() {
        // [MAGI S6 MS3 loop16, R2 Balthasar] RV09 accepts the number shown for a containing
        // unit: escaping the frame (fences, tags) must neither add nor drop a line.
        let item = ContextItem {
            kind: ContextKind::Unit,
            file: "a.rs".into(),
            label: "f".into(),
            text: "fn f() {\n    // ```\n    // </code>\n}".into(),
            candidate: false,
            lines: Some((10, 13)),
            old_side: false,
        };
        let rendered = escaped(item).rendered();
        let numbers: Vec<&str> = rendered
            .lines()
            .map(|l| l.trim_start().split(|c: char| !c.is_ascii_digit()).next())
            .map(|n| n.unwrap_or(""))
            .collect();
        assert_eq!(numbers, ["10", "11", "12", "13"], "{rendered}");
    }

    #[tokio::test]
    async fn a_withheld_batch_declares_its_own_cause_and_is_not_a_missing_embedder() {
        // [MAGI S6 MS3 loop16, R2 Caspar/Melchior] A batch withheld by its own text was reported
        // as `without_embedder` with no cause at all: the embedder answered nothing because it
        // was never asked, which the report must say, per batch and apart from a down embedder.
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        let end = concat!("-----", "END ", "PRIVATE KEY-----");
        let key = format!("{begin}\nMIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8w\n{end}");
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let rag = f.rag(true);
        let a = rag
            .context_for(&batch_on("a.rs", &key))
            .await
            .expect("context");
        assert!(!a.without_embedder, "the embedder is up: {:?}", a.trimmed);
        for kind in [ContextKind::Documentation, ContextKind::History] {
            assert!(
                a.trimmed
                    .iter()
                    .any(|(k, label)| *k == kind && label.contains("private-key block")),
                "{kind:?} declared: {:?}",
                a.trimmed
            );
        }
        let next = rag
            .context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("context");
        assert!(
            next.trimmed
                .iter()
                .all(|(_, label)| !label.contains("private-key block")),
            "the withhold does not outlive its batch: {:?}",
            next.trimmed
        );
    }

    #[tokio::test]
    async fn without_a_git_and_without_documentation_no_query_is_embedded() {
        // [MAGI S6 MS3 loop17, R2 Melchior/Balthasar] A no-git review still embedded a query and
        // read the history store per batch for entries `served_on_heads_line` always discards.
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let mut rag = f.rag(true);
        rag.docs = None;
        rag.history_git = None;
        let a = rag
            .context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("context");
        assert_eq!(f.embedder.calls(), 0, "nothing could be served");
        assert!(
            a.trimmed
                .iter()
                .any(|(k, label)| *k == ContextKind::History && label == HISTORY_NO_GIT),
            "the cause is still declared: {:?}",
            a.trimmed
        );
    }

    #[tokio::test]
    async fn a_review_without_git_declares_that_history_was_not_served() {
        // [MAGI S6 MS3 loop16, R2 Caspar] Without a git no entry can be checked against HEAD, so
        // none is served; that was said nowhere.
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let mut rag = f.rag(true);
        rag.history_git = None;
        let a = rag
            .context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("context");
        assert!(
            a.trimmed
                .iter()
                .any(|(k, label)| *k == ContextKind::History && label.contains("no git")),
            "{:?}",
            a.trimmed
        );
        let again = rag
            .context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("context");
        assert!(
            again
                .trimmed
                .iter()
                .all(|(_, label)| !label.contains("no git")),
            "declared once: {:?}",
            again.trimmed
        );
    }

    #[tokio::test]
    async fn corrupt_history_rows_are_declared_per_review_not_per_store() {
        // [MAGI S6 MS3 loop16, R2 Balthasar] `corrupt_rows` counts for the store's lifetime, so in
        // the TUI a row skipped by an earlier review made every later review declare it.
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        f.history.record(&[attended("bad", "d")]).expect("record");
        f.history
            .connection_for_test()
            .lock()
            .expect("lock")
            .execute(
                "UPDATE code_review_history SET finding_blob = x'00' WHERE id = 'bad'",
                [],
            )
            .expect("corrupt");
        f.history
            .validated_on_files(&["a.rs".to_string()])
            .expect("an earlier review reads it");
        assert!(f.history.corrupt_rows() > 0);
        f.history
            .connection_for_test()
            .lock()
            .expect("lock")
            .execute("DELETE FROM code_review_history", [])
            .expect("delete");
        let rag = f.rag(true);
        let a = rag
            .context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("context");
        assert!(
            a.trimmed.iter().all(|(_, label)| label != CORRUPT_HISTORY),
            "this review skipped none: {:?}",
            a.trimmed
        );
    }

    #[tokio::test]
    async fn homonym_definitions_leave_room_for_documentation() {
        // [MAGI S6 MS3 loop16, R2 Caspar/Melchior/Balthasar] A batch naming `new` built every
        // definition of it, charged them all, and the room check then skipped documentation and
        // history for the batch (`no room left`), the layers MS3 exists to add.
        let body = format!("fn new() {{\n{}}}\n", "    let x = 1;\n".repeat(30));
        let files: Vec<(String, String)> = (0..40)
            .map(|i| (format!("f{i:02}.rs"), body.clone()))
            .chain([("a.rs".to_string(), "fn a() {\n    new();\n}\n".to_string())])
            .collect();
        let borrowed: Vec<(&str, &str)> = files
            .iter()
            .map(|(p, t)| (p.as_str(), t.as_str()))
            .collect();
        let f = RagFixture::new(&borrowed).await;
        f.docs
            .reindex(
                &[(
                    src("README.md", "k1"),
                    vec![chunk("README.md", "A", "about new")],
                )],
                &f.embedder,
            )
            .await
            .expect("indexed");
        let mut rag = f.rag(true);
        rag.doc_sources = BTreeSet::from([src("README.md", "k1")]);
        rag.limit_tokens = 1_800;
        let a = rag
            .context_for(&batch_on("a.rs", "    new();"))
            .await
            .expect("context");
        assert!(
            a.items.iter().any(|i| i.kind == ContextKind::Documentation),
            "{:?}",
            a.trimmed
        );
        assert!(
            a.trimmed
                .iter()
                .any(|(k, label)| *k == ContextKind::Definition && label.contains("new")),
            "the cut is declared: {:?}",
            a.trimmed
        );
    }

    #[tokio::test]
    async fn no_docs_no_room_note() {
        // MAGI FIX (MS3 §6 loop12): a review that indexed no documentation source declared
        // `no room` for a layer that had nothing to retrieve.
        let f = RagFixture::new(&[("notes.txt", "plain\n")]).await;
        let b = batch_on("notes.txt", "plain");
        let mut rag = f.rag(true);
        rag.limit_tokens = b.tokens
            + magi_rs::review::window::estimate_tokens(magi_rs::review::pipeline::CONTEXT_HEADING);
        let a = rag.context_for(&b).await.expect("context");
        assert!(
            !a.trimmed
                .iter()
                .any(|(kind, label)| *kind == ContextKind::Documentation && label == NO_ROOM),
            "{:?}",
            a.trimmed
        );
    }

    #[tokio::test]
    async fn with_no_room_left_the_documentation_query_is_not_embedded() {
        // [CP2 MS3 loop 13] Nothing could enter: no embed call and no search are paid for it.
        let f = RagFixture::new(&[("notes.txt", "plain\n")]).await;
        let mut rag = f.rag(true);
        let b = batch_on("notes.txt", "plain");
        rag.limit_tokens = b.tokens
            + magi_rs::review::window::estimate_tokens(magi_rs::review::pipeline::CONTEXT_HEADING);
        rag.context_for(&b).await.expect("context");
        assert_eq!(f.embedder.calls(), 0);
    }

    #[test]
    fn the_outbound_reserve_covers_the_dimension_probe() {
        // [MAGI S6 MS3 loop1, R2 Melchior] `doc_store` may embed one probe (under
        // EMBED_QUERY_TIMEOUT_SECS) before the indexing: the reserve the budget check holds back
        // must cover it, or a run deadline can be overshot by up to that bound.
        assert_eq!(
            outbound_prep_reserve_secs(2, true),
            REINDEX_TIMEOUT_SECS + EMBED_QUERY_TIMEOUT_SECS + 2 * EMBED_QUERY_TIMEOUT_SECS
        );
        assert_eq!(
            outbound_prep_reserve_secs(2, false),
            0,
            "no call, no reserve"
        );
    }

    #[tokio::test]
    async fn the_two_stage_four_reserves_differ_only_by_what_runs_between_the_checks() {
        // [MAGI S6 MS3 loop4, R2 Melchior] Two budget checks, one bound: the runner checks with
        // `outbound_prep_reserve_secs` before the dimension probe and the indexing; the pipeline
        // checks again with `ReviewRag::outbound_reserve_secs` after them, when only the query
        // embeddings are left. Both count those queries with the same `query_reserve_secs`.
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let with = ReviewRag {
            docs: Some(&f.docs),
            history: Some(&f.history),
            registry: &f.registry,
            embedder: Some(&f.embedder),
            limit_tokens: 100_000,
            prep: DocPrep::default(),
            history_git: None,
        };
        let without = ReviewRag {
            embedder: None,
            prep: DocPrep::default(),
            ..with
        };
        for batches in [1, 3] {
            assert_eq!(
                with.outbound_reserve_secs(batches),
                outbound_prep_reserve_secs(batches, true)
                    - DIMENSION_PROBE_TIMEOUT_SECS
                    - REINDEX_TIMEOUT_SECS
            );
            assert_eq!(
                without.outbound_reserve_secs(batches),
                outbound_prep_reserve_secs(batches, false)
            );
        }
    }

    /// Answers two vectors for every text: a provider that does not keep one vector per input.
    struct TwoVectors;
    #[async_trait::async_trait]
    impl EmbeddingProvider for TwoVectors {
        async fn embed(
            &self,
            texts: &[String],
        ) -> Result<Vec<Vec<f32>>, crate::memory::error::EmbeddingError> {
            Ok(texts
                .iter()
                .flat_map(|_| [vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]])
                .collect())
        }
        fn model_id(&self) -> &str {
            "fake"
        }
        fn dim(&self) -> usize {
            3
        }
        fn query_prefix(&self) -> &str {
            ""
        }
        fn document_prefix(&self) -> &str {
            ""
        }
    }

    #[tokio::test]
    async fn a_query_answered_with_other_than_one_vector_is_a_down_embedder() {
        // [MAGI S6 MS3 loop2, R2 Caspar] One query, one vector: any other count says the answer
        // cannot be matched to the query, so no vector is guessed at (the last one was taken).
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let two = TwoVectors;
        let mut rag = f.rag(true);
        rag.embedder = Some(&two);
        let a = rag
            .context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("never fails the review");
        assert!(a.without_embedder, "{:?}", a.trimmed);
    }

    #[tokio::test]
    async fn an_unavailable_index_is_never_declared_as_lack_of_room() {
        // [MAGI S6 MS3 loop2, R2 Balthasar] With the documentation index unavailable, a full batch
        // declared it cut for room too, a second cause that was not the reason.
        let f = RagFixture::new(&[("notes.txt", "plain\n")]).await;
        let mut rag = f.rag(true);
        rag.docs_unavailable = true;
        let b = batch_on("notes.txt", "plain");
        rag.limit_tokens = b.tokens + estimate_tokens(CONTEXT_HEADING);
        let a = rag.context_for(&b).await.expect("context");
        let docs: Vec<&str> = a
            .trimmed
            .iter()
            .filter(|(kind, _)| *kind == ContextKind::Documentation)
            .map(|(_, label)| label.as_str())
            .collect();
        assert_eq!(docs, vec![INDEX_UNAVAILABLE], "{:?}", a.trimmed);
    }

    #[tokio::test]
    async fn rows_deleted_mid_review_are_a_declared_degradation() {
        // [MAGI S6 MS3 loop15, R1 Caspar] another process's clean-up deleted loaded rows
        // mid-review and the hits were silently fewer; it is now declared once, like any other
        // loss of the documentation index.
        let f = RagFixture::new(&[]).await;
        f.docs
            .reindex(
                &[
                    (src("a.md", "k1"), vec![chunk("a.md", "A", "x")]),
                    (src("b.md", "k2"), vec![chunk("b.md", "B", "y")]),
                ],
                &f.embedder,
            )
            .await
            .expect("indexed");
        let mut rag = f.rag(true);
        rag.doc_sources = BTreeSet::from([src("a.md", "k1"), src("b.md", "k2")]);
        let query = [0.5, 0.5, 0.5];
        let mut declared = Vec::new();
        assert_eq!(rag.documentation(&query, &mut declared).len(), 2);
        assert!(declared.is_empty(), "{declared:?}");
        f.docs
            .connection_for_test()
            .lock()
            .expect("lock")
            .execute(
                "DELETE FROM code_review_docs WHERE key = ?1",
                [f.docs.key_tag_for_test("k1")],
            )
            .expect("delete");
        let hits = rag.documentation(&query, &mut declared);
        assert_eq!(hits.len(), 1, "the surviving hit still serves");
        assert_eq!(
            declared,
            vec![(ContextKind::Documentation, INDEX_UNAVAILABLE.to_string())]
        );
        // [MAGI S6 MS3 loop16, R2 Caspar] The loss touched one row: the next batch still serves
        // the survivor and the declaration is not repeated.
        let again = rag.documentation(&query, &mut declared);
        assert_eq!(again.len(), 1, "documentation stays on for what survived");
        assert_eq!(declared.len(), 1, "declared once: {declared:?}");
    }

    #[tokio::test]
    async fn a_source_gone_before_the_load_serves_the_others_and_is_declared() {
        // [MAGI S6 MS3 loop16, R1 Caspar / R2] One allowed source no longer indexed failed the
        // whole documentation layer; only that source is lost now, declared once.
        let f = RagFixture::new(&[]).await;
        f.docs
            .reindex(
                &[
                    (src("a.md", "k1"), vec![chunk("a.md", "A", "x")]),
                    (src("b.md", "k2"), vec![chunk("b.md", "B", "y")]),
                ],
                &f.embedder,
            )
            .await
            .expect("indexed");
        let mut rag = f.rag(true);
        rag.doc_sources = BTreeSet::from([src("a.md", "k1"), src("b.md", "k2")]);
        f.docs
            .connection_for_test()
            .lock()
            .expect("lock")
            .execute(
                "DELETE FROM code_review_doc_sources WHERE key = ?1",
                [f.docs.key_tag_for_test("k1")],
            )
            .expect("delete");
        let query = [0.5, 0.5, 0.5];
        let mut declared = Vec::new();
        let hits = rag.documentation(&query, &mut declared);
        assert_eq!(hits.len(), 1, "the other source serves");
        assert_eq!(
            declared,
            vec![(ContextKind::Documentation, INDEX_UNAVAILABLE.to_string())]
        );
        let again = rag.documentation(&query, &mut declared);
        assert_eq!((again.len(), declared.len()), (1, 1));
    }

    #[tokio::test]
    async fn no_query_is_embedded_once_both_layers_have_failed() {
        // [MAGI S6 MS3 loop7, R2 Caspar] With documentation and history both out for the rest of
        // the review, every later batch still spent an embedder call (and its bound) for nothing.
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let rag = f.rag(true);
        rag.docs_failed.store(true, Ordering::SeqCst);
        rag.history_failed.store(true, Ordering::SeqCst);
        rag.context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("context");
        assert_eq!(f.embedder.calls(), 0);
    }

    #[tokio::test]
    async fn a_documentation_store_that_was_not_opened_is_declared_unavailable() {
        // [MAGI S6 MS3 loop7, R2 Melchior] With a database and an embedder but no store, the
        // documentation was absent with no cause, whatever the caller set in the preparation.
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let mut rag = f.rag(true);
        rag.docs = None;
        let a = rag
            .context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("context");
        assert!(
            a.trimmed
                .contains(&(ContextKind::Documentation, INDEX_UNAVAILABLE.to_string())),
            "{:?}",
            a.trimmed
        );
    }

    #[tokio::test]
    async fn two_hunks_of_one_unit_are_charged_once_when_checking_for_room() {
        // [MAGI S6 MS3 loop2, R2 Melchior] `assemble` charges a repeated unit once; the room check
        // charged it per hunk, so a batch with room for one more item was declared full and its
        // documentation query never sent.
        let f = RagFixture::new(&[(
            "a.rs",
            "fn a() {\n    let x = 1;\n    let y = 2;\n    let z = 3;\n}\n",
        )])
        .await;
        let mut rag = f.rag(true);
        let hunk = |line: u32| BatchItem::Hunk {
            file: "a.rs".into(),
            text: format!("+line {line}"),
            first_line: line,
            last_line: line,
            old_first_line: line,
            old_last_line: line,
        };
        let b = ReviewBatch {
            index: 0,
            tokens: 10,
            companion_tokens: 0,
            items: vec![hunk(2), hunk(4)],
        };
        let unit = rag.companion(&hunk(2)).expect("companion");
        assert_eq!(unit.len(), 1, "precondition: one containing unit");
        rag.limit_tokens = b.tokens
            + estimate_tokens(CONTEXT_HEADING)
            + unit[0].1
            + magi_rs::review::rag::CONTEXT_ITEM_FRAME_TOKENS;
        let a = rag.context_for(&b).await.expect("context");
        assert_eq!(
            f.embedder.calls(),
            1,
            "the query was embedded: {:?}",
            a.trimmed
        );
    }

    #[tokio::test]
    async fn documentation_and_history_left_out_for_lack_of_room_are_declared() {
        // [MAGI S6 MS3 loop1, R2 Melchior and Balthasar] RV07: the report declares what was cut,
        // also when nothing of (3) and (4) could enter and their query was never embedded.
        let f = RagFixture::new(&[("notes.txt", "plain\n")]).await;
        let mut rag = f.rag(true);
        // MAGI FIX (MS3 §6 loop12): a source was indexed, so the cut layer is declared.
        rag.doc_sources = BTreeSet::from([src("README.md", "k")]);
        let b = batch_on("notes.txt", "plain");
        rag.limit_tokens = b.tokens
            + magi_rs::review::window::estimate_tokens(magi_rs::review::pipeline::CONTEXT_HEADING);
        let a = rag.context_for(&b).await.expect("context");
        const NO_ROOM: &str = "no room left in the batch";
        let cut: Vec<&(ContextKind, String)> = a
            .trimmed
            .iter()
            .filter(|(_, label)| label == NO_ROOM)
            .collect();
        assert_eq!(
            cut,
            vec![
                &(ContextKind::Documentation, NO_ROOM.to_string()),
                &(ContextKind::History, NO_ROOM.to_string())
            ]
        );
        assert!(!a.without_embedder && !a.without_memory);
    }

    #[test]
    fn with_prep_moves_every_field_of_the_preparation() {
        // [CP2 MS3 loop 13] The DocPrep -> StoreRag copy main.rs does; docs_unavailable is the one RV-S38 depends on.
        let rt = tokio::runtime::Runtime::new().expect("rt");
        let f = rt.block_on(RagFixture::new(&[("a.rs", "fn a() {}\n")]));
        let prep = DocPrep {
            doc_sources: BTreeSet::from([src("README.md", "k")]),
            doc_index: IndexDeclarations {
                withheld: 2,
                truncated: true,
                ..Default::default()
            },
            docs_unavailable: true,
            index_ms: 7,
            withheld_sources: vec!["x.md".into()],
            masked_sources: vec![("y.md".into(), "openai-key")],
        };
        let rag = f.rag(true).with_prep(prep);
        assert_eq!(
            magi_rs::review::pipeline::RagSource::masked(&rag),
            vec![("y.md".to_string(), "openai-key")]
        );
        assert_eq!(rag.doc_sources, BTreeSet::from([src("README.md", "k")]));
        assert_eq!(
            rag.doc_index,
            IndexDeclarations {
                withheld: 2,
                truncated: true,
                ..Default::default()
            }
        );
        assert!(rag.docs_unavailable);
        assert_eq!(
            (rag.index_ms, rag.withheld_sources.clone()),
            (7, vec!["x.md".to_string()])
        );
    }

    #[tokio::test]
    async fn a_deletion_only_hunk_takes_its_unit_from_the_old_side() {
        // [CP2 MS3 loop 10] RV21: the old side for deleted lines; the working tree and the new side say something else.
        let f = RagFixture::new(&[("a.rs", "fn worktree() {}\n")]).await;
        let two = TwoSides {
            new: vec![("a.rs", "fn other() {}\n")],
            old: vec![("a.rs", "fn gone() {\n    x();\n}\n")],
        };
        let mut rag = f.rag(false);
        rag.sources = &two;
        let batch = ReviewBatch {
            index: 0,
            tokens: 10,
            companion_tokens: 0,
            items: vec![BatchItem::Hunk {
                file: "a.rs".into(),
                text: "-    x();".into(),
                first_line: 1,
                last_line: 0,
                old_first_line: 2,
                old_last_line: 2,
            }],
        };
        let a = rag.context_for(&batch).await.expect("context");
        let units: Vec<&str> = a
            .items
            .iter()
            .filter(|i| i.kind == ContextKind::Unit)
            .map(|i| i.text.as_str())
            .collect();
        assert!(units.iter().any(|t| t.contains("fn gone()")), "{units:?}");
    }

    #[tokio::test]
    async fn a_hunk_that_changes_two_functions_is_planned_and_served_with_both_units() {
        // [Loop1 r1, I1] A real `git diff` hunk (-U3) that opens on a blank line between units
        // and changes `f` and `g`: the context carries both units, and planning charged exactly
        // what the context adds.
        let f = RagFixture::new(&[("a.rs", "fn worktree() {}\n")]).await;
        let two = TwoSides {
            new: vec![(
                "a.rs",
                "fn e() {\n    a();\n}\n\nfn f() {\n    b();\n    cc();\n}\n\nfn g() {\n    dd();\n}\n",
            )],
            old: vec![(
                "a.rs",
                "fn e() {\n    a();\n}\n\nfn f() {\n    b();\n    c();\n}\n\nfn g() {\n    d();\n}\n",
            )],
        };
        let mut rag = f.rag(false);
        rag.sources = &two;
        let diff = "diff --git a/a.rs b/a.rs\nindex 31d2d81..eec0b67 100644\n--- a/a.rs\n+++ b/a.rs\n\
                    @@ -4,9 +4,9 @@ fn e() {\n \n fn f() {\n     b();\n-    c();\n+    cc();\n }\n \n fn g() {\n\
                    -    d();\n+    dd();\n }\n";
        let files = vec![magi_rs::review::input::ReviewFile {
            path: "a.rs".into(),
            diff: Some(diff.into()),
        }];
        let batches = magi_rs::review::batching::plan_batches_with(
            &files,
            &two,
            &f.registry,
            100_000,
            &|i| rag.companion(i),
        )
        .expect("plans");
        let a = rag.context_for(&batches[0]).await.expect("context");
        let units: Vec<&ContextItem> = a
            .items
            .iter()
            .filter(|i| i.kind == ContextKind::Unit)
            .collect();
        assert_eq!(
            units.iter().map(|u| u.label.as_str()).collect::<Vec<_>>(),
            vec!["f", "g"]
        );
        let charged: usize = units.iter().map(|u| item_cost(u)).sum();
        assert_eq!(batches[0].companion_tokens, charged);
    }

    #[tokio::test]
    async fn a_batch_packed_to_the_planners_limit_still_assembles_and_one_token_less_does_not() {
        // [MAGI S6 MS3 loop17, R2 Balthasar] Who owns the context heading across the planning /
        // assembly seam: the planner packs to the prompt limit less the heading (`plan_review`),
        // and `context_for` takes the heading off `limit_tokens` itself. A batch packed exactly to
        // the planner's limit must assemble; one token less is the visible error.
        let f = RagFixture::new(&[("a.rs", "fn worktree() {}\n")]).await;
        let two = TwoSides {
            new: vec![("a.rs", "fn f() {\n    b();\n    cc();\n}\n")],
            old: vec![("a.rs", "fn f() {\n    b();\n    c();\n}\n")],
        };
        let mut rag = f.rag(false);
        rag.sources = &two;
        let diff =
            "diff --git a/a.rs b/a.rs\nindex 31d2d81..eec0b67 100644\n--- a/a.rs\n+++ b/a.rs\n\
                    @@ -1,4 +1,4 @@\n fn f() {\n     b();\n-    c();\n+    cc();\n }\n";
        let files = vec![magi_rs::review::input::ReviewFile {
            path: "a.rs".into(),
            diff: Some(diff.into()),
        }];
        let heading = estimate_tokens(CONTEXT_HEADING);
        let plan = |limit: usize| {
            magi_rs::review::batching::plan_batches_with(&files, &two, &f.registry, limit, &|i| {
                rag.companion(i)
            })
        };
        let roomy = plan(100_000).expect("plans");
        let packed = roomy[0].tokens + roomy[0].companion_tokens;
        assert!(roomy[0].companion_tokens > 0, "the unit is charged");
        // The planner, at exactly that limit, packs the same batch (the limit is reached).
        let at_limit = plan(packed).expect("fits at its own size");
        assert_eq!(at_limit.len(), 1);
        rag.limit_tokens = packed + heading;
        rag.context_for(&at_limit[0])
            .await
            .expect("a batch packed to the planner's limit assembles");
        rag.limit_tokens = packed + heading - 1;
        assert!(
            matches!(
                rag.context_for(&at_limit[0]).await,
                Err(ReviewError::TooLargeForBatch { .. })
            ),
            "one token under, the batch and its unit no longer fit"
        );
    }

    #[tokio::test]
    async fn an_old_side_and_a_new_side_unit_on_the_same_lines_are_two_companions() {
        // [MS3 final review m2] Planning charges what assemble adds: both units, never one.
        let f = RagFixture::new(&[("a.rs", "fn worktree() {}\n")]).await;
        let two = TwoSides {
            new: vec![("a.rs", "fn kept() {\n    y();\n}\n")],
            old: vec![("a.rs", "fn gone() {\n    x();\n}\n")],
        };
        let mut rag = f.rag(false);
        rag.sources = &two;
        let hunk = |first_line, last_line| BatchItem::Hunk {
            file: "a.rs".into(),
            text: "x".into(),
            first_line,
            last_line,
            old_first_line: 2,
            old_last_line: 2,
        };
        let old = rag
            .companion(&hunk(2, 1))
            .expect("ok")
            .into_iter()
            .next()
            .expect("unit");
        let new = rag
            .companion(&hunk(2, 2))
            .expect("ok")
            .into_iter()
            .next()
            .expect("unit");
        assert_ne!(old.0, new.0, "{old:?} {new:?}");
    }

    #[tokio::test]
    async fn a_working_tree_review_drops_older_working_tree_versions() {
        // [CP2 MS3 loop 10] RV-S39 for the working tree: the current version stays, the one it replaced goes.
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let now = src("README.md", &magi_rs::review::docs::worktree_key(b"v2"));
        let old = src("README.md", &magi_rs::review::docs::worktree_key(b"v1"));
        f.docs
            .reindex(
                &[
                    (now.clone(), vec![chunk("README.md", "A", "x")]),
                    (old.clone(), vec![chunk("README.md", "A", "y")]),
                ],
                &f.embedder,
            )
            .await
            .expect("index");
        let removed = collect_garbage(
            &f.docs,
            None,
            &|_: &str| true,
            Some(&BTreeSet::from([now.clone()])),
        )
        .expect("ok");
        assert_eq!(removed, Some(1));
        assert_eq!(
            f.docs.missing(&[now, old.clone()]).expect("missing"),
            vec![old]
        );
    }

    /// More distinct refs than MAX_GC_REFS; every tree is empty. [CP2 MS3 loop 10]
    struct ManyRefsGit;
    impl GitCommand for ManyRefsGit {
        fn run_bytes(&self, args: &[&str]) -> Result<Vec<u8>, ReviewError> {
            match args.join(" ").as_str() {
                "rev-parse --verify HEAD^{commit}" => Ok(b"h0\n".to_vec()),
                "for-each-ref --format=%(objectname) %(*objectname) refs/heads refs/tags" => Ok((0
                    ..=magi_rs::review::docs::MAX_GC_REFS)
                    .map(|i| format!("h{i}\n"))
                    .collect::<String>()
                    .into_bytes()),
                _ => Ok(Vec::new()),
            }
        }
    }

    #[tokio::test]
    async fn past_the_ref_cap_the_clean_up_after_a_review_leaves_every_row() {
        // [CP2 MS3 loop 10] The safe direction: skipped, never partial.
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        f.docs
            .reindex(
                &[(src("README.md", "k1"), vec![chunk("README.md", "A", "x")])],
                &f.embedder,
            )
            .await
            .expect("index");
        assert_eq!(
            collect_garbage(&f.docs, Some(&ManyRefsGit), &|_: &str| true, None).expect("ok"),
            None
        );
        assert!(f
            .docs
            .missing(&[src("README.md", "k1")])
            .expect("missing")
            .is_empty());
    }

    /// HEAD and one branch at `h1`, whose tree holds README.md at `k1`. [RV21 D+B, 2026-10-20]
    struct OneRefGit;
    impl GitCommand for OneRefGit {
        fn run_bytes(&self, args: &[&str]) -> Result<Vec<u8>, ReviewError> {
            match args.join(" ").as_str() {
                "rev-parse --verify HEAD^{commit}"
                | "for-each-ref --format=%(objectname) %(*objectname) refs/heads refs/tags" => {
                    Ok(b"h1\n".to_vec())
                }
                "ls-tree -r -z h1" => Ok(b"100644 blob k1\tREADME.md\0".to_vec()),
                other => Err(ReviewError::Git(other.to_string())),
            }
        }
    }

    #[tokio::test]
    async fn the_clean_up_after_a_review_keeps_what_a_local_ref_reaches() {
        // [RV21 D+B, 2026-10-20] RV-S39: a README of a deleted branch goes, the one HEAD reaches stays.
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        // [CP2 MS3 loop 14 (pasada final)] And a working-tree version: a commit review never listed the tree, so it stays.
        let wt = src(
            "README.md",
            &magi_rs::review::docs::worktree_key(b"local edit"),
        );
        f.docs
            .reindex(
                &[
                    (src("README.md", "k1"), vec![chunk("README.md", "A", "x")]),
                    (src("README.md", "gone"), vec![chunk("README.md", "A", "y")]),
                    (wt.clone(), vec![chunk("README.md", "A", "z")]),
                ],
                &f.embedder,
            )
            .await
            .expect("index");
        let removed =
            collect_garbage(&f.docs, Some(&OneRefGit), &|_: &str| true, None).expect("ok");
        assert_eq!(removed, Some(1));
        assert!(f
            .docs
            .missing(&[src("README.md", "k1"), wt])
            .expect("missing")
            .is_empty());
    }

    /// HEAD and refs answer; listing the tree fails with a message that quotes a key. [CP2 MS3 loop 14 (pasada final)]
    struct LeakyGit;
    impl GitCommand for LeakyGit {
        fn run_bytes(&self, args: &[&str]) -> Result<Vec<u8>, ReviewError> {
            match args.join(" ").as_str() {
                "rev-parse --verify HEAD^{commit}"
                | "for-each-ref --format=%(objectname) %(*objectname) refs/heads refs/tags" => {
                    Ok(b"h1\n".to_vec())
                }
                _ => Err(ReviewError::Git(LEAKY.into())),
            }
        }
    }
    const LEAKY: &str = "fatal: path sk-ant-api03-EEEEEEEEEEEEEEEEEEEE not found"; // allow-secret-scan

    #[tokio::test]
    async fn a_git_error_during_the_clean_up_is_a_redacted_storage_error() {
        // [CP2 MS3 loop 14 (pasada final)] The declared error path, and its cause never carries the raw text out.
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let r = collect_garbage(&f.docs, Some(&LeakyGit), &|_: &str| true, None);
        assert!(
            matches!(&r, Err(MemoryError::Storage(m)) if !m.contains("sk-ant-")),
            "{r:?}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_hanging_embedder_costs_prepare_docs_one_timeout_not_one_per_batch() {
        // [CP2 MS3 loop 14 (pasada final)] Two batches of sources, a hung embedder: the remaining budget passes down and
        // the first expiry stops the rest, so the whole preparation ends near BOUND, never near 2 x BOUND.
        use crate::memory::review_docs::EMBED_BATCH_ITEMS;
        let big: String = (0..EMBED_BATCH_ITEMS)
            .map(|i| format!("# S{i}\nx{i}\n"))
            .collect();
        let f = RagFixture::new(&[("CLAUDE.md", "# B\ny\n"), ("README.md", big.as_str())]).await;
        let prep = tokio::time::timeout(
            BOUND + Duration::from_secs(5),
            prepare_docs(
                &f.docs,
                None,
                None,
                &f.source,
                &["CLAUDE.md".to_string(), "README.md".to_string()],
                &|_: &str| true,
                &f.registry,
                &HangingEmbedder,
                BOUND,
            ),
        )
        .await;
        assert!(
            matches!(&prep, Ok(p) if p.docs_unavailable),
            "{:?}",
            prep.as_ref().map(|p| p.docs_unavailable)
        );
    }

    /// Answers its first request after 100 s and never answers again.
    struct SlowThenHanging(std::sync::atomic::AtomicUsize);
    #[async_trait::async_trait]
    impl EmbeddingProvider for SlowThenHanging {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            if self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                tokio::time::sleep(Duration::from_secs(100)).await;
                Ok(texts.iter().map(|_| vec![0.5; 3]).collect())
            } else {
                std::future::pending().await
            }
        }
        fn model_id(&self) -> &str {
            "fake"
        }
        fn dim(&self) -> usize {
            3
        }
        fn query_prefix(&self) -> &str {
            ""
        }
        fn document_prefix(&self) -> &str {
            ""
        }
    }

    #[tokio::test(start_paused = true)]
    async fn a_later_batch_gets_only_what_is_left_of_the_preparation_bound() {
        // The first batch spends 100 of the 120 s; the second must get the 20 s left, not 120 more.
        use crate::memory::review_docs::EMBED_BATCH_ITEMS;
        let big: String = (0..EMBED_BATCH_ITEMS)
            .map(|i| format!("# S{i}\nx{i}\n"))
            .collect();
        let f = RagFixture::new(&[("CLAUDE.md", "# B\ny\n"), ("README.md", big.as_str())]).await;
        let embedder = SlowThenHanging(Default::default());
        let prep = tokio::time::timeout(
            BOUND + Duration::from_secs(5),
            prepare_docs(
                &f.docs,
                None,
                None,
                &f.source,
                &["CLAUDE.md".to_string(), "README.md".to_string()],
                &|_: &str| true,
                &f.registry,
                &embedder,
                BOUND,
            ),
        )
        .await;
        assert!(
            matches!(&prep, Ok(p) if p.docs_unavailable),
            "{:?}",
            prep.as_ref().map(|p| p.docs_unavailable)
        );
    }

    /// Fails its first request and answers every later one.
    struct FailsFirst(std::sync::atomic::AtomicUsize);
    #[async_trait::async_trait]
    impl EmbeddingProvider for FailsFirst {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            if self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                Err(EmbeddingError::Http("down".into()))
            } else {
                Ok(texts.iter().map(|_| vec![0.5; 3]).collect())
            }
        }
        fn model_id(&self) -> &str {
            "fake"
        }
        fn dim(&self) -> usize {
            3
        }
        fn query_prefix(&self) -> &str {
            ""
        }
        fn document_prefix(&self) -> &str {
            ""
        }
    }

    #[tokio::test]
    async fn the_first_failing_batch_of_sources_stops_the_rest() {
        // A failure degrades the review (B); the batches after it are not attempted.
        use crate::memory::review_docs::EMBED_BATCH_ITEMS;
        let big: String = (0..EMBED_BATCH_ITEMS)
            .map(|i| format!("# S{i}\nx{i}\n"))
            .collect();
        let f = RagFixture::new(&[("CLAUDE.md", "# B\ny\n"), ("README.md", big.as_str())]).await;
        let embedder = FailsFirst(Default::default());
        let prep = prepare_docs(
            &f.docs,
            None,
            None,
            &f.source,
            &["CLAUDE.md".to_string(), "README.md".to_string()],
            &|_: &str| true,
            &f.registry,
            &embedder,
            BOUND,
        )
        .await;
        assert!(prep.docs_unavailable);
        assert_eq!(embedder.0.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_documentation_source_that_is_not_utf8_is_left_out_of_the_revision() {
        // §0: never a silent skip. It is not served, so it is not one of the revision's sources.
        let f = RagFixture::new(&[("CLAUDE.md", "# B\ny\n")]).await;
        std::fs::write(f._ws.path().join("README.md"), [0xff, 0xfe, b'#']).expect("write");
        let prep = prepare_docs(
            &f.docs,
            None,
            None,
            &f.source,
            &["CLAUDE.md".to_string(), "README.md".to_string()],
            &|_: &str| true,
            &f.registry,
            &f.embedder,
            BOUND,
        )
        .await;
        let paths: Vec<&str> = prep.doc_sources.iter().map(|s| s.path.as_str()).collect();
        assert_eq!(paths, vec!["CLAUDE.md"]);
        assert!(!prep.docs_unavailable);
    }

    #[tokio::test]
    async fn a_down_embedder_is_seen_even_when_the_index_is_unavailable() {
        // [CP2 MS3 loop 14 (pasada final)] docs_unavailable skips the search, never the query embed: S51's signal holds.
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        f.history.record(&[attended("h1", "fine")]).expect("record");
        let down = CountingFailingEmbedder(Default::default());
        let mut rag = f.rag(true);
        rag.embedder = Some(&down);
        rag.docs_unavailable = true;
        let a = rag
            .context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("never fails the review");
        assert!(a.without_embedder, "the embedder was not asked");
        assert!(
            a.items.iter().all(|i| i.kind != ContextKind::History),
            "RV07: the history goes with the embedder"
        );
        assert_eq!(down.0.load(std::sync::atomic::Ordering::SeqCst), 1);
    }
}
