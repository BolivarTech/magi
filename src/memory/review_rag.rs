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

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rusqlite::Connection;
use tokio::time::Instant;

use magi_rs::review::batching::{BatchItem, Companion, ReviewBatch};
use magi_rs::review::docs::{
    bounded, chunk_source, doc_index_id, live_doc_sources, revision_doc_sources, without_secrets,
    worktree_key, yields_doc_chunks, DocChunk, DocSource,
};
use magi_rs::review::error::ReviewError;
use magi_rs::review::history::{reclassify_then_record, HistoryEntry, MAX_RECLASSIFY_PER_REVIEW};
use magi_rs::review::index::LexicalIndex;
use magi_rs::review::input::SourceKind;
use magi_rs::review::languages::LanguageRegistry;
use magi_rs::review::pipeline::{escape_fence, RagSource, ReviewMaterial, CONTEXT_HEADING};
use magi_rs::review::rag::{
    assemble, callers_for, charged_tokens, containing_units, declare_index, definitions_for,
    doc_search_query, item_cost, Assembled, ContextItem, ContextKind, IndexDeclarations,
    CONTEXT_ITEM_FRAME_TOKENS, MAX_CALLERS_PER_NAME,
};
use magi_rs::review::secrets::{
    display_path, find_secret, holds_key_block, mask_secrets, masked_pattern, outward_text,
};
use magi_rs::review::source::{ContentSource, GitCommand};
use magi_rs::review::window::estimate_tokens;
use magi_rs::vault::MaskedDek;

use crate::memory::embedding::EmbeddingProvider;
use crate::memory::error::MemoryError;
use crate::memory::review_docs::{DocVectors, ReviewDocStore, EMBED_BATCH_ITEMS};
use crate::memory::review_history::ReviewHistoryStore;

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

/// Bound on the whole documentation preparation of one review (listing, reads and every
/// reindex), checked between steps and sources; it runs before the review clock (RV06).
pub const REINDEX_TIMEOUT_SECS: u64 = 120;

/// Run-log target of this module's entries.
const LOG_TARGET: &str = "magi_rs::review";

/// The `trimmed` label of a documentation or history source that is unavailable for the review.
const INDEX_UNAVAILABLE: &str = "index unavailable";

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
    /// The review's content source.
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
    /// A history read failed once: no later batch reads it.
    pub history_failed: AtomicBool,
    /// The reviewed revision's documentation sources: `search` sees only these (RV21).
    pub doc_sources: BTreeSet<DocSource>,
    /// Documentation paths left out for a secret, as `display_path`, returned by `skipped`.
    pub withheld_sources: Vec<String>,
    /// Documentation paths indexed with a secret masked, with the first pattern masked in each,
    /// returned by `masked`.
    pub masked_sources: Vec<(String, &'static str)>,
    /// Milliseconds the documentation preparation took, returned by `index_ms`.
    pub index_ms: u64,
    /// The decrypted vectors of `doc_sources`, loaded by the first batch that searches and
    /// scored by every later one: a review decrypts each allowed vector once.
    pub doc_vectors: Mutex<Option<Arc<DocVectors>>>,
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
    /// Sources left out for a secret, as `display_path`.
    pub withheld_sources: Vec<String>,
    /// Sources indexed with a secret masked, as `display_path`, each with the first pattern
    /// masked in it.
    pub masked_sources: Vec<(String, &'static str)>,
}

/// What one [`record_review`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordStats {
    /// Due entries checked this review.
    pub reclassified: usize,
    /// New entries recorded.
    pub recorded: usize,
    /// Due entries left for a later review (the total minus those checked now).
    pub backlog: usize,
}

/// `item` with its text escaped for its frame, so what is charged is what is written.
fn escaped(item: ContextItem) -> ContextItem {
    ContextItem {
        text: escape_fence(&item.text),
        ..item
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

/// The names of the changed units: each containing unit's, and each unit item's. A whole file
/// names none (its callers would be every caller of the file).
fn changed_names(batch: &ReviewBatch, units: &[ContextItem]) -> Vec<String> {
    let mut seen = BTreeSet::new();
    units
        .iter()
        .map(|u| u.label.as_str())
        .chain(batch.items.iter().filter_map(|i| match i {
            BatchItem::Unit { name, .. } => Some(name.as_str()),
            _ => None,
        }))
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
        lines: e.line.map(|l| (l, l)),
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

    /// The query's vector, embedded under [`EMBED_QUERY_TIMEOUT_SECS`]; `None` when the embedder
    /// fails or does not answer, which marks it down for the rest of the review with one WARN.
    async fn query_vector(&self, embedder: &dyn EmbeddingProvider, text: &str) -> Option<Vec<f32>> {
        if self.embedder_down.load(Ordering::SeqCst) {
            return None;
        }
        let query = doc_search_query(embedder.query_prefix(), text);
        let bound = Duration::from_secs(EMBED_QUERY_TIMEOUT_SECS);
        let cause = match tokio::time::timeout(bound, embedder.embed(&[query])).await {
            // MAGI FIX (MS3 §6 loop2): one query, one vector; any other count is not guessed at.
            Ok(Ok(mut vectors)) => match (vectors.pop(), vectors.is_empty()) {
                (Some(v), true) if !v.is_empty() => return Some(v),
                (None, _) => "the embedder answered no vector".to_string(),
                _ => "the embedder did not answer one vector for the query".to_string(),
            },
            Ok(Err(e)) => outward_text(&e.to_string()),
            Err(_) => format!("no answer within {EMBED_QUERY_TIMEOUT_SECS} s"),
        };
        if !self.embedder_down.swap(true, Ordering::SeqCst) {
            tracing::event!(
                target: LOG_TARGET,
                tracing::Level::WARN,
                error = cause.as_str(),
                "the embedder did not answer: documentation and history are left out of this review"
            );
        }
        None
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
            Ok(hits) => hits.into_iter().map(|(chunk, _)| doc_item(chunk)).collect(),
            Err(e) => {
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
        }
    }

    /// The [`DOC_SEARCH_K`] chunks of the revision's sources closest to `query`, scored against
    /// [`Self::doc_vectors`]: the first call loads and decrypts them (the connection lock is held
    /// only for the raw read), every later call of the review reuses them. The cache lock is
    /// released before any store call.
    ///
    /// # Errors
    ///
    /// [`MemoryError`] of the store.
    fn search_docs(
        &self,
        docs: &ReviewDocStore,
        query: &[f32],
    ) -> Result<Vec<(DocChunk, f32)>, MemoryError> {
        let cached = self
            .doc_vectors
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let vectors = match cached {
            Some(vectors) if vectors.dim() == query.len() => vectors,
            _ => {
                let loaded = Arc::new(docs.load_vectors(query.len(), &self.doc_sources)?);
                *self
                    .doc_vectors
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Arc::clone(&loaded));
                loaded
            }
        };
        docs.search_loaded(&vectors, query, DOC_SEARCH_K)
    }

    /// The validated history on `files`, each one-line secret of a title or detail masked
    /// ([`mask_secrets`], RV22 amended); an entry whose title or detail holds a private-key block
    /// counts in `withheld` instead (RV22's exception). A read error leaves history out for the
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
        match history.validated_on_files(files) {
            Ok(entries) => entries
                .into_iter()
                .filter_map(|e| {
                    if holds_key_block(&e.title) || holds_key_block(&e.detail) {
                        *withheld = withheld.saturating_add(1);
                        None
                    } else {
                        Some(history_item(HistoryEntry {
                            title: mask_secrets(&e.title),
                            detail: mask_secrets(&e.detail),
                            ..e
                        }))
                    }
                })
                .collect(),
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

#[async_trait::async_trait]
impl RagSource for StoreRag<'_> {
    /// The batch's context: containing units, definitions and callers from the review's own
    /// content, then documentation and history when there is a database and an embedder.
    ///
    /// The query is embedded only when something could still enter the budget, and before any
    /// store is touched. A missing embedder is declared as `without_embedder`, a run without a
    /// database as `without_memory`; a failing embedder or store degrades, never fails. A
    /// configured embedder that fails also declares, once per review, "the embedder did not
    /// answer" for documentation and history, so the report tells it from a missing one.
    ///
    /// "Something could still enter" is checked against [`CONTEXT_ITEM_FRAME_TOKENS`], the floor
    /// of any item's cost: a real item also costs its text, so with little room left the query
    /// may be embedded for items that then do not fit (declared as trimmed). The check only
    /// saves the call when nothing at all can enter.
    ///
    /// The query is embedded even when the documentation cannot be searched (no store, or an
    /// index unavailable for the revision): the history goes with the embedder (RV07, CP2 MS3
    /// loop 14), so the call is also the review's check that the embedder answers
    /// (`a_down_embedder_is_seen_even_when_the_index_is_unavailable`). The query is masked
    /// ([`doc_search_query`]) like every other byte that leaves the process.
    async fn context_for(&self, batch: &ReviewBatch) -> Result<Assembled, ReviewError> {
        let text = batch_text(batch);
        let mut units = Vec::new();
        for item in &batch.items {
            units.extend(
                containing_units(item, self.sources, self.registry)?
                    .into_iter()
                    .map(escaped),
            );
        }
        let mut code = definitions_for(&text, self.index, self.sources)?;
        let names = changed_names(batch, &units);
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let (callers, capped) = callers_for(&refs, self.index, self.sources, self.registry)?;
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
        let (docs, history) = match (memory, self.embedder) {
            (false, _) | (true, None) => (None, None),
            (true, Some(_)) if !room => {
                // MAGI FIX (MS3 §6 loop1): RV07 declares what was cut, this case included.
                // MAGI FIX (MS3 §6 loop2): an unavailable index has its own cause, never room.
                let docs_down = self.docs_unavailable || self.docs_failed.load(Ordering::SeqCst);
                if self.docs.is_some() && !docs_down {
                    declared.push((ContextKind::Documentation, NO_ROOM.to_string()));
                }
                if self.history.is_some() {
                    declared.push((ContextKind::History, NO_ROOM.to_string()));
                }
                (Some(Vec::new()), Some(Vec::new()))
            }
            (true, Some(embedder)) => {
                let was_down = self.embedder_down.load(Ordering::SeqCst);
                match self.query_vector(embedder, &text).await {
                    None => {
                        // MAGI FIX (MS3 §6 loop2): a down embedder is said apart from a missing
                        // one, once per review (the run log keeps its error).
                        if !was_down {
                            if self.docs.is_some() {
                                declared
                                    .push((ContextKind::Documentation, EMBEDDER_DOWN.to_string()));
                            }
                            if self.history.is_some() {
                                declared.push((ContextKind::History, EMBEDDER_DOWN.to_string()));
                            }
                        }
                        (None, None)
                    }
                    Some(vector) => (
                        Some(self.documentation(&vector, &mut declared)),
                        Some(self.history_items(&batch_files(batch), &mut declared, &mut withheld)),
                    ),
                }
            }
        };
        let mut a = assemble(batch.tokens, limit, units, code, docs, history)?;
        if !memory {
            a.without_memory = true;
            a.without_embedder = false;
        }
        a.withheld = a.withheld.saturating_add(withheld);
        a.trimmed.extend(capped.into_iter().map(|name| {
            (
                ContextKind::Caller,
                format!("{name}: more than {MAX_CALLERS_PER_NAME} callers"),
            )
        }));
        a.trimmed.extend(declared);
        if !self.declared.swap(true, Ordering::SeqCst) {
            declare_index(&mut a, &self.doc_index);
            if memory && self.docs_unavailable {
                a.trimmed
                    .push((ContextKind::Documentation, INDEX_UNAVAILABLE.to_string()));
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
        Ok(containing_units(item, self.sources, self.registry)?
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
            .collect())
    }

    fn skipped(&self) -> Vec<String> {
        self.withheld_sources.clone()
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
    /// The documentation index, when it could be opened for this review.
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
    /// bound source will call the embedder (a store and an embedder); 0 otherwise.
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
            history_failed: AtomicBool::new(false),
            doc_sources: BTreeSet::new(),
            withheld_sources: Vec::new(),
            masked_sources: Vec::new(),
            index_ms: 0,
            doc_vectors: Mutex::new(None),
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
    /// [`MemoryError::Storage`] when the dimension is still unknown or the schema cannot be
    /// checked, [`MemoryError::Crypto`] when the key cannot be duplicated.
    pub async fn doc_store(
        &self,
        embedder: &dyn EmbeddingProvider,
    ) -> Result<ReviewDocStore, MemoryError> {
        if embedder.dim() == 0 {
            let bound = Duration::from_secs(DIMENSION_PROBE_TIMEOUT_SECS);
            let probe = vec![DIMENSION_PROBE.to_string()];
            // Its answer is not used: a successful call sets the dimension; a failed one leaves
            // it at 0, checked right below.
            let _ = tokio::time::timeout(bound, embedder.embed(&probe)).await;
        }
        if embedder.dim() == 0 {
            return Err(MemoryError::Storage(
                "the embedder has no known dimension".to_string(),
            ));
        }
        let key = self
            .dek
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .duplicate()
            .map_err(|e| MemoryError::Crypto(e.to_string()))?;
        ReviewDocStore::new(
            Arc::clone(&self.conn),
            key,
            doc_index_id(embedder.model_id(), embedder.dim()),
        )
    }
}

/// The revision whose documentation a review serves (RV21, U29): a revision review's new side,
/// or `None` for the working tree.
///
/// # Arguments
///
/// * `kind` - The resolved input's content source.
///
/// # Returns
///
/// The new side's revision, or `None`.
pub fn doc_revision(kind: &SourceKind) -> Option<&str> {
    match kind {
        SourceKind::Revision { new_rev, .. } => Some(new_rev.as_str()),
        SourceKind::Worktree { .. } => None,
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
/// through `sources` and keyed by [`worktree_key`]; a path that quotes a secret is never read.
///
/// A file that cannot be read (a hard link, a link that escapes the workspace or names a
/// directory, a file another process holds open) is left out with one WARN naming it, as MS2
/// leaves a listed file out of a review: one such file never costs the rest their documentation.
/// The size cap cuts the source (declared as truncated).
///
/// # Errors
///
/// A cause when the deadline passes.
fn list_worktree(
    sources: &dyn ContentSource,
    paths: &[String],
    wanted: &dyn Fn(&str) -> bool,
    deadline: Instant,
    prep: &mut DocPrep,
) -> Result<Vec<Listed>, String> {
    let mut listed = Vec::new();
    for path in paths.iter().filter(|p| wanted(p)) {
        if Instant::now() >= deadline {
            return Err(format!("not listed within {REINDEX_TIMEOUT_SECS} s"));
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
            Ok(None) => {}
            Err(e) if is_size_cap(&e) => prep.doc_index.truncated = true,
            Err(e) => {
                let (shown, error) = (display_path(path), outward_text(&e.to_string()));
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
/// 1. **List**: `rev` = the reviewed revision's new side, listed through `git` without reading
///    any file; `None` = the working tree, each of `worktree_paths` read through `sources` and
///    keyed by [`worktree_key`]. Only the paths `keep` accepts that can yield a chunk
///    ([`yields_doc_chunks`]: a Markdown doc source, or a file whose doc comments are chunked)
///    count; a path that quotes a secret is never read or stored (its
///    path would sit in a clear column) and counts as withheld.
/// 2. **Index what is missing**: each source the store lacks is read through `sources`, chunked
///    with `chunk_source` (which masks every one-line secret, RV22 amended), then passed through
///    `without_secrets` (every chunk, before any embedding) and `bounded`, in that order. A
///    source indexed with a mask is declared in `masked_sources`; a source that holds a
///    private-key block anywhere, or any chunk `without_secrets` drops, is withheld whole.
///    The rest is indexed in batches of at most [`EMBED_BATCH_ITEMS`] chunks, one
///    [`reindex_or_degrade`] each, given what is left of `timeout`; what a batch wrote stays.
/// 3. **Degrade (B)**: a listing, `missing`, read or reindex that fails, and `timeout` itself,
///    set `docs_unavailable` with a WARN; the first failing batch stops the rest. The size cap
///    only cuts (declared as truncated), it never degrades. The two read failures differ on
///    purpose: a revision's source that git lists but cannot give back means that revision's
///    index cannot be completed, and U29 then serves no documentation rather than a partial
///    one; a working-tree file that cannot be read is one the review itself leaves out (MS2's
///    rule for a listed file), so it is not part of the revision's documentation at all.
///
/// Never fails. `timeout` bounds the whole preparation, checked between steps and sources (a git
/// call or read in flight ends by its own bound). O(listed sources + bytes read).
///
/// # Arguments
///
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
    let wanted = |p: &str| keep(p) && (find_secret(p).is_some() || yields_doc_chunks(p, registry));
    let listed = match (rev, git) {
        (Some(rev), Some(git)) => revision_doc_sources(git, rev, &wanted)
            .map(|v| v.into_iter().map(|s| (s, None)).collect())
            .map_err(|e| outward_text(&e.to_string())),
        (Some(_), None) => Err("no repository to list the reviewed revision".to_string()),
        (None, _) => list_worktree(sources, worktree_paths, &wanted, deadline, &mut prep),
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
    let mut chunked = Vec::new();
    for (source, bytes) in pairs.into_iter().filter(|(s, _)| missing.contains(s)) {
        if Instant::now() >= deadline {
            let cause = format!("not indexed within {REINDEX_TIMEOUT_SECS} s");
            return degraded(prep, &cause, started);
        }
        let bytes = match bytes {
            Some(bytes) => bytes,
            None => match sources.read(&source.path) {
                Ok(Some(bytes)) => bytes,
                Ok(None) => continue,
                Err(e) if is_size_cap(&e) => {
                    prep.doc_index.truncated = true;
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
            prep.doc_sources.remove(&source);
            continue;
        };
        // MAGI FIX (MS3 §6 loop1): RV22 amended: a chunk's one-line secrets are masked by the
        // chunker and the file is declared as masked; only a private-key block, anywhere in the
        // file (its code included), still withholds the whole source.
        let (kept, dropped) = if holds_key_block(&text) {
            (Vec::new(), vec![display_path(&source.path)])
        } else {
            without_secrets(chunk_source(&source.path, &text, registry))
        };
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
    let (kept, truncated) = bounded(chunked);
    prep.doc_index.truncated |= truncated;
    for batch in reindex_batches(kept) {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() || !reindex_or_degrade(docs, &batch, embedder, left).await {
            let cause = "a batch of documentation sources was not indexed";
            return degraded(prep, cause, started);
        }
    }
    prep.index_ms = millis(started.elapsed());
    prep
}

/// One [`ReviewDocStore::reindex`] under `timeout`: `true` when it wrote, `false` (with a WARN)
/// when it failed or did not finish. A reindex is all or nothing, so a timeout writes nothing
/// and the next review tries the same sources again. Leaves an INFO with its duration, and the
/// fixed rebuilt line when the index changed identity.
///
/// # Arguments
///
/// * `docs` - The documentation index.
/// * `sources` - Whole sources with their chunks.
/// * `embedder` - The `[embedding]` embedder.
/// * `timeout` - The bound on this reindex.
///
/// # Returns
///
/// Whether the sources are indexed now.
pub async fn reindex_or_degrade(
    docs: &ReviewDocStore,
    sources: &[(DocSource, Vec<DocChunk>)],
    embedder: &dyn EmbeddingProvider,
    timeout: Duration,
) -> bool {
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
            true
        }
        Ok(Err(e)) => {
            let error = outward_text(&e.to_string());
            tracing::event!(
                target: LOG_TARGET,
                tracing::Level::WARN,
                error = error.as_str(),
                "the documentation index could not be updated"
            );
            false
        }
        Err(_) => {
            tracing::event!(
                target: LOG_TARGET,
                tracing::Level::WARN,
                ms = millis(started.elapsed()),
                "the documentation index was not updated in time"
            );
            false
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
/// [`MemoryError`] of the store, never swallowed.
pub fn record_review(
    history: &ReviewHistoryStore,
    findings: &[HistoryEntry],
    head: &str,
    git: &dyn GitCommand,
    now: i64,
    budget: Duration,
) -> Result<RecordStats, MemoryError> {
    let started = std::time::Instant::now();
    let (due, total) = history.due(MAX_RECLASSIFY_PER_REVIEW)?;
    let mut checked = 0usize;
    for entry in &due {
        if started.elapsed() >= budget {
            break;
        }
        let (updates, _) = reclassify_then_record(std::slice::from_ref(entry), &[], head, git);
        for (id, status) in updates {
            history.update_status(&id, status)?;
        }
        history.touch(std::slice::from_ref(&entry.id), now)?;
        checked = checked.saturating_add(1);
    }
    let recorded = history.record(findings)?;
    Ok(RecordStats {
        reclassified: checked,
        recorded,
        backlog: total.saturating_sub(checked),
    })
}

/// Deletes the documentation sources nothing reaches any more (RV-S39): with `git`, the live
/// set is every local ref's sources plus `worktree` when the review listed the working tree;
/// without git, `worktree` alone. A review that did not list the working tree keeps its
/// working-tree sources for the next one that does. Without git and without a listed working
/// tree nothing is known to be live, so nothing is deleted. O(refs x tree entries) git calls.
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
    docs.gc(&live, worktree.is_none()).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::review_docs::test_embedder::{chunk, src, FailingEmbedder, FakeEmbedder};
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
                history_failed: Default::default(),
                doc_sources: BTreeSet::new(),
                withheld_sources: Vec::new(),
                masked_sources: Vec::new(),
                index_ms: 0,
                doc_vectors: Mutex::new(None),
            }
        }
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
    async fn a_multi_batch_review_decrypts_the_allowed_vectors_once() {
        // [MS3 final review I3] The allowed vectors are decrypted once per review, never once per
        // batch, and every batch is served the same documentation.
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
            2,
            "each allowed row once"
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
            reindex_or_degrade(&f.docs, &docs, &f.embedder, bound).await,
            "a healthy embedder indexes"
        );
        // [CP2 MS3 loop 13] Another source: `docs` is indexed now and would be skipped without calling the embedder, so
        // the failing one must have something new to embed (with the old "failing" model id the index-id gate hid this).
        let other = [(
            src("README.md", "k2"),
            vec![chunk("README.md", "Usage", "how a works, v2")],
        )];
        assert!(!reindex_or_degrade(&f.docs, &other, &FailingEmbedder, bound).await);
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
            matches!(r, Ok(false)),
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

    #[test]
    fn the_documentation_revision_is_the_reviewed_new_side() {
        // [CP2 MS3 loop 11] main.rs takes `rev` from the resolved input: a revision review lists its new side; every
        // working-tree input (file, directory, uncommitted: MS2 resolve_input) lists the working tree.
        let rev = SourceKind::Revision {
            new_rev: "n1".into(),
            old_rev: Some("o1".into()),
        };
        assert_eq!(doc_revision(&rev), Some("n1"));
        assert_eq!(
            doc_revision(&SourceKind::Worktree {
                old_rev: Some("h".into())
            }),
            None
        );
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
                truncated: true
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
                "for-each-ref --format=%(objectname) refs/heads refs/tags" => Ok((0
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
                | "for-each-ref --format=%(objectname) refs/heads refs/tags" => {
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
                | "for-each-ref --format=%(objectname) refs/heads refs/tags" => {
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
