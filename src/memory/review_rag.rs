// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-11-05

//! The retrieved context of a `code-review` batch, from the encrypted stores and the embedder
//! (REQ-RV07, RV08, RV19, RV21, RV22).
#![allow(dead_code)] // Red stub: removed in Green

use std::collections::BTreeSet;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use magi_rs::review::batching::{BatchItem, Companion, ReviewBatch};
use magi_rs::review::docs::{DocChunk, DocSource};
use magi_rs::review::error::ReviewError;
use magi_rs::review::history::HistoryEntry;
use magi_rs::review::index::LexicalIndex;
use magi_rs::review::input::SourceKind;
use magi_rs::review::languages::LanguageRegistry;
use magi_rs::review::pipeline::RagSource;
use magi_rs::review::rag::{Assembled, IndexDeclarations};
use magi_rs::review::source::{ContentSource, GitCommand};

use crate::memory::embedding::EmbeddingProvider;
use crate::memory::error::MemoryError;
use crate::memory::review_docs::ReviewDocStore;
use crate::memory::review_history::ReviewHistoryStore;

/// Wall-clock bound on one reclassification pass of the review history.
pub const RECLASSIFY_BUDGET_SECS: u64 = 60;

/// Bound on embedding one batch's documentation query.
pub const EMBED_QUERY_TIMEOUT_SECS: u64 = 30;

/// Documentation chunks retrieved per batch before `assemble` trims to the budget.
pub const DOC_SEARCH_K: usize = 8;

/// Bound on the whole documentation preparation of one review.
pub const REINDEX_TIMEOUT_SECS: u64 = 120;

/// The context source of one review, over the stores and the review's own content.
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
    /// The batch limit in tokens.
    pub limit_tokens: usize,
    /// What indexing left out.
    pub doc_index: IndexDeclarations,
    /// Whether the index declarations were already reported.
    pub declared: AtomicBool,
    /// The documentation index could not be completed for this revision.
    pub docs_unavailable: bool,
    /// The query embedder failed once.
    pub embedder_down: AtomicBool,
    /// A documentation read failed once.
    pub docs_failed: AtomicBool,
    /// A history read failed once.
    pub history_failed: AtomicBool,
    /// The reviewed revision's documentation sources.
    pub doc_sources: BTreeSet<DocSource>,
    /// Documentation paths left out for a secret, as `display_path`.
    pub withheld_sources: Vec<String>,
    /// Milliseconds the documentation preparation took.
    pub index_ms: u64,
}

/// What the documentation preparation of one review produced.
#[derive(Debug, Default, Clone)]
pub struct DocPrep {
    /// The reviewed revision's documentation sources.
    pub doc_sources: BTreeSet<DocSource>,
    /// What indexing left out.
    pub doc_index: IndexDeclarations,
    /// The index could not be completed.
    pub docs_unavailable: bool,
    /// Milliseconds the preparation took.
    pub index_ms: u64,
    /// Sources left out for a secret.
    pub withheld_sources: Vec<String>,
}

/// What one [`record_review`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordStats {
    /// Due entries checked.
    pub reclassified: usize,
    /// New entries recorded.
    pub recorded: usize,
    /// Due entries left for a later review.
    pub backlog: usize,
}

impl<'a> StoreRag<'a> {
    /// Moves a preparation into the source.
    pub fn with_prep(self, prep: DocPrep) -> Self {
        let _ = prep;
        self
    }
}

#[async_trait::async_trait]
impl RagSource for StoreRag<'_> {
    async fn context_for(&self, _batch: &ReviewBatch) -> Result<Assembled, ReviewError> {
        todo!()
    }

    fn companion(&self, _item: &BatchItem) -> Result<Option<Companion>, ReviewError> {
        todo!()
    }
}

/// The revision whose documentation a review serves.
pub fn doc_revision(kind: &SourceKind) -> Option<&str> {
    let _ = kind;
    None
}

/// Indexes the review's documentation, degrading on any failure.
#[allow(clippy::too_many_arguments)]
pub async fn prepare_docs(
    docs: &ReviewDocStore,
    git: Option<&dyn GitCommand>,
    rev: Option<&str>,
    sources: &dyn ContentSource,
    worktree_paths: &[String],
    keep: &dyn Fn(&str) -> bool,
    registry: &LanguageRegistry,
    embedder: &dyn EmbeddingProvider,
    timeout: Duration,
) -> DocPrep {
    let _ = (
        docs,
        git,
        rev,
        sources,
        worktree_paths,
        keep,
        registry,
        embedder,
        timeout,
    );
    todo!()
}

/// One bounded reindex.
pub async fn reindex_or_degrade(
    docs: &ReviewDocStore,
    sources: &[(DocSource, Vec<DocChunk>)],
    embedder: &dyn EmbeddingProvider,
    timeout: Duration,
) -> bool {
    let _ = (docs, sources, embedder, timeout);
    todo!()
}

/// Reclassifies, then records.
pub fn record_review(
    history: &ReviewHistoryStore,
    findings: &[HistoryEntry],
    head: &str,
    git: &dyn GitCommand,
    now: i64,
    budget: Duration,
) -> Result<RecordStats, MemoryError> {
    let _ = (history, findings, head, git, now, budget);
    todo!()
}

/// Deletes what nothing reaches.
pub fn collect_garbage(
    docs: &ReviewDocStore,
    git: Option<&dyn GitCommand>,
    keep: &dyn Fn(&str) -> bool,
    worktree: Option<&BTreeSet<DocSource>>,
) -> Result<Option<usize>, MemoryError> {
    let _ = (docs, git, keep, worktree);
    todo!()
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
                index_ms: 0,
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
    async fn a_history_entry_quoting_a_secret_never_reaches_the_context() {
        let f = RagFixture::new(&[("a.rs", "fn a() {}\n")]).await;
        let leaked = "key sk-ant-api03-AAAAAAAAAAAAAAAAAAAA is hardcoded"; // allow-secret-scan
                                                                           // [CP2 MS3 loop 11] Title and detail are each scanned: one entry per field.
        let mut by_title = attended("h2", "fine");
        by_title.title = leaked.into();
        f.history
            .record(&[attended("h1", leaked), by_title])
            .expect("record");
        f.history
            .update_status("h1", HistoryStatus::Attended)
            .expect("status");
        f.history
            .update_status("h2", HistoryStatus::Attended)
            .expect("status");
        let a = f
            .rag(true)
            .context_for(&batch_on("a.rs", "fn a() {}"))
            .await
            .expect("context");
        assert!(
            a.items
                .iter()
                .all(|i| !i.text.contains("sk-ant-") && !i.label.contains("sk-ant-")),
            "a secret left the process"
        );
        assert_eq!(
            a.withheld, 2,
            "a title-only and a detail-only secret, each withheld"
        );
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
    async fn a_source_with_one_secret_section_is_withheld_whole() {
        // [CP2 MS3 loop 13] skipped's `secret` entry means nothing of the file left: its clean sections are not indexed either.
        let readme = "# Setup\nexport KEY=sk-ant-api03-AAAAAAAAAAAAAAAAAAAA\n# Other\nfine\n"; // allow-secret-scan
        let f = RagFixture::new(&[("README.md", readme)]).await;
        let prep = prepare_docs(
            &f.docs,
            None,
            None,
            &f.source,
            &["README.md".to_string()],
            &|_: &str| true,
            &f.registry,
            &f.embedder,
            BOUND,
        )
        .await;
        assert!(prep.doc_sources.is_empty(), "{:?}", prep.doc_sources);
        assert_eq!(prep.withheld_sources, vec!["README.md".to_string()]);
        assert_eq!(f.embedder.calls(), 0, "no section of it was embedded");
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
        };
        let rag = f.rag(true).with_prep(prep);
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
