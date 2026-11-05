// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-11-05

//! Encrypted index of project documentation for the `code-review` context (RV07, RV19, RV21).
//!
//! Stub: the Green commit replaces these bodies.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use crate::memory::embedding::EmbeddingProvider;
use crate::memory::error::MemoryError;
use magi_rs::review::docs::{DocChunk, DocSource};
use magi_rs::vault::MaskedDek;

/// Texts per `embed` request.
#[allow(dead_code)] // Red stub: removed in Green
pub const EMBED_BATCH_ITEMS: usize = 64;

/// Counts of one `reindex`.
#[allow(dead_code)] // Red stub: removed in Green
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReindexStats {
    pub embedded: usize,
    pub reused: usize,
    pub sources: usize,
    pub rebuilt: bool,
}

/// Stub store.
#[allow(dead_code)] // Red stub: removed in Green
pub struct ReviewDocStore {
    conn: Arc<Mutex<rusqlite::Connection>>,
}

#[allow(dead_code)] // Red stub: removed in Green
impl ReviewDocStore {
    #[cfg(test)]
    pub(crate) fn connection_for_test(&self) -> &Arc<Mutex<rusqlite::Connection>> {
        &self.conn
    }

    pub fn new(
        conn: Arc<Mutex<rusqlite::Connection>>,
        _dek: MaskedDek,
        _index_id: String,
    ) -> Result<Self, MemoryError> {
        Ok(Self { conn })
    }

    pub fn missing(&self, _wanted: &[DocSource]) -> Result<Vec<DocSource>, MemoryError> {
        todo!()
    }

    pub async fn reindex(
        &self,
        _sources: &[(DocSource, Vec<DocChunk>)],
        _embedder: &dyn EmbeddingProvider,
    ) -> Result<ReindexStats, MemoryError> {
        todo!()
    }

    pub fn search(
        &self,
        _query: &[f32],
        _k: usize,
        _allowed: &BTreeSet<DocSource>,
    ) -> Result<Vec<(DocChunk, f32)>, MemoryError> {
        todo!()
    }

    pub fn gc(
        &self,
        _live: &BTreeSet<DocSource>,
        _keep_worktree_keys: bool,
    ) -> Result<usize, MemoryError> {
        todo!()
    }
}

#[cfg(test)]
pub(crate) mod test_embedder {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{DocChunk, DocSource, EmbeddingProvider};
    use crate::memory::error::EmbeddingError;

    /// Counts embedded texts; a known text gets its fixed vector, anything else `[0.5; dim]`.
    /// `largest` = texts in the biggest single request. Model "fake" by default, like
    /// `store_for_test`'s index id: every fixture embedder passes the index-id gate unless a
    /// test means to fail it.
    pub(crate) struct FakeEmbedder {
        model: String,
        dim: usize,
        fixed: BTreeMap<String, Vec<f32>>,
        calls: AtomicUsize,
        largest: AtomicUsize,
    }

    impl FakeEmbedder {
        pub(crate) fn new(dim: usize) -> Self {
            Self {
                model: "fake".into(),
                dim,
                fixed: BTreeMap::new(),
                calls: AtomicUsize::new(0),
                largest: AtomicUsize::new(0),
            }
        }

        /// `dim` = the length of the first vector (the tests pass 3, the store's).
        pub(crate) fn with_vectors(v: &[(&str, Vec<f32>)]) -> Self {
            Self {
                fixed: v.iter().map(|(t, x)| (t.to_string(), x.clone())).collect(),
                ..Self::new(v.first().map_or(3, |(_, x)| x.len()))
            }
        }

        /// Another model name, for a store opened under `doc_index_id(model, dim)`.
        pub(crate) fn named(self, model: &str) -> Self {
            Self {
                model: model.into(),
                ..self
            }
        }

        pub(crate) fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }

        pub(crate) fn largest_request(&self) -> usize {
            self.largest.load(Ordering::SeqCst)
        }
    }

    #[async_trait::async_trait]
    impl EmbeddingProvider for FakeEmbedder {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            self.calls.fetch_add(texts.len(), Ordering::SeqCst);
            self.largest.fetch_max(texts.len(), Ordering::SeqCst);
            Ok(texts
                .iter()
                .map(|t| {
                    self.fixed
                        .get(t)
                        .cloned()
                        .unwrap_or_else(|| vec![0.5; self.dim])
                })
                .collect())
        }
        fn model_id(&self) -> &str {
            &self.model
        }
        fn dim(&self) -> usize {
            self.dim
        }
        fn query_prefix(&self) -> &str {
            ""
        }
        fn document_prefix(&self) -> &str {
            ""
        }
    }

    /// An embedder that always fails. Model "fake", dim 3 (the fixture stores'): it passes the
    /// index-id gate and fails in `embed()`, the path its tests guard.
    pub(crate) struct FailingEmbedder;

    #[async_trait::async_trait]
    impl EmbeddingProvider for FailingEmbedder {
        async fn embed(&self, _t: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            Err(EmbeddingError::Http("down".into()))
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

    /// A chunk of `path` with the hash `chunk_markdown` gives its section.
    pub(crate) fn chunk(path: &str, section: &str, text: &str) -> DocChunk {
        let hash = magi_rs::review::docs::chunk_markdown(path, &format!("# {section}\n{text}\n"))
            .into_iter()
            .next()
            .expect("one chunk")
            .hash;
        DocChunk {
            path: path.into(),
            section: section.into(),
            text: text.into(),
            hash,
        }
    }

    /// A documentation source: a path and the key of its content.
    pub(crate) fn src(path: &str, key: &str) -> DocSource {
        DocSource {
            path: path.into(),
            key: key.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_embedder::*;
    use super::*;
    use crate::memory::error::EmbeddingError;

    fn store_for_test() -> (ReviewDocStore, tempfile::NamedTempFile) {
        let tmp = tempfile::NamedTempFile::new().expect("tmp");
        let mem = crate::system::database::EncryptedSqliteMemory::new(
            tmp.path().to_path_buf(),
            zeroize::Zeroizing::new(crate::system::database::TEST_MASTER_PASSPHRASE.to_string()),
        )
        .expect("db");
        (
            ReviewDocStore::new(
                mem.shared_conn(),
                mem.data_key().expect("dek"),
                magi_rs::review::docs::doc_index_id("fake", 3),
            )
            .expect("store"),
            tmp,
        )
    }

    /// The `dim` column of every row.
    fn dims(store: &ReviewDocStore) -> Vec<i64> {
        store
            .connection_for_test()
            .lock()
            .expect("lock")
            .prepare("SELECT dim FROM code_review_docs ORDER BY key, chunk_hash")
            .expect("prep")
            .query_map([], |r| r.get(0))
            .expect("q")
            .collect::<Result<_, _>>()
            .expect("rows")
    }

    #[tokio::test]
    async fn an_indexed_source_is_never_read_or_embedded_again() {
        // `missing` is what the caller reads; a second reindex of the same source does nothing.
        let (store, _dir) = store_for_test();
        let e = FakeEmbedder::new(3);
        let wanted = [src("README.md", "k1"), src("docs/b.md", "k2")];
        assert_eq!(store.missing(&wanted).expect("missing").len(), 2);
        let s = store
            .reindex(
                &[(src("README.md", "k1"), vec![chunk("README.md", "A", "x")])],
                &e,
            )
            .await
            .expect("first");
        assert_eq!((s.embedded, s.sources), (1, 1));
        assert_eq!(
            store.missing(&wanted).expect("missing"),
            vec![src("docs/b.md", "k2")]
        );
        let again = store
            .reindex(
                &[(src("README.md", "k1"), vec![chunk("README.md", "A", "x")])],
                &e,
            )
            .await
            .expect("again");
        assert_eq!((again.embedded, again.sources, e.calls()), (0, 0, 1));
    }

    #[tokio::test]
    async fn a_source_without_doc_chunks_is_remembered_as_indexed() {
        // A file with no doc comment would otherwise be read again on every review.
        let (store, _dir) = store_for_test();
        let e = FakeEmbedder::new(3);
        store
            .reindex(&[(src("src/m.rs", "k3"), vec![])], &e)
            .await
            .expect("indexed");
        assert!(store
            .missing(&[src("src/m.rs", "k3")])
            .expect("missing")
            .is_empty());
        assert_eq!(e.calls(), 0);
    }

    #[tokio::test]
    async fn a_chunk_shared_by_two_revisions_is_embedded_once() {
        // RV-S37: the second README differs in one section only.
        let (store, _dir) = store_for_test();
        let e = FakeEmbedder::new(3);
        store
            .reindex(
                &[(
                    src("README.md", "k1"),
                    vec![chunk("README.md", "A", "x"), chunk("README.md", "B", "y")],
                )],
                &e,
            )
            .await
            .expect("first");
        let s = store
            .reindex(
                &[(
                    src("README.md", "k2"),
                    vec![chunk("README.md", "A", "x"), chunk("README.md", "B", "y2")],
                )],
                &e,
            )
            .await
            .expect("second");
        assert_eq!((s.embedded, s.reused), (1, 1));
        assert_eq!(e.calls(), 3);
        // The reused row is served under the second source (rows carry their source; the chunk
        // hash includes the path, so two paths never share a row).
        let hits = store
            .search(
                &[0.5, 0.5, 0.5],
                5,
                &BTreeSet::from([src("README.md", "k2")]),
            )
            .expect("search");
        assert_eq!(hits.len(), 2, "{hits:?}");
    }

    #[tokio::test]
    async fn search_only_considers_the_allowed_sources() {
        // RV-S36: the other revision's chunk is the more similar one, and it is never returned.
        let (store, _dir) = store_for_test();
        // Three dimensions, the store's: a 2-dim embedder was refused by the index-id gate.
        let e = FakeEmbedder::with_vectors(&[
            ("old docs", vec![1.0, 0.0, 0.0]),
            ("new docs", vec![0.0, 1.0, 0.0]),
        ]);
        store
            .reindex(
                &[
                    (
                        src("README.md", "old"),
                        vec![chunk("README.md", "U", "old docs")],
                    ),
                    (
                        src("README.md", "new"),
                        vec![chunk("README.md", "U", "new docs")],
                    ),
                ],
                &e,
            )
            .await
            .expect("ok");
        let hits = store
            .search(
                &[1.0, 0.0, 0.0],
                5,
                &BTreeSet::from([src("README.md", "new")]),
            )
            .expect("search");
        assert_eq!(
            hits.iter()
                .map(|(c, _)| c.text.as_str())
                .collect::<Vec<_>>(),
            vec!["new docs"]
        );
    }

    #[tokio::test]
    async fn search_returns_the_most_similar_chunk_first() {
        let (store, _dir) = store_for_test();
        let e = FakeEmbedder::with_vectors(&[
            ("about dogs", vec![1.0, 0.0, 0.0]),
            ("about cats", vec![0.0, 1.0, 0.0]),
        ]);
        store
            .reindex(
                &[(
                    src("d.md", "k"),
                    vec![
                        chunk("d.md", "dogs", "about dogs"),
                        chunk("d.md", "cats", "about cats"),
                    ],
                )],
                &e,
            )
            .await
            .expect("ok");
        let hits = store
            .search(&[0.9, 0.1, 0.0], 1, &BTreeSet::from([src("d.md", "k")]))
            .expect("search");
        assert_eq!(hits[0].0.section, "dogs");
    }

    #[tokio::test]
    async fn a_dimension_change_forces_a_reindex() {
        // The dimension is part of the index id (T3.2): `missing` sees the change before anything
        // is read, so a review indexes every source it wants instead of serving a subset.
        let tmp = tempfile::NamedTempFile::new().expect("tmp");
        let mem = crate::system::database::EncryptedSqliteMemory::new(
            tmp.path().to_path_buf(),
            zeroize::Zeroizing::new(crate::system::database::TEST_MASTER_PASSPHRASE.to_string()),
        )
        .expect("db");
        use magi_rs::review::docs::doc_index_id;
        let d3 = ReviewDocStore::new(
            mem.shared_conn(),
            mem.data_key().expect("dek"),
            doc_index_id("fake", 3),
        )
        .expect("d3");
        d3.reindex(
            &[(src("d.md", "k1"), vec![chunk("d.md", "A", "x")])],
            &FakeEmbedder::new(3),
        )
        .await
        .expect("dim 3");
        let d5 = ReviewDocStore::new(
            mem.shared_conn(),
            mem.data_key().expect("dek"),
            doc_index_id("fake", 5),
        )
        .expect("d5");
        assert_eq!(
            d5.missing(&[src("d.md", "k1")]).expect("missing"),
            vec![src("d.md", "k1")]
        );
        // The same chunk under another source: its stored vector has the old dimension, so it is
        // embedded again instead of reused, and the old source is gone with its rows.
        let s = d5
            .reindex(
                &[(
                    src("d.md", "k2"),
                    vec![chunk("d.md", "A", "x"), chunk("d.md", "B", "y")],
                )],
                &FakeEmbedder::new(5),
            )
            .await
            .expect("dim 5");
        assert_eq!((s.embedded, s.reused, s.rebuilt), (2, 0, true));
        assert_eq!(dims(&d5), vec![5, 5]);
    }

    #[tokio::test]
    async fn another_embedding_model_rebuilds_the_index_even_with_nothing_new() {
        // Same dimension, another model: the old vectors are not comparable. Every source is
        // missing again, so the next review re-indexes instead of serving no documentation,
        // review after review, in silence.
        let tmp = tempfile::NamedTempFile::new().expect("tmp");
        let mem = crate::system::database::EncryptedSqliteMemory::new(
            tmp.path().to_path_buf(),
            zeroize::Zeroizing::new(crate::system::database::TEST_MASTER_PASSPHRASE.to_string()),
        )
        .expect("db");
        use magi_rs::review::docs::doc_index_id;
        let old = ReviewDocStore::new(
            mem.shared_conn(),
            mem.data_key().expect("dek"),
            doc_index_id("m1", 3),
        )
        .expect("old");
        // Each store gets its own model's embedder (the gate refuses any other).
        old.reindex(
            &[(src("d.md", "k"), vec![chunk("d.md", "A", "x")])],
            &FakeEmbedder::new(3).named("m1"),
        )
        .await
        .expect("m1");
        let new = ReviewDocStore::new(
            mem.shared_conn(),
            mem.data_key().expect("dek"),
            doc_index_id("m2", 3),
        )
        .expect("new");
        let wanted = BTreeSet::from([src("d.md", "k")]);
        assert!(
            new.search(&[0.5, 0.5, 0.5], 5, &wanted)
                .expect("search")
                .is_empty(),
            "never compared across models"
        );
        assert_eq!(
            new.missing(&[src("d.md", "k")]).expect("missing"),
            vec![src("d.md", "k")]
        );
        let s = new
            .reindex(
                &[(src("d.md", "k"), vec![chunk("d.md", "A", "x")])],
                &FakeEmbedder::new(3).named("m2"),
            )
            .await
            .expect("m2");
        assert_eq!((s.embedded, s.reused, s.rebuilt), (1, 0, true));
        assert_eq!(
            new.search(&[0.5, 0.5, 0.5], 5, &wanted)
                .expect("search")
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn embedding_requests_are_bounded_and_written_once() {
        // Thousands of chunks never go in one request; all requests run before the one write.
        let (store, _dir) = store_for_test();
        let e = FakeEmbedder::new(3);
        let n = EMBED_BATCH_ITEMS * 2 + 1;
        let many: Vec<DocChunk> = (0..n)
            .map(|i| chunk("d.md", &format!("S{i}"), &format!("t{i}")))
            .collect();
        let s = store
            .reindex(&[(src("d.md", "k"), many)], &e)
            .await
            .expect("ok");
        assert_eq!((s.embedded, e.calls()), (n, n));
        assert_eq!(e.largest_request(), EMBED_BATCH_ITEMS);
        assert_eq!(dims(&store).len(), n);
    }

    #[tokio::test]
    async fn a_reindex_whose_embedder_fails_leaves_the_previous_index_whole() {
        // Embed first, write once.
        let (store, _dir) = store_for_test();
        store
            .reindex(
                &[(
                    src("d.md", "k1"),
                    vec![chunk("d.md", "A", "x"), chunk("d.md", "B", "y")],
                )],
                &FakeEmbedder::new(3),
            )
            .await
            .expect("dim 3");
        assert!(store
            .reindex(
                &[(src("d.md", "k2"), vec![chunk("d.md", "A", "x2")])],
                &FailingEmbedder
            )
            .await
            .is_err());
        assert_eq!(
            dims(&store),
            vec![3, 3],
            "the index before the failed reindex, whole"
        );
        assert_eq!(
            store.missing(&[src("d.md", "k2")]).expect("missing"),
            vec![src("d.md", "k2")],
            "nothing recorded"
        );
    }

    #[tokio::test]
    async fn search_accepts_more_allowed_sources_than_sqlite_parameters() {
        // `allowed` never goes as one `IN (...)` (999 parameters in older SQLite builds).
        let (store, _dir) = store_for_test();
        let all: Vec<(DocSource, Vec<DocChunk>)> = (0..1_200)
            .map(|i| {
                (
                    src(&format!("docs/{i}.md"), "k"),
                    vec![chunk(&format!("docs/{i}.md"), "S", &format!("t{i}"))],
                )
            })
            .collect();
        store
            .reindex(&all, &FakeEmbedder::new(3))
            .await
            .expect("index");
        let allowed: BTreeSet<DocSource> = all.iter().map(|(s, _)| s.clone()).collect();
        assert_eq!(
            store
                .search(&[0.5, 0.5, 0.5], 3, &allowed)
                .expect("search")
                .len(),
            3
        );
    }

    #[tokio::test]
    async fn the_clean_up_accepts_more_live_sources_than_sqlite_parameters() {
        // gc's `live` set never goes as one `IN (...)` either. Chunkless sources: no embed.
        let (store, _dir) = store_for_test();
        let live: Vec<(DocSource, Vec<DocChunk>)> = (0..1_200)
            .map(|i| (src(&format!("src/{i}.rs"), "k"), vec![]))
            .collect();
        let mut all = live.clone();
        all.push((src("src/gone.rs", "k"), vec![]));
        store
            .reindex(&all, &FakeEmbedder::new(3))
            .await
            .expect("index");
        let live_set: BTreeSet<DocSource> = live.iter().map(|(s, _)| s.clone()).collect();
        assert_eq!(
            store.gc(&live_set, false).expect("gc"),
            1,
            "only the source no ref reaches"
        );
        assert!(store
            .missing(&live.iter().map(|(s, _)| s.clone()).collect::<Vec<_>>())
            .expect("missing")
            .is_empty());
    }

    #[tokio::test]
    async fn a_source_without_chunks_is_asked_again_after_an_index_change() {
        // Its only trace is its source row: that row carries the index identity too.
        let tmp = tempfile::NamedTempFile::new().expect("tmp");
        let mem = crate::system::database::EncryptedSqliteMemory::new(
            tmp.path().to_path_buf(),
            zeroize::Zeroizing::new(crate::system::database::TEST_MASTER_PASSPHRASE.to_string()),
        )
        .expect("db");
        use magi_rs::review::docs::doc_index_id;
        let old = ReviewDocStore::new(
            mem.shared_conn(),
            mem.data_key().expect("dek"),
            doc_index_id("m1", 3),
        )
        .expect("old");
        old.reindex(
            &[(src("src/m.rs", "k"), vec![])],
            &FakeEmbedder::new(3).named("m1"),
        )
        .await
        .expect("m1");
        let new = ReviewDocStore::new(
            mem.shared_conn(),
            mem.data_key().expect("dek"),
            doc_index_id("m2", 3),
        )
        .expect("new");
        assert_eq!(
            new.missing(&[src("src/m.rs", "k")]).expect("missing"),
            vec![src("src/m.rs", "k")]
        );
    }

    #[tokio::test]
    async fn search_skips_rows_of_another_dimension() {
        // A query of another dimension skips every row, never compares them.
        let (store, _dir) = store_for_test();
        store
            .reindex(
                &[(src("d.md", "k"), vec![chunk("d.md", "A", "x")])],
                &FakeEmbedder::new(3),
            )
            .await
            .expect("dim 3");
        assert!(store
            .search(&[1.0, 0.0], 5, &BTreeSet::from([src("d.md", "k")]))
            .expect("search")
            .is_empty());
    }

    #[tokio::test]
    async fn the_clean_up_deletes_every_source_that_is_not_live() {
        // RV-S39: rows and source go together.
        let (store, _dir) = store_for_test();
        let e = FakeEmbedder::new(3);
        let wt = src("README.md", &magi_rs::review::docs::worktree_key(b"w"));
        store
            .reindex(
                &[
                    (src("README.md", "k1"), vec![chunk("README.md", "A", "x")]),
                    (src("README.md", "k2"), vec![chunk("README.md", "A", "y")]),
                    (wt.clone(), vec![chunk("README.md", "A", "z")]),
                ],
                &e,
            )
            .await
            .expect("index");
        assert_eq!(
            store
                .gc(&BTreeSet::from([src("README.md", "k2")]), false)
                .expect("gc"),
            2
        );
        assert_eq!(
            store
                .missing(&[src("README.md", "k1"), src("README.md", "k2"), wt.clone()])
                .expect("missing"),
            vec![src("README.md", "k1"), wt]
        );
        assert_eq!(dims(&store).len(), 1, "only k2's row is left");
    }

    #[tokio::test]
    async fn after_a_revision_review_the_clean_up_keeps_working_tree_sources() {
        // A revision review never listed the working tree, so it cannot judge those sources.
        let (store, _dir) = store_for_test();
        let e = FakeEmbedder::new(3);
        let wt = src("README.md", &magi_rs::review::docs::worktree_key(b"w"));
        store
            .reindex(
                &[
                    (src("README.md", "k1"), vec![chunk("README.md", "A", "x")]),
                    (wt.clone(), vec![]),
                ],
                &e,
            )
            .await
            .expect("index");
        assert_eq!(store.gc(&BTreeSet::new(), true).expect("gc"), 1);
        assert!(store.missing(&[wt]).expect("missing").is_empty());
    }

    #[tokio::test]
    async fn the_text_is_stored_encrypted() {
        let (store, _tmp) = store_for_test();
        store
            .reindex(
                &[(
                    src("d.md", "k"),
                    vec![chunk("d.md", "SECTION-MARKER-PT", "PLAINTEXT-MARKER")],
                )],
                &FakeEmbedder::new(3),
            )
            .await
            .expect("ok");
        let (text, section, vector): (Vec<u8>, Vec<u8>, Vec<u8>) = store
            .connection_for_test()
            .lock()
            .expect("lock")
            .query_row(
                "SELECT text_blob, section_blob, embedding_blob FROM code_review_docs",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .expect("row");
        assert!(
            !text.windows(16).any(|w| w == b"PLAINTEXT-MARKER"),
            "text stored in the clear"
        );
        // The section and the vector are encrypted too (FakeEmbedder answers 0.5 per dimension).
        assert!(
            !section.windows(17).any(|w| w == b"SECTION-MARKER-PT"),
            "section stored in the clear"
        );
        let clear: Vec<u8> = [0.5f32; 3].iter().flat_map(|f| f.to_le_bytes()).collect();
        assert!(
            !vector.windows(clear.len()).any(|w| w == clear.as_slice()),
            "embedding stored in the clear"
        );
    }

    #[tokio::test]
    async fn two_identical_chunks_in_one_reindex_are_stored_once() {
        // Repeated boilerplate sections hash alike, and (key, chunk_hash) is the primary key.
        let (store, _dir) = store_for_test();
        let e = FakeEmbedder::new(3);
        let s = store
            .reindex(
                &[(
                    src("d.md", "k"),
                    vec![chunk("d.md", "A", "x"), chunk("d.md", "A", "x")],
                )],
                &e,
            )
            .await
            .expect("not an error");
        assert_eq!((s.embedded, e.calls()), (1, 1));
    }

    #[tokio::test]
    async fn an_embedder_answering_mixed_dimensions_fails_the_reindex_whole() {
        // One stored dimension: a response mixing lengths writes nothing.
        let (store, _dir) = store_for_test();
        store
            .reindex(
                &[(src("d.md", "k1"), vec![chunk("d.md", "A", "x")])],
                &FakeEmbedder::new(3),
            )
            .await
            .expect("dim 3");
        // Declares 3 (first vector), the store's, so it passes the index-id gate; "z" is 2 long.
        let mixed =
            FakeEmbedder::with_vectors(&[("y", vec![1.0, 0.0, 0.0]), ("z", vec![1.0, 0.0])]);
        let all = vec![chunk("d.md", "B", "y"), chunk("d.md", "C", "z")];
        assert!(store
            .reindex(&[(src("d.md", "k2"), all)], &mixed)
            .await
            .is_err());
        assert_eq!(
            dims(&store),
            vec![3],
            "the index before the failed reindex, whole"
        );
    }

    /// Declares one dimension and answers another.
    struct MislabeledEmbedder;
    #[async_trait::async_trait]
    impl EmbeddingProvider for MislabeledEmbedder {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            Ok(texts.iter().map(|_| vec![0.5; 5]).collect())
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
    async fn an_embedder_answering_another_length_than_it_declares_fails_the_reindex() {
        // The index id carries dim(): a vector of another length would mislabel the whole index.
        let (store, _dir) = store_for_test();
        store
            .reindex(
                &[(src("d.md", "k1"), vec![chunk("d.md", "A", "x")])],
                &FakeEmbedder::new(3),
            )
            .await
            .expect("dim 3");
        assert!(store
            .reindex(
                &[(src("d.md", "k2"), vec![chunk("d.md", "B", "y")])],
                &MislabeledEmbedder
            )
            .await
            .is_err());
        assert_eq!(dims(&store), vec![3], "nothing written");
    }

    /// Another model with the same dimension as the store's.
    struct OtherModelEmbedder;
    #[async_trait::async_trait]
    impl EmbeddingProvider for OtherModelEmbedder {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            Ok(texts.iter().map(|_| vec![0.5; 3]).collect())
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

    #[tokio::test]
    async fn an_embedder_of_another_model_is_refused_even_with_the_same_dimension() {
        // The store's index_id names a model: a same-length vector of another model never lands
        // under it.
        let (store, _dir) = store_for_test();
        assert!(store
            .reindex(
                &[(src("d.md", "k1"), vec![chunk("d.md", "A", "x")])],
                &OtherModelEmbedder
            )
            .await
            .is_err());
        assert!(dims(&store).is_empty(), "nothing written");
    }

    #[tokio::test]
    async fn a_chunk_with_a_secret_is_refused_before_anything_is_embedded() {
        // Defense in depth (RV22): a caller that skipped without_secrets fails here, before
        // embed().
        let (store, _dir) = store_for_test();
        let e = FakeEmbedder::new(3);
        let text = "export KEY=sk-ant-api03-AAAAAAAAAAAAAAAAAAAA"; // allow-secret-scan
        let r = store
            .reindex(
                &[
                    (src("d.md", "k1"), vec![chunk("d.md", "A", "x")]),
                    (src("leak.md", "k2"), vec![chunk("leak.md", "Setup", text)]),
                ],
                &e,
            )
            .await;
        assert!(
            matches!(&r, Err(MemoryError::Storage(m)) if m.contains("leak.md") && !m.contains("sk-ant-")),
            "{:?}",
            r.as_ref().err()
        );
        assert_eq!(e.calls(), 0);
    }

    /// Embeds like FakeEmbedder and records whether the DB lock was taken while it ran.
    struct LockProbe {
        conn: Arc<Mutex<rusqlite::Connection>>,
        calls: std::sync::atomic::AtomicUsize,
        held: std::sync::atomic::AtomicBool,
    }
    #[async_trait::async_trait]
    impl EmbeddingProvider for LockProbe {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if self.conn.try_lock().is_err() {
                self.held.store(true, std::sync::atomic::Ordering::SeqCst);
            }
            Ok(texts.iter().map(|_| vec![0.5; 3]).collect())
        }
        fn model_id(&self) -> &str {
            "fake" // passes the index-id gate, so embed() runs
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
    async fn the_database_lock_is_free_while_the_embedder_runs() {
        // CLAUDE.md: never hold the DB lock across an await (it would block every other DB user).
        let (store, _dir) = store_for_test();
        let probe = LockProbe {
            conn: Arc::clone(store.connection_for_test()),
            calls: Default::default(),
            held: Default::default(),
        };
        store
            .reindex(&[(src("d.md", "k"), vec![chunk("d.md", "A", "x")])], &probe)
            .await
            .expect("ok");
        assert_eq!(probe.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert!(
            !probe.held.load(std::sync::atomic::Ordering::SeqCst),
            "the DB lock was held during embed()"
        );
    }
}
