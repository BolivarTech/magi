// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-11-05

//! Encrypted store of remembered review findings (RV07(4), RV08).

use std::sync::{Arc, Mutex};

use rusqlite::Connection;

use crate::memory::error::MemoryError;
use magi_rs::review::history::{HistoryEntry, HistoryStatus};
use magi_rs::vault::MaskedDek;

/// Red stub.
#[allow(dead_code)] // Red stub: removed in Green
pub struct ReviewHistoryStore {
    conn: Arc<Mutex<Connection>>,
}

#[allow(dead_code)] // Red stub: removed in Green
impl ReviewHistoryStore {
    #[cfg(test)]
    pub(crate) fn connection_for_test(&self) -> &Arc<Mutex<Connection>> {
        &self.conn
    }

    pub fn new(conn: Arc<Mutex<Connection>>, _dek: MaskedDek) -> Result<Self, MemoryError> {
        Ok(Self { conn })
    }

    pub fn record(&self, _entries: &[HistoryEntry]) -> Result<usize, MemoryError> {
        Ok(0)
    }

    #[cfg(test)]
    pub(crate) fn all(&self) -> Result<Vec<HistoryEntry>, MemoryError> {
        Ok(Vec::new())
    }

    pub fn due(&self, _limit: usize) -> Result<(Vec<HistoryEntry>, usize), MemoryError> {
        Ok((Vec::new(), 0))
    }

    pub fn validated_on_files(&self, _files: &[String]) -> Result<Vec<HistoryEntry>, MemoryError> {
        Ok(Vec::new())
    }

    pub fn touch(&self, _ids: &[String], _now: i64) -> Result<(), MemoryError> {
        Ok(())
    }

    pub fn update_status(&self, _id: &str, _status: HistoryStatus) -> Result<(), MemoryError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use magi_rs::review::history::{HistoryEntry, HistoryStatus};

    use crate::memory::error::MemoryError;

    fn store_for_test() -> (ReviewHistoryStore, tempfile::NamedTempFile) {
        let tmp = tempfile::NamedTempFile::new().expect("tmp");
        let mem = crate::system::database::EncryptedSqliteMemory::new(
            tmp.path().to_path_buf(),
            zeroize::Zeroizing::new(crate::system::database::TEST_MASTER_PASSPHRASE.to_string()),
        )
        .expect("db");
        (
            ReviewHistoryStore::new(mem.shared_conn(), mem.data_key().expect("dek"))
                .expect("store"),
            tmp,
        )
    }

    fn stored(id: &str, line: Option<u32>, commit: &str) -> HistoryEntry {
        HistoryEntry {
            id: id.into(),
            file: "src/a.rs".into(),
            line,
            reviewed_commit: commit.into(),
            title: format!("title {id}"),
            detail: format!("detail {id}"),
            status: HistoryStatus::Pending,
        }
    }

    fn sorted(mut v: Vec<HistoryEntry>) -> Vec<HistoryEntry> {
        v.sort_by(|a, b| a.id.cmp(&b.id));
        v
    }

    fn raw_status(store: &ReviewHistoryStore, id: &str) -> String {
        store
            .connection_for_test()
            .lock()
            .expect("lock")
            .query_row(
                "SELECT status FROM code_review_history WHERE id = ?1",
                [id],
                |r| r.get(0),
            )
            .expect("row")
    }

    #[test]
    fn recorded_entries_come_back_whole() {
        let (store, _tmp) = store_for_test();
        let entries = vec![stored("i1", Some(7), "c1"), stored("i2", None, "c1")];
        assert_eq!(store.record(&entries).expect("record"), 2);
        assert_eq!(sorted(store.all().expect("all")), sorted(entries));
    }

    #[test]
    fn a_merged_title_and_detail_with_newlines_come_back_whole() {
        // [CP2 MS3 loop 3] entries_from_review joins merged titles with newlines; a separator join would split them.
        let (store, _tmp) = store_for_test();
        let e = HistoryEntry {
            title: "t1\nt2".into(),
            detail: "d1\n\nd2\n".into(),
            ..stored("i1", Some(7), "c1")
        };
        store.record(std::slice::from_ref(&e)).expect("record");
        assert_eq!(store.all().expect("all"), vec![e]);
    }

    #[test]
    fn the_finding_text_is_stored_encrypted() {
        let (store, _tmp) = store_for_test();
        let mut e = stored("i1", Some(7), "c1");
        e.title = "TITLE-PLAINTEXT-MARKER".into();
        e.detail = "DETAIL-PLAINTEXT-MARKER".into();
        store.record(&[e]).expect("record");
        let raw: Vec<u8> = store
            .connection_for_test()
            .lock()
            .expect("lock")
            .query_row("SELECT finding_blob FROM code_review_history", [], |r| {
                r.get(0)
            })
            .expect("row");
        for marker in [
            &b"TITLE-PLAINTEXT-MARKER"[..],
            &b"DETAIL-PLAINTEXT-MARKER"[..],
        ] {
            assert!(
                !raw.windows(marker.len()).any(|w| w == marker),
                "finding text stored in the clear"
            );
        }
    }

    #[test]
    fn update_status_changes_only_that_entry() {
        let (store, _tmp) = store_for_test();
        store
            .record(&[stored("i1", Some(7), "c1"), stored("i2", Some(9), "c1")])
            .expect("record");
        store
            .update_status("i1", HistoryStatus::Attended)
            .expect("update");
        let all = sorted(store.all().expect("all"));
        assert_eq!(
            (all[0].id.as_str(), all[0].status),
            ("i1", HistoryStatus::Attended)
        );
        assert_eq!(
            (all[1].id.as_str(), all[1].status),
            ("i2", HistoryStatus::Pending)
        );
        // Persisted in the row, in the serde spelling, not only cached in the store.
        assert_eq!(raw_status(&store, "i1"), "attended");
        assert_eq!(raw_status(&store, "i2"), "pending");
    }

    #[test]
    fn updating_an_unknown_id_is_an_error() {
        let (store, _tmp) = store_for_test();
        store
            .record(&[stored("i1", Some(7), "c1")])
            .expect("record");
        let err = store
            .update_status("nope", HistoryStatus::Attended)
            .expect_err("unknown id");
        assert!(
            matches!(&err, MemoryError::Storage(m) if m.contains("nope")),
            "{err}"
        );
        assert_eq!(
            store.all().expect("all")[0].status,
            HistoryStatus::Pending,
            "the known entry is untouched"
        );
    }

    #[test]
    fn due_entries_rotate_so_a_stuck_one_never_starves_the_rest() {
        // [CP2 MS3 loop 6] An entry that stays Pending forever is checked again only after every other due entry.
        let (store, _tmp) = store_for_test();
        store
            .record(&[
                stored("a", Some(1), "c1"),
                stored("b", Some(2), "c1"),
                stored("c", Some(3), "c1"),
                stored("d", Some(4), "c1"),
            ])
            .expect("record");
        store
            .update_status("d", HistoryStatus::Attended)
            .expect("validated");
        let ids = |v: Vec<HistoryEntry>| v.into_iter().map(|e| e.id).collect::<Vec<_>>();
        let (first, total) = store.due(2).expect("due");
        // [CP2 MS3 loop 14 (pasada final)] Attended is due too (a revert can undo it), so the total counts all four.
        assert_eq!(
            (ids(first), total),
            (vec!["a".to_string(), "b".to_string()], 4)
        );
        store
            .touch(&["a".to_string(), "b".to_string()], 100)
            .expect("touch");
        let (second, _) = store.due(2).expect("due");
        assert_eq!(
            ids(second),
            vec!["c".to_string(), "d".to_string()],
            "least recently checked first"
        );
    }

    #[test]
    fn touching_an_unknown_id_is_an_error() {
        // [CP2 MS3 loop 7] Same contract as update_status: never a silent success.
        let (store, _tmp) = store_for_test();
        store
            .record(&[stored("i1", Some(7), "c1")])
            .expect("record");
        let err = store
            .touch(&["i1".to_string(), "nope".to_string()], 5)
            .expect_err("unknown id");
        assert!(
            matches!(&err, MemoryError::Storage(m) if m.contains("nope")),
            "{err}"
        );
    }

    #[test]
    fn only_validated_entries_on_the_given_files_are_read_back() {
        // [CP2 MS3 loop 7] The context read: validated entries of the batch's files; never Pending, never another file.
        let (store, _tmp) = store_for_test();
        let other = HistoryEntry {
            file: "src/b.rs".into(),
            ..stored("o", Some(1), "c1")
        };
        store
            .record(&[
                stored("a", Some(1), "c1"),
                stored("p", Some(2), "c1"),
                stored("w", Some(3), "c1"),
                other,
            ])
            .expect("record");
        store
            .update_status("a", HistoryStatus::Attended)
            .expect("a");
        store
            .update_status("w", HistoryStatus::WeakReject)
            .expect("w");
        store
            .update_status("o", HistoryStatus::Attended)
            .expect("o");
        let ids: Vec<String> = sorted(
            store
                .validated_on_files(&["src/a.rs".to_string()])
                .expect("read"),
        )
        .into_iter()
        .map(|e| e.id)
        .collect();
        assert_eq!(ids, vec!["a".to_string(), "w".to_string()]);
        assert!(store.validated_on_files(&[]).expect("read").is_empty());
    }

    #[test]
    fn validated_on_files_accepts_more_files_than_sqlite_parameters() {
        // [CP2 MS3 loop 14 (pasada final)] A mass rename touches more than 999 files: never one `IN (...)`.
        let (store, _tmp) = store_for_test();
        let a = HistoryEntry {
            file: "src/f1199.rs".into(),
            ..stored("a", Some(1), "c1")
        };
        store.record(&[a]).expect("record");
        store
            .update_status("a", HistoryStatus::Attended)
            .expect("a");
        let files: Vec<String> = (0..1_200).map(|i| format!("src/f{i}.rs")).collect();
        let ids: Vec<String> = store
            .validated_on_files(&files)
            .expect("read")
            .into_iter()
            .map(|e| e.id)
            .collect();
        assert_eq!(
            ids,
            vec!["a".to_string()],
            "the last file of the list is reached"
        );
    }

    #[test]
    fn recording_the_same_finding_again_keeps_the_first_sighting() {
        let (store, _tmp) = store_for_test();
        assert_eq!(
            store.record(&[stored("i1", Some(7), "c1")]).expect("first"),
            1
        );
        let mut again = stored("i1", Some(8), "c9");
        again.title = "reworded".into();
        assert_eq!(
            store.record(&[again]).expect("second"),
            0,
            "an existing id is not inserted again"
        );
        let all = store.all().expect("all");
        assert_eq!(all.len(), 1);
        assert_eq!(
            (
                all[0].reviewed_commit.as_str(),
                all[0].line,
                all[0].title.as_str()
            ),
            ("c1", Some(7), "title i1"),
            "the first sighting wins, so classify keeps comparing from the first reviewed commit"
        );
    }
}
