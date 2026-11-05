// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-11-05

//! Finding history for the `code-review` context (RV07(4), RV08).
//!
//! A finding the trio reported and the gate kept is remembered with the commit it was reviewed
//! at. Later, git alone says what became of it: the line changed in a later commit (attended),
//! or the code merged untouched (weak reject). A model never validates history (RV08).

use super::source::GitCommand;

/// What became of a remembered finding, judged from git alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoryStatus {
    /// No signal yet (no line, git failed, commit gone, or no later commit).
    Pending,
    /// The reported line changed in a later commit.
    Attended,
    /// A later commit exists and the reported line is untouched.
    WeakReject,
}

/// One remembered finding.
#[derive(Debug, Clone, PartialEq)]
pub struct HistoryEntry {
    /// The stable finding id.
    pub id: String,
    /// The reported file.
    pub file: String,
    /// The reported 1-based line.
    pub line: Option<u32>,
    /// The commit the finding was first reviewed at.
    pub reviewed_commit: String,
    /// The merged titles.
    pub title: String,
    /// The merged details.
    pub detail: String,
    /// The current status.
    pub status: HistoryStatus,
}

/// Most existing entries reclassified per review.
pub const MAX_RECLASSIFY_PER_REVIEW: usize = 200;

/// Red stub.
pub fn classify(_entry: &HistoryEntry, _head: &str, _git: &dyn GitCommand) -> HistoryStatus {
    HistoryStatus::Attended
}

/// Red stub.
pub fn reclassify_then_record(
    _existing: &[HistoryEntry],
    _new: &[HistoryEntry],
    _head: &str,
    _git: &dyn GitCommand,
) -> (Vec<(String, HistoryStatus)>, Vec<HistoryEntry>) {
    (Vec::new(), Vec::new())
}

/// Red stub.
pub fn entries_from_review(
    _findings: &[(
        crate::review::consolidate::ConsolidatedFinding,
        crate::review::gate::GateVerdict,
    )],
    _reviewed_commit: &str,
) -> Vec<HistoryEntry> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::review::test_support::FakeGit;

    fn entry(line: Option<u32>) -> HistoryEntry {
        HistoryEntry {
            id: "i".into(),
            file: "a.rs".into(),
            line,
            reviewed_commit: "c1".into(),
            title: "t".into(),
            detail: "d".into(),
            status: HistoryStatus::Pending,
        }
    }

    #[test]
    fn lines_changed_after_the_review_mean_attended() {
        // [CP2 MS3 loop 1] Anchored to the reviewed commit: `-4,3` covers lines 4..=6 of c1.
        let git = FakeGit::from(&[
            ("merge-base --is-ancestor c1 HEAD", Ok("")),
            (DIFF_C1_HEAD, Ok("@@ -4,3 +4,3 @@\n")),
        ]);
        assert_eq!(
            classify(&entry(Some(5)), "HEAD", &git),
            HistoryStatus::Attended
        );
    }

    const DIFF_C1_HEAD: &str = "diff --no-color --no-ext-diff --no-textconv --no-renames --unified=0 --relative c1 HEAD -- a.rs";
    /// [CP2 MS3 loop 12] A weak reject needs a commit after the reviewed one.
    const HEAD_IS_C2: &str = "rev-parse --verify HEAD^{commit}";

    #[test]
    fn lines_inserted_above_do_not_count_as_a_change_to_the_finding_line() {
        // [CP2 MS3 loop 1] Measured: log -L traced HEAD's line 5, a different line once two were inserted above.
        let git = FakeGit::from(&[
            ("merge-base --is-ancestor c1 HEAD", Ok("")),
            (DIFF_C1_HEAD, Ok("@@ -0,0 +1,2 @@\n")),
            (HEAD_IS_C2, Ok("c2\n")),
        ]);
        assert_eq!(
            classify(&entry(Some(5)), "HEAD", &git),
            HistoryStatus::WeakReject
        );
    }

    #[test]
    fn a_hunk_header_with_function_context_is_read() {
        // [CP2 MS3 loop 13] git writes the enclosing function after the second @@; a strict parser would leave every
        // entry on a real Rust file Pending.
        let git = FakeGit::from(&[
            ("merge-base --is-ancestor c1 HEAD", Ok("")),
            (DIFF_C1_HEAD, Ok("@@ -4,3 +4,3 @@ fn foo() {\n")),
        ]);
        assert_eq!(
            classify(&entry(Some(5)), "HEAD", &git),
            HistoryStatus::Attended
        );
    }

    #[test]
    fn an_unparsable_hunk_header_leaves_the_entry_pending() {
        // [CP2 MS3 loop 6] A diff format this parser does not know never decides attended or weak_reject.
        let git = FakeGit::from(&[
            ("merge-base --is-ancestor c1 HEAD", Ok("")),
            (DIFF_C1_HEAD, Ok("@@ garbage @@\n")),
        ]);
        assert_eq!(
            classify(&entry(Some(5)), "HEAD", &git),
            HistoryStatus::Pending
        );
    }

    #[test]
    fn a_failing_diff_leaves_the_entry_pending() {
        let git = FakeGit::from(&[
            ("merge-base --is-ancestor c1 HEAD", Ok("")),
            (DIFF_C1_HEAD, Err("fatal: bad revision")),
        ]);
        assert_eq!(
            classify(&entry(Some(5)), "HEAD", &git),
            HistoryStatus::Pending
        );
    }

    #[test]
    fn existing_pending_entries_are_reclassified_before_new_ones_are_recorded() {
        // [CP2 MS3 loop 1] A new entry reviewed at HEAD would classify WeakReject (HEAD..HEAD is empty) if it were
        // reclassified after recording; it must come back untouched, as Pending.
        let mut fresh = entry(Some(5));
        fresh.id = "new".into();
        fresh.reviewed_commit = "HEAD".into();
        let git = FakeGit::from(&[("merge-base --is-ancestor c1 HEAD", Ok("")),
                              (DIFF_C1_HEAD, Ok("@@ -5 +5 @@\n")),
                              ("merge-base --is-ancestor HEAD HEAD", Ok("")),
                              ("diff --no-color --no-ext-diff --no-textconv --no-renames --unified=0 --relative HEAD HEAD -- a.rs", Ok("")),
                              // [CP2 MS3 loop 13] Scripted, so reclassifying the new entry would make it WeakReject
                              // (HEAD resolves to a commit other than the literal "HEAD" it was reviewed at): the
                              // mutation goes red whatever FakeGit answers for an unscripted argv.
                              (HEAD_IS_C2, Ok("c9\n"))]);
        let (updates, to_record) =
            reclassify_then_record(&[entry(Some(5))], &[fresh.clone()], "HEAD", &git);
        assert_eq!(updates, vec![("i".to_string(), HistoryStatus::Attended)]);
        assert_eq!(to_record, vec![fresh]);
    }

    #[test]
    fn a_weak_reject_whose_line_changes_later_becomes_attended() {
        // [CP2 MS3 loop 2] RV08: a weak reject is provisional; a later change to the line attends it.
        let mut weak = entry(Some(5));
        weak.status = HistoryStatus::WeakReject;
        let git = FakeGit::from(&[
            ("merge-base --is-ancestor c1 HEAD", Ok("")),
            (DIFF_C1_HEAD, Ok("@@ -5 +5 @@\n")),
        ]);
        let (updates, _) = reclassify_then_record(&[weak], &[], "HEAD", &git);
        assert_eq!(updates, vec![("i".to_string(), HistoryStatus::Attended)]);
    }

    #[test]
    fn an_attended_entry_whose_change_was_reverted_becomes_a_weak_reject() {
        // [CP2 MS3 loop 14 (pasada final)] RV08 by net state: the reverted line is the reviewed code again.
        let mut done = entry(Some(5));
        done.status = HistoryStatus::Attended;
        let git = FakeGit::from(&[
            ("merge-base --is-ancestor c1 HEAD", Ok("")),
            (DIFF_C1_HEAD, Ok("")),
            (
                HEAD_IS_C2,
                Ok("c2
"),
            ),
        ]);
        let (updates, _) = reclassify_then_record(&[done], &[], "HEAD", &git);
        assert_eq!(updates, vec![("i".to_string(), HistoryStatus::WeakReject)]);
    }

    #[test]
    fn no_signal_never_downgrades_a_validated_entry() {
        // [CP2 MS3 loop 14 (pasada final)] A failing git classifies Pending; an Attended entry keeps its status.
        let mut done = entry(Some(5));
        done.status = HistoryStatus::Attended;
        let git = FakeGit::from(&[(
            "merge-base --is-ancestor c1 HEAD",
            Err("fatal: not a valid commit"),
        )]);
        let (updates, _) = reclassify_then_record(&[done], &[], "HEAD", &git);
        assert!(updates.is_empty(), "{updates:?}");
    }

    #[test]
    fn reclassification_is_capped_per_review_oldest_first() {
        // [CP2 MS3 loop 3] Without pruning the pending set grows; each review pays at most the cap.
        // [CP2 MS3 loop 4] FakeGit answers by argv, never consumed: all 201 share c1 and a.rs, so two keys answer every call.
        let olds: Vec<HistoryEntry> = (0..MAX_RECLASSIFY_PER_REVIEW + 1)
            .map(|i| HistoryEntry {
                id: format!("i{i:04}"),
                ..entry(Some(5))
            })
            .collect();
        let git = FakeGit::from(&[
            ("merge-base --is-ancestor c1 HEAD", Ok("")),
            (DIFF_C1_HEAD, Ok("@@ -5 +5 @@\n")),
        ]);
        let (updates, _) = reclassify_then_record(&olds, &[], "HEAD", &git);
        assert_eq!(updates.len(), MAX_RECLASSIFY_PER_REVIEW);
        assert_eq!(updates[0].0, "i0000", "the oldest goes first");
        assert!(updates
            .iter()
            .all(|(id, _)| *id != format!("i{MAX_RECLASSIFY_PER_REVIEW:04}")));
    }

    fn found(
        file: &str,
        line: u32,
        title: &str,
    ) -> crate::review::consolidate::ConsolidatedFinding {
        use magi_core::schema::Severity;
        crate::review::consolidate::ConsolidatedFinding {
            file: Some(file.into()),
            line: Some(line),
            severity: Severity::Warning,
            reported_severity: Severity::Warning,
            titles: vec![title.into()],
            details: vec!["d".into()],
            categories: vec![],
            sources: vec![],
            marks: vec![],
            batches: vec![0],
        }
    }

    fn judged(state: crate::review::gate::GateState) -> crate::review::gate::GateVerdict {
        crate::review::gate::GateVerdict {
            state,
            cause: None,
            causes: vec![],
            model: None,
            lean: None,
            confidence: None,
            skipped: vec![],
            status: None,
        }
    }

    #[test]
    fn a_review_records_its_main_list_anchored_to_the_reviewed_commit() {
        // [CP2 MS3 loop 2] RV08/RV19: what enters the history, and from which commit it is later validated.
        use crate::review::gate::GateState;
        let main = vec![
            (found("a.rs", 3, "kept"), judged(GateState::Kept)),
            (found("a.rs", 9, "low"), judged(GateState::LowConfidence)),
        ];
        let entries = entries_from_review(&main, "c1");
        assert_eq!(entries.len(), 2);
        assert!(entries
            .iter()
            .all(|e| e.reviewed_commit == "c1" && e.status == HistoryStatus::Pending));
        assert_eq!(
            entries[0].id,
            magi_core::finding_id::generate_finding_id(
                "a.rs",
                3,
                magi_core::schema::Category::Other
            )
        );
        assert_eq!(
            (entries[0].file.as_str(), entries[0].line),
            ("a.rs", Some(3))
        );
    }

    #[test]
    fn a_filtered_finding_is_never_recorded() {
        let main = vec![(
            found("a.rs", 3, "f"),
            judged(crate::review::gate::GateState::Filtered),
        )];
        assert!(entries_from_review(&main, "c1").is_empty());
    }

    #[test]
    fn an_unjudged_finding_is_never_recorded() {
        // [CP2 MS3 loop 4] A gate timeout is not a judgment; history is served back as validated context.
        let main = vec![(
            found("a.rs", 3, "u"),
            judged(crate::review::gate::GateState::Unfiltered),
        )];
        assert!(entries_from_review(&main, "c1").is_empty());
    }

    #[test]
    fn a_finding_without_a_location_is_not_recorded() {
        // [CP2 MS3 loop 3] classify needs file and line; such an entry would stay Pending forever.
        use crate::review::gate::GateState;
        let (mut no_line, mut no_file) = (found("a.rs", 3, "x"), found("a.rs", 3, "y"));
        no_line.line = None;
        no_file.file = None;
        let main = vec![
            (no_line, judged(GateState::Kept)),
            (no_file, judged(GateState::Kept)),
        ];
        assert!(entries_from_review(&main, "c1").is_empty());
    }

    #[test]
    fn merged_without_changes_is_a_weak_reject() {
        let git = FakeGit::from(&[
            ("merge-base --is-ancestor c1 HEAD", Ok("")),
            (DIFF_C1_HEAD, Ok("")),
            (HEAD_IS_C2, Ok("c2\n")),
        ]);
        assert_eq!(
            classify(&entry(Some(5)), "HEAD", &git),
            HistoryStatus::WeakReject
        );
    }

    #[test]
    fn a_finding_reviewed_at_head_stays_pending_until_a_later_commit() {
        // [CP2 MS3 loop 12] RV08 «en un commit posterior»: HEAD still at the reviewed commit is no signal yet.
        let at_head = FakeGit::from(&[
            ("merge-base --is-ancestor c1 HEAD", Ok("")),
            (DIFF_C1_HEAD, Ok("")),
            (HEAD_IS_C2, Ok("c1\n")),
        ]);
        assert_eq!(
            classify(&entry(Some(5)), "HEAD", &at_head),
            HistoryStatus::Pending
        );
    }

    #[test]
    fn a_vanished_commit_leaves_the_entry_pending() {
        let git = FakeGit::from(&[(
            "merge-base --is-ancestor c1 HEAD",
            Err("fatal: not a valid commit"),
        )]);
        assert_eq!(
            classify(&entry(Some(5)), "HEAD", &git),
            HistoryStatus::Pending
        );
    }

    #[test]
    fn an_entry_without_a_line_stays_pending() {
        let git = FakeGit::from(&[]);
        assert_eq!(classify(&entry(None), "HEAD", &git), HistoryStatus::Pending);
    }
}
