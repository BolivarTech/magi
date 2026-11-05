// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-11-05

//! Finding history for the `code-review` context (RV07(4), RV08).
//!
//! A finding the trio reported and the gate judged is remembered with the commit it was first
//! reviewed at. Later, git alone says what became of it: the reported line changed in a later
//! commit (attended), or the code reached `HEAD` untouched (weak reject). **No model ever
//! validates history** (RV08); there is no human mark either.
//!
//! The reading is by net state, between the reviewed commit and `HEAD`: a line changed and then
//! reverted is the reviewed code again and is a weak reject, not attended.
//!
//! # Examples
//!
//! ```ignore
//! let status = classify(&entry, "HEAD", &git);
//! let (updates, to_record) = reclassify_then_record(&existing, &fresh, "HEAD", &git);
//! ```

use magi_core::finding_id::generate_finding_id;
use magi_core::schema::Category;

use super::consolidate::ConsolidatedFinding;
use super::gate::{GateState, GateVerdict};
use super::source::GitCommand;

/// The category every located finding is keyed with: the neutral one consolidation uses
/// (`consolidate::NEUTRAL_CATEGORY`), so the id here is the id magi-core computed there.
const NEUTRAL_CATEGORY: Category = Category::Other;

/// Separator that joins the merged titles and details of a finding.
const MERGED_TEXT_SEPARATOR: &str = "\n";

/// The only hunk-header prefix read from a diff.
const HUNK_PREFIX: &str = "@@";

/// What became of a remembered finding, judged from git alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoryStatus {
    /// No signal yet: no line, git failed, the commit is gone, or no later commit exists.
    Pending,
    /// The reported line changed in a later commit.
    Attended,
    /// A later commit exists and the reported line is untouched.
    WeakReject,
}

impl HistoryStatus {
    /// The `snake_case` spelling stored in the `status` column (the `serde` one).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Attended => "attended",
            Self::WeakReject => "weak_reject",
        }
    }

    /// The status spelled `text`, or `None` for any other text.
    ///
    /// # Arguments
    /// * `text` - a value of the `status` column.
    pub fn parse(text: &str) -> Option<Self> {
        [Self::Pending, Self::Attended, Self::WeakReject]
            .into_iter()
            .find(|s| s.as_str() == text)
    }
}

/// One remembered finding.
#[derive(Debug, Clone, PartialEq)]
pub struct HistoryEntry {
    /// The stable finding id (magi-core's `generate_finding_id`).
    pub id: String,
    /// The reported file, relative to the workspace.
    pub file: String,
    /// The reported 1-based line; `None` only in a row edited by hand.
    pub line: Option<u32>,
    /// The commit the finding was first reviewed at.
    pub reviewed_commit: String,
    /// The merged titles, one per line.
    pub title: String,
    /// The merged details, one per line.
    pub detail: String,
    /// The current status.
    pub status: HistoryStatus,
}

/// The most existing entries [`reclassify_then_record`] looks at per review.
///
/// Each costs at most three bounded git calls (`merge-base`, `diff` and, only before a weak
/// reject, `rev-parse`), so one review pays at most `3 * MAX_RECLASSIFY_PER_REVIEW` of them.
pub const MAX_RECLASSIFY_PER_REVIEW: usize = 200;

/// Lines `a..a+b-1` of the old side that a hunk header `-a[,b]` covers; `b = 0` covers none.
fn parse_range(token: &str, sign: char) -> Option<(u64, u64)> {
    let body = token.strip_prefix(sign)?;
    let (start, count) = match body.split_once(',') {
        Some((a, b)) => (a, b),
        None => (body, "1"),
    };
    Some((start.parse().ok()?, count.parse().ok()?))
}

/// The old-side `(start, count)` of a `@@ -a[,b] +c[,d] @@ ...` header, or `None` if the line
/// does not have that shape. What follows the second `@@` (git's function context) is ignored.
fn parse_hunk_header(line: &str) -> Option<(u64, u64)> {
    let mut parts = line
        .strip_prefix(HUNK_PREFIX)?
        .strip_prefix(' ')?
        .split(' ');
    let old = parse_range(parts.next()?, '-')?;
    parse_range(parts.next()?, '+')?;
    (parts.next()? == HUNK_PREFIX).then_some(old)
}

/// Whether the diff `text` changes `line` of the old side: `Some(true)` if a hunk covers it,
/// `Some(false)` if none does, `None` if a hunk header cannot be read (never decide on a format
/// that is not understood).
///
/// Complexity: O(lines of the diff).
fn diff_covers(text: &str, line: u32) -> Option<bool> {
    let line = u64::from(line);
    let mut covered = false;
    for header in text.lines().filter(|l| l.starts_with(HUNK_PREFIX)) {
        let (start, count) = parse_hunk_header(header)?;
        covered |= count > 0 && line >= start && line < start.saturating_add(count);
    }
    Some(covered)
}

/// Judges one remembered finding from git alone, never from a model (RV08).
///
/// Anchored to the reviewed commit: `git diff --unified=0 --relative <commit> <head> -- <file>`
/// gives hunks in the coordinates of `<commit>`, the ones the finding's line uses, so lines
/// inserted above never move it. A hunk covering the line means [`HistoryStatus::Attended`]; a
/// deleted or renamed file is covered whole. Otherwise the line is untouched, and it is a
/// [`HistoryStatus::WeakReject`] only once `head` is a commit after the reviewed one.
///
/// Anything that is not a signal is [`HistoryStatus::Pending`]: no line, a reference that
/// starts with `-`, a commit that is not an ancestor of `head` (another branch, or gone), a git
/// failure, a hunk header that cannot be read, or `head` still at the reviewed commit.
///
/// Complexity: three git calls at most, O(size of the diff of one file).
///
/// # Arguments
/// * `entry` - the remembered finding.
/// * `head` - the revision to compare against.
/// * `git` - the git runner, rooted at the workspace so `--relative` and the paths agree.
///
/// # Returns
/// The status the evidence supports.
pub fn classify(entry: &HistoryEntry, head: &str, git: &dyn GitCommand) -> HistoryStatus {
    let commit = entry.reviewed_commit.as_str();
    let Some(line) = entry.line else {
        return HistoryStatus::Pending;
    };
    if commit.starts_with('-') || head.starts_with('-') {
        return HistoryStatus::Pending;
    }
    if git
        .run(&["merge-base", "--is-ancestor", commit, head])
        .is_err()
    {
        return HistoryStatus::Pending;
    }
    let diff = git.run(&[
        "diff",
        "--no-color",
        "--no-ext-diff",
        "--no-textconv",
        "--no-renames",
        "--unified=0",
        "--relative",
        commit,
        head,
        "--",
        entry.file.as_str(),
    ]);
    match diff.ok().and_then(|text| diff_covers(&text, line)) {
        None => HistoryStatus::Pending,
        Some(true) => HistoryStatus::Attended,
        Some(false) => weak_reject_if_later(commit, head, git),
    }
}

/// [`HistoryStatus::WeakReject`] when `head` resolves to a commit other than `commit`;
/// otherwise [`HistoryStatus::Pending`]: with no commit after the reviewed one there is no
/// signal yet, and a failing `rev-parse` is no signal either.
fn weak_reject_if_later(commit: &str, head: &str, git: &dyn GitCommand) -> HistoryStatus {
    match git.run(&["rev-parse", "--verify", &format!("{head}^{{commit}}")]) {
        Ok(resolved) if resolved.trim() != commit => HistoryStatus::WeakReject,
        _ => HistoryStatus::Pending,
    }
}

/// Reclassifies the existing entries, then hands back the new ones untouched.
///
/// The order matters: an entry reviewed at `head` would classify as a weak reject if it were
/// reclassified after being recorded (`head..head` is empty), so the new entries are only
/// returned, as `Pending`, for the caller to record.
///
/// Existing entries of every status are reclassified (a weak reject is provisional, and an
/// attended entry whose change was reverted goes back to a weak reject, RV08 by net state), but
/// only the first [`MAX_RECLASSIFY_PER_REVIEW`], in the order received: the caller passes the
/// least recently checked first (`ReviewHistoryStore::due`). The updates are only the statuses
/// that change, and a `Pending` result never replaces a validated status.
///
/// Complexity: O(min(existing, [`MAX_RECLASSIFY_PER_REVIEW`])) classifications.
///
/// # Arguments
/// * `existing` - entries already stored, least recently checked first.
/// * `new` - entries of this review, not stored yet.
/// * `head` - the revision to compare against.
/// * `git` - the git runner.
///
/// # Returns
/// `(id, new status)` for each existing entry whose status changed, and the new entries.
pub fn reclassify_then_record(
    existing: &[HistoryEntry],
    new: &[HistoryEntry],
    head: &str,
    git: &dyn GitCommand,
) -> (Vec<(String, HistoryStatus)>, Vec<HistoryEntry>) {
    let updates = existing
        .iter()
        .take(MAX_RECLASSIFY_PER_REVIEW)
        .filter_map(|e| {
            let status = classify(e, head, git);
            (status != HistoryStatus::Pending && status != e.status).then(|| (e.id.clone(), status))
        })
        .collect();
    (updates, new.to_vec())
}

/// What a review records: its judged main list, anchored to the reviewed commit.
///
/// Only `kept` and `low_confidence` findings enter. A `filtered` one is a false positive by the
/// gate's own judgment, and an `unfiltered` one was never judged (timeout, exhausted chain,
/// unauthorized), so recording either would serve it back later as validated context. A finding
/// without a file or a line is not recorded either: [`classify`] needs both and could never take
/// it out of `Pending`. The `id` is magi-core's `generate_finding_id` over file, line and the
/// neutral category consolidation uses, so it is stable across reviews while file and line do
/// not change; a failure that moves to another line is another id. It depends on magi-core's
/// algorithm (pinned `=4.2.0`): a bump that changes it leaves old rows with ids no new review
/// repeats, and must re-verify this.
///
/// Complexity: O(findings).
///
/// # Arguments
/// * `findings` - the judged main list with each finding's gate verdict.
/// * `reviewed_commit` - the pinned SHA of the reviewed revision.
///
/// # Returns
/// One `Pending` entry per recordable finding.
pub fn entries_from_review(
    findings: &[(ConsolidatedFinding, GateVerdict)],
    reviewed_commit: &str,
) -> Vec<HistoryEntry> {
    findings
        .iter()
        .filter(|(_, v)| matches!(v.state, GateState::Kept | GateState::LowConfidence))
        .filter_map(|(f, _)| {
            let (file, line) = (f.file.as_ref()?, f.line?);
            Some(HistoryEntry {
                id: generate_finding_id(file, line, NEUTRAL_CATEGORY),
                file: file.clone(),
                line: Some(line),
                reviewed_commit: reviewed_commit.to_string(),
                title: f.titles.join(MERGED_TEXT_SEPARATOR),
                detail: f.details.join(MERGED_TEXT_SEPARATOR),
                status: HistoryStatus::Pending,
            })
        })
        .collect()
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

    #[test]
    fn an_unreadable_hunk_header_never_becomes_a_weak_reject() {
        // Guardian: with a later commit scripted, reading `@@ garbage @@` as an empty hunk would
        // reach the weak-reject branch; an unreadable diff must stay Pending.
        let git = FakeGit::from(&[
            ("merge-base --is-ancestor c1 HEAD", Ok("")),
            (DIFF_C1_HEAD, Ok("@@ garbage @@\n")),
            (HEAD_IS_C2, Ok("c2\n")),
        ]);
        assert_eq!(
            classify(&entry(Some(5)), "HEAD", &git),
            HistoryStatus::Pending
        );
    }
}
