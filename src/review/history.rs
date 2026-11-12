// Author: Julian Bolivar
// Version: 0.25.0
// Date: 2026-11-08

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

use super::consolidate::{ConsolidatedFinding, NEUTRAL_CATEGORY};
use super::gate::{GateState, GateVerdict};
use super::secrets::{find_secret, holds_key_block, mask_secrets};
use super::source::GitCommand;

/// Separator that joins the merged titles and details of a finding.
const MERGED_TEXT_SEPARATOR: &str = "\n";

/// The only hunk-header prefix read from a diff.
const HUNK_PREFIX: &str = "@@";

/// Start of the line git writes, instead of hunks, for a file whose change is binary.
const BINARY_PREFIX: &str = "Binary files ";

/// End of that line.
const BINARY_SUFFIX: &str = " differ";

/// What opens the header of one file in git's diff.
const DIFF_HEADER_PREFIX: &str = "diff --git ";

/// Mode prefix of a regular file in `git ls-tree` (`100644`, `100755`); a symlink is `120000`.
const REGULAR_FILE_MODE_PREFIX: &str = "100";

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
    /// Whether the status is validated (RV08): attended or weak reject, never pending.
    pub fn is_validated(self) -> bool {
        matches!(self, Self::Attended | Self::WeakReject)
    }

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
/// Each costs at most five bounded local git calls ([`classify`]), so one review pays at most
/// `5 * MAX_RECLASSIFY_PER_REVIEW` of them.
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

/// Whether the diff `text` changes `line` of the old side: `Some(true)` if a hunk covers it or
/// the change is binary (git writes no hunk then, and the line is gone with the text), `Some(false)`
/// if none does, `None` if a hunk header cannot be read (never decide on a format that is not
/// understood).
///
/// It reads the subset of git's unified diff that `git diff --unified=0` prints for one file
/// (git-diff(1), "Combined diff format" excluded: two revisions are compared, never a merge's
/// parents): only the hunk headers, `@@ -a[,b] +c[,d] @@[ function context]`, whose old range
/// `a,b` (`b` absent means 1, `b` = 0 an insertion after line `a`) is what covers the line, and
/// git's `Binary files … differ` line. Body lines are not read.
///
/// Complexity: O(lines of the diff).
// MAGI FIX (MS3 §6 loop3): a binary change has no hunk; it read as untouched.
fn diff_covers(text: &str, line: u32) -> Option<bool> {
    if text
        .lines()
        .any(|l| l.starts_with(BINARY_PREFIX) && l.ends_with(BINARY_SUFFIX))
    {
        return Some(true);
    }
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
/// deleted or renamed file, or one whose change git shows as binary, is covered whole. Otherwise the line is untouched, and it is a
/// [`HistoryStatus::WeakReject`] only once `head` is a commit after the reviewed one.
///
/// Anything that is not a signal is [`HistoryStatus::Pending`]: no line, no file (an empty
/// pathspec would be the whole tree), a reviewed commit that is no commit id (`is_commit_id`:
/// a symbolic name moves, so it anchors nothing), a `head` that starts with `-`, a commit that is
/// not an ancestor of `head` (another branch, or gone), a git
/// failure, a hunk header that cannot be read, or `head` still at the reviewed commit.
///
/// Complexity: four git calls at most (`merge-base`, `diff`, `rev-parse` of `head`, and `ls-tree`
/// on an empty diff's weak reject), five for an abbreviated stored commit; O(diff of one file).
///
/// # Arguments
/// * `entry` - the remembered finding.
/// * `head` - the revision to compare against.
/// * `git` - the git runner, rooted at the workspace so `--relative` and the paths agree. It
///   must run git with `GIT_LITERAL_PATHSPECS=1`, as the production runner (`ScrubbedGit`,
///   `src/system/review_git.rs`, tested there) does: `entry.file` is passed as a pathspec
///   after `--`, and without literal pathspecs a stored name such as `:(glob)*.rs` or
///   `src[0].rs` would match other files and read their hunks as this line's.
///
/// # Returns
/// The status the evidence supports.
pub fn classify(entry: &HistoryEntry, head: &str, git: &dyn GitCommand) -> HistoryStatus {
    match signal(entry, head, git) {
        Signal::Status(status) => status,
        Signal::NoLaterCommit | Signal::OffLine | Signal::Unknown => HistoryStatus::Pending,
    }
}

/// Whether a stored `entry` may be served as validated context against `head`: only when its
/// status is validated ([`HistoryStatus::is_validated`], `pending_is_never_served`) and its
/// reviewed commit is an ancestor of `head` ([`on_heads_line`]), so a finding validated on another
/// branch never reaches a review of this one (RV08 reads git's signal on `head`'s line). The context
/// read applies it to every history entry it serves (`StoreRag::history_items`,
/// `memory::review_rag`), and the reclassification turns the entry `Pending` once git says so
/// for certain ([`reclassify_then_record`]). Fails closed: a commit that is no commit id, a
/// `head` that starts with `-`, or a git that does not answer, serves nothing
/// (`a_validated_entry_off_heads_line_goes_back_to_pending_and_is_never_served`). One local git
/// call.
///
/// # Arguments
/// * `entry` - the stored entry.
/// * `head` - the revision the review runs against.
/// * `git` - the git runner.
///
/// # Returns
/// Whether it may be served.
// MAGI FIX (MS3 §6 loop7): a validated entry was served whatever branch HEAD was on.
// MAGI FIX (MS3 §6 loop11): the status was not checked: a `Pending` entry was served.
pub fn served_as_validated(entry: &HistoryEntry, head: &str, git: &dyn GitCommand) -> bool {
    entry.status.is_validated() && on_heads_line(&entry.reviewed_commit, head, git)
}

/// The ancestry half of [`served_as_validated`]: whether `commit` is a commit id that is an
/// ancestor of `head`. Depends on `commit` and `head` alone, so a caller may memoize it per
/// commit (`StoreRag`). Fails closed as [`served_as_validated`]. One local git call.
///
/// # Arguments
/// * `commit` - the reviewed commit.
/// * `head` - the revision the review runs against.
/// * `git` - the git runner.
///
/// # Returns
/// Whether `commit` is on `head`'s line.
pub fn on_heads_line(commit: &str, head: &str, git: &dyn GitCommand) -> bool {
    is_commit_id(commit)
        && !head.starts_with('-')
        && git
            .run(&["merge-base", "--is-ancestor", commit, head])
            .is_ok()
}

/// Whether `rev` resolves to a commit: `rev-parse --verify <rev>^{commit}` answers. One call.
fn resolves(rev: &str, git: &dyn GitCommand) -> bool {
    git.run(&["rev-parse", "--verify", &format!("{rev}^{{commit}}")])
        .is_ok()
}

/// What git says of a remembered finding: a status, a definitive lack of signal, or no answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Signal {
    /// The status the evidence supports (attended or weak reject).
    Status(HistoryStatus),
    /// `head` resolves to the reviewed commit itself: nothing came after it, so there is no
    /// signal, and RV08 says the entry is pending. A definitive answer of git.
    NoLaterCommit,
    /// The reviewed commit is not an ancestor of `head`, and both resolve: `head` is on another
    /// line (another branch), so there is no signal there, a definitive answer of git, and a
    /// validated entry goes back to `Pending` like [`Self::NoLaterCommit`].
    ///
    /// Residual, declared (MAGI FIX, MS3 §6 loop13, R3 Melchior): `merge-base --is-ancestor`
    /// reports "not an ancestor" and a failure to run with the same non-zero exit, so a git that
    /// fails for another reason while both commits still resolve reads as `OffLine`. The effect
    /// is the safe direction (a validated entry goes back to `Pending` and is not served until
    /// the next review re-checks it); no entry is ever promoted by it.
    OffLine,
    /// No answer: an entry that cannot be checked, a git that failed, a diff not understood.
    Unknown,
}

/// [`classify`], telling a definitive lack of signal ([`Signal::NoLaterCommit`]) from no answer
/// ([`Signal::Unknown`]). Same cases and cost as [`classify`].
fn signal(entry: &HistoryEntry, head: &str, git: &dyn GitCommand) -> Signal {
    let commit = entry.reviewed_commit.as_str();
    let Some(line) = entry.line else {
        return Signal::Unknown;
    };
    // MAGI FIX (MS3 §6 loop3): an empty file would be an empty pathspec, the whole tree.
    // MAGI FIX (MS3 §6 loop5): a reviewed commit that is no commit id (a branch name, `HEAD~1`)
    // gives no signal on any branch, not only the weak reject's: a moving ref anchors nothing.
    if !is_commit_id(commit) || head.starts_with('-') || entry.file.is_empty() {
        return Signal::Unknown;
    }
    if git
        .run(&["merge-base", "--is-ancestor", commit, head])
        .is_err()
    {
        // MAGI FIX (MS3 §6 loop7): a definitive "not an ancestor" (both commits resolve) is told
        // apart from a git that failed; it was kept as validated, served on another branch.
        return if resolves(commit, git) && resolves(head, git) {
            Signal::OffLine
        } else {
            Signal::Unknown
        };
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
    // MAGI FIX (MS3 §6 loop6): a directory as `file` is a tree-wide pathspec; a diff that names
    // any other file is no signal about this entry.
    let Some(text) = diff.ok().filter(|text| names_only(text, &entry.file)) else {
        return Signal::Unknown;
    };
    match diff_covers(&text, line) {
        None => Signal::Unknown,
        Some(true) => Signal::Status(HistoryStatus::Attended),
        // MAGI FIX (MS3 §6 loop7): an empty diff names no file, so `names_only` passes for a
        // directory or a path that never existed; a weak reject then needs a regular file there.
        Some(false) => match weak_reject_if_later(commit, head, git) {
            Signal::Status(HistoryStatus::WeakReject)
                if text.trim().is_empty() && !is_regular_file_at(commit, &entry.file, git) =>
            {
                Signal::Unknown
            }
            other => other,
        },
    }
}

/// Whether `file` is a regular file (a blob of mode `100…`, never a tree, a submodule or a
/// symlink) at `commit`: `git ls-tree -z <commit> -- <file>` lists exactly that one entry. The
/// path is read relative to the runner's root, as the diff's `--relative` is. A git failure is
/// `false`, the safe direction (no weak reject). One local git call, O(bytes of the listing).
fn is_regular_file_at(commit: &str, file: &str, git: &dyn GitCommand) -> bool {
    let Ok(listing) = git.run(&["ls-tree", "-z", commit, "--", file]) else {
        return false;
    };
    let entries: Vec<&str> = listing.split('\0').filter(|e| !e.is_empty()).collect();
    let [only] = entries.as_slice() else {
        return false;
    };
    let Some((meta, path)) = only.split_once('\t') else {
        return false;
    };
    let fields: Vec<&str> = meta.split_whitespace().collect();
    path == file
        && matches!(fields.as_slice(), [mode, "blob", _] if mode.starts_with(REGULAR_FILE_MODE_PREFIX))
}

/// Whether every file header of the diff `text` (`diff --git a/<path> b/<path>`, `--no-renames`
/// so both paths are one) names `file` itself.
///
/// A directory as `file` (`src`, `.`) is a pathspec over a whole subtree even with literal
/// pathspecs, so its diff can hold another file's hunk at the entry's line; such a diff gives no
/// signal (`a_diff_of_another_file_than_the_entrys_gives_no_signal`). A diff without a header
/// (nothing changed) names nothing else. The production runner writes paths unquoted
/// (`core.quotePath=false`); a path git still quotes (a tab, a newline or a quote in its name)
/// does not match here either, so it gives no signal: `Pending`, the safe direction. O(lines).
fn names_only(text: &str, file: &str) -> bool {
    let plain = format!("{DIFF_HEADER_PREFIX}a/{file} b/{file}");
    text.lines()
        .filter(|l| l.starts_with(DIFF_HEADER_PREFIX))
        .all(|l| l == plain)
}

/// Whether `commit` reads as a commit id: non-empty hexadecimal, a full SHA as `record_history`
/// records it or an abbreviation of one. A branch name that is not all hex, `HEAD~1`, an empty
/// string or an option (`-x`) is not. O(len).
///
/// Declared limit: the check is lexical, so a branch named in hex (`deadbeef`) or a one-digit
/// "abbreviation" passes it. Production never stores either: `record_history` resolves the
/// reviewed revision with `rev-parse` to its full 40-hex SHA before `entries_from_review` runs,
/// so only a hand-edited row can hold one, and the worst it gets is a wrong history status
/// (`Pending` or a weak reject, never `Attended` without a hunk on its line); no byte leaves the
/// process. No minimum length is enforced, so the tests' short ids (`c1`) keep standing for
/// SHAs.
fn is_commit_id(commit: &str) -> bool {
    !commit.is_empty() && commit.bytes().all(|b| b.is_ascii_hexdigit())
}

/// [`HistoryStatus::WeakReject`] when `head` resolves to a commit other than `commit`;
/// [`Signal::NoLaterCommit`] when it resolves to `commit` itself (no commit after the reviewed
/// one: no signal, RV08), and [`Signal::Unknown`] when `rev-parse` fails.
///
/// `commit` is the SHA recorded at review time (`record_history` resolves it with `rev-parse`).
/// Both sides are compared as full names, exactly: `head` is resolved by `rev-parse --verify`,
/// which prints the full name whatever `core.abbrev` says, and a stored `commit` of another
/// length (an abbreviation in a hand-edited row) is resolved the same way first, one more call
/// (`stored_sha_is_compared_in_full`). A `commit` that is not hexadecimal gives no signal.
// MAGI FIX (MS3 §6 loop3): an abbreviated SHA of HEAD itself read as a later commit.
// MAGI FIX (MS3 §6 loop4): a non-SHA commit read as a later one, a false weak reject.
fn weak_reject_if_later(commit: &str, head: &str, git: &dyn GitCommand) -> Signal {
    if !is_commit_id(commit) {
        return Signal::Unknown;
    }
    let Ok(resolved) = git.run(&["rev-parse", "--verify", &format!("{head}^{{commit}}")]) else {
        return Signal::Unknown;
    };
    let resolved = resolved.trim();
    // MAGI FIX (MS3 §6 loop8): a stored abbreviation was compared as a prefix of HEAD's SHA;
    // it is resolved too, so both sides are full names and the comparison is exact.
    let stored = if commit.len() == resolved.len() {
        commit.to_string()
    } else {
        match git.run(&["rev-parse", "--verify", &format!("{commit}^{{commit}}")]) {
            Ok(full) => full.trim().to_string(),
            Err(_) => return Signal::Unknown,
        }
    };
    if stored == resolved {
        Signal::NoLaterCommit
    } else {
        Signal::Status(HistoryStatus::WeakReject)
    }
}

/// Reclassifies the existing entries, then hands back the new ones untouched.
///
/// The order matters: a new entry is never classified against its own commit. Anchored to the
/// full SHA `head` resolves to, it would only read as `Pending` (no commit after the reviewed
/// one yet, as `classify` reads it), and anchored to any other spelling of `head` as a false
/// weak reject; so the new entries are only returned, untouched and `Pending`, for the caller to
/// record after the existing ones were reclassified.
///
/// Existing entries of every status are reclassified (a weak reject is provisional, and an
/// attended entry whose change was reverted goes back to a weak reject, RV08 by net state), but
/// only the first [`MAX_RECLASSIFY_PER_REVIEW`], in the order received: the caller passes the
/// least recently checked first (`ReviewHistoryStore::due`). The updates are only the statuses
/// that change. A validated status goes back to `Pending` only on git's definitive answer that
/// `head` is the reviewed commit again (RV08: no later commit, no signal;
/// `a_validated_entry_goes_back_to_pending_once_head_is_the_reviewed_commit_again`) or that the
/// reviewed commit is on another line than `head`, both resolving
/// (`a_validated_entry_off_heads_line_goes_back_to_pending_and_is_never_served`); an entry git
/// could not answer for (a failure, a gone commit) keeps its status, so a broken or partial
/// clone never erases what was validated (`no_signal_never_downgrades_a_validated_entry`); the
/// context read still serves it only on `head`'s line ([`served_as_validated`]).
///
/// Complexity: O(min(existing, [`MAX_RECLASSIFY_PER_REVIEW`])) classifications.
///
/// Declared limits (MS3 §6 loop12). (1) An entry past the cap keeps the status it had until a
/// later review reaches it (the backlog is reported); if `HEAD` is reset to its reviewed commit
/// meanwhile, an `attended` entry can be served once more as validated, on a line that is
/// ancestor-correct (`on_heads_line`) though no commit follows it. The cost is one stale label on
/// a reference item the trio reads as data, bounded by the cap's queue; a cheap guard would add a
/// `rev-parse` per memoized commit to every context read. (2) Each entry costs up to five local
/// git calls and entries that share a `(reviewed commit, head)` pair repeat the ancestry call; the
/// bound is the cap (200) times those calls inside the 60 s sub-budget, nothing leaves the
/// process, and a per-pair memo would add state to a pure function for that. (3) The diff and
/// hunk-header parse reads git's unified output as git 2.x writes it (this module's tests feed
/// it git's literal output, `lines_changed_after_the_review_mean_attended` and its neighbours);
/// a line it does not understand gives `Unknown`, never `attended`.
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
        // MAGI FIX (MS3 §6 loop5): a definitive lack of signal is recorded as pending (RV08); it
        // was dropped like a git failure, so a reset to the reviewed commit kept it validated.
        .filter_map(|e| match signal(e, head, git) {
            Signal::Status(status) if status != e.status => Some((e.id.clone(), status)),
            Signal::NoLaterCommit | Signal::OffLine if e.status != HistoryStatus::Pending => {
                Some((e.id.clone(), HistoryStatus::Pending))
            }
            Signal::Status(_) | Signal::NoLaterCommit | Signal::OffLine | Signal::Unknown => None,
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
/// it out of `Pending`. The caller passes only findings whose line is a line of the reviewed
/// revision (`ReviewSummary::anchored_findings`): a finding whose number can only be a removed
/// line is not recorded, because that number belongs to the old file and [`classify`] would check
/// it against an unrelated new-side line. A number that is both a removed and a live line (an
/// in-place edit, or a removed line whose old number also falls in a new-side range of the file)
/// is ambiguous: it is read on the new side, the side the gate judged, and recorded there.
/// Text is stored with its one-line secrets masked, and a finding whose title or detail holds a
/// private-key block is not recorded (RV22: masking would keep the block's body; a block is
/// recognised by its header alone, so one split between title and detail is caught on the side
/// that holds the header, `a_key_block_whose_header_ends_the_title_and_body_opens_the_detail_is_not_stored`); the context
/// read masks again when it serves an entry back
/// (`a_history_entry_quoting_a_secret_reaches_the_context_masked`, `memory::review_rag`).
/// The `id` is magi-core's `generate_finding_id` over file, line and the neutral category
/// consolidation uses, so it is stable across reviews while file and line do not change; a
/// failure that moves to another line is another id. Declared limitation: two different
/// findings on one unchanged `file:line` in two reviews share the id, and the first sighting
/// wins (`ReviewHistoryStore::record`) unless the stored one is `attended`, whose code changed. It depends on magi-core's algorithm
/// (pinned `=4.2.0`): a bump that changes it leaves old rows with ids no new review
/// repeats, and must re-verify this.
///
/// Complexity: O(findings).
///
/// # Arguments
/// * `findings` - the judged main list with each finding's gate verdict.
/// * `reviewed_commit` - the pinned SHA of the reviewed revision.
///
/// # Returns
/// One `Pending` entry per recordable finding; none when `reviewed_commit` is no commit id
/// (`is_commit_id`: empty, as a working-tree review would have it, or symbolic), since RV08
/// records history only against a commit.
// MAGI FIX (Loop1 r3, Minor 1): the removed-line rule stated as the code applies it.
pub fn entries_from_review(
    findings: &[(ConsolidatedFinding, GateVerdict)],
    reviewed_commit: &str,
) -> Vec<HistoryEntry> {
    // MAGI FIX (MS3 §6 loop5): the anchor is checked where it is written, not only when it is
    // classified: no commit id, no history (RV08).
    if !is_commit_id(reviewed_commit) {
        return Vec::new();
    }
    findings
        .iter()
        .filter(|(_, v)| matches!(v.state, GateState::Kept | GateState::LowConfidence))
        .filter_map(|(f, _)| {
            let (file, line) = (f.file.as_ref()?, f.line?);
            // MAGI FIX (MS3 §6 loop13): no line to classify in an empty path, and a secret-shaped
            // one never reaches the clear `file` column (the read side withholds it as well).
            if file.is_empty() || find_secret(file).is_some() {
                return None;
            }
            let title = f.titles.join(MERGED_TEXT_SEPARATOR);
            let detail = f.details.join(MERGED_TEXT_SEPARATOR);
            // MAGI FIX (MS3 §6 loop4): a backstop at storage, on top of the masking at the
            // context read: a key block is left out (masking would keep its body).
            // MAGI FIX (MS3 §6 loop6): looked for across the mages' texts too, and by its END
            // line, not only by a header whole in one joined string.
            // MAGI FIX (MS3 §6 loop13): the END line is looked for over the joined (stored) forms
            // too: a footer opening the second text is at a line start only there.
            // MAGI FIX (MS3 §6 loop13): the END line is looked for over the joined (stored) forms
            // too: a footer opening the second text is at a line start only there.
            if [&title, &detail]
                .iter()
                .any(|t| holds_key_block(t) || holds_key_block_end(t))
                || quotes_a_split_key_block(&f.titles, &f.details)
            {
                return None;
            }
            Some(HistoryEntry {
                id: generate_finding_id(file, line, NEUTRAL_CATEGORY),
                file: file.clone(),
                line: Some(line),
                reviewed_commit: reviewed_commit.to_string(),
                title: mask_secrets(&title),
                detail: mask_secrets(&detail),
                status: HistoryStatus::Pending,
            })
        })
        .collect()
}

/// The five dashes that open and close a PEM or OpenPGP armor line.
const ARMOR_DASHES: &str = "-----";
/// The four dashes that open and close an SSH2 armor line.
const SSH2_DASHES: &str = "----";
/// The text every private-key END line holds.
const PRIVATE_KEY: &str = "PRIVATE KEY";

/// The END markers of the private-key blocks RV22 leaves out, each as `(dashes the line starts
/// with, word right after them, text the line must also hold)`, compared without ASCII case. The
/// first matches the PEM footers (plain, RSA, EC, OPENSSH, ENCRYPTED) and the OpenPGP one (PGP
/// ... BLOCK); the second the SSH2 one. Pinned by `canonical_footers_are_recognized`.
// MAGI FIX (MS3 §6 loop8): built from named dashes, the review tool mangled the joined literal.
const KEY_BLOCK_END_MARKERS: &[(&str, &str, &str)] = &[
    (ARMOR_DASHES, "END ", PRIVATE_KEY),
    (SSH2_DASHES, " END SSH2 ", PRIVATE_KEY),
];

/// Whether a finding's texts quote a private-key block that no single joined string shows by its
/// header: a header split across two mages' texts (the join puts a separator inside it), or
/// split between the title and the detail, is whole in the separator-free concatenations; a
/// header the detector cannot read still leaves the block's END line
/// (`a_key_block_header_split_across_two_mages_texts_or_known_by_its_end_line_is_not_stored`).
///
/// The safe direction of RV22's tie-break ("no byte of the secret leaves the process"): a false
/// match only keeps one finding out of the history. Limit, declared: a header broken inside one
/// string, with its END line gone too, is recognized by nothing, as everywhere else the detector
/// runs (`holds_key_block`). O(bytes of the texts).
fn quotes_a_split_key_block(titles: &[String], details: &[String]) -> bool {
    let (titles, details) = (titles.concat(), details.concat());
    let both = format!("{titles}{details}");
    [&titles, &details, &both]
        .iter()
        .any(|t| holds_key_block(t) || holds_key_block_end(t))
}

/// Whether `text` holds the END line of a private-key block ([`KEY_BLOCK_END_MARKERS`]), the
/// line and the markers compared with every ASCII space removed, so a footer joined at a token
/// boundary without its space (`…PRIVATE` + `KEY-----`) or an SSH2 one written without spaces is
/// still one (`split_footers_are_recognized`); END closes only at the line's start. O(n).
// MAGI FIX (MS3 §6 loop11): the exact spacing let a split footer through.
// MAGI FIX (MS3 §6 loop12): every ASCII whitespace goes, not the space alone (a tab evaded it).
fn holds_key_block_end(text: &str) -> bool {
    let spaceless = |s: &str| {
        s.chars()
            .filter(|c| !c.is_ascii_whitespace())
            .collect::<String>()
    };
    text.lines().any(|line| {
        let line = spaceless(line.trim()).to_ascii_uppercase();
        KEY_BLOCK_END_MARKERS.iter().any(|(dashes, word, holds)| {
            line.strip_prefix(dashes)
                .is_some_and(|rest| rest.starts_with(&spaceless(word)))
                && line.contains(&spaceless(holds))
        })
    })
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
    fn tabs_in_a_footer_still_end_a_block() {
        // MAGI FIX (MS3 §6 loop12): only ASCII spaces were removed before the END markers were
        // compared, so a hand-edited footer with tabs between its words was no END line.
        let (d, k) = (ARMOR_DASHES, PRIVATE_KEY);
        for footer in [
            format!("{d}END\tRSA\t{k}{d}"),
            format!("{d}END \tRSA {k}\t{d}"),
            format!("{SSH2_DASHES} END\tSSH2\tENCRYPTED\t{k} {SSH2_DASHES}"),
        ] {
            assert!(holds_key_block_end(&footer), "{footer:?}");
        }
        assert!(!holds_key_block_end("plain text\twith a tab"));
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

    #[test]
    fn a_diff_of_another_file_than_the_entrys_gives_no_signal() {
        // [MAGI S6 MS3 loop6, R3 Melchior] A directory as `file` (`src`) is a tree-wide pathspec
        // even with literal pathspecs: another file's hunk at the same line read as attended.
        let mut dir = entry(Some(5));
        dir.file = "src".into();
        let other = "diff --git a/src/b.rs b/src/b.rs\n@@ -4,3 +4,3 @@\n";
        let git = FakeGit::from(&[
            ("merge-base --is-ancestor c1 HEAD", Ok("")),
            (
                "diff --no-color --no-ext-diff --no-textconv --no-renames --unified=0 --relative c1 HEAD -- src",
                Ok(other),
            ),
            ("rev-parse --verify HEAD^{commit}", Ok("c2\n")),
        ]);
        assert_eq!(classify(&dir, "HEAD", &git), HistoryStatus::Pending);
        // The entry's own file named by the header is still read.
        let own = FakeGit::from(&[
            ("merge-base --is-ancestor c1 HEAD", Ok("")),
            (
                DIFF_C1_HEAD,
                Ok("diff --git a/a.rs b/a.rs\nindex 1..2 100644\n@@ -4,3 +4,3 @@\n"),
            ),
        ]);
        assert_eq!(
            classify(&entry(Some(5)), "HEAD", &own),
            HistoryStatus::Attended
        );
    }

    const DIFF_C1_HEAD: &str = "diff --no-color --no-ext-diff --no-textconv --no-renames --unified=0 --relative c1 HEAD -- a.rs";
    /// [CP2 MS3 loop 12] A weak reject needs a commit after the reviewed one.
    const HEAD_IS_C2: &str = "rev-parse --verify HEAD^{commit}";
    /// [MAGI S6 MS3 loop7] `a.rs` is a regular file at `c1` (an empty diff weak-rejects only one).
    const A_RS_IS_A_FILE_AT_C1: (&str, Result<&str, &str>) =
        ("ls-tree -z c1 -- a.rs", Ok("100644 blob bbbb\ta.rs\0"));

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
    fn a_symbolic_reviewed_commit_gives_no_signal_even_when_its_diff_covers_the_line() {
        // [MAGI S6 MS3 loop5, R3 Melchior, Caspar] The hexadecimal check ran only on the weak
        // reject branch: a branch name or `HEAD~1` in a hand-edited row that resolves as an
        // ancestor, with a diff covering the line, was marked attended against a moving ref.
        for commit in ["main", "HEAD~1"] {
            let mut e = entry(Some(5));
            e.reviewed_commit = commit.into();
            let diff = format!(
                "diff --no-color --no-ext-diff --no-textconv --no-renames --unified=0 \
                 --relative {commit} HEAD -- a.rs"
            );
            let ancestor = format!("merge-base --is-ancestor {commit} HEAD");
            let git = FakeGit::from(&[
                (ancestor.as_str(), Ok("")),
                (diff.as_str(), Ok("@@ -5 +5 @@\n")),
            ]);
            assert_eq!(
                classify(&e, "HEAD", &git),
                HistoryStatus::Pending,
                "{commit}"
            );
        }
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
        // [MAGI S6 MS3 loop5] anchored to a commit id (`c9`), since a symbolic anchor now gives no
        // signal at all and would keep this guardian green under its mutation.
        let mut fresh = entry(Some(5));
        fresh.id = "new".into();
        fresh.reviewed_commit = "c9".into();
        let git = FakeGit::from(&[("merge-base --is-ancestor c1 HEAD", Ok("")),
                              (DIFF_C1_HEAD, Ok("@@ -5 +5 @@\n")),
                              ("merge-base --is-ancestor c9 HEAD", Ok("")),
                              ("diff --no-color --no-ext-diff --no-textconv --no-renames --unified=0 --relative c9 HEAD -- a.rs", Ok("")),
                              // [CP2 MS3 loop 13] Scripted, so reclassifying the new entry would make it WeakReject
                              // (HEAD resolves to a commit other than the c9 it was reviewed at): the
                              // mutation goes red whatever FakeGit answers for an unscripted argv.
                              (HEAD_IS_C2, Ok("c8\n"))]);
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
            // [MAGI S6 MS3 loop7] An empty diff weak-rejects a regular file only.
            A_RS_IS_A_FILE_AT_C1,
        ]);
        let (updates, _) = reclassify_then_record(&[done], &[], "HEAD", &git);
        assert_eq!(updates, vec![("i".to_string(), HistoryStatus::WeakReject)]);
    }

    #[test]
    fn a_validated_entry_goes_back_to_pending_once_head_is_the_reviewed_commit_again() {
        // [MAGI S6 MS3 loop5, R3 Caspar] RV08: with no commit after the reviewed one there is
        // no signal, and the entry is pending. Attended after c1 -> c2, then HEAD reset to c1:
        // it stayed attended and kept being served as validated context. That answer is
        // definitive (git resolved HEAD to the reviewed commit), unlike a git that failed.
        for status in [HistoryStatus::Attended, HistoryStatus::WeakReject] {
            let mut done = entry(Some(5));
            done.status = status;
            let git = FakeGit::from(&[
                ("merge-base --is-ancestor c1 HEAD", Ok("")),
                (DIFF_C1_HEAD, Ok("")),
                (HEAD_IS_C2, Ok("c1\n")),
            ]);
            let (updates, _) = reclassify_then_record(&[done], &[], "HEAD", &git);
            assert_eq!(
                updates,
                vec![("i".to_string(), HistoryStatus::Pending)],
                "{status:?}"
            );
        }
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
    fn a_validated_entry_off_heads_line_goes_back_to_pending_and_is_never_served() {
        // [MAGI S6 MS3 loop7, R3 Caspar] Attended on `feature`, then HEAD on `main`, where the
        // reviewed commit is no ancestor: git's answer is definitive (both commits resolve), yet
        // the entry stayed attended and was served on `main`.
        let off_line = FakeGit::from(&[
            ("merge-base --is-ancestor c1 HEAD", Err("exit 1")),
            ("rev-parse --verify c1^{commit}", Ok("c1\n")),
            (HEAD_IS_C2, Ok("c2\n")),
        ]);
        for status in [HistoryStatus::Attended, HistoryStatus::WeakReject] {
            let mut done = entry(Some(5));
            done.status = status;
            assert!(!served_as_validated(&done, "HEAD", &off_line), "{status:?}");
            let (updates, _) = reclassify_then_record(&[done], &[], "HEAD", &off_line);
            assert_eq!(updates, vec![("i".to_string(), HistoryStatus::Pending)]);
        }
        let on_line = FakeGit::from(&[("merge-base --is-ancestor c1 HEAD", Ok(""))]);
        let mut done = entry(Some(5));
        done.status = HistoryStatus::Attended;
        assert!(served_as_validated(&done, "HEAD", &on_line));
        let broken = FakeGit::from(&[]);
        assert!(!served_as_validated(&done, "HEAD", &broken), "fails closed");
    }

    #[test]
    fn pending_is_never_served() {
        // MAGI FIX (MS3 §6 loop11): `served_as_validated` read ancestry only, not the status.
        let on_line = FakeGit::from(&[("merge-base --is-ancestor c1 HEAD", Ok(""))]);
        let pending = entry(Some(5));
        assert_eq!(pending.status, HistoryStatus::Pending);
        assert!(!served_as_validated(&pending, "HEAD", &on_line));
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
    fn a_finding_is_stored_with_its_secrets_masked_and_one_quoting_a_key_block_is_not_stored() {
        // [MAGI S6 MS3 loop4, R5 Balthasar] Backstop for the history, an egress point of RV22
        // when it is served back: a one-line secret quoted by the trio is stored masked, and a
        // finding quoting a private-key block is not stored (masking would leave its body).
        use crate::review::gate::GateState;
        let key = "sk-ant-api03-AAAAAAAAAAAAAAAAAAAA"; // allow-secret-scan
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        let mut keyed = found("a.rs", 9, "key");
        keyed.details = vec![format!("{begin}\nMIIEvQIBADANBg")];
        let main = vec![
            (
                found("a.rs", 3, &format!("leaks {key}")),
                judged(GateState::Kept),
            ),
            (keyed, judged(GateState::Kept)),
        ];
        let entries = entries_from_review(&main, "c1");
        assert_eq!(entries.len(), 1, "{entries:?}");
        assert!(
            !entries[0].title.contains("sk-ant-")
                && entries[0].title.contains("<redacted: anthropic-key>"),
            "{}",
            entries[0].title
        );
    }

    #[test]
    fn a_key_block_whose_header_ends_the_title_and_body_opens_the_detail_is_not_stored() {
        // [MAGI S6 MS3 loop5, R3 Melchior] Title and detail are checked apart; a block split
        // between them is still caught, because a block is recognised by its header alone,
        // whatever follows it (no END line is needed), and the header is whole in the title.
        use crate::review::gate::GateState;
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        let mut split = found("a.rs", 3, &format!("see {begin}"));
        split.details = vec!["MIIEvQIBADANBgkqhkiG9w0BAQEFAASC\n-----END PRIVATE KEY-----".into()];
        let main = vec![(split, judged(GateState::Kept))];
        assert!(entries_from_review(&main, "c1").is_empty());
    }

    #[test]
    fn a_key_block_header_split_across_two_mages_texts_or_known_by_its_end_line_is_not_stored() {
        // [MAGI S6 MS3 loop6, R3 Melchior] Two mages' texts are joined with a separator, so a
        // header split between them is whole in neither the parts nor the join; and a header the
        // detector cannot read leaves only the body and its END line. Both were stored.
        use crate::review::gate::GateState;
        let body = "MIIEvQIBADANBgkqhkiG9w0BAQEFAASC";
        let mut across = found("a.rs", 3, concat!("see -----", "BEGIN PRIV"));
        across.titles.push(concat!("ATE ", "KEY-----").into());
        across.details = vec![body.into()];
        let mut end_only = found("a.rs", 4, "see the key");
        end_only.details = vec![format!(
            "{}\n{body}\n{}",
            concat!("-----", "BEG IN ", "PRIVATE KEY-----"),
            concat!("-----", "END PRIVATE KEY-----")
        )];
        for (case, f) in [("across", across), ("end only", end_only)] {
            let main = vec![(f, judged(GateState::Kept))];
            assert!(entries_from_review(&main, "c1").is_empty(), "{case}");
        }
        let clean = vec![(found("a.rs", 5, "fine"), judged(GateState::Kept))];
        assert_eq!(entries_from_review(&clean, "c1").len(), 1);
    }

    #[test]
    fn odd_file_paths_are_not_stored() {
        // MAGI FIX (MS3 §6 loop13, R3 Caspar): an empty path has no line to classify, and a
        // secret-shaped one would be stored in the clear `file` column (the read side withholds
        // it, but the backstop is here too: RV22, no byte of a secret leaves).
        use crate::review::gate::GateState;
        let shaped = "src/sk-ant-api03-AAAAAAAAAAAAAAAAAAAA.rs"; // allow-secret-scan
        for (case, file) in [("empty", ""), ("secret", shaped)] {
            let main = vec![(found(file, 3, "a finding"), judged(GateState::Kept))];
            assert!(entries_from_review(&main, "c1").is_empty(), "{case}");
        }
    }

    #[test]
    fn joined_end_line_is_not_stored() {
        // MAGI FIX (MS3 §6 loop13, R3 Melchior): a footer that opens the second of two texts is
        // at a line start only in the stored (joined) form, never in the separator-free
        // concatenations, so the END check must also run over the joined title and detail.
        use crate::review::gate::GateState;
        let footer = concat!("-----", "END PRIVATE KEY-----");
        let mut in_detail = found("a.rs", 3, "see the key");
        in_detail.details = vec!["MIIEvQIBADANBgkqhkiG9w0BAQEFAASC".into(), footer.into()];
        let mut in_title = found("a.rs", 4, "lead");
        in_title.titles.push(footer.into());
        for (case, f) in [("detail", in_detail), ("title", in_title)] {
            let main = vec![(f, judged(GateState::Kept))];
            assert!(entries_from_review(&main, "c1").is_empty(), "{case}");
        }
    }

    #[test]
    fn every_canonical_private_key_footer_keeps_a_finding_with_a_disrupted_header_out() {
        // [MAGI S6 MS3 loop7, R3 Caspar] The END markers are checked against the real footers
        // (the review copy showed `-----END ` neutralized): with the header broken so no
        // detector reads it, each format's footer alone keeps the key body out of the history.
        use crate::review::gate::GateState;
        let body = "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7";
        for kind in ["", "RSA ", "ENCRYPTED ", "OPENSSH ", "PGP "] {
            let block = if kind == "PGP " {
                "PRIVATE KEY BLOCK"
            } else {
                "PRIVATE KEY"
            };
            let mut f = found("a.rs", 3, "see the key");
            f.details = vec![format!(
                "{}{kind}{block}-----\n{body}\n{}{kind}{block}-----",
                concat!("-----", "BEG IN "),
                concat!("-----", "END ")
            )];
            let main = vec![(f, judged(GateState::Kept))];
            assert!(entries_from_review(&main, "c1").is_empty(), "{kind}{block}");
        }
        let mut ssh2 = found("a.rs", 3, "see the key");
        ssh2.details = vec![format!(
            "---- BEG IN SSH2 ENCRYPTED PRIVATE KEY ----\n{body}\n{}",
            concat!("---- ", "END SSH2 ENCRYPTED PRIVATE KEY ----")
        )];
        assert!(entries_from_review(&[(ssh2, judged(GateState::Kept))], "c1").is_empty());
    }

    #[test]
    fn split_footers_are_recognized() {
        // MAGI FIX (MS3 §6 loop11): a footer joined without its space at a token boundary
        // ('...PRIVATE' + 'KEY-----'), or an SSH2 one without spaces, slipped the backstop.
        let d = ARMOR_DASHES;
        let s = SSH2_DASHES;
        for footer in [
            format!("{d}END RSA PRIVATEKEY{d}"),
            format!("{d}ENDRSA PRIVATE KEY{d}"),
            format!("{s}END SSH2 {PRIVATE_KEY}{s}"),
        ] {
            assert!(holds_key_block_end(&footer), "{footer}");
        }
        let mid = format!("x {d}END RSA {PRIVATE_KEY}{d}");
        assert!(!holds_key_block_end(&mid), "END closes only at line start");
    }

    #[test]
    fn canonical_footers_are_recognized() {
        // [MAGI S6 MS3 loop8, R3 Caspar/Melchior] The review tool showed the armor dashes
        // mangled; the markers are built from named dash constants, checked here against an
        // independent count, and every canonical footer built from them keeps the body out.
        use crate::review::gate::GateState;
        assert_eq!(
            (ARMOR_DASHES, SSH2_DASHES),
            ("-".repeat(5).as_str(), "-".repeat(4).as_str())
        );
        let d = ARMOR_DASHES;
        let footers = [
            format!("{d}END {PRIVATE_KEY}{d}"),
            format!("{d}END RSA {PRIVATE_KEY}{d}"),
            format!("{d}END OPENSSH {PRIVATE_KEY}{d}"),
            format!("{d}END PGP {PRIVATE_KEY} BLOCK{d}"),
            format!("{SSH2_DASHES} END SSH2 ENCRYPTED {PRIVATE_KEY} {SSH2_DASHES}"),
        ];
        for footer in footers {
            assert!(holds_key_block_end(&footer), "{footer}");
            let mut f = found("a.rs", 3, "see the key");
            f.details = vec![format!(
                "{d}BEG IN {PRIVATE_KEY}{d}\nMIIEvQIBADANBg\n{footer}"
            )];
            assert!(entries_from_review(&[(f, judged(GateState::Kept))], "c1").is_empty());
        }
    }

    #[test]
    fn a_review_with_no_commit_id_to_anchor_to_records_nothing() {
        // [MAGI S6 MS3 loop5, R3 Caspar] Any anchor string was serialized into a row: an empty
        // one (a working-tree path that supplied `""`, which RV08 forbids) or a symbolic one,
        // which classification can never resolve to a signal.
        let main = vec![(
            found("a.rs", 3, "k"),
            judged(crate::review::gate::GateState::Kept),
        )];
        for anchor in ["", "HEAD", "main", "HEAD~1", "-c"] {
            assert!(entries_from_review(&main, anchor).is_empty(), "{anchor:?}");
        }
        assert_eq!(entries_from_review(&main, "c1").len(), 1);
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
    fn an_empty_diff_is_a_weak_reject_only_for_a_regular_file_of_the_reviewed_commit() {
        // [MAGI S6 MS3 loop7, R3 Caspar] An empty diff names no file, so `names_only` passed
        // vacuously: an entry whose `file` is a directory (or a path that never existed) became a
        // weak reject though it refers to no file. A weak reject now needs a blob at that path.
        let for_path = |path: &str, tree: Result<&'static str, &'static str>| {
            let diff = DIFF_C1_HEAD.replace("-- a.rs", &format!("-- {path}"));
            let tree_call = format!("ls-tree -z c1 -- {path}");
            let git = FakeGit::from(&[
                ("merge-base --is-ancestor c1 HEAD", Ok("")),
                (diff.as_str(), Ok("")),
                (HEAD_IS_C2, Ok("c2\n")),
                (tree_call.as_str(), tree),
            ]);
            let mut e = entry(Some(5));
            e.file = path.into();
            classify(&e, "HEAD", &git)
        };
        assert_eq!(
            for_path("src", Ok("040000 tree aaaa\tsrc\0")),
            HistoryStatus::Pending
        );
        assert_eq!(for_path("gone.rs", Ok("")), HistoryStatus::Pending);
        assert_eq!(for_path("a.rs", Err("fatal")), HistoryStatus::Pending);
        assert_eq!(
            for_path("a.rs", Ok("100644 blob bbbb\ta.rs\0")),
            HistoryStatus::WeakReject
        );
    }

    #[test]
    fn merged_without_changes_is_a_weak_reject() {
        // [MAGI S6 MS3 loop7] An empty diff weak-rejects a regular file only: `a.rs` is one.
        let git = FakeGit::from(&[
            ("merge-base --is-ancestor c1 HEAD", Ok("")),
            (DIFF_C1_HEAD, Ok("")),
            (HEAD_IS_C2, Ok("c2\n")),
            A_RS_IS_A_FILE_AT_C1,
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
    fn a_file_that_turned_binary_is_attended() {
        // [MAGI S6 MS3 loop3, R3 Melchior] git gives no hunk for a binary change, only the
        // `Binary files ... differ` line, so the entry read as untouched (a weak reject) while
        // its line no longer exists. The whole file changed, as a deleted file is covered whole.
        let git = FakeGit::from(&[
            ("merge-base --is-ancestor c1 HEAD", Ok("")),
            (
                DIFF_C1_HEAD,
                Ok("diff --git a/a.rs b/a.rs\nindex 1..2 100644\nBinary files a/a.rs and b/a.rs differ\n"),
            ),
            (HEAD_IS_C2, Ok("c2\n")),
        ]);
        assert_eq!(
            classify(&entry(Some(5)), "HEAD", &git),
            HistoryStatus::Attended
        );
    }

    #[test]
    fn an_abbreviated_reviewed_commit_at_head_stays_pending() {
        // [MAGI S6 MS3 loop3, R3 all three] `rev-parse` answers the full SHA; an entry anchored
        // to an abbreviation of that same commit compared unequal and read as a weak reject.
        let full = "abc1234def5678901234567890123456789abcde";
        let git = FakeGit::from(&[
            ("merge-base --is-ancestor abc1234 HEAD", Ok("")),
            (
                "diff --no-color --no-ext-diff --no-textconv --no-renames --unified=0 --relative abc1234 HEAD -- a.rs",
                Ok(""),
            ),
            (HEAD_IS_C2, Ok(full)),
        ]);
        let short = HistoryEntry {
            reviewed_commit: "abc1234".into(),
            ..entry(Some(5))
        };
        assert_eq!(classify(&short, "HEAD", &git), HistoryStatus::Pending);
    }

    #[test]
    fn a_full_reviewed_sha_against_an_abbreviated_head_of_that_commit_stays_pending() {
        // [MAGI S6 MS3 loop7, R3 Caspar] The other direction: `head` is resolved by `rev-parse
        // --verify`, which prints the full name whatever `core.abbrev` says (pinned against real
        // git by `rev_parse_verify_prints_the_full_sha_whatever_core_abbrev_says`), so a full
        // stored SHA equals what an abbreviated `head` resolves to: no later commit, pending.
        let full = "abc1234def5678901234567890123456789abcde";
        let diff = format!(
            "diff --no-color --no-ext-diff --no-textconv --no-renames --unified=0 --relative {full} abc1234 -- a.rs"
        );
        let git = FakeGit::from(&[
            (
                format!("merge-base --is-ancestor {full} abc1234").as_str(),
                Ok(""),
            ),
            (diff.as_str(), Ok("")),
            ("rev-parse --verify abc1234^{commit}", Ok(full)),
        ]);
        let anchored = HistoryEntry {
            reviewed_commit: full.into(),
            ..entry(Some(5))
        };
        assert_eq!(classify(&anchored, "abc1234", &git), HistoryStatus::Pending);
    }

    #[test]
    fn stored_sha_is_compared_in_full() {
        // [MAGI S6 MS3 loop8, R3 Balthasar] An abbreviated stored commit that is a prefix of
        // HEAD's SHA but resolves to another commit read as HEAD itself (a prefix comparison).
        let head = "abc1234def5678901234567890123456789abcde";
        let other = "abc1234000000000000000000000000000000000";
        let git = FakeGit::from(&[
            ("merge-base --is-ancestor abc1234 HEAD", Ok("")),
            (
                "diff --no-color --no-ext-diff --no-textconv --no-renames --unified=0 --relative abc1234 HEAD -- a.rs",
                Ok(""),
            ),
            (HEAD_IS_C2, Ok(head)),
            ("rev-parse --verify abc1234^{commit}", Ok(other)),
            ("ls-tree -z abc1234 -- a.rs", Ok("100644 blob f00\ta.rs\0")),
        ]);
        let short = HistoryEntry {
            reviewed_commit: "abc1234".into(),
            ..entry(Some(5))
        };
        assert_eq!(classify(&short, "HEAD", &git), HistoryStatus::WeakReject);
    }

    #[test]
    fn a_reviewed_commit_that_is_not_a_sha_stays_pending_never_a_weak_reject() {
        // [MAGI S6 MS3 loop4, R3 Melchior] The prefix comparison is exact only for a SHA. A
        // symbolic name (a hand-edited row, a future caller) compared unequal to HEAD's SHA and
        // read as a later commit: unvalidated context served as validated. Now: no signal.
        let full = "abc1234def5678901234567890123456789abcde";
        let git = FakeGit::from(&[
            ("merge-base --is-ancestor main HEAD", Ok("")),
            (
                "diff --no-color --no-ext-diff --no-textconv --no-renames --unified=0 --relative main HEAD -- a.rs",
                Ok(""),
            ),
            (HEAD_IS_C2, Ok(full)),
        ]);
        let symbolic = HistoryEntry {
            reviewed_commit: "main".into(),
            ..entry(Some(5))
        };
        assert_eq!(classify(&symbolic, "HEAD", &git), HistoryStatus::Pending);
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

    /// Counts every git call and answers success with empty output.
    struct CountingGit(std::sync::atomic::AtomicUsize);

    impl GitCommand for CountingGit {
        fn run_bytes(&self, _args: &[&str]) -> Result<Vec<u8>, crate::review::error::ReviewError> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(Vec::new())
        }
    }

    #[test]
    fn an_entry_without_a_file_stays_pending_without_running_git() {
        // [MAGI S6 MS3 loop3, R3 Caspar] An empty pathspec after `--` would diff the whole
        // tree, and a hunk of any file at the same line number would read as attended.
        let git = CountingGit(std::sync::atomic::AtomicUsize::new(0));
        let no_file = HistoryEntry {
            file: String::new(),
            ..entry(Some(5))
        };
        assert_eq!(classify(&no_file, "HEAD", &git), HistoryStatus::Pending);
        assert_eq!(git.0.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[test]
    fn a_reference_starting_with_a_dash_is_never_passed_to_git() {
        // Security guardian: a stored commit or a head such as `--output=x` would be read by git as
        // an option; classify must stay Pending and must not run git at all.
        let calls = |e: &HistoryEntry, head: &str| {
            let git = CountingGit(std::sync::atomic::AtomicUsize::new(0));
            let status = classify(e, head, &git);
            (status, git.0.load(std::sync::atomic::Ordering::SeqCst))
        };
        assert_eq!(
            calls(&entry(Some(5)), "--output=x"),
            (HistoryStatus::Pending, 0)
        );
        let dashed = HistoryEntry {
            reviewed_commit: "--output=x".into(),
            ..entry(Some(5))
        };
        assert_eq!(calls(&dashed, "HEAD"), (HistoryStatus::Pending, 0));
    }
}
