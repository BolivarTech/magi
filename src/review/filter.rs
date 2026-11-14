// Author: Julian Bolivar
// Version: 0.25.0
// Date: 2026-11-08

//! What never leaves the process and what is not code: the per-file filter of a review.
//!
//! Every file of a review is classified here before any byte of it is batched: a path a review
//! never reads or `.gitignore` excludes, a path or content that holds a secret, a file over the
//! byte cap, a binary and a text that is not UTF-8 are left out with their cause; everything
//! else is kept, and a kept file with no language extractor is declared.

use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::path::Path;

use super::diffmask::{fallback_diff, rebuilt_diff, HUNK_START};
use super::error::ReviewError;
use super::input::{
    in_work_tree, is_binary_marker_line, is_binary_section, renamed_from, split_nul, ReviewFile,
    ReviewInput,
};
use super::languages::LanguageRegistry;
use super::secrets::{display_path, find_secret, holds_key_block, leaks_secret};
use super::source::{is_never_reviewed, ContentSource, Face, GitCommand, HARD_LINK_CAUSE};

/// The cause of [`ReviewError::NotReviewable`] for a file requested by name that does not exist.
pub(crate) const MISSING: &str = "missing";
/// The tracing target of the filter's run-log entries.
const LOG_TARGET: &str = "magi_rs::review";
/// What git writes after a last diff line that has no newline, newline included.
const NO_NEWLINE_AT_EOF: &str = "\n\\ No newline at end of file\n";
/// `ls-files` for a file named by the caller: tracked or untracked, when a rule excludes it.
const LS_FILES_IGNORED_ANY: &[&str] = &[
    "ls-files",
    "--cached",
    "--others",
    "--ignored",
    "--exclude-standard",
    "-z",
];
/// `ls-files` for a directory or uncommitted changes: the tracked files a rule excludes (the
/// enumeration lists them; untracked ignored files are never enumerated).
///
/// The untracked half of RV22 rests on the enumeration of `src/review/input.rs`
/// (`--exclude-standard`), not on this query: pinned with real git by
/// `an_untracked_ignored_file_is_never_listed_in_a_directory_or_uncommitted_review`
/// (`src/system/review_git.rs`), which turns red if a listing starts yielding them. Querying the
/// untracked ignored files here too (`--others`) would list every file of an ignored build
/// directory, a cost bounded only by git's output cap, for paths no input yields.
// MAGI FIX (S6 loop 13, A2-c3): declined with the guardian that holds the invariant named.
const LS_FILES_IGNORED_TRACKED: &[&str] = &[
    "ls-files",
    "--cached",
    "--ignored",
    "--exclude-standard",
    "-z",
];

/// `rev-parse` for the commit `HEAD` names: no output and exit 1 in a repository without commits.
const HEAD_COMMIT_QUERY: &[&str] = &["rev-parse", "--verify", "--quiet", "HEAD^{commit}"];
/// The `ls-files` option that also lists the paths of a tree, as if still in the index.
const WITH_TREE_OPTION: &str = "--with-tree=";

/// A file left out of a review, with why.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Skipped {
    /// The path, as it may leave the process ([`display_path`]).
    pub path: String,
    /// Why it was left out.
    pub cause: SkipCause,
}

/// Why a file was left out of a review.
///
/// Serialized as the text the report shows, spelled in one place (`SkipCause::as_str`, pinned by
/// `skip_causes_serialize_with_the_spec_texts`): `binary`, `not utf-8`, `secret`, `ignored`,
/// `outside workspace`, `too large`, `not a regular file`, `hard link`, `unreadable`,
/// `disappeared` and `repeated section` (a second, different diff for a path already listed).
/// A file named explicitly that is skipped fails with that same text as the cause
/// of [`ReviewError::NotReviewable`].
// MAGI FIX (S6 loop 13, A2-m6 / A2-b5): the list named 7 of the 10 causes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipCause {
    /// A binary file.
    Binary,
    /// Text that is not valid UTF-8.
    NotUtf8,
    /// A file whose path or content holds a secret.
    Secret,
    /// A file `.gitignore` excludes, or one a review never reads.
    Ignored,
    /// A file that resolves outside the workspace.
    OutsideWorkspace,
    /// A file over the per-file byte cap.
    TooLarge,
    /// A path whose target is not a regular file: a link to a directory inside the workspace, a
    /// FIFO, a socket, a device file, or a submodule (a diff with neither side).
    NotRegularFile,
    /// A working-tree file with more than one hard link: its path is inside the workspace, its
    /// inode may be any file on the same file system.
    HardLink,
    /// A listed file whose read failed (a sharing violation, a permission change).
    Unreadable,
    /// A listed file removed between the listing and its read.
    Disappeared,
    /// A second section for a path already listed, with a different diff: only the first is
    /// reviewed, and this entry declares the one left out.
    DuplicateSection,
}

impl serde::Serialize for SkipCause {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl std::fmt::Display for Skipped {
    /// `<path> (<cause>)`, the shown path and the cause's text.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.path, self.cause.as_str())
    }
}

impl SkipCause {
    /// The cause's text: the one place it is spelled.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Binary => "binary",
            Self::NotUtf8 => "not utf-8",
            Self::Secret => "secret",
            Self::Ignored => "ignored",
            Self::OutsideWorkspace => "outside workspace",
            Self::TooLarge => "too large",
            Self::NotRegularFile => "not a regular file",
            Self::HardLink => HARD_LINK_CAUSE,
            Self::Unreadable => "unreadable",
            Self::Disappeared => "disappeared",
            Self::DuplicateSection => "repeated section",
        }
    }
}

/// The outcome of filtering.
#[derive(Debug)]
pub struct Filtered {
    /// The files that are reviewed, in input order.
    pub kept: Vec<ReviewFile>,
    /// The files left out, with their cause, in input order.
    pub skipped: Vec<Skipped>,
    /// Kept files with no language extractor, as [`display_path`] shows them.
    pub no_extractor: Vec<String>,
    /// Kept files that held a secret, reviewed with each secret masked, in input order.
    pub masked: Vec<Masked>,
}

/// A file reviewed with its secrets masked (RV22): reviewed, never left out.
///
/// Its diff is rebuilt from the masked sides (diffmask); a body line that differs from its side's
/// beyond a mask is replaced by the side's text, and the count is declared in the run log (one WARN
/// per file), not per file in the report: the replaced text is always the masked side's, so the
/// declaration is about drift between git's diff and the file, never about a secret.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Masked {
    /// The path, as it may leave the process ([`display_path`]).
    pub path: String,
    /// Why it is declared: always [`SkipCause::Secret`], as RV22 amended names it; the list, not
    /// the cause, tells it from an omission
    /// (`a_masked_file_and_an_omitted_one_share_the_cause_and_never_the_list`).
    pub cause: SkipCause,
    /// The name of the first secret pattern found in it.
    pub pattern: &'static str,
}

/// What classifying one file decided.
enum Verdict {
    /// Reviewed, as [`Kept`] says.
    Keep(Kept),
    /// Left out, with its cause.
    Skip(SkipCause),
}

/// How a kept file is reviewed.
struct Kept {
    /// The diff it is reviewed through (`None`: reviewed whole), which replaces the file's own:
    /// rebuilt from the masked sides when a secret was masked, or built from the old side for a
    /// deleted text file whose diff git printed as binary.
    diff: Option<String>,
    /// Whether its old side stays held (`false`: reviewed whole from its new side, a text file
    /// whose diff git printed as binary).
    with_old: bool,
    /// The pattern of the first secret masked in it, if any.
    masked: Option<&'static str>,
}

/// Splits the files of a review into what is reviewed and what is left out.
///
/// For each file, in this order, the first cause that applies wins: a path a review never
/// reads (a `.git` or `.magi` component, or a name that starts with `.magi-rs-memory.db`,
/// whatever `is_ignored` says) or one `is_ignored` names, as the new path or as a rename's old
/// path → `ignored`; a path that holds a
/// secret → `secret`, without reading it; a file that does not exist → an error, never a skip;
/// over the byte cap → `too large`; a NUL anywhere in a side → `binary`; not UTF-8 →
/// `not utf-8`. A one-line secret in the content no longer leaves the file out (RV22 as amended
/// in T3.7): the source masks it in each side it holds ([`ContentSource::masked`]), the diff is
/// rebuilt from the masked sides, and the file is kept and declared in [`Filtered::masked`] (see
/// `kept_verdict` for the cases that still leave it out as `secret`). The exception is a
/// multi-line private-key block: the source holds such a side unmasked (a key body has no
/// pattern of its own), and this filter leaves the file out as `secret` when either side still
/// holds one, so no later stage reads the file. A side that still holds a secret of any kind,
/// which only a source that does not mask hands over, leaves the file out the same way.
/// A diff section whose header lines (before the first `@@`) carry git's binary marker
/// ([`is_binary_section`]) is judged by its bytes the same way, and when both sides are text it
/// is kept with no diff, reviewed whole from its new side; a side that cannot be read is left
/// out under its own cause, as for any file; a deleted text file is kept with the deletion diff
/// git would have printed, and it stays `binary` only when the file has neither side
/// (`suppressed_diff`).
/// The binary and UTF-8 checks run on both sides that exist (binary first). The
/// secret scan covers both whole sides, not only what is sent: the safe direction. A renamed
/// file's old side is read under its old path (`rename from`). Once every file is judged, the
/// source is sealed ([`ContentSource::seal`]): no later stage is served a side held raw with a
/// secret, whether this filter read it or not
/// (`once_sealed_the_cache_never_serves_a_side_raw_with_a_secret_even_one_never_classified`,
/// `src/review/source.rs`).
///
/// # Arguments
///
/// * `files` - The files the input resolved to.
/// * `source` - The review's single content source.
/// * `registry` - The language extractors, to declare kept files with none.
/// * `explicit` - Whether the file was requested by name: then a skip is an error.
/// * `is_ignored` - A positive set: whether `.gitignore` excludes a path (built by the caller
///   from [`ignored_paths`], once per review).
///
/// # Returns
///
/// The kept files, the skipped ones, the kept ones with no extractor and the kept ones with a
/// secret masked. Every path that leaves the process goes through [`display_path`]. Each
/// `secret` skip and each masked file writes one INFO entry with the shown path and the
/// pattern's name, never the value.
///
/// # Errors
///
/// [`ReviewError::NotReviewable`] when `explicit` and the file is skipped or missing,
/// [`ReviewError::Io`] when a file named explicitly cannot be read, [`ReviewError::NothingToReview`]
/// when no file is kept, [`ReviewError::InputTooLarge`] when the kept files pass the review's
/// total cap ([`ContentSource::settle`], checked after each file), and any other error of
/// `source`.
///
/// O(total bytes read) time.
pub fn filter_files(
    files: Vec<ReviewFile>,
    source: &dyn ContentSource,
    registry: &LanguageRegistry,
    explicit: bool,
    is_ignored: &dyn Fn(&str) -> bool,
) -> Result<Filtered, ReviewError> {
    let classified = classify_files(files, source, registry, explicit, is_ignored);
    // MAGI FIX (MS3 §6 loop5): the classification is over; from here no stage is served a side
    // held raw with a secret, a path this filter never saw included.
    // MAGI FIX (MS3 §6 loop6): on every path, an early error included, not on success alone
    // (`a_filter_that_fails_still_seals_the_source`).
    source.seal();
    let filtered = classified?;
    if filtered.kept.is_empty() {
        // MAGI FIX (S6 loop 3): the omissions and their causes travel with the error (RV20/RV22).
        return Err(ReviewError::NothingToReview {
            skipped: filtered.skipped,
        });
    }
    Ok(filtered)
}

/// The classification loop of [`filter_files`], which seals the source after it whatever it
/// returns.
///
/// # Errors
///
/// As [`filter_files`], but for `NothingToReview`, which the caller decides.
fn classify_files(
    files: Vec<ReviewFile>,
    source: &dyn ContentSource,
    registry: &LanguageRegistry,
    explicit: bool,
    is_ignored: &dyn Fn(&str) -> bool,
) -> Result<Filtered, ReviewError> {
    let mut filtered = Filtered {
        kept: Vec::new(),
        skipped: Vec::new(),
        no_extractor: Vec::new(),
        masked: Vec::new(),
    };
    // The sides a kept file was scanned with: never released, whoever else names their path.
    let mut kept_sides: HashSet<(Face, String)> = HashSet::new();
    // The paths kept (a second listing of one is always declared as a repeated section) and the
    // (path, diff hash) pairs already left out, so one skip is listed once.
    let mut kept_paths: HashSet<String> = HashSet::new();
    let mut skipped_seen: HashSet<(String, Option<u64>)> = HashSet::new();
    // The paths left out as `secret`: a later listing of one is never kept (RV22).
    let mut secret_paths: HashSet<String> = HashSet::new();
    for file in files {
        let sides = sides_of(&file);
        // MAGI FIX (MS3 §6 loop5): the diff moves into the verdict; it was copied for every file.
        let ReviewFile { path, diff } = file;
        let diff_key = diff.as_deref().map(diff_hash);
        match classify(&path, diff, source, is_ignored, explicit)? {
            Verdict::Keep(kept) => {
                // MAGI FIX (MS3 §6 loop12): a path is kept once, whoever lists it again (the
                // repeat reads the cached, masked side and would be reviewed and declared twice).
                // MAGI FIX (MS3 §6 loop13): a repeat with another diff is declared, not dropped.
                // MAGI FIX (MS3 §6 loop16, R6a1): the guard runs both ways: a path already left
                // out as `secret` is not kept by a later listing either.
                if kept_paths.contains(&path) || secret_paths.contains(&path) {
                    // MAGI FIX (MS3 §6 loop15): always declared, whatever the diff's hash says:
                    // a collision or an honest repeat is never a silent drop.
                    filtered.skipped.push(Skipped {
                        path: display_path(&path),
                        cause: SkipCause::DuplicateSection,
                    });
                    // MAGI FIX (MS3 §6 loop14): the repeat's own sides (a rename's other path)
                    // are released like any left-out file's, but never one a kept file read.
                    release_unkept(source, &kept_sides, sides);
                    continue;
                }
                kept_paths.insert(path.clone());
                if registry.get(&path).is_none() {
                    filtered.no_extractor.push(display_path(&path));
                }
                let mut sides = sides.into_iter();
                if kept.with_old {
                    // Both sides stay: the deleted lines are read from the old side.
                    kept_sides.extend(sides);
                } else {
                    // MAGI FIX (S6 loop 17, A2-2): only the new side is reviewed; the old side read
                    // to judge the file stayed cached and was charged against the review total.
                    kept_sides.extend(sides.next());
                    release_unkept(source, &kept_sides, sides);
                }
                if let Some(pattern) = kept.masked {
                    filtered.masked.push(masked_file(&path, pattern));
                }
                // MAGI FIX (MS3 §6 loop14): a kept path is never secret-shaped: `classify` skips
                // such a path as `secret` before any verdict (`a_path_shaped_like_a_key_is_never_sent`),
                // and every display goes through `display_path` besides.
                filtered.kept.push(ReviewFile {
                    path,
                    diff: kept.diff,
                });
            }
            Verdict::Skip(cause) if explicit => {
                return Err(ReviewError::NotReviewable {
                    path: display_path(&path),
                    cause: cause.as_str(),
                })
            }
            Verdict::Skip(cause) => {
                // MAGI FIX (MS3 §6 loop14): a path already kept is not also skipped under its
                // repeat's cause: the repeat is declared as the dropped section it is.
                let cause = match kept_paths.contains(&path) {
                    true => SkipCause::DuplicateSection,
                    false => cause,
                };
                // MAGI FIX (S6 loop 12, A2-c1 / A2-c5): a skipped file is never sent, so the
                // cache keeps none of it: the total cap bounds what is reviewed.
                // MAGI FIX (S6 loop 13, A2-m2 / A2-b1): side by side, and never a side a kept file
                // was scanned with: a rename's old path can be a kept file's path.
                release_unkept(source, &kept_sides, sides);
                if cause == SkipCause::Secret {
                    secret_paths.insert(path.clone());
                }
                if skipped_seen.insert((path.clone(), diff_key)) {
                    filtered.skipped.push(Skipped {
                        path: display_path(&path),
                        cause,
                    });
                }
            }
        }
        // MAGI FIX (S6 loop 15, A2-3): the total is checked after the verdict and its releases,
        // so a file left out near the cap no longer fails the review.
        source.settle()?;
    }
    Ok(filtered)
}

/// A hash of one diff section, to tell a repeated listing from a different one. O(len).
///
/// `DefaultHasher::new()` is a fixed-key hasher: it keys the (path, diff) pairs already left
/// out, so one skip is listed once. It is not collision resistance against an adversary, and
/// nothing rests on it for a kept path: every second listing of one is declared as a `repeated
/// section` whatever the hashes say (`an_identical_second_listing_is_declared`).
fn diff_hash(diff: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    diff.hash(&mut hasher);
    hasher.finish()
}

/// Releases each of `sides` from `source`, except a side a kept file was scanned with
/// (`kept_sides`): a path is not private to one file, so a rename's old path can be a kept
/// file's path. O(sides) expected.
///
/// # Arguments
///
/// * `source` - The review's content source.
/// * `kept_sides` - The sides kept files were scanned with.
/// * `sides` - The sides to forget.
fn release_unkept(
    source: &dyn ContentSource,
    kept_sides: &HashSet<(Face, String)>,
    sides: impl IntoIterator<Item = (Face, String)>,
) {
    for (face, path) in sides {
        if !kept_sides.contains(&(face, path.clone())) {
            source.release(face, &path);
        }
    }
}

/// The sides [`classify`] reads for `file`, each with the path it is read under: the new side
/// under its path and, for a diff, the old side under its `rename from` path or its own, so a
/// renamed file's old side is released under the key it was cached by, never the new path. O(1).
fn sides_of(file: &ReviewFile) -> Vec<(Face, String)> {
    let mut sides = vec![(Face::New, file.path.clone())];
    if let Some(diff) = file.diff.as_deref() {
        let old = renamed_from(diff).unwrap_or_else(|| file.path.clone());
        sides.push((Face::Old, old));
    }
    sides
}

/// Classifies one file (see [`filter_files`] for the order).
fn classify(
    path: &str,
    diff: Option<String>,
    source: &dyn ContentSource,
    is_ignored: &dyn Fn(&str) -> bool,
    explicit: bool,
) -> Result<Verdict, ReviewError> {
    // A renamed or copied file's old side lives under its old path.
    let old_path = diff.as_deref().and_then(renamed_from);
    // MAGI FIX (S6 loop 7): the old path meets the same name rule as the new one: a crafted
    // commit carried a blob stored under `.git/` (or an ignored name) in as a rename's old side.
    if [Some(path), old_path.as_deref()]
        .into_iter()
        .flatten()
        .any(|p| is_never_reviewed(p) || is_ignored(p))
    {
        return Ok(Verdict::Skip(SkipCause::Ignored));
    }
    // MAGI FIX (S6 loop 15, A2-4): the old path too, as for the name rule above: a rename's
    // key-shaped old path was read and cached under that name before the diff scan left it out.
    if let Some(name) = [Some(path), old_path.as_deref()]
        .into_iter()
        .flatten()
        .find_map(find_secret)
    {
        return Ok(secret(path, name));
    }
    if diff.as_deref().is_some_and(is_binary_section) {
        return suppressed_diff(path, diff.as_deref(), source, old_path.as_deref(), explicit);
    }
    let new = match read_side(source.read(path), path, explicit)? {
        Side::Content(bytes) => bytes,
        Side::Skip(cause) => return Ok(Verdict::Skip(cause)),
    };
    // MAGI FIX (S6 loop 14, A2-5): the old side's read failure is logged under the path it was
    // read under (a rename's old path), not the new one, which reads fine.
    let old_side = old_path.as_deref().unwrap_or(path);
    let old = match &diff {
        Some(_) => match read_side(source.read_old(old_side), old_side, explicit)? {
            Side::Content(bytes) => bytes,
            Side::Skip(cause) => return Ok(Verdict::Skip(cause)),
        },
        None => None,
    };
    // MAGI FIX (S6 loop 8): a diff with neither side is a gitlink (a submodule reads as absent
    // on both sides) or a path gone from both revisions; it was kept with no content and no
    // cause. `split_diff` drops gitlink sections before this, so this is the filter's own answer.
    if new.is_none() && old.is_none() && diff.is_some() {
        return Ok(Verdict::Skip(SkipCause::NotRegularFile));
    }
    // MAGI FIX (S6 loop 11): a dangling link reads as absent here too, but it never reaches this
    // branch as "disappeared": every listing of `input` moves enumerated links whose target is
    // gone to `ResolvedInput::dangling_links` before the filter runs (`without_dangling`, pinned
    // by `an_input_left_with_only_dangling_links_is_nothing_to_review_and_names_them`), and they
    // are reported as `SkipCause::NotRegularFile`. Only a file removed after listing lands here.
    if new.is_none() && diff.is_none() {
        if explicit {
            return Err(ReviewError::NotReviewable {
                path: display_path(path),
                cause: MISSING,
            });
        }
        // MAGI FIX (S6 loop 12, A2-m5 / A2-c2): an omission, not a failure: an editor's
        // write-then-rename or a checkout in another window unlinked a listed path for a moment
        // and the whole review died.
        return Ok(Verdict::Skip(SkipCause::Disappeared));
    }
    let sides = [new.as_deref(), old.as_deref()];
    // MAGI FIX (MS3 §6 loop12): what this call checks, in this file. The cached source answers a
    // side holding a multi-line key block RAW (`held_side` -> `holds_key_block`), never masked, so
    // `content_verdict`'s `find_secret` over each side's bytes (defined below in this file) finds
    // the block's header pattern and returns `Skip(secret)`: the only way a key-block file is left
    // out here. Pinned by `a_real_later_stage_reading_a_left_out_key_block_file_fails_closed`,
    // `key_file_listed_twice_skips_once` and
    // `a_pem_key_whose_header_is_outside_the_hunk_still_skips_the_file` (this file).
    if let Some(skip) = content_verdict(path, sides) {
        return Ok(skip);
    }
    let read = SidesRead {
        path,
        old_path: old_side,
        new: new.as_deref(),
        old: old.as_deref(),
    };
    // MAGI FIX (T3.7): a one-line secret in the content no longer leaves the file out; it is
    // masked and the file reviewed (RV22 amended, user decision 2026-11-06). A key-block side is
    // held unmasked by the cache, so `content_verdict` above already left that file out.
    Ok(kept_verdict(&read, source, diff, true))
}

/// The two sides [`classify`] read for a kept file, with the paths they were read under.
struct SidesRead<'a> {
    /// The new side's path.
    path: &'a str,
    /// The old side's path (a rename's old path).
    old_path: &'a str,
    /// The new side, if it exists.
    new: Option<&'a [u8]>,
    /// The old side, if it was read and exists.
    old: Option<&'a [u8]>,
}

/// The verdict on a file whose content may be reviewed, with any secret masked (RV22).
///
/// The sides come masked from the source ([`ContentSource::masked`]; a key-block side never
/// reaches here, [`content_verdict`] leaves it out). When anything was masked, the diff is
/// rebuilt from the masked sides ([`rebuilt_diff`]) and the file declared with the first
/// pattern found; when the diff cannot be rebuilt or the rebuilt one still holds a secret, the
/// raw diff is masked by line instead ([`fallback_diff`], `unrebuilt_diff_is_masked`), with a
/// WARN naming which. RV22's omission classes, all with the `secret` cause: a private-key block, a
/// secret-shaped path and the by-line fallback's refusal (its output still holds a secret, so the
/// file is left out, `a_refused_fallback_is_a_secret_skip`). Declared limit: a rebuild whose body lines differ beyond masking (sides changed after the diff was read;
/// RV21 reads both through one git access) keeps the diff's hunk ranges with the sides' text,
/// said by a run-log WARN only.
///
/// The diff is rebuilt even when it shows no secret of its own: a hunk shows only some lines, and
/// a pattern may need one it leaves out (a YAML value under its key). A file with no secret keeps
/// the very string git produced. O(diff + sides).
///
/// # Arguments
///
/// * `read` - The sides the filter read.
/// * `source` - The review's content source.
/// * `diff` - The diff the file would be reviewed through, if any.
/// * `with_old` - Whether its old side stays held.
///
/// # Returns
///
/// [`Verdict::Keep`], or a `secret` skip.
fn kept_verdict(
    read: &SidesRead<'_>,
    source: &dyn ContentSource,
    diff: Option<String>,
    with_old: bool,
) -> Verdict {
    let new_mask = read.new.and(source.masked(Face::New, read.path));
    let old_mask = read.old.and(source.masked(Face::Old, read.old_path));
    // MAGI FIX (MS3 §6 loop16, R6a1): `side_mask` is a mask that happened; a secret found only in
    // the diff (`pattern`) makes the diff rebuild, but declares the file masked only when the
    // by-line fallback redacted something.
    let side_mask = new_mask.or(old_mask).map(|m| m.pattern);
    let pattern = side_mask.or_else(|| diff.as_deref().and_then(find_secret));
    let Some(pattern) = pattern else {
        return Verdict::Keep(Kept {
            diff,
            with_old,
            masked: None,
        });
    };
    let mut declared = side_mask;
    let diff = match diff {
        Some(d) => match rebuilt_diff(&d, read.new, read.old) {
            Some((r, diverged)) if !leaks_secret(&r) => {
                if diverged > 0 {
                    let shown = display_path(read.path);
                    // MAGI FIX (MS3 §6 loop17, R6a1): the count has two causes the code cannot
                    // tell apart. A commit or a two-revision diff reads its sides and its diff
                    // through one git access, so a count there is a secret spanning a line
                    // break, masked differently on the whole side than on one diff line; a
                    // worktree review reads the diff from `git diff` and the sides from the
                    // file system, so a file saved in between is pure drift.
                    tracing::event!(
                        target: LOG_TARGET,
                        tracing::Level::WARN,
                        path = shown.as_str(),
                        lines = diverged,
                        "{}",
                        DIVERGENCE_WARNING
                    );
                }
                Some(r)
            }
            rebuilt => {
                // MAGI FIX (MS3 §6 loop4): the fallback was silent beyond the INFO of every
                // `secret` skip; a WARN makes a parser drift visible in the run log.
                let shown = display_path(read.path);
                let why = if rebuilt.is_none() {
                    "the diff disagrees with the file's sides"
                } else {
                    "the rebuilt diff still holds a secret"
                };
                tracing::event!(
                    target: LOG_TARGET,
                    tracing::Level::WARN,
                    path = shown.as_str(),
                    cause = why,
                    pattern = pattern,
                    "a masked file's diff could not be rebuilt from its sides; its own lines are masked"
                );
                // MAGI FIX (MS3 §6 loop11): it was a third omission class (RV22 allows two); the
                // raw diff is masked instead, by line, and the file reviewed.
                // MAGI FIX (MS3 §6 loop16, R6a1): the None arm below is reachable, with its cost
                // declared: a secret whose lines each sit verbatim in a clean side (an AWS key and
                // its value as adjacent context) survives by-line masking, the re-scan refuses it
                // and the whole file is left out as `secret`
                // (`a_refused_fallback_is_a_secret_skip`, `fallback_refusal_is_reachable`).
                match fallback_diff(&d, read.new, read.old) {
                    Some(masked) => {
                        declared = Some(declared.unwrap_or(pattern));
                        Some(masked)
                    }
                    None => return secret(read.path, pattern),
                }
            }
        },
        None => None,
    };
    if let Some(declared) = declared {
        log_masked(read.path, declared);
    }
    Verdict::Keep(Kept {
        diff,
        with_old,
        masked: declared,
    })
}

/// The skip the content of a file calls for, or `None` when it may be reviewed: a NUL in a side
/// gives `binary`, a side that is not UTF-8 `not utf-8`, a secret still in a side `secret`, in
/// that order. A side still holds a secret here when it holds a multi-line private-key block,
/// which the source never masks (REQ-RV22's exception), or when the source does not mask at
/// all ([`ContentSource::masked`]): either way the file is left out, as before masking existed.
/// O(bytes).
///
/// # Arguments
///
/// * `path` - The file's path, for the `secret` log entry.
/// * `sides` - The new and old sides that exist.
///
/// # Returns
///
/// The skip, or `None`.
// MAGI FIX (S6 loop 4): both sides, not the new one alone: a text file whose old side was
// binary or Latin-1 carried old-side bytes into the review.
// MAGI FIX (S6 loop 7): the whole side, not git's first 8 000 bytes: a text prefix hid a
// binary payload, which is valid UTF-8 (NUL is a code point). One pass over bytes already
// read and capped (MAX_REVIEW_FILE_BYTES), so the cost stays O(file).
fn content_verdict(path: &str, sides: [Option<&[u8]>; 2]) -> Option<Verdict> {
    if sides.iter().flatten().any(|side| side.contains(&0)) {
        return Some(Verdict::Skip(SkipCause::Binary));
    }
    // MAGI FIX (S6 loop 16, A2-5): each side is decoded once and scanned as the `&str` it is;
    // a lossy decode of text already validated was a second UTF-8 pass per side.
    let mut faces: Vec<&str> = Vec::with_capacity(sides.len() + 1);
    for side in sides.into_iter().flatten() {
        match std::str::from_utf8(side) {
            Ok(text) => faces.push(text),
            Err(_) => return Some(Verdict::Skip(SkipCause::NotUtf8)),
        }
    }
    // MAGI FIX (MS3 §6 loop13): the omission set must cover what the cache never masks. A side
    // `holds_key_block` reports (a header `find_secret` misses, such as two joined literals) is
    // held raw by `CachedSource`, so the file is left out here, never kept with its body.
    // MAGI FIX (MS3 §6 loop17, R6a2/R6a1): the verdict asks the one egress predicate
    // (`leaks_secret`), the question `held_side` answered when it held the side raw; a pattern
    // name is read from `find_secret` first, then from what only the stronger terms see.
    let named = faces.iter().copied().find_map(find_secret);
    match named {
        Some(name) => Some(secret(path, name)),
        None => faces.iter().find(|face| leaks_secret(face)).map(|face| {
            secret(
                path,
                if holds_key_block(face) {
                    KEY_BLOCK_PATTERN
                } else {
                    SPLIT_SECRET_PATTERN
                },
            )
        }),
    }
}

/// The run-log text for a masked file whose diff lines differ from its masked sides beyond a
/// mask: the code cannot tell which of the two causes it is, so it names both.
const DIVERGENCE_WARNING: &str = "diff lines differ from the file's masked sides: either the \
    file changed after the diff was read (drift), or a secret spans a line break and was masked \
    differently on the whole side than on one diff line; the file is reviewed from its sides";

/// The pattern name a file is left out under when a hidden or control character splits a secret
/// the contiguous scan cannot name.
const SPLIT_SECRET_PATTERN: &str = "split-secret";

/// The pattern name a file is left out under when a side holds a key block no pattern names.
const KEY_BLOCK_PATTERN: &str = "pem-private-key";

/// The verdict on a file whose diff section git printed as binary ([`is_binary_section`]).
///
/// The marker is a hint, never the answer: git prints it for any file the `diff` attribute
/// turns off (`-diff`, `binary`), and the reviewed tree writes its own `.gitattributes`, so a
/// change could mark itself out of the review. Both sides are read and judged by their bytes
/// ([`content_verdict`]); a text file passes and is reviewed whole, from its new side, since
/// git printed no hunks for it (one INFO entry says so). A side that fails to read is mapped
/// as for any file ([`read_side`]): it is left out under its own cause (`unreadable`, `too
/// large`, `outside workspace`, …), and an error that is not a skip (the review's total cap)
/// fails the review. A deletion (no new side) whose old side is text is kept with the diff git
/// would have printed ([`deletion_diff`]), so its deleted lines are reviewed like any deletion's;
/// it stays `binary`, the label the section carries, only when it has no old side either.
///
/// # Arguments
///
/// * `path` - The file's path.
/// * `section` - Its diff section, git's binary marker in it.
/// * `source` - The review's content source.
/// * `old_path` - Its `rename from` path, if any.
/// * `explicit` - Whether the file was requested by name.
///
/// # Returns
///
/// [`Verdict::Keep`] or the skip.
///
/// # Errors
///
/// Every read error that [`read_side`] does not map to a skip.
// MAGI FIX (S6 loop 13, A2-m3 / A2-c1, security): `evil.rs -diff` in the reviewed commit left
// `evil.rs` out as `binary` without a byte of it read.
// MAGI FIX (S6 loop 14, A2-4): a failed read was `binary`, and the total cap's `InputTooLarge`
// an omission that could end as `NothingToReview`; each now goes through `read_side`.
fn suppressed_diff(
    path: &str,
    section: Option<&str>,
    source: &dyn ContentSource,
    old_path: Option<&str>,
    explicit: bool,
) -> Result<Verdict, ReviewError> {
    let new = match read_side(source.read(path), path, explicit)? {
        Side::Content(bytes) => bytes,
        Side::Skip(cause) => return Ok(Verdict::Skip(cause)),
    };
    let old_side = old_path.unwrap_or(path);
    let old = match read_side(source.read_old(old_side), old_side, explicit)? {
        Side::Content(bytes) => bytes,
        Side::Skip(cause) => return Ok(Verdict::Skip(cause)),
    };
    let sides = [new.as_deref(), old.as_deref()];
    if let Some(skip) = content_verdict(path, sides) {
        return Ok(skip);
    }
    let read = SidesRead {
        path,
        old_path: old_side,
        new: new.as_deref(),
        old: old.as_deref(),
    };
    // MAGI FIX (review residual A2-b): a deleted text file marked `-diff` was left out as
    // `binary`, a false cause; its deletion is rebuilt from the old side, already scanned above.
    if new.is_none() {
        return Ok(match (old.as_deref().map(std::str::from_utf8), section) {
            (Some(Ok(text)), Some(section)) => {
                kept_verdict(&read, source, Some(deletion_diff(section, text)), true)
            }
            _ => Verdict::Skip(SkipCause::Binary),
        });
    }
    let shown = display_path(path);
    tracing::event!(
        target: LOG_TARGET,
        tracing::Level::INFO,
        path = shown.as_str(),
        "git printed a text file's diff as binary; it is reviewed whole"
    );
    Ok(kept_verdict(&read, source, None, false))
}

/// The unified diff of the deletion of a file whose old side is `old`, built on the header lines
/// of git's `section` (those before its binary marker): one hunk `@@ -1,N +0,0 @@` with every
/// old line as `-`, and git's `\ No newline at end of file` after a last line without one. An
/// empty `old` gives the header alone, with no hunk, as git prints an empty file's deletion.
/// O(old + section).
///
/// The text is only what `old` and `section` hold, both scanned for secrets by
/// [`content_verdict`]; the batches scan it again as any diff's on the way out.
fn deletion_diff(section: &str, old: &str) -> String {
    let mut diff = String::with_capacity(section.len() + old.len() * 2);
    for line in section
        .lines()
        .take_while(|l| !l.starts_with(HUNK_START) && !is_binary_marker_line(l))
    {
        diff.push_str(line);
        diff.push('\n');
    }
    let lines: Vec<&str> = old.split_inclusive('\n').collect();
    if lines.is_empty() {
        return diff;
    }
    diff.push_str(&format!(
        "{HUNK_START} -1,{} +0,0 {HUNK_START}\n",
        lines.len()
    ));
    for line in lines {
        diff.push('-');
        diff.push_str(line);
        if !line.ends_with('\n') {
            diff.push_str(NO_NEWLINE_AT_EOF);
        }
    }
    diff
}

/// The pattern a `secret` skip logs when the cache refused the read: the file was left out
/// earlier, and the refusal does not carry which pattern did it.
const REFUSED_BY_CACHE_PATTERN: &str = "left out earlier";

/// A `secret` skip, logged with the shown path and the pattern's name (never the value).
fn secret(path: &str, pattern: &'static str) -> Verdict {
    let shown = display_path(path);
    tracing::event!(
        target: LOG_TARGET,
        tracing::Level::INFO,
        path = shown.as_str(),
        pattern = pattern,
        "a file was left out of the review because it holds a secret"
    );
    Verdict::Skip(SkipCause::Secret)
}

/// Logs, at INFO with the shown path and the pattern's name (never the value), that a file is
/// reviewed with its secrets masked.
fn log_masked(path: &str, pattern: &'static str) {
    let shown = display_path(path);
    tracing::event!(
        target: LOG_TARGET,
        tracing::Level::INFO,
        path = shown.as_str(),
        pattern = pattern,
        "a file holds a secret; it is reviewed with the secret masked"
    );
}

/// The declaration of a file reviewed with its secrets masked, its path as it may leave the
/// process.
fn masked_file(path: &str, pattern: &'static str) -> Masked {
    Masked {
        path: display_path(path),
        cause: SkipCause::Secret,
        pattern,
    }
}

/// What reading one side gave.
enum Side {
    /// The content, or `None` when the side has no such file.
    Content(Option<Vec<u8>>),
    /// A refusal of the source that is a skip.
    Skip(SkipCause),
}

/// Maps a read to its content or to the skip its refusal stands for.
///
/// An I/O failure of a listed file (`explicit` false) is the skip [`SkipCause::Unreadable`],
/// logged at WARN with the shown path and the scrubbed error; for a file named explicitly it
/// stays an error. Nothing of the file is sent either way. A side the cache refuses for a
/// secret ([`ReviewError::EgressSecret`]: one already left out, such as a path listed twice) is
/// the skip `secret`. Past the filter the cache serves masked content and refuses only raw-secret
/// sides: the containing units and the lexical index read the kept files, where a refusal ends
/// the review (failing closed is RV22's tie-break); a definition or caller read
/// declares its item unreadable instead (`egress_secret_read_is_noted` in `rag`)
/// (`key_file_listed_twice_skips_once`,
/// `after_the_filter_no_stage_is_served_a_key_block_it_never_classified`).
///
/// # Arguments
///
/// * `read` - The source's answer.
/// * `path` - The path the side was read under (a rename's old path for its old side), for the
///   log entry.
/// * `explicit` - Whether the file was requested by name.
///
/// # Errors
///
/// Every error of `read` that is not a skip.
// MAGI FIX (S6 loop 12, A2-c2): one file held open by a sync client or an indexer (a Windows
// sharing violation) failed the whole review after its preparation.
fn read_side(
    read: Result<Option<Vec<u8>>, ReviewError>,
    path: &str,
    explicit: bool,
) -> Result<Side, ReviewError> {
    match read {
        Err(error @ ReviewError::Io(_)) if !explicit => {
            let shown = display_path(path);
            let error = error.to_string();
            tracing::event!(
                target: LOG_TARGET,
                tracing::Level::WARN,
                path = shown.as_str(),
                error = error.as_str(),
                "a listed file could not be read and was left out of the review"
            );
            Ok(Side::Skip(SkipCause::Unreadable))
        }
        Ok(bytes) => Ok(Side::Content(bytes)),
        Err(ReviewError::OutsideWorkspace(_)) => Ok(Side::Skip(SkipCause::OutsideWorkspace)),
        Err(ReviewError::TooLarge { .. }) => Ok(Side::Skip(SkipCause::TooLarge)),
        Err(ReviewError::Protected(_)) => Ok(Side::Skip(SkipCause::Ignored)),
        Err(ReviewError::NotRegularFile(_)) => Ok(Side::Skip(SkipCause::NotRegularFile)),
        // MAGI FIX (MS3 §6 loop5): the cache refuses a side already left out for a secret (a
        // path listed twice); inside the filter that is the same skip, not a failed review.
        // MAGI FIX (MS3 §6 loop15): logged like every other `secret` skip; the cache's refusal
        // does not carry the pattern that first left the file out, so the entry names why.
        Err(ReviewError::EgressSecret { .. }) => {
            let _ = secret(path, REFUSED_BY_CACHE_PATTERN);
            Ok(Side::Skip(SkipCause::Secret))
        }
        // The source's one `NotReviewable`, keyed on the constant it is built from.
        Err(ReviewError::NotReviewable {
            cause: HARD_LINK_CAUSE,
            ..
        }) => Ok(Side::Skip(SkipCause::HardLink)),
        Err(other) => Err(other),
    }
}

/// The paths `.gitignore` excludes that a review can still reach.
///
/// Paths are relative to `workspace` with `/` separators, the same space as
/// [`ReviewFile::path`]: the production git runs with `-C <workspace>`, `ls-files` prints paths
/// relative to its cwd (no `--full-name`), and every diff carries `--relative`. A review of S6
/// loop 13 (A2-m1) read `ls-files` as repository-root relative; measured with git 2.56 and
/// pinned, with the workspace in a subdirectory of the repository, by
/// `an_uncommitted_review_in_a_sub_workspace_names_ignored_files_like_its_diff` and
/// `a_tracked_ignored_file_in_a_subdirectory_is_named_like_the_enumeration_names_it`
/// (`src/system/review_git.rs`); `--full-name`, the proposed fix, turns the first red. History inputs
/// (`Commit`, `Range`, `Branches`) ignore nothing and call no git: `.gitignore` is a worktree
/// rule. Otherwise one `rev-parse --is-inside-work-tree`, then one `ls-files` call: for `File`,
/// tracked and untracked files a rule excludes (a file that does not exist is not listed, so it
/// stays `missing`); for `Directory` and `Uncommitted` (`pathspec` `.`), the tracked ones a rule
/// excludes, which the enumeration lists (untracked ignored files are never enumerated). For
/// `Uncommitted` the set also covers the paths of `HEAD`'s tree (`--with-tree`, after one more
/// `rev-parse`; none in a repository without commits), where its diff reads the old side: a
/// tracked ignored file staged for deletion or renamed away is no longer in the index. `HEAD` is
/// read again here, not the SHA the input pinned: a commit the operator makes during the
/// preparation makes the set follow the newer tree (declared, not closed).
/// Outside any repository the set is empty.
///
/// `pathspec` follows a `--`, so a name that begins with `-` (`-x.rs`, `--stdin`) is a path and
/// never an option of `ls-files`; pinned by
/// `a_file_named_like_a_git_option_is_a_pathspec_not_an_option` (`src/system/review_git.rs`).
///
/// `pathspec` is a literal path, never a glob: the production git sets
/// `GIT_LITERAL_PATHSPECS=1` on every call, so a name holding `*`, `?` or `[` matches only
/// itself. Measured with git 2.56 (reviews of S6 loops 5 and 6 raised it twice): a tracked or
/// untracked ignored file named `a[1].rs` is listed with or without the variable, because git
/// tries the exact name before the glob; without it the glob also names `a1.rs`. Pinned by
/// `a_tracked_ignored_file_named_with_glob_characters_keeps_its_ignore_verdict` and
/// `an_ignored_file_with_glob_characters_in_its_name_keeps_its_ignore_verdict`
/// (`src/system/review_git.rs`).
///
/// # Arguments
///
/// * `input` - What the caller asked to review.
/// * `pathspec` - The workspace-relative path the query is limited to.
/// * `workspace` - Where the `.git` search starts when `rev-parse` fails.
/// * `git` - The review's git.
///
/// # Errors
///
/// [`ReviewError::Git`] when git fails while a `.git` exists in `workspace` or an ancestor (an
/// empty set there would send an ignored file), and any error of `git`.
///
/// O(output bytes) time.
pub fn ignored_paths(
    input: &ReviewInput,
    pathspec: &str,
    workspace: &Path,
    git: &dyn GitCommand,
) -> Result<HashSet<String>, ReviewError> {
    let query = match input {
        ReviewInput::Commit(_) | ReviewInput::Range { .. } | ReviewInput::Branches { .. } => {
            return Ok(HashSet::new())
        }
        ReviewInput::File(_) => LS_FILES_IGNORED_ANY,
        ReviewInput::Directory(_) | ReviewInput::Uncommitted => LS_FILES_IGNORED_TRACKED,
    };
    if !in_work_tree(git, workspace, workspace)? {
        return Ok(HashSet::new());
    }
    let mut args = query.to_vec();
    // MAGI FIX (S6 loop 16, A2-2, security): the diff's old side comes from HEAD, the index
    // alone forgot a tracked ignored file staged for deletion or renamed away, and its HEAD
    // content was kept. `--with-tree` lists HEAD's paths as well (measured with git 2.56).
    let with_tree = match input {
        ReviewInput::Uncommitted => git
            .query(HEAD_COMMIT_QUERY)?
            .map(|sha| format!("{WITH_TREE_OPTION}{}", sha.trim())),
        _ => None,
    };
    if let Some(option) = &with_tree {
        args.push(option);
    }
    // MAGI FIX (S6 loop 10): D2-m2 refuted; the `--` keeps a leading-dash name a path.
    args.extend_from_slice(&["--", pathspec]);
    // A name that is not UTF-8 is never enumerated as a file to review (the input sets it
    // aside), so it needs no entry here.
    let (names, _not_utf8) = split_nul(&git.run_bytes(&args)?);
    Ok(names.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::review::diffmask::CONTEXT_VISITS;
    use crate::review::source::CachedSource;
    use crate::review::test_support::MapSource;

    /// The declaration of a file reviewed with its secrets masked.
    fn masked(path: &str, pattern: &'static str) -> Masked {
        Masked {
            path: path.into(),
            cause: SkipCause::Secret,
            pattern,
        }
    }

    #[test]
    fn a_secret_in_a_whole_file_is_reviewed_masked_and_an_explicit_one_too() {
        // [T3.7] RV22 amended: the file is reviewed with the secret masked, never left out, and
        // a file named explicitly is no error. The cache (production's single access) masks.
        let text = "const K: &str = \"ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\";\n"; // allow-secret-scan
        let inner = MapSource::new(&[("config.rs", text), ("a.rs", "fn f() {}\n")], &[]);
        let src = CachedSource::new(&inner);
        let files = vec![
            ReviewFile {
                path: "config.rs".into(),
                diff: None,
            },
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
        ];
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
            .expect("both kept");
        assert_eq!(f.skipped, vec![]);
        assert_eq!(f.masked, vec![masked("config.rs", "github-token")]);
        let kept: Vec<&str> = f.kept.iter().map(|k| k.path.as_str()).collect();
        assert_eq!(kept, ["config.rs", "a.rs"]);
        let one = vec![ReviewFile {
            path: "config.rs".into(),
            diff: None,
        }];
        let f = filter_files(one, &src, &LanguageRegistry::new(), true, &|_| false)
            .expect("an explicit file with a secret is reviewed");
        assert_eq!(f.masked, vec![masked("config.rs", "github-token")]);
    }

    #[test]
    fn after_the_filter_no_stage_is_served_a_key_block_it_never_classified() {
        // [MAGI S6 MS3 loop5, R6a Melchior] A later stage reading a path this filter never saw
        // (beyond the kept files) got a raw key block: the filter now seals the cache.
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        let keyed = format!("{begin}\nMIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8w\n");
        let inner = MapSource::new(
            &[("a.rs", "fn f() {}\n"), ("never.pem", keyed.as_str())],
            &[],
        );
        let src = CachedSource::new(&inner);
        let files = vec![ReviewFile {
            path: "a.rs".into(),
            diff: None,
        }];
        filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false).expect("kept");
        assert!(matches!(
            src.read("never.pem"),
            Err(ReviewError::EgressSecret { .. })
        ));
        assert!(src.read("a.rs").expect("kept file").is_some());
    }

    #[test]
    fn a_url_credential_file_is_reviewed_masked_and_its_mask_never_reads_as_a_secret() {
        // [MAGI S6 MS3 loop5, R6a Melchior] Masking holds only while a mask never matches a
        // pattern again: a mask that re-matched would turn masking back into a `secret` skip.
        // The placeholder lands inside the URL for this pattern, the one whose omission
        // motivated RV22's amendment (PM-S-14).
        let text = "let url = \"postgres://svc:pa55word@db.internal/app\";\n";
        let inner = MapSource::new(&[("db.rs", text)], &[]);
        let src = CachedSource::new(&inner);
        let files = vec![ReviewFile {
            path: "db.rs".into(),
            diff: None,
        }];
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
            .expect("kept masked");
        assert_eq!(f.masked, vec![masked("db.rs", "url-credential")]);
        let held =
            String::from_utf8(src.read("db.rs").expect("read").expect("present")).expect("utf-8");
        assert!(!held.contains("pa55word"), "{held}");
        assert_eq!(find_secret(&held), None, "{held}");
    }

    #[test]
    fn fallback_secret_is_refused() {
        // [MAGI S6 MS3 loop13, R6a Melchior] The by-line fallback was the one egress text never
        // re-scanned: a secret that exists only across adjacent diff lines (each line verbatim
        // on a clean side) survived it. `fallback_diff` scans its own output.
        let value = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"; // allow-secret-scan
        let new = format!("aws_secret_key =\nfn f() {{}}\n\"{value}\"\n");
        let diff = format!("@@ -1,2 +1,2 @@\n aws_secret_key =\n \"{value}\"\n");
        assert_eq!(find_secret(&new), None, "premise: each side is clean");
        assert_eq!(
            find_secret(&diff),
            Some("aws-secret-key"),
            "premise: the diff is not"
        );
        // Reachability: here by-line masking redacts both lines (no clean side shows them), so
        // the input is answered; `fallback_refusal_is_reachable` has the input the re-scan
        // refuses. The contract pinned is that whatever `fallback_diff` returns holds no secret.
        let masked = fallback_diff(&diff, Some(new.as_bytes()), None).expect("redacted by line");
        assert_eq!(find_secret(&masked), None, "{masked}");
        assert!(!masked.contains(value), "{masked}");
        let clean = "@@ -0,0 +1,1 @@\n+fn f() {}\n";
        assert!(fallback_diff(clean, Some(b"fn f() {}\n"), None).is_some());
    }
    #[test]
    fn the_content_verdict_omits_every_text_the_one_egress_predicate_withholds() {
        // [MAGI S6 MS3 loop17, R6a2 Melchior/R6a1] the filter's verdict asked `find_secret` (and
        // `holds_key_block`), not the module's egress predicate: a hidden-character split token
        // was held raw by the cache and judged clean here.
        for text in crate::review::secrets::withheld_corpus() {
            assert!(crate::review::secrets::leaks_secret(&text), "{text:?}");
            for sides in [[Some(text.as_bytes()), None], [None, Some(text.as_bytes())]] {
                assert!(
                    matches!(
                        content_verdict("a.rs", sides),
                        Some(Verdict::Skip(SkipCause::Secret))
                    ),
                    "{text:?}"
                );
            }
        }
    }

    #[test]
    fn joined_key_header_is_omitted() {
        // [MAGI S6 MS3 loop13, R6a Caspar, critical] `holds_key_block` (the hold decision)
        // saw a header joined across two literals that `find_secret` (the omission decision)
        // missed: the file was kept and its body went out raw. The filter now omits any file
        // whose served side holds a key block, so the omission set covers the never-masked set.
        let header = concat!(
            "\"",
            "-----",
            "BEGIN ",
            "\" \"",
            "RSA PRIVATE KEY-----",
            "\""
        );
        let text =
            format!("const H: &str = concat!({header});\nMIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8w\n");
        assert_eq!(
            find_secret(&text),
            None,
            "premise: the joined header matches no pattern"
        );
        let inner = MapSource::new(&[("a.rs", "fn f() {}\n"), ("h.rs", text.as_str())], &[]);
        let src = CachedSource::new(&inner);
        let file = |p: &str| ReviewFile {
            path: p.into(),
            diff: None,
        };
        let f = filter_files(
            vec![file("h.rs"), file("a.rs")],
            &src,
            &LanguageRegistry::new(),
            false,
            &|_| false,
        )
        .expect("a.rs kept");
        let skipped = Skipped {
            path: "h.rs".into(),
            cause: SkipCause::Secret,
        };
        assert_eq!(f.skipped, vec![skipped]);
        assert_eq!(f.kept.len(), 1);
    }
    #[test]
    fn key_file_listed_twice_skips_once() {
        // [MAGI S6 MS3 loop5, R6a Caspar] The second read of the released side was refused by
        // the cache, and that refusal ended the whole review; inside the filter it is the skip
        // the first read already chose.
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        let keyed = format!("{begin}\nMIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8w\n");
        let inner = MapSource::new(&[("a.rs", "fn f() {}\n"), ("k.pem", keyed.as_str())], &[]);
        let src = CachedSource::new(&inner);
        let file = |p: &str| ReviewFile {
            path: p.into(),
            diff: None,
        };
        let f = filter_files(
            vec![file("k.pem"), file("a.rs"), file("k.pem")],
            &src,
            &LanguageRegistry::new(),
            false,
            &|_| false,
        )
        .expect("a.rs kept");
        let secret = Skipped {
            path: "k.pem".into(),
            cause: SkipCause::Secret,
        };
        // MAGI FIX (MS3 §6 loop13): a repeated listing is declared once, kept or skipped.
        assert_eq!(f.skipped, vec![secret]);
        assert!(matches!(
            filter_files(
                vec![file("k.pem")],
                &src,
                &LanguageRegistry::new(),
                true,
                &|_| false
            ),
            Err(ReviewError::NotReviewable {
                cause: "secret",
                ..
            })
        ));
    }

    #[test]
    fn a_refused_fallback_is_a_secret_skip() {
        // [MAGI S6 MS3 loop14, R6a1 Caspar] The fallback's refusal is not a third omission
        // class: it is RV22's own `secret` cause, and this input really reaches it (a two-line
        // secret whose lines both sit verbatim in a clean old and new side).
        let value = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"; // allow-secret-scan
        let side = format!("aws_secret_key =\nfn f() {{}}\n\"{value}\"\n");
        let diff = format!(
            "@@ -1,5 +1,5 @@\n aws_secret_key =\n \"{value}\"\n fn f() {{}}\n fn f() {{}}\n fn f() {{}}\n"
        );
        let inner = MapSource::new(
            &[("a.rs", side.as_str()), ("b.rs", "fn b() {}\n")],
            &[("a.rs", side.as_str())],
        );
        let src = CachedSource::new(&inner);
        let files = vec![
            ReviewFile {
                path: "a.rs".into(),
                diff: Some(diff),
            },
            ReviewFile {
                path: "b.rs".into(),
                diff: None,
            },
        ];
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false);
        let f = f.expect("a skip, not an error");
        assert_eq!(f.kept.len(), 1, "{f:?}");
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: "a.rs".into(),
                cause: SkipCause::Secret
            }]
        );
    }

    #[test]
    fn a_repeated_path_is_never_kept_and_skipped() {
        // [MAGI S6 MS3 loop14, R6a1 Caspar] A second listing of a kept path, with a diff that
        // would be left out as `secret`, was reported as skipped while the path stayed kept.
        let value = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"; // allow-secret-scan
        let side = format!("aws_secret_key =\nfn f() {{}}\n\"{value}\"\n");
        let diff = format!(
            "@@ -1,5 +1,5 @@\n aws_secret_key =\n \"{value}\"\n fn f() {{}}\n fn f() {{}}\n fn f() {{}}\n"
        );
        let inner = MapSource::new(&[("a.rs", side.as_str())], &[("a.rs", side.as_str())]);
        let src = CachedSource::new(&inner);
        let file = |diff: Option<String>| ReviewFile {
            path: "a.rs".into(),
            diff,
        };
        let f = filter_files(
            vec![file(None), file(Some(diff))],
            &src,
            &LanguageRegistry::new(),
            false,
            &|_| false,
        )
        .expect("kept once");
        assert_eq!(f.kept.len(), 1, "{f:?}");
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: "a.rs".into(),
                cause: SkipCause::DuplicateSection
            }]
        );
    }

    #[test]
    fn a_file_is_declared_masked_only_when_a_mask_happened() {
        // [MAGI S6 MS3 loop16, R6a1 Melchior] `kept_verdict` took the pattern from
        // `find_secret(diff)` when neither side was masked: a diff quoting a credential that no
        // side holds (a stale line the rebuild replaces with the clean side's text) put the file
        // in `Filtered::masked` though no byte of it was masked.
        let token = concat!("ghp_", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"); // allow-secret-scan
        let diff = format!("@@ -1 +1 @@\n-fn a() {{}}\n+let t = \"{token}\";\n");
        let inner = MapSource::new(&[("a.rs", "fn b() {}\n")], &[("a.rs", "fn a() {}\n")]);
        let src = CachedSource::new(&inner);
        let files = vec![ReviewFile {
            path: "a.rs".into(),
            diff: Some(diff),
        }];
        let f =
            filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false).expect("kept");
        assert_eq!(f.kept.len(), 1, "{f:?}");
        let served = f.kept[0].diff.as_deref().unwrap_or_default();
        assert!(!served.contains(token), "{served}");
        assert!(f.masked.is_empty(), "nothing was masked: {:?}", f.masked);
    }

    #[test]
    fn a_path_left_out_as_secret_is_never_kept_by_a_later_listing() {
        // [MAGI S6 MS3 loop16, R6a1 Melchior] The repeated-path guard ran one way: a kept path
        // turned a later skip into a repeat, but a path left out as `secret` could be kept by a
        // later listing, so the report held it both skipped and kept.
        let value = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"; // allow-secret-scan
        let side = format!("aws_secret_key =\nfn f() {{}}\n\"{value}\"\n");
        let diff = format!(
            "@@ -1,5 +1,5 @@\n aws_secret_key =\n \"{value}\"\n fn f() {{}}\n fn f() {{}}\n fn f() {{}}\n"
        );
        let inner = MapSource::new(&[("a.rs", side.as_str())], &[("a.rs", side.as_str())]);
        let src = CachedSource::new(&inner);
        let file = |diff: Option<String>| ReviewFile {
            path: "a.rs".into(),
            diff,
        };
        let outcome = filter_files(
            vec![file(Some(diff)), file(None)],
            &src,
            &LanguageRegistry::new(),
            false,
            &|_| false,
        );
        match outcome {
            Err(ReviewError::NothingToReview { skipped }) => {
                assert_eq!(skipped.len(), 2, "{skipped:?}");
                assert_eq!(skipped[0].cause, SkipCause::Secret);
                assert_eq!(skipped[1].cause, SkipCause::DuplicateSection);
            }
            other => panic!("a left-out path was kept again: {other:?}"),
        }
    }

    #[test]
    fn a_source_that_does_not_mask_still_never_lets_a_secret_through() {
        // [T3.7] the fallback: a source with no mask (no production path builds one) hands the
        // filter raw content, so a file that holds a secret is left out as before, and an
        // explicit one is an error.
        let text = "const K: &str = \"ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\";\n"; // allow-secret-scan
        let src = MapSource::new(&[("config.rs", text), ("a.rs", "fn f() {}\n")], &[]);
        let files = vec![
            ReviewFile {
                path: "config.rs".into(),
                diff: None,
            },
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
        ];
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
            .expect("a.rs kept");
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: "config.rs".into(),
                cause: SkipCause::Secret
            }]
        );
        let one = vec![ReviewFile {
            path: "config.rs".into(),
            diff: None,
        }];
        assert!(matches!(
            filter_files(one, &src, &LanguageRegistry::new(), true, &|_| false),
            Err(ReviewError::NotReviewable {
                cause: "secret",
                ..
            })
        ));
    }

    #[test]
    fn a_path_shaped_like_a_key_is_never_sent() {
        // [CP2 MS2 S1b loop 13] RV22: every batch header carries the path, so the path is scanned like the content.
        // [CP2 MS2 S1b1 loop 16] and the key never leaves through the report, the log or an error either, whatever the cause.
        let name = concat!("gh", "p_", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", ".rs");
        let shown = "<path redacted: github-token>";
        let src = MapSource::new(&[(name, "fn f() {}\n"), ("a.rs", "fn f() {}\n")], &[]);
        let both = || {
            vec![
                ReviewFile {
                    path: name.into(),
                    diff: None,
                },
                ReviewFile {
                    path: "a.rs".into(),
                    diff: None,
                },
            ]
        };
        let log = crate::logging::testutil::capture(|| {
            let f = filter_files(both(), &src, &LanguageRegistry::new(), false, &|_| false)
                .expect("a.rs kept");
            assert_eq!(
                f.skipped,
                vec![Skipped {
                    path: shown.into(),
                    cause: SkipCause::Secret
                }]
            );
        });
        assert!(!log.contains("AAAAAAAAAAAAAAAA"), "{log}");
        let f = filter_files(both(), &src, &LanguageRegistry::new(), false, &|p| {
            p == name
        })
        .expect("a.rs kept");
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: shown.into(),
                cause: SkipCause::Ignored
            }]
        );
        let one = vec![ReviewFile {
            path: name.into(),
            diff: None,
        }];
        match filter_files(one, &src, &LanguageRegistry::new(), true, &|_| false) {
            Err(ReviewError::NotReviewable {
                path,
                cause: "secret",
            }) => assert_eq!(path, shown),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_secret_only_in_the_old_side_is_masked_and_the_file_reviewed() {
        // The diff and the new side are clean; only the revision before the change holds the key.
        // [CP2 MS2 S1b loop 15] a kept sibling makes the skip and its `secret` cause observable (RV-S30, second clause).
        // [T3.7] RV22 amended: masked, not omitted; the diff is unchanged, it quotes no secret.
        let diff = "@@ -2,1 +1,1 @@\n-fn g() {}\n+fn h() {}\n";
        let old = "const K: &str = \"AKIAABCDEFGHIJKLMNOP\";\nfn g() {}\n"; // allow-secret-scan
        let inner = MapSource::new(
            &[("a.rs", "fn h() {}\n"), ("b.rs", "fn b() {}\n")],
            &[("a.rs", old)],
        );
        let src = CachedSource::new(&inner);
        let files = vec![
            ReviewFile {
                path: "a.rs".into(),
                diff: Some(diff.into()),
            },
            ReviewFile {
                path: "b.rs".into(),
                diff: None,
            },
        ];
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
            .expect("both kept");
        assert_eq!(f.skipped, vec![]);
        assert_eq!(f.masked, vec![masked("a.rs", "aws-access-key")]);
        assert_eq!(
            f.kept.first().and_then(|k| k.diff.as_deref()),
            Some(diff),
            "{f:?}"
        );
        let old = String::from_utf8(src.read_old("a.rs").expect("read").expect("old side"))
            .expect("utf-8");
        assert!(!old.contains("AKIA"), "{old}");
    }

    #[test]
    fn a_failed_read_of_a_renamed_file_names_the_path_of_the_side_read() {
        // [MAGI S6 loop 14, A2-5] the old side's read failure was logged under the new path,
        // which reads fine, so the entry pointed at the wrong file.
        struct OldFails;
        impl ContentSource for OldFails {
            fn read(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Ok(Some(b"fn f() {}\n".to_vec()))
            }
            fn read_old(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Err(ReviewError::Io(format!("{path}: sharing violation")))
            }
        }
        let moved = "diff --git a/was_here.rs b/now_here.rs\nsimilarity index 90%\n\
                     rename from was_here.rs\nrename to now_here.rs\n--- a/was_here.rs\n\
                     +++ b/now_here.rs\n@@ -1,1 +1,1 @@\n-fn f() {}\n+fn f() {}\n";
        let files = || {
            vec![
                ReviewFile {
                    path: "now_here.rs".into(),
                    diff: Some(moved.into()),
                },
                ReviewFile {
                    path: "a.rs".into(),
                    diff: None,
                },
            ]
        };
        let log = crate::logging::testutil::capture(|| {
            let f = filter_files(files(), &OldFails, &LanguageRegistry::new(), false, &|_| {
                false
            })
            .expect("a.rs kept");
            assert_eq!(
                f.skipped,
                vec![Skipped {
                    path: "now_here.rs".into(),
                    cause: SkipCause::Unreadable
                }]
            );
        });
        assert!(log.contains("path=was_here.rs"), "{log}");
    }

    #[test]
    fn a_rename_whose_old_path_is_shaped_like_a_key_is_left_out_before_any_read() {
        // [MAGI S6 loop 15, A2-4] the path check ran on the new path only: a rename's old path
        // shaped like a key was read (and cached) under that name before the diff scan left the
        // file out, and a failed read of it was logged as `unreadable`, not `secret`.
        use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
        struct Counting(AtomicUsize);
        impl ContentSource for Counting {
            fn read(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                self.0.fetch_add(1, SeqCst);
                Ok(Some(b"fn f() {}\n".to_vec()))
            }
            fn read_old(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                self.0.fetch_add(1, SeqCst);
                Err(ReviewError::Io(format!("{path}: sharing violation")))
            }
        }
        let old = concat!("AK", "IA", "ABCDEFGHIJKLMNOP.rs");
        let moved = format!(
            "diff --git a/{old} b/now.rs\nsimilarity index 90%\nrename from {old}\n\
             rename to now.rs\n--- a/{old}\n+++ b/now.rs\n@@ -1,1 +1,1 @@\n-fn f() {{}}\n+fn f() {{}}\n"
        );
        let file = vec![ReviewFile {
            path: "now.rs".into(),
            diff: Some(moved),
        }];
        let src = Counting(AtomicUsize::new(0));
        let err = filter_files(file, &src, &LanguageRegistry::new(), false, &|_| false)
            .expect_err("the only file is left out");
        assert!(
            matches!(
                &err,
                ReviewError::NothingToReview { skipped }
                    if skipped == &[Skipped { path: "now.rs".into(), cause: SkipCause::Secret }]
            ),
            "{err:?}"
        );
        assert_eq!(src.0.load(SeqCst), 0, "no side was read");
    }

    #[test]
    fn a_masked_file_and_an_omitted_one_share_the_cause_and_never_the_list() {
        // [MAGI S6 MS3 loop7, R6a Balthasar] Both are `secret`; the list tells them apart.
        let key = concat!("sk-", "proj-", "AAAAAAAAAAAAAAAAAAAAAAAA");
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        let end = concat!("-----", "END ", "PRIVATE KEY-----");
        let block = format!("{begin}\nMIIBVQIBADANBgkqhkiG9w0BAQEFAAS\n{end}\n");
        let inner = MapSource::new(
            &[("m.rs", &format!("let k = \"{key}\";\n")), ("k.rs", &block)],
            &[],
        );
        let files = ["m.rs", "k.rs"].map(|p| ReviewFile {
            path: p.into(),
            diff: None,
        });
        let src = CachedSource::new(&inner);
        let f = filter_files(files.into(), &src, &LanguageRegistry::new(), false, &|_| {
            false
        })
        .expect("m.rs kept");
        assert_eq!(f.masked, vec![masked("m.rs", "openai-key")]);
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: "k.rs".into(),
                cause: SkipCause::Secret
            }]
        );
        assert_eq!(f.masked[0].cause, SkipCause::Secret);
    }

    #[test]
    fn a_renamed_file_is_masked_from_its_old_path_and_a_lone_key_header_keeps_it_out() {
        // [MAGI §6 MS2 A loop 1] a rename's old side lives under its old path: read by the new path it was empty,
        // and a key on a removed line or outside the hunk reached a model.
        // [T3.7] RV22 amended: new.rs is reviewed, its diff rebuilt from the masked sides (the key
        // the removed line quoted is gone). k_new.rs stays out: only its old side has the key's
        // `BEGIN` line, so its new side's body lines are a key no pattern names and cannot be masked.
        let removed = concat!("-", "sk-", "proj-", "AAAAAAAAAAAAAAAAAAAAAAAA");
        let moved = format!(
            "diff --git a/old.rs b/new.rs\nsimilarity index 90%\nrename from old.rs\nrename to new.rs\n\
             --- a/old.rs\n+++ b/new.rs\n@@ -1,2 +1,1 @@\n fn f() {{}}\n{removed}\n"
        );
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        let old_key = format!("{begin}\nMIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8wggE7AgEAAkEA\nBBBB\n-----END PRIVATE KEY-----\n");
        let new_key = old_key.replace("\nBBBB\n", "\nAAAA\n");
        let body_only = "diff --git a/k_old.rs b/k_new.rs\nsimilarity index 90%\nrename from k_old.rs\n\
                         rename to k_new.rs\n--- a/k_old.rs\n+++ b/k_new.rs\n@@ -3,1 +3,1 @@\n-BBBB\n+AAAA\n";
        // The new side of k_new.rs has no header either: only its old side holds the `BEGIN` line.
        let new_body = new_key.replace(begin, "");
        let inner = MapSource::new(
            &[
                ("new.rs", "fn f() {}\n"),
                ("k_new.rs", new_body.as_str()),
                ("a.rs", "fn a() {}\n"),
            ],
            &[
                ("old.rs", "fn f() {}\nfn g() {}\n"),
                ("k_old.rs", old_key.as_str()),
            ],
        );
        let files = vec![
            ReviewFile {
                path: "new.rs".into(),
                diff: Some(moved),
            },
            ReviewFile {
                path: "k_new.rs".into(),
                diff: Some(body_only.into()),
            },
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
        ];
        let src = CachedSource::new(&inner);
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
            .expect("a.rs kept");
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: "k_new.rs".into(),
                cause: SkipCause::Secret
            }]
        );
        // [MAGI S6 MS3 loop16, R6a1] The key sat only in the diff's removed line, which the
        // rebuild replaces with the clean old side: no byte was masked, so none is declared.
        assert!(f.masked.is_empty(), "{:?}", f.masked);
        let diff = f
            .kept
            .iter()
            .find(|k| k.path == "new.rs")
            .and_then(|k| k.diff.as_deref())
            .expect("new.rs kept with its diff");
        assert!(!diff.contains("AAAAAAAAAAAA"), "{diff}");
        assert!(
            diff.ends_with("@@ -1,2 +1,1 @@\n fn f() {}\n-fn g() {}\n"),
            "{diff}"
        );
    }

    /// Filters `key.rs` (its diff, its new and old sides) beside a clean `a.rs`, through the
    /// masking cache, and asserts that no byte of a key body (`BODY`) leaves in a kept file: in its
    /// diff, or in either side the batches, units and gate read, and that `key.rs` itself is left
    /// out as `secret`, never reviewed masked (RV22's exception for a multi-line key block).
    // [T3.7 fix round 3] the omission is required again: any side holding a key block leaves the
    // file out (round 2 had let the guard mask instead).
    fn assert_no_key_body_leaves(diff: Option<&str>, new: &str, old: Option<&str>) {
        let olds: Vec<(&str, &str)> = old.map(|o| ("key.rs", o)).into_iter().collect();
        let inner = MapSource::new(&[("key.rs", new), ("a.rs", "fn f() {}\n")], &olds);
        let src = CachedSource::new(&inner);
        let files = vec![
            ReviewFile {
                path: "key.rs".into(),
                diff: diff.map(str::to_string),
            },
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
        ];
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
            .expect("a.rs kept");
        let text = |b: Option<Vec<u8>>| String::from_utf8(b.unwrap_or_default()).expect("utf-8");
        for kept in &f.kept {
            let diff = kept.diff.as_deref().unwrap_or("");
            assert!(!diff.contains("BODY"), "{}: {diff}", kept.path);
            let new = text(src.read(&kept.path).expect("read"));
            assert!(!new.contains("BODY"), "{}: {new}", kept.path);
            let old = text(src.read_old(&kept.path).expect("read"));
            assert!(!old.contains("BODY"), "{}: {old}", kept.path);
        }
        // [T3.7 fix round 3] RV22's exception: a file any side of which holds a multi-line
        // private-key block is left out as `secret`, never reviewed masked.
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: "key.rs".into(),
                cause: SkipCause::Secret
            }],
            "{f:?}"
        );
        assert!(f.masked.is_empty(), "{:?}", f.masked);
    }

    /// A PEM block around one body line, its armor split so the source holds no header.
    fn pem_block(body: &str) -> String {
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        let end = concat!("-----", "END ", "PRIVATE KEY-----");
        format!("{begin}\n{body}\n{end}")
    }

    /// `n` numbered plain lines, `fn p<i>() {}`.
    fn plain_lines(n: usize) -> String {
        (1..=n).map(|i| format!("fn p{i}() {{}}\n")).collect()
    }

    #[test]
    fn a_key_body_moved_bare_to_another_hunk_never_leaves() {
        // [T3.7 fix round 2, re-review R1, security] the block is deleted in one hunk and its
        // body re-added bare twenty lines away, in a second hunk: no hunk crosses on its own.
        let body = "BODYONEMIIBVQIBADANBgkqhkiG9w0B";
        let plain = plain_lines(20);
        let old = format!("fn a() {{}}\n{}\n{plain}fn z() {{}}\n", pem_block(body));
        let new = format!("fn a() {{}}\n{plain}fn z() {{}}\n{body}\n");
        let deleted: String = pem_block(body).lines().map(|l| format!("-{l}\n")).collect();
        let diff = format!(
            "@@ -1,5 +1,2 @@\n fn a() {{}}\n{deleted} fn p1() {{}}\n\
             @@ -24,2 +21,3 @@\n fn p20() {{}}\n fn z() {{}}\n+{body}\n"
        );
        assert_no_key_body_leaves(Some(&diff), &new, Some(&old));
    }

    #[test]
    fn a_key_body_copied_elsewhere_beside_its_unchanged_block_never_leaves() {
        // [T3.7 fix round 2, re-review R1, security] the block is unchanged and its body line is
        // also added, bare, below: a positional check sees only a plain added line.
        let body = "BODYONEMIIBVQIBADANBgkqhkiG9w0B";
        let old = format!("fn a() {{}}\n{}\nfn b() {{}}\n", pem_block(body));
        // A copy on a line of its own, as in an array of key lines; a body embedded in other code
        // on the same line is outside what a line fingerprint matches (declared in the README).
        let new = format!("{old}    \"{body}\",\n");
        let diff = format!("@@ -5,0 +6 @@\n+    \"{body}\",\n");
        assert_no_key_body_leaves(Some(&diff), &new, Some(&old));
    }

    #[test]
    fn a_whole_file_with_a_key_block_and_a_bare_copy_of_its_body_never_leaves_it() {
        // [T3.7 fix round 2, re-review R1, security] one text, no diff: the block is masked and
        // its body line, copied raw elsewhere in the same file, matches no pattern on its own.
        let body = "BODYONEMIIBVQIBADANBgkqhkiG9w0B";
        let new = format!("{}\nfn b() {{}}\n    \"{body}\\n\" +\n", pem_block(body));
        assert_no_key_body_leaves(None, &new, None);
    }

    #[test]
    fn a_text_file_git_marks_binary_whose_old_key_block_lost_its_begin_line_is_left_out() {
        // [T3.7 fix round 2, re-review R2] the branch reviewed whole from its new side: the old
        // side holds the block, the new side the block's edited body without its armor, which no
        // fingerprint can match. Any side holding a key block leaves the file out (`secret`).
        let section = "diff --git a/key.rs b/key.rs\nindex 1111111..2222222 100644\n\
                       Binary files a/key.rs and b/key.rs differ\n";
        let old = format!(
            "{}\nfn b() {{}}\n",
            pem_block("BODYOLDMIIBVQIBADANBgkqhkiG9w0B")
        );
        let new = "BODYEDITEDVQIBADANBgkqhkiG9w0B\nfn b() {}\n".to_string();
        assert_no_key_body_leaves(Some(section), &new, Some(&old));
    }

    #[test]
    fn a_key_whose_begin_line_was_deleted_beside_another_added_key_never_leaves_its_body() {
        // [T3.7 fix round 1, review F1, security] both sides hold a key block, at different
        // lines: the old block lost its `BEGIN` line (its body is a context line now, masked on
        // the old side only) while another block was added below. A per-side flag passed it and
        // the body left raw from the new side.
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        let end = concat!("-----", "END ", "PRIVATE KEY-----");
        let old = format!("fn a() {{}}\n{begin}\nBODYONE\n{end}\nfn b() {{}}\n");
        let new = format!(
            "fn a() {{}}\nBODYONE\n{end}\nfn b() {{}}\nconst B: &str = \"{begin}\nQQQQ\n{end}\";\n"
        );
        let diff = format!(
            "@@ -1,5 +1,7 @@\n fn a() {{}}\n-{begin}\n BODYONE\n {end}\n fn b() {{}}\n\
             +const B: &str = \"{begin}\n+QQQQ\n+{end}\";\n"
        );
        assert_no_key_body_leaves(Some(&diff), &new, Some(&old));
    }

    #[test]
    fn a_key_whose_end_line_moved_up_never_leaves_the_body_lines_below_it() {
        // [T3.7 fix round 1, review F1, security] both sides hold one block, but the new side's
        // END line moved up: the body lines below it are context, masked on the old side only.
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        let end = concat!("-----", "END ", "PRIVATE KEY-----");
        let old =
            format!("fn a() {{}}\n{begin}\nBODYONE\nBODYTWO\nBODYTHREE\n{end}\nfn b() {{}}\n");
        let new =
            format!("fn a() {{}}\n{begin}\nBODYONE\n{end}\nBODYTWO\nBODYTHREE\nfn b() {{}}\n");
        let diff = format!(
            "@@ -1,7 +1,7 @@\n fn a() {{}}\n {begin}\n BODYONE\n+{end}\n BODYTWO\n BODYTHREE\n\
             -{end}\n fn b() {{}}\n"
        );
        assert_no_key_body_leaves(Some(&diff), &new, Some(&old));
    }

    #[test]
    fn a_key_added_whole_beside_an_unchanged_key_is_left_out() {
        // [T3.7 fix round 1] the line-precise guard kept this file, its bodies masked.
        // [T3.7 fix round 3] RV22's exception: any private-key block leaves the file out.
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        let end = concat!("-----", "END ", "PRIVATE KEY-----");
        let old = format!("{begin}\nBODYONE\n{end}\nfn b() {{}}\n");
        let new = format!("{begin}\nBODYONE\n{end}\nfn b() {{}}\n{begin}\nBODYTWO\n{end}\n");
        let diff = format!("@@ -4,0 +5,3 @@\n+{begin}\n+BODYTWO\n+{end}\n");
        assert_no_key_body_leaves(Some(&diff), &new, Some(&old));
    }

    #[test]
    fn skip_causes_serialize_with_the_spec_texts() {
        let got: Vec<String> = [
            SkipCause::Binary,
            SkipCause::NotUtf8,
            SkipCause::Secret,
            SkipCause::Ignored,
            SkipCause::OutsideWorkspace,
            SkipCause::TooLarge,
            SkipCause::NotRegularFile,
            SkipCause::DuplicateSection,
        ]
        .iter()
        .map(|c| serde_json::to_string(c).expect("serializes"))
        .collect();
        assert_eq!(
            got,
            vec![
                "\"binary\"",
                "\"not utf-8\"",
                "\"secret\"",
                "\"ignored\"",
                "\"outside workspace\"",
                "\"too large\"",
                "\"not a regular file\"",
                "\"repeated section\""
            ]
        );
    }

    #[test]
    fn a_pem_key_whose_header_is_outside_the_hunk_still_skips_the_file() {
        // [CP2 MS2 S1b loop 11] the hunk touches only the key body; both whole sides are scanned, so `BEGIN` is seen.
        // [T3.7 fix round 3] RV22's exception: a private-key block leaves the file out, as before
        // the amendment.
        // Split so tests/no_hardcoded_secrets.rs (which flags the literal `-----BEGIN`) does not read a header here.
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        let new = format!("{begin}\nMIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8wggE7AgEAAkEA\nAAAA\n-----END PRIVATE KEY-----\n");
        let old = new.replace("\nAAAA\n", "\nBBBB\n");
        let files = vec![
            ReviewFile {
                path: "key.rs".into(),
                diff: Some("@@ -3,1 +3,1 @@\n-BBBB\n+AAAA\n".into()),
            },
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
        ];
        let inner = MapSource::new(
            &[("key.rs", new.as_str()), ("a.rs", "fn f() {}\n")],
            &[("key.rs", old.as_str())],
        );
        let src = CachedSource::new(&inner);
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
            .expect("a.rs kept");
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: "key.rs".into(),
                cause: SkipCause::Secret
            }]
        );
        assert!(f.masked.is_empty(), "{:?}", f.masked);
        assert!(f.kept.iter().all(|k| k.path != "key.rs"), "{f:?}");
    }

    #[test]
    fn a_file_over_the_size_cap_is_skipped_and_an_explicit_one_is_an_error() {
        // [CP2 MS2 S1a loop 9] the content source refuses it (T2.3); the filter turns that into `too large`.
        struct Capped;
        impl ContentSource for Capped {
            fn read(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                if path == "huge.rs" {
                    Err(ReviewError::TooLarge {
                        path: path.into(),
                        limit: 1,
                    })
                }
                // [S1a loop 10] typed
                else {
                    Ok(Some(b"fn f() {}\n".to_vec()))
                }
            }
            fn read_old(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Ok(None)
            }
        }
        let files = vec![
            ReviewFile {
                path: "huge.rs".into(),
                diff: None,
            },
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
        ];
        let f = filter_files(files, &Capped, &LanguageRegistry::new(), false, &|_| false)
            .expect("a.rs kept");
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: "huge.rs".into(),
                cause: SkipCause::TooLarge
            }]
        );
        let one = vec![ReviewFile {
            path: "huge.rs".into(),
            diff: None,
        }];
        assert!(matches!(
            filter_files(one, &Capped, &LanguageRegistry::new(), true, &|_| false),
            Err(ReviewError::NotReviewable {
                cause: "too large",
                ..
            })
        ));
    }

    #[test]
    fn an_armored_pgp_private_key_skips_the_file_and_logs_only_its_pattern_name() {
        // [T3.7 fix round 3] RV22's exception: a private-key block leaves the file out.
        // Split so tests/no_hardcoded_secrets.rs (which flags the literal `-----BEGIN`) reads no
        // header here; the body is obviously fake.
        let begin = concat!("-----", "BEGIN PGP PRIVATE KEY BLOCK-----");
        let end = concat!("-----", "END PGP PRIVATE KEY BLOCK-----");
        let body = "lQOYBGZ0FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE";
        let text = format!("const K: &str = r\"\n{begin}\n\n{body}\n=FAKE\n{end}\n\";\n");
        let public = concat!("-----", "BEGIN PGP PUBLIC KEY BLOCK-----");
        let pubtext = format!("{public}\n\n{body}\n");
        let inner = MapSource::new(
            &[
                ("key.rs", text.as_str()),
                ("pub.asc", pubtext.as_str()),
                ("a.rs", "fn f() {}\n"),
            ],
            &[],
        );
        let files = vec![
            ReviewFile {
                path: "key.rs".into(),
                diff: None,
            },
            ReviewFile {
                path: "pub.asc".into(),
                diff: None,
            },
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
        ];
        let src = CachedSource::new(&inner);
        let line = crate::logging::testutil::capture(|| {
            let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
                .expect("a.rs kept");
            assert_eq!(
                f.skipped,
                vec![Skipped {
                    path: "key.rs".into(),
                    cause: SkipCause::Secret
                }]
            );
            assert!(f.masked.is_empty(), "{:?}", f.masked);
            // A public key is not a secret.
            assert!(f.kept.iter().any(|k| k.path == "pub.asc"), "{f:?}");
        });
        assert!(
            line.contains("key.rs") && line.contains("pgp-private-key"),
            "{line}"
        );
        assert!(!line.contains(body), "{line}");
    }

    #[test]
    fn a_masked_secret_logs_the_path_and_the_pattern_name_never_the_value() {
        // [CP2 MS2 S1b loop 9] the consumer of SecretPattern.name (loop 8), pinned.
        // [T3.7] the entry now says the file was reviewed with the secret masked.
        let key = "ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"; // allow-secret-scan
        let text = format!("const K: &str = \"{key}\";\n");
        let inner = MapSource::new(
            &[("config.rs", text.as_str()), ("a.rs", "fn f() {}\n")],
            &[],
        );
        let files = vec![
            ReviewFile {
                path: "config.rs".into(),
                diff: None,
            },
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
        ];
        let src = CachedSource::new(&inner);
        let line = crate::logging::testutil::capture(|| {
            filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
                .expect("a.rs kept");
        });
        assert!(
            line.contains("config.rs") && line.contains("github-token") && line.contains("masked"),
            "{line}"
        );
        assert!(!line.contains(key), "{line}");
    }

    #[test]
    fn a_hard_linked_file_is_left_out_as_a_hard_link_not_as_outside_the_workspace() {
        // [MAGI S6 loop 12, A2-b1 / A2-c4] the refusal stays, but its cause read `outside
        // workspace` for a path inside it: an operator looked for an escape that did not exist.
        let outer = tempfile::tempdir().expect("tempdir");
        let root = outer.path().join("ws");
        std::fs::create_dir_all(&root).expect("mkdir");
        std::fs::write(outer.path().join("store.rs"), "fn s() {}\n").expect("write");
        std::fs::hard_link(outer.path().join("store.rs"), root.join("linked.rs")).expect("link");
        std::fs::write(root.join("a.rs"), "fn f() {}\n").expect("write");
        let src = crate::review::source::WorktreeSource::new(root);
        let files = |names: &[&str]| {
            names
                .iter()
                .map(|n| ReviewFile {
                    path: (*n).into(),
                    diff: None,
                })
                .collect::<Vec<_>>()
        };
        let f = filter_files(
            files(&["linked.rs", "a.rs"]),
            &src,
            &LanguageRegistry::new(),
            false,
            &|_| false,
        )
        .expect("a.rs kept");
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: "linked.rs".into(),
                cause: SkipCause::HardLink
            }]
        );
        assert_eq!(f.skipped[0].to_string(), "linked.rs (hard link)");
        assert!(matches!(
            filter_files(
                files(&["linked.rs"]),
                &src,
                &LanguageRegistry::new(),
                true,
                &|_| false
            ),
            Err(ReviewError::NotReviewable {
                cause: "hard link",
                ..
            })
        ));
    }

    #[test]
    fn a_file_that_resolves_outside_the_workspace_is_skipped_and_an_explicit_one_is_an_error() {
        struct Escaping;
        impl ContentSource for Escaping {
            fn read(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                if path == "link.rs" {
                    Err(ReviewError::OutsideWorkspace(path.into()))
                } else {
                    Ok(Some(b"fn f() {}\n".to_vec()))
                }
            }
            fn read_old(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Ok(None)
            }
        }
        let files = vec![
            ReviewFile {
                path: "link.rs".into(),
                diff: None,
            },
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
        ];
        let f = filter_files(files, &Escaping, &LanguageRegistry::new(), false, &|_| {
            false
        })
        .expect("a.rs kept");
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: "link.rs".into(),
                cause: SkipCause::OutsideWorkspace
            }]
        );
        let one = vec![ReviewFile {
            path: "link.rs".into(),
            diff: None,
        }];
        assert!(matches!(
            filter_files(one, &Escaping, &LanguageRegistry::new(), true, &|_| false),
            Err(ReviewError::NotReviewable {
                cause: "outside workspace",
                ..
            })
        ));
    }

    #[test]
    fn a_secret_in_a_removed_line_is_masked_in_the_diff_and_the_file_reviewed() {
        // [CP2 MS2 S1b loop 15] with a kept sibling, so the skip and its `secret` cause are asserted, not inferred.
        // [T3.7] RV22 amended (RV-S30): the deleted line is shown, its key masked.
        let removed = "const K: &str = \"sk-proj-AAAAAAAAAAAAAAAAAAAAAAAA\";"; // allow-secret-scan
        let diff = format!("@@ -1,2 +1,1 @@\n-{removed}\n fn f() {{}}\n");
        let old = format!("{removed}\nfn f() {{}}\n");
        let files = vec![
            ReviewFile {
                path: "a.rs".into(),
                diff: Some(diff),
            },
            ReviewFile {
                path: "b.rs".into(),
                diff: None,
            },
        ];
        let inner = MapSource::new(
            &[("a.rs", "fn f() {}\n"), ("b.rs", "fn b() {}\n")],
            &[("a.rs", old.as_str())],
        );
        let src = CachedSource::new(&inner);
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
            .expect("both kept");
        assert_eq!(f.skipped, vec![]);
        assert_eq!(f.masked, vec![masked("a.rs", "openai-key")]);
        assert_eq!(
            f.kept.first().and_then(|k| k.diff.as_deref()),
            Some("@@ -1,2 +1,1 @@\n-const K: &str = \"<redacted: openai-key>\";\n fn f() {}\n")
        );
    }

    #[test]
    fn real_git_diffs_rebuild_unchanged_from_their_own_sides() {
        // [MAGI S6 MS3 loop2, R6a Balthasar] `rebuilt_diff` reads git's dialect: on real output it
        // must give the diff back byte for byte from unmasked sides (several hunks, a function
        // context, an edit at the first line, a last line without a newline, a deletion, an
        // addition), so a masked file is never left out for a parse it should have done.
        fn git(dir: &Path, args: &[&str]) -> String {
            let out = std::process::Command::new("git")
                .arg("-C")
                .arg(dir)
                .args(args)
                .output()
                .expect("git runs");
            assert!(out.status.success(), "git {args:?}");
            String::from_utf8(out.stdout).expect("utf-8")
        }
        let d = tempfile::tempdir().expect("tempdir");
        git(d.path(), &["init", "-q", "-b", "main"]);
        for (k, v) in [
            ("user.email", "t@example.invalid"),
            ("user.name", "t"),
            ("commit.gpgsign", "false"),
            ("core.autocrlf", "false"),
        ] {
            git(d.path(), &["config", k, v]);
        }
        let body: String = (1..=30).map(|i| format!("fn f{i}() {{}}\n")).collect();
        let cases: [(&str, Option<String>, Option<String>); 4] = [
            (
                "edit.rs",
                Some(body.clone()),
                Some(
                    body.replace("fn f1()", "fn g1()")
                        .replace("fn f15()", "fn g15()")
                        .replace("fn f30() {}\n", "fn f30() {}"),
                ),
            ),
            ("gone.rs", Some(body.clone()), None),
            ("new.rs", None, Some("fn n() {}\n".to_string())),
            (
                "tail.rs",
                Some("a\nb".to_string()),
                Some("a\nc\n".to_string()),
            ),
        ];
        for (path, old, _) in &cases {
            if let Some(old) = old {
                std::fs::write(d.path().join(path), old).expect("write");
            }
        }
        git(d.path(), &["add", "."]);
        git(d.path(), &["commit", "-q", "-m", "1"]);
        for (path, _, new) in &cases {
            match new {
                Some(new) => std::fs::write(d.path().join(path), new).expect("write"),
                None => std::fs::remove_file(d.path().join(path)).expect("rm"),
            }
        }
        git(d.path(), &["add", "-A"]);
        for (path, old, new) in &cases {
            let diff = git(
                d.path(),
                &[
                    "diff",
                    "--cached",
                    "--no-color",
                    "--no-ext-diff",
                    "--",
                    path,
                ],
            );
            assert!(diff.contains("@@"), "{path}: {diff}");
            let rebuilt = rebuilt_diff(
                &diff,
                new.as_deref().map(str::as_bytes),
                old.as_deref().map(str::as_bytes),
            )
            .map(|(rebuilt, _)| rebuilt);
            // [MAGI S6 MS3 loop6, R6a Melchior] Byte for byte, the hunk's function context
            // included: it is kept, masked, and no longer dropped.
            assert_eq!(rebuilt.as_deref(), Some(diff.as_str()), "{path}");
        }
    }

    #[test]
    fn a_hunk_context_that_no_masked_side_line_shows_is_dropped() {
        // [MAGI S6 MS3 loop6] A function context masked on its own is the diff-alone case: a
        // YAML value line has no pattern without its key on the line above, so masking the
        // context alone left the value raw. Only a context some masked side line shows as it is
        // is kept.
        let key = concat!("aws_secret", "_access_key");
        let value = concat!("wJalrXUtnFEMI/K7MDENG/", "bPxRfiCYEXAMPLEKEY"); // allow-secret-scan
        let side = format!("{key}: |\n{value}\nother: 1\n");
        let masked = crate::review::secrets::mask_secrets_keeping_lines(&side)
            .expect("masks")
            .text;
        assert!(
            !masked.contains(value),
            "precondition: the whole side masks it"
        );
        let diff = format!("@@ -3,1 +3,1 @@ {value}\n-other: 0\n+other: 1\n");
        let old = masked.replace("other: 1", "other: 0");
        let (rebuilt, _) =
            rebuilt_diff(&diff, Some(masked.as_bytes()), Some(old.as_bytes())).expect("rebuilds");
        assert!(!rebuilt.contains(value), "{rebuilt}");
        assert!(rebuilt.starts_with("@@ -3,1 +3,1 @@\n"), "{rebuilt}");
    }

    #[test]
    fn a_rebuilt_hunk_keeps_its_function_context_masked_and_counts_every_unexplained_difference() {
        // [MAGI S6 MS3 loop6, R6a Melchior] The reconstruction dropped the function context
        // after `@@`, a change git never makes; and a differing line was never counted once its
        // side held a mask, whether the mask explained the difference or not.
        let key = concat!("sk-ant-", "api03-", "AAAAAAAAAAAAAAAAAAAA"); // allow-secret-scan
        let new = format!("fn outer() {{\n    let k = \"{key}\";\n}}\n");
        let masked = crate::review::secrets::mask_secrets_keeping_lines(&new)
            .expect("masks")
            .text;
        let diff = format!(
            "@@ -1,3 +1,3 @@ fn outer() {{\n fn outer() {{\n-    let k = 1;\n+    let k = \"{key}\";\n }}\n"
        );
        let old: &[u8] = b"fn outer() {\n    let k = 1;\n}\n";
        let (rebuilt, diverged) =
            rebuilt_diff(&diff, Some(masked.as_bytes()), Some(old)).expect("rebuilds");
        let header = rebuilt.lines().next().unwrap_or_default();
        assert_eq!(header, "@@ -1,3 +1,3 @@ fn outer() {", "{rebuilt}");
        assert!(!rebuilt.contains("sk-ant-"), "{rebuilt}");
        assert_eq!(diverged, 0, "a difference the mask explains is not counted");
        // A context that carries the secret is no masked side line as it is: it is dropped.
        let leaky = diff.replacen("@@ fn outer() {", &format!("@@ fn outer() {{ // {key}"), 1);
        let (rebuilt, _) =
            rebuilt_diff(&leaky, Some(masked.as_bytes()), Some(old)).expect("rebuilds");
        assert!(rebuilt.starts_with("@@ -1,3 +1,3 @@\n"), "{rebuilt}");
        assert!(!rebuilt.contains("sk-ant-"), "{rebuilt}");
        // A line that differs beyond its mask is counted, though its side holds one.
        let stale = diff.replace("+    let k = \"", "+    let k = \"x");
        let (_, diverged) = rebuilt_diff(
            &stale,
            Some(masked.as_bytes()),
            Some(b"fn outer() {\n    let k = 1;\n}\n"),
        )
        .expect("rebuilds");
        assert_eq!(diverged, 1);
    }

    #[test]
    fn a_real_later_stage_reading_a_left_out_key_block_file_fails_closed() {
        // [MAGI S6 MS3 loop6, R6a Melchior] The seal's contract against a real stage, not the
        // cache alone: the containing-unit lookup of the RAG, handed a hunk of a file the filter
        // left out for its key block, is refused (`EgressSecret`, which ends the review) instead
        // of being served the raw side. Every production stage reads only kept files, so this
        // is the fail-closed direction of RV22's tie-break.
        use crate::review::batching::BatchItem;
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        let key = format!(
            "fn k() {{\n    let s = \"{begin}\nMIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8w\";\n}}\n"
        );
        let inner = MapSource::new(&[("k.rs", key.as_str()), ("a.rs", "fn a() {}\n")], &[]);
        let src = CachedSource::new(&inner);
        let files = vec![
            ReviewFile {
                path: "k.rs".into(),
                diff: None,
            },
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
        ];
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
            .expect("a.rs kept");
        assert_eq!(f.skipped.len(), 1, "precondition: k.rs left out");
        let hunk = BatchItem::Hunk {
            file: "k.rs".into(),
            text: "@@ -0,0 +1,2 @@\n+fn k() {\n+    let s = 1;\n".into(),
            first_line: 1,
            last_line: 2,
            old_first_line: 0,
            old_last_line: 0,
        };
        assert!(matches!(
            crate::review::rag::containing_units(&hunk, &src, &LanguageRegistry::new()),
            Err(ReviewError::EgressSecret { .. })
        ));
    }

    #[test]
    fn a_filter_that_fails_still_seals_the_source() {
        // [MAGI S6 MS3 loop6, R6a Melchior] The seal ran on the success path only: a filter that
        // returned early (a key-block file named explicitly) left the cache unsealed, so a later
        // read of that side would have been served raw had the caller gone on.
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        let key = format!("{begin}\nMIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8w\n");
        let inner = MapSource::new(&[("k.rs", key.as_str())], &[]);
        let src = CachedSource::new(&inner);
        let one = vec![ReviewFile {
            path: "k.rs".into(),
            diff: None,
        }];
        assert!(matches!(
            filter_files(one, &src, &LanguageRegistry::new(), true, &|_| false),
            Err(ReviewError::NotReviewable {
                cause: "secret",
                ..
            })
        ));
        assert!(matches!(
            src.read("k.rs"),
            Err(ReviewError::EgressSecret { .. })
        ));
    }

    #[test]
    fn a_real_filter_pass_over_the_production_cache_seals_and_reports_masks() {
        // [MAGI S6 MS3 loop16, R6a2 Melchior] `seal` and `masked` are defaulted no-ops on
        // `ContentSource`: a wrapper that forgets to forward them compiles and silently opens the
        // post-filter egress guard. Drive a real filter pass through the production `CachedSource`
        // and assert both effects: an unclassified key-block path is refused afterwards, and the
        // mask the cache applied is the one the filter declares.
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        let key = format!("{begin}\nMIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8w\n");
        let token = concat!("ghp_", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"); // allow-secret-scan
        let masked = format!("let t = \"{token}\";\n");
        let inner = MapSource::new(
            &[
                ("a.rs", "fn a() {}\n"),
                ("t.rs", masked.as_str()),
                ("never.pem", key.as_str()),
            ],
            &[],
        );
        let src = CachedSource::new(&inner);
        let files = ["a.rs", "t.rs"]
            .iter()
            .map(|p| ReviewFile {
                path: (*p).into(),
                diff: None,
            })
            .collect();
        let filtered = filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
            .expect("filters");
        assert_eq!(filtered.kept.len(), 2, "the masked file stays reviewed");
        assert_eq!(filtered.masked.len(), 1, "the cache's mask is declared");
        assert!(
            matches!(src.read("never.pem"), Err(ReviewError::EgressSecret { .. })),
            "a path the filter never classified was served raw after its pass"
        );
    }

    #[test]
    fn many_hunks_stay_near_linear() {
        // [MAGI S6 MS3 loop10, R6a Balthasar] Each hunk's context scanned both whole sides.
        let n = 2_000usize;
        let side: String = (0..n).map(|i| format!("fn f{i}() {{}}\n")).collect();
        let diff: String = (1..=n)
            .map(|i| format!("@@ -{i},1 +{i},1 @@ fn f{}\n fn f{}() {{}}\n", i - 1, i - 1))
            .collect();
        CONTEXT_VISITS.with(|c| c.set(0));
        let (_, diverged) =
            rebuilt_diff(&diff, Some(side.as_bytes()), Some(side.as_bytes())).expect("rebuilds");
        assert_eq!(diverged, 0);
        let visits = CONTEXT_VISITS.with(std::cell::Cell::get);
        assert!(visits <= 4 * n + 64 * n, "{visits}");
    }

    #[test]
    fn a_context_line_that_differs_on_the_old_side_counts_as_a_divergence() {
        // [MAGI S6 MS3 loop7, R6a Caspar] A context line never looked at the old side.
        let diff = "@@ -1,3 +1,3 @@\n a\n b\n c\n";
        let (rebuilt, diverged) =
            rebuilt_diff(diff, Some(b"a\nb\nc\n"), Some(b"a\nX\nc\n")).expect("rebuilds");
        assert_eq!(
            rebuilt, diff,
            "the old side's text is compared, never written"
        );
        assert_eq!(diverged, 1);
        let (_, clean) =
            rebuilt_diff(diff, Some(b"a\nb\nc\n"), Some(b"a\nb\nc\n")).expect("rebuilds");
        assert_eq!(clean, 0);
    }

    #[test]
    fn unrebuilt_diff_is_masked() {
        // [MAGI S6 MS3 loop2, R6a Balthasar] the safe fallback of `kept_verdict`: a diff that
        // names a line its masked side does not have cannot be rebuilt, so the raw diff, which
        // carries the key, is never kept as it is.
        // MAGI FIX (MS3 §6 loop11): it was a third omission class; the raw diff is masked now
        // (`masked_raw_diff`: a body line no masked side shows verbatim is redacted), the file
        // reviewed and declared masked, and a file named explicitly is no longer refused.
        let removed = "const K: &str = \"sk-proj-AAAAAAAAAAAAAAAAAAAAAAAA\";"; // allow-secret-scan
        let diff = format!("@@ -5,2 +5,1 @@\n-{removed}\n fn f() {{}}\n");
        let old = format!("{removed}\nfn f() {{}}\n");
        let inner = MapSource::new(&[("a.rs", "fn f() {}\n")], &[("a.rs", old.as_str())]);
        for explicit in [false, true] {
            let one = vec![ReviewFile {
                path: "a.rs".into(),
                diff: Some(diff.clone()),
            }];
            let src = CachedSource::new(&inner);
            let f = filter_files(one, &src, &LanguageRegistry::new(), explicit, &|_| false)
                .expect("kept, masked");
            assert!(f.skipped.is_empty(), "{:?}", f.skipped);
            assert_eq!(f.masked.len(), 1, "{:?}", f.masked);
            let sent = f.kept[0].diff.clone().expect("diff");
            assert!(!sent.contains("sk-proj-"), "{sent}");
            assert!(sent.contains("<redacted: "), "{sent}");
        }
    }

    #[test]
    fn second_diff_section_is_declared() {
        // [MAGI S6 MS3 loop13, R6a Caspar] The first section won and the second vanished with
        // nothing in the report; `input.rs` lists one section per path, so this is a guard.
        let inner = MapSource::new(&[("a.rs", "fn f() {}\n")], &[]);
        let src = CachedSource::new(&inner);
        let sect = |d: &str| ReviewFile {
            path: "a.rs".into(),
            diff: Some(format!("@@ -0,0 +1,1 @@\n+{d}\n")),
        };
        let f = filter_files(
            vec![sect("fn f() {}"), sect("fn f() {}"), sect("fn g() {}")],
            &src,
            &LanguageRegistry::new(),
            false,
            &|_| false,
        )
        .expect("kept once");
        assert_eq!(f.kept.len(), 1);
        let dropped = Skipped {
            path: "a.rs".into(),
            cause: SkipCause::DuplicateSection,
        };
        // MAGI FIX (MS3 §6 loop15): one entry per repeat, the identical one too (it was silent).
        assert_eq!(f.skipped, vec![dropped.clone(), dropped]);
    }
    #[test]
    fn a_listed_twice_file_is_kept_once() {
        // MAGI FIX (MS3 §6 loop12): the second listing of a masked file read the cached masked
        // side and was kept again, so the file was reviewed and declared twice.
        let secret = "const K: &str = \"sk-proj-AAAAAAAAAAAAAAAAAAAAAAAA\";\n"; // allow-secret-scan
        let inner = MapSource::new(&[("a.rs", secret)], &[]);
        let twice: Vec<ReviewFile> = (0..2)
            .map(|_| ReviewFile {
                path: "a.rs".into(),
                diff: None,
            })
            .collect();
        let src = CachedSource::new(&inner);
        let f = filter_files(twice, &src, &LanguageRegistry::new(), false, &|_| false)
            .expect("kept, masked");
        assert_eq!(f.kept.len(), 1, "{:?}", f.kept);
        assert_eq!(f.masked.len(), 1, "{:?}", f.masked);
    }

    #[test]
    fn an_identical_second_listing_is_declared() {
        // [MAGI S6 MS3 loop15, R6a1 Caspar] A second listing of a kept path whose diff hashed
        // the same (an honest repeat, or a collision) was dropped without a word, against the
        // doc of `diff_hash`: it is always declared as a repeated section.
        let diff = "@@ -1,1 +1,1 @@\n-fn a() {}\n+fn a() { 1; }\n";
        let inner = MapSource::new(&[("a.rs", "fn a() { 1; }\n")], &[("a.rs", "fn a() {}\n")]);
        let twice: Vec<ReviewFile> = (0..2)
            .map(|_| ReviewFile {
                path: "a.rs".into(),
                diff: Some(diff.into()),
            })
            .collect();
        let src = CachedSource::new(&inner);
        let f = filter_files(twice, &src, &LanguageRegistry::new(), false, &|_| false)
            .expect("kept once");
        assert_eq!(f.kept.len(), 1);
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: "a.rs".into(),
                cause: SkipCause::DuplicateSection
            }]
        );
    }

    #[test]
    fn raw_diff_masks_hunk_context() {
        // MAGI FIX (MS3 §6 loop12): the `@@` line was masked alone, so a secret value in the
        // function context (its key on another line) left raw; the context is kept only when a
        // masked side shows it, like `rebuilt_diff`'s.
        let removed = "const K: &str = \"sk-proj-AAAAAAAAAAAAAAAAAAAAAAAA\";"; // allow-secret-scan
        let value = concat!("wJalrXUtnFEMI/K7MDENG/", "bPxRfiCYEXAMPLEKEY"); // allow-secret-scan
        let diff = format!("@@ -5,2 +5,1 @@   {value}\n-{removed}\n fn f() {{}}\n");
        let old = format!("{removed}\nfn f() {{}}\n");
        let inner = MapSource::new(&[("a.rs", "fn f() {}\n")], &[("a.rs", old.as_str())]);
        let one = vec![ReviewFile {
            path: "a.rs".into(),
            diff: Some(diff),
        }];
        let src = CachedSource::new(&inner);
        let f = filter_files(one, &src, &LanguageRegistry::new(), false, &|_| false)
            .expect("kept, masked");
        let sent = f.kept[0].diff.clone().expect("diff");
        assert!(!sent.contains("wJalrXUtn"), "{sent}");
        assert!(sent.starts_with("@@ -5,2 +5,1 @@"), "{sent}");
    }

    #[test]
    fn diff_alone_is_unsafe_to_mask() {
        // [MAGI S6 MS3 loop3, R6a Caspar; the tie-break of RV22] Masking the raw diff text alone
        // is not enough: a hunk shows only some lines, and a pattern may need lines the hunk
        // leaves out: a YAML value on the line after its key. The side, read whole, masks it; the
        // diff alone holds the value with no key. So the fallback (`masked_raw_diff`, MS3 §6
        // loop11) redacts every body line no masked side shows verbatim, never `mask_secrets` alone.
        let key = concat!("aws_secret", "_access_key");
        let value = concat!("wJalrXUtnFEMI/K7MDENG/", "bPxRfiCYEXAMPLEKEY"); // allow-secret-scan
        let side = format!("{key}: |\n  {value}\n");
        assert!(
            find_secret(&side).is_some(),
            "precondition: the side shows it"
        );
        let hunk = format!("@@ -2,0 +2,1 @@\n+  {value}\n");
        assert!(
            crate::review::secrets::mask_secrets(&hunk).contains(value),
            "masking the diff alone leaves the value raw"
        );
    }

    #[test]
    fn a_rebuilt_line_that_differs_from_the_diff_beyond_its_mask_is_logged_at_warn() {
        // [MAGI S6 MS3 loop5, R6a Caspar] The side's text replaced a body line that disagreed
        // with it and nothing said so: a stale diff, or two revisions mixed (RV21), was papered
        // over. The file is still reviewed from its sides; the run log now says it diverged.
        use tracing_subscriber::layer::SubscriberExt;
        #[derive(Clone, Default)]
        struct Diverged(std::sync::Arc<std::sync::atomic::AtomicUsize>);
        impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for Diverged {
            fn on_event(
                &self,
                e: &tracing::Event<'_>,
                _: tracing_subscriber::layer::Context<'_, S>,
            ) {
                if e.metadata().target() == LOG_TARGET
                    && *e.metadata().level() == tracing::Level::WARN
                {
                    self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }
            }
        }
        let key = concat!("sk-ant-", "api03-", "AAAAAAAAAAAAAAAAAAAA"); // allow-secret-scan
        let new = format!("let k = \"{key}\";\nfn f() {{}}\n");
        let old = "let k = 1;\nfn f() {}\n";
        let run = |diff: &str| {
            let warns = Diverged::default();
            let _guard = tracing::subscriber::set_default(
                tracing_subscriber::registry().with(warns.clone()),
            );
            let inner = MapSource::new(&[("a.rs", new.as_str())], &[("a.rs", old)]);
            let src = CachedSource::new(&inner);
            let files = vec![ReviewFile {
                path: "a.rs".into(),
                diff: Some(diff.to_string()),
            }];
            let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
                .expect("kept");
            assert_eq!(f.kept.len(), 1);
            warns.0.load(std::sync::atomic::Ordering::SeqCst)
        };
        let agrees = format!("@@ -1,2 +1,2 @@\n-let k = 1;\n+let k = \"{key}\";\n fn f() {{}}\n");
        assert_eq!(run(&agrees), 0, "a masked line alone is no divergence");
        let stale = format!("@@ -1,2 +1,2 @@\n-let k = 1;\n+let k = \"{key}\";\n fn g() {{}}\n");
        assert_eq!(run(&stale), 1, "a stale context line is");
    }

    #[test]
    fn a_secret_skip_through_the_cache_refusal_is_logged_with_a_pattern() {
        // [MAGI S6 MS3 loop15, R6a1] the refusal arm of `read_side` skipped a file silently; every
        // `secret` skip logs the shown path and a pattern name (never the value).
        use tracing_subscriber::layer::SubscriberExt;
        #[derive(Clone, Default)]
        struct Infos(std::sync::Arc<std::sync::Mutex<Vec<String>>>);
        struct Pattern(Option<String>);
        impl tracing::field::Visit for Pattern {
            fn record_str(&mut self, f: &tracing::field::Field, v: &str) {
                if f.name() == "pattern" {
                    self.0 = Some(v.to_string());
                }
            }
            fn record_debug(&mut self, _: &tracing::field::Field, _: &dyn std::fmt::Debug) {}
        }
        impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for Infos {
            fn on_event(
                &self,
                e: &tracing::Event<'_>,
                _: tracing_subscriber::layer::Context<'_, S>,
            ) {
                if e.metadata().target() == LOG_TARGET
                    && *e.metadata().level() == tracing::Level::INFO
                {
                    let mut p = Pattern(None);
                    e.record(&mut p);
                    if let (Some(p), Ok(mut seen)) = (p.0, self.0.lock()) {
                        seen.push(p);
                    }
                }
            }
        }
        let infos = Infos::default();
        let _guard =
            tracing::subscriber::set_default(tracing_subscriber::registry().with(infos.clone()));
        let side = read_side(
            Err(ReviewError::EgressSecret {
                path: "k.pem".into(),
            }),
            "k.pem",
            false,
        )
        .expect("a skip");
        assert!(matches!(side, Side::Skip(SkipCause::Secret)));
        let seen = infos.0.lock().expect("lock").clone();
        assert_eq!(seen.len(), 1, "{seen:?}");
    }

    #[test]
    fn unrebuilt_diff_warns_and_masks() {
        // [MAGI S6 MS3 loop4, R6a Balthasar] The fallback branch itself: a masked file whose diff
        // disagrees with its sides (a body line without a marker) is left out, the safe
        // direction, and the run log says why at WARN so a parser drift is visible.
        // [MAGI S6 MS3 loop9] Its cause is `secret` (RV22); the warning still says why.
        use tracing_subscriber::layer::SubscriberExt;
        #[derive(Clone, Default)]
        struct Warns(std::sync::Arc<std::sync::atomic::AtomicUsize>);
        impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for Warns {
            fn on_event(
                &self,
                e: &tracing::Event<'_>,
                _: tracing_subscriber::layer::Context<'_, S>,
            ) {
                if e.metadata().target() == LOG_TARGET
                    && *e.metadata().level() == tracing::Level::WARN
                {
                    self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }
            }
        }
        let warns = Warns::default();
        let _guard =
            tracing::subscriber::set_default(tracing_subscriber::registry().with(warns.clone()));
        let key = concat!("sk-ant-", "api03-", "AAAAAAAAAAAAAAAAAAAA"); // allow-secret-scan
        let new = format!("let k = \"{key}\";\n");
        let inner = MapSource::new(&[("a.rs", new.as_str())], &[("a.rs", "let k = 1;\n")]);
        let src = CachedSource::new(&inner);
        let files = vec![ReviewFile {
            path: "a.rs".into(),
            diff: Some("@@ -1 +1 @@\nno marker here\n".into()),
        }];
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
            .expect("kept, masked (MS3 §6 loop11)");
        assert!(f.skipped.is_empty(), "{:?}", f.skipped);
        let sent = f.kept[0].diff.clone().expect("diff");
        assert!(!sent.contains("sk-ant-"), "{sent}");
        assert_eq!(warns.0.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn binaries_and_non_utf8_are_skipped_with_their_cause_and_unknown_text_is_kept() {
        let files = vec![
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
            ReviewFile {
                path: "logo.png".into(),
                diff: None,
            },
            ReviewFile {
                path: "conf.toml".into(),
                diff: None,
            },
            ReviewFile {
                path: "latin1.rs".into(),
                diff: None,
            },
            ReviewFile {
                path: "notes.txt".into(),
                diff: None,
            },
        ];
        let src = MapSource::with_bytes(&[
            ("a.rs", b"fn f() {}\n"),
            ("logo.png", b"\x89PNG\r\n\x1a\n\0\0"),
            ("conf.toml", b"a = 1\n"),
            ("latin1.rs", b"// caf\xe9\n"),
            ("notes.txt", b"plain\n"),
        ]);
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
            .expect("some kept");
        assert_eq!(
            f.kept.iter().map(|k| k.path.as_str()).collect::<Vec<_>>(),
            vec!["a.rs", "conf.toml", "notes.txt"]
        );
        assert_eq!(
            f.skipped,
            vec![
                Skipped {
                    path: "logo.png".into(),
                    cause: SkipCause::Binary
                },
                Skipped {
                    path: "latin1.rs".into(),
                    cause: SkipCause::NotUtf8
                }
            ]
        );
        // [CP2 MS2 S1 loop 3] config_files.rs is not ported (RV20(3), RV-S14): toml has no extractor either.
        assert_eq!(
            f.no_extractor,
            vec!["conf.toml".to_string(), "notes.txt".to_string()]
        );
    }

    #[test]
    fn an_explicit_binary_file_is_an_error_not_a_skip() {
        let files = vec![ReviewFile {
            path: "logo.png".into(),
            diff: None,
        }];
        let src = MapSource::with_bytes(&[("logo.png", b"\0\0\0")]);
        assert!(matches!(
            filter_files(files, &src, &LanguageRegistry::new(), true, &|_| false),
            Err(ReviewError::NotReviewable {
                cause: "binary",
                ..
            })
        ));
    }

    #[test]
    fn ignored_files_and_never_reviewed_paths_are_left_out() {
        let files = vec![
            ReviewFile {
                path: ".env".into(),
                diff: None,
            },
            ReviewFile {
                path: ".magi/magi.toml".into(),
                diff: None,
            },
            ReviewFile {
                path: "sub/.git".into(),
                diff: None,
            }, // [CP2 MS2 S1b loop 14] a worktree/submodule pointer file
            ReviewFile {
                path: ".magi-rs-memory.db-wal".into(),
                diff: None,
            }, // [S1b1 loop 16] the prefix rule
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
        ];
        // [CP2 MS2 S1b loop 13] `.env` holds a key: ignored wins over secret (the order above), so its cause is `ignored`.
        let env_text = "K=AKIAABCDEFGHIJKLMNOP\n"; // allow-secret-scan
        let src = MapSource::new(
            &[
                (".env", env_text),
                (".magi/magi.toml", "a=1\n"),
                ("sub/.git", "gitdir: ../.git/modules/sub\n"),
                (".magi-rs-memory.db-wal", "w\n"),
                ("a.rs", "fn f() {}\n"),
            ],
            &[],
        );
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|p| {
            p == ".env"
        })
        .expect("a.rs kept");
        assert_eq!(f.kept.len(), 1);
        assert!(f.skipped.iter().all(|s| s.cause == SkipCause::Ignored));
    }

    #[test]
    fn the_first_applicable_cause_wins_binary_then_not_utf8_then_secret() {
        let key = b"AKIAABCDEFGHIJKLMNOP"; // allow-secret-scan
        let bin: Vec<u8> = [b"\0\0".as_slice(), key].concat();
        let latin: Vec<u8> = [b"caf\xe9 ".as_slice(), key].concat();
        let files = vec![
            ReviewFile {
                path: "k.bin".into(),
                diff: None,
            },
            ReviewFile {
                path: "k.rs".into(),
                diff: None,
            },
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
        ];
        let src =
            MapSource::with_bytes(&[("k.bin", &bin), ("k.rs", &latin), ("a.rs", b"fn f() {}\n")]);
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
            .expect("a.rs kept");
        assert_eq!(
            f.skipped.iter().map(|s| s.cause).collect::<Vec<_>>(),
            vec![SkipCause::Binary, SkipCause::NotUtf8]
        );
    }

    #[test]
    fn git_binary_marker_is_confirmed_against_the_bytes_before_a_file_is_left_out() {
        // The section comes from real `git diff` output through the input's own splitter, so it
        // carries its `diff --git` and `index` header lines. `.gitattributes` makes git call the
        // file binary although its bytes are valid UTF-8 with no NUL.
        // [MAGI S6 loop 13, A2-m3 / A2-c1, security] the marker alone used to decide: a reviewed
        // commit that marked its own `evil.rs -diff` was listed as `binary` and never sent, so a
        // malicious change was certified unreviewed. The marker is now a hint: the sides are
        // read, and a text file whose diff git suppressed is reviewed whole. Modified from
        // `a_binary_section_as_git_prints_it_is_binary_without_reading_the_file`, which pinned
        // the opposite.
        fn git(dir: &Path, args: &[&str]) -> Vec<u8> {
            let out = std::process::Command::new("git")
                .arg("-C")
                .arg(dir)
                .args(args)
                .output()
                .expect("git runs");
            assert!(out.status.success(), "git {args:?}");
            out.stdout
        }
        let d = tempfile::tempdir().expect("tempdir");
        git(d.path(), &["init", "-q", "-b", "main"]);
        for (k, v) in [
            ("user.email", "t@example.invalid"),
            ("user.name", "t"),
            ("commit.gpgsign", "false"),
            ("core.autocrlf", "false"),
        ] {
            git(d.path(), &["config", k, v]);
        }
        std::fs::write(
            d.path().join(".gitattributes"),
            "*.dat binary\nevil.rs -diff\n",
        )
        .expect("write");
        std::fs::write(d.path().join("a.dat"), "hello\n").expect("write");
        std::fs::write(d.path().join("evil.rs"), "fn f() {}\n").expect("write");
        git(d.path(), &["add", "."]);
        git(d.path(), &["commit", "-q", "-m", "1"]);
        std::fs::write(d.path().join("a.dat"), "world\n").expect("write");
        std::fs::write(d.path().join("evil.rs"), "fn f() { danger() }\n").expect("write");
        let diff = git(
            d.path(),
            &["diff", "--no-color", "--no-ext-diff", "--no-textconv"],
        );
        let (sections, _) = crate::review::input::split_diff(&diff);
        let copy = |f: &ReviewFile| ReviewFile {
            path: f.path.clone(),
            diff: f.diff.clone(),
        };
        let section = |name: &str| {
            let s = sections
                .iter()
                .find(|f| f.path == name)
                .map(copy)
                .expect("git lists the file");
            assert!(
                s.diff.as_deref().is_some_and(is_binary_section),
                "git marks {name} binary: {s:?}"
            );
            s
        };
        let (dat, evil) = (section("a.dat"), section("evil.rs"));
        let both = || {
            vec![
                copy(&dat),
                ReviewFile {
                    path: "a.rs".into(),
                    diff: None,
                },
            ]
        };
        let binary = vec![Skipped {
            path: "a.dat".into(),
            cause: SkipCause::Binary,
        }];
        // [MAGI S6 loop 14, A2-4] When a side cannot be read, its own cause is named, as for any
        // file (`unreadable`, `too large`), and the review's total cap still fails the review: a
        // read failure became `binary`, and an `InputTooLarge` an omission.
        struct Unreadable(SkipOrFail);
        #[derive(Clone, Copy)]
        enum SkipOrFail {
            Fail,
            TooLarge,
            Total,
        }
        impl ContentSource for Unreadable {
            fn read(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                match (path, self.0) {
                    ("a.dat", SkipOrFail::Fail) => Err(ReviewError::Io("a.dat was read".into())),
                    ("a.dat", SkipOrFail::TooLarge) => Err(ReviewError::TooLarge {
                        path: path.into(),
                        limit: 1,
                    }),
                    ("a.dat", SkipOrFail::Total) => Err(ReviewError::InputTooLarge {
                        limit: 1,
                        held: 1,
                        path: path.into(),
                    }),
                    _ => Ok(Some(b"fn f() {}\n".to_vec())),
                }
            }
            fn read_old(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                self.read(path)
            }
        }
        let reg = LanguageRegistry::new();
        for (mode, cause) in [
            (SkipOrFail::Fail, SkipCause::Unreadable),
            (SkipOrFail::TooLarge, SkipCause::TooLarge),
        ] {
            let f = filter_files(both(), &Unreadable(mode), &reg, false, &|_| false)
                .expect("a.rs kept");
            assert_eq!(
                f.skipped,
                vec![Skipped {
                    path: "a.dat".into(),
                    cause
                }]
            );
        }
        let total = filter_files(both(), &Unreadable(SkipOrFail::Total), &reg, false, &|_| {
            false
        });
        assert!(
            matches!(total, Err(ReviewError::InputTooLarge { .. })),
            "{total:?}"
        );
        // Real binary bytes: still `binary`.
        let nul = MapSource::new(
            &[("a.dat", "wor\0ld\n"), ("a.rs", "fn f() {}\n")],
            &[("a.dat", "hello\n")],
        );
        let f = filter_files(both(), &nul, &reg, false, &|_| false).expect("a.rs kept");
        assert_eq!(f.skipped, binary);
        // Valid UTF-8 with no NUL: reviewed, whole, from its new side.
        let src = MapSource::new(
            &[
                ("a.dat", "world\n"),
                ("evil.rs", "fn f() { danger() }\n"),
                ("a.rs", "fn f() {}\n"),
            ],
            &[("a.dat", "hello\n"), ("evil.rs", "fn f() {}\n")],
        );
        let mut files = both();
        files.push(copy(&evil));
        let f = filter_files(files, &src, &reg, false, &|_| false).expect("all kept");
        assert_eq!(f.skipped, vec![]);
        assert_eq!(
            f.kept
                .iter()
                .map(|k| (k.path.as_str(), k.diff.is_none()))
                .collect::<Vec<_>>(),
            vec![("a.dat", true), ("a.rs", true), ("evil.rs", true)]
        );
        // [T3.7] A secret on the old side is masked, as for any diff; reviewed whole from its new side.
        let key = concat!("sk-", "proj-", "AAAAAAAAAAAAAAAAAAAAAAAA");
        let old_key = format!("const K: &str = \"{key}\";\n");
        let leaky_inner = MapSource::new(
            &[
                ("evil.rs", "fn f() { danger() }\n"),
                ("a.rs", "fn f() {}\n"),
            ],
            &[("evil.rs", old_key.as_str())],
        );
        let files = vec![
            evil,
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
        ];
        let leaky = CachedSource::new(&leaky_inner);
        let f = filter_files(files, &leaky, &reg, false, &|_| false).expect("both kept");
        assert_eq!(f.skipped, vec![]);
        assert_eq!(f.masked, vec![masked("evil.rs", "openai-key")]);
        assert!(
            f.kept
                .iter()
                .any(|k| k.path == "evil.rs" && k.diff.is_none()),
            "{f:?}"
        );
    }

    #[test]
    fn a_deleted_text_file_whose_diff_git_suppressed_is_reviewed_as_a_deletion() {
        // [Review residual A2-b] `gone.rs -diff` in `.gitattributes`, then `git rm gone.rs`:
        // git prints the section below (measured with git 2.56), with no hunk. The file was left
        // out as `binary`, a false cause, and its deleted lines were never reviewed.
        let section = concat!(
            "diff --git a/gone.rs b/gone.rs\n",
            "deleted file mode 100644\n",
            "index 83021ca..0000000\n",
            "Binary files a/gone.rs and /dev/null differ\n",
        );
        let deleted = || ReviewFile {
            path: "gone.rs".into(),
            diff: Some(section.into()),
        };
        let src = MapSource::new(&[], &[("gone.rs", "fn a() {}\nfn b() {}\n")]);
        let reg = LanguageRegistry::new();
        let f = filter_files(vec![deleted()], &src, &reg, false, &|_| false)
            .expect("the deletion is kept");
        assert_eq!(f.skipped, vec![]);
        let kept: Vec<_> = f.kept.iter().map(|k| k.path.as_str()).collect();
        assert_eq!(kept, vec!["gone.rs"]);
        let diff = f.kept.first().and_then(|k| k.diff.as_deref()).unwrap_or("");
        assert!(!is_binary_section(diff), "{diff:?}");
        assert!(
            diff.ends_with("@@ -1,2 +0,0 @@\n-fn a() {}\n-fn b() {}\n"),
            "{diff:?}"
        );
        // It is batched as any deletion: one hunk over the old lines 1-2.
        let batches = crate::review::batching::plan_batches(&f.kept, &src, &reg, 10_000)
            .expect("the deletion is batched");
        let hunks: Vec<_> = batches
            .iter()
            .flat_map(|b| &b.items)
            .filter_map(|i| match i {
                crate::review::batching::BatchItem::Hunk {
                    old_first_line,
                    old_last_line,
                    ..
                } => Some((*old_first_line, *old_last_line)),
                _ => None,
            })
            .collect();
        assert_eq!(hunks, vec![(1, 2)]);
        // [T3.7] A secret on the old side is masked in the rebuilt deletion, as for any deletion (RV22).
        let key = concat!("sk-", "proj-", "AAAAAAAAAAAAAAAAAAAAAAAA");
        let old_key = format!("const K: &str = \"{key}\";\n");
        let leaky_inner =
            MapSource::new(&[("a.rs", "fn f() {}\n")], &[("gone.rs", old_key.as_str())]);
        let leaky = CachedSource::new(&leaky_inner);
        let files = vec![
            deleted(),
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
        ];
        let f = filter_files(files, &leaky, &reg, false, &|_| false).expect("both kept");
        assert_eq!(f.skipped, vec![]);
        assert_eq!(f.masked, vec![masked("gone.rs", "openai-key")]);
        let diff = f.kept.first().and_then(|k| k.diff.as_deref()).unwrap_or("");
        assert!(
            diff.ends_with("@@ -1,1 +0,0 @@\n-const K: &str = \"<redacted: openai-key>\";\n"),
            "{diff:?}"
        );
    }

    #[test]
    fn a_binary_section_of_a_diff_is_skipped_as_binary() {
        let files = vec![
            ReviewFile {
                path: "logo.png".into(),
                diff: Some("Binary files a/logo.png and b/logo.png differ\n".into()),
            },
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
        ];
        let src = MapSource::new(&[("a.rs", "fn f() {}\n")], &[]);
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
            .expect("a.rs kept");
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: "logo.png".into(),
                cause: SkipCause::Binary
            }]
        );
    }

    #[test]
    fn an_explicit_ignored_file_is_an_error() {
        let files = vec![ReviewFile {
            path: ".env".into(),
            diff: None,
        }];
        let src = MapSource::new(&[(".env", "X=1\n")], &[]);
        assert!(matches!(
            filter_files(files, &src, &LanguageRegistry::new(), true, &|p| p
                == ".env"),
            Err(ReviewError::NotReviewable {
                cause: "ignored",
                ..
            })
        ));
    }

    #[test]
    fn a_tracked_file_that_gitignore_matches_is_left_out_of_a_directory_review() {
        // [CP2 MS2 S1b loop 10] `ls-files --cached` lists tracked files whatever .gitignore says (git 2.56, measured).
        use crate::review::input::ReviewInput;
        use crate::review::test_support::FakeGit;
        let git = FakeGit::from(&[
            ("rev-parse --is-inside-work-tree", Ok("true\n")),
            (
                "ls-files --cached --ignored --exclude-standard -z -- src",
                Ok("src/tracked.env\0"),
            ),
        ]);
        let ignored = ignored_paths(
            &ReviewInput::Directory("src".into()),
            "src",
            Path::new("/w"),
            &git,
        )
        .expect("one git call");
        let files = vec![
            ReviewFile {
                path: "src/tracked.env".into(),
                diff: None,
            },
            ReviewFile {
                path: "src/a.rs".into(),
                diff: None,
            },
        ];
        let src = MapSource::new(
            &[
                ("src/tracked.env", "HOST=internal\n"),
                ("src/a.rs", "fn f() {}\n"),
            ],
            &[],
        );
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|p| {
            ignored.contains(p)
        })
        .expect("a.rs kept");
        assert_eq!(
            f.kept.iter().map(|k| k.path.as_str()).collect::<Vec<_>>(),
            vec!["src/a.rs"]
        );
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: "src/tracked.env".into(),
                cause: SkipCause::Ignored
            }]
        );
    }

    #[test]
    fn a_missing_file_requested_by_name_is_missing_not_ignored() {
        // [CP2 MS2 S1b loop 10] a nonexistent file is not in the ignored set, so `missing` wins.
        use crate::review::input::ReviewInput;
        use crate::review::test_support::FakeGit;
        let git = FakeGit::from(&[
            ("rev-parse --is-inside-work-tree", Ok("true\n")),
            (
                "ls-files --cached --others --ignored --exclude-standard -z -- gone.rs",
                Ok(""),
            ),
            (
                "ls-files --cached --others --ignored --exclude-standard -z -- .env",
                Ok(".env\0"),
            ),
        ]);
        let gone = ignored_paths(
            &ReviewInput::File("gone.rs".into()),
            "gone.rs",
            Path::new("/w"),
            &git,
        )
        .expect("one git call");
        let one = vec![ReviewFile {
            path: "gone.rs".into(),
            diff: None,
        }];
        assert!(matches!(
            filter_files(
                one,
                &MapSource::new(&[], &[]),
                &LanguageRegistry::new(),
                true,
                &|p| gone.contains(p)
            ),
            Err(ReviewError::NotReviewable {
                cause: "missing",
                ..
            })
        ));
        let env = ignored_paths(
            &ReviewInput::File(".env".into()),
            ".env",
            Path::new("/w"),
            &git,
        )
        .expect("one git call");
        let one = vec![ReviewFile {
            path: ".env".into(),
            diff: None,
        }];
        let src = MapSource::new(&[(".env", "X=1\n")], &[]);
        assert!(matches!(
            filter_files(one, &src, &LanguageRegistry::new(), true, &|p| env
                .contains(p)),
            Err(ReviewError::NotReviewable {
                cause: "ignored",
                ..
            })
        ));
    }

    #[test]
    fn history_inputs_and_non_repos_ignore_nothing() {
        // [CP2 MS2 S1b loop 10] .gitignore is a worktree rule; an empty FakeGit fails any call it receives.
        use crate::review::input::ReviewInput;
        use crate::review::test_support::FakeGit;
        let none = FakeGit::from(&[]);
        for input in [
            ReviewInput::Commit(None),
            ReviewInput::Range {
                from: "a".into(),
                to: "b".into(),
            },
            ReviewInput::Branches {
                base: "a".into(),
                head: "b".into(),
            },
        ] {
            assert!(ignored_paths(&input, ".", Path::new("/w"), &none)
                .expect("no git call")
                .is_empty());
        }
        let outside = FakeGit::from(&[(
            "rev-parse --is-inside-work-tree",
            Err("fatal: not a git repository"),
        )]);
        let no_repo = tempfile::tempdir().expect("tempdir"); // no .git here or above
                                                             // [CP2 MS2 S1b loop 13] the assumption, made visible: a `.git` above TMPDIR makes the rule (correctly) fail closed.
        assert!(
            no_repo.path().ancestors().all(|a| !a.join(".git").exists()),
            "TMPDIR sits inside a git repository"
        );
        assert!(ignored_paths(
            &ReviewInput::Directory(".".into()),
            ".",
            no_repo.path(),
            &outside
        )
        .expect("not a repo")
        .is_empty());
    }

    #[test]
    fn history_inputs_ignore_nothing_even_inside_a_repository_whose_rules_match() {
        // The test above cannot see a git call for a history input: its empty FakeGit fails the
        // call, and outside a repository that failure reads as "ignore nothing". Here git
        // answers, and would name an ignored file, so only skipping git keeps the set empty.
        use crate::review::input::ReviewInput;
        use crate::review::test_support::FakeGit;
        let answering = FakeGit::from(&[
            ("rev-parse --is-inside-work-tree", Ok("true\n")),
            (
                "ls-files --cached --ignored --exclude-standard -z -- .",
                Ok(".env\0"),
            ),
            (
                "ls-files --cached --others --ignored --exclude-standard -z -- .",
                Ok(".env\0"),
            ),
        ]);
        for input in [
            ReviewInput::Commit(None),
            ReviewInput::Range {
                from: "a".into(),
                to: "b".into(),
            },
            ReviewInput::Branches {
                base: "a".into(),
                head: "b".into(),
            },
        ] {
            let ignored = ignored_paths(&input, ".", Path::new("/w"), &answering).expect("no git");
            assert!(ignored.is_empty(), "{input:?}: {ignored:?}");
        }
    }

    #[test]
    fn a_git_failure_inside_a_repository_is_an_error_not_an_empty_ignore_set() {
        // [CP2 MS2 S1a loop 11] an empty set here would review an explicitly requested ignored `.env`.
        use crate::review::input::ReviewInput;
        use crate::review::test_support::FakeGit;
        let d = tempfile::tempdir().expect("tempdir");
        std::fs::write(d.path().join(".git"), "gitdir: elsewhere\n")
            .expect("a worktree's .git file");
        let broken = FakeGit::from(&[(
            "rev-parse --is-inside-work-tree",
            Err("fatal: not a git repository"),
        )]);
        assert!(matches!(
            ignored_paths(&ReviewInput::File(".env".into()), ".env", d.path(), &broken),
            Err(ReviewError::Git(_))
        ));
    }

    #[test]
    fn a_diff_file_without_an_extractor_is_declared_too() {
        // [CP2 MS2 S1b loop 10] RV20(3) covers files reviewed by diff, not only whole files.
        let files = vec![ReviewFile {
            path: "conf.toml".into(),
            diff: Some("@@ -1,1 +1,1 @@\n-a = 1\n+a = 2\n".into()),
        }];
        let src = MapSource::new(&[("conf.toml", "a = 2\n")], &[("conf.toml", "a = 1\n")]);
        let f =
            filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false).expect("kept");
        assert_eq!(
            (f.kept.len(), f.no_extractor.clone()),
            (1, vec!["conf.toml".to_string()])
        );
    }

    #[test]
    fn a_missing_whole_file_is_never_skipped_silently() {
        let src = MapSource::new(&[], &[]);
        let one = || {
            vec![ReviewFile {
                path: "gone.rs".into(),
                diff: None,
            }]
        };
        assert!(matches!(
            filter_files(one(), &src, &LanguageRegistry::new(), true, &|_| false),
            Err(ReviewError::NotReviewable {
                cause: "missing",
                ..
            })
        ));
        // [MAGI S6 loop 12, A2-m5 / A2-c2] an enumerated file removed after the listing is an
        // omission with its own cause, not a failure of the whole review.
        match filter_files(one(), &src, &LanguageRegistry::new(), false, &|_| false) {
            Err(ReviewError::NothingToReview { skipped }) => assert_eq!(
                skipped,
                vec![Skipped {
                    path: "gone.rs".into(),
                    cause: SkipCause::Disappeared
                }]
            ),
            other => panic!("expected nothing to review, got {other:?}"),
        }
        // With a diff, an absent new side is a deletion and is kept.
        let del = vec![ReviewFile {
            path: "gone.rs".into(),
            diff: Some("@@ -1,1 +0,0 @@\n-x\n".into()),
        }];
        let src_old = MapSource::new(&[], &[("gone.rs", "x\n")]);
        assert_eq!(
            filter_files(del, &src_old, &LanguageRegistry::new(), false, &|_| false)
                .expect("kept")
                .kept
                .len(),
            1
        );
    }

    #[test]
    fn an_enumerated_file_that_cannot_be_read_is_left_out_and_a_named_one_fails() {
        // [MAGI S6 loop 12, A2-c2] a sharing violation (a file held open by a sync client or an
        // indexer) on one listed file failed the whole review after its preparation; RV20 omits
        // and declares. A file named explicitly still fails.
        struct Locked;
        impl ContentSource for Locked {
            fn read(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                if path == "locked.rs" {
                    Err(ReviewError::Io("sharing violation".into()))
                } else {
                    Ok(Some(b"fn f() {}\n".to_vec()))
                }
            }
            fn read_old(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Ok(None)
            }
        }
        let files = |names: &[&str]| {
            names
                .iter()
                .map(|n| ReviewFile {
                    path: (*n).into(),
                    diff: None,
                })
                .collect::<Vec<_>>()
        };
        let reg = LanguageRegistry::new();
        let f = filter_files(files(&["locked.rs", "a.rs"]), &Locked, &reg, false, &|_| {
            false
        })
        .expect("a.rs kept");
        assert_eq!(f.kept.len(), 1);
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: "locked.rs".into(),
                cause: SkipCause::Unreadable
            }]
        );
        assert_eq!(f.skipped[0].to_string(), "locked.rs (unreadable)");
        assert!(matches!(
            filter_files(files(&["locked.rs"]), &Locked, &reg, true, &|_| false),
            Err(ReviewError::Io(_))
        ));
    }

    #[test]
    fn a_skipped_file_is_not_charged_against_the_review_total() {
        // [MAGI S6 loop 12, A2-c1 / A2-c5] the cache kept every side the filter read, skipped
        // ones included: committed binaries crossed the total cap before the files under review
        // did, and a secret-bearing file's bytes stayed in memory for the whole run.
        let inner = MapSource::new(&[("big.dat", "\0abcdefghi"), ("a.rs", "fn f() {}\n")], &[]);
        let cache = crate::review::source::CachedSource::with_total_cap(&inner, 15);
        let files = ["big.dat", "a.rs"]
            .iter()
            .map(|n| ReviewFile {
                path: (*n).into(),
                diff: None,
            })
            .collect();
        let f = filter_files(files, &cache, &LanguageRegistry::new(), false, &|_| false)
            .expect("a.rs fits once big.dat is released");
        assert_eq!(f.kept.len(), 1);
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: "big.dat".into(),
                cause: SkipCause::Binary
            }]
        );
    }

    #[test]
    fn a_file_left_out_near_the_total_cap_does_not_fail_the_review() {
        // [MAGI S6 loop 15, A2-3] the cache checked the total before the filter judged the file:
        // with the kept files near the cap, reading a file that was then left out (a secret, a
        // binary) failed the whole review with `InputTooLarge`.
        // [T3.7] a secret no longer leaves a file out (it is masked), so the file left out after
        // its read is one that is not UTF-8.
        let kept = "fn f() {}\n";
        let other = "fn b() {}\n";
        let inner = MapSource::with_bytes(&[
            ("a.rs", kept.as_bytes()),
            ("key.rs", b"// caf\xe9\n"),
            ("big.dat", b"\0abcdefghijklmnopqrstuvwxyz"),
            ("b.rs", other.as_bytes()),
        ]);
        let cap = u64::try_from(kept.len() + other.len()).expect("fits");
        let cache = crate::review::source::CachedSource::with_total_cap(&inner, cap);
        let files = |names: &[&str]| -> Vec<ReviewFile> {
            names
                .iter()
                .map(|n| ReviewFile {
                    path: (*n).into(),
                    diff: None,
                })
                .collect()
        };
        let f = filter_files(
            files(&["a.rs", "key.rs", "big.dat", "b.rs"]),
            &cache,
            &LanguageRegistry::new(),
            false,
            &|_| false,
        )
        .expect("the omitted files are released before the total is checked");
        let kept_paths: Vec<&str> = f.kept.iter().map(|k| k.path.as_str()).collect();
        assert_eq!(kept_paths, ["a.rs", "b.rs"]);
        assert_eq!(
            f.skipped,
            vec![
                Skipped {
                    path: "key.rs".into(),
                    cause: SkipCause::NotUtf8
                },
                Skipped {
                    path: "big.dat".into(),
                    cause: SkipCause::Binary
                },
            ]
        );
        // What is kept still may not pass the cap: one byte less fails on the file that passes it.
        let tight = crate::review::source::CachedSource::with_total_cap(&inner, cap - 1);
        let err = filter_files(
            files(&["a.rs", "key.rs", "b.rs"]),
            &tight,
            &LanguageRegistry::new(),
            false,
            &|_| false,
        )
        .expect_err("a.rs and b.rs pass the cap");
        assert!(
            matches!(&err, ReviewError::InputTooLarge { path, .. } if path == "b.rs"),
            "{err:?}"
        );
    }

    #[test]
    fn an_empty_directory_is_nothing_to_review() {
        let src = MapSource::new(&[], &[]);
        assert!(matches!(
            filter_files(Vec::new(), &src, &LanguageRegistry::new(), false, &|_| {
                false
            }),
            Err(ReviewError::NothingToReview { .. })
        ));
    }

    #[test]
    fn a_non_empty_input_where_every_file_is_omitted_is_nothing_to_review() {
        // [CP2 MS2 S1b1 loop 16] RV20: the emptiness that counts is the kept list's, after every file was classified.
        // [T3.7] MapSource masks nothing, so the secret files take the fallback and stay out.
        let key = "const K: &str = \"AKIAABCDEFGHIJKLMNOP\";\n"; // allow-secret-scan
        let src = MapSource::new(&[(".env", "X=1\n"), ("a.rs", key), ("b.rs", key)], &[]);
        let ignored = vec![ReviewFile {
            path: ".env".into(),
            diff: None,
        }];
        assert!(matches!(
            filter_files(ignored, &src, &LanguageRegistry::new(), false, &|p| p
                == ".env"),
            Err(ReviewError::NothingToReview { .. })
        ));
        let secrets = vec![
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
            ReviewFile {
                path: "b.rs".into(),
                diff: None,
            },
        ];
        assert!(matches!(
            filter_files(secrets, &src, &LanguageRegistry::new(), false, &|_| false),
            Err(ReviewError::NothingToReview { .. })
        ));
    }

    #[test]
    fn an_all_omitted_input_keeps_every_omission_and_its_cause_in_the_error() {
        // [MAGI S6 loop 3, A2] RV20/RV22: with nothing kept, the omissions are still reported.
        // [T3.7] MapSource masks nothing, so the secret file takes the fallback and stays out.
        let key = "const K: &str = \"AKIAABCDEFGHIJKLMNOP\";
"; // allow-secret-scan
        let src = MapSource::new(
            &[
                (
                    ".env", "X=1
",
                ),
                ("a.rs", key),
                ("b.bin", "a b"),
            ],
            &[],
        );
        let files = [".env", "a.rs", "b.bin"]
            .iter()
            .map(|p| ReviewFile {
                path: (*p).into(),
                diff: None,
            })
            .collect();
        let e = filter_files(files, &src, &LanguageRegistry::new(), false, &|p| {
            p == ".env"
        })
        .expect_err("nothing kept");
        let ReviewError::NothingToReview { skipped } = &e else {
            panic!("expected NothingToReview, got {e:?}");
        };
        let causes: Vec<(&str, SkipCause)> =
            skipped.iter().map(|s| (s.path.as_str(), s.cause)).collect();
        assert_eq!(
            causes,
            vec![
                (".env", SkipCause::Ignored),
                ("a.rs", SkipCause::Secret),
                ("b.bin", SkipCause::Binary)
            ]
        );
        let shown = e.to_string();
        for part in [".env (ignored)", "a.rs (secret)", "b.bin (binary)"] {
            assert!(shown.contains(part), "{shown}");
        }
    }

    #[test]
    fn a_file_whose_target_is_protected_is_ignored_and_an_explicit_one_is_an_error() {
        // [CP2 MS2 S1b1 loop 16] T2.3's Protected: a symlink alias of a never-reviewed path.
        struct Aliased;
        impl ContentSource for Aliased {
            fn read(&self, p: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                if p == "notes.rs" {
                    Err(ReviewError::Protected(p.into()))
                } else {
                    Ok(Some(b"fn f() {}\n".to_vec()))
                }
            }
            fn read_old(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Ok(None)
            }
        }
        let files = vec![
            ReviewFile {
                path: "notes.rs".into(),
                diff: None,
            },
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
        ];
        let f = filter_files(files, &Aliased, &LanguageRegistry::new(), false, &|_| false)
            .expect("a.rs kept");
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: "notes.rs".into(),
                cause: SkipCause::Ignored
            }]
        );
        let one = vec![ReviewFile {
            path: "notes.rs".into(),
            diff: None,
        }];
        assert!(matches!(
            filter_files(one, &Aliased, &LanguageRegistry::new(), true, &|_| false),
            Err(ReviewError::NotReviewable {
                cause: "ignored",
                ..
            })
        ));
    }

    #[test]
    fn a_text_diff_that_quotes_the_binary_marker_is_still_text() {
        // [CP2 MS2 S1b1 loop 16] the marker is the whole section, never a line inside a hunk.
        let diff = "@@ -1,1 +1,2 @@\n x\n+Binary files a/logo.png and b/logo.png differ\n";
        let files = vec![ReviewFile {
            path: "notes.md".into(),
            diff: Some(diff.into()),
        }];
        let src = MapSource::new(
            &[(
                "notes.md",
                "x\nBinary files a/logo.png and b/logo.png differ\n",
            )],
            &[("notes.md", "x\n")],
        );
        let f =
            filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false).expect("kept");
        assert!(f.skipped.is_empty() && f.kept.len() == 1, "{f:?}");
    }
    #[test]
    fn an_old_side_that_is_binary_or_not_utf8_skips_the_file_even_if_the_new_one_is_text() {
        // [MAGI S6 MS2 loop 4, A2 Caspar] the checks ran on the new side only, so a text file
        // whose previous revision was binary or Latin-1 carried old-side bytes into the review.
        struct OldBytes;
        impl ContentSource for OldBytes {
            fn read(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Ok(Some(b"fn h() {}\n".to_vec()))
            }
            fn read_old(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Ok(Some(match path {
                    "bin.rs" => b"fn g() {}\0\n".to_vec(),
                    "latin1.rs" => b"// caf\xe9\nfn g() {}\n".to_vec(),
                    _ => b"fn g() {}\n".to_vec(),
                }))
            }
        }
        let diff = "@@ -2,1 +1,1 @@\n-fn g() {}\n+fn h() {}\n";
        let files = ["bin.rs", "latin1.rs", "a.rs"]
            .into_iter()
            .map(|p| ReviewFile {
                path: p.into(),
                diff: Some(diff.into()),
            })
            .collect();
        let f = filter_files(files, &OldBytes, &LanguageRegistry::new(), false, &|_| {
            false
        })
        .expect("a.rs kept");
        assert_eq!(
            f.skipped,
            vec![
                Skipped {
                    path: "bin.rs".into(),
                    cause: SkipCause::Binary
                },
                Skipped {
                    path: "latin1.rs".into(),
                    cause: SkipCause::NotUtf8
                }
            ]
        );
    }

    #[test]
    fn a_diff_file_with_neither_side_is_skipped_as_not_a_regular_file() {
        // [MAGI S6 loop 8, A2-c1 Caspar] a submodule's gitlink reads as absent on both sides
        // (`RevisionSource::holds`), so a diff section that reached the filter in that shape was
        // kept with no content and no cause. `split_diff` drops gitlink sections first; this pins
        // the filter's own answer for whatever brings that shape.
        let files = vec![
            ReviewFile {
                path: "vendored".into(),
                diff: Some("@@ -1 +1 @@\n-Subproject commit 1\n+Subproject commit 2\n".into()),
            },
            ReviewFile {
                path: "a.rs".into(),
                diff: Some("@@ -1 +1 @@\n-fn f() {}\n+fn g() {}\n".into()),
            },
        ];
        let src = MapSource::new(&[("a.rs", "fn g() {}\n")], &[("a.rs", "fn f() {}\n")]);
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
            .expect("a.rs kept");
        assert_eq!(
            f.kept.iter().map(|k| k.path.as_str()).collect::<Vec<_>>(),
            vec!["a.rs"]
        );
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: "vendored".into(),
                cause: SkipCause::NotRegularFile
            }]
        );
    }

    #[test]
    fn a_nul_anywhere_in_a_file_makes_it_binary() {
        // [MAGI S6 loop 7, A2-c2] only the first 8 000 bytes were sniffed: a text prefix hid a
        // binary payload, which is valid UTF-8 and was batched (RV20).
        let mut late: Vec<u8> = vec![b'a'; 9_000];
        late.extend_from_slice(b"\0payload\n");
        let files = vec![
            ReviewFile {
                path: "late.rs".into(),
                diff: None,
            },
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
        ];
        let src = MapSource::with_bytes(&[("late.rs", &late), ("a.rs", b"fn f() {}\n")]);
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
            .expect("a.rs kept");
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: "late.rs".into(),
                cause: SkipCause::Binary,
            }]
        );
    }

    #[test]
    fn a_rename_from_a_never_reviewed_or_ignored_path_is_ignored() {
        // [MAGI S6 loop 7, A2-m2] only the new path met the name rule: a crafted commit carried a
        // blob stored under `.git/` or an ignored name into the review as a rename's old side.
        let rename = |from: &str, to: &str| {
            format!(
                "diff --git a/{from} b/{to}\nsimilarity index 90%\nrename from {from}\n\
                 rename to {to}\n--- a/{from}\n+++ b/{to}\n@@ -1,1 +1,1 @@\n-fn g() {{}}\n+fn h() {{}}\n"
            )
        };
        let src = MapSource::new(
            &[
                ("x.rs", "fn h() {}\n"),
                ("y.rs", "fn h() {}\n"),
                ("a.rs", "fn a() {}\n"),
            ],
            &[
                (".git/hooks/x.rs", "fn g() {}\n"),
                ("local.env", "fn g() {}\n"),
            ],
        );
        let files = vec![
            ReviewFile {
                path: "x.rs".into(),
                diff: Some(rename(".git/hooks/x.rs", "x.rs")),
            },
            ReviewFile {
                path: "y.rs".into(),
                diff: Some(rename("local.env", "y.rs")),
            },
            ReviewFile {
                path: "a.rs".into(),
                diff: None,
            },
        ];
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|p| {
            p == "local.env"
        })
        .expect("a.rs kept");
        assert_eq!(
            f.skipped
                .iter()
                .map(|s| (s.path.as_str(), s.cause))
                .collect::<Vec<_>>(),
            vec![("x.rs", SkipCause::Ignored), ("y.rs", SkipCause::Ignored)]
        );
    }

    #[test]
    fn a_skipped_rename_never_evicts_the_sides_a_kept_file_was_scanned_with() {
        // [MAGI S6 loop 13, A2-m2 / A2-b1] `release` dropped both sides held under one path, and
        // a skipped rename released its old path too: when that path was a kept file of the same
        // review, its scanned bytes were evicted and a later stage read the disk again, so what
        // was batched was no longer what the detector saw. Kept a.rs (scanned first), then a
        // rename a.rs -> b.rs skipped for a secret: a.rs must stay held on both sides.
        use std::sync::Mutex;
        struct Counting {
            inner: MapSource,
            reads: Mutex<Vec<(bool, String)>>,
        }
        impl ContentSource for Counting {
            fn read(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                self.reads
                    .lock()
                    .expect("lock")
                    .push((true, path.to_string()));
                self.inner.read(path)
            }
            fn read_old(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                self.reads
                    .lock()
                    .expect("lock")
                    .push((false, path.to_string()));
                self.inner.read_old(path)
            }
        }
        // [T3.7] a secret is now masked and its file kept, so the rename is left out as binary.
        let leaked = "x\0\n".to_string();
        let counting = Counting {
            inner: MapSource::new(
                &[("a.rs", "fn a() { 1 }\n"), ("b.rs", leaked.as_str())],
                &[("a.rs", "fn a() {}\n")],
            ),
            reads: Mutex::new(Vec::new()),
        };
        let cache = crate::review::source::CachedSource::new(&counting);
        let files = vec![
            ReviewFile {
                path: "a.rs".into(),
                diff: Some("@@ -1 +1 @@\n-fn a() {}\n+fn a() { 1 }\n".into()),
            },
            ReviewFile {
                path: "b.rs".into(),
                diff: Some(
                    "diff --git a/a.rs b/b.rs\nsimilarity index 50%\nrename from a.rs\n\
                     rename to b.rs\n--- a/a.rs\n+++ b/b.rs\n@@ -1 +1 @@\n-fn a() {}\n+x\n"
                        .into(),
                ),
            },
        ];
        let f = filter_files(files, &cache, &LanguageRegistry::new(), false, &|_| false)
            .expect("a.rs kept");
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: "b.rs".into(),
                cause: SkipCause::Binary
            }]
        );
        cache.read("a.rs").expect("read");
        cache.read_old("a.rs").expect("read");
        let reads = counting.reads.lock().expect("lock").clone();
        for side in [true, false] {
            let n = reads
                .iter()
                .filter(|(s, p)| *s == side && p == "a.rs")
                .count();
            assert_eq!(n, 1, "side new={side} of a.rs was read again: {reads:?}");
        }
    }

    #[test]
    fn a_file_reviewed_whole_from_its_new_side_is_never_charged_for_its_old_side() {
        // [MAGI S6 loop 17, A2-2] a text file whose diff git printed as binary is kept with no
        // diff and reviewed from its new side alone, but the old side read to judge it stayed in
        // the cache and counted against the review total, so many such files failed a review as
        // `InputTooLarge` whose reviewed bytes fit the cap.
        let inner = MapSource::new(
            &[(
                "a.txt",
                "0123456789
",
            )],
            &[(
                "a.txt",
                "abcdefghij
",
            )],
        );
        let cache = crate::review::source::CachedSource::with_total_cap(&inner, 15);
        let files = vec![ReviewFile {
            path: "a.txt".into(),
            diff: Some(
                "Binary files a/a.txt and b/a.txt differ
"
                .into(),
            ),
        }];
        let f = filter_files(files, &cache, &LanguageRegistry::new(), false, &|_| false)
            .expect("the new side alone fits the cap");
        assert_eq!(
            f.kept
                .iter()
                .map(|k| (k.path.as_str(), k.diff.is_none()))
                .collect::<Vec<_>>(),
            vec![("a.txt", true)]
        );
        assert!(f.skipped.is_empty(), "{:?}", f.skipped);
        assert!(cache.settle().is_ok(), "only the new side is held");
    }
}
