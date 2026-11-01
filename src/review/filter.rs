// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! What never leaves the process and what is not code: the per-file filter of a review.
//!
//! Every file of a review is classified here before any byte of it is batched: a path a review
//! never reads or `.gitignore` excludes, a path or content that holds a secret, a file over the
//! byte cap, a binary and a text that is not UTF-8 are left out with their cause; everything
//! else is kept, and a kept file with no language extractor is declared.

use std::collections::HashSet;
use std::path::Path;

use super::error::ReviewError;
use super::input::{
    in_work_tree, is_binary_section, renamed_from, split_nul, ReviewFile, ReviewInput,
};
use super::languages::LanguageRegistry;
use super::secrets::{display_path, find_secret};
use super::source::{is_never_reviewed, ContentSource, Face, GitCommand, HARD_LINK_CAUSE};

/// The cause of [`ReviewError::NotReviewable`] for a file requested by name that does not exist.
pub(crate) const MISSING: &str = "missing";
/// The tracing target of the filter's run-log entries.
const LOG_TARGET: &str = "magi_rs::review";
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
/// `outside workspace`, `too large`, `not a regular file`, `hard link`, `unreadable` and
/// `disappeared`. A file named explicitly that is skipped fails with that same text as the cause
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
    fn as_str(self) -> &'static str {
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
}

/// What classifying one file decided.
enum Verdict {
    /// Reviewed.
    Keep,
    /// Reviewed whole, from its new side: a text file whose diff git printed as binary.
    KeepWhole,
    /// Left out, with its cause.
    Skip(SkipCause),
}

/// Splits the files of a review into what is reviewed and what is left out.
///
/// For each file, in this order, the first cause that applies wins: a path a review never
/// reads (a `.git` or `.magi` component, or a name that starts with `.magi-rs-memory.db`,
/// whatever `is_ignored` says) or one `is_ignored` names, as the new path or as a rename's old
/// path → `ignored`; a path that holds a
/// secret → `secret`, without reading it; a file that does not exist → an error, never a skip;
/// over the byte cap → `too large`; a NUL anywhere in a side → `binary`; not UTF-8 →
/// `not utf-8`; a secret anywhere in the new side, the old side or the diff text → `secret`.
/// A diff section whose header lines (before the first `@@`) carry git's binary marker
/// ([`is_binary_section`]) is judged by its bytes the same way, and when both sides are text it
/// is kept with no diff, reviewed whole from its new side; a side that cannot be read is left
/// out under its own cause, as for any file, and it stays `binary` only when the file has no new
/// side (`suppressed_diff`).
/// The binary and UTF-8 checks run on both sides that exist (binary first). The
/// secret scan covers both whole sides, not only what is sent: the safe direction. A renamed
/// file's old side is read under its old path (`rename from`).
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
/// The kept files, the skipped ones and the kept ones with no extractor. Every path that
/// leaves the process goes through [`display_path`]. Each `secret` skip writes one INFO entry
/// with the shown path and the pattern's name, never the value.
///
/// # Errors
///
/// [`ReviewError::NotReviewable`] when `explicit` and the file is skipped or missing,
/// [`ReviewError::Io`] when a file named explicitly cannot be read, [`ReviewError::NothingToReview`]
/// when no file is kept, and any other error of `source`.
///
/// O(total bytes read) time.
pub fn filter_files(
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
    };
    // The sides a kept file was scanned with: never released, whoever else names their path.
    let mut kept_sides: HashSet<(Face, String)> = HashSet::new();
    for file in files {
        match classify(&file, source, is_ignored, explicit)? {
            Verdict::Keep => {
                if registry.get(&file.path).is_none() {
                    filtered.no_extractor.push(display_path(&file.path));
                }
                kept_sides.extend(sides_of(&file));
                filtered.kept.push(file);
            }
            Verdict::KeepWhole => {
                if registry.get(&file.path).is_none() {
                    filtered.no_extractor.push(display_path(&file.path));
                }
                kept_sides.extend(sides_of(&file));
                filtered.kept.push(ReviewFile {
                    path: file.path,
                    diff: None,
                });
            }
            Verdict::Skip(cause) if explicit => {
                return Err(ReviewError::NotReviewable {
                    path: display_path(&file.path),
                    cause: cause.as_str(),
                })
            }
            Verdict::Skip(cause) => {
                // MAGI FIX (S6 loop 12, A2-c1 / A2-c5): a skipped file is never sent, so the
                // cache keeps none of it: the total cap bounds what is reviewed.
                // MAGI FIX (S6 loop 13, A2-m2 / A2-b1): side by side, and never a side a kept file
                // was scanned with: a rename's old path can be a kept file's path.
                for (face, path) in sides_of(&file) {
                    if !kept_sides.contains(&(face, path.clone())) {
                        source.release(face, &path);
                    }
                }
                filtered.skipped.push(Skipped {
                    path: display_path(&file.path),
                    cause,
                });
            }
        }
    }
    if filtered.kept.is_empty() {
        // MAGI FIX (S6 loop 3): the omissions and their causes travel with the error (RV20/RV22).
        return Err(ReviewError::NothingToReview {
            skipped: filtered.skipped,
        });
    }
    Ok(filtered)
}

/// The sides [`classify`] reads for `file`, each with the path it is read under: the new side
/// under its path and, for a diff, the old side under its `rename from` path or its own. O(1).
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
    file: &ReviewFile,
    source: &dyn ContentSource,
    is_ignored: &dyn Fn(&str) -> bool,
    explicit: bool,
) -> Result<Verdict, ReviewError> {
    let path = file.path.as_str();
    // A renamed or copied file's old side lives under its old path.
    let old_path = file.diff.as_deref().and_then(renamed_from);
    // MAGI FIX (S6 loop 7): the old path meets the same name rule as the new one: a crafted
    // commit carried a blob stored under `.git/` (or an ignored name) in as a rename's old side.
    if [Some(path), old_path.as_deref()]
        .into_iter()
        .flatten()
        .any(|p| is_never_reviewed(p) || is_ignored(p))
    {
        return Ok(Verdict::Skip(SkipCause::Ignored));
    }
    if let Some(name) = find_secret(path) {
        return Ok(secret(path, name));
    }
    if file.diff.as_deref().is_some_and(is_binary_section) {
        return suppressed_diff(file, source, old_path.as_deref(), explicit);
    }
    let new = match read_side(source.read(path), path, explicit)? {
        Side::Content(bytes) => bytes,
        Side::Skip(cause) => return Ok(Verdict::Skip(cause)),
    };
    // MAGI FIX (S6 loop 14, A2-5): the old side's read failure is logged under the path it was
    // read under (a rename's old path), not the new one, which reads fine.
    let old_side = old_path.as_deref().unwrap_or(path);
    let old = match &file.diff {
        Some(_) => match read_side(source.read_old(old_side), old_side, explicit)? {
            Side::Content(bytes) => bytes,
            Side::Skip(cause) => return Ok(Verdict::Skip(cause)),
        },
        None => None,
    };
    // MAGI FIX (S6 loop 8): a diff with neither side is a gitlink (a submodule reads as absent
    // on both sides) or a path gone from both revisions; it was kept with no content and no
    // cause. `split_diff` drops gitlink sections before this, so this is the filter's own answer.
    if new.is_none() && old.is_none() && file.diff.is_some() {
        return Ok(Verdict::Skip(SkipCause::NotRegularFile));
    }
    // MAGI FIX (S6 loop 11): a dangling link reads as absent here too, but it never reaches this
    // branch as "disappeared": every listing of `input` moves enumerated links whose target is
    // gone to `ResolvedInput::dangling_links` before the filter runs (`without_dangling`, pinned
    // by `an_input_left_with_only_dangling_links_is_nothing_to_review_and_names_them`), and they
    // are reported as `SkipCause::NotRegularFile`. Only a file removed after listing lands here.
    if new.is_none() && file.diff.is_none() {
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
    Ok(
        content_verdict(path, [new.as_deref(), old.as_deref()], file.diff.as_deref())
            .unwrap_or(Verdict::Keep),
    )
}

/// The skip the content of a file calls for, or `None` when it may be reviewed: a NUL in a side
/// gives `binary`, a side that is not UTF-8 `not utf-8`, a secret in a side or in `diff`
/// `secret`, in that order. O(bytes).
///
/// # Arguments
///
/// * `path` - The file's path, for the `secret` log entry.
/// * `sides` - The new and old sides that exist.
/// * `diff` - The file's diff section, if any.
///
/// # Returns
///
/// The skip, or `None`.
// MAGI FIX (S6 loop 4): both sides, not the new one alone: a text file whose old side was
// binary or Latin-1 carried old-side bytes into the review.
// MAGI FIX (S6 loop 7): the whole side, not git's first 8 000 bytes: a text prefix hid a
// binary payload, which is valid UTF-8 (NUL is a code point). One pass over bytes already
// read and capped (MAX_REVIEW_FILE_BYTES), so the cost stays O(file).
fn content_verdict(path: &str, sides: [Option<&[u8]>; 2], diff: Option<&str>) -> Option<Verdict> {
    if sides.iter().flatten().any(|side| side.contains(&0)) {
        return Some(Verdict::Skip(SkipCause::Binary));
    }
    if sides
        .iter()
        .flatten()
        .any(|side| std::str::from_utf8(side).is_err())
    {
        return Some(Verdict::Skip(SkipCause::NotUtf8));
    }
    let [new, old] = sides;
    [new, old, diff.map(str::as_bytes)]
        .into_iter()
        .flatten()
        .find_map(|face| find_secret(&String::from_utf8_lossy(face)))
        .map(|name| secret(path, name))
}

/// The verdict on a file whose diff section git printed as binary ([`is_binary_section`]).
///
/// The marker is a hint, never the answer: git prints it for any file the `diff` attribute
/// turns off (`-diff`, `binary`), and the reviewed tree writes its own `.gitattributes`, so a
/// change could mark itself out of the review. Both sides are read and judged by their bytes
/// ([`content_verdict`]); a text file passes and is reviewed whole, from its new side, since
/// git printed no hunks for it (one INFO entry says so). A side that fails to read is mapped
/// as for any file ([`read_side`]): it is left out under its own cause (`unreadable`, `too
/// large`, `outside workspace`, …), and an error that is not a skip (the review's total cap)
/// fails the review. Only a file with no new side (a deletion) stays `binary`, the label the
/// section carries.
///
/// # Arguments
///
/// * `file` - The file.
/// * `source` - The review's content source.
/// * `old_path` - Its `rename from` path, if any.
/// * `explicit` - Whether the file was requested by name.
///
/// # Returns
///
/// [`Verdict::KeepWhole`] or the skip.
///
/// # Errors
///
/// Every read error that [`read_side`] does not map to a skip.
// MAGI FIX (S6 loop 13, A2-m3 / A2-c1, security): `evil.rs -diff` in the reviewed commit left
// `evil.rs` out as `binary` without a byte of it read.
// MAGI FIX (S6 loop 14, A2-4): a failed read was `binary`, and the total cap's `InputTooLarge`
// an omission that could end as `NothingToReview`; each now goes through `read_side`.
fn suppressed_diff(
    file: &ReviewFile,
    source: &dyn ContentSource,
    old_path: Option<&str>,
    explicit: bool,
) -> Result<Verdict, ReviewError> {
    let path = file.path.as_str();
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
    if let Some(skip) = content_verdict(path, sides, file.diff.as_deref()) {
        return Ok(skip);
    }
    if new.is_none() {
        return Ok(Verdict::Skip(SkipCause::Binary));
    }
    let shown = display_path(path);
    tracing::event!(
        target: LOG_TARGET,
        tracing::Level::INFO,
        path = shown.as_str(),
        "git printed a text file's diff as binary; it is reviewed whole"
    );
    Ok(Verdict::KeepWhole)
}

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
/// stays an error. Nothing of the file is sent either way.
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
/// excludes, which the enumeration lists (untracked ignored files are never enumerated).
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
    use crate::review::test_support::MapSource;

    #[test]
    fn a_secret_in_a_whole_file_skips_it_and_an_explicit_one_is_an_error() {
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
    fn a_secret_only_in_the_old_side_skips_the_file() {
        // The diff and the new side are clean; only the revision before the change holds the key.
        // [CP2 MS2 S1b loop 15] a kept sibling makes the skip and its `secret` cause observable (RV-S30, second clause).
        let diff = "@@ -10,1 +10,1 @@\n-fn g() {}\n+fn h() {}\n";
        let old = "const K: &str = \"AKIAABCDEFGHIJKLMNOP\";\nfn g() {}\n"; // allow-secret-scan
        let src = MapSource::new(
            &[("a.rs", "fn h() {}\n"), ("b.rs", "fn b() {}\n")],
            &[("a.rs", old)],
        );
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
            .expect("b.rs kept");
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: "a.rs".into(),
                cause: SkipCause::Secret
            }]
        );
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
    fn a_secret_on_the_old_side_of_a_renamed_file_skips_it() {
        // [MAGI §6 MS2 A loop 1] a rename's old side lives under its old path: read by the new path it was empty,
        // and a key on a removed line or outside the hunk reached a model.
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
        let src = MapSource::new(
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
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
            .expect("a.rs kept");
        assert_eq!(
            f.skipped,
            vec![
                Skipped {
                    path: "new.rs".into(),
                    cause: SkipCause::Secret
                },
                Skipped {
                    path: "k_new.rs".into(),
                    cause: SkipCause::Secret
                }
            ]
        );
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
                "\"not a regular file\""
            ]
        );
    }

    #[test]
    fn a_pem_key_whose_header_is_outside_the_hunk_still_skips_the_file() {
        // [CP2 MS2 S1b loop 11] the hunk touches only the key body; both whole sides are scanned, so `BEGIN` is seen.
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
        let src = MapSource::new(
            &[("key.rs", new.as_str()), ("a.rs", "fn f() {}\n")],
            &[("key.rs", old.as_str())],
        );
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
            .expect("a.rs kept");
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: "key.rs".into(),
                cause: SkipCause::Secret
            }]
        );
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
        // Split so tests/no_hardcoded_secrets.rs (which flags the literal `-----BEGIN`) reads no
        // header here; the body is obviously fake.
        let begin = concat!("-----", "BEGIN PGP PRIVATE KEY BLOCK-----");
        let end = concat!("-----", "END PGP PRIVATE KEY BLOCK-----");
        let body = "lQOYBGZ0FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE";
        let text = format!("const K: &str = r\"\n{begin}\n\n{body}\n=FAKE\n{end}\n\";\n");
        let public = concat!("-----", "BEGIN PGP PUBLIC KEY BLOCK-----");
        let pubtext = format!("{public}\n\n{body}\n");
        let src = MapSource::new(
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
    fn a_secret_omission_logs_the_path_and_the_pattern_name_never_the_value() {
        // [CP2 MS2 S1b loop 9] the consumer of SecretPattern.name (loop 8), pinned.
        let key = "ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"; // allow-secret-scan
        let text = format!("const K: &str = \"{key}\";\n");
        let src = MapSource::new(
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
        let line = crate::logging::testutil::capture(|| {
            filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
                .expect("a.rs kept");
        });
        assert!(
            line.contains("config.rs") && line.contains("github-token"),
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
    fn a_secret_in_a_removed_line_skips_the_file() {
        // [CP2 MS2 S1b loop 15] with a kept sibling, so the skip and its `secret` cause are asserted, not inferred.
        let diff =
            "@@ -1,2 +1,1 @@\n-const K: &str = \"sk-proj-AAAAAAAAAAAAAAAAAAAAAAAA\";\n fn f() {}\n"; // allow-secret-scan
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
        let src = MapSource::new(&[("a.rs", "fn f() {}\n"), ("b.rs", "fn b() {}\n")], &[]);
        let f = filter_files(files, &src, &LanguageRegistry::new(), false, &|_| false)
            .expect("b.rs kept");
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: "a.rs".into(),
                cause: SkipCause::Secret
            }]
        );
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
        // A secret on the old side still keeps the file out, as for any diff.
        let key = concat!("sk-", "proj-", "AAAAAAAAAAAAAAAAAAAAAAAA");
        let old_key = format!("const K: &str = \"{key}\";\n");
        let leaky = MapSource::new(
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
        let f = filter_files(files, &leaky, &reg, false, &|_| false).expect("a.rs kept");
        assert_eq!(
            f.skipped,
            vec![Skipped {
                path: "evil.rs".into(),
                cause: SkipCause::Secret
            }]
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
        let key = concat!("sk-", "proj-", "AAAAAAAAAAAAAAAAAAAAAAAA");
        let leaked = format!("const K: &str = \"{key}\";\n");
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
                cause: SkipCause::Secret
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
}
