// Author: Julian Bolivar
// Version: 0.25.0
// Date: 2026-11-08

//! The single content source of a review: every stage that reads code reads it through here.
//!
//! A review reads its code from exactly one place, chosen once from the input: a git revision
//! ([`RevisionSource`], `git show <rev>:./<path>`) or the worktree ([`WorktreeSource`]), with an
//! optional old side for deleted lines. Nothing mixes the worktree with another revision.

use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use super::error::ReviewError;
use super::secrets::{
    find_secret, holds_key_block, mask_secrets_keeping_lines_within, BoundedMask,
};

/// Per-file byte cap: a file over it is reported as [`ReviewError::TooLarge`], never returned.
pub const MAX_REVIEW_FILE_BYTES: u64 = 8 * 1024 * 1024;
/// Byte cap of any one git output; past it git is stopped ([`ReviewError::GitOutputTooLarge`]).
pub const MAX_GIT_OUTPUT_BYTES: u64 = 64 * 1024 * 1024;
/// Bytes of a git call's stderr that are kept for the error message.
pub const MAX_GIT_STDERR_BYTES: u64 = 4 * 1024;
/// Byte cap of everything one review reads.
pub const MAX_REVIEW_TOTAL_BYTES: u64 = 256 * 1024 * 1024;
/// The cause a working-tree file with more than one hard link is refused with
/// ([`ReviewError::NotReviewable`]); the filter leaves an enumerated one out under it.
pub const HARD_LINK_CAUSE: &str = "hard link";
/// The git argument after which every argument is positional (git 2.24+): a revision placed
/// after it is never read as an option (`--output=<file>` would write a file).
const END_OF_OPTIONS: &str = "--end-of-options";
/// The object type `git ls-tree` prints for a file (a submodule is `commit`).
const LS_TREE_BLOB: &str = "blob";
/// The memory database's file name; its `-wal`/`-shm` companions share the prefix.
const MEMORY_DB_PREFIX: &str = ".magi-rs-memory.db";
/// Names a review never reads: the git and Magi directories and the memory database.
pub const NEVER_REVIEWED: &[&str] = &[".git", ".magi", MEMORY_DB_PREFIX];

/// Runs git for a review.
///
/// The production implementation neutralizes the reviewed repository's config and environment;
/// test doubles answer by the exact argument list.
pub trait GitCommand: Send + Sync {
    /// Runs git with `args` and returns its stdout as bytes.
    ///
    /// # Errors
    ///
    /// [`ReviewError::Git`] when git cannot run or exits with an error, and
    /// [`ReviewError::GitOutputTooLarge`] when its output passes the cap.
    fn run_bytes(&self, args: &[&str]) -> Result<Vec<u8>, ReviewError>;

    /// Runs git with `args` and returns its stdout, which may not pass `cap` bytes.
    ///
    /// The default checks the length after [`GitCommand::run_bytes`], which is what a test
    /// double needs; the production runner stops git at `cap` instead of reading past it.
    ///
    /// # Errors
    ///
    /// As [`GitCommand::run_bytes`], and [`ReviewError::GitOutputTooLarge`] with `cap` as the
    /// limit when the output passes it.
    // MAGI FIX (S6 loop 4): a blob over the per-file cap was read whole under the 64 MiB cap.
    fn run_bytes_capped(&self, args: &[&str], cap: u64) -> Result<Vec<u8>, ReviewError> {
        let out = self.run_bytes(args)?;
        if u64::try_from(out.len()).unwrap_or(u64::MAX) > cap {
            return Err(ReviewError::GitOutputTooLarge { limit: cap });
        }
        Ok(out)
    }

    /// Runs git and decodes its stdout as UTF-8.
    ///
    /// # Errors
    ///
    /// As [`GitCommand::run_bytes`], and [`ReviewError::Git`] when the output is not UTF-8.
    fn run(&self, args: &[&str]) -> Result<String, ReviewError> {
        String::from_utf8(self.run_bytes(args)?).map_err(|_| {
            ReviewError::Git(format!(
                "git {} printed output that is not UTF-8",
                args.first().copied().unwrap_or_default()
            ))
        })
    }

    /// Runs git with `args` against an empty repository of its own whose work tree is the
    /// workdir, for a directory outside any repository: git's ignore rules (`.gitignore` at every
    /// level, the operator's excludes file) then apply as they would inside one.
    ///
    /// The default runs `args` as [`GitCommand::run_bytes`] does, which is what a test double
    /// answering by argument list needs.
    ///
    /// # Errors
    ///
    /// As [`GitCommand::run_bytes`], and [`ReviewError::Io`] when the empty repository cannot be
    /// made.
    fn run_bytes_without_repository(&self, args: &[&str]) -> Result<Vec<u8>, ReviewError> {
        self.run_bytes(args)
    }

    /// Runs a query that git answers with exit status 1 and nothing on stderr when nothing
    /// matches (`rev-parse --verify --quiet`), and decodes its stdout as UTF-8.
    ///
    /// The default reads that answer from [`GitCommand::run`] as a [`ReviewError::Git`] with an
    /// empty message, which is how an implementation that reports git's stderr verbatim sees it;
    /// an implementation that words its failures overrides this with the exit status. The
    /// production runner (`ScrubbedGit`, `src/system/review_git.rs`) does: it decides by exit 1
    /// with an empty stderr, so wording its errors never turns a no-match into a failure. The
    /// default serves test doubles; pinned by
    /// `the_default_query_reads_only_an_empty_git_error_as_no_match`.
    // MAGI FIX (S6 loop 15, A2-6): the contract was implicit, and read as load-bearing for the
    // production runner, which overrides it.
    ///
    /// # Returns
    ///
    /// `Ok(None)` for "no match", `Ok(Some(stdout))` on success.
    ///
    /// # Errors
    ///
    /// As [`GitCommand::run`], for every other failure.
    fn query(&self, args: &[&str]) -> Result<Option<String>, ReviewError> {
        match self.run(args) {
            Ok(out) => Ok(Some(out)),
            Err(ReviewError::Git(msg)) if msg.is_empty() => Ok(None),
            Err(other) => Err(other),
        }
    }
}

/// Reads the content of a review, as bytes (binary and non-UTF-8 files are filtered later).
///
/// # Line contract
///
/// Every stage that reads through a source (hunks, the reviewed ranges of RV09, units, the gate
/// state) numbers lines the way git does: a line ends at `\n` only. A `\r\n` is one line end,
/// and stages that split with `str::lines` drop its `\r`. A lone `\r` is not a line break; it
/// stays inside its line. Counting as git counts is what keeps the hunk ranges git reports in
/// agreement with the lines every stage shows and checks. The consequence, kept on purpose: a
/// file with classic-Mac endings (lone `\r` only) is one line and so one unit; when that unit
/// does not fit a batch it is a typed error (RV03), never a silent cut. Pinned by
/// `a_lone_carriage_return_is_not_a_line_break` and
/// `a_crlf_source_still_splits_into_its_real_lines` (`review::languages::python`).
// MAGI FIX (B2b2): the lone-`\r` behaviour was undeclared; normalizing it here would number
// lines differently from git's hunks.
pub trait ContentSource: Send + Sync {
    /// Reads `path` (relative, `/`-separated) from the new side, or the worktree.
    ///
    /// # Returns
    ///
    /// `Ok(None)` when the path does not exist on that side.
    ///
    /// # Errors
    ///
    /// Implementation specific; see [`WorktreeSource`] and [`RevisionSource`].
    fn read(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError>;

    /// Reads `path` from the old side.
    ///
    /// # Returns
    ///
    /// `Ok(None)` when there is no old side or the path does not exist on it.
    ///
    /// # Errors
    ///
    /// As [`ContentSource::read`].
    fn read_old(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError>;

    /// Drops whatever is held for `path` on side `face` only, so a file the review left out
    /// costs nothing afterwards. A no-op for a source that holds nothing (the default); a later
    /// read of that side of `path` reads the inner source again.
    ///
    /// One side at a time because a path is not private to one file: a rename's old path can be
    /// another file's new path, and releasing both sides under it evicted the bytes that file was
    /// scanned with.
    ///
    /// # Arguments
    ///
    /// * `face` - The side to forget.
    /// * `path` - The path to forget on that side.
    // MAGI FIX (S6 loop 12, A2-c1 / A2-c5): the cache kept the sides of skipped files, charged
    // against the review total and, for a secret-bearing file, held in memory for the whole run.
    // MAGI FIX (S6 loop 13, A2-m2 / A2-b1): side-specific, so a skipped rename's old path no
    // longer evicts another file's new side.
    fn release(&self, face: Face, path: &str) {
        let _ = (face, path);
    }

    /// Checks that what the source holds, once the caller released what it left out, fits the
    /// review's total cap. A caller that reads a file to judge it calls this after the verdict
    /// and its releases, so a file it left out is never charged against the total.
    ///
    /// The default holds nothing and never fails.
    ///
    /// # Errors
    ///
    /// Implementation specific; see [`CachedSource`]'s [`ReviewError::InputTooLarge`].
    // MAGI FIX (S6 loop 15, A2-3): the total was checked at the read, before the filter judged the
    // file, so a secret or binary file near the cap failed a review it would have left out.
    fn settle(&self) -> Result<(), ReviewError> {
        Ok(())
    }

    /// Ends the filter's classification (§5.0 step 2): from now on, no side held raw with a
    /// secret in it is served, whether the filter read it or not. The filter has to read such a
    /// side to leave its file out; no later stage may. The default holds nothing and does
    /// nothing.
    fn seal(&self) {}

    /// What this source masked in side `face` of `path` (RV22): the secrets of a held text side
    /// are replaced by `<redacted: …>`, line breaks kept, before any stage reads it. A side that
    /// holds a multi-line private-key block is never masked (RV22's exception), so the filter
    /// leaves its file out.
    ///
    /// The default masks nothing and answers `None`; the filter then leaves out any file whose
    /// content still holds a secret, as it did before masking existed.
    ///
    /// # Arguments
    ///
    /// * `face` - The side.
    /// * `path` - The path it was read under.
    ///
    /// # Returns
    ///
    /// The mask, or `None` when the side held no secret or was never read.
    fn masked(&self, face: Face, path: &str) -> Option<SideMask> {
        let _ = (face, path);
        None
    }
}

/// What a source masked in one side of a file ([`ContentSource::masked`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SideMask {
    /// The name of the first secret pattern found in the side (table order).
    pub pattern: &'static str,
}

/// Reads all of `reader`, but never more than `cap + 1` bytes.
///
/// # Arguments
///
/// * `reader` - What to read.
/// * `cap` - The largest accepted size, in bytes.
///
/// # Returns
///
/// `Some(bytes)` when the content fits in `cap`, `None` when it is larger (what was read is
/// dropped). O(min(n, cap)) time and memory.
///
/// Limit: `cap` must be below `u64::MAX`. At `u64::MAX` the extra byte that tells "over" from
/// "exactly at" saturates away, so a longer content reads as `Some`; no reader can hold that
/// many bytes, and every caller passes a per-file or per-output cap (8 MiB, 64 MiB).
// MAGI FIX (S6 loop 15, A2-7): the saturating bound was undocumented.
///
/// # Errors
///
/// The reader's I/O error.
///
/// # Examples
///
/// ```
/// use magi_rs::review::source::read_bounded;
///
/// assert_eq!(read_bounded(&b"abc"[..], 3).unwrap(), Some(b"abc".to_vec()));
/// assert_eq!(read_bounded(&b"abcd"[..], 3).unwrap(), None);
/// ```
pub fn read_bounded(reader: impl Read, cap: u64) -> std::io::Result<Option<Vec<u8>>> {
    let mut buf = Vec::new();
    reader.take(cap.saturating_add(1)).read_to_end(&mut buf)?;
    let len = u64::try_from(buf.len()).unwrap_or(u64::MAX);
    Ok((len <= cap).then_some(buf))
}

/// Whether a path is one a review never reads.
///
/// True for a component `.git` or `.magi`, or any component that starts with
/// `.magi-rs-memory.db` (a directory named like the database hides nothing either). Components split on `/` and `\`, compared ASCII-case-insensitively after
/// trimming trailing `.` and ` ` (Windows resolves `.GIT`, `.git.` and `.git ` to `.git`; the
/// same rule everywhere over-blocks a `.Git` directory on Linux, the safe direction).
///
/// # Examples
///
/// ```
/// use magi_rs::review::source::is_never_reviewed;
///
/// assert!(is_never_reviewed("sub/.GIT/config"));
/// assert!(!is_never_reviewed(".gitignore"));
/// ```
pub fn is_never_reviewed(path: &str) -> bool {
    let components: Vec<&str> = path
        .split(['/', '\\'])
        .map(|c| c.trim_end_matches(['.', ' ']))
        .collect();
    let named = components
        .iter()
        .any(|c| NEVER_REVIEWED.iter().any(|n| c.eq_ignore_ascii_case(n)));
    // MAGI FIX (S6 loop 14, A2-2, security): every component, not the last one alone:
    // `.magi-rs-memory.db-wal/notes.txt` was reviewed.
    let database = components.iter().any(|c| {
        c.get(..MEMORY_DB_PREFIX.len())
            .is_some_and(|head| head.eq_ignore_ascii_case(MEMORY_DB_PREFIX))
    });
    named || database
}

/// Reads the worktree under a root, never outside it.
///
/// A root whose canonical path has a never-reviewed component (a `.git` or `.magi` directory, or
/// an alias resolving into one) reads nothing: every read is [`ReviewError::Protected`].
///
/// Each read canonicalizes `root.join(path)`, checks by path components that it stays under the
/// canonical root, refuses a resolved target that is a never-reviewed path (a symlink alias of
/// `.git/config`), and reads that canonical path with the per-file cap. After the open the path
/// is resolved again and must still lie under the root and name the very file opened (same
/// device and inode, or volume and file index): a component swapped for a link between the
/// check and the open is refused as outside the workspace.
///
/// A hard link names its inode from inside the workspace whatever file it is, so a file with
/// more than one link is refused too, on every platform (`link_count`), under its own cause:
/// [`ReviewError::NotReviewable`] with [`HARD_LINK_CAUSE`].
pub struct WorktreeSource<'g> {
    /// The root as given; paths are joined to it (a verbatim canonical root would not resolve `..`).
    root: PathBuf,
    /// The canonical root, or the error that kept it from resolving (every read then fails).
    canonical_root: Result<PathBuf, String>,
    /// The old side, when the worktree is compared against a commit.
    old: Option<RevisionSource<'g>>,
}

impl<'g> WorktreeSource<'g> {
    /// A worktree source with no old side.
    ///
    /// Infallible: if `root` cannot be canonicalized the error is kept, and every read returns it.
    pub fn new(root: PathBuf) -> Self {
        let canonical_root = std::fs::canonicalize(&root).map_err(|e| e.to_string());
        Self {
            root,
            canonical_root,
            old: None,
        }
    }

    /// A worktree source whose old side is the commit `old_rev` (`git show <old_rev>:./<path>`).
    pub fn with_old(root: PathBuf, git: &'g dyn GitCommand, old_rev: String) -> Self {
        Self {
            old: Some(RevisionSource::new(git, old_rev, None)),
            ..Self::new(root)
        }
    }
}

impl ContentSource for WorktreeSource<'_> {
    /// # Errors
    ///
    /// [`ReviewError::Io`] when the root did not resolve or a read fails,
    /// [`ReviewError::OutsideWorkspace`] when the path resolves outside the root or no longer
    /// names the file opened (see [`WorktreeSource`]), [`ReviewError::NotReviewable`] with
    /// [`HARD_LINK_CAUSE`] for a file with more than one hard link,
    /// [`ReviewError::Protected`] when it resolves to a never-reviewed path or the root is one,
    /// [`ReviewError::NotRegularFile`] when it resolves to something other than a regular file
    /// (a link to a directory), and [`ReviewError::TooLarge`] over [`MAX_REVIEW_FILE_BYTES`].
    /// A link whose target does not exist is `Ok(None)`, like a file that does not exist.
    fn read(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
        self.read_with(path, &open_for_review)
    }

    fn read_old(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
        match &self.old {
            Some(old) => old.read(path),
            None => Ok(None),
        }
    }
}

impl WorktreeSource<'_> {
    /// [`ContentSource::read`] with the file opened by `open`, so a test can stand in for a
    /// path swapped between the check and the open.
    ///
    /// # Errors
    ///
    /// As [`ContentSource::read`].
    fn read_with(
        &self,
        path: &str,
        open: &dyn Fn(&std::path::Path) -> std::io::Result<std::fs::File>,
    ) -> Result<Option<Vec<u8>>, ReviewError> {
        let io = |e: std::io::Error| ReviewError::Io(format!("{path}: {e}"));
        let canonical_root = self
            .canonical_root
            .as_ref()
            .map_err(|e| ReviewError::Io(format!("{}: {e}", self.root.display())))?;
        // MAGI FIX (S6 loop 16, A2-1, security): the containment check below strips the root, and
        // with it a `.git` / `.magi` component of the root itself (or of the directory an alias
        // root resolves to): under such a root nothing is read.
        if is_never_reviewed(&canonical_root.to_string_lossy()) {
            return Err(ReviewError::Protected(path.to_string()));
        }
        let joined = self.root.join(path);
        match std::fs::symlink_metadata(&joined) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(io(e)),
            Ok(_) => {}
        }
        // A link whose target is gone reads as a file that does not exist.
        let canonical = match std::fs::canonicalize(&joined) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            other => other.map_err(io)?,
        };
        let Ok(relative) = canonical.strip_prefix(canonical_root) else {
            return Err(ReviewError::OutsideWorkspace(path.to_string()));
        };
        if is_never_reviewed(&relative.to_string_lossy()) {
            return Err(ReviewError::Protected(path.to_string()));
        }
        // MAGI FIX (S6 loop 14, A2-6): a file removed between the canonicalization and the open
        // reads as absent, as one removed a moment earlier does, not as an I/O failure.
        let meta = match std::fs::metadata(&canonical) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            other => other.map_err(io)?,
        };
        if !meta.is_file() {
            return Err(ReviewError::NotRegularFile(path.to_string()));
        }
        let file = match open(&canonical) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            other => other.map_err(io)?,
        };
        // Checked again on the handle: the path may have been swapped between the check above
        // and the open (a FIFO, a device file).
        let opened_meta = file.metadata().map_err(io)?;
        if !opened_meta.is_file() {
            return Err(ReviewError::NotRegularFile(path.to_string()));
        }
        // MAGI FIX (S6 loop 6): a hard link's path is inside the workspace, its inode may be any
        // file on the filesystem; the containment check cannot tell, so more than one link is
        // refused.
        // MAGI FIX (S6 loop 7): on Windows too (it read the outside bytes there).
        // MAGI FIX (S6 loop 12, A2-b1 / A2-c4): refused under its own cause; `outside workspace`
        // named an escape the path does not have.
        if link_count(&file, &opened_meta).map_err(io)? > 1 {
            return Err(ReviewError::NotReviewable {
                path: path.to_string(),
                cause: HARD_LINK_CAUSE,
            });
        }
        let opened = same_file::Handle::from_file(file).map_err(io)?;
        // Re-resolved after the open: a component swapped for an escaping link before the open
        // either still escapes, or names another file than the one opened.
        // MAGI FIX (S6 loop 16, A2-4): a file unlinked right after the open reads as absent here
        // too, as it does at every earlier step; nothing of it is read either way.
        // MAGI FIX (review residual A2-c): any other failure is the I/O error it is, as at every
        // earlier step; it was `outside workspace`, an escape the path never had.
        let now = match std::fs::canonicalize(&joined) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            other => other.map_err(io)?,
        };
        let still_inside = now
            .strip_prefix(canonical_root)
            .is_ok_and(|r| !is_never_reviewed(&r.to_string_lossy()));
        if !still_inside {
            return Err(ReviewError::OutsideWorkspace(path.to_string()));
        }
        // MAGI FIX (S6 loop 7): the path is opened again through `open_for_review`, never a
        // plain open: a FIFO swapped in after the first open opens at once and fails the
        // identity comparison instead of blocking the preparation.
        let reopened = match open_for_review(&now) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            other => other.map_err(io)?,
        };
        let named = same_file::Handle::from_file(reopened).map_err(io)?;
        if named != opened {
            return Err(ReviewError::OutsideWorkspace(path.to_string()));
        }
        match read_bounded(opened.as_file(), MAX_REVIEW_FILE_BYTES).map_err(io)? {
            Some(bytes) => Ok(Some(bytes)),
            None => Err(ReviewError::TooLarge {
                path: path.to_string(),
                limit: MAX_REVIEW_FILE_BYTES,
            }),
        }
    }
}

/// How many hard links name the open file.
///
/// On Unix the count of its metadata (`st_nlink`). On Windows std exposes the count only on
/// nightly (`windows_by_handle`), so it is read through `winapi-util`, which wraps
/// `GetFileInformationByHandle` with no `unsafe` of ours (already in the build through
/// `same-file`).
///
/// # Arguments
/// * `file` - The open file.
/// * `meta` - Its metadata (the Unix count).
///
/// # Returns
/// The number of links; 1 for a file no hard link shares.
///
/// # Errors
/// On Windows, any error of the handle query.
#[cfg(unix)]
fn link_count(_file: &std::fs::File, meta: &std::fs::Metadata) -> std::io::Result<u64> {
    Ok(std::os::unix::fs::MetadataExt::nlink(meta))
}

/// See the Unix [`link_count`].
#[cfg(windows)]
fn link_count(file: &std::fs::File, _meta: &std::fs::Metadata) -> std::io::Result<u64> {
    Ok(winapi_util::file::information(file)?.number_of_links())
}

/// Opens a worktree file for reading without ever blocking in the open.
///
/// On unix the open carries `O_NONBLOCK`: a path swapped for a FIFO between the type check and
/// the open returns at once instead of waiting for a writer (the caller then refuses the handle,
/// which is not a regular file). On a regular file the flag changes nothing. Elsewhere it is a
/// plain open: Windows has no file type whose open blocks this way.
///
/// # Arguments
/// * `path` - The canonical path to open.
///
/// # Returns
/// The open file.
///
/// # Errors
/// Any error of the open.
// MAGI FIX (S6 loop 3): TOCTOU, a FIFO swapped in after the `is_file` check blocked the open.
pub fn open_for_review(path: &std::path::Path) -> std::io::Result<std::fs::File> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        let nonblock = i32::try_from(rustix::fs::OFlags::NONBLOCK.bits())
            .map_err(|_| std::io::Error::other("O_NONBLOCK does not fit an open flag"))?;
        options.custom_flags(nonblock);
    }
    options.open(path)
}

/// Reads a git revision (`git show <rev>:./<path>`), never the worktree.
///
/// "Absent" is decided by the revision's own entry for the path
/// (`ls-tree -z --end-of-options <rev> -- <path>`, one call per read), never by the text of a
/// git error: a path with no file entry is `Ok(None)`, and a failing `show` on a listed path is
/// an error. Every revision argument follows `--end-of-options`, so the source is safe on its
/// own: a rev shaped like an option (`--output=<file>`) is never one, whatever the caller checked.
///
/// Cost: two git processes per (side, path) that exists, one per absent side, each once thanks
/// to [`CachedSource`]. Measured 2026-10-31 on Windows 11 with git 2.56 (a 300-file commit, from
/// a shell loop whose own spawn cost a trivial executable puts at ~34 ms): ~58 ms per `ls-tree`
/// or `show`, so ~35 s for the new side of 300 added files and ~17 s more for their absent old
/// side, off the clock (RV06) and reported as preparation time (RV13); a few percent of one
/// batch's trio time. One `git cat-file --batch` session over stdin would answer type, size and
/// content in one process with the same per-path bound, but [`GitCommand`] has no stdin channel
/// and its production runner lives in `src/system/review_git.rs`; left as a follow-up.
// MAGI FIX (S6 loop 12, A2-m2 / A2-b3): measured and documented, not changed (see above).
///
/// A symlink (mode `120000`) is a blob whose content is its target path, so a revision review
/// reads that path as the file's text, where the worktree source follows the link: the same
/// logical file can read differently on the two sides. Nothing extra leaves the process: the
/// target string is repository data and passes the same secret scan (RV22).
///
/// Paths are workspace-relative and every call runs `git -C <workspace>`, so `<rev>:./<path>`
/// resolves from the workspace: a workspace below the repository root reads its own subtree,
/// never a same-named path at the root. That is why the source takes no root (unlike
/// [`WorktreeSource`]); pinned with the real git by
/// `a_workspace_below_the_repo_root_reviews_only_its_own_subtree` (S6 loop 7, A1-m3).
pub struct RevisionSource<'g> {
    git: &'g dyn GitCommand,
    new_rev: String,
    old_rev: Option<String>,
}

impl<'g> RevisionSource<'g> {
    /// A source over `new_rev`, with `old_rev` as its old side.
    pub fn new(git: &'g dyn GitCommand, new_rev: String, old_rev: Option<String>) -> Self {
        Self {
            git,
            new_rev,
            old_rev,
        }
    }

    /// Whether `rev` holds `path` as a file: `ls-tree -z <rev> -- <path>` lists the entry for
    /// that path alone (pathspecs are literal: `GIT_LITERAL_PATHSPECS`), so the output is one
    /// record or none whatever the size of the repository. O(1) records per call.
    ///
    /// The listed name is compared as bytes with the requested path, never decoded: a name that
    /// is not UTF-8 never stands for a requested path, which is always UTF-8. Only files count:
    /// a submodule (a gitlink, type `commit`) is not content of the revision, so a path that
    /// names one there reads as absent, like in the worktree; a directory (type `tree`) too.
    ///
    /// No `-r` is needed for a nested path: a pathspec makes `ls-tree` descend the trees it
    /// names and list the blob itself (`100644 blob <id>\tsrc/a/b.rs`, measured with git 2.56),
    /// pinned by `a_revision_file_in_a_nested_directory_is_read` (`src/system/review_git.rs`).
    // MAGI FIX (S6 loop 10): A2-c1 refuted with real git; the nested-path read is pinned.
    // MAGI FIX (S6 loop 7): the whole revision was listed (`ls-tree -r`) to answer for one path,
    // so a monorepo past the git output cap failed the review before any model call.
    fn holds(&self, rev: &str, path: &str) -> Result<bool, ReviewError> {
        // MAGI FIX (S6 loop 12, A2-c3): `rev` follows `--end-of-options` (git 2.24+), so a rev
        // shaped like an option is a revision here, whatever its caller validated.
        let out = self
            .git
            .run_bytes(&["ls-tree", "-z", END_OF_OPTIONS, rev, "--", path])?;
        // Each record is `<mode> SP <type> SP <object> TAB <path>`.
        Ok(out.split(|b| *b == 0).any(|record| {
            record
                .iter()
                .position(|b| *b == b'\t')
                .and_then(|tab| Some((record.get(..tab)?, record.get(tab + 1..)?)))
                .is_some_and(|(meta, name)| {
                    // MAGI FIX (S6 loop 4): a gitlink listed here failed the review on `git show`.
                    meta.split(|b| *b == b' ').nth(1) == Some(LS_TREE_BLOB.as_bytes())
                        && name == path.as_bytes()
                })
        }))
    }

    /// Reads `path` at `rev`, with the per-file cap.
    ///
    /// A path that is empty, absolute or holds a `..` is refused before any git call
    /// ([`ReviewError::OutsideWorkspace`]): `show <rev>:./<path>` resolves against the workspace,
    /// so `..` would read a file outside it, and the source now holds that line itself instead of
    /// relying on its callers (git-derived paths never hold one; explicit ones pass the guard).
    /// The rule is the input's own `is_confined` (`src/review/input.rs`), restated here because
    /// that one is private to the listing.
    ///
    /// A `.` component (`./a.rs`, `src/./b.rs`) is dropped before any git call: `ls-tree` names
    /// a path without one, so the check that `rev` holds the path ([`Self::holds`]) compares
    /// the path as git names it. A path of `.` components alone names no file and is refused
    /// like an empty one.
    // MAGI FIX (S6 loop 13, A2-c4): the revision side enforced no containment of its own.
    fn read_at(&self, rev: &str, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
        let confined = Path::new(path)
            .components()
            .all(|c| matches!(c, Component::Normal(_) | Component::CurDir));
        // MAGI FIX (S6 loop 17, A2-3): `./a.rs` never matched `ls-tree`'s `a.rs`, so a path named
        // with a leading `./` read as absent from the revision.
        let parts: Option<Vec<&str>> = Path::new(path)
            .components()
            .filter_map(|c| match c {
                Component::Normal(part) => Some(part.to_str()),
                _ => None,
            })
            .collect();
        let named = parts.map(|p| p.join("/")).unwrap_or_default();
        if !confined || named.is_empty() {
            return Err(ReviewError::OutsideWorkspace(path.to_string()));
        }
        let path = named.as_str();
        if !self.holds(rev, path)? {
            return Ok(None);
        }
        let too_large = || ReviewError::TooLarge {
            path: path.to_string(),
            limit: MAX_REVIEW_FILE_BYTES,
        };
        let spec = format!("{rev}:./{path}");
        match self.git.run_bytes_capped(
            &[
                "show",
                "--no-ext-diff",
                "--no-textconv",
                END_OF_OPTIONS,
                &spec,
            ],
            MAX_REVIEW_FILE_BYTES,
        ) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(ReviewError::GitOutputTooLarge { .. }) => Err(too_large()),
            Err(ReviewError::Git(msg)) => Err(ReviewError::Git(format!(
                "cannot read {path} at {rev}: {msg}; if this is a partial clone, fetch its objects \
                 first (git fetch --refetch, or a full clone)"
            ))),
            Err(other) => Err(other),
        }
    }
}

impl ContentSource for RevisionSource<'_> {
    /// # Errors
    ///
    /// [`ReviewError::Git`] when git fails (a missing blob names the path and the partial-clone
    /// remedy), [`ReviewError::TooLarge`] over [`MAX_REVIEW_FILE_BYTES`], and
    /// [`ReviewError::GitOutputTooLarge`] when the path list itself passes the git output cap.
    fn read(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
        self.read_at(&self.new_rev, path)
    }

    fn read_old(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
        match &self.old_rev {
            Some(rev) => self.read_at(rev, path),
            None => Ok(None),
        }
    }
}

/// Memoizes each successful read of an inner source, by side and path, for one review.
///
/// Every stage of a review (the filter, the batches, the index and the gate) reads through one
/// cache, so a file is read once per side and every stage sees the bytes the filter scanned,
/// even if the file changes on disk meanwhile. A text side is held with each one-line secret
/// masked, line breaks kept (RV22; [`ContentSource::masked`] says what was masked). The exception
/// is a side held raw with a secret in it (a multi-line private-key block, a text too dense to
/// mask, a binary side): safety comes from the filter leaving its file out, and the cache
/// enforces it. A released side that was, or on a later read would be, held raw with a secret is
/// refused ([`ReviewError::EgressSecret`]); a side withheld at its release stays refused even if
/// the file later loses the header that made it recognisable (its body matches no pattern). Once
/// the filter sealed the cache ([`ContentSource::seal`]), no such side is served to any stage,
/// read by the filter or not, so a stage reading beyond the kept files cannot get one either. What
/// no pattern recognises (a key body with no header anywhere in the file) is beyond any of these
/// checks, as it is beyond the detector. An `Err` is never kept, and the filter releases
/// every side of a file it leaves out that no kept file was scanned with
/// ([`ContentSource::release`]), so the cap and the memory count only
/// what is reviewed and no skipped secret stays held. The bytes held are
/// bounded by [`MAX_REVIEW_TOTAL_BYTES`], checked by [`ContentSource::settle`] once the caller
/// released what it left out; until then a read may pass it by both sides of one file at the
/// per-file cap ([`MAX_REVIEW_FILE_BYTES`]),
/// and one that would go further is refused and not kept. A read of what is already held costs
/// nothing. Memory is O(bytes reviewed); a lookup is
/// O(1) expected plus the copy it returns.
pub struct CachedSource<'a> {
    inner: &'a dyn ContentSource,
    cap: u64,
    /// How far past `cap` a read may go until [`ContentSource::settle`].
    headroom: u64,
    /// What is held. A `Mutex` because [`ContentSource`] reads through `&self` and must be
    /// `Send + Sync`; its scope, and why it is never contended, is documented at `get`.
    held: Mutex<Held>,
}

/// What a [`CachedSource`] holds.
#[derive(Default)]
struct Held {
    /// New-side reads by path.
    new: HashMap<String, Option<Vec<u8>>>,
    /// Old-side reads by path.
    old: HashMap<String, Option<Vec<u8>>>,
    /// What was masked in each held side, by side and path (RV22).
    masks: HashMap<(Face, String), SideMask>,
    /// Sides released while they were held raw with a secret in them ([`HeldSide::raw_secret`]:
    /// a private-key block, RV22's exception, a text too dense to mask, a binary side): never
    /// read again, so no stage after the filter can get that secret back.
    withheld: HashSet<(Face, String)>,
    /// Held sides that are raw with a secret in them ([`HeldSide::raw_secret`]).
    raw_secret: HashSet<(Face, String)>,
    /// Every side released, read or not (a file the filter left out by its name is released
    /// unread): a later read that would serve it raw with a secret is refused like a withheld
    /// side, so a side that gains a secret after its release is refused too.
    released: HashSet<(Face, String)>,
    /// Whether the filter has ended its classification ([`ContentSource::seal`]).
    sealed: bool,
    /// Bytes held across both sides.
    bytes: u64,
    /// The bytes held before, and the path of, the first read that passed the cap since the
    /// total last fit it: what [`ContentSource::settle`] reports.
    over: Option<(u64, String)>,
}

/// How far past the total cap a [`CachedSource`] read may go until the caller settles: both sides
/// of one file at the per-file cap, so the filter can read a file before it judges it.
const PROVISIONAL_HEADROOM_BYTES: u64 = 2 * MAX_REVIEW_FILE_BYTES;

/// Which side of a file a read is for, or a release is of ([`ContentSource::release`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Face {
    /// The new side, or the worktree.
    New,
    /// The old side.
    Old,
}

impl<'a> CachedSource<'a> {
    /// A cache over `inner`, bounded by [`MAX_REVIEW_TOTAL_BYTES`].
    ///
    /// # Arguments
    ///
    /// * `inner` - The review's single content source.
    ///
    /// # Returns
    ///
    /// An empty cache.
    pub fn new(inner: &'a dyn ContentSource) -> Self {
        Self::capped(inner, MAX_REVIEW_TOTAL_BYTES)
    }

    /// A cache over `inner` with an explicit total cap.
    #[cfg(test)]
    pub(crate) fn with_total_cap(inner: &'a dyn ContentSource, cap: u64) -> Self {
        Self::capped(inner, cap)
    }

    /// A cache over `inner` with an explicit total cap and provisional headroom.
    #[cfg(test)]
    pub(crate) fn with_caps(inner: &'a dyn ContentSource, cap: u64, headroom: u64) -> Self {
        Self {
            headroom,
            ..Self::capped(inner, cap)
        }
    }

    fn capped(inner: &'a dyn ContentSource, cap: u64) -> Self {
        Self {
            inner,
            cap,
            headroom: PROVISIONAL_HEADROOM_BYTES,
            held: Mutex::new(Held::default()),
        }
    }

    /// Reads one side of `path`, from the cache or from the inner source.
    ///
    /// The lock is held across the inner read, so a check and its insert never race and a side
    /// is read at most once. That serializes concurrent readers behind one slow read (a `git
    /// show` can take up to `GIT_CALL_TIMEOUT_SECS`); a review reads in series (RV05) and no
    /// source calls back into the cache, so there is no contention and no reentrant deadlock
    /// today. A stage that reads in parallel would need a read outside the lock and a re-check
    /// before the insert.
    // MAGI FIX (S6 loop 15, A2-5): the lock's scope was undocumented.
    fn get(&self, face: Face, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
        let mut held = self.held.lock().unwrap_or_else(PoisonError::into_inner);
        // MAGI FIX (MS3 §6 loop3): the filter left this file out for its key block; a later read
        // would have fetched the raw key again.
        // MAGI FIX (MS3 §6 loop5): once sealed, a side held raw with a secret is served to no
        // stage, released or not.
        let key = (face, path.to_string());
        if held.withheld.contains(&key) || (held.sealed && held.raw_secret.contains(&key)) {
            return Err(ReviewError::EgressSecret {
                path: path.to_string(),
            });
        }
        let side = match face {
            Face::New => &held.new,
            Face::Old => &held.old,
        };
        if let Some(bytes) = side.get(path) {
            return Ok(bytes.clone());
        }
        let raw = match face {
            Face::New => self.inner.read(path)?,
            Face::Old => self.inner.read_old(path)?,
        };
        // MAGI FIX (T3.7): RV22 masks a one-line secret instead of leaving its file out; every
        // stage reads through this cache, so masking here is the one place it happens. A side with
        // a key block is not masked (see `masked_side`): the filter leaves that file out.
        let HeldSide {
            bytes: read,
            mask,
            raw_secret,
        } = masked_side(raw);
        // MAGI FIX (MS3 §6 loop4): a side released unread (left out by its name) was fetched
        // fresh and, holding a key block, served raw; the rule now holds for those sides too.
        // MAGI FIX (MS3 §6 loop5): for every released side, read or not, and for any secret it
        // would be served raw with (a binary side, one too dense to mask, a key block it gained
        // after its release), not a key block alone.
        // MAGI FIX (MS3 §6 loop5): and once sealed, for a side the filter never read either.
        if raw_secret && (held.sealed || held.released.contains(&key)) {
            return Err(ReviewError::EgressSecret {
                path: path.to_string(),
            });
        }
        let len = read
            .as_ref()
            .map_or(0, |b| u64::try_from(b.len()).unwrap_or(u64::MAX));
        let total = held.bytes.saturating_add(len);
        // MAGI FIX (S6 loop 15, A2-3): past the cap a read is held provisionally, within the
        // headroom; `settle` decides once the caller released what it left out.
        if total > self.cap.saturating_add(self.headroom) {
            // MAGI FIX (S6 loop 5): the error named no path and no total, so an operator could
            // not tell a higher cap from fewer files.
            return Err(ReviewError::InputTooLarge {
                limit: self.cap,
                held: held.bytes,
                path: path.to_string(),
            });
        }
        if total > self.cap && held.over.is_none() {
            held.over = Some((held.bytes, path.to_string()));
        }
        held.bytes = total;
        if let Some(mask) = mask {
            held.masks.insert((face, path.to_string()), mask);
        }
        if raw_secret {
            held.raw_secret.insert((face, path.to_string()));
        }
        let side = match face {
            Face::New => &mut held.new,
            Face::Old => &mut held.old,
        };
        side.insert(path.to_string(), read.clone());
        Ok(read)
    }
}

/// A side as the cache holds it: a UTF-8 text with no NUL has each secret masked, its line
/// breaks kept ([`mask_secrets_keeping_lines_within`], bounded by `MAX_MASKED_SIDE_BYTES`: a
/// side too dense in secrets to mask within it is held as read and its file left out as
/// `secret`); anything else is held as read, since the filter
/// leaves a binary or non-UTF-8 side out (masking could hide the NUL that makes it binary). A
/// text that holds a multi-line private-key block is held as read too, so the filter's secret
/// check leaves its file out (REQ-RV22's exception: a key body has no pattern of its own, so no
/// mask could promise that none of it leaves). O(bytes).
// MAGI FIX (T3.7 fix round 3): three review rounds found key-body lines that masking let through.
fn masked_side(read: Option<Vec<u8>>) -> HeldSide {
    held_side(read, MAX_MASKED_SIDE_BYTES)
}

/// The largest masked side the cache builds: twice the per-file cap, since a mask can be longer
/// than its secret. A side whose mask would pass it is too dense in secrets to mask; it is held
/// as read, so the filter leaves its file out as `secret`, the safe direction of RV22.
const MAX_MASKED_SIDE_BYTES: usize = 2 * MAX_REVIEW_FILE_BYTES as usize;

/// [`masked_side`] with an explicit bound on the masked side, its bytes and mask only.
#[cfg(test)]
fn masked_side_within(
    read: Option<Vec<u8>>,
    max_bytes: usize,
) -> (Option<Vec<u8>>, Option<SideMask>) {
    let held = held_side(read, max_bytes);
    (held.bytes, held.mask)
}

/// A side as the cache holds it ([`masked_side`]), and whether it is held raw with a secret in it.
struct HeldSide {
    /// The bytes held.
    bytes: Option<Vec<u8>>,
    /// What was masked in them, if anything.
    mask: Option<SideMask>,
    /// The bytes held still hold a secret the detector finds: a private-key block, a text too
    /// dense to mask within the bound, or a binary or non-UTF-8 side (never masked) with a match
    /// in its lossy decoding. Such a side reaches no model only because the filter leaves its
    /// file out, so the cache never serves it again once released ([`CachedSource::get`]).
    raw_secret: bool,
}

/// [`masked_side`] with an explicit bound on the masked side. One secret scan for a clean or masked
/// text side, two for a binary one. O(bytes).
fn held_side(read: Option<Vec<u8>>, max_bytes: usize) -> HeldSide {
    // MAGI FIX (MS3 §6 loop2): the key-block scan ran over every side; a key block is a secret
    // match, so it is looked for only in a side that holds one (one scan for most files).
    // MAGI FIX (MS3 §6 loop4): the mask is bounded; a side too dense to mask is held as read.
    let text = read
        .as_deref()
        .filter(|bytes| !bytes.contains(&0))
        .and_then(|bytes| std::str::from_utf8(bytes).ok());
    let (masked, raw_secret) = match text {
        Some(text) => match mask_secrets_keeping_lines_within(text, max_bytes) {
            BoundedMask::Clean => (None, false),
            BoundedMask::Masked(m) if !holds_key_block(text) => (Some(m), false),
            BoundedMask::Masked(_) | BoundedMask::TooLarge => (None, true),
        },
        // MAGI FIX (MS3 §6 loop5): a binary side is never masked; whether it holds a secret is
        // what decides that it is never served again once released.
        None => (
            None,
            read.as_deref()
                .is_some_and(|bytes| find_secret(&String::from_utf8_lossy(bytes)).is_some()),
        ),
    };
    match masked {
        Some(m) => HeldSide {
            bytes: Some(m.text.into_bytes()),
            mask: Some(SideMask { pattern: m.pattern }),
            raw_secret,
        },
        None => HeldSide {
            bytes: read,
            mask: None,
            raw_secret,
        },
    }
}

impl ContentSource for CachedSource<'_> {
    /// # Errors
    ///
    /// The inner source's error (never kept), [`ReviewError::InputTooLarge`] when keeping the read
    /// would pass the total cap by more than the provisional headroom, and
    /// [`ReviewError::EgressSecret`] for a released side that was, or would be, held raw with a
    /// secret in it.
    fn read(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
        self.get(Face::New, path)
    }

    fn read_old(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
        self.get(Face::Old, path)
    }

    /// Removes side `face` of `path` and gives its bytes back to the total cap. A side held raw
    /// with a secret in it (a private-key block, a text too dense to mask, a binary side) is
    /// remembered, and a later read of it is refused; so is any later read of a released side,
    /// read before or not, that would serve it raw with a secret
    /// (`a_key_block_side_released_before_it_was_ever_read_is_refused_too`,
    /// `a_released_side_held_raw_for_any_reason_or_raw_on_its_next_read_is_refused`). O(1)
    /// expected.
    fn release(&self, face: Face, path: &str) {
        let mut held = self.held.lock().unwrap_or_else(PoisonError::into_inner);
        let removed = match face {
            Face::New => held.new.remove(path),
            Face::Old => held.old.remove(path),
        };
        let key = (face, path.to_string());
        held.masks.remove(&key);
        // MAGI FIX (MS3 §6 loop5): every side held raw with a secret, not a key block alone.
        if held.raw_secret.remove(&key) {
            held.withheld.insert(key.clone());
        }
        held.released.insert(key);
        let removed = removed.flatten();
        let freed = removed.map_or(0, |bytes| u64::try_from(bytes.len()).unwrap_or(u64::MAX));
        held.bytes = held.bytes.saturating_sub(freed);
        if held.bytes <= self.cap {
            held.over = None;
        }
    }

    /// # Errors
    ///
    /// [`ReviewError::InputTooLarge`] when what is held passes the total cap, naming the first
    /// read that passed it and what was held before that read. O(1).
    fn settle(&self) -> Result<(), ReviewError> {
        let mut held = self.held.lock().unwrap_or_else(PoisonError::into_inner);
        if held.bytes <= self.cap {
            held.over = None;
            return Ok(());
        }
        let (before, path) = held.over.clone().unwrap_or_default();
        Err(ReviewError::InputTooLarge {
            limit: self.cap,
            held: before,
            path,
        })
    }

    /// From now on no side held raw with a secret is served, read before or not
    /// (`once_sealed_the_cache_never_serves_a_side_raw_with_a_secret_even_one_never_classified`).
    /// A kept file never has such a side (the filter left it out), so only a read beyond the
    /// kept files is refused. O(1).
    fn seal(&self) {
        self.held
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .sealed = true;
    }

    /// What the cache masked when it read side `face` of `path`. O(1) expected.
    fn masked(&self, face: Face, path: &str) -> Option<SideMask> {
        let held = self.held.lock().unwrap_or_else(PoisonError::into_inner);
        held.masks.get(&(face, path.to_string())).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_released_key_block_side_is_never_read_again() {
        // [MAGI S6 MS3 loop3, R6a Melchior] A side holding a private-key block is held raw (RV22's
        // exception), and only the filter leaving its file out kept it from a model; a later
        // stage that read the path again got the raw key back. The cache now refuses that read.
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        let end = concat!("-----", "END ", "PRIVATE KEY-----");
        let keyed = format!("{begin}\nMIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8w\n{end}\n");
        let inner = crate::review::test_support::MapSource::new(
            &[("k.pem", keyed.as_str()), ("a.rs", "fn a() {}\n")],
            &[],
        );
        let cache = CachedSource::new(&inner);
        assert!(cache.read("k.pem").expect("read").is_some());
        assert!(cache.read("a.rs").expect("read").is_some());
        cache.release(Face::New, "k.pem");
        cache.release(Face::New, "a.rs");
        assert!(
            matches!(cache.read("k.pem"), Err(ReviewError::EgressSecret { .. })),
            "the raw key came back"
        );
        assert_eq!(
            cache.read("a.rs").expect("a clean side reads again"),
            Some(b"fn a() {}\n".to_vec())
        );
    }

    /// A one-file source whose new side can change between reads, as a worktree file can.
    struct SwapSource(Mutex<Vec<u8>>);

    impl ContentSource for SwapSource {
        fn read(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
            Ok(Some(
                self.0
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .clone(),
            ))
        }

        fn read_old(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
            Ok(None)
        }
    }

    #[test]
    fn a_binary_side_holding_a_secret_split_by_a_nul_is_held_as_raw_secret() {
        // [MAGI S6 MS3 loop6, R6a Caspar] A NUL inside a token (`ghp_\0AAAA…`) ends the token's
        // tail, so no pattern matches the side and it was not held as raw secret: once released
        // a later read served it raw. A binary side holding any pattern's prefix now is one.
        let split = b"x\0ghp_\0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n".to_vec(); // allow-secret-scan
        assert!(
            find_secret(&String::from_utf8_lossy(&split)).is_none(),
            "precondition: no whole pattern matches"
        );
        assert!(held_side(Some(split), MAX_MASKED_SIDE_BYTES).raw_secret);
        let clean = b"\x7fELF\0\0\x01plain bytes".to_vec();
        assert!(!held_side(Some(clean), MAX_MASKED_SIDE_BYTES).raw_secret);
    }

    #[test]
    fn a_released_side_held_raw_for_any_reason_or_raw_on_its_next_read_is_refused() {
        // [MAGI S6 MS3 loop5, R6a Caspar] Only a key block was withheld at release: a side held
        // raw because it is binary (a NUL stops masking) or too dense to mask came back raw on a
        // later read, and so did a clean side that gained a key block after its release.
        let token = concat!("ghp_", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"); // allow-secret-scan
        let inner = SwapSource(Mutex::new(format!("x\0{token}\n").into_bytes()));
        let cache = CachedSource::new(&inner);
        assert!(cache.read("b.bin").expect("read").is_some());
        cache.release(Face::New, "b.bin");
        assert!(
            matches!(cache.read("b.bin"), Err(ReviewError::EgressSecret { .. })),
            "a raw binary side came back"
        );
        let inner = SwapSource(Mutex::new(b"fn a() {}\n".to_vec()));
        let cache = CachedSource::new(&inner);
        assert!(cache.read("a.rs").expect("read").is_some());
        cache.release(Face::New, "a.rs");
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        *inner.0.lock().unwrap_or_else(PoisonError::into_inner) =
            format!("{begin}\nMIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8w\n").into_bytes();
        assert!(
            matches!(cache.read("a.rs"), Err(ReviewError::EgressSecret { .. })),
            "a key block gained after the release was served raw"
        );
    }

    #[test]
    fn once_sealed_the_cache_never_serves_a_side_raw_with_a_secret_even_one_never_classified() {
        // [MAGI S6 MS3 loop5, R6a Melchior] The refusals covered only sides the filter released;
        // a path it never classified (a later stage reading beyond the kept files) was fetched
        // fresh and served raw. After the filter seals the cache, no side held raw with a secret
        // is served, read before or not; clean and masked sides still are.
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        let keyed = format!("{begin}\nMIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8w\n");
        let token = concat!("ghp_", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"); // allow-secret-scan
        let masked = format!("let t = \"{token}\";\n");
        let inner = crate::review::test_support::MapSource::new(
            &[
                ("seen.pem", keyed.as_str()),
                ("never.pem", keyed.as_str()),
                ("a.rs", "fn a() {}\n"),
                ("t.rs", masked.as_str()),
            ],
            &[],
        );
        let cache = CachedSource::new(&inner);
        assert!(cache
            .read("seen.pem")
            .expect("the filter reads it")
            .is_some());
        cache.seal();
        for path in ["seen.pem", "never.pem"] {
            assert!(
                matches!(cache.read(path), Err(ReviewError::EgressSecret { .. })),
                "{path} was served raw after the seal"
            );
        }
        assert_eq!(
            cache.read("a.rs").expect("clean"),
            Some(b"fn a() {}\n".to_vec())
        );
        let t = cache.read("t.rs").expect("masked").expect("present");
        assert!(!String::from_utf8_lossy(&t).contains(token));
    }

    #[test]
    fn a_side_too_dense_in_secrets_is_held_as_read_so_its_file_is_left_out() {
        // [MAGI S6 MS3 loop4, R6a Caspar] Its mask would pass the bound: it is not masked, and
        // the filter's secret check then leaves the file out (declared `secret`).
        let dense = "a://:p@h ".repeat(1_000).into_bytes();
        let (held, mask) = masked_side_within(Some(dense.clone()), dense.len());
        assert_eq!((held, mask), (Some(dense.clone()), None));
        let (held, mask) = masked_side_within(Some(dense.clone()), 64 * dense.len());
        assert!(mask.is_some() && held != Some(dense));
    }

    #[test]
    fn a_key_block_side_released_before_it_was_ever_read_is_refused_too() {
        // [MAGI S6 MS3 loop4, R6a Melchior] The filter leaves some files out before reading them
        // (a secret-shaped path, an ignored name) and releases their sides unread; a later read
        // of such a side fetched it fresh and, holding a key block, got it raw.
        let begin = concat!("-----", "BEGIN ", "PRIVATE KEY-----");
        let end = concat!("-----", "END ", "PRIVATE KEY-----");
        let keyed = format!("{begin}\nMIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8w\n{end}\n");
        let inner = crate::review::test_support::MapSource::new(
            &[("k.pem", keyed.as_str()), ("a.rs", "fn a() {}\n")],
            &[],
        );
        let cache = CachedSource::new(&inner);
        cache.release(Face::New, "k.pem");
        cache.release(Face::New, "a.rs");
        assert!(
            matches!(cache.read("k.pem"), Err(ReviewError::EgressSecret { .. })),
            "the raw key came back"
        );
        assert_eq!(
            cache.read("a.rs").expect("a clean side reads"),
            Some(b"fn a() {}\n".to_vec())
        );
    }

    #[cfg(unix)] // a FIFO is a unix file type; Windows has no open that blocks this way.
    #[test]
    fn a_file_swapped_for_a_fifo_before_the_open_is_refused_without_blocking() {
        // [MAGI S6 loop 3, A5] TOCTOU: the type is checked before the open, so a regular file
        // swapped for a FIFO in that window used to block the open forever (the preparation runs
        // outside the review clock). The opener is the production one, pointed at the FIFO.
        let root = tempfile::tempdir().expect("tempdir");
        std::fs::write(root.path().join("a.rs"), "fn f() {}\n").expect("write");
        let fifo = root.path().join("pipe");
        let made = std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .expect("mkfifo runs");
        assert!(made.success(), "mkfifo");
        let src = WorktreeSource::new(root.path().to_path_buf());
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let swapped = |_: &std::path::Path| open_for_review(&fifo);
            let _ = tx.send(src.read_with("a.rs", &swapped));
        });
        match rx.recv_timeout(std::time::Duration::from_secs(30)) {
            Ok(Err(ReviewError::NotRegularFile(p))) => assert_eq!(p, "a.rs"),
            Ok(other) => panic!("expected NotRegularFile, got {other:?}"),
            Err(_) => panic!("opening the FIFO blocked"),
        }
    }
    #[cfg(unix)] // a FIFO is a unix file type; Windows has no open that blocks this way.
    #[test]
    fn a_file_swapped_for_a_fifo_after_the_open_is_refused_without_blocking() {
        // [MAGI S6 loop 7, A2-m1/A2-c1] TOCTOU: the re-verification after the open opened the
        // path again with a plain, blocking open, so a FIFO swapped in after the first open hung
        // the preparation (outside the review clock). The swap happens inside the first open.
        let root = tempfile::tempdir().expect("tempdir");
        std::fs::write(root.path().join("a.rs"), "fn f() {}\n").expect("write");
        let src = WorktreeSource::new(root.path().to_path_buf());
        let dir = root.path().to_path_buf();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let swap_after_open = |p: &std::path::Path| {
                let file = open_for_review(p)?;
                std::fs::rename(dir.join("a.rs"), dir.join("a.old"))?;
                let made = std::process::Command::new("mkfifo")
                    .arg(dir.join("a.rs"))
                    .status()?;
                assert!(made.success(), "mkfifo");
                Ok(file)
            };
            let _ = tx.send(src.read_with("a.rs", &swap_after_open));
        });
        match rx.recv_timeout(std::time::Duration::from_secs(30)) {
            Ok(Err(ReviewError::OutsideWorkspace(p))) => assert_eq!(p, "a.rs"),
            Ok(other) => panic!("expected OutsideWorkspace, got {other:?}"),
            Err(_) => panic!("the re-verification blocked on the FIFO"),
        }
    }
    use crate::review::test_support::FakeGit;

    #[test]
    fn a_cached_source_reads_each_side_of_a_path_once() {
        // [CP2 MS2 S1 loop 8] several stages read the same file; RevisionSource pays a `git show` per read.
        use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
        struct Counting(AtomicUsize);
        impl ContentSource for Counting {
            fn read(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                self.0.fetch_add(1, SeqCst);
                Ok(Some(b"x".to_vec()))
            }
            fn read_old(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                self.0.fetch_add(1, SeqCst);
                Ok(None)
            }
        }
        let inner = Counting(AtomicUsize::new(0));
        let cached = CachedSource::new(&inner);
        for _ in 0..3 {
            assert_eq!(cached.read("a.rs").expect("read"), Some(b"x".to_vec()));
            assert_eq!(cached.read_old("a.rs").expect("read_old"), None);
        }
        assert_eq!(inner.0.load(SeqCst), 2);
    }

    #[test]
    fn a_cached_source_masks_each_secret_it_holds_and_says_which() {
        // [T3.7] RV22 masks instead of omitting: every stage reads through the cache, so the
        // cache is where a secret is masked, once, with its lines kept.
        let secret = "let u = \"postgres://svc:S3cr3t@db/app\";\nfn f() {}\n";
        let inner = crate::review::test_support::MapSource::new(
            &[("a.rs", secret), ("b.rs", "fn b() {}\n")],
            &[("a.rs", secret)],
        );
        let cached = CachedSource::new(&inner);
        for read in [cached.read("a.rs"), cached.read_old("a.rs")] {
            let text = String::from_utf8(read.expect("read").expect("present")).expect("utf-8");
            assert!(!text.contains("S3cr3t"), "{text}");
            assert!(text.contains("<redacted: url-credential>"), "{text}");
            assert_eq!(text.lines().count(), 2);
        }
        let mask = SideMask {
            pattern: "url-credential",
        };
        assert_eq!(cached.masked(Face::New, "a.rs"), Some(mask));
        assert_eq!(cached.masked(Face::Old, "a.rs"), Some(mask));
        assert_eq!(
            cached.read("b.rs").expect("read"),
            Some(b"fn b() {}\n".to_vec())
        );
        assert_eq!(cached.masked(Face::New, "b.rs"), None);
        cached.release(Face::New, "a.rs");
        assert_eq!(cached.masked(Face::New, "a.rs"), None);
        assert_eq!(cached.masked(Face::Old, "a.rs"), Some(mask));
        // A binary side is held as read: masking could hide the NUL that makes it binary.
        let bin = crate::review::test_support::MapSource::with_bytes(&[(
            "k.bin",
            b"\0postgres://svc:S3cr3t@db\n",
        )]);
        let cached = CachedSource::new(&bin);
        assert_eq!(
            cached.read("k.bin").expect("read"),
            Some(b"\0postgres://svc:S3cr3t@db\n".to_vec())
        );
        assert_eq!(cached.masked(Face::New, "k.bin"), None);
    }

    #[test]
    fn a_cached_source_keeps_the_bytes_it_first_read_even_if_the_file_changes() {
        // [CP2 MS2 S1b loop 10] RV22: the filter scanned the first read; later stages must see those same bytes.
        use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
        struct Changing(AtomicUsize);
        impl ContentSource for Changing {
            fn read(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Ok(Some(if self.0.fetch_add(1, SeqCst) == 0 {
                    b"clean".to_vec()
                } else {
                    b"changed".to_vec()
                }))
            }
            fn read_old(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Ok(None)
            }
        }
        let inner = Changing(AtomicUsize::new(0));
        let cached = CachedSource::new(&inner);
        assert_eq!(cached.read("a.rs").expect("first"), Some(b"clean".to_vec()));
        assert_eq!(
            cached.read("a.rs").expect("second"),
            Some(b"clean".to_vec())
        );
    }

    #[test]
    fn a_cached_source_refuses_to_hold_more_than_the_total_cap() {
        // [CP2 MS2 S1a loop 10] the per-file cap bounds a file; this bounds the review, before any model call.
        struct Three;
        impl ContentSource for Three {
            fn read(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Ok(Some(b"abc".to_vec()))
            }
            fn read_old(&self, _: &str) -> Result<Option<Vec<u8>>, ReviewError> {
                Ok(Some(b"abc".to_vec()))
            }
        }
        // [MAGI S6 loop 15, A2-3] a read may pass the cap by the provisional headroom (3 here)
        // until the caller settles, so a file the filter then leaves out costs nothing.
        let cached = CachedSource::with_caps(&Three, 5, 3);
        assert_eq!(cached.read("a.rs").expect("fits"), Some(b"abc".to_vec()));
        assert!(cached.settle().is_ok());
        // [CP2 MS2 cierre (sin MAGI)] the old side is held too, so it counts toward the same cap.
        assert_eq!(
            cached.read_old("a.rs").expect("provisional"),
            Some(b"abc".to_vec())
        );
        assert!(matches!(
            cached.settle(),
            Err(ReviewError::InputTooLarge {
                limit: 5,
                held: 3,
                ..
            })
        ));
        cached.release(Face::Old, "a.rs");
        assert!(cached.settle().is_ok(), "released, the total fits again");
        // [MAGI S6 loop 5, A2-b1] the error says what was held and which read passed the cap.
        assert!(cached.read("b.rs").is_ok(), "within the headroom");
        let err = cached.settle().expect_err("over the cap");
        assert!(
            matches!(&err, ReviewError::InputTooLarge { limit: 5, held: 3, path } if path == "b.rs"),
            "{err:?}"
        );
        assert!(err.to_string().contains("b.rs"), "{err}");
        // Past the headroom a read is refused outright and not kept.
        let err = cached.read_old("b.rs").expect_err("past the headroom");
        assert!(
            matches!(&err, ReviewError::InputTooLarge { limit: 5, held: 6, path } if path == "b.rs"),
            "{err:?}"
        );
        assert_eq!(
            cached.read("a.rs").expect("already held: adds nothing"),
            Some(b"abc".to_vec())
        );
    }

    // [CP2 MS2 S1 loop 4: absent vs failed, decided by ls-tree, never by error text]
    #[test]
    fn a_path_absent_from_the_revision_is_none_and_a_failing_show_is_an_error() {
        let git = FakeGit::from(&[
            (
                "ls-tree -z --end-of-options X -- a.rs",
                Ok("100644 blob 0000000000000000000000000000000000000000\ta.rs\0"),
            ),
            (
                "ls-tree -z --end-of-options X -- b.rs",
                Ok("100644 blob 0000000000000000000000000000000000000000\tb.rs\0"),
            ),
            ("ls-tree -z --end-of-options X -- missing.rs", Ok("")),
            (
                "show --no-ext-diff --no-textconv --end-of-options X:./a.rs",
                Ok("fn a() {}\n"),
            ),
            (
                "show --no-ext-diff --no-textconv --end-of-options X:./b.rs",
                Err("fatal: bad object X:./b.rs"),
            ),
        ]);
        let src = RevisionSource::new(&git, "X".into(), None);
        assert_eq!(
            src.read("a.rs").expect("present"),
            Some(b"fn a() {}\n".to_vec())
        );
        assert_eq!(
            src.read("missing.rs").expect("absent is not an error"),
            None
        ); // no show call scripted for it
           // [CP2 MS2 S1a loop 13] a listed path whose blob is missing: typed, naming the path and what to do.
        match src.read("b.rs") {
            Err(ReviewError::Git(msg)) => assert!(
                msg.contains("b.rs") && msg.contains("partial clone"),
                "{msg}"
            ),
            other => panic!("expected a git error, got {other:?}"),
        }
    }

    #[test]
    fn a_missing_worktree_path_is_none() {
        let d = tempfile::tempdir().expect("tempdir");
        assert_eq!(
            WorktreeSource::new(d.path().to_path_buf())
                .read("nope.rs")
                .expect("not an error"),
            None
        );
    }

    #[test]
    fn a_worktree_source_whose_root_cannot_be_resolved_fails_every_read_closed() {
        // [CP2 MS2 S1a loop 10] `new` stays infallible: the canonicalize error is kept and every read returns it.
        let d = tempfile::tempdir().expect("tempdir");
        let src = WorktreeSource::new(d.path().join("gone"));
        assert!(matches!(src.read("a.rs"), Err(ReviewError::Io(_))));
    }

    // [CP2 MS2 S1 loop 2: per-file containment, security]
    #[test]
    fn a_worktree_read_never_leaves_the_root() {
        let outer = tempfile::tempdir().expect("tempdir");
        std::fs::write(outer.path().join("secret.txt"), "outside\n").expect("write");
        let root = outer.path().join("ws");
        std::fs::create_dir_all(&root).expect("mkdir");
        std::fs::write(root.join("a.rs"), "fn f() {}\n").expect("write");
        let src = WorktreeSource::new(root.clone());
        assert_eq!(
            src.read("a.rs").expect("inside"),
            Some(b"fn f() {}\n".to_vec())
        );
        assert!(matches!(
            src.read("../secret.txt"),
            Err(ReviewError::OutsideWorkspace(_))
        ));
    }

    #[test]
    fn a_sibling_directory_sharing_the_roots_name_prefix_is_outside() {
        // [CP2 MS2 S1a loop 9] containment is by path components: `ws-evil` is not inside `ws`.
        let outer = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(outer.path().join("ws-evil")).expect("mkdir");
        std::fs::write(outer.path().join("ws-evil").join("secret.txt"), "outside\n")
            .expect("write");
        std::fs::create_dir_all(outer.path().join("ws")).expect("mkdir");
        let src = WorktreeSource::new(outer.path().join("ws"));
        assert!(matches!(
            src.read("../ws-evil/secret.txt"),
            Err(ReviewError::OutsideWorkspace(_))
        ));
    }

    #[test]
    fn a_bounded_read_stops_one_byte_past_the_cap() {
        // [CP2 MS2 S1a loop 9] the one reader both sources and ScrubbedGit's stdout use.
        assert_eq!(
            read_bounded(&b"abcd"[..], 4).expect("reads"),
            Some(b"abcd".to_vec())
        );
        assert_eq!(read_bounded(&b"abcde"[..], 4).expect("reads"), None);
        assert_eq!(read_bounded(&b""[..], 0).expect("reads"), Some(Vec::new()));
    }

    #[test]
    fn the_default_query_reads_only_an_empty_git_error_as_no_match() {
        // [MAGI S6 loop 15, A2-6] the default `query` decides "no match" by an empty message:
        // that is the contract of a double that reports git's stderr verbatim. Any other failure
        // stays an error, so a worded failure never reads as an absent path.
        struct Answers;
        impl GitCommand for Answers {
            fn run_bytes(&self, args: &[&str]) -> Result<Vec<u8>, ReviewError> {
                match args.first().copied() {
                    Some("found") => Ok(b"abc\n".to_vec()),
                    Some("none") => Err(ReviewError::Git(String::new())),
                    _ => Err(ReviewError::Git("git rev-parse failed: exit 128".into())),
                }
            }
        }
        assert_eq!(Answers.query(&["found"]).expect("ok"), Some("abc\n".into()));
        assert_eq!(Answers.query(&["none"]).expect("no match"), None);
        assert!(matches!(
            Answers.query(&["broken"]),
            Err(ReviewError::Git(msg)) if !msg.is_empty()
        ));
    }

    #[test]
    fn a_worktree_file_over_the_cap_is_too_large_and_never_returned() {
        let d = tempfile::tempdir().expect("tempdir");
        let cap = usize::try_from(MAX_REVIEW_FILE_BYTES).expect("fits");
        std::fs::write(d.path().join("big.rs"), vec![b'a'; cap + 1]).expect("write");
        std::fs::write(d.path().join("edge.rs"), vec![b'a'; cap]).expect("write");
        let src = WorktreeSource::new(d.path().to_path_buf());
        assert!(matches!(
            src.read("big.rs"),
            Err(ReviewError::TooLarge { .. })
        )); // [S1a loop 10] typed
        assert_eq!(
            src.read("edge.rs")
                .expect("at the cap is fine")
                .map(|b| b.len()),
            Some(cap)
        );
    }

    #[test]
    fn a_revision_blob_over_the_cap_is_too_large() {
        // Either git returns the whole blob (checked against the cap) or ScrubbedGit cut it (GitOutputTooLarge).
        struct Big;
        impl GitCommand for Big {
            fn run_bytes(&self, args: &[&str]) -> Result<Vec<u8>, ReviewError> {
                match args.join(" ").as_str() {
                    "ls-tree -z --end-of-options X -- big.rs" => Ok(
                        b"100644 blob 0000000000000000000000000000000000000000\tbig.rs\0".to_vec(),
                    ),
                    "ls-tree -z --end-of-options X -- cut.rs" => Ok(
                        b"100644 blob 0000000000000000000000000000000000000000\tcut.rs\0".to_vec(),
                    ),
                    "show --no-ext-diff --no-textconv --end-of-options X:./big.rs" => Ok(vec![
                            b'a';
                            usize::try_from(MAX_REVIEW_FILE_BYTES).expect("fits")
                                + 1
                        ]),
                    "show --no-ext-diff --no-textconv --end-of-options X:./cut.rs" => {
                        Err(ReviewError::GitOutputTooLarge {
                            limit: MAX_GIT_OUTPUT_BYTES,
                        })
                    }
                    other => Err(ReviewError::Git(format!("unexpected: {other}"))),
                }
            }
        }
        let src = RevisionSource::new(&Big, "X".into(), None);
        for p in ["big.rs", "cut.rs"] {
            assert!(
                matches!(src.read(p), Err(ReviewError::TooLarge { .. })),
                "{p}"
            ); // [S1a loop 10] typed
        }
    }

    #[test]
    fn a_file_gone_between_the_check_and_the_open_reads_as_absent() {
        // [MAGI S6 loop 14, A2-6] a file removed between the canonicalization and the open was
        // an I/O error (`unreadable`), while one removed a moment earlier read as absent
        // (`disappeared`): the same event got two causes.
        let root = tempfile::tempdir().expect("tempdir");
        std::fs::write(root.path().join("a.rs"), "fn f() {}\n").expect("write");
        let src = WorktreeSource::new(root.path().to_path_buf());
        let gone = |_: &std::path::Path| -> std::io::Result<std::fs::File> {
            Err(std::io::Error::from(std::io::ErrorKind::NotFound))
        };
        assert_eq!(src.read_with("a.rs", &gone).expect("absent"), None);
        // Any other failure of the open stays an error.
        let denied = |_: &std::path::Path| -> std::io::Result<std::fs::File> {
            Err(std::io::Error::from(std::io::ErrorKind::PermissionDenied))
        };
        assert!(matches!(
            src.read_with("a.rs", &denied),
            Err(ReviewError::Io(_))
        ));
    }

    #[test]
    fn a_file_gone_right_after_the_open_reads_as_absent() {
        // [MAGI S6 loop 16, A2-4] the same unlink a moment later, between the open and the
        // re-verification, read as outside the workspace.
        let root = tempfile::tempdir().expect("tempdir");
        std::fs::write(root.path().join("a.rs"), "fn f() {}\n").expect("write");
        let src = WorktreeSource::new(root.path().to_path_buf());
        let open_then_unlink = |p: &std::path::Path| -> std::io::Result<std::fs::File> {
            let file = open_for_review(p)?;
            std::fs::remove_file(p)?;
            Ok(file)
        };
        assert_eq!(
            src.read_with("a.rs", &open_then_unlink).expect("absent"),
            None
        );
    }

    #[test]
    fn a_failed_re_resolution_after_the_open_is_an_io_error_not_an_escape() {
        // [Review residual A2-c] the re-resolution after the open mapped every failure that was
        // not `NotFound` to `OutsideWorkspace`, naming a sandbox escape for an I/O failure. The
        // opener stands in for a directory swapped, after the open, for a link to itself: the
        // path then fails to resolve (a loop), with no escape anywhere.
        fn self_loop(dir: &std::path::Path) -> std::io::Result<()> {
            #[cfg(unix)]
            {
                std::os::unix::fs::symlink(dir, dir)
            }
            #[cfg(windows)]
            {
                // A directory junction needs no privilege, unlike a symlink.
                let made = std::process::Command::new("cmd")
                    .args(["/C", "mklink", "/J"])
                    .arg(dir)
                    .arg(dir)
                    .output()?;
                if made.status.success() {
                    Ok(())
                } else {
                    Err(std::io::Error::other("mklink /J failed"))
                }
            }
        }
        let root = tempfile::tempdir().expect("tempdir");
        let dir = root.path().join("d");
        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::write(dir.join("a.rs"), "fn f() {}\n").expect("write");
        let src = WorktreeSource::new(root.path().to_path_buf());
        let open_then_loop = |p: &std::path::Path| -> std::io::Result<std::fs::File> {
            let file = open_for_review(p)?;
            std::fs::remove_dir_all(&dir)?;
            self_loop(&dir)?;
            Ok(file)
        };
        let read = src.read_with("d/a.rs", &open_then_loop);
        assert!(matches!(read, Err(ReviewError::Io(_))), "{read:?}");
    }

    #[test]
    fn a_file_swapped_between_the_check_and_the_open_is_refused() {
        // [MAGI §6 MS2 D loop 2, security] the opener stands in for a component swapped for an
        // escaping link after the containment check: what was opened is not what the path names.
        let outer = tempfile::tempdir().expect("tempdir");
        let outside = outer.path().join("secret.txt");
        std::fs::write(
            &outside, "outside
",
        )
        .expect("write");
        let root = outer.path().join("ws");
        std::fs::create_dir_all(&root).expect("mkdir");
        std::fs::write(
            root.join("a.rs"),
            "fn f() {}
",
        )
        .expect("write");
        let src = WorktreeSource::new(root);
        let swapped = |_: &std::path::Path| std::fs::File::open(&outside);
        assert!(matches!(
            src.read_with("a.rs", &swapped),
            Err(ReviewError::OutsideWorkspace(_))
        ));
        let honest = |p: &std::path::Path| std::fs::File::open(p);
        assert_eq!(
            src.read_with("a.rs", &honest).expect("inside"),
            Some(
                b"fn f() {}
"
                .to_vec()
            )
        );
    }

    #[test]
    fn a_hard_link_to_a_file_outside_the_workspace_is_refused() {
        // [MAGI S6 loop 6, F2-c1] a hard link's path is inside the workspace but its inode may
        // be anything on the same filesystem: a file with more than one link is never read.
        // [MAGI S6 loop 7, A2-b3] on every platform: Windows read the outside bytes.
        let outer = tempfile::tempdir().expect("tempdir");
        std::fs::write(outer.path().join("secret.txt"), "outside\n").expect("write");
        let root = outer.path().join("ws");
        std::fs::create_dir_all(&root).expect("mkdir");
        std::fs::hard_link(outer.path().join("secret.txt"), root.join("copy.rs")).expect("link");
        std::fs::write(root.join("own.rs"), "fn f() {}\n").expect("write");
        let src = WorktreeSource::new(root);
        // [MAGI S6 loop 12, A2-b1 / A2-c4] refused with its own cause, never `outside workspace`.
        assert!(
            matches!(
                src.read("copy.rs"),
                Err(ReviewError::NotReviewable { ref path, cause: HARD_LINK_CAUSE }) if path == "copy.rs"
            ),
            "{:?}",
            src.read("copy.rs")
        );
        assert_eq!(
            src.read("own.rs").expect("reads"),
            Some(b"fn f() {}\n".to_vec())
        );
    }

    #[cfg(unix)]
    // Windows CI cannot create symlinks without Developer Mode; the `..` case above covers the check there.
    #[test]
    fn a_tracked_symlink_that_points_outside_is_not_followed() {
        let outer = tempfile::tempdir().expect("tempdir");
        std::fs::write(outer.path().join("secret.txt"), "outside\n").expect("write");
        let root = outer.path().join("ws");
        std::fs::create_dir_all(&root).expect("mkdir");
        std::os::unix::fs::symlink(outer.path().join("secret.txt"), root.join("link.rs"))
            .expect("symlink");
        let src = WorktreeSource::new(root);
        assert!(matches!(
            src.read("link.rs"),
            Err(ReviewError::OutsideWorkspace(_))
        ));
    }

    #[cfg(unix)]
    // as above: creating a symlink needs Developer Mode on Windows; the lexical rule below runs everywhere.
    #[test]
    fn a_symlink_to_a_protected_path_inside_the_workspace_is_refused() {
        // [CP2 MS2 S1b1 loop 16, security] the names are innocuous and stay inside the root; the targets are not.
        let d = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(d.path().join(".git")).expect("mkdir");
        std::fs::write(
            d.path().join(".git").join("config"),
            "[core]\n\tbare = false\n",
        )
        .expect("write");
        std::fs::create_dir_all(d.path().join(".magi")).expect("mkdir");
        std::fs::write(d.path().join(".magi").join("magi.toml"), "a = 1\n").expect("write");
        std::os::unix::fs::symlink(
            d.path().join(".git").join("config"),
            d.path().join("notes.rs"),
        )
        .expect("symlink");
        std::os::unix::fs::symlink(d.path().join(".magi"), d.path().join("cfg")).expect("symlink");
        let src = WorktreeSource::new(d.path().to_path_buf());
        for p in ["notes.rs", "cfg/magi.toml"] {
            assert!(matches!(src.read(p), Err(ReviewError::Protected(_))), "{p}");
        }
    }

    #[test]
    fn a_worktree_source_rooted_in_a_protected_directory_reads_nothing() {
        // [S6 loop 16, A2-1, security] stripping the root drops its own `.git` / `.magi` component,
        // so the per-path check alone saw `config` as an ordinary file.
        let d = tempfile::tempdir().expect("tempdir");
        for (dir, file) in [
            (".git", "config"),
            (".magi", "magi.toml"),
            (".git", "hooks/x"),
        ] {
            let root = d.path().join(dir);
            let target = root.join(file);
            std::fs::create_dir_all(target.parent().expect("parent")).expect("mkdir");
            std::fs::write(&target, "[core]\n").expect("write");
            let nested = if file.contains('/') {
                root.join("hooks")
            } else {
                root.clone()
            };
            let name = target
                .file_name()
                .expect("name")
                .to_string_lossy()
                .to_string();
            let src = WorktreeSource::new(nested);
            assert!(
                matches!(src.read(&name), Err(ReviewError::Protected(_))),
                "{dir}/{file}"
            );
        }
    }

    #[cfg(unix)]
    // as above: creating a symlink needs Developer Mode on Windows.
    #[test]
    fn a_worktree_source_rooted_at_an_alias_of_a_protected_directory_reads_nothing() {
        // [S6 loop 16, A2-1, security] the alias canonicalizes to the protected directory.
        let d = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(d.path().join(".git")).expect("mkdir");
        std::fs::write(d.path().join(".git").join("config"), "[core]\n").expect("write");
        std::os::unix::fs::symlink(d.path().join(".git"), d.path().join("ws")).expect("symlink");
        let src = WorktreeSource::new(d.path().join("ws"));
        assert!(matches!(src.read("config"), Err(ReviewError::Protected(_))));
    }

    #[test]
    fn protected_names_match_whatever_their_case_or_trailing_dots() {
        // [CP2 MS2 S1b1 loop 16] Windows resolves `.GIT`, `.git.` and `.git ` to `.git`; the rule is the same everywhere.
        for p in [
            ".git/config",
            "sub/.GIT",
            ".git./config",
            ".Magi/magi.toml",
            ".git /x",
            "a\\.git\\config",
            ".magi-rs-memory.db-wal",
            "x/.MAGI-RS-MEMORY.DB-shm",
        ] {
            assert!(is_never_reviewed(p), "{p}");
        }
        for p in [
            "a.rs",
            ".github/workflows/ci.yml",
            ".gitignore",
            "magi/x.rs",
            "my.magi/x",
            "src/git.rs",
        ] {
            assert!(!is_never_reviewed(p), "{p}");
        }
    }
    #[test]
    fn a_directory_named_like_the_memory_database_is_never_reviewed() {
        // [MAGI S6 loop 14, A2-2, security] the database prefix was checked on the last component
        // only, so a directory named like the database or a companion hid what it holds.
        for p in [
            ".magi-rs-memory.db-wal/notes.txt",
            "a/.Magi-rs-memory.db.bak/x/y.rs",
            ".magi-rs-memory.db/x",
            "a\\.magi-rs-memory.db-shm\\b.rs",
        ] {
            assert!(is_never_reviewed(p), "{p}");
        }
        for p in [
            "magi-rs-memory.db/x",
            "a/x.magi-rs-memory.db.rs/y",
            "src/memory.rs",
        ] {
            assert!(!is_never_reviewed(p), "{p}");
        }
    }

    #[test]
    fn a_listed_name_that_is_not_utf8_never_stands_for_a_requested_path() {
        // [MAGI S6 MS2 loop 4, A2 Caspar] the listing was decoded lossily, so `a\xff.rs` read as
        // `a\u{FFFD}.rs`: a requested path that is valid UTF-8 matched a name it is not.
        struct Latin1Tree;
        impl GitCommand for Latin1Tree {
            fn run_bytes(&self, args: &[&str]) -> Result<Vec<u8>, ReviewError> {
                match args.join(" ").as_str() {
                    "ls-tree -z --end-of-options X -- a\u{FFFD}.rs" => Ok(
                        b"100644 blob 0000000000000000000000000000000000000000\ta\xff.rs\0"
                            .to_vec(),
                    ),
                    "ls-tree -z --end-of-options X -- b.rs" => Ok(
                        b"100644 blob 0000000000000000000000000000000000000000\tb.rs\0".to_vec(),
                    ),
                    "show --no-ext-diff --no-textconv --end-of-options X:./b.rs" => {
                        Ok(b"fn b() {}\n".to_vec())
                    }
                    other => Err(ReviewError::Git(format!("unexpected: {other}"))),
                }
            }
        }
        let src = RevisionSource::new(&Latin1Tree, "X".into(), None);
        assert_eq!(src.read("a\u{FFFD}.rs").expect("absent"), None);
        assert_eq!(
            src.read("b.rs").expect("present"),
            Some(b"fn b() {}\n".to_vec())
        );
    }
    #[test]
    fn a_revision_blob_is_read_with_the_per_file_cap_not_the_git_output_cap() {
        // [MAGI S6 MS2 loop 4, A2 Balthasar] `show` ran under the 64 MiB git output cap, so a
        // blob between 8 and 64 MiB was held whole only to be refused.
        struct Capped(Mutex<Vec<u64>>);
        impl GitCommand for Capped {
            fn run_bytes(&self, args: &[&str]) -> Result<Vec<u8>, ReviewError> {
                match args.join(" ").as_str() {
                    "ls-tree -z --end-of-options X -- a.rs" => Ok(
                        b"100644 blob 0000000000000000000000000000000000000000\ta.rs\0".to_vec(),
                    ),
                    other => Err(ReviewError::Git(format!("uncapped: {other}"))),
                }
            }
            fn run_bytes_capped(&self, args: &[&str], cap: u64) -> Result<Vec<u8>, ReviewError> {
                self.0
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push(cap);
                match args.join(" ").as_str() {
                    "show --no-ext-diff --no-textconv --end-of-options X:./a.rs" => {
                        Ok(b"fn a() {}\n".to_vec())
                    }
                    other => Err(ReviewError::Git(format!("unexpected: {other}"))),
                }
            }
        }
        let git = Capped(Mutex::new(Vec::new()));
        let src = RevisionSource::new(&git, "X".into(), None);
        assert_eq!(
            src.read("a.rs").expect("read"),
            Some(b"fn a() {}\n".to_vec())
        );
        assert_eq!(
            *git.0.lock().unwrap_or_else(PoisonError::into_inner),
            vec![MAX_REVIEW_FILE_BYTES]
        );
    }

    #[test]
    fn a_revision_shaped_like_an_option_reaches_git_after_end_of_options() {
        // [MAGI S6 loop 12, A2-c3, security] the revision reached `ls-tree` and `show` as a
        // positional argument: a rev of `--output=<file>` read as an option of `git show` and
        // wrote a file. `--end-of-options` makes the sink safe without trusting its caller.
        let git = FakeGit::from(&[
            (
                "ls-tree -z --end-of-options --output=x -- a.rs",
                Ok("100644 blob 0000000000000000000000000000000000000000\ta.rs\0"),
            ),
            (
                "show --no-ext-diff --no-textconv --end-of-options --output=x:./a.rs",
                Ok("fn a() {}\n"),
            ),
        ]);
        let src = RevisionSource::new(&git, "--output=x".into(), None);
        assert_eq!(
            src.read("a.rs").expect("read"),
            Some(b"fn a() {}\n".to_vec())
        );
    }

    #[test]
    fn a_revision_read_lists_only_the_requested_path_never_the_whole_tree() {
        // [MAGI S6 loop 7, A2-c5] the existence check listed every blob of the revision, so a
        // monorepo past the 64 MiB git output cap failed the review before any model call.
        let git = FakeGit::from(&[
            (
                "ls-tree -z --end-of-options X -- a.rs",
                Ok("100644 blob 0000000000000000000000000000000000000000\ta.rs\0"),
            ),
            ("ls-tree -z --end-of-options X -- gone.rs", Ok("")),
            (
                "show --no-ext-diff --no-textconv --end-of-options X:./a.rs",
                Ok("fn a() {}\n"),
            ),
        ]);
        let src = RevisionSource::new(&git, "X".into(), None);
        assert_eq!(
            src.read("a.rs").expect("read"),
            Some(b"fn a() {}\n".to_vec())
        );
        assert_eq!(src.read("gone.rs").expect("absent"), None);
    }

    #[test]
    fn a_revision_source_refuses_a_path_that_climbs_out_or_is_absolute_before_any_git_call() {
        // [MAGI S6 loop 13, A2-c4, security] `RevisionSource` took any path to `ls-tree` and
        // `show <rev>:./<path>`: a `..` resolved against the workspace read a file outside it
        // (inside the repository), and only the callers kept such a path out. No git call is
        // expected here: the fake answers none.
        let git = crate::review::test_support::FakeGit::from(&[]);
        let src = RevisionSource::new(&git, "abc".into(), Some("def".into()));
        for path in ["../secrets.env", "a/../../b.rs", "/etc/passwd"] {
            for read in [src.read(path), src.read_old(path)] {
                assert!(
                    matches!(read, Err(ReviewError::OutsideWorkspace(ref p)) if p == path),
                    "{path}: {read:?}"
                );
            }
        }
    }
}
