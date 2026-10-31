// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! The single content source of a review: every stage that reads code reads it through here.
//!
//! A review reads its code from exactly one place, chosen once from the input: a git revision
//! ([`RevisionSource`], `git show <rev>:./<path>`) or the worktree ([`WorktreeSource`]), with an
//! optional old side for deleted lines. Nothing mixes the worktree with another revision.

use std::collections::HashMap;
use std::io::Read;
use std::path::PathBuf;
use std::sync::{Mutex, PoisonError};

use super::error::ReviewError;

/// Per-file byte cap: a file over it is reported as [`ReviewError::TooLarge`], never returned.
pub const MAX_REVIEW_FILE_BYTES: u64 = 8 * 1024 * 1024;
/// Byte cap of any one git output; past it git is stopped ([`ReviewError::GitOutputTooLarge`]).
pub const MAX_GIT_OUTPUT_BYTES: u64 = 64 * 1024 * 1024;
/// Bytes of a git call's stderr that are kept for the error message.
pub const MAX_GIT_STDERR_BYTES: u64 = 4 * 1024;
/// Byte cap of everything one review reads.
pub const MAX_REVIEW_TOTAL_BYTES: u64 = 256 * 1024 * 1024;
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
    /// an implementation that words its failures overrides this with the exit status.
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
/// True for a component `.git` or `.magi`, or a last component that starts with
/// `.magi-rs-memory.db`. Components split on `/` and `\`, compared ASCII-case-insensitively after
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
    let database = components.last().is_some_and(|last| {
        last.get(..MEMORY_DB_PREFIX.len())
            .is_some_and(|head| head.eq_ignore_ascii_case(MEMORY_DB_PREFIX))
    });
    named || database
}

/// Reads the worktree under a root, never outside it.
///
/// Each read canonicalizes `root.join(path)`, checks by path components that it stays under the
/// canonical root, refuses a resolved target that is a never-reviewed path (a symlink alias of
/// `.git/config`), and reads that canonical path with the per-file cap. After the open the path
/// is resolved again and must still lie under the root and name the very file opened (same
/// device and inode, or volume and file index): a component swapped for a link between the
/// check and the open is refused as outside the workspace.
///
/// A hard link names its inode from inside the workspace whatever file it is, so a file with
/// more than one link is refused as outside the workspace too, on every platform (`link_count`).
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
    /// names the file opened (see [`WorktreeSource`]),
    /// [`ReviewError::Protected`] when it resolves to a never-reviewed path,
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
        if !std::fs::metadata(&canonical).map_err(io)?.is_file() {
            return Err(ReviewError::NotRegularFile(path.to_string()));
        }
        let file = open(&canonical).map_err(io)?;
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
        if link_count(&file, &opened_meta).map_err(io)? > 1 {
            return Err(ReviewError::OutsideWorkspace(path.to_string()));
        }
        let opened = same_file::Handle::from_file(file).map_err(io)?;
        // Re-resolved after the open: a component swapped for an escaping link before the open
        // either still escapes, or names another file than the one opened.
        let now = std::fs::canonicalize(&joined)
            .map_err(|_| ReviewError::OutsideWorkspace(path.to_string()))?;
        let still_inside = now
            .strip_prefix(canonical_root)
            .is_ok_and(|r| !is_never_reviewed(&r.to_string_lossy()));
        if !still_inside {
            return Err(ReviewError::OutsideWorkspace(path.to_string()));
        }
        // MAGI FIX (S6 loop 7): the path is opened again through `open_for_review`, never a
        // plain open: a FIFO swapped in after the first open opens at once and fails the
        // identity comparison instead of blocking the preparation.
        let named = same_file::Handle::from_file(open_for_review(&now).map_err(io)?).map_err(io)?;
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
/// "Absent" is decided by the revision's own entry for the path (`ls-tree -z <rev> -- <path>`,
/// one call per read), never by the text of a git error: a path with no file entry is
/// `Ok(None)`, and a failing `show` on a listed path is an error.
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
        let out = self.git.run_bytes(&["ls-tree", "-z", rev, "--", path])?;
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
    fn read_at(&self, rev: &str, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
        if !self.holds(rev, path)? {
            return Ok(None);
        }
        let too_large = || ReviewError::TooLarge {
            path: path.to_string(),
            limit: MAX_REVIEW_FILE_BYTES,
        };
        let spec = format!("{rev}:./{path}");
        match self.git.run_bytes_capped(
            &["show", "--no-ext-diff", "--no-textconv", &spec],
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
/// even if the file changes on disk meanwhile. An `Err` is never kept. The bytes held are
/// bounded by [`MAX_REVIEW_TOTAL_BYTES`]: a read that would pass it is refused and not kept,
/// while a read of what is already held costs nothing. Memory is O(bytes reviewed); a lookup is
/// O(1) expected plus the copy it returns.
pub struct CachedSource<'a> {
    inner: &'a dyn ContentSource,
    cap: u64,
    held: Mutex<Held>,
}

/// What a [`CachedSource`] holds.
#[derive(Default)]
struct Held {
    /// New-side reads by path.
    new: HashMap<String, Option<Vec<u8>>>,
    /// Old-side reads by path.
    old: HashMap<String, Option<Vec<u8>>>,
    /// Bytes held across both sides.
    bytes: u64,
}

/// Which side of a file a read is for.
#[derive(Clone, Copy)]
enum Face {
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

    fn capped(inner: &'a dyn ContentSource, cap: u64) -> Self {
        Self {
            inner,
            cap,
            held: Mutex::new(Held::default()),
        }
    }

    /// Reads one side of `path`, from the cache or from the inner source.
    fn get(&self, face: Face, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
        let mut held = self.held.lock().unwrap_or_else(PoisonError::into_inner);
        let side = match face {
            Face::New => &held.new,
            Face::Old => &held.old,
        };
        if let Some(bytes) = side.get(path) {
            return Ok(bytes.clone());
        }
        let read = match face {
            Face::New => self.inner.read(path)?,
            Face::Old => self.inner.read_old(path)?,
        };
        let len = read
            .as_ref()
            .map_or(0, |b| u64::try_from(b.len()).unwrap_or(u64::MAX));
        let total = held.bytes.saturating_add(len);
        if total > self.cap {
            // MAGI FIX (S6 loop 5): the error named no path and no total, so an operator could
            // not tell a higher cap from fewer files.
            return Err(ReviewError::InputTooLarge {
                limit: self.cap,
                held: held.bytes,
                path: path.to_string(),
            });
        }
        held.bytes = total;
        let side = match face {
            Face::New => &mut held.new,
            Face::Old => &mut held.old,
        };
        side.insert(path.to_string(), read.clone());
        Ok(read)
    }
}

impl ContentSource for CachedSource<'_> {
    /// # Errors
    ///
    /// The inner source's error (never kept), and [`ReviewError::InputTooLarge`] when keeping
    /// the read would pass the total cap.
    fn read(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
        self.get(Face::New, path)
    }

    fn read_old(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
        self.get(Face::Old, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let cached = CachedSource::with_total_cap(&Three, 5);
        assert_eq!(cached.read("a.rs").expect("fits"), Some(b"abc".to_vec()));
        // [CP2 MS2 cierre (sin MAGI)] the old side is held too, so it counts toward the same cap.
        assert!(matches!(
            cached.read_old("a.rs"),
            Err(ReviewError::InputTooLarge { limit: 5, .. })
        ));
        // [MAGI S6 loop 5, A2-b1] the error says what was held and which read passed the cap.
        let err = cached.read("b.rs").expect_err("over the cap");
        assert!(
            matches!(&err, ReviewError::InputTooLarge { limit: 5, held: 3, path } if path == "b.rs"),
            "{err:?}"
        );
        assert!(err.to_string().contains("b.rs"), "{err}");
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
                "ls-tree -z X -- a.rs",
                Ok("100644 blob 0000000000000000000000000000000000000000\ta.rs\0"),
            ),
            (
                "ls-tree -z X -- b.rs",
                Ok("100644 blob 0000000000000000000000000000000000000000\tb.rs\0"),
            ),
            ("ls-tree -z X -- missing.rs", Ok("")),
            (
                "show --no-ext-diff --no-textconv X:./a.rs",
                Ok("fn a() {}\n"),
            ),
            (
                "show --no-ext-diff --no-textconv X:./b.rs",
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
                    "ls-tree -z X -- big.rs" => Ok(
                        b"100644 blob 0000000000000000000000000000000000000000\tbig.rs\0".to_vec(),
                    ),
                    "ls-tree -z X -- cut.rs" => Ok(
                        b"100644 blob 0000000000000000000000000000000000000000\tcut.rs\0".to_vec(),
                    ),
                    "show --no-ext-diff --no-textconv X:./big.rs" => {
                        Ok(vec![
                            b'a';
                            usize::try_from(MAX_REVIEW_FILE_BYTES).expect("fits")
                                + 1
                        ])
                    }
                    "show --no-ext-diff --no-textconv X:./cut.rs" => {
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
        assert!(
            matches!(src.read("copy.rs"), Err(ReviewError::OutsideWorkspace(ref p)) if p == "copy.rs"),
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
    fn a_listed_name_that_is_not_utf8_never_stands_for_a_requested_path() {
        // [MAGI S6 MS2 loop 4, A2 Caspar] the listing was decoded lossily, so `a\xff.rs` read as
        // `a\u{FFFD}.rs`: a requested path that is valid UTF-8 matched a name it is not.
        struct Latin1Tree;
        impl GitCommand for Latin1Tree {
            fn run_bytes(&self, args: &[&str]) -> Result<Vec<u8>, ReviewError> {
                match args.join(" ").as_str() {
                    "ls-tree -z X -- a\u{FFFD}.rs" => Ok(
                        b"100644 blob 0000000000000000000000000000000000000000\ta\xff.rs\0"
                            .to_vec(),
                    ),
                    "ls-tree -z X -- b.rs" => Ok(
                        b"100644 blob 0000000000000000000000000000000000000000\tb.rs\0".to_vec(),
                    ),
                    "show --no-ext-diff --no-textconv X:./b.rs" => Ok(b"fn b() {}\n".to_vec()),
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
                    "ls-tree -z X -- a.rs" => Ok(
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
                    "show --no-ext-diff --no-textconv X:./a.rs" => Ok(b"fn a() {}\n".to_vec()),
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
    fn a_revision_read_lists_only_the_requested_path_never_the_whole_tree() {
        // [MAGI S6 loop 7, A2-c5] the existence check listed every blob of the revision, so a
        // monorepo past the 64 MiB git output cap failed the review before any model call.
        let git = FakeGit::from(&[
            (
                "ls-tree -z X -- a.rs",
                Ok("100644 blob 0000000000000000000000000000000000000000\ta.rs\0"),
            ),
            ("ls-tree -z X -- gone.rs", Ok("")),
            (
                "show --no-ext-diff --no-textconv X:./a.rs",
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
}
