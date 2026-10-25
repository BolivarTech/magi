// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! The single content source of a review: every stage that reads code reads it through here.
//!
//! A review reads its code from exactly one place, chosen once from the input: a git revision
//! ([`RevisionSource`], `git show <rev>:./<path>`) or the worktree ([`WorktreeSource`]), with an
//! optional old side for deleted lines. Nothing mixes the worktree with another revision.

use std::collections::{BTreeMap, BTreeSet, HashMap};
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
/// `.git/config`), and reads that canonical path with the per-file cap. A directory component
/// swapped for a symlink between the check and the open is a declared, accepted window.
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
    /// [`ReviewError::OutsideWorkspace`] when the path resolves outside the root,
    /// [`ReviewError::Protected`] when it resolves to a never-reviewed path,
    /// [`ReviewError::NotRegularFile`] when it resolves to something other than a regular file
    /// (a link to a directory), and [`ReviewError::TooLarge`] over [`MAX_REVIEW_FILE_BYTES`].
    /// A link whose target does not exist is `Ok(None)`, like a file that does not exist.
    fn read(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
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
        let file = std::fs::File::open(&canonical).map_err(io)?;
        match read_bounded(file, MAX_REVIEW_FILE_BYTES).map_err(io)? {
            Some(bytes) => Ok(Some(bytes)),
            None => Err(ReviewError::TooLarge {
                path: path.to_string(),
                limit: MAX_REVIEW_FILE_BYTES,
            }),
        }
    }

    fn read_old(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
        match &self.old {
            Some(old) => old.read(path),
            None => Ok(None),
        }
    }
}

/// Reads a git revision (`git show <rev>:./<path>`), never the worktree.
///
/// "Absent" is decided by the revision's path list (`ls-tree`, run once per revision), never by
/// the text of a git error: a path not listed is `Ok(None)`, and a failing `show` on a listed
/// path is an error.
pub struct RevisionSource<'g> {
    git: &'g dyn GitCommand,
    new_rev: String,
    old_rev: Option<String>,
    /// The path list of each revision read so far.
    listings: Mutex<BTreeMap<String, BTreeSet<String>>>,
}

impl<'g> RevisionSource<'g> {
    /// A source over `new_rev`, with `old_rev` as its old side.
    pub fn new(git: &'g dyn GitCommand, new_rev: String, old_rev: Option<String>) -> Self {
        Self {
            git,
            new_rev,
            old_rev,
            listings: Mutex::new(BTreeMap::new()),
        }
    }

    /// Whether `rev` holds `path`, listing `rev` once. O(log n) per lookup after the listing.
    fn holds(&self, rev: &str, path: &str) -> Result<bool, ReviewError> {
        let mut listings = self.listings.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(paths) = listings.get(rev) {
            return Ok(paths.contains(path));
        }
        let out = self
            .git
            .run_bytes(&["ls-tree", "-r", "--name-only", "-z", rev])?;
        let paths: BTreeSet<String> = out
            .split(|b| *b == 0)
            .filter(|name| !name.is_empty())
            .map(|name| String::from_utf8_lossy(name).into_owned())
            .collect();
        let holds = paths.contains(path);
        listings.insert(rev.to_string(), paths);
        Ok(holds)
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
        match self
            .git
            .run_bytes(&["show", "--no-ext-diff", "--no-textconv", &spec])
        {
            Ok(bytes) if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_REVIEW_FILE_BYTES => {
                Err(too_large())
            }
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
            return Err(ReviewError::InputTooLarge { limit: self.cap });
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
            Err(ReviewError::InputTooLarge { limit: 5 })
        ));
        assert!(matches!(
            cached.read("b.rs"),
            Err(ReviewError::InputTooLarge { limit: 5 })
        ));
        assert_eq!(
            cached.read("a.rs").expect("already held: adds nothing"),
            Some(b"abc".to_vec())
        );
    }

    // [CP2 MS2 S1 loop 4: absent vs failed, decided by ls-tree, never by error text]
    #[test]
    fn a_path_absent_from_the_revision_is_none_and_a_failing_show_is_an_error() {
        let git = FakeGit::from(&[
            ("ls-tree -r --name-only -z X", Ok("a.rs\0b.rs\0")),
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
                    "ls-tree -r --name-only -z X" => Ok(b"big.rs\0cut.rs\0".to_vec()),
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
}
