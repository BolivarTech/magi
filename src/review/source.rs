// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! The single content source of a review: every stage that reads code reads it through here.

use std::path::PathBuf;

use super::error::ReviewError;

/// Per-file byte cap: a larger file is never returned.
pub const MAX_REVIEW_FILE_BYTES: u64 = 8 * 1024 * 1024;
/// Byte cap of any one git output.
pub const MAX_GIT_OUTPUT_BYTES: u64 = 64 * 1024 * 1024;
/// Bytes of a failing git call's stderr that are kept.
pub const MAX_GIT_STDERR_BYTES: u64 = 4 * 1024;
/// Byte cap of everything one review reads.
pub const MAX_REVIEW_TOTAL_BYTES: u64 = 256 * 1024 * 1024;
/// Names a review never reads.
pub const NEVER_REVIEWED: &[&str] = &[".git", ".magi", ".magi-rs-memory.db"];

/// Runs git.
pub trait GitCommand: Send + Sync {
    /// Runs git with `args` and returns its stdout.
    ///
    /// # Errors
    ///
    /// [`ReviewError::Git`] when git fails.
    fn run_bytes(&self, args: &[&str]) -> Result<Vec<u8>, ReviewError>;

    /// Default: `run_bytes` decoded as UTF-8 (`Git` error if it is not).
    ///
    /// # Errors
    ///
    /// As [`GitCommand::run_bytes`].
    fn run(&self, args: &[&str]) -> Result<String, ReviewError> {
        let _ = args;
        Err(ReviewError::Io("not implemented".into()))
    }
}

/// Reads the content of a review.
pub trait ContentSource: Send + Sync {
    /// The new side (or the worktree).
    ///
    /// # Errors
    ///
    /// Implementation specific.
    fn read(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError>;
    /// The old side.
    ///
    /// # Errors
    ///
    /// Implementation specific.
    fn read_old(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError>;
}

/// Reads at most `cap` bytes.
///
/// # Errors
///
/// The reader's I/O error.
pub fn read_bounded(reader: impl std::io::Read, cap: u64) -> std::io::Result<Option<Vec<u8>>> {
    let _ = (reader, cap);
    Ok(None)
}

/// Whether a path is one a review never reads.
pub fn is_never_reviewed(path: &str) -> bool {
    let _ = path;
    false
}

/// The worktree.
pub struct WorktreeSource<'g> {
    _root: PathBuf,
    _git: Option<&'g dyn GitCommand>,
}

impl<'g> WorktreeSource<'g> {
    /// A worktree source with no old side.
    pub fn new(root: PathBuf) -> Self {
        Self {
            _root: root,
            _git: None,
        }
    }

    /// A worktree source whose old side is `old_rev`.
    pub fn with_old(root: PathBuf, git: &'g dyn GitCommand, old_rev: String) -> Self {
        let _ = old_rev;
        Self {
            _root: root,
            _git: Some(git),
        }
    }
}

impl ContentSource for WorktreeSource<'_> {
    fn read(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
        let _ = path;
        Err(ReviewError::Git("not implemented".into()))
    }

    fn read_old(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
        let _ = path;
        Err(ReviewError::Git("not implemented".into()))
    }
}

/// A git revision.
pub struct RevisionSource<'g> {
    _git: &'g dyn GitCommand,
}

impl<'g> RevisionSource<'g> {
    /// A source over `new_rev`, with `old_rev` as its old side.
    pub fn new(git: &'g dyn GitCommand, new_rev: String, old_rev: Option<String>) -> Self {
        let _ = (new_rev, old_rev);
        Self { _git: git }
    }
}

impl ContentSource for RevisionSource<'_> {
    fn read(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
        let _ = path;
        Err(ReviewError::Io("not implemented".into()))
    }

    fn read_old(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
        let _ = path;
        Err(ReviewError::Io("not implemented".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::review::test_support::FakeGit;

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
