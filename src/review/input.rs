// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! What a review reviews: its input kinds and how they resolve to files and a content source.

use std::path::{Path, PathBuf};

use super::error::ReviewError;
use super::source::GitCommand;

/// What the caller asked to review.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReviewInput {
    /// One file, reviewed whole from the worktree.
    File(PathBuf),
    /// A directory, every file reviewed whole from the worktree.
    Directory(PathBuf),
    /// One commit (`None` is `HEAD`).
    Commit(Option<String>),
    /// The diff between two commits.
    Range {
        /// The old end.
        from: String,
        /// The new end.
        to: String,
    },
    /// The diff of `head` against its merge base with `base`.
    Branches {
        /// The base branch.
        base: String,
        /// The reviewed branch.
        head: String,
    },
    /// Staged, unstaged and untracked changes.
    Uncommitted,
}

/// One file to review.
pub struct ReviewFile {
    /// Path relative to the workspace, with `/` separators.
    pub path: String,
    /// Its diff section, or `None` when it is reviewed whole.
    pub diff: Option<String>,
}

/// Where the content of a review comes from.
pub enum SourceKind {
    /// The worktree, with an optional old side.
    Worktree {
        /// The old side's commit.
        old_rev: Option<String>,
    },
    /// A git revision.
    Revision {
        /// The new side's commit.
        new_rev: String,
        /// The old side's commit.
        old_rev: Option<String>,
    },
}

/// A resolved input.
pub struct ResolvedInput {
    /// The files to review.
    pub files: Vec<ReviewFile>,
    /// Where their content comes from.
    pub source_kind: SourceKind,
    /// Diff sections or names that are not UTF-8, by their lossy path.
    pub not_utf8: Vec<String>,
}

/// Rejects a ref git could read as an option or a range.
///
/// # Errors
///
/// [`ReviewError::InvalidRef`].
pub fn validate_ref(r: &str) -> Result<(), ReviewError> {
    let _ = r;
    Ok(())
}

/// Resolves an input to its files and content source.
///
/// # Errors
///
/// See [`ReviewError`].
pub fn resolve_input(
    input: &ReviewInput,
    workspace: &Path,
    git: &dyn GitCommand,
    guard: &dyn Fn(&Path) -> Result<PathBuf, ReviewError>,
) -> Result<ResolvedInput, ReviewError> {
    let _ = (input, workspace, git, guard);
    Err(ReviewError::Io("not implemented".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::review::test_support::FakeGit;

    fn allow_all(p: &Path) -> Result<PathBuf, ReviewError> {
        Ok(p.to_path_buf())
    }

    #[test]
    fn a_ref_starting_with_a_dash_is_rejected_before_git_runs() {
        for bad in ["--output=x", "-p", "", "a b", "a\0b", "a..b"] {
            assert!(
                matches!(validate_ref(bad), Err(ReviewError::InvalidRef(_))),
                "{bad:?}"
            );
        }
        assert!(validate_ref("HEAD~1").is_ok());
        assert!(validate_ref("feature/x").is_ok());
    }

    #[test]
    fn no_git_argument_defaults_to_head() {
        let git = FakeGit(
            [
                (
                    "rev-parse --verify HEAD^{commit}".to_string(),
                    Ok("abc\n".to_string()),
                ),
                (
                    "rev-parse --verify HEAD^^{commit}".to_string(),
                    Ok("p\n".to_string()),
                ),
                (
                    "diff-tree -p -r --root --no-color --no-ext-diff --no-textconv --relative abc"
                        .to_string(),
                    Ok(
                        "diff --git a/a.rs b/a.rs\n--- a/a.rs\n+++ b/a.rs\n@@ -1 +1 @@\n-x\n+y\n"
                            .to_string(),
                    ),
                ),
            ]
            .into_iter()
            .collect(),
        );
        let r = resolve_input(
            &ReviewInput::Commit(None),
            Path::new("/w"),
            &git,
            &allow_all,
        )
        .expect("resolves");
        assert_eq!(r.files.len(), 1);
        assert_eq!(r.files[0].path, "a.rs");
        assert!(
            matches!(r.source_kind, SourceKind::Revision { ref new_rev, ref old_rev }
            if new_rev == "abc" && old_rev.as_deref() == Some("p"))
        ); // [CP2 MS2 S1 loop 7] pinned SHAs
    }

    #[test]
    fn a_root_commit_is_reviewed_against_the_empty_tree() {
        let git = FakeGit::from(&[
            ("rev-parse --verify HEAD^{commit}", Ok("r\n")),
            ("rev-parse --verify HEAD^^{commit}", Err("fatal: no parent")),
            ("diff-tree -p -r --root --no-color --no-ext-diff --no-textconv --relative r",
             Ok("diff --git a/a.rs b/a.rs\nnew file mode 100644\n--- /dev/null\n+++ b/a.rs\n@@ -0,0 +1 @@\n+fn f() {}\n")),
        ]);
        let r = resolve_input(
            &ReviewInput::Commit(None),
            Path::new("/w"),
            &git,
            &allow_all,
        )
        .expect("a root commit is reviewable");
        assert_eq!(r.files.len(), 1);
        assert!(
            matches!(r.source_kind, SourceKind::Revision { ref old_rev, .. } if old_rev.is_none())
        );
    }

    #[test]
    fn an_unknown_ref_is_a_typed_error() {
        let git = FakeGit(
            [(
                "rev-parse --verify nope^{commit}".to_string(),
                Err("fatal".to_string()),
            )]
            .into_iter()
            .collect(),
        );
        assert!(matches!(
            resolve_input(
                &ReviewInput::Commit(Some("nope".into())),
                Path::new("/w"),
                &git,
                &allow_all
            ),
            Err(ReviewError::RefNotFound(_))
        ));
    }

    #[test]
    fn a_path_outside_the_workspace_is_rejected_by_the_guard() {
        let deny = |p: &Path| -> Result<PathBuf, ReviewError> {
            Err(ReviewError::OutsideWorkspace(p.display().to_string()))
        };
        let git = FakeGit(Default::default());
        assert!(matches!(
            resolve_input(
                &ReviewInput::File("../fuera".into()),
                Path::new("/w"),
                &git,
                &deny
            ),
            Err(ReviewError::OutsideWorkspace(_))
        ));
    }

    // [CP2 MS2 S1 loop 1] the other RV01 kinds, and the diff-to-file split.
    const ONE_HUNK: &str =
        "diff --git a/a.rs b/a.rs\n--- a/a.rs\n+++ b/a.rs\n@@ -1 +1 @@\n-x\n+y\n";

    #[test]
    fn a_range_is_reviewed_from_its_right_end_against_its_left_end() {
        let git = FakeGit::from(&[
            ("rev-parse --verify v1^{commit}", Ok("a\n")),
            ("rev-parse --verify v2^{commit}", Ok("b\n")),
            (
                "diff --no-color --no-ext-diff --no-textconv --relative a..b",
                Ok(ONE_HUNK),
            ),
        ]);
        let r = resolve_input(
            &ReviewInput::Range {
                from: "v1".into(),
                to: "v2".into(),
            },
            Path::new("/w"),
            &git,
            &allow_all,
        )
        .expect("resolves");
        assert_eq!(
            r.files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(),
            vec!["a.rs"]
        );
        assert!(r.files[0].diff.is_some());
        assert!(
            matches!(r.source_kind, SourceKind::Revision { ref new_rev, ref old_rev }
            if new_rev == "b" && old_rev.as_deref() == Some("a"))
        );
    }

    #[test]
    fn branches_are_reviewed_from_their_merge_base() {
        let git = FakeGit::from(&[
            ("rev-parse --verify main^{commit}", Ok("a\n")),
            ("rev-parse --verify feature/x^{commit}", Ok("b\n")),
            ("merge-base a b", Ok("mb1\n")),
            (
                "diff --no-color --no-ext-diff --no-textconv --relative mb1..b",
                Ok(ONE_HUNK),
            ),
        ]);
        let r = resolve_input(
            &ReviewInput::Branches {
                base: "main".into(),
                head: "feature/x".into(),
            },
            Path::new("/w"),
            &git,
            &allow_all,
        )
        .expect("resolves");
        assert_eq!(r.files.len(), 1);
        assert!(
            matches!(r.source_kind, SourceKind::Revision { ref new_rev, ref old_rev }
            if new_rev == "b" && old_rev.as_deref() == Some("mb1"))
        );
    }

    #[test]
    fn branches_without_a_common_ancestor_are_a_typed_error_that_says_what_to_do() {
        // [CP2 MS2 S1a loop 11] unrelated histories: `merge-base` exits 1 with no output, never an empty message.
        let git = FakeGit::from(&[
            ("rev-parse --verify main^{commit}", Ok("a\n")),
            ("rev-parse --verify orphan^{commit}", Ok("b\n")),
            ("merge-base a b", Err("")),
        ]);
        match resolve_input(
            &ReviewInput::Branches {
                base: "main".into(),
                head: "orphan".into(),
            },
            Path::new("/w"),
            &git,
            &allow_all,
        ) {
            Err(ReviewError::Git(msg)) => assert!(
                msg.contains("main") && msg.contains("orphan") && msg.contains("--from"),
                "{msg}"
            ),
            // [RED-DRYRUN 2026-10-19] ResolvedInput is not Debug: only the error side is formatted.
            Err(other) => panic!("expected a Git error, got {other:?}"),
            Ok(_) => panic!("expected a Git error, got a resolved input"),
        }
    }

    #[test]
    fn an_unknown_range_end_is_a_typed_error() {
        let git = FakeGit::from(&[
            ("rev-parse --verify v1^{commit}", Ok("a\n")),
            ("rev-parse --verify nope^{commit}", Err("fatal")),
        ]);
        assert!(matches!(
            resolve_input(
                &ReviewInput::Range {
                    from: "v1".into(),
                    to: "nope".into()
                },
                Path::new("/w"),
                &git,
                &allow_all
            ),
            Err(ReviewError::RefNotFound(_))
        ));
    }

    #[test]
    fn uncommitted_changes_are_the_diff_against_head_plus_untracked_files_whole() {
        let git = FakeGit::from(&[
            ("rev-parse --verify HEAD^{commit}", Ok("abc\n")),
            (
                "diff --no-color --no-ext-diff --no-textconv --relative abc",
                Ok(ONE_HUNK),
            ),
            (
                "ls-files --others --exclude-standard -z",
                Ok("new.rs\0dir/other.rs\0"),
            ),
        ]);
        let r = resolve_input(&ReviewInput::Uncommitted, Path::new("/w"), &git, &allow_all)
            .expect("resolves");
        let got: Vec<(&str, bool)> = r
            .files
            .iter()
            .map(|f| (f.path.as_str(), f.diff.is_some()))
            .collect();
        assert_eq!(
            got,
            vec![("a.rs", true), ("new.rs", false), ("dir/other.rs", false)]
        );
        // [CP2 MS2 S1 loop 2] the old side of an uncommitted diff is HEAD.
        assert!(
            matches!(r.source_kind, SourceKind::Worktree { ref old_rev } if old_rev.as_deref() == Some("abc"))
        );
    }

    #[test]
    fn uncommitted_changes_in_a_repo_without_commits_have_no_old_side() {
        let git = FakeGit::from(&[
            (
                "rev-parse --verify HEAD^{commit}",
                Err("fatal: needed a single revision"),
            ),
            // [CP2 MS2 S1 loop 3] staged files (`git add`, no commit yet) are --cached, and must not be lost.
            (
                "ls-files --cached --others --exclude-standard -z",
                Ok("staged.rs\0new.rs\0"),
            ),
        ]);
        let r = resolve_input(&ReviewInput::Uncommitted, Path::new("/w"), &git, &allow_all)
            .expect("resolves");
        let got: Vec<(&str, bool)> = r
            .files
            .iter()
            .map(|f| (f.path.as_str(), f.diff.is_none()))
            .collect();
        assert_eq!(got, vec![("staged.rs", true), ("new.rs", true)]);
        assert!(matches!(
            r.source_kind,
            SourceKind::Worktree { old_rev: None }
        ));
    }

    #[test]
    fn a_hostile_ref_is_rejected_by_resolve_input_before_any_git_call() {
        // [CP2 MS2 S1 loop 2] FakeGit has no entries: any git call is a `Git` error, so only validate_ref yields InvalidRef.
        let git = FakeGit(Default::default());
        for input in [
            ReviewInput::Commit(Some("--output=x".into())),
            ReviewInput::Range {
                from: "v1".into(),
                to: "-p".into(),
            },
            ReviewInput::Branches {
                base: "--upload-pack=x".into(),
                head: "main".into(),
            },
        ] {
            assert!(
                matches!(
                    resolve_input(&input, Path::new("/w"), &git, &allow_all),
                    Err(ReviewError::InvalidRef(_))
                ),
                "{input:?}"
            );
        }
    }

    #[test]
    fn a_diff_section_that_is_not_utf8_is_set_aside_and_the_rest_is_reviewed() {
        struct BytesGit(Vec<u8>);
        impl GitCommand for BytesGit {
            fn run_bytes(&self, args: &[&str]) -> Result<Vec<u8>, ReviewError> {
                match args.join(" ").as_str() {
                    "rev-parse --verify v1^{commit}" | "rev-parse --verify v2^{commit}" => {
                        Ok(b"x\n".to_vec())
                    }
                    "diff --no-color --no-ext-diff --no-textconv --relative x..x" => {
                        Ok(self.0.clone())
                    }
                    other => Err(ReviewError::Git(format!("unexpected: {other}"))),
                }
            }
        }
        let mut diff = b"diff --git a/latin1.rs b/latin1.rs\n--- a/latin1.rs\n+++ b/latin1.rs\n@@ -1 +1 @@\n-x\n+caf\xe9\n".to_vec();
        diff.extend_from_slice(ONE_HUNK.as_bytes());
        let r = resolve_input(
            &ReviewInput::Range {
                from: "v1".into(),
                to: "v2".into(),
            },
            Path::new("/w"),
            &BytesGit(diff),
            &allow_all,
        )
        .expect("one bad section does not sink the review");
        assert_eq!(
            r.files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(),
            vec!["a.rs"]
        );
        assert_eq!(r.not_utf8, vec!["latin1.rs".to_string()]);
    }

    #[test]
    fn a_merge_commit_with_an_empty_diff_is_nothing_to_review() {
        let git = FakeGit::from(&[
            ("rev-parse --verify M^{commit}", Ok("m\n")),
            ("rev-parse --verify M^^{commit}", Ok("p\n")),
            (
                "diff-tree -p -r --root --no-color --no-ext-diff --no-textconv --relative m",
                Ok(""),
            ),
        ]);
        assert!(matches!(
            resolve_input(
                &ReviewInput::Commit(Some("M".into())),
                Path::new("/w"),
                &git,
                &allow_all
            ),
            Err(ReviewError::NothingToReview)
        ));
    }

    #[test]
    fn a_file_input_that_names_a_directory_is_not_reviewable() {
        // [CP2 MS2 S1a loop 11] the T2.3/T2.4 seam: a directory sent as a file is typed, not an opaque I/O error.
        let d = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(d.path().join("src")).expect("mkdir");
        let git = FakeGit(Default::default());
        assert!(matches!(
            resolve_input(&ReviewInput::File("src".into()), d.path(), &git, &allow_all),
            Err(ReviewError::NotReviewable {
                cause: "directory",
                ..
            })
        ));
    }

    #[test]
    fn a_file_input_is_reviewed_whole_from_the_worktree() {
        let git = FakeGit(Default::default());
        let r = resolve_input(
            &ReviewInput::File("src/a.rs".into()),
            Path::new("/w"),
            &git,
            &allow_all,
        )
        .expect("resolves");
        assert_eq!(r.files.len(), 1);
        assert_eq!(
            (r.files[0].path.as_str(), r.files[0].diff.is_none()),
            ("src/a.rs", true)
        );
        assert!(matches!(
            r.source_kind,
            SourceKind::Worktree { old_rev: None }
        ));
    }

    #[test]
    fn a_directory_in_a_repo_lists_what_git_does_not_ignore_and_that_exists() {
        let d = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(d.path().join("src")).expect("mkdir");
        std::fs::write(d.path().join("src/a.rs"), "fn a() {}\n").expect("write");
        std::fs::write(d.path().join("src/b.rs"), "fn b() {}\n").expect("write");
        std::fs::create_dir_all(d.path().join("src/vendored")).expect("mkdir"); // [CP2 MS2 S1 loop 8] a submodule
                                                                                // `src/gone.rs` is tracked but deleted from the tree; `.env` is ignored, so ls-files never names it.
        let git = FakeGit::from(&[
            ("rev-parse --is-inside-work-tree", Ok("true\n")),
            (
                "ls-files --cached --others --exclude-standard -z -- src",
                Ok("src/a.rs\0src/b.rs\0src/gone.rs\0src/vendored\0"),
            ),
        ]);
        let r = resolve_input(
            &ReviewInput::Directory("src".into()),
            d.path(),
            &git,
            &allow_all,
        )
        .expect("resolves");
        assert_eq!(
            r.files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(),
            vec!["src/a.rs", "src/b.rs"]
        );
        assert!(r.files.iter().all(|f| f.diff.is_none()));
    }

    #[test]
    fn the_workspace_itself_as_a_directory_uses_the_dot_pathspec() {
        let d = tempfile::tempdir().expect("tempdir");
        std::fs::write(d.path().join("a.rs"), "fn a() {}\n").expect("write");
        let git = FakeGit::from(&[
            ("rev-parse --is-inside-work-tree", Ok("true\n")),
            (
                "ls-files --cached --others --exclude-standard -z -- .",
                Ok("a.rs\0"),
            ),
        ]);
        let r = resolve_input(
            &ReviewInput::Directory(".".into()),
            d.path(),
            &git,
            &allow_all,
        )
        .expect("resolves");
        assert_eq!(
            r.files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(),
            vec!["a.rs"]
        );
    }

    #[test]
    fn a_directory_outside_a_repo_is_walked_in_order() {
        let d = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(d.path().join("src/sub")).expect("mkdir");
        std::fs::write(d.path().join("src/b.rs"), "b\n").expect("write");
        std::fs::write(d.path().join("src/sub/a.rs"), "a\n").expect("write");
        let git = FakeGit::from(&[(
            "rev-parse --is-inside-work-tree",
            Err("fatal: not a git repository"),
        )]);
        // [CP2 MS2 S1b loop 13] a `.git` above TMPDIR would make the production rule (correctly) fail closed.
        assert!(
            d.path().ancestors().all(|a| !a.join(".git").exists()),
            "TMPDIR sits inside a git repository"
        );
        let r = resolve_input(
            &ReviewInput::Directory("src".into()),
            d.path(),
            &git,
            &allow_all,
        )
        .expect("resolves");
        assert_eq!(
            r.files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(),
            vec!["src/b.rs", "src/sub/a.rs"]
        );
    }

    #[test]
    fn a_git_failure_inside_a_repository_fails_closed_instead_of_walking() {
        // [CP2 MS2 S1a loop 11] a broken git (missing binary, dubious ownership) must not drop .gitignore in silence.
        let d = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(d.path().join(".git")).expect("mkdir");
        std::fs::create_dir_all(d.path().join("src")).expect("mkdir");
        std::fs::write(d.path().join("src/a.rs"), "fn a() {}\n").expect("write");
        let git = FakeGit::from(&[(
            "rev-parse --is-inside-work-tree",
            Err("fatal: detected dubious ownership"),
        )]);
        assert!(matches!(
            resolve_input(
                &ReviewInput::Directory("src".into()),
                d.path(),
                &git,
                &allow_all
            ),
            Err(ReviewError::Git(_))
        ));
    }

    #[test]
    fn a_multi_file_diff_is_split_by_its_real_paths() {
        let diff = concat!(
            "diff --git a/src/añadir módulo.rs b/src/añadir módulo.rs\n",
            "--- a/src/añadir módulo.rs\t\n+++ b/src/añadir módulo.rs\t\n@@ -1 +1 @@\n-x\n+y\n",
            "diff --git a/old_name.rs b/new_name.rs\nsimilarity index 90%\nrename from old_name.rs\nrename to new_name.rs\n",
            "--- a/old_name.rs\n+++ b/new_name.rs\n@@ -1 +1 @@\n-x\n+y\n",
            "diff --git a/gone.rs b/gone.rs\ndeleted file mode 100644\n--- a/gone.rs\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n",
            "diff --git a/logo.png b/logo.png\nBinary files a/logo.png and b/logo.png differ\n",
            "diff --git a/run.sh b/run.sh\nold mode 100644\nnew mode 100755\n",
            "diff --git \"a/a\\tb.rs\" \"b/a\\tb.rs\"\n--- \"a/a\\tb.rs\"\n+++ \"b/a\\tb.rs\"\n@@ -1 +1 @@\n-x\n+y\n",
            // [CP2 MS2 S1 loop 8] a submodule bump is not reviewed; a named escape (`\v`) is decoded.
            "diff --git a/sub b/sub\nindex 1111111..2222222 160000\n--- a/sub\n+++ b/sub\n@@ -1 +1 @@\n",
            "-Subproject commit 1111111\n+Subproject commit 2222222\n",
            "diff --git \"a/b\\vc.rs\" \"b/b\\vc.rs\"\n--- \"a/b\\vc.rs\"\n+++ \"b/b\\vc.rs\"\n@@ -1 +1 @@\n-x\n+y\n",
        );
        let git = FakeGit::from(&[
            ("rev-parse --verify v1^{commit}", Ok("a\n")),
            ("rev-parse --verify v2^{commit}", Ok("b\n")),
            (
                "diff --no-color --no-ext-diff --no-textconv --relative a..b",
                Ok(diff),
            ),
        ]);
        let r = resolve_input(
            &ReviewInput::Range {
                from: "v1".into(),
                to: "v2".into(),
            },
            Path::new("/w"),
            &git,
            &allow_all,
        )
        .expect("resolves");
        assert_eq!(
            r.files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(),
            vec![
                "src/añadir módulo.rs",
                "new_name.rs",
                "gone.rs",
                "logo.png",
                "a\tb.rs",
                "b\u{b}c.rs"
            ]
        );
        // Each file carries only its own section.
        assert!(r.files[2]
            .diff
            .as_deref()
            .is_some_and(|t| t.contains("-x") && !t.contains("+y")));
    }

    #[test]
    fn no_prefix_of_a_hostile_diff_panics_the_splitter() {
        // [CP2 MS2 S1a loop 9] release is panic=abort: the diff split must survive any truncation of hostile input.
        let hostile = concat!(
            "diff --git \"a/x\\",
            "\ndiff --git a/ b/\n+++ \n--- \"\n",
            "diff --git a/s b/s\nindex 1..2 160000\n",
            "diff --git a/b.bin b/b.bin\nGIT binary patch\nliteral 12\nzcm\n",
            "diff --git \"a/\\9\" b/q\nrename to \"\\9\n",
            "+++ b/\t\n",
            "@@ -1,99999999999 +1 @@\r\n+x\r\n"
        );
        for end in hostile
            .char_indices()
            .map(|(i, _)| i)
            .chain([hostile.len()])
        {
            let git = FakeGit::from(&[
                ("rev-parse --verify v1^{commit}", Ok("a\n")),
                ("rev-parse --verify v2^{commit}", Ok("b\n")),
                (
                    "diff --no-color --no-ext-diff --no-textconv --relative a..b",
                    Ok(&hostile[..end]),
                ),
            ]);
            let _ = resolve_input(
                &ReviewInput::Range {
                    from: "v1".into(),
                    to: "v2".into(),
                },
                Path::new("/w"),
                &git,
                &allow_all,
            );
        }
    }

    #[test]
    fn a_header_only_section_takes_its_path_from_the_symmetric_split() {
        // [CP2 MS2 S1a loop 10, headers as git 2.56 writes them] binary sections carry only `diff --git`; the halves are equal.
        let diff = concat!(
            "diff --git a/lib b/util.so b/lib b/util.so\nindex 1..2 100644\nBinary files a/lib b/util.so and b/lib b/util.so differ\n",
            "diff --git a/my file.bin b/my file.bin\nindex 1..2 100644\nBinary files a/my file.bin and b/my file.bin differ\n",
            "diff --git \"a/q\\tz.bin\" \"b/q\\tz.bin\"\nindex 1..2 100644\nBinary files \"a/q\\tz.bin\" and \"b/q\\tz.bin\" differ\n",
            "diff --git a/old name.rs b/new name.rs\nsimilarity index 90%\nrename from old name.rs\nrename to new name.rs\n",
            "--- a/old name.rs\t\n+++ b/new name.rs\t\n@@ -1 +1 @@\n-x\n+y\n",
            "diff --git a/x b/y\nBinary files a/x and b/y differ\n", // asymmetric, no rename: git never writes it
        );
        let git = FakeGit::from(&[
            ("rev-parse --verify v1^{commit}", Ok("a\n")),
            ("rev-parse --verify v2^{commit}", Ok("b\n")),
            (
                "diff --no-color --no-ext-diff --no-textconv --relative a..b",
                Ok(diff),
            ),
        ]);
        let r = resolve_input(
            &ReviewInput::Range {
                from: "v1".into(),
                to: "v2".into(),
            },
            Path::new("/w"),
            &git,
            &allow_all,
        )
        .expect("resolves");
        assert_eq!(
            r.files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(),
            vec!["lib b/util.so", "my file.bin", "q\tz.bin", "new name.rs"]
        );
    }
}
