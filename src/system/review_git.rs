// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! The git runner of `code-review`: an untrusted repository's config and environment never run code.

use std::ffi::OsString;
use std::path::PathBuf;

use magi_rs::review::error::ReviewError;
use magi_rs::review::source::GitCommand;

/// git runner with the reviewed repository's config neutralized.
#[allow(dead_code)] // Red stub
pub(crate) struct ScrubbedGit {
    workdir: PathBuf,
}

#[allow(dead_code)] // Red stub
impl ScrubbedGit {
    /// A runner over `workdir`.
    pub(crate) fn new(workdir: PathBuf) -> Self {
        Self { workdir }
    }

    /// A runner with a custom stdout cap (tests only).
    #[cfg(test)]
    fn with_output_cap(workdir: PathBuf, cap: u64) -> Self {
        let _ = cap;
        Self { workdir }
    }

    /// The command line for `args`.
    pub(crate) fn argv(&self, args: &[&str]) -> Vec<OsString> {
        let _ = args;
        Vec::new()
    }
}

impl GitCommand for ScrubbedGit {
    fn run_bytes(&self, args: &[&str]) -> Result<Vec<u8>, ReviewError> {
        let _ = args;
        Err(ReviewError::Io("not implemented".into()))
    }
}

/// The `-c` values that neutralize the reviewed repository's filter drivers.
#[allow(dead_code)] // Red stub
pub(crate) fn filter_overrides(config_z: &[u8]) -> Result<Vec<String>, ReviewError> {
    let _ = config_z;
    Ok(Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use magi_rs::review::input::ReviewInput;
    use magi_rs::review::source::{ContentSource, RevisionSource, MAX_GIT_STDERR_BYTES};

    fn git(dir: &std::path::Path, args: &[&str]) {
        let st = std::process::Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .status()
            .expect("git runs");
        assert!(st.success(), "git {args:?}");
    }

    fn repo() -> tempfile::TempDir {
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
        d
    }

    #[test]
    fn a_commit_on_another_branch_is_read_from_the_revision_not_the_worktree() {
        let d = repo();
        std::fs::write(d.path().join("a.rs"), "one\n").expect("write");
        git(d.path(), &["add", "a.rs"]);
        git(d.path(), &["commit", "-q", "-m", "x"]);
        git(d.path(), &["tag", "X"]);
        git(d.path(), &["checkout", "-q", "-b", "other"]);
        std::fs::write(d.path().join("a.rs"), "two\n").expect("write");
        let g = ScrubbedGit::new(d.path().to_path_buf());
        let src = RevisionSource::new(&g, "X".into(), None);
        assert_eq!(src.read("a.rs").expect("read"), Some(b"one\n".to_vec()));
        assert_eq!(src.read("missing.rs").expect("read"), None);
    }

    #[test]
    fn a_workspace_below_the_repo_root_reviews_only_its_own_subtree() {
        // [CP2 MS2 S1 loop 4] the workspace need not be the repo root.
        let d = repo();
        std::fs::create_dir_all(d.path().join("ws")).expect("mkdir");
        std::fs::write(d.path().join("ws/a.rs"), "one\n").expect("write");
        std::fs::write(d.path().join("outside.rs"), "one\n").expect("write");
        git(d.path(), &["add", "-A"]);
        git(d.path(), &["commit", "-q", "-m", "1"]);
        std::fs::write(d.path().join("ws/a.rs"), "two\n").expect("write");
        std::fs::write(d.path().join("outside.rs"), "two\n").expect("write");
        git(d.path(), &["commit", "-q", "-am", "2"]);
        let g = ScrubbedGit::new(d.path().join("ws"));
        let allow = |p: &std::path::Path| -> Result<std::path::PathBuf, ReviewError> {
            Ok(p.to_path_buf())
        };
        let r = magi_rs::review::input::resolve_input(
            &ReviewInput::Commit(None),
            &d.path().join("ws"),
            &g,
            &allow,
        )
        .expect("resolves");
        assert_eq!(
            r.files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(),
            vec!["a.rs"]
        );
        let src = RevisionSource::new(&g, "HEAD".into(), None);
        assert_eq!(src.read("a.rs").expect("read"), Some(b"two\n".to_vec()));
    }

    #[test]
    fn every_git_call_neutralizes_the_reviewed_repos_config() {
        let g = ScrubbedGit::new(std::path::PathBuf::from("/w"));
        let argv: Vec<String> = g
            .argv(&["diff", "--no-color", "HEAD"])
            .into_iter()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        for needle in [
            "-C",
            "core.fsmonitor=false",
            "core.hooksPath=/dev/null",
            "diff.external=",
            "core.pager=cat",
            "core.quotePath=false",
            "diff.noprefix=false",
            "diff.mnemonicPrefix=false",
            "diff.srcPrefix=a/",
            "diff.dstPrefix=b/",
            "diff.context=3",
            "diff.interHunkContext=0",
            "protocol.allow=never",
        ] {
            assert!(
                argv.iter().any(|a| a.contains(needle)),
                "{needle} missing from {argv:?}"
            );
        }
        assert_eq!(argv.last().map(String::as_str), Some("HEAD"));
    }

    #[test]
    fn a_git_output_over_the_cap_is_cut_and_typed() {
        // [CP2 MS2 S1a loop 9] stdout goes through read_bounded; past the cap git is killed and the error is typed.
        let d = repo();
        std::fs::write(d.path().join("a.rs"), "x".repeat(100)).expect("write");
        git(d.path(), &["add", "a.rs"]);
        git(d.path(), &["commit", "-q", "-m", "x"]);
        let capped = ScrubbedGit::with_output_cap(d.path().to_path_buf(), 10);
        assert!(matches!(
            capped.run_bytes(&["show", "HEAD:./a.rs"]),
            Err(ReviewError::GitOutputTooLarge { limit: 10 })
        ));
        let whole = ScrubbedGit::new(d.path().to_path_buf())
            .run_bytes(&["show", "HEAD:./a.rs"])
            .expect("under the cap");
        assert_eq!(whole.len(), 100);
    }

    #[test]
    fn a_filter_driver_from_the_reviewed_repos_config_never_runs() {
        // [CP2 MS2 S1a loop 10, measured with git 2.56] `diff <sha>` against the worktree runs the repo's `clean` filter;
        // --no-ext-diff and --no-textconv do not stop it. The filter would create PWNED in the repo root.
        let d = repo();
        std::fs::write(d.path().join("a.rs"), "one\n").expect("write");
        git(d.path(), &["add", "a.rs"]);
        git(d.path(), &["commit", "-q", "-m", "1"]);
        std::fs::write(d.path().join(".gitattributes"), "* filter=evil\n").expect("write");
        git(
            d.path(),
            &["config", "filter.evil.clean", "touch PWNED; cat"],
        );
        std::fs::write(d.path().join("a.rs"), "two\n").expect("write");
        let g = ScrubbedGit::new(d.path().to_path_buf());
        let head = g
            .run(&["rev-parse", "--verify", "HEAD^{commit}"])
            .expect("head");
        let diff = g
            .run(&[
                "diff",
                "--no-color",
                "--no-ext-diff",
                "--no-textconv",
                "--relative",
                head.trim(),
            ])
            .expect("diff");
        assert!(diff.contains("+two"), "{diff}");
        assert!(
            !d.path().join("PWNED").exists(),
            "the reviewed repo's filter ran"
        );
    }

    #[test]
    fn only_filter_drivers_from_the_reviewed_repos_own_config_are_neutralized() {
        // [CP2 MS2 S1a loop 10] `config --null --show-scope --get-regexp ^filter\.`: scope NUL key LF value NUL.
        let z = b"global\0filter.lfs.clean\ngit-lfs clean -- %f\0local\0filter.evil.clean\nsh x\0\
                  worktree\0filter.a.b.process\ny\0local\0filter.evil.required\ntrue\0";
        let want: Vec<String> = ["a.b", "evil"]
            .iter()
            .flat_map(|n| {
                ["clean=", "smudge=", "process=", "required=false"]
                    .map(|v| format!("filter.{n}.{v}"))
            })
            .collect();
        assert_eq!(filter_overrides(z).expect("parses"), want);
        assert!(filter_overrides(b"").expect("no entries").is_empty());
        assert!(
            filter_overrides(b"local\0filter.x=y.clean\nz\0").is_err(),
            "`-c` cannot carry a name with `=`"
        );
        // [CP2 MS2 S1a loop 12] a local entry on an operator's driver name neutralizes the whole name (`-c` beats every scope).
        let hijack =
            b"global\0filter.lfs.clean\ngit-lfs clean -- %f\0local\0filter.lfs.smudge\nsh evil\0";
        assert_eq!(filter_overrides(hijack).expect("parses").len(), 4);
    }

    #[test]
    fn filter_overrides_reads_every_shape_git_prints_and_refuses_what_it_cannot_parse() {
        // [CP2 MS2 S1a loop 11, measured with git 2.56] a multi-line value, one key twice, an included file (reported with
        // the including file's scope) and a value-less boolean key (`[filter "b"] clean`: no LF before its NUL).
        let z = b"local\0filter.a.clean\nline1\nline2\0local\0filter.a.clean\nsecond\0local\0filter.inc.clean\nx\0\
                  local\0filter.b.clean\0";
        let want: Vec<String> = ["a", "b", "inc"]
            .iter()
            .flat_map(|n| {
                ["clean=", "smudge=", "process=", "required=false"]
                    .map(|v| format!("filter.{n}.{v}"))
            })
            .collect();
        assert_eq!(filter_overrides(z).expect("parses"), want);
        for bad in [
            &b"local\0filter.a.clean\nx"[..],
            b"local",
            b"local\0notfilter.a.clean\nx\0",
            b"local\0filter.clean\nx\0",
        ] {
            assert!(
                filter_overrides(bad).is_err(),
                "{:?}",
                String::from_utf8_lossy(bad)
            );
        }
    }

    #[test]
    fn too_many_filter_drivers_to_neutralize_fail_closed() {
        // [CP2 MS2 S1a loop 13] thousands of drivers would push every git call past the OS command-line limit.
        let mut z = Vec::new();
        for i in 0..400 {
            z.extend_from_slice(format!("local\0filter.d{i}.clean\ncat\0").as_bytes());
        }
        assert!(filter_overrides(&z).is_err());
    }

    #[test]
    fn a_failing_git_call_keeps_only_a_bounded_stderr() {
        // [CP2 MS2 S1a loop 10] stderr is piped (the TUI owns the terminal) and cut at the cap; git echoes the long path.
        let d = repo();
        std::fs::write(d.path().join("a.rs"), "x\n").expect("write");
        git(d.path(), &["add", "a.rs"]);
        git(d.path(), &["commit", "-q", "-m", "x"]);
        let long = format!("HEAD:./{}", "x".repeat(10_000));
        let cap = usize::try_from(MAX_GIT_STDERR_BYTES).expect("fits");
        match ScrubbedGit::new(d.path().to_path_buf()).run_bytes(&["show", &long]) {
            Err(ReviewError::Git(msg)) => assert!(
                !msg.is_empty() && msg.len() <= cap + 64,
                "{} bytes",
                msg.len()
            ),
            other => panic!("expected a Git error, got {other:?}"),
        }
    }

    #[test]
    fn a_succeeding_command_with_more_stderr_than_a_pipe_holds_neither_deadlocks_nor_fails() {
        // [CP2 MS2 S1a loop 15, measured with git 2.56] 800 LF files under a repo-local core.autocrlf=true make
        // `git diff <sha>` print ~107 KB of warnings on stderr with exit 0, over the ~64 KiB pipe buffer: a serial drain
        // deadlocks, and a drain that stops reading at the cap gets git killed by EPIPE.
        let d = repo();
        let name = |i: usize| format!("file_with_a_reasonably_long_name_{i:04}.txt");
        for i in 0..800 {
            std::fs::write(d.path().join(name(i)), "a\n").expect("write");
        }
        git(d.path(), &["add", "."]);
        git(d.path(), &["commit", "-q", "-m", "1"]);
        git(d.path(), &["config", "core.autocrlf", "true"]);
        for i in 0..800 {
            std::fs::write(d.path().join(name(i)), "a\nb\n").expect("write");
        }
        let head = std::process::Command::new("git")
            .arg("-C")
            .arg(d.path())
            .args(["rev-parse", "HEAD"])
            .output()
            .expect("git runs")
            .stdout;
        let sha = String::from_utf8(head).expect("utf-8").trim().to_string();
        // [CP2 MS2 S1a loop 16] precondition: if git stopped printing these warnings the test would pass vacuously.
        let pre = std::process::Command::new("git")
            .arg("-C")
            .arg(d.path())
            .args(["diff", "--no-ext-diff", "--no-textconv", &sha])
            .output()
            .expect("git runs");
        assert!(
            pre.stderr.len() > 65_536,
            "precondition: only {} bytes of stderr",
            pre.stderr.len()
        );
        let root = d.path().to_path_buf();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(ScrubbedGit::new(root).run_bytes(&[
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                &sha,
            ]));
        });
        match rx.recv_timeout(std::time::Duration::from_secs(30)) {
            Ok(Ok(out)) => assert!(!out.is_empty()),
            Ok(Err(e)) => panic!("a succeeding git call failed: {e:?}"),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                panic!("git did not finish in 30 s: stderr is not drained")
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => panic!("the call panicked"),
        }
    }

    #[test]
    fn a_directory_outside_any_repository_is_walked_through_the_real_git() {
        // [CP2 MS2 S1a loop 15] the composition FakeGit cannot see: outside a repo the filter probe reads "no local
        // entries" and rev-parse fails into the disk check and the ordered walk, never a fail-closed error.
        let d = tempfile::tempdir().expect("tempdir");
        assert!(
            d.path().ancestors().all(|a| !a.join(".git").exists()),
            "TMPDIR sits inside a git repository"
        );
        std::fs::create_dir_all(d.path().join("src")).expect("mkdir");
        std::fs::write(d.path().join("src/a.rs"), "fn a() {}\n").expect("write");
        let allow = |p: &std::path::Path| -> Result<std::path::PathBuf, ReviewError> {
            Ok(p.to_path_buf())
        };
        let r = magi_rs::review::input::resolve_input(
            &ReviewInput::Directory("src".into()),
            d.path(),
            &ScrubbedGit::new(d.path().to_path_buf()),
            &allow,
        )
        .expect("walks");
        assert_eq!(
            r.files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(),
            vec!["src/a.rs"]
        );
    }

    #[test]
    fn a_directory_name_with_glob_characters_is_matched_literally() {
        // [CP2 MS2 S1a loop 14, measured with git 2.56] without GIT_LITERAL_PATHSPECS `src[0]` is a glob that also
        // matches the sibling file `src0`. [S1a loop 15] Here, with T2.3's enumeration, not in T2.4.
        let d = repo();
        std::fs::create_dir_all(d.path().join("src[0]")).expect("mkdir");
        std::fs::write(d.path().join("src[0]/f.rs"), "fn f() {}\n").expect("write");
        std::fs::write(d.path().join("src0"), "fn g() {}\n").expect("write");
        git(d.path(), &["add", "."]);
        git(d.path(), &["commit", "-q", "-m", "1"]);
        let allow = |p: &std::path::Path| -> Result<std::path::PathBuf, ReviewError> {
            Ok(p.to_path_buf())
        };
        let input = magi_rs::review::input::ReviewInput::Directory("src[0]".into());
        let r = magi_rs::review::input::resolve_input(
            &input,
            d.path(),
            &ScrubbedGit::new(d.path().to_path_buf()),
            &allow,
        )
        .expect("resolves");
        assert_eq!(
            r.files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(),
            vec!["src[0]/f.rs"]
        );
    }

    #[test]
    fn a_path_with_spaces_and_unicode_is_read_from_the_revision() {
        let d = repo();
        std::fs::create_dir_all(d.path().join("src")).expect("mkdir");
        std::fs::write(d.path().join("src").join("añadir módulo.rs"), "fn f() {}\n")
            .expect("write");
        git(d.path(), &["add", "-A"]);
        git(d.path(), &["commit", "-q", "-m", "u"]);
        let g = ScrubbedGit::new(d.path().to_path_buf());
        let src = RevisionSource::new(&g, "HEAD".into(), None);
        assert_eq!(
            src.read("src/añadir módulo.rs").expect("read"),
            Some(b"fn f() {}\n".to_vec())
        );
    }
}
