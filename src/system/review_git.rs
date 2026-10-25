// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! The git runner of `code-review`: an untrusted repository's config and environment never run
//! code, and no git output takes more memory than its cap.
//!
//! Every call runs `git -C <workdir>` with [`NEUTRAL_GIT_CONFIG`] and, when the reviewed
//! repository's own config defines filter drivers, `-c` overrides that disable them; the child
//! gets only the tool environment allowlist plus `GIT_NO_LAZY_FETCH=1` (a partial clone never
//! downloads during a review) and `GIT_LITERAL_PATHSPECS=1` (a path is never a glob).

use std::collections::BTreeSet;
use std::ffi::OsString;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{mpsc, Mutex, PoisonError};
use std::time::Duration;

use magi_rs::review::error::ReviewError;
use magi_rs::review::source::{
    read_bounded, GitCommand, MAX_GIT_OUTPUT_BYTES, MAX_GIT_STDERR_BYTES,
};

use crate::system::path_guard::PathGuard;
use crate::tools::proc_group::tool_child_env;

/// `-c` settings every call carries, against the reviewed repository's config.
///
/// No hook, fsmonitor, external diff or pager runs; paths come back in UTF-8 and unquoted when
/// possible; the `a/`/`b/` prefixes and three lines of context the diff split relies on are fixed
/// (a repository's `diff.context=0` would silently shrink what the review sees); and
/// `protocol.allow=never` cuts every transport. On Git for Windows `/dev/null` names a directory
/// that does not exist, which disables hooks the same way.
pub(crate) const NEUTRAL_GIT_CONFIG: &[&str] = &[
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
];

/// The longest a git call may run: a git hung on a foreign lock or a network file system must
/// not hang the preparation, which runs before the review's clock starts.
pub(crate) const GIT_CALL_TIMEOUT_SECS: u64 = 300;

/// Total bytes the filter `-c` overrides may take: Windows caps a command line near 32 KiB, and a
/// repository with thousands of drivers would push every call past it and fail opaquely.
pub(crate) const MAX_FILTER_OVERRIDE_BYTES: usize = 8 * 1024;

/// The probe of the filter drivers git sees, with the scope of each entry (git 2.26 or later).
const FILTER_PROBE: &[&str] = &[
    "config",
    "--null",
    "--show-scope",
    "--get-regexp",
    "^filter\\.",
];

/// Scopes that belong to the reviewed repository, not to the operator.
const REPOSITORY_SCOPES: &[&str] = &["local", "worktree"];

/// What each neutralized driver is set to.
const NEUTRALIZED_FILTER_SETTINGS: &[&str] = &["clean=", "smudge=", "process=", "required=false"];

/// The exit code of `git config --get-regexp` when nothing matches.
const CONFIG_NO_MATCH_EXIT: i32 = 1;

/// Size of each read while draining stderr.
const DRAIN_CHUNK_BYTES: usize = 8 * 1024;

/// git runner with the reviewed repository's config and the caller's environment neutralized.
///
/// # Examples
///
/// ```ignore
/// let git = ScrubbedGit::new(workspace.to_path_buf());
/// let head = git.run(&["rev-parse", "--verify", "HEAD^{commit}"])?;
/// ```
pub(crate) struct ScrubbedGit {
    /// The directory every call runs in (`-C`).
    workdir: PathBuf,
    /// The stdout cap of each call.
    output_cap: u64,
    /// The filter overrides, probed once before the first command.
    filters: Mutex<Option<Vec<String>>>,
}

/// What a finished git call produced.
struct Finished {
    code: Option<i32>,
    stdout: Vec<u8>,
    stderr: String,
}

impl ScrubbedGit {
    /// A runner over `workdir`, with stdout capped at [`MAX_GIT_OUTPUT_BYTES`].
    pub(crate) fn new(workdir: PathBuf) -> Self {
        Self {
            workdir,
            output_cap: MAX_GIT_OUTPUT_BYTES,
            filters: Mutex::new(None),
        }
    }

    /// A runner with a custom stdout cap, so a test need not write 64 MiB.
    #[cfg(test)]
    fn with_output_cap(workdir: PathBuf, cap: u64) -> Self {
        Self {
            output_cap: cap,
            ..Self::new(workdir)
        }
    }

    /// The full command line for `args`: `-C <workdir>`, [`NEUTRAL_GIT_CONFIG`], the filter
    /// overrides probed so far, then `args` unchanged.
    pub(crate) fn argv(&self, args: &[&str]) -> Vec<OsString> {
        let filters = self
            .filters
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        let mut argv: Vec<OsString> = vec!["-C".into(), self.workdir.clone().into_os_string()];
        for setting in NEUTRAL_GIT_CONFIG
            .iter()
            .copied()
            .map(str::to_string)
            .chain(filters.into_iter().flatten())
        {
            argv.push("-c".into());
            argv.push(setting.into());
        }
        argv.extend(args.iter().map(OsString::from));
        argv
    }

    /// Probes the reviewed repository's filter drivers once (exit 1 is "none"); any other
    /// failure fails closed, so no command runs with a repository filter still active.
    fn ensure_filters(&self) -> Result<(), ReviewError> {
        if self
            .filters
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_some()
        {
            return Ok(());
        }
        let overrides = interpret_filter_probe(&self.execute(FILTER_PROBE)?)?;
        *self.filters.lock().unwrap_or_else(PoisonError::into_inner) = Some(overrides);
        Ok(())
    }

    /// Runs git with `args`, draining stdout (capped) and stderr (kept up to
    /// [`MAX_GIT_STDERR_BYTES`], read to the end) in parallel, within [`GIT_CALL_TIMEOUT_SECS`].
    fn execute(&self, args: &[&str]) -> Result<Finished, ReviewError> {
        let command = args.first().copied().unwrap_or_default();
        let mut child = Command::new("git")
            .args(self.argv(args))
            .env_clear()
            .envs(tool_child_env())
            .env("GIT_NO_LAZY_FETCH", "1")
            .env("GIT_LITERAL_PATHSPECS", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| ReviewError::Git(format!("cannot run git: {e}")))?;
        let (Some(stdout), Some(stderr)) = (child.stdout.take(), child.stderr.take()) else {
            kill(&mut child);
            return Err(ReviewError::Git("cannot read git's output".to_string()));
        };
        let stderr_reader = std::thread::spawn(move || drain_bounded(stderr, MAX_GIT_STDERR_BYTES));
        let (tx, rx) = mpsc::channel();
        let cap = self.output_cap;
        std::thread::spawn(move || {
            let _ = tx.send(read_bounded(stdout, cap));
        });
        let stdout = match rx.recv_timeout(Duration::from_secs(GIT_CALL_TIMEOUT_SECS)) {
            Ok(Ok(Some(bytes))) => bytes,
            Ok(Ok(None)) => {
                kill(&mut child);
                return Err(ReviewError::GitOutputTooLarge { limit: cap });
            }
            Ok(Err(e)) => {
                kill(&mut child);
                return Err(ReviewError::Git(format!("cannot read git's output: {e}")));
            }
            Err(_) => {
                kill(&mut child);
                return Err(ReviewError::Git(format!(
                    "git {command} did not finish in {GIT_CALL_TIMEOUT_SECS} s"
                )));
            }
        };
        let status = child
            .wait()
            .map_err(|e| ReviewError::Git(format!("cannot wait for git: {e}")))?;
        let stderr = stderr_reader.join().unwrap_or_default();
        Ok(Finished {
            code: status.code(),
            stdout,
            stderr: String::from_utf8_lossy(&stderr).trim().to_string(),
        })
    }
}

impl GitCommand for ScrubbedGit {
    /// # Errors
    ///
    /// [`ReviewError::Git`] when git cannot run, exits with an error (the message is its bounded
    /// stderr), or does not finish in time; [`ReviewError::GitOutputTooLarge`] past the cap.
    fn run_bytes(&self, args: &[&str]) -> Result<Vec<u8>, ReviewError> {
        self.ensure_filters()?;
        let finished = self.execute(args)?;
        if finished.code == Some(0) {
            Ok(finished.stdout)
        } else {
            Err(failure(
                args.first().copied().unwrap_or_default(),
                &finished,
            ))
        }
    }
}

/// What the filter probe means: the overrides to apply, or a fail-closed error.
///
/// Exit 0 is parsed by [`filter_overrides`]; exit 1 is "no filter entry anywhere"; any other exit
/// (or a signal) is an error, so no command runs with a repository filter left active.
///
/// # Errors
///
/// [`ReviewError::Git`] for a failing probe, and whatever [`filter_overrides`] refuses.
fn interpret_filter_probe(probe: &Finished) -> Result<Vec<String>, ReviewError> {
    match probe.code {
        Some(0) => filter_overrides(&probe.stdout),
        Some(CONFIG_NO_MATCH_EXIT) => Ok(Vec::new()),
        _ => Err(failure("config", probe)),
    }
}

/// The error of a failed call: its bounded stderr, or the exit code when stderr is empty.
fn failure(command: &str, finished: &Finished) -> ReviewError {
    if finished.stderr.is_empty() {
        let code = finished
            .code
            .map_or_else(|| "a signal".to_string(), |c| c.to_string());
        ReviewError::Git(format!("git {command} exited with {code}"))
    } else {
        ReviewError::Git(finished.stderr.clone())
    }
}

/// Kills a child and reaps it; both are best effort (it may already have exited).
fn kill(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

/// Reads `reader` to the end, keeping at most `cap` bytes: stopping early would kill the writer
/// with a broken pipe, and not draining would let it block on a full pipe.
fn drain_bounded(mut reader: impl Read, cap: u64) -> Vec<u8> {
    let cap = usize::try_from(cap).unwrap_or(usize::MAX);
    let mut kept = Vec::new();
    let mut chunk = [0u8; DRAIN_CHUNK_BYTES];
    while let Ok(n) = reader.read(&mut chunk) {
        if n == 0 {
            break;
        }
        let room = cap.saturating_sub(kept.len());
        kept.extend_from_slice(chunk.get(..n.min(room)).unwrap_or_default());
    }
    kept
}

/// The sandbox check of a review's `File`/`Directory` paths: [`PathGuard::validate`], whose
/// rejections (`..`, an absolute path elsewhere, a symlinked directory that leaves, a NUL byte)
/// all become [`ReviewError::OutsideWorkspace`] naming the path.
///
/// # Returns
///
/// The canonical path inside the workspace (a missing tail is appended lexically).
///
/// # Errors
///
/// [`ReviewError::OutsideWorkspace`] for any path the guard refuses.
pub(crate) fn review_path_guard(guard: &PathGuard, path: &Path) -> Result<PathBuf, ReviewError> {
    guard
        .validate(path)
        .map_err(|_| ReviewError::OutsideWorkspace(path.display().to_string()))
}

/// The `-c` values that neutralize the filter drivers the reviewed repository's own config
/// (scope `local` or `worktree`) defines, sorted by driver.
///
/// Reads the output of `git config --null --show-scope --get-regexp ^filter\.`: records of
/// `<scope> NUL <key>` followed by `LF <value> NUL`, or by `NUL` alone for a key without value.
/// The key ends at the first `LF` (a value may hold more). A name with a local entry is
/// neutralized whole, including what the operator's own config defines under it: the safe
/// direction. The operator's filters (`global`, `system`) keep running. O(n) in the input.
///
/// # Errors
///
/// [`ReviewError::Git`] for a truncated record, a key outside `filter.<name>.<variable>`, a
/// repository driver name with `=` (which `-c` cannot express), or overrides over
/// [`MAX_FILTER_OVERRIDE_BYTES`].
pub(crate) fn filter_overrides(config_z: &[u8]) -> Result<Vec<String>, ReviewError> {
    let unreadable = |what: String| {
        ReviewError::Git(format!(
            "cannot read the repository's filter configuration: {what}"
        ))
    };
    let mut names = BTreeSet::new();
    let mut rest = config_z;
    while !rest.is_empty() {
        let (scope, after) =
            split_at_nul(rest).ok_or_else(|| unreadable("a truncated record".into()))?;
        let (entry, after) =
            split_at_nul(after).ok_or_else(|| unreadable("a truncated record".into()))?;
        rest = after;
        let key = entry.split(|b| *b == b'\n').next().unwrap_or_default();
        let key = String::from_utf8_lossy(key);
        let name = key
            .strip_prefix("filter.")
            .and_then(|tail| tail.rsplit_once('.'))
            .filter(|(name, variable)| !name.is_empty() && !variable.is_empty())
            .map(|(name, _)| name)
            .ok_or_else(|| unreadable(format!("unexpected key {key:?}")))?;
        if REPOSITORY_SCOPES.iter().any(|s| s.as_bytes() == scope) {
            if name.contains('=') {
                return Err(unreadable(format!(
                    "a filter driver name with '=': {name:?}"
                )));
            }
            names.insert(name.to_string());
        }
    }
    let overrides: Vec<String> = names
        .iter()
        .flat_map(|name| {
            NEUTRALIZED_FILTER_SETTINGS
                .iter()
                .map(move |s| format!("filter.{name}.{s}"))
        })
        .collect();
    if overrides.iter().map(String::len).sum::<usize>() > MAX_FILTER_OVERRIDE_BYTES {
        let listed: Vec<&str> = names.iter().map(String::as_str).collect();
        return Err(ReviewError::Git(format!(
            "the repository defines more filter drivers than a review can disable ({} drivers: {})",
            listed.len(),
            listed.join(", ")
        )));
    }
    Ok(overrides)
}

/// Splits at the first NUL: the bytes before it and after it.
fn split_at_nul(bytes: &[u8]) -> Option<(&[u8], &[u8])> {
    let at = bytes.iter().position(|b| *b == 0)?;
    Some((bytes.get(..at)?, bytes.get(at + 1..)?))
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
    fn a_core_worktree_that_points_outside_the_workspace_is_never_read() {
        // [MAGI §6 MS2 A loop 1] core.worktree is repository config: an untrusted repository can point git's work
        // tree at a directory outside the workspace, and `git diff <HEAD>` then embeds that directory's files.
        let d = repo();
        let elsewhere = tempfile::tempdir().expect("tempdir");
        std::fs::write(d.path().join("a.rs"), "fn a() {}\n").expect("write");
        git(d.path(), &["add", "a.rs"]);
        git(d.path(), &["commit", "-q", "-m", "1"]);
        std::fs::write(elsewhere.path().join("a.rs"), "SENTINEL_OUTSIDE\n").expect("write");
        let target = elsewhere.path().to_str().expect("utf-8 tempdir");
        git(d.path(), &["config", "core.worktree", target]);
        let plain = std::process::Command::new("git")
            .arg("-C")
            .arg(d.path())
            .args(["diff", "--no-color", "HEAD"])
            .output()
            .expect("git runs");
        assert!(
            String::from_utf8_lossy(&plain.stdout).contains("SENTINEL_OUTSIDE"),
            "precondition: git reads the outside tree"
        );
        let allow = |p: &std::path::Path| -> Result<std::path::PathBuf, ReviewError> {
            Ok(p.to_path_buf())
        };
        match magi_rs::review::input::resolve_input(
            &ReviewInput::Uncommitted,
            d.path(),
            &ScrubbedGit::new(d.path().to_path_buf()),
            &allow,
        ) {
            Err(ReviewError::Git(msg)) => assert!(!msg.contains("SENTINEL"), "{msg}"),
            Err(other) => panic!("expected a Git error, got {other:?}"),
            Ok(r) => panic!(
                "the outside tree was read: {:?}",
                r.files.iter().map(|f| &f.diff).collect::<Vec<_>>()
            ),
        }
    }

    #[test]
    fn git_tells_a_missing_ref_and_a_root_commit_apart_from_a_broken_repository() {
        // [MAGI §6 MS2 A loop 1, measured with git 2.56] `rev-parse --verify --quiet` exits 1 with nothing on stderr
        // for a name that is no commit (and for a root commit's parent); a broken repository exits 128 with a message.
        let d = repo();
        std::fs::write(d.path().join("a.rs"), "fn a() {}\n").expect("write");
        git(d.path(), &["add", "a.rs"]);
        git(d.path(), &["commit", "-q", "-m", "root"]);
        let g = ScrubbedGit::new(d.path().to_path_buf());
        let allow = |p: &std::path::Path| -> Result<std::path::PathBuf, ReviewError> {
            Ok(p.to_path_buf())
        };
        let resolve = |input: ReviewInput| {
            magi_rs::review::input::resolve_input(&input, d.path(), &g, &allow)
        };
        assert!(matches!(
            resolve(ReviewInput::Commit(Some("nope".into()))),
            Err(ReviewError::RefNotFound(_))
        ));
        match resolve(ReviewInput::Commit(None)) {
            Ok(r) => assert!(matches!(
                r.source_kind,
                magi_rs::review::input::SourceKind::Revision { old_rev: None, .. }
            )),
            Err(e) => panic!("a root commit is reviewable: {e:?}"),
        }
        std::fs::write(d.path().join(".git").join("HEAD"), "garbage\n").expect("write");
        assert!(matches!(
            resolve(ReviewInput::Commit(None)),
            Err(ReviewError::Git(_))
        ));
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
        // A value with a `.` before its last LF: a key cut at the last LF would name another driver.
        let z = b"local\0filter.a.clean\nline1\nline2\0local\0filter.a.clean\nsecond\0local\0filter.inc.clean\nx\0\
                  local\0filter.b.clean\0local\0filter.a.clean\nx.y\nline2\0";
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
    fn the_filter_probe_reads_exit_one_as_no_drivers_and_fails_closed_on_any_other_failure() {
        // `git config --get-regexp` exits 1 when nothing matches: a host with no filter config at all.
        let probe = |code: Option<i32>, stdout: &[u8], stderr: &str| Finished {
            code,
            stdout: stdout.to_vec(),
            stderr: stderr.to_string(),
        };
        let none = interpret_filter_probe(&probe(Some(1), b"", "")).expect("exit 1 is no entries");
        assert!(none.is_empty());
        let local = probe(Some(0), b"local\0filter.evil.clean\nsh x\0", "");
        assert_eq!(interpret_filter_probe(&local).expect("parses").len(), 4);
        let operator = probe(
            Some(0),
            b"system\0filter.lfs.clean\ngit-lfs clean -- %f\0",
            "",
        );
        assert!(interpret_filter_probe(&operator)
            .expect("parses")
            .is_empty());
        let truncated = probe(Some(0), b"local\0filter.evil.clean\nsh x", "");
        assert!(interpret_filter_probe(&truncated).is_err());
        match interpret_filter_probe(&probe(Some(128), b"", "fatal: bad config line 1")) {
            Err(ReviewError::Git(msg)) => assert!(msg.contains("bad config line 1"), "{msg}"),
            other => panic!("expected a Git error, got {other:?}"),
        }
        match interpret_filter_probe(&probe(Some(3), b"", "")) {
            Err(ReviewError::Git(msg)) => {
                assert!(msg.contains("config") && msg.contains('3'), "{msg}")
            }
            other => panic!("expected a Git error, got {other:?}"),
        }
        assert!(matches!(
            interpret_filter_probe(&probe(None, b"", "")),
            Err(ReviewError::Git(_))
        ));
    }

    #[test]
    fn the_review_guard_maps_an_escaping_path_to_outside_workspace() {
        let outer = tempfile::tempdir().expect("tempdir");
        std::fs::write(outer.path().join("secret.txt"), "outside\n").expect("write");
        let ws = outer.path().join("ws");
        std::fs::create_dir_all(ws.join("src")).expect("mkdir");
        std::fs::write(ws.join("src").join("a.rs"), "fn a() {}\n").expect("write");
        let guard = PathGuard::new(ws.clone()).expect("guard");
        for escaping in [
            ws.join("..").join("fuera"),
            ws.join("../secret.txt"),
            outer.path().join("secret.txt"),
        ] {
            assert!(
                matches!(
                    review_path_guard(&guard, &escaping),
                    Err(ReviewError::OutsideWorkspace(_))
                ),
                "{}",
                escaping.display()
            );
        }
        let inside = review_path_guard(&guard, &ws.join("src").join("a.rs")).expect("inside");
        assert!(
            inside.starts_with(guard.workspace_root()),
            "{}",
            inside.display()
        );
    }

    #[cfg(unix)] // creating a symlink needs Developer Mode on Windows; the `..` cases above run everywhere.
    #[test]
    fn the_review_guard_refuses_a_symlinked_directory_that_escapes() {
        let outer = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(outer.path().join("elsewhere")).expect("mkdir");
        std::fs::write(outer.path().join("elsewhere").join("secret.rs"), "x\n").expect("write");
        let ws = outer.path().join("ws");
        std::fs::create_dir_all(&ws).expect("mkdir");
        std::os::unix::fs::symlink(outer.path().join("elsewhere"), ws.join("link"))
            .expect("symlink");
        let guard = PathGuard::new(ws.clone()).expect("guard");
        for escaping in [ws.join("link"), ws.join("link").join("secret.rs")] {
            assert!(
                matches!(
                    review_path_guard(&guard, &escaping),
                    Err(ReviewError::OutsideWorkspace(_))
                ),
                "{}",
                escaping.display()
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

    /// 800 LF files committed, then rewritten under a repo-local `core.autocrlf=true`: every git call
    /// that compares them with the worktree prints one warning per file, far over a pipe buffer.
    /// Returns the repository and the SHA of its commit.
    fn crlf_warning_repo() -> (tempfile::TempDir, String) {
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
        (d, sha)
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
        // git cuts a single `fatal:` line at 4096 bytes itself, so the call above cannot tell a capped
        // stderr from a whole one. This failing call prints one warning per file, many lines past the cap.
        let (crlf, sha) = crlf_warning_repo();
        let args = [
            "diff",
            "--exit-code",
            "--no-ext-diff",
            "--no-textconv",
            sha.as_str(),
        ];
        let pre = std::process::Command::new("git")
            .arg("-C")
            .arg(crlf.path())
            .args(args)
            .output()
            .expect("git runs");
        assert!(
            pre.status.code() == Some(1) && pre.stderr.len() > 4 * cap,
            "precondition: exit {:?}, {} bytes of stderr",
            pre.status.code(),
            pre.stderr.len()
        );
        match ScrubbedGit::new(crlf.path().to_path_buf()).run_bytes(&args) {
            Err(ReviewError::Git(msg)) => assert!(
                !msg.is_empty() && msg.len() <= cap + 64,
                "{} bytes",
                msg.len()
            ),
            Err(other) => panic!("expected a Git error, got {other:?}"),
            Ok(_) => panic!("expected a Git error, got a successful diff"),
        }
    }

    #[test]
    fn a_succeeding_command_with_more_stderr_than_a_pipe_holds_neither_deadlocks_nor_fails() {
        // [CP2 MS2 S1a loop 15, measured with git 2.56] 800 LF files under a repo-local core.autocrlf=true make
        // `git diff <sha>` print ~107 KB of warnings on stderr with exit 0, over the ~64 KiB pipe buffer: a serial drain
        // deadlocks, and a drain that stops reading at the cap gets git killed by EPIPE.
        let (d, sha) = crlf_warning_repo();
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

    #[test]
    fn a_tracked_ignored_file_in_a_subdirectory_is_named_like_the_enumeration_names_it() {
        // The workspace is below the repo root and .gitignore lives at the root: both functions must speak `sub/x`.
        let d = repo();
        std::fs::create_dir_all(d.path().join("ws").join("sub")).expect("mkdir");
        std::fs::write(d.path().join(".gitignore"), "*.env\n").expect("write");
        std::fs::write(d.path().join("ws/sub/tracked.env"), "HOST=internal\n").expect("write");
        std::fs::write(d.path().join("ws/sub/a.rs"), "fn a() {}\n").expect("write");
        git(
            d.path(),
            &[
                "add",
                "-f",
                ".gitignore",
                "ws/sub/tracked.env",
                "ws/sub/a.rs",
            ],
        );
        git(d.path(), &["commit", "-q", "-m", "1"]);
        let ws = d.path().join("ws");
        let g = ScrubbedGit::new(ws.clone());
        let allow = |p: &std::path::Path| -> Result<std::path::PathBuf, ReviewError> {
            Ok(p.to_path_buf())
        };
        let input = ReviewInput::Directory("sub".into());
        let listed =
            magi_rs::review::input::resolve_input(&input, &ws, &g, &allow).expect("resolves");
        let ignored =
            magi_rs::review::filter::ignored_paths(&input, "sub", &ws, &g).expect("one git call");
        assert_eq!(
            ignored.iter().map(String::as_str).collect::<Vec<_>>(),
            vec!["sub/tracked.env"]
        );
        let paths: Vec<&str> = listed.files.iter().map(|f| f.path.as_str()).collect();
        assert!(paths.contains(&"sub/tracked.env"), "{paths:?}");
    }

    #[test]
    fn an_uncommitted_review_in_a_sub_workspace_lists_nothing_outside_it() {
        // [CP2 MS2 S1a loop 12, pins T2.3's measured claim] `ls-files --others` without a pathspec stays in the cwd's subtree.
        let d = repo();
        std::fs::create_dir_all(d.path().join("ws")).expect("mkdir");
        std::fs::write(d.path().join("ws/a.rs"), "fn a() {}\n").expect("write");
        git(d.path(), &["add", "ws/a.rs"]);
        git(d.path(), &["commit", "-q", "-m", "1"]);
        std::fs::write(d.path().join("ws/new.rs"), "fn n() {}\n").expect("write");
        std::fs::write(d.path().join("outside.rs"), "fn o() {}\n").expect("write");
        let ws = d.path().join("ws");
        let allow = |p: &std::path::Path| -> Result<std::path::PathBuf, ReviewError> {
            Ok(p.to_path_buf())
        };
        let input = magi_rs::review::input::ReviewInput::Uncommitted;
        let r = magi_rs::review::input::resolve_input(
            &input,
            &ws,
            &ScrubbedGit::new(ws.clone()),
            &allow,
        )
        .expect("resolves");
        assert_eq!(
            r.files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(),
            vec!["new.rs"]
        );
    }

    #[test]
    fn the_ignored_query_takes_a_glob_character_literally() {
        // [CP2 MS2 S1b loop 14, measured with git 2.56] as a glob, `src[0]` named the tracked-and-ignored `src0`.
        let d = repo();
        std::fs::create_dir_all(d.path().join("src[0]")).expect("mkdir");
        std::fs::write(d.path().join("src[0]/f.rs"), "fn f() {}\n").expect("write");
        std::fs::write(d.path().join(".gitignore"), "src0\n").expect("write");
        std::fs::write(d.path().join("src0"), "x\n").expect("write");
        git(d.path(), &["add", "-f", "."]);
        git(d.path(), &["commit", "-q", "-m", "1"]);
        let input = magi_rs::review::input::ReviewInput::Directory("src[0]".into());
        let ignored = magi_rs::review::filter::ignored_paths(
            &input,
            "src[0]",
            d.path(),
            &ScrubbedGit::new(d.path().to_path_buf()),
        )
        .expect("one git call");
        assert!(ignored.is_empty(), "{ignored:?}");
    }

    #[test]
    fn a_file_requested_by_name_is_ignored_whether_tracked_or_not() {
        // [CP2 MS2 S1b loop 12, measured with git 2.56] the File query returns tracked-ignored AND untracked-ignored.
        let d = repo();
        std::fs::create_dir_all(d.path().join("ws").join("sub")).expect("mkdir");
        std::fs::write(d.path().join(".gitignore"), "*.env\n").expect("write");
        std::fs::write(d.path().join("ws/sub/tracked.env"), "HOST=internal\n").expect("write");
        std::fs::write(d.path().join("ws/sub/untracked.env"), "HOST=internal\n").expect("write");
        git(d.path(), &["add", "-f", ".gitignore", "ws/sub/tracked.env"]);
        git(d.path(), &["commit", "-q", "-m", "1"]);
        let ws = d.path().join("ws");
        let g = ScrubbedGit::new(ws.clone());
        for p in ["sub/tracked.env", "sub/untracked.env"] {
            let ignored =
                magi_rs::review::filter::ignored_paths(&ReviewInput::File(p.into()), p, &ws, &g)
                    .expect("one git call");
            assert!(ignored.contains(p), "{p}: {ignored:?}");
        }
    }
}
