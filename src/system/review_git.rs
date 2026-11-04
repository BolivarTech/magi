// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! The git runner of `code-review`: an untrusted repository's config and environment never run
//! code, and no git output takes more memory than its cap.
//!
//! Every call runs `git -C <workdir>` with [`NEUTRAL_GIT_CONFIG`] and, when git's config (any
//! scope: system, global, the repository's own) defines filter drivers, `-c` overrides that
//! disable them; the child
//! gets only the tool environment allowlist plus `GIT_NO_LAZY_FETCH=1` (a partial clone never
//! downloads during a review) and `GIT_LITERAL_PATHSPECS=1` (a path is never a glob).
//!
//! A repository's `[alias]` entries are not overridden because they cannot reach a call: every
//! subcommand the review runs (`config`, `rev-parse`, `cat-file`, `merge-base`, `diff`,
//! `diff-tree`, `ls-files`, `ls-tree`, `show`) is a git built-in, and git ignores an alias whose
//! name is a built-in command (documented in `git help config`, `alias.*`; measured with git
//! 2.56: a `!touch` alias named like each of them never runs, while one named `zz` does).
//! Pinned by `a_repo_alias_never_shadows_a_git_subcommand_the_review_runs`; a new call must use a
//! built-in and join that test's list.
// MAGI FIX (S6 loop 10): A1-c1 refuted with real git; the alias boundary is stated here.
//!
//! It parses untrusted bytes (the filter probe, git's listings, a bounded stderr), so it carries
//! the same no-panic net as `magi_rs::review`: `panic = "abort"` would end the process on hostile
//! input with no report.
// MAGI FIX (S6 loop 6): the review module's deny net now covers its git runner too.
#![cfg_attr(
    not(test),
    deny(
        clippy::indexing_slicing,
        clippy::string_slice,
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable
    )
)]

use std::collections::BTreeSet;
use std::ffi::OsString;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{mpsc, Mutex, PoisonError};
use std::time::{Duration, Instant};

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
/// (a repository's `diff.context=0` would silently shrink what the review sees);
/// `protocol.allow=never` cuts every transport; and `core.useReplaceRefs=false` reads the
/// objects the pinned commit names, never a repository's replacements. On Git for Windows
/// `/dev/null` names a directory that does not exist, which disables hooks the same way.
///
/// End-of-line conversion (`core.autocrlf`, `text`/`eol` attributes) is left as the repository
/// sets it: an uncommitted review's diff may show the converted form while the gate reads the
/// raw worktree bytes. Conversion keeps the line count, so hunk lines, the RV09 ranges and the
/// gate's window still align; only the line endings shown can differ.
///
/// `safe.directory` is deliberately not set: git's refusal of a repository another user owns is
/// a defense layer this list does not replace, so such a review fails with git's "dubious
/// ownership" error until the operator trusts the path in their own global config.
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
    // MAGI FIX (S6 loop 5): `refs/replace/*` is repository data; honoring it reviewed a
    // substitute object instead of what the pinned commit holds (RV21).
    "core.useReplaceRefs=false",
];

/// The file name git has in a PATH entry.
#[cfg(windows)]
const GIT_PROGRAM: &str = "git.exe";
/// The file name git has in a PATH entry.
#[cfg(not(windows))]
const GIT_PROGRAM: &str = "git";

/// The git executable a review runs: [`GIT_PROGRAM`] in the first entry of `path_var` (a
/// `PATH` value) that is absolute, lies outside `workdir` and outside the outermost repository
/// that encloses it (the farthest ancestor holding a `.git` entry), and holds it as a file.
///
/// A bare `Command::new("git")` let the platform search pick the program: a relative entry
/// (`.`, an empty one) resolves against the process's working directory, often the reviewed
/// repository, and an entry inside the repository (a vendored `tools/`) is repository data;
/// either could run a planted git before any setting of [`NEUTRAL_GIT_CONFIG`] applied. On
/// Windows the search also tries the directory of Magi's own executable first. (Rust 1.58 and
/// later no longer search the working directory itself on Windows; a relative entry still
/// does, which the test below reproduces.) Resolving to an absolute path once closes all of
/// these; the cost is that a git reached only through a relative PATH entry is not found.
/// When the workspace is a subdirectory of a repository (or of a submodule's superproject),
/// the whole repository is refused, not just the workspace: a git installed under a directory
/// that is itself a repository (a home directory kept under git) is therefore not found either.
///
/// # Arguments
/// * `path_var` - The `PATH` value to search.
/// * `workdir` - The reviewed workspace; no entry inside it, or inside the outermost repository
///   that encloses it, may provide git.
///
/// # Returns
/// The absolute path of git, or `None` when no entry qualifies or `workdir` cannot be
/// canonicalized (the check that git lies outside it needs its canonical form, so it fails
/// closed).
// MAGI FIX (S6 loop 9): git was spawned by bare name, so a planted git could run (A1-c1).
// MAGI FIX (S6 loop 33): a workspace that could not be canonicalized fell back to its raw
// path, so the outside-the-workspace check could stop protecting; it now fails closed.
// MAGI FIX (S6 loop 35): only the workspace was refused, so with the workspace inside a
// repository a git planted elsewhere in that repository ran; the repository is refused too.
fn resolve_git(path_var: &std::ffi::OsStr, workdir: &Path) -> Option<PathBuf> {
    let root = workdir.canonicalize().ok()?;
    let repository = root
        .ancestors()
        .filter(|dir| dir.join(".git").exists())
        .last()
        .unwrap_or(&root);
    std::env::split_paths(path_var)
        .filter(|dir| dir.is_absolute())
        .map(|dir| dir.join(GIT_PROGRAM))
        .filter(|candidate| candidate.is_file())
        .find(|candidate| {
            // `repository` is `root` or an ancestor of it, so this also refuses the workspace.
            candidate
                .canonicalize()
                .is_ok_and(|real| !real.starts_with(repository) && !candidate.starts_with(workdir))
        })
}

/// The longest a git call may run: a git hung on a foreign lock or a network file system must
/// not hang the preparation, which runs before the review's clock starts.
pub(crate) const GIT_CALL_TIMEOUT_SECS: u64 = 300;
/// How long git's stderr may still be drained after git exited, when the call's deadline is
/// closer: a descendant that kept the pipe open must not hold the preparation past it.
const STDERR_DRAIN_GRACE_MS: u64 = 1_000;

/// The first pause while waiting for a git whose output is closed to exit; it usually has.
const WAIT_POLL_FIRST: Duration = Duration::from_millis(1);

/// The longest pause between two checks of a git that has not exited yet.
const WAIT_POLL_MAX: Duration = Duration::from_millis(50);

/// Total bytes the filter `-c` overrides may take: Windows caps a command line near 32 KiB, and a
/// repository with thousands of drivers would push every call past it and fail opaquely.
pub(crate) const MAX_FILTER_OVERRIDE_BYTES: usize = 8 * 1024;
/// The key prefix of a filter driver setting (`filter.<name>.<variable>`).
const FILTER_KEY_PREFIX: &str = "filter.";
/// How many driver names the over-the-cap error names; the error stays short whatever the config.
const FILTER_DRIVERS_NAMED_IN_ERROR: usize = 10;

/// The probe of the filter drivers git sees, with the scope of each entry (git 2.26 or later).
const FILTER_PROBE: &[&str] = &[
    "config",
    "--null",
    "--show-scope",
    "--get-regexp",
    "^filter\\.",
];

/// What each neutralized driver is set to.
const NEUTRALIZED_FILTER_SETTINGS: &[&str] = &["clean=", "smudge=", "process=", "required=false"];

/// The exit code of `git config --get-regexp` when nothing matches.
const CONFIG_NO_MATCH_EXIT: i32 = 1;

/// The exit code of `rev-parse --verify --quiet` for a name that is no commit (with an empty
/// stderr); a broken repository exits 128 with a message.
const QUERY_NO_MATCH_EXIT: i32 = 1;

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
    /// The git executable, resolved once ([`resolve_git`]).
    program: std::sync::OnceLock<Option<PathBuf>>,
    /// Extra variables for every git child, so a test can point git at an operator config.
    #[cfg(test)]
    test_env: Vec<(String, String)>,
}

/// The prefix of an empty repository's directory name under the system temporary directory.
const EMPTY_GIT_DIR_PREFIX: &str = "magi-review-empty-git";

/// How many names [`EmptyGitDir::create`] tries before it gives up.
const EMPTY_GIT_DIR_ATTEMPTS: u32 = 16;

/// The smallest directory git accepts as a repository: `HEAD`, `objects/` and `refs/`.
const EMPTY_GIT_HEAD: &str = "ref: refs/heads/main\n";

/// An empty repository directory that lives as long as the value; nothing is ever written to
/// it but its skeleton, and it is removed on drop (best effort).
struct EmptyGitDir {
    path: PathBuf,
}

impl EmptyGitDir {
    /// Creates a fresh directory (never an existing one, never through a symlink: `create_dir`
    /// refuses any name that exists) under a random name, and its skeleton.
    ///
    /// # Errors
    ///
    /// [`ReviewError::Io`] when no fresh name can be created or the skeleton cannot be written.
    fn create() -> Result<Self, ReviewError> {
        let io = |e: std::io::Error| {
            ReviewError::Io(format!("cannot make an empty git repository: {e}"))
        };
        let mut last = None;
        for _ in 0..EMPTY_GIT_DIR_ATTEMPTS {
            // MAGI FIX (S6 loop 5): a sequence number was predictable, so another local user
            // could take the next names ahead of time; a random suffix leaves nothing to guess.
            let path = std::env::temp_dir().join(format!(
                "{EMPTY_GIT_DIR_PREFIX}-{}-{:016x}",
                std::process::id(),
                rand::random::<u64>()
            ));
            match std::fs::create_dir(&path) {
                Ok(()) => {
                    let dir = Self { path };
                    std::fs::create_dir(dir.path.join("objects")).map_err(io)?;
                    std::fs::create_dir(dir.path.join("refs")).map_err(io)?;
                    std::fs::write(dir.path.join("HEAD"), EMPTY_GIT_HEAD).map_err(io)?;
                    return Ok(dir);
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => last = Some(e),
                Err(e) => return Err(io(e)),
            }
        }
        Err(io(last.unwrap_or_else(|| {
            std::io::Error::other("no attempt was made")
        })))
    }

    /// The directory as text, for `--git-dir`.
    ///
    /// # Errors
    ///
    /// [`ReviewError::Io`] when the temporary directory's path is not UTF-8.
    fn path_str(&self) -> Result<&str, ReviewError> {
        self.path.to_str().ok_or_else(|| {
            ReviewError::Io(format!(
                "the temporary directory {} is not UTF-8",
                self.path.display()
            ))
        })
    }
}

impl Drop for EmptyGitDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
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
            program: std::sync::OnceLock::new(),
            #[cfg(test)]
            test_env: Vec::new(),
        }
    }

    /// A runner whose git children also get `env` (an operator-scope config, in a test).
    #[cfg(test)]
    fn with_test_env(workdir: PathBuf, env: Vec<(String, String)>) -> Self {
        Self {
            test_env: env,
            ..Self::new(workdir)
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

    /// [`ScrubbedGit::execute_capped`] under the runner's own stdout cap.
    fn execute(&self, args: &[&str]) -> Result<Finished, ReviewError> {
        self.execute_capped(args, self.output_cap)
    }

    /// Runs git with `args`, draining stdout (stopped past `cap` bytes) and stderr (kept up to
    /// [`MAX_GIT_STDERR_BYTES`], read to the end) in parallel, within [`GIT_CALL_TIMEOUT_SECS`].
    fn execute_capped(&self, args: &[&str], cap: u64) -> Result<Finished, ReviewError> {
        let command = args.first().copied().unwrap_or_default();
        let deadline = Instant::now() + Duration::from_secs(GIT_CALL_TIMEOUT_SECS);
        let program = self
            .program
            .get_or_init(|| {
                resolve_git(&std::env::var_os("PATH").unwrap_or_default(), &self.workdir)
            })
            .clone()
            .ok_or_else(|| {
                ReviewError::Git(
                    "cannot find git in an absolute PATH entry outside the workspace and its repository"
                        .to_string(),
                )
            })?;
        let mut command_line = Command::new(program);
        command_line
            .args(self.argv(args))
            .env_clear()
            .envs(tool_child_env())
            .env("GIT_NO_LAZY_FETCH", "1")
            .env("GIT_LITERAL_PATHSPECS", "1");
        #[cfg(test)]
        command_line.envs(self.test_env.iter().cloned());
        let mut child = command_line
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| ReviewError::Git(format!("cannot run git: {e}")))?;
        let (Some(stdout), Some(stderr)) = (child.stdout.take(), child.stderr.take()) else {
            kill(&mut child);
            return Err(ReviewError::Git("cannot read git's output".to_string()));
        };
        // MAGI FIX (S6 loop 10): A1-c4. The two readers are not joined on an early return: they
        // end when every writer of their pipe is gone, and `kill` ends git, the only writer. A
        // descendant that could keep a pipe open needs a code path git runs (hook, filter,
        // fsmonitor, pager, external diff, transport, or an alias, which cannot name the
        // built-ins this runner calls), and each is off in every call.
        let (err_tx, err_rx) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = err_tx.send(drain_bounded(stderr, MAX_GIT_STDERR_BYTES));
        });
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(read_bounded(stdout, cap));
        });
        let stdout = match rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
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
        let status = wait_within(&mut child, deadline)?;
        // MAGI FIX (S6 loop 8): the stderr drain was joined without a bound; a descendant that
        // inherited the pipe (hooks, filters and protocols are off, so unlikely) could hold it
        // open past the deadline. Its text is waited for until the deadline, or a short grace.
        let drain_for = deadline
            .saturating_duration_since(Instant::now())
            .max(Duration::from_millis(STDERR_DRAIN_GRACE_MS));
        let stderr = err_rx.recv_timeout(drain_for).unwrap_or_default();
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
        self.run_bytes_capped(args, self.output_cap)
    }

    /// Stops git once its stdout passes `cap` (or the runner's own cap, when lower), so a
    /// refused output is never held whole.
    ///
    /// # Errors
    ///
    /// As [`GitCommand::run_bytes`], with the lower cap as the limit of
    /// [`ReviewError::GitOutputTooLarge`].
    // MAGI FIX (S6 loop 4): a blob over the per-file cap was read whole under the 64 MiB cap.
    fn run_bytes_capped(&self, args: &[&str], cap: u64) -> Result<Vec<u8>, ReviewError> {
        self.ensure_filters()?;
        let finished = self.execute_capped(args, cap.min(self.output_cap))?;
        if finished.code == Some(0) {
            Ok(finished.stdout)
        } else {
            Err(failure(
                args.first().copied().unwrap_or_default(),
                &finished,
            ))
        }
    }

    /// Runs `args` with `--git-dir` set to a fresh empty repository under the system temporary
    /// directory (removed afterwards) and `--work-tree=.`, the workdir.
    ///
    /// # Errors
    ///
    /// [`ReviewError::Io`] when the empty repository cannot be made or its path is not UTF-8, and
    /// those of [`GitCommand::run_bytes`].
    fn run_bytes_without_repository(&self, args: &[&str]) -> Result<Vec<u8>, ReviewError> {
        let empty = EmptyGitDir::create()?;
        let git_dir = format!("--git-dir={}", empty.path_str()?);
        let mut full = vec![git_dir.as_str(), "--work-tree=."];
        full.extend_from_slice(args);
        self.run_bytes(&full)
    }

    /// # Errors
    ///
    /// As [`GitCommand::run_bytes`] for any exit but 0, or 1 with an empty stderr (`None`), and
    /// [`ReviewError::Git`] for an output that is not UTF-8.
    fn query(&self, args: &[&str]) -> Result<Option<String>, ReviewError> {
        self.ensure_filters()?;
        let finished = self.execute(args)?;
        let command = args.first().copied().unwrap_or_default();
        match finished.code {
            Some(0) => String::from_utf8(finished.stdout).map(Some).map_err(|_| {
                ReviewError::Git(format!("git {command} printed output that is not UTF-8"))
            }),
            Some(QUERY_NO_MATCH_EXIT) if finished.stderr.is_empty() => Ok(None),
            _ => Err(failure(command, &finished)),
        }
    }
}

/// What the filter probe means: the overrides to apply, or a fail-closed error.
///
/// Exit 0 is parsed by [`filter_overrides`]; exit 1 with nothing on stderr is "no filter entry
/// anywhere" (decided by the status, as in `query`, never by the text); any other exit
/// (or a signal) is an error, so no command runs with a repository filter left active.
///
/// # Errors
///
/// [`ReviewError::Git`] for a failing probe, and whatever [`filter_overrides`] refuses.
fn interpret_filter_probe(probe: &Finished) -> Result<Vec<String>, ReviewError> {
    match probe.code {
        Some(0) => filter_overrides(&probe.stdout),
        Some(CONFIG_NO_MATCH_EXIT) if probe.stderr.is_empty() => Ok(Vec::new()),
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

/// Waits for `child` to exit, killing it at `deadline`.
///
/// # Errors
///
/// [`ReviewError::Git`] when waiting fails or the deadline passes.
fn wait_within(child: &mut Child, deadline: Instant) -> Result<ExitStatus, ReviewError> {
    let mut pause = WAIT_POLL_FIRST;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status),
            Ok(None) => {}
            Err(e) => {
                kill(child);
                return Err(ReviewError::Git(format!("cannot wait for git: {e}")));
            }
        }
        let now = Instant::now();
        if now >= deadline {
            kill(child);
            return Err(ReviewError::Git(format!(
                "git did not exit in {GIT_CALL_TIMEOUT_SECS} s"
            )));
        }
        std::thread::sleep(pause.min(deadline - now));
        pause = (pause * 2).min(WAIT_POLL_MAX);
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
/// The canonical path inside the workspace (a missing tail is appended lexically). A link whose
/// target is gone cannot be canonicalized, so it comes back as itself under its guarded parent,
/// for the input to call it missing: nothing is ever read through it, whatever its target.
///
/// # Errors
///
/// [`ReviewError::OutsideWorkspace`] for any path the guard refuses.
// MAGI FIX (S6 loop 9): a dangling link was refused as outside the workspace (A1-c5).
pub(crate) fn review_path_guard(guard: &PathGuard, path: &Path) -> Result<PathBuf, ReviewError> {
    let outside = || ReviewError::OutsideWorkspace(path.display().to_string());
    guard.validate(path).or_else(|_| {
        let dangling = std::fs::symlink_metadata(path).is_ok_and(|m| m.file_type().is_symlink())
            && std::fs::metadata(path).is_err_and(|e| e.kind() == std::io::ErrorKind::NotFound);
        match (dangling, path.parent(), path.file_name()) {
            (true, Some(parent), Some(name)) => guard
                .validate(parent)
                .map(|p| p.join(name))
                .map_err(|_| outside()),
            _ => Err(outside()),
        }
    })
}

/// The `-c` values that neutralize every filter driver git's config defines, in any scope
/// (`system`, `global`, `local`, `worktree`, an included file), sorted by driver.
///
/// Reads the output of `git config --null --show-scope --get-regexp ^filter\.`: records of
/// `<scope> NUL <key>` followed by `LF <value> NUL`, or by `NUL` alone for a key without value.
/// The key ends at the first `LF` (a value may hold more). This shape is git's output, not a
/// documented format: `--show-scope` needs git 2.26 or later, the parser was measured against
/// git 2.56 (the tests pin each shape), and a record it cannot parse fails closed. No filter
/// runs during a review, the operator's included: a filter receives the worktree bytes before the
/// review's secret scan (REQ-RV22), so an LFS file is read as its worktree bytes, not its pointer.
/// O(n) in the input.
///
/// # Errors
///
/// [`ReviewError::Git`] for a truncated record, a key outside `filter.` or with an empty driver
/// name or variable (a two-level `filter.<key>` names no driver and is skipped), a
/// driver name with `=` or one that is not UTF-8 (which `-c` cannot express exactly), or
/// overrides over
/// [`MAX_FILTER_OVERRIDE_BYTES`].
pub(crate) fn filter_overrides(config_z: &[u8]) -> Result<Vec<String>, ReviewError> {
    let unreadable = |what: String| {
        ReviewError::Git(format!(
            "cannot read the repository's filter configuration: {what}"
        ))
    };
    let mut names = BTreeSet::new();
    let mut override_bytes = 0usize;
    let mut rest = config_z;
    while !rest.is_empty() {
        let (_scope, after) =
            split_at_nul(rest).ok_or_else(|| unreadable("a truncated record".into()))?;
        let (entry, after) =
            split_at_nul(after).ok_or_else(|| unreadable("a truncated record".into()))?;
        rest = after;
        let key = entry.split(|b| *b == b'\n').next().unwrap_or_default();
        // MAGI FIX (S6 loop 5): a lossy decode turned a non-UTF-8 driver name into U+FFFD, so the
        // override neutralized another driver and the repository's own ran; `-c` cannot carry
        // arbitrary bytes on every platform, so such a name fails closed before any command.
        let key = std::str::from_utf8(key).map_err(|_| {
            unreadable(format!(
                "a filter driver name that is not UTF-8: {:?}",
                String::from_utf8_lossy(key)
            ))
        })?;
        let unexpected = || unreadable(format!("unexpected key {key:?}"));
        let tail = key.strip_prefix(FILTER_KEY_PREFIX).ok_or_else(unexpected)?;
        // MAGI FIX (S6 loop 7): git reads a driver only from `filter.<driver>.<variable>`; a
        // two-level `filter.<key>` has no subsection, configures nothing, and failed the review.
        let Some((name, variable)) = tail.rsplit_once('.') else {
            continue;
        };
        if name.is_empty() || variable.is_empty() {
            return Err(unexpected());
        }
        // MAGI FIX (S6 loop 3): every scope, the operator's too: a filter gets the worktree bytes
        // before the review scans them (RV22), wherever it is defined.
        if name.contains('=') {
            return Err(unreadable(format!(
                "a filter driver name with '=': {name:?}"
            )));
        }
        // MAGI FIX (S6 loop 6): the cap is checked as each new driver arrives, so neither the
        // overrides nor the error grow past it on a config with millions of drivers.
        if names.insert(name.to_string()) {
            override_bytes += NEUTRALIZED_FILTER_SETTINGS
                .iter()
                .map(|s| FILTER_KEY_PREFIX.len() + name.len() + 1 + s.len())
                .sum::<usize>();
            if override_bytes > MAX_FILTER_OVERRIDE_BYTES {
                let shown: Vec<&str> = names
                    .iter()
                    .take(FILTER_DRIVERS_NAMED_IN_ERROR)
                    .map(String::as_str)
                    .collect();
                return Err(ReviewError::Git(format!(
                    "git's config defines more filter drivers than a review can disable (over \
                     {MAX_FILTER_OVERRIDE_BYTES} bytes of overrides after {} drivers, among them: {})",
                    names.len(),
                    shown.join(", ")
                )));
            }
        }
    }
    Ok(names
        .iter()
        .flat_map(|name| {
            NEUTRALIZED_FILTER_SETTINGS
                .iter()
                .map(move |s| format!("{FILTER_KEY_PREFIX}{name}.{s}"))
        })
        .collect())
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
    fn names_taken_ahead_in_the_temp_directory_never_block_the_empty_repository() {
        // [MAGI S6 loop 5, A1-c1] another local user could take `<prefix>-<pid>-<n>` for the next
        // attempts; a random name leaves nothing to predict. nextest runs each test in its own
        // process, so this process's sequence starts at 0.
        let taken: Vec<_> = (0..u64::from(EMPTY_GIT_DIR_ATTEMPTS) * 4)
            .map(|n| {
                std::env::temp_dir()
                    .join(format!("{EMPTY_GIT_DIR_PREFIX}-{}-{n}", std::process::id()))
            })
            .filter(|p| std::fs::create_dir(p).is_ok())
            .collect();
        let made = EmptyGitDir::create();
        for p in &taken {
            let _ = std::fs::remove_dir(p);
        }
        assert!(made.is_ok(), "{:?}", made.err());
    }

    #[test]
    fn an_ignored_file_with_glob_characters_in_its_name_keeps_its_ignore_verdict() {
        // [MAGI S6 loop 5, A2-m1, false positive] every call carries GIT_LITERAL_PATHSPECS=1, and
        // git also matches a pathspec against the exact name before trying it as a glob (measured:
        // the verdict holds with the variable removed), so `x[0].rs` keeps its verdict.
        let d = repo();
        std::fs::write(d.path().join(".gitignore"), "x\\[0\\].rs\n").expect("write");
        std::fs::write(d.path().join("x[0].rs"), "fn x() {}\n").expect("write");
        std::fs::write(d.path().join("x0.rs"), "fn y() {}\n").expect("write");
        let g = ScrubbedGit::new(d.path().to_path_buf());
        let ignored = magi_rs::review::filter::ignored_paths(
            &ReviewInput::File("x[0].rs".into()),
            "x[0].rs",
            d.path(),
            &g,
        )
        .expect("query runs");
        assert!(ignored.contains("x[0].rs"), "{ignored:?}");
    }

    #[test]
    fn a_tracked_ignored_file_named_with_glob_characters_keeps_its_ignore_verdict() {
        // [MAGI S6 loop 6, A2-c1, false positive] Caspar's case: a TRACKED file a rule ignores,
        // named explicitly, with `[` in its name, next to a sibling the glob would match.
        // Measured with git 2.56: the name is listed with and without GIT_LITERAL_PATHSPECS
        // (git tries the exact name before the glob); the variable makes the match exact.
        let d = repo();
        std::fs::write(d.path().join("a[1].rs"), "fn a() {}\n").expect("write");
        std::fs::write(d.path().join("a1.rs"), "fn b() {}\n").expect("write");
        git(d.path(), &["add", "--", "a[1].rs", "a1.rs"]);
        git(d.path(), &["commit", "-q", "-m", "1"]);
        std::fs::write(d.path().join(".gitignore"), "a*\n").expect("write");
        let g = ScrubbedGit::new(d.path().to_path_buf());
        let ignored = magi_rs::review::filter::ignored_paths(
            &ReviewInput::File("a[1].rs".into()),
            "a[1].rs",
            d.path(),
            &g,
        )
        .expect("query runs");
        assert_eq!(ignored, ["a[1].rs".to_string()].into_iter().collect());
    }

    #[test]
    fn a_replace_ref_never_changes_the_content_a_revision_review_reads() {
        // [MAGI S6 loop 5, A1-b2] `refs/replace/*` is repository data: honoring it would review
        // a substitute blob instead of what the pinned commit holds (RV21).
        let d = repo();
        std::fs::write(d.path().join("a.rs"), "real\n").expect("write");
        git(d.path(), &["add", "a.rs"]);
        git(d.path(), &["commit", "-q", "-m", "x"]);
        std::fs::write(d.path().join("fake.txt"), "fake\n").expect("write");
        let blob = |p: &str| {
            let out = std::process::Command::new("git")
                .arg("-C")
                .arg(d.path())
                .args(["hash-object", "-w", p])
                .output()
                .expect("git runs");
            String::from_utf8(out.stdout)
                .expect("utf-8")
                .trim()
                .to_string()
        };
        let (real, fake) = (blob("a.rs"), blob("fake.txt"));
        git(d.path(), &["replace", &real, &fake]);
        let g = ScrubbedGit::new(d.path().to_path_buf());
        let src = RevisionSource::new(&g, "HEAD".into(), None);
        assert_eq!(src.read("a.rs").expect("read"), Some(b"real\n".to_vec()));
    }

    #[test]
    fn a_graft_file_never_changes_the_reviewed_parent_or_merge_base() {
        // [MAGI S6 loop 36, A1, security] `$GIT_DIR/info/grafts` is repository data that
        // `core.useReplaceRefs=false` does not neutralize: a graft `C A` made git read A as C's
        // parent, so a review of C diffed A..C, and a graft to a commit with C's tree emptied it.
        let d = repo();
        let mut shas = Vec::new();
        for text in ["a\n", "b\n", "c\n"] {
            std::fs::write(d.path().join("a.rs"), text).expect("write");
            git(d.path(), &["add", "a.rs"]);
            git(d.path(), &["commit", "-q", "-m", text.trim()]);
            let out = std::process::Command::new("git")
                .arg("-C")
                .arg(d.path())
                .args(["rev-parse", "HEAD"])
                .output()
                .expect("git runs");
            shas.push(
                String::from_utf8(out.stdout)
                    .expect("utf-8")
                    .trim()
                    .to_string(),
            );
        }
        let (a, b, c) = (&shas[0], &shas[1], &shas[2]);
        std::fs::write(
            d.path().join(".git").join("info").join("grafts"),
            format!("{c} {a}\n"),
        )
        .expect("graft");
        let g = ScrubbedGit::new(d.path().to_path_buf());
        let allow = |p: &std::path::Path| -> Result<std::path::PathBuf, ReviewError> {
            Ok(p.to_path_buf())
        };
        let diff_of = |r: &magi_rs::review::input::ResolvedInput| {
            r.files
                .iter()
                .find(|f| f.path == "a.rs")
                .and_then(|f| f.diff.clone())
                .unwrap_or_default()
        };
        let commit = magi_rs::review::input::resolve_input(
            &ReviewInput::Commit(Some(c.clone())),
            d.path(),
            &g,
            &allow,
        )
        .expect("resolves");
        let parent = match &commit.source_kind {
            magi_rs::review::input::SourceKind::Revision { old_rev, .. } => old_rev.clone(),
            _ => None,
        };
        assert_eq!(
            parent.as_deref(),
            Some(b.as_str()),
            "the graft moved C's parent"
        );
        assert!(diff_of(&commit).contains("-b"), "{}", diff_of(&commit));
        let branches = magi_rs::review::input::resolve_input(
            &ReviewInput::Branches {
                base: b.clone(),
                head: c.clone(),
            },
            d.path(),
            &g,
            &allow,
        )
        .expect("resolves");
        assert!(diff_of(&branches).contains("-b"), "{}", diff_of(&branches));
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
    fn a_revision_diff_reads_attributes_from_the_reviewed_revision() {
        // [MAGI S6 loop 7, A1-b1, measured with git 2.56] a revision diff read `.gitattributes`
        // from the working tree, so a later commit marking `*.rs -diff` turned the reviewed
        // change into "Binary files differ" and the file was left out as binary.
        let d = repo();
        std::fs::write(d.path().join("a.rs"), "one\n").expect("write");
        git(d.path(), &["add", "a.rs"]);
        git(d.path(), &["commit", "-q", "-m", "1"]);
        std::fs::write(d.path().join("a.rs"), "two\n").expect("write");
        git(d.path(), &["commit", "-q", "-am", "2"]);
        std::fs::write(d.path().join(".gitattributes"), "*.rs -diff\n").expect("write");
        git(d.path(), &["add", ".gitattributes"]);
        git(d.path(), &["commit", "-q", "-m", "3"]);
        let g = ScrubbedGit::new(d.path().to_path_buf());
        let allow = |p: &std::path::Path| -> Result<std::path::PathBuf, ReviewError> {
            Ok(p.to_path_buf())
        };
        for (input, added) in [
            (ReviewInput::Commit(Some("HEAD~1".into())), "+two"),
            (
                ReviewInput::Range {
                    from: "HEAD~2".into(),
                    to: "HEAD~1".into(),
                },
                "+two",
            ),
            (ReviewInput::Commit(Some("HEAD~2".into())), "+one"),
        ] {
            let r = magi_rs::review::input::resolve_input(&input, d.path(), &g, &allow)
                .expect("resolves");
            let diff = r
                .files
                .iter()
                .find(|f| f.path == "a.rs")
                .and_then(|f| f.diff.clone())
                .expect("a.rs has a diff");
            assert!(diff.contains(added), "{diff}");
        }
    }

    #[test]
    fn a_merge_commit_is_reviewed_against_its_first_parent() {
        // [MAGI §6 MS2 A loop 2] `diff-tree <merge>` prints no diff, so a HEAD reached through
        // `git merge` used to be "nothing to review"; what the merge brought in is reviewed.
        let d = repo();
        std::fs::write(d.path().join("a.rs"), "fn a() {}\n").expect("write");
        git(d.path(), &["add", "a.rs"]);
        git(d.path(), &["commit", "-q", "-m", "root"]);
        git(d.path(), &["checkout", "-q", "-b", "topic"]);
        std::fs::write(d.path().join("b.rs"), "fn b() {}\n").expect("write");
        git(d.path(), &["add", "b.rs"]);
        git(d.path(), &["commit", "-q", "-m", "topic"]);
        git(d.path(), &["checkout", "-q", "main"]);
        std::fs::write(d.path().join("c.rs"), "fn c() {}\n").expect("write");
        git(d.path(), &["add", "c.rs"]);
        git(d.path(), &["commit", "-q", "-m", "main"]);
        git(
            d.path(),
            &["merge", "-q", "--no-ff", "-m", "merge", "topic"],
        );
        let first_parent = String::from_utf8(
            std::process::Command::new("git")
                .arg("-C")
                .arg(d.path())
                .args(["rev-parse", "HEAD^1"])
                .output()
                .expect("git runs")
                .stdout,
        )
        .expect("utf-8");
        let g = ScrubbedGit::new(d.path().to_path_buf());
        let allow = |p: &std::path::Path| -> Result<std::path::PathBuf, ReviewError> {
            Ok(p.to_path_buf())
        };
        for input in [
            ReviewInput::Commit(None),
            ReviewInput::Commit(Some("HEAD".into())),
        ] {
            let r = magi_rs::review::input::resolve_input(&input, d.path(), &g, &allow)
                .expect("a merge commit is reviewable");
            assert_eq!(
                r.files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(),
                vec!["b.rs"],
                "{input:?}"
            );
            assert!(
                matches!(&r.source_kind, magi_rs::review::input::SourceKind::Revision { old_rev, .. }
                    if old_rev.as_deref() == Some(first_parent.trim())),
                "{input:?}: old side is not the first parent"
            );
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
    fn a_git_that_keeps_running_after_its_output_is_killed_at_the_deadline() {
        // [MAGI §6 MS2 A loop 1] the call's time limit covered only stdout: a git that closed its output and then
        // hung (on a lock, say) blocked the preparation forever. `hash-object --stdin` waits on a stdin kept open.
        let mut child = std::process::Command::new("git")
            .args(["hash-object", "--stdin"])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("git runs");
        let stdin = child.stdin.take();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let waited = wait_within(&mut child, Instant::now() + Duration::from_millis(200));
            let _ = tx.send(waited.is_err());
        });
        let outcome = rx.recv_timeout(Duration::from_secs(30));
        drop(stdin);
        match outcome {
            Ok(true) => {}
            Ok(false) => panic!("a child that never exits was reported as finished"),
            Err(_) => panic!("the wait did not end at its deadline"),
        }
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
    fn a_revision_file_in_a_nested_directory_is_read() {
        // [MAGI S6 loop 10, A2-c1, false positive] `holds` runs `ls-tree` without `-r`. Measured
        // with git 2.56: a pathspec makes `ls-tree` descend the trees it names and list the blob
        // itself (`100644 blob <id>\tsrc/a/b.rs`), from the root and from a sub-workspace alike.
        let d = repo();
        std::fs::create_dir_all(d.path().join("src/a")).expect("mkdir");
        std::fs::write(d.path().join("src/a/b.rs"), "fn b() {}\n").expect("write");
        git(d.path(), &["add", "src/a/b.rs"]);
        git(d.path(), &["commit", "-q", "-m", "1"]);
        let g = ScrubbedGit::new(d.path().to_path_buf());
        let src = RevisionSource::new(&g, "HEAD".into(), None);
        assert_eq!(
            src.read("src/a/b.rs").expect("read"),
            Some(b"fn b() {}\n".to_vec())
        );
        assert_eq!(src.read("src/a").expect("read"), None, "a tree is no file");
        let sub = ScrubbedGit::new(d.path().join("src"));
        let src = RevisionSource::new(&sub, "HEAD".into(), None);
        assert_eq!(
            src.read("a/b.rs").expect("read"),
            Some(b"fn b() {}\n".to_vec())
        );
    }

    #[test]
    fn a_gitlink_replaced_by_a_file_or_the_reverse_keeps_the_regular_side() {
        // [MAGI S6 loop 10, A1-m1/A1-b1, false positive] measured with git 2.56: a type change
        // between a gitlink and a regular file is printed as two sections, a deletion and an
        // addition, never as one `old mode 160000`/`new mode 100644` section; only the gitlink's
        // own section is dropped, and the regular file's section reaches the review.
        let d = repo();
        std::fs::write(d.path().join("r.rs"), "fn r() {}\n").expect("write");
        git(d.path(), &["add", "r.rs"]);
        git(d.path(), &["commit", "-q", "-m", "0"]);
        let commit_id = "1111111111111111111111111111111111111111";
        git(
            d.path(),
            &[
                "update-index",
                "--add",
                "--cacheinfo",
                &format!("160000,{commit_id},sub"),
            ],
        );
        git(d.path(), &["commit", "-q", "-m", "gitlink"]);
        git(d.path(), &["tag", "G"]);
        git(d.path(), &["rm", "-q", "--cached", "sub"]);
        std::fs::write(d.path().join("sub"), "fn s() {}\n").expect("write");
        git(d.path(), &["add", "sub"]);
        git(d.path(), &["commit", "-q", "-m", "file"]);
        git(d.path(), &["tag", "F"]);
        let g = ScrubbedGit::new(d.path().to_path_buf());
        let allow = |p: &std::path::Path| -> Result<std::path::PathBuf, ReviewError> {
            Ok(p.to_path_buf())
        };
        for (from, to, line) in [("G", "F", "+fn s() {}"), ("F", "G", "-fn s() {}")] {
            let input = ReviewInput::Range {
                from: from.into(),
                to: to.into(),
            };
            let r = magi_rs::review::input::resolve_input(&input, d.path(), &g, &allow)
                .expect("resolves");
            let diffs: Vec<&str> = r
                .files
                .iter()
                .filter(|f| f.path == "sub")
                .filter_map(|f| f.diff.as_deref())
                .collect();
            assert_eq!(diffs.len(), 1, "{from}..{to}: {diffs:?}");
            let diff = diffs.first().copied().unwrap_or_default();
            assert!(diff.contains(line), "{from}..{to}: {diff}");
            assert!(!diff.contains("Subproject commit"), "{from}..{to}: {diff}");
        }
    }

    #[test]
    fn a_rename_from_outside_a_sub_workspace_is_reviewed_as_an_addition() {
        // [MAGI S6 loop 10, A1-m2] measured with git 2.56: `diff --relative` leaves out the
        // paths outside the workspace before rename detection, so a file moved in from outside
        // is an addition with no `rename from`, and no old side is read outside the workspace.
        let d = repo();
        std::fs::create_dir_all(d.path().join("ws")).expect("mkdir");
        std::fs::write(
            d.path().join("outside.rs"),
            "fn o() {}\nfn p() {}\nfn q() {}\n",
        )
        .expect("write");
        std::fs::write(d.path().join("ws/k.rs"), "fn k() {}\n").expect("write");
        git(d.path(), &["add", "."]);
        git(d.path(), &["commit", "-q", "-m", "0"]);
        git(d.path(), &["mv", "outside.rs", "ws/a.rs"]);
        git(d.path(), &["commit", "-q", "-m", "1"]);
        let ws = d.path().join("ws");
        let allow = |p: &std::path::Path| -> Result<std::path::PathBuf, ReviewError> {
            Ok(p.to_path_buf())
        };
        let r = magi_rs::review::input::resolve_input(
            &ReviewInput::Commit(None),
            &ws,
            &ScrubbedGit::new(ws.clone()),
            &allow,
        )
        .expect("resolves");
        assert_eq!(
            r.files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(),
            vec!["a.rs"]
        );
        let diff = r
            .files
            .first()
            .and_then(|f| f.diff.as_deref())
            .unwrap_or_default();
        assert!(diff.contains("new file mode"), "{diff}");
        assert!(!diff.contains("rename from"), "{diff}");
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
    fn a_repo_alias_never_shadows_a_git_subcommand_the_review_runs() {
        // [MAGI S6 loop 10, A1-c1, false positive] a `!` alias in the reviewed repo's config
        // named like each subcommand the review calls would create a sentinel file if git ran
        // it. Measured with git 2.56: git ignores an alias whose name is a built-in command.
        // The control alias `zz` (no built-in) proves the sentinel fires through this runner.
        let d = repo();
        std::fs::write(d.path().join("a.rs"), "one\n").expect("write");
        git(d.path(), &["add", "a.rs"]);
        git(d.path(), &["commit", "-q", "-m", "1"]);
        let calls: &[&[&str]] = &[
            &["config", "--get-regexp", "^core\\."],
            &["rev-parse", "--verify", "HEAD^{commit}"],
            &["cat-file", "commit", "HEAD"],
            &["merge-base", "HEAD", "HEAD"],
            &["diff", "--no-color", "HEAD"],
            &["diff-tree", "-p", "-r", "--root", "HEAD"],
            &["ls-files", "-z"],
            &["ls-tree", "-z", "HEAD", "--", "a.rs"],
            &["show", "HEAD:./a.rs"],
        ];
        for call in calls {
            let name = call.first().copied().expect("subcommand");
            git(
                d.path(),
                &[
                    "config",
                    &format!("alias.{name}"),
                    &format!("!touch PWNED-{name}"),
                ],
            );
        }
        git(d.path(), &["config", "alias.zz", "!touch PWNED-zz"]);
        let g = ScrubbedGit::new(d.path().to_path_buf());
        for call in calls {
            g.run_bytes(call).expect("the built-in runs");
            let name = call.first().copied().expect("subcommand");
            assert!(
                !d.path().join(format!("PWNED-{name}")).exists(),
                "the alias `{name}` ran"
            );
        }
        let _ = g.run_bytes(&["zz"]);
        assert!(
            d.path().join("PWNED-zz").exists(),
            "the control alias did not run: the sentinel proves nothing"
        );
    }

    #[test]
    fn a_shallow_commit_is_never_reviewed_as_a_root_commit() {
        // [MAGI S6 loop 5, A1-m2] the parent of a shallow clone's tip is not in the repository:
        // the review fails typed instead of sending the whole tree as an addition.
        let origin = repo();
        std::fs::write(origin.path().join("a.rs"), "one\n").expect("write");
        git(origin.path(), &["add", "a.rs"]);
        git(origin.path(), &["commit", "-q", "-m", "1"]);
        std::fs::write(origin.path().join("a.rs"), "two\n").expect("write");
        git(origin.path(), &["commit", "-q", "-am", "2"]);
        let clone = tempfile::tempdir().expect("tempdir");
        let url = format!(
            "file://{}",
            origin.path().to_str().expect("utf-8").replace('\\', "/")
        );
        git(
            clone.path(),
            &["clone", "-q", "--depth", "1", &url, "shallow"],
        );
        let shallow = clone.path().join("shallow");
        let g = ScrubbedGit::new(shallow.clone());
        let allow = |p: &std::path::Path| -> Result<std::path::PathBuf, ReviewError> {
            Ok(p.to_path_buf())
        };
        let r =
            magi_rs::review::input::resolve_input(&ReviewInput::Commit(None), &shallow, &g, &allow);
        assert!(
            matches!(r, Err(ReviewError::Git(_))),
            "{:?}",
            r.map(|r| r.files.len())
        );
    }

    #[test]
    fn a_filter_driver_with_a_non_utf8_name_never_runs() {
        // [MAGI S6 loop 5, A1-m1] a subsection name is bytes: a lossy decode would neutralize
        // `filter.\u{fffd}` and leave the attacker's `filter.\xff` active, so its `clean` would
        // run on the worktree bytes before the secret scan. The runner must fail closed instead.
        let d = repo();
        std::fs::write(d.path().join("a.rs"), "one\n").expect("write");
        git(d.path(), &["add", "a.rs"]);
        git(d.path(), &["commit", "-q", "-m", "1"]);
        std::fs::write(d.path().join(".gitattributes"), b"* filter=\xff\n").expect("write");
        let config = d.path().join(".git").join("config");
        let mut bytes = std::fs::read(&config).expect("read config");
        bytes.extend_from_slice(b"[filter \"\xff\"]\n\tclean = touch PWNED; cat\n");
        std::fs::write(&config, bytes).expect("write config");
        std::fs::write(d.path().join("a.rs"), "two\n").expect("write");
        let g = ScrubbedGit::new(d.path().to_path_buf());
        let diff = g.run(&[
            "diff",
            "--no-color",
            "--no-ext-diff",
            "--no-textconv",
            "HEAD",
        ]);
        assert!(
            !d.path().join("PWNED").exists(),
            "a non-UTF-8 filter driver ran: {diff:?}"
        );
        assert!(diff.is_err(), "the runner must fail closed: {diff:?}");
    }

    #[test]
    fn filter_overrides_refuses_a_driver_name_that_is_not_utf8() {
        // [MAGI S6 loop 5, A1-m1] `-c` cannot carry the name's bytes on every platform, so a
        // name that is not UTF-8 fails closed instead of neutralizing a lossy replacement.
        let err = filter_overrides(b"local\0filter.\xff.clean\ntouch PWNED\0")
            .expect_err("a non-UTF-8 driver name fails closed");
        assert!(matches!(err, ReviewError::Git(_)), "{err:?}");
    }

    #[test]
    fn a_filter_driver_from_the_operators_global_config_never_runs() {
        // [MAGI S6 loop 3, S3] RV22: a filter receives the worktree bytes before the review scans
        // them, so a driver from ANY scope git reads (the operator's global file here) must not run.
        let d = repo();
        std::fs::write(d.path().join("a.rs"), "one\n").expect("write");
        git(d.path(), &["add", "a.rs"]);
        git(d.path(), &["commit", "-q", "-m", "1"]);
        std::fs::write(d.path().join(".gitattributes"), "* filter=evil\n").expect("write");
        std::fs::write(d.path().join("a.rs"), "two\n").expect("write");
        let home = tempfile::tempdir().expect("tempdir");
        let global = home.path().join("gitconfig");
        std::fs::write(&global, "[filter \"evil\"]\n\tclean = touch PWNED; cat\n").expect("write");
        let env = vec![(
            "GIT_CONFIG_GLOBAL".to_string(),
            global.to_str().expect("utf-8 tempdir").to_string(),
        )];
        let g = ScrubbedGit::with_test_env(d.path().to_path_buf(), env);
        let head = g
            .run(&["rev-parse", "--verify", "HEAD^{commit}"])
            .expect("head");
        let diff = g
            .run(&[
                "diff",
                "--no-color",
                "--no-ext-diff",
                "--no-textconv",
                head.trim(),
            ])
            .expect("diff");
        assert!(diff.contains("+two"), "{diff}");
        assert!(
            !d.path().join("PWNED").exists(),
            "the operator's global filter ran on the reviewed bytes"
        );
    }

    #[test]
    fn no_textconv_or_external_diff_driver_of_any_scope_runs_in_a_review_diff() {
        // [MAGI S6 loop 3, S3] guardian: every diff a review takes (the worktree, a commit, a root
        // commit) carries --no-textconv and --no-ext-diff, so a `diff.<driver>.textconv` or
        // `.command` from the operator's global config never sees the reviewed bytes.
        let d = repo();
        std::fs::write(d.path().join(".gitattributes"), "* diff=evil\n").expect("write");
        std::fs::write(d.path().join("a.rs"), "one\n").expect("write");
        git(d.path(), &["add", "."]);
        git(d.path(), &["commit", "-q", "-m", "1"]);
        std::fs::write(d.path().join("a.rs"), "two\n").expect("write");
        git(d.path(), &["commit", "-q", "-am", "2"]);
        std::fs::write(d.path().join("a.rs"), "three\n").expect("write");
        let home = tempfile::tempdir().expect("tempdir");
        let global = home.path().join("gitconfig");
        std::fs::write(
            &global,
            "[diff \"evil\"]\n\ttextconv = touch PWNED_TEXTCONV; cat\n\tcommand = touch PWNED_EXT\n",
        )
        .expect("write");
        let env = vec![(
            "GIT_CONFIG_GLOBAL".to_string(),
            global.to_str().expect("utf-8 tempdir").to_string(),
        )];
        let g = ScrubbedGit::with_test_env(d.path().to_path_buf(), env);
        let allow = |p: &std::path::Path| -> Result<std::path::PathBuf, ReviewError> {
            Ok(p.to_path_buf())
        };
        let results: Vec<_> = [
            ReviewInput::Uncommitted,
            ReviewInput::Commit(None),
            ReviewInput::Commit(Some("HEAD~1".into())),
        ]
        .iter()
        .map(|input| magi_rs::review::input::resolve_input(input, d.path(), &g, &allow))
        .collect();
        for sentinel in ["PWNED_TEXTCONV", "PWNED_EXT"] {
            assert!(!d.path().join(sentinel).exists(), "{sentinel} ran");
        }
        for r in results {
            assert!(!r.expect("resolves").files.is_empty());
        }
    }

    #[test]
    fn filter_drivers_from_every_scope_are_neutralized() {
        // [CP2 MS2 S1a loop 10] `config --null --show-scope --get-regexp ^filter\.`: scope NUL key LF value NUL.
        let z = b"global\0filter.lfs.clean\ngit-lfs clean -- %f\0local\0filter.evil.clean\nsh x\0\
                  worktree\0filter.a.b.process\ny\0local\0filter.evil.required\ntrue\0";
        let want: Vec<String> = ["a.b", "evil", "lfs"]
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
            b"local\0filter..clean\nx\0",
        ] {
            assert!(
                filter_overrides(bad).is_err(),
                "{:?}",
                String::from_utf8_lossy(bad)
            );
        }
    }

    #[test]
    fn a_filter_key_with_no_driver_name_is_inert_and_never_refuses_the_review() {
        // [MAGI S6 loop 7, A1-m2] git reads a driver only from `filter.<driver>.<variable>`; a
        // two-level `filter.<key>` (no subsection) configures nothing, yet it failed the whole
        // review as "unexpected key". It is skipped; a real driver next to it is still neutralized.
        assert_eq!(
            filter_overrides(b"local\0filter.foo\ntrue\0global\0filter.bar\0").expect("inert"),
            Vec::<String>::new()
        );
        let want: Vec<String> = ["clean=", "smudge=", "process=", "required=false"]
            .map(|v| format!("filter.a.{v}"))
            .to_vec();
        assert_eq!(
            filter_overrides(b"local\0filter.foo\ntrue\0local\0filter.a.clean\nx\0")
                .expect("parses"),
            want
        );
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
        assert_eq!(interpret_filter_probe(&operator).expect("parses").len(), 4);
        let truncated = probe(Some(0), b"local\0filter.evil.clean\nsh x", "");
        assert!(interpret_filter_probe(&truncated).is_err());
        match interpret_filter_probe(&probe(Some(128), b"", "fatal: bad config line 1")) {
            Err(ReviewError::Git(msg)) => assert!(msg.contains("bad config line 1"), "{msg}"),
            other => panic!("expected a Git error, got {other:?}"),
        }
        // [MAGI §6 MS2 A loop 2] exit 1 is "nothing matched" only with nothing on stderr, as in
        // `query`; anything git says there fails closed.
        assert!(matches!(
            interpret_filter_probe(&probe(Some(1), b"", "error: something went wrong")),
            Err(ReviewError::Git(_))
        ));
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

    #[cfg(unix)] // creating a symlink needs Developer Mode on Windows.
    #[test]
    fn a_dangling_link_is_missing_not_outside_the_workspace() {
        // [MAGI S6 loop 9, A1-c5] the guard canonicalizes, and a link whose target is gone
        // cannot be canonicalized, so it was refused as outside the workspace before the input
        // could call it missing. Whatever its target, nothing is read through it.
        let outer = tempfile::tempdir().expect("tempdir");
        let ws = outer.path().join("ws");
        std::fs::create_dir_all(&ws).expect("mkdir");
        std::os::unix::fs::symlink(ws.join("gone.rs"), ws.join("inside.rs")).expect("symlink");
        std::os::unix::fs::symlink(outer.path().join("gone"), ws.join("outside")).expect("symlink");
        let guard = PathGuard::new(ws.clone()).expect("guard");
        let check = |p: &std::path::Path| review_path_guard(&guard, p);
        let git = ScrubbedGit::new(ws.clone());
        for input in [
            magi_rs::review::input::ReviewInput::File("inside.rs".into()),
            magi_rs::review::input::ReviewInput::File("outside".into()),
            magi_rs::review::input::ReviewInput::Directory("outside".into()),
        ] {
            match magi_rs::review::input::resolve_input(&input, &ws, &git, &check) {
                Err(ReviewError::NotReviewable { cause, .. }) => assert_eq!(cause, "missing"),
                Err(other) => panic!("{other}"),
                Ok(_) => panic!("a dangling link resolved"),
            }
        }
        assert!(matches!(
            review_path_guard(&guard, &ws.join("outside").join("x.rs")),
            Err(ReviewError::OutsideWorkspace(_))
        ));
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
    fn the_filter_driver_cap_stops_the_parse_and_bounds_the_error() {
        // [MAGI S6 loop 6, A1-c2] every override was built and every name joined into the error
        // before the cap was checked: a config with many drivers grew both without bound.
        let mut z = Vec::new();
        for i in 0..50_000 {
            z.extend_from_slice(format!("local\0filter.driver{i}.clean\ncat\0").as_bytes());
        }
        let message = filter_overrides(&z).expect_err("over the cap").to_string();
        assert!(
            message.len() <= 2 * MAX_FILTER_OVERRIDE_BYTES,
            "{} bytes",
            message.len()
        );
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
    fn git_answers_false_only_inside_a_git_directory_where_the_review_fails_closed() {
        // [MAGI S6 loop 7, A1-c1, false positive, measured with git 2.56] outside any repository
        // `rev-parse --is-inside-work-tree` exits 128 ("not a git repository"), so the walk is
        // reached (a_directory_outside_any_repository_is_walked_through_the_real_git). git
        // prints `false` with exit 0 only inside a git directory (a bare repository, a `.git`):
        // walking there would read the repository's internals, so the review refuses.
        let d = tempfile::tempdir().expect("tempdir");
        assert!(
            d.path().ancestors().all(|a| !a.join(".git").exists()),
            "TMPDIR sits inside a git repository"
        );
        let outside = std::process::Command::new("git")
            .arg("-C")
            .arg(d.path())
            .args(["rev-parse", "--is-inside-work-tree"])
            .output()
            .expect("git runs");
        assert!(!outside.status.success(), "outside a repository git fails");
        let bare = d.path().join("b.git");
        git(d.path(), &["init", "-q", "--bare", "b.git"]);
        std::fs::write(bare.join("x.rs"), "fn x() {}\n").expect("write");
        let allow = |p: &std::path::Path| -> Result<std::path::PathBuf, ReviewError> {
            Ok(p.to_path_buf())
        };
        let r = magi_rs::review::input::resolve_input(
            &ReviewInput::Directory(".".into()),
            &bare,
            &ScrubbedGit::new(bare.clone()),
            &allow,
        );
        match r {
            Err(ReviewError::Git(msg)) => {
                assert!(msg.contains("not inside a git work tree"), "{msg}")
            }
            Err(other) => panic!("expected a fail-closed git error, got {other:?}"),
            Ok(_) => panic!("a git directory was walked"),
        }
    }

    #[test]
    fn an_untracked_ignored_file_is_never_listed_in_a_directory_or_uncommitted_review() {
        // [MAGI S6 loop 10, A2-m3] the filter's ignored set holds only tracked ignored files for
        // these inputs; the untracked half of RV22 rests on the listing, pinned here with git.
        let d = repo();
        std::fs::write(d.path().join(".gitignore"), "*.env\n").expect("write");
        std::fs::write(d.path().join("a.rs"), "fn a() {}\n").expect("write");
        git(d.path(), &["add", ".gitignore", "a.rs"]);
        git(d.path(), &["commit", "-q", "-m", "1"]);
        std::fs::write(d.path().join("a.rs"), "fn a() { 1 }\n").expect("write");
        std::fs::write(d.path().join("b.env"), "HOST=internal\n").expect("write");
        let allow = |p: &std::path::Path| -> Result<std::path::PathBuf, ReviewError> {
            Ok(p.to_path_buf())
        };
        let g = ScrubbedGit::new(d.path().to_path_buf());
        for input in [ReviewInput::Directory(".".into()), ReviewInput::Uncommitted] {
            let r =
                magi_rs::review::input::resolve_input(&input, d.path(), &g, &allow).expect("lists");
            let paths: Vec<&str> = r.files.iter().map(|f| f.path.as_str()).collect();
            assert!(paths.contains(&"a.rs"), "{input:?}: {paths:?}");
            assert!(!paths.contains(&"b.env"), "{input:?}: {paths:?}");
        }
    }

    #[test]
    fn a_directory_outside_any_repository_still_honors_its_gitignore() {
        // [MAGI §6 MS2 A loop 1] RV22: a directory respects `.gitignore`, repository or not. Outside one, git
        // itself reads the rules (root and nested) through an empty repository of the review's own.
        let d = tempfile::tempdir().expect("tempdir");
        assert!(
            d.path().ancestors().all(|a| !a.join(".git").exists()),
            "TMPDIR sits inside a git repository"
        );
        std::fs::create_dir_all(d.path().join("src/deep")).expect("mkdir");
        std::fs::write(d.path().join(".gitignore"), "*.env\n").expect("write");
        std::fs::write(d.path().join("src/.gitignore"), "deep/\n").expect("write");
        std::fs::write(d.path().join("src/a.rs"), "fn a() {}\n").expect("write");
        std::fs::write(d.path().join("src/b.env"), "HOST=internal\n").expect("write");
        std::fs::write(d.path().join("src/deep/c.rs"), "fn c() {}\n").expect("write");
        let allow = |p: &std::path::Path| -> Result<std::path::PathBuf, ReviewError> {
            Ok(p.to_path_buf())
        };
        let r = magi_rs::review::input::resolve_input(
            &ReviewInput::Directory("src".into()),
            d.path(),
            &ScrubbedGit::new(d.path().to_path_buf()),
            &allow,
        )
        .expect("lists");
        assert_eq!(
            r.files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(),
            vec!["src/.gitignore", "src/a.rs"]
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
    fn an_uncommitted_review_keeps_an_unmerged_path_with_its_conflict_markers() {
        // [MAGI S6 loop 9, A1-c2, false positive] a review read `git diff HEAD` during a
        // conflict as a combined `diff --cc`, which `split_diff` drops. Measured with git 2.56:
        // a diff against a commit compares that tree with the working tree and prints a plain
        // `diff --git` section for an unmerged path (only the index-vs-worktree `git diff`
        // combines), so the conflicted file is reviewed with its markers.
        let d = repo();
        std::fs::write(
            d.path().join("a.rs"),
            "fn a() {}
",
        )
        .expect("write");
        git(d.path(), &["add", "a.rs"]);
        git(d.path(), &["commit", "-q", "-m", "1"]);
        git(d.path(), &["checkout", "-q", "-b", "side"]);
        std::fs::write(
            d.path().join("a.rs"),
            "fn b() {}
",
        )
        .expect("write");
        git(d.path(), &["commit", "-q", "-am", "b"]);
        git(d.path(), &["checkout", "-q", "main"]);
        std::fs::write(
            d.path().join("a.rs"),
            "fn c() {}
",
        )
        .expect("write");
        git(d.path(), &["commit", "-q", "-am", "c"]);
        let merged = std::process::Command::new("git")
            .arg("-C")
            .arg(d.path())
            .args(["merge", "-q", "side"])
            .output()
            .expect("git runs");
        assert!(!merged.status.success(), "the merge must conflict");
        let allow = |p: &std::path::Path| -> Result<std::path::PathBuf, ReviewError> {
            Ok(p.to_path_buf())
        };
        let r = magi_rs::review::input::resolve_input(
            &magi_rs::review::input::ReviewInput::Uncommitted,
            d.path(),
            &ScrubbedGit::new(d.path().to_path_buf()),
            &allow,
        )
        .expect("resolves");
        let a = r
            .files
            .iter()
            .find(|f| f.path == "a.rs")
            .expect("a.rs is reviewed");
        assert!(
            a.diff
                .as_deref()
                .is_some_and(|t| t.contains("+<<<<<<< HEAD")),
            "{:?}",
            a.diff
        );
    }

    #[test]
    fn a_non_revision_attribute_only_hides_a_revision_section_never_changes_its_bytes() {
        // [MAGI S6 loop 9, A1-c3, declared] attributes outside the reviewed tree (a git older
        // than 2.46 reads the working tree's `.gitattributes`; every git reads
        // `$GIT_DIR/info/attributes`) were read as a breach of RV21. They decide only whether a
        // section shows as binary, never which revision's bytes are diffed: the worst case is
        // the file reviewed whole from the revision, with an INFO run-log entry. With git 2.46 or later the
        // working tree's file is not read at all.
        let d = repo();
        std::fs::write(
            d.path().join("a.rs"),
            "fn a() {}
",
        )
        .expect("write");
        git(d.path(), &["add", "a.rs"]);
        git(d.path(), &["commit", "-q", "-m", "1"]);
        std::fs::write(
            d.path().join("a.rs"),
            "fn b() {}
",
        )
        .expect("write");
        git(d.path(), &["commit", "-q", "-am", "2"]);
        let allow = |p: &std::path::Path| -> Result<std::path::PathBuf, ReviewError> {
            Ok(p.to_path_buf())
        };
        let section = || {
            let r = magi_rs::review::input::resolve_input(
                &magi_rs::review::input::ReviewInput::Commit(None),
                d.path(),
                &ScrubbedGit::new(d.path().to_path_buf()),
                &allow,
            )
            .expect("resolves");
            r.files
                .into_iter()
                .find(|f| f.path == "a.rs")
                .and_then(|f| f.diff)
                .expect("a.rs has a section")
        };
        std::fs::write(
            d.path().join(".gitattributes"),
            "*.rs -diff
",
        )
        .expect("write");
        let shown = section();
        assert!(shown.contains("+fn b() {}"), "{shown}");
        std::fs::write(
            d.path().join(".git/info/attributes"),
            "*.rs -diff
",
        )
        .expect("write");
        let hidden = section();
        assert!(
            magi_rs::review::input::is_binary_section(&hidden),
            "{hidden}"
        );
        assert!(!hidden.contains("fn "), "{hidden}");
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
    fn a_file_named_like_a_git_option_is_a_pathspec_not_an_option() {
        // [MAGI S6 loop 10, D2-m2, false positive] `ignored_paths` puts `--` before the pathspec,
        // so a file named `-x.rs` (or `--stdin`) keeps its ignore verdict and is never parsed as
        // an option of `ls-files`.
        let d = repo();
        std::fs::write(d.path().join(".gitignore"), "-x.rs\n--stdin\n").expect("write");
        for name in ["-x.rs", "--stdin", "-ok.rs"] {
            std::fs::write(d.path().join(name), "fn f() {}\n").expect("write");
        }
        let g = ScrubbedGit::new(d.path().to_path_buf());
        for (name, ignored) in [("-x.rs", true), ("--stdin", true), ("-ok.rs", false)] {
            let input = magi_rs::review::input::ReviewInput::File(name.into());
            let set = magi_rs::review::filter::ignored_paths(&input, name, d.path(), &g)
                .expect("one git call");
            assert_eq!(set.contains(name), ignored, "{name}: {set:?}");
            assert!(set.len() <= 1, "{name}: {set:?}");
        }
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

    #[test]
    fn file_and_directory_inputs_refuse_a_work_tree_moved_out_of_the_workspace() {
        // [MAGI §6 MS2 A loop 2] every working-tree read that asks git needs the workspace inside
        // git's work tree; with `core.worktree` elsewhere, git's answers describe another tree.
        let d = repo();
        let elsewhere = tempfile::tempdir().expect("tempdir");
        std::fs::write(d.path().join("a.rs"), "fn a() {}\n").expect("write");
        git(d.path(), &["add", "a.rs"]);
        git(d.path(), &["commit", "-q", "-m", "1"]);
        let target = elsewhere.path().to_str().expect("utf-8 tempdir");
        git(d.path(), &["config", "core.worktree", target]);
        let g = ScrubbedGit::new(d.path().to_path_buf());
        let allow = |p: &std::path::Path| -> Result<std::path::PathBuf, ReviewError> {
            Ok(p.to_path_buf())
        };
        for input in [
            ReviewInput::File("a.rs".into()),
            ReviewInput::Directory(".".into()),
        ] {
            let pathspec = match &input {
                ReviewInput::File(_) => "a.rs",
                _ => ".",
            };
            assert!(
                matches!(
                    magi_rs::review::filter::ignored_paths(&input, pathspec, d.path(), &g),
                    Err(ReviewError::Git(_))
                ),
                "{input:?}: the ignore query ran against another tree"
            );
        }
        assert!(matches!(
            magi_rs::review::input::resolve_input(
                &ReviewInput::Directory(".".into()),
                d.path(),
                &g,
                &allow
            ),
            Err(ReviewError::Git(_))
        ));
    }

    /// Resolves `input` in `ws` with the real git and filters it through the review's
    /// worktree source, as a review does before batching.
    fn resolve_and_filter(
        ws: &std::path::Path,
        input: &ReviewInput,
    ) -> Result<magi_rs::review::filter::Filtered, ReviewError> {
        let allow = |p: &std::path::Path| -> Result<std::path::PathBuf, ReviewError> {
            Ok(p.to_path_buf())
        };
        let g = ScrubbedGit::new(ws.to_path_buf());
        let resolved = magi_rs::review::input::resolve_input(input, ws, &g, &allow)?;
        let src = magi_rs::review::source::WorktreeSource::new(ws.to_path_buf());
        let explicit = matches!(input, ReviewInput::File(_));
        magi_rs::review::filter::filter_files(
            resolved.files,
            &src,
            &magi_rs::review::languages::LanguageRegistry::new(),
            explicit,
            &|_| false,
        )
    }

    #[test]
    fn a_repository_listing_never_keeps_a_never_reviewed_path() {
        // [MAGI S6 loop 7, A1-m1/A1-c2] the repository branch of a directory review lists with
        // `ls-files` and does not prune `.magi`, `.git` or the memory database itself: the
        // filter is the one enforcement point (RV22 clause 1) for every listing. Pinned with a
        // repository that neither ignores nor untracks them.
        let d = repo();
        std::fs::create_dir_all(d.path().join(".magi")).expect("mkdir");
        std::fs::write(d.path().join(".magi/magi.toml"), "x = 1\n").expect("write");
        std::fs::write(d.path().join(".magi-rs-memory.db"), "db\n").expect("write");
        std::fs::write(d.path().join("a.rs"), "fn a() {}\n").expect("write");
        git(d.path(), &["add", "-f", ".magi-rs-memory.db", "a.rs"]);
        git(d.path(), &["commit", "-q", "-m", "1"]);
        let f = resolve_and_filter(d.path(), &ReviewInput::Directory(".".into())).expect("a.rs");
        assert_eq!(
            f.kept.iter().map(|k| k.path.as_str()).collect::<Vec<_>>(),
            vec!["a.rs"]
        );
        let mut skipped: Vec<_> = f
            .skipped
            .iter()
            .map(|s| (s.path.clone(), s.cause))
            .collect();
        skipped.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(
            skipped,
            vec![
                (
                    ".magi-rs-memory.db".to_string(),
                    magi_rs::review::filter::SkipCause::Ignored
                ),
                (
                    ".magi/magi.toml".to_string(),
                    magi_rs::review::filter::SkipCause::Ignored
                ),
            ]
        );
    }

    /// The marker the outside file holds; no kept file may carry it.
    #[cfg(unix)]
    const OUTSIDE_MARKER: &str = "OUTSIDE-THE-WORKSPACE";

    /// Asserts `link.rs` and `dir-link` were left out as outside the workspace, `a.rs` was kept,
    /// and no byte of the outside file reaches a kept file.
    #[cfg(unix)]
    fn assert_outside_links_skipped(ws: &std::path::Path, f: &magi_rs::review::filter::Filtered) {
        use magi_rs::review::filter::SkipCause;
        let mut kept: Vec<_> = f.kept.iter().map(|k| k.path.as_str()).collect();
        kept.retain(|p| *p != "base.rs");
        assert_eq!(kept, vec!["a.rs"]);
        let mut skipped: Vec<_> = f
            .skipped
            .iter()
            .map(|s| (s.path.as_str(), s.cause))
            .collect();
        skipped.sort_by_key(|(p, _)| p.to_string());
        assert_eq!(
            skipped,
            vec![
                ("dir-link", SkipCause::OutsideWorkspace),
                ("link.rs", SkipCause::OutsideWorkspace)
            ]
        );
        let src = magi_rs::review::source::WorktreeSource::new(ws.to_path_buf());
        for k in &f.kept {
            let bytes = src.read(&k.path).expect("read").expect("present");
            assert!(!String::from_utf8_lossy(&bytes).contains(OUTSIDE_MARKER));
        }
    }

    /// An outside directory holding a marked file, and a workspace with `a.rs`, `link.rs` (a link
    /// to the outside file) and `dir-link` (a link to the outside directory).
    #[cfg(unix)]
    fn workspace_with_outside_links(ws: &std::path::Path, outer: &std::path::Path) {
        std::fs::write(outer.join("secret.rs"), format!("// {OUTSIDE_MARKER}\n")).expect("write");
        std::fs::write(ws.join("a.rs"), "fn a() {}\n").expect("write");
        std::os::unix::fs::symlink(outer.join("secret.rs"), ws.join("link.rs")).expect("symlink");
        std::os::unix::fs::symlink(outer, ws.join("dir-link")).expect("symlink");
    }

    #[cfg(unix)]
    #[test]
    fn a_tracked_link_out_of_the_workspace_is_skipped_in_a_directory_review() {
        // [MAGI S6 MS2 loop 4, A1 Caspar, security] git lists a tracked link as a file; the read
        // through the worktree source refuses its outside target, so it is skipped, never sent.
        let d = repo();
        let outer = tempfile::tempdir().expect("tempdir");
        workspace_with_outside_links(d.path(), outer.path());
        git(d.path(), &["add", "."]);
        git(d.path(), &["commit", "-q", "-m", "1"]);
        let f = resolve_and_filter(d.path(), &ReviewInput::Directory(".".into())).expect("a.rs");
        assert_outside_links_skipped(d.path(), &f);
    }

    #[cfg(unix)]
    #[test]
    fn an_untracked_link_out_of_the_workspace_is_skipped_in_an_uncommitted_review() {
        // [MAGI S6 MS2 loop 4, A1 Caspar, security] untracked links are listed whole.
        let d = repo();
        std::fs::write(d.path().join("base.rs"), "fn b() {}\n").expect("write");
        git(d.path(), &["add", "."]);
        git(d.path(), &["commit", "-q", "-m", "1"]);
        let outer = tempfile::tempdir().expect("tempdir");
        workspace_with_outside_links(d.path(), outer.path());
        let f = resolve_and_filter(d.path(), &ReviewInput::Uncommitted).expect("a.rs");
        assert_outside_links_skipped(d.path(), &f);
    }

    #[cfg(unix)]
    #[test]
    fn a_link_out_of_the_workspace_is_skipped_in_a_walk_outside_any_repository() {
        // [MAGI S6 MS2 loop 4, A1 Caspar, security] the walk lists a link without following it.
        let d = tempfile::tempdir().expect("tempdir");
        assert!(
            d.path().ancestors().all(|a| !a.join(".git").exists()),
            "TMPDIR sits inside a git repository"
        );
        let ws = d.path().join("ws");
        std::fs::create_dir_all(&ws).expect("mkdir");
        let outer = tempfile::tempdir().expect("tempdir");
        workspace_with_outside_links(&ws, outer.path());
        let f = resolve_and_filter(&ws, &ReviewInput::Directory(".".into())).expect("a.rs");
        assert_outside_links_skipped(&ws, &f);
    }

    #[cfg(unix)]
    #[test]
    fn an_explicit_fifo_or_missing_file_is_a_typed_error_without_blocking() {
        // [MAGI S6 MS2 loop 4, A1 Caspar] the explicit file is checked when the review reads it:
        // a FIFO is refused by its type before any open, a missing file is `missing`.
        let d = repo();
        let made = std::process::Command::new("mkfifo")
            .arg(d.path().join("pipe.rs"))
            .status()
            .expect("mkfifo runs");
        assert!(made.success());
        let (tx, rx) = std::sync::mpsc::channel();
        let ws = d.path().to_path_buf();
        std::thread::spawn(move || {
            let fifo = resolve_and_filter(&ws, &ReviewInput::File("pipe.rs".into()));
            let missing = resolve_and_filter(&ws, &ReviewInput::File("gone.rs".into()));
            let _ = tx.send((fifo, missing));
        });
        let (fifo, missing) = rx
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect("neither input blocks");
        assert!(
            matches!(
                fifo,
                Err(ReviewError::NotReviewable {
                    cause: "not a regular file",
                    ..
                })
            ),
            "{fifo:?}"
        );
        assert!(
            matches!(
                missing,
                Err(ReviewError::NotReviewable {
                    cause: "missing",
                    ..
                })
            ),
            "{missing:?}"
        );
    }
    #[test]
    fn a_file_replaced_by_a_submodule_and_back_is_reviewed_without_a_git_error() {
        // [MAGI S6 MS2 loop 4, A2 Melchior, measured with git 2.56] the path list held the
        // gitlink, so reading the deleted file's new side ran `git show <rev>:./x.rs` on a
        // submodule commit ("bad object") and failed the whole review. Both directions.
        let d = repo();
        std::fs::write(d.path().join("x.rs"), "fn x() {}\n").expect("write");
        std::fs::write(d.path().join("k.rs"), "fn k() {}\n").expect("write");
        git(d.path(), &["add", "."]);
        git(d.path(), &["commit", "-q", "-m", "1"]);
        git(d.path(), &["rm", "-q", "--cached", "x.rs"]);
        std::fs::remove_file(d.path().join("x.rs")).expect("rm");
        let gitlink = "160000,1111111111111111111111111111111111111111,x.rs";
        git(d.path(), &["update-index", "--add", "--cacheinfo", gitlink]);
        git(d.path(), &["commit", "-q", "-m", "2"]);
        git(d.path(), &["rm", "-q", "--cached", "x.rs"]);
        std::fs::write(d.path().join("x.rs"), "fn y() {}\n").expect("write");
        git(d.path(), &["add", "x.rs"]);
        git(d.path(), &["commit", "-q", "-m", "3"]);
        let allow = |p: &std::path::Path| -> Result<std::path::PathBuf, ReviewError> {
            Ok(p.to_path_buf())
        };
        let g = ScrubbedGit::new(d.path().to_path_buf());
        for rev in ["HEAD~1", "HEAD"] {
            let input = ReviewInput::Commit(Some(rev.into()));
            let r = magi_rs::review::input::resolve_input(&input, d.path(), &g, &allow)
                .expect("resolves");
            let magi_rs::review::input::SourceKind::Revision { new_rev, old_rev } = r.source_kind
            else {
                panic!("a commit is read from its revision");
            };
            let src = RevisionSource::new(&g, new_rev, old_rev);
            let f = magi_rs::review::filter::filter_files(
                r.files,
                &src,
                &magi_rs::review::languages::LanguageRegistry::new(),
                false,
                &|_| false,
            )
            .unwrap_or_else(|e| panic!("{rev}: {e}"));
            assert!(
                f.kept.iter().any(|k| k.path == "x.rs"),
                "{rev}: {:?}",
                f.kept.iter().map(|k| &k.path).collect::<Vec<_>>()
            );
        }
    }
    #[test]
    fn a_capped_call_stops_git_at_its_own_cap() {
        // [MAGI S6 MS2 loop 4, A2 Balthasar] a per-call cap below the runner's: git is stopped
        // there instead of its output being read whole and then refused.
        let d = repo();
        std::fs::write(d.path().join("big.rs"), "a".repeat(2_048)).expect("write");
        git(d.path(), &["add", "big.rs"]);
        git(d.path(), &["commit", "-q", "-m", "1"]);
        let g = ScrubbedGit::new(d.path().to_path_buf());
        let show = ["show", "--no-ext-diff", "--no-textconv", "HEAD:./big.rs"];
        assert!(matches!(
            g.run_bytes_capped(&show, 1_024),
            Err(ReviewError::GitOutputTooLarge { limit: 1_024 })
        ));
        assert_eq!(g.run_bytes_capped(&show, 4_096).expect("fits").len(), 2_048);
    }

    #[test]
    fn git_is_resolved_from_an_absolute_path_entry_outside_the_workspace() {
        // [MAGI S6 loop 9, A1-c1, security] git was spawned by bare name, so a relative PATH
        // entry (`.`, an empty one) or one inside the reviewed repository could run a planted
        // git before any config was neutralized. Only an absolute entry outside the workspace
        // may provide it.
        let workspace = tempfile::tempdir().expect("workspace");
        let inner = workspace.path().join("tools");
        std::fs::create_dir(&inner).expect("inner dir");
        let trusted = tempfile::tempdir().expect("trusted");
        // Relative to the test's working directory (the package root under cargo and nextest).
        let planted = tempfile::tempdir_in("target").expect("planted dir in target");
        let relative = Path::new("target").join(planted.path().file_name().expect("name"));
        for dir in [
            workspace.path(),
            inner.as_path(),
            trusted.path(),
            planted.path(),
        ] {
            std::fs::write(dir.join(GIT_PROGRAM), b"").expect("plant");
        }
        assert!(
            relative.join(GIT_PROGRAM).is_file(),
            "the relative entry must resolve"
        );
        let path_var = std::env::join_paths([
            relative.as_path(),
            workspace.path(),
            inner.as_path(),
            trusted.path(),
        ])
        .expect("join");
        assert_eq!(
            resolve_git(&path_var, workspace.path()),
            Some(trusted.path().join(GIT_PROGRAM))
        );
        let none = std::env::join_paths([relative.as_path(), workspace.path()]).expect("join");
        assert_eq!(resolve_git(&none, workspace.path()), None);
    }

    #[test]
    fn git_is_not_resolved_when_the_workspace_cannot_be_canonicalized() {
        // [MAGI S6 loop 33, A1, security] a workspace that could not be canonicalized fell back
        // to its raw path, so the check that git lies outside the workspace compared a
        // canonical candidate against a non-canonical root and could stop protecting. It now
        // fails closed: no git is resolved.
        let trusted = tempfile::tempdir().expect("trusted");
        std::fs::write(trusted.path().join(GIT_PROGRAM), b"").expect("plant");
        let missing = tempfile::tempdir().expect("parent").path().join("missing");
        let path_var = std::env::join_paths([trusted.path()]).expect("join");
        assert_eq!(resolve_git(&path_var, &missing), None);
    }

    #[test]
    fn git_is_not_resolved_from_the_enclosing_repository_of_a_sub_workspace() {
        // [MAGI S6 loop 35, A1, security] only entries under the workspace were refused: with
        // the workspace a subdirectory of a repository, a git planted elsewhere in that
        // repository (a vendored `tools/`, an activated `.venv`) ran. Every directory of the
        // outermost enclosing repository is now repository data.
        let outer = tempfile::tempdir().expect("outer repo");
        std::fs::create_dir(outer.path().join(".git")).expect("outer .git");
        let inner_repo = outer.path().join("module");
        std::fs::create_dir(&inner_repo).expect("submodule dir");
        std::fs::write(inner_repo.join(".git"), b"gitdir: ../.git/modules/module\n")
            .expect("submodule .git file");
        let workdir = inner_repo.join("sub");
        std::fs::create_dir(&workdir).expect("workdir");
        let tools = outer.path().join("tools");
        let vendored = inner_repo.join("tools");
        let trusted = tempfile::tempdir().expect("trusted");
        for dir in [tools.as_path(), vendored.as_path(), trusted.path()] {
            std::fs::create_dir_all(dir).expect("dir");
            std::fs::write(dir.join(GIT_PROGRAM), b"").expect("plant");
        }
        let planted = std::env::join_paths([tools.as_path(), vendored.as_path()]).expect("join");
        assert_eq!(resolve_git(&planted, &workdir), None);
        let with_trusted =
            std::env::join_paths([tools.as_path(), vendored.as_path(), trusted.path()])
                .expect("join");
        assert_eq!(
            resolve_git(&with_trusted, &workdir),
            Some(trusted.path().join(GIT_PROGRAM))
        );
    }

    #[test]
    fn an_uncommitted_review_in_a_sub_workspace_names_ignored_files_like_its_diff() {
        // [MAGI S6 loop 13, A2-m1, false positive] the review read `ls-files` as repository-root
        // relative: it is relative to its cwd (the workspace, `-C`) unless `--full-name` is
        // given, and the diff carries `--relative`, so both speak workspace-relative paths.
        // Measured with git 2.56; adding `--full-name` (the proposed fix) turns this red.
        let d = repo();
        std::fs::create_dir_all(d.path().join("ws/sub")).expect("mkdir");
        std::fs::write(d.path().join(".gitignore"), "*.env\n").expect("write");
        std::fs::write(d.path().join("ws/sub/tracked.env"), "HOST=a\n").expect("write");
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
        std::fs::write(d.path().join("ws/sub/tracked.env"), "HOST=b\n").expect("write");
        std::fs::write(d.path().join("ws/sub/a.rs"), "fn a() { 1 }\n").expect("write");
        let ws = d.path().join("ws");
        let g = ScrubbedGit::new(ws.clone());
        let allow = |p: &std::path::Path| -> Result<std::path::PathBuf, ReviewError> {
            Ok(p.to_path_buf())
        };
        let input = ReviewInput::Uncommitted;
        let listed =
            magi_rs::review::input::resolve_input(&input, &ws, &g, &allow).expect("resolves");
        let paths: Vec<&str> = listed.files.iter().map(|f| f.path.as_str()).collect();
        assert!(paths.contains(&"sub/tracked.env"), "{paths:?}");
        assert!(
            listed
                .files
                .iter()
                .find(|f| f.path == "sub/tracked.env")
                .is_some_and(|f| f.diff.is_some()),
            "the path comes from the diff"
        );
        let ignored =
            magi_rs::review::filter::ignored_paths(&input, ".", &ws, &g).expect("one git call");
        assert_eq!(
            ignored.iter().map(String::as_str).collect::<Vec<_>>(),
            vec!["sub/tracked.env"]
        );
    }

    #[test]
    fn an_ignored_file_staged_for_deletion_or_renamed_never_sends_its_old_side() {
        // [MAGI S6 loop 16, A2-2, security] the ignore set came from the index alone, while the
        // uncommitted diff's old side comes from HEAD: a tracked ignored file staged for deletion,
        // or renamed away, was absent from the index and its HEAD content was kept.
        let d = repo();
        std::fs::write(
            d.path().join(".gitignore"),
            "*.env
",
        )
        .expect("write");
        std::fs::write(
            d.path().join("gone.env"),
            "HOST=a
",
        )
        .expect("write");
        std::fs::write(
            d.path().join("moved.env"),
            "HOST=b
",
        )
        .expect("write");
        std::fs::write(
            d.path().join("a.rs"),
            "fn a() {}
",
        )
        .expect("write");
        git(
            d.path(),
            &["add", "-f", ".gitignore", "gone.env", "moved.env", "a.rs"],
        );
        git(d.path(), &["commit", "-q", "-m", "1"]);
        git(d.path(), &["rm", "-q", "gone.env"]);
        git(d.path(), &["mv", "moved.env", "moved.rs"]);
        std::fs::write(
            d.path().join("moved.rs"),
            "HOST=c
",
        )
        .expect("write");
        std::fs::write(
            d.path().join("a.rs"),
            "fn a() { 1 }
",
        )
        .expect("write");
        let ws = d.path();
        let g = ScrubbedGit::new(ws.to_path_buf());
        let allow = |p: &std::path::Path| -> Result<std::path::PathBuf, ReviewError> {
            Ok(p.to_path_buf())
        };
        let input = ReviewInput::Uncommitted;
        let ignored =
            magi_rs::review::filter::ignored_paths(&input, ".", ws, &g).expect("ignore set");
        let resolved =
            magi_rs::review::input::resolve_input(&input, ws, &g, &allow).expect("resolves");
        let magi_rs::review::input::SourceKind::Worktree {
            old_rev: Some(old_rev),
        } = resolved.source_kind
        else {
            panic!("a worktree source with an old side");
        };
        let src = magi_rs::review::source::WorktreeSource::with_old(ws.to_path_buf(), &g, old_rev);
        let f = magi_rs::review::filter::filter_files(
            resolved.files,
            &src,
            &magi_rs::review::languages::LanguageRegistry::new(),
            false,
            &|p| ignored.contains(p),
        )
        .expect("a.rs kept");
        let kept: Vec<&str> = f.kept.iter().map(|k| k.path.as_str()).collect();
        assert!(kept.contains(&"a.rs"), "{kept:?}");
        for old in ["gone.env", "moved.env"] {
            assert!(!kept.contains(&old), "{old} was kept: {kept:?}");
        }
        assert!(
            f.kept.iter().all(|k| k
                .diff
                .as_deref()
                .is_none_or(|d| !d.contains("HOST=a") && !d.contains("HOST=b"))),
            "an ignored old side reached a kept diff"
        );
    }

    #[test]
    fn a_path_named_with_a_leading_dot_slash_is_read_from_a_revision() {
        // [MAGI S6 loop 17, A2-3] the object spec `<rev>:./<path>` with a `./` path is
        // `<rev>:././<path>` (not `<rev>:../<path>`, as the review read it), which git resolves;
        // the check that the revision holds the path compares `ls-tree`'s names, which never
        // carry a `./`, so the path must be read as git names it.
        let d = repo();
        std::fs::create_dir(d.path().join("src")).expect("mkdir");
        std::fs::write(d.path().join("a.rs"), "one\n").expect("write");
        std::fs::write(d.path().join("src/b.rs"), "two\n").expect("write");
        git(d.path(), &["add", "a.rs", "src/b.rs"]);
        git(d.path(), &["commit", "-q", "-m", "1"]);
        std::fs::write(d.path().join("a.rs"), "dirty\n").expect("write");
        let g = ScrubbedGit::new(d.path().to_path_buf());
        let src = RevisionSource::new(&g, "HEAD".into(), None);
        for (path, want) in [
            ("./a.rs", "one\n"),
            ("./src/b.rs", "two\n"),
            ("src/./b.rs", "two\n"),
        ] {
            assert_eq!(
                src.read(path).expect("read"),
                Some(want.as_bytes().to_vec()),
                "{path}"
            );
        }
    }

    #[test]
    fn the_uncommitted_ignore_set_names_heads_ignored_paths_gone_from_the_index() {
        // [MAGI S6 loop 17, A2-4] the uncommitted ignore set relies on `ls-files --with-tree`
        // listing HEAD's tree paths, which git-ls-files(1) documents only for
        // `--error-unmatch`; a git that stopped listing them would widen what is sent. The set
        // is pinned exactly: HEAD's ignored files staged for deletion or renamed away, nothing
        // else.
        let d = repo();
        std::fs::write(d.path().join(".gitignore"), "*.env\n").expect("write");
        std::fs::write(d.path().join("gone.env"), "HOST=a\n").expect("write");
        std::fs::write(d.path().join("moved.env"), "HOST=b\n").expect("write");
        std::fs::write(d.path().join("a.rs"), "fn a() {}\n").expect("write");
        git(
            d.path(),
            &["add", "-f", ".gitignore", "gone.env", "moved.env", "a.rs"],
        );
        git(d.path(), &["commit", "-q", "-m", "1"]);
        git(d.path(), &["rm", "-q", "gone.env"]);
        git(d.path(), &["mv", "moved.env", "moved.rs"]);
        let g = ScrubbedGit::new(d.path().to_path_buf());
        let ignored =
            magi_rs::review::filter::ignored_paths(&ReviewInput::Uncommitted, ".", d.path(), &g)
                .expect("ignore set");
        let mut names: Vec<&str> = ignored.iter().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(names, vec!["gone.env", "moved.env"]);
    }

    #[test]
    fn a_file_under_an_ignored_directory_named_explicitly_is_refused_as_ignored() {
        // [MAGI S6 loop 17, A2-5] the `File` ignore set comes from `ls-files --others --ignored`,
        // which could collapse an ignored directory to `dir/`; a file under it, named
        // explicitly, must be refused as `ignored` like a file a rule matches directly, tracked
        // or not.
        let d = repo();
        std::fs::write(d.path().join(".gitignore"), "dir/\n").expect("write");
        std::fs::create_dir_all(d.path().join("dir/deep")).expect("mkdir");
        for name in ["dir/tracked.rs", "dir/file.rs", "dir/deep/file.rs"] {
            std::fs::write(d.path().join(name), "fn f() {}\n").expect("write");
        }
        git(d.path(), &["add", "-f", ".gitignore", "dir/tracked.rs"]);
        git(d.path(), &["commit", "-q", "-m", "1"]);
        for name in ["dir/tracked.rs", "dir/file.rs", "dir/deep/file.rs"] {
            let input = ReviewInput::File(name.into());
            let g = ScrubbedGit::new(d.path().to_path_buf());
            let ignored = magi_rs::review::filter::ignored_paths(&input, name, d.path(), &g)
                .expect("ignore set");
            let src = magi_rs::review::source::WorktreeSource::new(d.path().to_path_buf());
            let one = vec![magi_rs::review::input::ReviewFile {
                path: name.into(),
                diff: None,
            }];
            let verdict = magi_rs::review::filter::filter_files(
                one,
                &src,
                &magi_rs::review::languages::LanguageRegistry::new(),
                true,
                &|p| ignored.contains(p),
            );
            assert!(
                matches!(
                    verdict,
                    Err(ReviewError::NotReviewable {
                        cause: "ignored",
                        ..
                    })
                ),
                "{name}: {verdict:?} with {ignored:?}"
            );
        }
    }
}
