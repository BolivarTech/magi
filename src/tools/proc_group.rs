// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-07-18

//! Cross-platform spawn of a child process inside a **killable group**, so a
//! wall-clock timeout (REQ-H36) can terminate the whole subprocess *tree* — not
//! just the direct child — when a run is cancelled. Without this, a
//! `powershell -Command "cargo build"` (or `bash -c`) leaves the real worker
//! (`cargo`/`rustc`) orphaned when only the shell is killed, and the wall-clock
//! bound would be a lie.
//!
//! # Mechanism
//! - **Windows**: the shell is assigned to a Win32 **Job Object** created with
//!   `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` (via the safe [`win32job`] crate).
//!   Dropping the job handle terminates every process in the job, so a
//!   grandchild cannot outlive the kill. Assignment happens immediately after
//!   spawn; if it fails (the shell is already in a job we cannot join) the
//!   direct-child [`kill_on_drop`](tokio::process::Command::kill_on_drop)
//!   backstop still applies.
//! - **POSIX**: the shell leads a **new process group**
//!   ([`process_group(0)`](tokio::process::Command::process_group)); the group
//!   is signalled with `SIGKILL` via the safe [`rustix`] wrapper.
//!
//! # Two kill paths, one guarantee
//! [`GroupKiller`] kills the tree **both** on an explicit [`GroupKiller::kill`]
//! (the cooperative cancel branch of the caller's `select!`) **and** on `Drop`
//! (the backstop for when the future is dropped outright, e.g. by
//! [`tokio::time::timeout`] elapsing — which never polls the cancel branch).
//! Either way the tree dies. On a clean completion the caller calls
//! [`GroupKiller::disarm`] so the `Drop` backstop does not signal an
//! already-reaped (possibly reused) group.
//!
//! # No `unsafe`
//! Every platform primitive is behind an audited safe crate ([`win32job`] /
//! [`rustix`]); this module contains no `unsafe`, honoring the crate-wide
//! `#![forbid(unsafe_code)]`.
//!
//! ## Deviation from the milestone note (`CREATE_SUSPENDED`)
//! The plan suggested spawning `CREATE_SUSPENDED`, assigning to the Job while
//! suspended, then resuming — to close the fork↔assign race. Resuming a
//! suspended Windows process requires `ResumeThread` on the child's main-thread
//! handle, which neither `std` nor `win32job` exposes safely (it needs raw
//! Win32 `unsafe`). Per the "no `unsafe`" hard constraint, this module instead
//! assigns immediately post-spawn (the approach used by the well-worn
//! `command-group` crate): for a shell that must initialize before it can spawn
//! any grandchild, the window between `CreateProcess` returning and assignment
//! is not observably exploitable, and the tree-kill guarantee itself is intact.

use std::env;

use tokio::process::{Child, Command};

use crate::tools::{ToolError, ToolResult};

/// Environment variable names a spawned tool subprocess is allowed to inherit
/// (REQ-H37, defense-in-depth).
///
/// This is the **single source** of the allowlist, imported by both the bash
/// tool ([`scrub_child_env`]) and `main.rs` (which owns the parent-process env
/// scrub). Membership is tested by **exact, literal name equality** — never a
/// prefix — so a subprocess inherits neither a scrubbed secret nor an
/// attacker-chosen variable such as `LC_EVIL`.
///
/// The list is deliberately **broad enough** for the whitelisted shells and
/// binaries (`powershell`/`bash`, `cargo`, `git`, `npm`, `node`, `python`,
/// `pytest`, `rg`) to still function after [`Command::env_clear`]: it carries
/// the OS bootstrap, locale, temp-dir, home, and (non-secret) toolchain
/// configuration variables those programs legitimately need on Windows and
/// POSIX. It is curated to **exclude every secret** — `MAGI_PASSPHRASE`, the API
/// keys, and any `*_KEY`/`*_TOKEN`/`*_SECRET`/`*_PASSWORD` — which are never
/// operationally required by a build/search/VCS command.
pub(crate) const TOOL_ENV_ALLOWLIST: &[&str] = &[
    // ── Windows OS bootstrap (powershell + child processes need these) ──
    "PATH",
    "PATHEXT",
    "SystemRoot",
    "SystemDrive",
    "ComSpec",
    "windir",
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
    "APPDATA",
    "LOCALAPPDATA",
    "TEMP",
    "TMP",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
    "OS",
    "PSModulePath",
    // ── Cross-platform / POSIX shell + locale ──
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LC_MESSAGES",
    "TZ",
    "TERM",
    "SHELL",
    "TMPDIR",
    "USER",
    "LOGNAME",
    // ── Non-secret toolchain configuration (cargo / rustup / python / node / git) ──
    "CARGO_HOME",
    "RUSTUP_HOME",
    "RUSTUP_TOOLCHAIN",
    "PYTHONPATH",
    "PYTHONHOME",
    "NODE_PATH",
    "GIT_EXEC_PATH",
    "GIT_TEMPLATE_DIR",
];

/// Returns `true` if `name` is an allowlisted environment variable — by **whole-name
/// equality**, never a prefix, so `LC_EVIL` never matches `LC_ALL` and no secret
/// slips through.
///
/// The comparison is **case-insensitive on Windows** (where the OS itself treats
/// environment variable names case-insensitively and the real `PATH` is commonly
/// stored as `Path`) and **case-sensitive on POSIX** (where names are
/// case-sensitive by contract). It stays equality on both platforms, so
/// broadening to case-insensitive on Windows cannot let a secret through — a
/// secret name equals no allowlist entry under any casing.
fn is_allowed_env_name(name: &str) -> bool {
    #[cfg(windows)]
    {
        TOOL_ENV_ALLOWLIST
            .iter()
            .any(|allowed| allowed.eq_ignore_ascii_case(name))
    }
    #[cfg(not(windows))]
    {
        TOOL_ENV_ALLOWLIST.contains(&name)
    }
}

/// Returns the current process environment filtered to exactly the names
/// [`is_allowed_env_name`] admits, for use as a spawned tool subprocess's
/// environment (REQ-H37).
///
/// Filtering is by whole-name equality, so the returned pairs contain neither
/// the scrubbed secrets (which `main.rs` also removes from the parent env at
/// startup) nor an arbitrary variable such as `LC_EVIL`.
pub(crate) fn tool_child_env() -> Vec<(String, String)> {
    env::vars()
        .filter(|(name, _)| is_allowed_env_name(name))
        .collect()
}

/// Replaces `cmd`'s inherited environment with exactly the allowlisted, non-secret
/// variables (REQ-H37, defense-in-depth for the bash tool spawn).
///
/// Clears the whole inherited environment ([`Command::env_clear`]) then re-adds
/// only [`tool_child_env`], so an in-workspace interpreter (`python`/`node`)
/// cannot read `MAGI_PASSPHRASE` or an API key out of its own environment — even
/// if a future code path failed to scrub the parent process. This layers on top
/// of the parent-env scrub in `main.rs`; the two together close the env-exfil
/// vector. The hard barriers (allowlist, metachar ban, `PathGuard`) are
/// untouched — this only narrows the child's environment.
pub(crate) fn scrub_child_env(cmd: &mut Command) {
    cmd.env_clear();
    cmd.envs(tool_child_env());
}

/// A spawned child paired with the OS resource that can kill its whole tree.
///
/// Split into `(Child, GroupKiller)` via [`GroupChild::into_parts`] so the
/// caller can move the `Child` into a wait future while retaining the
/// `GroupKiller` to fire from a cancellation branch — without a borrow
/// conflict.
pub(crate) struct GroupChild {
    /// The spawned child. The caller keeps `kill_on_drop(true)` set on the
    /// originating [`Command`], so dropping this also kills the direct child.
    pub(crate) child: Child,
    /// The tree-killer, held separately from `child`.
    pub(crate) killer: GroupKiller,
}

impl GroupChild {
    /// Splits into the child and its tree-killer.
    pub(crate) fn into_parts(self) -> (Child, GroupKiller) {
        (self.child, self.killer)
    }
}

/// Kills the process *tree* of a [`GroupChild`], on explicit request or on drop.
///
/// See the module docs for the two kill paths. On Windows the tree dies via the
/// Job Object's kill-on-close; on POSIX via `kill(-pgid, SIGKILL)`.
pub(crate) struct GroupKiller {
    /// Windows: the kill-on-close Job Object. `None` if the job could not be
    /// created or the child could not be assigned (direct-child kill backstops).
    #[cfg(windows)]
    job: Option<win32job::Job>,
    /// POSIX: the child's process-group id (equal to its pid, since the child
    /// leads a fresh group). `None` if the pid was unavailable at spawn.
    #[cfg(unix)]
    pgid: Option<i32>,
    /// When `true`, the `Drop` backstop performs no kill (the run completed
    /// cleanly and the group is already reaped).
    disarmed: bool,
}

impl GroupKiller {
    /// Kills the process tree now, then marks the killer disarmed so `Drop` does
    /// not act again. Best-effort: a kill failure (already exited) is ignored.
    pub(crate) fn kill(&mut self) {
        self.kill_impl();
        self.disarmed = true;
    }

    /// Marks the killer disarmed without killing — call after a clean wait so the
    /// `Drop` backstop does not signal an already-reaped (possibly reused) group.
    pub(crate) fn disarm(&mut self) {
        self.disarmed = true;
    }

    /// Platform-specific tree kill. Idempotent and best-effort.
    fn kill_impl(&mut self) {
        #[cfg(windows)]
        {
            // Dropping the taken job handle triggers
            // `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, terminating the whole tree.
            let _ = self.job.take();
        }
        #[cfg(unix)]
        {
            if let Some(pgid) = self.pgid.take() {
                if let Some(pid) = rustix::process::Pid::from_raw(pgid) {
                    let _ = rustix::process::kill_process_group(pid, rustix::process::Signal::Kill);
                }
            }
        }
    }
}

impl Drop for GroupKiller {
    fn drop(&mut self) {
        if self.disarmed {
            return;
        }
        self.kill_impl();
    }
}

/// Spawns `cmd` inside a killable group and returns the child plus its
/// tree-killer.
///
/// The caller must have already set every other [`Command`] property
/// (`current_dir`, piped stdio, `kill_on_drop(true)`, program + args). Only the
/// group/job wiring is applied here.
///
/// # Errors
/// Returns [`ToolError::ExecutionError`] if the process cannot be spawned. Job
/// creation / process-group assignment failures are non-fatal (the direct-child
/// `kill_on_drop` backstop remains), so they do not surface as errors.
pub(crate) fn spawn_in_group(cmd: &mut Command) -> ToolResult<GroupChild> {
    #[cfg(unix)]
    {
        // New process group led by the child, so the whole group can be signalled.
        cmd.process_group(0);
    }

    let child = cmd
        .spawn()
        .map_err(|e| ToolError::ExecutionError(format!("Failed to spawn process: {e}")))?;

    #[cfg(windows)]
    let killer = {
        // Best-effort: assign the child to a kill-on-close Job Object.
        let job = build_kill_on_close_job();
        let assigned = match (job, child.raw_handle()) {
            (Some(job), Some(handle)) => {
                if job.assign_process(handle as isize).is_ok() {
                    Some(job)
                } else {
                    None
                }
            }
            _ => None,
        };
        GroupKiller {
            job: assigned,
            disarmed: false,
        }
    };

    #[cfg(unix)]
    let killer = GroupKiller {
        pgid: child.id().map(|id| id as i32),
        disarmed: false,
    };

    #[cfg(not(any(windows, unix)))]
    let killer = GroupKiller { disarmed: false };

    Ok(GroupChild { child, killer })
}

/// Builds a Job Object whose kill-on-close limit terminates the whole tree when
/// its handle is dropped. `None` if creation fails (the caller degrades to the
/// direct-child kill backstop).
#[cfg(windows)]
fn build_kill_on_close_job() -> Option<win32job::Job> {
    let mut info = win32job::ExtendedLimitInfo::new();
    info.limit_kill_on_job_close();
    win32job::Job::create_with_limit_info(&info).ok()
}

/// Shared fixtures for the deterministic subprocess-kill tests, reused by both
/// the `bash` tool tests and the headless-runner timeout test (DRY).
#[cfg(test)]
pub(crate) mod test_support {
    /// How long the worker (or, for the grandchild fixture, the grandchild
    /// itself) sleeps before it would write `done.marker` if left to survive.
    /// Must comfortably outlast [`CANCEL_FIRE_DELAY_MS`] under CPU contention —
    /// see the module-level rustdoc on [`CANCEL_FIRE_DELAY_MS`] for the full
    /// budget rationale shared by `bash.rs` and `headless_runner.rs`.
    pub(crate) const WORKER_SLEEP_SECS: u64 = 15;

    /// How long the **parent** (direct child of the shell) sleeps in the
    /// grandchild fixture after spawning the grandchild, keeping the shell
    /// blocked so the run cannot complete cleanly before the cancel/timeout
    /// fires. Must exceed [`CANCEL_FIRE_DELAY_MS`] with generous margin for the
    /// same cold-start-under-contention reason as [`WORKER_SLEEP_SECS`].
    pub(crate) const GRANDCHILD_PARENT_SLEEP_SECS: u64 = 18;

    /// Delay before a test fires its cancellation/timeout trigger. Sized to
    /// absorb interpreter (and, on Windows, shell) **cold-start latency under
    /// full-suite CPU contention** — when hundreds of tests (including
    /// 120–220 s memory tests) saturate every core, `python`/`powershell`
    /// process creation can be delayed well past the ~2 s this used to be,
    /// which previously caused a spurious "START marker missing" failure
    /// (the kill fired before the worker had even started, not a kill-logic
    /// bug). 5 s gives comfortable headroom while still firing well before
    /// [`WORKER_SLEEP_SECS`]/[`GRANDCHILD_PARENT_SLEEP_SECS`] elapse, so the
    /// kill still genuinely pre-empts live work rather than racing completion.
    pub(crate) const CANCEL_FIRE_DELAY_MS: u64 = 5_000;

    /// How long a test waits, after the kill/timeout has fired, before
    /// asserting `done.marker` is absent. Must exceed the worker's *remaining*
    /// sleep at the moment of the kill (`WORKER_SLEEP_SECS * 1000 -
    /// CANCEL_FIRE_DELAY_MS` = 10 000 ms) with margin, so a surviving
    /// (un-killed) orphan would provably have written `done.marker` by the
    /// time we check — otherwise the assertion would be meaningless.
    pub(crate) const POST_KILL_WAIT_MS: u64 = 13_000;

    /// A Python worker that writes a `start.marker` immediately, sleeps well past
    /// any test's cancel, then — **only if it survives** — writes a `done.marker`.
    /// Paths are absolute (derived from the script's own directory) so the worker
    /// is cwd-independent. Used to prove a kill deterministically: START must
    /// exist (the real grandchild ran) and DONE must never appear (it was killed
    /// mid-work, not merely denied time — the test waits past the sleep).
    pub(crate) fn tree_kill_worker() -> String {
        format!(
            "\
import os, time\n\
d = os.path.dirname(os.path.abspath(__file__))\n\
open(os.path.join(d, 'start.marker'), 'w').close()\n\
time.sleep({WORKER_SLEEP_SECS})\n\
open(os.path.join(d, 'done.marker'), 'w').close()\n"
        )
    }

    /// A two-level worker that spawns a **grandchild** (relative to the shell:
    /// shell→python→python) and lingers so the shell keeps waiting. The grandchild
    /// writes `start.marker`, sleeps well past any test's cancel, then — **only if
    /// it survives** — writes `done.marker`. The parent sleep keeps the shell
    /// blocked past the cancel so the run does not complete cleanly first. Proves
    /// the Job Object / process group kills the **whole tree** (a detached
    /// grandchild has no job of its own), not merely the direct child: START must
    /// exist (the grandchild ran) and DONE must never appear (it was killed).
    pub(crate) fn tree_kill_grandchild_worker() -> String {
        format!(
            "\
import os, sys, subprocess, time\n\
d = os.path.dirname(os.path.abspath(__file__))\n\
gc = (\n\
    \"import os, time\\n\"\n\
    \"d = \" + repr(d) + \"\\n\"\n\
    \"open(os.path.join(d, 'start.marker'), 'w').close()\\n\"\n\
    \"time.sleep({WORKER_SLEEP_SECS})\\n\"\n\
    \"open(os.path.join(d, 'done.marker'), 'w').close()\\n\"\n\
)\n\
subprocess.Popen([sys.executable, \"-c\", gc])\n\
time.sleep({GRANDCHILD_PARENT_SLEEP_SECS})\n"
        )
    }

    /// A probe that reports whether it can read the three magi-managed secrets
    /// (`MAGI_PASSPHRASE`/`ANTHROPIC_API_KEY`/`OPENAI_API_KEY`) from its own
    /// environment and, on unix, from `/proc/self/environ`. It writes every name
    /// it *found* into `exfil.marker` (an empty file iff none leaked), so the test
    /// asserts the marker exists (the probe ran) **and** is empty (no secret
    /// reached the child — REQ-H37 pen-test).
    // NOTE: written with **no indented blocks** (single-line `try:`/`except:` and
    // `open(...).write(...)`), because Rust's `\`-line-continuation strips the
    // leading whitespace of the next source line — indented Python here would
    // become an `IndentationError`. Compound statements stay on one physical line.
    pub(crate) const EXFIL_PROBE_WORKER: &str = "\
import os\n\
d = os.path.dirname(os.path.abspath(__file__))\n\
names = ['MAGI_PASSPHRASE', 'ANTHROPIC_API_KEY', 'OPENAI_API_KEY']\n\
found = [n for n in names if n in os.environ]\n\
raw = b''\n\
try: raw = open('/proc/self/environ', 'rb').read()\n\
except Exception: raw = b''\n\
found += [n + ':proc' for n in names if (n + '=').encode() in raw]\n\
open(os.path.join(d, 'exfil.marker'), 'w').write(','.join(found))\n";

    /// `true` if a `python` interpreter can be launched — the kill mechanism
    /// cannot be verified without a real long-lived child, so callers skip when
    /// this is `false`.
    pub(crate) fn python_available() -> bool {
        std::process::Command::new("python")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
}
