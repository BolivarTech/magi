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

use tokio::process::{Child, Command};

use crate::tools::{ToolError, ToolResult};

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
    /// A Python worker that writes a `start.marker` immediately, sleeps well past
    /// any test's cancel, then — **only if it survives** — writes a `done.marker`.
    /// Paths are absolute (derived from the script's own directory) so the worker
    /// is cwd-independent. Used to prove a kill deterministically: START must
    /// exist (the real grandchild ran) and DONE must never appear (it was killed
    /// mid-work, not merely denied time — the test waits past the sleep).
    pub(crate) const TREE_KILL_WORKER: &str = "\
import os, time\n\
d = os.path.dirname(os.path.abspath(__file__))\n\
open(os.path.join(d, 'start.marker'), 'w').close()\n\
time.sleep(4)\n\
open(os.path.join(d, 'done.marker'), 'w').close()\n";

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
