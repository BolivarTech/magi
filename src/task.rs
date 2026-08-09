// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-08-09

//! Generic async task-cancellation primitives, homed at the crate root instead of inside
//! any one tool's module.
//!
//! [`AbortOnDrop`] used to live inside `tools::consult`, purely because that was its
//! first user. But it has a **second** caller, `headless_runner`'s direct `magi consult`
//! path (`analyze_direct`), which needs the exact same RAII backstop for its own spawned
//! MAGI analysis — and importing a general-purpose async guard through a specific tool's
//! module ties unrelated code to that tool's internals for no reason (MAGI S6 gate
//! finding). Neither caller is more "the" owner of this primitive than the other, so it
//! moved to a neutral module both can depend on without depending on each other.

use tokio::task::AbortHandle;

/// RAII backstop that aborts a spawned task when the guard is dropped.
///
/// Both [`ConsultTool::execute`][crate::tools::consult::ConsultTool::execute] and
/// `headless_runner`'s `analyze_direct` run a 3-perspective MAGI analysis on a
/// `tokio::spawn` task and await it under a `select!`. An explicit cancel arm aborts the
/// task on `--timeout`, but if the awaiting future itself is *dropped* before either arm
/// resolves (e.g. the caller drops the tool call), a bare spawned task would keep running
/// and orphan its three in-flight LLM calls. Holding this guard across the `select!`
/// aborts the task on that drop too, mirroring the `GroupKiller` backstop the `bash` tool
/// uses for its subprocess.
pub(crate) struct AbortOnDrop {
    /// Abort handle of the guarded task.
    handle: AbortHandle,
}

impl AbortOnDrop {
    /// Wraps a task's abort handle so dropping the guard aborts the task.
    pub(crate) fn new(handle: AbortHandle) -> Self {
        Self { handle }
    }

    /// Aborts the guarded task now. Idempotent: aborting an already-finished or
    /// already-aborted task is a no-op, so `Drop` re-invoking it is harmless.
    pub(crate) fn abort(&self) {
        self.handle.abort();
    }
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    /// The `AbortOnDrop` backstop aborts its guarded task the instant the guard is
    /// dropped, so a dropped `execute` future cannot orphan the spawned analysis (the
    /// drop path an explicit cancel arm does not cover). A bare dropped
    /// `JoinHandle`/`AbortHandle` would merely detach the task, leaving it to run to
    /// completion — this asserts the join reports cancellation and the task never
    /// reached its completion store.
    #[tokio::test]
    async fn test_abort_on_drop_aborts_task_when_guard_dropped() {
        let ran_to_completion = Arc::new(AtomicBool::new(false));
        let flag = ran_to_completion.clone();
        let handle = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(30)).await;
            flag.store(true, Ordering::SeqCst);
        });
        {
            let _guard = AbortOnDrop::new(handle.abort_handle());
            // `_guard` drops here without ever calling `abort()` explicitly.
        }
        let joined = handle.await;
        assert!(
            joined.as_ref().err().map(|e| e.is_cancelled()).unwrap_or(false),
            "dropping the guard must abort the task (join must report cancellation); got {joined:?}"
        );
        assert!(
            !ran_to_completion.load(Ordering::SeqCst),
            "aborted task must not have reached its completion store"
        );
    }

    /// `abort()` is idempotent: calling it explicitly and then letting the guard drop
    /// (which calls it again) must not panic — `AbortHandle::abort` is documented as
    /// safe to call on an already-finished or already-aborted task.
    #[tokio::test]
    async fn test_abort_is_idempotent_across_explicit_call_and_drop() {
        let handle = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(30)).await;
        });
        let guard = AbortOnDrop::new(handle.abort_handle());
        guard.abort();
        guard.abort(); // second explicit call, still before drop
        drop(guard); // Drop::drop calls abort() a third time
        let joined = handle.await;
        assert!(
            joined
                .as_ref()
                .err()
                .map(|e| e.is_cancelled())
                .unwrap_or(false),
            "task must report cancellation after repeated aborts; got {joined:?}"
        );
    }
}
