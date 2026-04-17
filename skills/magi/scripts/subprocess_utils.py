#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-04-17
"""Subprocess reap + diagnostics helpers for the MAGI orchestrator.

Extracted from ``run_magi.py`` so the orchestrator file no longer has
to hold the Windows kill-tree rules and stderr-drain budgets in the
same mental model as arg parsing, display lifecycle, and consensus
wiring. The helpers here are isolated from the consensus pipeline and
from the live display; they only need an ``asyncio.subprocess.Process``
handle and a destination directory.

Public contract:

* :func:`reap_and_drain_stderr` is the async reap entry point used on
  timeout paths. Non-timeout failures inside ``communicate()`` still
  need separate handling at the call site - see the ``launch_agent``
  docstring in ``run_magi``.
* :func:`windows_kill_tree` runs *before* ``proc.kill()`` on Windows so
  ``taskkill /T`` can walk the parent-child mapping while it is still
  intact. Collapsing the order (``kill()`` first) used to leave child
  trees orphaned - regression R3-1 in ``CLAUDE.md``.
* :func:`write_stderr_log` persists captured stderr to
  ``{agent}.stderr.log`` and is best-effort - callers on an already-
  failing path must wrap it in ``try/except OSError`` so a disk error
  cannot shadow the root-cause exception.
* :func:`format_stderr_excerpt` returns a human-readable tail suitable
  for inclusion in ``TimeoutError`` / ``RuntimeError`` messages.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys

STDERR_EXCERPT_MAX_CHARS = 500
PROC_WAIT_REAP_TIMEOUT = 5.0
PROC_STDERR_DRAIN_TIMEOUT = 2.0
#: ``taskkill /F /T`` gets its own budget so a slow invocation on a busy
#: Windows host cannot consume the whole ``PROC_WAIT_REAP_TIMEOUT`` and
#: leave ``proc.wait()`` with zero headroom. When this separation is
#: collapsed, the "may be orphaned" warning fires even after a successful
#: tree kill, producing misleading diagnostics.
TASKKILL_TIMEOUT = 5.0


def write_stderr_log(output_dir: str, agent_name: str, data: bytes) -> None:
    """Persist captured stderr to ``{agent_name}.stderr.log`` if non-empty.

    Raises:
        OSError: If the destination cannot be opened or written. Callers
            on an already-failing path (e.g. the timeout handler in
            ``launch_agent``) must wrap this call in ``try/except
            OSError`` so a disk error cannot shadow the root-cause
            exception they are about to raise.
    """
    if not data:
        return
    stderr_file = os.path.join(output_dir, f"{agent_name}.stderr.log")
    with open(stderr_file, "wb") as f:
        f.write(data)


def format_stderr_excerpt(data: bytes) -> str:
    """Return a ``: <tail>`` suffix for error messages, empty if no data.

    The excerpt is decoded as UTF-8 with replacement, stripped, and
    truncated to the last :data:`STDERR_EXCERPT_MAX_CHARS` characters so
    diagnostics stay readable in exception strings.
    """
    if not data:
        return ""
    decoded = data.decode("utf-8", errors="replace").strip()
    if len(decoded) > STDERR_EXCERPT_MAX_CHARS:
        decoded = "..." + decoded[-STDERR_EXCERPT_MAX_CHARS:]
    return f": {decoded}"


def windows_kill_tree(pid: int) -> None:
    """Force-terminate a Windows process tree rooted at *pid*.

    ``proc.kill()`` on Windows issues ``TerminateProcess`` against the
    top-level process only, leaving any children the ``claude`` CLI may
    have spawned as orphans under the original parent. ``taskkill /F /T
    /PID`` walks the tree and force-terminates every descendant,
    collapsing the orphan window that used to survive a MAGI timeout.

    Uses its own :data:`TASKKILL_TIMEOUT` (separate from
    :data:`PROC_WAIT_REAP_TIMEOUT`) so a slow invocation on a busy host
    does not consume the caller's wait budget and produce a misleading
    "may be orphaned" warning even when the tree was successfully
    killed.

    Best-effort: if ``taskkill`` is missing from PATH, the spawn fails,
    or the subprocess itself hangs, we return normally. The caller's
    existing ``proc.wait()`` with :data:`PROC_WAIT_REAP_TIMEOUT` still
    fires and emits the "may be orphaned" warning so operators can
    notice.
    """
    try:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=TASKKILL_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        pass


async def reap_and_drain_stderr(proc: asyncio.subprocess.Process) -> bytes:
    """Kill *proc* (and its children on Windows), await exit, drain stderr.

    Both the ``wait()`` and the ``stderr.read()`` are bounded by short
    timeouts so a misbehaving subprocess cannot stall the orchestrator.
    Non-timeout failures are swallowed - the caller is already on an
    error path and only needs best-effort diagnostics. A ``wait()``
    timeout, however, means the reaped subprocess never exited: on
    Windows this typically indicates an orphaned child-process tree,
    so we emit a warning naming the pid so operators can notice it.

    On Windows, ``proc.kill()`` only terminates the top-level process;
    any children the ``claude`` CLI spawned stay alive. We collapse the
    tree first with a best-effort ``taskkill /F /T`` while the parent
    PID is still alive - once ``proc.kill()`` issues
    ``TerminateProcess``, the kernel may have torn down the parent-child
    mapping that ``taskkill /T`` walks, leaving descendants orphaned
    despite the call. ``proc.kill()`` then runs as a fallback so the
    asyncio.subprocess wrapper observes the exit cleanly even when
    ``taskkill`` is missing or times out. Non-Windows platforms send
    ``SIGKILL`` directly to the top process and rely on ``claude`` not
    to fork independent sub-agents there - if it ever did, that would
    need its own platform-specific handling.
    """
    if sys.platform == "win32":
        windows_kill_tree(proc.pid)
    proc.kill()
    try:
        await asyncio.wait_for(proc.wait(), timeout=PROC_WAIT_REAP_TIMEOUT)
    except asyncio.TimeoutError:
        print(
            f"\u26a0 WARNING: subprocess pid={proc.pid} did not exit within "
            f"{PROC_WAIT_REAP_TIMEOUT}s after kill() \u2014 may be orphaned "
            f"(common on Windows with child-process trees)",
            file=sys.stderr,
        )
    except Exception:  # noqa: BLE001 — best-effort reap
        pass

    if proc.stderr is None:
        return b""
    try:
        return await asyncio.wait_for(proc.stderr.read(), timeout=PROC_STDERR_DRAIN_TIMEOUT)
    except Exception:  # noqa: BLE001 — best-effort drain
        return b""
