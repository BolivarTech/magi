#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-04-17
"""Temp-directory housekeeping for the MAGI orchestrator.

Extracted from ``run_magi.py`` so the orchestrator file no longer has to
hold the LRU + symlink-traversal + mtime-tie-break rules in the same
mental model as subprocess and display concerns. The helpers here are
pure filesystem manipulation — no asyncio, no subprocesses, no CLI —
and can be unit-tested independently of the orchestrator wiring.

Public contract:

* :data:`MAGI_DIR_PREFIX` is the single source of truth for the
  ``magi-run-*`` directory naming convention; both cleanup and creation
  must agree on it.
* :func:`cleanup_old_runs` is the LRU entry point. It is deliberately
  total (never raises on scan/stat errors, only on programmer errors)
  so the orchestrator can call it unconditionally.
* :func:`create_output_dir` is the counterpart that honors the same
  prefix when generating a temp dir.

``_scan_magi_dirs``, ``_safe_temp_prefix`` and ``_safe_rmtree_under``
are internal helpers exposed with leading underscores — they are only
module-public for the regression suite that drills into specific TOCTOU
and mtime tie-break behaviors.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

MAGI_DIR_PREFIX = "magi-run-"


def _scan_magi_dirs(tmp_root: str) -> list[tuple[float, str]]:
    """Return ``(mtime, path)`` tuples for every ``magi-run-*`` dir under *tmp_root*.

    Entries that disappear between scan and stat are silently skipped.
    """
    results: list[tuple[float, str]] = []
    for entry in os.scandir(tmp_root):
        if not (entry.is_dir() and entry.name.startswith(MAGI_DIR_PREFIX)):
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        results.append((mtime, entry.path))
    return results


def _safe_temp_prefix(tmp_root: str) -> str:
    """Return the normalized temp-root prefix used for traversal checks.

    Resolves symlinks in *tmp_root* before building the prefix so that
    ``os.path.realpath(entry.path).startswith(prefix)`` stays consistent
    when the temp root itself is a symlink (e.g. ``/tmp`` ->
    ``/private/tmp`` on macOS). Without this, every scanned entry
    resolves outside the advertised prefix and cleanup becomes a
    silent no-op.
    """
    prefix = os.path.normcase(os.path.realpath(tmp_root))
    if not prefix.endswith(os.sep):
        prefix += os.sep
    return prefix


def _safe_rmtree_under(path: str, safe_prefix: str) -> None:
    """Remove *path* only if it resolves strictly inside *safe_prefix*.

    The realpath check prevents symlink traversal attacks on shared
    systems. Failures are logged to stderr - cleanup must never raise.
    """
    resolved = os.path.normcase(os.path.realpath(path))
    if not resolved.startswith(safe_prefix):
        print(
            f"WARNING: Skipping cleanup of {path} (resolves outside temp root: {resolved})",
            file=sys.stderr,
        )
        return
    try:
        shutil.rmtree(resolved)
    except OSError as exc:
        print(
            f"WARNING: Failed to remove old run {resolved}: {exc}",
            file=sys.stderr,
        )


def cleanup_old_runs(keep: int) -> None:
    """Remove oldest MAGI temp directories, keeping the most recent ones.

    Scans the system temp directory for directories matching the
    :data:`MAGI_DIR_PREFIX` and removes the oldest so that at most
    ``keep`` remain. Entries are sorted by ``st_mtime`` descending and,
    for deterministic LRU under mtime ties, by path ascending - the
    lexicographically smallest path is treated as the canonical
    survivor. Symlinks are resolved and validated against the temp root
    before deletion to prevent traversal attacks on shared systems.

    Intended to be called **before** the current run's temp dir is
    created, so the caller should pass ``keep_runs - 1`` when they want
    a final on-disk count of ``keep_runs`` after :func:`create_output_dir`
    adds the new dir. Without the off-by-one adjustment the final count
    is always ``keep_runs + 1``.

    Args:
        keep: Maximum number of existing runs to retain.
            ``keep >= 0``: valid; ``keep == 0`` removes every matching
            dir (the caller is reserving the only slot for the run it
            is about to create). ``keep < 0`` disables cleanup entirely.
    """
    if keep < 0:
        return

    tmp_root = tempfile.gettempdir()
    magi_dirs = _scan_magi_dirs(tmp_root)

    # Fast path: nothing to prune - skip the sort and the per-entry loop.
    # Never triggered when keep == 0 and at least one dir exists, so the
    # "wipe everything" case falls through to the slice below.
    if len(magi_dirs) <= keep:
        return

    # Explicit key so the tie-breaking direction is documented and cannot
    # drift if someone later replaces the list of tuples with a different
    # container.
    magi_dirs.sort(key=lambda entry: (-entry[0], entry[1]))

    safe_prefix = _safe_temp_prefix(tmp_root)
    for _, path in magi_dirs[keep:]:
        _safe_rmtree_under(path, safe_prefix)


def create_output_dir(output_dir: str | None) -> str:
    """Create and return the output directory.

    Uses ``tempfile.mkdtemp`` for cross-platform compatibility.

    Args:
        output_dir: Explicit path, or None to create a temp dir.

    Returns:
        Path to the created output directory.
    """
    if output_dir is None:
        return tempfile.mkdtemp(prefix=MAGI_DIR_PREFIX)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
