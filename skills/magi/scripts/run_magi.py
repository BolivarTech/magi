#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 2.1.1
# Date: 2026-04-17
"""MAGI Orchestrator — async Python replacement for run_magi.sh.

Launches Melchior, Balthasar, and Caspar in parallel using asyncio,
collects their JSON outputs, validates them, and runs synthesis.

Usage:
    python run_magi.py <mode> <input> [--model opus] [--timeout 900] [--output-dir <dir>]

Exit codes:
    0 - Success: synthesis completed and report saved.
    1 - Failure: prerequisites missing, or fewer than 2 agents succeeded.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from models import MODEL_IDS, VALID_MODELS, resolve_model
from parse_agent_output import parse_agent_output as parse_raw_output
from status_display import StatusDisplay
from stderr_shim import _buffered_stderr_while
from synthesize import (
    determine_consensus,
    format_report,
    load_agent_output,
)
from validate import MAX_INPUT_FILE_SIZE

# Public star-import contract. Underscore-prefixed symbols from
# ``stderr_shim`` (``_StderrBufferShim``, ``_BinaryStderrBufferShim``,
# ``_buffered_stderr_while``) are intentionally excluded — they are
# private helpers of that module, and tests that need them import from
# ``stderr_shim`` directly. ``_buffered_stderr_while`` is still imported
# here for internal use inside ``run_orchestrator``.
__all__ = [
    "MODEL_IDS",
    "VALID_MODELS",
    "resolve_model",
]

AGENTS = ("melchior", "balthasar", "caspar")
MAX_HISTORY_RUNS = 5
VALID_MODES = ("code-review", "design", "analysis")
MAGI_DIR_PREFIX = "magi-run-"

_STDERR_EXCERPT_MAX_CHARS = 500
_PROC_WAIT_REAP_TIMEOUT = 5.0
_PROC_STDERR_DRAIN_TIMEOUT = 2.0
# ``taskkill /F /T`` gets its own budget so a slow invocation on a busy
# Windows host cannot consume the whole ``_PROC_WAIT_REAP_TIMEOUT`` and
# leave ``proc.wait()`` with zero headroom. When this separation is
# collapsed, the "may be orphaned" warning fires even after a successful
# tree kill, producing misleading diagnostics.
_TASKKILL_TIMEOUT = 5.0


def _write_stderr_log(output_dir: str, agent_name: str, data: bytes) -> None:
    """Persist captured stderr to ``{agent_name}.stderr.log`` if non-empty.

    Raises:
        OSError: If the destination cannot be opened or written. Callers
            on an already-failing path (e.g. the timeout handler in
            :func:`launch_agent`) must wrap this call in ``try/except
            OSError`` so a disk error cannot shadow the root-cause
            exception they are about to raise.
    """
    if not data:
        return
    stderr_file = os.path.join(output_dir, f"{agent_name}.stderr.log")
    with open(stderr_file, "wb") as f:
        f.write(data)


def _format_stderr_excerpt(data: bytes) -> str:
    """Return a ``: <tail>`` suffix for error messages, empty if no data.

    The excerpt is decoded as UTF-8 with replacement, stripped, and
    truncated to the last :data:`_STDERR_EXCERPT_MAX_CHARS` characters so
    diagnostics stay readable in exception strings.
    """
    if not data:
        return ""
    decoded = data.decode("utf-8", errors="replace").strip()
    if len(decoded) > _STDERR_EXCERPT_MAX_CHARS:
        decoded = "..." + decoded[-_STDERR_EXCERPT_MAX_CHARS:]
    return f": {decoded}"


def _windows_kill_tree(pid: int) -> None:
    """Force-terminate a Windows process tree rooted at *pid*.

    ``proc.kill()`` on Windows issues ``TerminateProcess`` against the
    top-level process only, leaving any children the ``claude`` CLI may
    have spawned as orphans under the original parent. ``taskkill /F /T
    /PID`` walks the tree and force-terminates every descendant,
    collapsing the orphan window that used to survive a MAGI timeout.

    Uses its own ``_TASKKILL_TIMEOUT`` (separate from
    ``_PROC_WAIT_REAP_TIMEOUT``) so a slow invocation on a busy host does
    not consume the caller's wait budget and produce a misleading "may
    be orphaned" warning even when the tree was successfully killed.

    Best-effort: if ``taskkill`` is missing from PATH, the spawn fails,
    or the subprocess itself hangs, we return normally. The caller's
    existing ``proc.wait()`` with ``_PROC_WAIT_REAP_TIMEOUT`` still fires
    and emits the "may be orphaned" warning so operators can notice.
    """
    try:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_TASKKILL_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        pass


async def _reap_and_drain_stderr(proc: asyncio.subprocess.Process) -> bytes:
    """Kill *proc* (and its children on Windows), await exit, drain stderr.

    Both the ``wait()`` and the ``stderr.read()`` are bounded by short
    timeouts so a misbehaving subprocess cannot stall the orchestrator.
    Non-timeout failures are swallowed — the caller is already on an
    error path and only needs best-effort diagnostics. A ``wait()``
    timeout, however, means the reaped subprocess never exited: on
    Windows this typically indicates an orphaned child-process tree,
    so we emit a warning naming the pid so operators can notice it.

    On Windows, ``proc.kill()`` only terminates the top-level process;
    any children the ``claude`` CLI spawned stay alive. We follow the
    kill with a best-effort ``taskkill /F /T`` to collapse the tree.
    Non-Windows platforms send ``SIGKILL`` directly to the top process
    and rely on ``claude`` not to fork independent sub-agents there —
    if it ever did, that would need its own platform-specific handling.
    """
    proc.kill()
    if sys.platform == "win32":
        _windows_kill_tree(proc.pid)
    try:
        await asyncio.wait_for(proc.wait(), timeout=_PROC_WAIT_REAP_TIMEOUT)
    except asyncio.TimeoutError:
        print(
            f"\u26a0 WARNING: subprocess pid={proc.pid} did not exit within "
            f"{_PROC_WAIT_REAP_TIMEOUT}s after kill() \u2014 may be orphaned "
            f"(common on Windows with child-process trees)",
            file=sys.stderr,
        )
    except Exception:  # noqa: BLE001 — best-effort reap
        pass

    if proc.stderr is None:
        return b""
    try:
        return await asyncio.wait_for(proc.stderr.read(), timeout=_PROC_STDERR_DRAIN_TIMEOUT)
    except Exception:  # noqa: BLE001 — best-effort drain
        return b""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed namespace with mode, input, timeout, output_dir.
    """
    parser = argparse.ArgumentParser(description="MAGI Orchestrator")
    parser.add_argument("mode", choices=VALID_MODES, help="Analysis mode")
    parser.add_argument("input", help="Path to file or inline text to analyze")
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Per-agent timeout in seconds (default: 900)",
    )
    parser.add_argument("--output-dir", help="Directory for agent outputs")
    parser.add_argument(
        "--model",
        choices=VALID_MODELS,
        default="opus",
        help="LLM model for all agents (default: opus)",
    )
    parser.add_argument(
        "--keep-runs",
        type=int,
        default=MAX_HISTORY_RUNS,
        help=(
            f"Final on-disk count of magi-run-* temp dirs, including the run "
            f"about to be created (default: {MAX_HISTORY_RUNS}). "
            f"``--keep-runs 1`` keeps only the current run and wipes all "
            f"prior ones. ``--keep-runs 0`` is rejected. ``--keep-runs -1`` "
            f"disables cleanup entirely."
        ),
    )
    parser.add_argument(
        "--no-status",
        dest="show_status",
        action="store_false",
        help="Disable the live status tree display",
    )
    parser.set_defaults(show_status=True)
    args = parser.parse_args(argv)
    # ``--keep-runs 0`` is ambiguous: a naive reading is "keep nothing"
    # (wipe), but the legacy contract for ``cleanup_old_runs(keep)`` treats
    # a negative result as "disabled". Rather than bake a surprise into the
    # CLI, we reject 0 explicitly so operators pick the side they mean:
    # ``--keep-runs 1`` to wipe everything except the current run, or
    # ``--keep-runs -1`` to disable cleanup entirely.
    if args.keep_runs == 0:
        parser.error(
            "--keep-runs 0 is ambiguous: use --keep-runs 1 to wipe all prior "
            "runs (keeping only the one about to be created), or --keep-runs "
            "-1 to disable cleanup entirely."
        )
    return args


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
    when the temp root itself is a symlink (e.g. ``/tmp`` →
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
    systems. Failures are logged to stderr — cleanup must never raise.
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
    for deterministic LRU under mtime ties, by path ascending — the
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

    # Fast path: nothing to prune — skip the sort and the per-entry loop.
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

    Uses tempfile.mkdtemp for cross-platform compatibility (fixes W2).

    Args:
        output_dir: Explicit path, or None to create a temp dir.

    Returns:
        Path to the created output directory.
    """
    if output_dir is None:
        return tempfile.mkdtemp(prefix=MAGI_DIR_PREFIX)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


async def launch_agent(
    agent_name: str,
    agents_dir: str,
    prompt: str,
    output_dir: str,
    timeout: int,
    model: str = "opus",
) -> dict[str, Any]:
    """Launch a single agent subprocess and return validated output.

    Runs ``claude -p`` with the agent's system prompt, applies timeout,
    parses the raw output, and validates against the agent JSON schema.
    The user prompt is sent via stdin to avoid OS CLI argument length
    limits.  A copy is also saved to ``{agent_name}.prompt.txt`` in
    *output_dir* as a debug artifact.

    Args:
        agent_name: One of 'melchior', 'balthasar', 'caspar'.
        agents_dir: Directory containing agent prompt .md files.
        prompt: The prompt payload to send to the agent.
        output_dir: Directory for raw and parsed output files.
        timeout: Timeout in seconds per agent.
        model: Model short name ('opus', 'sonnet', 'haiku').

    Returns:
        Validated agent output dictionary.

    Raises:
        TimeoutError: If the agent does not respond within timeout. On this
            path the subprocess is killed and reaped (``wait()``) and any
            buffered stderr is persisted to ``{agent_name}.stderr.log`` and
            included in the error message for post-mortem diagnosis.
        RuntimeError: If the subprocess exits with a non-zero code.
        ValidationError: If the agent output fails schema validation.
        ValueError: If *model* is not a recognised short name.
    """
    model_id = resolve_model(model)

    system_prompt_file = os.path.join(agents_dir, f"{agent_name}.md")
    raw_file = os.path.join(output_dir, f"{agent_name}.raw.json")
    parsed_file = os.path.join(output_dir, f"{agent_name}.json")

    # Write user prompt to a temp file and pass via stdin to avoid
    # OS CLI argument length limits (~32K on Windows).
    prompt_file = os.path.join(output_dir, f"{agent_name}.prompt.txt")
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt)

    proc = await asyncio.create_subprocess_exec(
        "claude",
        "-p",
        "--output-format",
        "json",
        "--model",
        model_id,
        "--system-prompt-file",
        system_prompt_file,
        "-",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=prompt.encode("utf-8")), timeout=timeout
        )
    except asyncio.TimeoutError:
        stderr_buffered = await _reap_and_drain_stderr(proc)
        # Persisting the log is best-effort. If it fails (disk full,
        # permission denied), surface a warning but do not let the
        # OSError shadow the TimeoutError the caller actually needs.
        try:
            _write_stderr_log(output_dir, agent_name, stderr_buffered)
        except OSError as log_exc:
            print(
                f"WARNING: Failed to persist {agent_name}.stderr.log on timeout: {log_exc}",
                file=sys.stderr,
            )
        raise TimeoutError(
            f"Agent '{agent_name}' timed out after {timeout}s"
            f"{_format_stderr_excerpt(stderr_buffered)}"
        ) from None

    with open(raw_file, "wb") as f:
        f.write(stdout)

    # The stderr log is a diagnostic artefact, not load-bearing. A disk
    # error here (disk full, permission drop, antivirus lock on Windows)
    # must not turn an otherwise-successful agent into a reported
    # failure. Mirror the timeout-path pattern: warn and continue.
    try:
        _write_stderr_log(output_dir, agent_name, stderr)
    except OSError as log_exc:
        print(
            f"WARNING: Failed to persist {agent_name}.stderr.log: {log_exc}",
            file=sys.stderr,
        )

    if proc.returncode != 0:
        stderr_text = stderr.decode("utf-8", errors="replace").strip() if stderr else "no stderr"
        raise RuntimeError(
            f"Agent '{agent_name}' exited with code {proc.returncode}: {stderr_text}"
        )

    parse_raw_output(raw_file, parsed_file)
    return load_agent_output(parsed_file)


class _DisplayLogGate:
    """Once-per-run gate that logs the first display-update failure.

    Owns the "has the first failure already been logged" flag that used
    to live as module-level mutable state. A fresh instance is created
    by :func:`run_orchestrator` for every run, so there is no residual
    state across runs and no ``global`` plumbing for tests to reset.
    Each :func:`_safe_display_update` call is threaded through the gate
    belonging to the enclosing orchestrator invocation.
    """

    __slots__ = ("_logged",)

    def __init__(self) -> None:
        self._logged: bool = False

    def emit_once(self, exc: BaseException) -> None:
        """Log *exc* to stderr exactly once for this gate's lifetime.

        Subsequent calls are no-ops. The helper must never propagate a
        new exception — doing so would mask the original shutdown signal
        the caller is already re-raising. Failures inside the ``print``
        itself (stream closed, etc.) are swallowed silently for the same
        reason.
        """
        if self._logged:
            return
        self._logged = True
        try:
            print(
                f"\u26a0 WARNING: status display update failed ({exc!r}) "
                f"\u2014 live tree may be stale for the rest of this run",
                file=sys.stderr,
            )
        except BaseException:  # noqa: BLE001 — never let logging shadow shutdown
            pass


def _safe_display_update(
    display: StatusDisplay | None,
    name: str,
    state: str,
    log_gate: _DisplayLogGate,
) -> None:
    """Update a status display, swallowing any exception on failure.

    During shutdown paths (``KeyboardInterrupt``, ``CancelledError``, event
    loop closing) the display's underlying stream may already be closed or
    in a broken state. In that case a ``display.update`` call can raise,
    and propagating that new exception would mask the original shutdown
    signal. This helper isolates the display update so that the caller's
    ``raise`` statement always preserves the real cause.

    The first exception per run is logged to stderr through *log_gate* so
    the operator knows the live tree is blind; subsequent exceptions stay
    silent to prevent the redraw path from flooding the log on every tick.

    Args:
        display: The status display, or ``None`` to skip the update.
        name: Agent name to update.
        state: New state for the agent row.
        log_gate: Run-scoped gate that enforces the once-per-run log rule.
    """
    if display is None:
        return
    try:
        display.update(name, state)
    except BaseException as exc:  # noqa: BLE001 — see docstring shutdown-path contract
        # Catches ``Exception`` subclasses plus ``CancelledError``,
        # ``KeyboardInterrupt``, and ``SystemExit``. The helper is invoked
        # from ``tracked_launch``'s ``except BaseException`` clause which
        # then re-raises the *original* signal — if we let the display's
        # own BaseException escape here, that outer ``raise`` never runs
        # and the real shutdown reason is lost.
        log_gate.emit_once(exc)


async def run_orchestrator(
    agents_dir: str,
    prompt: str,
    output_dir: str,
    timeout: int,
    model: str = "opus",
    *,
    show_status: bool = True,
) -> dict[str, Any]:
    """Run all three agents concurrently and synthesize results.

    Launches agents in parallel, collects results, alerts on failures,
    and runs consensus synthesis on successful outputs.

    Args:
        agents_dir: Directory containing agent prompt files.
        prompt: The prompt payload.
        output_dir: Directory for output files.
        timeout: Per-agent timeout in seconds.
        model: Model short name ('opus', 'sonnet', 'haiku').
        show_status: Render a live status tree while agents run. When the
            stream is not a TTY, plain one-line-per-event output is emitted
            instead.

    Returns:
        Report dict with 'agents', 'consensus', and optionally
        'degraded' and 'failed_agents' when < 3 agents succeed.

    Raises:
        RuntimeError: If fewer than 2 agents succeed.
    """
    successful: list[dict[str, Any]] = []
    failed: list[str] = []

    # Fresh log gate per run so the first display failure is always
    # surfaced, even in hosts that reuse the module across orchestrator
    # invocations (tests, long-lived services).
    log_gate = _DisplayLogGate()

    # Display lifecycle invariant (structurally enforced by the
    # ``_buffered_stderr_while`` context manager below): while the status
    # display is rendering, ``sys.stderr`` is replaced with a write-buffer, so
    # any diagnostic print that would otherwise collide with the in-place
    # redraw is deferred until after ``display.stop()`` returns.
    #
    # The display itself captures the *real* ``sys.stderr`` reference at
    # construction time (below), so its own writes go straight to the
    # terminal, not through the buffer.
    display: StatusDisplay | None = (
        StatusDisplay(list(AGENTS), stream=sys.stderr) if show_status else None
    )

    async def tracked_launch(name: str) -> dict[str, Any]:
        _safe_display_update(display, name, "running", log_gate)
        try:
            result = await launch_agent(name, agents_dir, prompt, output_dir, timeout, model)
        except (asyncio.TimeoutError, TimeoutError):
            _safe_display_update(display, name, "timeout", log_gate)
            raise
        except BaseException:
            # Catches asyncio.CancelledError (which is BaseException in 3.8+),
            # generic Exception subclasses, KeyboardInterrupt, and SystemExit.
            # We always re-raise — the display update is a best-effort side
            # effect (see ``_safe_display_update``) so a stream already closed
            # during shutdown can never mask the real shutdown signal.
            _safe_display_update(display, name, "failed", log_gate)
            raise
        _safe_display_update(display, name, "success", log_gate)
        return result

    tasks = {name: tracked_launch(name) for name in AGENTS}

    if display is not None:
        try:
            await display.start()
        except Exception as exc:
            # A display-start failure (event-loop issue, terminal problem) must
            # never block the actual analysis. Drop the display and fall
            # through — tracked_launch closures will see ``display is None``.
            print(
                f"\u26a0 WARNING: status display failed to start ({exc}) "
                f"\u2014 continuing without live status",
                file=sys.stderr,
            )
            display = None

    with _buffered_stderr_while(active=display is not None):
        try:
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        finally:
            if display is not None:
                await display.stop()

    for name, result in zip(tasks.keys(), results):
        if isinstance(result, BaseException):
            # CancelledError is BaseException in 3.8+ but we treat a cancelled
            # child task as a normal agent failure — the orchestrator itself is
            # not being cancelled, only one sub-agent was. Truly fatal signals
            # (KeyboardInterrupt, SystemExit) still propagate.
            if not isinstance(result, (Exception, asyncio.CancelledError)):
                raise result
            print(
                f"\u26a0 WARNING: Agent '{name}' failed ({result}) \u2014 excluded from synthesis",
                file=sys.stderr,
            )
            failed.append(name)
        else:
            successful.append(result)

    if len(successful) < 2:
        raise RuntimeError(
            f"Only {len(successful)} agent(s) succeeded \u2014 fewer than 2 required for synthesis"
        )

    if failed:
        print(
            f"\u26a0 WARNING: Running synthesis with "
            f"{len(successful)}/{len(AGENTS)} agents "
            f"\u2014 results may be biased",
            file=sys.stderr,
        )

    consensus = determine_consensus(successful)

    report: dict[str, Any] = {
        "agents": successful,
        "consensus": consensus,
    }

    if failed:
        report["degraded"] = True
        report["failed_agents"] = failed

    return report


def main() -> None:
    """CLI entry point for MAGI orchestrator."""
    args = parse_args()

    if os.path.isfile(args.input):
        file_size = os.path.getsize(args.input)
        if file_size > MAX_INPUT_FILE_SIZE:
            print(
                f"ERROR: Input file {args.input} is {file_size} bytes, "
                f"exceeding maximum of {MAX_INPUT_FILE_SIZE} bytes.",
                file=sys.stderr,
            )
            sys.exit(1)
        with open(args.input, encoding="utf-8") as f:
            input_content = f.read()
        input_label = f"File: {args.input}"
    else:
        input_content = args.input
        input_label = "Inline input"

    prompt = f"MODE: {args.mode}\nCONTEXT ({input_label}):\n\n{input_content}"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    skill_dir = os.path.dirname(script_dir)
    agents_dir = os.path.join(skill_dir, "agents")

    # Hard prerequisite check runs **before** any filesystem setup so a
    # missing CLI cannot leak a half-initialised temp directory on disk.
    if not shutil.which("claude"):
        print("ERROR: 'claude' CLI not found in PATH", file=sys.stderr)
        sys.exit(1)

    is_temp_dir = args.output_dir is None
    if is_temp_dir:
        # Prune to ``keep_runs - 1`` existing dirs so the run about to be
        # created below brings the total to exactly ``keep_runs``. Without
        # the ``- 1`` the final count is always ``keep_runs + 1``.
        cleanup_old_runs(args.keep_runs - 1)
    output_dir = create_output_dir(args.output_dir)

    print("+==================================================+")
    print("|          MAGI SYSTEM -- INITIALIZING              |")
    print("+==================================================+")
    print(f"|  Mode: {args.mode}")
    print(f"|  Model: {args.model} ({MODEL_IDS[args.model]})")
    print(f"|  Timeout: {args.timeout}s")
    print(f"|  Output: {output_dir}")
    print("+==================================================+")
    print(flush=True)

    # ``BaseException`` rather than ``Exception`` so KeyboardInterrupt and
    # SystemExit also trigger the temp-dir cleanup — otherwise Ctrl-C mid
    # run leaves orphaned ``magi-run-*`` dirs that ``cleanup_old_runs``
    # only prunes opportunistically on the *next* run.
    report: dict[str, Any] | None = None
    try:
        report = asyncio.run(
            run_orchestrator(
                agents_dir,
                prompt,
                output_dir,
                args.timeout,
                args.model,
                show_status=args.show_status,
            )
        )
    except BaseException:
        if is_temp_dir:
            try:
                shutil.rmtree(output_dir)
            except OSError as cleanup_exc:
                print(
                    f"WARNING: Failed to clean up {output_dir}: {cleanup_exc}",
                    file=sys.stderr,
                )
        raise

    print(format_report(report["agents"], report["consensus"]))

    report_path = os.path.join(output_dir, "magi-report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    main()
