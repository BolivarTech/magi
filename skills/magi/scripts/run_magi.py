#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-04-01
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
import contextlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from typing import Any

from parse_agent_output import parse_agent_output as parse_raw_output
from status_display import StatusDisplay
from synthesize import (
    determine_consensus,
    format_report,
    load_agent_output,
)

AGENTS = ("melchior", "balthasar", "caspar")
MAX_HISTORY_RUNS = 5
VALID_MODES = ("code-review", "design", "analysis")
MODEL_IDS: dict[str, str] = {
    "opus": "claude-opus-4-6",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
}
VALID_MODELS = tuple(MODEL_IDS.keys())


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
        help=f"Number of recent temp runs to keep (default: {MAX_HISTORY_RUNS})",
    )
    parser.add_argument(
        "--no-status",
        dest="show_status",
        action="store_false",
        help="Disable the live status tree display",
    )
    parser.set_defaults(show_status=True)
    return parser.parse_args(argv)


def cleanup_old_runs(keep: int) -> None:
    """Remove oldest MAGI temp directories, keeping the most recent ones.

    Scans the system temp directory for directories matching the
    ``magi-run-`` prefix, sorted by modification time, and removes the
    oldest ones so that at most ``keep`` remain.  Symlinks are resolved
    and validated against the temp root before deletion to prevent
    traversal attacks on shared systems.

    Args:
        keep: Maximum number of recent runs to retain.
            A value <= 0 disables cleanup.
    """
    if keep <= 0:
        return
    tmp_root = tempfile.gettempdir()
    magi_dirs: list[tuple[float, str]] = []
    for entry in os.scandir(tmp_root):
        if entry.is_dir() and entry.name.startswith("magi-run-"):
            try:
                mtime = entry.stat().st_mtime
            except OSError:
                continue
            magi_dirs.append((mtime, entry.path))

    magi_dirs.sort(reverse=True)
    for _, path in magi_dirs[keep:]:
        resolved = os.path.normcase(os.path.realpath(path))
        safe_prefix = os.path.normcase(tmp_root)
        if not safe_prefix.endswith(os.sep):
            safe_prefix += os.sep
        if not resolved.startswith(safe_prefix):
            print(
                f"WARNING: Skipping cleanup of {path} (resolves outside temp root: {resolved})",
                file=sys.stderr,
            )
            continue
        try:
            shutil.rmtree(resolved)
        except OSError as exc:
            print(
                f"WARNING: Failed to remove old run {resolved}: {exc}",
                file=sys.stderr,
            )


def create_output_dir(output_dir: str | None) -> str:
    """Create and return the output directory.

    Uses tempfile.mkdtemp for cross-platform compatibility (fixes W2).

    Args:
        output_dir: Explicit path, or None to create a temp dir.

    Returns:
        Path to the created output directory.
    """
    if output_dir is None:
        return tempfile.mkdtemp(prefix="magi-run-")
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
        TimeoutError: If the agent does not respond within timeout.
        RuntimeError: If the subprocess exits with a non-zero code.
        ValidationError: If the agent output fails schema validation.
        FileNotFoundError: If the agent prompt file is missing.
    """
    if model not in MODEL_IDS:
        raise ValueError(f"Unknown model '{model}'. Must be one of {sorted(MODEL_IDS.keys())}.")

    system_prompt_file = os.path.join(agents_dir, f"{agent_name}.md")
    raw_file = os.path.join(output_dir, f"{agent_name}.raw.json")
    parsed_file = os.path.join(output_dir, f"{agent_name}.json")
    model_id = MODEL_IDS[model]

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
        proc.kill()
        raise TimeoutError(f"Agent '{agent_name}' timed out after {timeout}s") from None

    with open(raw_file, "wb") as f:
        f.write(stdout)

    if stderr:
        stderr_file = os.path.join(output_dir, f"{agent_name}.stderr.log")
        with open(stderr_file, "wb") as f:
            f.write(stderr)

    if proc.returncode != 0:
        stderr_text = stderr.decode("utf-8", errors="replace").strip() if stderr else "no stderr"
        raise RuntimeError(
            f"Agent '{agent_name}' exited with code {proc.returncode}: {stderr_text}"
        )

    parse_raw_output(raw_file, parsed_file)
    return load_agent_output(parsed_file)


class _BinaryStderrBufferShim:
    """Binary write-buffering proxy for ``sys.stderr.buffer``.

    Callers that do ``sys.stderr.buffer.write(b"...")`` would otherwise
    bypass the text-mode :class:`_StderrBufferShim` and write directly
    to the real stream, breaking the display-active-stderr-quiet
    invariant. This shim catches binary writes, decodes them into the
    shared text buffer (UTF-8 with replacement for any undecodable
    bytes), and proxies every non-write attribute to the real binary
    buffer so operations like ``fileno`` continue to work.
    """

    def __init__(self, text_buffer: list[str], real_binary: Any) -> None:
        self._text_buffer = text_buffer
        self._real = real_binary

    def write(self, data: bytes) -> int:
        self._text_buffer.append(data.decode("utf-8", errors="replace"))
        return len(data)

    def flush(self) -> None:  # buffered until context exit
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _StderrBufferShim:
    """Write-buffering proxy for :data:`sys.stderr`.

    Forwards ``write`` / ``flush`` to an internal list and proxies every
    other attribute (``encoding``, ``fileno``, ``isatty``, ...) to the
    original stderr. Exposes a ``.buffer`` attribute backed by
    :class:`_BinaryStderrBufferShim` so binary writes through
    ``sys.stderr.buffer`` are also intercepted.

    Used by :func:`_buffered_stderr_while` to structurally enforce the
    display-active-stderr-quiet invariant.

    **Uncovered paths (documented in ``CLAUDE.md`` Known limitations):**
    - ``os.write(sys.stderr.fileno(), ...)`` bypasses Python-level proxies
      and writes directly to fd 2.
    - Hard process death (``SIGKILL``, segfault) skips the buffer flush
      in ``_buffered_stderr_while``'s ``finally`` clause.
    """

    def __init__(self, real_stderr: Any, buffer: list[str]) -> None:
        self._real = real_stderr
        self._buffer = buffer
        # Expose a binary shim when the real stream has a binary buffer.
        # ``io.StringIO`` and pytest's capture streams may not, so fall
        # back to proxying via ``__getattr__`` when absent.
        real_binary = getattr(real_stderr, "buffer", None)
        self.buffer: _BinaryStderrBufferShim | None = (
            _BinaryStderrBufferShim(buffer, real_binary) if real_binary is not None else None
        )

    def write(self, data: str) -> int:
        self._buffer.append(data)
        return len(data)

    def flush(self) -> None:
        """Intentional no-op: content stays buffered until context exit.

        The ``_buffered_stderr_while`` context manager replays the full
        buffer to the real stderr on ``__exit__``; flushing the shim
        mid-context would defeat the display-active invariant.
        """
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def _safe_display_update(display: StatusDisplay | None, name: str, state: str) -> None:
    """Update a status display, swallowing any exception on failure.

    During shutdown paths (``KeyboardInterrupt``, ``CancelledError``, event
    loop closing) the display's underlying stream may already be closed or
    in a broken state. In that case a ``display.update`` call can raise,
    and propagating that new exception would mask the original shutdown
    signal. This helper isolates the display update so that the caller's
    ``raise`` statement always preserves the real cause.

    Args:
        display: The status display, or ``None`` to skip the update.
        name: Agent name to update.
        state: New state for the agent row.
    """
    if display is None:
        return
    try:
        display.update(name, state)
    except Exception:  # noqa: BLE001 — best-effort update during shutdown
        pass


@contextlib.contextmanager
def _buffered_stderr_while(active: bool) -> Iterator[None]:
    """Buffer ``sys.stderr`` writes while ``active`` is True.

    When the status display is rendering live to ``sys.stderr``, any
    concurrent diagnostic write lands inside the in-place redraw region
    and is wiped on the next refresh tick. This context manager
    structurally enforces the display-active-stderr-quiet invariant:
    while the body runs, ``sys.stderr`` is replaced with a write-only
    buffer, and on exit the buffered content is replayed to the real
    stderr after the display has been stopped.

    When ``active`` is False, this is a no-op.

    Args:
        active: True when a live status display is running against
            ``sys.stderr``.

    Yields:
        Control to the caller.
    """
    if not active:
        yield
        return

    saved = sys.stderr
    buffer: list[str] = []
    sys.stderr = _StderrBufferShim(saved, buffer)
    try:
        yield
    finally:
        sys.stderr = saved
        if buffer:
            saved.write("".join(buffer))
            saved.flush()


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
        _safe_display_update(display, name, "running")
        try:
            result = await launch_agent(name, agents_dir, prompt, output_dir, timeout, model)
        except (asyncio.TimeoutError, TimeoutError):
            _safe_display_update(display, name, "timeout")
            raise
        except BaseException:
            # Catches asyncio.CancelledError (which is BaseException in 3.8+),
            # generic Exception subclasses, KeyboardInterrupt, and SystemExit.
            # We always re-raise — the display update is a best-effort side
            # effect (see ``_safe_display_update``) so a stream already closed
            # during shutdown can never mask the real shutdown signal.
            _safe_display_update(display, name, "failed")
            raise
        _safe_display_update(display, name, "success")
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

    _MAX_INPUT_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    if os.path.isfile(args.input):
        file_size = os.path.getsize(args.input)
        if file_size > _MAX_INPUT_FILE_SIZE:
            print(
                f"ERROR: Input file {args.input} is {file_size} bytes, "
                f"exceeding maximum of {_MAX_INPUT_FILE_SIZE} bytes.",
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
    is_temp_dir = args.output_dir is None
    if is_temp_dir:
        cleanup_old_runs(args.keep_runs)
    output_dir = create_output_dir(args.output_dir)

    if not shutil.which("claude"):
        print("ERROR: 'claude' CLI not found in PATH", file=sys.stderr)
        sys.exit(1)

    print("+==================================================+")
    print("|          MAGI SYSTEM -- INITIALIZING              |")
    print("+==================================================+")
    print(f"|  Mode: {args.mode}")
    print(f"|  Model: {args.model} ({MODEL_IDS[args.model]})")
    print(f"|  Timeout: {args.timeout}s")
    print(f"|  Output: {output_dir}")
    print("+==================================================+")
    print(flush=True)

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
    except Exception:
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
