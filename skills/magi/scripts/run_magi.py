#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 2.2.6
# Date: 2026-04-27
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
import sys
from typing import Any

from models import MODE_DEFAULT_MODELS, MODEL_IDS, VALID_MODELS, resolve_model
from parse_agent_output import parse_agent_output as parse_raw_output
from status_display import StatusDisplay
from stderr_shim import _buffered_stderr_while
from synthesize import (
    determine_consensus,
    format_report,
    load_agent_output,
)
from subprocess_utils import (
    format_stderr_excerpt as _format_stderr_excerpt,
    reap_and_drain_stderr as _reap_and_drain_stderr,
    write_stderr_log as _write_stderr_log,
)
from temp_dirs import (
    MAGI_DIR_PREFIX,
    cleanup_old_runs,
    create_output_dir,
)
from validate import MAX_INPUT_FILE_SIZE, ValidationError

# Public star-import contract. Underscore-prefixed symbols from
# ``stderr_shim`` (``_StderrBufferShim``, ``_BinaryStderrBufferShim``,
# ``_buffered_stderr_while``) are intentionally excluded — they are
# private helpers of that module, and tests that need them import from
# ``stderr_shim`` directly. ``_buffered_stderr_while`` is still imported
# here for internal use inside ``run_orchestrator``.
#
# The ``temp_dirs`` symbols (``cleanup_old_runs``, ``create_output_dir``,
# ``MAGI_DIR_PREFIX`` and the underscore-prefixed traversal helpers) are
# re-exported from here so the longstanding ``from run_magi import
# cleanup_old_runs`` pattern in callers and tests continues to work after
# the 2.1.3 split. Future code should import from ``temp_dirs`` directly.
__all__ = [
    "MAGI_DIR_PREFIX",
    "MODEL_IDS",
    "VALID_MODELS",
    "cleanup_old_runs",
    "create_output_dir",
    "resolve_model",
]

AGENTS = ("melchior", "balthasar", "caspar")
MAX_HISTORY_RUNS = 5
VALID_MODES = ("code-review", "design", "analysis")


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
        default=None,
        help=(
            "LLM model for all agents. When omitted, the default depends "
            "on the mode: opus for code-review and design, sonnet for "
            "analysis. Pass --model explicitly to override."
        ),
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
    # Per-mode default model resolution (2.2.3). ``argparse`` cannot express
    # "default depends on another arg" cleanly, so we resolve here. The mode
    # has already been validated by ``choices=VALID_MODES`` above, so the
    # ``MODE_DEFAULT_MODELS`` lookup is total — no KeyError path is reachable
    # while VALID_MODES and MODE_DEFAULT_MODELS stay in lockstep (a guarantee
    # the test suite pins).
    if args.model is None:
        args.model = MODE_DEFAULT_MODELS[args.mode]
    return args


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
        ValidationError: If the agent output fails schema validation. Caught
            and retried by ``run_orchestrator.tracked_launch`` (2.2.0).
        json.JSONDecodeError: If the parsed text is not valid JSON. Raised
            by ``parse_agent_output``, propagated through ``launch_agent``,
            and caught + retried by ``run_orchestrator.tracked_launch``
            (2.2.4).
        ValueError: From ``resolve_model`` for unknown model short names,
            from ``parse_agent_output`` for unrecognised CLI output shapes,
            or when the agent's raw stdout (``{agent_name}.raw.json``)
            exceeds :data:`validate.MAX_INPUT_FILE_SIZE`. NOT retried —
            these are configuration / structural failures that a re-roll
            cannot fix.
        asyncio.CancelledError: If the orchestrating task is cancelled
            while ``launch_agent`` is awaiting the subprocess. Propagated
            unchanged so the cancel reaches the surrounding
            ``asyncio.gather`` in ``run_orchestrator``; ``tracked_launch``
            treats this as a non-retryable failure (the run as a whole is
            shutting down).
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
                f"[!] WARNING: status display update failed ({exc!r}) "
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


def _build_retry_prompt(original_prompt: str, error: ValidationError | json.JSONDecodeError) -> str:
    """Return the retry prompt with corrective feedback appended.

    When :func:`launch_agent` raises :class:`ValidationError` (schema
    fail) or :class:`json.JSONDecodeError` (output is not parseable JSON)
    on the first attempt, :func:`run_orchestrator` calls this helper to
    build the replacement prompt for the single retry. The original
    user prompt is preserved verbatim so the agent's task is unchanged;
    the parser/validator error message is appended so the model can
    self-correct the specific defect — a missing key, a stray comma, a
    truncated output, an unbalanced brace, etc. The envelope delimiter
    ``---RETRY-FEEDBACK---`` is intentionally distinct from user input
    so the model can identify the corrective block even if the original
    prompt already contains arbitrary markdown.

    Args:
        original_prompt: The exact prompt sent on the first attempt.
        error: The exception that triggered the retry. Currently either
            :class:`ValidationError` (schema mismatch) or
            :class:`json.JSONDecodeError` (output not parseable as JSON).

    Returns:
        A new prompt string that concatenates the original prompt with a
        feedback block describing the failure and restating the schema
        contract.
    """
    return (
        f"{original_prompt}\n\n"
        f"---RETRY-FEEDBACK---\n"
        f"Your previous response was rejected by the parsing pipeline:\n"
        f"{error}\n\n"
        f"Re-emit your response as a complete, syntactically valid JSON "
        f"object containing ALL seven required top-level keys: agent, "
        f"verdict, confidence, summary, reasoning, findings, "
        f"recommendation. Do not omit any key, do not truncate, do not "
        f"emit anything outside the JSON object."
    )


def _load_input_content(input_arg: str) -> tuple[str, str]:
    """Resolve the CLI ``input`` argument to (content, label).

    If *input_arg* is a path to an existing file, the file is read
    with ``encoding="utf-8"`` and ``errors="replace"`` so that a
    cp1252-encoded source (default for Windows tooling that does not
    set an explicit encoding) does not crash MAGI on the first byte
    that is not a valid UTF-8 start byte. Invalid bytes are replaced
    with U+FFFD and the run continues; readable portions of the file
    survive verbatim. The size check still applies.

    If *input_arg* is not a file path, it is returned as inline text
    unchanged — Python str values cannot have an encoding mismatch.

    Args:
        input_arg: The raw value from ``argparse`` for the positional
            ``input`` argument. Either a path to a file or inline
            text.

    Returns:
        Tuple ``(content, label)`` where ``content`` is the prompt
        body and ``label`` is the source description used in the
        eventual prompt envelope (``"File: <path>"`` or
        ``"Inline input"``).

    Raises:
        ValueError: If *input_arg* is a path to a file that exceeds
            :data:`validate.MAX_INPUT_FILE_SIZE`.
    """
    if os.path.isfile(input_arg):
        file_size = os.path.getsize(input_arg)
        if file_size > MAX_INPUT_FILE_SIZE:
            raise ValueError(
                f"Input file {input_arg} is {file_size} bytes, "
                f"exceeding maximum of {MAX_INPUT_FILE_SIZE} bytes."
            )
        # ``errors="replace"`` is the cp1252 hardening shipped in
        # 2.2.6. Windows tooling that writes input files without an
        # explicit encoding produces cp1252 bytes; reading those with
        # strict UTF-8 raises ``UnicodeDecodeError`` on the first
        # byte ≥0x80 that is not a valid UTF-8 start byte. The
        # replacement character (U+FFFD) is preferable to crashing
        # the orchestrator before synthesis.
        with open(input_arg, encoding="utf-8", errors="replace") as f:
            return f.read(), f"File: {input_arg}"
    return input_arg, "Inline input"


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
    # Telemetry (2.2.1): names of agents whose first attempt raised
    # ValidationError, regardless of whether the retry recovered.
    # Composes with ``failed`` to give downstream consumers two derived
    # cohorts: ``retried - failed`` is "retry recovered",
    # ``retried & failed`` is "retry also failed".
    retried: set[str] = set()

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
        """Launch an agent with live status updates and one retry on schema fail.

        State machine emitted to the live display:

        * ``running`` once at entry.
        * ``retrying`` iff the first attempt raised :class:`ValidationError`.
          The retry receives the full ``timeout`` budget and a corrective
          feedback block appended by :func:`_build_retry_prompt`.
        * Terminal state (``success`` | ``timeout`` | ``failed``) emitted
          exactly once by the outer handler, regardless of which attempt
          reached the terminal condition. This is why the retry branch
          does **not** install its own terminal handlers — they would
          duplicate the outer ones and risk drifting out of sync.

        Scope of retry: :class:`ValidationError` only. ``TimeoutError``,
        subprocess exit errors, ``asyncio.CancelledError``, and
        ``BaseException`` subclasses (``KeyboardInterrupt``,
        ``SystemExit``) flow through the outer handler unchanged so the
        degraded-mode and signal paths keep the 2.1.x semantics.
        """
        _safe_display_update(display, name, "running", log_gate)
        try:
            try:
                result = await launch_agent(name, agents_dir, prompt, output_dir, timeout, model)
            except (ValidationError, json.JSONDecodeError) as err:
                # Single-shot retry (2.2.0 + 2.2.4): fires on schema
                # drift (ValidationError, 2.2.0 scope) AND on JSON parse
                # failures (json.JSONDecodeError, 2.2.4 scope expansion).
                # Never on timeout / subprocess failure / cancellation /
                # ValueError (config or parser-shape errors). The retry
                # gets a fresh ``timeout`` budget (not the residual of
                # the first attempt) and carries the parser/validator
                # text so the model can target the specific defect —
                # missing key, truncated output, unbalanced brace, etc.
                retried.add(name)
                _safe_display_update(display, name, "retrying", log_gate)
                result = await launch_agent(
                    name,
                    agents_dir,
                    _build_retry_prompt(prompt, err),
                    output_dir,
                    timeout,
                    model,
                )
        except (asyncio.TimeoutError, TimeoutError):
            _safe_display_update(display, name, "timeout", log_gate)
            raise
        except BaseException:
            # Catches asyncio.CancelledError (which is BaseException in 3.8+),
            # generic Exception subclasses (including a retry that itself
            # raised ValidationError), KeyboardInterrupt, and SystemExit.
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
                f"[!] WARNING: status display failed to start ({exc}) "
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
                f"[!] WARNING: Agent '{name}' failed ({result}) \u2014 excluded from synthesis",
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
            f"[!] WARNING: Running synthesis with "
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

    # Conditional presence mirrors degraded/failed_agents: the field is
    # introduced only when there is something to report so 2.2.0 consumers
    # that ignore unknown keys keep working unchanged.
    if retried:
        report["retried_agents"] = sorted(retried)

    return report


def main() -> None:
    """CLI entry point for MAGI orchestrator."""
    args = parse_args()

    try:
        input_content, input_label = _load_input_content(args.input)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

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
