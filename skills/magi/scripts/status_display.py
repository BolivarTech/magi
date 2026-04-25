# Author: Julian Bolivar
# Version: 1.1.3
# Date: 2026-04-17
"""Live tree-style status display for the MAGI orchestrator.

Renders a multi-line status tree showing the state of each agent
(pending, running, success, failed, timeout). Auto-detects TTY and
encoding support:

- On a terminal with ANSI support, the tree is redrawn in place using
  cursor-movement escape codes and a spinner frame.
- On a pipe or captured stream, each ``update()`` call emits a single
  plain-text line (no cursor manipulation, no ANSI codes).
- When the output stream's encoding cannot represent the UTF-8 glyphs
  (e.g., Windows cp1252 consoles), an ASCII-only glyph set is used.

**Write-path invariant**: plain-mode writes (from :meth:`StatusDisplay.update`)
and ANSI refresh writes (from the async ``_refresh_loop``) are mutually
exclusive by design — ``_use_ansi`` selects exactly one of the two paths.
Do not mix them: emitting plain events while the refresh loop is active
would race against the in-place redraws within the same event-loop tick.

Uses only the Python standard library — no external dependencies.
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass
from typing import TextIO

VALID_STATES: frozenset[str] = frozenset(
    {"pending", "running", "retrying", "success", "failed", "timeout"}
)

_TERMINAL_STATES: frozenset[str] = frozenset({"success", "failed", "timeout"})

_DEFAULT_HEADER: str = "MAGI Orchestrator"
_DEFAULT_REFRESH_INTERVAL: float = 0.2
_AGENT_COLUMN_WIDTH: int = 10

# Public UTF-8 spinner frames (kept for backward compatibility).
SPINNER_FRAMES: tuple[str, ...] = (
    "⠋",
    "⠙",
    "⠹",
    "⠸",
    "⠼",
    "⠴",
    "⠦",
    "⠧",
    "⠇",
    "⠏",
)


@dataclass
class GlyphSet:
    """Glyph bundle for one rendering capability (UTF-8 or ASCII).

    Attributes:
        root: Character used for the root node of the tree.
        branch_mid: Connector for non-last agent rows.
        branch_end: Connector for the last agent row.
        spinner: Ordered frames cycled while an agent is ``running``.
        icons: Mapping from non-running state names to a single glyph.
    """

    root: str
    branch_mid: str
    branch_end: str
    spinner: tuple[str, ...]
    icons: dict[str, str]


_UTF8_GLYPHS: GlyphSet = GlyphSet(
    root="●",
    branch_mid="├─",
    branch_end="└─",
    spinner=SPINNER_FRAMES,
    icons={
        "pending": "○",
        "retrying": "↻",
        "success": "✓",
        "failed": "✗",
        "timeout": "⏱",
    },
)

_ASCII_GLYPHS: GlyphSet = GlyphSet(
    root="*",
    branch_mid="|-",
    branch_end="\\-",
    spinner=("|", "/", "-", "\\"),
    icons={
        "pending": ".",
        # ``r`` (lowercase) is the retrying glyph; lowercase avoids visual
        # collision with the capital ``R`` that could appear in agent
        # names or state words. The ``retrying`` state word in the same
        # row carries the authoritative meaning.
        "retrying": "r",
        "success": "v",
        "failed": "x",
        # ``~`` (tilde) is used instead of ``T`` to avoid visual collision
        # with the letter T in agent names and state words. The glyph is
        # cosmetic — the ``timeout`` state word in the same row carries
        # the authoritative meaning.
        "timeout": "~",
    },
)

# Characters used to probe whether the stream encoding supports UTF-8.
# ``↻`` must be included so the probe fails fast on encodings that cannot
# render the retrying glyph — otherwise the probe would say "unicode OK"
# and the render would blow up on the first retry.
_UNICODE_PROBE: str = "●○↻✓✗⏱├─└─⠋"


def _stream_supports_unicode(stream: TextIO) -> bool:
    """Return True if ``stream`` can encode the UTF-8 glyph set.

    Streams without a bound encoding (e.g., :class:`io.StringIO`) hold
    ``str`` directly and are treated as unicode-capable.

    Args:
        stream: The output stream to probe.

    Returns:
        True if the stream accepts the probe glyphs, False otherwise.
    """
    encoding = getattr(stream, "encoding", None)
    if encoding is None:
        return True
    try:
        _UNICODE_PROBE.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


class StatusDisplay:
    """Live status tree renderer for the MAGI orchestrator.

    Tracks the state of each agent and renders a tree-style status display
    either inline (ANSI mode) or as append-only event lines (plain mode).

    Usage::

        display = StatusDisplay(["melchior", "balthasar", "caspar"])
        await display.start()
        try:
            display.update("melchior", "running")
            # ... launch work ...
            display.update("melchior", "success")
        finally:
            await display.stop()

    **Thread / write-path invariant**: plain-mode writes from
    :meth:`update` and ANSI refresh writes from the background loop are
    mutually exclusive by design. Never enable both write paths against
    the same stream.
    """

    def __init__(
        self,
        agents: list[str],
        header: str = _DEFAULT_HEADER,
        *,
        stream: TextIO | None = None,
        use_ansi: bool | None = None,
        refresh_interval: float = _DEFAULT_REFRESH_INTERVAL,
    ) -> None:
        """Initialize a new StatusDisplay.

        Args:
            agents: Ordered list of agent names to display. Must be non-empty.
            header: Text shown on the root line of the tree.
            stream: Output stream (defaults to ``sys.stdout``).
            use_ansi: Force ANSI mode on/off. ``None`` means auto-detect
                based on whether the stream is a TTY and, on Windows, whether
                virtual terminal processing can be enabled.
            refresh_interval: Seconds between spinner redraws in ANSI mode.

        Raises:
            ValueError: If ``agents`` is empty.
        """
        if not agents:
            raise ValueError("agents list cannot be empty")

        self._agents: list[str] = list(agents)
        self._header: str = header
        self._stream: TextIO = stream if stream is not None else sys.stdout
        self._refresh_interval: float = refresh_interval
        self._states: dict[str, str] = {name: "pending" for name in self._agents}
        self._start_times: dict[str, float] = {}
        self._end_times: dict[str, float] = {}
        self._spinner_idx: int = 0
        self._lines_drawn: int = 0
        self._refresh_task: asyncio.Task[None] | None = None
        self._running: bool = False
        self._stopped: bool = False
        self._use_ansi: bool = self._detect_ansi_support() if use_ansi is None else use_ansi

        glyphs: GlyphSet = _UTF8_GLYPHS if _stream_supports_unicode(self._stream) else _ASCII_GLYPHS
        self._root_glyph: str = glyphs.root
        self._branch_mid: str = glyphs.branch_mid
        self._branch_end: str = glyphs.branch_end
        self._spinner: tuple[str, ...] = glyphs.spinner
        self._icons: dict[str, str] = glyphs.icons

    def _detect_ansi_support(self) -> bool:
        """Return True if the output stream likely supports ANSI escapes."""
        is_tty = getattr(self._stream, "isatty", lambda: False)()
        if not is_tty:
            return False
        if sys.platform == "win32":
            return self._enable_windows_vt_mode(self._stream)
        return True

    @staticmethod
    def _enable_windows_vt_mode(stream: TextIO) -> bool:
        """Enable ANSI VT processing on the Windows console handle backing *stream*.

        The handle is derived from ``stream.fileno()`` so the VT flag is
        applied to whichever standard stream the display is actually
        writing to — stdout *or* stderr. MAGI renders to stderr, and an
        earlier version unconditionally enabled VT on ``STD_OUTPUT_HANDLE``
        (-11), leaving stderr-based redraws garbled on legacy Windows
        consoles. Now:

        * ``fd == 1`` resolves to ``STD_OUTPUT_HANDLE`` (-11)
        * ``fd == 2`` resolves to ``STD_ERROR_HANDLE`` (-12)
        * any other fd (wrapper streams, pipes, ``io.StringIO``) returns
          ``False`` so the caller falls through to plain mode.

        Args:
            stream: The output stream whose underlying console handle
                should receive VT processing.

        Returns:
            True on success, False if the fd is unknown, the stream has
            no fileno, or the console mode cannot be set.
        """
        try:
            fd = stream.fileno()
        except (AttributeError, OSError, ValueError):
            return False
        if fd == 1:
            std_handle = -11  # STD_OUTPUT_HANDLE
        elif fd == 2:
            std_handle = -12  # STD_ERROR_HANDLE
        else:
            return False

        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            enable_vt = 0x0004
            handle = kernel32.GetStdHandle(std_handle)
            mode = ctypes.c_ulong()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            return bool(kernel32.SetConsoleMode(handle, mode.value | enable_vt))
        except (OSError, AttributeError, ImportError):
            return False

    def update(self, agent: str, state: str) -> None:
        """Record a state change for a single agent.

        In plain mode this also appends a one-line event to the stream.
        In ANSI mode the stream is only updated by the refresh loop
        (see :meth:`start`) or by calling :meth:`_redraw` directly.

        Args:
            agent: Agent name; must be one of those passed to ``__init__``.
            state: New state; must be one of :data:`VALID_STATES`.

        Raises:
            ValueError: If ``agent`` is unknown or ``state`` is invalid.
        """
        if agent not in self._states:
            raise ValueError(f"Unknown agent: {agent}")
        if state not in VALID_STATES:
            raise ValueError(f"Invalid state: {state}")

        previous = self._states[agent]
        self._states[agent] = state
        now = time.monotonic()

        if state == "running" and previous != "running":
            self._start_times[agent] = now
        elif state in _TERMINAL_STATES and agent in self._start_times:
            self._end_times[agent] = now

        if not self._use_ansi:
            self._write_plain_event(agent)

    def _write_plain_event(self, agent: str) -> None:
        """Write a single status line for ``agent`` in plain mode.

        This method is the plain-mode write path. The ANSI refresh loop
        writes through :meth:`_redraw` instead. The two paths are mutually
        exclusive: a ``RuntimeError`` is raised if this method is ever
        reached while ANSI mode is active, so a future edit cannot
        accidentally introduce a race between the two write paths within
        a single asyncio tick. ``RuntimeError`` (not ``assert``) is used
        so the invariant survives ``python -O``.
        """
        if self._use_ansi:
            raise RuntimeError(
                "_write_plain_event must never run while ANSI mode is active — "
                "plain-mode and ANSI refresh writes are mutually exclusive"
            )
        icon = self._icon_for(agent)
        state = self._states[agent]
        elapsed = self._elapsed_for(agent)
        suffix = f" {elapsed}" if elapsed else ""
        line = f"  {icon} {agent:<{_AGENT_COLUMN_WIDTH}} {state}{suffix}\n"
        self._stream.write(line)
        self._stream.flush()

    def _icon_for(self, agent: str) -> str:
        """Return the current icon for ``agent`` based on its state."""
        state = self._states[agent]
        if state == "running":
            return self._spinner[self._spinner_idx % len(self._spinner)]
        return self._icons[state]

    def _elapsed_for(self, agent: str) -> str:
        """Return the elapsed-time string for ``agent`` or the empty string.

        ``_end_times`` is only populated after ``_start_times``, so both
        lookups are safe without defensive fallbacks.
        """
        if agent in self._end_times:
            return f"({self._end_times[agent] - self._start_times[agent]:.1f}s)"
        if agent in self._start_times:
            return f"({time.monotonic() - self._start_times[agent]:.1f}s)"
        return ""

    def render(self) -> str:
        """Render the current tree as a multi-line string."""
        lines: list[str] = [f"{self._root_glyph} {self._header}"]
        last_index = len(self._agents) - 1
        for index, agent in enumerate(self._agents):
            branch = self._branch_end if index == last_index else self._branch_mid
            icon = self._icon_for(agent)
            state = self._states[agent]
            elapsed = self._elapsed_for(agent)
            suffix = f"  {elapsed}" if elapsed else ""
            lines.append(f"  {branch} {icon} {agent:<{_AGENT_COLUMN_WIDTH}} {state}{suffix}")
        return "\n".join(lines) + "\n"

    def _redraw(self) -> None:
        """Redraw the tree in place (ANSI mode only).

        Clears each previously-drawn line with ``\\033[2K`` before writing
        the new content, avoiding the broader ``\\033[0J`` erase which
        would wipe content scrolled below the tree.
        """
        if not self._use_ansi:
            return
        output_lines = self.render().splitlines()
        if self._lines_drawn > 0:
            self._stream.write(f"\033[{self._lines_drawn}A")
        for line in output_lines:
            self._stream.write(f"\033[2K{line}\n")
        self._stream.flush()
        self._lines_drawn = len(output_lines)
        self._spinner_idx += 1

    async def _refresh_loop(self) -> None:
        """Periodically redraw while :attr:`_running` is True.

        Any ``Exception`` from ``_redraw`` (``OSError`` from a dead pipe
        or closed terminal; ``ValueError`` from a closed ``io.StringIO``;
        ``UnicodeEncodeError`` from a mis-probed encoding; any logic bug
        introduced by a future edit) stops the loop silently instead of
        bubbling out of the background task. ``stop()`` would otherwise
        re-raise the failure on ``await self._refresh_task`` and crash
        the orchestrator after all agent results had already been
        gathered — losing them to a UI-only failure. The contract here
        is: *the live display is never allowed to fail the run*.

        ``BaseException`` subclasses (``KeyboardInterrupt``,
        ``SystemExit``, ``GeneratorExit``) deliberately fall through —
        those are shutdown signals the loop must respect, not display
        errors to suppress. ``CancelledError`` is handled separately
        below because it is the expected ``stop()`` path.
        """
        try:
            while self._running:
                try:
                    self._redraw()
                except Exception:  # noqa: BLE001 — display failures must never fail the run
                    self._running = False
                    return
                await asyncio.sleep(self._refresh_interval)
        except asyncio.CancelledError:
            pass

    async def start(self) -> None:
        """Begin the background refresh loop (ANSI mode only)."""
        if not self._use_ansi:
            return
        self._running = True
        self._redraw()
        self._refresh_task = asyncio.create_task(self._refresh_loop())

    async def stop(self) -> None:
        """Stop the refresh loop and render the final state.

        Idempotent: calling ``stop()`` more than once or without a prior
        ``start()`` is safe and produces no additional output.
        """
        if self._stopped:
            return
        self._stopped = True
        self._running = False
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
            self._refresh_task = None
        if self._use_ansi:
            try:
                self._redraw()
            except Exception:  # noqa: BLE001 — see _refresh_loop rationale
                # Same rationale as ``_refresh_loop``: a UI-only failure
                # during the final redraw must not crash ``stop`` and
                # discard the gathered agent results. Covers the same
                # full ``Exception`` surface for the same reason —
                # ``OSError`` (dead pipe), ``ValueError`` (closed
                # StringIO), ``UnicodeEncodeError`` (mis-probed
                # encoding), and any future bug in ``_redraw`` must all
                # degrade to "no final redraw" rather than losing the
                # agent results.
                pass
