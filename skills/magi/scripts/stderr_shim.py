#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-04-13
"""Stderr write-buffering shim for MAGI's live status display.

When the status display renders live to ``sys.stderr``, any concurrent
diagnostic write lands inside the in-place redraw region and is wiped
on the next refresh tick. The primitives in this module structurally
enforce the display-active-stderr-quiet invariant: while the display is
active, ``sys.stderr`` is replaced with a write-only buffer, and on
exit the buffered content is replayed to the real stream after the
display has been stopped.

Extracted from :mod:`run_magi` so the orchestrator stays focused on
orchestration and the shim machinery is independently testable.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Iterator
from typing import Any


class _StreamProxy:
    """Common base for the stderr shims.

    Holds a reference to the real underlying stream, provides a no-op
    ``flush`` (the buffer is replayed at context exit, not mid-run),
    and proxies every attribute lookup that the subclasses do not
    override to the real stream — this keeps ``fileno``, ``encoding``,
    ``isatty`` and friends working transparently.
    """

    def __init__(self, real: Any) -> None:
        self._real = real

    def flush(self) -> None:
        """Intentional no-op: content stays buffered until context exit.

        The :func:`_buffered_stderr_while` context manager replays the
        full buffer to the real stderr on ``__exit__``; flushing the
        shim mid-context would defeat the display-active invariant.
        """
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _BinaryStderrBufferShim(_StreamProxy):
    """Binary write-buffering proxy for ``sys.stderr.buffer``.

    Callers that do ``sys.stderr.buffer.write(b"...")`` would otherwise
    bypass the text-mode :class:`_StderrBufferShim` and write directly
    to the real stream, breaking the display-active-stderr-quiet
    invariant. This shim catches binary writes, decodes them into the
    shared text buffer (UTF-8 with replacement for any undecodable
    bytes), and proxies every non-write attribute to the real binary
    buffer via :class:`_StreamProxy`.
    """

    def __init__(self, text_buffer: list[str], real_binary: Any) -> None:
        super().__init__(real_binary)
        self._text_buffer = text_buffer

    def write(self, data: bytes) -> int:
        self._text_buffer.append(data.decode("utf-8", errors="replace"))
        return len(data)


class _StderrBufferShim(_StreamProxy):
    """Write-buffering proxy for :data:`sys.stderr`.

    Forwards ``write`` to an internal list and proxies every other
    attribute (``encoding``, ``fileno``, ``isatty``, ...) to the
    original stderr via :class:`_StreamProxy`. Exposes a ``.buffer``
    attribute backed by :class:`_BinaryStderrBufferShim` so binary
    writes through ``sys.stderr.buffer`` are also intercepted.

    Used by :func:`_buffered_stderr_while` to structurally enforce the
    display-active-stderr-quiet invariant.

    **Uncovered paths (documented in ``CLAUDE.md`` Known limitations):**
    - ``os.write(sys.stderr.fileno(), ...)`` bypasses Python-level proxies
      and writes directly to fd 2.
    - Hard process death (``SIGKILL``, segfault) skips the buffer flush
      in ``_buffered_stderr_while``'s ``finally`` clause.
    """

    def __init__(self, real_stderr: Any, buffer: list[str]) -> None:
        super().__init__(real_stderr)
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
