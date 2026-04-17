# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-04-13
"""Tests for the StatusDisplay live-tree renderer."""

from __future__ import annotations

import asyncio
import io

import pytest

from status_display import VALID_STATES, StatusDisplay


class TestInit:
    def test_empty_agents_raises(self):
        with pytest.raises(ValueError):
            StatusDisplay([], stream=io.StringIO(), use_ansi=False)

    def test_agents_start_pending(self):
        d = StatusDisplay(["a", "b"], stream=io.StringIO(), use_ansi=False)
        out = d.render()
        assert out.count("pending") == 2

    def test_custom_header_in_render(self):
        d = StatusDisplay(["a"], header="MAGI Test", stream=io.StringIO(), use_ansi=False)
        assert "MAGI Test" in d.render()

    def test_default_header_in_render(self):
        d = StatusDisplay(["a"], stream=io.StringIO(), use_ansi=False)
        assert "MAGI Orchestrator" in d.render()


class TestUpdate:
    def _make(self):
        return StatusDisplay(
            ["melchior", "balthasar", "caspar"],
            stream=io.StringIO(),
            use_ansi=False,
        )

    def test_update_running(self):
        d = self._make()
        d.update("melchior", "running")
        assert "running" in d.render()

    def test_update_success(self):
        d = self._make()
        d.update("melchior", "running")
        d.update("melchior", "success")
        assert "success" in d.render()

    def test_update_failed(self):
        d = self._make()
        d.update("caspar", "failed")
        assert "failed" in d.render()

    def test_update_timeout(self):
        d = self._make()
        d.update("balthasar", "timeout")
        assert "timeout" in d.render()

    def test_unknown_agent_raises(self):
        d = self._make()
        with pytest.raises(ValueError, match="Unknown agent"):
            d.update("unknown", "running")

    def test_invalid_state_raises(self):
        d = self._make()
        with pytest.raises(ValueError, match="Invalid state"):
            d.update("melchior", "bogus")

    def test_all_valid_states_accepted(self):
        d = self._make()
        for state in VALID_STATES:
            d.update("melchior", state)


class TestRenderFormat:
    def test_tree_branches_present(self):
        d = StatusDisplay(["a", "b", "c"], stream=io.StringIO(), use_ansi=False)
        out = d.render()
        assert "├─" in out
        assert "└─" in out

    def test_last_agent_uses_end_branch(self):
        d = StatusDisplay(["a", "b"], stream=io.StringIO(), use_ansi=False)
        lines = d.render().splitlines()
        assert "├─" in lines[1]
        assert "└─" in lines[2]

    def test_single_agent_uses_end_branch(self):
        d = StatusDisplay(["only"], stream=io.StringIO(), use_ansi=False)
        out = d.render()
        assert "└─" in out
        assert "├─" not in out

    def test_icon_for_pending(self):
        d = StatusDisplay(["a"], stream=io.StringIO(), use_ansi=False)
        assert "○" in d.render()

    def test_icon_for_success(self):
        d = StatusDisplay(["a"], stream=io.StringIO(), use_ansi=False)
        d.update("a", "success")
        assert "✓" in d.render()

    def test_icon_for_failed(self):
        d = StatusDisplay(["a"], stream=io.StringIO(), use_ansi=False)
        d.update("a", "failed")
        assert "✗" in d.render()

    def test_icon_for_timeout(self):
        d = StatusDisplay(["a"], stream=io.StringIO(), use_ansi=False)
        d.update("a", "timeout")
        assert "⏱" in d.render()


class TestPlainMode:
    def test_update_writes_to_stream(self):
        buf = io.StringIO()
        d = StatusDisplay(["m"], stream=buf, use_ansi=False)
        d.update("m", "running")
        assert "running" in buf.getvalue()
        assert "m" in buf.getvalue()

    def test_output_has_no_ansi_codes(self):
        buf = io.StringIO()
        d = StatusDisplay(["m"], stream=buf, use_ansi=False)
        d.update("m", "running")
        d.update("m", "success")
        assert "\033[" not in buf.getvalue()


class TestAnsiMode:
    def test_update_does_not_write_immediately(self):
        buf = io.StringIO()
        d = StatusDisplay(["m"], stream=buf, use_ansi=True)
        d.update("m", "running")
        assert buf.getvalue() == ""

    def test_redraw_emits_content(self):
        buf = io.StringIO()
        d = StatusDisplay(["m"], stream=buf, use_ansi=True)
        d._redraw()
        assert "MAGI Orchestrator" in buf.getvalue()

    def test_second_redraw_emits_cursor_codes(self):
        buf = io.StringIO()
        d = StatusDisplay(["m"], stream=buf, use_ansi=True)
        d._redraw()
        buf.truncate(0)
        buf.seek(0)
        d._redraw()
        assert "\033[" in buf.getvalue()


class TestAsyncLifecycle:
    @pytest.mark.asyncio
    async def test_start_stop_plain_mode_is_noop(self):
        buf = io.StringIO()
        d = StatusDisplay(["m"], stream=buf, use_ansi=False)
        await d.start()
        await d.stop()

    @pytest.mark.asyncio
    async def test_start_stop_ansi_mode_writes_output(self):
        buf = io.StringIO()
        d = StatusDisplay(["m"], stream=buf, use_ansi=True, refresh_interval=0.01)
        await d.start()
        await asyncio.sleep(0.05)
        await d.stop()
        assert len(buf.getvalue()) > 0

    @pytest.mark.asyncio
    async def test_stop_without_start_plain_mode(self):
        buf = io.StringIO()
        d = StatusDisplay(["m"], stream=buf, use_ansi=False)
        await d.stop()  # must not raise

    @pytest.mark.asyncio
    async def test_stop_without_start_ansi_mode(self):
        buf = io.StringIO()
        d = StatusDisplay(["m"], stream=buf, use_ansi=True)
        await d.stop()  # must not raise

    @pytest.mark.asyncio
    async def test_double_stop_is_idempotent(self):
        buf = io.StringIO()
        d = StatusDisplay(["m"], stream=buf, use_ansi=True, refresh_interval=0.01)
        await d.start()
        await asyncio.sleep(0.02)
        await d.stop()
        snapshot = buf.getvalue()
        await d.stop()
        assert buf.getvalue() == snapshot

    @pytest.mark.asyncio
    async def test_refresh_loop_survives_oserror_from_redraw(self):
        """Regression (v2.1.2): if ``_redraw`` raises ``OSError`` /
        ``BrokenPipeError`` (stream pipe died, terminal closed mid-run),
        the refresh loop must stop silently. Pre-2.1.2 the unhandled
        exception killed the background task; the next ``await
        self._refresh_task`` in :meth:`stop` then re-raised it and
        crashed the orchestrator on the way out, swallowing the agent
        results that had already been gathered.

        ``stop()`` must not raise, and the loop must have actually
        attempted at least one post-``start`` redraw so the test is
        not trivially satisfied by an empty loop body.
        """
        buf = io.StringIO()
        d = StatusDisplay(["m"], stream=buf, use_ansi=True, refresh_interval=0.01)

        original_redraw = d._redraw
        redraw_calls = {"count": 0}

        def broken_after_first() -> None:
            redraw_calls["count"] += 1
            if redraw_calls["count"] >= 2:
                raise BrokenPipeError("terminal pipe closed mid-redraw")
            original_redraw()

        d._redraw = broken_after_first  # type: ignore[method-assign]

        await d.start()
        # Give the background loop time to fire the broken redraw.
        await asyncio.sleep(0.1)
        # stop() must complete without re-raising the BrokenPipeError.
        await d.stop()

        assert redraw_calls["count"] >= 2, (
            "refresh loop must have attempted at least one post-start redraw "
            f"to exercise the OSError path; got {redraw_calls['count']} calls."
        )

    @pytest.mark.asyncio
    async def test_refresh_loop_survives_non_oserror_from_redraw(self):
        """Regression (v2.1.3): if ``_redraw`` raises anything other than
        ``OSError`` (``ValueError`` from a closed ``io.StringIO``, a
        ``UnicodeEncodeError`` if the glyph probe misidentifies the
        stream, a ``ZeroDivisionError`` introduced by a future edit),
        the refresh loop must stop silently.

        Pre-2.1.3 the handler was narrowed to ``except OSError`` only, so
        a non-OSError failure propagated out of the background task,
        ``stop()`` re-raised it on ``await self._refresh_task``, and the
        orchestrator crashed after all agent results had been gathered —
        the exact regression ``test_refresh_loop_survives_oserror_from_redraw``
        was written to prevent, just via a different exception type.

        The loop must cover ``Exception`` broadly, not only ``OSError``.
        """
        buf = io.StringIO()
        d = StatusDisplay(["m"], stream=buf, use_ansi=True, refresh_interval=0.01)

        original_redraw = d._redraw
        redraw_calls = {"count": 0}

        def broken_after_first() -> None:
            redraw_calls["count"] += 1
            if redraw_calls["count"] >= 2:
                raise ValueError("simulated non-OSError failure mid-redraw")
            original_redraw()

        d._redraw = broken_after_first  # type: ignore[method-assign]

        await d.start()
        await asyncio.sleep(0.1)
        # stop() must complete without re-raising the ValueError.
        await d.stop()

        assert redraw_calls["count"] >= 2, (
            "refresh loop must have attempted at least one post-start redraw "
            f"to exercise the non-OSError path; got {redraw_calls['count']} calls."
        )


class TestAsciiFallback:
    def test_ascii_glyphs_when_encoding_is_cp1252(self):
        buf = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", newline="")
        d = StatusDisplay(["a", "b"], stream=buf, use_ansi=False)
        out = d.render()
        # No non-ASCII glyphs must appear in render output.
        assert all(ord(c) < 128 or c in ("\n",) for c in out)
        # Must still show tree structure using ASCII branches.
        assert "|-" in out
        assert "\\-" in out

    def test_utf8_glyphs_when_stream_is_utf8(self):
        buf = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", newline="")
        d = StatusDisplay(["a"], stream=buf, use_ansi=False)
        out = d.render()
        assert "└─" in out
        assert "○" in out

    def test_stringio_uses_utf8_glyphs(self):
        d = StatusDisplay(["a"], stream=io.StringIO(), use_ansi=False)
        out = d.render()
        assert "○" in out

    def test_cp1252_stream_does_not_raise_on_update(self):
        buf = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", newline="")
        d = StatusDisplay(["m"], stream=buf, use_ansi=False)
        d.update("m", "running")
        d.update("m", "success")
        d.update("m", "failed")
        d.update("m", "timeout")

    def test_ascii_timeout_glyph_is_not_a_letter(self):
        """Regression: earlier versions used 'T' which collides visually
        with the letter T in agent names and state words."""
        from status_display import _ASCII_GLYPHS

        timeout_glyph = _ASCII_GLYPHS.icons["timeout"]
        assert not timeout_glyph.isalpha(), (
            f"ASCII timeout glyph must not be a letter, got {timeout_glyph!r}"
        )
        assert timeout_glyph != "T"

    def test_ascii_glyphs_are_all_ascii(self):
        """All glyphs in the ASCII fallback set must be pure ASCII."""
        from status_display import _ASCII_GLYPHS

        assert all(ord(c) < 128 for c in _ASCII_GLYPHS.root)
        assert all(ord(c) < 128 for c in _ASCII_GLYPHS.branch_mid)
        assert all(ord(c) < 128 for c in _ASCII_GLYPHS.branch_end)
        for frame in _ASCII_GLYPHS.spinner:
            assert all(ord(c) < 128 for c in frame)
        for icon in _ASCII_GLYPHS.icons.values():
            assert all(ord(c) < 128 for c in icon)


class TestWritePathInvariantTripwire:
    """Guard against mixing plain-mode and ANSI refresh writes."""

    def test_plain_write_raises_when_ansi_mode_is_active(self):
        """Calling _write_plain_event while use_ansi=True must raise RuntimeError.

        ``RuntimeError`` survives ``python -O`` (assert statements are
        stripped under optimize mode) so the invariant is enforced in
        production-style runs, not just during development.
        """
        buf = io.StringIO()
        d = StatusDisplay(["m"], stream=buf, use_ansi=True)
        with pytest.raises(RuntimeError, match="mutually exclusive"):
            d._write_plain_event("m")


class TestWindowsVtModeStreamHandle:
    """Verify ``_enable_windows_vt_mode`` selects the handle matching *stream*.

    Earlier versions unconditionally enabled VT on ``STD_OUTPUT_HANDLE``
    even when the display wrote to ``stderr``, leaving the live tree
    unreadable on legacy Windows consoles. The fix is to pick the
    handle from the stream's ``fileno()``: 1 -> STD_OUTPUT_HANDLE (-11),
    2 -> STD_ERROR_HANDLE (-12), anything else -> False (plain mode).
    """

    def _make_stream_with_fd(self, fd: int):
        class _FakeStream:
            def fileno(self_inner) -> int:
                return fd

        return _FakeStream()

    def test_unknown_fd_returns_false(self):
        """A stream whose fd is not 1 or 2 (e.g. a pipe) must not enable VT."""
        stream = self._make_stream_with_fd(7)
        assert StatusDisplay._enable_windows_vt_mode(stream) is False

    def test_stream_without_fileno_returns_false(self):
        """A stream that cannot produce a fd (e.g. ``io.StringIO``) returns False."""
        assert StatusDisplay._enable_windows_vt_mode(io.StringIO()) is False

    def test_fd_two_targets_stderr_handle(self, monkeypatch):
        """fd == 2 must call GetStdHandle with STD_ERROR_HANDLE (-12), not -11.

        This is the regression: the display is instantiated against
        ``sys.stderr``, so VT must be enabled on the stderr console
        handle. Enabling it on stdout leaves stderr-rendered escape
        codes uninterpreted.
        """
        import sys as _sys

        if _sys.platform != "win32":
            pytest.skip("Windows-only path")

        import ctypes

        handles_requested: list[int] = []

        class _FakeKernel32:
            def GetStdHandle(self_inner, handle_id: int) -> int:
                handles_requested.append(handle_id)
                return 42

            def GetConsoleMode(self_inner, handle, mode_ref) -> int:
                mode_ref._obj.value = 0
                return 1

            def SetConsoleMode(self_inner, handle, new_mode) -> int:
                return 1

        class _FakeWindll:
            kernel32 = _FakeKernel32()

        monkeypatch.setattr(ctypes, "windll", _FakeWindll(), raising=False)

        stream = self._make_stream_with_fd(2)
        result = StatusDisplay._enable_windows_vt_mode(stream)

        assert result is True
        assert handles_requested == [-12], (
            f"fd == 2 must resolve to STD_ERROR_HANDLE (-12), got {handles_requested!r}"
        )

    def test_fd_one_targets_stdout_handle(self, monkeypatch):
        """fd == 1 must call GetStdHandle with STD_OUTPUT_HANDLE (-11)."""
        import sys as _sys

        if _sys.platform != "win32":
            pytest.skip("Windows-only path")

        import ctypes

        handles_requested: list[int] = []

        class _FakeKernel32:
            def GetStdHandle(self_inner, handle_id: int) -> int:
                handles_requested.append(handle_id)
                return 42

            def GetConsoleMode(self_inner, handle, mode_ref) -> int:
                mode_ref._obj.value = 0
                return 1

            def SetConsoleMode(self_inner, handle, new_mode) -> int:
                return 1

        class _FakeWindll:
            kernel32 = _FakeKernel32()

        monkeypatch.setattr(ctypes, "windll", _FakeWindll(), raising=False)

        stream = self._make_stream_with_fd(1)
        result = StatusDisplay._enable_windows_vt_mode(stream)

        assert result is True
        assert handles_requested == [-11]
