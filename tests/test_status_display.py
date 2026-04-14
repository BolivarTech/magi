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
