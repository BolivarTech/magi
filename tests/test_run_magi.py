# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-04-01
"""Tests for run_magi.py — async Python orchestrator."""

from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import patch

import pytest


class TestParseArgs:
    """Verify CLI argument parsing."""

    def test_minimal_args(self):
        from run_magi import parse_args

        args = parse_args(["code-review", "input.py"])
        assert args.mode == "code-review"
        assert args.input == "input.py"
        assert args.timeout == 900
        assert args.output_dir is None

    def test_custom_timeout(self):
        from run_magi import parse_args

        args = parse_args(["analysis", "file.txt", "--timeout", "60"])
        assert args.timeout == 60

    def test_custom_output_dir(self):
        from run_magi import parse_args

        args = parse_args(["design", "spec.md", "--output-dir", "/tmp/out"])
        assert args.output_dir == "/tmp/out"

    def test_invalid_mode_rejected(self):
        from run_magi import parse_args

        with pytest.raises(SystemExit):
            parse_args(["invalid-mode", "input.py"])

    def test_all_valid_modes(self):
        from run_magi import parse_args

        for mode in ("code-review", "design", "analysis"):
            args = parse_args([mode, "input.py"])
            assert args.mode == mode

    def test_default_model_is_opus(self):
        from run_magi import parse_args

        args = parse_args(["code-review", "input.py"])
        assert args.model == "opus"

    def test_custom_model(self):
        from run_magi import parse_args

        for model in ("opus", "sonnet", "haiku"):
            args = parse_args(["code-review", "input.py", "--model", model])
            assert args.model == model

    def test_invalid_model_rejected(self):
        from run_magi import parse_args

        with pytest.raises(SystemExit):
            parse_args(["code-review", "input.py", "--model", "gpt4"])

    def test_default_show_status_true(self):
        from run_magi import parse_args

        args = parse_args(["code-review", "input.py"])
        assert args.show_status is True

    def test_no_status_flag_sets_false(self):
        from run_magi import parse_args

        args = parse_args(["code-review", "input.py", "--no-status"])
        assert args.show_status is False


class TestCreateOutputDir:
    """Verify cross-platform temp directory creation."""

    def test_uses_tempfile_mkdtemp(self):
        from run_magi import create_output_dir

        output_dir = create_output_dir(None)
        assert os.path.isdir(output_dir)
        assert "magi-run-" in os.path.basename(output_dir)
        os.rmdir(output_dir)

    def test_respects_explicit_output_dir(self, tmp_path):
        from run_magi import create_output_dir

        output_dir = create_output_dir(str(tmp_path / "custom"))
        assert output_dir == str(tmp_path / "custom")
        assert os.path.isdir(output_dir)


class TestRunOrchestrator:
    """Verify full orchestration with mocked agents."""

    @pytest.mark.asyncio
    async def test_all_three_agents_success(self, tmp_path):
        from run_magi import run_orchestrator

        agent_results = {}
        for name in ("melchior", "balthasar", "caspar"):
            agent_results[name] = {
                "agent": name,
                "verdict": "approve",
                "confidence": 0.9,
                "summary": f"{name} OK",
                "reasoning": "Fine",
                "findings": [],
                "recommendation": "Merge",
            }

        async def mock_launch(agent_name, agents_dir, prompt, output_dir, timeout, model="opus"):
            return agent_results[agent_name]

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            result = await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
            )
            assert result["consensus"]["consensus"] == "STRONG GO"
            assert result.get("degraded") is not True
            assert len(result["agents"]) == 3

    @pytest.mark.asyncio
    async def test_one_agent_fails_degraded_mode(self, tmp_path):
        from run_magi import run_orchestrator

        async def mock_launch(agent_name, agents_dir, prompt, output_dir, timeout, model="opus"):
            if agent_name == "caspar":
                raise TimeoutError(f"Agent {agent_name} timed out")
            return {
                "agent": agent_name,
                "verdict": "approve",
                "confidence": 0.85,
                "summary": "OK",
                "reasoning": "Fine",
                "findings": [],
                "recommendation": "Merge",
            }

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            result = await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
            )
            assert result["degraded"] is True
            assert "caspar" in result["failed_agents"]
            assert len(result["agents"]) == 2

    @pytest.mark.asyncio
    async def test_all_agents_fail_raises(self, tmp_path):
        from run_magi import run_orchestrator

        async def mock_launch(agent_name, agents_dir, prompt, output_dir, timeout, model="opus"):
            raise TimeoutError(f"Agent {agent_name} timed out")

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            with pytest.raises(RuntimeError, match="fewer than 2"):
                await run_orchestrator(
                    agents_dir=str(tmp_path),
                    prompt="test",
                    output_dir=str(tmp_path),
                    timeout=300,
                )

    @pytest.mark.asyncio
    async def test_model_passed_to_launch_agent(self, tmp_path):
        """Verify that the model parameter propagates to launch_agent."""
        from run_magi import run_orchestrator

        captured_models: list[str] = []

        async def mock_launch(agent_name, agents_dir, prompt, output_dir, timeout, model="opus"):
            captured_models.append(model)
            return {
                "agent": agent_name,
                "verdict": "approve",
                "confidence": 0.9,
                "summary": "OK",
                "reasoning": "Fine",
                "findings": [],
                "recommendation": "Merge",
            }

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
                model="sonnet",
            )
            assert all(m == "sonnet" for m in captured_models)
            assert len(captured_models) == 3

    @pytest.mark.asyncio
    async def test_two_fail_one_succeeds_raises(self, tmp_path):
        from run_magi import run_orchestrator

        async def mock_launch(agent_name, agents_dir, prompt, output_dir, timeout, model="opus"):
            if agent_name != "melchior":
                raise TimeoutError(f"Agent {agent_name} timed out")
            return {
                "agent": "melchior",
                "verdict": "approve",
                "confidence": 0.9,
                "summary": "OK",
                "reasoning": "Fine",
                "findings": [],
                "recommendation": "Merge",
            }

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            with pytest.raises(RuntimeError, match="fewer than 2"):
                await run_orchestrator(
                    agents_dir=str(tmp_path),
                    prompt="test",
                    output_dir=str(tmp_path),
                    timeout=300,
                )


class TestCleanupOldRuns:
    """Verify LRU cleanup of old MAGI temp directories."""

    def test_keep_zero_disables_cleanup(self, tmp_path):
        """keep <= 0 should not scan or delete anything."""
        from run_magi import cleanup_old_runs

        with patch("run_magi.tempfile.gettempdir", return_value=str(tmp_path)):
            magi_dir = tmp_path / "magi-run-abc123"
            magi_dir.mkdir()
            cleanup_old_runs(0)
            assert magi_dir.exists()

    def test_keeps_most_recent(self, tmp_path):
        """Should keep the N most recent and remove the rest."""
        from run_magi import cleanup_old_runs

        dirs = []
        for i in range(4):
            d = tmp_path / f"magi-run-{i:04d}"
            d.mkdir()
            # Set different mtimes
            os.utime(d, (1000 + i, 1000 + i))
            dirs.append(d)

        with patch("run_magi.tempfile.gettempdir", return_value=str(tmp_path)):
            cleanup_old_runs(2)

        # Most recent (dirs[2], dirs[3]) should remain
        assert dirs[3].exists()
        assert dirs[2].exists()
        assert not dirs[0].exists()
        assert not dirs[1].exists()

    def test_mtime_tie_uses_path_ascending_tiebreaker(self, tmp_path):
        """B-2: on mtime ties, cleanup must keep the lex-smallest path.

        Two or more ``magi-run-*`` dirs with identical ``st_mtime`` must
        produce a deterministic survivor. The contract is: sort by mtime
        descending, then by path ascending. The lex-smallest path is
        treated as the canonical survivor — not whatever ``os.scandir``
        happened to yield first.
        """
        from run_magi import cleanup_old_runs

        names = ["magi-run-0003", "magi-run-0001", "magi-run-0002"]
        for name in names:
            d = tmp_path / name
            d.mkdir()
            os.utime(d, (1000, 1000))  # identical mtime across all three

        with patch("run_magi.tempfile.gettempdir", return_value=str(tmp_path)):
            cleanup_old_runs(1)

        survivors = sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("magi-run-"))
        assert survivors == ["magi-run-0001"], (
            f"Under mtime ties, the lex-smallest path must be kept, got {survivors}"
        )

    def test_mtime_tie_tiebreaker_keeps_top_n(self, tmp_path):
        """B-2: with keep=2 and all mtimes tied, the two lex-smallest survive."""
        from run_magi import cleanup_old_runs

        for name in ("magi-run-b", "magi-run-d", "magi-run-a", "magi-run-c"):
            d = tmp_path / name
            d.mkdir()
            os.utime(d, (2000, 2000))

        with patch("run_magi.tempfile.gettempdir", return_value=str(tmp_path)):
            cleanup_old_runs(2)

        survivors = sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("magi-run-"))
        assert survivors == ["magi-run-a", "magi-run-b"]

    def test_cleanup_noop_when_no_magi_dirs(self, tmp_path):
        """B-2: with no magi-run-* entries, cleanup is a no-op.

        Unrelated files and directories in the temp root must survive
        and no exception must escape.
        """
        from run_magi import cleanup_old_runs

        (tmp_path / "other-dir").mkdir()
        (tmp_path / "readme.txt").write_text("keep me")

        with patch("run_magi.tempfile.gettempdir", return_value=str(tmp_path)):
            cleanup_old_runs(1)

        assert (tmp_path / "other-dir").exists()
        assert (tmp_path / "readme.txt").exists()

    def test_symlink_outside_temp_root_skipped(self, tmp_path):
        """Symlinks resolving outside temp root should be skipped."""
        from run_magi import cleanup_old_runs

        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        symlink_path = tmp_path / "magi-run-evil"
        try:
            symlink_path.symlink_to(outside_dir, target_is_directory=True)
        except OSError:
            pytest.skip("Symlinks not supported on this platform")

        with patch("run_magi.tempfile.gettempdir", return_value=str(tmp_path)):
            cleanup_old_runs(0)
            # keep=0 disables, use keep=1 with 2 dirs to trigger cleanup
            real_dir = tmp_path / "magi-run-real"
            real_dir.mkdir()
            os.utime(real_dir, (2000, 2000))
            os.utime(symlink_path, (1000, 1000))
            cleanup_old_runs(1)

        # Outside dir should not be deleted
        assert outside_dir.exists()


class TestModelsModule:
    """C-1: MODEL_IDS and VALID_MODELS live in a dedicated models module.

    Bumping a model ID must be a one-line change to a data module, not
    an edit to the orchestration code in run_magi.py.
    """

    def test_models_module_importable(self):
        """The models module must be importable by its short name."""
        import importlib

        module = importlib.import_module("models")
        assert module is not None

    def test_model_ids_contains_expected_keys(self):
        """MODEL_IDS must map the three short names to Anthropic model IDs."""
        from models import MODEL_IDS

        assert set(MODEL_IDS.keys()) == {"opus", "sonnet", "haiku"}
        assert all(isinstance(v, str) and v for v in MODEL_IDS.values())

    def test_valid_models_derived_from_model_ids(self):
        """VALID_MODELS must stay in lockstep with MODEL_IDS.keys()."""
        from models import MODEL_IDS, VALID_MODELS

        assert set(VALID_MODELS) == set(MODEL_IDS.keys())

    def test_run_magi_reexports_model_ids_from_models_module(self):
        """run_magi.MODEL_IDS must be the same object as models.MODEL_IDS.

        Reference identity (``is``) — not merely equality — rules out
        accidental shadowing where run_magi keeps its own local copy
        that could drift from the canonical source.
        """
        import models
        import run_magi

        assert run_magi.MODEL_IDS is models.MODEL_IDS

    def test_run_magi_reexports_valid_models_from_models_module(self):
        """Same identity guarantee for VALID_MODELS."""
        import models
        import run_magi

        assert run_magi.VALID_MODELS is models.VALID_MODELS


class TestLaunchAgentValidation:
    """Verify launch_agent input validation."""

    @pytest.mark.asyncio
    async def test_invalid_model_raises_value_error(self, tmp_path):
        from run_magi import launch_agent

        with pytest.raises(ValueError, match="Unknown model"):
            await launch_agent(
                agent_name="melchior",
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
                model="gpt4",
            )


class _FakeDisplay:
    """Test double that records update() calls without writing to any stream."""

    def __init__(self, *args, **kwargs):
        self.calls: list[tuple[str, str]] = []

    def update(self, agent: str, state: str) -> None:
        self.calls.append((agent, state))

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


def _ok_result(name: str) -> dict:
    return {
        "agent": name,
        "verdict": "approve",
        "confidence": 0.9,
        "summary": f"{name} OK",
        "reasoning": "Fine",
        "findings": [],
        "recommendation": "Merge",
    }


class TestTrackedLaunchStatusUpdates:
    """Verify tracked_launch wiring between run_orchestrator and StatusDisplay."""

    @pytest.mark.asyncio
    async def test_success_path_emits_running_then_success(self, tmp_path, monkeypatch):
        import run_magi

        instances: list[_FakeDisplay] = []

        def factory(*args, **kwargs):
            inst = _FakeDisplay()
            instances.append(inst)
            return inst

        monkeypatch.setattr(run_magi, "StatusDisplay", factory)

        async def mock_launch(agent_name, *args, **kwargs):
            return _ok_result(agent_name)

        monkeypatch.setattr(run_magi, "launch_agent", mock_launch)

        await run_magi.run_orchestrator(
            agents_dir=str(tmp_path),
            prompt="test",
            output_dir=str(tmp_path),
            timeout=300,
        )

        assert len(instances) == 1
        calls = instances[0].calls
        for name in ("melchior", "balthasar", "caspar"):
            assert (name, "running") in calls
            assert (name, "success") in calls
            assert (name, "failed") not in calls
            assert (name, "timeout") not in calls

    @pytest.mark.asyncio
    async def test_builtin_timeout_error_emits_timeout(self, tmp_path, monkeypatch):
        import run_magi

        instances: list[_FakeDisplay] = []
        monkeypatch.setattr(
            run_magi,
            "StatusDisplay",
            lambda *a, **kw: instances.append(_FakeDisplay()) or instances[-1],
        )

        async def mock_launch(agent_name, *args, **kwargs):
            if agent_name == "caspar":
                raise TimeoutError("builtin timeout")
            return _ok_result(agent_name)

        monkeypatch.setattr(run_magi, "launch_agent", mock_launch)

        await run_magi.run_orchestrator(
            agents_dir=str(tmp_path),
            prompt="test",
            output_dir=str(tmp_path),
            timeout=300,
        )

        assert ("caspar", "timeout") in instances[0].calls
        assert ("caspar", "failed") not in instances[0].calls

    @pytest.mark.asyncio
    async def test_asyncio_timeout_error_emits_timeout(self, tmp_path, monkeypatch):
        """Python 3.9/3.10: asyncio.TimeoutError must be treated as timeout too."""
        import run_magi

        instances: list[_FakeDisplay] = []
        monkeypatch.setattr(
            run_magi,
            "StatusDisplay",
            lambda *a, **kw: instances.append(_FakeDisplay()) or instances[-1],
        )

        async def mock_launch(agent_name, *args, **kwargs):
            if agent_name == "caspar":
                raise asyncio.TimeoutError("asyncio timeout")
            return _ok_result(agent_name)

        monkeypatch.setattr(run_magi, "launch_agent", mock_launch)

        await run_magi.run_orchestrator(
            agents_dir=str(tmp_path),
            prompt="test",
            output_dir=str(tmp_path),
            timeout=300,
        )

        assert ("caspar", "timeout") in instances[0].calls
        assert ("caspar", "failed") not in instances[0].calls

    @pytest.mark.asyncio
    async def test_generic_exception_emits_failed(self, tmp_path, monkeypatch):
        import run_magi

        instances: list[_FakeDisplay] = []
        monkeypatch.setattr(
            run_magi,
            "StatusDisplay",
            lambda *a, **kw: instances.append(_FakeDisplay()) or instances[-1],
        )

        async def mock_launch(agent_name, *args, **kwargs):
            if agent_name == "caspar":
                raise RuntimeError("boom")
            return _ok_result(agent_name)

        monkeypatch.setattr(run_magi, "launch_agent", mock_launch)

        await run_magi.run_orchestrator(
            agents_dir=str(tmp_path),
            prompt="test",
            output_dir=str(tmp_path),
            timeout=300,
        )

        assert ("caspar", "failed") in instances[0].calls
        assert ("caspar", "timeout") not in instances[0].calls

    @pytest.mark.asyncio
    async def test_show_status_false_skips_display(self, tmp_path, monkeypatch):
        import run_magi

        created: list[int] = []
        monkeypatch.setattr(
            run_magi,
            "StatusDisplay",
            lambda *a, **kw: created.append(1) or _FakeDisplay(),
        )

        async def mock_launch(agent_name, *args, **kwargs):
            return _ok_result(agent_name)

        monkeypatch.setattr(run_magi, "launch_agent", mock_launch)

        await run_magi.run_orchestrator(
            agents_dir=str(tmp_path),
            prompt="test",
            output_dir=str(tmp_path),
            timeout=300,
            show_status=False,
        )

        assert created == []

    @pytest.mark.asyncio
    async def test_cancelled_error_marks_display_failed(self, tmp_path, monkeypatch):
        """W4: CancelledError in an agent must mark its display row as failed."""
        import run_magi

        instances: list[_FakeDisplay] = []
        monkeypatch.setattr(
            run_magi,
            "StatusDisplay",
            lambda *a, **kw: instances.append(_FakeDisplay()) or instances[-1],
        )

        async def mock_launch(agent_name, *args, **kwargs):
            if agent_name == "caspar":
                raise asyncio.CancelledError()
            return _ok_result(agent_name)

        monkeypatch.setattr(run_magi, "launch_agent", mock_launch)

        result = await run_magi.run_orchestrator(
            agents_dir=str(tmp_path),
            prompt="test",
            output_dir=str(tmp_path),
            timeout=300,
        )

        assert ("caspar", "running") in instances[0].calls
        assert ("caspar", "failed") in instances[0].calls
        # caspar's row must not be left in "running" state and must not
        # be marked as "success".
        assert ("caspar", "success") not in instances[0].calls
        assert result.get("degraded") is True

    @pytest.mark.asyncio
    async def test_display_start_failure_falls_through_gracefully(
        self, tmp_path, monkeypatch, capsys
    ):
        """A raised ``display.start()`` must not block the analysis."""
        import run_magi

        class _FailingStartDisplay:
            def __init__(self, *args, **kwargs):
                self.updates: list[tuple[str, str]] = []
                self.stop_called = False

            def update(self, agent: str, state: str) -> None:
                self.updates.append((agent, state))

            async def start(self) -> None:
                raise RuntimeError("simulated start failure")

            async def stop(self) -> None:
                self.stop_called = True

        instances: list[_FailingStartDisplay] = []

        def factory(*args, **kwargs):
            inst = _FailingStartDisplay()
            instances.append(inst)
            return inst

        monkeypatch.setattr(run_magi, "StatusDisplay", factory)

        async def mock_launch(agent_name, *args, **kwargs):
            return _ok_result(agent_name)

        monkeypatch.setattr(run_magi, "launch_agent", mock_launch)

        result = await run_magi.run_orchestrator(
            agents_dir=str(tmp_path),
            prompt="test",
            output_dir=str(tmp_path),
            timeout=300,
        )

        assert result["consensus"]["consensus"] == "STRONG GO"
        assert len(instances) == 1
        # Display was dropped, so stop() is never called and no further
        # ``update()`` calls reach it after the start() failure — the
        # tracked_launch closure must see ``display is None``.
        assert instances[0].stop_called is False
        assert instances[0].updates == [], (
            f"No updates must reach a failed-start display, got {instances[0].updates}"
        )

        captured = capsys.readouterr()
        assert "status display failed to start" in captured.err

    @pytest.mark.asyncio
    async def test_display_update_errors_do_not_mask_original_exception(
        self, tmp_path, monkeypatch
    ):
        """If display.update() raises during shutdown, the real error must win."""
        import run_magi

        class _BrokenDisplay:
            def __init__(self, *args, **kwargs):
                self.stop_called = False

            def update(self, agent: str, state: str) -> None:
                raise RuntimeError("display is broken")

            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                self.stop_called = True

        monkeypatch.setattr(run_magi, "StatusDisplay", _BrokenDisplay)

        async def mock_launch(agent_name, *args, **kwargs):
            if agent_name == "caspar":
                raise ValueError("original failure")
            return _ok_result(agent_name)

        monkeypatch.setattr(run_magi, "launch_agent", mock_launch)

        # The orchestrator must still return (degraded) — the BrokenDisplay
        # update call must not propagate and mask caspar's ValueError.
        result = await run_magi.run_orchestrator(
            agents_dir=str(tmp_path),
            prompt="test",
            output_dir=str(tmp_path),
            timeout=300,
        )
        assert result.get("degraded") is True
        assert "caspar" in result.get("failed_agents", [])


class _FakeTimeoutProc:
    """Fake asyncio subprocess for timeout-path testing.

    ``communicate()`` hangs indefinitely so ``asyncio.wait_for`` fires a
    ``TimeoutError``. ``kill()`` and ``wait()`` record call order so tests
    can verify zombie reaping. ``proc.stderr`` is a prefilled
    :class:`asyncio.StreamReader` so the production code can drain buffered
    diagnostics after killing the process.
    """

    def __init__(
        self,
        stdout_bytes: bytes = b"",
        stderr_bytes: bytes = b"",
    ) -> None:
        self.returncode: int | None = None
        self.kill_called = False
        self.wait_called = False
        self.call_order: list[str] = []
        self.stdout = asyncio.StreamReader()
        self.stdout.feed_data(stdout_bytes)
        self.stdout.feed_eof()
        self.stderr = asyncio.StreamReader()
        self.stderr.feed_data(stderr_bytes)
        self.stderr.feed_eof()
        self.stdin = None

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
        # Hang so wait_for raises TimeoutError.
        await asyncio.sleep(3600)
        return b"", b""

    def kill(self) -> None:
        self.kill_called = True
        self.call_order.append("kill")
        self.returncode = -9

    async def wait(self) -> int | None:
        self.wait_called = True
        self.call_order.append("wait")
        return self.returncode


class TestLaunchAgentTimeoutReaping:
    """A-1: zombie reaping and stderr capture on agent timeout."""

    @pytest.mark.asyncio
    async def test_wait_awaited_after_kill_on_timeout(self, tmp_path, monkeypatch):
        """``proc.kill()`` must be followed by ``await proc.wait()`` to reap."""
        import run_magi

        fake = _FakeTimeoutProc(stderr_bytes=b"")

        async def fake_create(*args, **kwargs):
            return fake

        monkeypatch.setattr(run_magi.asyncio, "create_subprocess_exec", fake_create)
        (tmp_path / "melchior.md").write_text("sys prompt", encoding="utf-8")

        with pytest.raises(TimeoutError):
            await run_magi.launch_agent(
                agent_name="melchior",
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=1,
            )

        assert fake.kill_called, "kill() must be called on timeout"
        assert fake.wait_called, "wait() must be awaited after kill() to reap zombie"
        assert fake.call_order == ["kill", "wait"], (
            f"Order must be kill→wait, got {fake.call_order}"
        )

    @pytest.mark.asyncio
    async def test_stderr_persisted_to_log_on_timeout(self, tmp_path, monkeypatch):
        """Buffered stderr must be written to ``{agent}.stderr.log`` on timeout."""
        import run_magi

        stderr_payload = b"agent started thinking\nmid-computation diag\n"
        fake = _FakeTimeoutProc(stderr_bytes=stderr_payload)

        async def fake_create(*args, **kwargs):
            return fake

        monkeypatch.setattr(run_magi.asyncio, "create_subprocess_exec", fake_create)
        (tmp_path / "melchior.md").write_text("sys prompt", encoding="utf-8")

        with pytest.raises(TimeoutError):
            await run_magi.launch_agent(
                agent_name="melchior",
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=1,
            )

        stderr_log = tmp_path / "melchior.stderr.log"
        assert stderr_log.exists(), (
            "Stderr log must be persisted on timeout for post-mortem diagnosis"
        )
        assert stderr_log.read_bytes() == stderr_payload

    @pytest.mark.asyncio
    async def test_timeout_error_surfaces_stderr_excerpt(self, tmp_path, monkeypatch):
        """TimeoutError message must include stderr excerpt so operators see why."""
        import run_magi

        fake = _FakeTimeoutProc(stderr_bytes=b"Connection refused to upstream API")

        async def fake_create(*args, **kwargs):
            return fake

        monkeypatch.setattr(run_magi.asyncio, "create_subprocess_exec", fake_create)
        (tmp_path / "melchior.md").write_text("sys prompt", encoding="utf-8")

        with pytest.raises(TimeoutError, match="Connection refused"):
            await run_magi.launch_agent(
                agent_name="melchior",
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=1,
            )

    @pytest.mark.asyncio
    async def test_empty_stderr_on_timeout_does_not_create_log(self, tmp_path, monkeypatch):
        """No stderr data ⇒ no empty .stderr.log file should be written."""
        import run_magi

        fake = _FakeTimeoutProc(stderr_bytes=b"")

        async def fake_create(*args, **kwargs):
            return fake

        monkeypatch.setattr(run_magi.asyncio, "create_subprocess_exec", fake_create)
        (tmp_path / "melchior.md").write_text("sys prompt", encoding="utf-8")

        with pytest.raises(TimeoutError):
            await run_magi.launch_agent(
                agent_name="melchior",
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=1,
            )

        assert not (tmp_path / "melchior.stderr.log").exists()


class TestSafeDisplayUpdate:
    """Verify ``_safe_display_update`` swallows display errors during shutdown."""

    def test_none_display_is_noop(self):
        from run_magi import _safe_display_update

        _safe_display_update(None, "melchior", "running")  # must not raise

    def test_exception_is_swallowed(self):
        from run_magi import _safe_display_update

        class _Broken:
            def update(self, agent: str, state: str) -> None:
                raise RuntimeError("broken")

        _safe_display_update(_Broken(), "melchior", "running")  # must not raise

    def test_successful_update_propagates(self):
        from run_magi import _safe_display_update

        class _Recorder:
            def __init__(self):
                self.calls: list[tuple[str, str]] = []

            def update(self, agent: str, state: str) -> None:
                self.calls.append((agent, state))

        rec = _Recorder()
        _safe_display_update(rec, "melchior", "running")
        assert rec.calls == [("melchior", "running")]


class TestBufferedStderrWhile:
    """Structural enforcement of the display-active stderr-quiet invariant (W3)."""

    def test_noop_when_inactive(self):
        """When active=False, sys.stderr is untouched and writes pass through."""
        from run_magi import _buffered_stderr_while

        original = sys.stderr
        with _buffered_stderr_while(active=False):
            assert sys.stderr is original

    def test_buffers_writes_when_active(self, capsys):
        """When active=True, writes are buffered and replayed on context exit."""
        from run_magi import _buffered_stderr_while

        with _buffered_stderr_while(active=True):
            print("line 1", file=sys.stderr)
            print("line 2", file=sys.stderr)
            # Nothing should have reached real stderr yet.
            captured_mid = capsys.readouterr()
            assert captured_mid.err == ""

        # After context exit, buffered content is replayed.
        captured_after = capsys.readouterr()
        assert "line 1" in captured_after.err
        assert "line 2" in captured_after.err

    def test_restores_original_stderr_on_exit(self):
        """The original sys.stderr reference must be restored after the context."""
        from run_magi import _buffered_stderr_while

        original = sys.stderr
        with _buffered_stderr_while(active=True):
            assert sys.stderr is not original
        assert sys.stderr is original

    def test_proxies_non_write_attributes(self):
        """The shim must proxy encoding/isatty/fileno to the real stderr."""
        from run_magi import _buffered_stderr_while

        real_encoding = getattr(sys.stderr, "encoding", None)
        with _buffered_stderr_while(active=True):
            # isatty() and encoding come from the real stderr via __getattr__.
            assert sys.stderr.encoding == real_encoding
            # The shim is not the real stream.
            assert sys.stderr is not sys.__stderr__

    def test_restores_stderr_even_on_exception(self):
        """Context manager must restore stderr when the body raises."""
        from run_magi import _buffered_stderr_while

        original = sys.stderr
        with pytest.raises(RuntimeError):
            with _buffered_stderr_while(active=True):
                raise RuntimeError("boom")
        assert sys.stderr is original

    def test_binary_buffer_writes_are_intercepted(self, capsys):
        """Writes through ``sys.stderr.buffer.write`` must also be buffered."""
        from run_magi import _buffered_stderr_while

        with _buffered_stderr_while(active=True):
            shim_buffer = getattr(sys.stderr, "buffer", None)
            if shim_buffer is None:
                pytest.skip("pytest capture stream has no .buffer attribute")
            shim_buffer.write(b"binary diag line\n")
            captured_mid = capsys.readouterr()
            assert captured_mid.err == ""

        captured_after = capsys.readouterr()
        assert "binary diag line" in captured_after.err

    def test_shim_buffer_attribute_exists_when_real_has_buffer(self):
        """The shim must expose a ``.buffer`` shim when the real stderr has one."""
        from run_magi import _BinaryStderrBufferShim, _StderrBufferShim

        class _FakeBinary:
            def write(self, data: bytes) -> int:
                return len(data)

            def flush(self) -> None:
                pass

        class _FakeStderr:
            def __init__(self):
                self.buffer = _FakeBinary()

            def write(self, data: str) -> int:
                return len(data)

            def flush(self) -> None:
                pass

        text_buffer: list[str] = []
        shim = _StderrBufferShim(_FakeStderr(), text_buffer)
        assert shim.buffer is not None
        assert isinstance(shim.buffer, _BinaryStderrBufferShim)

        shim.buffer.write(b"hello\n")
        assert text_buffer == ["hello\n"]

    def test_shim_buffer_none_when_real_has_no_buffer(self):
        """When the real stderr lacks ``.buffer``, the shim's ``.buffer`` is None."""
        import io

        from run_magi import _StderrBufferShim

        text_buffer: list[str] = []
        shim = _StderrBufferShim(io.StringIO(), text_buffer)
        assert shim.buffer is None

    @pytest.mark.asyncio
    async def test_orchestrator_buffers_stderr_during_gather(self, tmp_path, monkeypatch, capsys):
        """End-to-end: writes from tracked tasks are buffered, then flushed."""
        import run_magi

        monkeypatch.setattr(run_magi, "StatusDisplay", lambda *a, **kw: _FakeDisplay())

        async def mock_launch(agent_name, *args, **kwargs):
            # Simulate a task that writes to stderr mid-run.
            print(f"diag from {agent_name}", file=sys.stderr)
            return _ok_result(agent_name)

        monkeypatch.setattr(run_magi, "launch_agent", mock_launch)

        await run_magi.run_orchestrator(
            agents_dir=str(tmp_path),
            prompt="test",
            output_dir=str(tmp_path),
            timeout=300,
        )

        captured = capsys.readouterr()
        # Diagnostic writes must have been replayed after the display stopped.
        assert "diag from melchior" in captured.err
        assert "diag from balthasar" in captured.err
        assert "diag from caspar" in captured.err
