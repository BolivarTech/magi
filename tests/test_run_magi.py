# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-04-01
"""Tests for run_magi.py — async Python orchestrator."""

import os
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
