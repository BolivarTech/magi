# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-04-01
"""Tests for run_magi.py — async Python orchestrator."""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any
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

    def test_default_model_for_code_review_is_opus(self):
        """code-review keeps opus as the default — dense technical reasoning
        warrants the cost. Backward-compatible with the 2.0.x-2.2.x default.
        """
        from run_magi import parse_args

        args = parse_args(["code-review", "input.py"])
        assert args.model == "opus"

    def test_default_model_for_design_is_opus(self):
        """design defaults to opus — multi-level abstraction (architecture,
        scaling, hidden coupling) where smaller models drop confidence sharply.
        """
        from run_magi import parse_args

        args = parse_args(["design", "spec.md"])
        assert args.model == "opus"

    def test_default_model_for_analysis_is_opus(self):
        """analysis defaults to opus.

        2.2.3 (released 2026-04-25) switched analysis to sonnet for cost
        relief. 2.2.5 (this release, 2026-04-26) reverts based on
        production evidence: Caspar (the most-output agent by design,
        consistently producing 4-7K output tokens vs Mel/Bal at 2-3K)
        failed in ≥33% of sbtdd Loop verifications under the sonnet
        default. That is an order of magnitude above the 3.3% design
        assumption documented in CLAUDE.md "Post-release hardening".

        The 2.2.4 retry could not recover Caspar consistently because
        the failure was structural (output-ceiling pressure on sonnet's
        ~8K max), not stochastic. The second attempt with the same
        model hit the same ceiling. Reverting analysis to opus restores
        the 32K max-output budget and gives Caspar headroom.

        The 2.2.4 retry path remains active for all three modes; only
        the per-mode default for analysis flips back to opus.
        ``code-review`` and ``design`` were never on sonnet.
        """
        from run_magi import parse_args

        args = parse_args(["analysis", "input.txt"])
        assert args.model == "opus"

    def test_explicit_model_overrides_mode_default(self):
        """``--model X`` always wins over any per-mode default. Without this,
        operators who want to force opus for analysis (or haiku for code-review)
        would have no way to do it.
        """
        from run_magi import parse_args

        # sonnet for analysis (override the opus default re-established in 2.2.5)
        args = parse_args(["analysis", "input.txt", "--model", "sonnet"])
        assert args.model == "sonnet"

        # haiku for code-review (override the opus default)
        args = parse_args(["code-review", "input.py", "--model", "haiku"])
        assert args.model == "haiku"

        # sonnet for design (override the opus default)
        args = parse_args(["design", "spec.md", "--model", "sonnet"])
        assert args.model == "sonnet"

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

    def test_keep_runs_default(self):
        """Default --keep-runs value lines up with MAX_HISTORY_RUNS."""
        from run_magi import MAX_HISTORY_RUNS, parse_args

        args = parse_args(["code-review", "input.py"])
        assert args.keep_runs == MAX_HISTORY_RUNS

    def test_keep_runs_zero_rejected(self):
        """``--keep-runs 0`` is ambiguous and must be rejected at argparse.

        Regression for the v2.1.1 fix: previously, ``--keep-runs 0`` was
        silently interpreted as ``cleanup_old_runs(-1)`` ("disable
        cleanup"), producing unbounded accumulation — the opposite of
        what a user passing 0 would reasonably expect. The CLI now
        rejects 0 with an error that points to ``--keep-runs 1``
        (wipe-all) or ``--keep-runs -1`` (disable) as the disambiguating
        replacements.
        """
        from run_magi import parse_args

        with pytest.raises(SystemExit):
            parse_args(["code-review", "input.py", "--keep-runs", "0"])

    def test_keep_runs_negative_accepted(self):
        """``--keep-runs -1`` is the explicit "disable cleanup" value."""
        from run_magi import parse_args

        args = parse_args(["code-review", "input.py", "--keep-runs", "-1"])
        assert args.keep_runs == -1

    def test_keep_runs_one_accepted(self):
        """``--keep-runs 1`` is the explicit "wipe all prior" value."""
        from run_magi import parse_args

        args = parse_args(["code-review", "input.py", "--keep-runs", "1"])
        assert args.keep_runs == 1


class TestModeModelLockstepInvariant:
    """Pin the lockstep invariant claimed by the 2.2.3 docstrings.

    `MODE_DEFAULT_MODELS` (in models.py) and the inline comment in
    `run_magi.parse_args` both promise the test suite enforces:

      * Every key of MODE_DEFAULT_MODELS is a valid analysis mode.
      * Every value of MODE_DEFAULT_MODELS is a registered model.

    Without these tests, a future contributor adding a fourth mode to
    VALID_MODES (or removing one from MODE_DEFAULT_MODELS) would slip
    past CI and surface as a runtime KeyError on the
    `MODE_DEFAULT_MODELS[args.mode]` lookup. These tests convert the
    docstring promise into a regression-blocking guarantee.
    """

    def test_every_mode_has_a_default_model(self):
        from models import MODE_DEFAULT_MODELS
        from run_magi import VALID_MODES

        assert set(MODE_DEFAULT_MODELS.keys()) == set(VALID_MODES), (
            f"MODE_DEFAULT_MODELS keys {sorted(MODE_DEFAULT_MODELS.keys())} "
            f"must equal VALID_MODES {sorted(VALID_MODES)} — adding a mode "
            f"requires adding its default; removing a mode requires removing "
            f"its default. The post-parse resolution at run_magi.parse_args "
            f"depends on this set equality holding."
        )

    def test_every_mode_default_is_a_registered_model(self):
        from models import MODE_DEFAULT_MODELS, MODEL_IDS

        unknown = set(MODE_DEFAULT_MODELS.values()) - set(MODEL_IDS.keys())
        assert not unknown, (
            f"MODE_DEFAULT_MODELS contains short names not in MODEL_IDS: "
            f"{sorted(unknown)}. Every default must resolve through "
            f"resolve_model() at orchestrator startup, so the set of "
            f"values must be a subset of MODEL_IDS keys."
        )


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

    def test_negative_keep_disables_cleanup(self, tmp_path):
        """keep < 0 should not scan or delete anything."""
        from run_magi import cleanup_old_runs

        with patch("temp_dirs.tempfile.gettempdir", return_value=str(tmp_path)):
            magi_dir = tmp_path / "magi-run-abc123"
            magi_dir.mkdir()
            cleanup_old_runs(-1)
            assert magi_dir.exists()

    def test_keep_zero_deletes_all_magi_dirs(self, tmp_path):
        """keep == 0 should remove every magi-run-* dir (reserves slot for new run)."""
        from run_magi import cleanup_old_runs

        magi_dirs = []
        for i in range(3):
            d = tmp_path / f"magi-run-{i:04d}"
            d.mkdir()
            magi_dirs.append(d)

        with patch("temp_dirs.tempfile.gettempdir", return_value=str(tmp_path)):
            cleanup_old_runs(0)

        for d in magi_dirs:
            assert not d.exists(), f"{d} should have been deleted"

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

        with patch("temp_dirs.tempfile.gettempdir", return_value=str(tmp_path)):
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

        with patch("temp_dirs.tempfile.gettempdir", return_value=str(tmp_path)):
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

        with patch("temp_dirs.tempfile.gettempdir", return_value=str(tmp_path)):
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

        with patch("temp_dirs.tempfile.gettempdir", return_value=str(tmp_path)):
            cleanup_old_runs(1)

        assert (tmp_path / "other-dir").exists()
        assert (tmp_path / "readme.txt").exists()

    def test_cleanup_works_when_tmpdir_itself_is_symlink(self, tmp_path, monkeypatch):
        """D-1a: a symlinked TMPDIR must not disable cleanup entirely.

        On macOS ``/tmp`` is a symlink to ``/private/tmp``; ``gettempdir()``
        returns ``/tmp`` but ``os.path.realpath(entry.path)`` resolves
        through the root symlink, so every candidate appears to live
        outside the ``/tmp/`` prefix and the traversal guard skips
        everything. The fix is to resolve the temp root the same way
        before building the safe prefix.

        This test simulates the scenario by monkeypatching ``realpath``
        so it runs identically on platforms without symlink support
        (e.g. Windows under a non-admin pytest run).
        """
        import temp_dirs
        from run_magi import cleanup_old_runs

        older = tmp_path / "magi-run-0001"
        older.mkdir()
        os.utime(older, (1000, 1000))
        newer = tmp_path / "magi-run-0002"
        newer.mkdir()
        os.utime(newer, (2000, 2000))

        advertised_root = str(tmp_path).replace(os.sep + "tmp", os.sep + "resolved_tmp", 1)
        if advertised_root == str(tmp_path):
            # Fallback: prepend a fake segment so realpath differs from the advertised path.
            advertised_root = str(tmp_path) + "_advertised"
        real_root_str = str(tmp_path)

        real_realpath = os.path.realpath

        def fake_realpath(path: str) -> str:
            # Rewrite the advertised (symlinked) root to the real one so
            # both the candidate entries and — crucially — the temp
            # root itself resolve to the same physical directory.
            if path == advertised_root or path.startswith(advertised_root + os.sep):
                return real_realpath(real_root_str + path[len(advertised_root) :])
            return real_realpath(path)

        monkeypatch.setattr(temp_dirs.os.path, "realpath", fake_realpath)
        monkeypatch.setattr(temp_dirs.tempfile, "gettempdir", lambda: advertised_root)

        # Rewrite scandir so it iterates the real tmp_path when asked
        # for the advertised symlinked root. This mirrors the OS-level
        # behavior on macOS: scandir follows the symlink transparently.
        real_scandir = os.scandir

        def fake_scandir(path):
            if path == advertised_root:
                return real_scandir(real_root_str)
            return real_scandir(path)

        monkeypatch.setattr(temp_dirs.os, "scandir", fake_scandir)

        cleanup_old_runs(1)

        assert newer.exists(), "Newest magi-run dir must be retained"
        assert not older.exists(), (
            "Oldest magi-run dir must be deleted even when TMPDIR is a symlink to its realpath"
        )

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

        with patch("temp_dirs.tempfile.gettempdir", return_value=str(tmp_path)):
            cleanup_old_runs(0)
            # keep=0 disables, use keep=1 with 2 dirs to trigger cleanup
            real_dir = tmp_path / "magi-run-real"
            real_dir.mkdir()
            os.utime(real_dir, (2000, 2000))
            os.utime(symlink_path, (1000, 1000))
            cleanup_old_runs(1)

        # Outside dir should not be deleted
        assert outside_dir.exists()


class TestStderrShimModule:
    """C-2: the stderr-buffering machinery lives in its own module.

    ``_StderrBufferShim``, ``_BinaryStderrBufferShim``, and the
    ``_buffered_stderr_while`` context manager were embedded in
    run_magi.py, bloating the orchestrator. Extracting them to
    stderr_shim.py keeps run_magi focused on orchestration and makes
    the shim machinery independently testable.
    """

    def test_stderr_shim_module_importable(self):
        """The stderr_shim module must be importable by its short name."""
        import importlib

        module = importlib.import_module("stderr_shim")
        assert module is not None

    def test_stderr_shim_exposes_expected_symbols(self):
        """stderr_shim must export the three shim primitives."""
        import stderr_shim

        assert hasattr(stderr_shim, "_StderrBufferShim")
        assert hasattr(stderr_shim, "_BinaryStderrBufferShim")
        assert hasattr(stderr_shim, "_buffered_stderr_while")

    def test_run_magi_does_not_reexport_private_shim_names(self):
        """Regression (v2.1.1): ``run_magi`` must not re-export the
        underscored shim names.

        The earlier pattern ``__all__ = [..., "_StderrBufferShim", ...]``
        was contradictory: an underscore says "private", yet ``__all__``
        says "part of the star-import contract". Tests that need the
        shims import them from ``stderr_shim`` directly — the single
        owner of that API.
        """
        import run_magi

        for private in ("_StderrBufferShim", "_BinaryStderrBufferShim"):
            assert not hasattr(run_magi, private), (
                f"run_magi must not re-export {private}; import from stderr_shim instead."
            )
        # ``_buffered_stderr_while`` is still imported for internal use,
        # so it is reachable as an attribute, but it must not appear in
        # ``__all__`` — asserted separately in
        # ``TestAllDoesNotExportPrivateShimNames``.


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
        # Fake pid so the Windows tree-kill path in ``_reap_and_drain_stderr``
        # has something to pass to ``taskkill``. Test fixtures monkeypatch
        # ``subprocess.run`` so the call is inert.
        self.pid = 999_000

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

    @pytest.fixture(autouse=True)
    def _stub_taskkill(self, monkeypatch):
        """Stub ``subprocess.run`` so the Windows tree-kill path in
        ``reap_and_drain_stderr`` does not invoke the real ``taskkill``
        against a fake pid and slow each test down by several seconds.
        """
        import subprocess_utils

        def _noop_run(*args, **kwargs):
            class _Completed:
                returncode = 0

            return _Completed()

        monkeypatch.setattr(subprocess_utils.subprocess, "run", _noop_run)

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
    async def test_write_stderr_log_oserror_does_not_mask_timeout(self, tmp_path, monkeypatch):
        """D-1b: OSError from the stderr-log write must not shadow TimeoutError.

        If the disk is full or read-only when we try to persist buffered
        diagnostics on the timeout path, the caller must still see the
        original ``TimeoutError`` — swallowing it behind an ``OSError``
        hides the real cause from the orchestrator's failure summary.
        """
        import run_magi

        fake = _FakeTimeoutProc(stderr_bytes=b"partial diagnostics before hang")

        async def fake_create(*args, **kwargs):
            return fake

        monkeypatch.setattr(run_magi.asyncio, "create_subprocess_exec", fake_create)
        (tmp_path / "melchior.md").write_text("sys prompt", encoding="utf-8")

        def failing_write(output_dir, agent_name, data):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(run_magi, "_write_stderr_log", failing_write)

        with pytest.raises(TimeoutError, match="timed out after"):
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


_FAKE_AGENT_JSON = (
    '{"agent": "melchior", "verdict": "approve", "confidence": 0.8, '
    '"summary": "ok", "reasoning": "looks fine", "findings": [], '
    '"recommendation": "merge"}'
)
# The ``claude -p --output-format json`` envelope wraps the agent JSON
# as a string under ``result`` — match that shape so the real
# ``parse_agent_output`` pipeline accepts the mock.
_FAKE_CLAUDE_ENVELOPE = (
    '{"result": "{\\"agent\\": \\"melchior\\", \\"verdict\\": \\"approve\\", '
    '\\"confidence\\": 0.8, \\"summary\\": \\"ok\\", \\"reasoning\\": '
    '\\"looks fine\\", \\"findings\\": [], \\"recommendation\\": \\"merge\\"}"}'
).encode("utf-8")


class _FakeSuccessProc:
    """Fake asyncio subprocess that simulates a successful agent run.

    Used by regression tests that need the full happy path through
    ``launch_agent`` without spawning the real ``claude`` CLI.
    """

    def __init__(
        self,
        stdout_bytes: bytes = _FAKE_CLAUDE_ENVELOPE,
        stderr_bytes: bytes = b"some stderr",
    ) -> None:
        self._stdout = stdout_bytes
        self._stderr = stderr_bytes
        self.returncode: int | None = None
        self.stdin = None

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
        self.returncode = 0
        return self._stdout, self._stderr

    def kill(self) -> None:  # pragma: no cover — never called on success path
        pass

    async def wait(self) -> int | None:  # pragma: no cover
        return self.returncode


class TestLaunchAgentSuccessStderrLog:
    """Regression (v2.1.1): success-path stderr log write must not mask
    an otherwise-successful agent when disk/permission errors occur.
    """

    @pytest.mark.asyncio
    async def test_success_path_oserror_does_not_mask_result(self, tmp_path, monkeypatch, capsys):
        """D-1c: OSError from the stderr-log write on the success path
        must be caught and logged, not propagated — the agent's parsed
        JSON is already valid at that point.

        Pre-2.1.1, the success-path ``_write_stderr_log`` call was bare;
        a disk-full or antivirus-lock error on Windows would bubble up
        from ``launch_agent`` and be reported as an agent failure in
        ``tracked_launch`` even though the agent itself succeeded. The
        fix mirrors the timeout-path ``try/except OSError`` pattern and
        is covered by this test.
        """
        import run_magi

        fake = _FakeSuccessProc(stderr_bytes=b"diagnostic line")

        async def fake_create(*args, **kwargs):
            return fake

        def failing_write(output_dir, agent_name, data):
            raise OSError(13, "Permission denied")

        monkeypatch.setattr(run_magi.asyncio, "create_subprocess_exec", fake_create)
        monkeypatch.setattr(run_magi, "_write_stderr_log", failing_write)
        (tmp_path / "melchior.md").write_text("sys prompt", encoding="utf-8")

        result = await run_magi.launch_agent(
            agent_name="melchior",
            agents_dir=str(tmp_path),
            prompt="test",
            output_dir=str(tmp_path),
            timeout=5,
        )
        assert result["agent"] == "melchior"
        assert result["verdict"] == "approve"
        captured = capsys.readouterr()
        assert "Failed to persist" in captured.err
        assert "melchior.stderr.log" in captured.err


class TestTaskkillTimeoutBudget:
    """Regression (v2.1.1): ``_TASKKILL_TIMEOUT`` must be independent of
    ``_PROC_WAIT_REAP_TIMEOUT`` so a slow ``taskkill`` does not consume
    the ``proc.wait()`` budget and fire a misleading orphan warning.
    """

    def test_taskkill_timeout_is_separate_constant(self):
        """The two timeouts are distinct module-level constants and
        operators can tune them without conflating the budgets.
        """
        from subprocess_utils import PROC_WAIT_REAP_TIMEOUT, TASKKILL_TIMEOUT

        # Both are floats > 0 — the exact values may change over time,
        # but they must live in separate constants so one slow call does
        # not poison the other's observability.
        assert isinstance(TASKKILL_TIMEOUT, float)
        assert isinstance(PROC_WAIT_REAP_TIMEOUT, float)
        assert TASKKILL_TIMEOUT > 0
        assert PROC_WAIT_REAP_TIMEOUT > 0

    def test_windows_kill_tree_uses_taskkill_timeout(self, monkeypatch):
        """``windows_kill_tree`` must pass ``TASKKILL_TIMEOUT`` to
        ``subprocess.run``, not ``PROC_WAIT_REAP_TIMEOUT`` — otherwise
        collapsing the two constants back into one would pass silently.
        """
        import sys as _sys

        if _sys.platform != "win32":
            pytest.skip("Windows-only path")

        import subprocess_utils

        captured: dict = {}

        def fake_run(argv, **kwargs):
            captured.update(kwargs)

            class _Completed:
                returncode = 0

            return _Completed()

        monkeypatch.setattr(subprocess_utils.subprocess, "run", fake_run)
        subprocess_utils.windows_kill_tree(54321)
        assert captured.get("timeout") == subprocess_utils.TASKKILL_TIMEOUT


class TestAllDoesNotExportPrivateShimNames:
    """Regression (v2.1.1): ``__all__`` must not expose underscore-prefixed
    names from ``stderr_shim`` — the shims are private to that module
    and tests should import them from ``stderr_shim`` directly.
    """

    def test_all_has_no_underscore_entries(self):
        from run_magi import __all__

        underscored = [name for name in __all__ if name.startswith("_")]
        assert not underscored, (
            f"__all__ must not expose private names: {underscored!r}. "
            "Tests needing the shims should import from stderr_shim."
        )

    def test_all_exposes_public_api(self):
        """The public API kept in __all__ must still be reachable."""
        from run_magi import __all__

        assert "MODEL_IDS" in __all__
        assert "VALID_MODELS" in __all__
        assert "resolve_model" in __all__


class TestSafeDisplayUpdate:
    """Verify ``_safe_display_update`` swallows display errors during shutdown."""

    def test_none_display_is_noop(self):
        from run_magi import _DisplayLogGate, _safe_display_update

        _safe_display_update(None, "melchior", "running", _DisplayLogGate())  # must not raise

    def test_exception_is_swallowed(self):
        from run_magi import _DisplayLogGate, _safe_display_update

        class _Broken:
            def update(self, agent: str, state: str) -> None:
                raise RuntimeError("broken")

        _safe_display_update(_Broken(), "melchior", "running", _DisplayLogGate())

    def test_first_exception_logged_subsequent_silent(self, capsys):
        """A broken display must surface its first error to stderr so the
        operator knows the live tree is blind, but subsequent errors stay
        silent to prevent the redraw path from flooding the log on every
        tick. The real shutdown signal from the caller is still preserved
        because ``_safe_display_update`` never re-raises."""
        from run_magi import _DisplayLogGate, _safe_display_update

        gate = _DisplayLogGate()

        class _Broken:
            def update(self, agent: str, state: str) -> None:
                raise RuntimeError("boom")

        broken = _Broken()
        _safe_display_update(broken, "melchior", "running", gate)
        _safe_display_update(broken, "balthasar", "running", gate)
        _safe_display_update(broken, "caspar", "running", gate)

        captured = capsys.readouterr()
        assert captured.err.count("status display") == 1, (
            "First failure must be logged exactly once; subsequent failures must stay silent."
        )
        assert "boom" in captured.err

    def test_fresh_gate_per_run_rearms_log(self, capsys):
        """Each run gets a new ``_DisplayLogGate``, so the first failure of
        every run surfaces to stderr. Without per-run isolation a long-lived
        host that reuses the module would never see display failures after
        the first run.
        """
        from run_magi import _DisplayLogGate, _safe_display_update

        class _Broken:
            def update(self, agent: str, state: str) -> None:
                raise RuntimeError("boom")

        broken = _Broken()
        # Run 1.
        _safe_display_update(broken, "melchior", "running", _DisplayLogGate())
        # Run 2 (separate gate).
        _safe_display_update(broken, "melchior", "running", _DisplayLogGate())

        captured = capsys.readouterr()
        assert captured.err.count("status display") == 2, (
            "A fresh gate per run must re-arm the first-failure log."
        )

    def test_successful_update_propagates(self):
        from run_magi import _DisplayLogGate, _safe_display_update

        class _Recorder:
            def __init__(self):
                self.calls: list[tuple[str, str]] = []

            def update(self, agent: str, state: str) -> None:
                self.calls.append((agent, state))

        rec = _Recorder()
        _safe_display_update(rec, "melchior", "running", _DisplayLogGate())
        assert rec.calls == [("melchior", "running")]

    def test_base_exception_is_swallowed(self):
        """The helper's contract explicitly names ``CancelledError`` and
        ``KeyboardInterrupt`` (both ``BaseException`` subclasses) as
        shutdown-path failures it must not propagate. ``tracked_launch``
        is wrapped in ``except BaseException`` and relies on this helper
        returning normally so the outer ``raise`` re-raises the *original*
        signal instead of whatever the display raised on the way down.
        """
        import asyncio

        from run_magi import _DisplayLogGate, _safe_display_update

        gate = _DisplayLogGate()

        class _CancelledRaiser:
            def update(self, agent: str, state: str) -> None:
                raise asyncio.CancelledError("display cancelled mid-shutdown")

        class _SystemExitRaiser:
            def update(self, agent: str, state: str) -> None:
                raise SystemExit(2)

        # Neither call may propagate — the documented contract says the
        # helper swallows shutdown-path failures so the caller's own
        # ``raise`` preserves the original exception.
        _safe_display_update(_CancelledRaiser(), "melchior", "failed", gate)
        _safe_display_update(_SystemExitRaiser(), "caspar", "failed", gate)


class TestReapAndDrainStderr:
    """Verify timeout warning when a killed subprocess fails to exit."""

    def test_warns_when_proc_wait_times_out(self, capsys, monkeypatch):
        """If ``proc.wait()`` still hasn't returned within
        ``_PROC_WAIT_REAP_TIMEOUT`` seconds after ``kill()``, the caller
        must emit a warning to stderr so an operator can notice an
        orphaned subprocess (Windows child-process-tree case). The
        function must still return the best-effort stderr buffer and
        must not raise."""
        import asyncio

        from subprocess_utils import PROC_WAIT_REAP_TIMEOUT, reap_and_drain_stderr

        class _FakeStderr:
            async def read(self) -> bytes:
                return b""

        class _FakeProc:
            pid = 9999
            stderr = _FakeStderr()
            kill_called = False

            def kill(self) -> None:
                type(self).kill_called = True

            async def wait(self) -> int:
                await asyncio.sleep(10)  # simulate hang
                return 0

        async def _fake_wait_for(awaitable, timeout):
            # Consume the coroutine so asyncio doesn't warn about it,
            # then raise to simulate the reap timeout on the wait() call.
            if timeout == PROC_WAIT_REAP_TIMEOUT:
                if asyncio.iscoroutine(awaitable):
                    awaitable.close()
                raise asyncio.TimeoutError
            return await awaitable

        monkeypatch.setattr("subprocess_utils.asyncio.wait_for", _fake_wait_for)

        proc = _FakeProc()
        result = asyncio.run(reap_and_drain_stderr(proc))  # type: ignore[arg-type]

        assert result == b""
        assert _FakeProc.kill_called is True
        captured = capsys.readouterr()
        assert "9999" in captured.err, (
            "Warning must name the unreaped subprocess so operators can identify the orphan."
        )
        assert "did not exit" in captured.err or "orphan" in captured.err.lower()

    def test_windows_invokes_taskkill_tree(self, monkeypatch):
        """On Windows, the reap path must also issue ``taskkill /F /T /PID``
        so orphan child processes (a real hazard when ``claude`` spawns
        its own helpers) do not survive a MAGI timeout.

        The existing ``proc.kill()`` is kept for signalling, and
        ``taskkill`` is invoked in addition to it — not as a replacement
        — because ``taskkill`` may fail if the binary is missing or a
        timeout cuts it off. Calling both makes the reap more robust
        without regressing the single-process case.
        """
        import asyncio
        import sys as _sys

        if _sys.platform != "win32":
            pytest.skip("Windows-only path")

        import subprocess_utils

        recorded_argv: list[list[str]] = []

        def fake_run(argv, **kwargs):
            recorded_argv.append(list(argv))

            class _Completed:
                returncode = 0

            return _Completed()

        monkeypatch.setattr(subprocess_utils.subprocess, "run", fake_run)

        class _FakeStderr:
            async def read(self) -> bytes:
                return b""

        class _FakeProc:
            pid = 12345
            stderr = _FakeStderr()

            def kill(self) -> None:
                pass

            async def wait(self) -> int:
                return 0

        asyncio.run(subprocess_utils.reap_and_drain_stderr(_FakeProc()))  # type: ignore[arg-type]

        assert any(
            argv[:4] == ["taskkill", "/F", "/T", "/PID"] and argv[4] == "12345"
            for argv in recorded_argv
        ), f"Expected taskkill invocation for pid 12345, recorded: {recorded_argv!r}"

    def test_windows_taskkill_runs_before_proc_kill(self, monkeypatch):
        """On Windows, ``taskkill /F /T /PID`` must be invoked BEFORE
        ``proc.kill()``. Calling ``proc.kill()`` first issues
        ``TerminateProcess`` against the parent, after which the
        kernel may have torn down the parent-child relationship that
        ``taskkill /T`` walks to enumerate descendants — leaving the
        orphan window the function exists to close still open.

        This is a regression guard: pre-2.1.2 the order was inverted
        and the tree-kill was effectively a no-op for child processes
        the ``claude`` CLI had spawned.
        """
        import sys as _sys

        if _sys.platform != "win32":
            pytest.skip("Windows-only path")

        import subprocess_utils

        call_order: list[str] = []

        def fake_run(argv, **kwargs):
            call_order.append("taskkill")

            class _Completed:
                returncode = 0

            return _Completed()

        monkeypatch.setattr(subprocess_utils.subprocess, "run", fake_run)

        class _FakeStderr:
            async def read(self) -> bytes:
                return b""

        class _FakeProc:
            pid = 99999
            stderr = _FakeStderr()

            def kill(self) -> None:
                call_order.append("proc_kill")

            async def wait(self) -> int:
                return 0

        asyncio.run(subprocess_utils.reap_and_drain_stderr(_FakeProc()))  # type: ignore[arg-type]

        assert call_order, "expected at least one of taskkill / proc_kill to fire"
        assert call_order[0] == "taskkill", (
            f"taskkill must run before proc.kill(); recorded order: {call_order!r}"
        )
        assert "proc_kill" in call_order, (
            "proc.kill() must still be invoked after the tree-kill so the "
            "asyncio.subprocess wrapper observes the exit cleanly."
        )


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
        from stderr_shim import _BinaryStderrBufferShim, _StderrBufferShim

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

        from stderr_shim import _StderrBufferShim

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

    def test_replay_oserror_does_not_mask_body_exception(self):
        """If the buffered-stderr replay raises ``OSError`` (the real
        stderr is closed, the parent pipe is dead, the file descriptor
        is gone), the original exception in flight from the body must
        propagate — the write failure during cleanup must not shadow
        the root cause.

        Pre-2.1.2 the ``finally`` clause did ``saved.write(...);
        saved.flush()`` unguarded. A ``BrokenPipeError`` during replay
        would raise out of the context manager and overwrite the body's
        exception, hiding the real failure from the operator.
        """
        from stderr_shim import _buffered_stderr_while

        class _BrokenStderr:
            encoding = "utf-8"
            buffer = None

            def write(self, data: str) -> int:
                raise BrokenPipeError("pipe closed during replay")

            def flush(self) -> None:
                pass

            def isatty(self) -> bool:
                return False

        saved = sys.stderr
        sys.stderr = _BrokenStderr()  # type: ignore[assignment]
        try:
            with pytest.raises(RuntimeError, match="root cause"):
                with _buffered_stderr_while(active=True):
                    print("buffered diagnostic", file=sys.stderr)
                    raise RuntimeError("root cause")
        finally:
            sys.stderr = saved

    def test_replay_oserror_alone_is_swallowed(self):
        """When the body succeeds but the replay raises ``OSError``,
        the context manager must exit cleanly. Re-raising the write
        failure from a cleanup-only path would crash the orchestrator
        on the way out for what is purely a diagnostics-delivery
        problem.
        """
        from stderr_shim import _buffered_stderr_while

        class _BrokenStderr:
            encoding = "utf-8"
            buffer = None

            def write(self, data: str) -> int:
                raise OSError(32, "Broken pipe")

            def flush(self) -> None:
                pass

            def isatty(self) -> bool:
                return False

        saved = sys.stderr
        sys.stderr = _BrokenStderr()  # type: ignore[assignment]
        try:
            with _buffered_stderr_while(active=True):
                print("diag that will fail to replay", file=sys.stderr)
        finally:
            sys.stderr = saved


class TestSingleShotRetry:
    """2.2.0: single-shot retry when an agent fails schema validation.

    Contract driven by these tests:

    * When :func:`launch_agent` raises :class:`ValidationError`,
      :func:`run_orchestrator` retries that specific agent **once** with
      corrective feedback appended to the prompt.
    * Retry fires **only** on :class:`ValidationError`. ``TimeoutError``,
      ``RuntimeError``, ``ValueError``, ``asyncio.CancelledError``, and any
      other exception flow through the existing degraded-mode path
      unchanged.
    * Each attempt receives the full ``--timeout`` budget. The retry is
      not given a reduced ceiling, so operators never see a doubled wall
      clock but always see the full configured per-attempt budget.
    * A ``retrying`` display state is emitted between ``running`` and the
      terminal state (``success`` / ``failed``) for the retried agent.
    * If the retry succeeds, the run completes with full 3-agent
      consensus and ``degraded`` is **not** set.
    * If the retry also raises ``ValidationError`` (or any other
      exception), the agent is dropped and the run continues on the
      surviving agents under the pre-existing 2-agent minimum rule.
    """

    @staticmethod
    def _valid(agent: str) -> dict[str, Any]:
        """Helper: build a schema-valid agent output dict."""
        return {
            "agent": agent,
            "verdict": "approve",
            "confidence": 0.85,
            "summary": f"{agent} OK",
            "reasoning": "Fine",
            "findings": [],
            "recommendation": "Merge",
        }

    @pytest.mark.asyncio
    async def test_schema_failure_triggers_retry_success(self, tmp_path):
        """First call raises ValidationError, second call succeeds.

        The orchestrator must retry the single failing agent and emerge
        with a full 3-agent consensus, no ``degraded`` flag set.
        """
        from run_magi import run_orchestrator
        from validate import ValidationError

        call_counts = {"melchior": 0, "balthasar": 0, "caspar": 0}

        async def mock_launch(agent_name, agents_dir, prompt, output_dir, timeout, model="opus"):
            call_counts[agent_name] += 1
            if agent_name == "caspar" and call_counts[agent_name] == 1:
                raise ValidationError("missing keys: ['recommendation']")
            return TestSingleShotRetry._valid(agent_name)

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            result = await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
            )
            assert result.get("degraded") is not True, (
                "retry that succeeds must not leave degraded flag set"
            )
            assert len(result["agents"]) == 3
            assert call_counts["caspar"] == 2, "caspar must be retried exactly once"
            assert call_counts["melchior"] == 1, "melchior must not be retried"
            assert call_counts["balthasar"] == 1, "balthasar must not be retried"

    @pytest.mark.asyncio
    async def test_retry_also_fails_degraded_mode(self, tmp_path):
        """Both attempts raise ValidationError → agent dropped, degraded=True."""
        from run_magi import run_orchestrator
        from validate import ValidationError

        call_counts = {"melchior": 0, "balthasar": 0, "caspar": 0}

        async def mock_launch(agent_name, agents_dir, prompt, output_dir, timeout, model="opus"):
            call_counts[agent_name] += 1
            if agent_name == "caspar":
                raise ValidationError("missing keys: ['recommendation']")
            return TestSingleShotRetry._valid(agent_name)

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
            assert call_counts["caspar"] == 2, (
                "caspar must be attempted exactly twice (initial + one retry)"
            )

    @pytest.mark.asyncio
    async def test_two_agents_both_exhaust_retries_raises(self, tmp_path):
        """Two agents fail both attempts → only one survivor → RuntimeError.

        The 2-agent minimum is unchanged by retry. If two agents burn
        through their retry budget, the run must raise the same
        ``RuntimeError`` it raises today — retry does not lower the
        consensus floor.
        """
        from run_magi import run_orchestrator
        from validate import ValidationError

        async def mock_launch(agent_name, agents_dir, prompt, output_dir, timeout, model="opus"):
            if agent_name in ("caspar", "melchior"):
                raise ValidationError(f"missing keys for {agent_name}")
            return TestSingleShotRetry._valid(agent_name)

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            with pytest.raises(RuntimeError, match="fewer than 2"):
                await run_orchestrator(
                    agents_dir=str(tmp_path),
                    prompt="test",
                    output_dir=str(tmp_path),
                    timeout=300,
                )

    @pytest.mark.asyncio
    async def test_timeout_does_not_trigger_retry(self, tmp_path):
        """``TimeoutError`` must not trigger retry (non-goal for 2.2.0)."""
        from run_magi import run_orchestrator

        call_counts = {"melchior": 0, "balthasar": 0, "caspar": 0}

        async def mock_launch(agent_name, agents_dir, prompt, output_dir, timeout, model="opus"):
            call_counts[agent_name] += 1
            if agent_name == "caspar":
                raise TimeoutError(f"agent {agent_name} timed out")
            return TestSingleShotRetry._valid(agent_name)

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            result = await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
            )
            assert result["degraded"] is True
            assert "caspar" in result["failed_agents"]
            assert call_counts["caspar"] == 1, (
                "timeout must not be retried — retry scope is schema only"
            )

    @pytest.mark.asyncio
    async def test_runtime_error_does_not_trigger_retry(self, tmp_path):
        """``RuntimeError`` (non-zero exit) must not trigger retry."""
        from run_magi import run_orchestrator

        call_counts = {"melchior": 0, "balthasar": 0, "caspar": 0}

        async def mock_launch(agent_name, agents_dir, prompt, output_dir, timeout, model="opus"):
            call_counts[agent_name] += 1
            if agent_name == "caspar":
                raise RuntimeError(f"agent {agent_name} exited non-zero")
            return TestSingleShotRetry._valid(agent_name)

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            result = await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
            )
            assert result["degraded"] is True
            assert call_counts["caspar"] == 1, "subprocess exit errors must not be retried"

    @pytest.mark.asyncio
    async def test_retry_uses_full_timeout_budget(self, tmp_path):
        """Each attempt receives the full ``timeout`` kwarg, not a reduced one.

        Operators configure ``--timeout`` as a per-attempt ceiling. The
        retry must honor the same ceiling; halving it (or consuming the
        first attempt's remaining budget) would introduce silent behavior
        the docs do not promise.
        """
        from run_magi import run_orchestrator
        from validate import ValidationError

        captured_timeouts: dict[str, list[int]] = {
            "melchior": [],
            "balthasar": [],
            "caspar": [],
        }
        call_counts = {"melchior": 0, "balthasar": 0, "caspar": 0}

        async def mock_launch(agent_name, agents_dir, prompt, output_dir, timeout, model="opus"):
            captured_timeouts[agent_name].append(timeout)
            call_counts[agent_name] += 1
            if agent_name == "caspar" and call_counts[agent_name] == 1:
                raise ValidationError("schema fail, retry please")
            return TestSingleShotRetry._valid(agent_name)

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
            )
        assert captured_timeouts["caspar"] == [300, 300], (
            "retry must be launched with the full per-agent timeout budget"
        )

    @pytest.mark.asyncio
    async def test_retry_injects_validation_error_feedback(self, tmp_path):
        """The retry prompt must carry corrective feedback from the error.

        Contract: the second call to ``launch_agent`` receives a prompt
        that (a) contains the original user prompt and (b) contains the
        ValidationError message so the model can self-correct. The
        feedback block format is implementation-defined but the error
        text must be substring-present.
        """
        from run_magi import run_orchestrator
        from validate import ValidationError

        error_msg = "Agent output missing keys: ['recommendation']"
        captured_prompts: dict[str, list[str]] = {
            "melchior": [],
            "balthasar": [],
            "caspar": [],
        }
        call_counts = {"melchior": 0, "balthasar": 0, "caspar": 0}

        async def mock_launch(agent_name, agents_dir, prompt, output_dir, timeout, model="opus"):
            captured_prompts[agent_name].append(prompt)
            call_counts[agent_name] += 1
            if agent_name == "caspar" and call_counts[agent_name] == 1:
                raise ValidationError(error_msg)
            return TestSingleShotRetry._valid(agent_name)

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="ORIGINAL-USER-PROMPT-TOKEN",
                output_dir=str(tmp_path),
                timeout=300,
            )

        assert len(captured_prompts["caspar"]) == 2
        first_prompt, retry_prompt = captured_prompts["caspar"]
        assert first_prompt == "ORIGINAL-USER-PROMPT-TOKEN", (
            "first call must receive the untouched user prompt"
        )
        assert "ORIGINAL-USER-PROMPT-TOKEN" in retry_prompt, (
            "retry prompt must preserve the original user prompt"
        )
        assert "recommendation" in retry_prompt, (
            "retry prompt must surface the ValidationError message so the "
            "model can self-correct the specific missing field"
        )

    @pytest.mark.asyncio
    async def test_retry_emits_retrying_display_state(self, tmp_path):
        """A ``retrying`` display state must appear between running and the
        terminal state for the agent that hit ValidationError.

        Other agents must not see a ``retrying`` update.
        """
        from run_magi import run_orchestrator
        from validate import ValidationError

        call_counts = {"melchior": 0, "balthasar": 0, "caspar": 0}
        display_events: list[tuple[str, str]] = []

        async def mock_launch(agent_name, agents_dir, prompt, output_dir, timeout, model="opus"):
            call_counts[agent_name] += 1
            if agent_name == "caspar" and call_counts[agent_name] == 1:
                raise ValidationError("schema fail")
            return TestSingleShotRetry._valid(agent_name)

        def capture_update(display, name, state, log_gate):
            display_events.append((name, state))

        with (
            patch("run_magi.launch_agent", side_effect=mock_launch),
            patch("run_magi._safe_display_update", side_effect=capture_update),
        ):
            await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
                show_status=True,
            )

        caspar_states = [state for name, state in display_events if name == "caspar"]
        assert "retrying" in caspar_states, (
            "caspar must transition through a 'retrying' state before success"
        )
        # Order: running → retrying → success
        running_idx = caspar_states.index("running")
        retrying_idx = caspar_states.index("retrying")
        success_idx = caspar_states.index("success")
        assert running_idx < retrying_idx < success_idx, (
            f"caspar state order violated: {caspar_states}"
        )

        for other in ("melchior", "balthasar"):
            other_states = [state for name, state in display_events if name == other]
            assert "retrying" not in other_states, (
                f"{other} must not emit 'retrying' — only the failing agent retries"
            )


class TestJsonDecodeRetry:
    """2.2.4: retry also fires on ``json.JSONDecodeError`` from parse_agent_output.

    Background: 2.2.0 scoped retry to :class:`ValidationError` only. A
    production ``iter 2 catastrophic failure`` (post-2.2.3) lost two of
    three agents to ``json.JSONDecodeError`` raised inside
    :func:`parse_agent_output.parse_agent_output` BEFORE
    :func:`validate.load_agent_output` could wrap the failure into
    ``ValidationError``. Without retry, both agents were dropped and
    synthesis aborted on the 2-agent minimum.

    2.2.4 widens the retry trigger to ``(ValidationError,
    json.JSONDecodeError)``. ``ValueError`` is **not** added to the
    trigger set: ``ValueError`` is also raised by ``resolve_model`` for
    invalid model short names (where retry is pointless — same input
    yields the same error) and by ``parse_agent_output._extract_text``
    for unrecognized Anthropic CLI output shapes (a structural change
    that needs a parser update, not a retry).

    Telemetry contract is unchanged: the `retried_agents` field
    introduced in 2.2.1 records the agent regardless of whether the
    triggering exception was ValidationError or JSONDecodeError.
    """

    @staticmethod
    def _valid(agent: str) -> dict[str, Any]:
        return {
            "agent": agent,
            "verdict": "approve",
            "confidence": 0.85,
            "summary": f"{agent} OK",
            "reasoning": "Fine",
            "findings": [],
            "recommendation": "Merge",
        }

    @pytest.mark.asyncio
    async def test_json_decode_error_triggers_retry_success(self, tmp_path):
        """First attempt raises json.JSONDecodeError, retry succeeds.

        The orchestrator must retry the agent and emerge with a full
        3-agent consensus, no `degraded` flag, and the agent listed in
        `retried_agents` so downstream telemetry sees the recovery.
        """
        import json as _json

        from run_magi import run_orchestrator

        call_counts = {"melchior": 0, "balthasar": 0, "caspar": 0}

        async def mock_launch(agent_name, agents_dir, prompt, output_dir, timeout, model="opus"):
            call_counts[agent_name] += 1
            if agent_name == "melchior" and call_counts[agent_name] == 1:
                # Simulate the exact failure mode reported in production:
                # parse_agent_output called json.loads on truncated text
                # and json raised JSONDecodeError.
                raise _json.JSONDecodeError("Expecting value", "truncated output...", 142)
            return TestJsonDecodeRetry._valid(agent_name)

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            result = await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
            )
        assert result.get("degraded") is not True, (
            "JSONDecodeError that recovered on retry must not leave "
            "degraded set — full 3-agent consensus is restored"
        )
        assert len(result["agents"]) == 3
        assert result.get("retried_agents") == ["melchior"], (
            "retried_agents must record the recovery regardless of "
            "whether the triggering exception was ValidationError or "
            "JSONDecodeError"
        )
        assert call_counts["melchior"] == 2

    @pytest.mark.asyncio
    async def test_json_decode_error_retry_also_fails_degrades(self, tmp_path):
        """Both attempts raise json.JSONDecodeError → agent dropped.

        Mirrors the ValidationError "retry-also-fails" path: the agent
        appears in BOTH `retried_agents` (it took the retry path) AND
        `failed_agents` (it ultimately failed). The intersection
        identifies the retry-also-failed cohort downstream tooling
        cares about.
        """
        import json as _json

        from run_magi import run_orchestrator

        async def mock_launch(agent_name, agents_dir, prompt, output_dir, timeout, model="opus"):
            if agent_name == "balthasar":
                raise _json.JSONDecodeError("Unterminated string", "broken", 50)
            return TestJsonDecodeRetry._valid(agent_name)

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            result = await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
            )
        assert result["degraded"] is True
        assert "balthasar" in result["failed_agents"]
        assert result.get("retried_agents") == ["balthasar"], (
            "retried_agents records every agent that took the retry "
            "path, including those whose retry also failed"
        )

    @pytest.mark.asyncio
    async def test_value_error_from_parse_does_not_retry(self, tmp_path):
        """``ValueError`` is **out of scope** for retry — explicit boundary.

        ``ValueError`` is raised by both ``parse_agent_output`` (for
        unrecognized output shapes) and ``resolve_model`` (for invalid
        model short names). The latter is a configuration error where
        retry is pointless; the former is a structural change that
        needs a parser fix, not a retry. Catching ValueError would
        retry on both paths, masking the configuration bug and wasting
        a subprocess on a structural one.

        This test pins the boundary so a future ``except (ValidationError,
        JSONDecodeError, ValueError)`` change cannot slip past CI.
        """
        from run_magi import run_orchestrator

        call_counts = {"melchior": 0, "balthasar": 0, "caspar": 0}

        async def mock_launch(agent_name, agents_dir, prompt, output_dir, timeout, model="opus"):
            call_counts[agent_name] += 1
            if agent_name == "caspar":
                raise ValueError("Unexpected Claude CLI output type: int")
            return TestJsonDecodeRetry._valid(agent_name)

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            result = await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
            )
        assert result["degraded"] is True
        assert "caspar" in result["failed_agents"]
        assert call_counts["caspar"] == 1, (
            "ValueError must not trigger retry — that path is reserved "
            "for JSON parse failures, not parser-shape or config errors"
        )
        assert "retried_agents" not in result, (
            "no agent took the retry path, so the field must be omitted "
            "(conditional-presence convention)"
        )


class TestRetryTelemetry:
    """2.2.1: report exposes ``retried_agents`` for downstream telemetry.

    Until 2.2.0 the only auditable signal of retry activity was
    ``degraded=true`` + ``failed_agents`` — i.e. the worst-case where
    the retry also failed. A successful retry was indistinguishable
    from a clean first-attempt run, which is exactly the case
    operators need to size to evaluate the retry budget and count.

    Contract for the new ``retried_agents`` report field:

    * Lists every agent name that hit the retry path, regardless of
      whether the retry recovered or also failed.
    * Sorted alphabetically so the JSON serialisation is byte-stable
      across runs and platforms (cleanly diff-able in audit logs).
    * Conditionally present, mirroring the existing ``degraded`` and
      ``failed_agents`` keys: omitted entirely when no retry fired,
      so 2.2.0 consumers that ignore unknown keys are unaffected.
    * Composes with ``failed_agents`` to give two derived sets:
      ``set(retried_agents) - set(failed_agents)`` is "retry recovered",
      ``set(retried_agents) & set(failed_agents)`` is "retry also failed".
    """

    @staticmethod
    def _valid(agent: str) -> dict[str, Any]:
        return {
            "agent": agent,
            "verdict": "approve",
            "confidence": 0.85,
            "summary": f"{agent} OK",
            "reasoning": "Fine",
            "findings": [],
            "recommendation": "Merge",
        }

    @pytest.mark.asyncio
    async def test_report_lists_retried_agent_when_retry_succeeds(self, tmp_path):
        """Retry-recovered: agent in ``agents`` and in ``retried_agents``."""
        from run_magi import run_orchestrator
        from validate import ValidationError

        call_counts = {"melchior": 0, "balthasar": 0, "caspar": 0}

        async def mock_launch(agent_name, agents_dir, prompt, output_dir, timeout, model="opus"):
            call_counts[agent_name] += 1
            if agent_name == "caspar" and call_counts[agent_name] == 1:
                raise ValidationError("missing keys: ['recommendation']")
            return TestRetryTelemetry._valid(agent_name)

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            result = await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
            )
        assert result.get("retried_agents") == ["caspar"], (
            "successful retry must still be recorded in retried_agents"
        )
        assert result.get("degraded") is not True
        assert "failed_agents" not in result, (
            "no failures => failed_agents must be omitted (2.1.x contract preserved)"
        )

    @pytest.mark.asyncio
    async def test_report_lists_retried_agent_when_retry_also_fails(self, tmp_path):
        """Retry-also-failed: agent in retried_agents AND failed_agents."""
        from run_magi import run_orchestrator
        from validate import ValidationError

        async def mock_launch(agent_name, agents_dir, prompt, output_dir, timeout, model="opus"):
            if agent_name == "caspar":
                raise ValidationError("missing keys: ['recommendation']")
            return TestRetryTelemetry._valid(agent_name)

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            result = await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
            )
        assert result.get("retried_agents") == ["caspar"]
        assert result.get("degraded") is True
        assert result.get("failed_agents") == ["caspar"]
        # Composability check: the intersection identifies retry-also-failed
        retried = set(result["retried_agents"])
        failed = set(result["failed_agents"])
        assert retried & failed == {"caspar"}

    @pytest.mark.asyncio
    async def test_report_omits_retried_agents_field_when_no_retry(self, tmp_path):
        """Field absent (not empty list) on a clean run.

        Mirrors the conditional-presence convention used by ``degraded``
        and ``failed_agents``: keys are introduced only when their value
        is informative, so 2.2.0 consumers reading these reports never
        see a meaningless ``"retried_agents": []`` they have to filter.
        """
        from run_magi import run_orchestrator

        async def mock_launch(agent_name, agents_dir, prompt, output_dir, timeout, model="opus"):
            return TestRetryTelemetry._valid(agent_name)

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            result = await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
            )
        assert "retried_agents" not in result, (
            "retried_agents must be omitted entirely when no agent retried "
            "(do not emit an empty list)"
        )

    @pytest.mark.asyncio
    async def test_report_lists_multiple_retried_agents_sorted(self, tmp_path):
        """Two retries (one recovers, one fails) → both listed, sorted."""
        from run_magi import run_orchestrator
        from validate import ValidationError

        call_counts = {"melchior": 0, "balthasar": 0, "caspar": 0}

        async def mock_launch(agent_name, agents_dir, prompt, output_dir, timeout, model="opus"):
            call_counts[agent_name] += 1
            # melchior recovers on retry; caspar fails twice.
            if agent_name == "melchior" and call_counts[agent_name] == 1:
                raise ValidationError("missing keys for melchior")
            if agent_name == "caspar":
                raise ValidationError("missing keys for caspar")
            return TestRetryTelemetry._valid(agent_name)

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            result = await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
            )
        assert result.get("retried_agents") == ["caspar", "melchior"], (
            "retried_agents must list every agent that took the retry path, "
            "sorted alphabetically for byte-stable JSON"
        )
        # caspar failed both attempts; melchior recovered on retry.
        assert result.get("failed_agents") == ["caspar"]
        retried = set(result["retried_agents"])
        failed = set(result["failed_agents"])
        assert retried - failed == {"melchior"}, "retry-recovered set"
        assert retried & failed == {"caspar"}, "retry-also-failed set"


class TestCp1252Resilience:
    """2.2.6: orchestrator must not crash on Windows cp1252 environments.

    Two reproducible crash sites existed under cp1252:

    1. ``print(f"\\u26a0 WARNING: ...", file=sys.stderr)``: the warning sign
       ``\\u26a0`` (⚠) is not in cp1252's codepage (cp1252 covers
       U+0000-U+00FF plus a 0x80-0x9F extension; U+26A0 is outside).
       When MAGI runs as a subprocess (sbtdd's ``subprocess.run`` with
       ``capture_output=True``), ``sys.stderr`` is the locale-encoding
       text wrapper, which on Windows is cp1252. The print encodes
       with ``errors='strict'`` and raises ``UnicodeEncodeError``,
       crashing the orchestrator before the report is written.

    2. ``open(args.input, encoding='utf-8')``: any user input file
       written by Windows tooling with the default cp1252 encoding
       (Notepad, VS Code without explicit BOM, Python ``open()`` on
       Windows without ``encoding=``) raises ``UnicodeDecodeError`` on
       the first byte ≥0x80 that is not a valid UTF-8 start byte.

    These tests pin the contract that both sites stay non-crashing
    after 2.2.6 even when the environment locale is cp1252.
    """

    @pytest.mark.asyncio
    async def test_warning_print_does_not_crash_on_cp1252_stderr(
        self, tmp_path, monkeypatch
    ):
        """Repro for crash site 1: WARNING about a failed agent must not
        encode-fail when ``sys.stderr`` is a cp1252-strict text stream.

        Pre-fix: the ``\\u26a0`` warning sign in run_orchestrator's
        WARNING messages crashes Python's ``print`` with
        ``UnicodeEncodeError`` on cp1252 locales.

        Post-fix: the message uses ASCII-only markers (``[!]`` instead
        of ``⚠``) so the print survives any encoding the parent process
        chooses for the captured stderr.
        """
        import codecs
        import io

        from run_magi import run_orchestrator

        async def mock_launch(agent_name, agents_dir, prompt, output_dir, timeout, model="opus"):
            if agent_name == "caspar":
                # Use a non-retryable error class so retry does not fire
                # and we go straight to the WARNING-then-degraded path.
                raise RuntimeError("subprocess died for the test")
            return {
                "agent": agent_name,
                "verdict": "approve",
                "confidence": 0.85,
                "summary": f"{agent_name} OK",
                "reasoning": "Fine",
                "findings": [],
                "recommendation": "Merge",
            }

        # Cp1252-strict stream: bytes that fall outside cp1252 will raise
        # UnicodeEncodeError on write. This mirrors what Python gives the
        # orchestrator when it runs as a subprocess on Windows.
        cp1252_buffer = io.BytesIO()
        cp1252_stderr = codecs.getwriter("cp1252")(cp1252_buffer, errors="strict")

        monkeypatch.setattr(sys, "stderr", cp1252_stderr)

        with patch("run_magi.launch_agent", side_effect=mock_launch):
            # Must not raise UnicodeEncodeError. Pre-fix: it does.
            result = await run_orchestrator(
                agents_dir=str(tmp_path),
                prompt="test",
                output_dir=str(tmp_path),
                timeout=300,
                show_status=False,  # keep stderr writes direct, no buffer shim
            )

        assert result["degraded"] is True
        assert "caspar" in result["failed_agents"]

        # Sanity: the WARNING actually made it through to the cp1252 stream.
        cp1252_stderr.flush()
        emitted = cp1252_buffer.getvalue().decode("cp1252", errors="replace")
        assert "WARNING" in emitted, (
            "the test must actually exercise the WARNING-emitting path; "
            "if this assert fails the mock arrangement is wrong"
        )
        assert "⚠" not in emitted, (
            "the warning sign \\u26a0 must not be emitted to stderr — "
            "it is the exact codepoint that breaks cp1252 environments"
        )

    def test_input_file_read_handles_cp1252_bytes(self, tmp_path):
        """Repro for crash site 2: a cp1252-encoded input file must not
        crash MAGI's input loader.

        Pre-fix: the ``open(args.input, encoding='utf-8')`` at the top of
        ``main()`` raises ``UnicodeDecodeError`` on any byte ≥0x80 that
        is not a valid UTF-8 start byte (e.g., the cp1252 em dash 0x97).

        Post-fix: the read uses ``errors='replace'`` so the file content
        is decoded with replacement characters in place of invalid bytes,
        and MAGI continues with whatever readable content remains.
        """
        from run_magi import _load_input_content

        cp1252_file = tmp_path / "cp1252-input.txt"
        # b'Hello \x97 world' — \x97 is the cp1252 em dash, NOT a valid
        # UTF-8 start byte. Reading this with strict UTF-8 raises.
        cp1252_file.write_bytes("Hello — world".encode("cp1252"))

        # Must not raise UnicodeDecodeError. Pre-fix: it does.
        content, label = _load_input_content(str(cp1252_file))

        assert "Hello" in content, "ASCII portions of the file must survive"
        assert "world" in content, "ASCII portions on either side of the bad byte"
        assert label == f"File: {cp1252_file}"

    def test_load_input_content_treats_inline_text_as_string(self):
        """``_load_input_content`` returns inline text untouched when the
        argument is not a file path. This pins the boundary between
        file-read (cp1252-tolerant) and inline-text (no decode) paths.
        """
        from run_magi import _load_input_content

        content, label = _load_input_content("inline analysis text — not a path")
        assert content == "inline analysis text — not a path"
        assert label == "Inline input"
