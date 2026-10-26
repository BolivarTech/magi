# Author: Julian Bolivar
# Version: 0.24.3
# Date: 2026-10-25
"""Unit tests for the code-review runs R12-R15 and the files they write."""

import pathlib
import shutil
import subprocess
import tempfile
import types
import unittest
from unittest import mock

from smoke import runs
from smoke.errors import HarnessError


def _executor(root: pathlib.Path) -> runs.RunExecutor:
    """An executor over an environment double rooted at *root*.

    The double carries a ``.magi`` directory with a config file and a scratch
    area, the two things the review workspace is built from.
    """
    magi = root / ".magi"
    magi.mkdir(parents=True, exist_ok=True)
    (magi / "magi.toml").write_text("# toy config\n", encoding="utf-8")
    env = types.SimpleNamespace(root=root, magi_dir=magi,
                                scratch_dir=root / "scratch")
    return runs.RunExecutor(binary=None, env=env, config=None)


def _git_available() -> bool:
    """Whether a ``git`` executable is on the path."""
    return shutil.which("git") is not None


class ReviewInputTests(unittest.TestCase):

    def test_the_generated_input_fills_at_least_three_batches(self) -> None:
        total = sum(len(body) for _, body in runs.review_input_files())
        window = int(runs.UNMEASURED_WINDOW_TOKENS * runs.BATCH_WINDOW_FRACTION)
        self.assertGreaterEqual(total // runs.BYTES_PER_TOKEN,
                                runs.REVIEW_MIN_BATCHES * window)

    def test_the_generated_input_is_deterministic(self) -> None:
        self.assertEqual(runs.review_input_files(), runs.review_input_files())

    def test_every_generated_function_name_is_unique(self) -> None:
        names = [line for _, body in runs.review_input_files()
                 for line in body.decode().splitlines() if line.startswith("pub fn ")]
        self.assertEqual(len(names), len(set(names)))

    def test_the_generated_files_are_the_declared_paths(self) -> None:
        paths = tuple(path for path, _ in runs.review_input_files())
        self.assertEqual(runs.REVIEW_GEN_PATHS, paths)

    def test_the_skipped_files_hold_a_binary_and_a_key_shaped_secret(self) -> None:
        files = dict(runs.review_skipped_files())
        self.assertIn(b"\x00", files["skipped/pixel.png"])
        self.assertIn(b"sk-ant-", files["skipped/leak.txt"])  # allow-secret-scan


class ReviewRunDefinitionTests(unittest.TestCase):

    def test_the_hostile_path_run_names_a_path_outside_the_workspace(self) -> None:
        self.assertEqual(runs.REVIEW_OUTSIDE_PATH, runs.DEFINITIONS["R13"].argv[-1])

    def test_the_timeout_run_asks_for_a_total_below_any_floor(self) -> None:
        argv = list(runs.DEFINITIONS["R15"].argv)
        self.assertEqual(str(runs.REVIEW_TOO_SHORT_TIMEOUT_S),
                         argv[argv.index("--timeout") + 1])
        self.assertEqual("error", argv[argv.index("--log-level") + 1])

    def test_only_the_large_review_run_reaches_the_trio(self) -> None:
        reviewing = {rid for rid, d in runs.DEFINITIONS.items()
                     if d.argv[0] == "code-review" and d.needs_trio}
        self.assertEqual({"R12"}, reviewing)


class WriteFilesTests(unittest.TestCase):

    def test_files_land_under_the_given_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _executor(root)._write_files(runs.DEFINITIONS["R14"], root)
            self.assertTrue((root / "skipped" / "pixel.png").is_file())

    def test_a_path_leaving_the_root_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hostile = runs.RunDefinition(
                run_id="RX", argv=("x",), stdin=b"", needs_trio=False,
                payload_size=runs.PAYLOAD_SMALL, timeout_s=1,
                files=(("../escape.txt", b"x"),))
            root = pathlib.Path(tmp)
            with self.assertRaises(HarnessError):
                _executor(root)._write_files(hostile, root)

    def test_a_definition_without_files_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            executor = _executor(root)
            before = sorted(root.iterdir())
            executor._write_files(runs.DEFINITIONS["R1"], root)
            self.assertEqual(before, sorted(root.iterdir()))


class VisibilityPreflightTests(unittest.TestCase):
    """The harness checks git can see the input before the product is paid."""

    def _repo(self, tmp: str, ignore: str | None) -> pathlib.Path:
        root = pathlib.Path(tmp)
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        if ignore is not None:
            (root / ".gitignore").write_text(ignore, encoding="utf-8")
        (root / "gen").mkdir()
        (root / "gen" / "a.rs").write_text("fn a() {}\n", encoding="utf-8")
        return root

    @unittest.skipUnless(_git_available(), "git is required")
    def test_an_ignored_input_is_a_harness_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._repo(tmp, "*\n")
            with self.assertRaises(HarnessError):
                runs.verify_visible(root, ("gen/a.rs",))

    @unittest.skipUnless(_git_available(), "git is required")
    def test_a_visible_input_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._repo(tmp, None)
            runs.verify_visible(root, ("gen/a.rs",))


class ReviewWorkspaceTests(unittest.TestCase):
    """R12-R15 run in a nested git repository under the scratch area."""

    def test_the_review_runs_name_the_review_workspace(self) -> None:
        for run_id in ("R12", "R13", "R14", "R15"):
            argv = runs.DEFINITIONS[run_id].argv
            self.assertEqual(runs.REVIEW_WORKSPACE_TOKEN,
                             argv[argv.index("-w") + 1], run_id)

    def test_the_token_resolves_to_a_directory_under_the_scratch_area(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            command = _executor(root)._command(runs.DEFINITIONS["R13"])
            expected = str(root / "scratch" / runs.REVIEW_WORKSPACE_DIR)
            self.assertEqual(expected, command[command.index("-w") + 1])

    @unittest.skipUnless(_git_available(), "git is required")
    def test_the_workspace_is_its_own_repo_with_the_config_and_the_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            workspace = _executor(root)._prepare_review_workspace(
                runs.DEFINITIONS["R14"])
            self.assertTrue((workspace / ".git").exists())
            self.assertTrue((workspace / ".magi" / "magi.toml").is_file())
            self.assertTrue((workspace / "skipped" / "pixel.png").is_file())
            top = subprocess.run(
                ["git", "-C", str(workspace), "rev-parse", "--show-toplevel"],
                check=True, capture_output=True, text=True).stdout.strip()
            self.assertEqual(workspace.resolve(), pathlib.Path(top).resolve())

    @unittest.skipUnless(_git_available(), "git is required")
    def test_a_second_preparation_starts_from_a_clean_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            executor = _executor(root)
            workspace = executor._prepare_review_workspace(runs.DEFINITIONS["R14"])
            (workspace / "stale.rs").write_text("x", encoding="utf-8")
            executor._prepare_review_workspace(runs.DEFINITIONS["R14"])
            self.assertFalse((workspace / "stale.rs").exists())

    def test_a_workspace_resolving_outside_the_scratch_area_is_not_deleted(self) -> None:
        # [MAGI §6 MS2 F loop 2] rmtree runs only on a path strictly under the scratch area.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            executor = _executor(root)
            (root / "scratch").mkdir()
            outside = root / "elsewhere"
            outside.mkdir()
            (outside / "keep.txt").write_text("x", encoding="utf-8")
            with mock.patch.object(runs, "REVIEW_WORKSPACE_DIR", "../elsewhere"):
                with self.assertRaises(HarnessError):
                    executor._prepare_review_workspace(runs.DEFINITIONS["R14"])
            self.assertTrue((outside / "keep.txt").is_file())


class ReviewLineCountTests(unittest.TestCase):

    def test_the_line_counts_match_the_generated_bytes(self) -> None:
        counts = runs.review_gen_line_counts()
        self.assertEqual(set(runs.REVIEW_GEN_PATHS), set(counts))
        for path, body in runs.review_input_files():
            self.assertEqual(len(body.decode("utf-8").splitlines()), counts[path])


if __name__ == "__main__":
    unittest.main()
