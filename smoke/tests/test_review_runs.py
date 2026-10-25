# Author: Julian Bolivar
# Version: 0.24.3
# Date: 2026-10-25
"""Unit tests for the code-review runs R12-R15 and the files they write."""

import pathlib
import tempfile
import types
import unittest

from smoke import runs
from smoke.errors import HarnessError


def _executor(root: pathlib.Path) -> runs.RunExecutor:
    """An executor over a bare environment double rooted at *root*."""
    env = types.SimpleNamespace(root=root)
    return runs.RunExecutor(binary=None, env=env, config=None)


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

    def test_files_land_under_the_environment_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _executor(root)._write_files(runs.DEFINITIONS["R14"])
            self.assertTrue((root / "skipped" / "pixel.png").is_file())

    def test_a_rerun_leaves_identical_files_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            executor = _executor(root)
            executor._write_files(runs.DEFINITIONS["R14"])
            target = root / "skipped" / "leak.txt"
            before = target.stat().st_mtime_ns
            executor._write_files(runs.DEFINITIONS["R14"])
            self.assertEqual(before, target.stat().st_mtime_ns)

    def test_a_path_leaving_the_root_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hostile = runs.RunDefinition(
                run_id="RX", argv=("x",), stdin=b"", needs_trio=False,
                payload_size=runs.PAYLOAD_SMALL, timeout_s=1,
                files=(("../escape.txt", b"x"),))
            with self.assertRaises(HarnessError):
                _executor(pathlib.Path(tmp))._write_files(hostile)

    def test_a_definition_without_files_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _executor(pathlib.Path(tmp))._write_files(runs.DEFINITIONS["R1"])
            self.assertEqual([], list(pathlib.Path(tmp).iterdir()))


if __name__ == "__main__":
    unittest.main()
