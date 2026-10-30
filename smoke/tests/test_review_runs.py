# Author: Julian Bolivar
# Version: 0.24.3
# Date: 2026-10-25
"""Unit tests for the code-review runs R12-R16 and the files they write."""

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

    def test_the_too_short_timeout_refuses_one_batch_under_the_most_lenient_config(self) -> None:
        # [MAGI §6 MS2 loop 6, F1a-c1] the refusal must fire at ANY batch count: one batch needs the
        # wall clock the floor derives with no rotation and no retry (the smallest attempt factor).
        from smoke.scenarios import trio
        lenient = trio.floor_activation_threshold(trio.attempt_factor(0, True))
        self.assertLess(runs.REVIEW_TOO_SHORT_TIMEOUT_S, lenient)

    def test_the_root_run_reviews_the_workspace_root_with_a_too_short_timeout(self) -> None:
        # [MAGI §6 MS2 loop 3, F2] the root of the nested workspace, holding .magi and .git; a
        # protected file taken for review would end at the budget refusal, never at a model.
        definition = runs.DEFINITIONS["R16"]
        argv = list(definition.argv)
        self.assertEqual(runs.REVIEW_ROOT_PATH, argv[-1])
        self.assertEqual(".", runs.REVIEW_ROOT_PATH)
        self.assertEqual(str(runs.REVIEW_TOO_SHORT_TIMEOUT_S),
                         argv[argv.index("--timeout") + 1])
        self.assertEqual(runs.review_skipped_files(), definition.files)
        self.assertFalse(definition.needs_trio)

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


    @unittest.skipUnless(_git_available(), "git is required")
    def test_a_repository_selecting_environment_does_not_redirect_the_query(self) -> None:
        # [MAGI §6 MS2 loop 6, F1a-c2] an inherited GIT_DIR / GIT_WORK_TREE would point git at another
        # repository; the product scrubs its git environment, so the harness must too.
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
            root = self._repo(tmp, None)
            hostile = {"GIT_DIR": str(pathlib.Path(other) / "nowhere.git"),
                       "GIT_WORK_TREE": other, "GIT_INDEX_FILE": str(pathlib.Path(other) / "index")}
            with mock.patch.dict("os.environ", hostile):
                runs.verify_visible(root, ("gen/a.rs",))


class ReviewWorkspaceTests(unittest.TestCase):
    """R12-R15 run in a nested git repository under the scratch area."""

    def test_the_review_runs_name_the_review_workspace(self) -> None:
        for run_id in ("R12", "R13", "R14", "R15", "R16"):
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
    def test_an_inherited_git_dir_does_not_redirect_the_workspace_init(self) -> None:
        # [MAGI §6 MS2 loop 6, F1a-c2] `git init` under an inherited GIT_DIR initialises THAT directory.
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
            elsewhere = pathlib.Path(other) / "elsewhere.git"
            with mock.patch.dict("os.environ", {"GIT_DIR": str(elsewhere)}):
                workspace = _executor(pathlib.Path(tmp))._prepare_review_workspace(
                    runs.DEFINITIONS["R14"])
            self.assertTrue((workspace / ".git").is_dir())
            self.assertFalse(elsewhere.exists())

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

    def test_a_nonexistent_workspace_outside_the_scratch_area_is_never_created(self) -> None:
        # [MAGI §6 MS2 loop 3, F3] containment is checked before anything is created, not only
        # before an existing tree is deleted.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            executor = _executor(root)
            (root / "scratch").mkdir()
            outside = root / "elsewhere-new"
            with mock.patch.object(runs, "REVIEW_WORKSPACE_DIR", "../elsewhere-new"):
                with self.assertRaises(HarnessError):
                    executor._prepare_review_workspace(runs.DEFINITIONS["R14"])
            self.assertFalse(outside.exists())


class ReviewWorkingDirectoryTests(unittest.TestCase):
    """[MAGI §6 MS2 loop 4, F1-2] a review run's relative input means the same under either base."""

    def _invocation(self, root: pathlib.Path, run_id: str) -> dict:
        """The keyword arguments the executor hands the binary for *run_id*."""
        binary = mock.Mock()
        binary.invoke.return_value = runs.ProductOutput(stdout=b"", stderr=b"", exit_code=0, command=["magi-rs"])
        env = types.SimpleNamespace(root=root, magi_dir=root / ".magi", scratch_dir=root / "scratch")
        executor = runs.RunExecutor(binary=binary, env=env,
                                    config=types.SimpleNamespace(passphrase="toy"))
        executor._invoke_with(runs.DEFINITIONS[run_id], {}, b"")
        return binary.invoke.call_args.kwargs

    def test_every_review_run_starts_the_product_in_the_review_workspace(self) -> None:
        # 'gen', 'skipped', '.' and '../smoke-hostile-target' resolve against the workspace whether the
        # product joins them to -w or to its working directory: R16's '.' must never be the harness's cwd.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            for run_id in ("R12", "R13", "R14", "R15", "R16"):
                self.assertEqual(root / "scratch" / runs.REVIEW_WORKSPACE_DIR,
                                 self._invocation(root, run_id).get("cwd"), run_id)

    def test_a_run_without_a_review_workspace_keeps_the_harness_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(self._invocation(pathlib.Path(tmp), "R1").get("cwd"))


class ReviewLogCaptureTests(unittest.TestCase):
    """[MAGI §6 MS2 loop 3, F4] the run log is captured onto the result at run time."""

    def _execute(self, root: pathlib.Path, definition, write_log: bool) -> runs.RunResult:
        """Execute *definition* with a stand-in product that may write one log line."""
        executor = _executor(root)

        def fake_invoke(_definition, _stdin):
            logs = root / "scratch" / runs.REVIEW_WORKSPACE_DIR / ".magi" / "logs"
            if write_log:
                logs.mkdir(parents=True, exist_ok=True)
                (logs / "2026-10-29.log").write_bytes(b"ERROR run=1-ab refusal\n")
            return runs.ProductOutput(stdout=b"", stderr=b"", exit_code=2, command=["magi-rs"])

        with mock.patch.object(executor, "_invoke", side_effect=fake_invoke):
            return executor.execute(definition)

    @unittest.skipUnless(_git_available(), "git is required")
    def test_a_review_run_carries_its_log_after_the_workspace_is_gone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            result = self._execute(root, runs.DEFINITIONS["R15"], write_log=True)
            shutil.rmtree(root / "scratch" / runs.REVIEW_WORKSPACE_DIR)
            self.assertIsNotNone(result.review_log)
            self.assertTrue(any(b"run=1-ab refusal" in data for data in result.review_log))

    @unittest.skipUnless(_git_available(), "git is required")
    def test_a_review_run_without_a_log_directory_carries_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # The toy .magi the workspace is copied from has no logs directory.
            result = self._execute(pathlib.Path(tmp), runs.DEFINITIONS["R14"], write_log=False)
            self.assertIsNone(result.review_log)


class ReviewLineCountTests(unittest.TestCase):

    def test_the_line_counts_match_the_generated_bytes(self) -> None:
        counts = runs.review_gen_line_counts()
        self.assertEqual(set(runs.REVIEW_GEN_PATHS), set(counts))
        for path, body in runs.review_input_files():
            self.assertEqual(len(body.decode("utf-8").splitlines()), counts[path])


if __name__ == "__main__":
    unittest.main()
