# Author: Julian Bolivar
# Version: 0.24.3
# Date: 2026-10-25
"""Unit tests for the code-review runs R12-R16 and the files they write."""

import os
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


def _make_directory_link(target: pathlib.Path, link: pathlib.Path) -> None:
    """Make *link* a directory symlink to *target*, or a junction where symlinks need a privilege.

    Raises:
        unittest.SkipTest: When the platform allows neither.
    """
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    try:
        import _winapi  # Windows only; a junction needs no privilege
        _winapi.CreateJunction(str(target), str(link))
    except (ImportError, OSError) as exc:
        raise unittest.SkipTest("cannot create a directory link here: %s" % exc) from exc


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
                run_id="RX", argv=("x", runs.REVIEW_WORKSPACE_TOKEN), stdin=b"",
                needs_trio=False, payload_size=runs.PAYLOAD_SMALL, timeout_s=1,
                files=(("../escape.txt", b"x"),))
            root = pathlib.Path(tmp)
            with self.assertRaises(HarnessError):
                _executor(root)._write_files(hostile, root)

    def test_files_without_the_review_workspace_token_are_refused_at_definition(self) -> None:
        # [MAGI §6 MS2 loop 14, F1a1-m1] only a run that names the review workspace writes its files, so
        # files on any other run were silently never written and surfaced as a confusing scenario FAIL.
        with self.assertRaises(HarnessError) as caught:
            runs.RunDefinition(
                run_id="RX", argv=("consult",), stdin=b"", needs_trio=False,
                payload_size=runs.PAYLOAD_SMALL, timeout_s=1,
                files=(("a.rs", b"x"),))
        self.assertIn("RX", str(caught.exception))
        self.assertIn(runs.REVIEW_WORKSPACE_TOKEN, str(caught.exception))

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

    def test_a_relative_scratch_area_still_hands_the_product_an_absolute_workspace(self) -> None:
        # MAGI FIX (S6 loop 11, F1a-m1): the product starts with cwd = the review workspace, so a
        # relative -w would resolve a second time against it and name a directory that does not exist.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = pathlib.Path(tmp.name)
        self.addCleanup(os.chdir, os.getcwd())  # runs first: leave the directory before it is removed
        os.chdir(root)
        env = types.SimpleNamespace(root=pathlib.Path("."), magi_dir=pathlib.Path(".magi"),
                                    scratch_dir=pathlib.Path("scratch"))
        executor = runs.RunExecutor(binary=None, env=env, config=None)
        command = executor._command(runs.DEFINITIONS["R13"])
        workspace = pathlib.Path(command[command.index("-w") + 1])
        self.assertTrue(workspace.is_absolute(), workspace)
        self.assertEqual((root / "scratch" / runs.REVIEW_WORKSPACE_DIR).resolve(), workspace)

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
    def test_the_environments_old_logs_are_not_copied_into_the_workspace(self) -> None:
        # [MAGI §6 MS2 loop 6, F1a-m4] the captured review log must be what THIS run wrote, not a
        # superset carrying the environment's earlier runs.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            executor = _executor(root)
            old = root / ".magi" / "logs" / "magi-old.log"
            old.parent.mkdir(parents=True)
            old.write_bytes(b"ERROR run=old-run an earlier line\n")
            workspace = executor._prepare_review_workspace(runs.DEFINITIONS["R14"])
            self.assertTrue((workspace / ".magi" / "magi.toml").is_file())
            self.assertFalse((workspace / ".magi" / "logs").exists())
            self.assertIsNone(executor._capture_review_log(runs.DEFINITIONS["R14"]))

    def test_an_unreadable_log_file_is_named_not_silently_dropped(self) -> None:
        # [MAGI §6 MS2 loop 12, F1a-c2] the file that could not be read is reported, as logs.scan does.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            executor = _executor(root)
            logs_dir = root / "scratch" / runs.REVIEW_WORKSPACE_DIR / ".magi" / "logs"
            logs_dir.mkdir(parents=True)
            (logs_dir / "magi-locked.log").write_bytes(b"x")
            unreadable = []
            with mock.patch.object(pathlib.Path, "read_bytes", side_effect=PermissionError("locked")):
                captured = executor._capture_review_log(runs.DEFINITIONS["R15"], unreadable)
            self.assertEqual((), captured)
            self.assertEqual(1, len(unreadable))
            self.assertIn("magi-locked.log", unreadable[0])

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


class ReviewWorkspaceLinkTests(unittest.TestCase):
    """[MAGI §6 MS2 loop 10, F1a-c4] links never steer what is copied in or deleted."""

    def test_a_link_inside_the_environments_magi_is_refused_before_anything_changes(self) -> None:
        # copytree follows a link: a .magi entry pointing outside would copy what it points at into the
        # workspace the product reads and the run log may name.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            executor = _executor(root)
            outside = root / "outside"
            outside.mkdir()
            (outside / "secret.txt").write_text("not for the workspace", encoding="utf-8")
            _make_directory_link(outside, root / ".magi" / "linked")
            workspace = root / "scratch" / runs.REVIEW_WORKSPACE_DIR
            workspace.mkdir(parents=True)
            (workspace / "marker").write_text("x", encoding="utf-8")
            with self.assertRaises(HarnessError) as caught:
                executor._prepare_review_workspace(runs.DEFINITIONS["R14"])
            self.assertIn("linked", str(caught.exception))
            self.assertTrue((workspace / "marker").is_file())
            self.assertTrue((outside / "secret.txt").is_file())

    def test_a_review_workspace_that_is_a_link_is_never_deleted_through(self) -> None:
        # The workspace's own path being a link inside the scratch area passes the resolve() containment
        # check; deleting through it would empty the directory it points at.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            executor = _executor(root)
            other = root / "scratch" / "other"
            other.mkdir(parents=True)
            (other / "keep.txt").write_text("x", encoding="utf-8")
            _make_directory_link(other, root / "scratch" / runs.REVIEW_WORKSPACE_DIR)
            with self.assertRaises(HarnessError) as caught:
                executor._prepare_review_workspace(runs.DEFINITIONS["R14"])
            self.assertIn("link", str(caught.exception))
            self.assertTrue((other / "keep.txt").is_file())

    def test_a_linked_parent_inside_the_scratch_area_is_never_deleted_through(self) -> None:
        # [MAGI §6 MS2 loop 12, F1a-c1] a parent component that is a link resolves under the scratch area
        # too, so the containment check passes and rmtree would empty the tree the link names.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            executor = _executor(root)
            copy = root / "scratch" / "copy"
            (copy / runs.REVIEW_WORKSPACE_DIR).mkdir(parents=True)
            keep = copy / runs.REVIEW_WORKSPACE_DIR / "keep.txt"
            keep.write_text("x", encoding="utf-8")
            _make_directory_link(copy, root / "scratch" / "mirror")
            with mock.patch.object(runs, "REVIEW_WORKSPACE_DIR", "mirror/" + runs.REVIEW_WORKSPACE_DIR):
                with self.assertRaises(HarnessError) as caught:
                    executor._prepare_review_workspace(runs.DEFINITIONS["R14"])
            self.assertIn("mirror", str(caught.exception))
            self.assertTrue(keep.is_file())

    @unittest.skipUnless(os.name == "nt", "junctions exist on Windows only")
    def test_a_junction_is_a_link_without_os_path_isjunction(self) -> None:
        # [MAGI §6 MS2 loop 13, F1a-m3/b2/c2] os.path.isjunction exists from Python 3.12; on the 3.11 floor a
        # junction read as an ordinary directory, so copytree and rmtree went through it.
        import _winapi
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            target = root / "target"
            target.mkdir()
            junction = root / "junction"
            _winapi.CreateJunction(str(target), str(junction))
            with mock.patch.object(os.path, "isjunction", None):
                self.assertTrue(runs._is_link(junction))
                self.assertFalse(runs._is_link(target))
                self.assertFalse(runs._is_link(root / "absent"))
                with self.assertRaises(HarnessError):
                    runs._reject_links(root)

    @unittest.skipUnless(os.name == "nt", "junctions exist on Windows only")
    @unittest.skipUnless(_git_available(), "git is required")
    def test_rebuilding_the_workspace_never_deletes_through_a_junction_inside_it(self) -> None:
        # [MAGI §6 MS2 loop 14, F1a1-c1] the existing workspace is removed with shutil.rmtree before it is
        # rebuilt; a junction left inside it must be unlinked, never descended into, on the 3.11 floor too.
        import _winapi
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            executor = _executor(root)
            outside = root / "outside"
            outside.mkdir()
            keep = outside / "keep.txt"
            keep.write_text("not the workspace's", encoding="utf-8")
            workspace = root / "scratch" / runs.REVIEW_WORKSPACE_DIR
            (workspace / "nested").mkdir(parents=True)
            _winapi.CreateJunction(str(outside), str(workspace / "nested" / "junction"))
            rebuilt = executor._prepare_review_workspace(runs.DEFINITIONS["R14"])
            self.assertTrue(keep.is_file())
            self.assertFalse((rebuilt / "nested").exists())

    def test_an_unreadable_directory_under_magi_is_a_harness_error(self) -> None:
        # [MAGI §6 MS2 loop 13, F1a-c1] os.walk skips a directory it cannot list unless told otherwise, so a
        # link inside it went unchecked; the walk must fail as a harness error (exit 3), never pass.
        def walk(top, onerror=None, **_):
            if onerror is not None:
                onerror(PermissionError(13, "access denied", str(top)))
            return iter(())

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(runs.os, "walk", walk):
                with self.assertRaises(HarnessError) as caught:
                    runs._reject_links(pathlib.Path(tmp))
            self.assertIn("access denied", str(caught.exception))

    def test_an_os_error_in_the_link_checks_is_a_harness_error(self) -> None:
        # [MAGI §6 MS2 loop 13, F1a-c1] the link checks ran outside the try that turns an OSError into a
        # HarnessError, so a raised OSError escaped the executor as a traceback instead of exit 3.
        with tempfile.TemporaryDirectory() as tmp:
            executor = _executor(pathlib.Path(tmp))
            with mock.patch.object(runs, "_is_link", side_effect=PermissionError(13, "locked")):
                with self.assertRaises(HarnessError) as caught:
                    executor._prepare_review_workspace(runs.DEFINITIONS["R14"])
            self.assertIn("locked", str(caught.exception))


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

    def test_a_link_under_the_log_directory_is_never_read_through(self) -> None:
        # [MAGI §6 MS2 loop 14, F1a1-m3] rglob descends into a junction and read_bytes follows a link, so a
        # link planted under .magi/logs had the harness read what it points at into the result.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            executor = _executor(root)
            outside = root / "outside"
            outside.mkdir()
            (outside / "elsewhere.log").write_bytes(b"not this run's log")
            logs_dir = root / "scratch" / runs.REVIEW_WORKSPACE_DIR / ".magi" / "logs"
            logs_dir.mkdir(parents=True)
            (logs_dir / "magi-run.log").write_bytes(b"this run's log")
            _make_directory_link(outside, logs_dir / "linked")
            unreadable = []
            captured = executor._capture_review_log(runs.DEFINITIONS["R15"], unreadable)
            self.assertEqual((b"this run's log",), captured)
            self.assertEqual(1, len(unreadable))
            self.assertIn("linked", unreadable[0])

    @unittest.skipUnless(_git_available(), "git is required")
    def test_a_review_run_without_a_log_directory_carries_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # The toy .magi the workspace is copied from has no logs directory.
            result = self._execute(pathlib.Path(tmp), runs.DEFINITIONS["R14"], write_log=False)
            self.assertIsNone(result.review_log)


class ProtectedCandidateTests(unittest.TestCase):
    """[MAGI §6 MS2 loop 9, F1a-c1/F1a-m2] what tells S51 whether its exclusion half ran."""

    @unittest.skipUnless(_git_available(), "git is required")
    def test_a_review_run_records_the_magi_files_git_lists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            executor = _executor(root)
            output = runs.ProductOutput(stdout=b"", stderr=b"", exit_code=2, command=["magi-rs"])
            with mock.patch.object(executor, "_invoke", return_value=output):
                result = executor.execute(runs.DEFINITIONS["R16"])
            self.assertEqual((".magi/magi.toml",), result.protected_visible)

    @unittest.skipUnless(_git_available(), "git is required")
    def test_a_magi_directory_an_ignore_rule_hides_is_no_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / ".magi").mkdir()
            (root / ".magi" / "magi.toml").write_text("# toy\n", encoding="utf-8")
            (root / "a.rs").write_text("fn a() {}\n", encoding="utf-8")
            self.assertEqual((".magi/magi.toml",), runs.protected_candidates(root))
            (root / ".gitignore").write_text(".magi/\n", encoding="utf-8")
            self.assertEqual((), runs.protected_candidates(root))


class ReviewLineCountTests(unittest.TestCase):

    def test_the_line_counts_match_the_generated_bytes(self) -> None:
        counts = runs.review_gen_line_counts()
        self.assertEqual(set(runs.REVIEW_GEN_PATHS), set(counts))
        for path, body in runs.review_input_files():
            self.assertEqual(len(body.decode("utf-8").splitlines()), counts[path])


if __name__ == "__main__":
    unittest.main()
