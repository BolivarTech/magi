# Author: Julian Bolivar
# Version: 0.24.3
# Date: 2026-10-25
"""Unit tests for the code-review runs R12-R16 and the files they write."""

import io
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
            with mock.patch.object(runs, "_open_listed", side_effect=PermissionError("locked")):
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

    def test_a_failed_git_init_names_what_git_wrote_to_stderr(self) -> None:
        # [MAGI §6 MS2 loop 15, F1a1-b1] CalledProcessError's str() carries no captured stderr, so the
        # HarnessError said only "returned non-zero exit status" and dropped git's own reason.
        def run(command, **kwargs):
            raise subprocess.CalledProcessError(128, command, output=b"",
                                                stderr=b"fatal: cannot copy template \xff files")

        with tempfile.TemporaryDirectory() as tmp:
            executor = _executor(pathlib.Path(tmp))
            with mock.patch.object(runs.subprocess, "run", side_effect=run):
                with self.assertRaises(HarnessError) as caught:
                    executor._prepare_review_workspace(runs.DEFINITIONS["R14"])
            self.assertIn("fatal: cannot copy template", str(caught.exception))

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
            with mock.patch.object(os.path, "isjunction", None, create=True):
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

    @unittest.skipUnless(os.name == "nt", "a read-only file blocks its own deletion on Windows only")
    @unittest.skipUnless(_git_available(), "git is required")
    def test_rebuilding_the_workspace_removes_a_read_only_file_left_in_it(self) -> None:
        # [MAGI §6 MS2 loop 15, F1a1-m3] copytree keeps source modes and rmtree had no error handler, so a
        # read-only file left in the old workspace aborted every later review run with exit 3 on Windows.
        import stat
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            executor = _executor(root)
            workspace = executor._prepare_review_workspace(runs.DEFINITIONS["R14"])
            stale = workspace / ".magi" / "stale.lock"
            stale.write_bytes(b"x")
            os.chmod(stale, stat.S_IREAD)
            rebuilt = executor._prepare_review_workspace(runs.DEFINITIONS["R14"])
            self.assertFalse((rebuilt / ".magi" / "stale.lock").exists())
            self.assertTrue((rebuilt / ".magi" / "magi.toml").is_file())

    def test_a_failed_deletion_of_a_link_is_never_retried_through_a_mode_change(self) -> None:
        # [MAGI §6 MS2 loop 15, F1a1-m3] the retry clears the read-only bit; on a link that would change the
        # target's mode, so the original error stands instead.
        error = PermissionError(13, "access denied")
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "x"
            with mock.patch.object(runs, "_is_link", return_value=True), \
                    mock.patch.object(runs.os, "chmod") as chmod:
                with self.assertRaises(PermissionError):
                    runs._retry_writable(os.unlink, str(path), error)
            chmod.assert_not_called()

    def test_a_failed_deletion_of_a_file_is_retried_once_writable(self) -> None:
        import stat
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "x"
            path.write_bytes(b"x")
            os.chmod(path, stat.S_IREAD)
            runs._retry_writable(os.unlink, str(path), PermissionError(13, "access denied"))
            self.assertFalse(path.exists())

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

    def test_a_refused_or_unreadable_nested_log_entry_is_named_by_its_path_under_the_log_root(self) -> None:
        # [MAGI §6 MS2 loop 15, F1a1-b2] entries were named by path.name alone, so a nested link or locked
        # file lost the directory that tells which one was left out.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            executor = _executor(root)
            outside = root / "outside"
            outside.mkdir()
            nested = root / "scratch" / runs.REVIEW_WORKSPACE_DIR / ".magi" / "logs" / "sub"
            nested.mkdir(parents=True)
            _make_directory_link(outside, nested / "linked")
            unreadable = []
            executor._capture_review_log(runs.DEFINITIONS["R15"], unreadable)
            self.assertEqual(["sub/linked (%s)" % runs.LINK_NOT_FOLLOWED], unreadable)
            (nested / "magi-locked.log").write_bytes(b"x")
            unreadable = []
            with mock.patch.object(runs, "_open_listed", side_effect=PermissionError("locked")):
                executor._capture_review_log(runs.DEFINITIONS["R15"], unreadable)
            self.assertIn("sub/magi-locked.log (locked)", unreadable)

    def test_a_log_directory_that_is_itself_a_link_is_never_read_through(self) -> None:
        # [MAGI §6 MS2 loop 15, F1a1-m1] the walk tests every entry under .magi/logs but not the top: os.scandir
        # follows a symlinked or junction .magi/logs, so the harness read what it points at into the result.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            executor = _executor(root)
            outside = root / "outside"
            outside.mkdir()
            (outside / "elsewhere.log").write_bytes(b"not this run's log")
            magi = root / "scratch" / runs.REVIEW_WORKSPACE_DIR / ".magi"
            magi.mkdir(parents=True)
            _make_directory_link(outside, magi / "logs")
            unreadable = []
            captured = executor._capture_review_log(runs.DEFINITIONS["R15"], unreadable)
            self.assertEqual((), captured)
            self.assertEqual(1, len(unreadable))
            self.assertIn("logs", unreadable[0])
            self.assertIn(runs.LINK_NOT_FOLLOWED, unreadable[0])

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


class ProductSpellingTests(unittest.TestCase):
    """[MAGI §6 MS2 loop 16, F1a1-caspar] the product folds ASCII case and trims trailing dots and spaces."""

    def test_protected_candidates_match_the_magi_directory_whatever_its_spelling(self) -> None:
        listed = frozenset({".MAGI/a.toml", ".magi./b", ".Magi /c", ".magi/d",
                            ".magic/e", "src/.magi/f", "src/a.rs"})
        with mock.patch.object(runs, "visible_paths", return_value=listed):
            self.assertEqual((".MAGI/a.toml", ".Magi /c", ".magi./b", ".magi/d", "src/.magi/f"),
                             runs.protected_candidates(pathlib.Path(".")))

    def test_protected_candidates_match_a_magi_directory_at_any_depth(self) -> None:
        # [MAGI §6 MS2 loop 18, F1a1-caspar] the product's is_never_reviewed tests every component, so a
        # nested .magi is left out too; counting only the first component left it unexercised by S51.
        listed = frozenset({"a/b/.MAGI./x.toml", "a\\.magi\\y", "a/.magic/z", "a/magi/w", "a/b.magi/v"})
        with mock.patch.object(runs, "visible_paths", return_value=listed):
            self.assertEqual(("a/b/.MAGI./x.toml", "a\\.magi\\y"),
                             runs.protected_candidates(pathlib.Path(".")))

    def test_protected_candidates_never_fold_a_non_ascii_letter(self) -> None:
        # The product folds ASCII only: a dotted or dotless I is not the "i" of ".magi".
        listed = frozenset({".magİ/a", ".MAGı/b"})
        with mock.patch.object(runs, "visible_paths", return_value=listed):
            self.assertEqual((), runs.protected_candidates(pathlib.Path(".")))

    def test_the_copy_leaves_out_the_log_directory_whatever_its_spelling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ignore = runs._skip_top_level_logs(tmp)
            names = ["Logs", "logs.", "LOGS ", "magi.toml", "logsx", "logs"]
            self.assertEqual({"Logs", "logs.", "LOGS ", "logs"}, ignore(tmp, names))
            self.assertEqual(set(), ignore(os.path.join(tmp, "sub"), names))


class ReviewLogBoundTests(unittest.TestCase):
    """[MAGI §6 MS2 loop 16, F1a1-caspar, F1a1-melchior] a bounded capture, and nothing left out unnamed."""

    def _logs_dir(self, root: pathlib.Path) -> pathlib.Path:
        logs_dir = root / "scratch" / runs.REVIEW_WORKSPACE_DIR / ".magi" / "logs"
        logs_dir.mkdir(parents=True)
        return logs_dir

    def test_a_log_file_over_the_per_file_cap_is_named_not_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            executor = _executor(root)
            logs_dir = self._logs_dir(root)
            (logs_dir / "a.log").write_bytes(b"1234")
            (logs_dir / "b.log").write_bytes(b"12345")
            unreadable = []
            with mock.patch.object(runs, "MAX_REVIEW_LOG_FILE_BYTES", 4), \
                    mock.patch.object(runs, "_open_listed",
                                      side_effect=runs._open_listed) as read:
                captured = executor._capture_review_log(runs.DEFINITIONS["R15"], unreadable)
            self.assertEqual((b"1234",), captured)
            self.assertEqual(["a.log"], [pathlib.Path(call.args[0]).name for call in read.call_args_list])
            self.assertEqual(1, len(unreadable))
            self.assertTrue(unreadable[0].startswith("b.log ("))
            self.assertIn("4", unreadable[0])

    def test_log_files_past_the_total_cap_are_named_not_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            executor = _executor(root)
            logs_dir = self._logs_dir(root)
            for name in ("a.log", "b.log", "c.log"):
                (logs_dir / name).write_bytes(b"123")
            unreadable = []
            with mock.patch.object(runs, "MAX_REVIEW_LOG_TOTAL_BYTES", 6):
                captured = executor._capture_review_log(runs.DEFINITIONS["R15"], unreadable)
            self.assertEqual((b"123", b"123"), captured)
            self.assertEqual(1, len(unreadable))
            self.assertTrue(unreadable[0].startswith("c.log ("))

    def test_a_file_that_grew_past_the_cap_after_its_size_was_read_is_dropped_and_named(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            executor = _executor(root)
            logs_dir = self._logs_dir(root)
            (logs_dir / "a.log").write_bytes(b"12")
            unreadable = []
            with mock.patch.object(runs, "MAX_REVIEW_LOG_FILE_BYTES", 4), \
                    mock.patch.object(runs, "_open_listed", return_value=io.BytesIO(b"123456")):
                captured = executor._capture_review_log(runs.DEFINITIONS["R15"], unreadable)
            self.assertEqual((), captured)
            self.assertEqual(1, len(unreadable))
            self.assertTrue(unreadable[0].startswith("a.log ("))

    def _read_growing(self, spent_before: int) -> tuple:
        """Read one file whose size reads as 1 byte but which keeps growing, with a 4-byte file cap.

        A file that grew after its size was read, or one a detached writer keeps extending, must be
        read no further than the cap allows: the bytes the double hands out are counted.

        Returns:
            tuple: What was kept, what was refused, and how many bytes the double handed out.
        """
        class GrowingFile:
            """A file that has more bytes than any read asks for."""

            def __init__(self) -> None:
                self.handed_out = 0

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self, size: int = -1) -> bytes:
                size = 1 << 20 if size is None or size < 0 else size
                self.handed_out += size
                return b"x" * size

        growing = GrowingFile()
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "a.log"
            path.write_bytes(b"x")
            refused = []
            filler = [pathlib.Path(tmp) / "filler.log"] if spent_before else []
            for item in filler:
                item.write_bytes(b"y" * spent_before)
            real_open = runs._open_listed

            def open_(listed, expected):
                return growing if pathlib.Path(listed) == path else real_open(listed, expected)

            with mock.patch.object(runs, "MAX_REVIEW_LOG_FILE_BYTES", 4), \
                    mock.patch.object(runs, "MAX_REVIEW_LOG_TOTAL_BYTES", 6), \
                    mock.patch.object(runs, "_open_listed", open_):
                kept = runs._read_bounded(tmp, filler + [path], refused)
        return kept, refused, growing.handed_out

    def test_a_file_that_keeps_growing_is_read_no_further_than_the_file_cap(self) -> None:
        # [MAGI §6 MS2 loop 17, F1a1-caspar/melchior/balthasar] the cap was checked against stat() and the
        # file then read whole, so a file that grew in between was held in memory unbounded.
        kept, refused, handed_out = self._read_growing(spent_before=0)
        self.assertEqual([], kept)
        self.assertEqual(["a.log (%s)" % (runs.OVER_FILE_CAP % 4)], refused)
        self.assertLessEqual(handed_out, 4 + 1)

    def test_a_file_that_keeps_growing_is_read_no_further_than_the_total_left(self) -> None:
        kept, refused, handed_out = self._read_growing(spent_before=4)
        self.assertEqual([b"yyyy"], kept)
        self.assertEqual(["a.log (%s)" % (runs.OVER_TOTAL_CAP % 6)], refused)
        self.assertLessEqual(handed_out, 2 + 1)

    def test_the_caps_are_positive_and_the_total_holds_one_whole_file(self) -> None:
        self.assertGreater(runs.MAX_REVIEW_LOG_FILE_BYTES, 0)
        self.assertGreaterEqual(runs.MAX_REVIEW_LOG_TOTAL_BYTES, runs.MAX_REVIEW_LOG_FILE_BYTES)

    def test_a_non_regular_entry_under_the_log_directory_is_named_not_read(self) -> None:
        # A fifo, socket or device file is neither a directory nor a regular file. Windows makes none
        # without a privilege, so the entry is a double; the real file next to it is captured as usual.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            executor = _executor(root)
            logs_dir = self._logs_dir(root)
            (logs_dir / "magi-run.log").write_bytes(b"this run's log")
            fifo = types.SimpleNamespace(
                path=str(logs_dir / "pipe"), name="pipe",
                is_dir=lambda follow_symlinks=True: False,
                is_file=lambda follow_symlinks=True: False)
            real_scandir = os.scandir

            def scandir(path):
                entries = list(real_scandir(path))
                if pathlib.Path(path) == logs_dir:
                    entries.append(fifo)
                return iter(entries)

            unreadable = []
            with mock.patch.object(runs.os, "scandir", side_effect=scandir):
                captured = executor._capture_review_log(runs.DEFINITIONS["R15"], unreadable)
            self.assertEqual((b"this run's log",), captured)
            self.assertEqual(["pipe (%s)" % runs.NOT_A_REGULAR_FILE], unreadable)


class ReviewLogSwapTests(unittest.TestCase):
    """[MAGI §6 MS2 loop 19, F1a1-balthasar] a log file swapped after the walk listed it is never read.

    The walk tests each entry, but the read used to open the file again by path and follow whatever
    stood there by then, so a surviving child process could swap a listed file for a link (or a hard
    link) to bytes outside the workspace between the two.
    """

    _OUTSIDE = b"bytes outside the workspace"

    def _capture_with_swap(self, swap) -> tuple:
        """Capture R15's log while *swap* replaces ``a.log`` right after the walk listed it.

        Args:
            swap: Called with ``(a.log, outside file)`` once the walk has returned.

        Returns:
            tuple: What was captured, and the entries named as left out.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            executor = _executor(root)
            logs_dir = root / "scratch" / runs.REVIEW_WORKSPACE_DIR / ".magi" / "logs"
            logs_dir.mkdir(parents=True)
            listed = logs_dir / "a.log"
            listed.write_bytes(b"this run's log")
            outside = root / "outside.txt"
            outside.write_bytes(self._OUTSIDE)
            real_walk = runs._regular_files_without_links

            def walk_then_swap(*args, **kwargs):
                files = real_walk(*args, **kwargs)
                listed.unlink()
                swap(listed, outside)
                return files

            unreadable = []
            with mock.patch.object(runs, "_regular_files_without_links", side_effect=walk_then_swap):
                captured = executor._capture_review_log(runs.DEFINITIONS["R15"], unreadable)
        return captured, unreadable

    def test_a_log_file_replaced_by_a_hard_link_after_listing_is_named_not_read(self) -> None:
        # A hard link is a regular file to every path test, so only the identity recorded when the walk
        # listed the file tells it from the one that was listed. Needs no privilege on any platform.
        captured, unreadable = self._capture_with_swap(lambda link, target: os.link(target, link))
        self.assertNotIn(self._OUTSIDE, b"".join(captured))
        self.assertEqual(["a.log (%s)" % runs.CHANGED_AFTER_LISTING], unreadable)

    def test_a_log_file_swapped_for_a_symlink_after_listing_is_never_read_through(self) -> None:
        def symlink(link, target):
            try:
                os.symlink(target, link)
            except (OSError, NotImplementedError) as exc:
                raise unittest.SkipTest("cannot create a symlink here: %s" % exc) from exc

        captured, unreadable = self._capture_with_swap(symlink)
        self.assertNotIn(self._OUTSIDE, b"".join(captured))
        self.assertEqual(1, len(unreadable))
        self.assertTrue(unreadable[0].startswith("a.log ("), unreadable)


def _make_symlink(target: pathlib.Path, link: pathlib.Path) -> None:
    """Make *link* a real directory symlink to *target*.

    Raises:
        unittest.SkipTest: When the platform or the account cannot make one (Windows without the
            symlink privilege).
    """
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        raise unittest.SkipTest("cannot create a symlink here: %s" % exc) from exc


def _make_junction(target: pathlib.Path, link: pathlib.Path) -> None:
    """Make *link* a real Windows junction to *target*.

    Raises:
        unittest.SkipTest: Off Windows, where junctions do not exist.
    """
    try:
        import _winapi  # Windows only; a junction needs no privilege
    except ImportError as exc:
        raise unittest.SkipTest("junctions exist on Windows only") from exc
    _winapi.CreateJunction(str(target), str(link))


class ReviewChainLinkTests(unittest.TestCase):
    """[MAGI §6 MS2 loop 17, F1a1-caspar] no link from the repository root down to the workspace."""

    def _linked_scratch(self, root: pathlib.Path, make_link) -> tuple:
        """A scratch area that is itself a link to *root*/outside, holding a workspace to keep."""
        outside = root / "outside"
        keep = outside / runs.REVIEW_WORKSPACE_DIR / "keep.txt"
        keep.parent.mkdir(parents=True)
        keep.write_text("not the harness's", encoding="utf-8")
        make_link(outside, root / "scratch")
        return outside, keep

    def _assert_scratch_link_refused(self, make_link) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            executor = _executor(root)
            outside, keep = self._linked_scratch(root, make_link)
            with mock.patch.object(runs, "REPOSITORY_ROOT", root):
                with self.assertRaises(HarnessError) as caught:
                    executor._prepare_review_workspace(runs.DEFINITIONS["R14"])
            self.assertIn("scratch", str(caught.exception))
            self.assertTrue(keep.is_file())
            self.assertFalse((outside / runs.REVIEW_WORKSPACE_DIR / ".magi").exists())

    def test_a_scratch_area_that_is_a_symlink_is_never_deleted_or_copied_through(self) -> None:
        # The containment check resolves the scratch area on both sides, so a linked scratch area passed it,
        # and the link walk started below it: rmtree emptied <outside>/review-ws and copytree wrote into it.
        self._assert_scratch_link_refused(_make_symlink)

    def test_a_scratch_area_that_is_a_junction_is_never_deleted_or_copied_through(self) -> None:
        self._assert_scratch_link_refused(_make_junction)

    def test_a_link_between_the_repository_root_and_the_scratch_area_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            real = root / "real-smoke"
            keep = real / "scratch" / runs.REVIEW_WORKSPACE_DIR / "keep.txt"
            keep.parent.mkdir(parents=True)
            keep.write_text("x", encoding="utf-8")
            _make_junction(real, root / "smoke")
            magi = root / ".magi"
            magi.mkdir()
            env = types.SimpleNamespace(root=root / "smoke" / "env", magi_dir=magi,
                                        scratch_dir=root / "smoke" / "scratch")
            executor = runs.RunExecutor(binary=None, env=env, config=None)
            with mock.patch.object(runs, "REPOSITORY_ROOT", root):
                with self.assertRaises(HarnessError) as caught:
                    executor._prepare_review_workspace(runs.DEFINITIONS["R14"])
            self.assertIn("smoke", str(caught.exception))
            self.assertTrue(keep.is_file())


class ReviewLogChainTests(unittest.TestCase):
    """[MAGI §6 MS2 loop 17, F1a1-caspar/melchior] the log capture re-tests the chain the product ran in."""

    def _outside_log(self, root: pathlib.Path) -> pathlib.Path:
        """A tree outside the scratch area shaped like a workspace, with a log that is not this run's."""
        outside = root / "outside"
        logs = outside / ".magi" / "logs"
        logs.mkdir(parents=True)
        (logs / "elsewhere.log").write_bytes(b"not this run's log")
        return outside

    def _capture(self, root: pathlib.Path) -> tuple:
        """Capture R15's log with the repository root at *root*; return the capture and what was refused."""
        executor = _executor(root)
        unreadable = []
        with mock.patch.object(runs, "REPOSITORY_ROOT", root):
            captured = executor._capture_review_log(runs.DEFINITIONS["R15"], unreadable)
        return captured, unreadable

    def _assert_workspace_link_not_read(self, make_link) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            outside = self._outside_log(root)
            (root / "scratch").mkdir()
            make_link(outside, root / "scratch" / runs.REVIEW_WORKSPACE_DIR)
            captured, unreadable = self._capture(root)
            self.assertEqual((), captured)
            self.assertEqual(1, len(unreadable))
            self.assertIn(runs.REVIEW_WORKSPACE_DIR, unreadable[0])
            self.assertIn(runs.LINK_NOT_FOLLOWED, unreadable[0])

    def test_a_workspace_swapped_for_a_junction_is_never_read_through(self) -> None:
        self._assert_workspace_link_not_read(_make_junction)

    def test_a_workspace_swapped_for_a_symlink_is_never_read_through(self) -> None:
        self._assert_workspace_link_not_read(_make_symlink)

    def test_a_scratch_area_swapped_for_a_junction_is_never_read_through(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            holder = root / "holder"
            holder.mkdir()
            shutil.move(str(self._outside_log(root)), str(holder / runs.REVIEW_WORKSPACE_DIR))
            _make_junction(holder, root / "scratch")
            captured, unreadable = self._capture(root)
            self.assertEqual((), captured)
            self.assertEqual(1, len(unreadable))
            self.assertIn("scratch", unreadable[0])
            self.assertIn(runs.LINK_NOT_FOLLOWED, unreadable[0])

    @unittest.skipUnless(_git_available(), "git is required")
    def test_a_product_that_swaps_its_workspace_for_a_link_has_nothing_read_through_it(self) -> None:
        # The chain was tested before the product ran; the product runs with the workspace as its cwd and
        # can replace it before it exits.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            executor = _executor(root)
            outside = self._outside_log(root)
            workspace = root / "scratch" / runs.REVIEW_WORKSPACE_DIR

            def swap(_definition, _stdin):
                shutil.rmtree(workspace)
                _make_junction(outside, workspace)
                return runs.ProductOutput(stdout=b"", stderr=b"", exit_code=2, command=["magi-rs"])

            with mock.patch.object(runs, "REPOSITORY_ROOT", root), \
                    mock.patch.object(executor, "_invoke", side_effect=swap):
                result = executor.execute(runs.DEFINITIONS["R15"])
            self.assertEqual((), result.review_log)
            self.assertTrue(any(runs.LINK_NOT_FOLLOWED in entry
                                for entry in result.review_log_unreadable))

    def test_a_log_directory_swapped_for_a_link_after_the_chain_check_is_never_read_through(self) -> None:
        # [MAGI §6 MS2 loop 19, F1a1-balthasar] a surviving child swaps .magi/logs between the chain check
        # and the walk: the walk lists, and the read opens, what the link points at.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            executor = _executor(root)
            outside = self._outside_log(root)
            logs_dir = root / "scratch" / runs.REVIEW_WORKSPACE_DIR / ".magi" / "logs"
            logs_dir.mkdir(parents=True)
            (logs_dir / "magi-run.log").write_bytes(b"this run's log")
            real_check = executor._first_link_down_to
            checks = []

            def check_then_swap(*below):
                found = real_check(*below)
                checks.append(below)
                if len(checks) == 1:
                    shutil.rmtree(logs_dir)
                    _make_junction(outside / ".magi" / "logs", logs_dir)
                return found

            unreadable = []
            with mock.patch.object(runs, "REPOSITORY_ROOT", root), \
                    mock.patch.object(executor, "_first_link_down_to", side_effect=check_then_swap):
                captured = executor._capture_review_log(runs.DEFINITIONS["R15"], unreadable)
            self.assertNotIn(b"not this run's log", b"".join(captured or ()))
            self.assertEqual((), captured)
            self.assertTrue(any(runs.LINK_NOT_FOLLOWED in entry for entry in unreadable), unreadable)


class RefusalRunModelCallTests(unittest.TestCase):
    """[MAGI §6 MS2 loop 19, F1a1-balthasar] R13-R16 prove that no model was contacted.

    Their exit code and stderr are what a refusal looks like whether or not a model was paid for first,
    so every endpoint their workspace declares is pointed at a local counter that answers nothing, and
    the run records each request past the capability probe.
    """

    _CONFIG = ('provider = "ollama"\n'
               'base_url = "http://localhost:11434/v1"\n'
               "[magi]\n"
               '# base_url = "http://commented.invalid/v1"\n'
               'melchior_model = "m"\n'
               "[embedding]\n"
               '  base_url = "http://elsewhere.invalid:11434/v1"\n')

    def _execute(self, run_id: str, paths: tuple, config: str = _CONFIG) -> tuple:
        """Execute *run_id* with a stand-in product that POSTs to *paths* under its configured daemon.

        Returns:
            tuple: The result, the workspace config the product saw, and its environment.
        """
        import tomllib
        import urllib.error
        import urllib.request
        seen = {}

        def invoke(argv, stdin, env, timeout, cwd):
            text = (pathlib.Path(cwd) / ".magi" / "magi.toml").read_text(encoding="utf-8")
            seen["config"], seen["env"] = text, env
            base = tomllib.loads(text)["base_url"]
            daemon = base[:-len("/v1")] if base.endswith("/v1") else base
            for path in paths:
                request = urllib.request.Request(daemon + path, data=b"{}", method="POST")
                try:
                    urllib.request.urlopen(request, timeout=5).close()
                except urllib.error.HTTPError as exc:
                    exc.close()
            return runs.ProductOutput(stdout=b"", stderr=b"", exit_code=2, command=list(argv))

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            magi = root / ".magi"
            magi.mkdir()
            (magi / "magi.toml").write_text(config, encoding="utf-8")
            binary = mock.Mock()
            binary.invoke.side_effect = invoke
            env = types.SimpleNamespace(root=root, magi_dir=magi, scratch_dir=root / "scratch")
            executor = runs.RunExecutor(binary=binary, env=env,
                                        config=types.SimpleNamespace(passphrase="toy"))
            result = executor.execute(runs.DEFINITIONS[run_id])
        return result, seen["config"], seen["env"]

    def test_the_refusal_runs_and_only_they_forbid_model_calls(self) -> None:
        forbidding = {run_id for run_id, definition in runs.DEFINITIONS.items()
                      if definition.forbids_model_calls}
        self.assertEqual({"R13", "R14", "R15", "R16"}, forbidding)

    def test_forbidding_model_calls_without_a_review_workspace_is_refused(self) -> None:
        with self.assertRaises(HarnessError):
            runs.RunDefinition(run_id="Rx", argv=("query",), stdin=b"", needs_trio=False,
                               payload_size=runs.PAYLOAD_SMALL, timeout_s=1,
                               forbids_model_calls=True)

    @unittest.skipUnless(_git_available(), "git is required")
    def test_a_refusal_run_that_reaches_a_model_records_the_request(self) -> None:
        for run_id in ("R13", "R14", "R15", "R16"):
            result, _, _ = self._execute(run_id, ("/api/show", "/api/chat?stream=1"))
            self.assertEqual(("POST /api/chat",), result.model_requests, run_id)

    @unittest.skipUnless(_git_available(), "git is required")
    def test_a_refusal_run_that_only_probes_records_nothing(self) -> None:
        result, _, _ = self._execute("R15", ("/api/show", "/api/tags"))
        self.assertEqual((), result.model_requests)

    @unittest.skipUnless(_git_available(), "git is required")
    def test_every_endpoint_a_refusal_run_declares_points_at_the_counter(self) -> None:
        _, config, env = self._execute("R13", ())
        active = [line.split("=", 1)[1].strip() for line in config.splitlines()
                  if line.strip().startswith("base_url")]
        self.assertEqual(2, len(active))
        self.assertEqual(1, len(set(active)))
        url = active[0].strip('"')
        self.assertTrue(url.startswith("http://127.0.0.1:"), url)
        self.assertEqual(url, env[runs._BACKEND_URL_VARIABLE])
        self.assertIn('# base_url = "http://commented.invalid/v1"', config)

    @unittest.skipUnless(_git_available(), "git is required")
    def test_a_config_without_a_root_endpoint_gets_one_pointing_at_the_counter(self) -> None:
        # Without a root base_url the product's built-in default, a real local daemon, would apply.
        _, config, _ = self._execute("R13", (), config='[magi]\nmelchior_model = "m"\n')
        import tomllib
        self.assertTrue(tomllib.loads(config)["base_url"].startswith("http://127.0.0.1:"))

    @unittest.skipUnless(_git_available(), "git is required")
    def test_a_run_allowed_to_reach_models_keeps_its_config(self) -> None:
        result, config, _ = self._execute("R12", ())
        self.assertEqual(self._CONFIG, config)
        self.assertEqual((), result.model_requests)


if __name__ == "__main__":
    unittest.main()
