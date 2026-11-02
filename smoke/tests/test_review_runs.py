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
from smoke.errors import HarnessError, TimedOut
from smoke.tests import support


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
        # [MAGI §6 MS2 loop 27, F1b1-melchior] whole declaration lines were compared, so one name declared
        # twice with different parameters passed; the identifier is what must not repeat.
        import re
        lines = [line for _, body in runs.review_input_files()
                 for line in body.decode().splitlines() if line.startswith("pub fn ")]
        # [MAGI §6 MS2 loop 28, F1b1-melchior] a declaration the pattern missed raised AttributeError on
        # None; each must parse, generic parameters included, and fail as an assertion naming the line.
        matches = [(line, re.match(r"pub fn ([A-Za-z_][A-Za-z0-9_]*)\s*[(<]", line)) for line in lines]
        for line, match in matches:
            self.assertIsNotNone(match, line)
        names = [match.group(1) for _, match in matches]
        self.assertTrue(names)
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
        # [MAGI §6 MS2 loop 29, F1b1a-melchior] the name alone was pinned; the path must also leave the
        # workspace it is joined to, or R13 reviews something inside it and calls models.
        workspace = os.path.abspath(os.path.join("scratch", runs.REVIEW_WORKSPACE_DIR))
        joined = os.path.normpath(os.path.join(workspace, runs.REVIEW_OUTSIDE_PATH))
        self.assertNotEqual(workspace, os.path.commonpath([workspace, joined]))

    def test_the_review_runs_read_the_inputs_they_write(self) -> None:
        # [MAGI §6 MS2 loop 29, F1b1a-melchior] R12 reviews the generated files and R14 the skipped ones.
        self.assertEqual(runs.REVIEW_GEN_DIR, runs.DEFINITIONS["R12"].argv[-1])
        self.assertTrue(all(path.startswith(runs.REVIEW_GEN_DIR + "/")
                            for path, _ in runs.DEFINITIONS["R12"].files))
        self.assertEqual(runs.REVIEW_SKIPPED_DIR, runs.DEFINITIONS["R14"].argv[-1])
        self.assertTrue(all(path.startswith(runs.REVIEW_SKIPPED_DIR + "/")
                            for path, _ in runs.DEFINITIONS["R14"].files))

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

    def test_a_failed_deletion_of_a_directory_keeps_its_search_bit(self) -> None:
        # [smoke review, Melchior] 0o600 on a directory drops its search bit on POSIX, so the retry
        # could not even list it; a directory is made 0o700, a file stays 0o600.
        import stat
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp) / "d"
            directory.mkdir()
            with mock.patch.object(runs, "_set_mode") as set_mode:
                runs._retry_writable(os.rmdir, str(directory), PermissionError(13, "access denied"))
            self.assertFalse(directory.exists())
        set_mode.assert_called_once_with(str(directory), stat.S_IRWXU)

    @unittest.skipUnless(os.chmod in os.supports_follow_symlinks,
                         "this platform's chmod cannot leave a link unfollowed")
    def test_the_mode_change_asks_chmod_not_to_follow_a_link(self) -> None:
        # [MAGI §6 MS2 loop 27, F1a1-caspar] _retry_writable tests for a link and then changed the mode by path,
        # which follows a link swapped in between: the mode of whatever it points at changed.
        import stat
        with mock.patch.object(runs.os, "chmod") as chmod:
            runs._set_mode("x", stat.S_IRWXU)
        chmod.assert_called_once_with("x", stat.S_IRWXU, follow_symlinks=False)

    @unittest.skipUnless(os.chmod not in os.supports_follow_symlinks and hasattr(os, "O_NOFOLLOW"),
                         "only where chmod follows every link and O_NOFOLLOW exists (Linux)")
    def test_the_mode_change_goes_through_a_descriptor_opened_without_following(self) -> None:
        import stat
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "x"
            path.write_bytes(b"x")
            real_open = os.open
            flags = []

            def recording_open(target, flag, *args, **kwargs):
                flags.append(flag)
                return real_open(target, flag, *args, **kwargs)

            with mock.patch.object(runs.os, "open", side_effect=recording_open):
                runs._set_mode(str(path), stat.S_IREAD | stat.S_IWRITE)
            self.assertTrue(flags and all(flag & os.O_NOFOLLOW for flag in flags), flags)
            self.assertEqual(stat.S_IREAD | stat.S_IWRITE, stat.S_IMODE(os.stat(path).st_mode))

    def test_the_mode_change_never_reaches_through_a_symlink(self) -> None:
        import stat
        with tempfile.TemporaryDirectory() as tmp:
            target = pathlib.Path(tmp) / "outside"
            target.write_bytes(b"x")
            os.chmod(target, stat.S_IREAD)
            before = os.stat(target).st_mode
            link = pathlib.Path(tmp) / "link"
            try:
                os.symlink(target, link)
            except (OSError, NotImplementedError) as exc:
                raise unittest.SkipTest("cannot create a symlink here: %s" % exc) from exc
            try:
                runs._set_mode(str(link), stat.S_IREAD | stat.S_IWRITE)
            except OSError:
                pass  # refusing the link outright is as good as changing the link itself
            self.assertEqual(before, os.stat(target).st_mode)
            os.chmod(target, stat.S_IREAD | stat.S_IWRITE)

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

    def test_a_path_that_cannot_be_resolved_is_a_harness_error(self) -> None:
        # [MAGI §6 MS2 loop 28, F1a1-caspar] Path.resolve raises RuntimeError on a symlink loop (OSError
        # from 3.13), outside the try that names a harness error, so the run ended in a traceback.
        for error in (RuntimeError("Symlink loop from 'x'"), OSError(40, "Too many levels of symbolic links")):
            with tempfile.TemporaryDirectory() as tmp:
                executor = _executor(pathlib.Path(tmp))
                with mock.patch.object(pathlib.Path, "resolve", side_effect=error):
                    with self.assertRaises(HarnessError, msg=repr(error)):
                        executor._prepare_review_workspace(runs.DEFINITIONS["R14"])
                    with self.assertRaises(HarnessError, msg=repr(error)):
                        executor._write_files(runs.DEFINITIONS["R12"], pathlib.Path(tmp))
                    with self.assertRaises(HarnessError, msg=repr(error)):
                        executor._command(runs.DEFINITIONS["R14"])

    def test_an_environment_magi_that_is_itself_a_link_is_never_copied(self) -> None:
        # [MAGI §6 MS2 loop 28, F1b1-balthasar] copytree follows a .magi that is a link and copies what it
        # points at into the workspace the product reads.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            outside = root / "outside"
            outside.mkdir()
            (outside / "elsewhere.toml").write_text("not the environment's\n", encoding="utf-8")
            env_dir = root / "env"
            env_dir.mkdir()
            _make_directory_link(outside, env_dir / ".magi")
            env = types.SimpleNamespace(root=env_dir, magi_dir=env_dir / ".magi", scratch_dir=root / "scratch")
            executor = runs.RunExecutor(binary=None, env=env, config=None)
            with mock.patch.object(runs, "REPOSITORY_ROOT", root):
                with self.assertRaises(HarnessError) as caught:
                    executor._prepare_review_workspace(runs.DEFINITIONS["R14"])
            self.assertIn(".magi", str(caught.exception))
            self.assertFalse((root / "scratch" / runs.REVIEW_WORKSPACE_DIR / ".magi").exists())

    def test_an_environment_magi_under_a_linked_directory_is_never_copied(self) -> None:
        # The link test covered .magi and what lies under it, not the directories above it: an environment
        # directory swapped for a link had copytree copy the tree it points at.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            outside = root / "outside"
            (outside / ".magi").mkdir(parents=True)
            (outside / ".magi" / "magi.toml").write_text("# not the environment's\n", encoding="utf-8")
            _make_directory_link(outside, root / "env")
            env = types.SimpleNamespace(root=root / "env", magi_dir=root / "env" / ".magi",
                                        scratch_dir=root / "scratch")
            executor = runs.RunExecutor(binary=None, env=env, config=None)
            with mock.patch.object(runs, "REPOSITORY_ROOT", root):
                with self.assertRaises(HarnessError) as caught:
                    executor._prepare_review_workspace(runs.DEFINITIONS["R14"])
            self.assertIn("env", str(caught.exception))
            self.assertFalse((root / "scratch" / runs.REVIEW_WORKSPACE_DIR / ".magi").exists())


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
        # The backslash path holds on POSIX too: is_never_reviewed splits on '/' and '\\' on every platform
        # (pinned by test_the_name_folding_is_the_one_is_never_reviewed_applies), and so does the harness.
        listed = frozenset({"a/b/.MAGI./x.toml", "a\\.magi\\y", "a/.magic/z", "a/magi/w", "a/b.magi/v"})
        with mock.patch.object(runs, "visible_paths", return_value=listed):
            self.assertEqual(("a/b/.MAGI./x.toml", "a\\.magi\\y"),
                             runs.protected_candidates(pathlib.Path(".")))

    def test_protected_candidates_are_sorted_whatever_order_git_lists_them_in(self) -> None:
        # [MAGI §6 MS2 loop 29, F1b1a-caspar] the tests above hand over a frozenset, whose order follows the
        # hash seed, and expect one fixed tuple: they rely on the sort. The sort is pinned here on an
        # ordered listing, reversed, so dropping it fails every time rather than under some seeds.
        listed = (".magi/z", "src/.magi/y", ".MAGI/b", ".magi/a")
        with mock.patch.object(runs, "visible_paths", return_value=listed):
            self.assertEqual((".MAGI/b", ".magi/a", ".magi/z", "src/.magi/y"),
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
            unreadable = []
            if hasattr(os, "mkfifo"):
                # [MAGI §6 MS2 loop 28] the double compared its argument with a path, and POSIX lists the
                # directory through a descriptor, so the test errored there; a real fifo needs no double.
                os.mkfifo(logs_dir / "pipe")
                captured = executor._capture_review_log(runs.DEFINITIONS["R15"], unreadable)
            else:
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

                with mock.patch.object(runs.os, "scandir", side_effect=scandir):
                    captured = executor._capture_review_log(runs.DEFINITIONS["R15"], unreadable)
            self.assertEqual((b"this run's log",), captured)
            self.assertEqual(["pipe (%s)" % runs.NOT_A_REGULAR_FILE], unreadable)

    def test_a_hard_link_already_under_the_log_directory_is_named_not_read(self) -> None:
        # [MAGI §6 MS2 loop 28, F1a1-melchior] a hard link is a regular file to every test, and its identity
        # is the one the walk lists, so a link to a file outside the workspace planted before the capture was
        # read like the run's own log. The product never writes one; a file with more links is refused.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            executor = _executor(root)
            logs_dir = self._logs_dir(root)
            (logs_dir / "magi-run.log").write_bytes(b"this run's log")
            outside = root / "outside.txt"
            outside.write_bytes(b"bytes outside the workspace")
            os.link(outside, logs_dir / "planted.log")
            unreadable = []
            captured = executor._capture_review_log(runs.DEFINITIONS["R15"], unreadable)
            self.assertEqual((b"this run's log",), captured)
            self.assertEqual(["planted.log (%s)" % runs.HARD_LINKED], unreadable)

    def test_a_log_directory_with_more_entries_than_the_cap_is_listed_no_further(self) -> None:
        # [MAGI §6 MS2 loop 28, F1a1-melchior] the byte caps bound what is read, not what is listed: a product
        # that wrote a vast number of empty files had every entry materialised, at no captured byte.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            executor = _executor(root)
            logs_dir = self._logs_dir(root)
            for index in range(6):
                (logs_dir / ("%d.log" % index)).write_bytes(b"x")
            real_scandir = os.scandir
            pulled = []

            class Counting:
                """Counts every entry the walk takes from a listing."""

                def __init__(self, listing) -> None:
                    self._listing = listing

                def __iter__(self):
                    return self

                def __next__(self):
                    entry = next(self._listing)
                    pulled.append(entry.name)
                    return entry

                def __enter__(self):
                    return self

                def __exit__(self, *_):
                    self.close()
                    return False

                def close(self) -> None:
                    self._listing.close()

            unreadable = []
            with mock.patch.object(runs, "MAX_REVIEW_LOG_ENTRIES", 3), \
                    mock.patch.object(runs.os, "scandir", side_effect=lambda arg: Counting(real_scandir(arg))):
                captured = executor._capture_review_log(runs.DEFINITIONS["R15"], unreadable)
            self.assertLessEqual(len(pulled), 3 + 1)
            self.assertLessEqual(len(captured), 3)
            self.assertEqual(["logs (%s)" % (runs.OVER_ENTRY_CAP % 3)], unreadable)

    def test_the_entry_cap_is_positive(self) -> None:
        self.assertGreater(runs.MAX_REVIEW_LOG_ENTRIES, 0)


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


class LinkHelperTests(unittest.TestCase):
    """[MAGI §6 MS2 loop 27, F1b1-melchior] the test helpers' links point where the tests mean them to.

    ``_winapi.CreateJunction(src, dst)`` takes the existing target first and the junction to create second
    (CPython's ``Lib/test`` uses it so). A reversed call would make every junction test exercise a link to
    the wrong directory, or skip, so the direction is pinned on a real link.
    """

    def _assert_points_at(self, make_link) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = pathlib.Path(tmp) / "target"
            target.mkdir()
            (target / "marker").write_bytes(b"x")
            link = pathlib.Path(tmp) / "link"
            make_link(target, link)
            self.assertTrue(runs._is_link(link))
            self.assertFalse(runs._is_link(target))
            self.assertEqual(os.path.realpath(target), os.path.realpath(link))
            self.assertTrue((link / "marker").is_file())

    def test_a_junction_points_at_its_target(self) -> None:
        self._assert_points_at(_make_junction)

    def test_a_directory_link_points_at_its_target(self) -> None:
        self._assert_points_at(_make_directory_link)


class ReviewNestedLogDirectorySwapTests(unittest.TestCase):
    """[MAGI §6 MS2 loop 25, F1a1-caspar] a nested log directory swapped after its link test is never listed.

    The walk tested each entry for a link and listed an approved directory later, by path, so a directory
    under ``.magi/logs`` replaced by a link in between had the walk list, and the read open, what the link
    points at. The post-read chain test stops at ``.magi/logs`` and never saw it.
    """

    _OUTSIDE = b"bytes outside the workspace"

    def test_a_nested_log_directory_swapped_for_a_link_after_its_test_is_never_read_through(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            executor = _executor(root)
            logs_dir = root / "scratch" / runs.REVIEW_WORKSPACE_DIR / ".magi" / "logs"
            nested = logs_dir / "sub"
            nested.mkdir(parents=True)
            (logs_dir / "a.log").write_bytes(b"this run's log")
            (nested / "b.log").write_bytes(b"nested log")
            outside = root / "outside"
            outside.mkdir()
            (outside / "b.log").write_bytes(self._OUTSIDE)
            real_is_link = runs._is_link
            swapped = []

            def test_then_swap(path) -> bool:
                found = real_is_link(path)
                if pathlib.Path(path) == nested and not swapped:
                    swapped.append(path)
                    shutil.rmtree(nested)
                    _make_directory_link(outside, nested)
                return found

            unreadable = []
            with mock.patch.object(runs, "REPOSITORY_ROOT", root), \
                    mock.patch.object(runs, "_is_link", side_effect=test_then_swap):
                captured = executor._capture_review_log(runs.DEFINITIONS["R15"], unreadable)
            self.assertTrue(swapped, "the walk never tested the nested directory")
            self.assertNotIn(self._OUTSIDE, b"".join(captured or ()))
            self.assertIn(b"this run's log", b"".join(captured or ()))
            self.assertTrue(any(entry.startswith("sub (") for entry in unreadable), unreadable)


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
            # [MAGI §6 MS2 loop 28, F1b1-melchior] a symlink where it can be made, so POSIX runs it too.
            _make_directory_link(real, root / "smoke")
            magi = root / ".magi"
            magi.mkdir()
            env = types.SimpleNamespace(root=root / "smoke" / "env", magi_dir=magi,
                                        scratch_dir=root / "smoke" / "scratch")
            executor = runs.RunExecutor(binary=None, env=env, config=None)
            with mock.patch.object(runs, "REPOSITORY_ROOT", root):
                with self.assertRaises(HarnessError) as caught:
                    executor._prepare_review_workspace(runs.DEFINITIONS["R14"])
            # [MAGI §6 MS2 loop 29, F1b1b-melchior] "smoke" alone is in any error naming a path through the
            # link; the refusal must name the linked component and say it is a link.
            self.assertIn("%s is a link" % (root / "smoke"), str(caught.exception))
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
        self._assert_scratch_link_not_read(_make_junction)

    def test_a_scratch_area_swapped_for_a_symlink_is_never_read_through(self) -> None:
        # [MAGI §6 MS2 loop 28, sweep of F1b1-melchior] the junction twin skips off Windows.
        self._assert_scratch_link_not_read(_make_symlink)

    def _assert_scratch_link_not_read(self, make_link) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            holder = root / "holder"
            holder.mkdir()
            shutil.move(str(self._outside_log(root)), str(holder / runs.REVIEW_WORKSPACE_DIR))
            make_link(holder, root / "scratch")
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
                _make_directory_link(outside, workspace)
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
                    _make_directory_link(outside / ".magi" / "logs", logs_dir)
                return found

            unreadable = []
            with mock.patch.object(runs, "REPOSITORY_ROOT", root), \
                    mock.patch.object(executor, "_first_link_down_to", side_effect=check_then_swap):
                captured = executor._capture_review_log(runs.DEFINITIONS["R15"], unreadable)
            self.assertNotIn(b"not this run's log", b"".join(captured or ()))
            self.assertEqual((), captured)
            self.assertTrue(any(runs.LINK_NOT_FOLLOWED in entry for entry in unreadable), unreadable)

    def test_a_log_directory_swapped_for_a_link_before_the_walk_is_never_read_once(self) -> None:
        # [MAGI §6 MS2 loop 27, F1a1-melchior] the walk listed its root by path, so a link swapped onto
        # .magi/logs after the chain check had the walk list, and the read open, what it points at; the
        # post-read check then dropped the capture, but the bytes outside the workspace had been read.
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
                    _make_directory_link(outside / ".magi" / "logs", logs_dir)
                return found

            real_read = runs._read_bounded
            read = []

            def recording_read(*args, **kwargs):
                captured = real_read(*args, **kwargs)
                read.extend(captured)
                return captured

            with mock.patch.object(runs, "REPOSITORY_ROOT", root), \
                    mock.patch.object(executor, "_first_link_down_to", side_effect=check_then_swap), \
                    mock.patch.object(runs, "_read_bounded", side_effect=recording_read):
                executor._capture_review_log(runs.DEFINITIONS["R15"], [])
            self.assertNotIn(b"not this run's log", b"".join(read))

    def test_an_ancestor_swapped_for_a_link_before_the_walk_is_never_read_once(self) -> None:
        # [MAGI §6 MS2 loop 28, F1a1-melchior] O_NOFOLLOW guarded only the last component: a .magi swapped
        # for a link after the chain check had the walk list, and the read open, the log behind it, before
        # the post-read check dropped the capture. Every component is now opened relative to its parent.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            executor = _executor(root)
            outside = self._outside_log(root)
            magi = root / "scratch" / runs.REVIEW_WORKSPACE_DIR / ".magi"
            (magi / "logs").mkdir(parents=True)
            (magi / "logs" / "magi-run.log").write_bytes(b"this run's log")
            real_check = executor._first_link_down_to
            checks = []

            def check_then_swap(*below):
                found = real_check(*below)
                checks.append(below)
                if len(checks) == 1:
                    shutil.rmtree(magi)
                    _make_directory_link(outside / ".magi", magi)
                return found

            real_read = runs._read_bounded
            read = []

            def recording_read(*args, **kwargs):
                captured = real_read(*args, **kwargs)
                read.extend(captured)
                return captured

            unreadable = []
            with mock.patch.object(runs, "REPOSITORY_ROOT", root), \
                    mock.patch.object(executor, "_first_link_down_to", side_effect=check_then_swap), \
                    mock.patch.object(runs, "_read_bounded", side_effect=recording_read):
                captured = executor._capture_review_log(runs.DEFINITIONS["R15"], unreadable)
            self.assertNotIn(b"not this run's log", b"".join(read))
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
                    support.loopback_open(request).close()
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
    def test_every_section_that_names_an_endpoint_is_pointed_at_the_counter(self) -> None:
        # [MAGI §6 MS2 loop 27, F1b1-melchior] the product's endpoint keys are the root's and those of
        # [magi], [embedding], [decision] and [code_review]; the gate's own endpoint is pinned here too.
        import tomllib
        sections = ("magi", "embedding", "decision", "code_review")
        config = 'base_url = "http://localhost:11434/v1"\n' + "".join(
            '[%s]\nbase_url = "http://%s.invalid/v1"\n' % (name, name) for name in sections)
        _, seen, env = self._execute("R13", (), config=config)
        document = tomllib.loads(seen)
        url = env[runs._BACKEND_URL_VARIABLE]
        self.assertEqual({"": url, **{name: url for name in sections}},
                         {"": document["base_url"],
                          **{name: document[name]["base_url"] for name in sections}})

    @unittest.skipUnless(_git_available(), "git is required")
    def test_a_config_without_a_root_endpoint_gets_one_pointing_at_the_counter(self) -> None:
        # Without a root base_url the product's built-in default, a real local daemon, would apply.
        _, config, _ = self._execute("R13", (), config='[magi]\nmelchior_model = "m"\n')
        import tomllib
        self.assertTrue(tomllib.loads(config)["base_url"].startswith("http://127.0.0.1:"))

    @unittest.skipUnless(_git_available(), "git is required")
    def test_an_endpoint_the_rewrite_cannot_reach_is_a_harness_error(self) -> None:
        # [smoke review, Melchior] an endpoint the line rewrite misses would reach a real backend, and
        # the counter, seeing nothing, would let R13-R16 pass vacuously. The rewritten config is
        # parsed and every endpoint checked before the run; a miss is the harness's setup (exit 3).
        for config in ('base_url = "http://localhost:11434/v1"\n'
                       '[embedding]\n"base_url" = "http://elsewhere.invalid/v1"\n',
                       'base_url = "http://localhost:11434/v1"\n'
                       'decision = { base_url = "http://elsewhere.invalid/v1" }\n',
                       'base_url = "http://localhost:11434/v1"\n'
                       '[[magi.fallback]]\n"base_url" = "http://elsewhere.invalid/v1"\n'):
            with self.assertRaises(HarnessError, msg=config):
                self._execute("R13", (), config=config)

    @unittest.skipUnless(_git_available(), "git is required")
    def test_an_anthropic_kind_in_a_refusal_run_is_a_harness_error(self) -> None:
        # An anthropic seat posts to a fixed URL that no base_url steers, so the counter cannot see it.
        for config in ('provider = "anthropic"\n', '[magi]\nkind = "anthropic"\n'):
            with self.assertRaises(HarnessError, msg=config):
                self._execute("R13", (), config=config)

    @unittest.skipUnless(_git_available(), "git is required")
    def test_a_refusal_run_accepts_only_ollama_where_the_product_reads_a_provider_kind(self) -> None:
        # [smoke review, Caspar] the probe allowlist is Ollama's, and the harness writes the product's
        # own scaffold (provider = "ollama"), so any other kind is a setup the counter was not built for.
        for config in ('provider = "openai-compat"\n', '[magi]\nkind = "openai-compat"\n',
                       'provider = "Ollama"\n'):
            with self.assertRaises(HarnessError, msg=config):
                self._execute("R13", (), config=config)

    @unittest.skipUnless(_git_available(), "git is required")
    def test_a_refusal_run_keeps_the_embedding_provider_label_and_blank_kinds(self) -> None:
        # [embedding].provider is a label with its own vocabulary ("openai" by default), never a
        # provider kind, and a blank kind is absent to the product.
        for config in ('provider = " ollama "\n[magi]\nkind = ""\n[embedding]\nprovider = "openai"\n',
                       '[magi]\nkind = "ollama"\n'):
            result, _, _ = self._execute("R13", (), config=config)
            self.assertEqual((), result.model_requests, config)

    @unittest.skipUnless(_git_available(), "git is required")
    def test_a_rewrite_that_changes_anything_but_an_endpoint_is_a_harness_error(self) -> None:
        # [smoke review, Caspar] the line rewrite can reach inside a multi-line string; the parsed
        # result must equal the original everywhere but the endpoints.
        multiline = ('base_url = "http://localhost:11434/v1"\n[magi]\nmelchior_model = """\n'
                     'base_url = "kept verbatim"\n"""\n')
        with self.assertRaises(HarnessError):
            self._execute("R13", (), config=multiline)

        original = runs.point_config_at

        def rewrite_and_drift(path, url):
            original(path, url)
            text = pathlib.Path(path).read_text(encoding="utf-8")
            pathlib.Path(path).write_text(text.replace('"m"', '"drifted"'), encoding="utf-8")

        with mock.patch.object(runs, "point_config_at", rewrite_and_drift):
            with self.assertRaises(HarnessError):
                self._execute("R13", ())

    @unittest.skipUnless(_git_available(), "git is required")
    def test_a_refusal_run_blanks_the_provider_override_of_the_harness_environment(self) -> None:
        # MAGI_PROVIDER=anthropic in the harness's own shell would steer the run past the counter;
        # the product reads a blank variable as absent.
        _, _, env = self._execute("R13", ())
        self.assertEqual("", env.get("MAGI_PROVIDER"))

    @unittest.skipUnless(_git_available(), "git is required")
    def test_a_refusal_run_never_inherits_a_proxy_from_the_harness_environment(self) -> None:
        # [MAGI loop 26] reqwest honours the system proxy and does not exempt 127.0.0.1, so an inherited
        # HTTP_PROXY sent the counter's requests to the proxy and S48-S51 passed with nothing counted.
        from smoke import binary
        _, _, env = self._execute("R13", ())
        inherited = {"HTTP_PROXY": "http://corp:3128", "https_proxy": "http://corp:3128",
                     "ALL_PROXY": "socks5://corp:1080", "all_proxy": "socks5://corp:1080",
                     "http_proxy": "http://corp:3128", "no_proxy": "corp.example"}
        with mock.patch.dict(os.environ, inherited):
            child = binary.child_environment(env)
        proxies = {name for name in child
                   if name.upper() in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"}}
        self.assertEqual(set(), proxies)
        self.assertEqual({"NO_PROXY": "127.0.0.1,localhost,::1"},
                         {name: value for name, value in child.items() if name.upper() == "NO_PROXY"})

    @unittest.skipUnless(_git_available(), "git is required")
    def test_a_refusal_run_that_times_out_still_records_what_it_sent(self) -> None:
        # [smoke review, Caspar] pins that the requests seen before a timeout are reported.
        def invoke(argv, stdin, env, timeout, cwd):
            import tomllib
            import urllib.error
            import urllib.request
            base = tomllib.loads((pathlib.Path(cwd) / ".magi" / "magi.toml")
                                 .read_text(encoding="utf-8"))["base_url"]
            request = urllib.request.Request(base[:-len("/v1")] + "/api/chat", data=b"{}",
                                             method="POST")
            try:
                support.loopback_open(request).close()
            except urllib.error.HTTPError as exc:
                exc.close()
            raise TimedOut("expired", output=None)

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / ".magi").mkdir()
            (root / ".magi" / "magi.toml").write_text(self._CONFIG, encoding="utf-8")
            binary = mock.Mock()
            binary.invoke.side_effect = invoke
            env = types.SimpleNamespace(root=root, magi_dir=root / ".magi",
                                        scratch_dir=root / "scratch")
            result = runs.RunExecutor(binary=binary, env=env,
                                      config=types.SimpleNamespace(passphrase="toy")
                                      ).execute(runs.DEFINITIONS["R15"])
        self.assertTrue(result.timed_out)
        self.assertEqual(("POST /api/chat",), result.model_requests)

    @unittest.skipUnless(_git_available(), "git is required")
    def test_a_refusal_run_ending_in_a_harness_failure_still_names_what_it_sent(self) -> None:
        # [MAGI §6 MS2 loop 27, F1a1-melchior] only a timeout was caught inside the counted block, so any other
        # exception skipped the record and the exit-3 report lost the requests that would explain it.
        def invoke(argv, stdin, env, timeout, cwd):
            import tomllib
            import urllib.error
            import urllib.request
            base = tomllib.loads((pathlib.Path(cwd) / ".magi" / "magi.toml")
                                 .read_text(encoding="utf-8"))["base_url"]
            request = urllib.request.Request(base[:-len("/v1")] + "/api/chat", data=b"{}",
                                             method="POST")
            try:
                support.loopback_open(request).close()
            except urllib.error.HTTPError as exc:
                exc.close()
            raise HarnessError("the stand-in failed after sending")

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / ".magi").mkdir()
            (root / ".magi" / "magi.toml").write_text(self._CONFIG, encoding="utf-8")
            binary = mock.Mock()
            binary.invoke.side_effect = invoke
            env = types.SimpleNamespace(root=root, magi_dir=root / ".magi",
                                        scratch_dir=root / "scratch")
            executor = runs.RunExecutor(binary=binary, env=env,
                                        config=types.SimpleNamespace(passphrase="toy"))
            with self.assertRaises(HarnessError) as caught:
                executor.execute(runs.DEFINITIONS["R15"])
        notes = getattr(caught.exception, "__notes__", [])
        self.assertTrue(any("POST /api/chat" in note for note in notes), notes)

    @unittest.skipUnless(_git_available(), "git is required")
    def test_a_refusal_run_flooding_the_counter_still_names_the_request_past_the_cap(self) -> None:
        with mock.patch.object(runs, "MODEL_COUNTER_MAX_RECORDED", 2):
            result, _, _ = self._execute("R15", ("/api/show",) * 3 + ("/api/chat",))
        self.assertEqual(1, len(result.model_requests))
        self.assertIn("1 request past the probe", result.model_requests[0])
        self.assertIn("overflowed", result.model_requests[0])

    @unittest.skipUnless(_git_available(), "git is required")
    def test_a_run_allowed_to_reach_models_keeps_its_config(self) -> None:
        result, config, _ = self._execute("R12", ())
        self.assertEqual(self._CONFIG, config)
        self.assertEqual((), result.model_requests)


class RequestLogTests(unittest.TestCase):
    """The model counter keeps a bounded record, yet counts every request it saw."""

    def test_a_flood_past_the_cap_keeps_only_the_first_requests_but_counts_them_all(self) -> None:
        log = runs.RequestLog(cap=3)
        for index in range(5):
            log.record("POST", "/api/chat/%d" % index)
        self.assertEqual((("POST", "/api/chat/0"), ("POST", "/api/chat/1"), ("POST", "/api/chat/2")),
                         log.recorded)
        self.assertEqual(5, log.total)
        self.assertEqual(5, log.past_the_probe_total)
        self.assertTrue(log.overflowed)

    def test_the_default_cap_is_the_named_constant(self) -> None:
        log = runs.RequestLog()
        for _ in range(runs.MODEL_COUNTER_MAX_RECORDED + 1):
            log.record("POST", "/api/chat")
        self.assertEqual(runs.MODEL_COUNTER_MAX_RECORDED, len(log.recorded))
        self.assertTrue(log.overflowed)

    def test_a_request_past_the_probe_beyond_the_cap_is_still_reported(self) -> None:
        log = runs.RequestLog(cap=2)
        for _ in range(3):
            log.record("POST", "/api/show")
        log.record("POST", "/api/chat")
        self.assertEqual(1, log.past_the_probe_total)
        reported = runs.requests_past_the_probe(log)
        self.assertEqual(1, len(reported))
        self.assertIn("1 request past the probe", reported[0])
        self.assertIn("overflowed", reported[0])

    def test_a_probe_only_flood_past_the_cap_reports_nothing(self) -> None:
        log = runs.RequestLog(cap=2)
        for _ in range(10):
            log.record("POST", "/api/tags")
        self.assertEqual(10, log.total)
        self.assertEqual(0, log.past_the_probe_total)
        self.assertTrue(log.overflowed)
        self.assertEqual((), runs.requests_past_the_probe(log))

    def test_recorded_requests_past_the_probe_are_named_with_the_overflow_total(self) -> None:
        log = runs.RequestLog(cap=2)
        log.record("POST", "/api/show")
        log.record("POST", "/api/chat")
        log.record("POST", "/v1/embeddings")
        log.record("POST", "/v1/embeddings")
        self.assertEqual(("POST /api/chat",
                          "3 requests past the probe in all; the counter overflowed after "
                          "recording the first 2 requests"),
                         runs.requests_past_the_probe(log))

    def test_within_the_cap_only_the_requests_past_the_probe_are_named(self) -> None:
        log = runs.RequestLog(cap=4)
        log.record("POST", "/api/show")
        log.record("POST", "/api/chat")
        self.assertFalse(log.overflowed)
        self.assertEqual(("POST /api/chat",), runs.requests_past_the_probe(log))


class ModelCounterShutdownTests(unittest.TestCase):
    """[MAGI §6 MS2 loop 25, F1a1-caspar] a request still waiting to be accepted at shutdown is counted.

    ``serve_forever`` stops as soon as shutdown is requested, even with a connection ready to accept, so a
    request the product sent just before it exited (or was killed) stayed in the listen backlog, was dropped
    by ``server_close`` and never counted: a refusal run that paid for a model could pass.
    """

    def test_a_request_left_in_the_backlog_at_shutdown_is_still_counted(self) -> None:
        # [MAGI §6 MS2 loop 27, F1b1-balthasar/caspar] synchronised by an event, not by polling a private
        # flag of BaseServer: the loop accepts nothing and returns once shutdown is asked for, which is what
        # serve_forever does with a connection still waiting.
        import http.server
        import socket
        import threading
        shutdown_requested = threading.Event()

        def serve_accepting_nothing(server, poll_interval=0.5):
            shutdown_requested.wait(30)

        def request_shutdown(server):
            shutdown_requested.set()

        with mock.patch.object(http.server.ThreadingHTTPServer, "serve_forever", serve_accepting_nothing), \
                mock.patch.object(http.server.ThreadingHTTPServer, "shutdown", request_shutdown):
            with runs.model_counter() as (url, log):
                port = int(url.rsplit(":", 1)[1].split("/", 1)[0])
                client = socket.create_connection(("127.0.0.1", port))
                client.sendall(b"POST /api/chat HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n")
                client.shutdown(socket.SHUT_WR)
            client.close()
        self.assertEqual(1, log.past_the_probe_total)
        self.assertEqual(("POST /api/chat",), runs.requests_past_the_probe(log))

    def test_a_request_drained_at_shutdown_is_recorded_before_the_block_ends(self) -> None:
        # [MAGI §6 MS2 loop 28, F1a1-caspar] handle_request starts the drained request on a thread and
        # returns at once; the log is complete only because server_close joins that thread
        # (ThreadingMixIn, daemon_threads off). The record is slowed down so a missing join shows.
        import http.server
        import socket
        import threading
        import time
        shutdown_requested = threading.Event()
        real_record = runs.RequestLog.record

        def slow_record(log, method, path):
            time.sleep(0.5)
            real_record(log, method, path)

        def serve_accepting_nothing(server, poll_interval=0.5):
            shutdown_requested.wait(30)

        with mock.patch.object(http.server.ThreadingHTTPServer, "serve_forever", serve_accepting_nothing), \
                mock.patch.object(http.server.ThreadingHTTPServer, "shutdown",
                                  lambda server: shutdown_requested.set()), \
                mock.patch.object(runs.RequestLog, "record", slow_record):
            with runs.model_counter() as (url, log):
                port = int(url.rsplit(":", 1)[1].split("/", 1)[0])
                client = socket.create_connection(("127.0.0.1", port))
                client.sendall(b"POST /api/chat HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n")
                client.shutdown(socket.SHUT_WR)
            client.close()
            self.assertEqual(("POST /api/chat",), runs.requests_past_the_probe(log))

    def test_a_client_trickling_its_body_cannot_hold_the_counter_open(self) -> None:
        # [MAGI §6 MS2 loop 27, F1a1-caspar] the body drain was bounded only per read, and server_close joins
        # every handler thread, so a product that declared a large body and sent a byte at a time held the
        # counter -- and the whole smoke run -- open for as long as it kept sending.
        import socket
        import threading
        import time
        stop = threading.Event()
        exited = threading.Event()
        seen = {}

        def trickle(port: int) -> None:
            client = socket.create_connection(("127.0.0.1", port))
            try:
                client.sendall(b"POST /api/chat HTTP/1.1\r\nHost: x\r\nContent-Length: 1000000\r\n\r\n")
                while not stop.wait(0.05):
                    client.sendall(b"x")
            except OSError:
                pass
            finally:
                client.close()

        def serve() -> None:
            with runs.model_counter() as (url, log):
                seen["log"] = log
                port = int(url.rsplit(":", 1)[1].split("/", 1)[0])
                threading.Thread(target=trickle, args=(port,), daemon=True).start()
                deadline = time.monotonic() + 30
                while log.total == 0 and time.monotonic() < deadline:
                    time.sleep(0.01)
            exited.set()

        with mock.patch.object(runs, "MODEL_COUNTER_CONNECTION_DEADLINE_S", 0.5):
            threading.Thread(target=serve, daemon=True).start()
            finished = exited.wait(30)
        stop.set()
        self.assertTrue(finished, "the counter never closed while a client kept trickling its body")
        self.assertEqual(("POST /api/chat",), runs.requests_past_the_probe(seen["log"]))


def _send_raw(port: int, data: bytes) -> None:
    """Send *data* to the counter on *port*, half-close, and read until the counter hangs up."""
    import socket
    with socket.create_connection(("127.0.0.1", port), timeout=30) as client:
        try:
            client.sendall(data)
            client.shutdown(socket.SHUT_WR)
            while client.recv(65536):
                pass
        except OSError:
            pass  # the counter may hang up before reading everything (an over-long line)


class ErrorBackendBodyTests(unittest.TestCase):
    """[MAGI §6 MS2 loop 28, coordinator] the echoing error backend answers whatever Content-Length it gets.

    It read ``int(Content-Length)`` bytes in one call: a non-numeric value raised in the handler and the
    product got no answer, a negative one read to EOF, and a huge one asked the reader for that many bytes at
    once. Each must still get the backend's 401, as the counter's bounded drain does.
    """

    def _answer(self, length: bytes, half_close: bool) -> bytes:
        """Send one POST declaring *length* with a 2-byte body; return the first bytes of the answer."""
        import socket
        with runs.error_backend() as url:
            port = int(url.rsplit(":", 1)[1].split("/", 1)[0])
            with socket.create_connection(("127.0.0.1", port), timeout=3) as client:
                client.sendall(b"POST /v1/chat HTTP/1.1\r\nHost: x\r\nContent-Length: " + length + b"\r\n\r\n{}")
                if half_close:
                    client.shutdown(socket.SHUT_WR)
                try:
                    return client.recv(64)
                except OSError:
                    return b""

    def test_a_non_numeric_length_is_still_answered(self) -> None:
        self.assertTrue(self._answer(b"abc", half_close=False).startswith(b"HTTP/1.0 401"))

    def test_a_negative_length_does_not_read_to_the_end_of_the_stream(self) -> None:
        # The client keeps its side open: a read to EOF waits until the per-read timeout instead of answering.
        self.assertTrue(self._answer(b"-1", half_close=False).startswith(b"HTTP/1.0 401"))

    def test_a_huge_length_is_drained_in_bounded_chunks_and_answered(self) -> None:
        reads = []
        real_drain = runs._drain_body

        def recording(rfile, headers):
            class Reader:
                def read(self, size=-1):
                    reads.append(size)
                    return rfile.read(size)
            return real_drain(Reader(), headers)

        with mock.patch.object(runs, "_drain_body", side_effect=recording):
            answer = self._answer(b"1000000000000000", half_close=True)
        self.assertTrue(answer.startswith(b"HTTP/1.0 401"), answer)
        self.assertTrue(reads)
        self.assertTrue(all(0 < size <= runs.MODEL_COUNTER_CHUNK_BYTES for size in reads), reads)


class LoopbackOpenTests(unittest.TestCase):
    """[MAGI §6 MS2 loop 28, F1b1-melchior] the stand-in doubles reach their loopback server on any host.

    ``urllib.request.urlopen`` honours ``HTTP_PROXY`` when ``no_proxy`` does not cover 127.0.0.1, so on a
    proxied host a stand-in's request never reached the counter and a test that asserts nothing was counted
    passed vacuously.
    """

    def test_a_loopback_request_never_goes_through_the_hosts_proxy(self) -> None:
        import urllib.error
        import urllib.request
        proxied = {"HTTP_PROXY": "http://127.0.0.1:9", "http_proxy": "http://127.0.0.1:9",
                   "ALL_PROXY": "http://127.0.0.1:9", "all_proxy": "http://127.0.0.1:9",
                   "NO_PROXY": "", "no_proxy": ""}
        with mock.patch.dict(os.environ, proxied), runs.model_counter() as (url, log):
            request = urllib.request.Request(url[:-len("/v1")] + "/api/chat", data=b"{}", method="POST")
            try:
                support.loopback_open(request).close()
            except urllib.error.HTTPError as exc:
                exc.close()
        self.assertEqual(("POST /api/chat",), runs.requests_past_the_probe(log))


class ModelCounterEveryRequestTests(unittest.TestCase):
    """[MAGI §6 MS2 loop 28, F1a1-melchior] every request reaches the record, whatever its method or shape.

    The counter recorded only from its ``do_*`` methods, so a method it does not implement (OPTIONS, TRACE,
    an unknown verb) was answered 501 by the base class, and a malformed or over-long request line 400 or
    414, all without a record: a refusal run whose product sent one passed as having reached no model.
    """

    def _counted(self, *requests: bytes) -> runs.RequestLog:
        """Send each raw request on its own connection to a counter, and return its log once it closed."""
        with runs.model_counter() as (url, log):
            port = int(url.rsplit(":", 1)[1].split("/", 1)[0])
            for data in requests:
                _send_raw(port, data)
        return log

    def test_a_method_the_counter_does_not_implement_is_still_counted(self) -> None:
        methods = ("OPTIONS", "TRACE", "CONNECT", "BREW")
        log = self._counted(*(b"%s /api/chat HTTP/1.1\r\nHost: x\r\n\r\n" % m.encode() for m in methods))
        self.assertEqual(tuple("%s /api/chat" % m for m in methods), runs.requests_past_the_probe(log))

    def test_a_malformed_request_is_still_counted(self) -> None:
        malformed = (b"GARBAGE\r\n\r\n",                         # no method and path
                     b"GET /api/chat HTTP/9.9\r\n\r\n",           # a version the server refuses
                     b"GET /" + b"a" * 70000 + b" HTTP/1.1\r\n",   # a line past the 64 KiB limit
                     b"POS")                                       # a line cut before its end
        log = self._counted(*malformed)
        self.assertEqual(len(malformed), log.past_the_probe_total)
        self.assertEqual(len(malformed), len(runs.requests_past_the_probe(log)))

    def test_a_path_that_does_not_parse_as_a_url_is_still_counted(self) -> None:
        # urlsplit raises on an unclosed IPv6 host, which killed the handler before its record.
        log = self._counted(b"GET http://[x/api/chat HTTP/1.1\r\nHost: x\r\n\r\n")
        self.assertEqual(1, log.past_the_probe_total)
        self.assertEqual(1, len(runs.requests_past_the_probe(log)))

    def test_a_connection_that_sends_nothing_is_no_request(self) -> None:
        log = self._counted(b"")
        self.assertEqual(0, log.total)

    def test_a_probe_with_any_method_is_still_the_probe(self) -> None:
        log = self._counted(b"POST /api/show HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n",
                            b"GET /api/tags HTTP/1.1\r\nHost: x\r\n\r\n")
        self.assertEqual(2, log.total)
        self.assertEqual((), runs.requests_past_the_probe(log))


if __name__ == "__main__":
    unittest.main()
