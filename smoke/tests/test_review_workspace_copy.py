# Author: Julian Bolivar
# Version: 0.25.0
# Date: 2026-11-02
"""Unit tests for how the review workspace copies the environment's ``.magi``.

The copy is the step that puts bytes in front of the product, so a link or a
hard link that reaches it must be refused there, not only by a check made
before it (a check and a later copy by path are two looks at the tree).
"""

import os
import pathlib
import shutil
import stat
import tempfile
import types
import unittest
from unittest import mock

from smoke import runs
from smoke.errors import HarnessError


def _git_available() -> bool:
    """Whether a ``git`` executable is on the path."""
    return shutil.which("git") is not None


def _environment(root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """A ``.magi`` with a config file and a nested directory, plus a secret outside it.

    Returns:
        tuple[pathlib.Path, pathlib.Path]: The ``.magi`` directory and the
        file outside it that must never be copied.
    """
    magi = root / "env" / ".magi"
    (magi / "nested").mkdir(parents=True)
    (magi / "magi.toml").write_text("# toy config\n", encoding="utf-8")
    (magi / "nested" / "kept.bin").write_bytes(b"\x00kept")
    outside = root / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("not for the workspace", encoding="utf-8")
    return magi, secret


def _hard_link(target: pathlib.Path, link: pathlib.Path) -> None:
    """Make *link* a hard link to *target*, or skip where the file system refuses one.

    Raises:
        unittest.SkipTest: When hard links cannot be made here.
    """
    try:
        os.link(target, link)
    except (OSError, NotImplementedError) as exc:
        raise unittest.SkipTest("cannot create a hard link here: %s" % exc) from exc


def _symlink(target: pathlib.Path, link: pathlib.Path, directory: bool = False) -> None:
    """Make *link* a symlink to *target* (or, for a directory, a junction where symlinks need a privilege).

    Raises:
        unittest.SkipTest: When the platform allows neither.
    """
    try:
        os.symlink(target, link, target_is_directory=directory)
        return
    except (OSError, NotImplementedError):
        pass
    if directory:
        try:
            import _winapi  # Windows only; a junction needs no privilege
            _winapi.CreateJunction(str(target), str(link))
            return
        except (ImportError, OSError):
            pass
    raise unittest.SkipTest("cannot create a link here")


class CopyWithoutLinksTests(unittest.TestCase):
    """[MAGI §6 MS2 loop 29, F1a1b-melchior, F1a1b-caspar] the copy itself refuses links and hard links."""

    def _copy(self, magi: pathlib.Path, destination: pathlib.Path) -> None:
        runs._copy_tree_without_links(magi, destination, runs._skip_top_level_logs(magi))

    def test_an_ordinary_tree_is_copied_whole_with_its_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            magi, _ = _environment(root)
            (magi / "logs").mkdir()
            (magi / "logs" / "old.log").write_bytes(b"earlier runs")
            destination = root / "copy"
            self._copy(magi, destination)
            self.assertEqual(b"# toy config\n", (destination / "magi.toml").read_bytes())
            self.assertEqual(b"\x00kept", (destination / "nested" / "kept.bin").read_bytes())
            self.assertFalse((destination / "logs").exists())

    def test_a_symlink_that_appears_after_the_preflight_is_refused_by_the_copy(self) -> None:
        # The preflight walk and the copy were two looks at the tree; a link swapped in between was followed
        # by copytree and what it points at reached the workspace. The preflight is bypassed here, which is
        # the state the race leaves.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            magi, secret = _environment(root)
            _symlink(secret, magi / "nested" / "swapped.txt")
            destination = root / "copy"
            with self.assertRaises(HarnessError) as caught:
                self._copy(magi, destination)
            self.assertIn("swapped.txt", str(caught.exception))
            self.assertFalse((destination / "nested" / "swapped.txt").exists())

    def test_a_directory_link_that_appears_after_the_preflight_is_refused_by_the_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            magi, secret = _environment(root)
            _symlink(secret.parent, magi / "swapped", directory=True)
            destination = root / "copy"
            with self.assertRaises(HarnessError) as caught:
                self._copy(magi, destination)
            self.assertIn("swapped", str(caught.exception))
            self.assertFalse((destination / "swapped" / "secret.txt").exists())

    def test_a_hard_link_to_a_file_outside_is_refused_by_the_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            magi, secret = _environment(root)
            _hard_link(secret, magi / "nested" / "hard.txt")
            destination = root / "copy"
            with self.assertRaises(HarnessError) as caught:
                self._copy(magi, destination)
            self.assertIn("hard.txt", str(caught.exception))
            self.assertFalse((destination / "nested" / "hard.txt").exists())

    def test_a_magi_that_is_itself_a_link_is_refused_by_the_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            magi, secret = _environment(root)
            linked = root / "linked-magi"
            _symlink(secret.parent, linked, directory=True)
            destination = root / "copy"
            with self.assertRaises(HarnessError):
                runs._copy_tree_without_links(linked, destination, runs._skip_top_level_logs(linked))
            self.assertFalse((destination / "secret.txt").exists())

    def test_a_file_swapped_between_the_listing_and_the_open_is_refused(self) -> None:
        # The file read must be the file the listing saw: a swap after the listing changes its identity.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            magi, _ = _environment(root)
            destination = root / "copy"
            real_open = runs._open_listed

            def swapped(path, expected):
                return real_open(path, (expected[0], expected[1] + 1))

            with mock.patch.object(runs, "_open_listed", swapped):
                with self.assertRaises(HarnessError) as caught:
                    self._copy(magi, destination)
            self.assertIn(runs.CHANGED_AFTER_LISTING, str(caught.exception))

    def test_a_tree_past_the_entry_cap_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            magi, _ = _environment(root)
            destination = root / "copy"
            with mock.patch.object(runs, "MAX_COPIED_MAGI_ENTRIES", 2):
                with self.assertRaises(HarnessError) as caught:
                    self._copy(magi, destination)
            self.assertIn("entries", str(caught.exception))

    @unittest.skipIf(os.name == "nt", "POSIX permission bits")
    def test_file_and_directory_modes_are_kept(self) -> None:
        # copytree kept the source modes (copy2); the encrypted store is 0600 in the environment.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            magi, _ = _environment(root)
            os.chmod(magi / "magi.toml", 0o600)
            os.chmod(magi / "nested", 0o700)
            destination = root / "copy"
            self._copy(magi, destination)
            self.assertEqual(0o600, stat.S_IMODE(os.stat(destination / "magi.toml").st_mode))
            self.assertEqual(0o700, stat.S_IMODE(os.stat(destination / "nested").st_mode))

    def test_a_read_only_file_is_copied_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            magi, _ = _environment(root)
            os.chmod(magi / "magi.toml", stat.S_IREAD)
            destination = root / "copy"
            try:
                self._copy(magi, destination)
                self.assertFalse(os.stat(destination / "magi.toml").st_mode & stat.S_IWRITE)
                self.assertEqual(b"# toy config\n", (destination / "magi.toml").read_bytes())
            finally:
                os.chmod(magi / "magi.toml", stat.S_IREAD | stat.S_IWRITE)
                if (destination / "magi.toml").exists():
                    os.chmod(destination / "magi.toml", stat.S_IREAD | stat.S_IWRITE)


class PreflightHardLinkTests(unittest.TestCase):
    """[MAGI §6 MS2 loop 29, F1a1b-caspar] a hard link is refused before the old workspace is deleted."""

    def test_reject_links_refuses_a_hard_linked_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            magi, secret = _environment(root)
            _hard_link(secret, magi / "hard.txt")
            with self.assertRaises(HarnessError) as caught:
                runs._reject_links(magi)
            self.assertIn("hard.txt", str(caught.exception))

    @unittest.skipUnless(_git_available(), "git is required")
    def test_a_hard_link_in_the_environment_never_reaches_the_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            magi, secret = _environment(root)
            _hard_link(secret, magi / "hard.txt")
            workspace = root / "scratch" / runs.REVIEW_WORKSPACE_DIR
            workspace.mkdir(parents=True)
            (workspace / "marker").write_text("x", encoding="utf-8")
            env = types.SimpleNamespace(root=root / "env", magi_dir=magi, scratch_dir=root / "scratch")
            executor = runs.RunExecutor(binary=None, env=env, config=None)
            with mock.patch.object(runs, "REPOSITORY_ROOT", root):
                with self.assertRaises(HarnessError):
                    executor._prepare_review_workspace(runs.DEFINITIONS["R14"])
            self.assertTrue((workspace / "marker").is_file())
            self.assertFalse((workspace / ".magi" / "hard.txt").exists())

    @unittest.skipUnless(_git_available(), "git is required")
    def test_the_workspace_is_built_by_the_link_refusing_copy(self) -> None:
        # Mutation guard: the workspace must come from _copy_tree_without_links, never copytree.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            magi, _ = _environment(root)
            env = types.SimpleNamespace(root=root / "env", magi_dir=magi, scratch_dir=root / "scratch")
            executor = runs.RunExecutor(binary=None, env=env, config=None)
            with mock.patch.object(runs, "REPOSITORY_ROOT", root), \
                    mock.patch.object(runs, "_copy_tree_without_links",
                                      wraps=runs._copy_tree_without_links) as copied, \
                    mock.patch.object(runs.shutil, "copytree", side_effect=AssertionError("copytree")):
                workspace = executor._prepare_review_workspace(runs.DEFINITIONS["R14"])
            copied.assert_called_once()
            self.assertEqual(b"# toy config\n", (workspace / ".magi" / "magi.toml").read_bytes())


if __name__ == "__main__":
    unittest.main()
