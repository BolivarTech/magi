# Author: Julian Bolivar
# Version: 0.24.3
# Date: 2026-11-02
"""Unit tests for every harness read of a log directory the product writes.

[MAGI §6 MS2 loop 29, F1a1b-melchior] the persistent environment's log was
read whole into memory with no byte bound, so one huge file a faulty product
wrote exhausted the harness before a scenario could report it. Swept: the
same unbounded, link-following read lived in the decide scenarios'
``_read_logs`` and in the vault scenario's ``_new_log_bytes``.
"""

import os
import pathlib
import tempfile
import unittest
from unittest import mock

from smoke import runs
from smoke.scenarios import decide, vault


def _hard_link(target: pathlib.Path, link: pathlib.Path) -> None:
    """Make *link* a hard link to *target*, or skip where the file system refuses one.

    Raises:
        unittest.SkipTest: When hard links cannot be made here.
    """
    try:
        os.link(target, link)
    except (OSError, NotImplementedError) as exc:
        raise unittest.SkipTest("cannot create a hard link here: %s" % exc) from exc


class ReadLogFilesBoundTests(unittest.TestCase):
    """``runs.read_log_files`` holds at most one bounded file at a time."""

    def _logs(self, tmp: str) -> pathlib.Path:
        directory = pathlib.Path(tmp) / "logs"
        directory.mkdir()
        (directory / "a.log").write_bytes(b"12")
        (directory / "b.log").write_bytes(b"123456")
        (directory / "c.log").write_bytes(b"34")
        return directory

    def test_a_file_over_the_per_file_cap_is_named_and_not_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = self._logs(tmp)
            unreadable = []
            with mock.patch.object(runs, "MAX_ENVIRONMENT_LOG_FILE_BYTES", 4):
                data = list(runs.read_log_files(directory, unreadable))
            self.assertEqual([b"12", b"34"], data)
            self.assertEqual(["b.log (%s)" % (runs.OVER_FILE_CAP % 4)], unreadable)

    def test_files_past_the_total_cap_are_named_and_not_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = self._logs(tmp)
            unreadable = []
            with mock.patch.object(runs, "MAX_ENVIRONMENT_LOG_TOTAL_BYTES", 9):
                data = list(runs.read_log_files(directory, unreadable))
            self.assertEqual([b"12", b"123456"], data)
            self.assertEqual(["c.log (%s)" % (runs.OVER_TOTAL_CAP % 9)], unreadable)

    def test_the_read_never_asks_for_more_than_the_cap_allows(self) -> None:
        # Mutation guard for the bounded read: an unbounded handle.read() asks for everything at once.
        with tempfile.TemporaryDirectory() as tmp:
            directory = self._logs(tmp)
            with mock.patch.object(runs, "_read_at_most", wraps=runs._read_at_most) as bounded:
                list(runs.read_log_files(directory, []))
            self.assertEqual(3, bounded.call_count)

    def test_offsets_skip_the_bytes_already_seen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = self._logs(tmp)
            offsets = {directory / "b.log": 4, directory / "c.log": 9}
            data = list(runs.read_log_files(directory, [], offsets))
            self.assertEqual([b"12", b"56", b""], data)

    def test_a_link_swapped_above_the_directory_during_the_read_drops_what_was_read(self) -> None:
        # [MAGI F1a1] Windows lists and opens by path, so a link swapped onto a component above the log
        # directory during the read had the bytes behind it reach a scenario: unlike the review-log capture,
        # the live scan never tested the chain again after reading. The chain test flips after the first open.
        with tempfile.TemporaryDirectory() as tmp:
            directory = self._logs(tmp)
            unreadable, swapped = [], []
            real_open, real_ancestor = runs._open_listed, runs._linked_ancestor

            def opening(path, expected):
                swapped.append(path)
                return real_open(path, expected)

            def ancestor(path):
                return pathlib.Path(tmp) if swapped else real_ancestor(path)

            with mock.patch.object(runs, "_open_listed", opening), \
                    mock.patch.object(runs, "_linked_ancestor", ancestor):
                data = list(runs.read_log_files(directory, unreadable))
            self.assertEqual([], data)
            self.assertTrue(any(runs.LINK_NOT_FOLLOWED in entry for entry in unreadable), unreadable)

    def test_the_caps_hold_every_file_the_environment_writes_in_a_day(self) -> None:
        # The bounds must never refuse a healthy environment: a day file is a few hundred KB.
        self.assertGreaterEqual(runs.MAX_ENVIRONMENT_LOG_FILE_BYTES, runs.MAX_REVIEW_LOG_FILE_BYTES)
        self.assertGreaterEqual(runs.MAX_ENVIRONMENT_LOG_TOTAL_BYTES, runs.MAX_ENVIRONMENT_LOG_FILE_BYTES)


class AnchoredOpenDescriptorTests(unittest.TestCase):
    """[MAGI F1a1b] ``_open_anchored`` never leaks a descriptor when closing the one above fails."""

    def test_a_failed_close_of_the_parent_closes_the_child_and_raises(self) -> None:
        import types
        opened, closed = [], []

        def fake_open(path, flags, dir_fd=None):
            opened.append(100 + len(opened))
            return opened[-1]

        def fake_close(descriptor):
            closed.append(descriptor)
            if descriptor == 100 and closed.count(100) == 1:
                raise OSError("close failed")

        fake_os = types.SimpleNamespace(open=fake_open, close=fake_close, O_RDONLY=0, O_DIRECTORY=0)
        with mock.patch.object(runs, "_chain_from_repository",
                               return_value=(pathlib.Path("base"), pathlib.PurePosixPath("a/b/c"))), \
                mock.patch.object(runs, "os", fake_os):
            with self.assertRaises(OSError):
                runs._open_anchored("base/a/b/c", 0)
        # The base (100) and the first child (101) were opened; each is closed exactly once.
        self.assertEqual([100, 101], opened)
        self.assertEqual([100, 101], closed)


class DecideLogReadTests(unittest.TestCase):
    """The decide and keys scenarios read a workspace log without following a link."""

    def test_a_hard_link_under_the_log_directory_is_never_read_and_is_named(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            directory = root.joinpath(*runs.LOG_DIR_PARTS)
            directory.mkdir(parents=True)
            (directory / "own.log").write_bytes(b"own line")
            outside = root / "outside.txt"
            outside.write_bytes(b"bytes outside the workspace")
            _hard_link(outside, directory / "planted.log")
            log = decide._read_logs(root)
        self.assertIn(b"own line", log)
        self.assertNotIn(b"bytes outside the workspace", log)
        self.assertIn(b"planted.log", log)

    def test_an_absent_directory_is_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(decide._read_logs(pathlib.Path(tmp)))


class VaultLogReadTests(unittest.TestCase):
    """The vault scenario reads only what was appended, without following a link."""

    def test_a_hard_link_under_the_log_directory_is_never_read_and_is_named(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            directory = root / "logs"
            directory.mkdir()
            own = directory / "own.log"
            own.write_bytes(b"before|after")
            outside = root / "outside.txt"
            outside.write_bytes(b"bytes outside the environment")
            _hard_link(outside, directory / "planted.log")
            with mock.patch.object(vault.logs, "log_directory", return_value=directory):
                appended = vault._new_log_bytes({own: len(b"before|")})
        self.assertIn(b"after", appended)
        self.assertNotIn(b"before", appended)
        self.assertNotIn(b"bytes outside the environment", appended)
        self.assertIn(b"planted.log", appended)


if __name__ == "__main__":
    unittest.main()
