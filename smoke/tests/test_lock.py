# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Tests for the single-run lock."""

import pathlib
import subprocess
import sys
import textwrap
import unittest

from smoke.lock import RunLock
from smoke.tests import support


class LockTests(unittest.TestCase):
    """One run at a time, and the lock outlives what it protects."""

    def setUp(self) -> None:
        self.directory = support.scratch_dir(self)
        self.path = self.directory / ".lock"

    def test_acquiring_creates_a_non_empty_file(self) -> None:
        with RunLock(self.path):
            self.assertTrue(self.path.exists())
            self.assertGreater(self.path.stat().st_size, 0)

    def test_the_file_records_the_owning_pid_as_diagnostic_text(self) -> None:
        with RunLock(self.path):
            recorded = self.path.read_text(encoding="utf-8").strip()
        self.assertTrue(recorded.isdigit())

    def test_a_second_holder_in_another_process_is_refused(self) -> None:
        script = textwrap.dedent(
            """
            import sys
            sys.path.insert(0, sys.argv[1])
            from smoke.errors import PreflightError
            from smoke.lock import RunLock
            try:
                with RunLock(sys.argv[2]):
                    sys.exit(0)
            except PreflightError:
                sys.exit(42)
            """
        )
        with RunLock(self.path):
            result = subprocess.run(
                [sys.executable, "-c", script, str(pathlib.Path.cwd()), str(self.path)],
                capture_output=True,
                check=False,
            )
        self.assertEqual(42, result.returncode, result.stderr.decode())

    def test_the_lock_is_released_when_the_holder_exits(self) -> None:
        with RunLock(self.path):
            pass
        with RunLock(self.path):
            pass



    def test_a_lock_file_that_is_not_utf8_still_names_a_holder(self) -> None:
        """The owner is read for a MESSAGE, so it must not become the failure.

        ``read_text`` caught only OSError, and a lock file holding bytes that
        are not UTF-8 -- a foreign writer, a truncated write, a filesystem
        that handed back garbage -- raises UnicodeDecodeError instead. That
        is a ValueError, so it escaped the read and reached the last-resort
        catch: a second run got a traceback where it should have been told
        another run is in progress.
        """
        path = support.scratch_dir(self) / ".lock"
        path.write_bytes(bytes([0xFF, 0xFE]) + b" not utf-8 at all")
        self.assertEqual("unknown", RunLock(path)._recorded_owner())

if __name__ == "__main__":
    unittest.main()
