# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Tests for the preflight's ordering and its hard cuts."""

import os
import pathlib
import stat
import sys
import tempfile
import unittest
from unittest import mock

from smoke.binary import ReleaseBinary
from smoke.config import SmokeConfig
from smoke.env import Environment
from smoke.errors import PreflightError
from smoke.lock import RunLock
from smoke.preflight import BackendStatus, Preflight, check_config_permissions


@unittest.skipIf(sys.platform == "win32", "POSIX permission bits")
class PosixPermissionTests(unittest.TestCase):
    """smoke.toml holds the passphrase in the clear, so others cannot read it."""

    def setUp(self) -> None:
        self.path = pathlib.Path(tempfile.mkdtemp()) / "smoke.toml"
        self.path.write_text("[env]\n", encoding="utf-8")

    def test_owner_only_is_accepted(self) -> None:
        os.chmod(self.path, stat.S_IRUSR | stat.S_IWUSR)
        check_config_permissions(self.path)

    def test_group_readable_is_refused(self) -> None:
        os.chmod(self.path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP)
        with self.assertRaises(PreflightError):
            check_config_permissions(self.path)

    def test_world_readable_is_refused_naming_the_fix(self) -> None:
        os.chmod(self.path, 0o644)
        with self.assertRaises(PreflightError) as caught:
            check_config_permissions(self.path)
        self.assertIn("chmod", str(caught.exception))


@unittest.skipUnless(sys.platform == "win32", "Windows ACL")
class WindowsPermissionTests(unittest.TestCase):
    """The ACL is read as SDDL, never as the localised human listing."""

    def test_the_check_reads_sddl_not_translated_text(self) -> None:
        source = pathlib.Path(__file__).resolve().parent.parent / "preflight.py"
        text = source.read_text(encoding="utf-8")
        self.assertIn("/save", text)
        for translated in ("Everyone", "Todos", "Users", "Usuarios"):
            self.assertNotIn('"%s"' % translated, text)

    def test_an_unreadable_acl_is_not_measurable_rather_than_passing(self) -> None:
        """The third option that shows up on its own is checking something
        easier and calling it green. If the SDDL cannot be obtained, the
        harness must say so: BackendStatus-style 'not measured', never a pass
        for having failed to look.
        """
        path = pathlib.Path(tempfile.mkdtemp()) / "smoke.toml"
        path.write_text("[env]\n", encoding="utf-8")
        with mock.patch("smoke.preflight._read_sddl", return_value=None):
            with self.assertRaises(PreflightError) as caught:
                check_config_permissions(path)
        self.assertIn("could not be read", str(caught.exception))


class OrderingTests(unittest.TestCase):
    """The lock precedes every mutation of the environment."""

    def test_the_lock_is_taken_before_the_environment_is_touched(self) -> None:
        """Round 2's CRITICAL: normalising outside the lock lets a concurrent
        run overwrite magi.toml between the write and its use, so a certifying
        run can end up on the cheap models under a certificate claiming the
        defaults. Swap steps 2 and 7b and this goes red.
        """
        order: list[str] = []
        directory = pathlib.Path(tempfile.mkdtemp())
        env = mock.Mock(spec=Environment)
        env.exists.return_value = True
        env.normalize_magi_toml.side_effect = lambda profile: order.append("normalize")
        with mock.patch.object(RunLock, "acquire",
                               side_effect=lambda: order.append("lock")):
            with mock.patch.object(RunLock, "release"):
                Preflight(mock.Mock(spec=SmokeConfig), env,
                          mock.Mock(spec=ReleaseBinary),
                          RunLock(directory / ".lock")).run(False, None)
        self.assertEqual(["lock", "normalize"], order)

    def test_a_missing_environment_cuts_before_it_is_normalised(self) -> None:
        """Step 5 before step 7b: normalising an environment that does not
        exist fails with a low-level error instead of the --init-env message
        step 5 already knows how to give.
        """
        directory = pathlib.Path(tempfile.mkdtemp())
        env = mock.Mock(spec=Environment)
        env.exists.return_value = False
        with mock.patch.object(RunLock, "acquire"):
            with mock.patch.object(RunLock, "release"):
                with self.assertRaises(PreflightError) as caught:
                    Preflight(mock.Mock(spec=SmokeConfig), env,
                              mock.Mock(spec=ReleaseBinary),
                              RunLock(directory / ".lock")).run(False, None)
        self.assertIn("--init-env", str(caught.exception))
        env.normalize_magi_toml.assert_not_called()


class BackendStatusTests(unittest.TestCase):
    """A backend that does not answer is reported, never guessed at."""

    def test_an_unreachable_backend_carries_its_cause(self) -> None:
        status = BackendStatus(reachable=False, cause="connection refused")
        self.assertFalse(status.reachable)
        self.assertIn("refused", status.cause)


if __name__ == "__main__":
    unittest.main()
