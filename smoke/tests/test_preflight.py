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
from smoke.product import ProductOutput


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


class RotationRestoreTests(unittest.TestCase):
    """Step 6 must RESTORE, not narrate.

    The first version of this step printed "restoring the backend credential"
    and then did nothing: it never re-set the value and never removed the
    marker, so every subsequent run found the marker again and announced the
    same restore. A message claiming work that did not happen is worse than
    silence -- it reads as evidence forever.
    """

    def _preflight(self, listing, credential="the-real-credential"):
        """Build a preflight whose vault answers *listing*.

        Args:
            listing: What ``vault ls`` prints.
            credential: What the environment holds for the backend key.

        Returns:
            tuple: The preflight and the list its binary records calls into.
        """
        calls = []
        binary = mock.create_autospec(ReleaseBinary, instance=True)

        def answer(args, stdin=None, env=None, timeout=None, cwd=None):
            calls.append(list(args))
            text = listing if args[:2] == ["vault", "ls"] else ""
            return ProductOutput(stdout=text.encode("utf-8"), stderr=b"",
                                 exit_code=0, command=["magi-rs"] + list(args))

        binary.invoke.side_effect = answer
        config = mock.Mock(spec=SmokeConfig)
        config.passphrase = "correct horse battery staple"
        config.backend_key_env = "SMOKE_TEST_KEY"
        env = mock.Mock(spec=Environment)
        env.exists.return_value = True
        os.environ["SMOKE_TEST_KEY"] = credential
        self.addCleanup(os.environ.pop, "SMOKE_TEST_KEY", None)
        return Preflight(config, env, binary, mock.Mock(spec=RunLock)), calls

    def test_a_left_over_marker_is_restored_and_removed(self) -> None:
        """Detection alone leaves the environment broken for the next run."""
        preflight, calls = self._preflight("SMOKE_R7_ROTATION\nOPENAI_API_KEY\n")
        preflight._restore_rotation_if_left_over()
        subcommands = [c[:2] for c in calls]
        self.assertIn(["vault", "set"], subcommands)
        self.assertIn(["vault", "rm"], subcommands)
        self.assertLess(subcommands.index(["vault", "set"]),
                        subcommands.index(["vault", "rm"]),
                        "the credential is restored before the marker is dropped")

    def test_no_marker_means_no_vault_writes_at_all(self) -> None:
        preflight, calls = self._preflight("OPENAI_API_KEY\n")
        preflight._restore_rotation_if_left_over()
        self.assertEqual([["vault", "ls"]], [c[:2] for c in calls])

    def test_the_vault_listing_uses_the_binary_s_real_signature(self) -> None:
        """The first version passed ``timeout_s=``; the binary takes ``timeout``.

        The resulting TypeError was swallowed by a bare ``except Exception``,
        so the listing silently answered "cannot tell" on every run and the
        marker was never detected at all. An autospec double refuses the wrong
        keyword, which is what makes this checkable.
        """
        preflight, _ = self._preflight("OPENAI_API_KEY\n")
        self.assertIsNotNone(preflight._vault_names())

    def test_a_restore_with_no_credential_to_restore_cuts(self) -> None:
        """Cutting is the honest answer; printing a restore is not."""
        preflight, _ = self._preflight("SMOKE_R7_ROTATION\n", credential="")
        os.environ.pop("SMOKE_TEST_KEY", None)
        with self.assertRaises(PreflightError):
            preflight._restore_rotation_if_left_over()


class BackendStatusTests(unittest.TestCase):
    """A backend that does not answer is reported, never guessed at."""

    def test_an_unreachable_backend_carries_its_cause(self) -> None:
        status = BackendStatus(reachable=False, cause="connection refused")
        self.assertFalse(status.reachable)
        self.assertIn("refused", status.cause)


if __name__ == "__main__":
    unittest.main()
