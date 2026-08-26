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


#: A minimal, valid smoke.toml. The key variable is interpolated so a test
#: can name the one it exported.
_CONFIG_TEXT = (
    '[env]\npassphrase = "correct-horse-battery-staple"\n'
    '[backend]\nkind = "ollama"\n'
    'base_url = "http://localhost:11434/v1"\nkey_env = "%s"\n'
)


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


def _verb(argv: list[str]) -> str:
    """The vault subcommand an invocation carries.

    Read by NAME rather than by position: the workspace flag sits between the
    subcommand group and its verb, so an index would have to move every time
    the argv gains a flag.

    Args:
        argv: The invocation.

    Returns:
        str: The first argument that is not the group, a flag, or a flag's
        value.
    """
    rest = list(argv[1:])
    while rest and rest[0].startswith("-"):
        rest = rest[2:]
    return rest[0] if rest else ""


def _quiet_permissions(case: unittest.TestCase) -> None:
    """Patch out step 4 for a test whose subject is a different step.

    The permission check reads a real file's ACL, which a temporary file in a
    test does not reliably satisfy. Patching it keeps these tests about the
    step they name; the check has its own tests, and its WIRING has
    :class:`PermissionCheckWiringTests`.

    Args:
        case: The test to register the cleanup on.
    """
    patcher = mock.patch("smoke.preflight.check_config_permissions")
    case.addCleanup(patcher.stop)
    patcher.start()


def _resolvable_config(case: unittest.TestCase):
    """Build a configuration double whose backend credential resolves.

    A configuration that names no resolvable credential is cut at step 3, so a
    double meant to reach a later step has to carry one.

    Args:
        case: The test to register the environment cleanup on.

    Returns:
        mock.Mock: The double.
    """
    os.environ["SMOKE_ORDERING_KEY"] = "the-real-credential"
    case.addCleanup(os.environ.pop, "SMOKE_ORDERING_KEY", None)
    _quiet_permissions(case)
    config = mock.Mock(spec=SmokeConfig)
    config.passphrase = "correct horse battery staple"
    config.backend_key_env = "SMOKE_ORDERING_KEY"
    config.path = str(pathlib.Path(tempfile.mkdtemp()) / "smoke.toml")
    return config


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
        config = _resolvable_config(self)
        env = mock.Mock(spec=Environment)
        env.exists.return_value = True
        env.normalize_magi_toml.side_effect = lambda profile: order.append("normalize")
        with mock.patch.object(RunLock, "acquire",
                               side_effect=lambda: order.append("lock")):
            with mock.patch.object(RunLock, "release"):
                Preflight(config, env,
                          mock.Mock(spec=ReleaseBinary),
                          RunLock(directory / ".lock")).run(False, None)
        self.assertEqual(["lock", "normalize"], order)

    def test_a_missing_environment_cuts_before_it_is_normalised(self) -> None:
        """Step 5 before step 7b: normalising an environment that does not
        exist fails with a low-level error instead of the --init-env message
        step 5 already knows how to give.
        """
        directory = pathlib.Path(tempfile.mkdtemp())
        config = _resolvable_config(self)
        env = mock.Mock(spec=Environment)
        env.exists.return_value = False
        with mock.patch.object(RunLock, "acquire"):
            with mock.patch.object(RunLock, "release"):
                with self.assertRaises(PreflightError) as caught:
                    Preflight(config, env,
                              mock.Mock(spec=ReleaseBinary),
                              RunLock(directory / ".lock")).run(False, None)
        self.assertIn("--init-env", str(caught.exception))
        env.normalize_magi_toml.assert_not_called()


class PermissionCheckWiringTests(unittest.TestCase):
    """Step 4 has to RUN, and only a Preflight-level test can say that it does.

    ``_require_config`` read ``getattr(self.config, "path", None)`` and
    ``SmokeConfig`` has no ``path`` attribute, so the default swallowed it and
    the check was skipped on every real run. The existing tests call
    ``check_config_permissions`` directly, so they passed over dead wiring --
    and the doubles are ``mock.Mock(spec=SmokeConfig)``, which cannot grow the
    attribute either. The file holds the vault passphrase in cleartext.
    """

    def test_the_loaded_config_knows_where_it_came_from(self) -> None:
        directory = pathlib.Path(tempfile.mkdtemp())
        path = directory / "smoke.toml"
        path.write_text(_CONFIG_TEXT % "K", encoding="utf-8")
        self.assertEqual(path, pathlib.Path(SmokeConfig.load(path).path))

    def test_the_preflight_checks_the_permissions_of_a_real_config(self) -> None:
        """Driven through a real ``SmokeConfig``, not a double.

        A double can be given a ``path`` by hand, and then the call site fires
        and the test passes whether or not the production object carries one.
        Only the real type discriminates.
        """
        os.environ["SMOKE_WIRING_KEY"] = "the-real-credential"
        self.addCleanup(os.environ.pop, "SMOKE_WIRING_KEY", None)
        directory = pathlib.Path(tempfile.mkdtemp())
        path = directory / "smoke.toml"
        path.write_text(
            _CONFIG_TEXT % "SMOKE_WIRING_KEY", encoding="utf-8")
        env = mock.Mock(spec=Environment)
        env.exists.return_value = True
        with mock.patch.object(RunLock, "acquire"):
            with mock.patch.object(RunLock, "release"):
                with mock.patch("smoke.preflight.check_config_permissions") as check:
                    Preflight(SmokeConfig.load(path), env,
                              mock.Mock(spec=ReleaseBinary),
                              RunLock(directory / ".lock")).run(False, None)
        check.assert_called_once()


class CredentialResolutionTests(unittest.TestCase):
    """Step 3 resolves the backend credential, and it does so BEFORE spending.

    R7 rotates that credential and has to put the real one back, so a variable
    that resolves to nothing is fatal to the run. Discovering it inside R7
    costs every earlier backend run first, which is exactly what happened: four
    completed runs, then ``exit 3`` over a precondition step 3 already owns.
    Failing before spending is the whole reason this cut sits where it does.
    """

    def _config(self, variable: str):
        """Build a configuration double naming *variable* as the key holder.

        Args:
            variable: The environment variable the backend credential lives in.

        Returns:
            mock.Mock: The double.
        """
        _quiet_permissions(self)
        config = mock.Mock(spec=SmokeConfig)
        config.passphrase = "correct horse battery staple"
        config.backend_key_env = variable
        config.path = str(pathlib.Path(tempfile.mkdtemp()) / "smoke.toml")
        return config

    def test_an_unresolvable_credential_cuts_naming_the_variable(self) -> None:
        os.environ.pop("SMOKE_ABSENT_KEY", None)
        directory = pathlib.Path(tempfile.mkdtemp())
        env = mock.Mock(spec=Environment)
        env.exists.return_value = True
        with mock.patch.object(RunLock, "acquire"):
            with mock.patch.object(RunLock, "release"):
                with self.assertRaises(PreflightError) as caught:
                    Preflight(self._config("SMOKE_ABSENT_KEY"), env,
                              mock.Mock(spec=ReleaseBinary),
                              RunLock(directory / ".lock")).run(False, None)
        self.assertIn("SMOKE_ABSENT_KEY", str(caught.exception))

    def test_it_cuts_before_the_environment_is_touched_at_all(self) -> None:
        """Step 3 precedes steps 5, 6 and 7b. A cut that happens after the
        environment has been listed or normalised has already paid for the
        thing it exists to avoid paying for.
        """
        os.environ.pop("SMOKE_ABSENT_KEY", None)
        directory = pathlib.Path(tempfile.mkdtemp())
        env = mock.Mock(spec=Environment)
        env.exists.return_value = True
        binary = mock.Mock(spec=ReleaseBinary)
        with mock.patch.object(RunLock, "acquire"):
            with mock.patch.object(RunLock, "release"):
                with self.assertRaises(PreflightError):
                    Preflight(self._config("SMOKE_ABSENT_KEY"), env, binary,
                              RunLock(directory / ".lock")).run(False, None)
        env.normalize_magi_toml.assert_not_called()
        binary.invoke.assert_not_called()

    def test_a_resolvable_credential_passes_the_cut(self) -> None:
        """A permanent red would look like a working guard, so the double has
        to be able to get through.
        """
        os.environ["SMOKE_PRESENT_KEY"] = "the-real-credential"
        self.addCleanup(os.environ.pop, "SMOKE_PRESENT_KEY", None)
        preflight = Preflight(self._config("SMOKE_PRESENT_KEY"),
                              mock.Mock(spec=Environment),
                              mock.Mock(spec=ReleaseBinary),
                              mock.Mock(spec=RunLock))
        preflight._require_config()


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
            text = listing if _verb(args) == "ls" else ""
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

    def test_every_vault_call_names_the_environment(self) -> None:
        """Without ``-w`` the product walks UP from the harness's own working
        directory, which is the repository root, not ``smoke/env/``.

        Two consequences, and the second is the one that matters. ``vault ls``
        exits non-zero where there is no ``.magi/``, so ``_vault_names``
        answers None and the marker is never detected: step 6, the layer that
        covers a power cut, does not exist. And if any ancestor of the launch
        directory does carry a ``.magi/``, the restore writes the operator's
        real backend credential into that unrelated workspace's vault.

        ``RunExecutor._vault_set`` has always passed ``-w``; this path did not.
        """
        preflight, calls = self._preflight("SMOKE_R7_ROTATION\n")
        preflight._restore_rotation_if_left_over()
        self.assertTrue(calls, "no vault call was made at all")
        for call in calls:
            self.assertIn("-w", call, "vault call without -w: %r" % (call,))

    def test_a_left_over_marker_is_restored_and_removed(self) -> None:
        """Detection alone leaves the environment broken for the next run."""
        preflight, calls = self._preflight("SMOKE_R7_ROTATION\nOPENAI_API_KEY\n")
        preflight._restore_rotation_if_left_over()
        verbs = [_verb(c) for c in calls]
        self.assertIn("set", verbs)
        self.assertIn("rm", verbs)
        self.assertLess(verbs.index("set"), verbs.index("rm"),
                        "the credential is restored before the marker is dropped")

    def test_no_marker_means_no_vault_writes_at_all(self) -> None:
        preflight, calls = self._preflight("OPENAI_API_KEY\n")
        preflight._restore_rotation_if_left_over()
        self.assertEqual(["ls"], [_verb(c) for c in calls])

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
