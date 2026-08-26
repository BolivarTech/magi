# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Tests for the preflight's ordering and its hard cuts."""

import contextlib
import os
import pathlib
import socket
import stat
import sys
import threading
import unittest
from unittest import mock

from smoke.binary import ReleaseBinary
from smoke.config import SmokeConfig
from smoke.env import Environment
from smoke.errors import PreflightError
from smoke.lock import RunLock
from smoke import preflight as preflight_module
from smoke.preflight import BackendStatus, Preflight, check_config_permissions
from smoke.product import ProductOutput
from smoke.tests import support


#: How long a test waits for its listener thread to notice the socket closed
#: under it. Generous on purpose: the deadline is a failure bound, not a
#: measurement, and the thread is a daemon so overrunning it costs nothing.
THREAD_JOIN_TIMEOUT_S = 30

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
        self.path = support.scratch_dir(self) / "smoke.toml"
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
        path = support.scratch_dir(self) / "smoke.toml"
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


def _quiet_binary():
    """A binary double whose ``vault ls`` answers an empty, healthy listing.

    A bare ``mock.Mock(spec=ReleaseBinary)`` returns a Mock from ``invoke``,
    so ``exit_code`` is a Mock too. That stayed invisible while step 6
    answered "cannot tell" and returned. Now that an unlistable vault CUTS, a
    double that does not say what the vault holds makes every test merely
    passing through step 6 fail on the formatting of a Mock -- so it says,
    which is what the real one always does.

    Returns:
        mock.Mock: The double.
    """
    binary = mock.Mock(spec=ReleaseBinary)
    binary.invoke.return_value = ProductOutput(
        stdout=b"(vault empty)", stderr=b"", exit_code=0,
        command=["magi-rs", "vault", "ls"])
    return binary


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
    config.path = str(support.scratch_dir(case) / "smoke.toml")
    config.backend_base_url = "http://127.0.0.1:1/v1"
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
        directory = support.scratch_dir(self)
        config = _resolvable_config(self)
        env = mock.Mock(spec=Environment)
        env.exists.return_value = True
        env.declared_base_url.return_value = None
        env.normalize_magi_toml.side_effect = lambda profile: order.append("normalize")
        with mock.patch.object(RunLock, "acquire",
                               side_effect=lambda: order.append("lock")):
            with mock.patch.object(RunLock, "release"):
                Preflight(config, env,
                          _quiet_binary(),
                          RunLock(directory / ".lock")).run(False, None)
        self.assertEqual(["lock", "normalize"], order)

    def test_a_missing_environment_cuts_before_it_is_normalised(self) -> None:
        """Step 5 before step 7b: normalising an environment that does not
        exist fails with a low-level error instead of the --init-env message
        step 5 already knows how to give.
        """
        directory = support.scratch_dir(self)
        config = _resolvable_config(self)
        env = mock.Mock(spec=Environment)
        env.exists.return_value = False
        with mock.patch.object(RunLock, "acquire"):
            with mock.patch.object(RunLock, "release"):
                with self.assertRaises(PreflightError) as caught:
                    Preflight(config, env,
                              _quiet_binary(),
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
        directory = support.scratch_dir(self)
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
        directory = support.scratch_dir(self)
        path = directory / "smoke.toml"
        path.write_text(
            _CONFIG_TEXT % "SMOKE_WIRING_KEY", encoding="utf-8")
        env = mock.Mock(spec=Environment)
        env.exists.return_value = True
        env.declared_base_url.return_value = None
        env.declared_models.return_value = set()
        # The backend probe is patched out: the subject here is whether step 4
        # RUNS, and a unit suite that reaches a real daemon on localhost gives a
        # different answer depending on whose machine it is -- and hangs for the
        # probe's whole timeout on one that blackholes the port.
        patched = BackendStatus(reachable=False, cause="patched")
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(RunLock, "acquire"))
            stack.enter_context(mock.patch.object(RunLock, "release"))
            stack.enter_context(mock.patch.object(Preflight, "_probe_backend",
                                                  return_value=patched))
            check = stack.enter_context(
                mock.patch("smoke.preflight.check_config_permissions"))
            Preflight(SmokeConfig.load(path), env,
                      _quiet_binary(),
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
        config.path = str(support.scratch_dir(self) / "smoke.toml")
        config.backend_base_url = "http://127.0.0.1:1/v1"
        return config

    def test_an_unresolvable_credential_cuts_naming_the_variable(self) -> None:
        os.environ.pop("SMOKE_ABSENT_KEY", None)
        directory = support.scratch_dir(self)
        env = mock.Mock(spec=Environment)
        env.exists.return_value = True
        env.declared_base_url.return_value = None
        with mock.patch.object(RunLock, "acquire"):
            with mock.patch.object(RunLock, "release"):
                with self.assertRaises(PreflightError) as caught:
                    Preflight(self._config("SMOKE_ABSENT_KEY"), env,
                              _quiet_binary(),
                              RunLock(directory / ".lock")).run(False, None)
        self.assertIn("SMOKE_ABSENT_KEY", str(caught.exception))

    def test_it_cuts_before_the_environment_is_touched_at_all(self) -> None:
        """Step 3 precedes steps 5, 6 and 7b. A cut that happens after the
        environment has been listed or normalised has already paid for the
        thing it exists to avoid paying for.
        """
        os.environ.pop("SMOKE_ABSENT_KEY", None)
        directory = support.scratch_dir(self)
        env = mock.Mock(spec=Environment)
        env.exists.return_value = True
        env.declared_base_url.return_value = None
        binary = _quiet_binary()
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
                              _quiet_binary(),
                              mock.Mock(spec=RunLock))
        preflight._require_config()


class EndpointAgreementTests(unittest.TestCase):
    """The endpoint the preflight PROBES is the one the runs USE.

    ``[backend].base_url`` is read by exactly one place, the reachability
    probe, while every run reaches whatever the environment's ``magi.toml``
    declares at its root. The two are separate settings and nothing compared
    them: this repository ran for a whole session with the probe pointed at a
    machine on the LAN and the runs going to localhost, and it stayed
    invisible because both daemons served the same tags. A preflight that
    certifies "the backend answers" about a host nothing talks to is a
    guardian aimed the wrong way.
    """

    def _preflight(self, declared, probed):
        """Build a preflight whose two endpoints are *declared* and *probed*.

        Args:
            declared: What the environment's magi.toml says at its root.
            probed: What smoke.toml names for the probe.

        Returns:
            Preflight: Ready to run.
        """
        os.environ["SMOKE_AGREE_KEY"] = "the-real-credential"
        self.addCleanup(os.environ.pop, "SMOKE_AGREE_KEY", None)
        _quiet_permissions(self)
        config = mock.Mock(spec=SmokeConfig)
        config.passphrase = "correct horse battery staple"
        config.backend_key_env = "SMOKE_AGREE_KEY"
        config.path = str(support.scratch_dir(self) / "smoke.toml")
        config.backend_base_url = probed
        env = mock.Mock(spec=Environment)
        env.exists.return_value = True
        env.declared_base_url.return_value = declared
        return Preflight(config, env, _quiet_binary(),
                         RunLock(support.scratch_dir(self) / ".lock"))

    def test_two_different_endpoints_cut(self) -> None:
        preflight = self._preflight("http://localhost:11434/v1",
                                    "http://192.168.0.30:11434/v1")
        with self.assertRaises(PreflightError) as caught:
            preflight._require_one_endpoint()
        message = str(caught.exception)
        self.assertIn("localhost", message)
        self.assertIn("192.168.0.30", message)

    def test_the_same_endpoint_passes(self) -> None:
        preflight = self._preflight("http://localhost:11434/v1",
                                    "http://localhost:11434/v1")
        preflight._require_one_endpoint()

    def test_an_unreadable_environment_config_does_not_cut(self) -> None:
        """Not measurable is not a disagreement.

        Step 5 already refuses an environment that is not there, and a config
        the harness cannot parse is step 7b's problem, not this check's.
        """
        preflight = self._preflight(None, "http://localhost:11434/v1")
        preflight._require_one_endpoint()


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

    def test_a_vault_that_cannot_be_listed_cuts(self) -> None:
        """Spec section 5.2: it cuts with exit 2, naming the cause.

        "No se arranca sin saber si hay una rotacion pendiente." Returning
        quietly is the worst of the three options: a run killed mid-rotation
        leaves a sentinel credential in the environment, the next run cannot
        tell, every backend invocation then authenticates with the sentinel
        and dies of an opaque auth error, and S16 renders a verdict over a
        half-rotated database. Nothing in the report says the recovery was
        skipped -- the announcement only covers the branch that succeeded.
        """
        preflight, _ = self._preflight("")
        preflight.binary.invoke.side_effect = OSError("no such binary")
        with self.assertRaises(PreflightError) as caught:
            preflight._restore_rotation_if_left_over()
        self.assertIn("no such binary", str(caught.exception))

    def test_left_over_placeholders_are_swept(self) -> None:
        """The crash path the placeholder removal never covered.

        R6 takes its four entries back out in a ``finally``, which handles a
        run that fails and one that times out. It does not handle the run that
        is KILLED: there is no finally to reach, and unlike the rotation
        marker nothing looked for them afterwards. The removal now warns
        rather than raising and says the next preflight sweeps them, so the
        sweep has to exist.
        """
        preflight, calls = self._preflight(
            "OPENAI_API_KEY\nBASE_URL_USER\nMAGI_BASE_URL_PASSWORD\n")
        preflight._restore_rotation_if_left_over()
        removed = [call[call.index("rm") + 1]
                   for call in calls if _verb(call) == "rm"]
        self.assertEqual(["BASE_URL_USER", "MAGI_BASE_URL_PASSWORD"],
                         sorted(removed))

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


class ModelExistenceTests(unittest.TestCase):
    """Step 8: a model the environment names has to exist on the backend.

    This is failure #5 -- a run that reaches a healthy backend and asks it for
    something it does not have. The spec makes it a hard cut, and it belongs
    in the preflight rather than in a scenario: every trio run would otherwise
    spend its whole ceiling discovering it.

    It only applies where the backend can be asked. ``openai-compat`` and
    ``anthropic`` publish no tag listing, so there the check degrades to not
    performed rather than to a guess.
    """

    def test_a_missing_model_cuts_naming_it(self) -> None:
        with self.assertRaises(PreflightError) as caught:
            preflight_module.require_declared_models(
                {"present:cloud", "absent:cloud"}, {"present:cloud"})
        message = str(caught.exception)
        self.assertIn("absent:cloud", message)
        self.assertNotIn("present:cloud", message)

    def test_every_model_present_passes(self) -> None:
        preflight_module.require_declared_models(
            {"a:cloud", "b:cloud"}, {"a:cloud", "b:cloud", "spare:cloud"})

    def test_an_untagged_name_matches_the_backend_s_latest(self) -> None:
        """Ollama always answers fully tagged, and an operator does not type it.

        ``/api/tags`` returns ``nomic-embed-text-v2-moe:latest``; a config that
        says ``nomic-embed-text-v2-moe`` -- which is what ``ollama run`` takes
        and what somebody filling in the cheap profile writes -- would be
        reported missing and the preflight would refuse to run, telling them to
        pull a model they already have.
        """
        preflight_module.require_declared_models({"foo"}, {"foo:latest"})

    def test_an_explicit_tag_still_has_to_match(self) -> None:
        with self.assertRaises(PreflightError):
            preflight_module.require_declared_models({"foo:v2"}, {"foo:latest"})

    def test_a_backend_that_lists_nothing_is_not_measurable(self) -> None:
        """An empty listing is "could not be asked", not "nothing exists".

        Cutting there would refuse every backend without a tag endpoint, which
        is every backend but Ollama.
        """
        preflight_module.require_declared_models({"a:cloud"}, None)

    def test_declaring_no_models_passes(self) -> None:
        preflight_module.require_declared_models(set(), {"a:cloud"})


class BackendStatusTests(unittest.TestCase):
    """A backend that does not answer is reported, never guessed at."""

    def test_an_unreachable_backend_carries_its_cause(self) -> None:
        status = BackendStatus(reachable=False, cause="connection refused")
        self.assertFalse(status.reachable)
        self.assertIn("refused", status.cause)


class BackendSpeakingGarbageTests(unittest.TestCase):
    """A daemon that answers something other than HTTP is not a harness bug.

    ``http.client`` raises ``BadStatusLine`` and ``IncompleteRead`` out of
    ``getresponse()``, and neither is an ``OSError`` -- ``urlopen``'s handler
    does not wrap what the response raises. An except tuple that names only
    ``URLError``/``OSError``/``ValueError`` therefore lets it escape the
    preflight, escape main's two typed handlers, and land on the last-resort
    catch, which exits 3. Exit 3 says the HARNESS failed; a misbehaving
    backend is D-17's "could not be reached", which is exit 1 territory at
    worst and a degraded status at best.
    """

    def _garbage_endpoint(self) -> str:
        """Stand up a socket that answers with something that is not HTTP.

        Returns:
            str: The base URL of a server that will accept a connection and
            reply with a status line no HTTP client can parse.
        """
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(8)
        self.addCleanup(listener.close)

        def serve() -> None:
            while True:
                try:
                    connection, _ = listener.accept()
                except OSError:
                    return
                with connection:
                    try:
                        connection.recv(4096)
                        connection.sendall(b"GARBAGE NOT HTTP\r\n\r\n")
                    except OSError:
                        pass

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        # Closed FIRST, joined second: the thread is blocked in accept() and
        # only the close wakes it. Daemon so it can never hang the suite, and
        # joined anyway so two of these do not accumulate.
        self.addCleanup(thread.join, THREAD_JOIN_TIMEOUT_S)
        return "http://127.0.0.1:%d/v1" % listener.getsockname()[1]

    def _preflight(self, url: str) -> Preflight:
        """Build a preflight pointed at *url*.

        Args:
            url: The endpoint both the probe and the model listing will use.

        Returns:
            Preflight: Ready to have either step called on it.
        """
        config = mock.Mock(spec=SmokeConfig)
        config.backend_base_url = url
        config.backend_kind = "ollama"
        return Preflight(config, mock.Mock(spec=Environment),
                         _quiet_binary(),
                         mock.Mock(spec=RunLock))

    def test_the_reachability_probe_degrades_rather_than_raises(self) -> None:
        """The probe, which runs FIRST and decides whether step 8 runs.

        The guard on the model listing was added for this failure mode and
        cannot fire for it: a daemon answering garbage dies here, several
        statements earlier.
        """
        preflight = self._preflight(self._garbage_endpoint())
        status = preflight._probe_backend()
        self.assertFalse(status.reachable)
        self.assertTrue(status.cause)

    def test_the_model_listing_degrades_rather_than_raises(self) -> None:
        preflight = self._preflight(self._garbage_endpoint())
        self.assertIsNone(preflight._available_models())


if __name__ == "__main__":
    unittest.main()
