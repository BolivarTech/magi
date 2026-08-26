# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Tests for the one door between a scenario and the product."""

import contextlib
import io
import pathlib
import unittest
import urllib.error
import urllib.parse
import urllib.request
from unittest import mock

from smoke import runs
from smoke.binary import ReleaseBinary
from smoke.config import ROTATION_MARKER, SmokeConfig
from smoke.env import Environment
from smoke.errors import HarnessError, ProductOutputError, TimedOut
from smoke.product import ProductOutput
from smoke.registry import Registry, ScenarioEntry
from smoke.secrets import PlantedSecret
from smoke.tests import support


def _http_post(url: str, credential: bytes) -> tuple[int, bytes]:
    """Post to *url* carrying *credential* as a bearer token.

    Args:
        url: Where to post.
        credential: What to put in the Authorization header.

    Returns:
        tuple[int, bytes]: The status code and the body.

    Raises:
        OSError: If nothing is listening.
    """
    request = urllib.request.Request(
        url.rstrip("/") + "/chat/completions", data=b"{}",
        headers={"Authorization": b"Bearer " + credential,
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=5) as answer:
            return answer.status, answer.read()
    except urllib.error.HTTPError as refused:
        return refused.code, refused.read()


def _config_double():
    """A configuration double that declares what :mod:`smoke.runs` reads.

    Archiving scrubs the passphrase, so a double without one is
    under-specified rather than minimal -- and it fails as an
    ``AttributeError`` from inside the archive, which is the loud version and
    the one this project prefers to a swallowed default.

    Returns:
        mock.Mock: The double.
    """
    config = mock.Mock(spec=SmokeConfig)
    config.passphrase = support.FAKE_PASSPHRASE
    return config


def _query_call(binary):
    """The invocation that ran the product, not the vault bookkeeping.

    The authenticated backend plants entries before the run and removes them
    after, so the LAST recorded call is a ``vault rm``, not the run.

    Args:
        binary: The fake binary.

    Returns:
        support.Call: The recorded query invocation.
    """
    for call in reversed(binary.calls):
        if list(call.args[:1]) != ["vault"]:
            return call
    raise AssertionError("no non-vault invocation was recorded")


class ConfigureGuardTests(unittest.TestCase):
    """Module state is write-once at startup, and that is checkable."""

    def setUp(self) -> None:
        runs.reset_for_test()

    def tearDown(self) -> None:
        runs.reset_for_test()

    def test_invoking_before_configuring_is_a_harness_error(self) -> None:
        """The guard is what turns "call configure first" from a rule someone
        can forget into a failure that names itself. Without it the first
        symptom is an AttributeError deep inside a scenario.
        """
        with self.assertRaises(HarnessError) as caught:
            runs.invoke(["--version"], timeout_s=5, label="probe")
        self.assertIn("configure", str(caught.exception))

    def test_a_configured_invoke_reaches_the_binary(self) -> None:
        binary = mock.Mock(spec=ReleaseBinary)
        binary.invoke.return_value = ProductOutput(
            stdout=b"out", stderr=b"", exit_code=0, command=["magi-rs", "--version"]
        )
        env = mock.Mock(spec=Environment)
        env.runs_dir = support.scratch_dir(self)
        runs.configure(binary, env, _config_double())
        output = runs.invoke(["--version"], timeout_s=5, label="probe")
        self.assertEqual(0, output.exit_code)
        binary.invoke.assert_called_once()


class ArchiveTests(unittest.TestCase):
    """The returned capture is raw; only what reaches disk is scrubbed."""

    def setUp(self) -> None:
        runs.reset_for_test()
        self.root = support.scratch_dir(self)
        self.env = mock.Mock(spec=Environment)
        self.env.runs_dir = self.root
        self.secret = PlantedSecret("s3cr3t-value", "backend credential")
        self.binary = mock.Mock(spec=ReleaseBinary)
        self.binary.invoke.return_value = ProductOutput(
            stdout=b"leaked s3cr3t-value here",
            stderr=b"",
            exit_code=0,
            command=["magi-rs", "query"],
        )
        # A configuration double that does not declare a passphrase is an
        # under-specified double: archiving scrubs it, and the real config
        # always has one.
        config = mock.Mock(spec=SmokeConfig)
        config.passphrase = support.FAKE_PASSPHRASE
        runs.configure(self.binary, self.env, config)

    def tearDown(self) -> None:
        runs.reset_for_test()

    def test_the_returned_output_keeps_the_secret_at_full_fidelity(self) -> None:
        """Scrubbing before asserting would turn S10 and S3 into empty greens.
        The two moments are distinct and the order is half the fix.
        """
        output = runs.invoke(["query"], timeout_s=5, label="r6",
                             planted=(self.secret,))
        self.assertIn(b"s3cr3t-value", output.raw())

    def test_the_archived_copy_is_scrubbed(self) -> None:
        runs.invoke(["query"], timeout_s=5, label="r6", planted=(self.secret,))
        written = b"".join(p.read_bytes() for p in self.root.rglob("*")
                           if p.is_file())
        self.assertNotIn(b"s3cr3t-value", written)
        self.assertIn(b"backend credential", written)

    def test_a_secret_the_caller_did_not_declare_is_not_removed(self) -> None:
        """``planted`` is what the scrubber knows about. A caller that plants a
        secret and does not declare it writes it to disk in clear -- so the
        parameter is the contract, and this test says what forgetting it costs.
        """
        runs.invoke(["query"], timeout_s=5, label="r6")
        written = b"".join(p.read_bytes() for p in self.root.rglob("*")
                           if p.is_file())
        self.assertIn(b"s3cr3t-value", written)


class InvocationOptionsTests(unittest.TestCase):
    """A scenario cannot chdir or export, so the door has to carry both."""

    def setUp(self) -> None:
        runs.reset_for_test()
        self.root = support.scratch_dir(self)
        self.env = mock.Mock(spec=Environment)
        self.env.runs_dir = self.root
        self.binary = mock.Mock(spec=ReleaseBinary)
        self.binary.invoke.return_value = ProductOutput(
            stdout=b"", stderr=b"", exit_code=0, command=["magi-rs", "init"]
        )
        runs.configure(self.binary, self.env, _config_double())

    def tearDown(self) -> None:
        runs.reset_for_test()

    def test_the_working_directory_reaches_the_binary(self) -> None:
        """S2 and S14 seed a workspace by running ``init`` with the target as
        the process's cwd, precisely so the precondition never uses the
        ``-w`` flag S14 exists to test. A door that dropped the directory would
        scaffold in the harness's own working directory instead.
        """
        runs.invoke(["init"], timeout_s=5, label="seed", cwd=self.root)
        self.assertEqual(self.root, self.binary.invoke.call_args.kwargs["cwd"])

    def test_the_environment_overlay_reaches_the_binary(self) -> None:
        runs.invoke(["vault", "ls"], timeout_s=5, label="probe",
                    env={"MAGI_PASSPHRASE": "opensesame1234"})
        self.assertEqual({"MAGI_PASSPHRASE": "opensesame1234"},
                         self.binary.invoke.call_args.kwargs["env"])

    def test_an_attempt_reports_a_failure_instead_of_raising(self) -> None:
        """A scenario declares several assertions and the runner turns a raised
        ProductOutputError into ONE finding, so an invocation that never
        completed would erase every other assertion from the report.
        """
        self.binary.invoke.side_effect = ProductOutputError("did not finish")
        attempt = runs.attempt(["query"], timeout_s=5, label="probe")
        self.assertFalse(attempt.ok)
        self.assertIn("did not finish", attempt.failure)
        self.assertIsNone(attempt.output)

    def test_a_successful_attempt_carries_the_capture(self) -> None:
        attempt = runs.attempt(["init"], timeout_s=5, label="seed")
        self.assertTrue(attempt.ok)
        self.assertEqual("", attempt.failure)
        self.assertEqual(0, attempt.output.exit_code)


class AccessorTests(unittest.TestCase):
    """What a scenario is allowed to know about where the product runs."""

    def setUp(self) -> None:
        runs.reset_for_test()
        self.root = support.scratch_dir(self)
        self.env = mock.Mock(spec=Environment)
        self.env.runs_dir = self.root / "runs"
        self.env.root = self.root
        self.env.scratch_dir = self.root / "scratch"
        self.binary = mock.Mock(spec=ReleaseBinary)
        self.binary.repo_root = self.root / "checkout"

    def tearDown(self) -> None:
        runs.reset_for_test()

    def test_every_accessor_refuses_before_configure(self) -> None:
        for accessor in (runs.repo_root, runs.workspace_root,
                         runs.scratch_root, runs.passphrase):
            with self.subTest(accessor=accessor.__name__):
                with self.assertRaises(HarnessError):
                    accessor()

    def test_the_accessors_answer_what_was_configured(self) -> None:
        config = mock.Mock(spec_set=["passphrase"])
        config.passphrase = "opensesame1234"
        runs.configure(self.binary, self.env, config)
        self.assertEqual(self.root / "checkout", runs.repo_root())
        self.assertEqual(self.root, runs.workspace_root())
        self.assertEqual("opensesame1234", runs.passphrase())

    def test_the_scratch_area_is_created_on_demand(self) -> None:
        """``--init-env`` makes it, but a scenario asking for it after a manual
        cleanup should get a usable directory rather than a missing one.

        The ENVIRONMENT is what creates it, so the directory arrives with the
        ignore rule that keeps its throwaway workspaces out of ``git status``.
        A bare ``mkdir`` here would have given the caller a usable directory
        and made the harness leave a trace.
        """
        self.env.prepare_scratch.side_effect = (
            lambda: self.env.scratch_dir.mkdir(parents=True, exist_ok=True))
        runs.configure(self.binary, self.env, _config_double())
        self.assertFalse(self.env.scratch_dir.exists())
        self.assertTrue(runs.scratch_root().is_dir())
        self.env.prepare_scratch.assert_called_once_with()


class DefinitionTableTests(unittest.TestCase):
    """The eight shared runs, declared complete in one place."""

    def test_every_run_of_the_table_is_declared(self) -> None:
        """All eight land here, not one per later task.

        A run whose only owner is a task nobody wrote never gets written at
        all, and the table then names something that does not exist.
        """
        self.assertEqual(["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"],
                         list(runs.DEFINITIONS))

    def test_every_invocation_asks_for_json_and_names_the_workspace(self) -> None:
        for run_id, definition in runs.DEFINITIONS.items():
            with self.subTest(run=run_id):
                argv = list(definition.argv)
                self.assertIn("--output-format", argv)
                self.assertEqual("json", argv[argv.index("--output-format") + 1])
                self.assertIn("-w", argv)

    def test_no_definition_puts_the_passphrase_on_the_command_line(self) -> None:
        """``-p`` is global, so it would ride in the archived command line of
        every run -- and any process on the machine can read a live one."""
        for run_id, definition in runs.DEFINITIONS.items():
            with self.subTest(run=run_id):
                self.assertNotIn("-p", definition.argv)
                self.assertNotIn("--passphrase", definition.argv)

    def test_the_prompt_travels_on_standard_input(self) -> None:
        for run_id, definition in runs.DEFINITIONS.items():
            with self.subTest(run=run_id):
                self.assertIsInstance(definition.stdin, bytes)

    def test_every_definition_bounds_its_own_wall_clock(self) -> None:
        """One shared constant would be wrong by an order of magnitude: the
        run that pushes a large payload through the trio and the one designed
        to fail authentication are not the same wait."""
        for run_id, definition in runs.DEFINITIONS.items():
            with self.subTest(run=run_id):
                self.assertGreater(definition.timeout_s, 0)

    def test_only_the_consulting_runs_declare_the_trio(self) -> None:
        self.assertEqual(
            {"R4", "R5", "R8"},
            {run_id for run_id, item in runs.DEFINITIONS.items()
             if item.needs_trio})

    def test_only_one_run_carries_the_large_payload(self) -> None:
        self.assertEqual(
            {"R4"},
            {run_id for run_id, item in runs.DEFINITIONS.items()
             if item.payload_size == runs.PAYLOAD_LARGE})

    def test_only_the_credential_run_plants_secrets(self) -> None:
        """The scenario searches for exactly what went in, so the definition
        is where what goes in is written down."""
        self.assertEqual(
            {"R6"},
            {run_id for run_id, item in runs.DEFINITIONS.items() if item.planted})

    def test_the_consult_run_authorises_the_tool_it_exists_to_observe(self) -> None:
        """S19 reads the consult tool's RESULT inside tool_calls[]. Under the
        default tier the tool is DENIED, so the only result is a denial
        message and S19 can report nothing but CANNOT_TEST -- which is what it
        did, measured rather than predicted. A run whose subject is the shape
        of a tool result has to be allowed to produce one.
        """
        self.assertIn("--auto", runs.DEFINITIONS["R5"].argv)

    def test_the_large_payload_run_carries_the_measured_ceiling(self) -> None:
        """MEASURED, not chosen. At --timeout 300 the product derives 40s per
        mage and 24s per attempt, and the 250 kB consult abandoned after 81.7s
        with a typed provider timeout: three FAILs in S6, one in S7, one in S8
        and three in S18, all from one under-dimensioned number. The same
        payload against the same backend at --timeout 1800 -- 249s per mage,
        149s per attempt -- completed in 103s with three real verdicts.
        """
        argv = list(runs.DEFINITIONS["R4"].argv)
        self.assertIn("--timeout", argv)
        self.assertEqual(str(runs.LARGE_CONSULT_TIMEOUT_S),
                         argv[argv.index("--timeout") + 1])

    def test_the_harness_ceiling_sits_above_the_products_own(self) -> None:
        """Not a second knob: below it, the harness would kill the very
        abandonment the run exists to observe.
        """
        self.assertGreater(runs.DEFINITIONS["R4"].timeout_s,
                           runs.LARGE_CONSULT_TIMEOUT_S)

    def test_only_one_run_rotates_a_credential(self) -> None:
        self.assertEqual(
            {"R7"},
            {run_id for run_id, item in runs.DEFINITIONS.items() if item.rotates})


class CarriedPayloadTests(unittest.TestCase):
    """What a run CARRIES is measured, never what it was declared to carry.

    ``payload_bytes`` read the length of the definition's declared prompt,
    which for R4 is 54 bytes, while the executor really did append the 250 kB
    body. S8 turned that into one CANNOT_TEST -- "the large input was never
    sent" -- and, worse, one PASS: "the generated payload stayed under the
    product's input cap", asserted over 54 bytes. A green measuring nothing.
    """

    def test_the_result_reports_the_bytes_that_were_sent(self) -> None:
        binary = support.install_fake_runs(self)
        executor = runs.RunExecutor(runs._binary, runs._env, runs._config)
        result = executor.execute(runs.DEFINITIONS["R4"])
        self.assertEqual(len(binary.calls[-1].stdin or b""),
                         result.stdin_bytes)

    def test_a_large_run_carries_more_than_its_declared_prompt(self) -> None:
        support.install_fake_runs(self)
        executor = runs.RunExecutor(runs._binary, runs._env, runs._config)
        result = executor.execute(runs.DEFINITIONS["R4"])
        self.assertGreater(result.stdin_bytes,
                           len(runs.DEFINITIONS["R4"].stdin))

    def test_a_small_run_carries_exactly_its_prompt(self) -> None:
        support.install_fake_runs(self)
        executor = runs.RunExecutor(runs._binary, runs._env, runs._config)
        result = executor.execute(runs.DEFINITIONS["R3"])
        self.assertEqual(len(runs.DEFINITIONS["R3"].stdin),
                         result.stdin_bytes)


class ControlRunTests(unittest.TestCase):
    """The control has to carry the SAME prompt as the run it controls.

    R2 duplicated R1's planting prompt while S9 compared it against R3, whose
    prompt is a quarter of the length. The difference between their token
    counts therefore mixed prompt length with what the assembler injected, and
    the run that measured 8 tokens of "injection" was measuring neither. With
    the same prompt on both sides, memory on against memory off came out at
    1668 against 392 input tokens: the assembler contributes 1276, and the
    declared margin of 100 is nowhere near it.
    """

    def test_the_control_carries_the_recall_prompt(self) -> None:
        self.assertEqual(runs.DEFINITIONS["R3"].stdin,
                         runs.DEFINITIONS["R2"].stdin)

    def test_the_control_is_the_one_that_disables_memory(self) -> None:
        self.assertIn("--no-memory", runs.DEFINITIONS["R2"].argv)
        self.assertNotIn("--no-memory", runs.DEFINITIONS["R3"].argv)


class ShortTreeTests(unittest.TestCase):
    """A tree too small for the payload degrades the run, never the harness.

    ``PayloadBuilder.build`` raises, and the raise reached ``main``'s
    ``except HarnessError`` -- exit 3, which is the code reserved for a bug in
    the HARNESS. Section 7.2 and S8's own trap text both say the answer is
    ``CANNOT_TEST``: the harness must not accuse the product of a size it was
    never sent, and it must not accuse itself of a bug either. S8 already had
    the branch for it, and nothing could reach it.
    """

    def test_a_tree_too_small_leaves_the_run_carrying_its_prompt(self) -> None:
        support.install_fake_runs(self)
        executor = runs.RunExecutor(
            runs._binary, runs._env,
            support.FakeConfig(support.FAKE_PASSPHRASE,
                               payload_target_bytes=10 ** 12))
        result = executor.execute(runs.DEFINITIONS["R4"])
        self.assertEqual(len(runs.DEFINITIONS["R4"].stdin), result.stdin_bytes)

    def test_any_other_harness_error_still_stops_the_run(self) -> None:
        """The degrade is for a tree too small and for nothing else.

        Catching every ``HarnessError`` here reads the same as catching the
        one that matters, and it is not: a source file that cannot be read, a
        target that makes no sense, a builder that fails for any reason at all
        would silently become "the large run carried its prompt alone" -- and
        S8, whose whole subject is what the product does with a large input,
        would report on a payload that was never assembled. Widen the except
        and this goes green while S8 quietly measures nothing.
        """
        support.install_fake_runs(self)
        executor = runs.RunExecutor(
            runs._binary, runs._env,
            support.FakeConfig(support.FAKE_PASSPHRASE,
                               payload_target_bytes=4096))
        with mock.patch.object(runs.PayloadBuilder, "build",
                               side_effect=HarnessError("unreadable source")):
            with self.assertRaises(HarnessError):
                executor.execute(runs.DEFINITIONS["R4"])


class BaselineCaptureTests(unittest.TestCase):
    """The rotation's baseline is produced by the EXECUTOR, and nothing tested it.

    S16's three interesting cases inject a hand-built ``baseline=`` dict, and
    the one production test drove a responder that answered ``{}`` to
    everything -- so ``_history_counts`` ran, parsed nothing, returned None,
    and its result was never asserted. If it never produced a baseline in
    production, S16 assertion 2 would be CANNOT_TEST forever, the gate would
    block on every run, and the whole suite would stay green.
    """

    def _executor(self, report: bytes):
        """An executor whose vault answers *report* to ``diagnose``.

        Args:
            report: What the product should print for the diagnose call.

        Returns:
            RunExecutor: Configured against the double.
        """
        def responder(call):
            args = list(call.args)
            if args[:1] != ["vault"]:
                return None
            body = report if "diagnose" in args else b""
            return ProductOutput(stdout=body, stderr=b"", exit_code=0,
                                 command=["magi-rs"] + args)

        support.install_fake_runs(self, responder=responder)
        return runs.RunExecutor(runs._binary, runs._env, runs._config,
                                credential="the-real-credential")

    def test_the_rotation_records_the_counts_it_read(self) -> None:
        report = ("envelope: present" + chr(10) + "counts:" + chr(10) +
                  "  vault: 1" + chr(10) + "  sessions: 2" + chr(10) +
                  "  messages: 9" + chr(10)).encode("utf-8")
        result = self._executor(report).execute(runs.DEFINITIONS["R7"])
        self.assertEqual({"vault": 1, "sessions": 2, "messages": 9},
                         result.baseline)

    def test_a_report_it_cannot_read_records_no_baseline(self) -> None:
        """None means NOT MEASURED, and S16 refuses rather than assuming."""
        result = self._executor(b"envelope: absent").execute(
            runs.DEFINITIONS["R7"])
        self.assertIsNone(result.baseline)

    def test_a_run_that_does_not_rotate_records_none(self) -> None:
        result = self._executor(b"counts:").execute(runs.DEFINITIONS["R3"])
        self.assertIsNone(result.baseline)


class PlaceholderLifetimeTests(unittest.TestCase):
    """R6's four vault entries are planted and removed as a unit.

    They were planted OUTSIDE the try, so a `vault set` that failed part-way --
    a 180 s Argon2 open under load is documented in this repository, not
    hypothetical -- left the earlier entries behind with no cleanup path at
    all. And the removal loop had no per-entry guard, so the first failure
    skipped the rest. Nothing recovers them either: the preflight's rotation
    recovery knows only about the marker.
    """

    def test_a_failure_part_way_through_still_removes_what_was_planted(self) -> None:
        planted, removed = [], []

        def flaky(call):
            args = list(call.args)
            if args[:1] != ["vault"]:
                return None
            verb = "set" if "set" in args else ("rm" if "rm" in args else "")
            name = args[args.index(verb) + 1] if verb else ""
            if verb == "set":
                if len(planted) == 2:
                    raise ProductOutputError("vault set timed out")
                planted.append(name)
            if verb == "rm":
                removed.append(name)
            return ProductOutput(stdout=b"", stderr=b"", exit_code=0,
                                 command=["magi-rs"] + args)

        support.install_fake_runs(self, responder=flaky)
        executor = runs.RunExecutor(runs._binary, runs._env, runs._config,
                                    credential="the-real-credential")
        with self.assertRaises(ProductOutputError):
            executor.execute(runs.DEFINITIONS["R6"])
        self.assertEqual(set(planted), set(removed),
                         "everything planted has to be taken back out")

    def test_a_failed_restore_is_typed_and_keeps_the_run_s_own_failure(self):
        """The same defect the placeholder removal had, in R7's finally.

        A restore that fails is NOT inert -- the environment is left holding
        a sentinel credential and every backend run after it dies of an
        opaque auth error -- so unlike a leftover placeholder it must stop
        the harness. What it must not do is stop it as an untyped
        ProductOutputError escaping into main's last-resort catch: that is a
        traceback and exit 3, the code reserved for a defect in the harness,
        with the report discarded and every finding from every completed run
        lost. A typed HarnessError says the same thing through the handler
        that prints a cause and returns a code.
        """
        def stubborn(call):
            args = list(call.args)
            if args[:1] != ["vault"]:
                return None
            if "set" in args and runs.ROTATION_MARKER not in args:
                raise ProductOutputError("vault set refused")
            return ProductOutput(stdout=b"", stderr=b"", exit_code=0,
                                 command=["magi-rs"] + args)

        support.install_fake_runs(self, responder=stubborn)
        executor = runs.RunExecutor(runs._binary, runs._env, runs._config,
                                    credential="the-real-credential")
        with self.assertRaises(HarnessError) as caught:
            executor.execute(runs.DEFINITIONS["R7"])
        self.assertIn("credential", str(caught.exception))

    def test_a_failed_removal_warns_instead_of_taking_the_harness_down(self):
        """A leftover entry is a note, never a verdict on the product.

        Raising out of the ``finally`` replaced whatever was already in
        flight -- R6's own ``TimedOut`` among them, which the runner needs to
        substitute CANNOT_TEST for S10 -- and then escaped ``execute``. Main
        types only PreflightError and HarnessError, so it reached the
        last-resort catch: a traceback, exit 3, no report, and every finding
        from every completed run discarded. Exit 3 says the HARNESS failed;
        the event is one ``vault rm`` the product refused.
        """
        def stubborn(call):
            args = list(call.args)
            if args[:1] != ["vault"]:
                return None
            if "rm" in args:
                raise ProductOutputError("vault rm refused")
            return ProductOutput(stdout=b"", stderr=b"", exit_code=0,
                                 command=["magi-rs"] + args)

        support.install_fake_runs(self, responder=stubborn)
        executor = runs.RunExecutor(runs._binary, runs._env, runs._config,
                                    credential="the-real-credential")
        noise = io.StringIO()
        with contextlib.redirect_stderr(noise):
            result = executor.execute(runs.DEFINITIONS["R6"])
        self.assertEqual("R6", result.run_id)
        self.assertIn("placeholder", noise.getvalue())

    def test_one_failed_removal_does_not_skip_the_rest(self) -> None:
        removed = []

        def stubborn(call):
            args = list(call.args)
            if args[:1] != ["vault"]:
                return None
            if "rm" in args:
                name = args[args.index("rm") + 1]
                if not removed:
                    removed.append(name)
                    raise ProductOutputError("vault rm timed out")
                removed.append(name)
            return ProductOutput(stdout=b"", stderr=b"", exit_code=0,
                                 command=["magi-rs"] + args)

        support.install_fake_runs(self, responder=stubborn)
        executor = runs.RunExecutor(runs._binary, runs._env, runs._config,
                                    credential="the-real-credential")
        noise = io.StringIO()
        with contextlib.redirect_stderr(noise):
            executor.execute(runs.DEFINITIONS["R6"])
        self.assertEqual(len(runs.PLACEHOLDER_ENTRIES), len(removed),
                         "every entry gets its own removal attempt")
        self.assertIn("left placeholder entries", noise.getvalue())


class ArchiveScrubbingTests(unittest.TestCase):
    """The passphrase is scrubbed on BOTH archive paths, not just one.

    ``RunExecutor.archive`` adds it deliberately and explains why. The
    ``invoke()`` path -- every autonomous scenario: S2, S3, S4, S5, S9, S14,
    S15, S17 -- passes only the secrets the caller declared, while the child
    carries the same passphrase in its environment. One path guarded, one not,
    for the secret that opens the whole vault.
    """

    def test_the_passphrase_never_reaches_an_archived_invocation(self) -> None:
        """Driven by a product that DOES echo it, because one that does not
        cannot tell a guarded path from an unguarded one.
        """
        def leaks(call):
            return ProductOutput(
                stdout=b"error: could not open with " +
                       support.FAKE_PASSPHRASE.encode("utf-8"),
                stderr=b"", exit_code=1, command=["magi-rs"] + list(call.args))

        support.install_fake_runs(self, responder=leaks)
        runs.invoke(["vault", "ls"], timeout_s=5, label="probe")
        archived = b"".join(path.read_bytes() for path in
                            pathlib.Path(runs.archive_root()).rglob("*")
                            if path.is_file())
        self.assertTrue(archived, "nothing was archived")
        self.assertNotIn(support.FAKE_PASSPHRASE.encode("utf-8"), archived)


class AuthenticatedBackendTests(unittest.TestCase):
    """R6 has to put the credential in the URL, not only in a header.

    The spec picks R6 as "an authenticated ``base_url`` that fails" for one
    reason: the defect it guards lives in the vault percent-encoding a
    credential INTO the authority, where a reserved character moves the last
    ``@`` that ``redact_url`` anchors on. A credential carried in
    ``Authorization`` never travels that path, so the percent-encoded forms in
    ``PlantedSecret.forms()`` were unreachable and the shape detector had no
    authority to look at anywhere in 33 kB of output.

    Measured against the product: a literal credential in ``base_url`` is
    refused at load, naming the placeholder mechanism; the placeholders then
    need BOTH the root pair and the ``MAGI_``-prefixed pair, because the trio
    section inherits the root endpoint. With all four planted the run reaches
    the backend, and the product's error reads ``"Basic [REDACTED]"``.
    """

    def test_the_run_declares_the_authenticated_token(self) -> None:
        declared = dict(runs.DEFINITIONS["R6"].env)
        self.assertEqual(runs.ERROR_BACKEND_TOKEN,
                         declared["OPENAI_BASE_URL"])

    def test_the_resolved_url_carries_the_placeholders(self) -> None:
        binary = support.install_fake_runs(self)
        executor = runs.RunExecutor(runs._binary, runs._env, runs._config,
                                    credential="the-real-credential")
        executor.execute(runs.DEFINITIONS["R6"])
        resolved = (_query_call(binary).env or {}).get("OPENAI_BASE_URL", "")
        self.assertIn(runs.USER_PLACEHOLDER, resolved)
        self.assertIn(runs.PASSWORD_PLACEHOLDER, resolved)

    def test_the_credential_is_planted_in_the_vault_and_removed(self) -> None:
        """It travels through the vault, never as an argument, and it does not
        outlive the run: a leftover entry would make the next run's endpoint
        authenticated by accident.
        """
        binary = support.install_fake_runs(self)
        executor = runs.RunExecutor(runs._binary, runs._env, runs._config,
                                    credential="the-real-credential")
        executor.execute(runs.DEFINITIONS["R6"])
        planted, removed = [], []
        for call in binary.calls:
            args = list(call.args)
            if args[:1] != ["vault"]:
                continue
            if "set" in args:
                planted.append(args[args.index("set") + 1])
            if "rm" in args:
                removed.append(args[args.index("rm") + 1])
        self.assertEqual(set(runs.PLACEHOLDER_ENTRIES), set(planted))
        self.assertEqual(set(runs.PLACEHOLDER_ENTRIES), set(removed))
        for call in binary.calls:
            self.assertNotIn(runs._R6_CREDENTIAL.value, " ".join(call.args))


class ErrorBackendTests(unittest.TestCase):
    """R6 has to reach an endpoint that ANSWERS with an error, fast.

    Two mechanisms were tried and measured before this one. ``127.0.0.1:9``
    was declared "the discard service, reserved and unused"; where that
    service is running the connection is accepted and never answered, and R6
    hung past its ceiling, was killed with no output, and S10 reported
    CANNOT_TEST over three assertions it never got to search. An ephemeral
    port bound and released does not help either: this machine drops the
    packet instead of refusing it, so the wait is the same. Bounding the run
    with the product's own ``--timeout`` does finish, and it is worse than
    both -- the measured output carried neither the credential nor the
    endpoint, so all three assertions would have passed over nothing.

    What is left is the mechanism that was always the strongest: answer. A
    local endpoint that returns 401 and ECHOES the Authorization header into
    its body puts the credential onto the product's error path deliberately,
    which is where the redaction defect this repository already fixed once
    actually lived. Without it S10 can only show the product invented no leak;
    with it, S10 shows the product does not repeat one it was handed.
    """

    def test_it_answers_every_request_with_an_error(self) -> None:
        with runs.error_backend() as url:
            status, _ = _http_post(url, b"the-planted-credential")
        self.assertEqual(runs.ERROR_BACKEND_STATUS, status)

    def test_the_body_echoes_the_credential_it_was_sent(self) -> None:
        """This is the plant. A body that does not carry the credential
        cannot show whether the product would repeat one.
        """
        with runs.error_backend() as url:
            _, body = _http_post(url, b"the-planted-credential")
        self.assertIn(b"the-planted-credential", body)

    def test_it_stops_listening_once_the_run_is_over(self) -> None:
        """One leaked listener per run is a leaked thread and a leaked port."""
        with runs.error_backend() as url:
            port = urllib.parse.urlsplit(url).port
        with self.assertRaises(OSError):
            _http_post("http://127.0.0.1:%d/v1" % port, b"x")

    def test_the_run_declares_the_token_rather_than_a_fixed_url(self) -> None:
        declared = dict(runs.DEFINITIONS["R6"].env)
        self.assertEqual(runs.ERROR_BACKEND_TOKEN,
                         declared["OPENAI_BASE_URL"])

    def test_the_token_is_resolved_before_the_product_sees_it(self) -> None:
        binary = support.install_fake_runs(self)
        executor = runs.RunExecutor(runs._binary, runs._env, runs._config,
                                    credential="the-real-credential")
        executor.execute(runs.DEFINITIONS["R6"])
        overlay = _query_call(binary).env or {}
        self.assertNotIn(runs.ERROR_BACKEND_TOKEN,
                         overlay.get("OPENAI_BASE_URL", ""))
        self.assertIn("@127.0.0.1:", overlay.get("OPENAI_BASE_URL", ""))


class NeededRunsTests(unittest.TestCase):
    """Nothing expensive is paid for by a scenario that cannot execute."""

    @staticmethod
    def _registry(*entries) -> Registry:
        """Build a registry holding exactly *entries*.

        Args:
            entries: The scenarios to record.

        Returns:
            Registry: Its own, so no test touches the process-wide one.
        """
        registry = Registry()
        for entry in entries:
            registry.add(entry)
        return registry

    @staticmethod
    def _entry(scenario_id, run, needs_backend=False) -> ScenarioEntry:
        """Build one registered scenario.

        Args:
            scenario_id: The id.
            run: The run id, tuple of ids, or None.
            needs_backend: Whether it needs a reachable backend.

        Returns:
            ScenarioEntry: The entry, with a body nothing invokes.
        """
        return ScenarioEntry(scenario_id, lambda run: iter(()), run,
                             needs_backend, False, False)

    def test_a_run_no_scenario_declares_is_never_executed(self) -> None:
        self.assertEqual([], runs.needed_runs(self._registry(), True))

    def test_a_declared_run_is_needed(self) -> None:
        registry = self._registry(self._entry("S1", "R1"))
        self.assertEqual(["R1"], runs.needed_runs(registry, True))

    def test_a_scenario_that_cannot_execute_pays_for_nothing(self) -> None:
        """With the backend down the expensive half would produce nothing but
        CANNOT_TEST, so it is not spent."""
        registry = self._registry(self._entry("S6", "R4", needs_backend=True))
        self.assertEqual([], runs.needed_runs(registry, False))
        self.assertEqual(["R4"], runs.needed_runs(registry, True))

    def test_a_scenario_reading_several_runs_needs_all_of_them(self) -> None:
        registry = self._registry(self._entry("S9", ("R1", "R2", "R3")))
        self.assertEqual(["R1", "R2", "R3"], runs.needed_runs(registry, True))

    def test_the_order_is_the_table_and_not_the_registration(self) -> None:
        """The expensive runs must not reorder between invocations."""
        registry = self._registry(self._entry("S18", "R8"),
                                  self._entry("S1", "R1"))
        self.assertEqual(["R1", "R8"], runs.needed_runs(registry, True))

    def test_one_run_declared_twice_is_executed_once(self) -> None:
        registry = self._registry(self._entry("S1", "R1"),
                                  self._entry("S5", "R1"))
        self.assertEqual(["R1"], runs.needed_runs(registry, True))


class _Recorder:
    """A responder that answers every invocation with one canned capture."""

    def __init__(self, stdout=b"", stderr=b"", exit_code=0) -> None:
        """Create the responder.

        Args:
            stdout: What the fake product prints.
            stderr: What it prints to stderr.
            exit_code: What it exits with.
        """
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code

    def __call__(self, call: support.Call) -> ProductOutput:
        """Answer one invocation.

        Args:
            call: What the fake binary was asked to run.

        Returns:
            ProductOutput: The canned capture.
        """
        return ProductOutput(stdout=self.stdout, stderr=self.stderr,
                             exit_code=self.exit_code,
                             command=["magi-rs"] + list(call.args))


class ExecutorTests(unittest.TestCase):
    """One shared run: executed at full fidelity, archived scrubbed."""

    def setUp(self) -> None:
        self.secret = PlantedSecret("s3cr3t-value", "backend credential")
        self.binary = support.install_fake_runs(
            self, _Recorder(stdout=b'{"response":"leaked s3cr3t-value"}'))
        self.executor = runs.RunExecutor(runs._binary, runs._env, runs._config)
        self.definition = runs.RunDefinition(
            run_id="RT", argv=("query", "--output-format", "json", "-w", "."),
            stdin=b"say ok", needs_trio=False,
            payload_size=runs.PAYLOAD_SMALL, timeout_s=30,
            planted=(self.secret,))

    def test_the_result_carries_what_the_run_planted(self) -> None:
        """A result with output and no idea what to search for hands the
        scenario a haystack and no needle."""
        result = self.executor.execute(self.definition)
        self.assertEqual((self.secret,), result.planted)
        self.assertEqual("RT", result.run_id)

    def test_the_prompt_reaches_the_child_on_standard_input(self) -> None:
        self.executor.execute(self.definition)
        self.assertEqual(b"say ok", self.binary.calls[0].stdin)

    def test_the_passphrase_travels_in_the_environment(self) -> None:
        self.executor.execute(self.definition)
        self.assertEqual(support.FAKE_PASSPHRASE,
                         self.binary.calls[0].env["MAGI_PASSPHRASE"])

    def test_the_run_is_timed(self) -> None:
        result = self.executor.execute(self.definition)
        self.assertGreaterEqual(result.duration_s, 0.0)
        self.assertFalse(result.timed_out)

    def test_the_result_keeps_the_secret_at_full_fidelity(self) -> None:
        result = self.executor.execute(self.definition)
        self.assertIn(b"s3cr3t-value", result.output.raw())

    def test_the_archive_records_the_four_parts_of_an_invocation(self) -> None:
        result = self.executor.execute(self.definition)
        self.executor.archive(result)
        directory = pathlib.Path(runs._env.runs_dir) / "RT"
        for name in ("command", "stdout", "stderr", "exit_code"):
            with self.subTest(part=name):
                self.assertTrue((directory / name).is_file())
        self.assertEqual(
            "0", (directory / "exit_code").read_text(encoding="utf-8").strip())

    def test_the_archived_bytes_are_scrubbed_and_say_what_was_removed(self) -> None:
        """Scrubbing before the assertions run would turn the security
        scenarios into empty greens; not scrubbing at all writes a credential
        into a directory that outlives the run."""
        result = self.executor.execute(self.definition)
        self.executor.archive(result)
        written = b"".join(
            path.read_bytes()
            for path in (pathlib.Path(runs._env.runs_dir) / "RT").iterdir())
        self.assertNotIn(b"s3cr3t-value", written)
        self.assertIn(b"backend credential", written)


class TimeoutTests(unittest.TestCase):
    """A run that hangs is a result, not the end of the harness."""

    def setUp(self) -> None:
        self.partial = ProductOutput(
            stdout=b'{"applied_caps":{"ceiling_floored":true}}', stderr=b"",
            exit_code=-1, command=["magi-rs", "consult"])

        def responder(call: support.Call) -> ProductOutput:
            raise TimedOut("the product did not finish", output=self.partial)

        support.install_fake_runs(self, responder)
        self.executor = runs.RunExecutor(runs._binary, runs._env, runs._config)
        self.definition = runs.RunDefinition(
            run_id="RT", argv=("consult", "--output-format", "json", "-w", "."),
            stdin=b"say ok", needs_trio=True,
            payload_size=runs.PAYLOAD_SMALL, timeout_s=1)

    def test_a_timeout_becomes_a_result_rather_than_an_exception(self) -> None:
        """Raising would abort the whole harness over one slow provider and
        lose every scenario that had nothing to do with it."""
        result = self.executor.execute(self.definition)
        self.assertTrue(result.timed_out)
        self.assertEqual("RT", result.run_id)

    def test_whatever_was_captured_before_the_hang_survives(self) -> None:
        """S7's exemption rests entirely on reading applied_caps out of what
        the run emitted before it hung. An expiry path that returns empty
        output removes the one case the design grants.
        """
        result = self.executor.execute(self.definition)
        self.assertIn(b"applied_caps", result.output.raw())

    def test_a_hang_that_emitted_nothing_is_still_a_result(self) -> None:
        """Partial output is what the design asks for, not what it requires:
        a process killed before its first write has none, and the run still
        has to come back as timed out rather than as an exception.
        """

        def silent(call: support.Call) -> ProductOutput:
            raise TimedOut("the product did not finish")

        support.install_fake_runs(self, silent)
        executor = runs.RunExecutor(runs._binary, runs._env, runs._config)
        result = executor.execute(self.definition)
        self.assertTrue(result.timed_out)
        self.assertEqual(b"", result.output.stdout)


class RotationTests(unittest.TestCase):
    """R7 undoes itself, and the undo is not on the happy path."""

    def setUp(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.real = "the-real-backend-credential"
        self.raise_on_query = False

        def responder(call: support.Call) -> ProductOutput:
            self.calls.append(call.args)
            if self.raise_on_query and call.args[0] == "query":
                raise ProductOutputError("the product could not be started")
            return ProductOutput(stdout=b"{}", stderr=b"", exit_code=0,
                                 command=["magi-rs"] + list(call.args))

        support.install_fake_runs(self, responder)
        self.key_env = "OPENAI_API_KEY"
        runs._config.backend_key_env = self.key_env
        self.executor = runs.RunExecutor(runs._binary, runs._env, runs._config,
                                         credential=self.real)
        self.definition = runs.DEFINITIONS["R7"]

    def _subcommands(self) -> list[tuple[str, ...]]:
        """The first three words of every invocation, in order.

        Returns:
            list[tuple[str, ...]]: One entry per call.
        """
        return [args[:3] for args in self.calls]

    def test_the_marker_is_written_before_the_credential_moves(self) -> None:
        """The marker is what a later preflight sees after a power cut.

        Written AFTER the rotation it would be useless in exactly the case it
        exists for: the process dies in between, the credential is a sentinel,
        and nothing on disk says so.
        """
        self.executor.execute(self.definition)
        order = self._subcommands()
        self.assertLess(order.index(("vault", "set", ROTATION_MARKER)),
                        order.index(("vault", "set", self.key_env)))

    def test_the_credential_is_restored_and_the_marker_removed(self) -> None:
        self.executor.execute(self.definition)
        self.assertEqual(("vault", "rm", ROTATION_MARKER),
                         self._subcommands()[-1])
        written = [call for call in self.calls
                   if call[:3] == ("vault", "set", self.key_env)]
        self.assertEqual(2, len(written))

    def test_the_restore_runs_even_when_the_run_raises(self) -> None:
        """The undo lives in a finally, not after the happy path."""
        self.raise_on_query = True
        with self.assertRaises(ProductOutputError):
            self.executor.execute(self.definition)
        self.assertEqual(("vault", "rm", ROTATION_MARKER),
                         self._subcommands()[-1])
        self.assertIn(("vault", "set", self.key_env),
                      self._subcommands()[-3:])

    def test_the_sentinel_is_not_a_second_real_credential(self) -> None:
        """S16 only needs the stored value to CHANGE. Another valid credential
        would add nothing to any assertion and one more secret to protect."""
        self.assertNotIn(self.real.encode("utf-8"), runs.ROTATION_SENTINEL)

    def test_rotating_without_a_credential_to_restore_is_refused(self) -> None:
        """Rotating a value the harness cannot put back would destroy it.

        The refusal is a HarnessError -- the harness was not given what it
        needs -- and never a verdict on the product.
        """
        executor = runs.RunExecutor(runs._binary, runs._env, runs._config,
                                    credential="")
        with self.assertRaises(HarnessError):
            executor.execute(self.definition)
        self.assertEqual([], self.calls)




class LargePayloadWiringTests(unittest.TestCase):
    """A run declaring the large payload must actually be SENT one."""

    def setUp(self) -> None:
        runs.reset_for_test()
        self.addCleanup(runs.reset_for_test)

    def test_a_large_payload_run_is_sent_more_than_its_literal_prompt(self) -> None:
        """``payload_size`` was declared on every definition and read by
        nothing, so R4 -- the whole reason the large payload exists -- went out
        carrying a 53-byte literal. S8 would then have asserted a token floor
        against a size the product was never sent, which is precisely what S8's
        own trap text forbids: never accuse the product of a size it did not
        receive.
        """
        sent = {}
        binary = mock.Mock(spec=ReleaseBinary)

        def answer(args, stdin=None, env=None, timeout=None, cwd=None):
            sent["stdin"] = stdin
            return ProductOutput(stdout=b"{}", stderr=b"", exit_code=0,
                                 command=["magi-rs"] + list(args))

        binary.invoke.side_effect = answer
        binary.repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
        env = mock.Mock(spec=Environment)
        env.runs_dir = support.scratch_dir(self)
        config = mock.Mock(spec=SmokeConfig)
        config.passphrase = "correct horse battery staple"
        config.payload_target_bytes = 4096
        config.payload_token_floor = 10
        config.backend_key_env = "OPENAI_API_KEY"
        runs.configure(binary, env, config)
        runs.RunExecutor(binary, env, config).execute(runs.DEFINITIONS["R4"])
        sent_bytes = len(sent["stdin"] or b"")
        self.assertGreaterEqual(sent_bytes, config.payload_target_bytes)
        # Upper bound too, and it is the half that guards the CONFIG. Asserting
        # only "at least the target" passes just as well when the size is taken
        # from the module constant, which is exactly the copy-that-forgets-to-
        # update this reads from the config to avoid.
        self.assertLess(sent_bytes, config.payload_target_bytes * 2,
                        "the size must come from the config, not the default")

    def test_a_small_payload_run_is_not_inflated(self) -> None:
        """The large payload is expensive and exactly one run carries it."""
        sent = {}
        binary = mock.Mock(spec=ReleaseBinary)

        def answer(args, stdin=None, env=None, timeout=None, cwd=None):
            sent["stdin"] = stdin
            return ProductOutput(stdout=b"{}", stderr=b"", exit_code=0,
                                 command=["magi-rs"] + list(args))

        binary.invoke.side_effect = answer
        binary.repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
        env = mock.Mock(spec=Environment)
        env.runs_dir = support.scratch_dir(self)
        config = mock.Mock(spec=SmokeConfig)
        config.passphrase = "correct horse battery staple"
        config.payload_target_bytes = 4096
        config.payload_token_floor = 10
        config.backend_key_env = "OPENAI_API_KEY"
        runs.configure(binary, env, config)
        runs.RunExecutor(binary, env, config).execute(runs.DEFINITIONS["R1"])
        self.assertLess(len(sent["stdin"] or b""), 4096)


if __name__ == "__main__":
    unittest.main()
