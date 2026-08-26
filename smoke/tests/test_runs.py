# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Tests for the one door between a scenario and the product."""

import pathlib
import tempfile
import unittest
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
        env.runs_dir = pathlib.Path(tempfile.mkdtemp())
        runs.configure(binary, env, mock.Mock(spec=SmokeConfig))
        output = runs.invoke(["--version"], timeout_s=5, label="probe")
        self.assertEqual(0, output.exit_code)
        binary.invoke.assert_called_once()


class ArchiveTests(unittest.TestCase):
    """The returned capture is raw; only what reaches disk is scrubbed."""

    def setUp(self) -> None:
        runs.reset_for_test()
        self.root = pathlib.Path(tempfile.mkdtemp())
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
        runs.configure(self.binary, self.env, mock.Mock(spec=SmokeConfig))

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
        self.root = pathlib.Path(tempfile.mkdtemp())
        self.env = mock.Mock(spec=Environment)
        self.env.runs_dir = self.root
        self.binary = mock.Mock(spec=ReleaseBinary)
        self.binary.invoke.return_value = ProductOutput(
            stdout=b"", stderr=b"", exit_code=0, command=["magi-rs", "init"]
        )
        runs.configure(self.binary, self.env, mock.Mock(spec=SmokeConfig))

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
        self.root = pathlib.Path(tempfile.mkdtemp())
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
        runs.configure(self.binary, self.env, mock.Mock(spec=SmokeConfig))
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

    def test_only_one_run_rotates_a_credential(self) -> None:
        self.assertEqual(
            {"R7"},
            {run_id for run_id, item in runs.DEFINITIONS.items() if item.rotates})


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


if __name__ == "__main__":
    unittest.main()


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
        env = mock.Mock(spec=Environment)
        env.runs_dir = pathlib.Path(tempfile.mkdtemp())
        config = mock.Mock(spec=SmokeConfig)
        config.passphrase = "correct horse battery staple"
        config.payload_target_bytes = 4096
        config.payload_token_floor = 10
        config.backend_key_env = "OPENAI_API_KEY"
        runs.configure(binary, env, config)
        runs.RunExecutor(binary, env, config).execute(runs.DEFINITIONS["R4"])
        self.assertGreaterEqual(len(sent["stdin"] or b""),
                                config.payload_target_bytes)

    def test_a_small_payload_run_is_not_inflated(self) -> None:
        """The large payload is expensive and exactly one run carries it."""
        sent = {}
        binary = mock.Mock(spec=ReleaseBinary)

        def answer(args, stdin=None, env=None, timeout=None, cwd=None):
            sent["stdin"] = stdin
            return ProductOutput(stdout=b"{}", stderr=b"", exit_code=0,
                                 command=["magi-rs"] + list(args))

        binary.invoke.side_effect = answer
        env = mock.Mock(spec=Environment)
        env.runs_dir = pathlib.Path(tempfile.mkdtemp())
        config = mock.Mock(spec=SmokeConfig)
        config.passphrase = "correct horse battery staple"
        config.payload_target_bytes = 4096
        config.payload_token_floor = 10
        config.backend_key_env = "OPENAI_API_KEY"
        runs.configure(binary, env, config)
        runs.RunExecutor(binary, env, config).execute(runs.DEFINITIONS["R1"])
        self.assertLess(len(sent["stdin"] or b""), 4096)
