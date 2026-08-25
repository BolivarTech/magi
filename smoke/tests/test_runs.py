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
from smoke.config import SmokeConfig
from smoke.env import Environment
from smoke.errors import HarnessError, ProductOutputError
from smoke.product import ProductOutput
from smoke.secrets import PlantedSecret


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


if __name__ == "__main__":
    unittest.main()
