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
from smoke.errors import HarnessError
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
        runs.configure(binary, mock.Mock(spec=Environment), mock.Mock(spec=SmokeConfig))
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


if __name__ == "__main__":
    unittest.main()
