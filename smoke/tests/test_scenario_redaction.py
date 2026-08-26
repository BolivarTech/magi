# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Unit tests for the S10 scenario's own shape."""

import pathlib
import tempfile
import unittest
import urllib.parse
from unittest import mock

from smoke.outcome import Outcome
from smoke.registry import DEFAULT_REGISTRY
from smoke.scenarios import redaction  # noqa: F401 - import registers it
from smoke.secrets import PlantedSecret, mint_credential

#: RESERVED characters on purpose. Without them the percent-encoded form
#: equals the raw one -- ``quote`` leaves letters, digits and ``-._~``
#: alone -- so the encoding test would exercise no encoding at all and a
#: raw-only search would pass it. The mutation said so before this line
#: did.
_SECRET = "sup3r@s3cr3t/value#1"
_LABEL = "backend credential"


class _FakeRun:
    """A RunResult double carrying a capture and what was planted in it.

    Attributes:
        stdout: What the product wrote to standard output.
        stderr: What it wrote to standard error.
        planted: The secrets the executor put in front of it.
    """

    def __init__(self, stdout=b"", stderr=b"", planted=None, timed_out=False):
        """Build the double.

        Args:
            stdout: Standard output bytes.
            stderr: Standard error bytes.
            planted: Secrets the run was given, or None for the default one.
            timed_out: Whether the run exceeded its ceiling.
        """
        self.output = _FakeOutput(stdout, stderr)
        self.planted = planted if planted is not None else (
            PlantedSecret(_SECRET, _LABEL),)
        self.timed_out = timed_out
        self.run_id = "R6"


class _FakeOutput:
    """The capture half of the double.

    Attributes:
        stdout: Standard output bytes.
        stderr: Standard error bytes.
    """

    def __init__(self, stdout, stderr):
        """Build the capture.

        Args:
            stdout: Standard output bytes.
            stderr: Standard error bytes.
        """
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = 1
        self.command = ["magi-rs", "query"]

    def raw(self):
        """Both streams, joined by one newline.

        Returns:
            bytes: The capture, unscrubbed.
        """
        return self.stdout + b"\n" + self.stderr


def _outcomes(run) -> dict[str, Outcome]:
    """Run S10 against *run* and index its outcomes by assertion.

    Args:
        run: The run result double.

    Returns:
        dict[str, Outcome]: What each assertion concluded.
    """
    return {finding.assertion: finding.outcome
            for finding in DEFAULT_REGISTRY.get("S10").func(run)}


class RedactionScenarioTests(unittest.TestCase):
    """S10 registers under the right id, over the run that fails on purpose."""

    def test_it_registers_under_s10_against_r6(self) -> None:
        entry = DEFAULT_REGISTRY.get("S10")
        self.assertEqual("R6", entry.run)
        self.assertTrue(entry.needs_backend)

    def test_the_assertion_texts_are_the_spec_texts(self) -> None:
        self.assertEqual(
            [
                "the credential appears in neither stdout nor stderr",
                "nor in the JSON output",
                "nor in the run log",
                "every authority emitted anywhere — JSON, stdout, stderr, run "
                "log — is redacted; asserted on shape, not on the secret",
            ],
            list(redaction.ASSERTIONS),
        )


class RedactionBodyTests(unittest.TestCase):
    """What S10 concludes about output that leaks, and output that does not."""

    def test_a_clean_capture_passes_the_three_it_can_evaluate(self) -> None:
        """With no archive configured, assertion 3 has nothing to search.

        It reports CANNOT_TEST rather than PASS, which is the discipline this
        harness runs on: finding no leak in a channel you never opened is not
        evidence of anything.
        """
        run = _FakeRun(stdout=b'{"ok": true}',
                       stderr=b"error: request to https://***@host/v1 failed")
        outcomes = _outcomes(run)
        for index in (0, 1, 3):
            self.assertEqual(Outcome.PASS, outcomes[redaction.ASSERTIONS[index]])
        self.assertEqual(Outcome.CANNOT_TEST,
                         outcomes[redaction.ASSERTIONS[2]])

    def test_a_clean_archive_passes_the_third(self) -> None:
        """The other half: with a log present and clean, assertion 3 evaluates."""
        directory = pathlib.Path(tempfile.mkdtemp()) / "R6"
        directory.mkdir(parents=True)
        (directory / "stderr").write_bytes(
            b"$ magi-rs query\n<redacted: backend credential, 1 occurrence>")
        with mock.patch.object(redaction.runs, "archive_root",
                               return_value=directory.parent):
            outcomes = _outcomes(_FakeRun(stdout=b'{"ok": true}'))
        self.assertEqual(Outcome.PASS, outcomes[redaction.ASSERTIONS[2]])

    def test_a_leak_in_the_archive_fails_the_third(self) -> None:
        """The archive is written SCRUBBED, so a secret found there is one the
        scrubber was never told about -- the case worth reporting.
        """
        directory = pathlib.Path(tempfile.mkdtemp()) / "R6"
        directory.mkdir(parents=True)
        (directory / "stderr").write_bytes(
            ("$ magi-rs query\n%s" % _SECRET).encode("utf-8"))
        with mock.patch.object(redaction.runs, "archive_root",
                               return_value=directory.parent):
            outcomes = _outcomes(_FakeRun(stdout=b'{"ok": true}'))
        self.assertEqual(Outcome.FAIL, outcomes[redaction.ASSERTIONS[2]])

    def test_another_scenario_s_archive_is_not_this_run_s(self) -> None:
        """S10 concatenated EVERY invocation.log under the archive root, so it
        read fixtures other scenarios had planted there. One of them is the
        placeholder the product documents and requires,
        ``https://[user]:[password]@host``, and S10 duly reported the
        harness's own fixture as an authority the product had leaked.
        """
        root = pathlib.Path(tempfile.mkdtemp())
        other = root / "s13-config-readme-2-20260101T000000"
        other.mkdir(parents=True)
        other.joinpath("invocation.log").write_bytes(
            b'base_url = "https://[user]:[password]@example.invalid/v1"')
        mine = root / "R6"
        mine.mkdir(parents=True)
        mine.joinpath("stdout").write_bytes(b'{"ok": true}')
        with mock.patch.object(redaction.runs, "archive_root",
                               return_value=root):
            outcomes = _outcomes(_FakeRun(stdout=b'{"ok": true}'))
        self.assertEqual(Outcome.PASS, outcomes[redaction.ASSERTIONS[3]])

    def test_the_raw_credential_in_stderr_fails_the_first(self) -> None:
        run = _FakeRun(stderr=("error: %s rejected" % _SECRET).encode("utf-8"))
        self.assertEqual(Outcome.FAIL,
                         _outcomes(run)[redaction.ASSERTIONS[0]])

    def test_the_percent_encoded_credential_is_found_too(self) -> None:
        """The vault percent-encodes a value before it enters a URL, so in the
        error path the credential arrives encoded. A raw-only search does not
        find it, and reports clean over a real leak.
        """
        encoded = urllib.parse.quote(_SECRET, safe="")
        run = _FakeRun(stderr=("url=https://u:%s@host/v1" % encoded).encode("utf-8"))
        self.assertEqual(Outcome.FAIL,
                         _outcomes(run)[redaction.ASSERTIONS[0]])

    def test_an_unredacted_authority_fails_the_fourth_on_shape(self) -> None:
        """The fourth does not look for the value, and that is its value.

        The first three share one derived set, so an encoding nobody foresaw
        fools all three at once. This one asks whether an authority came out
        with its userinfo intact, which breaks regardless of how the credential
        is spelled inside -- and fires on a credential the harness never
        planted.
        """
        run = _FakeRun(stderr=b"error: https://someone:unplanted@host/v1 refused")
        outcomes = _outcomes(run)
        self.assertEqual(Outcome.PASS, outcomes[redaction.ASSERTIONS[0]])
        self.assertEqual(Outcome.FAIL, outcomes[redaction.ASSERTIONS[3]])

    def test_a_leak_in_stdout_is_caught_as_well_as_one_in_stderr(self) -> None:
        """Four channels, and the defect this scenario protects lived in prose
        on stderr rather than in a JSON field. Checking one channel would
        inspect precisely where the product never failed.
        """
        run = _FakeRun(stdout=("{\"url\": \"%s\"}" % _SECRET).encode("utf-8"))
        self.assertEqual(Outcome.FAIL,
                         _outcomes(run)[redaction.ASSERTIONS[1]])

    def test_the_scenario_searches_what_was_planted_not_what_it_mints(self) -> None:
        """A scenario that minted its own credential would search for a
        different secret than the one that went in, and report clean over a
        real leak -- the exact false green S10 exists to prevent.
        """
        other = mint_credential()
        run = _FakeRun(stderr=other.value.encode("utf-8"),
                       planted=(other,))
        self.assertEqual(Outcome.FAIL,
                         _outcomes(run)[redaction.ASSERTIONS[0]])

    def test_a_timed_out_run_is_the_runner_s_to_answer(self) -> None:
        """S10 does not classify a timeout, and that is the invariant.

        It used to carry its own ``timed_out`` branch, which production could
        never reach: a scenario that does not declare ``inspects_timeouts``
        has already been answered by the runner before its body runs. The
        branch was dead code shaped like a guard, and the test that exercised
        it asserted on a path nothing takes. What has to hold is the
        declaration.
        """
        self.assertFalse(DEFAULT_REGISTRY.get("S10").inspects_timeouts)

    def test_a_run_that_never_happened_cannot_test(self) -> None:
        self.assertEqual({Outcome.CANNOT_TEST}, set(_outcomes(None).values()))


if __name__ == "__main__":
    unittest.main()
