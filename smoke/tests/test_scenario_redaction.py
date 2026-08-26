# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Unit tests for the S10 scenario's own shape."""

import unittest
import urllib.parse

from smoke.outcome import Outcome
from smoke.registry import DEFAULT_REGISTRY
from smoke.scenarios import redaction  # noqa: F401 - import registers it
from smoke.secrets import PlantedSecret, mint_credential

_SECRET = "sup3rs3cr3t-value"
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
                "every authority emitted anywhere is redacted",
            ],
            list(redaction.ASSERTIONS),
        )


class RedactionBodyTests(unittest.TestCase):
    """What S10 concludes about output that leaks, and output that does not."""

    def test_a_clean_capture_passes_every_assertion(self) -> None:
        run = _FakeRun(stdout=b'{"ok": true}',
                       stderr=b"error: request to https://***@host/v1 failed")
        self.assertEqual({Outcome.PASS}, set(_outcomes(run).values()))

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

    def test_a_timed_out_run_cannot_test_rather_than_pass(self) -> None:
        """Finding no leak in output that was cut short proves nothing."""
        run = _FakeRun(stdout=b"", stderr=b"", timed_out=True)
        self.assertEqual({Outcome.CANNOT_TEST}, set(_outcomes(run).values()))

    def test_a_run_that_never_happened_cannot_test(self) -> None:
        self.assertEqual({Outcome.CANNOT_TEST}, set(_outcomes(None).values()))


if __name__ == "__main__":
    unittest.main()
