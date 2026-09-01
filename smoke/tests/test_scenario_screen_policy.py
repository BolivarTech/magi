# Author: Julian Bolivar
# Version: 0.18.1
# Date: 2026-08-31
"""Unit tests for S24: which half a missing run id is allowed to silence.

S24 asserts two different things about one marker, and they do not need the
same evidence. "It is not on stderr" is answered by the run's own capture.
"It IS in the day's log" is answered by the shared, persistent log directory,
where a line only belongs to this run if the run published an id to bind the
search to. Deferring both when the id is absent hides a leak that was fully
observed -- which is the reverse of what ``CANNOT_TEST`` is for.

The half that does still depend on the id is the VACUITY check, and it is kept:
with the marker on neither surface, "the screen stayed quiet" is true of a
build whose memory subsystem never attached, and no reading of stderr alone can
tell those apart.
"""

import json
import unittest

from smoke.outcome import Outcome
from smoke.product import ProductOutput
from smoke.registry import DEFAULT_REGISTRY
from smoke.runs import RunResult
from smoke.scenarios import screen_policy  # noqa: F401 - import registers it
from smoke.tests import support

#: A run id shaped like the one REQ-L63 publishes.
_RUN_ID = "424242-deadbeefcafef00d"

#: A stderr stream carrying the leak S24's first assertion exists to catch.
_LEAKING = ("memory: 12 active, 0 archived, 0 %s~34 KB index)\n"
            % screen_policy.DIAGNOSTIC_MARKER).encode("utf-8")


def _result(stderr) -> RunResult:
    """Build R1's result with a given stderr stream.

    Args:
        stderr: What the run wrote to the error stream.

    Returns:
        RunResult: The real type, not a double.
    """
    output = ProductOutput(stdout=json.dumps({"ok": True}).encode(),
                           stderr=stderr, exit_code=0,
                           command=["magi-rs", "query"])
    return RunResult(run_id=screen_policy.RUN_ID, output=output,
                     duration_s=1.0, timed_out=False, planted=())


def _outcomes(run) -> dict[str, Outcome]:
    """Run S24 and index what it concluded by assertion text.

    Args:
        run: The result to hand it, or None.

    Returns:
        dict[str, Outcome]: What each assertion concluded.
    """
    findings = list(DEFAULT_REGISTRY.get("S24").func(run))
    return {finding.assertion: finding.outcome for finding in findings}


class MissingRunIdTests(unittest.TestCase):
    """Without an id, only the half that needs one may be silenced."""

    def setUp(self) -> None:
        support.install_fake_runs(self)

    def test_a_leak_is_reported_even_with_no_run_id(self) -> None:
        """The marker is on stderr, which is the whole of that assertion's
        evidence. Correlating a log line would add nothing to it, so an
        absent run id must not turn a FAIL into a deferral.
        """
        outcomes = _outcomes(_result(_LEAKING))
        self.assertEqual(Outcome.FAIL,
                         outcomes[screen_policy.ASSERTIONS[0]])
        self.assertEqual(Outcome.CANNOT_TEST,
                         outcomes[screen_policy.ASSERTIONS[1]])

    def test_a_quiet_screen_with_no_run_id_still_defers(self) -> None:
        """The vacuity check is the half that DOES need the id.

        With the marker on neither surface, a compliant build and one whose
        memory subsystem never attached look identical from stderr, and the
        log is what tells them apart. Reporting PASS here would be the
        vacuous green the scenario's own docstring refuses.
        """
        outcomes = _outcomes(_result(b"nothing to see\n"))
        self.assertEqual(Outcome.CANNOT_TEST,
                         outcomes[screen_policy.ASSERTIONS[0]])
        self.assertEqual(Outcome.CANNOT_TEST,
                         outcomes[screen_policy.ASSERTIONS[1]])

    def test_every_assertion_is_still_reported(self) -> None:
        """A scenario answers all of what it declared or the reconciliation
        is measuring something other than what ran.
        """
        for stderr in (_LEAKING, b"nothing to see\n"):
            with self.subTest(stderr=stderr):
                self.assertEqual(set(screen_policy.ASSERTIONS),
                                 set(_outcomes(_result(stderr))))


class RunIdPresentTests(unittest.TestCase):
    """With an id, both halves are judged as they always were."""

    def setUp(self) -> None:
        support.install_fake_runs(self)

    def test_a_leak_with_a_run_id_still_fails_the_first(self) -> None:
        stderr = ("run: %s\n" % _RUN_ID).encode("utf-8") + _LEAKING
        outcomes = _outcomes(_result(stderr))
        self.assertEqual(Outcome.FAIL,
                         outcomes[screen_policy.ASSERTIONS[0]])


if __name__ == "__main__":
    unittest.main()
