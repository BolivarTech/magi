# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-26
"""Tests for the certificate: who may emit it, and what it must say.

A certificate that is wrong is the worst defect this harness can produce,
because it is the artifact somebody trusts a year later without re-running
anything.
"""

import pathlib
import tempfile
import unittest

from smoke.certificate import (Certificate, OUT_OF_SCOPE_DECLARATION,
                               RoundCounter, may_certify)
from smoke.env import Growth
from smoke.outcome import Outcome
from smoke.registry import DECLARED_SCENARIO_COUNT, DEFAULT_REGISTRY
from smoke.runner import StampedFinding
from smoke import scenarios  # noqa: F401 - importing registers them


def _finding(scenario: str, assertion: str, outcome: Outcome = Outcome.PASS,
             run_id: str | None = None) -> StampedFinding:
    """Build one stamped finding.

    Args:
        scenario: The scenario id.
        assertion: The assertion text.
        outcome: What became of it.
        run_id: The shared run, when there is one.

    Returns:
        StampedFinding: The finding.
    """
    return StampedFinding(scenario=scenario, assertion=assertion,
                          outcome=outcome, detail="", run_id=run_id)


def _clean() -> list[StampedFinding]:
    """Findings with nothing blocking.

    Returns:
        list[StampedFinding]: Two passes.
    """
    return [_finding("S1", "first", run_id="R1"), _finding("S2", "second")]


class MayCertifyTests(unittest.TestCase):
    """Three conditions, all necessary: REQ-S23, REQ-S35 and D-17."""

    def test_smoke_1_never_certifies(self) -> None:
        self.assertFalse(may_certify(smoke_2=False, profile=None,
                                     findings=_clean(),
                                     evaluated=DECLARED_SCENARIO_COUNT))

    def test_a_profile_never_certifies_however_green(self) -> None:
        """REQ-S35. A cheap-profile run certifying under a document that says
        "product defaults" is false at the one point its whole value rests on.
        """
        self.assertFalse(may_certify(smoke_2=True, profile=object(),
                                     findings=_clean(),
                                     evaluated=DECLARED_SCENARIO_COUNT))

    def test_a_cannot_test_blocks_the_certificate(self) -> None:
        """D-17: what was promised and could not run is not a green."""
        findings = _clean() + [_finding("S3", "third", Outcome.CANNOT_TEST)]
        self.assertFalse(may_certify(smoke_2=True, profile=None,
                                     findings=findings,
                                     evaluated=DECLARED_SCENARIO_COUNT))

    def test_a_fail_blocks_the_certificate(self) -> None:
        findings = _clean() + [_finding("S3", "third", Outcome.FAIL)]
        self.assertFalse(may_certify(smoke_2=True, profile=None,
                                     findings=findings,
                                     evaluated=DECLARED_SCENARIO_COUNT))

    def test_out_of_scope_alone_does_not_block(self) -> None:
        findings = _clean() + [_finding("S3", "third", Outcome.OUT_OF_SCOPE)]
        self.assertTrue(may_certify(smoke_2=True, profile=None,
                                    findings=findings,
                                    evaluated=DECLARED_SCENARIO_COUNT))

    def test_a_clean_smoke_2_with_no_profile_certifies(self) -> None:
        self.assertTrue(may_certify(smoke_2=True, profile=None,
                                    findings=_clean(),
                                    evaluated=DECLARED_SCENARIO_COUNT))


    def test_a_run_that_evaluated_fewer_scenarios_does_not_certify(self) -> None:
        """The headline is "N of N", and nothing checked the first N.

        Drop a module from ``scenarios/__init__.py`` and reconciliation still
        passes -- it compares what was invoked against what reported, and both
        shrink together. The certificate would then read "18 of 19" and be
        emitted anyway, which is honest arithmetic over silently reduced
        coverage.
        """
        self.assertFalse(may_certify(smoke_2=True, profile=None,
                                     findings=_clean(), evaluated=18))

    def test_a_full_run_certifies(self) -> None:
        self.assertTrue(may_certify(smoke_2=True, profile=None,
                                    findings=_clean(),
                                    evaluated=DECLARED_SCENARIO_COUNT))


class CertificateRenderingTests(unittest.TestCase):
    """Everything a reader needs to know what this document covers."""

    def _certificate(self, findings=None) -> Certificate:
        """Build a certificate over *findings*.

        Args:
            findings: What to certify; the clean pair when omitted.

        Returns:
            Certificate: The document.
        """
        return Certificate(
            version="0.16.0",
            commit="abc1234",
            date="2026-08-26",
            binary_sha256="a" * 64,
            run_count=8,
            duration_s=912.4,
            rounds=3,
            evaluated=19,
            growth=Growth(db_bytes=2048, runs_bytes=4096, active_memories=7),
            findings=tuple(_clean() if findings is None else findings),
        )

    def test_it_carries_the_version_commit_and_binary_identity(self) -> None:
        rendered = self._certificate().render()
        self.assertIn("- version: 0.16.0", rendered)
        self.assertIn("- commit: abc1234", rendered)
        self.assertIn("- binary: sha256 %s" % ("a" * 64), rendered)

    def test_it_declares_how_many_rounds_it_took(self) -> None:
        """The cost series. A release that needed three is information about
        that release, and ``git log -p`` over the fixed path turns the single
        number into the series that reveals a harness getting dearer.
        """
        self.assertIn("- rounds needed: 3", self._certificate().render())

    def test_it_declares_the_profile_it_was_produced_under(self) -> None:
        self.assertIn("- profile: product defaults (no --profile)",
                      self._certificate().render())

    def test_it_declares_the_scope_against_the_declared_count(self) -> None:
        self.assertIn("- scope: 19 of %d scenarios evaluated"
                      % DECLARED_SCENARIO_COUNT,
                      self._certificate().render())

    def test_it_declares_what_it_does_not_cover(self) -> None:
        """Without this a green reads as coverage of the whole contract, which
        is the lie this harness exists not to tell.
        """
        rendered = self._certificate().render()
        self.assertIn("contract coverage:", rendered)
        self.assertIn(OUT_OF_SCOPE_DECLARATION, rendered)

    def test_it_reports_what_the_run_really_cost(self) -> None:
        self.assertIn("- real cost: 8 backend run(s) in 912s",
                      self._certificate().render())

    def test_the_assertion_lines_keep_the_order_they_were_given(self) -> None:
        """The runner already orders them deterministically, so the document's
        diff never moves on its own between two runs that concluded the same.
        """
        lines = [line for line in self._certificate().render().splitlines()
                 if line.startswith("[")]
        self.assertEqual(["[PASS] S1 run=R1 - first", "[PASS] S2 - second"],
                         lines[:2])

    def test_writing_it_creates_the_directory_and_the_file(self) -> None:
        target = (pathlib.Path(tempfile.mkdtemp()) / "docs" / "test"
                  / "smoke-certificate.md")
        certificate = self._certificate()
        certificate.write(target)
        self.assertEqual(certificate.render(),
                         target.read_text(encoding="utf-8"))

    def test_writing_it_replaces_what_was_there(self) -> None:
        """A fixed name, replacing the previous one: git's history IS the
        archive, and a versioned filename would force a reader to know what
        the old one was called.
        """
        target = pathlib.Path(tempfile.mkdtemp()) / "smoke-certificate.md"
        target.write_text("an older certificate", encoding="utf-8")
        self._certificate().write(target)
        self.assertNotIn("an older certificate",
                         target.read_text(encoding="utf-8"))


class DeclaredScenarioCountTests(unittest.TestCase):
    """The headline number is published, so it is asserted where it is set."""

    def test_the_registry_holds_exactly_the_declared_number(self) -> None:
        """The certificate publishes "N of N". Deleting a scenario module
        would make it say "18 of 18" -- honest arithmetic over silently
        reduced coverage. test_registry's import guard catches an UNIMPORTED
        module, never a deleted one.
        """
        self.assertEqual(DECLARED_SCENARIO_COUNT,
                         len(DEFAULT_REGISTRY.registered_ids()))


class RoundCounterTests(unittest.TestCase):
    """How many attempts this release needed, counted where it survives."""

    def setUp(self) -> None:
        self.path = pathlib.Path(tempfile.mkdtemp()) / "rounds"

    def test_the_first_round_is_one(self) -> None:
        self.assertEqual(1, RoundCounter(self.path).bump())

    def test_it_counts_across_processes(self) -> None:
        """Each iteration of smoke 2 is a separate invocation, so a counter
        held in memory would report 1 forever.
        """
        RoundCounter(self.path).bump()
        self.assertEqual(2, RoundCounter(self.path).bump())

    def test_it_starts_again_once_a_certificate_is_emitted(self) -> None:
        counter = RoundCounter(self.path)
        counter.bump()
        counter.bump()
        counter.reset()
        self.assertEqual(1, RoundCounter(self.path).bump())

    def test_an_unreadable_counter_starts_over_rather_than_failing(self) -> None:
        """A corrupt counter is one wrong line in the document. Refusing to
        run over it would cost the whole expensive run.
        """
        self.path.write_text("not a number", encoding="utf-8")
        self.assertEqual(1, RoundCounter(self.path).bump())


if __name__ == "__main__":
    unittest.main()
