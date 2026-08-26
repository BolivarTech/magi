# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-26
"""Tests for the report: every finding, and every cause, reaches the reader."""

import unittest

from smoke.env import Growth
from smoke.outcome import Outcome
from smoke.report import Report
from smoke.runner import StampedFinding


def _finding(scenario: str, assertion: str, outcome: Outcome = Outcome.PASS,
             detail: str = "", run_id: str | None = None) -> StampedFinding:
    """Build one stamped finding.

    Args:
        scenario: The scenario id.
        assertion: The assertion text.
        outcome: What became of it.
        detail: The cause, when there is one.
        run_id: The shared run, when there is one.

    Returns:
        StampedFinding: The finding.
    """
    return StampedFinding(scenario=scenario, assertion=assertion,
                          outcome=outcome, detail=detail, run_id=run_id)


class ReportRenderingTests(unittest.TestCase):
    """What the harness concluded has to be readable without the source."""

    def test_every_finding_gets_a_line_in_the_order_given(self) -> None:
        report = Report([_finding("S1", "first", run_id="R1"),
                         _finding("S2", "second")])
        lines = [line for line in report.render().splitlines()
                 if line.startswith("[")]
        self.assertEqual(["[PASS] S1 run=R1 - first", "[PASS] S2 - second"],
                         lines)

    def test_the_cause_of_a_non_pass_reaches_the_output(self) -> None:
        """The whole reason this module exists. Before it, a FAIL printed its
        assertion and swallowed the detail, so the reader was told that
        something had broken and never what.
        """
        report = Report([_finding("S5", "the sandbox holds", Outcome.FAIL,
                                  detail="view accepted ../../etc/passwd",
                                  run_id="R1")])
        self.assertIn("view accepted ../../etc/passwd", report.render())

    def test_several_non_pass_sharing_a_cause_are_reported_once(self) -> None:
        """Three assertions refused for one reason is one thing that went
        wrong, not three. Repeating the cause per assertion buries it.
        """
        cause = "R6 exceeded its ceiling, so its output is truncated"
        report = Report([
            _finding("S10", "not in stdout", Outcome.CANNOT_TEST, cause, "R6"),
            _finding("S10", "not in the JSON", Outcome.CANNOT_TEST, cause,
                     "R6"),
            _finding("S10", "nor in the log", Outcome.CANNOT_TEST, cause,
                     "R6"),
        ])
        self.assertEqual(1, report.render().count(cause))

    def test_the_grouped_cause_still_names_every_assertion_it_covers(self) -> None:
        """Collapsing the cause must not collapse what it happened to."""
        cause = "R6 exceeded its ceiling"
        report = Report([
            _finding("S10", "not in stdout", Outcome.CANNOT_TEST, cause, "R6"),
            _finding("S10", "nor in the log", Outcome.CANNOT_TEST, cause,
                     "R6"),
        ])
        rendered = report.render()
        self.assertIn("not in stdout", rendered)
        self.assertIn("nor in the log", rendered)

    def test_two_different_causes_are_both_reported(self) -> None:
        """Grouping collapses repetition, never distinct causes."""
        report = Report([
            _finding("S5", "a", Outcome.FAIL, "the first cause", "R1"),
            _finding("S9", "b", Outcome.FAIL, "the second cause", "R2"),
        ])
        rendered = report.render()
        self.assertIn("the first cause", rendered)
        self.assertIn("the second cause", rendered)

    def test_the_same_text_from_two_runs_is_two_causes(self) -> None:
        """Two runs that failed the same way failed twice. Keying the group on
        the text alone would report one of them and hide the other run's id.
        """
        report = Report([
            _finding("S5", "a", Outcome.FAIL, "it timed out", "R1"),
            _finding("S9", "b", Outcome.FAIL, "it timed out", "R2"),
        ])
        rendered = report.render()
        self.assertIn("R1", rendered)
        self.assertIn("R2", rendered)
        self.assertEqual(2, rendered.count("it timed out"))

    def test_a_report_with_nothing_wrong_carries_no_causes_section(self) -> None:
        self.assertNotIn("causes:", Report([_finding("S1", "first")]).render())


class ReportCountingTests(unittest.TestCase):
    """The summary is the line most people read, so it counts exactly."""

    def test_fail_and_cannot_test_are_what_block_a_green_run(self) -> None:
        """Asserted on the outcome itself, which is what production reads.

        This used to go through ``Report.blocking()``, an accessor whose only
        caller was this test: ``exit_code_for`` reads ``outcome.blocks_gate``
        directly. A test is not a consumer, so the accessor went and the
        guarantee stayed -- pointed at the path that actually decides the
        exit code.
        """
        blocking = {outcome for outcome in Outcome if outcome.blocks_gate}
        self.assertEqual({Outcome.FAIL, Outcome.CANNOT_TEST}, blocking)

    def test_out_of_scope_counts_as_neither_passed_nor_not_passed(self) -> None:
        """It was never promised, so counting it either way misstates the run:
        as a pass it inflates coverage, as a failure it blocks a gate over
        something nobody undertook to do.
        """
        report = Report([
            _finding("S1", "a"),
            _finding("S2", "b", Outcome.FAIL, "broke"),
            _finding("S3", "c", Outcome.OUT_OF_SCOPE, "never promised"),
        ])
        self.assertIn("result: 1 passed, 1 not passed, 3 total",
                      report.render())

    def test_an_unmeasured_memory_count_is_not_rendered_as_zero(self) -> None:
        """None means NOT MEASURED. Printing it as 0 is the same lie the
        harness refuses everywhere else: a number nobody took.
        """
        report = Report([_finding("S1", "a")],
                        growth=Growth(db_bytes=2048, runs_bytes=4096,
                                      active_memories=None))
        rendered = report.render()
        self.assertIn("active memories not measured", rendered)
        self.assertNotIn("0 active memories", rendered)

    def test_a_measured_memory_count_is_rendered(self) -> None:
        report = Report([_finding("S1", "a")],
                        growth=Growth(db_bytes=2048, runs_bytes=4096,
                                      active_memories=7))
        self.assertIn("7 active memories", report.render())

    def test_no_growth_means_no_environment_line(self) -> None:
        self.assertNotIn("environment:", Report([_finding("S1", "a")]).render())


if __name__ == "__main__":
    unittest.main()
