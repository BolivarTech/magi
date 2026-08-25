# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Tests for the report's data contract."""

import dataclasses
import unittest

from smoke.outcome import Finding, Outcome


class OutcomeTests(unittest.TestCase):
    """The four outcomes, and what each one means for the gate."""

    def test_exactly_four_outcomes_exist(self) -> None:
        self.assertEqual(
            {"PASS", "FAIL", "CANNOT_TEST", "OUT_OF_SCOPE"},
            {member.value for member in Outcome},
        )

    def test_cannot_test_blocks_the_gate_and_out_of_scope_does_not(self) -> None:
        self.assertTrue(Outcome.CANNOT_TEST.blocks_gate)
        self.assertTrue(Outcome.FAIL.blocks_gate)
        self.assertFalse(Outcome.OUT_OF_SCOPE.blocks_gate)
        self.assertFalse(Outcome.PASS.blocks_gate)


class FindingTests(unittest.TestCase):
    """A Finding never carries its own scenario id: the runner stamps it."""

    def test_finding_has_no_scenario_field(self) -> None:
        names = {field.name for field in dataclasses.fields(Finding)}
        self.assertNotIn("scenario", names)
        self.assertEqual({"assertion", "outcome", "detail", "run_id"}, names)

    def test_finding_is_frozen(self) -> None:
        finding = Finding("it holds", Outcome.PASS, "", None)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            finding.detail = "mutated"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
