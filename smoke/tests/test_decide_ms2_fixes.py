# Author: Julian Bolivar
# Version: 0.23.0
# Date: 2026-10-11
"""Regression tests for the three harness defects SMOKE #1 found in D1, D11 and D13.

The product was correct in all three; the scenarios judged more than the
property they name. Each test drives the scenario (or its fixture) with the
capture that exposed the defect.
"""

import json
import unittest
from unittest import mock

from smoke import runs
from smoke.outcome import Outcome
from smoke.product import ProductOutput
from smoke.scenarios import decide

_ERROR_ENVELOPE = json.dumps({
    "schema_version": 1,
    "chain": [{"model": "nimble", "outcome": "low_confidence"}],
    "error": {"kind": "decision_unavailable", "message": "chain exhausted"},
})
#: How the agent loop frames a tool that ran and returned an error.
_EXECUTION_FAILED = "Execution failed: " + _ERROR_ENVELOPE
_TIER_DENIAL = ("Tool 'decide' denied: not authorized in the current "
                "authorization tier")


def _query_attempt(calls) -> runs.Attempt:
    """An attempt whose stdout is a query envelope carrying ``tool_calls``."""
    stdout = json.dumps({"tool_calls": calls}).encode("utf-8")
    return runs.Attempt(
        output=ProductOutput(stdout=stdout, stderr=b"", exit_code=0,
                             command=["magi-rs"]),
        failure="")


def _call(result, ok) -> dict:
    """One ``decide`` entry of ``tool_calls``."""
    return {"name": "decide", "ok": ok, "result": result,
            "input": {"state": "<omitted>",
                      "questions": {"kind": {"type": "choice"}}}}


class D1FixtureTests(unittest.TestCase):
    """D1's score question has an answer the state fixes."""

    def test_the_score_question_is_not_the_ambiguous_urgency_one(self) -> None:
        score = decide.THREE_TYPES_REQUEST["questions"]["urgency"]
        self.assertEqual("score", score["type"])
        self.assertNotIn("Urgent", score["criteria"])
        self.assertNotIn("urgent", score["instructions"].lower())
        self.assertEqual(decide.LEVEL_COUNT, len(score["criteria"]))

    def test_the_three_types_are_still_one_of_each(self) -> None:
        kinds = sorted(q["type"] for q in
                       decide.THREE_TYPES_REQUEST["questions"].values())
        self.assertEqual(["choice", "noul", "score"], kinds)


class D11ApprovalTests(unittest.TestCase):
    """D11 judges the tier's approval, not whether the call succeeded."""

    def _finding(self, calls):
        with mock.patch.object(decide, "_query_in_environment",
                               return_value=_query_attempt(calls)):
            return list(decide.the_agent_calls_decide_in_the_default_tier(None))[0]

    def test_an_executed_call_whose_result_is_an_error_passes(self) -> None:
        finding = self._finding([_call(_EXECUTION_FAILED, False)])
        self.assertEqual(Outcome.PASS, finding.outcome)

    def test_a_tier_denial_fails(self) -> None:
        finding = self._finding([_call(_TIER_DENIAL, False)])
        self.assertEqual(Outcome.FAIL, finding.outcome)

    def test_a_successful_call_passes(self) -> None:
        finding = self._finding([_call('{"schema_version": 1}', True)])
        self.assertEqual(Outcome.PASS, finding.outcome)


class D13KeptResultTests(unittest.TestCase):
    """D13 keeps any result envelope the agent received, success or error."""

    def _second(self, result):
        questions = {"kind": {"type": "choice"}}
        return decide._questions_and_result_finding(
            _call(result, True), questions)

    def test_an_error_envelope_counts_as_a_kept_result(self) -> None:
        self.assertEqual(Outcome.PASS, self._second(_ERROR_ENVELOPE).outcome)

    def test_an_error_envelope_behind_the_failure_prefix_counts(self) -> None:
        self.assertEqual(Outcome.PASS, self._second(_EXECUTION_FAILED).outcome)

    def test_a_success_envelope_counts_as_a_kept_result(self) -> None:
        self.assertEqual(
            Outcome.PASS, self._second('{"schema_version": 1}').outcome)

    def test_an_empty_result_fails(self) -> None:
        self.assertEqual(Outcome.FAIL, self._second("").outcome)

    def test_a_result_that_is_not_an_envelope_fails(self) -> None:
        self.assertEqual(Outcome.FAIL, self._second("boom").outcome)


if __name__ == "__main__":
    unittest.main()
