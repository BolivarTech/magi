# Author: Julian Bolivar
# Version: 0.24.3
# Date: 2026-10-25
"""Unit tests for S47-S50 (code-review against the release binary)."""

import json
import unittest

from smoke.outcome import Outcome
from smoke.product import ProductOutput
from smoke.registry import DEFAULT_REGISTRY
from smoke.runs import RunResult
from smoke.scenarios import code_review  # noqa: F401 - import registers it

def _result(run_id: str, exit_code: int = 0, document=None, stderr: bytes = b"") -> RunResult:
    """A real ``RunResult`` for *run_id* whose stdout is *document* serialised (``None`` → empty stdout)."""
    body = b"" if document is None else json.dumps(document).encode()
    return RunResult(run_id=run_id,
                     output=ProductOutput(stdout=body, stderr=stderr, exit_code=exit_code, command=["magi-rs"]),
                     duration_s=1.0, timed_out=False, planted=())


def _outcomes(scenario_id: str, result) -> dict[str, Outcome]:
    return {f.assertion: f.outcome for f in DEFAULT_REGISTRY.get(scenario_id).func(result)}


def _review(batches: int = 3, findings=None, discarded=None) -> dict:
    # [CP2 MS2 S2 loop 1] the reviewed set is code_review.S47_FILES, the scenario's own generated input.
    # [CP2 MS2 S2 loop 2] the main findings live in code_review.findings (summary_json), not in consult.
    # [CP2 MS2 S2b loop 6] code_review.discarded, RV-S5's other half, each entry with its cause.
    # [CP2 MS2 S2b loop 8] unmeasured_models: RV04's window; with one unmeasured model it is the 32 768 default.
    return {"consult": {"consensus": {"verdict": "approve"}},
            "code_review": {"batches_planned": batches, "batches_completed": batches,
                            "unmeasured_models": ["kimi-k2.6:cloud"],
                            "findings": [{"file": "gen/a.rs", "line": 1}] if findings is None else findings,
                            "discarded": [{"cause": "file not reviewed"}] if discarded is None else discarded}}


class CodeReviewScenarioTests(unittest.TestCase):

    def test_registration(self) -> None:
        self.assertTrue(DEFAULT_REGISTRY.get("S47").needs_backend)
        for sid in ("S48", "S49", "S50"):
            self.assertFalse(DEFAULT_REGISTRY.get(sid).needs_backend, sid)

    def test_s47_passes_on_one_report_over_three_batches_with_findings_inside(self) -> None:
        r = _result("R12", 0, _review(3, [{"file": "gen/b.rs", "line": 4}]))
        self.assertEqual({Outcome.PASS}, set(_outcomes("S47", r).values()))

    def test_s47_fails_below_three_batches(self) -> None:
        r = _result("R12", 0, _review(2))
        self.assertEqual(Outcome.FAIL, _outcomes("S47", r)[code_review.S47_ASSERTIONS[2]])

    def test_s47_below_three_batches_with_every_model_measured_cannot_test(self) -> None:
        # [CP2 MS2 S2b loop 8] the generator is sized for the 32 768-token unmeasured window (RV04); a pool that
        # measured a larger window plans fewer batches for an environmental reason: CANNOT_TEST, never FAIL.
        doc = _review(2)
        doc["code_review"]["unmeasured_models"] = []
        self.assertEqual(Outcome.CANNOT_TEST, _outcomes("S47", _result("R12", 0, doc))[code_review.S47_ASSERTIONS[2]])

    def test_s47_fails_on_a_finding_outside_the_reviewed_files(self) -> None:
        r = _result("R12", 0, _review(3, [{"file": "elsewhere.rs", "line": 1}]))
        self.assertEqual(Outcome.FAIL, _outcomes("S47", r)[code_review.S47_ASSERTIONS[4]])

    def test_s47_skips_findings_without_a_file_in_the_location_check(self) -> None:
        # [CP2 MS2 S2b loop 9] an unlocated finding is not an invented location (RV09).
        r = _result("R12", 0, _review(3, [{"file": None, "line": None}, {"file": "gen/a.rs", "line": 1}]))
        self.assertEqual(Outcome.PASS, _outcomes("S47", r)[code_review.S47_ASSERTIONS[4]])
        r = _result("R12", 0, _review(3, [{"file": None, "line": None}]))
        self.assertEqual(Outcome.CANNOT_TEST, _outcomes("S47", r)[code_review.S47_ASSERTIONS[4]])

    def test_s48_passes_on_a_misuse_exit_naming_the_path(self) -> None:
        r = _result("R13", 2, None, b"error: path is outside the workspace: ../fuera\n")
        self.assertEqual({Outcome.PASS}, set(_outcomes("S48", r).values()))

    def test_s49_passes_on_nothing_to_review(self) -> None:
        r = _result("R14", 2, None, b"error: nothing to review: no reviewable file in the input\n")
        self.assertEqual({Outcome.PASS}, set(_outcomes("S49", r).values()))

    def test_s50_passes_when_the_floor_and_a_minimum_timeout_are_named(self) -> None:
        r = _result("R15", 2, None, b"error: 3 batches leave 10 s per batch, less than the 783 s a batch needs at the 15 s floor (per-batch ceiling 0 s); pass --timeout 2349 or more\n")
        self.assertEqual({Outcome.PASS}, set(_outcomes("S50", r).values()))

    def test_s50_fails_without_a_minimum(self) -> None:
        r = _result("R15", 2, None, b"error: below the 15 s floor\n")
        self.assertEqual(Outcome.FAIL, _outcomes("S50", r)[code_review.S50_ASSERTIONS[1]])

    def test_s47_without_the_findings_key_fails_never_an_empty_pass(self) -> None:
        doc = _review(3)
        del doc["code_review"]["findings"]
        self.assertEqual(Outcome.FAIL, _outcomes("S47", _result("R12", 0, doc))[code_review.S47_ASSERTIONS[4]])

    def test_s47_without_the_discarded_key_fails(self) -> None:
        # [CP2 MS2 S2b loop 10] symmetric to the findings key: an absent key is a shape drift, never "nothing discarded".
        doc = _review(3)
        del doc["code_review"]["discarded"]
        self.assertEqual(Outcome.FAIL, _outcomes("S47", _result("R12", 0, doc))[code_review.S47_ASSERTIONS[5]])

    def test_s47_without_the_unmeasured_key_treats_the_pool_as_unmeasured(self) -> None:
        # [CP2 MS2 S2b loop 10] an absent key never softens the batch assertion to CANNOT_TEST.
        doc = _review(2)
        del doc["code_review"]["unmeasured_models"]
        self.assertEqual(Outcome.FAIL, _outcomes("S47", _result("R12", 0, doc))[code_review.S47_ASSERTIONS[2]])

    def test_s47_with_no_findings_cannot_test_the_location_property(self) -> None:
        r = _result("R12", 0, _review(3, []))
        self.assertEqual(Outcome.CANNOT_TEST, _outcomes("S47", r)[code_review.S47_ASSERTIONS[4]])

    def test_s47_fails_when_a_discarded_finding_has_no_cause(self) -> None:
        r = _result("R12", 0, _review(3, None, [{"cause": "line out of range"}, {"cause": ""}]))
        self.assertEqual(Outcome.FAIL, _outcomes("S47", r)[code_review.S47_ASSERTIONS[5]])

    def test_s47_with_nothing_discarded_cannot_test_the_cause_property(self) -> None:
        r = _result("R12", 0, _review(3, None, []))
        self.assertEqual(Outcome.CANNOT_TEST, _outcomes("S47", r)[code_review.S47_ASSERTIONS[5]])

    def test_s47_fails_on_a_nonzero_exit_unparsed_stdout_or_no_consensus(self) -> None:
        # [CP2 MS2 S2b loop 7] the negative half of each guarded S47 property.
        self.assertEqual(Outcome.FAIL, _outcomes("S47", _result("R12", 1, _review(3)))[code_review.S47_ASSERTIONS[0]])
        self.assertEqual(Outcome.FAIL, _outcomes("S47", _result("R12", 0, None))[code_review.S47_ASSERTIONS[1]])
        doc = _review(3)
        del doc["consult"]["consensus"]
        self.assertEqual(Outcome.FAIL, _outcomes("S47", _result("R12", 0, doc))[code_review.S47_ASSERTIONS[3]])

    def test_s48_and_s49_fail_on_a_wrong_exit_or_a_silent_stderr(self) -> None:
        for sid, run, asserts in (("S48", "R13", code_review.S48_ASSERTIONS), ("S49", "R14", code_review.S49_ASSERTIONS)):
            self.assertEqual(Outcome.FAIL, _outcomes(sid, _result(run, 0, None, b""))[asserts[0]], sid)
            self.assertEqual(Outcome.FAIL, _outcomes(sid, _result(run, 2, None, b"error: something else\n"))[asserts[1]], sid)
