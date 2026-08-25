# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Unit tests for the S1 scenario's own shape and its exact-key guard."""

import copy
import json
import unittest

from smoke.outcome import Outcome
from smoke.product import ProductOutput
from smoke.registry import DEFAULT_REGISTRY
from smoke.runs import RunResult
from smoke.scenarios import contract  # noqa: F401 - import registers it

#: A capture that satisfies the whole contract, shaped exactly like
#: ``tests/golden/headless_output_v1.json``. Every body test starts from a copy
#: of this and breaks ONE thing, so a red names the break rather than the fixture.
_GOLDEN = {
    "applied_caps": {
        "ceiling_above_sanity": False,
        "ceiling_floored": False,
        "floor_activation_threshold_secs": 114,
        "max_rotations_effective": 2,
        "max_tool_calls": 15,
        "max_tool_calls_clamped": False,
        "operation_budget_secs": 149,
        "system_override_applied": False,
        "timeout_secs": None,
    },
    "consult": None,
    "error": None,
    "model": "kimi-k2.6:cloud",
    "provider": "ollama",
    "response": "Hello from magi.",
    "schema_version": 1,
    "stop_reason": "done",
    "timings": {"per_turn_ms": [600], "total_ms": 1234, "ttfb_ms": 200},
    "tool_calls": [],
    "transcript": [],
    "usage": {"input_tokens": 100, "output_tokens": 50},
}


def _result(document=None, exit_code=0, stdout=None) -> RunResult:
    """Build an R1 result carrying *document* as the product's JSON output.

    Args:
        document: The object to serialise as stdout; ``None`` uses the golden.
        exit_code: What the product exited with.
        stdout: Raw bytes to send instead of serialising *document*, for the
            test that hands the scenario something that is not JSON at all.

    Returns:
        RunResult: The real type, not a double -- a scenario that reads a field
        the type does not carry must fail here rather than pass against a
        laxer stand-in.
    """
    body = (stdout if stdout is not None
            else json.dumps(_GOLDEN if document is None else document).encode())
    return RunResult(
        run_id="R1",
        output=ProductOutput(stdout=body, stderr=b"", exit_code=exit_code,
                             command=["magi-rs", "query"]),
        duration_s=1.0,
        timed_out=False,
        planted=(),
    )


def _outcomes(result) -> dict[str, Outcome]:
    """Run S1 over *result*, indexed by assertion text.

    Args:
        result: What to hand the scenario as its run.

    Returns:
        dict[str, Outcome]: What each assertion concluded.
    """
    findings = list(DEFAULT_REGISTRY.get("S1").func(result))
    return {finding.assertion: finding.outcome for finding in findings}


class ContractScenarioShapeTests(unittest.TestCase):
    """S1 is registered against R1 and declares the spec's four assertions."""

    def test_s1_is_registered_with_its_declared_run(self) -> None:
        entry = DEFAULT_REGISTRY.get("S1")
        self.assertEqual("R1", entry.run)
        self.assertTrue(entry.needs_backend)

    def test_the_assertion_texts_are_the_spec_texts(self) -> None:
        self.assertEqual(
            [
                "exit 0 and stdout parses as JSON",
                "schema_version is 1",
                "every top-level contract key is present",
                "every nested key of applied_caps is present",
            ],
            list(contract.ASSERTIONS),
        )


class ContractScenarioBodyTests(unittest.TestCase):
    """The four assertions, and the silent defect the fourth exists for."""

    def test_the_golden_contract_passes_every_assertion(self) -> None:
        self.assertEqual({Outcome.PASS}, set(_outcomes(_result()).values()))

    def test_a_flatten_collision_that_swallowed_a_nested_key_fails(self) -> None:
        """The defect assertion 4 exists for.

        Two fields sharing a name across the ``#[serde(flatten)]`` boundary
        make ``serde_json`` write the key twice into one map: the second write
        wins and the first value vanishes with no error and still-valid JSON.
        A subset check cannot see it, because nothing INVALID appears -- one
        expected key is simply gone.
        """
        broken = copy.deepcopy(_GOLDEN)
        del broken["applied_caps"]["operation_budget_secs"]
        outcomes = _outcomes(_result(broken))
        self.assertEqual(Outcome.FAIL, outcomes[contract.ASSERTIONS[3]])
        self.assertEqual(Outcome.PASS, outcomes[contract.ASSERTIONS[2]])

    def test_an_unreviewed_nested_key_fails(self) -> None:
        """Exactness runs both ways inside ``applied_caps``.

        The flattened pair is where a name collision hides, so a key nobody
        declared arriving there is the same class of event as one vanishing:
        the two structs are in different files and neither mentions the
        other's names.
        """
        widened = copy.deepcopy(_GOLDEN)
        widened["applied_caps"]["retry_disabled"] = False
        self.assertEqual(Outcome.FAIL,
                         _outcomes(_result(widened))[contract.ASSERTIONS[3]])

    def test_a_missing_top_level_key_fails_the_third(self) -> None:
        broken = copy.deepcopy(_GOLDEN)
        del broken["usage"]
        self.assertEqual(Outcome.FAIL,
                         _outcomes(_result(broken))[contract.ASSERTIONS[2]])

    def test_an_added_top_level_key_still_passes_the_third(self) -> None:
        """Top level is a SUPERSET check, and the asymmetry is deliberate.

        REQ-H14 declares a new top-level key additive, so a consumer must
        tolerate one. The exactness lives one level down, where the flatten
        makes an addition capable of deleting a sibling.
        """
        widened = copy.deepcopy(_GOLDEN)
        widened["a_future_key"] = 1
        self.assertEqual(Outcome.PASS,
                         _outcomes(_result(widened))[contract.ASSERTIONS[2]])

    def test_a_wrong_schema_version_fails_the_second(self) -> None:
        bumped = copy.deepcopy(_GOLDEN)
        bumped["schema_version"] = 2
        self.assertEqual(Outcome.FAIL,
                         _outcomes(_result(bumped))[contract.ASSERTIONS[1]])

    def test_a_non_zero_exit_fails_the_first(self) -> None:
        self.assertEqual(Outcome.FAIL,
                         _outcomes(_result(exit_code=1))[contract.ASSERTIONS[0]])

    def test_output_that_is_not_json_reports_all_four(self) -> None:
        """The parse failure is assertion 1's own subject, so S1 catches it.

        Letting ``ProductOutputError`` escape would collapse four assertions
        into the runner's single "could not be interpreted" line, and three of
        them would vanish from the report with nobody noticing they were never
        evaluated.
        """
        outcomes = _outcomes(_result(stdout=b"not json at all"))
        self.assertEqual(list(contract.ASSERTIONS), list(outcomes))
        self.assertEqual(Outcome.FAIL, outcomes[contract.ASSERTIONS[0]])
        self.assertEqual(
            {Outcome.CANNOT_TEST},
            {outcomes[text] for text in contract.ASSERTIONS[1:]},
        )

    def test_a_missing_run_reports_all_four_as_cannot_test(self) -> None:
        outcomes = _outcomes(None)
        self.assertEqual(list(contract.ASSERTIONS), list(outcomes))
        self.assertEqual({Outcome.CANNOT_TEST}, set(outcomes.values()))


if __name__ == "__main__":
    unittest.main()
