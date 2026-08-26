# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Unit tests for S19: the assertion has to look INSIDE the tool result."""

import copy
import json
import unittest

from smoke.outcome import Outcome
from smoke.product import ProductOutput
from smoke.registry import DEFAULT_REGISTRY
from smoke.runs import RunResult
from smoke.scenarios import agent_path  # noqa: F401 - import registers it

#: What the consult tool hands back to the agent: the envelope WITHOUT the two
#: structured keys, which is the whole of REQ-EA02.
_TOOL_RESULT = {
    "report": "the trio agreed",
    "degraded": False,
    "mode": "analysis",
    "mode_source": "agent-chosen",
    "extraction_failures": {},
    "input_size": {"estimated_tokens": 900, "warn_threshold": 100000,
                   "exceeded": False},
    "report_truncated": "none",
    "endpoint_divergence": False,
    "timeout_below_formula": False,
    "failed_agents": {},
}


def _record(result, name=None):
    """Build one tool-call record.

    Args:
        result: What the tool returned, serialised if it is not already text.
        name: The tool's name; defaults to the consult tool's.

    Returns:
        dict: The record.
    """
    return {
        "name": agent_path.CONSULT_TOOL_NAME if name is None else name,
        "input": {"prompt": "should we adopt a logging framework?"},
        "result": result if isinstance(result, str) else json.dumps(result),
        "ms": 4200,
        "ok": True,
    }


def _result(records, top_level_leak=False) -> RunResult:
    """Build R5's result.

    Args:
        records: The ``tool_calls`` array.
        top_level_leak: Whether to put the two structured keys at the TOP level
            of the response, which is where a scenario looking in the wrong
            place would find them.

    Returns:
        RunResult: The real type, not a double.
    """
    document = {"schema_version": 1, "tool_calls": records, "consult": None,
                "error": None}
    if top_level_leak:
        document["agents"] = [{"agent": "melchior"}]
        document["consensus"] = {"consensus_verdict": "approve"}
    return RunResult(
        run_id="R5",
        output=ProductOutput(stdout=json.dumps(document).encode(), stderr=b"",
                             exit_code=0, command=["magi-rs", "query"]),
        duration_s=1.0, timed_out=False, planted=())


def _outcomes(run) -> dict[str, Outcome]:
    """Run S19 over *run*, indexed by assertion text.

    Args:
        run: What to hand the scenario as R5's result.

    Returns:
        dict[str, Outcome]: What each assertion concluded.
    """
    findings = list(DEFAULT_REGISTRY.get("S19").func(run))
    return {finding.assertion: finding.outcome for finding in findings}


class AgentPathScenarioShapeTests(unittest.TestCase):
    """S19 is registered against R5 and declares the spec's two assertions."""

    def test_s19_is_registered_with_its_declared_run(self) -> None:
        entry = DEFAULT_REGISTRY.get("S19")
        self.assertEqual("R5", entry.run)
        self.assertTrue(entry.needs_backend)

    def test_the_assertion_texts_are_the_spec_texts(self) -> None:
        self.assertEqual(
            [
                "the consult tool result inside tool_calls[] contains no "
                "agents key",
                "it contains no consensus key",
            ],
            list(agent_path.ASSERTIONS),
        )


class AgentPathScenarioBodyTests(unittest.TestCase):
    """The cap holds, and the check is one level down from where it looks."""

    def test_a_capped_tool_result_passes_both(self) -> None:
        self.assertEqual({Outcome.PASS},
                         set(_outcomes(_result([_record(_TOOL_RESULT)])).values()))

    def test_the_agents_key_inside_the_result_fails_the_first(self) -> None:
        """The defect: the trio's full output enters the agent's context on
        EVERY consult, and an interactive session resends that context per
        turn, so the cost goes from once to per turn. Nothing breaks -- valid
        JSON, correct verdict, same exit code.
        """
        leaked = dict(copy.deepcopy(_TOOL_RESULT),
                      agents=[{"agent": "melchior"}])
        outcomes = _outcomes(_result([_record(leaked)]))
        self.assertEqual(Outcome.FAIL, outcomes[agent_path.ASSERTIONS[0]])

    def test_the_consensus_key_inside_the_result_fails_the_second(self) -> None:
        leaked = dict(copy.deepcopy(_TOOL_RESULT),
                      consensus={"consensus_verdict": "approve"})
        outcomes = _outcomes(_result([_record(leaked)]))
        self.assertEqual(Outcome.FAIL, outcomes[agent_path.ASSERTIONS[1]])

    def test_the_keys_at_the_TOP_level_do_not_fail_it(self) -> None:
        """A scenario reading the top-level object would pass whether or not
        the defect is present -- and would ALSO go red on the response
        envelope, which legitimately carries them under the flag. This test is
        what pins which object is being read.
        """
        run = _result([_record(_TOOL_RESULT)], top_level_leak=True)
        self.assertEqual({Outcome.PASS}, set(_outcomes(run).values()))

    def test_a_leak_in_the_SECOND_consult_call_still_fails(self) -> None:
        """An agent may consult more than once in a turn, and a check that
        stopped at the first record would miss every later one."""
        leaked = dict(copy.deepcopy(_TOOL_RESULT),
                      agents=[{"agent": "caspar"}])
        outcomes = _outcomes(_result([_record(_TOOL_RESULT), _record(leaked)]))
        self.assertEqual(Outcome.FAIL, outcomes[agent_path.ASSERTIONS[0]])

    def test_another_tool_carrying_the_key_is_not_this_scenario(self) -> None:
        """Attribution is by tool NAME. A different tool that happens to
        return an ``agents`` key says nothing about REQ-EA02, and crediting it
        here would be a red nobody can act on.
        """
        leaked = dict(copy.deepcopy(_TOOL_RESULT),
                      agents=[{"agent": "melchior"}])
        run = _result([_record(_TOOL_RESULT),
                       _record(leaked, name="project_knowledge")])
        self.assertEqual({Outcome.PASS}, set(_outcomes(run).values()))

    def test_no_consult_call_reports_both_as_cannot_test(self) -> None:
        """Whether the agent reaches for the tool is the MODEL's behaviour."""
        outcomes = _outcomes(_result([_record(_TOOL_RESULT,
                                              name="project_knowledge")]))
        self.assertEqual(list(agent_path.ASSERTIONS), list(outcomes))
        self.assertEqual({Outcome.CANNOT_TEST}, set(outcomes.values()))

    def test_a_result_that_is_not_json_reports_both_as_cannot_test(self) -> None:
        """The tool result is capped in bytes, so a very large envelope can
        reach the record already cut. A fragment is not evidence either way.
        """
        outcomes = _outcomes(_result([_record('{"report": "cut off here')]))
        self.assertEqual({Outcome.CANNOT_TEST}, set(outcomes.values()))

    def test_a_missing_run_reports_both_as_cannot_test(self) -> None:
        outcomes = _outcomes(None)
        self.assertEqual(list(agent_path.ASSERTIONS), list(outcomes))
        self.assertEqual({Outcome.CANNOT_TEST}, set(outcomes.values()))


if __name__ == "__main__":
    unittest.main()
