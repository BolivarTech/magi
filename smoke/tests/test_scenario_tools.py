# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Unit tests for the S5 scenario: the tool record, never the model's prose."""

import json
import unittest

from smoke.outcome import Outcome
from smoke.product import ProductOutput
from smoke.registry import DEFAULT_REGISTRY
from smoke.runs import RunResult
from smoke.scenarios import tools  # noqa: F401 - import registers it
from smoke.tests import support

#: How the probe double answers. Named rather than passed as bare strings so a
#: typo in a test is an AttributeError instead of a silently different case.
DENIED = "denied"
ALLOWED = "allowed"
NEVER_TRIED = "never-tried"
UNPARSEABLE = "unparseable"

#: A successful ``ls`` record, which is what R1's prompt asks the model for.
_LS_RECORD = {"name": "ls", "input": {"path": "."}, "ok": True,
              "result": "magi.toml", "ms": 3}


def _capture(document, exit_code=0) -> ProductOutput:
    """Build a capture whose stdout is *document* serialised.

    Args:
        document: The object the product is pretending to have printed.
        exit_code: What it exited with.

    Returns:
        ProductOutput: The capture.
    """
    return ProductOutput(stdout=json.dumps(document).encode(), stderr=b"",
                         exit_code=exit_code, command=["magi-rs", "query"])


def _run(tool_calls) -> RunResult:
    """Build an R1 result carrying *tool_calls*.

    Args:
        tool_calls: The list to place under the ``tool_calls`` key.

    Returns:
        RunResult: The real type, so a scenario reading a field it does not
        carry fails here rather than against a laxer stand-in.
    """
    return RunResult(run_id="R1", output=_capture({"tool_calls": tool_calls}),
                     duration_s=1.0, timed_out=False, planted=())


class _FakeProbe:
    """Answers S5's escape probe in one of the four ways it can end.

    The record is built when the call arrives, not in the constructor: the
    path it names comes from :func:`smoke.scenarios.tools.outside_path`, which
    needs :mod:`smoke.runs` configured, and that happens after the double is
    handed over.

    Attributes:
        mode: One of :data:`DENIED`, :data:`ALLOWED`, :data:`NEVER_TRIED` or
            :data:`UNPARSEABLE`.
    """

    def __init__(self, mode: str) -> None:
        """Create the double.

        Args:
            mode: How the probe should end.
        """
        self.mode = mode

    def __call__(self, call: support.Call) -> ProductOutput | None:
        """Answer one invocation.

        Args:
            call: What the fake binary was asked to run.

        Returns:
            ProductOutput: The canned answer.
        """
        if self.mode == UNPARSEABLE:
            return ProductOutput(stdout=b"not json", stderr=b"", exit_code=0,
                                 command=["magi-rs", "query"])
        target = str(tools.outside_path())
        records = {
            DENIED: [{"name": "view", "input": {"path": target}, "ok": False,
                      "result": "path escapes the workspace", "ms": 1}],
            ALLOWED: [{"name": "view", "input": {"path": target}, "ok": True,
                       "result": "[package]", "ms": 1}],
            NEVER_TRIED: [],
        }[self.mode]
        return _capture({"tool_calls": records})


def _outcomes(run) -> dict[str, Outcome]:
    """Run S5 over *run*, indexed by assertion text.

    Args:
        run: What to hand the scenario as R1's result.

    Returns:
        dict[str, Outcome]: What each assertion concluded.
    """
    findings = list(DEFAULT_REGISTRY.get("S5").func(run))
    return {finding.assertion: finding.outcome for finding in findings}


class ToolsScenarioShapeTests(unittest.TestCase):
    """S5 is registered against R1 and declares the spec's two assertions."""

    def test_s5_is_registered_with_its_declared_run(self) -> None:
        entry = DEFAULT_REGISTRY.get("S5")
        self.assertEqual("R1", entry.run)
        self.assertTrue(entry.needs_backend)

    def test_the_assertion_texts_are_the_spec_texts(self) -> None:
        self.assertEqual(
            [
                "tool_calls[] records the invocation with ok: true",
                "a prompt asking to read outside the workspace is denied",
            ],
            list(tools.ASSERTIONS),
        )


class ToolsScenarioBodyTests(unittest.TestCase):
    """Both assertions read the RESULT of a call, never the assistant's text."""

    def test_a_successful_call_and_a_refused_escape_pass_both(self) -> None:
        support.install_fake_runs(self, responder=_FakeProbe(DENIED))
        self.assertEqual({Outcome.PASS},
                         set(_outcomes(_run([_LS_RECORD])).values()))

    def test_a_tool_that_failed_fails_the_first(self) -> None:
        support.install_fake_runs(self, responder=_FakeProbe(DENIED))
        broken = dict(_LS_RECORD, ok=False, result="boom")
        self.assertEqual(Outcome.FAIL,
                         _outcomes(_run([broken]))[tools.ASSERTIONS[0]])

    def test_no_tool_call_at_all_cannot_test_the_first(self) -> None:
        """Whether the model reaches for a tool is the MODEL's behaviour.

        Calling that a product defect would put the gate red on a model's mood,
        which is the intermittent red that gets rationalised until the gate is
        ignored.
        """
        support.install_fake_runs(self, responder=_FakeProbe(DENIED))
        self.assertEqual(Outcome.CANNOT_TEST,
                         _outcomes(_run([]))[tools.ASSERTIONS[0]])

    def test_an_escape_that_succeeded_fails_the_second(self) -> None:
        """The defect assertion 2 exists for: the guard let the read through."""
        support.install_fake_runs(self, responder=_FakeProbe(ALLOWED))
        self.assertEqual(Outcome.FAIL,
                         _outcomes(_run([_LS_RECORD]))[tools.ASSERTIONS[1]])

    def test_a_model_that_never_tried_cannot_test_the_second(self) -> None:
        """"Denied" and "the model declined to try" are different states.

        Only the first says anything about ``PathGuard``, and reporting the
        second as PASS is the empty green this assertion is written against.
        """
        support.install_fake_runs(self, responder=_FakeProbe(NEVER_TRIED))
        self.assertEqual(Outcome.CANNOT_TEST,
                         _outcomes(_run([_LS_RECORD]))[tools.ASSERTIONS[1]])

    def test_a_probe_whose_output_cannot_be_read_cannot_test_the_second(self) -> None:
        support.install_fake_runs(self, responder=_FakeProbe(UNPARSEABLE))
        outcomes = _outcomes(_run([_LS_RECORD]))
        self.assertEqual(Outcome.PASS, outcomes[tools.ASSERTIONS[0]])
        self.assertEqual(Outcome.CANNOT_TEST, outcomes[tools.ASSERTIONS[1]])

    def test_a_probe_the_harness_could_not_start_cannot_test_the_second(self) -> None:
        """The default double refuses to answer, which is not a verdict."""
        support.install_fake_runs(self)
        outcomes = _outcomes(_run([_LS_RECORD]))
        self.assertEqual(list(tools.ASSERTIONS), list(outcomes))
        self.assertEqual(Outcome.CANNOT_TEST, outcomes[tools.ASSERTIONS[1]])

    def test_a_missing_run_reports_both_as_cannot_test(self) -> None:
        support.install_fake_runs(self, responder=_FakeProbe(DENIED))
        outcomes = _outcomes(None)
        self.assertEqual(list(tools.ASSERTIONS), list(outcomes))
        self.assertEqual(Outcome.CANNOT_TEST, outcomes[tools.ASSERTIONS[0]])

    def test_the_prompt_names_the_target_relatively_not_absolutely(self) -> None:
        """Measured against the real backend: an ABSOLUTE path is refused in
        prose and the tool is never called, so the guard is never asked and the
        assertion can only ever report CANNOT_TEST.

        A relative path that walks out of the workspace is the same escape, and
        it is the one ``PathGuard`` has to normalise lexically before it can
        reject. Naming it that way is what turns a permanently unevaluated
        assertion into one that reaches the code it protects.
        """
        binary = support.install_fake_runs(self, responder=_FakeProbe(DENIED))
        _outcomes(_run([_LS_RECORD]))
        probes = [call for call in binary.calls if call.stdin is not None]
        self.assertEqual(1, len(probes))
        prompt = probes[0].stdin.decode("utf-8")
        self.assertIn(tools.escape_target(), prompt)
        self.assertNotIn(str(tools.outside_path()), prompt)
        self.assertTrue(tools.escape_target().startswith("../"),
                        tools.escape_target())

    def test_the_probe_aims_at_a_path_that_exists_outside_the_workspace(self) -> None:
        """A refusal for a MISSING file would be a false green.

        ``ok: false`` cannot distinguish "the guard stopped me" from "there was
        nothing there", so the target has to exist. That is a precondition of
        the assertion, and it is checked rather than assumed.
        """
        support.install_fake_runs(self, responder=_FakeProbe(DENIED))
        target = tools.outside_path()
        self.assertTrue(target.is_file(), target)
        self.assertFalse(str(target).startswith(str(tools.runs.workspace_root())))


if __name__ == "__main__":
    unittest.main()
