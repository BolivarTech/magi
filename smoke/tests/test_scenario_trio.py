# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Unit tests for the four scenarios that hang off R4."""

import copy
import json
import unittest

from smoke.outcome import Outcome
from smoke.product import ProductOutput
from smoke.registry import DEFAULT_REGISTRY
from smoke import runs as runs_module
from smoke.runs import RunResult
from smoke.scenarios import trio  # noqa: F401 - import registers it

#: ``applied_caps`` for R4's own command line: ``--timeout 300`` with the
#: product's default rotation count and retry enabled. Every number is what the
#: documented formula produces, computed by hand:
#:
#:   factor    = 2 attempts * (2 + 1) models * 120 = 720
#:   raw       = (300 - 6) * 100 // 720 = 40
#:   ceiling   = max(40, 15) = 40
#:   budget    = max(40 * 6 // 10, 10) = 24
#:   client    = max(40 * 3 // 10, 5)  = 12   (24 + 12 = 36 <= 40)
#:   threshold = 6 + ceil(15 * 720 / 100) = 114
_R4_CAPS = {
    "ceiling_above_sanity": False,
    "ceiling_floored": False,
    "floor_activation_threshold_secs": 114,
    "max_rotations_effective": 2,
    "max_tool_calls": 15,
    "max_tool_calls_clamped": False,
    "operation_budget_secs": 24,
    "system_override_applied": False,
    "timeout_secs": 300,
}

#: One seat's verdict, with the exact seven keys the product maps by hand.
_AGENT = {
    "agent": "melchior",
    "verdict": "approve",
    "confidence": 0.9,
    "summary": "s",
    "reasoning": "r",
    "findings": [],
    "recommendation": "rec",
}

#: One finding, with the exact six keys.
_FINDING = {
    "severity": "low",
    "title": "t",
    "detail": "d",
    "file": None,
    "line": None,
    "category": "correctness",
}

#: The consult envelope a healthy R4 emits.
_ENVELOPE = {
    "report": "the trio agreed",
    "degraded": False,
    "mode": "code-review",
    "mode_source": "explicit",
    "extraction_failures": {},
    "input_size": {"estimated_tokens": 60000, "warn_threshold": 100000,
                   "exceeded": False},
    "report_truncated": "none",
    "endpoint_divergence": False,
    "timeout_below_formula": False,
    "failed_agents": {},
    "agents": [dict(_AGENT, agent=seat)
               for seat in ("melchior", "balthasar", "caspar")],
    "consensus": {"consensus": "GO (3-0)", "consensus_verdict": "approve",
                  "confidence": 0.9, "score": 1.0, "agent_count": 3,
                  "votes": {"approve": 3}, "majority_summary": "m"},
}


def _document(envelope=None, caps=None, input_tokens=60000, error=None):
    """Build a complete headless output document.

    Args:
        envelope: What to put under ``consult``.
        caps: What to put under ``applied_caps``.
        input_tokens: What ``usage.input_tokens`` reports.
        error: The error payload, or None.

    Returns:
        dict: The document.
    """
    return {
        "schema_version": 1,
        "consult": copy.deepcopy(_ENVELOPE if envelope is None else envelope),
        "applied_caps": copy.deepcopy(_R4_CAPS if caps is None else caps),
        "usage": {"input_tokens": input_tokens, "output_tokens": 100},
        "error": error,
    }


def _result(run_id="R4", document=None, exit_code=0, timed_out=False,
            stdout=None) -> RunResult:
    """Build one shared run's result.

    Args:
        run_id: Which definition it stands for.
        document: The output object; None uses the healthy default.
        exit_code: What the product exited with.
        timed_out: Whether the harness's ceiling expired.
        stdout: Raw bytes to send instead of serialising *document*.

    Returns:
        RunResult: The real type, not a double.
    """
    body = (stdout if stdout is not None
            else json.dumps(_document() if document is None
                            else document).encode())
    return RunResult(
        run_id=run_id,
        output=ProductOutput(stdout=body, stderr=b"", exit_code=exit_code,
                             command=["magi-rs", "consult"]),
        duration_s=1.0, timed_out=timed_out, planted=())


def _outcomes(scenario_id, run) -> dict[str, Outcome]:
    """Run one scenario and index what it concluded by assertion text.

    Args:
        scenario_id: Which scenario to invoke.
        run: What to hand it.

    Returns:
        dict[str, Outcome]: What each assertion concluded.
    """
    findings = list(DEFAULT_REGISTRY.get(scenario_id).func(run))
    return {finding.assertion: finding.outcome for finding in findings}


class TrioScenarioShapeTests(unittest.TestCase):
    """The four declare the runs they read and the spec's assertion texts."""

    def test_s6_is_registered_with_its_declared_run(self) -> None:
        entry = DEFAULT_REGISTRY.get("S6")
        self.assertEqual("R4", entry.run)
        self.assertTrue(entry.needs_backend)
        self.assertFalse(entry.inspects_timeouts)

    def test_s7_is_the_only_one_that_inspects_timeouts(self) -> None:
        """It is the only scenario that can extract signal from a hang.

        It classifies from ``applied_caps`` -- evidence the run emitted before
        hanging -- never from configuration.
        """
        self.assertTrue(DEFAULT_REGISTRY.get("S7").inspects_timeouts)
        for other in ("S6", "S8", "S18"):
            self.assertFalse(DEFAULT_REGISTRY.get(other).inspects_timeouts,
                             other)

    def test_s18_is_registered_against_both_runs(self) -> None:
        entry = DEFAULT_REGISTRY.get("S18")
        self.assertEqual(("R4", "R8"), entry.run)
        self.assertTrue(entry.needs_backend)

    def test_the_assertion_texts_are_the_spec_texts(self) -> None:
        self.assertEqual(
            [
                "three verdicts are present",
                "a consensus was computed",
                "the run is not degraded",
            ],
            list(trio.S6_ASSERTIONS),
        )
        self.assertEqual(
            [
                "applied_caps satisfies the derived relation between the "
                "operation budget, the client timeout and the ceiling",
                "the trio is not degraded under a high --timeout",
                "ceiling_floored and ceiling_above_sanity report what "
                "corresponds",
            ],
            list(trio.S7_ASSERTIONS),
        )
        self.assertEqual(
            [
                "the large-payload run completes without truncation",
                "usage.input_tokens confirms the size in tokens, above the "
                "declared floor",
                "the generated payload stayed under the input cap",
            ],
            list(trio.S8_ASSERTIONS),
        )
        self.assertEqual(
            [
                "with the flag, agents and consensus are both present",
                "without it, both are absent",
                "agents[] exposes exactly 7 keys",
                "findings[] exposes exactly 6 keys",
            ],
            list(trio.S18_ASSERTIONS),
        )


class DerivationTests(unittest.TestCase):
    """The budget arithmetic, over a fixed table of published telemetry."""

    def test_the_declared_formula_over_a_fixed_table(self) -> None:
        for timeout, rotations, retry_off, expected in (
            # (timeout_secs, max_rotations, retry_disabled, (ceiling, budget,
            #  client, floored))
            (300, 2, False, (40, 24, 12, False)),
            (300, 0, False, (122, 73, 36, False)),
            (600, 2, False, (82, 49, 24, False)),
            (30, 2, False, (15, 10, 5, True)),
            (7200, 0, True, (5995, 3597, 1798, False)),
        ):
            with self.subTest(timeout=timeout, rotations=rotations):
                factor = trio.attempt_factor(rotations, retry_off)
                caps = {"timeout_secs": timeout,
                        "max_rotations_effective": rotations,
                        "floor_activation_threshold_secs":
                            trio.floor_activation_threshold(factor),
                        "operation_budget_secs": expected[1],
                        "ceiling_floored": expected[3],
                        "ceiling_above_sanity": expected[0] > 600}
                derived = trio.derive(caps)
                self.assertIsNotNone(derived)
                self.assertEqual(expected,
                                 (derived.ceiling, derived.budget,
                                  derived.client, derived.floored))

    def test_the_relation_holds_for_every_derivation_in_the_table(self) -> None:
        """REQ-A04: the two inner layers always fit inside the ceiling."""
        for timeout in (16, 30, 114, 300, 600, 1200):
            for rotations in (0, 1, 2, 4):
                for retry_off in (False, True):
                    with self.subTest(timeout=timeout, rotations=rotations,
                                      retry_off=retry_off):
                        factor = trio.attempt_factor(rotations, retry_off)
                        raw = trio.raw_ceiling(timeout, factor)
                        ceiling = max(raw, trio.CEILING_FLOOR_SECS)
                        self.assertLessEqual(
                            trio.operation_budget(ceiling)
                            + trio.client_timeout(ceiling), ceiling)

    def test_the_rotation_settings_are_recovered_from_the_threshold(self) -> None:
        """``retry_disabled`` is not published, so it is RECOVERED.

        The published ``floor_activation_threshold_secs`` is a function of the
        factor alone, and the two candidate factors give different thresholds,
        so the run's own telemetry says which one it used. Reading it from the
        harness's configuration instead would be the second source of truth
        that disagrees the first time somebody changes a setting.
        """
        for rotations, retry_off in ((2, False), (2, True), (0, False)):
            with self.subTest(rotations=rotations, retry_off=retry_off):
                factor = trio.attempt_factor(rotations, retry_off)
                caps = {"timeout_secs": 300,
                        "max_rotations_effective": rotations,
                        "floor_activation_threshold_secs":
                            trio.floor_activation_threshold(factor)}
                self.assertEqual(factor, trio.derive(caps).factor)

    def test_a_threshold_no_factor_explains_refuses_to_derive(self) -> None:
        caps = {"timeout_secs": 300, "max_rotations_effective": 2,
                "floor_activation_threshold_secs": 999}
        self.assertIsNone(trio.derive(caps))

    def test_an_absent_wall_clock_refuses_to_derive(self) -> None:
        """No ``--timeout`` means the ceiling came from the configuration.

        That value is not in ``applied_caps``, so there is nothing to derive
        from, and the harness does not go looking for the environment's
        ``agent_timeout_secs``: the assertion is about what the run PUBLISHED.
        """
        caps = dict(_R4_CAPS, timeout_secs=None)
        self.assertIsNone(trio.derive(caps))


class S6Tests(unittest.TestCase):
    """Three verdicts, a consensus, and no degradation."""

    def test_a_healthy_trio_passes_all_three(self) -> None:
        self.assertEqual({Outcome.PASS},
                         set(_outcomes("S6", _result()).values()))

    def test_two_verdicts_fail_the_first(self) -> None:
        envelope = copy.deepcopy(_ENVELOPE)
        envelope["agents"] = envelope["agents"][:2]
        outcomes = _outcomes("S6", _result(document=_document(envelope)))
        self.assertEqual(Outcome.FAIL, outcomes[trio.S6_ASSERTIONS[0]])

    def test_a_missing_consensus_verdict_fails_the_second(self) -> None:
        envelope = copy.deepcopy(_ENVELOPE)
        envelope["consensus"] = {"consensus_verdict": ""}
        outcomes = _outcomes("S6", _result(document=_document(envelope)))
        self.assertEqual(Outcome.FAIL, outcomes[trio.S6_ASSERTIONS[1]])

    def test_a_degraded_run_fails_the_third(self) -> None:
        envelope = dict(copy.deepcopy(_ENVELOPE), degraded=True)
        outcomes = _outcomes("S6", _result(document=_document(envelope)))
        self.assertEqual(Outcome.FAIL, outcomes[trio.S6_ASSERTIONS[2]])

    def test_a_failed_seat_fails_the_third(self) -> None:
        """``degraded`` is not the only signal, and ``failed_agents`` is typed.

        A seat that fell and was replaced leaves a cause here, and a scenario
        reading only the boolean would call that healthy.
        """
        envelope = copy.deepcopy(_ENVELOPE)
        envelope["failed_agents"] = {"caspar": "the endpoint refused"}
        outcomes = _outcomes("S6", _result(document=_document(envelope)))
        self.assertEqual(Outcome.FAIL, outcomes[trio.S6_ASSERTIONS[2]])

    def test_a_provider_error_cannot_test_every_assertion(self) -> None:
        """The provider's failure, not the product's -- see SS5.1's table."""
        broken = _document(error={"kind": "provider", "message": "502"})
        broken["consult"] = None
        outcomes = _outcomes("S6", _result(document=broken, exit_code=1))
        self.assertEqual({Outcome.CANNOT_TEST}, set(outcomes.values()))

    def test_a_runtime_error_fails_every_assertion(self) -> None:
        """The product spoke and said no, which is a verdict about it."""
        broken = _document(error={"kind": "runtime", "message": "boom"})
        broken["consult"] = None
        outcomes = _outcomes("S6", _result(document=broken, exit_code=1))
        self.assertEqual({Outcome.FAIL}, set(outcomes.values()))

    def test_a_consult_run_with_no_envelope_fails(self) -> None:
        outcomes = _outcomes("S6", _result(document=_document(envelope=None)
                                           | {"consult": None}))
        self.assertEqual({Outcome.FAIL}, set(outcomes.values()))


class S7Tests(unittest.TestCase):
    """The derived ceiling, read off evidence and never off configuration."""

    def test_the_published_telemetry_passes_all_three(self) -> None:
        self.assertEqual({Outcome.PASS},
                         set(_outcomes("S7", _result()).values()))

    def test_a_budget_the_formula_does_not_explain_fails_the_first(self) -> None:
        """Failure #4's mechanism: the ceiling stopped following --timeout."""
        caps = dict(_R4_CAPS, operation_budget_secs=54)
        outcomes = _outcomes("S7", _result(document=_document(caps=caps)))
        self.assertEqual(Outcome.FAIL, outcomes[trio.S7_ASSERTIONS[0]])

    def test_a_degraded_trio_fails_the_second(self) -> None:
        envelope = dict(copy.deepcopy(_ENVELOPE), degraded=True)
        outcomes = _outcomes("S7", _result(document=_document(envelope)))
        self.assertEqual(Outcome.FAIL, outcomes[trio.S7_ASSERTIONS[1]])

    def test_a_misreported_floor_flag_fails_the_third(self) -> None:
        caps = dict(_R4_CAPS, ceiling_floored=True)
        outcomes = _outcomes("S7", _result(document=_document(caps=caps)))
        self.assertEqual(Outcome.FAIL, outcomes[trio.S7_ASSERTIONS[2]])

    def test_a_misreported_sanity_flag_fails_the_third(self) -> None:
        caps = dict(_R4_CAPS, ceiling_above_sanity=True)
        outcomes = _outcomes("S7", _result(document=_document(caps=caps)))
        self.assertEqual(Outcome.FAIL, outcomes[trio.S7_ASSERTIONS[2]])

    def test_a_timeout_above_the_threshold_cannot_test_the_second(self) -> None:
        """A slow provider under an adequate ceiling is not a product defect.

        Calling it FAIL would turn the gate red on somebody else's load, which
        is the intermittent red that gets rationalised until the gate is
        ignored.
        """
        caps = dict(_R4_CAPS, timeout_secs=300,
                    floor_activation_threshold_secs=114)
        outcomes = _outcomes("S7", _result(document=_document(caps=caps),
                                           timed_out=True))
        self.assertEqual(Outcome.CANNOT_TEST, outcomes[trio.S7_ASSERTIONS[1]])

    def test_a_timeout_under_a_floored_ceiling_fails_the_second(self) -> None:
        """This one IS failure #4, and the evidence separating them is
        ``applied_caps``, which the run emits before it hangs."""
        caps = dict(_R4_CAPS, timeout_secs=30, operation_budget_secs=10,
                    ceiling_floored=True)
        outcomes = _outcomes("S7", _result(document=_document(caps=caps),
                                           timed_out=True))
        self.assertEqual(Outcome.FAIL, outcomes[trio.S7_ASSERTIONS[1]])

    def test_a_timeout_with_no_telemetry_cannot_test_anything(self) -> None:
        """Without ``applied_caps`` there is nothing to tell a degraded
        ceiling from a slow provider, and inventing the distinction falls the
        wrong way: guessing FAIL accuses the product with no evidence."""
        outcomes = _outcomes("S7", _result(stdout=b"", timed_out=True))
        self.assertEqual(list(trio.S7_ASSERTIONS), list(outcomes))
        self.assertEqual({Outcome.CANNOT_TEST}, set(outcomes.values()))

    def test_an_absent_wall_clock_cannot_test_the_arithmetic(self) -> None:
        caps = dict(_R4_CAPS, timeout_secs=None)
        outcomes = _outcomes("S7", _result(document=_document(caps=caps)))
        self.assertEqual(Outcome.CANNOT_TEST, outcomes[trio.S7_ASSERTIONS[0]])
        self.assertEqual(Outcome.CANNOT_TEST, outcomes[trio.S7_ASSERTIONS[2]])


class S8Tests(unittest.TestCase):
    """A large input arrives large, or the scenario says it never was sent."""

    def test_a_complete_large_run_passes_all_three(self) -> None:
        result = _result(document=_document(input_tokens=60000))
        with _payload(sent=250000, target=250000, floor=50000):
            self.assertEqual({Outcome.PASS},
                             set(_outcomes("S8", result).values()))

    def test_a_truncated_report_fails_the_first(self) -> None:
        envelope = dict(copy.deepcopy(_ENVELOPE), report_truncated="bytes")
        with _payload(sent=250000, target=250000, floor=50000):
            outcomes = _outcomes("S8", _result(document=_document(envelope)))
        self.assertEqual(Outcome.FAIL, outcomes[trio.S8_ASSERTIONS[0]])

    def test_too_few_tokens_fail_the_second(self) -> None:
        with _payload(sent=250000, target=250000, floor=50000):
            outcomes = _outcomes(
                "S8", _result(document=_document(input_tokens=100)))
        self.assertEqual(Outcome.FAIL, outcomes[trio.S8_ASSERTIONS[1]])

    def test_a_payload_that_was_never_sent_cannot_test_the_second(self) -> None:
        """The harness must not accuse the product of a size it never sent.

        The spec states this rule for a tree too small to build the payload;
        it applies identically when the run simply does not carry it.
        """
        with _payload(sent=53, target=250000, floor=50000):
            outcomes = _outcomes(
                "S8", _result(document=_document(input_tokens=100)))
        self.assertEqual(Outcome.CANNOT_TEST, outcomes[trio.S8_ASSERTIONS[1]])

    def test_a_payload_over_the_input_cap_fails_the_third(self) -> None:
        with _payload(sent=trio.MAX_QUERY_BYTES + 1, target=250000,
                      floor=50000):
            outcomes = _outcomes("S8", _result())
        self.assertEqual(Outcome.FAIL, outcomes[trio.S8_ASSERTIONS[2]])


class S18Tests(unittest.TestCase):
    """The shape varies by FLAG, never by DATA, and the counts are EXACT."""

    def _pair(self, r4=None, r8=None):
        """Build the two-run mapping S18 receives.

        Args:
            r4: R4's document, or None for the healthy default.
            r8: R8's document, or None for a flagless envelope.

        Returns:
            dict[str, RunResult]: Keyed by run id.
        """
        if r8 is None:
            bare = {key: value for key, value in copy.deepcopy(_ENVELOPE).items()
                    if key not in ("agents", "consensus")}
            r8 = _document(bare)
        return {"R4": _result("R4", document=r4),
                "R8": _result("R8", document=r8)}

    def test_the_declared_shapes_pass_the_first_three(self) -> None:
        """The fourth is covered separately: the default envelope carries no
        finding, and an empty array is the positive certificate that nothing
        was produced rather than a shape that is wrong."""
        outcomes = _outcomes("S18", self._pair())
        for text in trio.S18_ASSERTIONS[:3]:
            self.assertEqual(Outcome.PASS, outcomes[text], text)

    def test_a_missing_key_under_the_flag_fails_the_first(self) -> None:
        envelope = copy.deepcopy(_ENVELOPE)
        del envelope["consensus"]
        outcomes = _outcomes("S18", self._pair(r4=_document(envelope)))
        self.assertEqual(Outcome.FAIL, outcomes[trio.S18_ASSERTIONS[0]])

    def test_a_key_leaking_without_the_flag_fails_the_second(self) -> None:
        """REQ-EA02's cost: the trio's full output in every consumer's JSON."""
        outcomes = _outcomes("S18", self._pair(r8=_document(_ENVELOPE)))
        self.assertEqual(Outcome.FAIL, outcomes[trio.S18_ASSERTIONS[1]])

    def test_an_eighth_agent_key_fails_the_third(self) -> None:
        """The maintenance contract, accepted and not re-litigated.

        ``AgentOutput`` is ``#[non_exhaustive]``: a field magi-core adds in a
        minor release reaches the public JSON with no review here the moment
        somebody replaces the explicit mapping with a direct interpolation. An
        "at least 7" detects none of it.
        """
        envelope = copy.deepcopy(_ENVELOPE)
        envelope["agents"][0]["cost_usd"] = 0.01
        outcomes = _outcomes("S18", self._pair(r4=_document(envelope)))
        self.assertEqual(Outcome.FAIL, outcomes[trio.S18_ASSERTIONS[2]])

    def test_a_seventh_agent_key_removed_fails_the_third(self) -> None:
        envelope = copy.deepcopy(_ENVELOPE)
        del envelope["agents"][0]["recommendation"]
        outcomes = _outcomes("S18", self._pair(r4=_document(envelope)))
        self.assertEqual(Outcome.FAIL, outcomes[trio.S18_ASSERTIONS[2]])

    def test_a_seventh_finding_key_fails_the_fourth(self) -> None:
        envelope = copy.deepcopy(_ENVELOPE)
        envelope["agents"][0]["findings"] = [dict(_FINDING, cwe="CWE-20")]
        outcomes = _outcomes("S18", self._pair(r4=_document(envelope)))
        self.assertEqual(Outcome.FAIL, outcomes[trio.S18_ASSERTIONS[3]])

    def test_a_declared_finding_shape_passes_the_fourth(self) -> None:
        envelope = copy.deepcopy(_ENVELOPE)
        envelope["agents"][0]["findings"] = [copy.deepcopy(_FINDING)]
        outcomes = _outcomes("S18", self._pair(r4=_document(envelope)))
        self.assertEqual(Outcome.PASS, outcomes[trio.S18_ASSERTIONS[3]])

    def test_no_seat_completed_cannot_test_the_counts(self) -> None:
        """An empty ``agents`` is the positive certificate that no seat
        completed, so there is no shape to count -- not a shape that is
        wrong."""
        envelope = copy.deepcopy(_ENVELOPE)
        envelope["agents"] = []
        outcomes = _outcomes("S18", self._pair(r4=_document(envelope)))
        self.assertEqual(Outcome.CANNOT_TEST, outcomes[trio.S18_ASSERTIONS[2]])

    def test_no_finding_at_all_cannot_test_the_fourth(self) -> None:
        outcomes = _outcomes("S18", self._pair())
        self.assertEqual(Outcome.CANNOT_TEST, outcomes[trio.S18_ASSERTIONS[3]])

    def test_a_missing_second_run_reports_all_four(self) -> None:
        outcomes = _outcomes("S18", {"R4": _result()})
        self.assertEqual(list(trio.S18_ASSERTIONS), list(outcomes))
        self.assertEqual(Outcome.CANNOT_TEST, outcomes[trio.S18_ASSERTIONS[1]])


class _payload:
    """Stand in for the three payload facts S8 reads, for one block.

    Restoration happens in ``__exit__`` rather than on the happy path, so a
    test that fails inside the block still leaves the module as it found it --
    otherwise every test after it inherits this one's numbers.

    Attributes:
        sent: How many bytes R4's definition actually carries.
        target: The configured payload size.
        floor: The configured token floor.
    """

    #: The accessors this stands in for, named once so adding a fourth cannot
    #: be half-done: saved, replaced and restored all iterate this tuple.
    NAMES = ("payload_bytes", "payload_target", "payload_floor")

    def __init__(self, sent: int, target: int, floor: int) -> None:
        """Record what to answer.

        Args:
            sent: Bytes R4 carries.
            target: The configured size.
            floor: The configured floor.
        """
        self._answers = {
            "payload_bytes": lambda run_id: sent,
            "payload_target": lambda: target,
            "payload_floor": lambda: floor,
        }
        self._saved: dict = {}

    def __enter__(self) -> "_payload":
        """Install the stand-ins.

        Returns:
            _payload: Self.
        """
        self._saved = {name: getattr(runs_module, name) for name in self.NAMES}
        for name in self.NAMES:
            setattr(runs_module, name, self._answers[name])
        return self

    def __exit__(self, *exc) -> bool:
        """Put every accessor back, whatever happened.

        Args:
            *exc: The exception triple, ignored.

        Returns:
            bool: False, so nothing is suppressed.
        """
        for name, original in self._saved.items():
            setattr(runs_module, name, original)
        return False


if __name__ == "__main__":
    unittest.main()
