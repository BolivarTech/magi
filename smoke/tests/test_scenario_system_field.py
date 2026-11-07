# Author: Julian Bolivar
# Version: 0.25.0
# Date: 2026-11-08
"""Unit tests for S44-S46 (envelope system field, REQ-SY02/SY03/SY06)."""

import dataclasses
import json
import unittest

from smoke.outcome import Outcome
from smoke.product import ProductOutput
from smoke.registry import DEFAULT_REGISTRY
from smoke.runs import RunResult
from smoke.scenarios import system_field  # noqa: F401 - import registers it

WARNING = (b"warning: the request's \"system\" field was ignored; pass --allow-system-override "
           b"or set [headless] allow_system_override = true to apply it\n")


def _result(run_id: str, exit_code: int = 0, document=None, stderr: bytes = b"") -> RunResult:
    """A real ``RunResult`` for *run_id* whose stdout is *document* serialised (``None`` -> empty stdout)."""
    body = b"" if document is None else json.dumps(document).encode()
    return RunResult(run_id=run_id,
                     output=ProductOutput(stdout=body, stderr=stderr, exit_code=exit_code, command=["magi-rs"]),
                     duration_s=1.0, timed_out=False, planted=())


def _outcomes(scenario_id: str, result) -> dict[str, Outcome]:
    return {f.assertion: f.outcome for f in DEFAULT_REGISTRY.get(scenario_id).func(result)}


def _caps(**overrides) -> dict:
    caps = {"system_ignored": False, "system_override_applied": False}
    caps.update(overrides)
    return {"applied_caps": caps}


class SystemFieldScenarioTests(unittest.TestCase):

    def test_registration_and_declared_texts(self) -> None:
        self.assertEqual(("R9", "R10", "R11"), tuple(DEFAULT_REGISTRY.get(s).run for s in ("S44", "S45", "S46")))
        self.assertEqual(4, len(system_field.S44_ASSERTIONS))
        self.assertEqual(3, len(system_field.S45_ASSERTIONS))
        self.assertEqual(3, len(system_field.S46_ASSERTIONS))

    def test_s44_passes_on_an_ignored_system_with_all_signals(self) -> None:
        r = _result("R9", 0, _caps(system_ignored=True), WARNING)
        self.assertEqual({Outcome.PASS}, set(_outcomes("S44", r).values()))

    def test_s44_fails_when_the_warning_is_missing_from_stderr(self) -> None:
        r = _result("R9", 0, _caps(system_ignored=True), b"")
        self.assertEqual(Outcome.FAIL, _outcomes("S44", r)[system_field.S44_ASSERTIONS[1]])

    def test_s44_fails_when_the_field_says_not_ignored(self) -> None:
        r = _result("R9", 0, _caps(system_ignored=False), WARNING)
        self.assertEqual(Outcome.FAIL, _outcomes("S44", r)[system_field.S44_ASSERTIONS[2]])

    def test_s44_reads_an_environmental_error_as_untested_not_failed(self) -> None:
        # [F1a2] a backend that passed preflight and failed mid-run is the environment, as in S45.
        for kind in ("provider", "timeout"):
            r = _result("R9", 1, {**_caps(system_ignored=True),
                                  "error": {"kind": kind, "message": "quota exhausted"}}, WARNING)
            outcomes = _outcomes("S44", r)
            self.assertEqual(Outcome.CANNOT_TEST, outcomes[system_field.S44_ASSERTIONS[0]], kind)
            self.assertEqual(Outcome.PASS, outcomes[system_field.S44_ASSERTIONS[2]], kind)

    def test_s44_still_fails_the_exit_on_an_error_of_the_product_s_own(self) -> None:
        r = _result("R9", 2, {**_caps(system_ignored=True),
                              "error": {"kind": "misuse", "message": "refused"}}, WARNING)
        self.assertEqual(Outcome.FAIL, _outcomes("S44", r)[system_field.S44_ASSERTIONS[0]])

    def test_s45_passes_when_the_override_reached_the_mages(self) -> None:
        r = _result("R10", 0, {**_caps(system_override_applied=True), "consult": {}})
        self.assertEqual({Outcome.PASS}, set(_outcomes("S45", r).values()))

    def test_s45_fails_on_a_false_applied_flag(self) -> None:
        r = _result("R10", 0, {**_caps(system_override_applied=False), "consult": {}})
        self.assertEqual(Outcome.FAIL, _outcomes("S45", r)[system_field.S45_ASSERTIONS[1]])

    def test_s45_reads_the_flag_from_an_error_envelope(self) -> None:
        # A provider failure leaves the trio unproven but the installed flag measured (CP2 loop 6).
        r = _result("R10", 1, {**_caps(system_override_applied=True),
                               "error": {"kind": "provider", "message": "down"}})
        outcomes = _outcomes("S45", r)
        self.assertEqual(Outcome.CANNOT_TEST, outcomes[system_field.S45_ASSERTIONS[0]])
        self.assertEqual(Outcome.PASS, outcomes[system_field.S45_ASSERTIONS[1]])
        self.assertEqual(Outcome.PASS, outcomes[system_field.S45_ASSERTIONS[2]])

    def test_s45_fails_the_flag_on_an_error_envelope_that_says_not_installed(self) -> None:
        r = _result("R10", 1, {**_caps(system_override_applied=False),
                               "error": {"kind": "provider", "message": "down"}})
        self.assertEqual(Outcome.FAIL, _outcomes("S45", r)[system_field.S45_ASSERTIONS[1]])

    def test_s45_is_cannot_test_only_without_a_capture(self) -> None:
        self.assertEqual({Outcome.CANNOT_TEST}, set(_outcomes("S45", None).values()))

    def test_s45_fails_the_flag_when_the_envelope_has_no_applied_caps(self) -> None:
        r = _result("R10", 1, {"error": {"kind": "provider", "message": "down"}})
        self.assertEqual(Outcome.FAIL, _outcomes("S45", r)[system_field.S45_ASSERTIONS[1]])

    def test_s46_passes_on_a_misuse_exit_naming_the_cap(self) -> None:
        r = _result("R11", 2, None, b"error: the request's \"system\" field exceeds [headless] max_system_bytes = 16384 bytes\n")
        self.assertEqual({Outcome.PASS}, set(_outcomes("S46", r).values()))

    def test_s46_fails_when_the_run_succeeded(self) -> None:
        r = _result("R11", 0, _caps())
        self.assertEqual(Outcome.FAIL, _outcomes("S46", r)[system_field.S46_ASSERTIONS[0]])

    def test_s46_reads_a_hung_refusal_as_a_failure(self) -> None:
        # [MAGI F1a2] R11 is refused before any model, so a timeout is a product hang, never a slow provider:
        # S46 must see timed-out runs (the runner otherwise turns them into CANNOT_TEST) and fail every
        # assertion, without judging the synthetic exit 0 and empty stdout the executor gives a killed run.
        self.assertTrue(DEFAULT_REGISTRY.get("S46").inspects_timeouts)
        r = dataclasses.replace(_result("R11", 0, None, b""), timed_out=True)
        findings = list(DEFAULT_REGISTRY.get("S46").func(r))
        self.assertEqual([Outcome.FAIL] * len(system_field.S46_ASSERTIONS), [f.outcome for f in findings])
        self.assertTrue(all("timed out" in f.detail for f in findings))

    def test_s46_fails_when_stdout_carries_model_output(self) -> None:
        # [MAGI F1b2b] an emptied OUTPUT_KEYS made the loop below run nothing and pass; the keys are pinned.
        self.assertEqual(("response", "consult"), system_field.OUTPUT_KEYS)
        for key in system_field.OUTPUT_KEYS:
            with self.subTest(key=key):
                r = _result("R11", 2, {key: {"text": "x"}},
                            b"error: the request's \"system\" field exceeds [headless] max_system_bytes = 16384 bytes\n")
                self.assertEqual(Outcome.FAIL, _outcomes("S46", r)[system_field.S46_ASSERTIONS[2]])
