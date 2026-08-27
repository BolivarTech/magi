# Author: Julian Bolivar
# Version: 0.17.0
# Date: 2026-08-27
"""Unit tests for the three scenarios the magi-core 4.0.0 move needs.

These test the HARNESS. Every product answer here is a double, so what is
under test is the mapping from a published document to an outcome -- and in
particular that S21 reaches ``CANNOT_TEST`` by a NAMED condition rather than a
generic one, which is the only thing separating an honest not-run from a
vacuous green.
"""

import copy
import json
import unittest

from smoke.outcome import Outcome
from smoke.product import ProductOutput
from smoke.registry import DEFAULT_REGISTRY
from smoke import runs as runs_module
from smoke.runs import RunResult
from smoke.scenarios import migration  # noqa: F401 - import registers it

#: How large R4's payload is declared to be, and how many bytes the doubles say
#: it carried. Both are the harness's own numbers, so S21's "was the recipe
#: applied?" branch is exercised against a value it can actually reach.
PAYLOAD_TARGET = 250000
PAYLOAD_SENT = 250054

#: One completion attempt, with the exact five keys the product maps by hand
#: and a cap equal to the declared one.
_ATTEMPT = {
    "model": "glm-5.2:cloud",
    "cap": migration.DECLARED_COMPLETION_CAP,
    "finish": "stop",
    "completion_tokens": 900,
    "prompt_tokens": 62513,
}

#: One rotation hop that names an empty completion, mage-local.
_HOP = {
    "from_lineage": "glm",
    "to_lineage": "qwen",
    "model_resolved": "qwen3.5:397b-cloud",
    "cause": migration.EMPTY_COMPLETION_CAUSE,
    "mage_local": True,
    "detail": "mage-local: the completion was empty",
}

#: One seat's verdict. Only its presence and the count matter here; the exact
#: key set is S18's subject and is not re-litigated.
_AGENT = {
    "agent": "melchior",
    "verdict": "approve",
    "confidence": 0.9,
    "summary": "s",
    "reasoning": "r",
    "findings": [],
    "recommendation": "rec",
}

#: ``applied_caps`` for R4's own command line, with the threshold the
#: attempt-factor formula produces for two rotations with retry enabled:
#: ``factor = 2 * (2 + 1) * 120 = 720`` and ``6 + ceil(15 * 720 / 100) = 114``.
_CAPS = {
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

#: The consult envelope a healthy R4 emits under ``--structured-verdicts``.
_ENVELOPE = {
    "report": "the trio agreed",
    "degraded": False,
    "mode": "code-review",
    "report_truncated": "none",
    "failed_agents": {},
    "rotations": [],
    "ran_unmeasured": [],
    "completions": {
        "balthasar": [copy.deepcopy(_ATTEMPT)],
        "caspar": [copy.deepcopy(_ATTEMPT)],
        "melchior": [copy.deepcopy(_ATTEMPT)],
    },
    "pool_eligibility": {},
    "agents": [dict(_AGENT, agent=seat)
               for seat in ("melchior", "balthasar", "caspar")],
    "consensus": {"consensus": "GO (3-0)", "consensus_verdict": "approve"},
}


def _document(envelope=None, caps=None, error=None):
    """Build a complete headless output document.

    Args:
        envelope: What to put under ``consult``; None uses the healthy default.
        caps: What to put under ``applied_caps``; None uses the default.
        error: The error payload, or None.

    Returns:
        dict: The document.
    """
    return {
        "schema_version": 1,
        "consult": copy.deepcopy(_ENVELOPE if envelope is None else envelope),
        "applied_caps": copy.deepcopy(_CAPS if caps is None else caps),
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "error": error,
    }


def _result(document=None, exit_code=0, stdout=None,
            stdin_bytes=PAYLOAD_SENT) -> RunResult:
    """Build R4's result.

    Args:
        document: The output object; None uses the healthy default.
        exit_code: What the product exited with.
        stdout: Raw bytes to send instead of serialising *document*.
        stdin_bytes: How many bytes the run CARRIED. Read off the result the
            way the scenario reads it, never off a module accessor answering
            the declared prompt length.

    Returns:
        RunResult: The real type, not a double.
    """
    body = (stdout if stdout is not None
            else json.dumps(_document() if document is None
                            else document).encode())
    return RunResult(
        run_id=migration.MIGRATION_RUN,
        output=ProductOutput(stdout=body, stderr=b"", exit_code=exit_code,
                             command=["magi-rs", "consult"]),
        duration_s=1.0, timed_out=False, planted=(),
        stdin_bytes=stdin_bytes)


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


def _details(scenario_id, run) -> dict[str, str]:
    """Run one scenario and index the causes it wrote by assertion text.

    Args:
        scenario_id: Which scenario to invoke.
        run: What to hand it.

    Returns:
        dict[str, str]: What each assertion said about itself.
    """
    findings = list(DEFAULT_REGISTRY.get(scenario_id).func(run))
    return {finding.assertion: finding.detail for finding in findings}


def _envelope_with(**changes):
    """The healthy envelope with some keys replaced.

    Args:
        **changes: Keys to overwrite.

    Returns:
        dict: A fresh envelope.
    """
    envelope = copy.deepcopy(_ENVELOPE)
    envelope.update(copy.deepcopy(changes))
    return envelope


def _seat_attempts(*records):
    """A ``completions`` map holding one seat with the given attempts.

    Args:
        *records: The attempt objects, in order.

    Returns:
        dict: The map, keyed by a single seat.
    """
    return {"melchior": [copy.deepcopy(record) for record in records]}


class _payload:
    """Stand in for the payload size S21 reads, for one block.

    Restoration happens in ``__exit__`` rather than on the happy path, so a
    test failing inside the block still leaves the module as it found it --
    otherwise every test after it inherits this one's number.
    """

    #: The accessors this stands in for, named once so adding a second cannot
    #: be half-done: saved, replaced and restored all iterate this tuple.
    NAMES = ("payload_target",)

    def __init__(self, target: int = PAYLOAD_TARGET) -> None:
        """Record what to answer.

        Args:
            target: The configured payload size, in bytes.
        """
        self._answers = {"payload_target": lambda: target}
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


class MigrationScenarioShapeTests(unittest.TestCase):
    """All three declare R4, need a backend, and cannot read a timed-out run."""

    def test_each_scenario_is_registered_against_the_trio_run(self) -> None:
        for scenario_id in ("S20", "S21", "S22"):
            with self.subTest(scenario=scenario_id):
                entry = DEFAULT_REGISTRY.get(scenario_id)
                self.assertEqual(migration.MIGRATION_RUN, entry.run)
                self.assertTrue(entry.needs_backend)

    def test_none_of_them_claims_to_read_a_timed_out_run(self) -> None:
        """S7 is the only scenario that can classify a hang, and stays so.

        These three read an envelope the product emits at the END of a consult,
        so a run that never finished leaves them nothing -- claiming otherwise
        would have them assert over a partial capture.
        """
        for scenario_id in ("S20", "S21", "S22"):
            with self.subTest(scenario=scenario_id):
                self.assertFalse(
                    DEFAULT_REGISTRY.get(scenario_id).inspects_timeouts)

    def test_the_assertion_texts_are_the_declared_texts(self) -> None:
        self.assertEqual(
            [
                "the trio completed against the native wire",
                "every transmitted attempt carries the declared completion cap",
                "completions are recorded per attempt",
                "the published per-mage threshold agrees with the "
                "attempt-factor formula",
            ],
            list(migration.S20_ASSERTIONS),
        )
        self.assertEqual(
            [
                "the rotation cause reads empty_completion and not transport",
                "the empty completion is reported as mage-local",
                "finish tells budget exhaustion from a genuinely empty answer",
            ],
            list(migration.S21_ASSERTIONS),
        )
        self.assertEqual(
            [
                "pool_eligibility is present even when empty",
                "all three notions of degradation are derivable from "
                "published keys",
                "degraded is false for a three-verdict run",
            ],
            list(migration.S22_ASSERTIONS),
        )

    def test_the_declared_cap_is_the_milestones_own_number(self) -> None:
        """Mirrored from ``DECLARED_COMPLETION_CAP`` in ``main.rs``.

        The maintenance contract is accepted, not re-litigated: moving the
        product's cap turns S20 red and forces a change here. A check that
        adjusted itself to whatever the product reported would detect nothing,
        which is exactly the trap this cap sits in -- the declared value IS
        magi-core's own default.
        """
        self.assertEqual(16384, migration.DECLARED_COMPLETION_CAP)


class S20Tests(unittest.TestCase):
    """The trio answered, and the cap it carried was ours."""

    def test_a_healthy_run_passes_all_four(self) -> None:
        self.assertEqual({Outcome.PASS},
                         set(_outcomes("S20", _result()).values()))

    def test_a_non_zero_exit_fails_the_first(self) -> None:
        outcomes = _outcomes("S20", _result(exit_code=1))
        self.assertEqual(Outcome.FAIL, outcomes[migration.S20_ASSERTIONS[0]])

    def test_no_seat_answering_fails_the_first(self) -> None:
        """A verdict cannot exist without a completed round trip, so an empty
        ``agents`` on a zero exit says nothing crossed the wire."""
        document = _document(_envelope_with(agents=[]))
        outcomes = _outcomes("S20", _result(document=document))
        self.assertEqual(Outcome.FAIL, outcomes[migration.S20_ASSERTIONS[0]])

    def test_a_flagless_run_cannot_test_the_first(self) -> None:
        envelope = copy.deepcopy(_ENVELOPE)
        del envelope["agents"]
        outcomes = _outcomes("S20", _result(document=_document(envelope)))
        self.assertEqual(Outcome.CANNOT_TEST,
                         outcomes[migration.S20_ASSERTIONS[0]])

    def test_an_attempt_without_a_cap_fails_the_second(self) -> None:
        """A record with no cap fails, and says so in its own words.

        The distinction is for the reader, not for detection: an absent cap
        and a wrong one have different remedies, and a message reading as an
        equality mismatch sends the next person to inspect a value that was
        never transmitted. What the equality half cannot settle -- a declared
        cap against an inherited one, since the declared value IS the crate's
        default -- belongs to the Rust-side wiring trace and is not claimed
        here.
        """
        bare = {key: value for key, value in _ATTEMPT.items() if key != "cap"}
        document = _document(_envelope_with(completions=_seat_attempts(bare)))
        outcomes = _outcomes("S20", _result(document=document))
        self.assertEqual(Outcome.FAIL, outcomes[migration.S20_ASSERTIONS[1]])

    def test_the_absent_cap_message_names_presence_before_equality(self) -> None:
        """A failure that reads as an equality mismatch sends the next reader
        looking at the value rather than at the deleted call site."""
        bare = {key: value for key, value in _ATTEMPT.items() if key != "cap"}
        document = _document(_envelope_with(completions=_seat_attempts(bare)))
        detail = _details("S20", _result(document=document))
        self.assertIn("no cap was transmitted",
                      detail[migration.S20_ASSERTIONS[1]])
        self.assertNotIn("carries", detail[migration.S20_ASSERTIONS[1]])

    def test_the_crates_older_cap_fails_the_second(self) -> None:
        """4096 is what 3.2.0 shipped, and what the recorded failure ran under."""
        document = _document(
            _envelope_with(completions=_seat_attempts(dict(_ATTEMPT,
                                                           cap=4096))))
        outcomes = _outcomes("S20", _result(document=document))
        self.assertEqual(Outcome.FAIL, outcomes[migration.S20_ASSERTIONS[1]])

    def test_no_attempt_at_all_cannot_test_the_second(self) -> None:
        document = _document(_envelope_with(completions={}))
        outcomes = _outcomes("S20", _result(document=document))
        self.assertEqual(Outcome.CANNOT_TEST,
                         outcomes[migration.S20_ASSERTIONS[1]])

    def test_no_attempt_at_all_fails_the_third(self) -> None:
        """The second and the third disagree ON PURPOSE over the same input.

        With no record there is no transmitted cap to read, which is a
        ``CANNOT_TEST``; but "recorded per attempt" is precisely the claim that
        a completed consult leaves records, so the same emptiness is a FAIL
        there.
        """
        document = _document(_envelope_with(completions={}))
        outcomes = _outcomes("S20", _result(document=document))
        self.assertEqual(Outcome.FAIL, outcomes[migration.S20_ASSERTIONS[2]])

    def test_a_sixth_attempt_key_fails_the_third(self) -> None:
        """``CompletionRecord`` is ``#[non_exhaustive]`` and ``reasoning`` is
        deliberately not rendered: an extra key is as much a defect as a
        missing one, and an "at least five" would see neither."""
        document = _document(
            _envelope_with(completions=_seat_attempts(
                dict(_ATTEMPT, reasoning="disabled"))))
        outcomes = _outcomes("S20", _result(document=document))
        self.assertEqual(Outcome.FAIL, outcomes[migration.S20_ASSERTIONS[2]])

    def test_a_missing_attempt_key_fails_the_third(self) -> None:
        bare = {key: value for key, value in _ATTEMPT.items()
                if key != "prompt_tokens"}
        document = _document(_envelope_with(completions=_seat_attempts(bare)))
        outcomes = _outcomes("S20", _result(document=document))
        self.assertEqual(Outcome.FAIL, outcomes[migration.S20_ASSERTIONS[2]])

    def test_a_seat_series_that_is_not_an_array_fails_the_third(self) -> None:
        """The array under a seat IS the attempt series, and its length is a
        fact about the run: a per-seat total cannot be disaggregated back into
        which model was cut."""
        document = _document(
            _envelope_with(completions={"melchior": copy.deepcopy(_ATTEMPT)}))
        outcomes = _outcomes("S20", _result(document=document))
        self.assertEqual(Outcome.FAIL, outcomes[migration.S20_ASSERTIONS[2]])

    def test_several_attempts_for_one_seat_pass_the_third(self) -> None:
        """A rotating seat spends several attempts on several models, and each
        one is its own record."""
        document = _document(
            _envelope_with(completions=_seat_attempts(_ATTEMPT, _ATTEMPT,
                                                      _ATTEMPT)))
        outcomes = _outcomes("S20", _result(document=document))
        self.assertEqual(Outcome.PASS, outcomes[migration.S20_ASSERTIONS[2]])

    def test_a_threshold_no_factor_explains_fails_the_fourth(self) -> None:
        """The cross-check recomputes the threshold from the run's OWN
        rotation count, so it agrees with the product only when the product
        agrees with the formula."""
        caps = dict(_CAPS, floor_activation_threshold_secs=999)
        outcomes = _outcomes("S20", _result(document=_document(caps=caps)))
        self.assertEqual(Outcome.FAIL, outcomes[migration.S20_ASSERTIONS[3]])

    def test_a_threshold_for_a_different_rotation_count_fails_the_fourth(self):
        """114 is right for two rotations and wrong for none, and the check
        reads the count the run published rather than a configured one."""
        caps = dict(_CAPS, max_rotations_effective=0)
        outcomes = _outcomes("S20", _result(document=_document(caps=caps)))
        self.assertEqual(Outcome.FAIL, outcomes[migration.S20_ASSERTIONS[3]])

    def test_an_absent_wall_clock_cannot_test_the_fourth(self) -> None:
        caps = dict(_CAPS, timeout_secs=None)
        outcomes = _outcomes("S20", _result(document=_document(caps=caps)))
        self.assertEqual(Outcome.CANNOT_TEST,
                         outcomes[migration.S20_ASSERTIONS[3]])

    def test_a_provider_error_cannot_test_the_envelope_assertions(self) -> None:
        """The provider's failure, not the product's, so the three assertions
        that read the envelope degrade.

        The fourth does not, and that is right rather than an oversight:
        ``applied_caps`` is emitted whatever the trio did, so the arithmetic
        really was checked. Reporting it as not run would hide a cross-check
        that ran and held.
        """
        broken = _document(error={"kind": "provider", "message": "502"})
        broken["consult"] = None
        outcomes = _outcomes("S20", _result(document=broken, exit_code=1))
        self.assertEqual(list(migration.S20_ASSERTIONS), list(outcomes))
        for text in migration.S20_ASSERTIONS[:3]:
            self.assertEqual(Outcome.CANNOT_TEST, outcomes[text], text)
        self.assertEqual(Outcome.PASS, outcomes[migration.S20_ASSERTIONS[3]])

    def test_a_runtime_error_fails_the_first_three(self) -> None:
        """The product spoke and said no, which is a verdict about it. The
        fourth reads ``applied_caps``, which the run still published."""
        broken = _document(error={"kind": "runtime", "message": "boom"})
        broken["consult"] = None
        outcomes = _outcomes("S20", _result(document=broken, exit_code=1))
        for text in migration.S20_ASSERTIONS[:3]:
            self.assertEqual(Outcome.FAIL, outcomes[text], text)

    def test_a_missing_run_reports_all_four(self) -> None:
        outcomes = _outcomes("S20", None)
        self.assertEqual(list(migration.S20_ASSERTIONS), list(outcomes))
        self.assertEqual({Outcome.CANNOT_TEST}, set(outcomes.values()))


class S21Tests(unittest.TestCase):
    """The empty completion names itself, or the scenario names what was missing."""

    def _forced(self, attempt=None, hop=None, seat="melchior"):
        """A run in which one seat came back empty and rotated.

        Args:
            attempt: The empty attempt; None uses zero tokens and ``length``.
            hop: The rotation hop; None uses the healthy one.
            seat: Which seat the attempt and the hop belong to.

        Returns:
            RunResult: The double.
        """
        empty = (dict(_ATTEMPT, completion_tokens=0, finish="length")
                 if attempt is None else attempt)
        envelope = _envelope_with(
            completions={seat: [copy.deepcopy(empty)]},
            rotations=[{"agent": seat,
                        "model_configured": "glm-5.2:cloud",
                        "model_used": "qwen3.5:397b-cloud",
                        "ran_unmeasured": False,
                        "chain": [copy.deepcopy(_HOP if hop is None
                                                else hop)]}])
        return _result(document=_document(envelope))

    def test_a_forced_empty_completion_passes_all_three(self) -> None:
        with _payload():
            self.assertEqual({Outcome.PASS},
                             set(_outcomes("S21", self._forced()).values()))

    def test_a_transport_cause_fails_the_first(self) -> None:
        """``transport`` is the label a wildcard produces for a cause it does
        not know, and reading it here says the run blamed the network for
        something local to one mage."""
        with _payload():
            outcomes = _outcomes(
                "S21",
                self._forced(hop=dict(_HOP,
                                      cause=migration.TRANSPORT_CAUSE)))
        self.assertEqual(Outcome.FAIL, outcomes[migration.S21_ASSERTIONS[0]])

    def test_the_transport_failure_says_what_transport_would_mean(self) -> None:
        with _payload():
            detail = _details(
                "S21",
                self._forced(hop=dict(_HOP,
                                      cause=migration.TRANSPORT_CAUSE)))
        self.assertIn("blames the network",
                      detail[migration.S21_ASSERTIONS[0]])

    def test_a_run_wide_locality_fails_the_second(self) -> None:
        """``mage_local`` false says a cause condemned the whole run, which is
        what decides whether the other two seats keep going."""
        with _payload():
            outcomes = _outcomes("S21",
                                 self._forced(hop=dict(_HOP,
                                                       mage_local=False)))
        self.assertEqual(Outcome.FAIL, outcomes[migration.S21_ASSERTIONS[1]])

    def test_an_absent_locality_flag_fails_the_second(self) -> None:
        hop = {key: value for key, value in _HOP.items() if key != "mage_local"}
        with _payload():
            outcomes = _outcomes("S21", self._forced(hop=hop))
        self.assertEqual(Outcome.FAIL, outcomes[migration.S21_ASSERTIONS[1]])

    def test_a_genuinely_empty_answer_passes_the_third(self) -> None:
        """``stop`` on a zero-token attempt is the backend saying the model
        finished and produced nothing, which is the other side of the
        distinction the assertion is about."""
        with _payload():
            outcomes = _outcomes(
                "S21",
                self._forced(attempt=dict(_ATTEMPT, completion_tokens=0,
                                          finish="stop")))
        self.assertEqual(Outcome.PASS, outcomes[migration.S21_ASSERTIONS[2]])

    def test_an_unreported_finish_cannot_test_the_third(self) -> None:
        """``null`` means the backend did not say why the model stopped.

        Substituting a default there would assert a measurement nobody took --
        the confusion that made a completion scraping its cap look healthy --
        so the run simply cannot answer.
        """
        with _payload():
            outcomes = _outcomes(
                "S21",
                self._forced(attempt=dict(_ATTEMPT, completion_tokens=0,
                                          finish=None)))
        self.assertEqual(Outcome.CANNOT_TEST,
                         outcomes[migration.S21_ASSERTIONS[2]])

    def test_an_absent_finish_key_fails_the_third(self) -> None:
        attempt = {key: value for key, value in _ATTEMPT.items()
                   if key != "finish"}
        attempt["completion_tokens"] = 0
        with _payload():
            outcomes = _outcomes("S21", self._forced(attempt=attempt))
        self.assertEqual(Outcome.FAIL, outcomes[migration.S21_ASSERTIONS[2]])

    def test_a_finish_outside_the_vocabulary_fails_the_third(self) -> None:
        with _payload():
            outcomes = _outcomes(
                "S21",
                self._forced(attempt=dict(_ATTEMPT, completion_tokens=0,
                                          finish=17)))
        self.assertEqual(Outcome.FAIL, outcomes[migration.S21_ASSERTIONS[2]])

    def test_an_empty_completion_with_no_rotation_cannot_test_the_cause(self):
        """No cause was published, so there is nothing to read -- which is a
        different report from a cause that read wrong."""
        envelope = _envelope_with(
            completions=_seat_attempts(dict(_ATTEMPT, completion_tokens=0)))
        with _payload():
            outcomes = _outcomes("S21", _result(document=_document(envelope)))
        self.assertEqual(Outcome.CANNOT_TEST,
                         outcomes[migration.S21_ASSERTIONS[0]])

    def test_a_model_that_finished_cannot_test_and_says_so(self) -> None:
        """The recipe's most likely legitimate failure at a 16384 cap.

        The product did nothing wrong and the environment would not produce
        the state, so this is CANNOT_TEST -- and the report names the
        condition, because a generic reason is the vacuous outcome wearing an
        honest label.
        """
        with _payload():
            outcomes = _outcomes("S21", _result())
            detail = _details("S21", _result())
        self.assertEqual({Outcome.CANNOT_TEST}, set(outcomes.values()))
        self.assertIn("completed within the declared cap",
                      detail[migration.S21_ASSERTIONS[0]])

    def test_an_exhausted_budget_that_still_answered_reports_itself(self) -> None:
        """Distinct from the previous one: the cap DID run out and the model
        answered anyway, which is evidence about the cap rather than about the
        models configured."""
        document = _document(
            _envelope_with(completions=_seat_attempts(
                dict(_ATTEMPT, finish="length", completion_tokens=16384))))
        with _payload():
            outcomes = _outcomes("S21", _result(document=document))
            detail = _details("S21", _result(document=document))
        self.assertEqual({Outcome.CANNOT_TEST}, set(outcomes.values()))
        self.assertIn("exhausted the declared cap",
                      detail[migration.S21_ASSERTIONS[0]])

    def test_unreported_token_counts_report_the_recipe_unobservable(self) -> None:
        document = _document(
            _envelope_with(completions=_seat_attempts(
                dict(_ATTEMPT, completion_tokens=None))))
        with _payload():
            detail = _details("S21", _result(document=document))
        self.assertIn("no attempt reported a completion token count",
                      detail[migration.S21_ASSERTIONS[0]])

    def test_no_attempt_at_all_reports_the_recipe_unobservable(self) -> None:
        document = _document(_envelope_with(completions={}))
        with _payload():
            outcomes = _outcomes("S21", _result(document=document))
            detail = _details("S21", _result(document=document))
        self.assertEqual({Outcome.CANNOT_TEST}, set(outcomes.values()))
        self.assertIn("recorded no completion attempt",
                      detail[migration.S21_ASSERTIONS[0]])

    def test_a_payload_that_was_never_sent_reports_the_recipe_not_applied(self):
        """The recipe's ONE load-bearing property is the token count.

        A run that carried its bare prompt never made any model reason at
        length, so nothing about it says whether an empty completion names
        itself -- and blaming the product for a size it was never sent is what
        the harness's own trap text forbids.
        """
        with _payload():
            outcomes = _outcomes("S21", self._forced_short())
            detail = _details("S21", self._forced_short())
        self.assertEqual({Outcome.CANNOT_TEST}, set(outcomes.values()))
        self.assertIn("the recipe was not applied",
                      detail[migration.S21_ASSERTIONS[0]])

    def _forced_short(self):
        """A run that forced the state but never carried the payload.

        Returns:
            RunResult: The double, with a bare prompt's worth of stdin.
        """
        forced = self._forced()
        return _result(document=json.loads(forced.output.stdout),
                       stdin_bytes=54)

    def test_the_four_cannot_test_reasons_are_all_different(self) -> None:
        """A bare CANNOT_TEST with a generic reason is the vacuous outcome
        wearing an honest label, so the conditions that differ must read
        differently."""
        documents = {
            "completed": _document(),
            "exhausted": _document(_envelope_with(
                completions=_seat_attempts(dict(_ATTEMPT, finish="length",
                                                completion_tokens=16384)))),
            "unmeasured": _document(_envelope_with(
                completions=_seat_attempts(dict(_ATTEMPT,
                                                completion_tokens=None)))),
            "no attempt": _document(_envelope_with(completions={})),
        }
        seen = set()
        with _payload():
            for label, document in documents.items():
                with self.subTest(condition=label):
                    detail = _details("S21", _result(document=document))
                    seen.add(detail[migration.S21_ASSERTIONS[0]])
        self.assertEqual(len(documents), len(seen))

    def test_a_provider_error_cannot_test_every_assertion(self) -> None:
        broken = _document(error={"kind": "provider", "message": "502"})
        broken["consult"] = None
        with _payload():
            outcomes = _outcomes("S21", _result(document=broken, exit_code=1))
        self.assertEqual(list(migration.S21_ASSERTIONS), list(outcomes))
        self.assertEqual({Outcome.CANNOT_TEST}, set(outcomes.values()))

    def test_a_missing_run_reports_all_three(self) -> None:
        with _payload():
            outcomes = _outcomes("S21", None)
        self.assertEqual(list(migration.S21_ASSERTIONS), list(outcomes))
        self.assertEqual({Outcome.CANNOT_TEST}, set(outcomes.values()))


class S22Tests(unittest.TestCase):
    """The report is complete: present keys, derivable notions, an unchanged bit."""

    def test_a_healthy_run_passes_all_three(self) -> None:
        self.assertEqual({Outcome.PASS},
                         set(_outcomes("S22", _result()).values()))

    def test_an_empty_eligibility_snapshot_still_passes_the_first(self) -> None:
        """Emptiness is the positive certificate that the snapshot ran and
        rejected nothing, so it is a PASS and only absence is a failure."""
        document = _document(_envelope_with(pool_eligibility={}))
        outcomes = _outcomes("S22", _result(document=document))
        self.assertEqual(Outcome.PASS, outcomes[migration.S22_ASSERTIONS[0]])

    def test_an_absent_eligibility_snapshot_fails_the_first(self) -> None:
        """Absent says "not computed"; empty says "computed, nothing to
        reject". A run with no pool declared and one with a healthy pool would
        otherwise read identically."""
        envelope = copy.deepcopy(_ENVELOPE)
        del envelope["pool_eligibility"]
        outcomes = _outcomes("S22", _result(document=_document(envelope)))
        self.assertEqual(Outcome.FAIL, outcomes[migration.S22_ASSERTIONS[0]])

    def test_a_null_eligibility_snapshot_fails_the_first(self) -> None:
        document = _document(_envelope_with(pool_eligibility=None))
        outcomes = _outcomes("S22", _result(document=document))
        self.assertEqual(Outcome.FAIL, outcomes[migration.S22_ASSERTIONS[0]])

    def test_a_rotation_without_both_model_names_fails_the_second(self) -> None:
        """A mage that fell to another backend is derived from
        ``model_configured != model_used``, and one of the two missing makes
        the comparison unanswerable rather than false."""
        document = _document(_envelope_with(rotations=[
            {"agent": "melchior", "model_configured": "glm-5.2:cloud",
             "ran_unmeasured": False, "chain": []}]))
        outcomes = _outcomes("S22", _result(document=document))
        self.assertEqual(Outcome.FAIL, outcomes[migration.S22_ASSERTIONS[1]])

    def test_a_rotation_that_did_fall_back_still_passes_the_second(self) -> None:
        """The assertion is derivABILITY, not health: a seat that really
        rotated publishes both names and is exactly the case the derivation
        exists for."""
        document = _document(_envelope_with(rotations=[
            {"agent": "melchior", "model_configured": "glm-5.2:cloud",
             "model_used": "qwen3.5:397b-cloud", "ran_unmeasured": False,
             "chain": [copy.deepcopy(_HOP)]}]))
        outcomes = _outcomes("S22", _result(document=document))
        self.assertEqual(Outcome.PASS, outcomes[migration.S22_ASSERTIONS[1]])

    def test_an_attempt_without_finish_fails_the_second(self) -> None:
        """Truncation is derived from ``finish == "length"``, so the key has to
        be there even when it is null."""
        bare = {key: value for key, value in _ATTEMPT.items()
                if key != "finish"}
        document = _document(_envelope_with(completions=_seat_attempts(bare)))
        outcomes = _outcomes("S22", _result(document=document))
        self.assertEqual(Outcome.FAIL, outcomes[migration.S22_ASSERTIONS[1]])

    def test_a_null_finish_still_passes_the_second(self) -> None:
        """Null is the backend not having said, and it is still a published
        key: what the assertion checks is that the notion HAS somewhere to be
        derived from."""
        document = _document(_envelope_with(
            completions=_seat_attempts(dict(_ATTEMPT, finish=None))))
        outcomes = _outcomes("S22", _result(document=document))
        self.assertEqual(Outcome.PASS, outcomes[migration.S22_ASSERTIONS[1]])

    def test_a_non_boolean_degraded_fails_the_second(self) -> None:
        document = _document(_envelope_with(degraded="no"))
        outcomes = _outcomes("S22", _result(document=document))
        self.assertEqual(Outcome.FAIL, outcomes[migration.S22_ASSERTIONS[1]])

    def test_no_attempt_at_all_cannot_test_the_second(self) -> None:
        document = _document(_envelope_with(completions={}))
        outcomes = _outcomes("S22", _result(document=document))
        self.assertEqual(Outcome.CANNOT_TEST,
                         outcomes[migration.S22_ASSERTIONS[1]])

    def test_a_degraded_three_verdict_run_fails_the_third(self) -> None:
        document = _document(_envelope_with(degraded=True))
        outcomes = _outcomes("S22", _result(document=document))
        self.assertEqual(Outcome.FAIL, outcomes[migration.S22_ASSERTIONS[2]])

    def test_a_two_verdict_run_cannot_test_the_third(self) -> None:
        """On a run that produced fewer verdicts a ``degraded`` of true is the
        bit WORKING, so asserting false there would assert the opposite of the
        contract."""
        envelope = copy.deepcopy(_ENVELOPE)
        envelope["agents"] = envelope["agents"][:2]
        envelope["degraded"] = True
        outcomes = _outcomes("S22", _result(document=_document(envelope)))
        self.assertEqual(Outcome.CANNOT_TEST,
                         outcomes[migration.S22_ASSERTIONS[2]])

    def test_a_flagless_run_cannot_test_the_third(self) -> None:
        envelope = copy.deepcopy(_ENVELOPE)
        del envelope["agents"]
        outcomes = _outcomes("S22", _result(document=_document(envelope)))
        self.assertEqual(Outcome.CANNOT_TEST,
                         outcomes[migration.S22_ASSERTIONS[2]])

    def test_a_provider_error_cannot_test_every_assertion(self) -> None:
        broken = _document(error={"kind": "provider", "message": "502"})
        broken["consult"] = None
        outcomes = _outcomes("S22", _result(document=broken, exit_code=1))
        self.assertEqual(list(migration.S22_ASSERTIONS), list(outcomes))
        self.assertEqual({Outcome.CANNOT_TEST}, set(outcomes.values()))

    def test_a_missing_run_reports_all_three(self) -> None:
        outcomes = _outcomes("S22", None)
        self.assertEqual(list(migration.S22_ASSERTIONS), list(outcomes))
        self.assertEqual({Outcome.CANNOT_TEST}, set(outcomes.values()))


if __name__ == "__main__":
    unittest.main()
