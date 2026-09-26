# Author: Julian Bolivar
# Version: 0.20.0
# Date: 2026-09-26
"""S20, S21, S22, S25 and S26 -- the reasoning-and-migration scenarios.

S20, S21 and S22 are the three scenarios the magi-core 4.0.0 move needs. S25
and S26 (v0.20.0, REQ-EE-1/EE-3) check that the reasoning instrumentation
round-trips: S26 reads R4 alongside them (no ``reasoning`` key declared, so
every attempt reports the default control); S25 declares its own standalone
run, on its own scratch ``magi.toml``, because its property -- ``reasoning =
"disabled"`` round-tripping -- has to hold on the OPERATOR's configured trio,
never on a value this module invents.

All three read **R4**, the trio consult that already carries the large
deterministic payload. Nothing here declares a run of its own, and that is the
same fusion argument ``trio.py`` records: R4 is the harness's most expensive
invocation, the three properties below are orthogonal to each other, and a
fourth trio run would multiply the only expensive thing the harness does. The
coupling is declared rather than hidden -- if R4 falls, all three fall with it,
and every finding carries R4's id so a reader sees one cause rather than three
defects.

**S20 is the milestone's highest-value scenario because nothing else can see
its subject.** magi-core 4.0.0 moved the trio off ``POST /v1/chat/completions``
and onto ``POST {base}/api/chat``. That happened with no code change here and
no compile error: the pinned crate's ``OllamaProvider`` stopped delegating to
an inner OpenAI-compatible provider and started posting to its own route. A
unit suite cannot observe a protocol it never speaks, so only a run against a
real daemon says whether the product still completes.

**The cap is read from what the run TRANSMITTED, never from magi-rs's own
configuration echoed back at itself.** A cap that is declared and never reaches
the wire passes every unit test there is, so the only place it can be observed
is the per-attempt record the backend's answer produced.

**What this assertion could NOT prove through the 4.1.0 pin, and now can.**
Through magi-core 4.1.0 the declared cap was numerically the crate's own
default, so a run whose call site was deleted transmitted the same number and
this assertion stayed green regardless. magi-core 4.2.0 moved its own default
to 32768 while magi-rs's declared cap stays 16384 (v0.20.0 changes no default,
REQ-EE-5), so a deleted call site now transmits a DIFFERENT number and the
equality half catches it on its own -- see
``test_the_crates_own_default_cap_fails_the_second``. Distinguishing a
declared cap from an inherited one is still, independently, a question about
the BUILDER, answered by the Rust-side wiring trace, which asserts the builder
was handed a configuration rather than a value; that trace is not made
redundant by the pin move, it is corroborated by it. S20's share of the work
is the other half: that whatever was configured actually travelled. The
per-attempt ``cap`` is therefore asserted present first and equal second, so a
record carrying no cap is reported as nothing having been transmitted rather
than as a value that failed to match -- the two send the next reader to
different places.

**S21 asserts a SHAPE and forces nothing, and that is a redesign made on
measurement rather than on taste.** It used to try to CAUSE an empty completion
-- a reasoning model spending its whole output budget on thought against a
250 kB payload -- and then assert the product named the state correctly. Six
probes of ``deepseek-v4-pro:cloud`` on 2026-08-27 killed the recipe: the same
prompt, the same payload and the same 16 384-token cap gave ``length`` with
16 384 tokens and empty content on one run and ``stop`` with 10 312 on the
next. Cloud inference is not reproducible at ``temperature: 0`` and that model
sits exactly on the boundary; ``minimax-m3:cloud`` is the same coin flip, only
slower.

**That made S21 an assertion about the MODEL rather than about the product**,
which this harness's own doctrine forbids: a gate cannot go red over something
the product does not control. It was worse than red, in fact -- the three
outcomes it reported were ``CANNOT_TEST``, which BLOCKS certification, so
identical code would have been certified on some runs and refused on others.

So S21 now asserts what the product emits on **every** run, whether or not any
model overruns: that each recorded attempt carries a ``finish`` this build
knows -- one of :data:`KNOWN_FINISH_LABELS`, or an explicit JSON null meaning
*"the backend did not say"* -- and that the rotation report is published with
every hop naming a cause :data:`KNOWN_ROTATION_CAUSES` holds and saying whether
that cause was local to one mage.

**What was given up is real and is not hidden.** There is no longer an
end-to-end demonstration that a genuinely overrunning model is reported
correctly. That claim keeps its unit proof -- the mapping is a pure function
over a constructed ``CompletionRecord`` and is provable there and only there --
and a real run that happens to overrun still exercises it, opportunistically
and unasserted. What S21 buys back is a scenario whose colour is decided by
this build's code.

**Two assertions, not three, and the count moved with it.** The spec names
exactly two properties for S21, and a third invented to keep the number would
be padding -- which is the defect this redesign exists to remove, not a shape
to preserve. ``DECLARED_ASSERTION_COUNT`` drops by one accordingly.

**S22 asks whether the report is COMPLETE, not whether the run was healthy.**
Its subject is published keys: ``pool_eligibility`` present even when empty,
because absent and empty are different facts -- *"not computed"* against
*"computed, nothing to reject"* -- and the three separate notions of a degraded
run each derivable from something the JSON carries. A mage that fell to another
backend shows as ``model_configured != model_used``; truncation shows as
``finish == "length"``; fewer than three verdicts stays where it already was,
in the ``degraded`` bit, whose meaning this milestone does not touch.
"""

import dataclasses
import pathlib
import re
import shutil
import tempfile

from smoke import runs
from smoke.errors import ProductOutputError
from smoke.outcome import Finding, Outcome
from smoke.registry import scenario
# The budget arithmetic and the shared vocabulary of the headless output
# contract live in ``trio``, and they are imported rather than copied. A second
# transcription of the attempt-factor formula is precisely the second source of
# truth the harness's own doctrine warns about: both copies would keep passing
# while they disagreed with the product. Imported by FULL module path so the
# order of the imports in ``scenarios/__init__`` cannot decide whether this
# module loads.
from smoke.scenarios.trio import (AGENTS_KEY, APPLIED_CAPS_KEY, CONSULT_KEY,
                                  DEGRADED_KEY, ENVIRONMENTAL_ERROR_KINDS,
                                  ERROR_KEY, SUCCESS_EXIT_CODE, TRIO_SIZE,
                                  attempt_factor, derive,
                                  floor_activation_threshold)

#: The verbatim assertion texts, one tuple per scenario. Declared as the
#: module's own constants and handed to the decorator by name: a literal there
#: would be a second copy the completeness check cannot see drifting.
S20_ASSERTIONS = (
    "the trio completed against the native wire",
    "every transmitted attempt carries the declared completion cap",
    "completions are recorded per attempt",
    "the published per-mage threshold agrees with the attempt-factor formula",
)
S21_ASSERTIONS = (
    "every recorded attempt reports a finish this build knows, or an explicit "
    "null",
    "the rotations report is published and every hop names a known cause and "
    "its locality",
)
S22_ASSERTIONS = (
    "pool_eligibility is present even when empty",
    "all three notions of degradation are derivable from published keys",
    "degraded is false for a three-verdict run",
)
S25_ASSERTIONS = (
    "a consult under reasoning = disabled reports the disabled control on "
    "every recorded attempt",
    "under reasoning = disabled no attempt reports a measured reasoning count "
    "above zero",
    "no attempt record carries reasoning text, even with the trace opted in",
)
S26_ASSERTIONS = (
    "with no reasoning key declared every recorded attempt reports the "
    "default control",
    "every recorded attempt reports its reasoning state in magi-core's own "
    "shape",
)

#: The run all three read. See the module docstring for why there is one.
MIGRATION_RUN = "R4"

#: The completion cap magi-rs DECLARES, mirrored from the product's own
#: constant. The maintenance contract is accepted rather than re-litigated: a
#: cap the product moves turns S20 red and forces a change here, and a check
#: that adjusted itself to whatever the product reported would detect nothing.
#: Through magi-core 4.1.0 this value was numerically the crate's own default
#: too, which is what bounded what this scenario could prove -- see the module
#: docstring. magi-core 4.2.0 moved its own default to 32768 while this stays
#: 16384 (v0.20.0 changes no default, REQ-EE-5), so the two are now
#: distinguishable and the equality half of assertion 2 catches a deleted call
#: site on its own.
DECLARED_COMPLETION_CAP = 16384

#: Keys of the consult envelope this module reaches for.
COMPLETIONS_KEY = "completions"
POOL_ELIGIBILITY_KEY = "pool_eligibility"
ROTATIONS_KEY = "rotations"
CHAIN_KEY = "chain"
MODEL_CONFIGURED_KEY = "model_configured"
MODEL_USED_KEY = "model_used"

#: Keys of one rotation hop.
CAUSE_KEY = "cause"
MAGE_LOCAL_KEY = "mage_local"

#: The seven keys one completion attempt exposes, mapped field by field in
#: ``src/magi/completion_report.rs``: the five of 0.19.x plus ``reasoning``
#: and ``control``, which REQ-EE-1 adds in v0.20.0. Counted EXACTLY, for the
#: same reason S18 counts the verdict keys exactly: ``CompletionRecord`` is
#: ``#[non_exhaustive]``, so a field magi-core adds in a minor release reaches
#: this public JSON the moment somebody replaces the explicit mapping with a
#: direct interpolation, and an "at least seven" detects none of it -- an
#: unexpected key is as much a defect here as a missing one.
ATTEMPT_KEYS = ("model", "cap", "finish", "completion_tokens", "prompt_tokens",
                "reasoning", "control")
CAP_KEY = "cap"
FINISH_KEY = "finish"

#: Every finish label this build can produce for a reason it RECOGNISES,
#: mirrored from ``FinishReason``'s hand-written ``Serialize`` in the pinned
#: magi-core. Three, and no more: the crate's own capture campaign observed
#: exactly ``stop``, ``length`` and ``load``, and its ``from_wire`` folds every
#: word any vendor publishes -- ``tool_calls``, ``end_turn``, ``refusal``,
#: ``max_tokens``, ``model_context_window_exceeded`` and the rest -- into one
#: of these three before a record is ever built.
#:
#: A fourth string therefore is not a vendor word magi-rs failed to anticipate;
#: it is ``FinishReason::Other``, which the crate documents as *"a value no
#: vendor publishes"*. Reporting that is signal about the wire, and it is
#: DETERMINISTIC for a given backend -- which is the whole difference between
#: this check and the forcing recipe it replaced.
KNOWN_FINISH_LABELS = ("stop", "length", "load")

#: Every rotation cause this build can name, mirrored from ``cause_label`` in
#: ``src/magi/rotation_report.rs``. Seven, matching ``RotationKind``'s variants
#: one for one at the pin. The mirroring is deliberate and carries the same
#: maintenance contract as :data:`DECLARED_COMPLETION_CAP`: a cause magi-core
#: adds turns S21 red until this tuple moves, which is the point. A check that
#: accepted whatever string arrived would accept the wildcard label a
#: hand-written match invents, and that wildcard is exactly what REQ-V4-02
#: forbids.
KNOWN_ROTATION_CAUSES = ("transport", "timeout", "schema", "oversized_response",
                         "external_failure", "empty_completion",
                         "response_contract")

#: Where the per-mage threshold S20 cross-checks is published.
THRESHOLD_KEY = "floor_activation_threshold_secs"
TIMEOUT_KEY = "timeout_secs"
ROTATIONS_EFFECTIVE_KEY = "max_rotations_effective"


def _is_count(value):
    """Whether a JSON value is a usable non-negative integer.

    Args:
        value: The value to check.

    Returns:
        bool: True for a non-negative ``int`` that is not a bool. JSON's
        booleans are Python ints, and one arriving where a count belongs is a
        contract break rather than a zero.
    """
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


@dataclasses.dataclass(frozen=True)
class _Attempt:
    """One recorded completion attempt, with the seat it belongs to.

    Attributes:
        seat: The lowercase seat label the record was filed under.
        index: Its position in that seat's attempt series, so a message can
            name which attempt without the reader counting.
        record: The attempt object as published.
    """

    seat: str
    index: int
    record: object

    @property
    def label(self):
        """How this attempt is named in a finding's detail.

        Returns:
            str: ``"melchior[0]"`` and so on.
        """
        return "%s[%d]" % (self.seat, self.index)


class _Capture:
    """One run reduced to what these scenarios read, or to why there is none.

    Attributes:
        envelope: The consult envelope, or None.
        exit_code: What the product exited with, or None when there was no
            capture at all.
        outcome: What every assertion over this run must report when
            *envelope* is None.
        detail: Why.
    """

    def __init__(self, envelope, exit_code=None, outcome=None, detail=""):
        """Store the reduction.

        Args:
            envelope: The consult envelope, or None.
            exit_code: The product's exit code, when there was a capture.
            outcome: The outcome to inherit; None when *envelope* is set.
            detail: The cause.
        """
        self.envelope = envelope
        self.exit_code = exit_code
        self.outcome = outcome
        self.detail = detail


def _capture_of(result, run_id):
    """Reduce one shared run to its consult envelope or to a shared cause.

    The classification is the one ``trio`` already records: a product error of
    its own is the product having spoken, an unreachable or failing backend is
    the environment refusing, and a capture that cannot be read at all is a
    broken output contract.

    Args:
        result: The run's ``RunResult``, or None.
        run_id: Which run, for the message.

    Returns:
        _Capture: The envelope, or the outcome every assertion inherits.
    """
    if result is None:
        return _Capture(None, None, Outcome.CANNOT_TEST,
                        "run %s produced no capture to inspect" % run_id)
    try:
        document = result.output.json()
    except ProductOutputError as exc:
        return _Capture(None, result.output.exit_code, Outcome.FAIL,
                        "run %s: %s" % (run_id, exc))
    error = document.get(ERROR_KEY)
    if isinstance(error, dict):
        kind = error.get("kind")
        outcome = (Outcome.CANNOT_TEST if kind in ENVIRONMENTAL_ERROR_KINDS
                   else Outcome.FAIL)
        return _Capture(None, result.output.exit_code, outcome,
                        "run %s reported a %s error: %s"
                        % (run_id, kind, error.get("message", "")))
    envelope = document.get(CONSULT_KEY)
    if not isinstance(envelope, dict):
        return _Capture(None, result.output.exit_code, Outcome.FAIL,
                        "run %s exited %d but carries no consult envelope, so "
                        "the trio produced nothing to read"
                        % (run_id, result.output.exit_code))
    return _Capture(envelope, result.output.exit_code)


def _attempts_of(envelope):
    """Flatten every recorded completion attempt across every seat.

    Complexity: ``O(seats x attempts)`` -- one pass, over a trio.

    Args:
        envelope: The consult envelope, or None.

    Returns:
        tuple: ``(attempts, failure)``. *attempts* is the flattened list in
        seat order, possibly empty; it is None when ``completions`` cannot be
        read as a per-seat map at all, and *failure* then says what was found
        instead. An EMPTY list is a different answer from None: it says the
        map was readable and held nothing.
    """
    if envelope is None:
        return None, "there is no consult envelope to read %s from" % (
            COMPLETIONS_KEY,)
    records = envelope.get(COMPLETIONS_KEY)
    if not isinstance(records, dict):
        return None, ("%s is %s, expected an object keyed by seat"
                      % (COMPLETIONS_KEY, type(records).__name__))
    flattened = []
    for seat in sorted(records):
        series = records[seat]
        if not isinstance(series, list):
            return None, ("%s.%s is %s, expected the seat's attempt series as "
                          "an array" % (COMPLETIONS_KEY, seat,
                                        type(series).__name__))
        for index, record in enumerate(series):
            flattened.append(_Attempt(seat=seat, index=index, record=record))
    return flattened, ""


def _finding(texts, index, outcome, detail):
    """Build one finding against the shared run.

    Args:
        texts: The scenario's assertion texts.
        index: Position in *texts*.
        outcome: What became of it.
        detail: The cause when the outcome is not PASS.

    Returns:
        Finding: The finding, carrying R4's id.
    """
    return Finding(assertion=texts[index], outcome=outcome, detail=detail,
                   run_id=MIGRATION_RUN)


@scenario("S20", assertions=S20_ASSERTIONS, run=MIGRATION_RUN,
          needs_backend=True)
def the_trio_completes_on_the_native_wire(run):
    """Assert the trio answered, and that the cap it carried was ours.

    Args:
        run: R4's ``RunResult``, or None.

    Yields:
        Finding: One per entry of :data:`S20_ASSERTIONS`, in that order.
    """
    capture = _capture_of(run, MIGRATION_RUN)
    attempts, failure = _attempts_of(capture.envelope)
    yield _wire_finding(capture)
    yield _cap_finding(capture, attempts, failure)
    yield _per_attempt_finding(capture, attempts, failure)
    yield _threshold_finding(run)


def _wire_finding(capture):
    """Judge assertion 1: the trio answered end to end.

    A verdict cannot exist without a completed round trip, whatever route the
    pinned crate takes, so a seat's verdict is the evidence that the wire
    carried the traffic. That is the property the protocol move puts at risk
    and the one no unit test can reach.

    Args:
        capture: R4's reduction.

    Returns:
        Finding: PASS when the product exited zero and at least one seat
        produced a verdict.
    """
    if capture.envelope is None:
        return _finding(S20_ASSERTIONS, 0, capture.outcome, capture.detail)
    if capture.exit_code != SUCCESS_EXIT_CODE:
        return _finding(S20_ASSERTIONS, 0, Outcome.FAIL,
                        "the product exited %d" % capture.exit_code)
    agents = capture.envelope.get(AGENTS_KEY)
    if not isinstance(agents, list):
        return _finding(S20_ASSERTIONS, 0, Outcome.CANNOT_TEST,
                        "the run carried no structured verdicts, so there is "
                        "no per-seat evidence that the wire answered")
    if not agents:
        return _finding(S20_ASSERTIONS, 0, Outcome.FAIL,
                        "the consult completed and no seat produced a "
                        "verdict, so nothing crossed the wire")
    return _finding(S20_ASSERTIONS, 0, Outcome.PASS, "")


def _cap_finding(capture, attempts, failure):
    """Judge assertion 2: present first, equal second, in that order.

    What is checked is the cap the run TRANSMITTED, taken from the record the
    backend's answer produced -- never magi-rs's own configuration read back,
    which would agree with itself whatever reached the wire.

    The order matters for the report rather than for detection: an absent
    ``cap`` and a wrong one are different events with different remedies, and
    collapsing them into one equality mismatch sends the reader to inspect a
    value that was never there. What the equality half used to be unable to
    prove, through the 4.1.0 pin, is recorded in the module docstring: it
    coincided numerically with the crate's own default, so a deleted call site
    transmitted the same number; that coincidence ended at the 4.2.0 pin. The
    builder question -- was the wire handed a configuration or a bare value --
    stays the Rust-side wiring trace's, independently of what this half can
    now also catch.

    Args:
        capture: R4's reduction.
        attempts: Every recorded attempt, or None.
        failure: Why there are none.

    Returns:
        Finding: PASS when every attempt records the declared cap.
    """
    if capture.envelope is None:
        return _finding(S20_ASSERTIONS, 1, capture.outcome, capture.detail)
    if attempts is None:
        return _finding(S20_ASSERTIONS, 1, Outcome.FAIL, failure)
    if not attempts:
        return _finding(S20_ASSERTIONS, 1, Outcome.CANNOT_TEST,
                        "the run recorded no completion attempt, so no "
                        "transmitted cap can be read")
    absent = [item.label for item in attempts
              if not isinstance(item.record, dict)
              or CAP_KEY not in item.record]
    if absent:
        return _finding(S20_ASSERTIONS, 1, Outcome.FAIL,
                        "no cap was transmitted for %s, so there is no "
                        "recorded value to compare against the declared %d"
                        % (", ".join(absent), DECLARED_COMPLETION_CAP))
    wrong = ["%s carries %r" % (item.label, item.record[CAP_KEY])
             for item in attempts
             if item.record[CAP_KEY] != DECLARED_COMPLETION_CAP]
    if wrong:
        return _finding(S20_ASSERTIONS, 1, Outcome.FAIL,
                        "the declared cap is %d: %s"
                        % (DECLARED_COMPLETION_CAP, "; ".join(wrong)))
    return _finding(S20_ASSERTIONS, 1, Outcome.PASS, "")


def _per_attempt_finding(capture, attempts, failure):
    """Judge assertion 3: one record per attempt, with exactly seven keys.

    A per-seat total cannot be disaggregated back into which model was cut, so
    the array under a seat label is the attempt series and its length is a fact
    about the run. The key count is EXACT in both directions: an eighth key is
    as much a defect as a missing one, because ``CompletionRecord`` is
    ``#[non_exhaustive]`` -- ``reasoning`` and ``control`` are the two REQ-EE-1
    adds in v0.20.0, rendered exactly, never a wildcard for whatever comes
    after them.

    Args:
        capture: R4's reduction.
        attempts: Every recorded attempt, or None.
        failure: Why there are none.

    Returns:
        Finding: PASS when every attempt is an object of exactly
        :data:`ATTEMPT_KEYS`.
    """
    if capture.envelope is None:
        return _finding(S20_ASSERTIONS, 2, capture.outcome, capture.detail)
    if attempts is None:
        return _finding(S20_ASSERTIONS, 2, Outcome.FAIL, failure)
    if not attempts:
        return _finding(S20_ASSERTIONS, 2, Outcome.FAIL,
                        "the consult completed and recorded no completion "
                        "attempt at all, so nothing was recorded per attempt")
    wanted = set(ATTEMPT_KEYS)
    problems = []
    for item in attempts:
        if not isinstance(item.record, dict):
            problems.append("%s is not an object" % item.label)
            continue
        missing = sorted(wanted - set(item.record))
        unexpected = sorted(set(item.record) - wanted)
        if missing or unexpected:
            problems.append(
                "%s has %d keys, expected %d: missing [%s], unexpected [%s]"
                % (item.label, len(item.record), len(wanted),
                   ", ".join(missing), ", ".join(unexpected)))
    if problems:
        return _finding(S20_ASSERTIONS, 2, Outcome.FAIL, "; ".join(problems))
    return _finding(S20_ASSERTIONS, 2, Outcome.PASS, "")


def _threshold_finding(run):
    """Judge assertion 4: the published threshold is the formula's own.

    ``floor_activation_threshold_secs`` is a function of the attempt factor
    alone, and the factor is a function of the rotation count the same object
    publishes. So the run's telemetry can be checked against the formula
    without reading a single setting from the harness's configuration -- which
    is what makes it a cross-check rather than an echo.

    Args:
        run: R4's ``RunResult``, or None.

    Returns:
        Finding: PASS when some admissible attempt factor reproduces the
        published threshold.
    """
    if run is None:
        return _finding(S20_ASSERTIONS, 3, Outcome.CANNOT_TEST,
                        "run %s produced no capture to inspect"
                        % MIGRATION_RUN)
    try:
        caps = run.output.key(APPLIED_CAPS_KEY)
    except ProductOutputError as exc:
        return _finding(S20_ASSERTIONS, 3, Outcome.CANNOT_TEST, str(exc))
    if not isinstance(caps, dict):
        return _finding(S20_ASSERTIONS, 3, Outcome.FAIL,
                        "%s is %s, expected an object"
                        % (APPLIED_CAPS_KEY, type(caps).__name__))
    rotations = caps.get(ROTATIONS_EFFECTIVE_KEY)
    if not _is_count(caps.get(TIMEOUT_KEY)) or not _is_count(rotations):
        return _finding(S20_ASSERTIONS, 3, Outcome.CANNOT_TEST,
                        "the run published no wall clock and rotation count, "
                        "so the per-mage threshold has nothing to be checked "
                        "against")
    if derive(caps) is None:
        return _finding(
            S20_ASSERTIONS, 3, Outcome.FAIL,
            "the run published a threshold of %r; over %d rotations the "
            "formula gives %d with retry and %d without"
            % (caps.get(THRESHOLD_KEY), rotations,
               floor_activation_threshold(attempt_factor(rotations, False)),
               floor_activation_threshold(attempt_factor(rotations, True))))
    return _finding(S20_ASSERTIONS, 3, Outcome.PASS, "")


#: The written cause when the run recorded no completion attempt at all. It is
#: the ONE condition that can empty assertion 1's collection, and it is named
#: rather than left to a generic message: an "every attempt reports X" loop
#: over nothing is a green that asserted nothing, which is the vacuity this
#: harness's doctrine treats as the worst available outcome.
REASON_NO_ATTEMPTS = (
    "the run recorded no completion attempt, so \"every attempt reports a "
    "finish this build knows\" would hold over an empty collection"
)


def _finish_vocabulary_finding(capture, attempts, failure):
    """Judge assertion 1: every attempt names a finish this build can produce.

    Unconditional by construction. It reads the shape of what the product
    emitted rather than trying to provoke a state, so it holds on a run where
    nothing overran, on a run where something did, and on a run where the
    backend reported no reason at all.

    **Null is a value, absence is a defect, and they are separated here.**
    ``finish`` renders as JSON null when the backend did not say why the model
    stopped, and substituting a word there would assert a measurement nobody
    took -- the exact confusion that once made a completion scraping its cap
    look healthy. So null PASSES. A ``finish`` key that is missing altogether
    is a broken output contract and fails.

    **The vacuity guard is a CANNOT_TEST, and which one it is matters.** With
    no attempts recorded the loop below iterates nothing and would report a
    green that checked nothing, so the empty collection is reported by name
    instead. It is not a FAIL because *"a completed consult leaves records"* is
    already S20's third assertion over this very run, and duplicating that
    verdict here would report one defect as two. It also cannot become S21's
    resting state: a healthy R4 records attempts for all three seats, and a run
    that records none has already blocked the gate through S20.

    Args:
        capture: R4's reduction.
        attempts: Every recorded attempt, or None when ``completions`` could
            not be read as a per-seat map.
        failure: Why there are none.

    Returns:
        Finding: PASS when every attempt's ``finish`` is one of
        :data:`KNOWN_FINISH_LABELS` or null.
    """
    if capture.envelope is None:
        return _finding(S21_ASSERTIONS, 0, capture.outcome, capture.detail)
    if attempts is None:
        return _finding(S21_ASSERTIONS, 0, Outcome.FAIL, failure)
    if not attempts:
        return _finding(S21_ASSERTIONS, 0, Outcome.CANNOT_TEST,
                        REASON_NO_ATTEMPTS)
    problems = [problem
                for problem in (_finish_problem(item) for item in attempts)
                if problem]
    if problems:
        return _finding(S21_ASSERTIONS, 0, Outcome.FAIL, "; ".join(problems))
    return _finding(S21_ASSERTIONS, 0, Outcome.PASS, "")


def _finish_problem(item):
    """How one attempt fails the finish vocabulary, if it does.

    Args:
        item: The attempt to judge.

    Returns:
        str: The problem, or the empty string when the attempt is fine.
    """
    if not isinstance(item.record, dict):
        return "%s is not an object, so it reports no %s at all" % (
            item.label, FINISH_KEY)
    if FINISH_KEY not in item.record:
        return ("%s omits %s, so nothing says whether the backend reported a "
                "reason or reported none" % (item.label, FINISH_KEY))
    finish = item.record[FINISH_KEY]
    if finish is None or finish in KNOWN_FINISH_LABELS:
        return ""
    return ("%s reports %s as %r, which is outside the vocabulary this build "
            "renders (%s, or null for not reported)"
            % (item.label, FINISH_KEY, finish,
               ", ".join(KNOWN_FINISH_LABELS)))


def _rotation_consistency_finding(capture):
    """Judge assertion 2: the rotation report is published and well formed.

    Two halves, and the first is what keeps the second from being vacuous.

    **The report is published even when nobody rotated.** ``rotations`` is an
    array magi-rs emits on every consult, empty when no seat hopped, and that
    presence is the same fact ``pool_eligibility`` carries in S22: absent says
    *"not computed"*, empty says *"computed, nothing to report"*. So this half
    has content on every run, the healthy no-rotation one included, and there
    is no input for which this assertion checks nothing.

    **Each hop names a cause this build knows, and its locality.** ``cause`` is
    derived from the crate's own serde rather than a hand-written match
    precisely so a new ``RotationKind`` cannot ship as an invented wildcard
    label, and ``mage_local`` comes from magi-core's ``is_mage_local`` because
    it decides whether the other two seats keep going. Every hop is checked for
    both, and the cause against :data:`KNOWN_ROTATION_CAUSES`.

    A seat with no hops is not a failure and asserts nothing further, which is
    exactly why the first half has to carry the unconditional weight.

    Complexity: ``O(entries x hops)`` -- one pass over each.

    Args:
        capture: R4's reduction.

    Returns:
        Finding: PASS when the array is published and every hop it carries is
        well formed.
    """
    if capture.envelope is None:
        return _finding(S21_ASSERTIONS, 1, capture.outcome, capture.detail)
    if ROTATIONS_KEY not in capture.envelope:
        return _finding(S21_ASSERTIONS, 1, Outcome.FAIL,
                        "the envelope carries no %s; absent says the rotation "
                        "report was not computed, which is a different fact "
                        "from an empty one saying nobody hopped"
                        % ROTATIONS_KEY)
    entries = capture.envelope[ROTATIONS_KEY]
    if not isinstance(entries, list):
        return _finding(S21_ASSERTIONS, 1, Outcome.FAIL,
                        "%s is %s, expected an array of rotating seats"
                        % (ROTATIONS_KEY, type(entries).__name__))
    problems = []
    for position, entry in enumerate(entries):
        problems.extend(_entry_problems(position, entry))
    if problems:
        return _finding(S21_ASSERTIONS, 1, Outcome.FAIL, "; ".join(problems))
    return _finding(S21_ASSERTIONS, 1, Outcome.PASS, "")


def _entry_problems(position, entry):
    """Every way one rotation entry fails to publish a readable chain.

    Args:
        position: The entry's index, so a message names it without the reader
            counting.
        entry: The entry as published.

    Returns:
        list[str]: One message per problem; empty when the entry and every hop
        under it are well formed.
    """
    where = "%s[%d]" % (ROTATIONS_KEY, position)
    if not isinstance(entry, dict):
        return ["%s is not an object" % where]
    if CHAIN_KEY not in entry:
        return ["%s publishes no %s, so its hops cannot be read"
                % (where, CHAIN_KEY)]
    chain = entry[CHAIN_KEY]
    if not isinstance(chain, list):
        return ["%s.%s is %s, expected an array of hops"
                % (where, CHAIN_KEY, type(chain).__name__)]
    problems = []
    for index, hop in enumerate(chain):
        problems.extend(_hop_problems("%s.%s[%d]" % (where, CHAIN_KEY, index),
                                      hop))
    return problems


def _hop_problems(where, hop):
    """Every way one hop fails to name a known cause and its locality.

    Args:
        where: How to name this hop in a message.
        hop: The hop as published.

    Returns:
        list[str]: One message per problem; empty when the hop is well formed.
    """
    if not isinstance(hop, dict):
        return ["%s is not an object" % where]
    problems = []
    if CAUSE_KEY not in hop:
        problems.append("%s publishes no %s" % (where, CAUSE_KEY))
    elif hop[CAUSE_KEY] not in KNOWN_ROTATION_CAUSES:
        problems.append(
            "%s names %s %r, which is not one of the causes this build renders"
            " (%s)" % (where, CAUSE_KEY, hop[CAUSE_KEY],
                       ", ".join(KNOWN_ROTATION_CAUSES)))
    if MAGE_LOCAL_KEY not in hop:
        problems.append("%s publishes no %s, so nothing says whether the cause "
                        "condemned one mage or the run"
                        % (where, MAGE_LOCAL_KEY))
    elif not isinstance(hop[MAGE_LOCAL_KEY], bool):
        problems.append("%s reports %s as %r, which is not a locality"
                        % (where, MAGE_LOCAL_KEY, hop[MAGE_LOCAL_KEY]))
    return problems


@scenario("S21", assertions=S21_ASSERTIONS, run=MIGRATION_RUN,
          needs_backend=True)
def every_attempt_reports_a_finish_this_build_knows(run):
    """Assert the finish vocabulary and the rotation report's own shape.

    Both assertions are unconditional: they read what the product emitted on
    whatever run happened, rather than depending on a state a model has to be
    talked into producing. See the module docstring for the measurement that
    retired the forcing recipe.

    Args:
        run: R4's ``RunResult``, or None.

    Yields:
        Finding: One per entry of :data:`S21_ASSERTIONS`, in that order.
    """
    capture = _capture_of(run, MIGRATION_RUN)
    attempts, failure = _attempts_of(capture.envelope)
    yield _finish_vocabulary_finding(capture, attempts, failure)
    yield _rotation_consistency_finding(capture)


@scenario("S22", assertions=S22_ASSERTIONS, run=MIGRATION_RUN,
          needs_backend=True)
def the_rotation_report_is_complete(run):
    """Assert the report publishes what a consumer needs to derive degradation.

    Args:
        run: R4's ``RunResult``, or None.

    Yields:
        Finding: One per entry of :data:`S22_ASSERTIONS`, in that order.
    """
    capture = _capture_of(run, MIGRATION_RUN)
    attempts, failure = _attempts_of(capture.envelope)
    yield _eligibility_finding(capture)
    yield _derivability_finding(capture, attempts, failure)
    yield _degraded_finding(capture)


def _eligibility_finding(capture):
    """Judge assertion 1: the snapshot is published even when it rejects none.

    Absent and empty are different facts. Absent says the snapshot was not
    computed; empty says it was computed and found nothing to reject. A run
    with no pool declared and a run with a healthy pool would otherwise read
    identically, and the inert-pool case is what this telemetry exists to make
    visible -- so emptiness is a PASS and only absence is a failure.

    Args:
        capture: R4's reduction.

    Returns:
        Finding: PASS when the key is present and an object, empty included.
    """
    if capture.envelope is None:
        return _finding(S22_ASSERTIONS, 0, capture.outcome, capture.detail)
    if POOL_ELIGIBILITY_KEY not in capture.envelope:
        return _finding(S22_ASSERTIONS, 0, Outcome.FAIL,
                        "the envelope carries no %s; absent says the snapshot "
                        "was not computed, which is a different fact from an "
                        "empty one that rejected nothing"
                        % POOL_ELIGIBILITY_KEY)
    value = capture.envelope[POOL_ELIGIBILITY_KEY]
    if not isinstance(value, dict):
        return _finding(S22_ASSERTIONS, 0, Outcome.FAIL,
                        "%s is %s, expected an object keyed by seat"
                        % (POOL_ELIGIBILITY_KEY, type(value).__name__))
    return _finding(S22_ASSERTIONS, 0, Outcome.PASS, "")


def _derivability_finding(capture, attempts, failure):
    """Judge assertion 2: each notion of degradation has a published key.

    Three separate notions, three separate keys, and the check is over
    PRESENCE because that is what derivability means. A mage that fell to
    another backend is ``model_configured != model_used``; truncation is
    ``finish == "length"``, so the key has to be there even when it is null;
    fewer than three verdicts stays in ``degraded``.

    Args:
        capture: R4's reduction.
        attempts: Every recorded attempt, or None.
        failure: Why there are none.

    Returns:
        Finding: PASS when all three notions have something to be derived from.
    """
    if capture.envelope is None:
        return _finding(S22_ASSERTIONS, 1, capture.outcome, capture.detail)
    problems = _rotation_key_problems(capture.envelope)
    if not isinstance(capture.envelope.get(DEGRADED_KEY), bool):
        problems.append("%s is %r, so the fewer-than-three-verdicts notion "
                        "has no boolean to be read from"
                        % (DEGRADED_KEY, capture.envelope.get(DEGRADED_KEY)))
    if attempts is None:
        problems.append(failure)
    elif not attempts:
        # The truncation notion is genuinely underivable here, but whatever was already
        # collected is a REAL defect and outranks it: returning CANNOT_TEST and dropping
        # `problems` would file a rotation defect under "nothing to test", which is the one
        # way this harness can hide a failure behind an honest-looking label.
        untestable = ("the run recorded no completion attempt, so the truncation notion "
                      "has no published key to be derived from")
        if problems:
            return _finding(S22_ASSERTIONS, 1, Outcome.FAIL,
                            "; ".join([*problems, untestable]))
        return _finding(S22_ASSERTIONS, 1, Outcome.CANNOT_TEST, untestable)
    else:
        problems.extend(
            "%s omits %s, so truncation cannot be derived for it"
            % (item.label, FINISH_KEY)
            for item in attempts
            if not isinstance(item.record, dict)
            or FINISH_KEY not in item.record)
    if problems:
        return _finding(S22_ASSERTIONS, 1, Outcome.FAIL, "; ".join(problems))
    return _finding(S22_ASSERTIONS, 1, Outcome.PASS, "")


def _rotation_key_problems(envelope):
    """Every way the rotations array fails to carry the fallback notion.

    Complexity: ``O(entries)``.

    Args:
        envelope: The consult envelope.

    Returns:
        list[str]: One message per problem; empty when every entry publishes
        both model names.
    """
    entries = envelope.get(ROTATIONS_KEY)
    if not isinstance(entries, list):
        return ["%s is %s, expected an array, so a mage falling to another "
                "backend cannot be derived"
                % (ROTATIONS_KEY, type(entries).__name__)]
    problems = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            problems.append("%s[%d] is not an object" % (ROTATIONS_KEY,
                                                         position))
            continue
        missing = [key for key in (MODEL_CONFIGURED_KEY, MODEL_USED_KEY)
                   if not isinstance(entry.get(key), str)]
        if missing:
            problems.append(
                "%s[%d] publishes no %s, so a mage falling to another backend "
                "cannot be derived" % (ROTATIONS_KEY, position,
                                       " and no ".join(missing)))
    return problems


def _degraded_finding(capture):
    """Judge assertion 3: the bit still means what it always meant.

    The premise is a three-verdict run, and it is checked rather than assumed:
    on a run that produced fewer verdicts a ``degraded`` of true is the bit
    working, so asserting false there would be asserting the opposite of the
    contract.

    Args:
        capture: R4's reduction.

    Returns:
        Finding: PASS when three verdicts came back and the bit is false.
    """
    if capture.envelope is None:
        return _finding(S22_ASSERTIONS, 2, capture.outcome, capture.detail)
    agents = capture.envelope.get(AGENTS_KEY)
    if not isinstance(agents, list):
        return _finding(S22_ASSERTIONS, 2, Outcome.CANNOT_TEST,
                        "the run carried no structured verdicts, so it is not "
                        "observably a three-verdict run")
    if len(agents) != TRIO_SIZE:
        return _finding(S22_ASSERTIONS, 2, Outcome.CANNOT_TEST,
                        "the run produced %d verdicts, so the three-verdict "
                        "premise does not hold and the bit is not being asked "
                        "the same question" % len(agents))
    degraded = capture.envelope.get(DEGRADED_KEY)
    if degraded is False:
        return _finding(S22_ASSERTIONS, 2, Outcome.PASS, "")
    return _finding(S22_ASSERTIONS, 2, Outcome.FAIL,
                    "three verdicts came back and %s is %r"
                    % (DEGRADED_KEY, degraded))


#: Keys REQ-EE-1 adds to every completion record: the reasoning channel's
#: state, and which control the attempt was sent under.
REASONING_KEY = "reasoning"
CONTROL_KEY = "control"
#: The inner member of a ``Measured``/``Unsupported`` state that would carry
#: model text -- always JSON null as this build publishes it (REQ-EE-1).
REASONING_TEXT_KEY = "text"

#: ``ReasoningControl``'s kebab-case tags, mirrored from magi-core 4.2.0's
#: serde (``provider.rs:15-34``). Consumer: ``[magi] reasoning`` (REQ-EE-3).
KNOWN_CONTROL_TAGS = ("default", "disabled", "enabled")
DEFAULT_CONTROL = "default"
DISABLED_CONTROL = "disabled"

#: ``ReasoningState``'s variants at the pin, with the exact inner keys each
#: tagged variant serializes (externally tagged, ``provider.rs:437-493``).
#: ``NotMeasured`` carries no inner object at all.
NOT_MEASURED_STATE = "NotMeasured"
STATE_INNER_KEYS = {"Measured": ("chars", "text"),
                    "Unsupported": ("backend", "chars", "text")}

#: S25's own invocation, its prompt, and the label it archives under.
S25_LABEL = "s25-disabled-consult"
S25_PROMPT = b"Is a one-line rename worth a full review? Answer briefly.\n"
#: The two lines S25 installs in the [magi] table of its own scratch magi.toml.
S25_INSTALLED_LINES = ('reasoning = "disabled"', "reasoning_trace = true")

_INIT_SUBCOMMAND = "init"
_CONSULT_SUBCOMMAND = "consult"
_MAGI_DIR_NAME = ".magi"
_MAGI_TOML_NAME = "magi.toml"
_INIT_TIMEOUT_S = 120
#: What the product exits with when it refuses a configuration, mirrored from
#: ``config_fatal.CONFIG_EXIT_CODE`` (not imported: ``config_fatal`` imports
#: :func:`with_magi_lines` from here, so the reverse import would be circular).
_CONFIG_EXIT_CODE = 2

#: The written cause when S25/S26 recorded no completion attempt at all.
_REASON_NO_ATTEMPTS_S25 = (
    "the run recorded no completion attempt, so a claim about every attempt "
    "would hold over an empty collection"
)

_KEY_TOKEN = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=')
_TABLE_HEADER_LINE = re.compile(r'^\s*\[')
_MAGI_HEADER_LINE = re.compile(r'^\s*\[magi\]\s*$')


def with_magi_lines(generated: str, lines: tuple[str, ...]) -> str:
    """Install *lines* in the ``[magi]`` table of *generated*.

    Replaces any ACTIVE line that declares the same key as one of *lines*, in
    place; installs a line for a key that has no active declaration just
    before the table closes (before the pool, which TOML requires to come
    last); leaves every commented line exactly as it is, including one that
    happens to declare the same key. A key that is a PREFIX of another
    (``reasoning`` vs. ``reasoning_trace``/``reasoning_spelling``) is matched
    on the whole identifier, never the prefix.

    Complexity: O(lines of the file).

    Args:
        generated: A ``magi.toml`` the product wrote.
        lines: The lines to install, each ``key = value``.

    Returns:
        str: The configuration text, with *lines* installed.
    """
    wanted: dict[str, str] = {}
    for line in lines:
        match = _KEY_TOKEN.match(line)
        if match is not None:
            wanted[match.group(1)] = line
    installed: set[str] = set()
    result: list[str] = []
    inside = False
    for line in generated.splitlines():
        if not inside:
            result.append(line)
            if _MAGI_HEADER_LINE.match(line):
                inside = True
            continue
        stripped = line.strip()
        if stripped.startswith("#"):
            result.append(line)
            continue
        if _TABLE_HEADER_LINE.match(line):
            result.extend(new_line for key, new_line in wanted.items()
                         if key not in installed)
            installed.update(wanted)
            inside = False
            result.append(line)
            continue
        match = _KEY_TOKEN.match(line)
        if match is not None and match.group(1) in wanted:
            key = match.group(1)
            result.append(wanted[key])
            installed.add(key)
            continue
        result.append(line)
    if inside:
        result.extend(new_line for key, new_line in wanted.items()
                     if key not in installed)
    return "\n".join(result) + ("\n" if generated.endswith("\n") else "")


def _excerpt(output, limit=600):
    """Render the beginning of a capture for a finding's detail.

    Args:
        output: The capture to quote.
        limit: How many bytes to keep.

    Returns:
        str: The first *limit* bytes of both streams, decoded leniently.
    """
    return output.raw()[:limit].decode("utf-8", errors="replace").strip()


def _s25_finding(index, outcome, detail):
    """Build one of S25's three findings.

    Args:
        index: Position in :data:`S25_ASSERTIONS`.
        outcome: What became of it.
        detail: The cause when the outcome is not PASS.

    Returns:
        Finding: With no run id -- S25 is standalone.
    """
    return Finding(assertion=S25_ASSERTIONS[index], outcome=outcome,
                   detail=detail, run_id=None)


def _state_tag(record):
    """The ``(tag, inner)`` pair of one attempt's reasoning state.

    Args:
        record: The attempt object, expected to carry :data:`REASONING_KEY`.

    Returns:
        tuple[str | None, object]: ``(NOT_MEASURED_STATE, None)`` for the
        string state; ``(tag, inner)`` for a single-key tagged object; and
        ``(None, state)`` -- the caller's signal that the value has no
        recognisable shape at all -- for anything else.
    """
    state = record.get(REASONING_KEY) if isinstance(record, dict) else None
    if state == NOT_MEASURED_STATE:
        return NOT_MEASURED_STATE, None
    if isinstance(state, dict) and len(state) == 1:
        ((tag, inner),) = state.items()
        return tag, inner
    return None, state


def _control_problem(attempts, expected):
    """Every attempt whose :data:`CONTROL_KEY` differs from *expected*.

    Args:
        attempts: Every recorded attempt.
        expected: The control tag every attempt is supposed to carry.

    Returns:
        list[str]: One message per offender; empty when all agree.
    """
    problems = []
    for item in attempts:
        if not isinstance(item.record, dict) or CONTROL_KEY not in item.record:
            problems.append("%s publishes no %s" % (item.label, CONTROL_KEY))
            continue
        control = item.record[CONTROL_KEY]
        if control != expected:
            problems.append("%s reports %s as %r, expected %r"
                            % (item.label, CONTROL_KEY, control, expected))
    return problems


def _no_measured_overrun_problems(attempts):
    """Every attempt reporting a measured reasoning count above zero.

    A state this build does not recognise at all is reported too: it is
    exactly as untrustworthy a signal of "the switch worked" as a positive
    count would be.

    Args:
        attempts: Every recorded attempt.

    Returns:
        list[str]: One message per offender; empty when none overran.
    """
    problems = []
    for item in attempts:
        if not isinstance(item.record, dict) or REASONING_KEY not in item.record:
            problems.append("%s publishes no %s" % (item.label, REASONING_KEY))
            continue
        tag, inner = _state_tag(item.record)
        if tag is None or (tag != NOT_MEASURED_STATE
                           and tag not in STATE_INNER_KEYS):
            problems.append(
                "%s reports %s as %r, which is not a state this build "
                "recognises" % (item.label, REASONING_KEY,
                                item.record[REASONING_KEY]))
            continue
        if tag != "Measured":
            continue
        chars = inner.get("chars") if isinstance(inner, dict) else None
        if isinstance(chars, int) and not isinstance(chars, bool) and chars > 0:
            problems.append(
                "%s reports a measured reasoning count of %d chars while "
                "reasoning = disabled" % (item.label, chars))
    return problems


def _no_leaked_text_problems(attempts):
    """Every attempt whose reasoning state carries non-null text.

    The message cites the LENGTH of the leaked text, never its content: the
    text is the model's, unvalidated, and a finding's detail reaches the
    report and the certificate.

    Args:
        attempts: Every recorded attempt.

    Returns:
        list[str]: One message per offender; empty when none leaked.
    """
    problems = []
    for item in attempts:
        if not isinstance(item.record, dict) or REASONING_KEY not in item.record:
            problems.append("%s publishes no %s" % (item.label, REASONING_KEY))
            continue
        tag, inner = _state_tag(item.record)
        if tag is None or tag == NOT_MEASURED_STATE or not isinstance(inner, dict):
            continue
        text = inner.get(REASONING_TEXT_KEY)
        if text is not None:
            problems.append(
                "%s.%s.%s carries %d character(s) of model text"
                % (item.label, tag, REASONING_TEXT_KEY, len(str(text))))
    return problems


def _s25_control_finding(capture, attempts, failure):
    """Judge assertion 1: every attempt reports the disabled control.

    Args:
        capture: The consult's reduction.
        attempts: Every recorded attempt, or None.
        failure: Why there are none.

    Returns:
        Finding: PASS when every attempt carries ``control: "disabled"``.
    """
    if capture.envelope is None:
        return _s25_finding(0, capture.outcome, capture.detail)
    if attempts is None:
        return _s25_finding(0, Outcome.FAIL, failure)
    if not attempts:
        return _s25_finding(0, Outcome.CANNOT_TEST, _REASON_NO_ATTEMPTS_S25)
    problems = _control_problem(attempts, DISABLED_CONTROL)
    if problems:
        return _s25_finding(0, Outcome.FAIL, "; ".join(problems))
    return _s25_finding(0, Outcome.PASS, "")


def _s25_no_overrun_finding(capture, attempts, failure):
    """Judge assertion 2: no attempt reports a measured count above zero.

    Args:
        capture: The consult's reduction.
        attempts: Every recorded attempt, or None.
        failure: Why there are none.

    Returns:
        Finding: PASS when no attempt overran under ``Measured``.
    """
    if capture.envelope is None:
        return _s25_finding(1, capture.outcome, capture.detail)
    if attempts is None:
        return _s25_finding(1, Outcome.FAIL, failure)
    if not attempts:
        return _s25_finding(1, Outcome.CANNOT_TEST, _REASON_NO_ATTEMPTS_S25)
    problems = _no_measured_overrun_problems(attempts)
    if problems:
        return _s25_finding(1, Outcome.FAIL, "; ".join(problems))
    return _s25_finding(1, Outcome.PASS, "")


def _s25_no_leak_finding(capture, attempts, failure):
    """Judge assertion 3: no attempt record carries reasoning text.

    Args:
        capture: The consult's reduction.
        attempts: Every recorded attempt, or None.
        failure: Why there are none.

    Returns:
        Finding: PASS when every attempt's reasoning text is null.
    """
    if capture.envelope is None:
        return _s25_finding(2, capture.outcome, capture.detail)
    if attempts is None:
        return _s25_finding(2, Outcome.FAIL, failure)
    if not attempts:
        return _s25_finding(2, Outcome.CANNOT_TEST, _REASON_NO_ATTEMPTS_S25)
    problems = _no_leaked_text_problems(attempts)
    if problems:
        return _s25_finding(2, Outcome.FAIL, "; ".join(problems))
    return _s25_finding(2, Outcome.PASS, "")


@scenario("S25", assertions=S25_ASSERTIONS, needs_backend=True)
def a_disabled_reasoning_control_round_trips(run):
    """Assert ``reasoning = "disabled"`` round-trips on the default trio.

    Standalone (``run=None``): it scaffolds its OWN workspace by cwd, under
    ``needs_backend=True``, rather than reading a shared run, because the
    property has to hold on the trio the OPERATOR configured (OQ-8) -- a
    profile that overrides ``smoke.toml`` never rewrites this scratch
    ``magi.toml``. ``reasoning_trace = true`` is installed on purpose: without
    it magi-core never captures text at all, and assertion 3 would be a
    guardian that cannot fail (see the module's own contract note below).

    Args:
        run: Always ``None``; S25 declares no shared run.

    Yields:
        Finding: One per entry of :data:`S25_ASSERTIONS`, in that order.
    """
    root = pathlib.Path(
        tempfile.mkdtemp(prefix="s25-", dir=str(runs.scratch_root())))
    try:
        seeded = runs.attempt(
            [_INIT_SUBCOMMAND], stdin=b"", timeout_s=_INIT_TIMEOUT_S,
            label="s25-init", cwd=root,
            env={"MAGI_PASSPHRASE": runs.passphrase()})
        if not seeded.ok or not (root / _MAGI_DIR_NAME).is_dir():
            for index in range(len(S25_ASSERTIONS)):
                yield _s25_finding(
                    index, Outcome.CANNOT_TEST,
                    "the product's %s did not scaffold a workspace to "
                    "configure" % _INIT_SUBCOMMAND)
            return
        toml_path = root / _MAGI_DIR_NAME / _MAGI_TOML_NAME
        generated = toml_path.read_text(encoding="utf-8")
        toml_path.write_text(with_magi_lines(generated, S25_INSTALLED_LINES),
                             encoding="utf-8")
        attempt = runs.attempt(
            [_CONSULT_SUBCOMMAND, "--output-format", "json",
             "--timeout", str(runs.LARGE_CONSULT_TIMEOUT_S)],
            stdin=S25_PROMPT, timeout_s=runs.LARGE_CONSULT_CEILING_S,
            label=S25_LABEL, cwd=root,
            env={"MAGI_PASSPHRASE": runs.passphrase()})
        if not attempt.ok:
            for index in range(len(S25_ASSERTIONS)):
                yield _s25_finding(index, Outcome.CANNOT_TEST, attempt.failure)
            return
        if attempt.output.exit_code == _CONFIG_EXIT_CODE:
            detail = ("the product refused a configuration declaring "
                      "reasoning = disabled: %s" % _excerpt(attempt.output))
            for index in range(len(S25_ASSERTIONS)):
                yield _s25_finding(index, Outcome.FAIL, detail)
            return
        capture = _capture_of(attempt, S25_LABEL)
        attempts, failure = _attempts_of(capture.envelope)
        yield _s25_control_finding(capture, attempts, failure)
        yield _s25_no_overrun_finding(capture, attempts, failure)
        yield _s25_no_leak_finding(capture, attempts, failure)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _control_default_finding(capture, attempts, failure):
    """Judge assertion 1: every attempt reports the default control.

    Args:
        capture: R4's reduction.
        attempts: Every recorded attempt, or None.
        failure: Why there are none.

    Returns:
        Finding: PASS when every attempt carries ``control: "default"``.
    """
    if capture.envelope is None:
        return _finding(S26_ASSERTIONS, 0, capture.outcome, capture.detail)
    if attempts is None:
        return _finding(S26_ASSERTIONS, 0, Outcome.FAIL, failure)
    if not attempts:
        return _finding(S26_ASSERTIONS, 0, Outcome.CANNOT_TEST,
                        REASON_NO_ATTEMPTS)
    problems = _control_problem(attempts, DEFAULT_CONTROL)
    if problems:
        return _finding(S26_ASSERTIONS, 0, Outcome.FAIL, "; ".join(problems))
    return _finding(S26_ASSERTIONS, 0, Outcome.PASS, "")


def _reasoning_state_problems(item):
    """Every way one attempt's reasoning state is not one magi-core could
    have serialized, mirrored field for field from :data:`STATE_INNER_KEYS`.

    Args:
        item: The attempt to judge.

    Returns:
        list[str]: One message per problem; empty when the state is well
        formed.
    """
    record = item.record
    if not isinstance(record, dict) or REASONING_KEY not in record:
        return ["%s publishes no %s" % (item.label, REASONING_KEY)]
    tag, inner = _state_tag(record)
    if tag == NOT_MEASURED_STATE:
        return []
    if tag is None:
        return ["%s reports %s as %r, which is neither %s nor a single-key "
                "object" % (item.label, REASONING_KEY, record[REASONING_KEY],
                           NOT_MEASURED_STATE)]
    if tag not in STATE_INNER_KEYS:
        return ["%s reports %s tag %r, which is not one of %s"
                % (item.label, REASONING_KEY, tag,
                   ", ".join(sorted(STATE_INNER_KEYS)))]
    wanted = set(STATE_INNER_KEYS[tag])
    if not isinstance(inner, dict) or set(inner) != wanted:
        return ["%s.%s has keys %s, expected exactly %s"
                % (item.label, tag,
                   sorted(inner) if isinstance(inner, dict) else inner,
                   sorted(wanted))]
    problems = []
    chars = inner.get("chars")
    if chars is None:
        if tag != "Unsupported":
            problems.append("%s.%s.chars is null, which only Unsupported "
                            "allows" % (item.label, tag))
    elif not isinstance(chars, int) or isinstance(chars, bool) or chars < 0:
        problems.append("%s.%s.chars is %r, expected a non-negative integer"
                        % (item.label, tag, chars))
    if tag == "Unsupported" and not isinstance(inner.get("backend"), str):
        problems.append("%s.%s.backend is %r, expected a string"
                        % (item.label, tag, inner.get("backend")))
    if inner.get(REASONING_TEXT_KEY) is not None:
        problems.append("%s.%s.%s is not null"
                        % (item.label, tag, REASONING_TEXT_KEY))
    return problems


def _state_shape_finding(capture, attempts, failure):
    """Judge assertion 2: every attempt's reasoning state is well formed.

    Args:
        capture: R4's reduction.
        attempts: Every recorded attempt, or None.
        failure: Why there are none.

    Returns:
        Finding: PASS when every attempt's reasoning state matches a shape
        magi-core's serde could have produced.
    """
    if capture.envelope is None:
        return _finding(S26_ASSERTIONS, 1, capture.outcome, capture.detail)
    if attempts is None:
        return _finding(S26_ASSERTIONS, 1, Outcome.FAIL, failure)
    if not attempts:
        return _finding(S26_ASSERTIONS, 1, Outcome.CANNOT_TEST,
                        REASON_NO_ATTEMPTS)
    problems = []
    for item in attempts:
        problems.extend(_reasoning_state_problems(item))
    if problems:
        return _finding(S26_ASSERTIONS, 1, Outcome.FAIL, "; ".join(problems))
    return _finding(S26_ASSERTIONS, 1, Outcome.PASS, "")


@scenario("S26", assertions=S26_ASSERTIONS, run=MIGRATION_RUN,
          needs_backend=True)
def every_attempt_reports_its_reasoning_control_and_state(run):
    """Assert R4's attempts report the default control in magi-core's shape.

    R4 declares no ``reasoning`` key, so REQ-EE-3's default applies, and
    REQ-EE-1's rendering is checked structurally rather than against any one
    model's behaviour (OQ-9/§6 doctrine: never assert on what the model
    answered).

    Args:
        run: R4's ``RunResult``, or None.

    Yields:
        Finding: One per entry of :data:`S26_ASSERTIONS`, in that order.
    """
    capture = _capture_of(run, MIGRATION_RUN)
    attempts, failure = _attempts_of(capture.envelope)
    yield _control_default_finding(capture, attempts, failure)
    yield _state_shape_finding(capture, attempts, failure)


