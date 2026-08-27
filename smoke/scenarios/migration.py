# Author: Julian Bolivar
# Version: 0.17.0
# Date: 2026-08-27
"""S20, S21 and S22 -- the three scenarios the magi-core 4.0.0 move needs.

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

**What this assertion does NOT prove, said plainly.** The declared cap is
numerically magi-core 4.0.0's own default, so a run whose call site was deleted
transmits the same number and this assertion stays green. Distinguishing a
declared cap from an inherited one is a question about the BUILDER, and it is
answered where it can be -- the Rust-side wiring trace, which asserts the
builder was handed a configuration rather than a value. S20's share of the
work is the other half: that whatever was configured actually travelled. The
per-attempt ``cap`` is therefore asserted present first and equal second, so a
record carrying no cap is reported as nothing having been transmitted rather
than as a value that failed to match -- the two send the next reader to
different places.

**S21 forces a state and is allowed not to reach it.** A reasoning model
returns empty content when the prompt is large enough that it spends its whole
output budget thinking; the recorded failure used roughly 250 kB of Rust
against a 4096-token cap. Only the TOKEN COUNT is load-bearing -- the text does
not matter -- and R4's payload is exactly that, built from the tree by
:mod:`smoke.payload` and bounded by the product's own ``MAX_QUERY_BYTES``. The
cap is now four times what it was, so the model may simply finish. **That is
``CANNOT_TEST``, never ``FAIL``**: the product did nothing wrong, the
environment would not produce the state. Which condition was missing is
reported by name, because a bare ``CANNOT_TEST`` with a generic reason is the
vacuous outcome wearing an honest label.

**The one thing S21 cannot observe, stated rather than papered over.**
``CompletionRecord::reasoning`` is deliberately not rendered into the run JSON
-- no consumer reads it -- so the harness cannot ask *"did a model reason?"*
directly. What it can read is the finish reason and the token counts, and those
separate the reports that matter: an attempt that exhausted the cap and still
answered is a different fact from one that stopped normally well inside it, and
both differ from a backend that reported neither. Where two readings genuinely
collapse into one observable, the message names both rather than picking one.

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
                                  ERROR_KEY, PAYLOAD_SENT_FRACTION,
                                  SUCCESS_EXIT_CODE, TRIO_SIZE,
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
    "the rotation cause reads empty_completion and not transport",
    "the empty completion is reported as mage-local",
    "finish tells budget exhaustion from a genuinely empty answer",
)
S22_ASSERTIONS = (
    "pool_eligibility is present even when empty",
    "all three notions of degradation are derivable from published keys",
    "degraded is false for a three-verdict run",
)

#: The run all three read. See the module docstring for why there is one.
MIGRATION_RUN = "R4"

#: The completion cap magi-rs DECLARES, mirrored from the product's own
#: constant. The maintenance contract is accepted rather than re-litigated: a
#: cap the product moves turns S20 red and forces a change here, and a check
#: that adjusted itself to whatever the product reported would detect nothing.
#: The value is numerically magi-core 4.0.0's own default, which is what bounds
#: what this scenario can prove -- see the module docstring.
DECLARED_COMPLETION_CAP = 16384

#: Keys of the consult envelope this module reaches for.
COMPLETIONS_KEY = "completions"
POOL_ELIGIBILITY_KEY = "pool_eligibility"
ROTATIONS_KEY = "rotations"
CHAIN_KEY = "chain"
MODEL_CONFIGURED_KEY = "model_configured"
MODEL_USED_KEY = "model_used"
AGENT_KEY = "agent"

#: Keys of one rotation hop.
CAUSE_KEY = "cause"
MAGE_LOCAL_KEY = "mage_local"

#: The five keys one completion attempt exposes, mapped field by field in
#: ``src/magi/completion_report.rs``. Counted EXACTLY, for the same reason S18
#: counts the verdict keys exactly: ``CompletionRecord`` is
#: ``#[non_exhaustive]``, so a field magi-core adds in a minor release reaches
#: this public JSON the moment somebody replaces the explicit mapping with a
#: direct interpolation, and an "at least five" detects none of it. The sixth
#: field the record carries, ``reasoning``, is deliberately NOT rendered -- an
#: unexpected key is as much a defect here as a missing one.
ATTEMPT_KEYS = ("model", "cap", "finish", "completion_tokens", "prompt_tokens")
CAP_KEY = "cap"
FINISH_KEY = "finish"
COMPLETION_TOKENS_KEY = "completion_tokens"

#: The cause a seat's rotation must name when its completion came back empty,
#: and the one it must NOT name. ``transport`` is the label a wildcard produces
#: when a hand-written match meets a cause it does not know, so reading it here
#: says the run blamed the network for something local to one mage.
EMPTY_COMPLETION_CAUSE = "empty_completion"
TRANSPORT_CAUSE = "transport"

#: The finish reason that means the output budget ran out.
LENGTH_FINISH = "length"

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
    value that was never there. The limit of what the equality half can prove
    is recorded in the module docstring: :data:`DECLARED_COMPLETION_CAP` IS the
    crate's own default, so a deleted call site transmits the same number, and
    the builder question is settled by the Rust-side wiring trace instead.

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
    """Judge assertion 3: one record per attempt, with exactly five keys.

    A per-seat total cannot be disaggregated back into which model was cut, so
    the array under a seat label is the attempt series and its length is a fact
    about the run. The key count is EXACT in both directions: a sixth key is as
    much a defect as a missing one, because ``CompletionRecord`` is
    ``#[non_exhaustive]`` and ``reasoning`` is deliberately not rendered.

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


#: Why S21 could not observe the state it forces. Each is a WRITTEN cause: a
#: bare CANNOT_TEST with a generic reason is the vacuous outcome wearing an
#: honest label, and these are the conditions that actually differ.
REASON_PAYLOAD_NOT_SENT = (
    "the recipe was not applied: run %s carried %d bytes against a declared "
    "payload of %d, so the prompt was never large enough to make a model "
    "spend its output budget"
)
REASON_NO_ATTEMPTS = (
    "the recipe cannot be observed: the run recorded no completion attempt, "
    "so nothing says how any model stopped"
)
REASON_NO_TOKEN_COUNTS = (
    "the recipe cannot be observed: no attempt reported a completion token "
    "count, so an empty answer cannot be told from an unreported one"
)
REASON_EXHAUSTED_AND_ANSWERED = (
    "an attempt exhausted the declared cap of %d and still returned %d "
    "completion tokens, so budget exhaustion did not produce an empty "
    "completion here"
)
REASON_COMPLETED_WITHIN_CAP = (
    "the model completed within the declared cap of %d, spending at most %d "
    "completion tokens and never reporting %r: either no configured model "
    "reasons at length, or the cap is now large enough for the ones that do"
)
REASON_NO_ROTATION = (
    "attempt %s returned zero completion tokens and its seat recorded no "
    "rotation, so no cause was published to read"
)


@dataclasses.dataclass(frozen=True)
class _EmptyCompletion:
    """The forced state, or the named reason it was not reached.

    Attributes:
        attempt: The attempt that came back with zero completion tokens, or
            None.
        hop: The rotation hop the attempt's seat recorded, or None.
        outcome: What every assertion reports when *attempt* is None, or when
            *attempt* is set and *hop* is not.
        detail: The written cause.
    """

    attempt: object
    hop: object
    outcome: object
    detail: str


def _empty_completion_state(run, capture, attempts, failure):
    """Find the empty completion, or say precisely which condition was missing.

    Complexity: ``O(attempts + hops)`` -- one pass over each.

    Args:
        run: R4's ``RunResult``, or None.
        capture: R4's reduction.
        attempts: Every recorded attempt, or None.
        failure: Why there are none.

    Returns:
        _EmptyCompletion: The state, or the reason.
    """
    if capture.envelope is None:
        return _EmptyCompletion(None, None, capture.outcome, capture.detail)
    target = runs.payload_target()
    if run.stdin_bytes < target * PAYLOAD_SENT_FRACTION:
        return _EmptyCompletion(
            None, None, Outcome.CANNOT_TEST,
            REASON_PAYLOAD_NOT_SENT % (MIGRATION_RUN, run.stdin_bytes, target))
    if attempts is None:
        return _EmptyCompletion(None, None, Outcome.FAIL, failure)
    if not attempts:
        return _EmptyCompletion(None, None, Outcome.CANNOT_TEST,
                                REASON_NO_ATTEMPTS)
    empty = _first_empty_attempt(attempts)
    if empty is None:
        return _EmptyCompletion(None, None, Outcome.CANNOT_TEST,
                                _why_not_empty(attempts))
    hop = _empty_completion_hop(capture.envelope, empty.seat)
    if hop is None:
        return _EmptyCompletion(empty, None, Outcome.CANNOT_TEST,
                                REASON_NO_ROTATION % empty.label)
    return _EmptyCompletion(empty, hop, None, "")


def _first_empty_attempt(attempts):
    """The first attempt that returned nothing at all.

    Args:
        attempts: Every recorded attempt.

    Returns:
        _Attempt | None: The attempt reporting exactly zero completion tokens.
    """
    for item in attempts:
        if not isinstance(item.record, dict):
            continue
        tokens = item.record.get(COMPLETION_TOKENS_KEY)
        if _is_count(tokens) and tokens == 0:
            return item
    return None


def _why_not_empty(attempts):
    """The written cause when every attempt came back with content.

    The three branches are the three genuinely different observations, and
    they are separated by what the report carries rather than by a threshold
    somebody invented. What the harness CANNOT separate -- a model that does
    not reason from one that reasons inside the cap -- is named in one message
    rather than guessed at, because ``reasoning`` is deliberately not rendered.

    Args:
        attempts: Every recorded attempt, all of them non-empty.

    Returns:
        str: The cause.
    """
    counts = [item.record.get(COMPLETION_TOKENS_KEY) for item in attempts
              if isinstance(item.record, dict)]
    measured = [count for count in counts if _is_count(count)]
    if not measured:
        return REASON_NO_TOKEN_COUNTS
    exhausted = [item for item in attempts
                 if isinstance(item.record, dict)
                 and item.record.get(FINISH_KEY) == LENGTH_FINISH]
    if exhausted:
        spent = exhausted[0].record.get(COMPLETION_TOKENS_KEY)
        return REASON_EXHAUSTED_AND_ANSWERED % (DECLARED_COMPLETION_CAP, spent)
    return REASON_COMPLETED_WITHIN_CAP % (DECLARED_COMPLETION_CAP,
                                          max(measured), LENGTH_FINISH)


def _empty_completion_hop(envelope, seat):
    """The rotation hop one seat recorded, preferring the empty-completion one.

    Complexity: ``O(hops)``.

    Args:
        envelope: The consult envelope.
        seat: The seat label to look up.

    Returns:
        dict | None: The seat's ``empty_completion`` hop when it has one, its
        first hop otherwise -- because a hop naming another cause is what
        assertion 1 exists to fail on -- and None when the seat did not rotate.
    """
    entries = envelope.get(ROTATIONS_KEY)
    if not isinstance(entries, list):
        return None
    chain = []
    for entry in entries:
        if isinstance(entry, dict) and entry.get(AGENT_KEY) == seat:
            found = entry.get(CHAIN_KEY)
            chain = found if isinstance(found, list) else []
            break
    hops = [hop for hop in chain if isinstance(hop, dict)]
    for hop in hops:
        if hop.get(CAUSE_KEY) == EMPTY_COMPLETION_CAUSE:
            return hop
    return hops[0] if hops else None


@scenario("S21", assertions=S21_ASSERTIONS, run=MIGRATION_RUN,
          needs_backend=True)
def an_empty_completion_names_itself(run):
    """Assert the empty completion is named, local, and told from exhaustion.

    Args:
        run: R4's ``RunResult``, or None.

    Yields:
        Finding: One per entry of :data:`S21_ASSERTIONS`, in that order.
    """
    capture = _capture_of(run, MIGRATION_RUN)
    attempts, failure = _attempts_of(capture.envelope)
    state = _empty_completion_state(run, capture, attempts, failure)
    yield _cause_finding(state)
    yield _locality_finding(state)
    yield _finish_finding(state)


def _cause_finding(state):
    """Judge assertion 1: the cause names the empty completion, not transport.

    ``transport`` is the label a wildcard produces for a cause it does not
    know, and it says the run blamed the network for something that happened
    inside one mage. That is why the two are named against each other rather
    than the cause merely being checked for presence.

    Args:
        state: What the run turned out to hold.

    Returns:
        Finding: PASS when the hop's cause is ``empty_completion``.
    """
    if state.hop is None:
        return _finding(S21_ASSERTIONS, 0, state.outcome, state.detail)
    cause = state.hop.get(CAUSE_KEY)
    if cause == EMPTY_COMPLETION_CAUSE:
        return _finding(S21_ASSERTIONS, 0, Outcome.PASS, "")
    blamed = (", which blames the network for a condition local to one mage"
              if cause == TRANSPORT_CAUSE else "")
    return _finding(S21_ASSERTIONS, 0, Outcome.FAIL,
                    "attempt %s returned zero completion tokens and its seat "
                    "rotated with cause %r rather than %r%s"
                    % (state.attempt.label, cause, EMPTY_COMPLETION_CAUSE,
                       blamed))


def _locality_finding(state):
    """Judge assertion 2: the hop reports the cause as mage-local.

    ``mage_local`` false says a cause condemned the whole run, which is what
    decides whether the other two seats keep going. Reading it off the hop is
    the only way to see it: the flag comes from magi-core's own
    ``is_mage_local``, not from a match magi-rs maintains.

    Args:
        state: What the run turned out to hold.

    Returns:
        Finding: PASS when ``mage_local`` is exactly True.
    """
    if state.hop is None:
        return _finding(S21_ASSERTIONS, 1, state.outcome, state.detail)
    local = state.hop.get(MAGE_LOCAL_KEY)
    if local is True:
        return _finding(S21_ASSERTIONS, 1, Outcome.PASS, "")
    if MAGE_LOCAL_KEY not in state.hop:
        return _finding(S21_ASSERTIONS, 1, Outcome.FAIL,
                        "the hop for %s publishes no %s, so nothing says "
                        "whether the cause condemned one mage or the run"
                        % (state.attempt.label, MAGE_LOCAL_KEY))
    return _finding(S21_ASSERTIONS, 1, Outcome.FAIL,
                    "the hop for %s reports %s as %r, so an empty completion "
                    "is being treated as a run-wide condition"
                    % (state.attempt.label, MAGE_LOCAL_KEY, local))


def _finish_finding(state):
    """Judge assertion 3: exhaustion and an empty answer stay distinguishable.

    ``finish`` is null when the backend did not say why the model stopped, and
    that distinction is the assertion's subject: substituting a default would
    assert a measurement nobody took, which is exactly what made a completion
    scraping its cap look healthy. So a null is ``CANNOT_TEST`` -- the run
    cannot answer -- while an absent key is a broken contract and a value that
    is not a string is a shape the vocabulary does not have.

    Args:
        state: What the run turned out to hold.

    Returns:
        Finding: PASS when the empty attempt carries a finish reason that
        tells ``length`` from anything else.
    """
    if state.attempt is None:
        return _finding(S21_ASSERTIONS, 2, state.outcome, state.detail)
    record = state.attempt.record
    if FINISH_KEY not in record:
        return _finding(S21_ASSERTIONS, 2, Outcome.FAIL,
                        "attempt %s omits %s, so budget exhaustion cannot be "
                        "told from a genuinely empty answer"
                        % (state.attempt.label, FINISH_KEY))
    finish = record[FINISH_KEY]
    if finish is None:
        return _finding(S21_ASSERTIONS, 2, Outcome.CANNOT_TEST,
                        "the backend reported no finish reason for attempt "
                        "%s, so this run cannot tell budget exhaustion from a "
                        "genuinely empty answer" % state.attempt.label)
    if not isinstance(finish, str) or not finish:
        return _finding(S21_ASSERTIONS, 2, Outcome.FAIL,
                        "attempt %s reports %s as %r, which is not a finish "
                        "reason the wire vocabulary has"
                        % (state.attempt.label, FINISH_KEY, finish))
    return _finding(S21_ASSERTIONS, 2, Outcome.PASS, "")


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
