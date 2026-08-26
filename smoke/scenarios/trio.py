# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""S6, S7, S8 and S18 -- the four scenarios that hang off R4.

R4 is the fusion that saves the most: ``--timeout 300``,
``--structured-verdicts`` and the large payload are orthogonal, so one trio run
delivers the derived ceiling S7 reads, the structured keys S6 and S18 count, and
the large token count S8 asserts.

**The coupling is an accepted limitation, declared rather than pending.** If R4
falls, all four fall together and read as four defects when they are one.
Splitting them into four runs would multiply the harness's only expensive run by
four. What the design does instead is make the shared cause visible: each
assertion carries R4's id, and the outcome the four inherit depends on WHY the
run fell -- a product error of its own is ``FAIL``, an unreachable or failing
backend is ``CANNOT_TEST``, and a timeout is ``CANNOT_TEST`` for everyone except
S7.

**S7 is the exception because it classifies on EVIDENCE.** ``applied_caps`` is
emitted before the run hangs, so a ceiling that degraded the trio can be told
from a provider that was merely slow -- the first is failure #4 and is ``FAIL``,
the second is somebody else's load and is ``CANNOT_TEST``. **And when
``applied_caps`` never got emitted, S7 degrades too**: without that object there
is nothing to tell them apart, and inventing the distinction falls the wrong way,
because guessing ``FAIL`` accuses the product with no evidence.

**The budget arithmetic is mirrored here on purpose, and it is a maintenance
contract.** The ceiling itself is not published -- only the operation budget
derived from it -- so the relation REQ-A04 states cannot be checked without
recomputing it from what IS published: the run's wall clock and its rotation
count. When the product's own constants move, S7 goes red and this module has to
follow. That is the same bargain S18 takes on for the key counts, and for the
same reason: a check that adjusts itself to whatever the product says detects
nothing.

``retry_disabled`` is the one input that is not published, and it is
**recovered** rather than read from the harness's configuration:
``floor_activation_threshold_secs`` is a function of the attempt factor alone,
and the two candidate factors produce different thresholds, so the run's own
telemetry says which one it used.

**S18's counts are EXACT and that is where the value is.** ``AgentOutput`` and
``Finding`` are ``#[non_exhaustive]`` in magi-core: if the explicit mapping were
ever replaced by a direct interpolation, any field magi-core adds in a minor
release would reach the public JSON with no review here. An "at least 7" detects
none of it. The maintenance contract is accepted and not re-litigated: a
backward-compatible addition upstream turns S18 red and forces a harness change
even though the product is healthy. That IS the signal.
"""

import dataclasses
import math

from smoke import runs
from smoke.errors import ProductOutputError
from smoke.outcome import Finding, Outcome
from smoke.registry import scenario

#: The verbatim assertion texts of the spec's section 8, per scenario.
S6_ASSERTIONS = (
    "three verdicts are present",
    "a consensus was computed",
    "the run is not degraded",
)
S7_ASSERTIONS = (
    "applied_caps satisfies the derived relation between the operation "
    "budget, the client timeout and the ceiling",
    "the trio is not degraded under a high --timeout",
    "ceiling_floored and ceiling_above_sanity report what corresponds",
)
S8_ASSERTIONS = (
    "the large-payload run completes without truncation",
    "usage.input_tokens confirms the size in tokens, above the declared floor",
    "the generated payload stayed under the input cap",
)
S18_ASSERTIONS = (
    "with the flag, agents and consensus are both present",
    "without it, both are absent",
    "agents[] exposes exactly 7 keys",
    "findings[] exposes exactly 6 keys",
)

#: The runs these four read.
TRIO_RUN = "R4"
BARE_RUN = "R8"

#: A MAGI trio is three seats.
TRIO_SIZE = 3

#: Keys of the headless output contract these scenarios reach for.
CONSULT_KEY = "consult"
APPLIED_CAPS_KEY = "applied_caps"
ERROR_KEY = "error"
INPUT_TOKENS_PATH = "usage.input_tokens"

#: Keys of the consult envelope.
AGENTS_KEY = "agents"
CONSENSUS_KEY = "consensus"
DEGRADED_KEY = "degraded"
FAILED_AGENTS_KEY = "failed_agents"
REPORT_TRUNCATED_KEY = "report_truncated"
CONSENSUS_VERDICT_KEY = "consensus_verdict"

#: The label ``report_truncated`` carries when nothing was cut.
NOT_TRUNCATED = "none"

#: The exit code a healthy run reports.
SUCCESS_EXIT_CODE = 0

#: Error classes that are the ENVIRONMENT's failure rather than the product's,
#: so a scenario reading them degrades instead of accusing. Everything else the
#: product can report is the product having spoken and said no.
ENVIRONMENTAL_ERROR_KINDS = ("provider", "timeout")

#: The seven keys ``agents[]`` exposes, mapped field by field in
#: ``src/tools/consult.rs``. Counted EXACTLY.
AGENT_KEYS = ("agent", "verdict", "confidence", "summary", "reasoning",
              "findings", "recommendation")

#: The six keys ``findings[]`` exposes. Counted EXACTLY, same reasoning.
FINDING_KEYS = ("severity", "title", "detail", "file", "line", "category")

#: The product's own budget constants, mirrored (see the module docstring).
#: ``magi/mod.rs``: the classifier's slice of the wall clock, the headless slack
#: percentage, the ceiling's absolute floor, and the sanity bound above which a
#: derived ceiling is reported as a probable typo but is NOT clamped.
CLASSIFY_TIMEOUT_SECS = 6
TIMEOUT_SLACK_PCT = 20
CEILING_FLOOR_SECS = 15
CEILING_SANITY_SECS = 600

#: The two derived layers, as the numerator/denominator pairs the product uses,
#: each with the minimum it is floored at. 6/10 + 3/10 leaves the 10 % margin
#: that makes abandonment surface as a typed error rather than an opaque cutoff.
BUDGET_NUM, BUDGET_DEN, BUDGET_MIN = 6, 10, 10
CLIENT_NUM, CLIENT_DEN, CLIENT_MIN = 3, 10, 5

#: Attempts per model, with and without retry.
ATTEMPTS_WITH_RETRY = 2
ATTEMPTS_WITHOUT_RETRY = 1

#: The product's input cap. A payload above it is REJECTED, not truncated.
MAX_QUERY_BYTES = 256 * 1024

#: How far below its declared size the payload may land before the scenario
#: concludes the run never carried it. Half is generous: nothing legitimately
#: halves the payload, and a run sending a one-line prompt is nowhere near it.
PAYLOAD_SENT_FRACTION = 0.5


@dataclasses.dataclass(frozen=True)
class Derivation:
    """The budget layers recomputed from one run's published telemetry.

    Attributes:
        factor: The attempt factor the run used, recovered from its threshold.
        raw: The ceiling before the floor is applied.
        ceiling: The ceiling that governed the run.
        budget: The per-attempt operation budget the ceiling implies.
        client: The client timeout it implies.
        floored: Whether the raw derivation fell below the absolute floor.
        above_sanity: Whether the ceiling exceeded the sanity bound.
    """

    factor: int
    raw: int
    ceiling: int
    budget: int
    client: int
    floored: bool
    above_sanity: bool


def attempt_factor(max_rotations, retry_disabled):
    """How many attempt-widths the wall clock has to cover.

    Args:
        max_rotations: The resolved rotation count.
        retry_disabled: Whether retry is off.

    Returns:
        int: Attempts per model times models per mage times the slack
        percentage, in hundredths.
    """
    attempts = ATTEMPTS_WITHOUT_RETRY if retry_disabled else ATTEMPTS_WITH_RETRY
    return attempts * (max_rotations + 1) * (100 + TIMEOUT_SLACK_PCT)


def raw_ceiling(timeout_secs, factor):
    """The per-mage ceiling a wall clock derives, before the floor.

    Args:
        timeout_secs: The run's wall clock.
        factor: The attempt factor.

    Returns:
        int: The raw ceiling in seconds, never negative -- the subtraction
        saturates, because a wall clock at or under the classifier's own slice
        derives a minimal ceiling, not an astronomical one.
    """
    return max(timeout_secs - CLASSIFY_TIMEOUT_SECS, 0) * 100 // factor


def floor_activation_threshold(factor):
    """The smallest wall clock whose ceiling still reaches the floor.

    Args:
        factor: The attempt factor.

    Returns:
        int: The threshold in seconds. Rounded UP, because what is wanted is
        the smallest dividend whose TRUNCATING division still reaches the
        floor.
    """
    return CLASSIFY_TIMEOUT_SECS + math.ceil(CEILING_FLOOR_SECS * factor / 100)


def operation_budget(ceiling_secs):
    """The per-attempt budget a ceiling implies.

    Args:
        ceiling_secs: The governing ceiling.

    Returns:
        int: The budget in seconds, floored at :data:`BUDGET_MIN`.
    """
    return max(ceiling_secs * BUDGET_NUM // BUDGET_DEN, BUDGET_MIN)


def client_timeout(ceiling_secs):
    """The client timeout a ceiling implies.

    Args:
        ceiling_secs: The governing ceiling.

    Returns:
        int: The timeout in seconds, floored at :data:`CLIENT_MIN`.
    """
    return max(ceiling_secs * CLIENT_NUM // CLIENT_DEN, CLIENT_MIN)


def derive(caps):
    """Recompute the budget layers from what one run published.

    Complexity: ``O(1)`` -- the factor search is over the two values
    ``retry_disabled`` can take.

    Args:
        caps: The run's ``applied_caps``.

    Returns:
        Derivation | None: The recomputation, or ``None`` when the run
        published no wall clock (the configured path, whose ceiling is not in
        this object) or when no attempt factor explains the threshold it
        published, which is the telemetry contradicting itself.
    """
    timeout_secs = caps.get("timeout_secs")
    rotations = caps.get("max_rotations_effective")
    threshold = caps.get("floor_activation_threshold_secs")
    if not _is_count(timeout_secs) or not _is_count(rotations):
        return None
    for retry_disabled in (False, True):
        factor = attempt_factor(rotations, retry_disabled)
        if floor_activation_threshold(factor) != threshold:
            continue
        raw = raw_ceiling(timeout_secs, factor)
        ceiling = max(raw, CEILING_FLOOR_SECS)
        return Derivation(
            factor=factor, raw=raw, ceiling=ceiling,
            budget=operation_budget(ceiling), client=client_timeout(ceiling),
            floored=raw < CEILING_FLOOR_SECS,
            above_sanity=ceiling > CEILING_SANITY_SECS)
    return None


def _is_count(value):
    """Whether a JSON value is a usable non-negative integer.

    Args:
        value: The value to check.

    Returns:
        bool: True for a non-negative ``int`` that is not a bool. JSON's
        booleans are Python ints, and one arriving where a count belongs is a
        contract break, not a zero.
    """
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


class _Envelope:
    """One run reduced to the consult object, or to why there is none.

    Attributes:
        body: The consult envelope, or None.
        outcome: What every assertion over this run must report when *body* is
            None.
        detail: Why.
    """

    def __init__(self, body, outcome=None, detail=""):
        """Store the reduction.

        Args:
            body: The envelope, or None.
            outcome: The outcome to inherit; None when *body* is set.
            detail: The cause.
        """
        self.body = body
        self.outcome = outcome
        self.detail = detail


def _envelope_of(result, run_id):
    """Reduce one shared run to its consult envelope or to a shared cause.

    The classification is SS5.1's table: a product error of its own is the
    product having spoken, an unreachable or failing backend is the
    environment, and a capture that cannot be read at all is a broken contract.

    Args:
        result: The run's ``RunResult``, or None.
        run_id: Which run, for the message.

    Returns:
        _Envelope: The envelope, or the outcome every assertion inherits.
    """
    if result is None:
        return _Envelope(None, Outcome.CANNOT_TEST,
                         "run %s produced no capture to inspect" % run_id)
    try:
        document = result.output.json()
    except ProductOutputError as exc:
        return _Envelope(None, Outcome.FAIL,
                         "run %s: %s" % (run_id, exc))
    error = document.get(ERROR_KEY)
    if isinstance(error, dict):
        kind = error.get("kind")
        outcome = (Outcome.CANNOT_TEST if kind in ENVIRONMENTAL_ERROR_KINDS
                   else Outcome.FAIL)
        return _Envelope(None, outcome,
                         "run %s reported a %s error: %s"
                         % (run_id, kind, error.get("message", "")))
    body = document.get(CONSULT_KEY)
    if not isinstance(body, dict):
        return _Envelope(None, Outcome.FAIL,
                         "run %s exited %d but carries no consult envelope, "
                         "so the trio produced nothing to read"
                         % (run_id, result.output.exit_code))
    return _Envelope(body)


def _degradation_cause(body):
    """Why an envelope counts as degraded, or the empty string.

    ``degraded`` is not the only signal: a seat that fell leaves a typed cause
    in ``failed_agents``, and a check reading only the boolean would call that
    run healthy.

    Args:
        body: The consult envelope.

    Returns:
        str: The cause, or "" when the run is not degraded.
    """
    if body.get(DEGRADED_KEY) is True:
        return "the envelope reports degraded"
    failed = body.get(FAILED_AGENTS_KEY)
    if isinstance(failed, dict) and failed:
        return "seats failed: %s" % ", ".join(sorted(failed))
    return ""


@scenario("S6", run=TRIO_RUN, needs_backend=True)
def the_real_trio_reaches_consensus(run):
    """Count the verdicts, check the consensus, and refuse a degraded run.

    Args:
        run: R4's ``RunResult``, or None.

    Yields:
        Finding: One per entry of :data:`S6_ASSERTIONS`, in that order.
    """
    envelope = _envelope_of(run, TRIO_RUN)
    if envelope.body is None:
        for index in range(len(S6_ASSERTIONS)):
            yield _finding(S6_ASSERTIONS, index, envelope.outcome, TRIO_RUN,
                           envelope.detail)
        return
    body = envelope.body
    agents = body.get(AGENTS_KEY)
    if not isinstance(agents, list) or len(agents) != TRIO_SIZE:
        yield _finding(S6_ASSERTIONS, 0, Outcome.FAIL, TRIO_RUN,
                       "the envelope carries %s verdicts, expected %d"
                       % (len(agents) if isinstance(agents, list) else "no",
                          TRIO_SIZE))
    else:
        yield _finding(S6_ASSERTIONS, 0, Outcome.PASS, TRIO_RUN, "")

    consensus = body.get(CONSENSUS_KEY)
    verdict = (consensus.get(CONSENSUS_VERDICT_KEY)
               if isinstance(consensus, dict) else None)
    if not isinstance(verdict, str) or not verdict.strip():
        yield _finding(S6_ASSERTIONS, 1, Outcome.FAIL, TRIO_RUN,
                       "no consensus verdict was computed: %r" % (verdict,))
    else:
        yield _finding(S6_ASSERTIONS, 1, Outcome.PASS, TRIO_RUN, "")

    cause = _degradation_cause(body)
    yield _finding(S6_ASSERTIONS, 2,
                   Outcome.FAIL if cause else Outcome.PASS, TRIO_RUN, cause)


@scenario("S7", run=TRIO_RUN, needs_backend=True, inspects_timeouts=True)
def the_derived_ceiling_does_not_degrade_the_trio(run):
    """Recompute the budget layers, then judge the run against them.

    Args:
        run: R4's ``RunResult``, which S7 receives even when the run timed out.

    Yields:
        Finding: One per entry of :data:`S7_ASSERTIONS`, in that order.
    """
    caps, caps_failure = _caps_of(run)
    derived = derive(caps) if caps is not None else None
    yield _relation_finding(derived, caps, caps_failure)
    yield _trio_health_finding(run, derived, caps_failure)
    yield _flags_finding(derived, caps, caps_failure)


def _caps_of(run):
    """The ``applied_caps`` of a run, even a partial one.

    Args:
        run: The run's result, or None.

    Returns:
        tuple: ``(caps, failure)``; exactly one of the two is set.
    """
    if run is None:
        return None, "run %s produced no capture to inspect" % TRIO_RUN
    try:
        caps = run.output.key(APPLIED_CAPS_KEY)
    except ProductOutputError as exc:
        return None, (
            "run %s emitted no %s, so there is nothing to tell a degraded "
            "ceiling from a slow provider: %s"
            % (TRIO_RUN, APPLIED_CAPS_KEY, exc)
        )
    if not isinstance(caps, dict):
        return None, ("%s is %s, expected an object"
                      % (APPLIED_CAPS_KEY, type(caps).__name__))
    return caps, ""


def _relation_finding(derived, caps, failure):
    """Judge assertion 1: the published budget is the one the formula gives.

    Args:
        derived: The recomputation, or None.
        caps: The published telemetry, or None.
        failure: Why there is none.

    Returns:
        Finding: PASS when the published budget matches AND the two inner
        layers fit inside the ceiling.
    """
    if caps is None:
        return _finding(S7_ASSERTIONS, 0, Outcome.CANNOT_TEST, TRIO_RUN,
                        failure)
    if derived is None:
        return _finding(S7_ASSERTIONS, 0, Outcome.CANNOT_TEST, TRIO_RUN,
                        "the run published no wall clock, or no attempt factor "
                        "explains the floor threshold it published, so the "
                        "ceiling cannot be recomputed from evidence")
    published = caps.get("operation_budget_secs")
    if published != derived.budget:
        return _finding(S7_ASSERTIONS, 0, Outcome.FAIL, TRIO_RUN,
                        "the run published an operation budget of %rs; a "
                        "ceiling of %ds derived from a %ds wall clock implies "
                        "%ds"
                        % (published, derived.ceiling,
                           caps.get("timeout_secs"), derived.budget))
    if derived.budget + derived.client > derived.ceiling:
        return _finding(S7_ASSERTIONS, 0, Outcome.FAIL, TRIO_RUN,
                        "REQ-A04 is broken: %ds of budget plus %ds of client "
                        "timeout exceed the %ds ceiling"
                        % (derived.budget, derived.client, derived.ceiling))
    return _finding(S7_ASSERTIONS, 0, Outcome.PASS, TRIO_RUN, "")


def _trio_health_finding(run, derived, failure):
    """Judge assertion 2, telling a degraded ceiling from a slow provider.

    Args:
        run: The run's result, or None.
        derived: The recomputation, or None.
        failure: Why there is no telemetry.

    Returns:
        Finding: FAIL when the trio degraded, or when the run hung under a
        ceiling that had been floored -- that second case IS failure #4.
        CANNOT_TEST when it hung under an adequate ceiling, because a slow
        provider is not a defect of the product.
    """
    if run is not None and run.timed_out:
        if derived is None:
            return _finding(S7_ASSERTIONS, 1, Outcome.CANNOT_TEST, TRIO_RUN,
                            failure or "the run hung and published no usable "
                                       "telemetry")
        if derived.floored:
            return _finding(S7_ASSERTIONS, 1, Outcome.FAIL, TRIO_RUN,
                            "the run hung under a ceiling the wall clock had "
                            "driven below the %ds floor: that is the derived "
                            "ceiling degrading the trio, not a slow provider"
                            % CEILING_FLOOR_SECS)
        return _finding(S7_ASSERTIONS, 1, Outcome.CANNOT_TEST, TRIO_RUN,
                        "the run hung under an adequate ceiling of %ds, which "
                        "is the provider's pace and not the product's defect"
                        % derived.ceiling)
    envelope = _envelope_of(run, TRIO_RUN)
    if envelope.body is None:
        return _finding(S7_ASSERTIONS, 1, envelope.outcome, TRIO_RUN,
                        envelope.detail)
    cause = _degradation_cause(envelope.body)
    return _finding(S7_ASSERTIONS, 1,
                    Outcome.FAIL if cause else Outcome.PASS, TRIO_RUN, cause)


def _flags_finding(derived, caps, failure):
    """Judge assertion 3: both booleans say what the arithmetic says.

    Args:
        derived: The recomputation, or None.
        caps: The published telemetry, or None.
        failure: Why there is none.

    Returns:
        Finding: PASS when both flags agree with the recomputation.
    """
    if caps is None:
        return _finding(S7_ASSERTIONS, 2, Outcome.CANNOT_TEST, TRIO_RUN,
                        failure)
    if derived is None:
        return _finding(S7_ASSERTIONS, 2, Outcome.CANNOT_TEST, TRIO_RUN,
                        "the ceiling cannot be recomputed, so neither flag "
                        "has anything to be checked against")
    disagreements = [
        "%s is %r, the arithmetic says %r" % (key, caps.get(key), expected)
        for key, expected in (("ceiling_floored", derived.floored),
                              ("ceiling_above_sanity", derived.above_sanity))
        if caps.get(key) is not expected
    ]
    if disagreements:
        return _finding(S7_ASSERTIONS, 2, Outcome.FAIL, TRIO_RUN,
                        "; ".join(disagreements))
    return _finding(S7_ASSERTIONS, 2, Outcome.PASS, TRIO_RUN, "")


@scenario("S8", run=TRIO_RUN, needs_backend=True)
def a_large_input_arrives_large(run):
    """Check the run completed whole, arrived large, and stayed under the cap.

    Args:
        run: R4's ``RunResult``, or None.

    Yields:
        Finding: One per entry of :data:`S8_ASSERTIONS`, in that order.
    """
    envelope = _envelope_of(run, TRIO_RUN)
    yield _untruncated_finding(run, envelope)
    yield _token_floor_finding(run)
    yield _under_cap_finding(run)


def _untruncated_finding(run, envelope):
    """Judge assertion 1: the run finished and nothing was cut.

    Args:
        run: The run's result, or None.
        envelope: Its reduction.

    Returns:
        Finding: PASS when the product exited zero and reported no truncation.
    """
    if envelope.body is None:
        return _finding(S8_ASSERTIONS, 0, envelope.outcome, TRIO_RUN,
                        envelope.detail)
    if run.output.exit_code != SUCCESS_EXIT_CODE:
        return _finding(S8_ASSERTIONS, 0, Outcome.FAIL, TRIO_RUN,
                        "the product exited %d" % run.output.exit_code)
    level = envelope.body.get(REPORT_TRUNCATED_KEY)
    if level != NOT_TRUNCATED:
        return _finding(S8_ASSERTIONS, 0, Outcome.FAIL, TRIO_RUN,
                        "the report was truncated at the %r level" % (level,))
    return _finding(S8_ASSERTIONS, 0, Outcome.PASS, TRIO_RUN, "")


def _token_floor_finding(run):
    """Judge assertion 2: the product counted more tokens than the floor.

    This is an ASSERTION, not a verified estimate: the harness picks the bytes
    and asks the product how many tokens they became. A failure reports the
    observed count beside the byte size, which are the two numbers that decide
    whether the declared size or the model is the problem.

    Args:
        run: The run's result, or None.

    Returns:
        Finding: CANNOT_TEST when the run never carried the payload -- the
        harness must not accuse the product of a size it was never sent.
    """
    if run is None:
        return _finding(S8_ASSERTIONS, 1, Outcome.CANNOT_TEST, TRIO_RUN,
                        "run %s produced no capture to inspect" % TRIO_RUN)
    sent = run.stdin_bytes
    target = runs.payload_target()
    if sent < target * PAYLOAD_SENT_FRACTION:
        return _finding(S8_ASSERTIONS, 1, Outcome.CANNOT_TEST, TRIO_RUN,
                        "run %s carried %d bytes against a declared payload of "
                        "%d, so the large input was never sent and the token "
                        "count says nothing about it"
                        % (TRIO_RUN, sent, target))
    floor = runs.payload_floor()
    try:
        observed = run.output.key(INPUT_TOKENS_PATH)
    except ProductOutputError as exc:
        return _finding(S8_ASSERTIONS, 1, Outcome.CANNOT_TEST, TRIO_RUN,
                        str(exc))
    if not _is_count(observed):
        return _finding(S8_ASSERTIONS, 1, Outcome.CANNOT_TEST, TRIO_RUN,
                        "%s is %r, which is not a token count"
                        % (INPUT_TOKENS_PATH, observed))
    if observed < floor:
        return _finding(S8_ASSERTIONS, 1, Outcome.FAIL, TRIO_RUN,
                        "%d bytes became %d input tokens, below the declared "
                        "floor of %d" % (sent, observed, floor))
    return _finding(S8_ASSERTIONS, 1, Outcome.PASS, TRIO_RUN, "")


def _under_cap_finding(run):
    """Judge assertion 3: the payload fits inside the product's input cap.

    The product REJECTS an oversized query rather than truncating it, so a
    payload over the cap does not produce a smaller run -- it produces no run
    at all. Checking the bytes the harness chose is what keeps that from being
    discovered as an opaque product error.

    Args:
        run: The run's result, or None.

    Returns:
        Finding: PASS when what the run CARRIED is under the cap. Reading the
        declaration instead is what let this pass over a 54-byte prompt while
        claiming to have checked a quarter of a megabyte.
    """
    if run is None:
        return _finding(S8_ASSERTIONS, 2, Outcome.CANNOT_TEST, TRIO_RUN,
                        "run %s produced no capture, so what it carried is "
                        "unknown" % TRIO_RUN)
    sent = run.stdin_bytes
    if sent >= MAX_QUERY_BYTES:
        return _finding(S8_ASSERTIONS, 2, Outcome.FAIL, TRIO_RUN,
                        "run %s carries %d bytes, at or above the product's "
                        "%d-byte input cap, which rejects rather than "
                        "truncates" % (TRIO_RUN, sent, MAX_QUERY_BYTES))
    return _finding(S8_ASSERTIONS, 2, Outcome.PASS, TRIO_RUN, "")


@scenario("S18", run=(TRIO_RUN, BARE_RUN), needs_backend=True)
def the_shape_varies_by_flag_never_by_data(run):
    """Compare the flagged envelope against the flagless one, and count keys.

    Args:
        run: The two ``RunResult`` objects keyed by id.

    Yields:
        Finding: One per entry of :data:`S18_ASSERTIONS`, in that order.
    """
    results = run or {}
    flagged = _envelope_of(results.get(TRIO_RUN), TRIO_RUN)
    bare = _envelope_of(results.get(BARE_RUN), BARE_RUN)
    yield _both_present_finding(flagged)
    yield _both_absent_finding(bare)
    agents = (flagged.body.get(AGENTS_KEY) if flagged.body is not None
              else None)
    yield _exact_keys_finding(S18_ASSERTIONS, 2, agents, AGENT_KEYS,
                              AGENTS_KEY, flagged)
    yield _exact_keys_finding(S18_ASSERTIONS, 3, _findings_of(agents),
                              FINDING_KEYS, "findings", flagged)


def _both_present_finding(flagged):
    """Judge assertion 1: the flag put both keys in.

    Args:
        flagged: R4's reduction.

    Returns:
        Finding: PASS when both keys are present.
    """
    if flagged.body is None:
        return _finding(S18_ASSERTIONS, 0, flagged.outcome, TRIO_RUN,
                        flagged.detail)
    absent = [key for key in (AGENTS_KEY, CONSENSUS_KEY)
              if key not in flagged.body]
    if absent:
        return _finding(S18_ASSERTIONS, 0, Outcome.FAIL, TRIO_RUN,
                        "the flag was passed and %s missing from the envelope"
                        % " and ".join(absent))
    return _finding(S18_ASSERTIONS, 0, Outcome.PASS, TRIO_RUN, "")


def _both_absent_finding(bare):
    """Judge assertion 2: without the flag neither key appears.

    Args:
        bare: R8's reduction.

    Returns:
        Finding: PASS when neither key is present.
    """
    if bare.body is None:
        return _finding(S18_ASSERTIONS, 1, bare.outcome, BARE_RUN, bare.detail)
    present = [key for key in (AGENTS_KEY, CONSENSUS_KEY) if key in bare.body]
    if present:
        return _finding(S18_ASSERTIONS, 1, Outcome.FAIL, BARE_RUN,
                        "no flag was passed and %s reached the envelope anyway"
                        % " and ".join(present))
    return _finding(S18_ASSERTIONS, 1, Outcome.PASS, BARE_RUN, "")


def _findings_of(agents):
    """Every finding across every seat, or None when there are no seats.

    Args:
        agents: The ``agents`` array, or None.

    Returns:
        list | None: The findings, or None when the seats could not be read.
    """
    if not isinstance(agents, list):
        return None
    collected = []
    for seat in agents:
        if isinstance(seat, dict) and isinstance(seat.get("findings"), list):
            collected.extend(seat["findings"])
    return collected


def _exact_keys_finding(texts, index, entries, expected, label, flagged):
    """Judge one EXACT key count over a list of objects.

    Complexity: ``O(number of entries)``, each compared against a set of fixed
    size.

    Args:
        texts: The scenario's assertion texts.
        index: Which assertion.
        entries: The objects to count, or None when they could not be read.
        expected: The exact key names.
        label: What the array is called, for the message.
        flagged: R4's reduction, for the cause when there is no envelope.

    Returns:
        Finding: PASS on exact equality for every entry; CANNOT_TEST when the
        array is empty, because an empty array is the positive certificate that
        nothing was produced, not a shape that is wrong.
    """
    if flagged.body is None:
        return _finding(texts, index, flagged.outcome, TRIO_RUN, flagged.detail)
    if entries is None:
        return _finding(texts, index, Outcome.FAIL, TRIO_RUN,
                        "%s is not an array, so its shape cannot be counted"
                        % label)
    if not entries:
        return _finding(texts, index, Outcome.CANNOT_TEST, TRIO_RUN,
                        "the run produced no %s, so there is no shape to "
                        "count" % label)
    wanted = set(expected)
    problems = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            problems.append("%s[%d] is not an object" % (label, position))
            continue
        missing = sorted(wanted - set(entry))
        unexpected = sorted(set(entry) - wanted)
        if missing or unexpected:
            problems.append(
                "%s[%d] has %d keys, expected %d: missing [%s], unexpected "
                "[%s]" % (label, position, len(entry), len(wanted),
                          ", ".join(missing), ", ".join(unexpected)))
    if problems:
        return _finding(texts, index, Outcome.FAIL, TRIO_RUN,
                        "; ".join(problems))
    return _finding(texts, index, Outcome.PASS, TRIO_RUN, "")


def _finding(texts, index, outcome, run_id, detail):
    """Build one finding.

    Args:
        texts: The scenario's assertion texts.
        index: Position in *texts*.
        outcome: What became of it.
        run_id: The shared run it came from.
        detail: The cause when the outcome is not PASS.

    Returns:
        Finding: The finding.
    """
    return Finding(assertion=texts[index], outcome=outcome, detail=detail,
                   run_id=run_id)
