# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""S1 -- the output contract still holds.

Protects ``tests/golden/headless_output_v1.json``. The golden file pins the
shape against the ``Cargo.lock`` of the day it was written; this scenario pins
it against the binary an operator actually runs, built from today's resolved
dependencies.

**Assertion 4 is where a defect would be silent, and it is the only reason the
scenario earns its place.** ``AppliedCaps`` carries ``#[serde(flatten)]`` over
``BudgetTelemetry``, so two fields sharing a name across that boundary make
``serde_json`` write the key twice into one map: the second write wins and the
first value **vanishes from the output with no error, no warning, and a
still-valid JSON document**. Nothing invalid appears, so every check that asks
"is what I found well formed?" passes. Only an EXACT key set sees it -- a
subset check cannot detect a key that is simply gone.

**Assertion 3 is a superset check and assertion 4 is an exact one, on purpose.**
REQ-H14 declares a new top-level key additive: a consumer must tolerate one, so
demanding exactness there would turn every compatible release into a red. One
level down the flatten changes the stakes, because an addition there is capable
of deleting a sibling.

The scenario catches its own parse failure rather than letting it reach the
runner. The runner turns a ``ProductOutputError`` into ONE finding for the whole
scenario, which is right for a scenario built around a single question and wrong
here: three assertions would disappear from the report without anyone noticing
they were never evaluated, and assertion 1's subject IS whether the output
parses.
"""

from smoke.errors import ProductOutputError
from smoke.outcome import Finding, Outcome
from smoke.registry import scenario

#: The verbatim assertion texts of the spec's section 8, for S1.
ASSERTIONS = (
    "exit 0 and stdout parses as JSON",
    "schema_version is 1",
    "every top-level contract key is present",
    "every nested key of applied_caps is present",
)

#: The run S1 reads, named once so the finding's ``run_id`` and the decorator
#: cannot drift apart.
RUN_ID = "R1"

#: The exit code a healthy ``query`` run reports.
SUCCESS_EXIT_CODE = 0

#: The contract version this scenario was written against. A bump is a
#: deliberate act by whoever changes the contract, so it belongs in the harness
#: as a constant to update, never as "whatever the product said".
EXPECTED_SCHEMA_VERSION = 1

#: The key holding the flattened pair, and the one assertion 4 walks.
APPLIED_CAPS_KEY = "applied_caps"

#: Every key ``RunOutcome`` serialises (``src/headless/types.rs``). Checked as a
#: SUBSET -- see the module docstring for why this level and the next differ.
TOP_LEVEL_KEYS = (
    "applied_caps",
    "consult",
    "error",
    "model",
    "provider",
    "response",
    "schema_version",
    "stop_reason",
    "timings",
    "tool_calls",
    "transcript",
    "usage",
)

#: The nine keys ``applied_caps`` serialises: four of its own plus the five
#: ``BudgetTelemetry`` flattens in beside them. Checked for EXACT equality,
#: which is the whole point of the assertion.
APPLIED_CAPS_KEYS = (
    "ceiling_above_sanity",
    "ceiling_floored",
    "floor_activation_threshold_secs",
    "max_rotations_effective",
    "max_tool_calls",
    "max_tool_calls_clamped",
    "operation_budget_secs",
    "system_override_applied",
    "timeout_secs",
)


@scenario("S1", run=RUN_ID, needs_backend=True)
def the_output_contract_still_holds(run):
    """Check the four contract properties over R1's capture.

    Args:
        run: R1's ``RunResult``, or ``None`` when the run never produced one.

    Yields:
        Finding: One per entry of :data:`ASSERTIONS`, in that order, whatever
        happened -- a scenario that stopped early would be read as three
        assertions that passed.
    """
    if run is None:
        for index in range(len(ASSERTIONS)):
            yield _s1(index, Outcome.CANNOT_TEST,
                      "run %s produced no capture to inspect" % RUN_ID)
        return

    output = run.output
    try:
        document = output.json()
    except ProductOutputError as exc:
        yield _s1(0, Outcome.FAIL, str(exc))
        for index in range(1, len(ASSERTIONS)):
            yield _s1(index, Outcome.CANNOT_TEST,
                      "stdout did not parse, so there is no object to read")
        return

    yield _exit_and_parse_finding(output)
    yield _schema_version_finding(document)
    yield _top_level_finding(document)
    yield _applied_caps_finding(output)


def _exit_and_parse_finding(output):
    """Judge assertion 1: the process succeeded and stdout was an object.

    The parse already happened by the time this runs, so only the exit code is
    left to decide.

    Args:
        output: R1's capture.

    Returns:
        Finding: PASS when the product exited zero.
    """
    if output.exit_code != SUCCESS_EXIT_CODE:
        return _s1(0, Outcome.FAIL,
                   "the product exited %d; stdout parsed, so the contract "
                   "question is the exit code" % output.exit_code)
    return _s1(0, Outcome.PASS, "")


def _schema_version_finding(document):
    """Judge assertion 2 against the version this harness was written for.

    Args:
        document: The parsed output.

    Returns:
        Finding: PASS when the declared version is the expected one.
    """
    declared = document.get("schema_version")
    if declared != EXPECTED_SCHEMA_VERSION:
        return _s1(1, Outcome.FAIL,
                   "schema_version is %r, expected %d"
                   % (declared, EXPECTED_SCHEMA_VERSION))
    return _s1(1, Outcome.PASS, "")


def _top_level_finding(document):
    """Judge assertion 3: no declared top-level key went missing.

    Complexity: ``O(len(TOP_LEVEL_KEYS))``.

    Args:
        document: The parsed output.

    Returns:
        Finding: PASS when every expected key is present. An UNEXPECTED key is
        not a failure here; REQ-H14 declares additions at this level
        compatible.
    """
    missing = sorted(set(TOP_LEVEL_KEYS) - set(document))
    if missing:
        return _s1(2, Outcome.FAIL,
                   "the output is missing %d top-level contract key%s: %s"
                   % (len(missing), "" if len(missing) == 1 else "s",
                      ", ".join(missing)))
    return _s1(2, Outcome.PASS, "")


def _applied_caps_finding(output):
    """Judge assertion 4 by comparing the EXACT key set of ``applied_caps``.

    Read through :meth:`~smoke.product.ProductOutput.key` so an output whose
    ``applied_caps`` is absent or is not an object becomes this assertion's own
    failure rather than an exception from somewhere deeper.

    Complexity: ``O(len(APPLIED_CAPS_KEYS))``.

    Args:
        output: R1's capture.

    Returns:
        Finding: PASS only on exact equality. Both directions are named
        separately in the detail, because a vanished key and an unreviewed one
        are different events that happen to be one comparison.
    """
    try:
        caps = output.key(APPLIED_CAPS_KEY)
    except ProductOutputError as exc:
        return _s1(3, Outcome.FAIL, str(exc))
    if not isinstance(caps, dict):
        return _s1(3, Outcome.FAIL,
                   "%s is %s, expected an object"
                   % (APPLIED_CAPS_KEY, type(caps).__name__))
    expected = set(APPLIED_CAPS_KEYS)
    present = set(caps)
    missing = sorted(expected - present)
    unexpected = sorted(present - expected)
    if missing or unexpected:
        return _s1(3, Outcome.FAIL,
                   "%s carries %d keys, expected %d: missing [%s], "
                   "unexpected [%s]"
                   % (APPLIED_CAPS_KEY, len(present), len(expected),
                      ", ".join(missing), ", ".join(unexpected)))
    return _s1(3, Outcome.PASS, "")


def _s1(index, outcome, detail):
    """Build the finding for one entry of :data:`ASSERTIONS`.

    Args:
        index: Position in :data:`ASSERTIONS`.
        outcome: What became of it.
        detail: The cause when the outcome is not PASS.

    Returns:
        Finding: The finding, stamped with the run it came from.
    """
    return Finding(assertion=ASSERTIONS[index], outcome=outcome, detail=detail,
                   run_id=RUN_ID)
