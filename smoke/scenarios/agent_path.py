# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""S19 -- the agent's consult cap is still protected.

Protects REQ-EA02, and it is the scenario that best illustrates why a harness
like this exists at all.

**Its failure breaks nothing.** No exception, no different exit code, valid
JSON, a correct verdict. What happens is that the trio's full output enters the
agent's context window on **every** consult, and because an interactive session
resends that context each turn, the cost goes from once to per turn. No unit
violates its contract; the integrated application simply becomes expensive.

**Because nothing breaks, the assertion has to look one level down** -- at the
JSON string the consult tool returned to the agent, not at the top-level
response. A scenario that read the top-level object would pass whether or not
the defect is present, and would additionally go red on ``magi consult``, whose
response envelope carries those keys legitimately under the flag. The two
surfaces are different on purpose, and that difference is exactly what R5
exercises against R4: merging the two runs would delete the scenario.

Attribution is by tool NAME. Another tool returning something that happens to
carry an ``agents`` key says nothing about REQ-EA02, and crediting it here
would produce a red nobody can act on.

**Two absences degrade rather than accuse**: an agent that never reached for the
tool leaves no result to read, which is the model's behaviour and not the
product's contract; and a result that does not parse is a result the tool-result
cap cut, which is a fragment rather than evidence either way.
"""

import json

from smoke.errors import ProductOutputError
from smoke.outcome import Finding, Outcome
from smoke.registry import scenario

#: The verbatim assertion texts of the spec's section 8, for S19.
ASSERTIONS = (
    "the consult tool result inside tool_calls[] contains no agents key",
    "it contains no consensus key",
)

#: The run S19 reads: ``query --consult``, the AGENT's path to the trio.
RUN_ID = "R5"

#: The name the consult tool registers under.
CONSULT_TOOL_NAME = "consult"

#: Where the auditable record of every tool invocation lives.
TOOL_CALLS_KEY = "tool_calls"

#: The two keys that must never reach the agent-facing result, in assertion
#: order, so the loop below and :data:`ASSERTIONS` cannot drift apart.
FORBIDDEN_KEYS = ("agents", "consensus")


@scenario("S19", assertions=ASSERTIONS, run=RUN_ID, needs_backend=True)
def the_agents_consult_cap_is_still_protected(run):
    """Read every consult tool result R5 recorded and check both keys.

    Complexity: ``O(number of tool records)``, each with one JSON parse of its
    own result.

    Args:
        run: R5's ``RunResult``, or ``None`` when the run produced no capture.

    Yields:
        Finding: One per entry of :data:`ASSERTIONS`, in that order.
    """
    results, failure = _consult_results(run)
    for index, key in enumerate(FORBIDDEN_KEYS):
        if results is None:
            yield _s19(index, Outcome.CANNOT_TEST, failure)
            continue
        carrying = [position for position, body in enumerate(results)
                    if key in body]
        if carrying:
            yield _s19(index, Outcome.FAIL,
                       "%d of %d consult tool result%s carries a %r key, so "
                       "the trio's full output enters the agent's context on "
                       "every consult and is resent every turn"
                       % (len(carrying), len(results),
                          "" if len(results) == 1 else "s", key))
        else:
            yield _s19(index, Outcome.PASS, "")


def _consult_results(run):
    """Every consult tool result of the run, parsed.

    Args:
        run: R5's result, or None.

    Returns:
        tuple: ``(results, failure)``; exactly one of the two is set. *results*
        is the list of parsed tool results, never empty -- an empty one is
        reported as the failure it is, because a scenario that found no consult
        call has nothing to conclude.
    """
    if run is None:
        return None, "run %s produced no capture to inspect" % RUN_ID
    try:
        records = run.output.key(TOOL_CALLS_KEY)
    except ProductOutputError as exc:
        return None, str(exc)
    if not isinstance(records, list):
        return None, ("%s is %s, expected a list"
                      % (TOOL_CALLS_KEY, type(records).__name__))
    parsed = []
    unreadable = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("name") != CONSULT_TOOL_NAME:
            continue
        try:
            body = json.loads(record.get("result", ""))
        except (TypeError, ValueError):
            unreadable += 1
            continue
        if isinstance(body, dict):
            parsed.append(body)
        else:
            unreadable += 1
    if parsed:
        return parsed, ""
    if unreadable:
        return None, (
            "%d consult tool result%s could not be read as an object, which is "
            "what the tool-result cap leaves behind when it cuts a large "
            "envelope; a fragment is not evidence either way"
            % (unreadable, "" if unreadable == 1 else "s")
        )
    return None, (
        "the agent made no %s tool call, so there is no result to read. "
        "Whether it reaches for the tool is its behaviour, not the product's "
        "contract." % CONSULT_TOOL_NAME
    )


def _s19(index, outcome, detail):
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
