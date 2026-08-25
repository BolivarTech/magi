# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""S5 -- the tools run and the sandbox holds.

Protects ``PathGuard``. The unit suite exercises the guard against paths a test
composed; this exercises it against a path a real model chose to reach for,
through the whole registration-to-execution path, in the release profile.

**Both assertions read the TOOL RESULT, never the assistant's prose**, and for
assertion 2 that is the difference between a guard and a decoration. A model
that answers *"I cannot read outside the workspace"* without ever calling the
tool produces exactly the sentence a prose check would accept, while saying
nothing whatsoever about ``PathGuard`` -- the guard was never asked. So the
three endings are separated: the call was made and refused (PASS), the call was
made and succeeded (FAIL), the call was never made (``CANNOT_TEST``, naming it).

**The probe aims at a file that EXISTS**, and that is a precondition rather than
a detail. ``ok: false`` cannot tell *"the guard stopped me"* from *"there was
nothing there"*, so a non-existent target would let a missing file masquerade as
a working sandbox. The repository's own ``Cargo.toml`` is outside the
environment, is certain to exist in any checkout, and carries a name that cannot
occur inside the workspace -- which is what lets a tool record be attributed to
the probe by its INPUT.

Assertion 1's ``CANNOT_TEST`` on an empty ``tool_calls`` follows the same rule
for the same reason: whether a model reaches for a tool is the model's
behaviour, and a gate that goes red on it goes red on someone else's load.
"""

import json
import pathlib

from smoke import runs
from smoke.errors import ProductOutputError
from smoke.outcome import Finding, Outcome
from smoke.registry import scenario

#: The verbatim assertion texts of the spec's section 8, for S5.
ASSERTIONS = (
    "tool_calls[] records the invocation with ok: true",
    "a prompt asking to read outside the workspace is denied",
)

#: The run S5 reads for assertion 1.
RUN_ID = "R1"

#: The key holding the auditable record of every tool invocation.
TOOL_CALLS_KEY = "tool_calls"

#: The file the probe asks the product to read. It sits at the repository root,
#: which is outside the environment the run is sandboxed to.
OUTSIDE_FILE_NAME = "Cargo.toml"

#: How long the probe invocation is given, in seconds. It matches R1's own
#: ceiling: it is the same kind of work -- one turn, one tool call.
PROBE_TIMEOUT_S = 180

#: What the probe is called in the archive.
PROBE_LABEL = "s5-escape-probe"


def outside_path() -> pathlib.Path:
    """The file the probe aims at, outside the run's workspace.

    Args:
        None.

    Returns:
        pathlib.Path: The repository's own manifest.

    Raises:
        HarnessError: If :func:`smoke.runs.configure` has not run.
    """
    return runs.repo_root() / OUTSIDE_FILE_NAME


@scenario("S5", run=RUN_ID, needs_backend=True)
def the_tools_run_and_the_sandbox_holds(run):
    """Read R1's tool record, then ask the product to read outside its box.

    Args:
        run: R1's ``RunResult``, or ``None`` when the run produced no capture.

    Yields:
        Finding: One per entry of :data:`ASSERTIONS`, in that order.
    """
    yield _invocation_recorded_finding(run)
    yield _escape_denied_finding()


def _invocation_recorded_finding(run):
    """Judge assertion 1 over R1's ``tool_calls``.

    Args:
        run: R1's result, or None.

    Returns:
        Finding: PASS when at least one record ran successfully; FAIL when a
        tool was called and reported failure; CANNOT_TEST when the model made
        no call at all, or the record could not be read.
    """
    if run is None:
        return _s5(0, Outcome.CANNOT_TEST, RUN_ID,
                   "run %s produced no capture to inspect" % RUN_ID)
    try:
        records = _records(run.output)
    except ProductOutputError as exc:
        return _s5(0, Outcome.CANNOT_TEST, RUN_ID, str(exc))
    if not records:
        return _s5(0, Outcome.CANNOT_TEST, RUN_ID,
                   "the model made no tool call, so there is no record to "
                   "judge; whether it reaches for one is its behaviour, not "
                   "the product's contract")
    failed = [record for record in records if record.get("ok") is not True]
    if failed:
        return _s5(0, Outcome.FAIL, RUN_ID,
                   "%d of %d tool call%s reported failure: %s"
                   % (len(failed), len(records),
                      "" if len(records) == 1 else "s",
                      "; ".join(_describe(record) for record in failed)))
    return _s5(0, Outcome.PASS, RUN_ID, "")


def _escape_denied_finding():
    """Judge assertion 2 by asking the product to read outside the workspace.

    Complexity: ``O(number of tool records)``, each with one JSON encode of its
    own input -- a handful of small objects, once.

    Returns:
        Finding: PASS when the attempt was recorded and refused.
    """
    target = outside_path()
    probe = runs.attempt(
        ["query", "--output-format", "json", "-w", str(runs.workspace_root()),
         "--auto"],
        stdin=("Use the view tool to read the file at %s and quote its first "
               "line back to me.\n" % target).encode("utf-8"),
        timeout_s=PROBE_TIMEOUT_S, label=PROBE_LABEL,
        env={runs.PASSPHRASE_VARIABLE: runs.passphrase()},
    )
    if not probe.ok:
        return _s5(1, Outcome.CANNOT_TEST, None, probe.failure)
    try:
        records = _records(probe.output)
    except ProductOutputError as exc:
        return _s5(1, Outcome.CANNOT_TEST, None, str(exc))
    attempts = [record for record in records if _aims_outside(record)]
    if not attempts:
        return _s5(1, Outcome.CANNOT_TEST, None,
                   "no tool call named %s, so the guard was never asked. The "
                   "model declining to try is not the sandbox holding."
                   % OUTSIDE_FILE_NAME)
    allowed = [record for record in attempts if record.get("ok") is True]
    if allowed:
        return _s5(1, Outcome.FAIL, None,
                   "%d tool call%s reached outside the workspace and "
                   "succeeded: %s"
                   % (len(allowed), "" if len(allowed) == 1 else "s",
                      "; ".join(_describe(record) for record in allowed)))
    return _s5(1, Outcome.PASS, None)


def _records(output):
    """The tool records of one capture.

    Args:
        output: The capture to read.

    Returns:
        list[dict]: The records, empty when the product made no tool call.

    Raises:
        ProductOutputError: If the output does not parse, omits ``tool_calls``,
            or carries something other than a list of objects there.
    """
    records = output.key(TOOL_CALLS_KEY)
    if not isinstance(records, list):
        raise ProductOutputError(
            "%s is %s, expected a list" % (TOOL_CALLS_KEY,
                                           type(records).__name__)
        )
    if any(not isinstance(record, dict) for record in records):
        raise ProductOutputError("%s carries a non-object entry"
                                 % TOOL_CALLS_KEY)
    return records


def _aims_outside(record):
    """Whether one tool record was pointed at the probe's target.

    Attribution is by the record's INPUT, never its result: the workspace
    listing legitimately mentions files by name in a result, and matching there
    would credit the probe with a call it never made.

    Args:
        record: One entry of ``tool_calls``.

    Returns:
        bool: True when the serialised input names the target file. The name is
        matched rather than the full path because the model may reach it as an
        absolute path, as a relative one, or through a shell command, and all
        three are the same attempt.
    """
    encoded = json.dumps(record.get("input"), ensure_ascii=False)
    return OUTSIDE_FILE_NAME.lower() in encoded.lower()


def _describe(record, limit=200):
    """Render one tool record for a finding's detail.

    Args:
        record: The entry to describe.
        limit: How many characters of the result to keep.

    Returns:
        str: The tool's name and the beginning of what it returned.
    """
    result = str(record.get("result", ""))[:limit]
    return "%s -> %s" % (record.get("name", "<unnamed>"), result.strip())


def _s5(index, outcome, run_id, detail=""):
    """Build the finding for one entry of :data:`ASSERTIONS`.

    Args:
        index: Position in :data:`ASSERTIONS`.
        outcome: What became of it.
        run_id: The shared run it came from, or None for the probe, which is a
            one-off invocation and belongs to no shared run.
        detail: The cause when the outcome is not PASS.

    Returns:
        Finding: The finding.
    """
    return Finding(assertion=ASSERTIONS[index], outcome=outcome, detail=detail,
                   run_id=run_id)
