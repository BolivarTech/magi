# Author: Julian Bolivar
# Version: 0.18.1
# Date: 2026-09-02
"""S23 -- the log file a real run leaves behind.

Anchored to **REQ-L63**: one run id, published on every surface a reader might
reach for, and the same one in each.

**Why the suite cannot answer this.** Every unit test of the logging layer
builds its own ``LoggingConfig`` over a temporary directory and installs the
subscriber by hand. None of them exercises ``main.rs``'s wiring, none runs in
the release profile, and none observes what survives process exit -- the writer
lives on its own thread, so whether anything reaches the disk depends on the
handle being dropped on the way out rather than on the code that formats the
line. A log that is written correctly and never flushed is, to a user, no log.

**The persistent environment is what shapes assertion 2.** ``smoke/env/``
accumulates across runs, so ``.magi/logs`` already holds files from every
earlier invocation. "A log file exists" would therefore have passed before this
milestone was written -- the fixture would not have created the condition it
claims to measure, which is this repository's most frequent defect in its own
guardians. So the assertion is not that a file exists but that **this run's id
is inside one**, which no leftover can satisfy.

Assertion 1 is what makes 2 meaningful: it establishes that the id the log is
searched for is the id the run actually published, on both of the surfaces a CI
job might read.
"""

from smoke import logs
from smoke.outcome import Finding, Outcome
from smoke.registry import scenario

#: The run this reads. R1 is the cheapest invocation the harness makes and it
#: already executes for two other scenarios, so S23 adds no backend cost.
RUN_ID = "R1"

# There is NO assertion here that the retired JSONL run log is gone, and the
# reason is the same persistent environment that shapes assertion 2 -- read in
# the other direction. ``smoke/env/.magi/logs`` still holds sixteen ``.jsonl``
# files written before the format changed, so "the directory holds no JSONL"
# goes red over leftovers rather than over a writer that is still wired. The
# scenario reads its run AFTER it has finished, so it cannot snapshot the
# directory beforehand and cannot tell the two apart. An assertion that fails
# for the wrong reason is worse than an absent one, and padding the count to
# keep a round number is what the S21 redesign refused to do.

#: The verbatim assertion texts, in the order they are yielded.
ASSERTIONS = (
    "the run id printed on stderr is the one carried in the JSON envelope",
    "that same run id appears inside a file in the workspace log directory, "
    "so the log survived process exit and belongs to this run",
)


@scenario("S23", assertions=ASSERTIONS, run=RUN_ID, needs_backend=True)
def the_run_leaves_a_correlated_log(run):
    """Correlate stderr, the envelope and the log file for one run.

    Complexity: ``O(bytes in the log directory)``.

    Args:
        run: The completed R1, or ``None`` when it did not execute.

    Yields:
        Finding: One per entry of :data:`ASSERTIONS`, in that order.
    """
    if run is None:
        for text in ASSERTIONS:
            yield Finding(text, Outcome.CANNOT_TEST,
                          "R1 did not execute, so no run published an id",
                          RUN_ID)
        return

    stderr_id = logs.id_on_stderr(run.output.stderr)
    envelope_id = logs.id_in_envelope(run.output)

    yield _correlation_finding(stderr_id, envelope_id)

    published = logs.resolve_id(run.output)
    if published is None:
        yield Finding(ASSERTIONS[1], Outcome.CANNOT_TEST,
                      "neither surface published a run id, so there is nothing "
                      "to search the log for", RUN_ID)
    else:
        yield _log_finding(published)


def _correlation_finding(stderr_id, envelope_id):
    """Judge assertion 1.

    Args:
        stderr_id: What stderr published, or None.
        envelope_id: What the envelope published, or None.

    Returns:
        Finding: PASS only when both are present and equal.
    """
    if stderr_id is None and envelope_id is None:
        return Finding(ASSERTIONS[0], Outcome.FAIL,
                       "neither stderr nor the JSON envelope published a run "
                       "id; REQ-L63 requires both", RUN_ID)
    if stderr_id is None:
        return Finding(ASSERTIONS[0], Outcome.FAIL,
                       "the envelope published %r but stderr carried no "
                       "run-id line" % envelope_id, RUN_ID)
    if envelope_id is None:
        return Finding(ASSERTIONS[0], Outcome.FAIL,
                       "stderr published %r but the envelope carried no "
                       "run_id" % stderr_id, RUN_ID)
    if stderr_id != envelope_id:
        return Finding(ASSERTIONS[0], Outcome.FAIL,
                       "stderr says %r and the envelope says %r, so the two "
                       "surfaces name different runs"
                       % (stderr_id, envelope_id), RUN_ID)
    return Finding(ASSERTIONS[0], Outcome.PASS, "", RUN_ID)


def _log_finding(published):
    """Judge assertion 2 by searching the log directory for *published*.

    Args:
        published: The run id both surfaces agreed on, or the one that was
            available.

    Returns:
        Finding: PASS when some file in the directory contains it.
    """
    found, dir_existed, unreadable = logs.contains(published)
    if found:
        return Finding(ASSERTIONS[1], Outcome.PASS, "", RUN_ID)
    directory = logs.log_directory()
    if not dir_existed:
        return Finding(ASSERTIONS[1], Outcome.FAIL,
                       "%s does not exist, so the run wrote no log at all"
                       % directory, RUN_ID)
    if unreadable:
        # Unreadable is not absent. Naming the file keeps a permissions
        # problem from being reported as a missing log.
        return Finding(ASSERTIONS[1], Outcome.CANNOT_TEST,
                       "run id %s was not found, but these files could not be "
                       "read: %s" % (published, "; ".join(unreadable)), RUN_ID)
    return Finding(ASSERTIONS[1], Outcome.FAIL,
                   "run id %s appears in no file under %s; either nothing was "
                   "written or it was not flushed before exit"
                   % (published, directory), RUN_ID)

