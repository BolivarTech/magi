# Author: Julian Bolivar
# Version: 0.18.1
# Date: 2026-08-31
"""S24 -- MS2's own objective: a clean run's diagnostics stay off the screen.

Anchored to **SC-L14** and **REQ-L19**: `ERROR` and `WARN` reach the screen,
`INFO` reaches only the day's log file. Nothing in the unit suite can see
whether that split survives past `main.rs`'s wiring -- `src/notices.rs` builds
its own subscriber over a temporary directory, never the release binary's real
startup, and never observes what a headless run actually writes to `stderr`
versus `.magi/logs`.

**Why this is one scenario with two halves, and why the second is the one
that matters.** The milestone's visible promise is "the screen goes quiet",
and a scenario that only checked that would pass just as well against a build
where the logging layer stopped writing anywhere at all -- silence on screen
for the wrong reason. So the marker this scenario searches for has to be
absent from the screen **and** present in the day's log, on the SAME run: only
the second half tells a compliant build apart from a broken one that merely
went quiet everywhere.

**"The screen" is BOTH standard streams, not `stderr` alone** (MS2 gate S4
fourth pass, Caspar). It read `stderr` only, which made the guard specific to
where the notice happens to be written today rather than to the property
REQ-L19 states: an `INFO` diagnostic does not reach the user's terminal. A
regression that moved the whole notice list to `stdout` -- one `eprintln!`
becoming a `println!` -- leaks it to exactly the same terminal, and to the
pipe a caller redirects, and this scenario passed. Both streams are the
screen, so both are searched.

**The marker.** `attach_persistent_memory` (`src/main.rs`) builds an `INFO`
notice reading `"memory: {N} active, {N} archived, {N} pending re-embed (~{N}
KB index)"` on every headless run whose memory subsystem attaches -- the exact
"how many memories" diagnostic the milestone brief names. `"pending re-embed"`
is the fingerprint: unique to that one notice, present nowhere else a run's
output could plausibly carry it.

**R1, reused.** The cheapest invocation the harness makes, already paid for by
S23, so this scenario adds no backend cost of its own.

**The log search is bound to THIS run's id, and that is correctness rather
than tidiness.** ``smoke/env/`` is persistent by design, so ``.magi/logs``
accumulates every run the environment has ever made -- and R1 writes its own
startup line there on every one of them. A search for the bare marker answers
"has this environment EVER written it", which is a question a broken build
passes: on a run where the memory subsystem never attaches, the leftover from
an earlier run keeps ``in_log`` true, the ``CANNOT_TEST`` branch below is
skipped, and both assertions report PASS having checked nothing about this
run. So the matcher requires one line to carry both ``run=<this run's id>``
(REQ-L63) and the marker, as S9 does. S23 is deliberately coarser -- it asks
whether the run id appears in ANY file under the directory -- so it is not the
precedent for the line-bound form, only for searching the directory at all.

**The one thing this cannot promise: that the marker was produced at all.**
Attaching the memory subsystem depends on the embedder answering, which is a
backend concern outside this run's control. A build that never produces the
notice would let both halves pass VACUOUSLY -- the same shape the S18/S19
"talked out of their own subject" trap already named in this project's
`CLAUDE.local.md` -- so when the marker is on neither surface AND the log
directory itself is there to search, both assertions defer to
`CANNOT_TEST` rather than reporting a green that checked nothing.

**A log directory that does not exist is the SAME deferral, not a failure**
(MS2 gate S4 fourth pass, Caspar). It used to fail, on the reading that
"nothing was written to disk this run". That is a verdict this scenario has
no evidence for: with no directory there is nothing to search, so a run that
never produced the notice and a run that produced it and lost it are
indistinguishable from here -- and only the second is a statement about the
product. It is the exact case the paragraph above already calls untestable,
reached by a different route. S9 is the model: where a green would no longer
mean what it says, the honest answer is `CANNOT_TEST` naming what was
missing, not a red aimed at the product for an environment the harness could
not establish.

**Which is also why a missing run id silences one half and not both.** Absent
evidence is what defers an assertion, and the two halves do not rest on the
same evidence: the log search needs the id, because the directory is shared;
the screen assertion reads the run's own capture. So a marker found ON the
screen fails the first assertion whether or not an id was published -- there
is nothing left for a log line to add. Only the vacuity case above still
needs the id, and it keeps deferring.
"""

from smoke import logs
from smoke.outcome import Finding, Outcome
from smoke.registry import scenario

#: The run this reads. R1 is the cheapest invocation the harness makes and it
#: already executes for S23 (and others), so S24 adds no backend cost.
RUN_ID = "R1"

#: The fingerprint of the memory-count `INFO` notice `attach_persistent_memory`
#: builds in `src/main.rs`: `"memory: {N} active, {N} archived, {N} pending
#: re-embed (~{N} KB index)"`. This fragment names nothing else in the tree
#: (checked by grep against `src/`): the one other occurrence is a test
#: fixture's `"pending re-embed memory"` label, which does not share this
#: substring's trailing space and open paren.
DIAGNOSTIC_MARKER = "pending re-embed ("

#: The verbatim assertion texts, in the order they are yielded.
ASSERTIONS = (
    "the memory-count diagnostic notice never reaches stdout or stderr, the "
    "headless run's screen",
    "that same notice is written into the day's log file under the "
    "workspace's log directory",
)


@scenario("S24", assertions=ASSERTIONS, run=RUN_ID, needs_backend=True)
def a_clean_startup_shows_nothing_on_screen(run):
    """Correlate the screen and the day's log for the memory-count notice.

    The log half is bound to this run's own id, so a marker left behind by an
    earlier run against the persistent environment cannot answer for one that
    never wrote it. Without a run id the log half has nothing to bind to and
    defers; the screen half defers only when the marker is absent, because
    that is the case a log line would have had to disambiguate. A marker
    sitting ON the screen is a leak the capture proves by itself.

    Complexity: ``O(bytes in the log directory)``.

    Args:
        run: The completed R1, or ``None`` when it did not execute.

    Yields:
        Finding: One per entry of :data:`ASSERTIONS`, in that order.
    """
    if run is None:
        for text in ASSERTIONS:
            yield Finding(text, Outcome.CANNOT_TEST,
                          "R1 did not execute, so no startup diagnostics were "
                          "produced", RUN_ID)
        return

    on_screen = _on_screen(run.output)

    run_id = logs.resolve_id(run.output)
    if run_id is None:
        # **Only the half that needs the id is silenced.** The log search is
        # the half that does: the directory is shared, so without an id there
        # is nothing to tell this run's line from an earlier one's. The screen
        # half is answered by the run's own capture -- if the marker IS on
        # the screen, that is a leak fully observed, and deferring it would
        # hide a FAIL behind evidence it never needed. What still defers is the
        # VACUITY case: with the marker on neither surface, a compliant build
        # and one whose memory subsystem never attached look identical from
        # the screen alone, and the log is what separates them.
        no_id = ("run %s published no run id on stderr or in its JSON "
                 "envelope, so its own log line cannot be told apart from an "
                 "earlier run's in the shared log directory" % RUN_ID)
        if on_screen:
            yield _screen_finding(on_screen)
        else:
            yield Finding(ASSERTIONS[0], Outcome.CANNOT_TEST,
                          "%s -- and the marker is on neither surface, so a "
                          "quiet screen cannot be told from a run that never "
                          "produced the notice" % no_id, RUN_ID)
        yield Finding(ASSERTIONS[1], Outcome.CANNOT_TEST, no_id, RUN_ID)
        return

    found, dir_existed, unreadable = logs.scan(_marker_matcher(run_id))
    in_log = bool(found)
    directory = logs.log_directory()

    if not dir_existed:
        # **Not a product verdict.** With no directory there is nothing to
        # search, so a run that never produced the notice and one that
        # produced it and failed to persist it read identically from here --
        # and only the second says anything about the product. The screen
        # half keeps its own evidence: a marker sitting ON the screen is a
        # leak the capture proves by itself, and no log line could excuse it.
        missing = ("%s does not exist, so there is nothing to search: a run "
                   "that never produced the notice cannot be told from one "
                   "that produced it and did not persist it" % directory)
        if on_screen:
            yield _screen_finding(on_screen)
        else:
            yield Finding(ASSERTIONS[0], Outcome.CANNOT_TEST, missing, RUN_ID)
        yield Finding(ASSERTIONS[1], Outcome.CANNOT_TEST, missing, RUN_ID)
        return

    if not on_screen and not in_log and not unreadable:
        detail = ("the memory diagnostics notice appears on neither stdout "
                  "nor stderr, nor on any line carrying run=%s under %s; this "
                  "run's memory subsystem never attached, so there was "
                  "nothing to check on either surface" % (run_id, directory))
        for text in ASSERTIONS:
            yield Finding(text, Outcome.CANNOT_TEST, detail, RUN_ID)
        return

    yield _screen_finding(on_screen)
    yield _log_finding(in_log, unreadable, run_id)


def _on_screen(output):
    """Whether the marker reached either standard stream.

    Both are the screen: the user sees one terminal, and a regression that
    turns an ``eprintln!`` into a ``println!`` leaks the notice just as far.
    Searched separately rather than on the concatenation, so a marker split
    across the tail of one stream and the head of the other cannot be
    invented by the join.

    Args:
        output: The completed run's captured output.

    Returns:
        bool: True when :data:`DIAGNOSTIC_MARKER` is on stdout or on stderr.

    Complexity: ``O(bytes captured)``.
    """
    return any(
        DIAGNOSTIC_MARKER in stream.decode("utf-8", errors="replace")
        for stream in (output.stdout, output.stderr)
    )


def _marker_matcher(run_id):
    """Build a :func:`smoke.logs.scan` matcher bound to one run's own lines.

    Splitting on ``b"\\n"`` is safe because the logging layer escapes every
    control character out of a rendered event before it is written (REQ-L64
    stage 3), so one physical line is always exactly one event -- a foreign
    string can never fold two events together or split one apart.

    **The needle ends at the id, not at the separator after it.** It used to
    carry a trailing space, which made the guard depend on how the renderer
    joins fields -- a detail REQ-L63 never promised. Anchoring on the id alone
    cannot widen the match either, because an id is ``<pid>-<16 hex>`` with the
    hex half a fixed width: no id is a prefix of a different one, so a bare
    ``run=<id>`` still binds the line to exactly this run.

    Args:
        run_id: The id to bind the search to.

    Returns:
        Callable[[bytes], bool | None]: A matcher over one file's raw bytes,
        returning True for the first line carrying both this run's id and
        :data:`DIAGNOSTIC_MARKER`, or None when the file has no such line.
    """
    id_needle = ("run=%s" % run_id).encode("utf-8")
    marker = DIAGNOSTIC_MARKER.encode("utf-8")

    def matcher(data):
        for line in data.split(b"\n"):
            if id_needle in line and marker in line:
                return True
        return None

    return matcher


def _screen_finding(on_screen):
    """Judge assertion 1: the notice must not be on either standard stream.

    Args:
        on_screen: Whether the marker was found on the run's stdout or stderr.

    Returns:
        Finding: FAIL when the notice leaked to the screen, PASS otherwise.
    """
    if on_screen:
        return Finding(ASSERTIONS[0], Outcome.FAIL,
                       "the memory diagnostics notice appears on stdout or "
                       "stderr; REQ-L19 sends an INFO notice to the day's "
                       "file alone", RUN_ID)
    return Finding(ASSERTIONS[0], Outcome.PASS, "", RUN_ID)


def _log_finding(in_log, unreadable, run_id):
    """Judge assertion 2 from what :func:`smoke.logs.scan` found.

    Only reached with the log directory present: its absence is judged by the
    caller as ``CANNOT_TEST``, because there is nothing to search and so
    nothing to say about the product.

    Args:
        in_log: Whether a line carrying both this run's id and the marker was
            found under the log directory.
        unreadable: Files that could not be opened while searching.
        run_id: This run's id, named in every failure detail so a reader can
            see which run was searched for.

    Returns:
        Finding: PASS when found; FAIL when the directory was searched clean;
        CANNOT_TEST when a file blocked the search.
    """
    if in_log:
        return Finding(ASSERTIONS[1], Outcome.PASS, "", RUN_ID)
    directory = logs.log_directory()
    if unreadable:
        return Finding(ASSERTIONS[1], Outcome.CANNOT_TEST,
                       "no line carrying run=%s and the marker was found, but "
                       "these files could not be read: %s"
                       % (run_id, "; ".join(unreadable)), RUN_ID)
    return Finding(ASSERTIONS[1], Outcome.FAIL,
                   "no line under %s carries both run=%s and the memory "
                   "diagnostics marker; either nothing was written or it was "
                   "not flushed before exit" % (directory, run_id), RUN_ID)
