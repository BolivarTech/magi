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
absent from `stderr` **and** present in the day's log, on the SAME run: only
the second half tells a compliant build apart from a broken one that merely
went quiet everywhere.

**The marker.** `attach_persistent_memory` (`src/main.rs`) builds an `INFO`
notice reading `"memory: {N} active, {N} archived, {N} pending re-embed (~{N}
KB index)"` on every headless run whose memory subsystem attaches -- the exact
"how many memories" diagnostic the milestone brief names. `"pending re-embed"`
is the fingerprint: unique to that one notice, present nowhere else a run's
output could plausibly carry it.

**R1, reused.** The cheapest invocation the harness makes, already paid for by
S23, so this scenario adds no backend cost of its own.

**The one thing this cannot promise: that the marker was produced at all.**
Attaching the memory subsystem depends on the embedder answering, which is a
backend concern outside this run's control. A build that never produces the
notice would let both halves pass VACUOUSLY -- the same shape the S18/S19
"talked out of their own subject" trap already named in this project's
`CLAUDE.local.md` -- so when the marker is on neither surface AND the log
directory itself is there to search, both assertions defer to
`CANNOT_TEST` rather than reporting a green that checked nothing. A log
directory that does not exist at all is judged differently: that is not an
absent notice, it is nothing having been written to disk this run, and it
fails.
"""

from smoke import runs
from smoke.outcome import Finding, Outcome
from smoke.registry import scenario

#: The run this reads. R1 is the cheapest invocation the harness makes and it
#: already executes for S23 (and others), so S24 adds no backend cost.
RUN_ID = "R1"

#: Where the product writes its log, relative to the workspace. Same parts
#: ``logging.py`` uses.
LOG_DIR_PARTS = (".magi", "logs")

#: The fingerprint of the memory-count `INFO` notice `attach_persistent_memory`
#: builds in `src/main.rs`: `"memory: {N} active, {N} archived, {N} pending
#: re-embed (~{N} KB index)"`. This fragment names nothing else in the tree
#: (checked by grep against `src/`): the one other occurrence is a test
#: fixture's `"pending re-embed memory"` label, which does not share this
#: substring's trailing space and open paren.
DIAGNOSTIC_MARKER = "pending re-embed ("

#: The verbatim assertion texts, in the order they are yielded.
ASSERTIONS = (
    "the memory-count diagnostic notice never reaches stderr, the headless "
    "run's screen",
    "that same notice is written into the day's log file under the "
    "workspace's log directory",
)


@scenario("S24", assertions=ASSERTIONS, run=RUN_ID, needs_backend=True)
def a_clean_startup_shows_nothing_on_screen(run):
    """Correlate stderr and the day's log for the memory-count notice.

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

    on_screen = DIAGNOSTIC_MARKER in run.output.stderr.decode(
        "utf-8", errors="replace")
    in_log, dir_existed, unreadable = _search_log(DIAGNOSTIC_MARKER)

    if not on_screen and not in_log and dir_existed and not unreadable:
        directory = runs.workspace_root().joinpath(*LOG_DIR_PARTS)
        detail = ("the memory diagnostics notice appears on neither stderr "
                   "nor in any file under %s; this run's memory subsystem "
                   "never attached, so there was nothing to check on either "
                   "surface" % directory)
        for text in ASSERTIONS:
            yield Finding(text, Outcome.CANNOT_TEST, detail, RUN_ID)
        return

    yield _screen_finding(on_screen)
    yield _log_finding(in_log, dir_existed, unreadable)


def _screen_finding(on_screen):
    """Judge assertion 1: the notice must not be on stderr.

    Args:
        on_screen: Whether the marker was found on the run's stderr.

    Returns:
        Finding: FAIL when the notice leaked to the screen, PASS otherwise.
    """
    if on_screen:
        return Finding(ASSERTIONS[0], Outcome.FAIL,
                       "the memory diagnostics notice appears on stderr; "
                       "REQ-L19 sends an INFO notice to the day's file alone",
                       RUN_ID)
    return Finding(ASSERTIONS[0], Outcome.PASS, "", RUN_ID)


def _log_finding(in_log, dir_existed, unreadable):
    """Judge assertion 2 from what :func:`_search_log` found.

    Args:
        in_log: Whether the marker was found in some file under the log
            directory.
        dir_existed: Whether the log directory existed at all.
        unreadable: Files that could not be opened while searching.

    Returns:
        Finding: PASS when found; FAIL when the directory is missing or was
        searched clean; CANNOT_TEST when a file blocked the search.
    """
    if in_log:
        return Finding(ASSERTIONS[1], Outcome.PASS, "", RUN_ID)
    directory = runs.workspace_root().joinpath(*LOG_DIR_PARTS)
    if not dir_existed:
        return Finding(ASSERTIONS[1], Outcome.FAIL,
                       "%s does not exist, so nothing was written to disk "
                       "this run" % directory, RUN_ID)
    if unreadable:
        return Finding(ASSERTIONS[1], Outcome.CANNOT_TEST,
                       "the marker was not found, but these files could not "
                       "be read: %s" % "; ".join(unreadable), RUN_ID)
    return Finding(ASSERTIONS[1], Outcome.FAIL,
                   "the memory diagnostics notice appears in no file under "
                   "%s; either nothing was written or it was not flushed "
                   "before exit" % directory, RUN_ID)


def _search_log(marker):
    """Search the workspace's log directory for *marker*.

    Args:
        marker: The text to search for, as it would appear decoded.

    Returns:
        tuple[bool, bool, list[str]]: ``(found, directory existed,
        unreadable file names)``. The directory flag is kept separate from
        "found" because a missing directory (nothing written at all) and an
        existing one searched clean (the notice was never produced) are
        different findings.
    """
    directory = runs.workspace_root().joinpath(*LOG_DIR_PARTS)
    if not directory.is_dir():
        return False, False, []
    needle = marker.encode("utf-8")
    unreadable = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        try:
            if needle in path.read_bytes():
                return True, True, []
        except OSError as exc:
            unreadable.append("%s (%s)" % (path.name, exc))
    return False, True, unreadable
