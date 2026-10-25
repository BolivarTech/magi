# Author: Julian Bolivar
# Version: 0.24.3
# Date: 2026-10-25
"""S47-S50 -- ``code-review`` against the release binary.

The suite proves each stage of a review with doubles. What it cannot see is the
release binary doing it end to end on an input that really splits into batches,
and refusing the inputs it must refuse before a model is paid for.

* **S47** (R12, the real trio): a generated tree that plans at least three
  batches yields ONE report. Its two property assertions read the JSON the
  product publishes (``code_review.findings`` and ``code_review.discarded``;
  the batch numbers inside them are 1-based, and S47 reads only the count); a missing key is a ``FAIL`` and never an
  empty list that would pass, while an empty list is ``CANNOT_TEST`` because a
  real trio cannot be forced to hallucinate a location or to have something
  discarded.
* **S48** (R13): a path outside the workspace is refused with exit 2.
* **S49** (R14): a directory holding only a binary and a file with a secret has
  nothing to review: exit 2.
* **S50** (R15): ``--timeout 30`` over three batches is refused before any model
  call, naming the floor and the ``--timeout`` that would do.

S48-S50 need no backend. That no model was contacted is not observable from
outside the process; the Rust integration tests pin it with a stand-in that
records every body.
"""

import re

from smoke import runs
from smoke.errors import ProductOutputError
from smoke.outcome import Finding, Outcome
from smoke.registry import scenario

#: The verbatim assertion texts of S47-S50.
S47_ASSERTIONS = (
    "exit 0",
    "stdout parses as JSON",
    "code_review.batches_planned is at least 3, every one completed",
    "consult has one consensus",
    "every main finding points inside the reviewed files",
    "every discarded finding carries its cause",
)
S48_ASSERTIONS = ("exit 2", "stderr names the outside path")
S49_ASSERTIONS = ("exit 2", "stderr says nothing to review")
S50_ASSERTIONS = ("exit 2", "stderr names the 15 s floor and a minimum --timeout")

#: Exit codes of a successful run and of CLI misuse.
EXIT_OK = 0
EXIT_MISUSE = 2

#: Batches S47 requires, and the keys of the product's JSON it reads.
MIN_BATCHES = runs.REVIEW_MIN_BATCHES
CODE_REVIEW_KEY = "code_review"
CONSULT_KEY = "consult"
CONSENSUS_KEY = "consensus"

#: The files S47 generated, as the product names them (relative to the
#: workspace). The scenario compares findings against what it knows on its own
#: account, so the product's summary gains no field for the purpose.
S47_FILES = runs.REVIEW_GEN_PATHS

#: The product's wording that S48 and S49 look for in stderr.
OUTSIDE_WORKSPACE_TEXT = "outside the workspace"
NOTHING_TO_REVIEW_TEXT = "nothing to review"

#: The floor of a per-batch ceiling in seconds
#: (``AGENT_TIMEOUT_ABSOLUTE_FLOOR_SECS``), the phrase S50 looks for, and the
#: shape of the minimum it must name.
FLOOR_SECS = 15
FLOOR_TEXT = "%d s floor" % FLOOR_SECS
MINIMUM_TIMEOUT_PATTERN = re.compile(r"--timeout\s+\d+")


def _finding(texts, index, outcome, run_id, detail=""):
    """Build the finding for assertion *index* of *texts*.

    Args:
        texts: The scenario's assertion texts.
        index: Position in *texts*.
        outcome: What became of it.
        run_id: The shared run it came from.
        detail: The cause when the outcome is not ``PASS``.

    Returns:
        Finding: The finding.
    """
    return Finding(assertion=texts[index], outcome=outcome, detail=detail,
                   run_id=run_id)


def _verdict(texts, index, run_id, held, detail):
    """Turn a boolean check into a PASS or FAIL finding.

    Args:
        texts: The scenario's assertion texts.
        index: Position in *texts*.
        run_id: The shared run it came from.
        held: Whether the property held.
        detail: The cause reported when it did not.

    Returns:
        Finding: ``PASS`` when *held*, otherwise ``FAIL`` carrying *detail*.
    """
    return _finding(texts, index, Outcome.PASS if held else Outcome.FAIL,
                    run_id, "" if held else detail)


def _all_cannot_test(texts, run_id, detail):
    """One ``CANNOT_TEST`` finding per assertion of a scenario.

    Args:
        texts: The scenario's assertion texts.
        run_id: The shared run that produced nothing.
        detail: The cause.

    Yields:
        Finding: One per entry of *texts*, in order.
    """
    for index in range(len(texts)):
        yield _finding(texts, index, Outcome.CANNOT_TEST, run_id, detail)


def _stderr_text(run):
    """Decode a run's stderr leniently.

    Args:
        run: The run's ``RunResult``.

    Returns:
        str: The decoded text.
    """
    return run.output.stderr.decode("utf-8", errors="replace")


def _refusal(texts, run_id, run, stderr_ok, stderr_detail):
    """The two assertions shared by the refusal scenarios S48-S50.

    Args:
        texts: The scenario's two assertion texts.
        run_id: The shared run.
        run: Its ``RunResult``, or ``None``.
        stderr_ok: Predicate over the decoded stderr.
        stderr_detail: The cause reported when it does not hold.

    Yields:
        Finding: ``exit 2`` first, then the stderr assertion.
    """
    if run is None:
        yield from _all_cannot_test(
            texts, run_id, "run %s produced no capture to inspect" % run_id)
        return
    code = run.output.exit_code
    yield _verdict(texts, 0, run_id, code == EXIT_MISUSE,
                   "exited %d, expected %d" % (code, EXIT_MISUSE))
    yield _verdict(texts, 1, run_id, stderr_ok(_stderr_text(run)), stderr_detail)


def _list_under(section, key):
    """Read a list the product publishes under ``code_review``.

    Args:
        section: The ``code_review`` object.
        key: The list's key.

    Returns:
        tuple: ``(items, failure)``: the list, or ``None`` and why it is
        unusable. A missing key is a failure, never an empty list.
    """
    items = section.get(key)
    if not isinstance(items, list):
        return None, "code_review.%s is missing or not a list" % key
    return items, ""


def _batches_finding(section, run_id):
    """Judge ``code_review.batches_planned`` and ``code_review.batches_completed``.

    Args:
        section: The ``code_review`` object.
        run_id: The shared run.

    Returns:
        Finding: ``FAIL`` when ``batches_completed`` is not an integer equal to
        ``batches_planned`` (one report over batches that never ran is not the
        multi-batch report RV-S2 asks for). Otherwise ``PASS`` at three batches
        or more; below that, ``CANNOT_TEST`` when the whole pool measured a
        window (the input is sized for the unmeasured default, so fewer batches
        is the environment's doing), else ``FAIL``. An absent
        ``unmeasured_models`` counts as unmeasured.
    """
    planned = section.get("batches_planned")
    completed = section.get("batches_completed")
    if not isinstance(completed, int) or completed != planned:
        return _finding(
            S47_ASSERTIONS, 2, Outcome.FAIL, run_id,
            "batches_completed is %r, expected every one of the %r planned"
            % (completed, planned))
    if isinstance(planned, int) and planned >= MIN_BATCHES:
        return _finding(S47_ASSERTIONS, 2, Outcome.PASS, run_id)
    detail = "batches_planned is %r, expected at least %d" % (planned, MIN_BATCHES)
    unmeasured = section.get("unmeasured_models")
    if isinstance(unmeasured, list) and not unmeasured:
        return _finding(
            S47_ASSERTIONS, 2, Outcome.CANNOT_TEST, run_id,
            detail + "; every model measured a window, larger than the one "
            "the input is sized for")
    return _finding(S47_ASSERTIONS, 2, Outcome.FAIL, run_id, detail)


def _locations_finding(section, run_id):
    """Judge that every located main finding points inside the generated files.

    Args:
        section: The ``code_review`` object.
        run_id: The shared run.

    Returns:
        Finding: ``FAIL`` on a missing list or a file outside
        :data:`S47_FILES`; ``CANNOT_TEST`` when no finding carries a file;
        otherwise ``PASS``. A finding without a file is unlocated, not invented.
    """
    findings, failure = _list_under(section, "findings")
    if findings is None:
        return _finding(S47_ASSERTIONS, 4, Outcome.FAIL, run_id, failure)
    located = [f for f in findings
               if isinstance(f, dict) and f.get("file") is not None]
    if not located:
        return _finding(S47_ASSERTIONS, 4, Outcome.CANNOT_TEST, run_id,
                        "the trio reported no located finding, so there is "
                        "no location to check")
    outside = [f.get("file") for f in located if f.get("file") not in S47_FILES]
    return _verdict(S47_ASSERTIONS, 4, run_id, not outside,
                    "findings point outside the reviewed files: %s" % outside)


def _causes_finding(section, run_id):
    """Judge that every discarded finding names why it was discarded.

    Args:
        section: The ``code_review`` object.
        run_id: The shared run.

    Returns:
        Finding: ``FAIL`` on a missing list or an empty cause;
        ``CANNOT_TEST`` when nothing was discarded; otherwise ``PASS``.
    """
    discarded, failure = _list_under(section, "discarded")
    if discarded is None:
        return _finding(S47_ASSERTIONS, 5, Outcome.FAIL, run_id, failure)
    if not discarded:
        return _finding(S47_ASSERTIONS, 5, Outcome.CANNOT_TEST, run_id,
                        "nothing was discarded, so there is no cause to check")
    bare = [d for d in discarded
            if not (isinstance(d, dict) and isinstance(d.get("cause"), str)
                    and d["cause"].strip())]
    return _verdict(S47_ASSERTIONS, 5, run_id, not bare,
                    "%d discarded finding(s) carry no cause" % len(bare))


@scenario("S47", assertions=S47_ASSERTIONS, run="R12", needs_backend=True)
def a_large_review_yields_one_report(run):
    """Check that a review over three or more batches yields one report.

    Args:
        run: R12's ``RunResult``, or ``None`` when the run produced no capture.

    Yields:
        Finding: One per entry of :data:`S47_ASSERTIONS`, in that order.
    """
    if run is None:
        yield from _all_cannot_test(S47_ASSERTIONS, "R12",
                                    "run R12 produced no capture to inspect")
        return
    code = run.output.exit_code
    yield _verdict(S47_ASSERTIONS, 0, "R12", code == EXIT_OK, "exited %d" % code)
    try:
        document = run.output.json()
    except ProductOutputError as exc:
        for index in range(1, len(S47_ASSERTIONS)):
            yield _finding(S47_ASSERTIONS, index, Outcome.FAIL, "R12",
                           "stdout is not a JSON document: %s" % exc)
        return
    yield _finding(S47_ASSERTIONS, 1, Outcome.PASS, "R12")
    section = document.get(CODE_REVIEW_KEY)
    if not isinstance(section, dict):
        section = {}
    yield _batches_finding(section, "R12")
    consult = document.get(CONSULT_KEY)
    has_consensus = isinstance(consult, dict) and bool(consult.get(CONSENSUS_KEY))
    yield _verdict(S47_ASSERTIONS, 3, "R12", has_consensus,
                   "the document carries no consult.consensus")
    yield _locations_finding(section, "R12")
    yield _causes_finding(section, "R12")


@scenario("S48", assertions=S48_ASSERTIONS, run="R13")
def a_hostile_path_is_refused(run):
    """Check that a path outside the workspace is refused with exit 2.

    Both the product's fixed phrase and the hostile path's last component must
    appear: the component is the separator-independent part of the path the
    product prints, and it occurs in no fixed text.

    Args:
        run: R13's ``RunResult``, or ``None`` when the run produced no capture.

    Yields:
        Finding: One per entry of :data:`S48_ASSERTIONS`, in that order.
    """
    yield from _refusal(S48_ASSERTIONS, "R13", run,
                        lambda text: OUTSIDE_WORKSPACE_TEXT in text
                        and runs.REVIEW_OUTSIDE_COMPONENT in text,
                        "stderr does not name %s as outside the workspace"
                        % runs.REVIEW_OUTSIDE_COMPONENT)


@scenario("S49", assertions=S49_ASSERTIONS, run="R14")
def only_skipped_files_are_nothing_to_review(run):
    """Check that a directory of only skipped files is nothing to review.

    Args:
        run: R14's ``RunResult``, or ``None`` when the run produced no capture.

    Yields:
        Finding: One per entry of :data:`S49_ASSERTIONS`, in that order.
    """
    yield from _refusal(S49_ASSERTIONS, "R14", run,
                        lambda text: NOTHING_TO_REVIEW_TEXT in text,
                        "stderr does not say there is nothing to review")


@scenario("S50", assertions=S50_ASSERTIONS, run="R15")
def an_insufficient_timeout_is_refused_before_spending(run):
    """Check that a too-short ``--timeout`` is refused naming floor and minimum.

    The numbers in the message come from the product's formula and are not
    compared; only the floor text and the presence of a minimum are.

    Args:
        run: R15's ``RunResult``, or ``None`` when the run produced no capture.

    Yields:
        Finding: One per entry of :data:`S50_ASSERTIONS`, in that order.
    """
    yield from _refusal(
        S50_ASSERTIONS, "R15", run,
        lambda text: FLOOR_TEXT in text
        and bool(MINIMUM_TIMEOUT_PATTERN.search(text)),
        "stderr does not name the %s and a minimum --timeout" % FLOOR_TEXT)
