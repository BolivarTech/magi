# Author: Julian Bolivar
# Version: 0.25.0
# Date: 2026-11-08
"""S52, S55 -- ``code-review`` retrieves its documentation and history, or says it could not.

* **S52** (R17, the real trio): the review workspace's ``[embedding]`` points at
  an endpoint that answers an HTTP error at once
  (``runs.embedder_pointed_away``), so documentation and history cannot be
  retrieved. The review must still run to its report, exit 0, and say
  so: ``code_review.without_embedder`` is ``true`` and ``code_review.trimmed``
  names the cause (an unavailable index, an embedder that did not answer),
  RV07's declaration of what was left out.

The unit suite proves each half with doubles (``src/memory/review_rag.rs``:
``a_failing_embedder_degrades_to_no_documentation_or_history``,
``a_failing_reindex_degrades_and_never_fails_the_review``); what it cannot see is
the release binary meeting a failing embedder endpoint and
still delivering a report. A timed-out R17 never reaches this body: S52 does not
declare ``inspects_timeouts``, so the runner reports it as ``CANNOT_TEST``.

* **S55** (R18, the real trio and the environment's own embedder, the one S9
  needs answering): the same machinery with nothing pointed away must not
  degrade. ``code_review.without_embedder`` is ``false`` and ``code_review.trimmed``
  names neither an unavailable documentation index nor an embedder that did not
  answer. Without it the suite could not tell a working RAG from one that
  degrades on every run: S52 even expects the degradation. Whether a chunk was
  relevant is the model side and is never asserted; a trim for room is the
  budget at work and passes. R18 does call the embedder: the file it reviews
  carries a ``///`` doc comment, a documentation source of its own, so the
  documentation index is built and each batch's query embedded. A history the review would retrieve needs a chain
  of revision reviews, which no smoke run makes, so it stays in the unit suite.
"""

from smoke.errors import ProductOutputError
from smoke.outcome import Finding, Outcome
from smoke.registry import scenario

#: The verbatim assertion texts of S52.
S52_ASSERTIONS = (
    "exit 0",
    "code_review.without_embedder is true",
    "code_review.trimmed names an unavailable index or an embedder that did not answer",
)

#: Exit code of a review that produced its report.
EXIT_OK = 0

#: The keys of the product's JSON S52 reads.
CODE_REVIEW_KEY = "code_review"
WITHOUT_EMBEDDER_KEY = "without_embedder"

#: The run S52 reads.
RUN_ID = "R17"

#: The key of the context the review left out, and of each entry's cause.
TRIMMED_KEY = "trimmed"
TRIMMED_LABEL_KEY = "label"

#: The ``trimmed`` labels of a degraded RAG (``INDEX_UNAVAILABLE`` and
#: ``EMBEDDER_DOWN`` in ``src/memory/review_rag.rs``); a trim for room is not one.
DEGRADED_LABELS = ("index unavailable", "the embedder did not answer")


def _finding(index, outcome, detail=""):
    """Build the finding for assertion *index* of :data:`S52_ASSERTIONS`.

    Args:
        index: Position in :data:`S52_ASSERTIONS`.
        outcome: What became of it.
        detail: The cause when the outcome is not ``PASS``.

    Returns:
        Finding: The finding.
    """
    return Finding(assertion=S52_ASSERTIONS[index], outcome=outcome, detail=detail,
                   run_id=RUN_ID)


def _verdict(index, held, detail):
    """Turn a boolean check into a ``PASS`` or ``FAIL`` finding.

    Args:
        index: Position in :data:`S52_ASSERTIONS`.
        held: Whether the property held.
        detail: The cause reported when it did not.

    Returns:
        Finding: ``PASS`` when *held*, otherwise ``FAIL`` carrying *detail*.

    Example:
        >>> _verdict(0, True, "x").outcome.value
        'PASS'
    """
    return _finding(index, Outcome.PASS if held else Outcome.FAIL, "" if held else detail)


@scenario("S52", assertions=S52_ASSERTIONS, run=RUN_ID, needs_backend=True)
def a_review_without_an_embedder_runs_and_says_so(run):
    """Check that a review whose embedder is unreachable still reports, and declares it.

    The declaration must be the boolean ``true``: a missing key, or any other
    value, is shape drift and a ``FAIL``, never a pass by absence.

    Args:
        run: R17's ``RunResult``, or ``None`` when the run produced no capture.

    Yields:
        Finding: One per entry of :data:`S52_ASSERTIONS`, in that order.
    """
    if run is None:
        for index in range(len(S52_ASSERTIONS)):
            yield _finding(index, Outcome.CANNOT_TEST,
                           "run %s produced no capture to inspect" % RUN_ID)
        return
    code = run.output.exit_code
    yield _verdict(0, code == EXIT_OK, "exited %d" % code)
    # A root that is not an object (`null`, a list) never reaches `.get`: `ProductOutput.json`
    # raises ProductOutputError for it, a FAIL below (`test_s52_fails_on_valid_json_that_is_not_an_object`,
    # `test_s55_fails_on_valid_json_that_is_not_an_object`).
    try:
        report = run.output.json()
    except ProductOutputError as exc:
        for index in range(1, len(S52_ASSERTIONS)):
            yield _finding(index, Outcome.FAIL, "stdout is not a JSON document: %s" % exc)
        return
    summary = report.get(CODE_REVIEW_KEY)
    summary = summary if isinstance(summary, dict) else {}
    value = summary.get(WITHOUT_EMBEDDER_KEY)
    yield _verdict(1, value is True,
                   "code_review.without_embedder is %r, not true" % (value,))
    # MAGI FIX (S6 MS3 loop3): the boolean alone was read; RV07 declares the cause in `trimmed`.
    degraded = _degraded_labels(summary.get(TRIMMED_KEY))
    if degraded is None:
        yield _finding(2, Outcome.FAIL, "code_review.trimmed is missing or not a list of entries")
    else:
        yield _verdict(2, bool(degraded), "code_review.trimmed names no degraded cause")


#: The verbatim assertion texts of S55.
S55_ASSERTIONS = (
    "code_review.without_embedder is false: the embedder answered the review",
    "code_review.trimmed names no unavailable index and no embedder that did not answer",
)

#: The run S55 reads: a review with the environment's own embedder.
S55_RUN_ID = "R18"


def _s55_finding(index, outcome, detail=""):
    """Build the finding for assertion *index* of :data:`S55_ASSERTIONS`.

    Args:
        index: Position in :data:`S55_ASSERTIONS`.
        outcome: What became of it.
        detail: The cause when the outcome is not ``PASS``.

    Returns:
        Finding: The finding.
    """
    return Finding(assertion=S55_ASSERTIONS[index], outcome=outcome, detail=detail,
                   run_id=S55_RUN_ID)


def _degraded_labels(trimmed):
    """The degraded causes *trimmed* names, or ``None`` when it is not a list of entries.

    Args:
        trimmed: The ``code_review.trimmed`` value read from the product.

    Returns:
        list: Each :data:`DEGRADED_LABELS` label found, in order; ``None`` for a
        shape the harness does not understand, an entry without a string
        ``label`` included (a missing label is shape drift, never a pass).

    Example:
        >>> _degraded_labels([{"kind": "history", "label": "no room left in the batch"}])
        []
        >>> _degraded_labels("x") is None
        True
        >>> _degraded_labels([{"kind": "history"}]) is None
        True
    """
    # MAGI FIX (S6 MS3 loop3): an entry without a string label was skipped, not refused.
    if not isinstance(trimmed, list) or not all(
            isinstance(t, dict) and isinstance(t.get(TRIMMED_LABEL_KEY), str) for t in trimmed):
        return None
    return [t.get(TRIMMED_LABEL_KEY) for t in trimmed
            if t.get(TRIMMED_LABEL_KEY) in DEGRADED_LABELS]


@scenario("S55", assertions=S55_ASSERTIONS, run=S55_RUN_ID, needs_backend=True)
def a_review_with_its_embedder_up_is_not_degraded(run):
    """Check that a review whose embedder answers retrieves without degrading.

    Both declarations must be present and well typed: a missing key is shape
    drift and a ``FAIL``, never a pass by absence.

    Args:
        run: R18's ``RunResult``, or ``None`` when the run produced no capture.

    Yields:
        Finding: One per entry of :data:`S55_ASSERTIONS`, in that order.
    """
    # MAGI FIX (S6 MS3 loop2): no scenario observed a working RAG; S52 only its degradation.
    if run is None:
        for index in range(len(S55_ASSERTIONS)):
            yield _s55_finding(index, Outcome.CANNOT_TEST,
                               "run %s produced no capture to inspect" % S55_RUN_ID)
        return
    try:
        # `.json()` raises for a root that is not an object, so `.get` is always on a dict.
        summary = run.output.json().get(CODE_REVIEW_KEY)
    except ProductOutputError as exc:
        for index in range(len(S55_ASSERTIONS)):
            yield _s55_finding(index, Outcome.FAIL, "stdout is not a JSON document: %s" % exc)
        return
    summary = summary if isinstance(summary, dict) else {}
    value = summary.get(WITHOUT_EMBEDDER_KEY)
    held = value is False
    yield _s55_finding(0, Outcome.PASS if held else Outcome.FAIL,
                       "" if held else "code_review.without_embedder is %r, not false" % (value,))
    degraded = _degraded_labels(summary.get(TRIMMED_KEY))
    if degraded is None:
        yield _s55_finding(1, Outcome.FAIL, "code_review.trimmed is missing or not a list of entries")
    elif degraded:
        yield _s55_finding(1, Outcome.FAIL, "code_review.trimmed declares %s" % ", ".join(degraded))
    else:
        yield _s55_finding(1, Outcome.PASS)