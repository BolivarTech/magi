# Author: Julian Bolivar
# Version: 0.24.3
# Date: 2026-11-05
"""S52 -- ``code-review`` declares the documentation and history it could not retrieve.

* **S52** (R17, the real trio): the review workspace's ``[embedding]`` points at
  a closed port (``runs.UNREACHABLE_EMBEDDER_URL``), so documentation and history
  cannot be retrieved. The review must still run to its report, exit 0, and say
  so: ``code_review.without_embedder`` is ``true``.

The unit suite proves each half with doubles (``src/memory/review_rag.rs``:
``a_failing_embedder_degrades_to_no_documentation_or_history``,
``a_failing_reindex_degrades_and_never_fails_the_review``); what it cannot see is
the release binary meeting a refused connection on its way to the embedder and
still delivering a report. A timed-out R17 never reaches this body: S52 does not
declare ``inspects_timeouts``, so the runner reports it as ``CANNOT_TEST``.
"""

from smoke.errors import ProductOutputError
from smoke.outcome import Finding, Outcome
from smoke.registry import scenario

#: The verbatim assertion texts of S52.
S52_ASSERTIONS = ("exit 0", "code_review.without_embedder is true")

#: Exit code of a review that produced its report.
EXIT_OK = 0

#: The keys of the product's JSON S52 reads.
CODE_REVIEW_KEY = "code_review"
WITHOUT_EMBEDDER_KEY = "without_embedder"

#: The run S52 reads.
RUN_ID = "R17"


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
    try:
        report = run.output.json()
    except ProductOutputError as exc:
        yield _finding(1, Outcome.FAIL, "stdout is not a JSON document: %s" % exc)
        return
    summary = report.get(CODE_REVIEW_KEY)
    value = summary.get(WITHOUT_EMBEDDER_KEY) if isinstance(summary, dict) else None
    yield _verdict(1, value is True,
                   "code_review.without_embedder is %r, not true" % (value,))
