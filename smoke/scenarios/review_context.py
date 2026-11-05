# Author: Julian Bolivar
# Version: 0.24.3
# Date: 2026-11-05
"""S52 -- ``code-review`` declares the documentation and history it could not retrieve.

* **S52** (R17, the real trio): the review workspace's ``[embedding]`` points at
  a closed port, so documentation and history cannot be retrieved. The review
  must still run to its report, exit 0, and say so: ``code_review.without_embedder``
  is ``true``.
"""

from smoke.outcome import Finding, Outcome
from smoke.registry import scenario

#: The verbatim assertion texts of S52.
S52_ASSERTIONS = ("exit 0", "code_review.without_embedder is true")


@scenario("S52", assertions=S52_ASSERTIONS, run="R17", needs_backend=True)
def a_review_without_an_embedder_runs_and_says_so(run):
    """Check that a review whose embedder is unreachable still reports, and declares it.

    Args:
        run: R17's ``RunResult``, or ``None`` when the run produced no capture.

    Yields:
        Finding: One per entry of :data:`S52_ASSERTIONS`, in that order.
    """
    for text in S52_ASSERTIONS:
        yield Finding(assertion=text, outcome=Outcome.FAIL, detail="not implemented", run_id="R17")
