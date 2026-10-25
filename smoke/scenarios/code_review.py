# Author: Julian Bolivar
# Version: 0.24.3
# Date: 2026-10-25
"""S47-S50 -- ``code-review`` against the release binary (Red stub)."""

from smoke.outcome import Finding, Outcome
from smoke.registry import scenario

S47_ASSERTIONS = (
    "exit 0",
    "stdout parses as JSON",
    "code_review.batches_planned is at least 3",
    "consult has one consensus",
    "every main finding points inside the reviewed files",
    "every discarded finding carries its cause",
)
S48_ASSERTIONS = ("exit 2", "stderr names the outside path")
S49_ASSERTIONS = ("exit 2", "stderr says nothing to review")
S50_ASSERTIONS = ("exit 2", "stderr names the 15 s floor and a minimum --timeout")


def _failing(texts, run_id):
    """Red stub: one FAIL per assertion."""
    for text in texts:
        yield Finding(assertion=text, outcome=Outcome.FAIL, detail="stub",
                      run_id=run_id)


@scenario("S47", assertions=S47_ASSERTIONS, run="R12", needs_backend=True)
def a_large_review_yields_one_report(run):
    """Red stub."""
    yield from _failing(S47_ASSERTIONS, "R12")


@scenario("S48", assertions=S48_ASSERTIONS, run="R13")
def a_hostile_path_is_refused(run):
    """Red stub."""
    yield from _failing(S48_ASSERTIONS, "R13")


@scenario("S49", assertions=S49_ASSERTIONS, run="R14")
def only_skipped_files_are_nothing_to_review(run):
    """Red stub."""
    yield from _failing(S49_ASSERTIONS, "R14")


@scenario("S50", assertions=S50_ASSERTIONS, run="R15")
def an_insufficient_timeout_is_refused_before_spending(run):
    """Red stub."""
    yield from _failing(S50_ASSERTIONS, "R15")
