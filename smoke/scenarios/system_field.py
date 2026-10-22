# Author: Julian Bolivar
# Version: 0.24.3
# Date: 2026-10-22
"""S44-S46 -- the envelope ``system`` field (Red stub)."""

from smoke.outcome import Finding, Outcome
from smoke.registry import scenario

S44_ASSERTIONS = (
    "exit 0",
    "stderr contains the ignored-system warning even at --log-level error",
    "applied_caps.system_ignored is true",
    "applied_caps.system_override_applied is false",
)
S45_ASSERTIONS = (
    "the run produced a consult envelope (a provider or timeout error is CANNOT_TEST)",
    "applied_caps.system_override_applied is true in any envelope the run produced",
    "applied_caps.system_ignored is false in any envelope the run produced",
)
S46_ASSERTIONS = (
    "exit 2",
    "stderr names max_system_bytes",
    "no consult or query output on stdout",
)


def _all_fail(assertions, run_id):
    for text in assertions:
        yield Finding(assertion=text, outcome=Outcome.FAIL, detail="Red stub", run_id=run_id)


@scenario("S44", assertions=S44_ASSERTIONS, run="R9", needs_backend=True)
def an_ignored_system_is_announced(run):
    """Red stub: fails every assertion."""
    yield from _all_fail(S44_ASSERTIONS, "R9")


@scenario("S45", assertions=S45_ASSERTIONS, run="R10", needs_backend=True)
def an_applied_system_reaches_the_mages(run):
    """Red stub: fails every assertion."""
    yield from _all_fail(S45_ASSERTIONS, "R10")


@scenario("S46", assertions=S46_ASSERTIONS, run="R11")
def an_oversized_system_is_refused(run):
    """Red stub: fails every assertion."""
    yield from _all_fail(S46_ASSERTIONS, "R11")
