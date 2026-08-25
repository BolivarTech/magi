# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""The report's data contract: what a scenario concludes, and how it says so."""

import dataclasses
import enum


class Outcome(enum.Enum):
    """What a scenario concluded about one assertion.

    ``CANNOT_TEST`` and ``OUT_OF_SCOPE`` are separated by what was *promised*,
    not by who decided (D-17): the first was in scope and the environment
    refused to let it run, so it blocks the gate; the second was never
    promised, so it does not. Collapsing them is what makes a degraded run
    read as a complete one.
    """

    PASS = "PASS"
    FAIL = "FAIL"
    CANNOT_TEST = "CANNOT_TEST"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"

    @property
    def blocks_gate(self) -> bool:
        """Whether this outcome forbids a green run and a certificate.

        Returns:
            True for ``FAIL`` and ``CANNOT_TEST``; False otherwise.
        """
        return self in (Outcome.FAIL, Outcome.CANNOT_TEST)


@dataclasses.dataclass(frozen=True)
class Finding:
    """One assertion and what became of it.

    The scenario id is deliberately absent: the runner stamps it on receipt.
    A field the scenario author fills in by hand gets copy-pasted wrong, and
    the reconciliation then credits the wrong scenario, so the one that really
    ran looks silent while the silent one looks evaluated. Both halves of the
    deception from one typo, and neither raises an error.

    Args:
        assertion: The English text of what was asserted. It reaches the
            certificate verbatim, so it is written once, here.
        outcome: What became of it.
        detail: The cause when the outcome is not ``PASS``; empty on ``PASS``.
        run_id: The shared run that produced it, or None when the scenario is
            standalone.
    """

    assertion: str
    outcome: Outcome
    detail: str
    run_id: str | None
