# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""The harness's domain exceptions.

The split that matters is between a defect in the PRODUCT and a defect in the
HARNESS: the first is a verdict, the second is exit code 3. Never a bare
``except``, and never an ``except Exception`` that swallows the cause.
"""


class SmokeError(Exception):
    """Base for everything this harness raises."""


class HarnessError(SmokeError):
    """A defect in the harness itself. Never a verdict on the product."""


class ProductOutputError(SmokeError):
    """The product's output could not be interpreted.

    Raised only while reading what the product emitted: parsing, shape,
    missing keys, an unexpected exit code. The runner maps this to
    ``Outcome.FAIL``, because malformed output is the product's defect even
    though it reaches the harness as an exception.
    """


class TimedOut(ProductOutputError):
    """The product did not finish inside the wall clock it was given.

    A SUBCLASS on purpose: every caller that already treats a timeout as
    ``ProductOutputError`` keeps working unchanged, and the one caller that has
    to tell a hang from malformed output -- the shared-run executor, which
    turns a hang into ``CANNOT_TEST`` rather than ``FAIL`` -- can catch this
    without reading the message text.

    It carries whatever was captured before the expiry, and that is a
    requirement rather than a courtesy: the scenario exempted from the timeout
    rule reads ``applied_caps`` out of what the run emitted before it hung, so
    an expiry path that discarded the partial streams would silently remove the
    one case the design grants. ``None`` is still possible -- a process killed
    before its first write has nothing -- so the field says "nothing was
    captured", never "nothing was emitted".
    """

    def __init__(self, message: str, output=None) -> None:
        """Record the failure and whatever the product had already emitted.

        Args:
            message: What went wrong.
            output: The partial capture, or ``None`` when there is none.
        """
        super().__init__(message)
        self.output = output


class PreflightError(SmokeError):
    """A precondition failed before any scenario could run. Exit code 2."""
