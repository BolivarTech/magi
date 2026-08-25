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


class PreflightError(SmokeError):
    """A precondition failed before any scenario could run. Exit code 2."""
