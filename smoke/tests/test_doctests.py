# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-26
"""Every ``Example:`` in the harness is executed, not decoration.

The docstrings carried worked examples from the first task and nothing ever
ran one, so they were prose that happened to be shaped like code -- and prose
drifts. Three more were added in a single review round before anybody checked
whether any of them still evaluated.

The loader walks the package rather than naming modules, so a new module's
examples are covered the moment it exists. ``smoke.tests`` itself is skipped:
its doubles carry examples that only make sense with the module configured,
and collecting them here would run them against an unconfigured one.
"""

import doctest
import importlib
import pkgutil
import unittest

import smoke

#: Subpackages whose examples are not collected, and why.
_SKIPPED = ("smoke.tests",)


def load_tests(loader, tests, ignore):  # noqa: ARG001 - unittest protocol
    """Add every module's doctests to the suite.

    Args:
        loader: The unittest loader. Unused; the protocol supplies it.
        tests: The suite collected so far.
        ignore: Unused; the protocol supplies it.

    Returns:
        unittest.TestSuite: The suite, with one case per module that has
        examples. A module with none contributes nothing, which is why an
        empty package does not fail here.
    """
    for module in pkgutil.walk_packages(smoke.__path__, "smoke."):
        if module.name.startswith(_SKIPPED):
            continue
        imported = importlib.import_module(module.name)
        if doctest.DocTestFinder().find(imported):
            tests.addTests(doctest.DocTestSuite(imported))
    return tests


class DoctestCoverageTests(unittest.TestCase):
    """The LOADER has to find something, not a copy of the loader.

    The first version of this walked the package itself and counted what it
    found, which guards the wrong half. Rename ``load_tests`` and no example
    executes anywhere, yet a test that re-implements the walk still passes:
    it certified that examples EXIST, while the suite quietly went back to
    running none of them. Mutation said so -- the rename left this file
    green.

    So it calls the real entry point. That goes red on a renamed, mistyped
    or exception-swallowing loader, and still goes red if every example is
    deleted.
    """

    def test_the_loader_produces_cases_to_run(self) -> None:
        suite = load_tests(unittest.TestLoader(), unittest.TestSuite(), None)
        self.assertGreater(suite.countTestCases(), 0,
                           "the loader produced no doctest cases, so no "
                           "Example: in the package is being executed")

    def test_the_loader_covers_more_than_one_module(self) -> None:
        """One module's examples is not the package's examples.

        A walk that broke after the first import would satisfy the count
        above while covering almost nothing.
        """
        suite = load_tests(unittest.TestLoader(), unittest.TestSuite(), None)
        modules = {case.id().rsplit(".", 1)[0] for case in suite}
        self.assertGreater(len(modules), 1, "the walk stopped early")
