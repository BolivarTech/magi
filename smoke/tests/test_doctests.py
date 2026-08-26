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
    """The loader has to actually find something.

    Without this, deleting every example -- or breaking the walk so it
    imports nothing -- leaves a green suite that certifies no examples at
    all, which is the shape of guardian this harness exists to refuse.
    """

    def test_the_walk_finds_examples_to_run(self) -> None:
        found = 0
        for module in pkgutil.walk_packages(smoke.__path__, "smoke."):
            if module.name.startswith(_SKIPPED):
                continue
            imported = importlib.import_module(module.name)
            found += sum(len(t.examples)
                         for t in doctest.DocTestFinder().find(imported))
        self.assertGreater(found, 0, "no Example: is being executed")
