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
        unittest.TestSuite: The suite, with one case per DOCSTRING that
        carries examples -- a module with several contributes several. A
        module with none contributes nothing, which is why an empty package
        does not fail here.
    """
    for module in pkgutil.walk_packages(smoke.__path__, "smoke."):
        if module.name.startswith(_SKIPPED):
            continue
        imported = importlib.import_module(module.name)
        if doctest.DocTestFinder().find(imported):
            tests.addTests(doctest.DocTestSuite(imported))
    return tests


def _doctest_cases():
    """The doctest cases UNITTEST would run for this module.

    Loaded through ``defaultTestLoader``, never by calling :func:`load_tests`
    by name, and that distinction is the whole guard. Two earlier versions of
    this check failed the same way one layer apart: the first re-implemented
    the walk, the second called the walk. Neither can observe the thing that
    actually has to hold -- that a module attribute literally spelled
    ``load_tests`` exists and is what unittest invokes. Rename it
    consistently, the way an editor's rename-symbol does, and both of those
    stayed green while zero examples executed anywhere.

    Returns:
        list[doctest.DocTestCase]: One per DOCSTRING that carries examples,
        flattened out of the nested suites the loader builds. Not one per
        example -- ``DocTestSuite`` groups a docstring's examples into a
        single case, so this tree yields 11 cases over 24 examples, and a
        later floor written against the wrong unit would be red on arrival.
    """
    module = importlib.import_module(__name__)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)

    def flatten(item):
        if isinstance(item, unittest.TestSuite):
            for child in item:
                yield from flatten(child)
        else:
            yield item

    return [case for case in flatten(suite)
            if isinstance(case, doctest.DocTestCase)]


class DoctestCoverageTests(unittest.TestCase):
    """Unittest has to be delivering the examples, not merely able to.

    Goes red on all three ways the wiring dies: the loader renamed or moved
    so the protocol stops calling it, a walk that yields nothing, and every
    ``Example:`` deleted. The first is the one two previous versions of this
    check could not see.
    """

    def test_unittest_delivers_the_examples(self) -> None:
        self.assertGreater(len(_doctest_cases()), 0,
                           "unittest is running no doctest at all, so every "
                           "Example: in the package is unexecuted prose")

    def test_the_walk_covers_more_than_one_module(self) -> None:
        """One module's examples is not the package's examples.

        A walk that broke after the first import would satisfy the count
        above while covering almost nothing.
        """
        modules = {case.id().rsplit(".", 1)[0] for case in _doctest_cases()}
        self.assertGreater(len(modules), 1, "the walk stopped early")
