# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Unit tests for the S2 scenario's own shape."""

import unittest

from smoke.outcome import Outcome
from smoke.registry import DEFAULT_REGISTRY
from smoke.scenarios import workspace  # noqa: F401 - import registers it
from smoke.tests import support


class WorkspaceScenarioTests(unittest.TestCase):
    """S2 is registered standalone and declares its four assertions."""

    def test_s2_is_registered_without_a_run(self) -> None:
        entry = DEFAULT_REGISTRY.get("S2")
        self.assertIsNone(entry.run)
        self.assertFalse(entry.needs_backend)

    def test_the_assertion_texts_are_the_spec_texts(self) -> None:
        self.assertEqual(
            [
                "magi init creates .magi/ in an empty directory",
                "the permissions are restrictive — POSIX bits on Unix, ACL on Windows",
                "a second init refuses and leaves the directory unchanged",
                "query from a nested subdirectory finds the ancestor .magi/",
            ],
            list(workspace.ASSERTIONS),
        )


class WorkspaceScenarioBodyTests(unittest.TestCase):
    """Every assertion is reported, whatever the product did."""

    def test_a_product_that_does_nothing_still_reports_all_four(self) -> None:
        """The reconciliation of section 2.4 calls silence a harness failure,
        so a scenario that gives up half way is a defect even when the product
        is the thing that failed. Against a binary that scaffolds nothing,
        every assertion must still come back -- and none of them as PASS.
        """
        support.install_fake_runs(self)
        entry = DEFAULT_REGISTRY.get("S2")
        findings = list(entry.func(None))
        self.assertEqual(list(workspace.ASSERTIONS),
                         [finding.assertion for finding in findings])
        self.assertNotIn(Outcome.PASS,
                         {finding.outcome for finding in findings})

    def test_the_seed_runs_init_in_the_scratch_area_by_cwd(self) -> None:
        """The precondition may not use ``-w``: that is the flag S14 tests, and
        a seed that used it could not fail when the flag stopped working.
        """
        binary = support.install_fake_runs(self)
        list(DEFAULT_REGISTRY.get("S2").func(None))
        seeds = [call for call in binary.calls if call.args[:1] == ("init",)]
        self.assertTrue(seeds, "S2 never ran the product's init")
        for call in seeds:
            self.assertIsNotNone(call.cwd, "the seed must name a cwd")
            self.assertNotIn("-w", call.args)
            self.assertNotIn("--workdir", call.args)


if __name__ == "__main__":
    unittest.main()
