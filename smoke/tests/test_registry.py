# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Tests for scenario registration and its silence-detection guard."""

import pathlib
import ast
import unittest

from smoke.errors import HarnessError
from smoke.registry import DECLARED_SCENARIO_COUNT, Registry, scenario


class RegistryTests(unittest.TestCase):
    """Registration records what a scenario needs before anything runs."""

    def setUp(self) -> None:
        self.registry = Registry()

    def _register(self, scenario_id: str, **kwargs: object) -> None:
        scenario(scenario_id, registry=self.registry, **kwargs)(
            lambda run: iter(())
        )

    def test_decorator_records_run_and_backend_need(self) -> None:
        self._register("S7", run="R4", needs_backend=True)
        entry = self.registry.get("S7")
        self.assertEqual("R4", entry.run)
        self.assertTrue(entry.needs_backend)

    def test_standalone_scenario_declares_no_run(self) -> None:
        self._register("S2")
        self.assertIsNone(self.registry.get("S2").run)
        self.assertFalse(self.registry.get("S2").needs_backend)

    def test_duplicate_id_is_a_harness_error(self) -> None:
        self._register("S2")
        with self.assertRaises(HarnessError):
            self._register("S2")

    def test_registered_ids_sort_numerically_within_the_prefix(self) -> None:
        for scenario_id in ("S13", "S2", "S9"):
            self._register(scenario_id)
        self.assertEqual(["S2", "S9", "S13"], self.registry.registered_ids())


class ScenarioModuleImportTests(unittest.TestCase):
    """A module nobody imported never registers, so nothing can miss it."""

    def test_every_scenario_module_is_imported_by_the_package(self) -> None:
        package = pathlib.Path(__file__).resolve().parent.parent / "scenarios"
        on_disk = {p.stem for p in package.glob("*.py") if p.stem != "__init__"}
        source = (package / "__init__.py").read_text(encoding="utf-8")
        missing = {name for name in on_disk if f"from smoke.scenarios import {name}" not in source}
        self.assertEqual(
            set(),
            missing,
            f"scenario modules present but never imported: {sorted(missing)}",
        )


class AssertionSourceTests(unittest.TestCase):
    """The decorator is handed the module's own constant, never a copy.

    The texts are declared once, as a module constant, and reach the
    certificate verbatim from there. A decorator given a literal instead would
    create a second copy that the completeness check cannot see drifting: it
    compares findings against the decorator's tuple, so both sides of a
    divergence would agree with each other and disagree with the constant the
    scenario actually builds its findings from.
    """

    def test_every_registration_names_a_constant(self) -> None:
        """Parsed, not pattern-matched.

        A regex over the decorator line reads argument ORDER, so
        ``@scenario("S20", run="R1", assertions=ASSERTIONS)`` -- which is
        legal and which somebody will write -- would not match and the guard
        would go red over a correct registration. Reading the call from the
        syntax tree asks what was actually passed.
        """
        checked = 0
        for path in sorted(pathlib.Path("smoke/scenarios").glob("*.py")):
            if path.name == "__init__.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            declared = {node.targets[0].id for node in tree.body
                        if isinstance(node, ast.Assign)
                        and getattr(node.targets[0], "id", "").endswith("ASSERTIONS")}
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call)
                        and getattr(node.func, "id", "") == "scenario"):
                    continue
                passed = {kw.arg: kw.value for kw in node.keywords}
                self.assertIn("assertions", passed,
                              "%s registers a scenario without declaring its "
                              "assertions" % path.name)
                value = passed["assertions"]
                self.assertIsInstance(
                    value, ast.Name,
                    "%s passes a literal rather than the module's own "
                    "constant" % path.name)
                self.assertIn(value.id, declared,
                              "%s passes %r, which is not a constant it "
                              "declares" % (path.name, value.id))
                checked += 1
        self.assertEqual(DECLARED_SCENARIO_COUNT, checked,
                         "every registered scenario must declare its "
                         "assertions to the decorator")


if __name__ == "__main__":
    unittest.main()
