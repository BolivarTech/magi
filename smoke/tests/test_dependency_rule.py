# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""The dependency rule of section 2.1, checked instead of trusted."""

import ast
import pathlib
import sys
import unittest

FORBIDDEN_IMPORT = "smoke.binary"


class DependencyRuleTests(unittest.TestCase):
    """No scenario reaches the binary directly; it goes through runs."""

    def test_no_scenario_imports_binary(self) -> None:
        """Parsed, not grepped.

        A substring search flags a comment that merely NAMES the module -- and
        the honest comment explaining why a scenario must not import it is
        exactly such a comment, so the naive check punishes the documentation
        it should encourage. ``ast`` sees imports and nothing else.
        """
        package = pathlib.Path(__file__).resolve().parent.parent / "scenarios"
        offenders = []
        for path in package.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                if any(name == FORBIDDEN_IMPORT or
                       name.startswith(FORBIDDEN_IMPORT + ".") for name in names):
                    offenders.append(path.name)
        self.assertEqual(
            [],
            offenders,
            "scenarios must reach the product through runs.py, so the "
            f"invocation count stays auditable; offenders: {offenders}",
        )

    def test_no_third_party_imports_anywhere(self) -> None:
        package = pathlib.Path(__file__).resolve().parent.parent
        offenders: list[str] = []
        for path in package.rglob("*.py"):
            # Parsed, not split on whitespace. ``import os, sys`` names two
            # modules and the string version reads only the first, so a
            # third-party module in second position walks straight past the
            # guard -- a dependency-rule test that admits dependencies.
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                else:
                    continue
                for dotted in modules:
                    root = dotted.split(".")[0]
                    if root in sys.stdlib_module_names or root == "smoke":
                        continue
                    offenders.append(f"{path.name}: {dotted}")
        self.assertEqual([], offenders, f"REQ-S02 allows stdlib only: {offenders}")


if __name__ == "__main__":
    unittest.main()
