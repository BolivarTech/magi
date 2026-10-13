# Author: Julian Bolivar
# Version: 0.23.0
# Date: 2026-10-11
"""The MS2 decide scenarios: registration, declared totals and verbatim assertion texts."""

import pathlib
import unittest

from smoke.registry import (DECLARED_ASSERTION_COUNT, DECLARED_SCENARIO_COUNT,
                            DEFAULT_REGISTRY)

_SCENARIOS = pathlib.Path(__file__).resolve().parents[1] / "scenarios"

#: The assertion texts are the contract (T11's table): they must appear verbatim.
_DECIDE_ASSERTIONS = (
    "when the agent calls decide in the default tier the call is approved",
    "a consult with a dead decision endpoint still completes",
    "the run log records one mode fallback with its cause",
    "nothing about the fallback reaches stderr or stdout",
    "the decide call in tool_calls stores the state as <omitted>",
    "the decide call in tool_calls keeps its questions and its result",
    "the state marker reaches neither the decide record nor the decide log lines",
)
_DOCS_ASSERTIONS = (
    "the documented /decide forms and the decide tool exist",
    "no help screen of the release binary shows a development id",
)


def _source(name: str) -> str:
    """Return a scenario module's source with line endings normalised."""
    return (_SCENARIOS / name).read_text(encoding="utf-8").replace("\r", "")


class DecideMs2RegistryTests(unittest.TestCase):
    """D11-D13 are registered and the declared totals moved with them."""

    def test_the_three_ms2_decide_scenarios_are_registered(self) -> None:
        import smoke.scenarios  # noqa: F401 -- importing registers every scenario
        ids = DEFAULT_REGISTRY.registered_ids()
        for scenario_id in ("D11", "D12", "D13"):
            self.assertIn(scenario_id, ids)

    def test_the_declared_totals_include_them(self) -> None:
        self.assertEqual(42, DECLARED_SCENARIO_COUNT)
        self.assertEqual(125, DECLARED_ASSERTION_COUNT)


class DecideMs2AssertionTextTests(unittest.TestCase):
    """Every declared assertion text is present verbatim in its scenario module."""

    def test_the_decide_assertion_texts_are_verbatim(self) -> None:
        source = _source("decide.py")
        for text in _DECIDE_ASSERTIONS:
            with self.subTest(text=text):
                self.assertIn(text, source)

    def test_the_two_documentation_assertion_texts_are_verbatim(self) -> None:
        source = _source("docs.py")
        for text in _DOCS_ASSERTIONS:
            with self.subTest(text=text):
                self.assertIn(text, source)


if __name__ == "__main__":
    unittest.main()
