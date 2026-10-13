# Author: Julian Bolivar
# Version: 0.24.0
# Date: 2026-10-13
"""The adoption scenario D14: registration, verbatim texts and judging helpers."""

import pathlib
import unittest

from smoke.registry import DEFAULT_REGISTRY

_SCENARIOS = pathlib.Path(__file__).resolve().parents[1] / "scenarios"

#: The assertion texts are the contract (T19's table): they must appear verbatim.
_D14 = (
    "the scaffold names clef-flash with its threshold active",
    "magi decide answers the example request with clef-flash",
)

_SCAFFOLD_DECISION = (
    '[decision]\nbase_url = "http://localhost:11434"\nmodel = "clef-flash"\n'
    "min_confidence = 0.75\n\n"
    '[[decision.fallback]]\nmodel = "nimble"\nmin_confidence = 0.75\n'
)


class DecideAdoptionRegistrationTests(unittest.TestCase):
    """D14 is registered with its exact assertions and needs a backend."""

    def test_d14_is_registered_with_its_assertion_texts(self) -> None:
        import smoke.scenarios  # noqa: F401 -- importing registers every scenario
        self.assertIn("D14", DEFAULT_REGISTRY.registered_ids())
        self.assertEqual(_D14, DEFAULT_REGISTRY.get("D14").assertions)

    def test_d14_needs_a_backend(self) -> None:
        import smoke.scenarios  # noqa: F401
        self.assertTrue(DEFAULT_REGISTRY.get("D14").needs_backend)

    def test_the_assertion_texts_are_verbatim_in_the_module(self) -> None:
        source = (_SCENARIOS / "decide.py").read_text(encoding="utf-8").replace("\r", "")
        for text in _D14:
            with self.subTest(text=text):
                self.assertIn(text, source)


class ScaffoldJudgingTests(unittest.TestCase):
    """The scaffold judgement reads the parsed file, not its comments."""

    def test_a_scaffold_with_the_adopted_chain_is_accepted(self) -> None:
        from smoke.scenarios import decide
        self.assertEqual("", decide.scaffold_adoption_problem(_SCAFFOLD_DECISION))

    def test_another_primary_model_is_reported(self) -> None:
        from smoke.scenarios import decide
        text = _SCAFFOLD_DECISION.replace('model = "clef-flash"', 'model = "nimble"', 1)
        self.assertIn("nimble", decide.scaffold_adoption_problem(text))

    def test_a_commented_threshold_is_not_active(self) -> None:
        from smoke.scenarios import decide
        text = _SCAFFOLD_DECISION.replace("min_confidence = 0.75\n\n", "# min_confidence = 0.75\n\n", 1)
        self.assertIn("min_confidence", decide.scaffold_adoption_problem(text))

    def test_a_different_fallback_is_reported(self) -> None:
        from smoke.scenarios import decide
        text = _SCAFFOLD_DECISION.replace('"nimble"', '"tev1"')
        self.assertIn("tev1", decide.scaffold_adoption_problem(text))

    def test_text_that_does_not_parse_is_reported(self) -> None:
        from smoke.scenarios import decide
        self.assertNotEqual("", decide.scaffold_adoption_problem("[decision\n"))

    def test_a_file_without_a_decision_section_is_reported(self) -> None:
        from smoke.scenarios import decide
        self.assertNotEqual("", decide.scaffold_adoption_problem('provider = "ollama"\n'))


if __name__ == "__main__":
    unittest.main()
