# Author: Julian Bolivar
# Version: 0.23.0
# Date: 2026-10-11
"""The cheap profile's decision model, the preflight's view of it, and the decide scenario count."""

import tomllib
import unittest

from smoke.config import ModelProfile, Seat
from smoke.env import Environment, _apply_profile, table_header
from smoke.errors import PreflightError
from smoke.scenarios import decide
from smoke.registry import (DECLARED_ASSERTION_COUNT, DECLARED_SCENARIO_COUNT,
                            DEFAULT_REGISTRY)
from smoke.tests import support
from smoke.tests.test_config import _MINIMAL

_GENERATED = (
    '[openai]\nmodel = "kimi-k2.6:cloud"\n\n'
    '[decision]\nbase_url = "http://localhost:11434"\nmodel = "nimble"\n'
    '# min_confidence = 0.5\n\n'
    '[[decision.fallback]]\nmodel = "tev1"\n\n'
    '[[decision.fallback]]\nmodel = "tev1:0.8b"\n\n'
    '[magi]\nmelchior_model = "m"\n'
)


def _profile(decision):
    """Build a three-seat profile with the given decision model (or None)."""
    return ModelProfile(
        model="cheap-main",
        trio=[Seat(model="a", lineage="x"), Seat(model="b", lineage="y"),
              Seat(model="c", lineage="z")],
        decision=decision,
    )


class DecisionProfileRewriteTests(unittest.TestCase):
    """One cheap decision model replaces the primary and the chain goes."""

    def test_the_primary_is_replaced_and_the_fallbacks_are_dropped(self) -> None:
        doc = tomllib.loads(_apply_profile(_GENERATED, _profile("tev1:0.8b")))
        self.assertEqual("tev1:0.8b", doc["decision"]["model"])
        self.assertNotIn("fallback", doc["decision"])
        self.assertEqual("http://localhost:11434", doc["decision"]["base_url"])
        self.assertEqual("a", doc["magi"]["melchior_model"], "the next table survives the drop")

    def test_a_commented_threshold_outside_a_fallback_stays_commented(self) -> None:
        self.assertIn("# min_confidence = 0.5", _apply_profile(_GENERATED, _profile("tev1:0.8b")))

    def test_without_a_decision_model_the_section_is_left_as_written(self) -> None:
        doc = tomllib.loads(_apply_profile(_GENERATED, _profile(None)))
        self.assertEqual("nimble", doc["decision"]["model"])
        self.assertEqual(["tev1", "tev1:0.8b"],
                         [f["model"] for f in doc["decision"]["fallback"]])


class FallbackHeaderSpellingTests(unittest.TestCase):
    """Header detection follows TOML, not one literal spelling."""

    _SPACED = (
        '[decision]\nmodel = "nimble"\n\n'
        '[[ decision.fallback ]]\nmodel = "tev1"\n\n'
        '[magi]\nmelchior_model = "m"\n'
    )
    _COMMENTED = (
        '[decision]\nmodel = "nimble"\n'
        '# [[decision.fallback]]\n# model = "tev1"\n\n'
        '[magi]\nmelchior_model = "m"\n# [[decision.fallback]]\nkept = 1\n'
    )

    def test_a_spaced_fallback_header_is_dropped_by_the_profile(self) -> None:
        doc = tomllib.loads(_apply_profile(self._SPACED, _profile("cheap")))
        self.assertNotIn("fallback", doc["decision"])
        self.assertEqual("a", doc["magi"]["melchior_model"])

    def test_a_spaced_fallback_header_is_dropped_by_without_decision(self) -> None:
        doc = tomllib.loads(decide._without_decision(self._SPACED))
        self.assertNotIn("decision", doc)
        self.assertEqual("m", doc["magi"]["melchior_model"])

    def test_a_commented_header_is_not_a_header(self) -> None:
        rewritten = _apply_profile(self._COMMENTED, _profile("cheap"))
        self.assertIn("# [[decision.fallback]]", rewritten)
        self.assertIn('# model = "tev1"', rewritten)
        stripped = tomllib.loads(decide._without_decision(self._COMMENTED))
        self.assertEqual(1, stripped["magi"]["kept"])


class DecisionProfileLoadTests(unittest.TestCase):
    """`decision` in [profile.cheap] is one model name, optional."""

    _CHEAP = (
        '\n[profile.cheap]\nmodel = "m"\ntrio = [\n'
        '  { model = "a", lineage = "alpha" },\n'
        '  { model = "b", lineage = "beta" },\n'
        '  { model = "c", lineage = "gamma" },\n]\n'
    )

    def _write(self, text):
        path = support.scratch_dir(self) / "smoke.toml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_a_declared_decision_model_is_read(self) -> None:
        profile = ModelProfile.load(self._write(_MINIMAL + self._CHEAP + 'decision = "tev1:0.8b"\n'))
        self.assertEqual("tev1:0.8b", profile.decision)

    def test_an_absent_decision_model_is_none(self) -> None:
        self.assertIsNone(ModelProfile.load(self._write(_MINIMAL + self._CHEAP)).decision)

    def test_a_blank_or_non_text_decision_model_is_refused(self) -> None:
        for bad in ('decision = ""\n', 'decision = ["tev1:0.8b"]\n'):
            with self.subTest(bad=bad):
                with self.assertRaises(PreflightError) as caught:
                    ModelProfile.load(self._write(_MINIMAL + self._CHEAP + bad))
                self.assertIn("decision", str(caught.exception))


class DecisionDeclaredModelTests(unittest.TestCase):
    """The preflight asks for the decision primary, never for its fallbacks."""

    def test_the_decision_primary_is_declared_and_its_fallbacks_are_not(self) -> None:
        root = support.scratch_dir(self)
        (root / ".magi").mkdir()
        (root / ".magi" / "magi.toml").write_text(_GENERATED, encoding="utf-8")
        declared = Environment(root).declared_models()
        self.assertIn("nimble", declared)
        self.assertNotIn("tev1", declared)
        self.assertNotIn("tev1:0.8b", declared)


class DecideRegistryTests(unittest.TestCase):
    """The ten decide scenarios are registered and the totals moved with them."""

    def test_the_ten_decide_scenarios_are_registered(self) -> None:
        import smoke.scenarios  # noqa: F401 -- importing registers every scenario
        ids = DEFAULT_REGISTRY.registered_ids()
        for n in range(1, 11):
            self.assertIn("D%d" % n, ids)

    def test_the_declared_totals_include_them(self) -> None:
        self.assertEqual(43, DECLARED_SCENARIO_COUNT)
        self.assertEqual(127, DECLARED_ASSERTION_COUNT)


class TableHeaderTests(unittest.TestCase):
    """The shared header reading ignores blanks and trailing comments."""

    def test_an_inline_comment_after_a_header_is_ignored(self) -> None:
        self.assertEqual("[[decision.fallback]]",
                         table_header("[[decision.fallback]]  # tuned"))
        self.assertEqual("[decision]", table_header("[ decision ] # note"))

    def test_a_commented_or_plain_line_is_not_a_header(self) -> None:
        self.assertIsNone(table_header("# [[decision.fallback]]"))
        self.assertIsNone(table_header('model = "x" # [a]'))


if __name__ == "__main__":
    unittest.main()
