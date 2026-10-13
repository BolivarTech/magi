# Author: Julian Bolivar
# Version: 0.23.0
# Date: 2026-10-13
"""The per-section key scenarios K1-K3: registration, totals, texts and judging helpers."""

import pathlib
import tomllib
import unittest

from smoke.product import ProductOutput
from smoke.registry import (DECLARED_ASSERTION_COUNT, DECLARED_SCENARIO_COUNT,
                            DEFAULT_REGISTRY)

_SCENARIOS = pathlib.Path(__file__).resolve().parents[1] / "scenarios"

#: The assertion texts are the contract (T14's table): they must appear verbatim.
_K1 = (
    "the run log has exactly one entry naming MAGI_API_KEY and "
    "http://127.0.0.1:11434",
    "the screen shows nothing about the retained key",
)
_K2 = (
    "a sk-ant- MAGI_API_KEY under openai-compat warns naming MAGI_API_KEY and "
    "[magi].kind, without the value",
    "a sk-or- MAGI_API_KEY under anthropic warns the same way",
    "a sk-or- MAGI_API_KEY under openai-compat does not warn",
)
_K3 = (
    "with MAGI_API_KEY the OpenRouter trio answers a consult",
    "without it the consult failure names MAGI_API_KEY",
)

#: What ``magi init`` writes, reduced to the parts the scenarios rewrite.
_SCAFFOLD = (
    'provider = "ollama"\nbase_url = "http://localhost:11434/v1"\n\n'
    '[decision]\nbase_url = "http://localhost:11434"\nmodel = "nimble"\n\n'
    '[magi]  # the trio\n'
    'melchior_model  = "glm-5.3:cloud"\nbalthasar_model = "kimi-k2.6:cloud"\n'
    'caspar_model    = "deepseek-v4-pro:cloud"\n'
    'melchior_lineage  = "zhipu"\nbalthasar_lineage = "moonshot"\n'
    'caspar_lineage    = "deepseek"\nmax_rotations   = 2\n\n'
    '[[magi.fallback]]\nmodel   = "minimax-m3:cloud"\nlineage = "minimax"\n'
)


def _capture(stdout=b"", stderr=b"", exit_code=1):
    """Build a canned capture."""
    return ProductOutput(stdout=stdout, stderr=stderr, exit_code=exit_code,
                         command=["magi-rs"])


def _source():
    """The scenario module's source with line endings normalised."""
    return (_SCENARIOS / "keys.py").read_text(encoding="utf-8").replace("\r", "")


class KeysRegistrationTests(unittest.TestCase):
    """K1-K3 are registered, exact in their assertions, and the totals moved."""

    def test_the_three_key_scenarios_are_registered_with_their_assertion_counts(self):
        import smoke.scenarios  # noqa: F401 -- importing registers every scenario
        for scenario_id, texts in (("K1", _K1), ("K2", _K2), ("K3", _K3)):
            with self.subTest(scenario=scenario_id):
                self.assertIn(scenario_id, DEFAULT_REGISTRY.registered_ids())
                self.assertEqual(texts, DEFAULT_REGISTRY.get(scenario_id).assertions)

    def test_the_declared_totals_include_them(self):
        self.assertEqual(42, DECLARED_SCENARIO_COUNT)
        self.assertEqual(125, DECLARED_ASSERTION_COUNT)

    def test_the_assertion_texts_are_verbatim_in_the_module(self):
        source = _source()
        for text in _K1 + _K2 + _K3:
            with self.subTest(text=text):
                self.assertIn(text, source)

    def test_k1_needs_a_backend_and_the_other_two_do_not(self):
        import smoke.scenarios  # noqa: F401
        self.assertTrue(DEFAULT_REGISTRY.get("K1").needs_backend)
        self.assertFalse(DEFAULT_REGISTRY.get("K2").needs_backend)
        self.assertFalse(DEFAULT_REGISTRY.get("K3").needs_backend)


class KeysJudgingHelperTests(unittest.TestCase):
    """The pure pieces the scenarios judge with."""

    def test_the_shape_warning_is_read_from_stderr_only(self):
        from smoke.scenarios import keys
        warned = _capture(stderr=b"WARN MAGI_API_KEY does not look like a key "
                                 b"for the provider of [magi].kind")
        self.assertTrue(keys.shape_warning_shown(warned))
        self.assertFalse(keys.shape_warning_shown(_capture(stderr=b"quiet")))

    def test_a_warning_that_repeats_the_key_value_is_a_leak(self):
        from smoke.scenarios import keys
        output = _capture(stderr=b"MAGI_API_KEY does not look like a key sk-ant-x")
        self.assertTrue(keys.leaks_value(output, "sk-ant-x"))
        self.assertFalse(keys.leaks_value(_capture(stderr=b"nothing"), "sk-ant-x"))

    def test_log_entries_count_lines_naming_both_the_variable_and_the_origin(self):
        from smoke.scenarios import keys
        log = (b"INFO OPENAI_API_KEY was not sent to http://127.0.0.1:11434/v1: "
               b"magi-rs vault set MAGI_API_KEY\n"
               b"INFO unrelated MAGI_API_KEY only\n"
               b"INFO unrelated http://127.0.0.1:11434 only\n")
        self.assertEqual(1, keys.retained_entries(log))
        self.assertEqual(0, keys.retained_entries(b""))

    def test_the_screen_check_finds_the_notice_on_either_stream(self):
        from smoke.scenarios import keys
        self.assertTrue(keys.retained_on_screen(
            _capture(stdout=b"OPENAI_API_KEY was not sent to http://x")))
        self.assertTrue(keys.retained_on_screen(
            _capture(stderr=b"OPENAI_API_KEY was not sent to http://x")))
        self.assertFalse(keys.retained_on_screen(_capture(stderr=b"fine")))


class KeysConfigurationRewriteTests(unittest.TestCase):
    """The scaffold rewrites produce parseable files with the intended values."""

    def test_the_magi_section_gets_a_kind_and_an_endpoint(self):
        from smoke.scenarios import keys
        text = keys.with_magi_endpoint(_SCAFFOLD, "openai-compat",
                                       "http://127.0.0.1:11434/v1")
        document = tomllib.loads(text)
        self.assertEqual("openai-compat", document["magi"]["kind"])
        self.assertEqual("http://127.0.0.1:11434/v1", document["magi"]["base_url"])
        self.assertEqual("glm-5.3:cloud", document["magi"]["melchior_model"])

    def test_a_kind_without_an_endpoint_leaves_the_endpoint_inherited(self):
        from smoke.scenarios import keys
        document = tomllib.loads(keys.with_magi_endpoint(_SCAFFOLD, "anthropic"))
        self.assertEqual("anthropic", document["magi"]["kind"])
        self.assertNotIn("base_url", document["magi"])

    def test_a_file_without_a_magi_section_is_not_rewritten(self):
        from smoke.scenarios import keys
        self.assertIsNone(keys.with_magi_endpoint('provider = "ollama"\n',
                                                  "anthropic"))

    def test_the_root_endpoint_is_replaced_not_the_decision_one(self):
        from smoke.scenarios import keys
        document = tomllib.loads(keys.with_root_endpoint(
            _SCAFFOLD, "http://127.0.0.1:9/v1"))
        self.assertEqual("http://127.0.0.1:9/v1", document["base_url"])
        self.assertEqual("http://localhost:11434", document["decision"]["base_url"])

    def test_the_hosted_trio_has_three_seats_no_pool_and_no_rotation(self):
        from smoke.scenarios import keys
        text = keys.with_hosted_trio(_SCAFFOLD, "https://openrouter.ai/api/v1")
        document = tomllib.loads(text)
        magi = document["magi"]
        self.assertEqual("openai-compat", magi["kind"])
        self.assertEqual("https://openrouter.ai/api/v1", magi["base_url"])
        self.assertEqual(0, magi["max_rotations"])
        self.assertNotIn("fallback", magi)
        lineages = {magi["melchior_lineage"], magi["balthasar_lineage"],
                    magi["caspar_lineage"]}
        self.assertEqual(3, len(lineages))
        for seat in ("melchior", "balthasar", "caspar"):
            self.assertNotIn("cloud", magi[seat + "_model"])


if __name__ == "__main__":
    unittest.main()
