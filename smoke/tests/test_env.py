# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Tests for the test environment's lifecycle."""

import pathlib
import tempfile
import unittest

from smoke.config import ModelProfile, Seat
from smoke.env import Environment, _apply_profile
from smoke.errors import PreflightError


class LifecycleTests(unittest.TestCase):
    """init creates, reset rebuilds, and neither leaves a half-built tree."""

    def setUp(self) -> None:
        self.root = pathlib.Path(tempfile.mkdtemp()) / "env"

    def test_init_creates_every_declared_subdirectory(self) -> None:
        env = Environment(self.root)
        env.init()
        for path in (env.runs_dir, env.payload_dir, env.scratch_dir):
            self.assertTrue(path.is_dir(), f"{path} was not created")

    def test_init_writes_the_self_protecting_gitignore(self) -> None:
        env = Environment(self.root)
        env.init()
        text = (self.root / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("*", text)
        self.assertIn("!.gitignore", text)

    def test_init_on_an_existing_environment_is_refused(self) -> None:
        env = Environment(self.root)
        env.init()
        with self.assertRaises(PreflightError):
            env.init()

    def test_reset_destroys_and_rebuilds(self) -> None:
        env = Environment(self.root)
        env.init()
        marker = env.runs_dir / "R1" / "stdout"
        marker.parent.mkdir(parents=True)
        marker.write_text("old", encoding="utf-8")
        env.reset()
        self.assertFalse(marker.exists())
        self.assertTrue(env.runs_dir.is_dir())

    def test_exists_is_false_before_init_and_true_after(self) -> None:
        env = Environment(self.root)
        self.assertFalse(env.exists())
        env.init()
        self.assertTrue(env.exists())


class GrowthTests(unittest.TestCase):
    """Growth is unbounded on purpose, so it is made visible."""

    def test_growth_reports_sizes_for_an_empty_environment(self) -> None:
        env = Environment(pathlib.Path(tempfile.mkdtemp()) / "env")
        env.init()
        growth = env.growth()
        self.assertEqual(0, growth.db_bytes)
        self.assertEqual(0, growth.runs_bytes)


if __name__ == "__main__":
    unittest.main()


class ProfileRewriteTests(unittest.TestCase):
    """A profiled seat is rewritten whole, or the file it produces is broken."""

    _GENERATED = (
        '[openai]\nmodel = "kimi-k2.6:cloud"\n\n'
        '[magi]\n'
        'melchior_model  = "qwen3.5:397b-cloud"\n'
        'balthasar_model = "gpt-oss:120b-cloud"\n'
        'caspar_model    = "deepseek-v4-pro:cloud"\n'
        'melchior_lineage  = "alibaba"\n'
        'balthasar_lineage = "openai"\n'
        'caspar_lineage    = "deepseek"\n'
        '# melchior_model = "commented out"\n'
        '[anthropic]\nmodel = "claude-sonnet-4-6"\n'
    )

    def _profile(self) -> ModelProfile:
        """Build a three-seat profile whose lineages differ from the file's.

        Returns:
            ModelProfile: The profile under test.
        """
        return ModelProfile(
            model="cheap-main",
            trio=[
                Seat(model="cheap-a", lineage="alpha"),
                Seat(model="cheap-b", lineage="beta"),
                Seat(model="cheap-c", lineage="gamma"),
            ],
        )

    def test_each_seat_has_both_of_its_keys_rewritten(self) -> None:
        """Model and lineage move together.

        Rewriting the model alone leaves the file declaring a failure domain
        the seat no longer belongs to -- and since v0.13.0 the product treats a
        seat whose halves disagree as a load error, so a half-rewrite does not
        degrade gracefully, it fails to start. Drop the lineage line from
        ``_apply_profile`` and this goes red.
        """
        rewritten = _apply_profile(self._GENERATED, self._profile())
        for model, lineage in (("cheap-a", "alpha"), ("cheap-b", "beta"),
                               ("cheap-c", "gamma")):
            self.assertIn('"%s"' % model, rewritten)
            self.assertIn('"%s"' % lineage, rewritten)
        for stale in ("qwen3.5:397b-cloud", "alibaba", "deepseek"):
            self.assertNotIn('= "%s"' % stale, rewritten)

    def test_a_key_of_the_same_name_in_another_table_is_untouched(self) -> None:
        """``model`` exists under [openai] and under [anthropic]; only the
        first is the main agent's. A line-wise rewrite that forgot which table
        it was in would silently repoint the product's Anthropic backend.
        """
        rewritten = _apply_profile(self._GENERATED, self._profile())
        self.assertIn('"claude-sonnet-4-6"', rewritten)
        self.assertIn('"cheap-main"', rewritten)

    def test_a_commented_line_is_left_alone(self) -> None:
        """Commented configuration is documentation. Rewriting it would put a
        cheap model into an example the operator reads as the product's.
        """
        rewritten = _apply_profile(self._GENERATED, self._profile())
        self.assertIn('# melchior_model = "commented out"', rewritten)
