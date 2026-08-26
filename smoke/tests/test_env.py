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

    def test_the_scratch_area_is_outside_the_product_workspace(self) -> None:
        """It exists to hold trees that carry NO ``.magi/``, so it cannot live
        under one.

        The environment root holds the product's workspace, and ``magi init``
        refuses to nest a second ``.magi/`` inside an existing one. A scratch
        area under the root therefore fails the one thing it was created for:
        every scenario that seeds a workspace there gets
        ``refusing to nest a second .magi/ inside it`` and reports the
        product's correct behaviour as its own inability to run.
        """
        env = Environment(self.root)
        env.init()
        self.assertFalse(
            env.scratch_dir.resolve().is_relative_to(env.root.resolve()),
            "magi init cannot scaffold inside the environment's own workspace",
        )

    def test_the_scratch_area_protects_itself_from_git(self) -> None:
        """It sits beside the environment, so it needs its own ignore rule.

        The environment's ``.gitignore`` covers the environment and nothing
        else, and the scenario that asserts the harness left no trace reads
        ``git status``.
        """
        env = Environment(self.root)
        env.init()
        text = (env.scratch_dir / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("*", text)
        self.assertIn("!.gitignore", text)

    def test_reset_empties_the_scratch_area_too(self) -> None:
        """``--reset-env`` is the recovery for an environment that grew.

        The workspaces scenarios seed there are throwaway and accumulate one
        per run, so leaving them behind would make the reset a partial one.
        """
        env = Environment(self.root)
        env.init()
        stale = env.scratch_dir / "s11-old" / "marker"
        stale.parent.mkdir(parents=True)
        stale.write_text("old", encoding="utf-8")
        env.reset()
        self.assertFalse(stale.exists())
        self.assertTrue(env.scratch_dir.is_dir())

    def test_exists_is_false_before_init_and_true_after(self) -> None:
        env = Environment(self.root)
        self.assertFalse(env.exists())
        env.init()
        self.assertTrue(env.exists())


class InitialisedTests(unittest.TestCase):
    """"The directory is there" and "the environment exists" are not the same."""

    def test_a_directory_holding_only_its_gitignore_is_not_initialised(self) -> None:
        """The tracked ``smoke/env/.gitignore`` makes the directory exist in
        every clone, so reading existence off the directory means ``--init-env``
        refuses before it has ever run. Found by running the harness, not by a
        unit test: the other tests build environments in temporary directories,
        where that tracked file is not present.
        """
        root = pathlib.Path(tempfile.mkdtemp()) / "env"
        root.mkdir()
        (root / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")
        self.assertFalse(Environment(root).exists())

    def test_init_succeeds_into_that_directory(self) -> None:
        """The consequence, end to end: a fresh clone can build its environment."""
        root = pathlib.Path(tempfile.mkdtemp()) / "env"
        root.mkdir()
        (root / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")
        env = Environment(root)
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


class MemorySettingsTests(unittest.TestCase):
    """A default the product COMMENTED OUT is still a value it declared."""

    def _write_env(self, body: str) -> Environment:
        """Build an environment whose magi.toml holds *body*.

        Args:
            body: The file's contents.

        Returns:
            Environment: Ready to read.
        """
        root = pathlib.Path(tempfile.mkdtemp()) / "env"
        env = Environment(root)
        env.init()
        (env.magi_dir / "magi.toml").write_text(body, encoding="utf-8")
        return env

    def test_a_live_key_is_read(self) -> None:
        env = self._write_env("[memory]\ncontext_budget_tokens = 8000\n")
        self.assertEqual(8000, env.memory_settings()["context_budget_tokens"])

    def test_a_commented_default_is_read_as_a_default(self) -> None:
        """``magi init`` writes two of the three fields commented, WITH their
        real values interpolated -- so the file carries the product's own
        defaults as data. Reading only the live keys made S9's ceiling
        permanently underivable, including on the certifying run, and the
        alternative (copying the defaults into the harness) is the second
        source of truth this project rejects everywhere else. The value comes
        from the line the product wrote.
        """
        env = self._write_env(
            "[memory]\ncontext_budget_tokens = 8000\n"
            "# response_headroom_tokens = 1024    # reserved for the reply\n"
            "# safety_margin_ratio = 0.1           # heuristic guard\n"
        )
        settings = env.memory_settings()
        self.assertEqual(1024, settings["response_headroom_tokens"])
        self.assertEqual(0.1, settings["safety_margin_ratio"])

    def test_a_live_key_wins_over_a_commented_one(self) -> None:
        """An operator who uncommented and changed a value means it."""
        env = self._write_env(
            "[memory]\n# response_headroom_tokens = 1024\n"
            "response_headroom_tokens = 4096\n"
        )
        self.assertEqual(4096, env.memory_settings()["response_headroom_tokens"])

    def test_a_key_that_appears_nowhere_is_still_absent(self) -> None:
        """Absent is not zero, and this is the half that keeps it so."""
        env = self._write_env("[memory]\ncontext_budget_tokens = 8000\n")
        self.assertNotIn("safety_margin_ratio", env.memory_settings())

    def test_a_commented_key_of_another_table_is_not_borrowed(self) -> None:
        """The scan is table-aware; a commented key under [openai] is not a
        memory setting just because it is commented.
        """
        env = self._write_env(
            "[openai]\n# safety_margin_ratio = 9.9\n"
            "[memory]\ncontext_budget_tokens = 8000\n"
        )
        self.assertNotIn("safety_margin_ratio", env.memory_settings())
