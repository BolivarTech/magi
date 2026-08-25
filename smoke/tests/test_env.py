# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Tests for the test environment's lifecycle."""

import pathlib
import tempfile
import unittest

from smoke.env import Environment
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
