# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Tests for the scoped-test selector's path classification.

The property under test is not "which filter comes out" but "which paths are
allowed to widen". A selector that widens on everything still reports success,
which is the failure mode ``scoped_tests.py`` is written to avoid, so the cases
that must NOT widen are asserted individually.
"""

import pathlib
import sys
import subprocess
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import scoped_tests  # noqa: E402 - the path insert above has to run first


def _repo_with_a_commit() -> pathlib.Path:
    """A git repository with one commit, so ``git diff HEAD`` has a HEAD.

    Returns:
        pathlib.Path: The repository root.
    """
    root = pathlib.Path(tempfile.mkdtemp())
    run = lambda *args: subprocess.run(args, cwd=str(root), check=True,
                                       capture_output=True)
    run("git", "init", "-q")
    run("git", "config", "user.email", "harness@example.invalid")
    run("git", "config", "user.name", "harness")
    (root / "seed").write_text("seed", encoding="utf-8")
    run("git", "add", "seed")
    run("git", "commit", "-q", "-m", "seed")
    return root


class HarnessCategoryTests(unittest.TestCase):
    """``smoke/`` is classified, not unknown, and not conflated with 'no diff'."""

    def test_a_harness_source_file_is_its_own_category(self) -> None:
        """Not 'full': cargo does not compile it, so no Rust test can break.

        Not ``None`` either -- that routes to the full suite as well, which is
        the whole cost this category removes.
        """
        self.assertEqual("harness", scoped_tests.module_filter("smoke/runner.py"))

    def test_a_harness_toml_is_the_same_category(self) -> None:
        """``.toml`` is in DOC_SUFFIXES, so without an explicit harness check
        this path returns None and widens to the full suite.
        """
        self.assertEqual("harness", scoped_tests.module_filter("smoke/smoke.toml"))

    def test_a_harness_only_diff_does_not_widen(self) -> None:
        expression, reason = scoped_tests.build_filter(
            ["smoke/runner.py", "smoke/tests/test_runner.py"], None)
        self.assertEqual("harness", expression)
        self.assertNotIn("full", reason)

    def test_the_harness_message_never_claims_the_files_are_gitignored(self) -> None:
        """``smoke/`` is TRACKED. Reusing the empty-diff message would tell the
        reader the opposite of the truth about their own working tree.
        """
        _, reason = scoped_tests.build_filter(["smoke/runner.py"], None)
        self.assertNotIn("ignored", reason.lower())
        self.assertNotIn("invisible", reason.lower())

    def test_rust_alongside_the_harness_still_selects_the_rust_tests(self) -> None:
        """The harness contributes no filter; it must not erase one either."""
        expression, _ = scoped_tests.build_filter(
            ["smoke/runner.py", "src/agent/mod.rs"], None)
        self.assertIn("agent", expression)
        self.assertNotEqual("harness", expression)

    def test_a_new_untracked_file_is_a_change(self) -> None:
        """``git diff`` does not list what git has never been told about.

        A brand-new source file that has not been staged is invisible to
        ``git diff --name-only HEAD``, so the script answered "git reports no
        change -- nothing to verify, so nothing is run" and ran no tests at
        all, on the one commit that adds a file. The empty-diff shortcut is
        the script's single exception to widening, and it has to mean "there
        is no change", not "git was not looking".
        """
        root = _repo_with_a_commit()
        (root / "src").mkdir()
        (root / "src" / "brand_new.rs").write_text("fn main() {}\n",
                                                   encoding="utf-8")
        listed = scoped_tests.changed_files(None, cwd=str(root))
        self.assertIn("src/brand_new.rs", listed)

    def test_an_ignored_new_file_is_still_not_a_change(self) -> None:
        """Widening on an ignored path would run the whole suite for a log."""
        root = _repo_with_a_commit()
        (root / ".gitignore").write_text("noise.log\n", encoding="utf-8")
        (root / "noise.log").write_text("x\n", encoding="utf-8")
        listed = scoped_tests.changed_files(None, cwd=str(root))
        self.assertNotIn("noise.log", listed)

    def test_a_script_change_selects_the_scripts_own_tests(self) -> None:
        """These nine tests were run by nothing.

        ``module_filter`` answered "full" for this very file, so editing it ran
        the 400-600 s Rust suite and never the Python tests it contains, and
        nothing else in the repository referenced it. A test file no gate runs
        is a test file that stops being true without anyone noticing.
        """
        self.assertEqual("harness",
                         scoped_tests.module_filter("scripts/test_scoped_tests.py"))

    def test_an_empty_diff_is_still_its_own_answer(self) -> None:
        """The pre-existing contract must not regress: no diff means nothing to
        verify, which is different from 'only the harness changed'.
        """
        expression, _ = scoped_tests.build_filter([], None)
        self.assertEqual("none", expression)

    def test_an_unknown_path_still_widens(self) -> None:
        """The 'when in doubt, widen' rule is what this change must not weaken:
        ``smoke/`` stops being unknown; nothing else does.
        """
        self.assertEqual("full", scoped_tests.module_filter("unmapped/thing.py"))


if __name__ == "__main__":
    unittest.main()
