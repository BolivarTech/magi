# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Unit tests for the S12 scenario's own shape."""

import pathlib
import shutil
import unittest
import uuid

from smoke.config import CERTIFICATE_PATH
from smoke.outcome import Outcome
from smoke.registry import DEFAULT_REGISTRY
from smoke.runner import Ambient, capture_tree
from smoke.scenarios import hygiene  # noqa: F401 - import registers it
from smoke.tests import support

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

#: What a self-protecting environment's own gitignore says. Everything under it
#: is hidden from git, which is the whole of assertion 2.
_IGNORE_EVERYTHING = "*\n"


def _ambient(snapshot) -> Ambient:
    """Wrap a tree snapshot in the ambient state a scenario receives.

    Args:
        snapshot: The pre-run snapshot, or None.

    Returns:
        Ambient: The state, with the fields S12 does not read left empty.
    """
    return Ambient(tree_snapshot=snapshot,
                   ceiling_fraction=None, memory_settings={})


def _outcomes(ambient: Ambient) -> dict[str, Outcome]:
    """Run S12 against *ambient* and index its outcomes by assertion.

    Args:
        ambient: The state captured before any run started.

    Returns:
        dict[str, Outcome]: What each assertion concluded.
    """
    findings = list(DEFAULT_REGISTRY.get("S12").func(None, ambient))
    return {finding.assertion: finding.outcome for finding in findings}


class HygieneScenarioTests(unittest.TestCase):
    """S12 is registered standalone, needs ambient state, and declares two."""

    def test_s12_is_registered_needing_ambient_state(self) -> None:
        entry = DEFAULT_REGISTRY.get("S12")
        self.assertIsNone(entry.run)
        self.assertFalse(entry.needs_backend)
        self.assertTrue(entry.needs_ambient)

    def test_the_assertion_texts_are_the_spec_texts(self) -> None:
        self.assertEqual(
            [
                "the run adds no entry to git status --porcelain beyond the "
                "certificate",
                "git status --ignored smoke/env/ shows the whole environment "
                "on the ignored side",
            ],
            list(hygiene.ASSERTIONS),
        )


class HygieneScenarioBodyTests(unittest.TestCase):
    """The diff, and what it deliberately does not claim."""

    def _probe_dir(self, ignore: str | None) -> pathlib.Path:
        """Create a throwaway directory inside the repository.

        Args:
            ignore: What to write into its own ``.gitignore``, or None to
                leave it visible to git.

        Returns:
            pathlib.Path: The directory, removed when the test ends.
        """
        directory = REPO_ROOT / ("s12_probe_%s" % uuid.uuid4().hex)
        directory.mkdir()
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        (directory / "payload.bin").write_bytes(b"probe")
        if ignore is not None:
            (directory / ".gitignore").write_text(ignore, encoding="utf-8")
        return directory

    def _probe_file(self) -> pathlib.Path:
        """Create a throwaway untracked file inside the repository.

        Returns:
            pathlib.Path: The file, removed when the test ends.
        """
        path = REPO_ROOT / ("s12_probe_%s.txt" % uuid.uuid4().hex)
        self.addCleanup(path.unlink, missing_ok=True)
        path.write_text("probe", encoding="utf-8")
        return path

    def test_without_a_snapshot_neither_assertion_is_answered(self) -> None:
        """The snapshot is taken by the runner before anything ran. Taking one
        inside S12 would already include whatever earlier scenarios wrote, so
        when there is none the honest answer is that nothing was measured.
        """
        support.install_fake_runs(self)
        self.assertEqual({Outcome.CANNOT_TEST},
                         set(_outcomes(_ambient(None)).values()))

    def test_a_dirty_entry_present_in_both_snapshots_passes(self) -> None:
        """The heart of the scenario. The harness runs mostly DURING
        development, where uncommitted changes are expected, so asserting a
        clean tree would report the developer's own state as a product defect.
        The invariant is measured by subtracting.
        """
        self._probe_file()
        support.install_fake_runs(self)
        outcomes = _outcomes(_ambient(capture_tree(REPO_ROOT)))
        self.assertEqual(Outcome.PASS, outcomes[hygiene.ASSERTIONS[0]])

    def test_an_entry_that_appeared_after_the_snapshot_fails(self) -> None:
        support.install_fake_runs(self)
        before = capture_tree(REPO_ROOT)
        appeared = self._probe_file()
        outcomes = _outcomes(_ambient(before))
        self.assertEqual(Outcome.FAIL, outcomes[hygiene.ASSERTIONS[0]])
        self.assertTrue(appeared.exists())

    def _borrow_certificate_path(self):
        """Take the real certificate path, and guarantee it comes back.

        This test writes to ``docs/test/smoke-certificate.md`` itself, because
        S12's subject is what GIT sees and a temporary path elsewhere is not
        the file git would report. Borrowing it is therefore necessary; not
        putting it back was a defect, and one that could only appear once the
        file existed.

        The first version handled the absent case -- unlink on cleanup -- and
        not the present one. Every certificate this harness had ever emitted
        was written by a run that then exited, so the tree never carried one
        while the suite ran, and the gap was invisible until a release put a
        real certificate there. Running the unit suite then replaced a
        released artifact with nine bytes, and the suite stayed green: the
        test that asserts the harness leaves no trace was the thing leaving
        one.

        Returns:
            pathlib.Path: The certificate path, restored to exactly its prior
            content -- or to not existing -- when the test ends.
        """
        certificate = REPO_ROOT / CERTIFICATE_PATH
        if certificate.exists():
            saved = certificate.read_bytes()
            self.addCleanup(certificate.write_bytes, saved)
        else:
            certificate.parent.mkdir(parents=True, exist_ok=True)
            self.addCleanup(certificate.unlink, True)
        return certificate

    def test_the_certificate_alone_is_allowed_to_appear(self) -> None:
        """It is the one file the run is supposed to leave behind."""
        support.install_fake_runs(self)
        before = capture_tree(REPO_ROOT)
        certificate = self._borrow_certificate_path()
        certificate.write_text("# probe\n", encoding="utf-8")
        outcomes = _outcomes(_ambient(before))
        self.assertEqual(Outcome.PASS, outcomes[hygiene.ASSERTIONS[0]])

    def test_an_existing_certificate_survives_this_module(self) -> None:
        """The guardian for the borrow, and it drives the real cleanup.

        Asserting inside the borrowing test cannot work: the restore runs in
        a cleanup, after every assertion in that test has already passed. So
        this one plants a sentinel, runs the borrowing test through unittest
        so its cleanups actually fire, and then reads the file.
        """
        certificate = REPO_ROOT / CERTIFICATE_PATH
        certificate.parent.mkdir(parents=True, exist_ok=True)
        existed = certificate.exists()
        saved = certificate.read_bytes() if existed else None
        self.addCleanup(certificate.write_bytes, saved) if existed else \
            self.addCleanup(certificate.unlink, True)

        sentinel = b"# a released certificate nobody may overwrite\n"
        certificate.write_bytes(sentinel)
        case = HygieneScenarioBodyTests(
            "test_the_certificate_alone_is_allowed_to_appear")
        result = case.run()
        self.assertTrue(result.wasSuccessful(),
                        "the borrowing test itself failed: %s" % result.errors)
        self.assertEqual(sentinel, certificate.read_bytes(),
                         "the suite overwrote a certificate it only borrowed")

    def test_an_environment_git_cannot_see_passes(self) -> None:
        directory = self._probe_dir(_IGNORE_EVERYTHING)
        support.install_fake_runs(self, env_root=directory)
        outcomes = _outcomes(_ambient(capture_tree(REPO_ROOT)))
        self.assertEqual(Outcome.PASS, outcomes[hygiene.ASSERTIONS[1]])

    def test_an_environment_git_reports_as_untracked_fails(self) -> None:
        """Assertion 2 is about what git SEES, not what is on disk: the
        environment goes on existing on purpose (REQ-S30), and the guarantee is
        that none of it can be committed by accident.
        """
        directory = self._probe_dir(None)
        support.install_fake_runs(self, env_root=directory)
        outcomes = _outcomes(_ambient(capture_tree(REPO_ROOT)))
        self.assertEqual(Outcome.FAIL, outcomes[hygiene.ASSERTIONS[1]])

    def test_a_missing_environment_cannot_be_judged(self) -> None:
        directory = REPO_ROOT / ("s12_absent_%s" % uuid.uuid4().hex)
        support.install_fake_runs(self, env_root=directory)
        outcomes = _outcomes(_ambient(capture_tree(REPO_ROOT)))
        self.assertEqual(Outcome.CANNOT_TEST, outcomes[hygiene.ASSERTIONS[1]])


if __name__ == "__main__":
    unittest.main()
