# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Tests for scenario invocation, id stamping and reconciliation."""

import dataclasses
import pathlib
import subprocess
import tempfile
import unittest

from smoke.errors import HarnessError, ProductOutputError
from smoke.outcome import Finding, Outcome
from smoke.registry import Registry, scenario
from smoke.runner import Ambient, Runner, capture_tree

#: No scenario under test reads ambient state, so every case below passes the
#: same empty one. S12 is the only scenario that declares needs_ambient, and it
#: is tested in its own module against a real snapshot.
_NO_AMBIENT = Ambient(tree_snapshot=None,
                      ceiling_fraction=None, memory_settings={})


@dataclasses.dataclass(frozen=True)
class _TimedOut:
    """The minimum a RunResult needs to be for the Runner's timeout branch.

    A double rather than the real ``RunResult``: that type belongs to Task 21,
    and this test file is written in Task 3.
    """

    run_id: str
    timed_out: bool = True


class StampingTests(unittest.TestCase):
    """The runner attaches the id, so a scenario cannot mislabel itself."""

    def setUp(self) -> None:
        self.registry = Registry()

    def test_runner_stamps_the_id_it_invoked(self) -> None:
        @scenario("S2", registry=self.registry)
        def workspace(run):
            yield Finding("it holds", Outcome.PASS, "", None)

        findings = Runner(self.registry, {}, backend_reachable=True,
                       ambient=_NO_AMBIENT).run()
        self.assertEqual(["S2"], [f.scenario for f in findings])

    def test_findings_keep_assertion_outcome_and_detail(self) -> None:
        @scenario("S2", registry=self.registry)
        def workspace(run):
            yield Finding("it holds", Outcome.FAIL, "because reasons", None)

        finding = Runner(self.registry, {}, backend_reachable=True,
                       ambient=_NO_AMBIENT).run()[0]
        self.assertEqual("it holds", finding.assertion)
        self.assertEqual(Outcome.FAIL, finding.outcome)
        self.assertEqual("because reasons", finding.detail)


class TreeSnapshotTests(unittest.TestCase):
    """capture_tree is what makes S12's claim checkable, so it is checked."""

    def test_it_records_a_hash_per_file_git_reports(self) -> None:
        """A path->hash map, not one hash of everything: S12 has to name WHICH
        file appeared, and a diff nobody can read is a red nobody acts on.
        """
        root = pathlib.Path(tempfile.mkdtemp())
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        (root / "kept.txt").write_text("one", encoding="utf-8")
        snapshot = capture_tree(root)
        self.assertIn("kept.txt", snapshot.entries)
        self.assertEqual(64, len(snapshot.entries["kept.txt"]))

    def test_a_changed_file_changes_its_hash(self) -> None:
        """Without this the snapshot could return constant hashes and S12 would
        pass over a modified tree.
        """
        root = pathlib.Path(tempfile.mkdtemp())
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        target = root / "kept.txt"
        target.write_text("one", encoding="utf-8")
        before = capture_tree(root).entries["kept.txt"]
        target.write_text("two", encoding="utf-8")
        self.assertNotEqual(before, capture_tree(root).entries["kept.txt"])


class ReconciliationTests(unittest.TestCase):
    """The two ways a scenario disappears get two different messages."""

    def setUp(self) -> None:
        self.registry = Registry()

    def test_invoked_but_silent_is_a_harness_error(self) -> None:
        @scenario("S2", registry=self.registry)
        def silent(run):
            yield from ()

        with self.assertRaises(HarnessError) as caught:
            Runner(self.registry, {}, backend_reachable=True,
                       ambient=_NO_AMBIENT).run()
        self.assertIn("invoked but reported nothing", str(caught.exception))

    def test_registered_but_never_invoked_is_a_different_message(self) -> None:
        runner = Runner(self.registry, {}, backend_reachable=True,
                       ambient=_NO_AMBIENT)
        with self.assertRaises(HarnessError) as caught:
            runner.reconcile(registered={"S2", "S3"}, invoked={"S2"}, reported={"S2"})
        self.assertIn("registered but never invoked", str(caught.exception))

    def test_a_clean_run_reconciles_without_raising(self) -> None:
        @scenario("S2", registry=self.registry)
        def speaks(run):
            yield Finding("it holds", Outcome.PASS, "", None)

        Runner(self.registry, {}, backend_reachable=True,
                       ambient=_NO_AMBIENT).run()


class ProductOutputBoundaryTests(unittest.TestCase):
    """A malformed product output is the product's defect, not the harness's."""

    def setUp(self) -> None:
        self.registry = Registry()

    def test_product_output_error_becomes_a_fail_not_a_crash(self) -> None:
        @scenario("S1", registry=self.registry)
        def contract(run):
            raise ProductOutputError("stdout is not JSON")
            yield  # pragma: no cover - makes the function a generator

        findings = Runner(self.registry, {}, backend_reachable=True,
                       ambient=_NO_AMBIENT).run()
        self.assertEqual([Outcome.FAIL], [f.outcome for f in findings])
        self.assertIn("stdout is not JSON", findings[0].detail)

    def _run_with(self, result: _TimedOut, inspects: bool) -> list:
        """Register one scenario against ``result``'s run and invoke it.

        Args:
            result: The timed-out double the Runner will read.
            inspects: Whether the scenario declares ``inspects_timeouts``.

        Returns:
            The findings the run produced.
        """
        self.invoked: list[str] = []

        @scenario("S7", run=result.run_id, inspects_timeouts=inspects,
                  registry=self.registry)
        def hung(run):
            self.invoked.append("S7")
            yield Finding(assertion="read applied_caps", outcome=Outcome.PASS,
                          detail="", run_id=result.run_id)

        return Runner(self.registry, {result.run_id: result},
                      backend_reachable=True, ambient=_NO_AMBIENT).run()

    def test_a_hung_run_is_cannot_test_for_a_scenario_that_does_not_inspect(self) -> None:
        """SS5.1: a timeout is never FAIL by default. A slow provider is not a
        product defect, and calling it one puts the gate red on someone else's
        load -- the intermittent red that gets rationalised away.
        """
        findings = self._run_with(_TimedOut("R4"), inspects=False)
        self.assertEqual([Outcome.CANNOT_TEST], [f.outcome for f in findings])
        self.assertEqual([], self.invoked)

    def test_a_scenario_that_inspects_timeouts_receives_the_partial_result(self) -> None:
        """S7 is the one exception SS5.1 grants, and it is granted on evidence:
        it reads applied_caps out of what the run emitted before hanging. The
        Runner must therefore hand it the timed-out result instead of deciding
        for it. Drop the inspects_timeouts branch and this goes red, and S7
        stops being able to detect failure #4 at all.
        """
        findings = self._run_with(_TimedOut("R4"), inspects=True)
        self.assertEqual(["S7"], self.invoked)
        self.assertEqual([Outcome.PASS], [f.outcome for f in findings])

    def test_a_scenario_needing_a_down_backend_is_cannot_test_not_fail(self) -> None:
        """D-17: a backend that did not answer is not a product defect.

        FAIL would accuse the product of something the harness never observed,
        and PASS would certify over an untested scenario. CANNOT_TEST is the
        only honest answer, and it still blocks the gate. Delete the branch in
        ``_invoke`` and this goes red -- the scenario body would run and yield
        whatever it yields against a backend that is not there.
        """
        invoked: list[str] = []

        @scenario("S7", needs_backend=True, registry=self.registry)
        def needs_it(run):
            invoked.append("S7")
            yield Finding(assertion="never reached", outcome=Outcome.PASS,
                          detail="", run_id=None)

        findings = Runner(self.registry, {}, backend_reachable=False,
                       ambient=_NO_AMBIENT).run()
        self.assertEqual([Outcome.CANNOT_TEST], [f.outcome for f in findings])
        self.assertEqual([], invoked)

    def test_a_scenario_not_needing_the_backend_runs_while_it_is_down(self) -> None:
        """The gate is on ``needs_backend``, not on the backend. Widening it to
        skip everything would turn a down backend into a blank report."""
        @scenario("S2", registry=self.registry)
        def offline(run):
            yield Finding(assertion="runs offline", outcome=Outcome.PASS,
                          detail="", run_id=None)

        findings = Runner(self.registry, {}, backend_reachable=False,
                       ambient=_NO_AMBIENT).run()
        self.assertEqual([Outcome.PASS], [f.outcome for f in findings])

    def test_a_harness_bug_still_propagates(self) -> None:
        @scenario("S1", registry=self.registry)
        def buggy(run):
            raise ZeroDivisionError("harness bug")
            yield  # pragma: no cover

        with self.assertRaises(ZeroDivisionError):
            Runner(self.registry, {}, backend_reachable=True,
                       ambient=_NO_AMBIENT).run()


if __name__ == "__main__":
    unittest.main()
