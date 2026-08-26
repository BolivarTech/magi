# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Tests for the CLI surface and the four exit codes."""

import contextlib
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from smoke import __main__ as main
from smoke.lock import RunLock
from smoke.preflight import BackendStatus
from smoke.product import ProductOutput
from smoke.runs import RunResult
from smoke.env import Growth
from smoke.errors import HarnessError, PreflightError
from smoke.registry import DECLARED_SCENARIO_COUNT, DEFAULT_REGISTRY
from smoke.__main__ import (
    EXIT_HARNESS,
    EXIT_NOT_PASSED,
    EXIT_OK,
    EXIT_PREFLIGHT,
    exit_code_for,
    parse_args,
)
from smoke.outcome import Outcome
from smoke.runner import StampedFinding

#: How long the child interpreter is given to import the entry point, in
#: seconds. It imports modules and touches no network, so anything past this is
#: hung rather than slow.
_IMPORT_TIMEOUT_S = 120


def _finding(outcome: Outcome) -> StampedFinding:
    return StampedFinding("S1", "it holds", outcome, "", None)


class ExitCodeTests(unittest.TestCase):
    """Each code means one thing, and 3 is never a verdict."""

    def test_all_pass_exits_zero(self) -> None:
        self.assertEqual(EXIT_OK, exit_code_for([_finding(Outcome.PASS)]))

    def test_out_of_scope_does_not_count_against_a_green_run(self) -> None:
        findings = [_finding(Outcome.PASS), _finding(Outcome.OUT_OF_SCOPE)]
        self.assertEqual(EXIT_OK, exit_code_for(findings))

    def test_a_fail_exits_one(self) -> None:
        self.assertEqual(EXIT_NOT_PASSED, exit_code_for([_finding(Outcome.FAIL)]))

    def test_a_cannot_test_also_exits_one(self) -> None:
        self.assertEqual(
            EXIT_NOT_PASSED, exit_code_for([_finding(Outcome.CANNOT_TEST)])
        )

    def test_the_four_codes_are_distinct(self) -> None:
        self.assertEqual(
            4, len({EXIT_OK, EXIT_NOT_PASSED, EXIT_PREFLIGHT, EXIT_HARNESS})
        )


class ArgumentTests(unittest.TestCase):
    """No --defaults flag exists; absent --profile is the certifying mode."""

    def test_absent_profile_is_none(self) -> None:
        self.assertIsNone(parse_args(["--smoke-2"]).profile)

    def test_profile_is_captured(self) -> None:
        self.assertEqual("smoke/smoke.toml", parse_args(["--smoke-1", "--profile", "smoke/smoke.toml"]).profile)

    def test_smoke_1_and_smoke_2_are_mutually_exclusive(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--smoke-1", "--smoke-2"])

    def test_init_env_and_reset_env_are_mutually_exclusive(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--init-env", "--reset-env"])


class AmbientOrderingTests(unittest.TestCase):
    """The environment's [memory] table is read AFTER the preflight writes it.

    Step 7b is what normalises ``magi.toml``. Reading the table before it
    means S9's saturation ceiling derives from the PREVIOUS run's file, or
    from nothing at all on a first run where the file does not exist yet --
    and then 3b is CANNOT_TEST, assertion 3 degrades with it, and no
    certificate can be emitted on a fresh environment. The tree snapshot has
    the opposite requirement and must still be taken before.

    Checked against the source: the two reads have to sit on opposite sides of
    the preflight call, and nothing else in this module can say which side.
    """

    def test_the_memory_table_is_read_after_the_preflight_runs(self) -> None:
        text = pathlib.Path(main.__file__).read_text(encoding="utf-8")
        preflight = text.index("Preflight(config, env, binary")
        settings = text.index("env.memory_settings()")
        snapshot = text.index("capture_tree(REPO_ROOT)")
        self.assertLess(snapshot, preflight,
                        "the tree snapshot must precede the preflight")
        self.assertGreater(settings, preflight,
                           "the memory table must be read after step 7b "
                           "writes magi.toml")

class RegistrationOnTheProductPathTests(unittest.TestCase):
    """The scenarios have to be registered on the path the OPERATOR takes."""

    def test_running_the_module_registers_the_scenarios(self) -> None:
        """A SUBPROCESS, and that is the whole value of this test.

        In this process every scenario module has already been imported by the
        scenario tests, so ``DEFAULT_REGISTRY`` is populated no matter what
        ``smoke.__main__`` does. Only a fresh interpreter that imports nothing
        but the entry point can answer whether the operator's ``python -m
        smoke`` sees any scenario at all.

        Without this the harness ran to completion, evaluated NOTHING, and
        exited 0 -- a green run over an empty registry, which is the exact
        shape of guardian this harness exists to refuse. The reconciliation
        cannot catch it: with nothing registered, registered, invoked and
        reported are all empty and every set difference is empty too.
        """
        result = subprocess.run(
            [sys.executable, "-c",
             "import smoke.__main__;"
             "from smoke.registry import DEFAULT_REGISTRY;"
             "print(len(DEFAULT_REGISTRY.registered_ids()))"],
            cwd=str(pathlib.Path(__file__).resolve().parent.parent.parent),
            capture_output=True, text=True, timeout=_IMPORT_TIMEOUT_S,
            check=True,
        )
        self.assertGreater(int(result.stdout.strip()), 0,
                           "python -m smoke would evaluate nothing and exit 0")


class DeclaredCountAtRuntimeTests(unittest.TestCase):
    """The registry is checked against the declared count BEFORE anything runs.

    A unit test already guards the two against drifting apart, and a test is
    the wrong place for the whole guarantee: it catches the change that is made
    with the suite watching, not the deployment where a module quietly failed
    to import. The count is what the certificate publishes as "N of N", so the
    harness verifies it about ITSELF, at startup, and a mismatch is exit 3 --
    a defect in the harness, never a verdict on the product.
    """

    def test_a_registry_short_of_the_declared_count_is_refused(self) -> None:
        with self.assertRaises(HarnessError) as caught:
            main.require_declared_count(18)
        self.assertIn("18", str(caught.exception))
        self.assertIn(str(DECLARED_SCENARIO_COUNT), str(caught.exception))

    def test_the_real_registry_matches(self) -> None:
        main.require_declared_count(len(DEFAULT_REGISTRY.registered_ids()))


#: What the patched preflight answers: a reachable backend, so the run
#: proceeds to the executor where the lock is actually checked.
_REACHABLE = BackendStatus(reachable=True, cause="")


class LockLifetimeTests(unittest.TestCase):
    """The lock is held while the RUNS execute, not only while the preflight does.

    ``Preflight.run`` acquires it and nothing released it, which read as "held
    for the process". It was not: ``main`` passed ``RunLock(LOCK_PATH)`` as a
    temporary, so the object was collected at the end of that statement, its
    file handle finalized, and both ``flock`` and ``msvcrt.locking`` release on
    close. The eight product runs -- the only part that mutates the shared
    persistent environment -- then ran unlocked, which is the exact race
    ``smoke/lock.py`` says produces a false verdict rather than an error.
    """

    def test_a_second_run_cannot_start_while_the_runs_execute(self) -> None:
        root = pathlib.Path(tempfile.mkdtemp())
        held: list[bool] = []

        def executing(self, definition):
            """Stand in for one product run, and try to take the lock."""
            try:
                RunLock(root / ".lock").acquire()
                held.append(False)
            except PreflightError:
                held.append(True)
            return RunResult(
                run_id=definition.run_id,
                output=ProductOutput(stdout=b"{}", stderr=b"", exit_code=0,
                                     command=["magi-rs"]),
                duration_s=0.0, timed_out=False, planted=())

        def preflight(self, certifying, profile):
            """Stand in for the preflight: take the lock, as step 2 does."""
            self.lock.acquire()
            return BackendStatus(reachable=True, cause="")

        environment = mock.Mock()
        environment.return_value.growth.return_value = Growth(
            db_bytes=0, runs_bytes=0, active_memories=None)

        patches = [
            mock.patch.object(main, "LOCK_PATH", root / ".lock"),
            mock.patch.object(main, "ENV_ROOT", root / "env"),
            mock.patch.object(main, "CONFIG_PATH", root / "smoke.toml"),
            mock.patch.object(main.SmokeConfig, "load"),
            mock.patch.object(main, "Environment", environment),
            mock.patch.object(main, "ReleaseBinary"),
            mock.patch.object(main, "capture_tree"),
            mock.patch.object(main.runs, "configure"),
            mock.patch.object(main, "require_declared_count"),
            mock.patch.object(main, "needed_runs", return_value=["R1"]),
            mock.patch.object(main, "DEFINITIONS", {"R1": mock.Mock()}),
            mock.patch.object(main.RunExecutor, "execute", executing),
            mock.patch.object(main.RunExecutor, "archive"),
            mock.patch.object(main.Preflight, "run", preflight),
            mock.patch.object(main.Runner, "run", return_value=[]),
        ]
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            main.main(["--smoke-1"])

        self.assertEqual([True], held,
                         "the lock must still be held while a run executes")

    def test_a_second_run_cannot_start_while_the_certificate_is_written(self):
        """The same race, moved later rather than closed.

        Freeing the lock in a ``finally`` around the runs leaves everything
        after it unlocked, and what comes after is not bookkeeping: the growth
        figures the certificate publishes, ``RoundCounter``'s bump and reset --
        which WRITE inside the environment -- the binary's digest, and the
        document itself. A second run may take the lock the instant the
        ``finally`` executes, and ``--reset-env`` would then rmtree the
        directory the counter is about to write into, while the first run is
        still deciding what its certificate says.
        """
        root = pathlib.Path(tempfile.mkdtemp())
        held: list[bool] = []

        def certifying(*args, **kwargs):
            """Stand in for the certificate, and try to take the lock."""
            try:
                RunLock(root / ".lock").acquire()
                held.append(False)
            except PreflightError:
                held.append(True)

        def preflight(self, certifying_run, profile):
            """Stand in for the preflight: take the lock, as step 2 does."""
            self.lock.acquire()
            return _REACHABLE

        environment = mock.Mock()
        environment.return_value.growth.return_value = Growth(
            db_bytes=0, runs_bytes=0, active_memories=None)

        patches = [
            mock.patch.object(main, "LOCK_PATH", root / ".lock"),
            mock.patch.object(main, "ENV_ROOT", root / "env"),
            mock.patch.object(main, "CONFIG_PATH", root / "smoke.toml"),
            mock.patch.object(main.SmokeConfig, "load"),
            mock.patch.object(main, "Environment", environment),
            mock.patch.object(main, "ReleaseBinary"),
            mock.patch.object(main, "capture_tree"),
            mock.patch.object(main.runs, "configure"),
            mock.patch.object(main, "require_declared_count"),
            mock.patch.object(main, "needed_runs", return_value=[]),
            mock.patch.object(main, "DEFINITIONS", {}),
            mock.patch.object(main.Preflight, "run", preflight),
            mock.patch.object(main.Runner, "run", return_value=[]),
            mock.patch.object(main, "certify", certifying),
        ]
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            main.main(["--smoke-2"])

        self.assertEqual([True], held,
                         "the lock must still be held while the certificate "
                         "is decided and written")


if __name__ == "__main__":
    unittest.main()
