# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Unit tests for the S2 scenario's own shape."""

import pathlib
import subprocess
import unittest
from unittest import mock

from smoke.outcome import Outcome
from smoke.product import ProductOutput
from smoke.registry import DEFAULT_REGISTRY
from smoke.scenarios import workspace  # noqa: F401 - import registers it
from smoke.tests import support


class WorkspaceScenarioTests(unittest.TestCase):
    """S2 is registered standalone and declares its four assertions."""

    def test_s2_is_registered_without_a_run(self) -> None:
        entry = DEFAULT_REGISTRY.get("S2")
        self.assertIsNone(entry.run)
        self.assertFalse(entry.needs_backend)

    def test_the_assertion_texts_are_the_spec_texts(self) -> None:
        self.assertEqual(
            [
                "magi init creates .magi/ in an empty directory",
                "the permissions are restrictive — POSIX bits on Unix, ACL on Windows",
                "a second init refuses and leaves the directory unchanged",
                "query from a nested subdirectory finds the ancestor .magi/",
            ],
            list(workspace.ASSERTIONS),
        )


class WorkspaceScenarioBodyTests(unittest.TestCase):
    """Every assertion is reported, whatever the product did."""

    def test_a_product_that_does_nothing_still_reports_all_four(self) -> None:
        """The reconciliation of section 2.4 calls silence a harness failure,
        so a scenario that gives up half way is a defect even when the product
        is the thing that failed. Against a binary that scaffolds nothing,
        every assertion must still come back -- and none of them as PASS.
        """
        support.install_fake_runs(self)
        entry = DEFAULT_REGISTRY.get("S2")
        findings = list(entry.func(None))
        self.assertEqual(list(workspace.ASSERTIONS),
                         [finding.assertion for finding in findings])
        self.assertNotIn(Outcome.PASS,
                         {finding.outcome for finding in findings})

    def test_the_seed_runs_init_in_the_scratch_area_by_cwd(self) -> None:
        """The precondition may not use ``-w``: that is the flag S14 tests, and
        a seed that used it could not fail when the flag stopped working.
        """
        binary = support.install_fake_runs(self)
        list(DEFAULT_REGISTRY.get("S2").func(None))
        seeds = [call for call in binary.calls if call.args[:1] == ("init",)]
        self.assertTrue(seeds, "S2 never ran the product's init")
        for call in seeds:
            self.assertIsNotNone(call.cwd, "the seed must name a cwd")
            self.assertNotIn("-w", call.args)
            self.assertNotIn("--workdir", call.args)


class _FakeWorkdir:
    """A product double whose ``-w`` behaves, or deliberately does not.

    Attributes:
        honors_flag: Whether ``-w`` is read at all. With it False the double
            resolves every invocation against the process's cwd, which is the
            silent degradation the release profile can no longer catch with a
            ``debug_assert!`` -- and the whole reason S14 exists.
        innermost_wins: Whether the LAST ``-w`` wins when it is given twice.
        init_rejects_repeats: Whether ``init`` refuses a repeated ``-w`` the
            way clap does for a non-global argument.
    """

    def __init__(self, honors_flag: bool = True, innermost_wins: bool = True,
                 init_rejects_repeats: bool = True) -> None:
        """Create the double.

        Args:
            honors_flag: Read ``-w`` rather than always using the cwd.
            innermost_wins: Let the last ``-w`` win over the first.
            init_rejects_repeats: Make a repeated ``-w`` a parse error.
        """
        self.honors_flag = honors_flag
        self.innermost_wins = innermost_wins
        self.init_rejects_repeats = init_rejects_repeats
        self.entries: dict[str, set[str]] = {}

    def __call__(self, call: support.Call) -> ProductOutput | None:
        """Answer one invocation.

        Args:
            call: What the fake binary was asked to run.

        Returns:
            ProductOutput: The canned answer, or None for the failed default.
        """
        args = list(call.args)
        flags = [args[index + 1] for index, token in enumerate(args)
                 if token == "-w" and index + 1 < len(args)]
        if args[:1] == ["init"]:
            return self._init(call, flags)
        if args[:1] == ["vault"]:
            return self._vault(args, self._resolve(call, flags))
        return None

    def _resolve(self, call: support.Call, flags: list[str]) -> str:
        """Decide which directory an invocation acts on.

        Args:
            call: The invocation.
            flags: Every ``-w`` value it carried, in order.

        Returns:
            str: The resolved directory.
        """
        if not self.honors_flag or not flags:
            return str(call.cwd)
        return flags[-1] if self.innermost_wins else flags[0]

    def _init(self, call: support.Call,
              flags: list[str]) -> ProductOutput:
        """Scaffold, or refuse a repeated flag the way clap does.

        Args:
            call: The invocation.
            flags: Every ``-w`` value it carried.

        Returns:
            ProductOutput: The capture.
        """
        if len(flags) > 1 and self.init_rejects_repeats:
            return _capture(
                b"",
                b"error: the argument '--workdir <WORKDIR>' cannot be used "
                b"multiple times\n\nUsage: magi-rs init [OPTIONS]\n",
                2,
            )
        target = pathlib.Path(self._resolve(call, flags))
        magi = target / ".magi"
        if magi.exists():
            return _capture(b"", b"error: .magi/ already exists\n", 1)
        magi.mkdir(parents=True)
        (magi / "magi.toml").write_text("# generated\n", encoding="utf-8")
        return _capture(b"", b"", 0)

    def _vault(self, args: list[str], resolved: str) -> ProductOutput:
        """Answer ``vault ls`` or ``vault set`` for one resolved directory.

        Args:
            args: The full argv.
            resolved: The directory the invocation acts on.

        Returns:
            ProductOutput: The capture.
        """
        if not (pathlib.Path(resolved) / ".magi").is_dir():
            return _capture(
                b"", b"error: no .magi/ state directory found in this "
                     b"directory or any parent\n", 1)
        held = self.entries.setdefault(resolved, set())
        if "set" in args:
            held.add(args[args.index("set") + 1])
            return _capture(b"stored\n", b"", 0)
        if not held:
            return _capture(b"(vault empty)\n", b"", 0)
        listed = "\n".join("%s · t · t" % name for name in sorted(held))
        return _capture(listed.encode("utf-8") + b"\n", b"", 0)


def _capture(stdout: bytes, stderr: bytes, exit_code: int) -> ProductOutput:
    """Build a canned capture.

    Args:
        stdout: What the product printed.
        stderr: What it wrote to the error stream.
        exit_code: The code it exited with.

    Returns:
        ProductOutput: The capture.
    """
    return ProductOutput(stdout=stdout, stderr=stderr, exit_code=exit_code,
                         command=["magi-rs"])


def _workdir_outcomes() -> dict[str, Outcome]:
    """Run S14 against whatever double is installed, indexed by assertion.

    Returns:
        dict[str, Outcome]: What each assertion concluded.
    """
    findings = list(DEFAULT_REGISTRY.get("S14").func(None))
    return {finding.assertion: finding.outcome for finding in findings}


class WorkdirFlagScenarioTests(unittest.TestCase):
    """S14 is registered standalone and declares its four assertions."""

    def test_s14_is_registered_without_a_run(self) -> None:
        entry = DEFAULT_REGISTRY.get("S14")
        self.assertIsNone(entry.run)
        self.assertFalse(entry.needs_backend)

    def test_the_assertion_texts_are_the_spec_texts(self) -> None:
        self.assertEqual(
            [
                "init -w <dir> scaffolds into <dir> and leaves the current "
                "directory untouched",
                "vault -w <dir> ls and vault ls -w <dir> both parse",
                "given twice, the innermost wins",
                "on init it is not global, so repeating it is a clap error",
            ],
            list(workspace.S14_ASSERTIONS),
        )


class WorkdirFlagScenarioBodyTests(unittest.TestCase):
    """The scenario with the most ways to guard nothing."""

    def test_a_product_that_does_nothing_still_reports_all_four(self) -> None:
        support.install_fake_runs(self)
        findings = list(DEFAULT_REGISTRY.get("S14").func(None))
        self.assertEqual(list(workspace.S14_ASSERTIONS),
                         [finding.assertion for finding in findings])
        self.assertNotIn(Outcome.PASS,
                         {finding.outcome for finding in findings})

    def test_a_product_that_honors_the_flag_passes_every_assertion(self) -> None:
        support.install_fake_runs(self, responder=_FakeWorkdir())
        self.assertEqual({Outcome.PASS}, set(_workdir_outcomes().values()))

    def test_a_failed_control_invocation_cannot_test_the_third(self) -> None:
        """The control's exit code was never read.

        Assertion 3 concludes from "the marker is absent from the outer
        listing". An outer invocation that exited non-zero with empty stdout
        satisfies that too, so the assertion passed on half a measurement.
        """
        class _OuterFails(_FakeWorkdir):
            def __call__(self, call):
                answer = super().__call__(call)
                flags = [call.args[i + 1] for i, a in enumerate(call.args)
                         if a == "-w"]
                if answer is not None and len(flags) == 2:
                    return ProductOutput(stdout=b"", stderr=b"boom",
                                         exit_code=1,
                                         command=["magi-rs"] + list(call.args))
                return answer

        support.install_fake_runs(self, responder=_OuterFails())
        self.assertEqual(Outcome.CANNOT_TEST,
                         _workdir_outcomes()[workspace.S14_ASSERTIONS[2]])

    def test_a_resolver_that_ignores_the_flag_fails(self) -> None:
        """The mutation, run rather than described -- at the level of the
        double. With ``-w`` ignored the product acts on the process's cwd, and
        the assertions that depend on the flag must go red. A precondition that
        used the flag could not: the seed would land in the current directory
        and the assertion would then resolve that same directory, find the
        workspace it expected, and pass.
        """
        support.install_fake_runs(self, responder=_FakeWorkdir(honors_flag=False))
        outcomes = _workdir_outcomes()
        self.assertEqual(Outcome.FAIL, outcomes[workspace.S14_ASSERTIONS[0]])
        self.assertEqual(Outcome.FAIL, outcomes[workspace.S14_ASSERTIONS[2]])

    def test_an_outermost_wins_resolver_fails(self) -> None:
        support.install_fake_runs(self,
                                  responder=_FakeWorkdir(innermost_wins=False))
        outcomes = _workdir_outcomes()
        self.assertEqual(Outcome.FAIL, outcomes[workspace.S14_ASSERTIONS[2]])

    def test_an_init_that_accepts_a_repeated_flag_fails(self) -> None:
        support.install_fake_runs(
            self, responder=_FakeWorkdir(init_rejects_repeats=False))
        outcomes = _workdir_outcomes()
        self.assertEqual(Outcome.FAIL, outcomes[workspace.S14_ASSERTIONS[3]])

    def test_every_seed_uses_the_cwd_and_never_the_flag(self) -> None:
        """The precondition may not use the mechanism under test. Reading this
        rule is not enough to enforce it, so it is asserted.
        """
        binary = support.install_fake_runs(self, responder=_FakeWorkdir())
        list(DEFAULT_REGISTRY.get("S14").func(None))
        seeds = [call for call in binary.calls
                 if call.args[:1] == ("init",) and "-w" not in call.args]
        self.assertTrue(seeds, "S14 seeded no workspace by cwd")
        for call in seeds:
            self.assertIsNotNone(call.cwd)


class WindowsPrincipalCountTests(unittest.TestCase):
    """The question is HOW MANY ACCOUNTS, not how many entries."""

    _TARGET = "C:\\ws\\.magi"
    _OWNER_TWICE = (
        "C:\\ws\\.magi THOTH\\jbolivarg:(OI)(CI)(F)\n"
        "                THOTH\\jbolivarg:(F)\n"
    )
    _TWO_ACCOUNTS = (
        "C:\\ws\\.magi THOTH\\jbolivarg:(OI)(CI)(F)\n"
        "                THOTH\\someone-else:(R)\n"
    )
    #: ONE account, and it is the broad one. Pairing Everyone with the owner
    #: would make the count check fire too, so dropping the broad check
    #: would still FAIL and the test would guard nothing.
    _BROAD = "C:\\ws\\.magi Everyone:(F)\n"

    def _verdict(self, listing):
        """Run the permission finding against a canned icacls listing.

        Args:
            listing: What icacls prints.

        Returns:
            Finding: What the scenario concluded.
        """
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=listing.encode("utf-8"), stderr=b"")
        with mock.patch.object(workspace.subprocess, "run",
                               return_value=completed):
            return workspace._windows_permission_finding(
                pathlib.Path(self._TARGET))

    def test_two_entries_for_the_same_account_pass(self) -> None:
        """icacls lists a principal once per access entry, and the product's
        own restriction leaves the owner with two -- one inheritable for what
        the directory contains, one for the directory itself. Counting entries
        reported "2 accounts (THOTH\\jbolivarg, THOTH\\jbolivarg)": the
        same name twice, widening access to nobody, and the assertion could
        never pass on a correctly restricted workspace.
        """
        self.assertEqual(Outcome.PASS, self._verdict(self._OWNER_TWICE).outcome)

    def test_a_second_real_account_still_fails(self) -> None:
        """The half that must not be lost. Another individual account carries
        no well-known name to match, so counting distinct principals is the
        only thing that sees it.
        """
        self.assertEqual(Outcome.FAIL, self._verdict(self._TWO_ACCOUNTS).outcome)

    def test_a_broad_principal_still_fails(self) -> None:
        self.assertEqual(Outcome.FAIL, self._verdict(self._BROAD).outcome)


if __name__ == "__main__":
    unittest.main()
