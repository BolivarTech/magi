# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Unit tests for the S3 scenario's own shape."""

import unittest

from smoke.outcome import Outcome
from smoke.product import ProductOutput
from smoke.registry import DEFAULT_REGISTRY
from smoke.scenarios import vault  # noqa: F401 - import registers it
from smoke.tests import support

#: What a well-behaved ``vault ls`` prints: name, created, updated.
_LS_LINE = "%s · 2026-08-25T10:00:00Z · 2026-08-25T10:00:00Z"


class _FakeVault:
    """A product double whose vault behaves, so S3 can be seen to PASS.

    A scenario that only ever reports FAIL against a double is untestable in
    the direction that matters: nothing proves it CAN conclude PASS, so a
    permanent red would look like a working guardian.
    """

    def __init__(self, leak: bool = False) -> None:
        """Create the double.

        Args:
            leak: Whether ``ls`` should echo the value it was handed, which is
                the defect assertion 3 exists to catch.
        """
        self.leak = leak
        self.value = b""
        self.stored = False

    def __call__(self, call: support.Call) -> ProductOutput | None:
        """Answer one invocation.

        Args:
            call: What the fake binary was asked to run.

        Returns:
            ProductOutput: The canned answer, or None for the failed default.
        """
        args = list(call.args)
        if args[:1] != ["vault"]:
            return None
        if "set" in args:
            self.value = (call.stdin or b"").strip()
            self.stored = True
            return self._ok(b"secret '%s' stored" % vault.PROBE_NAME.encode())
        if "rm" in args:
            self.stored = False
            return self._ok(b"secret '%s' removed" % vault.PROBE_NAME.encode())
        if "--help" in args:
            return self._ok(b"Commands:\n  ls\n  set\n  rm\n  passwd\n  diagnose\n")
        if "ls" in args:
            if not self.stored:
                return self._ok(b"(vault empty)")
            line = (_LS_LINE % vault.PROBE_NAME).encode("utf-8")
            if self.leak:
                line += b" " + self.value
            return self._ok(line)
        return None

    @staticmethod
    def _ok(stdout: bytes) -> ProductOutput:
        """Wrap *stdout* in a successful capture.

        Args:
            stdout: What the product printed.

        Returns:
            ProductOutput: Exit 0 with that output.
        """
        return ProductOutput(stdout=stdout, stderr=b"", exit_code=0,
                             command=["magi-rs", "vault"])


def _outcomes(scenario_id: str) -> dict[str, Outcome]:
    """Run one registered scenario and index its outcomes by assertion.

    Args:
        scenario_id: The id to invoke.

    Returns:
        dict[str, Outcome]: What each assertion concluded.
    """
    findings = list(DEFAULT_REGISTRY.get(scenario_id).func(None))
    return {finding.assertion: finding.outcome for finding in findings}


class VaultScenarioTests(unittest.TestCase):
    """S3 is registered standalone and declares its five assertions."""

    def test_s3_is_registered_without_a_run(self) -> None:
        entry = DEFAULT_REGISTRY.get("S3")
        self.assertIsNone(entry.run)
        self.assertFalse(entry.needs_backend)

    def test_the_assertion_texts_are_the_spec_texts(self) -> None:
        self.assertEqual(
            [
                "vault set accepts the value from stdin and vault ls lists its name",
                "vault ls prints name and timestamps, never the value",
                "the planted value appears nowhere in stdout, stderr or the run log",
                "no subcommand exists that prints a stored value",
                "vault rm removes it and ls no longer lists it",
            ],
            list(vault.S3_ASSERTIONS),
        )


class VaultScenarioBodyTests(unittest.TestCase):
    """What S3 concludes, against a product that behaves and one that leaks."""

    def test_a_product_that_does_nothing_still_reports_all_five(self) -> None:
        support.install_fake_runs(self)
        findings = list(DEFAULT_REGISTRY.get("S3").func(None))
        self.assertEqual(list(vault.S3_ASSERTIONS),
                         [finding.assertion for finding in findings])
        self.assertNotIn(Outcome.PASS,
                         {finding.outcome for finding in findings})

    def test_a_well_behaved_vault_passes_every_assertion(self) -> None:
        support.install_fake_runs(self, responder=_FakeVault())
        self.assertEqual({Outcome.PASS}, set(_outcomes("S3").values()))

    def test_a_listing_that_echoes_the_value_fails_the_search(self) -> None:
        """The one assertion that cannot be read off the code. It searches the
        capture for every derived form of the planted value, so a product that
        prints it -- in any encoding -- goes red here and nowhere else.
        """
        support.install_fake_runs(self, responder=_FakeVault(leak=True))
        outcomes = _outcomes("S3")
        leaked = vault.S3_ASSERTIONS[2]
        self.assertEqual(Outcome.FAIL, outcomes[leaked])

    def test_the_value_never_travels_as_an_argument(self) -> None:
        """REQ-V10: the value is read from stdin, never from the command line,
        where any process on the machine can read it while the child lives.
        """
        binary = support.install_fake_runs(self, responder=_FakeVault())
        list(DEFAULT_REGISTRY.get("S3").func(None))
        sets = [call for call in binary.calls if "set" in call.args]
        self.assertTrue(sets, "S3 never ran vault set")
        for call in sets:
            self.assertTrue(call.stdin, "the value must travel on stdin")
            planted = call.stdin.strip().decode("utf-8")
            self.assertNotIn(planted, call.args)

    def test_a_help_listing_a_reveal_subcommand_fails(self) -> None:
        """Assertion 4 is verified against the parser's own output. "I am not
        aware of a get subcommand" is not evidence; a Commands list is.
        """
        class _Revealing(_FakeVault):
            def __call__(self, call: support.Call) -> ProductOutput | None:
                if "--help" in call.args:
                    return self._ok(b"Commands:\n  ls\n  get\n  set\n")
                return super().__call__(call)

        support.install_fake_runs(self, responder=_Revealing())
        outcomes = _outcomes("S3")
        self.assertEqual(Outcome.FAIL, outcomes[vault.S3_ASSERTIONS[3]])


if __name__ == "__main__":
    unittest.main()
