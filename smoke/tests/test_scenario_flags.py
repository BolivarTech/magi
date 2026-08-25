# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Unit tests for the S17 scenario's own shape."""

import unittest

from smoke.outcome import Outcome
from smoke.product import ProductOutput
from smoke.registry import DEFAULT_REGISTRY
from smoke.scenarios import flags  # noqa: F401 - import registers it
from smoke.tests import support

#: What clap prints when it refuses an argument a subcommand does not carry.
_PARSE_ERROR = (
    b"error: unexpected argument '--structured-verdicts' found\n\n"
    b"Usage: magi-rs query [OPTIONS]\n"
)

#: What the product prints when the flag exists but its precondition does not.
_RUNTIME_REFUSAL = (
    b"error: --structured-verdicts requires --output-format json\n"
)


class _FakeFlags:
    """A product double whose ``--structured-verdicts`` surface can be moved.

    Attributes:
        query_answer: How ``query`` reacts to the flag: ``"parse-error"``,
            ``"accepted"`` or ``"runtime"``. The three are what "it didn't
            work" can mean, and only the first is the contract.
        consult_parses: Whether ``consult`` still carries the flag at all. With
            it False the refusal becomes a parse error -- which also exits 2,
            and would pass an assertion that only read the exit code.
    """

    def __init__(self, query_answer: str = "parse-error",
                 consult_parses: bool = True) -> None:
        """Create the double.

        Args:
            query_answer: How ``query`` reacts to the flag.
            consult_parses: Whether ``consult`` accepts the flag.
        """
        self.query_answer = query_answer
        self.consult_parses = consult_parses

    def __call__(self, call: support.Call) -> ProductOutput | None:
        """Answer one invocation.

        Args:
            call: What the fake binary was asked to run.

        Returns:
            ProductOutput: The canned answer, or None for the failed default.
        """
        args = list(call.args)
        if flags.STRUCTURED_VERDICTS_FLAG not in args:
            return None
        if args[:1] == ["query"]:
            return {
                "parse-error": _capture(b"", _PARSE_ERROR, 2),
                "accepted": _capture(b'{"response":"ok"}', b"", 0),
                "runtime": _capture(b"", b"error: something else went wrong\n",
                                    2),
            }[self.query_answer]
        if args[:1] == ["consult"]:
            if not self.consult_parses:
                return _capture(b"", _PARSE_ERROR.replace(b"query", b"consult"),
                                2)
            return _capture(b"", _RUNTIME_REFUSAL, 2)
        return None


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


def _outcomes() -> dict[str, Outcome]:
    """Run S17 against whatever double is installed, indexed by assertion.

    Returns:
        dict[str, Outcome]: What each assertion concluded.
    """
    findings = list(DEFAULT_REGISTRY.get("S17").func(None))
    return {finding.assertion: finding.outcome for finding in findings}


class FlagsScenarioTests(unittest.TestCase):
    """S17 is registered standalone and declares its two assertions."""

    def test_s17_is_registered_without_a_run(self) -> None:
        entry = DEFAULT_REGISTRY.get("S17")
        self.assertIsNone(entry.run)
        self.assertFalse(entry.needs_backend)

    def test_the_assertion_texts_are_the_spec_texts(self) -> None:
        self.assertEqual(
            [
                "query --structured-verdicts is a clap parse error — not an "
                "accepted no-op, not a runtime exit 2",
                "consult --structured-verdicts --output-format text exits 2",
            ],
            list(flags.ASSERTIONS),
        )


class FlagsScenarioBodyTests(unittest.TestCase):
    """Three outcomes look like "it didn't work"; only one is the contract."""

    def test_a_product_that_does_nothing_still_reports_both(self) -> None:
        support.install_fake_runs(self)
        findings = list(DEFAULT_REGISTRY.get("S17").func(None))
        self.assertEqual(list(flags.ASSERTIONS),
                         [finding.assertion for finding in findings])
        self.assertNotIn(Outcome.PASS,
                         {finding.outcome for finding in findings})

    def test_the_declared_surface_passes_both(self) -> None:
        support.install_fake_runs(self, responder=_FakeFlags())
        self.assertEqual({Outcome.PASS}, set(_outcomes().values()))

    def test_a_flag_query_silently_accepts_fails(self) -> None:
        """The defect an operator only discovers when the field they expected
        is missing from a batch run's output.
        """
        support.install_fake_runs(self,
                                  responder=_FakeFlags(query_answer="accepted"))
        self.assertEqual(Outcome.FAIL, _outcomes()[flags.ASSERTIONS[0]])

    def test_a_runtime_exit_two_on_query_fails(self) -> None:
        """A different defect with the same exit code: the flag parsed on a
        subcommand that should not have it, and failed later.
        """
        support.install_fake_runs(self,
                                  responder=_FakeFlags(query_answer="runtime"))
        self.assertEqual(Outcome.FAIL, _outcomes()[flags.ASSERTIONS[0]])

    def test_a_consult_that_no_longer_carries_the_flag_fails(self) -> None:
        """The empty green this assertion is written against. A parse error
        also exits 2, so reading the exit code alone would report the feature
        as protected on the very run that removed it.
        """
        support.install_fake_runs(self,
                                  responder=_FakeFlags(consult_parses=False))
        self.assertEqual(Outcome.FAIL, _outcomes()[flags.ASSERTIONS[1]])


if __name__ == "__main__":
    unittest.main()
