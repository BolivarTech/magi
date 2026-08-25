# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Tests for the CLI surface and the four exit codes."""

import unittest

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


if __name__ == "__main__":
    unittest.main()
