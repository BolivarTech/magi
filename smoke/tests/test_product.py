# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Tests for the only path to the product's output."""

import unittest

from smoke.errors import ProductOutputError
from smoke.product import diagnose_counts, ProductOutput


def _output(stdout: bytes = b"{}", exit_code: int = 0) -> ProductOutput:
    return ProductOutput(stdout=stdout, stderr=b"", exit_code=exit_code,
                         command=["magi-rs", "query"])


class JsonTests(unittest.TestCase):
    """Malformed output is the product's defect, reported as such."""

    def test_valid_json_parses(self) -> None:
        self.assertEqual({"a": 1}, _output(b'{"a": 1}').json())

    def test_malformed_json_raises_product_output_error(self) -> None:
        with self.assertRaises(ProductOutputError):
            _output(b"not json").json()

    def test_missing_key_raises_product_output_error(self) -> None:
        with self.assertRaises(ProductOutputError):
            _output(b'{"a": 1}').key("applied_caps.ceiling_floored")

    def test_nested_key_resolves(self) -> None:
        payload = b'{"applied_caps": {"ceiling_floored": true}}'
        self.assertIs(True, _output(payload).key("applied_caps.ceiling_floored"))

    def test_unexpected_exit_code_raises_product_output_error(self) -> None:
        with self.assertRaises(ProductOutputError):
            _output(exit_code=2).require_exit(0)


class RawTests(unittest.TestCase):
    """raw() is the named exemption, and it cannot fail."""

    def test_raw_returns_every_byte_of_both_streams(self) -> None:
        output = ProductOutput(stdout=b"out", stderr=b"err", exit_code=0, command=[])
        self.assertIn(b"out", output.raw())
        self.assertIn(b"err", output.raw())

    def test_raw_never_raises_on_malformed_bytes(self) -> None:
        output = ProductOutput(stdout=b"\xff\xfe not utf-8", stderr=b"", exit_code=0, command=[])
        self.assertIn(b"\xff\xfe", output.raw())


class DiagnoseCountsTests(unittest.TestCase):
    """One parser for the product's counts block, with the semantics written.

    There were two -- one in ``runs.py`` for the baseline, one in
    ``scenarios/vault.py`` for the reading -- with the same regex and different
    behaviour: only the second filtered to the history tables, and they
    disagreed about an unreadable report. The divergence was already being
    compensated for at a call site, and it was found by a live run reporting
    "vault went from 1 to 0", not by a test.
    """

    def test_it_reads_the_counts_block(self) -> None:
        report = ("envelope: present\nfec: ok\nverdict: healthy\ncounts:\n"
                  "  vault: 2\n  sessions: 1\n  messages: 8\n")
        self.assertEqual({"vault": 2, "sessions": 1, "messages": 8},
                         diagnose_counts(report.encode("utf-8")))

    def test_a_missing_table_does_not_end_the_block(self) -> None:
        """``vault`` is the FIRST label the product prints and it renders
        ``missing`` for a table it creates lazily. Stopping at the first
        non-numeric line dropped every history table that followed it, and the
        baseline then came back empty for a healthy database.
        """
        report = ("counts:\n  vault: missing\n  sessions: 1\n"
                  "  messages: 8\n")
        self.assertEqual({"sessions": 1, "messages": 8},
                         diagnose_counts(report.encode("utf-8")))

    def test_a_missing_table_is_absent_rather_than_zero(self) -> None:
        """Unknown is not empty, which is what the docstrings already claimed."""
        report = "counts:\n  vault: missing\n"
        self.assertNotIn("vault", diagnose_counts(report.encode("utf-8")))

    def test_a_report_without_the_block_reads_as_nothing(self) -> None:
        self.assertEqual({}, diagnose_counts(b"envelope: absent\n"))


if __name__ == "__main__":
    unittest.main()
