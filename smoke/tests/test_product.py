# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Tests for the only path to the product's output."""

import unittest

from smoke.errors import ProductOutputError
from smoke.product import ProductOutput


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


if __name__ == "__main__":
    unittest.main()
