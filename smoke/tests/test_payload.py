# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Tests for the large payload generator.

Bytes are what the generator can control without a tokenizer, so bytes are what
it promises. The token floor is asserted at run time against what the product
reports it counted -- that is a different file's job, and deliberately so.
"""

import pathlib
import re
import shutil
import tempfile
import unittest

from smoke.config import PAYLOAD_TARGET_BYTES
from smoke.errors import HarnessError
from smoke.payload import PayloadBuilder
from smoke.tests import support

#: Where the product declares the cap it rejects a payload above, and the
#: pattern that reads it. Parsed from the source rather than copied here: a
#: copied number is a second source of truth that stops being true silently,
#: and the whole point of the assertion is that the harness never generates
#: something the product will refuse.
_CAP_SOURCE = pathlib.Path("src") / "magi" / "mod.rs"
_CAP = re.compile(r"MAX_QUERY_BYTES:\s*usize\s*=\s*(\d+)\s*\*\s*(\d+)\s*;")


def _plant(root: pathlib.Path, relative: str, body: bytes) -> None:
    """Write one source file into a fixture tree.

    Args:
        root: The fixture repository root.
        relative: The path under it, with forward slashes.
        body: The bytes to write, written WITHOUT translation.
    """
    path = root.joinpath(*relative.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)


class BuildTests(unittest.TestCase):
    """What the builder promises about the bytes it returns."""

    def setUp(self) -> None:
        self.root = support.scratch_dir(self)
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        _plant(self.root, "src/alpha.rs", b"a" * 500)
        _plant(self.root, "src/nested/beta.rs", b"b" * 500)
        self.builder = PayloadBuilder(self.root)

    def test_the_payload_is_exactly_the_requested_size(self) -> None:
        self.assertEqual(777, len(self.builder.build(777)))

    def test_a_zero_sized_payload_is_empty(self) -> None:
        self.assertEqual(b"", self.builder.build(0))

    def test_two_builds_over_an_unchanged_tree_are_identical(self) -> None:
        """Determinism is the property, and nothing else checked it.

        Two runs of the same commit have to send the SAME bytes or their
        results are not comparable, and filesystem iteration order does not
        promise that on any platform.
        """
        self.assertEqual(self.builder.build(600), self.builder.build(600))

    def test_available_bytes_reports_what_the_tree_holds(self) -> None:
        self.assertEqual(1000, self.builder.available_bytes())

    def test_asking_for_more_than_the_tree_holds_is_refused(self) -> None:
        """The scenario turns this into CANNOT_TEST, never FAIL.

        Sending a short payload silently would let the run pass green without
        testing what it claims, and accusing the product of a size it was
        never sent would be worse.
        """
        with self.assertRaises(HarnessError):
            self.builder.build(1001)

    def test_a_negative_size_is_refused(self) -> None:
        with self.assertRaises(HarnessError):
            self.builder.build(-1)

    def test_a_tree_with_no_sources_reports_nothing_available(self) -> None:
        empty = support.scratch_dir(self)
        self.addCleanup(shutil.rmtree, empty, ignore_errors=True)
        self.assertEqual(0, PayloadBuilder(empty).available_bytes())


class OrderingTests(unittest.TestCase):
    """The order is the byte order of the POSIX-normalised relative path."""

    def setUp(self) -> None:
        self.root = support.scratch_dir(self)
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_the_separator_is_normalised_before_the_comparison(self) -> None:
        """``/`` is 0x2F and ``\\`` is 0x5C, with ``A`` at 0x41 between them.

        So ``src/a/z.rs`` precedes ``src/aA.rs`` under POSIX spelling and
        FOLLOWS it under the native Windows one. Sorting the native string
        would therefore produce a different payload on Windows than on Linux
        from the same commit -- and the token floor would be measuring a
        different input on each.
        """
        _plant(self.root, "src/aA.rs", b"A")
        _plant(self.root, "src/a/z.rs", b"Z")
        self.assertEqual(b"ZA", PayloadBuilder(self.root).build(2))

    def test_the_comparison_is_by_byte_and_not_by_case(self) -> None:
        """``B`` is 0x42 and ``a`` is 0x61, so byte order puts ``B`` first
        while any case-insensitive ordering puts ``a`` first."""
        _plant(self.root, "src/B.rs", b"B")
        _plant(self.root, "src/a.rs", b"a")
        self.assertEqual(b"Ba", PayloadBuilder(self.root).build(2))

    def test_creation_order_does_not_reach_the_payload(self) -> None:
        for name in ("m", "c", "x", "b"):
            _plant(self.root, "src/%s.rs" % name, name.encode("ascii"))
        self.assertEqual(b"bcmx", PayloadBuilder(self.root).build(4))


class BinaryReadTests(unittest.TestCase):
    """Text mode would translate line endings and send something else."""

    def test_a_carriage_return_survives_into_the_payload(self) -> None:
        """On a CRLF checkout a text-mode read changes the content actually
        sent, so the payload would differ between two clones of one commit."""
        root = support.scratch_dir(self)
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        _plant(root, "src/crlf.rs", b"one\r\ntwo\r\n")
        self.assertIn(b"\r\n", PayloadBuilder(root).build(10))


class CapTests(unittest.TestCase):
    """The declared size must be one the product will accept."""

    def test_the_target_sits_below_the_product_cap(self) -> None:
        """The product REJECTS rather than truncates, so a target above its
        cap turns every run of the scenario into a refusal."""
        root = pathlib.Path(__file__).resolve().parent.parent.parent
        match = _CAP.search((root / _CAP_SOURCE).read_text(encoding="utf-8"))
        self.assertIsNotNone(
            match, "the product's input cap could not be read from %s; the "
                   "assertion below would otherwise pass by never running"
                   % _CAP_SOURCE.as_posix())
        cap = int(match.group(1)) * int(match.group(2))
        self.assertLess(PAYLOAD_TARGET_BYTES, cap)


if __name__ == "__main__":
    unittest.main()
