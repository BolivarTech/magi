# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Tests for locating, hashing and invoking the release binary."""

import hashlib
import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock

from smoke.binary import ReleaseBinary
from smoke.errors import PreflightError, ProductOutputError, TimedOut


class LocationTests(unittest.TestCase):
    """The harness uses target/release, never cargo run (REQ-S07)."""

    def setUp(self) -> None:
        self.root = pathlib.Path(tempfile.mkdtemp())
        (self.root / "target" / "release").mkdir(parents=True)

    def _plant(self, content: bytes = b"binary") -> pathlib.Path:
        binary = ReleaseBinary(self.root).path
        binary.write_bytes(content)
        return binary

    def test_path_points_at_target_release(self) -> None:
        parts = ReleaseBinary(self.root).path.parts
        self.assertIn("target", parts)
        self.assertIn("release", parts)

    def test_a_missing_binary_is_a_preflight_error(self) -> None:
        with self.assertRaises(PreflightError):
            ReleaseBinary(self.root).sha256()

    def test_sha256_matches_the_file_on_disk(self) -> None:
        binary = self._plant(b"deterministic content")
        expected = hashlib.sha256(binary.read_bytes()).hexdigest()
        self.assertEqual(expected, ReleaseBinary(self.root).sha256())

    def test_sha256_is_64_lowercase_hex_characters(self) -> None:
        self._plant()
        digest = ReleaseBinary(self.root).sha256()
        self.assertEqual(64, len(digest))
        self.assertEqual(digest, digest.lower())
        self.assertTrue(all(c in "0123456789abcdef" for c in digest))

    def test_a_changed_binary_changes_the_hash(self) -> None:
        self._plant(b"one")
        first = ReleaseBinary(self.root).sha256()
        self._plant(b"two")
        self.assertNotEqual(first, ReleaseBinary(self.root).sha256())


class ExpiryTests(unittest.TestCase):
    """A run that did not finish still carries what it had already written."""

    def setUp(self) -> None:
        self.binary = ReleaseBinary(pathlib.Path(tempfile.mkdtemp()))

    def test_a_timeout_carries_the_streams_the_child_had_written(self) -> None:
        """Discarding them would remove the ONE case the timeout rule grants.

        A consult that hangs may already have emitted the block a scenario
        reads to tell a degraded ceiling from a slow provider. Re-running to
        recover it would be a second run of the thing that hung.
        """
        expired = subprocess.TimeoutExpired(cmd=["magi-rs"], timeout=1,
                                            output=b"partial", stderr=b"warn")
        with mock.patch("smoke.binary.subprocess.run", side_effect=expired):
            with self.assertRaises(TimedOut) as caught:
                self.binary.invoke(["consult"], timeout=1)
        self.assertEqual(b"partial", caught.exception.output.stdout)
        self.assertEqual(b"warn", caught.exception.output.stderr)

    def test_a_timeout_with_nothing_captured_still_carries_a_capture(self) -> None:
        """POSIX leaves the streams unset on expiry, so ``None`` is a real
        shape. Empty bytes say "nothing was captured", which is not the same
        claim as "the product emitted nothing" -- and a caller that had to
        handle ``None`` here would eventually forget."""
        expired = subprocess.TimeoutExpired(cmd=["magi-rs"], timeout=1)
        with mock.patch("smoke.binary.subprocess.run", side_effect=expired):
            with self.assertRaises(TimedOut) as caught:
                self.binary.invoke(["consult"], timeout=1)
        self.assertEqual(b"", caught.exception.output.stdout)

    def test_a_timeout_is_still_a_product_output_error(self) -> None:
        """Every caller that already treats an expiry as one keeps working."""
        self.assertTrue(issubclass(TimedOut, ProductOutputError))


if __name__ == "__main__":
    unittest.main()
