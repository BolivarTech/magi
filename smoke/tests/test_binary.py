# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Tests for locating, hashing and invoking the release binary."""

import hashlib
import pathlib
import tempfile
import unittest

from smoke.binary import ReleaseBinary
from smoke.errors import PreflightError


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


if __name__ == "__main__":
    unittest.main()
