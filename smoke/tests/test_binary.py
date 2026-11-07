# Author: Julian Bolivar
# Version: 0.25.0
# Date: 2026-11-08
"""Tests for locating, hashing and invoking the release binary."""

import hashlib
import json
import os
import pathlib
import subprocess
import sys
import unittest
from unittest import mock

from smoke import binary
from smoke.binary import ReleaseBinary
from smoke.errors import PreflightError, ProductOutputError, TimedOut
from smoke.tests import support


class LocationTests(unittest.TestCase):
    """The harness uses target/release, never cargo run (REQ-S07)."""

    def setUp(self) -> None:
        self.root = support.scratch_dir(self)
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
        self.binary = ReleaseBinary(support.scratch_dir(self))

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


class WorkingDirectoryTests(unittest.TestCase):
    """[MAGI §6 MS2 loop 18, F1a1-caspar] the review runs pass ``cwd`` as a ``pathlib.Path``."""

    def test_a_path_working_directory_reaches_the_child(self) -> None:
        """R16's relative input ``.`` names the review workspace only if the
        child really starts there; a ``Path`` must be accepted, not only a str.
        The child is the running interpreter, standing in for the product.

        [F1b2c] The binary's repository root is a DIFFERENT directory from the
        ``cwd`` asked for: when they were the same, an ``invoke`` that started
        the child in the repository root passed as well.
        """
        scratch = support.scratch_dir(self).resolve()
        repo_root, workdir = scratch / "repo", scratch / "work"
        repo_root.mkdir()
        workdir.mkdir()
        with mock.patch.object(ReleaseBinary, "path", new_callable=mock.PropertyMock,
                               return_value=pathlib.Path(sys.executable)):
            output = ReleaseBinary(repo_root).invoke(
                ["-c", "import os; print(os.getcwd())"], timeout=30, cwd=workdir)
        self.assertEqual(0, output.exit_code)
        # realpath on both sides: TEMP may arrive in its 8.3 short form.
        self.assertEqual(os.path.realpath(workdir),
                         os.path.realpath(output.stdout.decode("utf-8").strip()))


class InvokeEnvironmentTests(unittest.TestCase):
    """[F1b1b] the overlay reaches the child on the REAL invoke path, not only through the helper.

    The proxy-egress pins called :func:`binary.child_environment` directly, so an ``invoke`` that built the
    child's environment from ``os.environ`` alone, dropping the overlay, left every one of them green.
    """

    def test_an_overlay_removal_reaches_the_child_started_by_invoke(self) -> None:
        """The child is the running interpreter, standing in for the product, printing its environment."""
        script = ("import json, os; print(json.dumps({k.upper(): v for k, v in os.environ.items()"
                  " if k.upper() in ('HTTP_PROXY', 'SMOKE_OVERLAY_PROBE')}))")
        with mock.patch.dict("os.environ", {"HTTP_PROXY": "http://corp:3128"}), \
                mock.patch.object(ReleaseBinary, "path", new_callable=mock.PropertyMock,
                                  return_value=pathlib.Path(sys.executable)):
            output = ReleaseBinary(support.scratch_dir(self)).invoke(
                ["-c", script], timeout=30, env={"http_proxy": None, "SMOKE_OVERLAY_PROBE": "set"})
        self.assertEqual(0, output.exit_code, output.stderr)
        self.assertEqual({"SMOKE_OVERLAY_PROBE": "set"}, json.loads(output.stdout.decode("utf-8")))


class ChildEnvironmentTests(unittest.TestCase):
    """[MAGI loop 26] the overlay can REMOVE a variable, and matches names whatever their case.

    A refusal run has to take the proxy variables out of the child's environment, and an overlay that
    could only add values could not express that.
    """

    def test_a_none_value_removes_every_case_variant_of_the_name(self) -> None:
        base = {"HTTP_PROXY": "http://corp:3128", "http_proxy": "http://corp:3128", "PATH": "p"}
        self.assertEqual({"PATH": "p"}, binary.child_environment({"Http_Proxy": None}, base=base))

    def test_a_value_replaces_every_case_variant_of_the_name(self) -> None:
        base = {"no_proxy": "corp.example", "PATH": "p"}
        self.assertEqual({"NO_PROXY": "127.0.0.1", "PATH": "p"},
                         binary.child_environment({"NO_PROXY": "127.0.0.1"}, base=base))

    def test_no_overlay_copies_the_base_without_touching_it(self) -> None:
        base = {"PATH": "p"}
        environment = binary.child_environment(None, base=base)
        environment["EXTRA"] = "x"
        self.assertEqual({"PATH": "p"}, base)

    def test_the_default_base_is_the_harness_environment(self) -> None:
        with mock.patch.dict("os.environ", {"SMOKE_CHILD_ENV_PROBE": "seen"}):
            self.assertEqual("seen", binary.child_environment()["SMOKE_CHILD_ENV_PROBE"])


if __name__ == "__main__":
    unittest.main()
