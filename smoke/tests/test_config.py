# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Tests for smoke.toml parsing."""

import pathlib
import tempfile
import unittest

from smoke.config import (
    PAYLOAD_TARGET_BYTES,
    PAYLOAD_TOKEN_FLOOR,
    ModelProfile,
    SmokeConfig,
)
from smoke.errors import PreflightError

_MINIMAL = """
[env]
passphrase = "correct horse battery staple"

[backend]
kind = "ollama"
base_url = "http://localhost:11434/v1"
key_env = "OPENAI_API_KEY"
"""


class LoadTests(unittest.TestCase):
    """A broken config fails before anything runs, naming what broke."""

    def _write(self, text: str) -> pathlib.Path:
        """Write *text* to a throwaway smoke.toml.

        Args:
            text: The file's contents.

        Returns:
            The path written.
        """
        directory = pathlib.Path(tempfile.mkdtemp())
        path = directory / "smoke.toml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_minimal_config_loads(self) -> None:
        config = SmokeConfig.load(self._write(_MINIMAL))
        self.assertEqual("ollama", config.backend_kind)
        self.assertEqual("OPENAI_API_KEY", config.backend_key_env)

    def test_the_calibration_and_payload_fields_parse(self) -> None:
        """Four numbers reach real assertions -- S9's derived ceiling and S8's
        token floor -- and none of them had a parsing test. A field that is
        read but never parsed under test is a field that silently arrives as
        its default.
        """
        path = self._write(
            _MINIMAL + '\n[calibration]\nmargin_tokens = 2048\n'
            'ceiling_fraction = 0.8\n[payload]\ntarget_bytes = 1000\n'
            'token_floor = 200\n'
        )
        config = SmokeConfig.load(path)
        self.assertEqual(2048, config.margin_tokens)
        self.assertEqual(0.8, config.ceiling_fraction)
        self.assertEqual(1000, config.payload_target_bytes)
        self.assertEqual(200, config.payload_token_floor)

    def test_an_absent_payload_section_falls_back_to_the_declared_defaults(self) -> None:
        """Absent is the common case and must not be an error; the example
        ships the section, but a minimal config is legal.
        """
        config = SmokeConfig.load(self._write(_MINIMAL))
        self.assertEqual(PAYLOAD_TARGET_BYTES, config.payload_target_bytes)
        self.assertEqual(PAYLOAD_TOKEN_FLOOR, config.payload_token_floor)

    def test_the_config_has_no_profile_field_even_when_the_section_is_present(self) -> None:
        """REQ-S35: a certifying run cannot pick up a profile it never asked for."""
        path = self._write(
            _MINIMAL + '\n[profile.cheap]\nmodel = "m"\ntrio = ["a", "b", "c"]\n'
        )
        self.assertFalse(hasattr(SmokeConfig.load(path), "profile"))

    def test_missing_file_is_a_preflight_error(self) -> None:
        with self.assertRaises(PreflightError):
            SmokeConfig.load(pathlib.Path("does/not/exist.toml"))

    def test_malformed_toml_is_a_preflight_error_naming_the_file(self) -> None:
        path = self._write("this is not = = toml")
        with self.assertRaises(PreflightError) as caught:
            SmokeConfig.load(path)
        self.assertIn("smoke.toml", str(caught.exception))

    def test_a_weak_passphrase_is_refused(self) -> None:
        path = self._write(_MINIMAL.replace("correct horse battery staple", "short"))
        with self.assertRaises(PreflightError) as caught:
            SmokeConfig.load(path)
        self.assertIn("12", str(caught.exception))

    def test_the_error_never_repeats_the_passphrase(self) -> None:
        path = self._write(_MINIMAL.replace("correct horse battery staple", "hunter2"))
        with self.assertRaises(PreflightError) as caught:
            SmokeConfig.load(path)
        self.assertNotIn("hunter2", str(caught.exception))

    def test_a_declared_profile_is_read_when_it_is_asked_for(self) -> None:
        path = self._write(
            _MINIMAL + '\n[profile.cheap]\nmodel = "m"\ntrio = ["a", "b", "c"]\n'
        )
        self.assertEqual(["a", "b", "c"], ModelProfile.load(path).trio)

    def test_a_trio_that_is_not_three_models_is_refused(self) -> None:
        path = self._write(
            _MINIMAL + '\n[profile.cheap]\nmodel = "m"\ntrio = ["a", "b"]\n'
        )
        with self.assertRaises(PreflightError):
            ModelProfile.load(path)

    def test_asking_for_a_profile_that_is_not_declared_is_a_preflight_error(self) -> None:
        """--profile against a file with no [profile.cheap] must cut, never fall
        back to the defaults: that fallback would certify by accident.
        """
        with self.assertRaises(PreflightError) as caught:
            ModelProfile.load(self._write(_MINIMAL))
        self.assertIn("profile.cheap", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
