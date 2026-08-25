# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Tests for derived secret forms, scrubbing and authority detection."""

import unittest
import urllib.parse

from smoke.secrets import (
    MARKER_CHARS,
    PlantedSecret,
    find_secret,
    find_unredacted_authorities,
    mint_credential,
    scrub,
)


class DerivedFormTests(unittest.TestCase):
    """"Literal" does not mean "one form": the product may encode it."""

    def setUp(self) -> None:
        self.secret = PlantedSecret("p@ss/w#rd", "backend credential")

    def test_the_raw_form_is_included(self) -> None:
        self.assertIn(b"p@ss/w#rd", self.secret.forms())

    def test_the_percent_encoded_form_is_included(self) -> None:
        encoded = urllib.parse.quote("p@ss/w#rd", safe="").encode("utf-8")
        self.assertIn(encoded, self.secret.forms())

    def test_encoding_uses_safe_empty_so_slash_is_encoded(self) -> None:
        self.assertTrue(any(b"%2F" in f.upper() for f in self.secret.forms()))

    def test_lowercase_hex_is_found_too(self) -> None:
        leaked = b"url=http://u:p%40ss%2fw%23rd@host/v1"
        self.assertIn("backend credential", find_secret(leaked, [self.secret]))

    def test_uppercase_hex_is_found_too(self) -> None:
        leaked = b"url=http://u:p%40ss%2Fw%23rd@host/v1"
        self.assertIn("backend credential", find_secret(leaked, [self.secret]))


class MarkerTests(unittest.TestCase):
    """The marker survives any encoding, and is long enough not to collide."""

    def test_a_minted_credential_carries_reserved_characters(self) -> None:
        secret = mint_credential()
        self.assertTrue(any(c in secret.value for c in "@/?#"))

    def test_the_marker_is_alphanumeric_so_encoding_leaves_it_alone(self) -> None:
        marker = mint_credential().marker
        self.assertTrue(marker.isalnum())
        self.assertEqual(marker, urllib.parse.quote(marker, safe=""))

    def test_the_marker_is_long_enough_to_avoid_collision(self) -> None:
        self.assertGreaterEqual(len(mint_credential().marker), MARKER_CHARS)

    def test_two_minted_credentials_differ(self) -> None:
        self.assertNotEqual(mint_credential().value, mint_credential().value)

    def test_a_marker_below_the_floor_is_not_searched(self) -> None:
        """The other side of the guard, so the asymmetry is a decision and not
        an accident.

        A secret the harness did not mint -- the passphrase, an existing
        credential -- has whatever marker its value happens to contain. Below
        the floor it is searched by value alone, which is weaker coverage,
        accepted because a short marker matching by chance would put S10 red
        with no leak at all. If this goes red the guard was dropped, and every
        planted passphrase became a source of false positives.
        """
        weak = PlantedSecret(value="ab/cd", label="hand made")
        self.assertNotIn(weak.marker.encode("utf-8"), weak.forms())

    def test_the_marker_is_one_of_the_searched_forms(self) -> None:
        """Invariance is worthless if nothing searches for it.

        An earlier round asserted the marker survives double encoding and
        stopped there -- but ``find_secret`` searches ``forms()``, so a marker
        outside that set is a property the harness proves and never uses. Drop
        the marker from ``forms()`` and this goes red.
        """
        secret = mint_credential()
        self.assertIn(secret.marker.encode("utf-8"), secret.forms())

    def test_a_double_encoded_secret_is_found_through_the_marker(self) -> None:
        """End to end: the encoding nobody derived is still detected.

        ``%2540`` shares no substring with the raw secret or with ``%40``, so
        deriving level two would only invite level three. The alphanumeric
        marker ends the arms race instead of entering it.
        """
        secret = mint_credential()
        once = urllib.parse.quote(secret.value, safe="")
        twice = urllib.parse.quote(once, safe="")
        leaked = ("url=http://u:%s@host/x" % twice).encode("utf-8")
        self.assertNotEqual([], find_secret(leaked, [secret]))


class ScrubTests(unittest.TestCase):
    """Scrubbing announces what it removed; it never deletes in silence."""

    def setUp(self) -> None:
        self.secret = PlantedSecret("s3cr3t-value", "backend credential")

    def test_the_value_is_gone_from_the_scrubbed_bytes(self) -> None:
        scrubbed = scrub(b"before s3cr3t-value after", [self.secret])
        self.assertNotIn(b"s3cr3t-value", scrubbed)

    def test_the_replacement_names_the_secret_and_the_count(self) -> None:
        scrubbed = scrub(b"s3cr3t-value and s3cr3t-value", [self.secret])
        self.assertIn(b"backend credential", scrubbed)
        self.assertIn(b"2 occurrences", scrubbed)

    def test_surrounding_text_survives_so_diagnosis_stays_possible(self) -> None:
        scrubbed = scrub(b"before s3cr3t-value after", [self.secret])
        self.assertIn(b"before", scrubbed)
        self.assertIn(b"after", scrubbed)

    def test_clean_data_is_returned_unchanged(self) -> None:
        self.assertEqual(b"nothing here", scrub(b"nothing here", [self.secret]))


class AuthorityShapeTests(unittest.TestCase):
    """The independent detector: shape, not the secret's value."""

    def test_an_unredacted_authority_is_found_in_plain_prose(self) -> None:
        prose = b"request failed: https://user:pw@api.example.com/v1 returned 401"
        self.assertEqual(1, len(find_unredacted_authorities(prose)))

    def test_a_redacted_authority_is_not_reported(self) -> None:
        prose = b"request failed: https://***@api.example.com/v1 returned 401"
        self.assertEqual([], find_unredacted_authorities(prose))

    def test_a_url_with_no_authority_is_not_reported(self) -> None:
        self.assertEqual([], find_unredacted_authorities(b"see http://host/v1"))


if __name__ == "__main__":
    unittest.main()
