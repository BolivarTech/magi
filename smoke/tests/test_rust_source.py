# Author: Julian Bolivar
# Version: 0.24.3
# Date: 2026-11-02
"""Tests for the helpers the source-sync guards read the Rust tree through.

[MAGI loop 26] the guards searched the whole file, so a constant that drifted
while its old line survived in a ``///`` comment above it was read from the
comment and the guard stayed green; two definitions were never noticed either.
"""

import importlib
import unittest

_DRIFTED = ("/// Was `pub const BYTES_PER_TOKEN: usize = 4;` before the change.\n"
            "pub const BYTES_PER_TOKEN: usize = 5;\n")


def _helpers():
    """The module under test, imported when a test runs so one missing name fails per test."""
    return importlib.import_module("smoke.tests.rust_source")


class OneDefinitionTests(unittest.TestCase):
    """``one_definition`` finds the single real line a pattern starts."""

    def test_a_match_only_inside_a_comment_is_rejected(self) -> None:
        text = "// pub const LIMIT: usize = 4;\n/// pub const LIMIT: usize = 4;\nfn f() {}\n"
        with self.assertRaisesRegex(AssertionError, r"x\.rs.*LIMIT.*found 0"):
            _helpers().one_definition(text, r"pub const LIMIT: usize = (\d+);", "x.rs")

    def test_a_pattern_that_could_start_on_the_slashes_still_skips_a_comment_line(self) -> None:
        # The anchor alone cannot exclude a comment when the pattern itself may start with '/'.
        text = "/// LIMIT = 4;\nfn f() {}\n"
        with self.assertRaisesRegex(AssertionError, "found 0"):
            _helpers().one_definition(text, r".*LIMIT = (\d+);", "x.rs")

    def test_the_real_definition_wins_over_an_old_one_left_in_a_doc_comment(self) -> None:
        match = _helpers().one_definition(_DRIFTED, r"pub const BYTES_PER_TOKEN: usize = (\d+);", "w.rs")
        self.assertEqual("5", match.group(1))

    def test_two_real_definitions_are_rejected(self) -> None:
        text = "pub const LIMIT: usize = 4;\nmod inner {\n    pub const LIMIT: usize = 4;\n}\n"
        with self.assertRaisesRegex(AssertionError, r"x\.rs.*found 2"):
            _helpers().one_definition(text, r"pub const LIMIT: usize = (\d+);", "x.rs")

    def test_exactly_one_indented_definition_is_accepted(self) -> None:
        text = "fn f() {\n    match x {\n        Ok(n) => remaining_secs / n,\n    }\n}\n"
        match = _helpers().one_definition(text, r"Ok\(n\) => remaining_secs / n,", "c.rs")
        self.assertEqual("        Ok(n) => remaining_secs / n,", match.group(0))

    def test_a_pattern_never_starts_mid_line(self) -> None:
        text = "let old = 1; let per_batch = split(total);\n"
        with self.assertRaises(AssertionError):
            _helpers().one_definition(text, r"let per_batch = split\(total\);", "c.rs")


class OneOccurrenceTests(unittest.TestCase):
    """``one_occurrence`` finds a literal on exactly one code line, wherever it sits on the line."""

    def test_a_literal_mid_line_is_found(self) -> None:
        text = '    #[error("path is outside the workspace: {}", display_path(.0))]\n'
        match = _helpers().one_occurrence(text, '"path is outside the workspace: {}"', "e.rs")
        self.assertIn('"path is outside the workspace: {}"', match.group(0))

    def test_a_literal_only_in_a_comment_or_after_one_is_rejected(self) -> None:
        for text in ('// Ok(n) => remaining_secs / n,\n',
                     'let a = 1; // Ok(n) => remaining_secs / n,\n'):
            with self.assertRaisesRegex(AssertionError, "found 0", msg=text):
                _helpers().one_occurrence(text, "Ok(n) => remaining_secs / n,", "c.rs")

    def test_a_literal_on_two_code_lines_is_rejected(self) -> None:
        text = 'a("x");\nb("x");\n'
        with self.assertRaisesRegex(AssertionError, "found 2"):
            _helpers().one_occurrence(text, '("x")', "c.rs")

    def test_a_literal_may_span_lines(self) -> None:
        text = '        "consensus".into(),\n        consensus_json(&c),\n'
        match = _helpers().one_occurrence(text, '"consensus".into(),\n        consensus_json(&c)', "t.rs")
        self.assertTrue(match.group(0).endswith("consensus_json(&c)"))


class CodeOnlyTests(unittest.TestCase):
    """``code_only`` blanks comment lines and keeps every other line where it was."""

    def test_comment_lines_are_blanked_and_line_numbers_kept(self) -> None:
        text = 'a = [\n    "kept",\n    // "dropped",\n    /// "doc",\n]\n'
        self.assertEqual('a = [\n    "kept",\n\n\n]\n', _helpers().code_only(text))

    def test_a_line_with_code_before_a_comment_is_kept_whole(self) -> None:
        text = 'let x = "a"; // trailing\n'
        self.assertEqual(text, _helpers().code_only(text))


class GuardRoutingTests(unittest.TestCase):
    """The sync guards themselves read through the helpers."""

    def test_the_constant_guard_reads_the_definition_not_the_comment_above_it(self) -> None:
        from smoke.tests import test_rust_sync
        self.assertEqual(5, test_rust_sync._usize_const(_DRIFTED, "BYTES_PER_TOKEN"))


if __name__ == "__main__":
    unittest.main()
