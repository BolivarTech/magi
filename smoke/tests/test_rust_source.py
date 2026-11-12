# Author: Julian Bolivar
# Version: 0.25.0
# Date: 2026-11-08
"""Tests for the helpers the source-sync guards read the Rust tree through.

[MAGI loop 26] the guards searched the whole file, so a constant that drifted
while its old line survived in a ``///`` comment above it was read from the
comment and the guard stayed green; two definitions were never noticed either.
"""

import importlib
import re
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

    def test_a_line_inside_a_string_literal_is_not_a_definition(self) -> None:
        # [MAGI F1b2a] a fixture line inside a multi-line string started its own line, so it was read as the
        # definition: alone it answered for a constant the file never defines, beside the real one it made two.
        for fixture in ('pub const FIXTURE: &str = r#"\npub const LIMIT: usize = 9;\n"#;\n',
                        'pub const FIXTURE: &str = "\npub const LIMIT: usize = 9;\n";\n',
                        'const B: &[u8] = br"\npub const LIMIT: usize = 9;\n";\n'):
            with self.assertRaisesRegex(AssertionError, "found 0", msg=fixture):
                _helpers().one_definition(fixture, r"pub const LIMIT: usize = (\d+);", "x.rs")
            match = _helpers().one_definition(fixture + "pub const LIMIT: usize = 5;\n",
                                              r"pub const LIMIT: usize = (\d+);", "x.rs")
            self.assertEqual("5", match.group(1), fixture)

    def test_a_definition_that_opens_a_string_still_reads_its_contents(self) -> None:
        text = 'pub const CAUSE: &str = "secret";\n'
        self.assertEqual("secret", _helpers().one_definition(text, r'pub const CAUSE: &str = "([^"]+)";', "x.rs")
                         .group(1))

    def test_a_pattern_never_starts_mid_line(self) -> None:
        text = "let old = 1; let per_batch = split(total);\n"
        with self.assertRaises(AssertionError):
            _helpers().one_definition(text, r"let per_batch = split\(total\);", "c.rs")

    def test_a_wildcard_pattern_never_reads_a_value_from_a_trailing_comment(self) -> None:
        # MAGI FIX (S6 loop 34): a trailing // comment stayed in the searched view, so a pattern with a
        # leading wildcard read an old copy left in that comment as the definition.
        text = "fn f() {} // pub const X: usize = 4;\n"
        with self.assertRaisesRegex(AssertionError, "found 0"):
            _helpers().one_definition(text, r".*pub const X: usize = (\d+);", "f")

    def test_a_wildcard_pattern_never_reads_a_value_from_inside_a_literal(self) -> None:
        # [MAGI S6 loop 46, F1b2a-caspar] only the match's start was checked, so a leading or middle wildcard ran
        # into a string and captured a stale copy's value from inside it.
        for text, pattern in (('let s = "pub const LIMIT: usize = 4;";\n', r".*pub const LIMIT: usize = (\d+);"),
                              ('pub const LIMIT: u8 = f("= 4;");\n', r"pub const LIMIT.*= (\d+);")):
            with self.subTest(text=text), self.assertRaisesRegex(AssertionError, "found 0"):
                _helpers().one_definition(text, pattern, "x.rs")
        self.assertEqual("5", _helpers().one_definition('let s = "x";\npub const LIMIT: usize = 5;\n',
                                                        r".*pub const LIMIT: usize = (\d+);", "x.rs").group(1))

    def test_a_body_capture_spanning_lines_leaves_a_trailing_comment_out(self) -> None:
        # MAGI FIX (S6 loop 34): a re.S body capture carried a trailing comment's text with it.
        text = "fn f() {\n    let a = 1; // stale: let a = 2;\n}\n"
        body = _helpers().one_definition(text, r"fn f\(\) \{(.*?)\n\}", "f", flags=re.S).group(1)
        self.assertIn("let a = 1;", body)
        self.assertNotIn("stale", body)
        self.assertNotIn("//", body)


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

    def test_a_copy_inside_a_multi_line_literal_is_not_counted(self) -> None:
        # [MAGI S6 loop 38, F1b2a-melchior] a fixture line inside a multi-line string was counted, so with the
        # production line gone a stale copy in a test fixture still answered for it.
        fixture = 'let fixture = r#"\n    format!(", and {rest} more")\n"#;\n'
        with self.assertRaisesRegex(AssertionError, "found 0"):
            _helpers().one_occurrence(fixture, 'format!(", and {rest} more")', "e.rs")
        production = '    format!(", and {rest} more")\n'
        _helpers().one_occurrence(fixture + production, 'format!(", and {rest} more")', "e.rs")

    def test_a_copy_in_a_test_item_is_not_counted(self) -> None:
        # [MAGI S6 loop 44, F1b2a-melchior] a one-line fixture in a test module was a code line, so with the
        # production line gone it still answered for the wording.
        tests = ('#[cfg(test)]\nmod tests {\n    #[test]\n    fn t() {\n        let stale = "x is gone";\n'
                 '        assert!(f() != "{");\n    }\n}\n')
        for text in (tests, '#[cfg(test)]\nconst STALE: &str = "x is gone";\n'):
            with self.subTest(text=text), self.assertRaisesRegex(AssertionError, "found 0"):
                _helpers().one_occurrence(text, '"x is gone"', "e.rs")
        production = 'fn f() -> &str {\n    "x is gone"\n}\n'
        self.assertIn('"x is gone"', _helpers().one_occurrence(production + tests, '"x is gone"', "e.rs").group(0))

    def test_every_test_only_form_is_skipped_and_nothing_after_it(self) -> None:
        # [MAGI S6 loop 45, F1b2a-caspar, F1b2a-balthasar] a ``const fn`` was read as a ``;`` item and blanked the
        # production code after it, and ``cfg(all(test, ...))``, fields and statements were not skipped at all.
        production = 'fn prod() -> &str {\n    "x is gone"\n}\n'
        for fixture in ('#[cfg(test)]\nconst fn h() -> u8 { 1 }\n',
                        '#[cfg(test)]\npub(crate) const unsafe fn h() -> u8 { 1 }\n',
                        '#[cfg(all(test, unix))]\nfn t() { f("x is gone"); }\n',
                        '#[cfg(all(unix, any(test)))]\nconst STALE: &str = "x is gone";\n',
                        'fn s() -> S {\n    S {\n        #[cfg(test)]\n        msg: "x is gone",\n        n: 1,\n    }\n}\n',
                        'fn s() {\n    #[cfg(test)]\n    f("x is gone");\n    g();\n}\n',
                        'fn s(v: u8) {\n    match v {\n        #[cfg(test)]\n        0 => { f("x is gone") }\n    }\n}\n',
                        # [MAGI S6 loop 47, F1b2a1-caspar, F1b2a1-melchior] a generic argument's comma ended the
                        # item, and so did the ``}`` of a struct pattern, leaving the copy after it counted.
                        '#[cfg(test)]\nconst M: Foo<u8, u8> = Foo::new("x is gone");\n',
                        'fn s() {\n    #[cfg(test)]\n    foo::<A, B>("x is gone");\n    g();\n}\n',
                        'fn s() {\n    #[cfg(test)]\n    let x = <A as B<C, D>>::f("x is gone");\n    g();\n}\n',
                        'fn s(p: P) {\n    match p {\n        #[cfg(test)]\n        P { x: 0 } => f("x is gone"),\n'
                        '        _ => {}\n    }\n}\n',
                        'fn s(p: P) {\n    match p {\n        #[cfg(test)]\n        P { x } if x > 0 => f("x is gone"),\n'
                        '        _ => {}\n    }\n}\n',
                        'fn s() {\n    #[cfg(test)]\n    let a = { 1 } + f("x is gone");\n    g();\n}\n',
                        'fn s() {\n    #[cfg(test)]\n    let x = async {\n        1\n    }\n    .f("x is gone");\n'
                        '    g();\n}\n',
                        'fn s() {\n    #[cfg(test)]\n    let f = |a| g("x is gone");\n    g();\n}\n'):
            with self.subTest(fixture=fixture):
                with self.assertRaisesRegex(AssertionError, "found 0"):
                    _helpers().one_occurrence(fixture, '"x is gone"', "e.rs")
                _helpers().one_occurrence(fixture + production, '"x is gone"', "e.rs")
                _helpers().one_occurrence(production + fixture, '"x is gone"', "e.rs")

    def test_an_ambiguous_test_only_item_fails_loudly(self) -> None:
        # [MAGI S6 loop 47] each loop found another comma or brace the delimiter misread; a comma it cannot place
        # (a closure's parameters, a comparison) or a generic that never closes now fails instead of guessing.
        for fixture in ('fn s() {\n    #[cfg(test)]\n    let f = |a, b| g("x is gone");\n    g();\n}\n',
                        'fn s() -> S {\n    S {\n        #[cfg(test)]\n        ok: a < b,\n        n: f("x is gone"),\n'
                        '    }\n}\n',
                        'fn s() {\n    #[cfg(test)]\n    let ok = a<b;\n    f("x is gone");\n}\n'):
            with self.subTest(fixture=fixture), self.assertRaisesRegex(AssertionError, "ambiguous"):
                _helpers().one_occurrence(fixture, '"x is gone"', "e.rs")

    def test_a_cfg_that_does_not_require_test_is_production(self) -> None:
        for attribute in ("#[cfg(not(test))]", "#[cfg(any(test, unix))]", "#[cfg(unix)]",
                          '#[cfg(feature = "test")]'):
            with self.subTest(attribute=attribute):
                _helpers().one_occurrence(attribute + '\nfn f() { g("x is gone"); }\n', '"x is gone"', "e.rs")

    def test_a_needle_that_starts_inside_a_literal_is_not_counted(self) -> None:
        # [MAGI S6 loop 44, F1b2a-melchior] a needle found inside any string answered for the production code.
        with self.assertRaisesRegex(AssertionError, "found 0"):
            _helpers().one_occurrence('let s = "the path is outside";\n', "path is outside", "e.rs")


class CodeOnlyTests(unittest.TestCase):
    """``code_only`` blanks comment lines and keeps every other line where it was."""

    def test_comment_lines_are_blanked_and_line_numbers_kept(self) -> None:
        text = 'a = [\n    "kept",\n    // "dropped",\n    /// "doc",\n]\n'
        self.assertEqual('a = [\n    "kept",\n\n\n]\n', _helpers().code_only(text))

    def test_a_line_with_code_before_a_comment_is_kept_whole(self) -> None:
        text = 'let x = "a"; // trailing\n'
        self.assertEqual(text, _helpers().code_only(text))


class BlockCommentTests(unittest.TestCase):
    """[MAGI §6 MS2 loop 28, F1b2a-caspar] a block comment never answers for code.

    Only ``//`` lines were recognised, so a definition left inside ``/* ... */`` counted as code: with the
    real line gone, the guard read the stale value from the comment and stayed green.
    """

    def test_a_definition_only_inside_a_block_comment_is_rejected(self) -> None:
        text = "/*\npub const LIMIT: usize = 4;\n*/\nfn f() {}\n"
        with self.assertRaisesRegex(AssertionError, "found 0"):
            _helpers().one_definition(text, r"pub const LIMIT: usize = (\d+);", "x.rs")

    def test_a_definition_only_inside_a_test_module_is_rejected(self) -> None:
        # [MAGI S6 loop 44, F1b2a-melchior] a copy in a test module answered for a production definition.
        text = "#[cfg(test)]\nmod tests {\n    pub const LIMIT: usize = 4;\n}\n"
        with self.assertRaisesRegex(AssertionError, "found 0"):
            _helpers().one_definition(text, r"pub const LIMIT: usize = (\d+);", "x.rs")

    def test_the_real_definition_wins_over_an_old_one_in_a_block_comment(self) -> None:
        text = "/* was:\npub const LIMIT: usize = 4;\n*/\npub const LIMIT: usize = 5;\n"
        match = _helpers().one_definition(text, r"pub const LIMIT: usize = (\d+);", "x.rs")
        self.assertEqual("5", match.group(1))

    def test_a_literal_only_inside_an_inline_block_comment_is_rejected(self) -> None:
        text = 'let a = f(/* "path is outside" */ 1);\n'
        with self.assertRaisesRegex(AssertionError, "found 0"):
            _helpers().one_occurrence(text, '"path is outside"', "e.rs")

    def test_nested_block_comments_close_where_rust_closes_them(self) -> None:
        text = "/* a /* b */ pub const N: u8 = 1; */\npub const N: u8 = 2;\n"
        self.assertEqual("2", _helpers().one_definition(text, r"pub const N: u8 = (\d+);", "n.rs").group(1))

    def test_a_comment_opener_inside_a_string_or_char_is_code(self) -> None:
        text = ("let glob = \"src/**/*.rs\";\nlet c = '/';\nlet r = r#\"/* \"x\" */\"#;\n"
                "fn f<'a>(x: &'a str) {}\npub const N: u8 = 3;\n")
        self.assertEqual("3", _helpers().one_definition(text, r"pub const N: u8 = (\d+);", "n.rs").group(1))
        self.assertEqual(text, _helpers().code_only(text))

    def test_a_block_comment_opener_inside_a_line_comment_opens_nothing(self) -> None:
        text = "// see /* here\npub const N: u8 = 4;\n"
        self.assertEqual("4", _helpers().one_definition(text, r"pub const N: u8 = (\d+);", "n.rs").group(1))

    def test_code_only_blanks_a_block_comment_and_keeps_the_lines(self) -> None:
        text = 'a = [\n    "kept", /* "gone" */\n    /*\n    "gone too",\n    */\n]\n'
        blanked = _helpers().code_only(text)
        self.assertEqual(text.count("\n"), blanked.count("\n"))
        self.assertNotIn("gone", blanked)
        self.assertIn('"kept"', blanked)

    def test_an_unterminated_block_comment_fails_loudly(self) -> None:
        with self.assertRaisesRegex(AssertionError, "unterminated"):
            _helpers().code_only("pub const N: u8 = 1; /* never closed\n")


class CharListTests(unittest.TestCase):
    """[MAGI §6 MS2 loop 28, F1b2a-melchior] a Rust char-literal array is read whole, escapes included.

    The trim-list pin read ``'(.)'``, which cannot see an escaped literal such as ``'\\\\'``: the guard read
    fewer characters than the product lists and pinned the harness's copy to the shorter list.
    """

    def test_plain_literals_are_read_in_order(self) -> None:
        self.assertEqual([".", " "], _helpers().char_list("'.', ' '"))

    def test_escaped_literals_are_unescaped(self) -> None:
        self.assertEqual(["/", "\\", "'", "\n", "\t", "\0", "A", "é"],
                         _helpers().char_list(r"'/', '\\', '\'', '\n', '\t', '\0', '\x41', '\u{e9}'"))

    def test_a_trailing_comma_and_line_breaks_are_allowed(self) -> None:
        self.assertEqual([".", " "], _helpers().char_list("\n    '.',\n    ' ',\n"))

    def test_anything_but_a_char_literal_fails_loudly(self) -> None:
        # [F1b2a] ``'\u{FFFFFF}'`` fits the six-digit pattern but is no Unicode scalar: chr raised ValueError.
        for text in ("'.', SEPARATOR", "'.', \"x\"", "'ab'", "", r"'\u{FFFFFF}'", r"'\u{D800}'"):
            with self.assertRaises(AssertionError, msg=text):
                _helpers().char_list(text)


class ReadSourceTests(unittest.TestCase):
    """[MAGI §6 MS2 loop 27, F1b2a-melchior] ``read_source``, which every guard stands on, does what it says."""

    def _read(self, data: bytes) -> str:
        """Write *data* under a temporary root and read it back through ``read_source`` with that root."""
        import pathlib
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "x.rs").write_bytes(data)
            return _helpers().read_source("src/x.rs", root)

    def test_crlf_line_endings_are_normalised(self) -> None:
        self.assertEqual("const A: u8 = 1;\nconst B: u8 = 2;\n",
                         self._read(b"const A: u8 = 1;\r\nconst B: u8 = 2;\r\n"))

    def test_a_byte_that_is_not_utf8_is_replaced_not_fatal(self) -> None:
        self.assertEqual("let b = b\"�\";\nconst N: u8 = 3;\n",
                         self._read(b"let b = b\"\xff\";\nconst N: u8 = 3;\n"))

    def test_the_root_override_is_where_the_file_is_read_from(self) -> None:
        self.assertEqual("only under the given root\n", self._read(b"only under the given root\n"))

    def test_a_path_leaving_the_root_is_refused(self) -> None:
        # [MAGI §6 MS2 loop 28, F1b2a-balthasar] the helper joined any relative path to the root.
        import pathlib
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "repo"
            root.mkdir()
            (pathlib.Path(tmp) / "outside.rs").write_bytes(b"secret\n")
            for relative in ("../outside.rs", str(pathlib.Path(tmp) / "outside.rs")):
                with self.assertRaises(ValueError, msg=relative):
                    _helpers().read_source(relative, root)

    def test_without_a_root_the_repository_is_read(self) -> None:
        helpers = _helpers()
        self.assertTrue((helpers.REPO_ROOT / "Cargo.toml").is_file())
        self.assertIn("[package]", helpers.read_source("Cargo.toml"))


class SecretOpeningsTests(unittest.TestCase):
    """[MAGI §6 MS3 loop 13, R6a] ``secret_openings`` fails loudly when the table it reads matches
    nothing: a filter that matches nothing raises no error, so a drifted source shape would
    silently drop every opening from the scenarios that read them."""

    _TOKENS = 'SecretPattern { prefixes: &["ghp_", "sk-"] }\n'

    def _openings(self, source: str):
        """Run ``secret_openings`` over *source* written as the product's secrets file."""
        import pathlib
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            target = root / "src" / "review"
            target.mkdir(parents=True)
            (target / "secrets.rs").write_text(source, encoding="utf-8")
            return _helpers().secret_openings(root)

    def test_the_real_table_yields_its_header_and_token_openings(self) -> None:
        import pathlib
        openings = _helpers().secret_openings(pathlib.Path(__file__).resolve().parents[2])
        self.assertIn("-----begin ", openings)
        self.assertGreater(len(openings), 10)

    def test_a_table_with_no_fixed_openings_is_refused(self) -> None:
        source = self._TOKENS + "const NON_TOKEN_PREFIXES: &[&str] = &[];\n"
        with self.assertRaises(AssertionError):
            self._openings(source)

    def test_a_table_with_fixed_openings_parses(self) -> None:
        source = self._TOKENS + 'const NON_TOKEN_PREFIXES: &[&str] = &["-----begin "];\n'
        self.assertEqual(["ghp_", "sk-", "-----begin "], self._openings(source))

class GuardRoutingTests(unittest.TestCase):
    """The sync guards themselves read through the helpers."""

    def test_the_constant_guard_reads_the_definition_not_the_comment_above_it(self) -> None:
        from smoke.tests import test_rust_sync
        self.assertEqual(5, test_rust_sync._usize_const(_DRIFTED, "BYTES_PER_TOKEN"))


class LiteralScanTests(unittest.TestCase):
    """[MAGI §6 MS2 loop 29, F1b2a] what the literal scan does with an open literal and a byte literal."""

    def test_an_unterminated_string_fails_loudly_instead_of_swallowing_the_file(self) -> None:
        # It consumed the rest of the file as string content, so no block comment after it was blanked and an
        # old definition left in one answered for a real definition that no longer exists.
        text = 'let s = "never closed;\n/*\npub const N: u8 = 1;\n*/\n'
        with self.assertRaisesRegex(AssertionError, "unterminated"):
            _helpers().one_definition(text, r"pub const N: u8 = (\d+);", "n.rs")

    def test_an_unterminated_raw_string_fails_loudly(self) -> None:
        with self.assertRaisesRegex(AssertionError, "unterminated"):
            _helpers().code_only('let s = r#"never closed;\npub const N: u8 = 1;\n')

    def test_a_string_may_span_lines_as_rust_allows(self) -> None:
        text = 'let s = "one\n/* two */\n";\npub const N: u8 = 3;\n'
        self.assertEqual("3", _helpers().one_definition(text, r"pub const N: u8 = (\d+);", "n.rs").group(1))
        self.assertIn("/* two */", _helpers().code_only(text))

    def test_a_comment_opener_inside_a_byte_literal_opens_nothing(self) -> None:
        text = "let b = b\"/*\";\nlet c = b'/';\nlet r = br#\"/*\"#;\npub const N: u8 = 4;\n"
        self.assertEqual("4", _helpers().one_definition(text, r"pub const N: u8 = (\d+);", "n.rs").group(1))
        self.assertEqual(text, _helpers().code_only(text))

    def test_a_comment_opener_inside_a_c_string_literal_opens_nothing(self) -> None:
        # [F1b2a] ``cr#"..."#`` was not taken for a raw string: its ``r`` follows a letter, so the scan read
        # ``"a"`` as a plain string and ``/*"#;`` as an unterminated block comment. ``cr"\"`` holds a backslash
        # that a plain-string reading takes for an escape.
        text = ('let a = cr#"a"/*"#;\nlet b = cr"\\";\nlet c = c"/*";\n'
                "pub const N: u8 = 5;\n")
        self.assertEqual("5", _helpers().one_definition(text, r"pub const N: u8 = (\d+);", "n.rs").group(1))
        self.assertEqual(text, _helpers().code_only(text))

    def test_an_identifier_ending_in_c_before_r_is_not_a_raw_string(self) -> None:
        # The prefix is the whole token: ``abcr#"`` is an identifier, never a C raw string.
        text = 'let s = "x"; let abcr = 1; /* old */\npub const N: u8 = 6;\n'
        self.assertNotIn("old", _helpers().code_only(text))


class WithoutCommentsTests(unittest.TestCase):
    """[MAGI §6 MS2 loop 29, F1b2a-melchior] a trailing comment is not code either."""

    def test_a_trailing_line_comment_is_blanked_and_the_code_before_it_kept(self) -> None:
        stripped = _helpers().without_comments("let x = 1; // context assembly failed\n")
        self.assertNotIn("context assembly failed", stripped)
        self.assertIn("let x = 1;", stripped)

    def test_slashes_inside_a_string_are_not_a_comment(self) -> None:
        text = 'let u = "http://host"; let m = "context assembly failed";\n'
        self.assertEqual(text, _helpers().without_comments(text))

    def test_block_and_doc_comments_are_blanked_and_the_lines_kept(self) -> None:
        text = "/// doc marker\nlet a = 1; /* block marker */\n//! inner marker\n"
        stripped = _helpers().without_comments(text)
        self.assertEqual(text.count("\n"), stripped.count("\n"))
        self.assertNotIn("marker", stripped)
        self.assertIn("let a = 1;", stripped)


if __name__ == "__main__":
    unittest.main()
